import asyncio
import json
import time
import uuid

import aiohttp
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
import uvicorn
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Inicializa a aplicação FastAPI que atua como gateway HTTP para os nós do Ollama
app = FastAPI(title="Gateway de LLMs Abertos - Projeto ERUS")

# Limite de gerações simultâneas POR NÓ de inferência
LIMITE_CONCORRENCIA_POR_NO = 2

# Configuração da conexão com o banco vetorial Qdrant (RAG e cache semântico)
QDRANT_URL = "localhost"
NOME_CACHE = "cache_respostas"
cliente_qdrant = QdrantClient(host=QDRANT_URL, port=6333)


def configurar_cache(resetar: bool = False):
    """
    Garante que a coleção de cache semântico exista no Qdrant.
    Se resetar=True, recria a coleção vazia para experimentos controlados.
    """
    if resetar and cliente_qdrant.collection_exists(NOME_CACHE):
        cliente_qdrant.delete_collection(NOME_CACHE)

    if not cliente_qdrant.collection_exists(NOME_CACHE):
        cliente_qdrant.create_collection(
            collection_name=NOME_CACHE,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )


configurar_cache()

# Tabela de roteamento por domínio
DOMINIOS = {
    "erus-normas": {
        "modelo": "qwen2.5-coder:1.5b",
        "sistema": (
            "Você é o assistente da equipe ERUS. Responda de forma natural e direta, "
            "utilizando APENAS os dados fornecidos no texto de referência."
        ),
        "colecao": "site_erus",
        "url_geracao": "http://127.0.0.1:11434/api/generate",     # Nó 1 (Geração)
        "url_embedding": "http://127.0.0.1:11434/api/embeddings"  # Nó 1 (Embedding)
    },
    "erus-redes": {
        "modelo": "qwen2.5-coder:1.5b",
        "sistema": (
            "Você é o engenheiro de redes da equipe ERUS. Responda APENAS usando o texto "
            "de referência. Se a informação não estiver no texto, responda EXATAMENTE: "
            "'Não encontrei essa informação no material da disciplina'."
        ),
        "colecao": "material_redes",
        "url_geracao": "http://127.0.0.1:11435/api/generate",     # Nó 2 (Geração Distribuída)
        "url_embedding": "http://127.0.0.1:11434/api/embeddings"  # Nó 1 (Embedding Centralizado)
    }
}

# Fila / semáforo independente para cada nó de geração configurado
SEMAFOROS_POR_NO = {
    cfg["url_geracao"]: asyncio.Semaphore(LIMITE_CONCORRENCIA_POR_NO)
    for cfg in DOMINIOS.values()
}


class RequisicaoCliente(BaseModel):
    dominio: str
    pergunta: str
    usar_cache: bool = True  # False permite medir carga pura do LLM sem cache hit


async def gerar_vetor(texto: str, url_embedding: str) -> list[float]:
    """
    Gera o embedding da pergunta de forma assíncrona via Ollama ('nomic-embed-text').
    """
    payload = {"model": "nomic-embed-text", "prompt": texto}
    timeout = aiohttp.ClientTimeout(total=120)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url_embedding, json=payload) as resp:
                resposta = await resp.json()
                if "embedding" not in resposta:
                    print(f"\n❌ Erro interno do Ollama no embedding: {resposta}")
                    return []
                return resposta.get("embedding", [])
    except Exception as e:
        print(f"\n❌ Erro de conexão ao tentar gerar embedding: {e}")
        return []


# ================= CACHE SEMÂNTICO =================
def _buscar_no_cache_sync(vetor: list[float]):
    if not vetor:
        return None
    resultados = cliente_qdrant.query_points(
        collection_name=NOME_CACHE,
        query=vetor,
        limit=1,
        score_threshold=0.95
    ).points
    if resultados:
        return resultados[0].payload.get("resposta")
    return None


def _salvar_no_cache_sync(vetor: list[float], pergunta: str, resposta_completa: str):
    if not vetor or not resposta_completa.strip():
        return
    cliente_qdrant.upsert(
        collection_name=NOME_CACHE,
        points=[PointStruct(
            id=str(uuid.uuid4()),
            vector=vetor,
            payload={"pergunta": pergunta, "resposta": resposta_completa}
        )]
    )


# ================= RAG =================
def _buscar_no_banco_sync(vetor_pergunta: list[float], nome_colecao: str) -> str:
    if not vetor_pergunta or not cliente_qdrant.collection_exists(nome_colecao):
        return ""
    resultados = cliente_qdrant.query_points(
        collection_name=nome_colecao,
        query=vetor_pergunta,
        limit=3
    ).points
    return "\n".join([r.payload.get("conteudo", "") for r in resultados])


# ================= GERAÇÃO EM STREAMING =================
async def chamar_ollama_streaming(
    pergunta: str,
    config: dict,
    vetor_pergunta: list[float],
    usar_cache: bool,
    inicio_requisicao: float,
):
    """
    Respeita a fila (semáforo) específica do nó de destino, executa o RAG e
    transmite os tokens via aiohttp sem bloquear o event loop do gateway.
    """
    semaforo_no = SEMAFOROS_POR_NO[config["url_geracao"]]

    async with semaforo_no:
        espera_fila = time.perf_counter() - inicio_requisicao
        contexto = await asyncio.to_thread(_buscar_no_banco_sync, vetor_pergunta, config["colecao"])

        prompt_enriquecido = (
            "Por favor, responda à pergunta do usuário usando as informações "
            f"do texto de referência abaixo.\n\nTexto de referência:\n{contexto}\n\n"
            f"Pergunta: {pergunta}"
        )

        payload = {
            "model": config["modelo"],
            "prompt": prompt_enriquecido,
            "system": config["sistema"],
            "stream": True,
            "options": {"temperature": 0.3}
        }

        inicio_geracao = time.perf_counter()
        primeiro_token = True
        tokens = 0
        resposta_completa = ""
        timeout = aiohttp.ClientTimeout(total=600)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(config["url_geracao"], json=payload) as res:
                if res.status != 200:
                    erro_txt = await res.text()
                    print(f"\n❌ Erro HTTP {res.status} no nó {config['url_geracao']}: {erro_txt}")
                    yield f"[Erro no nó de inferência: HTTP {res.status}]"
                    return

                async for linha in res.content:
                    linha = linha.strip()
                    if not linha:
                        continue
                    chunk = json.loads(linha.decode("utf-8"))
                    token = chunk.get("response", "")
                    if token:
                        resposta_completa += token
                        if primeiro_token:
                            ttft_fim_a_fim = time.perf_counter() - inicio_requisicao
                            ttft_no = time.perf_counter() - inicio_geracao
                            print(
                                f"\n📊 Nó={config['url_geracao']} | Fila={espera_fila:.3f}s | "
                                f"TTFT(nó)={ttft_no:.3f}s | TTFT(total)={ttft_fim_a_fim:.3f}s"
                            )
                            primeiro_token = False

                        tokens += 1
                        yield token

        if usar_cache:
            await asyncio.to_thread(_salvar_no_cache_sync, vetor_pergunta, pergunta, resposta_completa)

        tempo_geracao = max(time.perf_counter() - inicio_geracao, 1e-6)
        tempo_total = max(time.perf_counter() - inicio_requisicao, 1e-6)
        print(
            f"📊 Latência total: {tempo_total:.3f}s | "
            f"Vazão(nó): {tokens / tempo_geracao:.2f} tok/s | Tokens: {tokens}"
        )


@app.delete("/cache")
async def rota_limpar_cache():
    """Limpa a coleção de cache semântico para iniciar baterias de teste do zero."""
    await asyncio.to_thread(configurar_cache, True)
    print("\n🧹 Cache semântico limpo com sucesso.")
    return {"status": "cache_limpo"}


@app.post("/perguntar")
async def rota_perguntar(req: RequisicaoCliente):
    inicio_requisicao = time.perf_counter()
    dominio = req.dominio.lower()
    if dominio not in DOMINIOS:
        return {"erro": "Domínio inválido. Use 'erus-normas' ou 'erus-redes'."}

    config = DOMINIOS[dominio]
    vetor_pergunta = await gerar_vetor(req.pergunta, config["url_embedding"])

    if req.usar_cache:
        resposta_cache = await asyncio.to_thread(_buscar_no_cache_sync, vetor_pergunta)
        if resposta_cache:
            latencia_cache = time.perf_counter() - inicio_requisicao
            print(f"\n⚡ CACHE HIT! Latência (embedding + busca): {latencia_cache:.4f}s")
            return PlainTextResponse(f"[Cache Rápido] {resposta_cache}")

    return StreamingResponse(
        chamar_ollama_streaming(
            req.pergunta,
            config,
            vetor_pergunta,
            req.usar_cache,
            inicio_requisicao,
        ),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)