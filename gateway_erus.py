import asyncio
import json
import time
import uuid

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, PlainTextResponse
from pydantic import BaseModel
import requests
import uvicorn
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

# Inicializa a aplicação FastAPI que atua como gateway HTTP para os nós do Ollama
app = FastAPI(title="Gateway de LLMs Abertos - Projeto ERUS")

# Controle de concorrência: limita quantas gerações simultâneas podem rodar ao mesmo tempo
LIMITE_CONCORRENCIA = 2
semaforo = asyncio.Semaphore(LIMITE_CONCORRENCIA)

# Configuração da conexão com o banco vetorial Qdrant (usado para RAG e cache semântico)
QDRANT_URL = "localhost"
NOME_CACHE = "cache_respostas"
cliente_qdrant = QdrantClient(host=QDRANT_URL, port=6333)


def configurar_cache():
    """
    Garante que a coleção usada para o cache semântico exista no Qdrant.
    Os vetores têm dimensão 768 (compatível com o modelo 'nomic-embed-text')
    e usam similaridade de cosseno (COSINE) para comparar perguntas.
    """
    if not cliente_qdrant.collection_exists(NOME_CACHE):
        cliente_qdrant.create_collection(
            collection_name=NOME_CACHE,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )


# Executa a configuração inicial do cache ao carregar o módulo
configurar_cache()

# Tabela de roteamento por domínio:
# Cada domínio define seu modelo gerador, prompt de sistema, coleção RAG no Qdrant
# e os endpoints (nós do Ollama) responsáveis por geração e embeddings.
DOMINIOS = {
    "erus-normas": {
        "modelo": "qwen2.5-coder:1.5b",
        "sistema": "Você é o assistente da equipe ERUS. Responda de forma natural e direta, utilizando APENAS os dados fornecidos no texto de referência.",
        "colecao": "site_erus",
        "url_geracao": "http://127.0.0.1:11434/api/generate",    # Nó 1 (Geração)
        "url_embedding": "http://127.0.0.1:11434/api/embeddings"  # Nó 1 (Embedding)
    },
    "erus-redes": {
        "modelo": "qwen2.5-coder:1.5b",
        "sistema": "Você é o engenheiro de redes da equipe ERUS. Responda APENAS usando o texto de referência. Se a informação não estiver no texto, responda EXATAMENTE: 'Não encontrei essa informação no material da disciplina'.",
        "colecao": "material_redes",
        "url_geracao": "http://127.0.0.1:11434/api/generate",    # Nó 2 (Geração Distribuída)
        "url_embedding": "http://127.0.0.1:11434/api/embeddings"  # Nó 1 (Embedding Centralizado)
    }
}


# Esquema de validação do corpo JSON recebido no endpoint POST /perguntar
class RequisicaoCliente(BaseModel):
    dominio: str
    pergunta: str


def gerar_vetor(texto: str, url_embedding: str):
    """
    Converte um texto (pergunta do usuário) em um vetor numérico (embedding)
    consultando o modelo 'nomic-embed-text' no nó do Ollama informado.
    """
    payload = {"model": "nomic-embed-text", "prompt": texto}
    try:
        resposta = requests.post(url_embedding, json=payload).json()
        if "embedding" not in resposta:
            print(f"\n❌ Erro interno do Ollama no embedding: {resposta}")
            return []  # Retorna lista vazia para sinalizar falha sem derrubar o servidor
        return resposta.get("embedding", [])
    except Exception as e:
        print(f"\n❌ Erro de conexão ao tentar gerar embedding: {e}")
        return []


# ================= CACHE SEMÂNTICO =================
def buscar_no_cache(vetor):
    """
    Procura no Qdrant uma pergunta já respondida com similaridade >= 0.95.
    Se encontrar, retorna a resposta salva em cache; caso contrário, retorna None.
    """
    resultados = cliente_qdrant.query_points(
        collection_name=NOME_CACHE,
        query=vetor,
        limit=1,
        score_threshold=0.95
    ).points
    if resultados:
        return resultados[0].payload.get("resposta")
    return None


def salvar_no_cache(vetor, pergunta, resposta_completa):
    """
    Armazena o vetor da pergunta e o par (pergunta, resposta) na coleção de cache
    do Qdrant usando um identificador UUID único.
    """
    cliente_qdrant.upsert(
        collection_name=NOME_CACHE,
        points=[PointStruct(
            id=str(uuid.uuid4()),
            vector=vetor,
            payload={"pergunta": pergunta, "resposta": resposta_completa}
        )]
    )


# ================= RAG =================
def buscar_no_banco(vetor_pergunta, nome_colecao):
    """
    Recupera os 3 trechos mais relevantes na coleção RAG do domínio informado
    e os concatena em um único bloco de texto para servir de contexto ao LLM.
    """
    if not cliente_qdrant.collection_exists(nome_colecao):
        return ""
    resultados = cliente_qdrant.query_points(
        collection_name=nome_colecao,
        query=vetor_pergunta,
        limit=3
    ).points
    return "\n".join([r.payload.get("conteudo", "") for r in resultados])


# ================= GERAÇÃO =================
async def chamar_ollama_streaming(pergunta: str, config: dict, vetor_pergunta: list):
    """
    Executa o fluxo de RAG + geração em streaming respeitando o semáforo de concorrência.
    Envia cada token assim que chega do Ollama, registra métricas (TTFT, latência e vazão)
    e salva a resposta completa no cache semântico ao final.
    """
    # Aguarda vaga no semáforo caso o limite de requisições simultâneas esteja ocupado
    async with semaforo:
        # 1. Busca o contexto relevante no Qdrant (etapa de Retrieval do RAG)
        contexto = buscar_no_banco(vetor_pergunta, config["colecao"])
        prompt_enriquecido = (
            "Por favor, responda à pergunta do usuário usando as informações "
            f"do texto de referência abaixo.\n\nTexto de referência:\n{contexto}\n\n"
            f"Pergunta: {pergunta}"
        )

        # 2. Monta a requisição de geração em modo streaming com baixa temperatura (mais factual)
        payload = {
            "model": config["modelo"],
            "prompt": prompt_enriquecido,
            "system": config["sistema"],
            "stream": True,
            "options": {"temperature": 0.3}
        }

        # Variáveis para instrumentação de desempenho da requisição
        inicio = time.time()
        primeiro_token = True
        tokens = 0
        resposta_completa = ""

        # Executa a abertura da conexão HTTP bloqueante fora do loop principal do asyncio
        loop = asyncio.get_running_loop()
        res = await loop.run_in_executor(
            None,
            lambda: requests.post(config["url_geracao"], json=payload, stream=True)
        )

        # 3. Lê o stream linha a linha (cada linha é um objeto JSON retornado pelo Ollama)
        for linha in res.iter_lines():
            if linha:
                chunk = json.loads(linha)
                token = chunk.get("response", "")
                if token:
                    resposta_completa += token

                    # Mede o TTFT (Time To First Token) na chegada do primeiro token válido
                    if primeiro_token:
                        ttft = time.time() - inicio
                        print(f"\n📊 TTFT (LLM Gerador): {ttft:.4f}s")
                        primeiro_token = False

                    tokens += 1
                    # Envia o token imediatamente ao cliente HTTP
                    yield token

        # 4. Persiste a resposta no cache semântico e imprime latência total e vazão
        salvar_no_cache(vetor_pergunta, pergunta, resposta_completa)
        tempo_total = time.time() - inicio
        print(f"📊 Latência: {tempo_total:.4f}s | Vazão: {tokens / tempo_total:.2f} tok/s")


@app.post("/perguntar")
async def rota_perguntar(req: RequisicaoCliente):
    """
    Endpoint principal do gateway.
    Valida o domínio, gera o embedding da pergunta, verifica o cache semântico e,
    em caso de cache miss, encaminha a geração em streaming para o nó correspondente.
    """
    dominio = req.dominio.lower()
    if dominio not in DOMINIOS:
        return {"erro": "Domínio inválido. Use 'erus-normas' ou 'erus-redes'."}

    # Seleciona as configurações (modelo, coleção e URLs) correspondentes ao domínio
    config = DOMINIOS[dominio]
    vetor_pergunta = gerar_vetor(req.pergunta, config["url_embedding"])

    # Verifica se já existe uma resposta suficientemente semelhante no cache semântico
    resposta_cache = buscar_no_cache(vetor_pergunta)
    if resposta_cache:
        print("\n⚡ CACHE HIT: Pergunta similar encontrada!")
        print(f"📊 TTFT (Cache): 0.00s | GPU não utilizada.")
        return PlainTextResponse(f"[Cache Rápido] {resposta_cache}")

    # Em caso de cache miss, retorna um fluxo contínuo com os tokens gerados pelo Ollama
    return StreamingResponse(
        chamar_ollama_streaming(req.pergunta, config, vetor_pergunta),
        media_type="text/event-stream"
    )


if __name__ == "__main__":
    # Sobe o servidor Uvicorn escutando em todas as interfaces de rede na porta 8080
    uvicorn.run(app, host="0.0.0.0", port=8080)