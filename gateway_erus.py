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

app = FastAPI(title="Gateway de LLMs Abertos - Projeto ERUS")

LIMITE_CONCORRENCIA = 2
semaforo = asyncio.Semaphore(LIMITE_CONCORRENCIA)

QDRANT_URL = "localhost"
NOME_CACHE = "cache_respostas"
cliente_qdrant = QdrantClient(host=QDRANT_URL, port=6333)

def configurar_cache():
    if not cliente_qdrant.collection_exists(NOME_CACHE):
        cliente_qdrant.create_collection(
            collection_name=NOME_CACHE,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
configurar_cache()

# As URLs do Ollama foram movidas para dentro de cada domínio
DOMINIOS = {
    "erus-normas": {
        "modelo": "qwen2.5-coder:1.5b",
        "sistema": "Você é o assistente da equipe ERUS. Responda de forma natural e direta, utilizando APENAS os dados fornecidos no texto de referência.",
        "colecao": "site_erus",
        "url_geracao": "http://127.0.0.1:11434/api/generate",   # Nó 1 (Geração)
        "url_embedding": "http://127.0.0.1:11434/api/embeddings" # Nó 1 (Embedding)
    },
    "erus-redes": {
        "modelo": "qwen2.5-coder:1.5b",
        "sistema": "Você é o engenheiro de redes da equipe ERUS. Responda APENAS usando o texto de referência. Se a informação não estiver no texto, responda EXATAMENTE: 'Não encontrei essa informação no material da disciplina'.",
        "colecao": "material_redes",
        "url_geracao": "http://127.0.0.1:11435/api/generate",   # Nó 2 (Geração Distribuída)
        "url_embedding": "http://127.0.0.1:11434/api/embeddings" # Nó 1 (Embedding Centralizado)
    }
}

class RequisicaoCliente(BaseModel):
    dominio: str
    pergunta: str

def gerar_vetor(texto: str, url_embedding: str):
    payload = {"model": "nomic-embed-text", "prompt": texto}
    try:
        resposta = requests.post(url_embedding, json=payload).json()
        if "embedding" not in resposta:
            print(f"\n❌ Erro interno do Ollama no embedding: {resposta}")
            return [] # Retorna vazio, mas o erro fica claro no terminal
        return resposta.get("embedding", [])
    except Exception as e:
        print(f"\n❌ Erro de conexão ao tentar gerar embedding: {e}")
        return []

# ================= CACHE SEMÂNTICO =================
def buscar_no_cache(vetor):
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
    async with semaforo:
        contexto = buscar_no_banco(vetor_pergunta, config["colecao"])
        prompt_enriquecido = f"Por favor, responda à pergunta do usuário usando as informações do texto de referência abaixo.\n\nTexto de referência:\n{contexto}\n\nPergunta: {pergunta}"
        
        payload = {
            "model": config["modelo"],
            "prompt": prompt_enriquecido,
            "system": config["sistema"],
            "stream": True,
            "options": {"temperature": 0.3} 
        }
        
        inicio = time.time()
        primeiro_token = True
        tokens = 0
        resposta_completa = ""
        
        loop = asyncio.get_running_loop()
        # Aqui o requests.post agora usa a URL de geração dinâmica
        res = await loop.run_in_executor(None, lambda: requests.post(config["url_geracao"], json=payload, stream=True))
        
        for linha in res.iter_lines():
            if linha:
                chunk = json.loads(linha)
                token = chunk.get("response", "")
                if token:
                    resposta_completa += token
                    if primeiro_token:
                        ttft = time.time() - inicio
                        print(f"\n📊 TTFT (LLM Gerador): {ttft:.4f}s")
                        primeiro_token = False
                    
                    tokens += 1
                    yield token
                    
        salvar_no_cache(vetor_pergunta, pergunta, resposta_completa)
        tempo_total = time.time() - inicio
        print(f"📊 Latência: {tempo_total:.4f}s | Vazão: {tokens / tempo_total:.2f} tok/s")

@app.post("/perguntar")
async def rota_perguntar(req: RequisicaoCliente):
    dominio = req.dominio.lower()
    if dominio not in DOMINIOS:
        return {"erro": "Domínio inválido. Use 'erus-normas' ou 'erus-redes'."}
        
    # Pega as configurações do domínio antes para saber pra qual URL mandar
    config = DOMINIOS[dominio]
    vetor_pergunta = gerar_vetor(req.pergunta, config["url_embedding"])
    
    resposta_cache = buscar_no_cache(vetor_pergunta)
    if resposta_cache:
        print("\n⚡ CACHE HIT: Pergunta similar encontrada!")
        print(f"📊 TTFT (Cache): 0.00s | GPU não utilizada.")
        return PlainTextResponse(f"[Cache Rápido] {resposta_cache}")
        
    return StreamingResponse(
        chamar_ollama_streaming(req.pergunta, config, vetor_pergunta),
        media_type="text/event-stream"
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)