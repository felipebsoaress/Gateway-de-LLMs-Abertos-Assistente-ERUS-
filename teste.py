import requests
from qdrant_client import QdrantClient

OLLAMA_URL_EMBEDDING = "http://localhost:11434/api/embeddings"
OLLAMA_URL_GENERATE = "http://localhost:11434/api/generate"
QDRANT_URL = "localhost"
COLECAO_RAG = "site_erus"

cliente_qdrant = QdrantClient(host=QDRANT_URL, port=6333)

def gerar_embedding(texto):
    payload = {"model": "nomic-embed-text", "prompt": texto}
    resposta = requests.post(OLLAMA_URL_EMBEDDING, json=payload).json()
    return resposta.get("embedding", [])

def buscar_no_site(pergunta):
    vetor = gerar_embedding(pergunta)
    resultados = cliente_qdrant.query_points(
        collection_name=COLECAO_RAG,
        query=vetor,
        limit=3
    ).points
    return "\n".join([r.payload.get("conteudo", "") for r in resultados])

def perguntar_ia(pergunta, contexto):
    # Prompt ajustado com o tom colaborativo para evitar travar o filtro do modelo
    prompt = f"Por favor, responda à pergunta do usuário usando as informações do texto de referência abaixo.\n\nTexto de referência:\n{contexto}\n\nPergunta: {pergunta}"
    
    payload = {
        "model": "qwen2.5-coder:1.5b",
        "prompt": prompt,
        "system": "Você é o assistente da equipe ERUS. Responda de forma natural e direta em português, utilizando apenas os dados fornecidos no texto de referência.",
        "stream": False,
        "options": {
            "temperature": 0.3 # Temperatura baixa para garantir a factualidade
        }
    }
    res = requests.post(OLLAMA_URL_GENERATE, json=payload).json()
    return res.get("response", "")

if __name__ == "__main__":
    # Altere a pergunta aqui para testar diferentes conhecimentos!
    pergunta = "Quais sao os membros da equipeERUS?"
    print(f"🕵️ Buscando no site: '{pergunta}'...\n")
    
    contexto = buscar_no_site(pergunta)
    print("🤖 Resposta da IA:")
    print(perguntar_ia(pergunta, contexto))