import requests
import uuid
from bs4 import BeautifulSoup
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct


import warnings
warnings.filterwarnings("ignore")

# URLs que queremos ler do site público
URLS_ERUS = [
    "https://www.erus.ufes.br/",
    "https://www.erus.ufes.br/sobre/",
    "https://www.erus.ufes.br/projetos/",
    "https://www.erus.ufes.br/membros/",
    "https://www.erus.ufes.br/conquistas/"
]

OLLAMA_URL = "http://localhost:11434/api/embeddings"
QDRANT_URL = "localhost"

cliente_qdrant = QdrantClient(host=QDRANT_URL, port=6333)

def configurar_banco():
    nome_colecao = "site_erus"
    # Reseta a coleção se existir para uma instalação limpa
    if cliente_qdrant.collection_exists(nome_colecao):
        cliente_qdrant.delete_collection(nome_colecao)
        
    cliente_qdrant.create_collection(
        collection_name=nome_colecao,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
    return nome_colecao

def extrair_texto_html(url):
    try:
        html = requests.get(url).text
        soup = BeautifulSoup(html, "html.parser")
        
        # Remove tags inúteis que sujam o texto do RAG
        for tag in soup(["script", "style", "nav", "footer"]):
            tag.decompose()
            
        # Puxa apenas o texto limpo
        return soup.get_text(separator='\n', strip=True)
    except Exception as e:
        print(f"Erro ao ler {url}: {e}")
        return ""

def fatiar_texto(texto, tamanho=1500, overlap=200):
    chunks = []
    inicio = 0
    while inicio < len(texto):
        fim = inicio + tamanho
        chunks.append(texto[inicio:fim])
        inicio = fim - overlap
    return chunks

def gerar_embedding(texto):
    payload = {"model": "nomic-embed-text", "prompt": texto}
    resposta = requests.post(OLLAMA_URL, json=payload).json()
    return resposta.get("embedding", [])

def indexar_site():
    colecao = configurar_banco()
    print("🌐 Iniciando varredura do site oficial ERUS...")
    
    for url in URLS_ERUS:
        texto = extrair_texto_html(url)
        if not texto:
            continue
            
        chunks = fatiar_texto(texto)
        pontos = []
        
        for trecho in chunks:
            # Ignora pedaços muito curtos
            if len(trecho) < 20: 
                continue
                
            vetor = gerar_embedding(trecho)
            if vetor:
                pontos.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vetor,
                    payload={"fonte": url, "conteudo": trecho}
                ))
        
        if pontos:
            cliente_qdrant.upsert(collection_name=colecao, points=pontos)
            print(f"✅ {len(pontos)} pedaços extraídos de {url} salvos no Qdrant.")

if __name__ == "__main__":
    indexar_site()