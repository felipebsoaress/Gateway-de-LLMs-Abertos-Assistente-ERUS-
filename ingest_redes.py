import os
import requests
import uuid
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct

import warnings
import logging
warnings.filterwarnings("ignore")
logging.getLogger("pypdf").setLevel(logging.ERROR)
# ------------------------------------------------------------------------

PASTA_SLIDES = "./slides_redes"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
QDRANT_URL = "localhost"

cliente_qdrant = QdrantClient(host=QDRANT_URL, port=6333)

def configurar_banco_redes():
    nome_colecao = "material_redes"
    if cliente_qdrant.collection_exists(nome_colecao):
        cliente_qdrant.delete_collection(nome_colecao)
        
    cliente_qdrant.create_collection(
        collection_name=nome_colecao,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE)
    )
    return nome_colecao

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

def indexar_slides():
    if not os.path.exists(PASTA_SLIDES):
        print(f"⚠️ A pasta {PASTA_SLIDES} não existe. Crie a pasta e coloque os PDFs lá.")
        return

    colecao = configurar_banco_redes()
    print("📚 Iniciando extração dos slides de Redes...")

    for arquivo in os.listdir(PASTA_SLIDES):
        if arquivo.endswith(".pdf"):
            caminho = os.path.join(PASTA_SLIDES, arquivo)
            leitor = PdfReader(caminho)
            texto_pdf = ""
            
            for pagina in leitor.pages:
                texto_pdf += pagina.extract_text() + "\n"
            
            chunks = fatiar_texto(texto_pdf)
            pontos = []
            
            for trecho in chunks:
                if len(trecho.strip()) < 20: 
                    continue
                    
                vetor = gerar_embedding(trecho)
                if vetor:
                    pontos.append(PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vetor,
                        payload={"fonte": arquivo, "conteudo": trecho}
                    ))
            
            if pontos:
                cliente_qdrant.upsert(collection_name=colecao, points=pontos)
                print(f"✅ {len(pontos)} pedaços extraídos de '{arquivo}' salvos no Qdrant.")

if __name__ == "__main__":
    indexar_slides()