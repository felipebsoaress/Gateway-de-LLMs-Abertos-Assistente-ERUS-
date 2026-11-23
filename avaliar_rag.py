import csv
import json
import os
import requests
from qdrant_client import QdrantClient

OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embeddings"
OLLAMA_GEN_URL = "http://127.0.0.1:11434/api/generate"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLECAO_SITE = "site_erus"
MODELO_LLM = "qwen2.5-coder:1.5b"
MODELO_EMBED = "nomic-embed-text"
SAIDA_CSV = "avaliacao_rag_20_perguntas.csv"
SISTEMA_ERUS = (
    "Você é o assistente da equipe ERUS. Responda de forma natural e direta, "
    "utilizando APENAS os dados fornecidos no texto de referência."
)

CAMPOS = [
    "id",
    "categoria",
    "pergunta",
    "resposta_esperada",
    "resposta_sem_rag",
    "resposta_com_rag",
    "nota_sem_rag_0_a_2",
    "nota_com_rag_0_a_2",
]

qdrant = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)


def embed(texto: str) -> list[float]:
    resp = requests.post(
        OLLAMA_EMBED_URL,
        json={"model": MODELO_EMBED, "prompt": texto, "keep_alive": "15m"},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json().get("embedding", [])


def recuperar_contexto(pergunta: str, top_k: int = 2, max_chars_por_trecho: int = 900) -> str:
    vetor = embed(pergunta)
    if not vetor or not qdrant.collection_exists(COLECAO_SITE):
        return ""
    pontos = qdrant.query_points(collection_name=COLECAO_SITE, query=vetor, limit=top_k).points
    trechos = [p.payload.get("conteudo", "")[:max_chars_por_trecho] for p in pontos]
    return "\n---\n".join(trechos)


def gerar_resposta(prompt: str, sistema: str) -> str:
    payload = {
        "model": MODELO_LLM,
        "prompt": prompt,
        "system": sistema,
        "stream": False,
        "keep_alive": "15m",
        "options": {"temperature": 0.3},
    }
    resp = requests.post(OLLAMA_GEN_URL, json=payload, timeout=500)
    resp.raise_for_status()
    return resp.json().get("response", "").strip()


def carregar_existentes(caminho: str) -> dict[int, dict]:
    if not os.path.exists(caminho):
        return {}
    existentes = {}
    with open(caminho, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                idx = int(row["id"])
                if row.get("resposta_sem_rag") and row.get("resposta_com_rag"):
                    existentes[idx] = row
            except (KeyError, ValueError):
                continue
    return existentes


def salvar_csv(caminho: str, linhas: list[dict]) -> None:
    with open(caminho, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CAMPOS)
        writer.writeheader()
        writer.writerows(linhas)


def main() -> None:
    with open("gabarito_erus_20.json", "r", encoding="utf-8") as f:
        itens = json.load(f)

    existentes = carregar_existentes(SAIDA_CSV)
    linhas_saida: list[dict] = []

    for item in itens:
        item_id = int(item["id"])
        if item_id in existentes:
            linhas_saida.append(existentes[item_id])
            print(f"[{item_id:02d}/20] Já processado, pulando...")
            continue

        pergunta = item["pergunta"]
        contexto = recuperar_contexto(pergunta, top_k=2, max_chars_por_trecho=900)

        prompt_sem_rag = f"Responda à pergunta a seguir:\n\nPergunta: {pergunta}"
        resp_sem_rag = gerar_resposta(prompt_sem_rag, SISTEMA_ERUS)

        prompt_com_rag = (
            "Por favor, responda à pergunta do usuário usando as informações "
            f"do texto de referência abaixo.\n\nTexto de referência:\n{contexto}\n\n"
            f"Pergunta: {pergunta}"
        )
        resp_com_rag = gerar_resposta(prompt_com_rag, SISTEMA_ERUS)

        linhas_saida.append({
            "id": item_id,
            "categoria": item["categoria"],
            "pergunta": pergunta,
            "resposta_esperada": item["resposta_esperada"],
            "resposta_sem_rag": resp_sem_rag,
            "resposta_com_rag": resp_com_rag,
            "nota_sem_rag_0_a_2": "",
            "nota_com_rag_0_a_2": "",
        })
        salvar_csv(SAIDA_CSV, linhas_saida)
        print(f"[{item_id:02d}/20] Concluído e salvo: {pergunta[:50]}...")

    print(f"\n✅ Avaliação concluída em '{SAIDA_CSV}'.")


if __name__ == "__main__":
    main()
