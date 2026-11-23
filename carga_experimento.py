import asyncio
import csv
import statistics
import time
import aiohttp

# ================= CONFIGURAÇÕES DO EXPERIMENTO =================
# Endpoint do gateway FastAPI que recebe as perguntas via HTTP POST
URL = "http://127.0.0.1:8080/perguntar"

# Níveis de clientes simultâneos exigidos no experimento (1, 2, 4, 8 e 16)
NIVEIS_CONCORRENCIA = [1, 2, 4, 8, 16]

# Quantidade de repetições por nível de concorrência para dar validade estatística
REPETICOES = 3

# Nome do arquivo CSV onde as métricas individuais de cada requisição serão salvas
SAIDA_CSV = "resultados_experimento_m2_m3.csv"

# Conjunto de perguntas-base sobre a ERUS usadas de forma cíclica entre os clientes
PERGUNTAS_BASE = [
    "Quais são os projetos desenvolvidos pela equipe ERUS?",
    "Como funciona a participação de membros na equipe ERUS?",
    "Quais são as principais conquistas da ERUS?",
    "Qual é o objetivo do projeto de extensão ERUS na UFES?",
    "Como é organizada a estrutura de trabalho da equipe ERUS?",
    "Quais áreas de engenharia fazem parte da ERUS?",
    "Onde posso encontrar informações sobre a história da ERUS?",
    "Quais competições ou eventos a ERUS participa?",
]


async def enviar_requisicao(
    session: aiohttp.ClientSession,
    concorrencia: int,
    repeticao: int,
    cliente_idx: int,
    pergunta: str,
) -> dict:
    """
    Simula um cliente individual enviando uma pergunta em streaming ao gateway.
    Mede o TTFT (tempo até o primeiro chunk/token), a duração total da resposta,
    estima a quantidade de tokens recebidos e identifica se houve acerto no cache.
    """
    # Monta o JSON esperado pelo gateway; o sufixo identifica a execução,
    # embora embeddings ainda possam atingir similaridade >= 0.95 no cache semântico
    payload = {
        "dominio": "erus-normas",
        "pergunta": f"{pergunta} [run={repeticao}-c={concorrencia}-id={cliente_idx}]",
    }

    # Relógio de alta precisão para medir latência do lado do cliente
    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    texto = []

    try:
        # Abre a conexão POST assíncrona e lê o corpo da resposta conforme os chunks chegam
        async with session.post(URL, json=payload) as resp:
            async for raw_line in resp.content:
                if not raw_line:
                    continue

                agora = time.perf_counter()
                # Registra o TTFT exatamente no instante em que o primeiro pedaço não vazio chega
                if ttft is None:
                    ttft = agora - t0

                parte = raw_line.decode("utf-8", errors="ignore")
                texto.append(parte)
                chunks += 1

    except Exception as exc:
        # Em caso de timeout ou falha de rede, registra o erro sem interromper toda a bateria
        return {
            "concorrencia": concorrencia,
            "repeticao": repeticao,
            "cliente_idx": cliente_idx,
            "ttft_s": None,
            "duracao_s": round(time.perf_counter() - t0, 4),
            "tokens_aprox": 0,
            "cache_hit": False,
            "erro": str(exc),
        }

    # Instante em que o servidor encerrou o envio da resposta completa
    t1 = time.perf_counter()
    resposta_completa = "".join(texto)

    # Estimativa simples de tokens (~4 caracteres por token em português)
    tokens_aprox = max(1, len(resposta_completa) // 4) if resposta_completa else 0

    # Verifica se a resposta veio do cache semântico do gateway ("[Cache Rápido]")
    cache_hit = "[Cache Rápido]" in resposta_completa

    return {
        "concorrencia": concorrencia,
        "repeticao": repeticao,
        "cliente_idx": cliente_idx,
        "ttft_s": round(ttft if ttft is not None else (t1 - t0), 4),
        "duracao_s": round(t1 - t0, 4),
        "tokens_aprox": tokens_aprox,
        "cache_hit": cache_hit,
        "erro": "",
    }


async def rodar_patamar(concorrencia: int, repeticao: int) -> list[dict]:
    """
    Dispara 'concorrencia' clientes simultâneos em uma única rodada e calcula
    as métricas agregadas da janela (TTFT médio e vazão agregada em tokens/s).
    """
    # Timeout alto (300s) para suportar filas longas sob alta concorrência em CPU
    timeout = aiohttp.ClientTimeout(total=300)
    t_inicio_rodada = time.perf_counter()

    # Cria uma sessão HTTP compartilhada e lança todas as requisições em paralelo
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tarefas = [
            enviar_requisicao(
                session=session,
                concorrencia=concorrencia,
                repeticao=repeticao,
                cliente_idx=i,
                pergunta=PERGUNTAS_BASE[i % len(PERGUNTAS_BASE)],
            )
            for i in range(concorrencia)
        ]
        # Aguarda todos os clientes simultâneos concluírem suas respostas
        resultados = await asyncio.gather(*tarefas)

    # Tempo total desde o disparo simultâneo até o término do último cliente da rodada
    duracao_rodada = max(time.perf_counter() - t_inicio_rodada, 1e-6)

    # Soma todos os tokens entregues sem erro e divide pela janela total da rodada
    total_tokens = sum(r["tokens_aprox"] for r in resultados if not r["erro"])
    vazao_agregada = total_tokens / duracao_rodada

    # Calcula o TTFT médio entre as requisições bem-sucedidas do patamar
    ttfts = [r["ttft_s"] for r in resultados if r["ttft_s"] is not None and not r["erro"]]
    ttft_medio = statistics.mean(ttfts) if ttfts else 0.0

    print(
        f"c={concorrencia:2d} | rep={repeticao} | "
        f"TTFT médio={ttft_medio:.3f}s | Vazão agregada={vazao_agregada:.2f} tok/s | "
        f"janela={duracao_rodada:.2f}s"
    )
    return resultados


async def main() -> None:
    """
    Orquestra todas as combinações de concorrência e repetições,
    consolidando os resultados brutos em um arquivo CSV para análise e gráficos.
    """
    todos_registros: list[dict] = []

    # Percorre cada nível de carga (1, 2, 4, 8, 16) e executa suas repetições
    for c in NIVEIS_CONCORRENCIA:
        for rep in range(1, REPETICOES + 1):
            registros = await rodar_patamar(c, rep)
            todos_registros.extend(registros)

    # Define o cabeçalho do CSV e grava todas as medições coletadas
    campos = [
        "concorrencia",
        "repeticao",
        "cliente_idx",
        "ttft_s",
        "duracao_s",
        "tokens_aprox",
        "cache_hit",
        "erro",
    ]
    with open(SAIDA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(todos_registros)

    print(f"\n✅ Dados salvos em {SAIDA_CSV}")


if __name__ == "__main__":
    # Inicia o event loop assíncrono do Python para executar o teste de carga
    asyncio.run(main())