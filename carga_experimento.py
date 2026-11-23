import asyncio
import csv
import statistics
import time
import aiohttp

URL = "http://127.0.0.1:8080/perguntar"
NIVEIS_CONCORRENCIA = [1, 2, 4, 8, 16]
REPETICOES = 3
SAIDA_CSV = "resultados_experimento_m2_m3.csv"

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
    payload = {
        "dominio": "erus-normas",
        "pergunta": f"{pergunta} [run={repeticao}-c={concorrencia}-id={cliente_idx}]",
    }

    t0 = time.perf_counter()
    ttft = None
    chunks = 0
    texto = []

    try:
        async with session.post(URL, json=payload) as resp:
            async for raw_line in resp.content:
                if not raw_line:
                    continue
                agora = time.perf_counter()
                if ttft is None:
                    ttft = agora - t0
                parte = raw_line.decode("utf-8", errors="ignore")
                texto.append(parte)
                chunks += 1
    except Exception as exc:
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

    t1 = time.perf_counter()
    resposta_completa = "".join(texto)
    # Aproximação simples (~4 caracteres por token em português/código)
    tokens_aprox = max(1, len(resposta_completa) // 4) if resposta_completa else 0
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
    timeout = aiohttp.ClientTimeout(total=300)
    t_inicio_rodada = time.perf_counter()

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
        resultados = await asyncio.gather(*tarefas)

    duracao_rodada = max(time.perf_counter() - t_inicio_rodada, 1e-6)
    total_tokens = sum(r["tokens_aprox"] for r in resultados if not r["erro"])
    vazao_agregada = total_tokens / duracao_rodada

    ttfts = [r["ttft_s"] for r in resultados if r["ttft_s"] is not None and not r["erro"]]
    ttft_medio = statistics.mean(ttfts) if ttfts else 0.0

    print(
        f"c={concorrencia:2d} | rep={repeticao} | "
        f"TTFT médio={ttft_medio:.3f}s | Vazão agregada={vazao_agregada:.2f} tok/s | "
        f"janela={duracao_rodada:.2f}s"
    )
    return resultados


async def main() -> None:
    todos_registros: list[dict] = []

    for c in NIVEIS_CONCORRENCIA:
        for rep in range(1, REPETICOES + 1):
            registros = await rodar_patamar(c, rep)
            todos_registros.extend(registros)

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
    asyncio.run(main())
