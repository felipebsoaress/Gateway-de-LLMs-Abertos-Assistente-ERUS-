import argparse
import asyncio
import csv
import statistics
import time
import aiohttp

URL_PERGUNTAR = "http://127.0.0.1:8080/perguntar"
URL_LIMPAR_CACHE = "http://127.0.0.1:8080/cache"

NIVEIS_CONCORRENCIA = [1, 2, 4, 8, 16]
CSV_CARGA = "resultados_carga_sem_cache.csv"
CSV_CACHE = "resultados_cache_semantico.csv"

PERGUNTAS_CARGA = [
    "Quais são os projetos desenvolvidos pela equipe ERUS?",
    "Como funciona a participação de membros na equipe ERUS?",
    "Quais são as principais conquistas da ERUS?",
    "Qual é o objetivo do projeto de extensão ERUS na UFES?",
    "Como é organizada a estrutura de trabalho da equipe ERUS?",
    "Quais áreas de engenharia fazem parte da ERUS?",
    "Onde a equipe ERUS está sediada na UFES?",
    "Quais competições ou eventos a ERUS participa?",
]

# Conjunto controlado para avaliar taxa de acerto (hit rate) e economia de tempo do cache
PERGUNTAS_CACHE = [
    # Fase 1: 5 perguntas inéditas (devem dar MISS e popular o cache)
    ("inedita", "Qual é o objetivo principal da equipe ERUS na UFES?"),
    ("inedita", "Onde a equipe ERUS está sediada?"),
    ("inedita", "Quais são os principais projetos da equipe ERUS?"),
    ("inedita", "Quais conquistas ou competições a ERUS registra no site?"),
    ("inedita", "Como estudantes podem participar da equipe ERUS?"),
    # Fase 2: 5 repetições exatas (devem dar HIT)
    ("exata", "Qual é o objetivo principal da equipe ERUS na UFES?"),
    ("exata", "Onde a equipe ERUS está sediada?"),
    ("exata", "Quais são os principais projetos da equipe ERUS?"),
    ("exata", "Quais conquistas ou competições a ERUS registra no site?"),
    ("exata", "Como estudantes podem participar da equipe ERUS?"),
    # Fase 3: 5 paráfrases próximas (testam a generalização semântica do limiar 0.95)
    ("parafrase", "Qual o principal objetivo da equipe ERUS na UFES?"),
    ("parafrase", "Onde fica localizada a sede da equipe ERUS?"),
    ("parafrase", "Quais projetos principais são desenvolvidos pela ERUS?"),
    ("parafrase", "Que competições e conquistas aparecem no site da ERUS?"),
    ("parafrase", "De que maneira um aluno pode participar da equipe ERUS?"),
]


async def enviar_requisicao(
    session: aiohttp.ClientSession,
    pergunta: str,
    dominio: str,
    usar_cache: bool,
    metadados: dict,
) -> dict:
    payload = {
        "dominio": dominio,
        "pergunta": pergunta,
        "usar_cache": usar_cache,
    }

    t0 = time.perf_counter()
    ttft = None
    texto = []

    try:
        async with session.post(URL_PERGUNTAR, json=payload) as resp:
            async for raw_chunk in resp.content.iter_any():
                if not raw_chunk:
                    continue
                agora = time.perf_counter()
                if ttft is None:
                    ttft = agora - t0
                texto.append(raw_chunk.decode("utf-8", errors="ignore"))
    except Exception as exc:
        return {
            **metadados,
            "usar_cache": usar_cache,
            "ttft_s": None,
            "duracao_s": round(time.perf_counter() - t0, 4),
            "tokens_aprox": 0,
            "cache_hit": False,
            "erro": str(exc),
        }

    t1 = time.perf_counter()
    resposta_completa = "".join(texto)
    tokens_aprox = max(1, len(resposta_completa) // 4) if resposta_completa else 0
    cache_hit = "[Cache Rápido]" in resposta_completa

    return {
        **metadados,
        "usar_cache": usar_cache,
        "ttft_s": round(ttft if ttft is not None else (t1 - t0), 4),
        "duracao_s": round(t1 - t0, 4),
        "tokens_aprox": tokens_aprox,
        "cache_hit": cache_hit,
        "erro": "",
    }


async def executar_teste_carga(repeticoes: int, dominio: str) -> None:
    print(f"\n🚀 Iniciando Experimento de Carga SEM CACHE (domínio={dominio}, repetições={repeticoes})")
    timeout = aiohttp.ClientTimeout(total=600)

    # Aquecimento: garante que o modelo já esteja carregado na memória antes de cronometrar
    async with aiohttp.ClientSession(timeout=timeout) as session:
        print("🔥 Executando 1 requisição de aquecimento (não contabilizada)...")
        await enviar_requisicao(
            session=session,
            pergunta="O que é a ERUS?",
            dominio=dominio,
            usar_cache=False,
            metadados={"concorrencia": 0, "repeticao": 0, "cliente_idx": 0},
        )

    todos_registros: list[dict] = []
    resumo_rodadas: dict[int, list[tuple[float, float]]] = {c: [] for c in NIVEIS_CONCORRENCIA}

    for c in NIVEIS_CONCORRENCIA:
        for rep in range(1, repeticoes + 1):
            t_inicio_rodada = time.perf_counter()
            async with aiohttp.ClientSession(timeout=timeout) as session:
                tarefas = [
                    enviar_requisicao(
                        session=session,
                        pergunta=PERGUNTAS_CARGA[i % len(PERGUNTAS_CARGA)],
                        dominio=dominio,
                        usar_cache=False,
                        metadados={"concorrencia": c, "repeticao": rep, "cliente_idx": i},
                    )
                    for i in range(c)
                ]
                resultados = await asyncio.gather(*tarefas)

            duracao_rodada = max(time.perf_counter() - t_inicio_rodada, 1e-6)
            validos = [r for r in resultados if not r["erro"] and r["ttft_s"] is not None]
            total_tokens = sum(r["tokens_aprox"] for r in validos)
            vazao_agregada = total_tokens / duracao_rodada
            ttft_medio = statistics.mean(r["ttft_s"] for r in validos) if validos else 0.0

            resumo_rodadas[c].append((ttft_medio, vazao_agregada))
            todos_registros.extend(resultados)

            print(
                f"c={c:2d} | rep={rep} | TTFT médio={ttft_medio:.3f}s | "
                f"Vazão agregada={vazao_agregada:.2f} tok/s | janela={duracao_rodada:.2f}s"
            )

    campos = [
        "concorrencia", "repeticao", "cliente_idx", "usar_cache",
        "ttft_s", "duracao_s", "tokens_aprox", "cache_hit", "erro"
    ]
    with open(CSV_CARGA, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(todos_registros)

    print("\n📊 Resumo consolidado por concorrência (média ± desvio padrão entre repetições):")
    for c in NIVEIS_CONCORRENCIA:
        ttfts = [x[0] for x in resumo_rodadas[c]]
        vazoes = [x[1] for x in resumo_rodadas[c]]
        ttft_std = statistics.stdev(ttfts) if len(ttfts) > 1 else 0.0
        vazao_std = statistics.stdev(vazoes) if len(vazoes) > 1 else 0.0
        print(
            f"  c={c:2d} -> TTFT: {statistics.mean(ttfts):.3f} ± {ttft_std:.3f}s | "
            f"Vazão: {statistics.mean(vazoes):.2f} ± {vazao_std:.2f} tok/s"
        )
    print(f"✅ Dados salvos em '{CSV_CARGA}'.")


async def executar_teste_cache(dominio: str) -> None:
    print(f"\n⚡ Iniciando Avaliação Dedicada de Cache Semântico (domínio={dominio})")
    timeout = aiohttp.ClientTimeout(total=300)
    registros: list[dict] = []

    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Zera o cache para garantir reprodutibilidade
        async with session.delete(URL_LIMPAR_CACHE) as resp:
            if resp.status != 200:
                print(f"⚠️ Aviso ao limpar cache: HTTP {resp.status}")

        for idx, (fase, pergunta) in enumerate(PERGUNTAS_CACHE, start=1):
            reg = await enviar_requisicao(
                session=session,
                pergunta=pergunta,
                dominio=dominio,
                usar_cache=True,
                metadados={"ordem": idx, "fase": fase, "pergunta": pergunta},
            )
            registros.append(reg)
            status = "HIT " if reg["cache_hit"] else "MISS"
            print(
                f"[{idx:02d}/{len(PERGUNTAS_CACHE)}] {fase:9s} | {status} | "
                f"TTFT={reg['ttft_s']}s | Duração={reg['duracao_s']}s"
            )

    campos = [
        "ordem", "fase", "pergunta", "usar_cache",
        "ttft_s", "duracao_s", "tokens_aprox", "cache_hit", "erro"
    ]
    with open(CSV_CACHE, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=campos)
        writer.writeheader()
        writer.writerows(registros)

    misses = [r for r in registros if not r["cache_hit"] and not r["erro"]]
    hits = [r for r in registros if r["cache_hit"] and not r["erro"]]
    exatas = [r for r in registros if r["fase"] == "exata"]
    parafrases = [r for r in registros if r["fase"] == "parafrase"]

    print("\n📊 Resumo do Cache Semântico:")
    print(f"  Taxa de acerto (exatas):     {sum(r['cache_hit'] for r in exatas)}/{len(exatas)}")
    print(f"  Taxa de acerto (paráfrases): {sum(r['cache_hit'] for r in parafrases)}/{len(parafrases)}")
    if misses and hits:
        ttft_miss = statistics.mean(r["ttft_s"] for r in misses)
        ttft_hit = statistics.mean(r["ttft_s"] for r in hits)
        dur_miss = statistics.mean(r["duracao_s"] for r in misses)
        dur_hit = statistics.mean(r["duracao_s"] for r in hits)
        economia_ttft = (1 - ttft_hit / ttft_miss) * 100
        economia_dur = (1 - dur_hit / dur_miss) * 100
        print(f"  TTFT médio:    MISS={ttft_miss:.3f}s -> HIT={ttft_hit:.3f}s (economia de {economia_ttft:.1f}%)")
        print(f"  Duração média: MISS={dur_miss:.3f}s -> HIT={dur_hit:.3f}s (economia de {economia_dur:.1f}%)")
    print(f"✅ Dados salvos em '{CSV_CACHE}'.")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Experimentos de carga e cache do Gateway ERUS")
    parser.add_argument("--modo", choices=["carga", "cache", "ambos"], default="carga")
    parser.add_argument("--repeticoes", type=int, default=3)
    parser.add_argument("--dominio", default="erus-normas")
    args = parser.parse_args()

    if args.modo in ("carga", "ambos"):
        await executar_teste_carga(repeticoes=args.repeticoes, dominio=args.dominio)
    if args.modo in ("cache", "ambos"):
        await executar_teste_cache(dominio=args.dominio)


if __name__ == "__main__":
    asyncio.run(main())