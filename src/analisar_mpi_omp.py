#!/usr/bin/env python3
"""
Gera tabela-resumo e gráficos a partir de csv_time/6_conv_mpi_omp.csv.

Uso (na pasta do projeto, onde existe csv_time/):
    python3 analisar_mpi_omp.py

Ou informe outro CSV:
    python3 analisar_mpi_omp.py caminho/arquivo.csv

Requer:
    pip install pandas matplotlib

O speedup é calculado em relação à configuração P=1, T=1 para o mesmo N
e modo. Se essa configuração não existir, usa a configuração com menor
P*T disponível para aquele N e modo.
"""
import argparse
import os
import sys

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLUNAS_OBRIGATORIAS = {
    "N", "P", "T", "modo", "rep", "iters", "tempo_s",
    "comm_s", "energia_J", "potencia_W", "mflops",
}


def carregar_csv(caminho):
    if not os.path.isfile(caminho):
        raise FileNotFoundError(
            f"CSV não encontrado: {caminho}\n"
            "Execute primeiro o programa C++ para gerar csv_time/6_conv_mpi_omp.csv."
        )

    df = pd.read_csv(caminho, sep=";")
    df.columns = df.columns.str.strip()

    faltantes = COLUNAS_OBRIGATORIAS - set(df.columns)
    if faltantes:
        raise ValueError(
            "O CSV não contém as colunas esperadas: " + ", ".join(sorted(faltantes))
        )

    numericas = [
        "N", "P", "T", "modo", "rep", "iters", "tempo_s",
        "comm_s", "energia_J", "potencia_W", "mflops",
    ]
    for coluna in numericas:
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")

    df = df.dropna(subset=["N", "P", "T", "modo", "tempo_s"])
    if df.empty:
        raise ValueError("O CSV não possui linhas válidas.")

    return df


def resumir(df):
    metricas = [
        c for c in [
            "tempo_s", "comm_s", "energia_J", "potencia_W", "mflops",
            "maxdiff_vs_serial", "checksum",
        ] if c in df.columns
    ]

    # Valores -1 indicam que a leitura de energia/RAPL não estava disponível.
    for coluna in ["energia_J", "potencia_W"]:
        if coluna in df.columns:
            df.loc[df[coluna] < 0, coluna] = float("nan")

    chaves = ["N", "P", "T", "modo"]
    medias = df.groupby(chaves)[metricas].mean().reset_index()
    desvios = (
        df.groupby(chaves)["tempo_s"]
        .std()
        .reset_index(name="tempo_std")
    )
    contagens = (
        df.groupby(chaves)
        .size()
        .reset_index(name="repeticoes")
    )

    res = medias.merge(desvios, on=chaves).merge(contagens, on=chaves)

    # Speedup e eficiência: referência P=1, T=1, para o mesmo N e modo.
    bases = {}
    for (n, modo), grupo in res.groupby(["N", "modo"]):
        base = grupo[(grupo["P"] == 1) & (grupo["T"] == 1)]
        if base.empty:
            base = grupo.assign(workers=grupo["P"] * grupo["T"]).sort_values(
                ["workers", "tempo_s"]
            ).head(1)
            print(
                f"Aviso: N={n:g}, modo={modo:g}: não há P=1,T=1; "
                "speedup usa a configuração com menor P*T.",
                file=sys.stderr,
            )
        bases[(n, modo)] = float(base.iloc[0]["tempo_s"])

    res["workers"] = res["P"] * res["T"]
    res["speedup"] = [
        bases[(n, modo)] / tempo if tempo and tempo > 0 else float("nan")
        for n, modo, tempo in zip(res["N"], res["modo"], res["tempo_s"])
    ]
    res["eficiencia"] = res["speedup"] / res["workers"]

    if "mflops" in res and "potencia_W" in res:
        res["MFLOPS_por_W"] = res["mflops"] / res["potencia_W"].where(
            res["potencia_W"] > 0
        )
    if "mflops" in res and "energia_J" in res:
        res["MFLOPS_por_J"] = res["mflops"] / res["energia_J"].where(
            res["energia_J"] > 0
        )
    if "energia_J" in res:
        res["pixels_por_J"] = (
            res["N"] * res["N"]
        ) / res["energia_J"].where(res["energia_J"] > 0)

    return res.sort_values(["N", "modo", "P", "T"]).reset_index(drop=True)


def plot_por_parametro(res, y, titulo, arquivo, eixo_x, legenda_extra,
                       ylabel=None, ideal=False):
    if y not in res.columns:
        return

    plt.figure(figsize=(8, 5))
    agrupadores = ["N", "modo", legenda_extra]
    for (n, modo, extra), grupo in res.groupby(agrupadores):
        grupo = grupo.sort_values(eixo_x)
        plt.plot(
            grupo[eixo_x], grupo[y], "o-",
            label=f"N={n:g}, modo={modo:g}, {legenda_extra}={extra:g}",
        )

    if ideal and eixo_x == "workers":
        valores = sorted(res["workers"].dropna().unique())
        if valores:
            base_x = min(valores)
            plt.plot(valores, [x / base_x for x in valores], "k--", label="ideal")

    plt.xlabel({"P": "Processos MPI (P)", "T": "Threads OpenMP (T)",
                "workers": "P × T (processos × threads)"}.get(eixo_x, eixo_x))
    plt.ylabel(ylabel or y)
    plt.title(titulo)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig(f"graficos/{arquivo}.png", dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Analisa os resultados da convolução híbrida MPI + OpenMP."
    )
    parser.add_argument(
        "csv", nargs="?", default="csv_time/6_conv_mpi_omp.csv",
        help="caminho do CSV (padrão: csv_time/6_conv_mpi_omp.csv)",
    )
    args = parser.parse_args()

    df = carregar_csv(args.csv)
    res = resumir(df)

    os.makedirs("graficos", exist_ok=True)
    saida = "graficos/resumo_mpi_omp.csv"
    res.to_csv(saida, sep=";", index=False, float_format="%.6g")

    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 40)
    print(res.round(4).to_string(index=False))

    # Variação de P mantendo T fixo.
    plot_por_parametro(
        res, "tempo_s", "Tempo por execução × processos MPI",
        "tempo_por_P", "P", "T", ylabel="Tempo por iteração (s)",
    )
    plot_por_parametro(
        res, "speedup", "Speedup × processos MPI",
        "speedup_por_P", "P", "T", ylabel="Speedup",
    )
    plot_por_parametro(
        res, "eficiencia", "Eficiência paralela × processos MPI",
        "eficiencia_por_P", "P", "T", ylabel="Speedup / (P × T)",
    )

    # Variação de T mantendo P fixo.
    plot_por_parametro(
        res, "tempo_s", "Tempo por execução × threads OpenMP",
        "tempo_por_T", "T", "P", ylabel="Tempo por iteração (s)",
    )
    plot_por_parametro(
        res, "speedup", "Speedup × threads OpenMP",
        "speedup_por_T", "T", "P", ylabel="Speedup",
    )
    plot_por_parametro(
        res, "eficiencia", "Eficiência paralela × threads OpenMP",
        "eficiencia_por_T", "T", "P", ylabel="Speedup / (P × T)",
    )

    # Métricas gerais: cada combinação P/T aparece como uma série.
    for coluna, titulo, arquivo, ylabel in [
        ("mflops", "Desempenho (MFLOPS)", "mflops", "MFLOPS"),
        ("comm_s", "Tempo de comunicação por iteração", "comunicacao", "s"),
        ("potencia_W", "Potência média", "potencia", "W"),
        ("energia_J", "Energia por iteração", "energia", "J"),
        ("MFLOPS_por_W", "Eficiência energética", "mflops_watt", "MFLOPS/W"),
        ("MFLOPS_por_J", "Desempenho por energia", "mflops_joule", "MFLOPS/J"),
        ("pixels_por_J", "Pixels processados por Joule", "pixels_joule", "pixels/J"),
    ]:
        if coluna not in res.columns:
            continue
        plt.figure(figsize=(8, 5))
        for (n, modo), grupo_n in res.groupby(["N", "modo"]):
            grupo_n = grupo_n.sort_values("workers")
            plt.plot(
                grupo_n["workers"], grupo_n[coluna], "o-",
                label=f"N={n:g}, modo={modo:g}",
            )
        plt.xlabel("P × T (processos × threads)")
        plt.ylabel(ylabel)
        plt.title(titulo)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(f"graficos/{arquivo}.png", dpi=150)
        plt.close()

    print(f"\nGráficos salvos em graficos/ | tabela em {saida}")


if __name__ == "__main__":
    main()
