#!/usr/bin/env python3
"""Gera tabela-resumo e gráficos a partir de csv_time/5_conv_omp.csv e perfs_omp/*.csv
Uso (dentro de src/):  python3 analisar.py
Requer: pip install pandas matplotlib
"""
import glob, os, re
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("csv_time/5_conv_omp.csv", sep=";")

# ---- lê contadores do perf (valores TOTAIS da região medida = ITERS execuções)
EVENTOS = {
    "l1-dcache-load-misses": "L1_miss",
    "l2_rqsts.all_demand_miss": "L2_miss",
    "mem_load_retired.l3_miss": "L3_miss",
    "cycles": "cycles",
    "instructions": "instr",
}
linhas = []
for f in glob.glob("perfs_omp/N*_T*_r*.csv"):
    N, T, r = map(int, re.findall(r"N(\d+)_T(\d+)_r(\d+)", f)[0])
    d = {"N": N, "threads": T, "rep": r}
    for ln in open(f):
        if ln.startswith("#") or not ln.strip():
            continue
        p = ln.strip().split(",")
        if len(p) < 3:
            continue
        ev = p[2].split(":")[0]
        if ev in EVENTOS:
            try:
                d[EVENTOS[ev]] = float(p[0])
            except ValueError:
                d[EVENTOS[ev]] = float("nan")   # <not counted>/<not supported>
    linhas.append(d)
perf = pd.DataFrame(linhas)
df = df.merge(perf, on=["N", "threads", "rep"], how="left")

# contadores por execução do kernel
for c in ["L1_miss", "L2_miss", "L3_miss", "cycles", "instr"]:
    if c in df:
        df[c] = df[c] / df["iters"]

# ---- média das repetições
med = df.groupby(["N", "threads"]).mean(numeric_only=True).reset_index()
std = df.groupby(["N", "threads"])["tempo_s"].std().reset_index(name="tempo_std")
med = med.merge(std, on=["N", "threads"])

# ---- métricas derivadas
res = []
for N, g in med.groupby("N"):
    g = g.sort_values("threads").copy()
    t1 = g.loc[g.threads == g.threads.min(), "tempo_s"].iloc[0]
    g["speedup"] = t1 / g["tempo_s"]
    g["eficiencia"] = g["speedup"] / g["threads"]
    g["MFLOPS_por_W"] = g["mflops"] / g["potencia_W"]
    g["MFLOPS_por_J"] = g["mflops"] / g["energia_J"]
    g["pixels_por_J"] = (N * N) / g["energia_J"]
    res.append(g)
res = pd.concat(res)
os.makedirs("graficos", exist_ok=True)
res.to_csv("graficos/resumo_omp.csv", sep=";", index=False, float_format="%.6g")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(res.round(4).to_string(index=False))

# ---- gráficos
def plot(y, titulo, arq, ideal=False, ylabel=None):
    plt.figure(figsize=(6.5, 4.2))
    for N, g in res.groupby("N"):
        plt.plot(g.threads, g[y], "o-", label=f"N={N}")
    if ideal:
        mx = res.threads.max()
        plt.plot([1, mx], [1, mx], "k--", label="ideal")
    plt.xlabel("Threads"); plt.ylabel(ylabel or y); plt.title(titulo)
    plt.grid(alpha=.3); plt.legend(); plt.tight_layout()
    plt.savefig(f"graficos/{arq}.png", dpi=150); plt.close()

plot("tempo_s", "Tempo por execução", "tempo", ylabel="s")
plot("speedup", "Speedup", "speedup", ideal=True)
plot("eficiencia", "Eficiência (speedup/threads)", "eficiencia")
plot("mflops", "MFLOPS", "mflops")
plot("potencia_W", "Potência média (pacote CPU)", "potencia", ylabel="W")
plot("energia_J", "Energia por execução", "energia", ylabel="J")
plot("MFLOPS_por_W", "Eficiência energética (MFLOPS/Watt)", "mflops_watt")
plot("pixels_por_J", "Pixels processados por Joule", "pixels_joule")
for c in ["L1_miss", "L2_miss", "L3_miss"]:
    if c in res:
        plot(c, f"Cache misses {c[:2]} por execução", c.lower())
print("\nGráficos salvos em graficos/  |  tabela em graficos/resumo_omp.csv")
