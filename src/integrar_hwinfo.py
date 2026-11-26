import pandas as pd

ARQ_BENCH = "csv_time/6_conv_mpi_omp.csv"
ARQ_TS    = "csv_time/timestamps_mpi_omp.csv"
ARQ_HW    = "hwinfo_log.csv"

# 1. Lê o log do HWiNFO64 com utf-8-sig
hw = pd.read_csv(ARQ_HW, encoding="utf-8-sig", low_memory=False)
hw = hw[hw["Time"] != "Time"].copy()

cols_pot = [
    c for c in hw.columns
    if "Potência total da CPU" in c or "CPU Package Power" in c or "Package Power" in c
]
if not cols_pot:
    raise ValueError(f"Coluna de potência não encontrada. Colunas disponíveis: {list(hw.columns)}")

COL_WATTS = cols_pot[0]
print(f"Usando sensor: '{COL_WATTS}'")

hw[COL_WATTS] = pd.to_numeric(hw[COL_WATTS], errors="coerce")
hw["td"] = pd.to_timedelta(hw["Time"].str.strip())

# 2. Lê os resultados do C++ e os timestamps usando encoding='latin1'
df_bench = pd.read_csv(ARQ_BENCH, sep=";", encoding="latin1")
df_ts    = pd.read_csv(ARQ_TS, sep=";", encoding="latin1")

df_ts["td_ini"] = pd.to_timedelta(df_ts["inicio"].str.strip())
df_ts["td_fim"] = pd.to_timedelta(df_ts["fim"].str.strip())

# 3. Cruza cada execução pela janela [inicio, fim]
potencias = []
energias  = []

for idx, row in df_bench.iterrows():
    match = df_ts[
        (df_ts["N"] == row["N"]) &
        (df_ts["P"] == row["P"]) &
        (df_ts["T"] == row["T"]) &
        (df_ts["modo"] == row["modo"]) &
        (df_ts["rep"] == row["rep"])
    ]
    if match.empty:
        potencias.append(-1.0)
        energias.append(-1.0)
        continue

    t0 = match.iloc[0]["td_ini"]
    t1 = match.iloc[0]["td_fim"]

    amostras = hw[(hw["td"] >= t0) & (hw["td"] <= t1)][COL_WATTS].dropna()

    if amostras.empty:
        idx_prox = (hw["td"] - t0).abs().idxmin()
        w_medio = float(hw.loc[idx_prox, COL_WATTS])
    else:
        w_medio = float(amostras.mean())

    e_joules = w_medio * float(row["tempo_s"])
    potencias.append(round(w_medio, 4))
    energias.append(round(e_joules, 6))

df_bench["potencia_W"] = potencias
df_bench["energia_J"]  = energias

df_bench.to_csv(ARQ_BENCH, sep=";", index=False, encoding="utf-8")
print(f"Atualizado com sucesso: {ARQ_BENCH}")
