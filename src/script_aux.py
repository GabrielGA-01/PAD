import os
import glob
import re
import shutil
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURAÇÃO DE FUSO HORÁRIO (TIMEZONE)
# ==========================================
TIMEZONE_OFFSET_HOURS = -3

print("Carregando 'energia.CSV'...")
try:
    df_energia = pd.read_csv("energia.CSV", encoding='latin1', low_memory=False)
except FileNotFoundError:
    print("Erro: O arquivo 'energia.CSV' não foi encontrado na pasta atual.")
    exit()

# Limpeza e tratamento do HWiNFO64
df_energia = df_energia[df_energia['CPU Package Power [W]'] != 'CPU Package Power [W]']
df_energia = df_energia[~df_energia['CPU Package Power [W]'].astype(str).str.contains(r'CPU \[#0\]', na=False)]

df_energia['CPU Package Power [W]'] = pd.to_numeric(df_energia['CPU Package Power [W]'], errors='coerce')
df_energia = df_energia.dropna(subset=['CPU Package Power [W]'])

timestamp_series = pd.to_datetime(
    df_energia['Date'].astype(str).str.strip() + ' ' + df_energia['Time'].astype(str).str.strip(),
    format='mixed',
    dayfirst=True,
    errors='coerce'
)

df_energia['Timestamp'] = timestamp_series
df_energia = df_energia.dropna(subset=['Timestamp'])
df_energia = df_energia.copy()


def calcular_energia_intervalo(t_ini, t_fim, nome_cenario):
    """Calcula energia total (J) e potência média (W) aplicando o ajuste de fuso horário."""
    t_ini_dt = pd.to_datetime(t_ini, unit='s') + pd.Timedelta(hours=TIMEZONE_OFFSET_HOURS)
    t_fim_dt = pd.to_datetime(t_fim, unit='s') + pd.Timedelta(hours=TIMEZONE_OFFSET_HOURS)

    df_filtrado = df_energia[(df_energia['Timestamp'] >= t_ini_dt) & (df_energia['Timestamp'] <= t_fim_dt)]

    if df_filtrado.empty:
        t_ini_tol = t_ini_dt - pd.Timedelta(seconds=2)
        t_fim_tol = t_fim_dt + pd.Timedelta(seconds=2)
        df_filtrado = df_energia[(df_energia['Timestamp'] >= t_ini_tol) & (df_energia['Timestamp'] <= t_fim_tol)]

        if df_filtrado.empty:
            print(f"[Aviso] Sem amostras de energia para {nome_cenario}")
            print(f"        Procurando entre: {t_ini_dt} e {t_fim_dt} (Ajustado por offset {TIMEZONE_OFFSET_HOURS}h)")
            return 0.0, 0.0

    df_f = df_filtrado.copy()
    df_f['Delta_Tempo'] = df_f['Timestamp'].diff().dt.total_seconds().fillna(0)

    if len(df_f) == 1 or df_f['Delta_Tempo'].sum() == 0:
        duracao_teste = t_fim - t_ini
        potencia_media = df_f['CPU Package Power [W]'].mean()
        energia_total = potencia_media * duracao_teste
        return (potencia_media if not pd.isna(potencia_media) else 0.0,
                energia_total if not pd.isna(energia_total) else 0.0)

    df_f['Energia_Parcial_J'] = df_f['CPU Package Power [W]'] * df_f['Delta_Tempo']
    energia_total = df_f['Energia_Parcial_J'].sum()
    tempo_total = df_f['Delta_Tempo'].sum()
    potencia_media = energia_total / tempo_total if tempo_total > 0 else 0.0

    return potencia_media, energia_total


def extrair_perf_stats(perf_path):
    stats = {'cycles': 0, 'instructions': 0, 'l1_misses': 0, 'l2_misses': 0, 'l3_misses': 0}
    if not os.path.exists(perf_path):
        return stats
    with open(perf_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            parts = line.strip().split()
            if not parts: continue
            val_str = parts[0].replace('.', '').replace(',', '')
            if not val_str.isdigit(): continue
            val = int(val_str)
            line_lower = line.lower()
            if 'cycles' in line_lower and 'ref' not in line_lower: stats['cycles'] = val
            elif 'instructions' in line_lower: stats['instructions'] = val
            elif 'l1-dcache-load-misses' in line_lower: stats['l1_misses'] = val
            elif 'l2' in line_lower and 'miss' in line_lower: stats['l2_misses'] = val
            elif 'l3' in line_lower and 'miss' in line_lower: stats['l3_misses'] = val
    return stats


def seguro_div(a, b):
    return a / b if b > 0 else 0.0


csv_files = glob.glob("csv_time/*.csv")
if not csv_files:
    print("Nenhum arquivo CSV encontrado na pasta 'csv_time/'.")
    exit()

if os.path.exists("relatorios_metricas"):
    shutil.rmtree("relatorios_metricas")
os.makedirs("relatorios_metricas")
print("\nProcessando cenários com correção de fuso horário...")

resultados = {}

for csv_path in csv_files:
    nome_arquivo = os.path.splitext(os.path.basename(csv_path))[0]
    correspondencia_iteracao = re.fullmatch(r"(.*)_I(\d+)_T(\d+)", nome_arquivo)
    if correspondencia_iteracao:
        nome_algoritmo = correspondencia_iteracao.group(1)
        iteracao_arquivo = int(correspondencia_iteracao.group(2))
        threads_arquivo = int(correspondencia_iteracao.group(3))
    else:
        correspondencia_iteracao = re.fullmatch(r"(.*)_I(\d+)", nome_arquivo)
        if correspondencia_iteracao:
            nome_algoritmo = correspondencia_iteracao.group(1)
            iteracao_arquivo = int(correspondencia_iteracao.group(2))
        else:
            nome_algoritmo = nome_arquivo
            iteracao_arquivo = 1
        threads_arquivo = None

    df_tempo = pd.read_csv(csv_path, sep=';')

    # Formato novo (3A/3B): t0..t3 + tempos por fase. Formato antigo: inicio/fim/tempo_s.
    formato_novo = 'tempo_stencil_s' in df_tempo.columns

    for _, row in df_tempo.iterrows():
        iteracao = int(row['iteracao']) if 'iteracao' in df_tempo.columns and pd.notna(row['iteracao']) else iteracao_arquivo
        threads = int(row['threads']) if 'threads' in df_tempo.columns and pd.notna(row['threads']) else threads_arquivo
        N = int(row['N'])
        bloco = int(row['bloco']) if 'bloco' in df_tempo.columns else None

        if formato_novo:
            t0 = float(row['t0_epoch'])
            t1 = float(row['t1_epoch'])
            t2 = float(row['t2_epoch'])
            t3 = float(row['t3_epoch'])
            tempo_divisao = float(row['tempo_divisao_s'])
            tempo_stencil = float(row['tempo_stencil_s'])
            tempo_juncao = float(row['tempo_juncao_s'])
            tempo_total = float(row['tempo_total_s'])
            tempo_s = tempo_stencil          # tempo de referência para MFLOPS = só o stencil
            t_ini_ref, t_fim_ref = t1, t2    # intervalo de referência para energia = só o stencil
        else:
            t_ini_ref = float(row['inicio_epoch'])
            t_fim_ref = float(row['fim_epoch'])
            tempo_s = float(row['tempo_s'])

        perf_base = f"{nome_algoritmo}_N{N}_B{bloco}" if bloco is not None else f"{nome_algoritmo}_N{N}"
        if threads is not None:
            perf_path = os.path.join("perfs", f"{perf_base}_I{iteracao}_T{threads}.txt")
        else:
            perf_path_iteracao = os.path.join("perfs", f"{perf_base}_I{iteracao}.txt")
            perf_path_legado = os.path.join("perfs", f"{perf_base}.txt")
            perf_path = perf_path_iteracao if os.path.exists(perf_path_iteracao) else perf_path_legado
        perf_data = extrair_perf_stats(perf_path)

        total_pixels = (N - 2) * (N - 2)
        total_flops = total_pixels * 18.0
        mflops = seguro_div(total_flops / 1e6, tempo_s)
        cpi = seguro_div(perf_data['cycles'], perf_data['instructions'])

        cenario_nome = f"{nome_algoritmo} (N={N}{f', B={bloco}' if bloco else ''}{f', threads={threads}' if threads is not None else ''})"

        # Energia do stencil (ou do intervalo único, no formato antigo)
        potencia_media, energia_total = calcular_energia_intervalo(t_ini_ref, t_fim_ref, cenario_nome + " [stencil]")
        mflops_por_watt = seguro_div(mflops, potencia_media)
        mflops_por_joule = seguro_div(mflops, energia_total)

        metricas = {
            'tempo_s': tempo_s,
            'mflops': mflops,
            'cpi': cpi,
            'cycles': perf_data['cycles'],
            'instructions': perf_data['instructions'],
            'l1_misses': perf_data['l1_misses'],
            'l2_misses': perf_data['l2_misses'],
            'l3_misses': perf_data['l3_misses'],
            'potencia_media': potencia_media,
            'energia_total': energia_total,
            'mflops_por_watt': mflops_por_watt,
            'mflops_por_joule': mflops_por_joule,
        }

        if formato_novo:
            # Energia das demais fases e do total (divisão + stencil + junção)
            pot_div, en_div = calcular_energia_intervalo(t0, t1, cenario_nome + " [divisão]")
            pot_jun, en_jun = calcular_energia_intervalo(t2, t3, cenario_nome + " [junção]")
            pot_tot, en_tot = calcular_energia_intervalo(t0, t3, cenario_nome + " [total]")
            mflops_total = seguro_div(total_flops / 1e6, tempo_total)
            mflops_por_watt_tot = seguro_div(mflops_total, pot_tot)
            mflops_por_joule_tot = seguro_div(mflops_total, en_tot)

            metricas.update({
                'tempo_divisao': tempo_divisao,
                'tempo_stencil': tempo_stencil,
                'tempo_juncao': tempo_juncao,
                'tempo_total': tempo_total,
                'mflops_total': mflops_total,
                'pot_divisao': pot_div,
                'energia_divisao': en_div,
                'pot_juncao': pot_jun,
                'energia_juncao': en_jun,
                'pot_total': pot_tot,
                'energia_total_fases': en_tot,
                'mflops_por_watt_total': mflops_por_watt_tot,
                'mflops_por_joule_total': mflops_por_joule_tot,
            })

        chave = (nome_algoritmo, N, bloco, threads)
        resultados.setdefault(chave, []).append(metricas)

for (nome_algoritmo, N, bloco, threads), execucoes in resultados.items():
    metricas = {
        nome: float(np.mean([execucao[nome] for execucao in execucoes]))
        for nome in execucoes[0]
    }
    formato_novo = 'tempo_total' in metricas

    sufixo_bloco = f"_B{bloco}" if bloco is not None else ""
    sufixo_threads = f"_T{threads}" if threads is not None else ""
    rel_nome = f"relatorios_metricas/{nome_algoritmo}_N{N}{sufixo_bloco}{sufixo_threads}.txt"
    cenario_str = f"Algoritmo: {nome_algoritmo} | N = {N}" + (f" | Bloco = {bloco}" if bloco is not None else "")
    if threads is not None:
        cenario_str += f" | Threads = {threads}"

    with open(rel_nome, "w", encoding="utf-8") as rel:
        rel.write("==================================================\n")
        rel.write("        RELATÓRIO DE DESEMPENHO E ENERGIA         \n")
        rel.write("==================================================\n")
        rel.write(f"{cenario_str}\n")
        rel.write(f"Execuções consideradas na média: {len(execucoes)}\n\n")

        rel.write("--- 1. DESEMPENHO TEMPORAL E COMPUTACIONAL ---\n")
        if formato_novo:
            rel.write(f"Tempo da Divisão em Blocos: {metricas['tempo_divisao']:.6f} s\n")
            rel.write(f"Tempo da Aplicação do Stencil: {metricas['tempo_stencil']:.6f} s\n")
            rel.write(f"Tempo da Junção da Matriz: {metricas['tempo_juncao']:.6f} s\n")
            rel.write(f"Tempo Total (Wall-Clock): {metricas['tempo_total']:.6f} s\n")
            rel.write(f"Desempenho Computacional (só stencil): {metricas['mflops']:.2f} MFLOPS\n")
            rel.write(f"Desempenho Computacional (total): {metricas['mflops_total']:.2f} MFLOPS\n")
        else:
            rel.write(f"Tempo de Execução (Wall-Clock): {metricas['tempo_s']:.6f} s\n")
            rel.write(f"Desempenho Computacional: {metricas['mflops']:.2f} MFLOPS\n")
        rel.write(f"CPI (Cycles Per Instruction): {metricas['cpi']:.4f}\n\n")

        rel.write("--- 2. HARDWARE E HIERARQUIA DE CACHE (perf) ---\n")
        rel.write(f"Ciclos de CPU (Cycles): {metricas['cycles']:.2f}\n")
        rel.write(f"Instruções Executadas (Instructions): {metricas['instructions']:.2f}\n")
        rel.write(f"L1 Data Cache Misses: {metricas['l1_misses']:.2f}\n")
        rel.write(f"L2 Cache Misses: {metricas['l2_misses']:.2f}\n")
        rel.write(f"L3 Cache Misses: {metricas['l3_misses']:.2f}\n\n")

        rel.write("--- 3. CONSUMO DE ENERGIA (HWiNFO64) ---\n")
        if formato_novo:
            rel.write(f"[Stencil]  Potência Média: {metricas['potencia_media']:.2f} W | Energia: {metricas['energia_total']:.4f} J\n")
            rel.write(f"[Divisão]  Potência Média: {metricas['pot_divisao']:.2f} W | Energia: {metricas['energia_divisao']:.4f} J\n")
            rel.write(f"[Junção]   Potência Média: {metricas['pot_juncao']:.2f} W | Energia: {metricas['energia_juncao']:.4f} J\n")
            rel.write(f"[Total]    Potência Média: {metricas['pot_total']:.2f} W | Energia: {metricas['energia_total_fases']:.4f} J\n\n")
        else:
            rel.write(f"Potência Média Consumida: {metricas['potencia_media']:.2f} W\n")
            rel.write(f"Energia Total Consumida (CPU Package): {metricas['energia_total']:.4f} Joules\n\n")

        rel.write("--- 4. EFICIÊNCIA ENERGÉTICA ---\n")
        if formato_novo:
            rel.write(f"Stencil - MFLOPS/Watt: {metricas['mflops_por_watt']:.4f}\n")
            rel.write(f"Stencil - MFLOPS/Joule: {metricas['mflops_por_joule']:.4f}\n")
            rel.write(f"Total   - MFLOPS/Watt: {metricas['mflops_por_watt_total']:.4f}\n")
            rel.write(f"Total   - MFLOPS/Joule: {metricas['mflops_por_joule_total']:.4f}\n")
        else:
            rel.write(f"Eficiência por Potência (MFLOPS/Watt): {metricas['mflops_por_watt']:.4f}\n")
            rel.write(f"Eficiência por Energia (MFLOPS/Joule): {metricas['mflops_por_joule']:.4f}\n")
        rel.write("==================================================\n")

print("\nProcessamento concluído! Verifique a pasta 'relatorios_metricas/'.")