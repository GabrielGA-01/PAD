import os
import glob
import pandas as pd
import numpy as np

# ==========================================
# CONFIGURAÇÃO DE FUSO HORÁRIO (TIMEZONE)
# ==========================================
# Se o HWiNFO está em horário local (ex: UTC-3) e o Linux/C++ em UTC,
# ajuste o offset aqui em horas (ex: -3 para o Brasil, 0 se estiverem no mesmo fuso).
TIMEZONE_OFFSET_HOURS = -3  

print("Carregando 'energia.CSV'...")
try:
    df_energia = pd.read_csv("energia.CSV", encoding='latin1')
except FileNotFoundError:
    print("Erro: O arquivo 'energia.CSV' não foi encontrado na pasta atual.")
    exit()

# Limpeza e tratamento do HWiNFO64
df_energia = df_energia[df_energia['CPU Package Power [W]'] != 'CPU Package Power [W]']
df_energia = df_energia[~df_energia['CPU Package Power [W]'].astype(str).str.contains(r'CPU \[#0\]', na=False)]

df_energia['CPU Package Power [W]'] = pd.to_numeric(df_energia['CPU Package Power [W]'], errors='coerce')
df_energia = df_energia.dropna(subset=['CPU Package Power [W]'])

# Combina Data e Hora do HWiNFO
df_energia['Timestamp'] = pd.to_datetime(
    df_energia['Date'].astype(str).str.strip() + ' ' + df_energia['Time'].astype(str).str.strip(), 
    format='mixed', 
    dayfirst=True, 
    errors='coerce'
)
df_energia = df_energia.dropna(subset=['Timestamp'])

def calcular_energia_intervalo(t_ini, t_fim, nome_cenario):
    """Calcula energia total (J) e potência média (W) aplicando o ajuste de fuso horário."""
    # Converte o epoch do C++ e aplica o offset de fuso horário para alinhar com o HWiNFO
    t_ini_dt = pd.to_datetime(t_ini, unit='s') + pd.Timedelta(hours=TIMEZONE_OFFSET_HOURS)
    t_fim_dt = pd.to_datetime(t_fim, unit='s') + pd.Timedelta(hours=TIMEZONE_OFFSET_HOURS)
    
    # Filtra o dataframe de energia dentro da janela ajustada
    df_filtrado = df_energia[(df_energia['Timestamp'] >= t_ini_dt) & (df_energia['Timestamp'] <= t_fim_dt)]
    
    # Se cair numa lacuna de amostragem exata, aplica tolerância de 2 segundos
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

csv_files = glob.glob("csv_time/*.csv")
if not csv_files:
    print("Nenhum arquivo CSV encontrado na pasta 'csv_time/'.")
    exit()

os.makedirs("relatorios_metricas", exist_ok=True)
print("\nProcessando cenários com correção de fuso horário...")

for csv_path in csv_files:
    nome_algoritmo = os.path.splitext(os.path.basename(csv_path))[0]
    df_tempo = pd.read_csv(csv_path, sep=';')
    
    for index, row in df_tempo.iterrows():
        N = int(row['N'])
        t0 = float(row['inicio_epoch'])
        t1 = float(row['fim_epoch'])
        tempo_s = float(row['tempo_s'])
        bloco = int(row['bloco']) if 'bloco' in df_tempo.columns else None
        
        perf_filename = f"{nome_algoritmo}_N{N}_B{bloco}.txt" if bloco is not None else f"{nome_algoritmo}_N{N}.txt"
        perf_path = os.path.join("perfs", perf_filename)
        perf_data = extrair_perf_stats(perf_path)
        
        total_pixels = (N - 2) * (N - 2)
        total_flops = total_pixels * 18.0
        mflops = (total_flops / 1e6) / tempo_s if tempo_s > 0 else 0.0
        cpi = (perf_data['cycles'] / perf_data['instructions']) if perf_data['instructions'] > 0 else 0.0
        
        cenario_nome = f"{nome_algoritmo} (N={N}{f', B={bloco}' if bloco else ''})"
        potencia_media, energia_total = calcular_energia_intervalo(t0, t1, cenario_nome)
        
        mflops_por_watt = (mflops / potencia_media) if potencia_media > 0 else 0.0
        mflops_por_joule = (mflops / energia_total) if energia_total > 0 else 0.0
        
        rel_nome = f"relatorios_metricas/{nome_algoritmo}_N{N}_B{bloco}.txt" if bloco is not None else f"relatorios_metricas/{nome_algoritmo}_N{N}.txt"
        cenario_str = f"Algoritmo: {nome_algoritmo} | N = {N}" + (f" | Bloco = {bloco}" if bloco is not None else "")
            
        with open(rel_nome, "w", encoding="utf-8") as rel:
            rel.write("==================================================\n")
            rel.write(f"        RELATÓRIO DE DESEMPENHO E ENERGIA         \n")
            rel.write("==================================================\n")
            rel.write(f"{cenario_str}\n\n")
            rel.write("--- 1. DESEMPENHO TEMPORAL E COMPUTACIONAL ---\n")
            rel.write(f"Tempo de Execução (Wall-Clock): {tempo_s:.6f} s\n")
            rel.write(f"Desempenho Computacional: {mflops:.2f} MFLOPS\n")
            rel.write(f"CPI (Cycles Per Instruction): {cpi:.4f}\n\n")
            rel.write("--- 2. HARDWARE E HIERARQUIA DE CACHE (perf) ---\n")
            rel.write(f5 := f"Ciclos de CPU (Cycles): {perf_data['cycles']}\n")
            rel.write(f"Instruções Executadas (Instructions): {perf_data['instructions']}\n")
            rel.write(f"L1 Data Cache Misses: {perf_data['l1_misses']}\n")
            rel.write(f"L2 Cache Misses: {perf_data['l2_misses']}\n")
            rel.write(f"L3 Cache Misses: {perf_data['l3_misses']}\n\n")
            rel.write("--- 3. CONSUMO DE ENERGIA (HWiNFO64) ---\n")
            rel.write(f"Potência Média Consumida: {potencia_media:.2f} W\n")
            rel.write(f"Energia Total Consumida (CPU Package): {energia_total:.4f} Joules\n\n")
            rel.write("--- 4. EFICIÊNCIA ENERGÉTICA ---\n")
            rel.write(f"Eficiência por Potência (MFLOPS/Watt): {mflops_por_watt:.4f}\n")
            rel.write(f"Eficiência por Energia (MFLOPS/Joule): {mflops_por_joule:.4f}\n")
            rel.write("==================================================\n")

print("\nProcessamento concluído! Verifique a pasta 'relatorios_metricas/'.")