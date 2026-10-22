import pandas as pd

# 1. Lê os horários gravados pelo C++
try:
    with open("exec_tempo.txt", "r") as f:
        linhas = f.readlines()
        str_inicio = linhas[0].strip()
        str_fim = linhas[1].strip()
    print(f"Início do Teste: {str_inicio}")
    print(f"Fim do Teste:    {str_fim}")
except FileNotFoundError:
    print("Erro: O arquivo 'exec_tempo.txt' não foi encontrado. Execute o programa C++ primeiro.")
    exit()

# Converte para o formato de data/hora do Pandas
t_inicio = pd.to_datetime(str_inicio, format='%d.%m.%Y %H:%M:%S.%f')
t_fim = pd.to_datetime(str_fim, format='%d.%m.%Y %H:%M:%S.%f')

# 2. Carrega o log de energia do HWiNFO64
df = pd.read_csv("Log_energia.CSV", encoding='latin1')

# Remove linhas indesejadas onde o valor da potência é o próprio cabeçalho ou metadados
df = df[df['CPU Package Power [W]'] != 'CPU Package Power [W]']
df = df[~df['CPU Package Power [W]'].astype(str).str.contains('CPU \[\#0\]', na=False)]

# Força a conversão para numérico, transformando eventuais textos restantes em NaN e os removendo
df['CPU Package Power [W]'] = pd.to_numeric(df['CPU Package Power [W]'], errors='coerce')
df = df.dropna(subset=['CPU Package Power [W]'])

# 3. Cria a coluna de timestamp com base no HWiNFO
df['Timestamp'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d.%m.%Y %H:%M:%S.%f')

# 4. Filtra estritamente o intervalo de tempo em que o C++ executou o teste
df_filtrado = df[(df['Timestamp'] >= t_inicio) & (df['Timestamp'] <= t_fim)]

if df_filtrado.empty:
    print("\nAviso: Nenhuma amostra foi encontrada dentro do intervalo exato. Verifique se os horários do relógio do sistema estão sincronizados.")
else:
    # 5. Calcula a energia gasta no período filtrado
    df_filtrado = df_filtrado.copy()
    df_filtrado['Delta_Tempo'] = df_filtrado['Timestamp'].diff().dt.total_seconds().fillna(0)
    df_filtrado['Energia_Parcial_J'] = df_filtrado['CPU Package Power [W]'] * df_filtrado['Delta_Tempo']

    energia_total = df_filtrado['Energia_Parcial_J'].sum()
    tempo_total = df_filtrado['Delta_Tempo'].sum()
    potencia_media = energia_total / tempo_total if tempo_total > 0 else 0

    print(f"\n--- RESULTADO DA MEDIÇÃO DE ENERGIA ---")
    print(f"Duração do Teste: {tempo_total:.3f} segundos")
    print(f"Energia Consumida (CPU Package): {energia_total:.2f} Joules")
    print(f"Potência Média durante o teste: {potencia_media:.2f} Watts")