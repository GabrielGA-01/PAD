#!/bin/bash

# Nome do arquivo de saída
ARQ_SAIDA="hardware_info.txt"

# Garante que a pasta perfs existe
mkdir -p perfs

echo "==================================================" > "$ARQ_SAIDA"
echo "        RELATÓRIO DE HARDWARE DO SISTEMA          " >> "$ARQ_SAIDA"
echo "==================================================" >> "$ARQ_SAIDA"
echo "Data da coleta: $(date)" >> "$ARQ_SAIDA"
echo "" >> "$ARQ_SAIDA"

echo "--------------------------------------------------" >> "$ARQ_SAIDA"
echo "1. MODELO DA CPU" >> "$ARQ_SAIDA"
echo "--------------------------------------------------" >> "$ARQ_SAIDA"
if command -v lscpu &> /dev/null; then
    lscpu | grep "Model name" >> "$ARQ_SAIDA"
    lscpu | grep "Architecture" >> "$ARQ_SAIDA"
    lscpu | grep "CPU(s):" >> "$ARQ_SAIDA"
else
    cat /proc/cpuinfo | grep "model name" | head -n 1 >> "$ARQ_SAIDA"
fi

echo "" >> "$ARQ_SAIDA"
echo "--------------------------------------------------" >> "$ARQ_SAIDA"
echo "2. MEMÓRIA RAM (INTERNA)" >> "$ARQ_SAIDA"
echo "--------------------------------------------------" >> "$ARQ_SAIDA"
free -h >> "$ARQ_SAIDA"

echo "" >> "$ARQ_SAIDA"
echo "--------------------------------------------------" >> "$ARQ_SAIDA"
echo "3. SUPORTE A INSTRUÇÕES VETORIAIS (AVX / AVX2 / AVX-512)" >> "$ARQ_SAIDA"
echo "--------------------------------------------------" >> "$ARQ_SAIDA"

if grep -q "avx" /proc/cpuinfo; then
    echo "[✔] AVX: Suportado" >> "$ARQ_SAIDA"
else
    echo "[X] AVX: Não suportado" >> "$ARQ_SAIDA"
fi

if grep -q "avx2" /proc/cpuinfo; then
    echo "[✔] AVX2: Suportado" >> "$ARQ_SAIDA"
else
    echo "[X] AVX2: Não suportado" >> "$ARQ_SAIDA"
fi

if grep -q "avx512f" /proc/cpuinfo; then
    echo "[✔] AVX-512 (Foundation): Suportado" >> "$ARQ_SAIDA"
else
    echo "[X] AVX-512 (Foundation): Não suportado" >> "$ARQ_SAIDA"
fi

echo "" >> "$ARQ_SAIDA"
echo "--- Lista completa de flags vetoriais encontradas ---" >> "$ARQ_SAIDA"
grep -o -E '(avx|avx2|avx512f|avx512bw|avx512cd|avx512dq|avx512vl|sse4_1|sse4_2)' /proc/cpuinfo | sort -u | tr '\n' ' ' >> "$ARQ_SAIDA"

echo "" >> "$ARQ_SAIDA"
echo "" >> "$ARQ_SAIDA"
echo "--------------------------------------------------" >> "$ARQ_SAIDA"
echo "4. FERRAMENTA DE MEDIÇÃO DE ENERGIA E HARDWARE" >> "$ARQ_SAIDA"
echo "--------------------------------------------------" >> "$ARQ_SAIDA"
echo "Ferramenta utilizada: HWiNFO64" >> "$ARQ_SAIDA"

echo "" >> "$ARQ_SAIDA"
echo "==================================================" >> "$ARQ_SAIDA"

echo "Informações de hardware e energia salvas com sucesso em '$ARQ_SAIDA'!"