#!/bin/bash

# Configurações de tamanhos e blocos
TAMANHOS=(8096 12288 16384)
BLOCOS=(8 16 32 64)

echo "=================================================="
echo "Removendo pastas de resultados e binários antigos..."
echo "=================================================="
rm -rf objects perfs csv_time

echo "=================================================="
echo "Criando novas pastas de diretórios..."
echo "=================================================="
mkdir -p objects
mkdir -p perfs
mkdir -p csv_time

echo "=================================================="
echo "Iniciando Compilação de Todos os Programas..."
echo "=================================================="

OPENCV_FLAGS=$(pkg-config --cflags --libs opencv4 2>/dev/null || pkg-config --cflags --libs opencv)

# Compilação de todos os algoritmos
g++ 1A_conv_linhas.cpp -o objects/1A_conv_linhas $OPENCV_FLAGS
g++ 1B_conv_colunas.cpp -o objects/1B_conv_colunas $OPENCV_FLAGS
g++ -O3 -mavx2 -fopenmp-simd 2_conv_simd.cpp -o objects/2_conv_simd $OPENCV_FLAGS
g++ 3A_conv_blocos.cpp -o objects/3A_conv_blocos $OPENCV_FLAGS
g++ -O3 -mavx2 -fopenmp-simd 3B_conv_blocos.cpp -o objects/3B_conv_blocos $OPENCV_FLAGS
g++ -O3 -mavx2 -fopenmp-simd 4_conv_opencv.cpp -o objects/4_conv_opencv $OPENCV_FLAGS

echo "Compilação concluída com sucesso!"
echo "=================================================="
echo "Executando Benchmarks (perf + tempos)..."
echo "=================================================="

PERF_EVENTS="cycles,instructions,l1-dcache-loads,l1-dcache-load-misses,l2_rqsts.all_demand_data_rd,l2_rqsts.all_demand_miss,mem_load_retired.l3_hit,mem_load_retired.l3_miss"

for N in "${TAMANHOS[@]}"; do
    echo "===================================="
    echo "--- Testando Tamanho N = $N ---"
    echo "===================================="
    
    # 1A. Conv Linhas
    perf stat -o "perfs/1A_conv_linhas_N${N}.txt" -e "$PERF_EVENTS" ./objects/1A_conv_linhas "$N"
    sleep 1

    # 1B. Conv Colunas
    perf stat -o "perfs/1B_conv_colunas_N${N}.txt" -e "$PERF_EVENTS" ./objects/1B_conv_colunas "$N"
    sleep 1

    # 2. Conv SIMD
    perf stat -o "perfs/2_conv_simd_N${N}.txt" -e "$PERF_EVENTS" ./objects/2_conv_simd "$N"
    sleep 1

    # 3A. Conv Blocos (Scalar)
    for B in "${BLOCOS[@]}"; do
        perf stat -o "perfs/3A_conv_blocos_N${N}_B${B}.txt" -e "$PERF_EVENTS" ./objects/3A_conv_blocos "$N" "$B"
        sleep 1
    done

    # 3B. Conv Blocos (Vectorized)
    for B in "${BLOCOS[@]}"; do
        perf stat -o "perfs/3B_conv_blocos_N${N}_B${B}.txt" -e "$PERF_EVENTS" ./objects/3B_conv_blocos "$N" "$B"
        sleep 1
    done

    # 4. OpenCV Baseline
    perf stat -o "perfs/4_conv_opencv_N${N}.txt" -e "$PERF_EVENTS" ./objects/4_conv_opencv "$N"
    sleep 1
done

echo "=================================================="
echo "Bateria completa de testes finalizada!"
echo "=================================================="