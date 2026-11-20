#!/bin/bash

# Configurações de tamanhos massivos e blocos
# TAMANHOS=(26500 24576 23552)
TAMANHOS=(1000 2000)

BLOCOS=(8 16 32 64)
EXECUCOES=5
THREAD_COUNTS=(1 2)

export OMP_DYNAMIC=FALSE

echo "=================================================="
echo "Removendo pastas de resultados e binários antigos..."
echo "=================================================="
rm -rf objects perfs csv_time *.bin

echo "=================================================="
echo "Criando novas pastas de diretórios..."
echo "=================================================="
mkdir -p objects
mkdir -p perfs
mkdir -p csv_time

echo "=================================================="
echo "Compilando Utilitário de Geração de Matrizes..."
echo "=================================================="

g++ -O3 -fopenmp gerar_matriz.cpp -o objects/gerar_matriz

echo "=================================================="
echo "Iniciando Compilação de Todos os Programas..."
echo "=================================================="

OPENCV_FLAGS=$(pkg-config --cflags --libs opencv4 2>/dev/null || pkg-config --cflags --libs opencv)

# Compilação de todos os algoritmos (com -fopenmp incluído)
g++ -O3 -mavx2 -fopenmp-simd -fopenmp 1A_conv_linhas.cpp -o objects/1A_conv_linhas $OPENCV_FLAGS
g++ -O3 -mavx2 -fopenmp-simd -fopenmp 1B_conv_colunas.cpp -o objects/1B_conv_colunas $OPENCV_FLAGS
g++ -O3 -mavx2 -fopenmp-simd -fopenmp 2_conv_simd.cpp -o objects/2_conv_simd $OPENCV_FLAGS
g++ -O3 -mavx2 -fopenmp-simd -fopenmp 3A_conv_blocos.cpp -o objects/3A_conv_blocos $OPENCV_FLAGS
g++ -O3 -mavx2 -fopenmp-simd -fopenmp 3B_conv_blocos.cpp -o objects/3B_conv_blocos $OPENCV_FLAGS
g++ -O3 -mavx2 -fopenmp-simd -fopenmp 4_conv_opencv.cpp -o objects/4_conv_opencv $OPENCV_FLAGS

echo "Compilação concluída com sucesso!"
echo "=================================================="
echo "Executando Benchmarks (perf + tempos)..."
echo "=================================================="

PERF_EVENTS="cycles,instructions,l1-dcache-loads,l1-dcache-load-misses,l2_rqsts.all_demand_data_rd,l2_rqsts.all_demand_miss,mem_load_retired.l3_hit,mem_load_retired.l3_miss"

for N in "${TAMANHOS[@]}"; do
    echo ==================================================
    echo "--- Gerando Matriz Determinística (N=$N) ---"
    echo ==================================================
    ./objects/gerar_matriz "$N"

    echo ==================================================
    echo "--- Testando Tamanho N = $N ---"
    echo ==================================================

    for ((I = 1; I <= EXECUCOES; I++)); do
        for THREADS in "${THREAD_COUNTS[@]}"; do
            export OMP_NUM_THREADS="$THREADS"
            echo "--- Iteração $I/$EXECUCOES (N=$N, threads=$THREADS) ---"

            # 1A. Conv Linhas
            ITERACAO="$I" perf stat -o "perfs/1A_conv_linhas_N${N}_I${I}_T${THREADS}.txt" -e "$PERF_EVENTS" ./objects/1A_conv_linhas "$N"
            sleep 1

            # 1B. Conv Colunas
            ITERACAO="$I" perf stat -o "perfs/1B_conv_colunas_N${N}_I${I}_T${THREADS}.txt" -e "$PERF_EVENTS" ./objects/1B_conv_colunas "$N"
            sleep 1

            # 2. Conv SIMD
            ITERACAO="$I" perf stat -o "perfs/2_conv_simd_N${N}_I${I}_T${THREADS}.txt" -e "$PERF_EVENTS" ./objects/2_conv_simd "$N"
            sleep 1

            # 3A. Conv Blocos (Scalar)
            for B in "${BLOCOS[@]}"; do
                ITERACAO="$I" perf stat -o "perfs/3A_conv_blocos_N${N}_B${B}_I${I}_T${THREADS}.txt" -e "$PERF_EVENTS" ./objects/3A_conv_blocos "$N" "$B"
                sleep 1
            done

            # 3B. Conv Blocos (Vectorized)
            for B in "${BLOCOS[@]}"; do
                ITERACAO="$I" perf stat -o "perfs/3B_conv_blocos_N${N}_B${B}_I${I}_T${THREADS}.txt" -e "$PERF_EVENTS" ./objects/3B_conv_blocos "$N" "$B"
                sleep 1
            done

            # 4. OpenCV Baseline
            ITERACAO="$I" perf stat -o "perfs/4_conv_opencv_N${N}_I${I}_T${THREADS}.txt" -e "$PERF_EVENTS" ./objects/4_conv_opencv "$N"
            sleep 1
        done
    done
done

# Opcional: limpa os arquivos binários gigantes ao finalizar para não ocupar espaço em disco
rm -f *.bin

echo "=================================================="
echo "Bateria completa de testes finalizada!"
echo "=================================================="