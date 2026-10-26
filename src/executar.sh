#!/bin/bash

# Configurações de tamanhos massivos e blocos
TAMANHOS=(26500 24576 23552)
BLOCOS=(8 16 32 64)

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

# Cria um gerador temporário otimizado e paralelo para criar a matriz binária
cat << 'EOF' > gerar_matriz.cpp
#include <iostream>
#include <vector>
#include <random>
#include <fstream>
#include <cstdlib>
#include <omp.h>

using namespace std;

int main(int argc, char** argv) {
    int N = (argc > 1) ? atoi(argv[1]) : 1024;
    string filename = "matriz_in_N" + to_string(N) + ".bin";
    size_t total_elementos = (size_t)N * N;
    
    vector<float> in(total_elementos);
    
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        size_t chunk_size = total_elementos / nthreads;
        size_t start = tid * chunk_size;
        size_t end = (tid == nthreads - 1) ? total_elementos : start + chunk_size;

        mt19937 gen(42);
        gen.discard(start);
        uniform_real_distribution<float> dis(0.0f, 255.0f);

        for (size_t k = start; k < end; k++) {
            in[k] = dis(gen);
        }
    }

    ofstream file(filename, ios::binary);
    file.write(reinterpret_cast<char*>(in.data()), total_elementos * sizeof(float));
    file.close();
    return 0;
}
EOF

g++ -O3 -fopenmp gerar_matriz.cpp -o objects/gerar_matriz
rm -f gerar_matriz.cpp

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

# Opcional: limpa os arquivos binários gigantes ao finalizar para não ocupar espaço em disco
rm -f *.bin

echo "=================================================="
echo "Bateria completa de testes finalizada!"
echo "=================================================="