#!/bin/bash
# executar_omp.sh - Benchmark da versão OpenMP variando o nº de threads
# Rodar dentro da pasta src/:   chmod +x executar_omp.sh && ./executar_omp.sh

TAMANHOS=(8192 16384)   # lados da imagem (NxN)
REPS=5                  # repetições por configuração
ITERS=5                 # execuções do kernel dentro da região medida
PERF_EVENTS="cycles,instructions,l1-dcache-loads,l1-dcache-load-misses,l2_rqsts.all_demand_data_rd,l2_rqsts.all_demand_miss,mem_load_retired.l3_hit,mem_load_retired.l3_miss"

# ---- lista de threads: 1,2,4,8,... até o nº lógico de CPUs + nº de núcleos físicos
NP=$(nproc)
PHYS=$(lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l)
LISTA=(); t=1
while [ $t -lt $NP ]; do LISTA+=($t); t=$((t*2)); done
LISTA+=($NP $PHYS)
THREADS=($(printf "%s\n" "${LISTA[@]}" | sort -nu))
echo "CPUs lógicas: $NP | núcleos físicos: $PHYS | threads a testar: ${THREADS[*]}"

echo "=== Preparando pastas ==="
rm -rf perfs_omp csv_time/5_conv_omp.csv
mkdir -p objects perfs_omp csv_time

echo "=== Compilando ==="
OPENCV_FLAGS=$(pkg-config --cflags --libs opencv4 2>/dev/null || pkg-config --cflags --libs opencv)
g++ -O3 -mavx2 -fopenmp 5_conv_omp.cpp -o objects/5_conv_omp $OPENCV_FLAGS || { echo "Falha na compilação"; exit 1; }

# Afinidade: threads presas aos núcleos (resultado estável e reprodutível)
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "=== Executando ==="
for N in "${TAMANHOS[@]}"; do
  for T in "${THREADS[@]}"; do
    for r in $(seq 1 $REPS); do
      export OMP_NUM_THREADS=$T
      echo "--- N=$N  threads=$T  rep=$r ---"

      # FIFOs para o perf ligar/desligar a contagem só em volta do kernel
      rm -f /tmp/pad_ctl /tmp/pad_ack
      mkfifo /tmp/pad_ctl /tmp/pad_ack
      exec {CTL}<>/tmp/pad_ctl
      exec {ACK}<>/tmp/pad_ack

      PERF_CTL_FD=$CTL PERF_ACK_FD=$ACK \
      perf stat -D -1 --control fd:${CTL},${ACK} -x, \
          -o "perfs_omp/N${N}_T${T}_r${r}.csv" -e "$PERF_EVENTS" \
          ./objects/5_conv_omp "$N" "$ITERS" "$r"

      exec {CTL}>&-
      exec {ACK}>&-
      rm -f /tmp/pad_ctl /tmp/pad_ack
      sleep 2
    done
  done
done

echo "=== Fim. Resultados em csv_time/5_conv_omp.csv e perfs_omp/ ==="
echo "Agora rode: python3 analisar.py"
