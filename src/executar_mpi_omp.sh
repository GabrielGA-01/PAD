#!/bin/bash
# executar_mpi_omp.sh - Benchmark Híbrido (MPI + OpenMP) sem perf, sincronizado com HWiNFO64

TAMANHOS=(8192 16384)   # lados da imagem (NxN)
MODOS=(0 1)             # 0 = bloqueante | 1 = sobreposição (Isend/Irecv)
REPS=5                  # repetições por configuração
ITERS=50                # iterações por execução (dura alguns segundos para o HWiNFO64 capturar)

NP=$(nproc)
PHYS=$(lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l)

LISTA=(); x=1
while [ $x -lt $NP ]; do LISTA+=($x); x=$((x*2)); done
LISTA+=($NP $PHYS)
VALORES=($(printf "%s\n" "${LISTA[@]}" | sort -nu))

PARES_PT=()
for P in "${VALORES[@]}"; do
  for T in "${VALORES[@]}"; do
    if [ $((P * T)) -le $NP ]; then
      PARES_PT+=("$P $T")
    fi
  done
done

echo "CPUs lógicas: $NP | núcleos físicos: $PHYS"
echo "Combinações (P x T): ${PARES_PT[*]}"

rm -rf csv_time/6_conv_mpi_omp.csv csv_time/timestamps_mpi_omp.csv
mkdir -p objects csv_time
echo "N;P;T;modo;rep;iters;inicio;fim" > csv_time/timestamps_mpi_omp.csv

mpic++ -O3 -mavx2 -fopenmp 6_conv_mpi_omp.cpp -o objects/6_conv_mpi_omp || { echo "Falha na compilação"; exit 1; }

export OMP_PROC_BIND=close
export OMP_PLACES=cores

for N in "${TAMANHOS[@]}"; do
  for pt in "${PARES_PT[@]}"; do
    read -r P T <<< "$pt"
    for modo in "${MODOS[@]}"; do
      for r in $(seq 1 $REPS); do
        export OMP_NUM_THREADS=$T
        INICIO=$(date +"%H:%M:%S.%3N")
        echo "--- N=$N  P=$P  T=$T  modo=$modo  rep=$r ($INICIO) ---"

        mpirun --oversubscribe --bind-to none \
               -x OMP_NUM_THREADS -x OMP_PROC_BIND -x OMP_PLACES \
               -np "$P" ./objects/6_conv_mpi_omp "$N" "$ITERS" "$r" "$modo"

        FIM=$(date +"%H:%M:%S.%3N")
        echo "$N;$P;$T;$modo;$r;$ITERS;$INICIO;$FIM" >> csv_time/timestamps_mpi_omp.csv
        sleep 2
      done
    done
  done
done

echo "=== Fim! Pare o log do HWiNFO64 e rode: python3 integrar_hwinfo.py ==="
