#!/bin/bash

# Запускать на кластере в каталоге, где лежит ./sol3
# Использование: ./run_sol3.sh [LOG_DIR_SUFFIX]
# Если SUFFIX не передан, используется пустая строка (папки log3 и launches3)
# Если передан (например, "_run1"), папки будут log3_run1 и launches3_run1

SUFFIX="${1:-}"
LOG_DIR="log3${SUFFIX}"
LAUNCH_DIR="launches3${SUFFIX}"

# Загружаем модули и компилируем
module load SpectrumMPI
export OMPI_CXX=mpicxx

# Компиляция MPI + OpenMP
#mpicxx -O3 -fopenmp -std=c++11 solution3.cpp -o sol3

mkdir -p "$LAUNCH_DIR"
mkdir -p "$LOG_DIR"

GRIDS=(
    "400 600"
    "800 1200"
)

# Списки для перебора всех комбинаций
MPI_PROCS=(1 2 4 8 16)
OMP_THREADS=(1 2 4 8 16)

# Для каждой сетки перебираем все комбинации MPI процессов × OpenMP потоков
for pair in "${GRIDS[@]}"; do
  set -- $pair
  N=$1
  M=$2
  
  if [ "$N" -ge 4000 ]; then
      WTIME="00:45"
  else
      WTIME="00:15"
  fi

  # Перебираем все комбинации threads × np
  for p in "${MPI_PROCS[@]}"; do
    for omp_threads in "${OMP_THREADS[@]}"; do
      out="./${LAUNCH_DIR}/sol3_${N}x${M}_p${p}_t${omp_threads}.lsf"

      echo "#BSUB -J sol3_p${p}_t${omp_threads}_${N}x${M}${SUFFIX}"  >  "$out"
      echo "#BSUB -n ${p} -q normal"                                  >> "$out"
      echo "#BSUB -W ${WTIME}"                                        >> "$out"
      echo "#BSUB -o ./${LOG_DIR}/${N}x${M}-p${p}-t${omp_threads}.out" >> "$out"
      echo "#BSUB -e ./${LOG_DIR}/${N}x${M}-p${p}-t${omp_threads}.err" >> "$out"
      
      # Настройка распределения по узлам
      if [ "$p" -eq 1 ]; then
          echo "#BSUB -R \"span[hosts=1]\""                           >> "$out"
      else
          echo "#BSUB -R \"span[ptile=2]\""                           >> "$out"
      fi
      
      echo "export OMP_NUM_THREADS=${omp_threads}"                    >> "$out"
      echo "mpirun -np ${p} ./sol3 ${N} ${M}"                         >> "$out"

      bsub < "$out"
    done
  done
done
