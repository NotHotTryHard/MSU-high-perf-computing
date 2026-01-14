#!/bin/bash

# Запускать на кластере в каталоге, где лежит ./sol3
# Использование: ./run_sol3.sh [LOG_DIR_SUFFIX]
# Если SUFFIX не передан, используется пустая строка (папки log3 и launches3)
# Если передан (например, "_run1"), папки будут log3_run1 и launches3_run1

SUFFIX="${1:-}"
LOG_DIR="log4${SUFFIX}"
LAUNCH_DIR="launches4${SUFFIX}"

# Загружаем модули и компилируем
module load openmpi
module load pgi
export OMPI_CXX=g++

# Компиляция
nvcc -O3 -std=gnu++11 -arch=sm_60 -ccbin mpicxx solution4.cu -o sol4

mkdir -p "$LAUNCH_DIR"
mkdir -p "$LOG_DIR"

GRIDS=(
    "10 10"
    "20 20"
    "40 40"
    "80 80"
    "400 600"
    "800 1200"
    "4000 4000"
)

# Для каждой сетки запускаем на p = 1, 2, 4
for pair in "${GRIDS[@]}"; do
  set -- $pair
  N=$1
  M=$2
  
  if [ "$N" -ge 4000 ]; then
      WTIME="00:45"
  else
      WTIME="00:15"
  fi

  for p in 1 2 3 4 6; do
    out="./${LAUNCH_DIR}/sol4_${N}x${M}_p${p}.lsf"

    echo "#BSUB -J sol4_p${p}_${N}x${M}${SUFFIX}"       >  "$out"
    echo "#BSUB -n ${p} -q normal"                      >> "$out"
    echo "#BSUB -x"                                     >> "$out"
    echo "#BSUB -W ${WTIME}"                            >> "$out"
    echo "#BSUB -o ./${LOG_DIR}/${N}x${M}-p${p}.out"    >> "$out"
    echo "#BSUB -e ./${LOG_DIR}/${N}x${M}-p${p}.err"    >> "$out"
    
    if [ "$p" -eq 1 ]; then
        echo "#BSUB -R \"span[hosts=1]\""                   >> "$out"
        echo "#BSUB -gpu \"num=1:mode=exclusive_process\""  >> "$out"
    else
        echo "#BSUB -R \"span[ptile=2]\""                   >> "$out"
        echo "#BSUB -gpu \"num=2:mode=exclusive_process\""  >> "$out"
    fi

    echo "mpirun -np ${p} ./sol4 ${N} ${M}"             >> "$out"

    bsub < "$out"
  done
done
