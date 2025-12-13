#!/bin/bash

# Запускать на кластере в каталоге, где лежит ./solution2

# Загружаем модули и компилируем (если нужно)
module load pgi
module load openmpi
export OMPI_CXX=g++

# Создаем папки для логов
mkdir -p launches2_new
mkdir -p log2_new

# Сетки и процессы как в задании с CUDA
# Grids: 10x10, 20x20, 40x40, 80x80, 400x600, 800x1200, 4000x4000, 6000x6000
# Processes: 1, 2, 3, 4, 6

grids=("10 10" "20 20" "40 40" "80 80" "400 600" "800 1200" "4000 4000" "6000 6000")
procs=(1 2 4 8 16)

for pair in "${grids[@]}"; do
  set -- $pair
  N=$1
  M=$2

  for p in "${procs[@]}"; do
    out=./launches2_new/run_${N}x${M}_p${p}.lsf
    
    # Формируем скрипт для LSF
    echo "#BSUB -J sol2_${N}x${M}_p${p}"              >  "$out"
    echo "#BSUB -n ${p} -q normal"                    >> "$out"
    echo "#BSUB -W 00:20"                             >> "$out" # 20 минут должно хватить
    echo "#BSUB -o ./log2_new/${N}x${M}-p${p}.out"    >> "$out"
    echo "#BSUB -e ./log2_new/${N}x${M}-p${p}.err"    >> "$out"
    
    echo "#BSUB -R \"span[hosts=1]\""                 >> "$out"

    echo "mpirun -np ${p} ./solution2 ${N} ${M}"      >> "$out"

    bsub < "$out"
    echo "Submitted ${N}x${M} p=${p}"
  done
done

