#!/bin/bash

# Запускать на кластере в каталоге, где лежит ./sol3

# Загружаем модули и компилируем
module load pgi/18.4
module load openmpi/2.1.2/2018

# Компиляция

mkdir -p launches3
mkdir -p log3

###############################
# 1) p = 1 2 4 8 16 32 64
#    сетки 400x600 и 800x1200
###############################
for pair in "400 600" "800 1200"; do
  set -- $pair
  N=$1
  M=$2

  for p in 1 2 4 8 16 32; do
    out=./launches3/part1_${N}x${M}_p${p}.lsf

    echo "#BSUB -J solution3_p${p}_${N}x${M}_part1"     >  "$out"
    echo "#BSUB -n ${p} -q normal"                      >> "$out"
    echo "#BSUB -W 00:32"                               >> "$out"
    echo "#BSUB -o ./log3/part1-${N}x${M}-p${p}.out"     >> "$out"
    echo "#BSUB -e ./log3/part1-${N}x${M}-p${p}.err"     >> "$out"
    echo "#BSUB -R \"span[hosts=1]\""                   >> "$out"

    echo "mpirun -np ${p} ./sol3 ${N} ${M}"        >> "$out"

    bsub < "$out"
  done
done

###################################
# 2) p = 1, сетки 10x10 20x20 40x40
###################################
p=1
for pair in "10 10" "20 20" "40 40"; do
  set -- $pair
  N=$1
  M=$2

  out=./launches3/part2_${N}x${M}_p${p}.lsf

  echo "#BSUB -J solution3_p${p}_${N}x${M}_part2"       >  "$out"
  echo "#BSUB -n ${p} -q normal"                        >> "$out"
  echo "#BSUB -W 00:10"                                 >> "$out"
  echo "#BSUB -o ./log3/part2-${N}x${M}-p${p}.out"      >> "$out"
  echo "#BSUB -e ./log3/part2-${N}x${M}-p${p}.err"      >> "$out"
  echo "#BSUB -R \"span[hosts=1]\""                     >> "$out"

  echo "mpirun -np ${p} ./sol3 ${N} ${M}"           >> "$out"

  bsub < "$out"
done

################################
# 3) 
#p = 1 4 16, сетка 40x40
###############################
N=40
M=40
for p in 1 4 16; do
  out=./launches3/part3_${N}x${M}_p${p}.lsf

  echo "#BSUB -J solution3_p${p}_${N}x${M}_part3"       >  "$out"
  echo "#BSUB -n ${p} -q normal"                        >> "$out"
  echo "#BSUB -W 00:10"                                 >> "$out"
  echo "#BSUB -o ./log3/part3-${N}x${M}-p${p}.out"      >> "$out"
  echo "#BSUB -e ./log3/part3-${N}x${M}-p${p}.err"      >> "$out"
  echo "#BSUB -R \"span[hosts=1]\""                     >> "$out"

  echo "mpirun -np ${p} ./sol3 ${N} ${M}"           >> "$out"

  bsub < "$out"
done

