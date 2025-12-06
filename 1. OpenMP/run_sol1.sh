#!/bin/bash

# Запускать на кластере в каталоге, где лежит ./sol1

mkdir -p launches
mkdir -p log

###############################
# 1) p = 1 2 4 8 16 32 64
#    сетки 400x600 и 800x1200
###############################
for pair in "400 600" "800 1200"; do
  set -- $pair
  N=$1
  M=$2

  for p in 1 2 4 8 16 32 64; do
    out=./launches/part1_${N}x${M}_p${p}.lsf

    echo "#BSUB -J sol1_p${p}_${N}x${M}_part1"     >  "$out"
    echo "#BSUB -n 1 -q normal"                    >> "$out"
    echo "#BSUB -W 00:32"                          >> "$out"
    echo "#BSUB -o ./log/part1-${N}x${M}-p${p}.out" >> "$out"
    echo "#BSUB -e ./log/part1-${N}x${M}-p${p}.err" >> "$out"
    echo "#BSUB -R \"span[hosts=1]\""              >> "$out"

    echo "OMP_NUM_THREADS=${p} ./sol1 ${N} ${M}"   >> "$out"

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

  out=./launches/part2_${N}x${M}_p${p}.lsf

  echo "#BSUB -J sol1_p${p}_${N}x${M}_part2"       >  "$out"
  echo "#BSUB -n 1 -q normal"                      >> "$out"
  echo "#BSUB -W 00:10"                            >> "$out"
  echo "#BSUB -o ./log/part2-${N}x${M}-p${p}.out"  >> "$out"
  echo "#BSUB -e ./log/part2-${N}x${M}-p${p}.err"  >> "$out"
  echo "#BSUB -R \"span[hosts=1]\""                >> "$out"

  echo "OMP_NUM_THREADS=${p} ./sol1 ${N} ${M}"     >> "$out"

  bsub < "$out"
done

################################
# 3) 
#p = 1 4 16, сетка 40x40
###############################
N=40
M=40
for p in 1 4 16; do
  out=./launches/part3_${N}x${M}_p${p}.lsf

  echo "#BSUB -J sol1_p${p}_${N}x${M}_part3"       >  "$out"
  echo "#BSUB -n 1 -q normal"                      >> "$out"
  echo "#BSUB -W 00:10"                            >> "$out"
  echo "#BSUB -o ./log/part3-${N}x${M}-p${p}.out"  >> "$out"
  echo "#BSUB -e ./log/part3-${N}x${M}-p${p}.err"  >> "$out"
  echo "#BSUB -R \"span[hosts=1]\""                >> "$out"

  echo "OMP_NUM_THREADS=${p} ./sol1 ${N} ${M}"     >> "$out"

  bsub < "$out"
done
