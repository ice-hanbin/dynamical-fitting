#!/bin/bash
#SBATCH --export=ALL
#SBATCH --job-name=diff_test
#SBATCH -N 1 -n 1 -c 1
#SBATCH --tasks-per-node=1 -p rtx3090 --nodelist gpu004
#SBATCH --mem=10000mb -o out -e err

export OPENMM_CPU_THREADS=1
export OMP_NUM_THREADS=1

module load cuda/11.4


echo "***** start time *****"
date

python diffusion.py > log

echo "***** finish time *****"
date

