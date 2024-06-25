#!/bin/bash
shopt -s expand_aliases
source /ssc-bin/cleancmd.sh
shopt -s expand_aliases
source /ssc-bin/cleancmd.sh
shopt -s expand_aliases
source /ssc-bin/cleancmd.sh
shopt -s expand_aliases
source /ssc-bin/cleancmd.sh
shopt -s expand_aliases
source /ssc-bin/cleancmd.sh
shopt -s expand_aliases
source /ssc-bin/cleancmd.sh
shopt -s expand_aliases
source /ssc-bin/cleancmd.sh
# create the right environment to run client: note client runs in python3
# while i-pi server runs in python2

# module load gcc/8.3.0
# module load fftw/3.3.8/single-threads
# module load compiler/intel/ips2018/u1
# module load mkl/intel/ips2018/u1
# module load cuda/11.4

export OMP_NUM_THREADS=1

addr=unix_dmff
port=1234
socktype=unix

python ./client_dmff.py water_new.pdb forcefield.xml residues.xml params.pickle  $addr $port $socktype 









