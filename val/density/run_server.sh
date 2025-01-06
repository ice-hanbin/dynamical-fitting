#!/bin/bash
# shopt -s expand_aliases
# source /ssc-bin/cleancmd.sh
export OMP_NUM_THREADS=1

cat input.xml | sed -e "s/<address> \([a-zA-Z_]\+\)/<address> \1_${SLURM_JOB_ID}/" > .tmp.xml
# cat input.xml | sed -e "s/<address> \([a-zA-Z_]\+\)/<address> \1/" > .tmp.xml
# i-pi simulation.restart >& logfile &
i-pi .tmp.xml >& logfile &
wait


