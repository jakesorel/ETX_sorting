#!/bin/bash                                                                                                                           


for i in $(seq 0 $(($1-1)))
do
    sbatch run_jobhet_WTT.sh "$i"
done

