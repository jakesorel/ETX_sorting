#!/bin/bash

#Submit this script with: sbatch thefilename

#SBATCH --time=1:00:00   # walltime
#SBATCH --ntasks=20   # number of processor cores (i.e. tasks)
#SBATCH -J "cell_sorting_jakecs"   # job name
#SBATCH --output=output.out
#SBATCH --error=output.out   
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=jakecs@caltech.edu

source activate apical_domain

python vary_heterogeneity_WTT.py "$1"