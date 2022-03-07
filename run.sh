#!/bin/zsh
#SBATCH --ntasks=10
srun python curry/main.py $@
