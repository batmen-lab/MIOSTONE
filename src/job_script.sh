#!/bin/bash

#SBATCH --job-name=myJob
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=../slurm_logs/myJob_%A_%a.out

# Load the arguments
IFS=$'\n' read -d '' -r -a lines < args.txt
args="${lines[$SLURM_ARRAY_TASK_ID]}"

# Run the command
srun python train.py $args