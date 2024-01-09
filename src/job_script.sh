#!/bin/bash

#SBATCH --job-name=myJob
#SBATCH --gpus-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --output=../slurm_logs/myJob_%A_%a.out

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <args_filename>"
    exit 1
fi

# Read the filename from the command line argument
args_file=$1

# Check if the file exists
if [ ! -f "$args_file" ]; then
    echo "Error: File not found: $args_file"
    exit 1
fi

# Load the arguments for the current task
IFS=$'\n' read -d '' -r -a lines < "$args_file"
args="${lines[$SLURM_ARRAY_TASK_ID]}"

# Determine which script to run
if [[ $args_file == 'train_args.txt' ]]; then
    # Run train.py
    srun python train.py $args
else
    # Run transfer_learning.py
    srun python transfer_learning.py $args
fi