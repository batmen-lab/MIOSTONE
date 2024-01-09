#!/bin/bash

# Check if an argument is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <filename>"
    exit 1
fi

# Read the filename from the command line argument
filename=$1

# Check if the file exists
if [ ! -f "$filename" ]; then
    echo "Error: File not found: $filename"
    exit 1
fi

# Count the number of lines in the file
num_lines=$(wc -l < "$filename")

# Submit the job array to Slurm
sbatch --array=0-$((num_lines-1)) job_script.sh "$filename"
