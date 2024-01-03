#!/bin/bash

num_lines=$(wc -l < args.txt)
sbatch --array=0-$((num_lines)) job_script.sh
