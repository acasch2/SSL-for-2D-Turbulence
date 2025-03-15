#!/bin/bash

## Job Name
#SBATCH --job-name=myjob

## Resources
#SBATCH -A atm170004p
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH --gpus=1
### #SBATCH -N 1
### #SBATCH -p GPU
### #SBATCH --gres=gpu:v100-16:8
#SBATCH -t 48:00:00

## Export all environment variables to the job
#SBATCH --export=all

source /jet/home/jakhar/.bashrc
conda activate fourks
module load nvhpc # for cuda, NVIDIA

## Run the python script
python3 -u -m analyze_model config.yaml