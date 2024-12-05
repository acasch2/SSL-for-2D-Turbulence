#!/bin/bash
#SBATCH --time=0-15:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH -o run_analysis.out
#SBATCH -e run_analysis.err

#SBATCH --account=145439188689

cd /home/u.dp200518/SSL-Wavelets/eval

source activate /scratch/user/u.dp200518/.conda/envs/2DTurbEmulator

python analysis.py
