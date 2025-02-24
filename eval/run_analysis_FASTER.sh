#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:t4:1
#SBATCH -o run_analysis_video_v1.out
#SBATCH -e run_analysis_video_v1.err

#SBATCH --account=145439188689

cd /home/u.dp200518/SSL-Wavelets/eval

source activate /scratch/user/u.dp200518/.conda/envs/2DTurbEmulator_v1

python -u analyze_model.py config_video_v1_FASTER.yaml
