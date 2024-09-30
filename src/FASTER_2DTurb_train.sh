#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a10:4
#SBATCH --output=/scratch/user/u.dp200518/Logfiles/2DTurb_Finetune_%x.out

#SBATCH --account=145439188689

# COMMAND: sbatch -J 'xxxx' test_wandb.sh 'xxxx'  - where 'xxxx' is RUN_NUM

set -x

cd /home/u.dp200518/SSL-Wavelets/src


#module load Anaconda3
source activate /scratch/user/u.dp200518/.conda/envs/2DTurbEmulator

# ------ Define all DDP vars ------ #
source export_DDP_vars.sh
export NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
#export WORLD_SIZE=${SLURM_NTASKS}
export OMP_NUM_THREADS=1

# ------ WANDB ------ #

source $HOME/set_wandb_key_dpp94.sh

# ------ Define all input args ------ #

YAML_CONFIG=/home/u.dp200518/SSL-Wavelets/src/config/mae_finetune.yaml
CONFIG=MAE_FINETUNE


# ------ Run main script ------ #

torchrun --nproc_per_node=${NUM_TASKS_PER_NODE} mae_finetune.py --yaml_config $YAML_CONFIG --config $CONFIG --run_num $1 --fresh_start
