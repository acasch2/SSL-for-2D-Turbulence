#!/bin/bash -l                                                          
#SBATCH --time=1-00:00:00                                                     
#SBATCH -C 'gpu&hbm40g'
#SBATCH --account=m4416                                         
#SBATCH -q regular                                                      
#SBATCH --nodes=1                                                     
#SBATCH --ntasks-per-node=4                                             
#SBATCH --gpus-per-node=4                                               
#SBATCH --cpus-per-task=8
#SBATCH --module=gpu,nccl-2.18                                          
#SBATCH -o /pscratch/sd/d/dpp94/Logfiles/2DTurb_finetune_%x.out
#SBATCH --mail-type=begin,end,fail                                      
#SBATCH --mail-user=dpp94@uchicago.edu 


# COMMAND: sbatch -J 'xxxx' test_wandb.sh 'xxxx'  - where 'xxxx' is RUN_NUM

set -x

cd /global/homes/d/dpp94/SSL-for-2D-Turbulence/src

# Activate conda env
ml conda
conda activate 2DTurbEmulator

# ------ Define all DDP vars ------ #
source export_DDP_vars.sh
export NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
#export WORLD_SIZE=${SLURM_NTASKS}
export OMP_NUM_THREADS=1

# ------ WANDB ------ #

source $HOME/set_wandb_key_dpp94.sh

# ------ Define all input args ------ #

YAML_CONFIG=/global/homes/d/dpp94/SSL-for-2D-Turbulence/src/config/vitnet_PM.yaml
CONFIG=MAE_FINETUNE


# ------ Run main script ------ #

torchrun --nproc_per_node=${NUM_TASKS_PER_NODE} --nnodes=${SLURM_NNODES} mae_finetune.py --yaml_config $YAML_CONFIG --config $CONFIG --run_num $1 --fresh_start
