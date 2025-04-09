#!/bin/bash
#SBATCH --time=04:00:00
#SBATCH -N 1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem=250G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:2
#SBATCH --output=/scratch/user/u.aa286850/Logfiles/2DTurb_Base_%x.out
#SBATCH --error=/scratch/user/u.aa286850/Logfiles/2DTurb_Base_%x.err

#SBATCH --account=145439188689

# COMMAND: sbatch -J 'xxxx' test_wandb.sh 'xxxx'  - where 'xxxx' is RUN_NUM

set -x

cd /home/u.aa286850/SSL-for-2D-Turbulence/src


module load Anaconda3
source activate /scratch/user/u.aa286850/.conda/envs/2DTurbEmulator

# ------ Define all DDP vars ------ #
source export_DDP_vars.sh
export NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
#export WORLD_SIZE=${SLURM_NTASKS}
export OMP_NUM_THREADS=1

# ------ WANDB ------ #

source $HOME/wandb_setup_acasch.sh

# ------ Define all input args ------ #

YAML_CONFIG=/home/u.aa286850/SSL-for-2D-Turbulence/src/config/vitnet_FASTER.yaml
CONFIG=BASE


# ------ Run main script ------ #

torchrun --nproc_per_node=${NUM_TASKS_PER_NODE} --nnodes=${SLURM_NNODES} train.py --yaml_config $YAML_CONFIG --config $CONFIG --run_num $1 --fresh_start
