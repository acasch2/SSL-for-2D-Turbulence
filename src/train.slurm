#!/bin/bash
## Job Name
#SBATCH --job-name=myjob
#SBATCH -A atm170004p
#SBATCH -p GPU-shared
#SBATCH --gpus=4
### #SBATCH -N 1
### #SBATCH -p GPU
### #SBATCH --gres=gpu:v100-16:8
#SBATCH -t 48:00:00

#SBATCH --export=all
### #SBATCH --output=/ocean/projects/phy220045p/jakhar/2d_emulator_vision/SSL-for-2D-Turbulence/results/%x_%j.out     # Output file name with job name and job ID
### #SBATCH --error=/ocean/projects/phy220045p/jakhar/2d_emulator_vision/SSL-for-2D-Turbulence/results/%x_%j.err      # Error file name with job name and job ID

## Resources: GPU
### request multiple of 8 GPUs for GPU parition
### v100-16:8 - 8 GPUs per node, 16 GB per GPU, 192 GB RAM per node
### v100-32:8 - 8 GPUs per node, 32 GB per GPU, 512 GB RAM per node
### v100-32:16 - 16 GPUs per node, 32 GB per GPU, 1.5 TB RAM per node


## Source bashrc and activate conda environment
source /jet/home/jakhar/.bashrc
conda activate fourks
module load nvhpc # for cuda, NVIDIA

# ------ Define all DDP vars ------ #
source export_DDP_vars.sh
export NUM_TASKS_PER_NODE=$(nvidia-smi -L | wc -l)
#export WORLD_SIZE=${SLURM_NTASKS}
export OMP_NUM_THREADS=1


# ------ Define all input args ------ #

YAML_CONFIG=/ocean/projects/phy220045p/jakhar/2d_emulator_vision/SSL-for-2D-Turbulence/src/config/base_emulator.yaml
CONFIG=BASE

# ------ Run main script ------ #

torchrun --nproc_per_node=${NUM_TASKS_PER_NODE} train.py --yaml_config $YAML_CONFIG --config $CONFIG --run_num base