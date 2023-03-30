#!/bin/sh

#SBATCH --partition=lyceum
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=03:00:00
#SBATCH --mail-user=ik3g20@soton.ac.uk
#SBATCH --output=slurm_outs/voc_e50_train_history.out
#SBATCH --error=slurm_outs/voc_e50_train_error_history.err

module load cuda/11.7
module load gcc/11.0

export PATH=/scratch/ik3g20/anaconda3/envs/ip_iridis/bin/:$PATH
export LD_LIBRARY_PATH=/scratch/ik3g20/anaconda3/envs/ip_iridis/lib/:$LD_LIBRARY_PATH

python ./train.py VOC_E50_256-256.yaml


