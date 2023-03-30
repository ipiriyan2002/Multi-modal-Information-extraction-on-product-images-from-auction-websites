#!/bin/sh

#SBATCH --partition=lyceum
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mail-user=ik3g20@soton.ac.uk
#SBATCH --output=slurm_outs/voc2007/download.out
#SBATCH --error=slurm_outs/voc2007/download_error.err

module load cuda/11.7

export PATH=/scratch/ik3g20/anaconda3/envs/ip_iridis/bin/:$PATH
export LD_LIBRARY_PATH=/scratch/ik3g20/anaconda3/envs/ip_iridis/lib/:$LD_LIBRARY_PATH

pwd
python ./init_train.py --target-dir /scratch/ik3g20/data/voc2007/ --split train
python ./init_train.py --target-dir /scratch/ik3g20/data/voc2007/ --split test

