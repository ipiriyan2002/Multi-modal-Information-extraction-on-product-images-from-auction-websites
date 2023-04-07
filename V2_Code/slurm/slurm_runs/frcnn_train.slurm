#!/bin/sh

#SBATCH --partition=lyceum
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=30G
#SBATCH --gres=gpu:4
#SBATCH --time=2-12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=ik3g20@soton.ac.uk
#SBATCH --output=slurm/slurm_outs/cord_e20_train.out
#SBATCH --error=slurm/slurm_outs/cord_e20_train_error.err

module load cuda/11.7
module load gcc/11.0

export PATH=/scratch/ik3g20/anaconda3/envs/ip_iridis/bin/:$PATH
export LD_LIBRARY_PATH=/scratch/ik3g20/anaconda3/envs/ip_iridis/lib/:$LD_LIBRARY_PATH
export PYTHON_PATH=/lyceum/ik3g20/Multi-modal-Information-extraction-on-product-images-from-auction-websites/V2_Code$PYTHON_PATH

python predefined/frcnn_train.py CORD_E20_256-256.yaml


