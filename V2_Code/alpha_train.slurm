#!/bin/sh

#SBATCH --partition=ecsstudents
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --mail-user=ik3g20@soton.ac.uk
#SBATCH --output=train_history.out
#SBATCH --error=train_error_history.err

module load python/3.10.4
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

pip install --no-index -r requirements.txt

python ./init_train.py
python ./train.py VOC_E50_256-256.yaml


