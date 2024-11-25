#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=2-10:30:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100:1

#SBATCH --job-name=cosfire_model_training

module load Python/3.11.5-GCCcore-13.2.0

source ../../env/bin/activate

python train.py