#!/bin/bash
#SBATCH --job-name=train_hackathon
#SBATCH --nodes=1             
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=rtx8000,pascal
#SBATCH --time=0-06:00:00
#SBATCH --output=train_hackathon.out
#SBATCH --qos=6hours

# activate conda env
export PATH=$HOME/miniconda3/bin:$PATH
source activate hackathon

srun python main.py fit -c $1