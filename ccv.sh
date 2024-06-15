#!/bin/bash
# https://docs.ccv.brown.edu/oscar/submitting-jobs/batch
# EXECUTE BY running sbatch ccv.sh

#SBATCH -n 4
#SBATCH --time=4:00:00
#SBATCH --mem=8G
#SBATCH -J LatentODE
#SBATCH -o LatentODE-%j.out
#SBATCH -e LatentODE-%j.err
#SBATCH --mail-type=ALL

#SBATCH -p gpu --gres=gpu:1

module load cuda
module load cudnn

source ~/LatentODESwarm/.venv/bin/activate

# Run a command
python learner_zhen/example_maze.py --gpu --iters 100000
