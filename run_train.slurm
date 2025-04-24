#!/bin/bash
#SBATCH --job-name=ball_action_train
#SBATCH --output=logs/ball_action_train_%j.log
#SBATCH --error=logs/ball_action_train_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=gpus  # <--- CHANGE HERE
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Load modules
module load docker

# Build and run inside Docker
make build
make run
make exec COMMAND="python3 scripts/ball_action/train.py --experiment sampling_weights_001"