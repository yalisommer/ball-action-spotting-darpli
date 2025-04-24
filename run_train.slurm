#!/bin/bash
#SBATCH --job-name=ball_action_train
#SBATCH --output=logs/ball_action_train_%j.log
#SBATCH --error=logs/ball_action_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1  # Request one GPU

# Load any necessary modules
module load docker

# Build and run the Docker container
make build
make run

# Execute the training script inside the Docker container
make exec COMMAND="python3 scripts/ball_action/train.py --experiment sampling_weights_001"