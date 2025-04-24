#!/bin/bash
#SBATCH --job-name=ball_action_train
#SBATCH --output=logs/ball_action_train_%j.log
#SBATCH --error=logs/ball_action_train_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Load modules (if needed for CUDA or Python)
module load python/3.11.11  # or whatever version is required
# module load cuda        # if you're using GPU code directly

# Activate your virtual environment
source ~/myEnv/bin/activate

# Run the training script
echo "sanity test"
# python3 scripts/ball_action/train.py --experiment sampling_weights_001