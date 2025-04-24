#!/bin/bash
#SBATCH --job-name=ball_action_train
#SBATCH --output=logs/ball_action_train_%j.log
#SBATCH --error=logs/ball_action_train_%j.err
#SBATCH --time=00:15:00
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --export=ALL

# Ensure logs directory exists
mkdir -p logs

# Load module system
source /etc/profile
module load python/3.10  # Or whatever version you want

# Activate virtual environment
source ~/myEnv/bin/activate

# Add current directory to Python path
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Run your training script
python3 scripts/ball_action/predict.py --experiment sampling_weights_001