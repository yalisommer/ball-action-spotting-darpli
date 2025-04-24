#!/bin/bash
#SBATCH --job-name=ball_action_train
#SBATCH --output=logs/ball_action_train_%j.log
#SBATCH --error=logs/ball_action_train_%j.err
#SBATCH --time=00:10:00
#SBATCH --partition=gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --export=ALL  # Ensure environment variables are preserved

# Confirm you're using bash
echo "Shell: $SHELL"

# Optional: load modules
module load python/3.10  # Try a safer default module if 3.11 isn't available

# Confirm Python works
which python3
python3 --version

# Try a minimal test
echo "sanity test"
# python3 scripts/ball_action/train.py --experiment sampling_weights_001