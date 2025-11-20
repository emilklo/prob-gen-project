#!/bin/bash
#SBATCH --job-name=vae-train
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="v100|a100"
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/vae_%j.out
#SBATCH --error=logs/vae_%j.err

echo "Starting VAE training job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"

cd /cluster/work/emilkl/prob-gen-project

# Show GPU info
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

nvidia-smi

# Run training (use -m to properly resolve module imports)
uv run python -m src.train --config config/default.yaml --mode vae

echo "Training complete"
