#!/bin/bash
#SBATCH --job-name=mdnrnn-train
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/rnn_%j.out
#SBATCH --error=logs/rnn_%j.err

echo "=============================================="
echo "  MDN-RNN Training"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

cd /cluster/work/emilkl/prob-gen-project

mkdir -p logs

# Show GPU info
nvidia-smi
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo ""

# Run MDN-RNN training
echo "Training MDN-RNN using VAE checkpoint from config run_name..."
uv run python -m src.train_mdnrnn

echo ""
echo "RNN training complete!"
echo "End time: $(date)"
