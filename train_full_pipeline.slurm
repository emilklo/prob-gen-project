#!/bin/bash
#SBATCH --job-name=world-model-full
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/full_pipeline_%j.out
#SBATCH --error=logs/full_pipeline_%j.err

# Run name from cuda.yaml config
RUN_NAME="cuda_z128_full"

echo "=============================================="
echo "  Full World Model Training Pipeline"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Run name: $RUN_NAME"
echo "Start time: $(date)"
echo ""

cd /cluster/work/emilkl/prob-gen-project

# Create logs directory if it doesn't exist
mkdir -p logs

# Show GPU info
echo "--- GPU Information ---"
nvidia-smi
uv run python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
echo ""

# ============================================
# PHASE 1: Train ConvVAE
# ============================================
echo "=============================================="
echo "  PHASE 1: Training ConvVAE"
echo "=============================================="
echo "Config: default.yaml + cuda.yaml (auto-merged)"
echo "Output: outputs/$RUN_NAME/checkpoints/"
echo ""

VAE_START=$(date +%s)
uv run python -m src.train --mode vae
VAE_STATUS=$?
VAE_END=$(date +%s)

if [ $VAE_STATUS -ne 0 ]; then
    echo "ERROR: VAE training failed with exit code $VAE_STATUS"
    exit 1
fi

VAE_DURATION=$((VAE_END - VAE_START))
echo ""
echo "VAE training completed in $((VAE_DURATION / 60)) minutes $((VAE_DURATION % 60)) seconds"
echo ""

# Verify VAE checkpoint exists
VAE_CHECKPOINT="outputs/$RUN_NAME/checkpoints/vae_final.pth"
if [ ! -f "$VAE_CHECKPOINT" ]; then
    echo "ERROR: VAE checkpoint not found at $VAE_CHECKPOINT"
    exit 1
fi
echo "VAE checkpoint saved: $VAE_CHECKPOINT"
echo ""

# ============================================
# PHASE 2: Train MDN-RNN
# ============================================
echo "=============================================="
echo "  PHASE 2: Training MDN-RNN"
echo "=============================================="
echo "Using VAE checkpoint: $VAE_CHECKPOINT"
echo "Output: outputs/$RUN_NAME/rnn_checkpoints/"
echo ""

RNN_START=$(date +%s)
uv run python -m src.train_mdnrnn
RNN_STATUS=$?
RNN_END=$(date +%s)

if [ $RNN_STATUS -ne 0 ]; then
    echo "ERROR: MDN-RNN training failed with exit code $RNN_STATUS"
    exit 1
fi

RNN_DURATION=$((RNN_END - RNN_START))
echo ""
echo "MDN-RNN training completed in $((RNN_DURATION / 60)) minutes $((RNN_DURATION % 60)) seconds"
echo ""

# ============================================
# SUMMARY
# ============================================
echo "=============================================="
echo "  TRAINING COMPLETE - SUMMARY"
echo "=============================================="
echo "End time: $(date)"
echo ""
echo "--- Timing ---"
echo "VAE training:     $((VAE_DURATION / 60))m $((VAE_DURATION % 60))s"
echo "MDN-RNN training: $((RNN_DURATION / 60))m $((RNN_DURATION % 60))s"
TOTAL_DURATION=$((VAE_DURATION + RNN_DURATION))
echo "Total:            $((TOTAL_DURATION / 60))m $((TOTAL_DURATION % 60))s"
echo ""
echo "--- Output Files ---"
echo "VAE Checkpoints:"
ls -lh outputs/$RUN_NAME/checkpoints/*.pth 2>/dev/null || echo "  No checkpoints found"
echo ""
echo "MDN-RNN Checkpoints:"
ls -lh outputs/$RUN_NAME/rnn_checkpoints/*.pth 2>/dev/null || echo "  No checkpoints found"
echo ""
echo "--- Configuration Used ---"
echo "Run name:      $RUN_NAME"
echo "Batch size:    128"
echo "Epochs:        100"
echo "Latent dim:    128"
echo "RNN hidden:    512"
echo "RNN layers:    2"
echo "Image size:    192x640"
echo ""
echo "--- Best Models ---"
echo "VAE:     outputs/$RUN_NAME/checkpoints/vae_final.pth"
echo "MDN-RNN: outputs/$RUN_NAME/rnn_checkpoints/rnn_best.pth"
echo ""
echo "Training pipeline completed successfully!"
