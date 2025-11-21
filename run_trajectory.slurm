#!/bin/bash
#SBATCH --job-name=trajectory-eval
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --constraint="a100"
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/trajectory_%j.out
#SBATCH --error=logs/trajectory_%j.err

echo "Running trajectory evaluation"
echo "Job ID: $SLURM_JOB_ID"

cd /cluster/work/emilkl/prob-gen-project

nvidia-smi

uv run python -m src.utils.trajectory

echo "Done - check trajectory_result.png"
