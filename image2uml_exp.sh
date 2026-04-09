#!/bin/bash
# ==========================================================
# Image → UML Experiment Runner - SLURM Job Script
# ==========================================================
# Starts a vLLM server (Qwen/Qwen3-VL-4B-Instruct, 1xH100),
# then runs image2uml.py against it.
#
# Usage: sbatch image2uml_exp.sh
# ==========================================================

#SBATCH --job-name=image2uml-vllm
#SBATCH --account=def-syriani
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:1                  # 4B model fits nicely on a single H100
#SBATCH --cpus-per-task=12                      # sufficient CPUs
#SBATCH --output=/scratch/gauransh/logs/image2uml-%j.out
#SBATCH --error=/scratch/gauransh/logs/image2uml-%j.err
#SBATCH --mail-user=gauransh.kumar@umontreal.ca # Set properly
#SBATCH --mail-type=ALL

# ==========================================================
# Configuration
# ==========================================================
export TENSOR_PARALLEL_SIZE=1                   # Fits on 1 GPU
export VLLM_PORT=8000
export VLLM_MODEL="Qwen/Qwen3-VL-4B-Instruct"   # The Vision language model
export LOG_DIR="/scratch/gauransh/logs"

# Ensure vLLM loads with proper vision handling limits (optional depending on exact package but safe to override)
export VLLM_ATTENTION_BACKEND="FLASH_ATTN"

# ==========================================================
# Safety: Enable error handling
# ==========================================================
set -e
set -u
set -o pipefail

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "Image → UML Experiment Starting"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID:-local}"
echo "Node: $(hostname)"
echo "Model: $VLLM_MODEL"
echo "GPUs: $TENSOR_PARALLEL_SIZE × H100"
echo "Start time: $(date)"
echo "=========================================="

# ==========================================================
# Step 1: Setup vLLM and start the inference server
# ==========================================================
echo "[INFO] Setting up vLLM server..."
bash "$SLURM_SUBMIT_DIR/setup_vllm.sh"

# ==========================================================
# Step 2: Run image2uml.py
# ==========================================================
# Temporarily disable set -e so cleanup always runs even on failure
set +e
bash "$SLURM_SUBMIT_DIR/run_image2uml.sh"
EXPERIMENT_EXIT_CODE=$?
set -e

# ==========================================================
# Step 3: Cleanup
# ==========================================================
echo "=========================================="
echo "Shutting down vLLM server..."
echo "=========================================="
pkill -f "vllm.entrypoints.openai.api_server" || true

echo "=========================================="
echo "Experiment completed"
echo "End time: $(date)"
echo "Exit code: $EXPERIMENT_EXIT_CODE"
echo "=========================================="

exit $EXPERIMENT_EXIT_CODE
