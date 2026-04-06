#!/bin/bash
# ==========================================================
# TikZ → UML Experiment Runner - SLURM Job Script
# ==========================================================
# Starts a vLLM server (Qwen/Qwen3-Coder-Next, 3×H100),
# then runs tikz2uml.py against it.
#
# Usage: sbatch tikz2uml_exp.sh
# ==========================================================

#SBATCH --job-name=tikz2uml-vllm
#SBATCH --account=def-syriani
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --gpus-per-node=h100:3
#SBATCH --cpus-per-task=12               # 4 CPUs per GPU
#SBATCH --output=/scratch/gauransh/logs/tikz2uml-%j.out
#SBATCH --error=/scratch/gauransh/logs/tikz2uml-%j.err
#SBATCH --mail-user=gauransh.kumar@umontreal.ca
#SBATCH --mail-type=ALL

# ==========================================================
# Configuration
# ==========================================================
export TENSOR_PARALLEL_SIZE=3
export VLLM_PORT=8000
export VLLM_MODEL="Qwen/Qwen3-Coder-Next"
export LOG_DIR="/scratch/gauransh/logs"

# ==========================================================
# Safety: Enable error handling
# ==========================================================
set -e
set -u
set -o pipefail

mkdir -p "$LOG_DIR"

echo "=========================================="
echo "TikZ → UML Experiment Starting"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "Model: $VLLM_MODEL"
echo "GPUs: $TENSOR_PARALLEL_SIZE × H100"
echo "Start time: $(date)"
echo "=========================================="

# ==========================================================
# Step 1: Setup vLLM and start the inference server
# ==========================================================
echo "[INFO] Setting up vLLM server..."
bash "$(dirname "$0")/setup_vllm.sh"

# ==========================================================
# Step 2: Run tikz2uml.py
# ==========================================================
# Temporarily disable set -e so cleanup always runs even on failure
set +e
bash "$(dirname "$0")/run_tikz2uml.sh"
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
