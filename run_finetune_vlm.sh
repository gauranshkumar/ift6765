#!/bin/bash
#SBATCH --job-name=qwen_vlm_finetune
#SBATCH --account=def-syriani
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/scratch/gauransh/logs/finetune_vlm_%j.out
#SBATCH --error=/scratch/gauransh/logs/finetune_vlm_%j.err

# Exit on first fault
set -e
set -o pipefail

# Setup comprehensive internal logging immediately to catch all module loading or setup tracebacks
mkdir -p /scratch/gauransh/logs
LOG_FILE="/scratch/gauransh/logs/finetune_vlm_exec_$(date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee -i "$LOG_FILE") 2>&1

echo "=========================================================="
echo "Starting VLM QLoRA Fine-tuning Setup (2x H100 GPUs)"
echo "=========================================================="

echo "[INFO] Loading Compute Canada core modules..."
module load StdEnv/2023    2>/dev/null || echo "[WARN] StdEnv/2023 not found"
module load gcc            2>/dev/null || echo "[WARN] gcc not found"
module load cuda/12.2      2>/dev/null || echo "[WARN] cuda/12.2 not found"
# IMPORTANT: arrow MUST be loaded before venv activation so that the CC-provided
# pyarrow wheel is linkable. Loading it after activation causes the dummy-wheel
# build error.
module load arrow/17.0.0   2>/dev/null || echo "[WARN] arrow/17.0.0 not found"
module load python/3.11    2>/dev/null || echo "[WARN] python/3.11 not found"

# Create a highly responsive virtual environment natively in RAM/fast-storage
ENV_DIR="${SLURM_TMPDIR:-/tmp}/vlm_finetune_env"
echo "[INFO] Constructing fresh virtual environment -> $ENV_DIR"
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

echo "[INFO] Upgrading standard build dependencies..."
pip install --upgrade pip wheel setuptools

echo "[INFO] Installing PyTorch..."
pip install --no-index torch torchvision torchaudio || pip install torch torchvision torchaudio

echo "[INFO] Installing pyarrow from CC local wheel (must come before datasets)..."
# --no-index forces pip to use the CC-provided wheel that links against the loaded
# arrow/17.0.0 module. Installing this first means datasets won't try to pull
# pyarrow from PyPI (which triggers the dummy-wheel build error).
pip install --no-index pyarrow

echo "[INFO] Installing HuggingFace stack + WandB..."
pip install transformers peft trl accelerate bitsandbytes pillow wandb python-dotenv datasets

echo "[INFO] Jumping to execution directory..."
cd /project/def-syriani/gauransh/ift6765

echo "=========================================================="
echo "Triggering Distributed Training (2x GPUs, DDP)"
echo "=========================================================="
# --num_processes=2  → one process per GPU (DDP)
# --mixed_precision=bf16 → use bfloat16 amp natively on H100
# Do NOT use device_map="auto" in the Python script when launching this way.
accelerate launch \
    --num_processes=2 \
    --num_machines=1 \
    --mixed_precision=bf16 \
    --dynamo_backend=no \
    finetune_vlm.py --epochs 8 --batch-size 4

echo "=========================================================="
echo "[SUCCESS] Script Execution Fully Complete"
echo "=========================================================="
