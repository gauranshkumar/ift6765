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
echo "Starting VLM QLoRA Fine-tuning Setup (2x GPUs)"
echo "=========================================================="

echo "[INFO] Loading Compute Canada core modules..."
module load StdEnv/2023    2>/dev/null || echo "[WARN] StdEnv/2023 not found"
module load gcc            2>/dev/null || echo "[WARN] gcc not found"
module load cuda/12.2      2>/dev/null || echo "[WARN] cuda/12.2 not found"
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
# Prioritizing fast local ComputeCanada wheels first, falling back to network
pip install --no-index torch torchvision torchaudio || pip install torch torchvision torchaudio

echo "[INFO] Linking isolated ComputeCanada Modules (PyArrow/Pandas)..."
pip install --no-index pyarrow pandas || pip install pyarrow pandas

echo "[INFO] Installing HuggingFace distributed training frameworks + WandB..."
# bitsandbytes handles massive multi-GPU scaling structures even without quantization
pip install transformers peft trl datasets accelerate bitsandbytes pillow wandb python-dotenv

echo "[INFO] Jumping to execution directory..."
cd /project/def-syriani/gauransh/ift6765

echo "=========================================================="
echo "Triggering Distributed Accelerated Execution "
echo "=========================================================="
# Accelerate will natively detect the 2 GPUs implicitly due to device_map and DP setups
accelerate launch finetune_vlm.py --epochs 8 --batch-size 8

echo "=========================================================="
echo "[SUCCESS] Script Execution Fully Complete"
echo "=========================================================="
