#!/bin/bash
#SBATCH --job-name=export_test_split
#SBATCH --account=def-syriani
#SBATCH --time=0:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/scratch/gauransh/logs/export_test_split_%j.out
#SBATCH --error=/scratch/gauransh/logs/export_test_split_%j.err

# No GPU requested — this is a pure CPU/pandas/HuggingFace job.

set -e
set -o pipefail

mkdir -p /scratch/gauransh/logs
LOG_FILE="/scratch/gauransh/logs/export_test_split_$(date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee -i "$LOG_FILE") 2>&1

echo "=========================================================="
echo "Export Test Split (CPU-only)"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo "=========================================================="

# ── Modules (arrow MUST be loaded before venv activation) ────────────────────
echo "[INFO] Loading modules..."
module load StdEnv/2023    2>/dev/null || echo "[WARN] StdEnv/2023 not found"
module load gcc            2>/dev/null || echo "[WARN] gcc not found"
module load arrow/17.0.0   2>/dev/null || echo "[WARN] arrow/17.0.0 not found"
module load python/3.11    2>/dev/null || echo "[WARN] python/3.11 not found"
# No cuda module needed — CPU only

# ── Virtual environment ───────────────────────────────────────────────────────
ENV_DIR="${SLURM_TMPDIR:-/tmp}/export_split_env"
echo "[INFO] Building virtual environment -> $ENV_DIR"
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

pip install --upgrade pip wheel setuptools --quiet

# torch (CPU-only build is fine for data processing — much smaller download)
pip install --no-index torch || pip install torch --index-url https://download.pytorch.org/whl/cpu

echo "[INFO] Installing data + HuggingFace packages..."
pip install --no-index pyarrow || true
pip install pandas "datasets<=2.19.0" transformers pillow python-dotenv

# ── Run export ────────────────────────────────────────────────────────────────
PROJECT_DIR="/project/def-syriani/gauransh/ift6765"
echo "[INFO] Jumping to project directory..."
cd "$PROJECT_DIR"

echo "[INFO] Running export-test-split..."
python finetune_vlm.py --export-test-split

echo "=========================================================="
echo "[SUCCESS] Test split exported."
echo "Find it at: ${PROJECT_DIR}/output/qwen_lora_test_split.parquet"
echo "End: $(date)"
echo "=========================================================="
