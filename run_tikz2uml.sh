#!/bin/bash
# ==========================================================
# TikZ → UML Experiment Execution Script
# ==========================================================
# Prerequisites:
#   - vLLM server running (started by setup_vllm.sh)
#   - Project checked out at BASE_DIR
#
# Environment variables (set by tikz2uml_exp.sh):
#   VLLM_PORT - Port where vLLM server is running (default: 8000)
#   LOG_DIR   - Directory for logs (default: /scratch/gauransh/logs)
# ==========================================================

set -e
set -u
set -o pipefail

VLLM_PORT=${VLLM_PORT:-8000}
LOG_DIR=${LOG_DIR:-"/scratch/gauransh/logs"}

BASE_DIR="/project/def-syriani/gauransh/ift6765"
VENV_DIR="$BASE_DIR/.venv"

echo "=========================================="
echo "TikZ → UML Experiment Configuration"
echo "=========================================="
echo "Base directory : $BASE_DIR"
echo "Log directory  : $LOG_DIR"
echo "vLLM endpoint  : http://localhost:$VLLM_PORT/v1"
echo "=========================================="

# ==========================================================
# 1) Load modules needed for the experiment (arrow for parquet)
# ==========================================================
echo "[INFO] Loading required modules..."
module load StdEnv/2023
module load gcc
module load arrow/17.0.0
module load python/3.11

# ==========================================================
# 2) Set up Python virtual environment
# ==========================================================
if [ ! -d "$VENV_DIR" ]; then
    echo "[INFO] Creating virtual environment at $VENV_DIR..."
    python -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
echo "[INFO] Python: $(which python)"

echo "[INFO] Installing dependencies..."
pip install --quiet --upgrade pip
pip install --quiet --no-index pyarrow pandas
pip install --quiet --no-index tqdm || pip install --quiet tqdm

# ==========================================================
# 3) Verify vLLM server is accessible
# ==========================================================
echo "[INFO] Verifying vLLM server connectivity..."
MAX_RETRIES=5
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s -f "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
        echo "[SUCCESS] vLLM server is accessible"
        break
    else
        RETRY_COUNT=$((RETRY_COUNT + 1))
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo "[WARNING] vLLM not ready, retrying in 15s ($RETRY_COUNT/$MAX_RETRIES)..."
            sleep 15
        else
            echo "[ERROR] vLLM server not accessible at http://localhost:$VLLM_PORT"
            exit 1
        fi
    fi
done

# ==========================================================
# 4) Run tikz2uml.py
# ==========================================================
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/tikz2uml_${TIMESTAMP}.log"

echo "=========================================="
echo "Running tikz2uml.py..."
echo "Log: $LOG_FILE"
echo "=========================================="

cd "$BASE_DIR"

if python tikz2uml.py 2>&1 | tee -a "$LOG_FILE"; then
    echo "=========================================="
    echo "[SUCCESS] Conversion completed!"
    echo "[INFO] Results log: $LOG_FILE"
    echo "=========================================="
    exit 0
else
    EXIT_CODE=$?
    echo "=========================================="
    echo "[ERROR] tikz2uml.py failed (exit code: $EXIT_CODE)"
    echo "[ERROR] Check: $LOG_FILE"
    echo "=========================================="
    exit $EXIT_CODE
fi
