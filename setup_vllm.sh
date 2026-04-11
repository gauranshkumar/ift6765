#!/bin/bash
# ==========================================================
# vLLM Setup Script for Compute Canada
# ==========================================================
# This script:
# 1. Loads required modules (CUDA, Python, etc.)
# 2. Creates a virtual environment on local SSD
# 3. Installs PyTorch, Triton, and vLLM
# 4. Starts the vLLM OpenAI-compatible API server
# 5. Waits for server to be healthy before returning
#
# Environment variables (set by promptslr_exp.sh):
#   TENSOR_PARALLEL_SIZE - Number of GPUs for tensor parallelism (default: 1)
#   VLLM_PORT - Port for vLLM server (default: 8000)
# ==========================================================

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# ==========================================================
# Configuration
# ==========================================================
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
VLLM_PORT=${VLLM_PORT:-8000}
LOG_DIR=${LOG_DIR:-"/scratch/gauransh/logs"}
MODEL_NAME=${VLLM_MODEL:-"Qwen/Qwen3-Coder-Next"}
DOWNLOAD_DIR="/scratch/gauransh/hf_models"
MAX_WAIT_TIME=3600  # Maximum seconds to wait for server (1 hour)
RUN_ID="${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}"
VLLM_LOG="$LOG_DIR/vllm_server_${RUN_ID}.log"

echo "=========================================="
echo "vLLM Setup Configuration"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Tensor Parallel Size: $TENSOR_PARALLEL_SIZE"
echo "Port: $VLLM_PORT"
echo "Download Dir: $DOWNLOAD_DIR"
echo "=========================================="

# ==========================================================
# 1) Load Compute Canada modules
# ==========================================================
echo "[INFO] Loading Compute Canada modules..."
module load StdEnv/2023
module load cuda/12.2
module load python/3.11
module load gcc
module load opencv || true

echo "[INFO] Job running on: $(hostname)"
echo "[INFO] SLURM_TMPDIR: $SLURM_TMPDIR"
echo "[INFO] Available GPUs: $(nvidia-smi -L | wc -l)"
echo "[INFO] CUDA version (nvidia-smi): $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1) / CUDA $(nvidia-smi | grep -oP 'CUDA Version: \K[0-9.]+')"
echo "[INFO] NVCC version: $(nvcc --version | grep release | grep -oP 'release \K[0-9.]+')"

# ==========================================================
# 2) Create venv inside local SSD for max speed
# ==========================================================
echo "[INFO] Creating virtual environment..."
ENV_DIR="$SLURM_TMPDIR/vllm-env"
python -m venv "$ENV_DIR"

# Activate venv
source "$ENV_DIR/bin/activate"
echo "[INFO] Python from venv: $(which python)"

# ==========================================================
# 3) Install PyTorch from Compute Canada wheelhouse
#    (pip auto-detects wheelhouse, no WHEEL_DIR needed)
# ==========================================================
echo "[INFO] Upgrading pip and installing build tools..."
pip install --upgrade pip wheel setuptools

echo "[INFO] Installing PyTorch from Compute Canada wheels..."
pip install --no-index torch torchvision torchaudio

# ==========================================================
# 4) Install vLLM (brings its own compatible Triton)
# ==========================================================
# Do NOT pre-install a pinned Triton. vLLM pins its own Triton dependency
# internally; overriding it causes kernel type-checking failures (e.g.
# uint32/int32 signedness errors in topk_topp_triton.py).
echo "[INFO] Installing vLLM..."
pip install "vllm>=0.5.1" "flashinfer-jit-cache" 

# ==========================================================
# 5b) vLLM engine selection and attention backend
# ==========================================================
# vLLM v1 engine's topk_topp_triton.py mixes uint32/int32 in a Triton
# kernel division, which Triton 3.0.0's stricter compiler rejects with:
#   "Cannot use // with triton.language.uint32 and triton.language.int32"
# Force the v0 engine which uses a plain PyTorch sampler path instead.
# export VLLM_USE_V1=0

# FlashInfer requires a JIT build step that fails on Compute Canada.
# FLASH_ATTN ships pre-built inside the vLLM wheel and works on H100s
# without any compilation — use it instead.
# export VLLM_ATTENTION_BACKEND=FLASH_ATTN
# Disable GDN-specific flashinfer kernel
# export VLLM_USE_FLASHINFER_GDN=0

# Or force pure PyTorch fallback for linear attn
# export VLLM_MAMBA_USE_PYTORCH_FALLBACK=1

# ==========================================================
# 5c) MIG UUID remapping
# ==========================================================
# vLLM calls int() on CUDA_VISIBLE_DEVICES entries; MIG UUIDs like
# "MIG-f0fb8b1e-..." are not integers and cause a ValueError in
# vllm/platforms/cuda.py:device_id_to_physical_device_id.
# Remap each UUID to its ordinal index (0, 1, 2, ...) so vLLM
# receives plain integers while SLURM still isolates the right GPU.
if echo "${CUDA_VISIBLE_DEVICES:-}" | grep -q "MIG-"; then
    NUM_DEVS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l | tr -d ' ')
    NEW_IDS=$(seq -s',' 0 $((NUM_DEVS - 1)))
    export CUDA_VISIBLE_DEVICES="$NEW_IDS"
    echo "[INFO] MIG UUIDs detected — remapped CUDA_VISIBLE_DEVICES to: $CUDA_VISIBLE_DEVICES"
fi

# ==========================================================
# 6) Validation

# ==========================================================
echo "[INFO] Validating installation..."
python - << 'EOF'
import torch, vllm, triton, transformers

print("Torch:          ", torch.__version__)
print("Torch CUDA:     ", torch.version.cuda)
print("Triton:         ", triton.__version__)
print("vLLM:           ", vllm.__version__)
print("Transformers:   ", transformers.__version__)

try:
    import flashinfer
    print("FlashInfer:     ", flashinfer.__version__)
except ImportError:
    print("FlashInfer:      not installed")

print("CUDA available: ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Number of GPUs: ", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}:       ", torch.cuda.get_device_name(i))
EOF

echo "[INFO] vLLM setup finished successfully!"

# ==========================================================
# 7) Start vLLM server in background
# ==========================================================
echo "[INFO] Starting vLLM OpenAI-compatible API server..."
echo "[INFO] This may take several minutes to download and load the model..."

# Create logs directory if needed
mkdir -p "$LOG_DIR"

# Start server in background with proper logging
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --host 0.0.0.0 \
    --port $VLLM_PORT \
    --download-dir "$DOWNLOAD_DIR" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    # --max-model-len 32768 \
    # --dtype bfloat16 \
    # --disable-custom-all-reduce \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "[INFO] vLLM server started with PID: $VLLM_PID"
echo $VLLM_PID > "$LOG_DIR/vllm_server.pid"

# ==========================================================
# 8) Wait for server to be healthy
# ==========================================================
echo "[INFO] Waiting for vLLM server to be ready (timeout: ${MAX_WAIT_TIME}s)..."

ELAPSED=0
HEALTH_CHECK_URL="http://localhost:$VLLM_PORT/health"

while [ $ELAPSED -lt $MAX_WAIT_TIME ]; do
    # Check if process is still running
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "[ERROR] vLLM server process died! Check $VLLM_LOG"
        exit 1
    fi

    # Check health endpoint
    if curl -s -f "$HEALTH_CHECK_URL" > /dev/null 2>&1; then
        echo "[SUCCESS] vLLM server is ready!"
        echo "[INFO] Server endpoint: http://localhost:$VLLM_PORT/v1"
        echo "[INFO] API docs: http://localhost:$VLLM_PORT/docs"
        echo "[INFO] Server logs: $VLLM_LOG"
        echo "[INFO] Run ID: $RUN_ID"
        echo "=========================================="
        exit 0
    fi

    echo -n "."
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

# Timeout reached
echo ""
echo "[ERROR] vLLM server failed to become healthy within ${MAX_WAIT_TIME}s"
echo "[ERROR] Check $VLLM_LOG for details"
tail -50 "$VLLM_LOG"
exit 1
