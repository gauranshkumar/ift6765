#!/bin/bash
# ==========================================================
# Test vLLM Setup Script for Compute Canada
# ==========================================================
# This script:
# 1. Loads required modules (CUDA, Python, etc.)
# 2. Creates a virtual environment in SLURM_TMPDIR or /tmp
# 3. Installs PyTorch, Triton, and vLLM
# 4. Authenticates or tests installation
# 5. Runs a simple offline vLLM generation test with a tiny model
#    to verify the installation is working correctly.
# ==========================================================

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# ==========================================================
# Configuration
# ==========================================================
# We use a very small model for testing to avoid huge downloads and long loading times
MODEL_NAME=${VLLM_MODEL:-"facebook/opt-125m"}
DOWNLOAD_DIR="/scratch/gauransh/hf_models"
TMP_DIR="${SLURM_TMPDIR:-/tmp/vllm_test_$$}"

echo "=========================================="
echo "vLLM Test Setup Configuration"
echo "=========================================="
echo "Test Model: $MODEL_NAME"
echo "Download Dir: $DOWNLOAD_DIR"
echo "Temp Dir: $TMP_DIR"
echo "=========================================="

# ==========================================================
# 1) Load Compute Canada modules
# ==========================================================
echo "[INFO] Loading Compute Canada modules..."
# We use || true to prevent script from crashing if tested on a non-Compute Canada machine
module load StdEnv/2023 2>/dev/null || echo "[WARN] Module StdEnv not found (maybe not on CC?)"
module load cuda/12.2 2>/dev/null || echo "[WARN] Module cuda/12.2 not found"
module load python/3.11 2>/dev/null || echo "[WARN] Module python/3.11 not found"
module load gcc 2>/dev/null || true
module load opencv 2>/dev/null || true

# Add modules from run_tikz2uml.sh
module load arrow/17.0.0 2>/dev/null || echo "[WARN] Module arrow/17.0.0 not found"

echo "[INFO] Job running on: $(hostname)"
if command -v nvidia-smi &> /dev/null; then
    echo "[INFO] Available GPUs:"
    nvidia-smi -L | wc -l
else
    echo "[WARN] nvidia-smi not found. Are you on a GPU node?"
fi

# ==========================================================
# 2) Create venv
# ==========================================================
echo "[INFO] Creating virtual environment at $TMP_DIR..."
mkdir -p "$TMP_DIR"
ENV_DIR="$TMP_DIR/vllm-test-env"
python3 -m venv "$ENV_DIR"

# Activate venv
source "$ENV_DIR/bin/activate"
echo "[INFO] Python from venv: $(which python)"

# ==========================================================
# 3) Install PyTorch and vLLM
# ==========================================================
echo "[INFO] Upgrading pip and installing build tools..."
pip install --upgrade pip wheel setuptools

echo "[INFO] Installing PyTorch..."
# On compute canada this uses local wheels if available
pip install --no-index torch torchvision torchaudio || pip install torch torchvision torchaudio

echo "[INFO] Installing compatible Triton version..."
pip install --upgrade "triton==3.0.0"

echo "[INFO] Installing vLLM..."
pip install "vllm>=0.5.1"

echo "[INFO] Installing dependencies from run_tikz2uml.sh..."
pip install --no-index pyarrow pandas || pip install pyarrow pandas
pip install tqdm

# ==========================================================
# 4) Basic Validation
# ==========================================================
echo "[INFO] Validating basic imports and CUDA..."
python - << 'EOF'
import torch
import triton
import vllm
import pyarrow as pa
import pandas as pd
import tqdm
print("Torch:", torch.__version__)
print("Triton:", triton.__version__)
print("vLLM:", vllm.__version__)
print("PyArrow:", pa.__version__)
print("Pandas:", pd.__version__)
print("tqdm:", tqdm.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Number of GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:", torch.cuda.get_device_name(i))
EOF

# ==========================================================
# 5) End-to-end Offline Inference Test
# ==========================================================
echo "[INFO] Running end-to-end offline inference test..."
export HF_HOME="$DOWNLOAD_DIR"

cat << 'EOF' > "$TMP_DIR/test_inference.py"
import os
import sys
from vllm import LLM, SamplingParams

# Get model name from environment or use default
model_name = os.environ.get("VLLM_MODEL_NAME", "facebook/opt-125m")

print(f"[TEST] Loading model {model_name}...")
try:
    # Initialize the LLM
    # Setting enforcement to false and small max_model_len to fit gracefully in tight test GPUs
    llm = LLM(model=model_name, max_model_len=512, enforce_eager=True)
    
    # Prepare promising prompts
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
    ]
    
    # Create sampling params
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=30)
    
    # Generate texts
    print("[TEST] Generating text...")
    outputs = llm.generate(prompts, sampling_params)
    
    # Print outputs
    print("\n[TEST] Generation Results:")
    print("-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}")
        print(f"Generated text: {generated_text!r}")
        print("-" * 60)
        
    print("\n[SUCCESS] Inference output looks good!")
except Exception as e:
    print(f"\n[TEST ERROR] Inference failed with error: {e}")
    sys.exit(1)
EOF

VLLM_MODEL_NAME=$MODEL_NAME python "$TMP_DIR/test_inference.py"

echo "=========================================="
echo "[SUCCESS] vLLM test script finished successfully!"
echo "Everything seems to be installed fine."
echo "=========================================="
