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
# SmolLM-135M: 135 MB, uses LlamaForCausalLM — supported in all vLLM versions including 0.5.x
MODEL_NAME=${VLLM_MODEL:-"HuggingFaceTB/SmolLM-135M"}
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
pip install --no-index tqdm || pip install tqdm

# ==========================================================
# 3b) vLLM engine selection
# ==========================================================
# vLLM v1 engine's topk_topp_triton.py mixes uint32/int32 in a Triton
# kernel division, which Triton 3.0.0's stricter compiler rejects with:
#   "Cannot use // with triton.language.uint32 and triton.language.int32"
# Force the v0 engine which uses a plain PyTorch sampler path instead.
export VLLM_USE_V1=0

# ==========================================================
# 3c) MIG UUID remapping
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
model_name = os.environ.get("VLLM_MODEL_NAME", "HuggingFaceTB/SmolLM-135M")

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

# ==========================================================
# 6) HTTP Server Test (mirrors setup_vllm.sh steps 7-8)
# ==========================================================
echo "[INFO] Testing vLLM HTTP server startup (mirrors setup_vllm.sh)..."

TEST_PORT=8765   # separate port to avoid conflicts with any live server
SERVER_LOG="$TMP_DIR/vllm_server_test.log"
MAX_SERVER_WAIT=300  # 5 minutes — enough for the tiny opt-125m model

export HF_HOME="$DOWNLOAD_DIR"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_NAME" \
    --host 127.0.0.1 \
    --port "$TEST_PORT" \
    --max-model-len 512 \
    --enforce-eager \
    > "$SERVER_LOG" 2>&1 &
SERVER_PID=$!
echo "[INFO] Test vLLM server started with PID: $SERVER_PID"

# Wait for health endpoint
ELAPSED=0
SERVER_READY=0
echo -n "[INFO] Waiting for server health"
while [ $ELAPSED -lt $MAX_SERVER_WAIT ]; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo ""
        echo "[ERROR] vLLM server process died during startup. Last log lines:"
        tail -20 "$SERVER_LOG"
        exit 1
    fi
    if curl -s -f "http://127.0.0.1:$TEST_PORT/health" > /dev/null 2>&1; then
        SERVER_READY=1
        echo ""
        echo "[SUCCESS] vLLM HTTP server is healthy"
        break
    fi
    echo -n "."
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [ $SERVER_READY -eq 0 ]; then
    echo ""
    echo "[ERROR] vLLM server failed to become healthy within ${MAX_SERVER_WAIT}s"
    tail -30 "$SERVER_LOG"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# ==========================================================
# 7) OpenAI-Compatible API Test (mirrors tikz2uml.py call_vllm)
# ==========================================================
echo "[INFO] Testing OpenAI-compatible API — /v1/models and /v1/chat/completions..."

python - << EOF
import json, sys, urllib.request, urllib.error

port = $TEST_PORT
model = "$MODEL_NAME"
base = f"http://127.0.0.1:{port}"

# -- /v1/models --
try:
    with urllib.request.urlopen(f"{base}/v1/models", timeout=10) as r:
        models = json.loads(r.read())
    print("[TEST] /v1/models OK —", [m["id"] for m in models.get("data", [])])
except Exception as e:
    print(f"[ERROR] /v1/models failed: {e}")
    sys.exit(1)

# -- /v1/chat/completions (mirrors call_vllm in tikz2uml.py) --
payload = json.dumps({
    "model": model,
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",   "content": "Say the word hello."},
    ],
    "temperature": 0.0,
    "max_tokens": 10,
}).encode("utf-8")

req = urllib.request.Request(
    f"{base}/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)
try:
    with urllib.request.urlopen(req, timeout=30) as r:
        resp = json.loads(r.read())
    reply = resp["choices"][0]["message"]["content"].strip()
    print(f"[TEST] /v1/chat/completions OK — reply: {reply!r}")
except urllib.error.HTTPError as e:
    print(f"[ERROR] /v1/chat/completions HTTP {e.code}: {e.reason}")
    sys.exit(1)
except Exception as e:
    print(f"[ERROR] /v1/chat/completions failed: {e}")
    sys.exit(1)

print("[SUCCESS] OpenAI-compatible API is working correctly")
EOF

# Cleanup test server
echo "[INFO] Stopping test server (PID: $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true
echo "[INFO] Test server stopped"

echo "=========================================="
echo "[SUCCESS] vLLM test script finished successfully!"
echo "Everything seems to be installed fine."
echo "=========================================="
