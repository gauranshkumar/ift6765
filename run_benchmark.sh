#!/bin/bash
#SBATCH --job-name=qwen_vlm_benchmark
#SBATCH --account=def-syriani
#SBATCH --time=3:00:00
#SBATCH --nodes=1
#SBATCH --gpus-per-node=h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=/scratch/gauransh/logs/benchmark_vlm_%j.out
#SBATCH --error=/scratch/gauransh/logs/benchmark_vlm_%j.err

set -e
set -o pipefail

mkdir -p /scratch/gauransh/logs
LOG_FILE="/scratch/gauransh/logs/benchmark_vlm_exec_$(date +%Y%m%d_%H%M%S)_$$.log"
exec > >(tee -i "$LOG_FILE") 2>&1

echo "=========================================================="
echo "VLM Benchmark Runner"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo "=========================================================="

# ── Module loading (arrow MUST come before venv activation) ──────────────────
echo "[INFO] Loading modules..."
module load StdEnv/2023    2>/dev/null || echo "[WARN] StdEnv/2023 not found"
module load gcc            2>/dev/null || echo "[WARN] gcc not found"
module load cuda/12.2      2>/dev/null || echo "[WARN] cuda/12.2 not found"
module load arrow/17.0.0   2>/dev/null || echo "[WARN] arrow/17.0.0 not found"
# opencv MUST be loaded before venv activation (same pattern as arrow/pyarrow)
module load opencv         2>/dev/null || echo "[WARN] opencv not found"
module load python/3.11    2>/dev/null || echo "[WARN] python/3.11 not found"

# ── Virtual environment ───────────────────────────────────────────────────────
ENV_DIR="${SLURM_TMPDIR:-/tmp}/vlm_benchmark_env"
echo "[INFO] Building virtual environment -> $ENV_DIR"
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

CONSTRAINTS="/project/def-syriani/gauransh/ift6765/constraints.txt"

pip install --upgrade pip wheel setuptools --quiet
pip install --no-index torch torchvision torchaudio || pip install torch torchvision torchaudio
pip install --no-index pyarrow || true
# Use CC's prebuilt vllm wheel — it includes a compatible opencv build internally
pip install --no-index vllm || pip install -c "$CONSTRAINTS" vllm
pip install -c "$CONSTRAINTS" transformers peft accelerate pillow pandas python-dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR="/project/def-syriani/gauransh/ift6765"
MODEL_DIR="${PROJECT_DIR}/output/qwen_merged_final"   # merged LoRA model
TEST_PARQUET="${PROJECT_DIR}/output/qwen_lora_test_split.parquet"
RESULTS_PATH="${PROJECT_DIR}/output/benchmark_results.parquet"
VLLM_PORT=8100
VLLM_LOG="/scratch/gauransh/logs/vllm_benchmark_${SLURM_JOB_ID:-local}.log"

# ── Step 1: Merge LoRA if merged model doesn't exist yet ─────────────────────
if [ ! -d "$MODEL_DIR" ]; then
    echo "[INFO] Merged model not found. Running merge_lora.py..."
    LORA_PATH="${PROJECT_DIR}/output/qwen_lora_finetuned/final"
    python "${PROJECT_DIR}/merge_lora.py" \
        --lora-path   "$LORA_PATH" \
        --output-dir  "$MODEL_DIR"
else
    echo "[INFO] Using existing merged model at $MODEL_DIR"
fi

# ── Step 2: Start vLLM server serving the merged model ───────────────────────
echo "[INFO] Launching vLLM server on port $VLLM_PORT..."

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --host 127.0.0.1 \
    --port "$VLLM_PORT" \
    --tensor-parallel-size 2 \
    --dtype bfloat16 \
    --trust-remote-code \
    > "$VLLM_LOG" 2>&1 &

SERVER_PID=$!
echo "[INFO] vLLM PID: $SERVER_PID"

# Wait for server to be healthy
echo -n "[INFO] Waiting for vLLM health"
MAX_WAIT=300
ELAPSED=0
while [ $ELAPSED -lt $MAX_WAIT ]; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo ""
        echo "[ERROR] vLLM server process died. Last log:"
        tail -20 "$VLLM_LOG"
        exit 1
    fi
    if curl -sf "http://127.0.0.1:${VLLM_PORT}/health" > /dev/null 2>&1; then
        echo " ready!"
        break
    fi
    echo -n "."
    sleep 5
    ELAPSED=$((ELAPSED + 5))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
    echo ""
    echo "[ERROR] vLLM server failed to start within ${MAX_WAIT}s"
    tail -30 "$VLLM_LOG"
    kill $SERVER_PID 2>/dev/null || true
    exit 1
fi

# ── Step 3: Run the benchmark ─────────────────────────────────────────────────
echo "[INFO] Starting benchmark evaluation..."
set +e   # don't exit so cleanup always runs
python "${PROJECT_DIR}/benchmark_vlm.py" \
    --test-parquet  "$TEST_PARQUET" \
    --output-path   "$RESULTS_PATH" \
    --endpoint      "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" \
    --model-name    "$MODEL_DIR" \
    --workers       8
BENCH_EXIT=$?
set -e

# ── Cleanup ───────────────────────────────────────────────────────────────────
echo "[INFO] Stopping vLLM server (PID $SERVER_PID)..."
kill $SERVER_PID 2>/dev/null || true
wait $SERVER_PID 2>/dev/null || true

echo "=========================================================="
echo "Benchmark complete — Exit code: $BENCH_EXIT"
echo "Results saved to: $RESULTS_PATH"
echo "End: $(date)"
echo "=========================================================="

exit $BENCH_EXIT
