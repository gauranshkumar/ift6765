#!/bin/bash
#SBATCH --job-name=qwen_vlm_benchmark
#SBATCH --account=def-syriani
#SBATCH --time=6:00:00
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
echo "VLM Benchmark Runner — base vs finetuned"
echo "Job ID : ${SLURM_JOB_ID:-local}"
echo "Node   : $(hostname)"
echo "Start  : $(date)"
echo "=========================================================="

# ── Module loading ────────────────────────────────────────────────────────────
echo "[INFO] Loading modules..."
module --force purge
module load StdEnv/2023    2>/dev/null || echo "[WARN] StdEnv/2023 not found"
module load gcc            2>/dev/null || echo "[WARN] gcc not found"
module load cuda/12.2      2>/dev/null || echo "[WARN] cuda/12.2 not found"
module load arrow/17.0.0   2>/dev/null || echo "[WARN] arrow/17.0.0 not found"
module load opencv         2>/dev/null || echo "[WARN] opencv not found"
module load scipy-stack    2>/dev/null || echo "[WARN] scipy-stack not found"
module load python/3.11    2>/dev/null || echo "[WARN] python/3.11 not found"

# ── Virtual environment ───────────────────────────────────────────────────────
ENV_DIR="${SLURM_TMPDIR:-/tmp}/vlm_benchmark_env"
echo "[INFO] Building virtual environment -> $ENV_DIR"
python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

pip install --upgrade pip wheel setuptools --quiet

# 1. scipy-stack packages — install from CC's local wheels (same binaries the
#    module provides, so no ABI mismatch with anything else CC compiled).
pip install --no-index numpy scipy pandas

# 2. opencv — CC blocks pip-install with an intentionally-failing dummy tarball;
#    the real cv2 is already on PYTHONPATH from the loaded module.
#    Stub the dist-info so every subsequent pip resolver step sees it as satisfied.
python3 - <<'PYSTUB'
import site, os
di = os.path.join(site.getsitepackages()[0], "opencv_python_headless-4.13.0.dist-info")
os.makedirs(di, exist_ok=True)
open(os.path.join(di, "METADATA"),  "w").write(
    "Metadata-Version: 2.1\nName: opencv-python-headless\nVersion: 4.13.0\n")
open(os.path.join(di, "INSTALLER"), "w").write("pip\n")
open(os.path.join(di, "RECORD"),    "w").write("")
print(f"[stub] opencv-python-headless 4.13.0 → {di}")
PYSTUB

# 3. torch — must come from CC's local wheel (CUDA-compiled for CC hardware).
pip install --no-index torch torchvision torchaudio || pip install torch torchvision torchaudio

# 4. pyarrow — must come from CC's local wheel (linked against arrow/17.0.0 module).
pip install --no-index pyarrow || true

# 5. vllm + everything else — latest from PyPI.
#    Latest vllm natively supports Qwen3-VL and is written for current transformers,
#    eliminating all the version-pinning and monkey-patching we needed with 0.8.x.
pip install vllm transformers peft accelerate pillow python-dotenv

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_DIR="/project/def-syriani/gauransh/ift6765"
BASE_MODEL="Qwen/Qwen3-VL-4B-Instruct"
FINETUNED_MODEL="${PROJECT_DIR}/output/qwen_merged_final"
TEST_PARQUET="${PROJECT_DIR}/output/qwen_lora_test_split.parquet"
RESULTS_BASE="${PROJECT_DIR}/output/benchmark_results_base.parquet"
RESULTS_FT="${PROJECT_DIR}/output/benchmark_results_finetuned.parquet"
VLLM_PORT=8100

# ── Step 1: Merge LoRA if merged model doesn't exist yet ─────────────────────
if [ ! -d "$FINETUNED_MODEL" ]; then
    echo "[INFO] Merged model not found. Running merge_lora.py..."
    LORA_PATH="${PROJECT_DIR}/output/qwen_lora_finetuned/final"
    python "${PROJECT_DIR}/merge_lora.py" \
        --lora-path   "$LORA_PATH" \
        --output-dir  "$FINETUNED_MODEL"
else
    echo "[INFO] Using existing merged model at $FINETUNED_MODEL"
fi

# ── Helper: start vLLM, wait for health, run benchmark, stop ─────────────────
# Usage: run_model_benchmark <model_path_or_id> <served_name> <results_parquet> <log_file>
run_model_benchmark() {
    local MODEL_PATH="$1"
    local SERVED_NAME="$2"
    local RESULTS_PATH="$3"
    local VLLM_LOG="$4"

    echo ""
    echo "=========================================================="
    echo "[INFO] Benchmarking: $SERVED_NAME"
    echo "  model  : $MODEL_PATH"
    echo "  results: $RESULTS_PATH"
    echo "=========================================================="

    # Start vLLM server
    python -m vllm.entrypoints.openai.api_server \
        --model               "$MODEL_PATH" \
        --served-model-name   "$SERVED_NAME" \
        --host                127.0.0.1 \
        --port                "$VLLM_PORT" \
        --tensor-parallel-size 2 \
        --dtype               bfloat16 \
        --trust-remote-code \
        --max-model-len       8192 \
        > "$VLLM_LOG" 2>&1 &
    local SERVER_PID=$!
    echo "[INFO] vLLM PID: $SERVER_PID"

    # Wait for health
    echo -n "[INFO] Waiting for vLLM health"
    local MAX_WAIT=300 ELAPSED=0
    while [ $ELAPSED -lt $MAX_WAIT ]; do
        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo ""
            echo "[ERROR] vLLM server died. Last log:"
            tail -20 "$VLLM_LOG"
            return 1
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
        return 1
    fi

    # Run benchmark
    set +e
    python "${PROJECT_DIR}/benchmark_vlm.py" \
        --test-parquet  "$TEST_PARQUET" \
        --output-path   "$RESULTS_PATH" \
        --endpoint      "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" \
        --model-name    "$SERVED_NAME" \
        --workers       8
    local BENCH_EXIT=$?
    set -e

    # Stop server
    echo "[INFO] Stopping vLLM server (PID $SERVER_PID)..."
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo "[INFO] Server stopped."

    return $BENCH_EXIT
}

# ── Step 2: Benchmark base model ─────────────────────────────────────────────
VLLM_LOG_BASE="/scratch/gauransh/logs/vllm_base_${SLURM_JOB_ID:-local}.log"
set +e
run_model_benchmark \
    "$BASE_MODEL" \
    "base" \
    "$RESULTS_BASE" \
    "$VLLM_LOG_BASE"
EXIT_BASE=$?
set -e
echo "[INFO] Base model benchmark exit code: $EXIT_BASE"

# ── Step 3: Benchmark finetuned model ────────────────────────────────────────
VLLM_LOG_FT="/scratch/gauransh/logs/vllm_finetuned_${SLURM_JOB_ID:-local}.log"
set +e
run_model_benchmark \
    "$FINETUNED_MODEL" \
    "finetuned" \
    "$RESULTS_FT" \
    "$VLLM_LOG_FT"
EXIT_FT=$?
set -e
echo "[INFO] Finetuned model benchmark exit code: $EXIT_FT"

# ── Final comparison summary ──────────────────────────────────────────────────
echo ""
RESULTS_BASE="$RESULTS_BASE" RESULTS_FT="$RESULTS_FT" python - <<'PYEOF'
import sys, os
import pandas as pd

base_path = os.environ.get("RESULTS_BASE", "")
ft_path   = os.environ.get("RESULTS_FT", "")

def stats(path, label):
    if not os.path.exists(path):
        print(f"  {label}: results file not found ({path})")
        return None
    df = pd.read_parquet(path)
    n        = len(df)
    n_valid  = df["uml_valid"].sum()
    n_failed = df["llm_failed"].sum()
    n_invalid = n - n_valid - n_failed
    return dict(label=label, n=n, valid=n_valid, failed=n_failed, invalid=n_invalid)

s_base = stats(base_path, "base")
s_ft   = stats(ft_path,   "finetuned")

rows = [r for r in [s_base, s_ft] if r]
if not rows:
    print("No result files to compare.")
    sys.exit(0)

print("=" * 62)
print(f"{'':20s} {'base':>12s} {'finetuned':>12s}  {'delta':>8s}")
print("=" * 62)
for key, label in [("valid", "Valid PlantUML"), ("invalid", "Invalid"), ("failed", "LLM failures")]:
    vals = {r["label"]: r[key] for r in rows}
    ns   = {r["label"]: r["n"]  for r in rows}
    b    = vals.get("base",      0); bn = ns.get("base",      1)
    f    = vals.get("finetuned", 0); fn = ns.get("finetuned", 1)
    bp   = 100 * b / bn if bn else 0
    fp   = 100 * f / fn if fn else 0
    delta = fp - bp
    sign  = "+" if delta >= 0 else ""
    print(f"  {label:18s} {b:4d} ({bp:5.1f}%)  {f:4d} ({fp:5.1f}%)  {sign}{delta:+.1f}pp")
print("=" * 62)
PYEOF

echo ""
echo "=========================================================="
echo "ALL BENCHMARKS COMPLETE"
echo "  Base model results   : $RESULTS_BASE  (exit $EXIT_BASE)"
echo "  Finetuned results    : $RESULTS_FT  (exit $EXIT_FT)"
echo "End: $(date)"
echo "=========================================================="

# Exit non-zero if either run failed
FINAL_EXIT=$(( EXIT_BASE > EXIT_FT ? EXIT_BASE : EXIT_FT ))
exit $FINAL_EXIT
