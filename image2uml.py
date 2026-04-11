# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas",
#     "pyarrow",
#     "tqdm",
#     "openai",
#     "tenacity",
#     "python-dotenv"
# ]
# ///

import os
import re
import pandas as pd
import glob
import json
import logging
import urllib.error
import argparse
import sys
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from openai import OpenAI
import openai
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_exception_type
from dotenv import load_dotenv

from utils.UML import PlantUMLWebValidator

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

OPENAI_MODEL_NAME = "gpt-4.5-preview"
VLLM_MODEL_NAME   = "Qwen/Qwen3-VL-4B-Instruct"
VLLM_ENDPOINT     = "http://localhost:8000/v1/chat/completions"

if "--no-hpc" in sys.argv:
    BASE_DIR         = "/Tmp/kumargau/ift6765"
    DATA_DIR         = f"{BASE_DIR}/data"
    OUTPUT_DIR       = f"{BASE_DIR}/output"
    LOG_DIR          = f"{BASE_DIR}/logs"
else:
    BASE_DIR         = "/project/def-syriani/gauransh/ift6765"
    DATA_DIR         = f"{BASE_DIR}/data"
    OUTPUT_DIR       = f"{BASE_DIR}/output"
    LOG_DIR          = "/scratch/gauransh/logs"
# Unique run ID — SLURM job ID when running via sbatch, timestamp otherwise
RUN_ID           = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
OUTPUT_PATH      = f"{OUTPUT_DIR}/image2uml_{RUN_ID}.parquet"
CHECKPOINT_PATH  = f"{OUTPUT_DIR}/image2uml_{RUN_ID}_checkpoint.parquet"

REQUEST_TIMEOUT  = 120                        # seconds per LLM call (longer for vision)
PLANTUML_SERVER  = "https://www.plantuml.com/plantuml"
BATCH_SIZE       = 32                         # rows submitted to the server at once

# ─────────────────────────────────────────────────────────────────────────────
# Auth / Client
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    pass

print(api_key[:4] if api_key else "None")
openai_client = OpenAI(api_key=api_key)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{LOG_DIR}/vision_conversion.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert in converting UML diagram sketches to PlantUML code. "
    "Given an image of a UML diagram, output only the equivalent PlantUML source "
    "wrapped in @startuml / @enduml. "
    "Do not include any explanation, markdown fences, or extra text — "
    "only the raw PlantUML source starting with @startuml."
)

def build_messages(image_bytes: bytes) -> list:
    """Builds the message format required for OpenAI-compatible vision endpoints."""
    b64_image = base64.b64encode(image_bytes).decode('utf-8')
    image_url = f"data:image/png;base64,{b64_image}"
    
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": [
            {"type": "text", "text": "Convert this UML sketch into valid PlantUML code."},
            {"type": "input_image", "image_url": {"url": image_url}}
        ]}
    ]

# ─────────────────────────────────────────────────────────────────────────────
# vLLM call
# ─────────────────────────────────────────────────────────────────────────────

def call_vllm(image_bytes: bytes) -> str | None:
    """
    Sends the image to the vLLM OpenAI-compatible vision endpoint.
    Returns the generated PlantUML string, or None on failure.
    """
    payload = json.dumps({
        "model": VLLM_MODEL_NAME,
        "messages": build_messages(image_bytes),
        "temperature": 0.0,
        "max_tokens": 8192,
    }).encode("utf-8")

    req = urllib.request.Request(
        VLLM_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            # Strip any residual <think>…</think> block
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            return content.strip()
    except urllib.error.URLError as e:
        log.error(f"vLLM connection error: {e.reason}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        log.error(f"Unexpected vLLM response format: {e}")
    except Exception as e:
        log.error(f"vLLM call failed: {e}")

    return None

# ─────────────────────────────────────────────────────────────────────────────
# OpenAI call
# ─────────────────────────────────────────────────────────────────────────────

@retry(
    wait=wait_random_exponential(min=1, max=60), 
    stop=stop_after_attempt(6),
    retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError))
)
def call_openai_vision_with_retry(image_bytes: bytes) -> str:
    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL_NAME,
        messages=build_messages(image_bytes),
        temperature=0.0,
        max_tokens=8192,
        timeout=REQUEST_TIMEOUT
    )
    return response.choices[0].message.content


def call_openai_vision(image_bytes: bytes) -> str | None:
    """
    Sends the image to the official OpenAI API endpoint.
    Returns the generated PlantUML string, or None on failure.
    """
    try:
        content = call_openai_vision_with_retry(image_bytes)
        # Strip any residual <think>…</think> block
        content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
        return content.strip()
    except Exception as e:
        log.error(f"OpenAI call failed after retries: {e}")
        return None

# ─────────────────────────────────────────────────────────────────────────────
# Per-row processing
# ─────────────────────────────────────────────────────────────────────────────

validator = PlantUMLWebValidator(server=PLANTUML_SERVER)

def process_row(image_bytes: bytes, provider: str = "vllm") -> dict:
    """
    Returns a dict with keys:
      uml_code        - generated PlantUML string (or empty string on LLM failure)
      uml_valid       - True / False
      uml_error       - error message string, empty if valid
      llm_failed      - True / False
    """
    result = {"uml_code": "", "uml_valid": False, "uml_error": "", "llm_failed": False}

    # ── 1. LLM conversion ────────────────────────────────────────────────────
    if provider == "openai":
        uml_code = call_openai_vision(image_bytes)
    else:
        uml_code = call_vllm(image_bytes)
        
    if uml_code is None:
        result["uml_valid"]  = False
        result["llm_failed"] = True
        result["uml_error"]  = "LLM call failed — see vision_conversion.log"
        log.warning("Skipping validation: LLM did not return output.")
        return result

    result["uml_code"] = uml_code

    # ── 2. PlantUML web validation ───────────────────────────────────────────
    try:
        validation = validator.validate(uml_code)
        result["uml_valid"] = validation["valid"]
        if not validation["valid"]:
            errors = "; ".join(validation.get("errors", []))
            result["uml_error"] = errors
            log.warning(f"UML validation failed: {errors}")
        else:
            log.info("UML validation passed.")
    except Exception as e:
        result["uml_error"] = f"Validator exception: {e}"
        log.error(f"Validator raised an exception: {e}")

    return result

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(provider: str, max_workers: int):
    # ── Load data ─────────────────────────────────────────────────────────────
    # Exclude checkpoint and output files from the input glob
    all_parquet = glob.glob(f"{DATA_DIR}/*.parquet")
    train_files = [
        p for p in all_parquet
        if os.path.basename(p) not in {
            os.path.basename(OUTPUT_PATH),
            os.path.basename(CHECKPOINT_PATH),
        }
    ]
    if not train_files:
        log.error(f"No parquet files found in {DATA_DIR}/")
        return

    log.info(f"Loading {len(train_files)} parquet file(s)...")
    split_dfs = []
    for path in sorted(train_files):
        # Extract split name from filename, e.g. "train-00001-of-00004.parquet" → "train"
        basename = os.path.basename(path)
        split_name = basename.split("-")[0]  # "train" / "test" / "validation"
        part_df = pd.read_parquet(path, columns=["sketch_image", "tikz_code", "tool"])
        part_df["split"] = split_name
        split_dfs.append(part_df)
        log.info(f"  {split_name:12s} — {len(part_df):5d} rows  ({basename})")

    df = pd.concat(split_dfs, ignore_index=True)
    log.info(f"Loaded {df.shape[0]} rows total across {df['split'].nunique()} split(s): "
             f"{sorted(df['split'].unique())}")

    # ── Resume from checkpoint if available ───────────────────────────────────
    image_bytes_list = [row_img["bytes"] for row_img in df["sketch_image"]]
    n_total  = len(image_bytes_list)
    results  = [None] * n_total
    n_done   = 0

    if os.path.exists(CHECKPOINT_PATH):
        log.info(f"Resuming from checkpoint: {CHECKPOINT_PATH}")
        ckpt_df = pd.read_parquet(CHECKPOINT_PATH)
        n_done  = len(ckpt_df)
        if n_done >= n_total:
            log.info("Checkpoint covers all rows — skipping inference, writing final output.")
            for i, row in enumerate(ckpt_df.to_dict("records")):
                results[i] = row
            n_done = n_total
        else:
            for i, row in enumerate(ckpt_df.to_dict("records")):
                results[i] = row
            log.info(f"Checkpoint has {n_done}/{n_total} rows — resuming from row {n_done}.")

    # ── Process in batches to avoid overloading the vLLM server ──────────────
    n_batches = (n_total + BATCH_SIZE - 1) // BATCH_SIZE

    with tqdm(total=n_total, initial=n_done, desc="Converting Image → UML") as pbar:
        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            end   = min(start + BATCH_SIZE, n_total)

            # Skip batches already covered by the checkpoint
            if end <= n_done:
                continue

            # For the first partial batch after a resume, start mid-batch
            row_start = max(start, n_done)
            batch     = image_bytes_list[row_start:end]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(process_row, img_bytes, provider): row_start + i
                    for i, img_bytes in enumerate(batch)
                }
                for future in as_completed(futures):
                    idx = futures[future]
                    results[idx] = future.result()
                    pbar.update(1)

            # ── Checkpoint: persist all completed rows after every batch ──────
            completed = [r for r in results if r is not None]
            pd.DataFrame(completed).to_parquet(CHECKPOINT_PATH, index=False)
            log.info(
                f"Batch {batch_idx + 1}/{n_batches} done "
                f"({end}/{n_total} rows) — checkpoint saved."
            )

    # ── Merge results into DataFrame ──────────────────────────────────────────
    results_df = pd.DataFrame(results, index=df.index)
    df = pd.concat([df, results_df], axis=1)

    # ── Summary ───────────────────────────────────────────────────────────────
    total        = len(df)
    valid        = df["uml_valid"].sum()
    llm_failures = df["llm_failed"].sum()
    uml_invalid  = total - valid - llm_failures

    log.info("─" * 50)
    log.info(f"Total rows      : {total}")
    log.info(f"LLM failures    : {llm_failures}")
    log.info(f"UML invalid     : {uml_invalid}")
    log.info(f"UML valid       : {valid}  ({100*valid/total:.1f}%)")
    log.info("─" * 50)
    for split_name, grp in df.groupby("split"):
        s_valid = grp["uml_valid"].sum()
        s_fail  = grp["llm_failed"].sum()
        log.info(
            f"  {split_name:12s} — {len(grp):5d} rows | "
            f"valid: {s_valid} ({100*s_valid/len(grp):.1f}%) | "
            f"llm_fail: {s_fail}"
        )
    log.info("─" * 50)

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_parquet(OUTPUT_PATH, index=False)
    log.info(f"Saved to {OUTPUT_PATH}")

    # Don't print the huge sketch_image dict out to the terminal
    sample_df = df[["split", "tikz_code", "uml_code", "uml_valid", "llm_failed", "uml_error"]].head()
    print(sample_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert images to UML PlantUML")
    parser.add_argument("--provider", choices=["vllm", "openai"], default="vllm", 
                        help="Choose the model provider (vllm or openai, defaults to vllm)")
    parser.add_argument("--workers", type=int, default=None,
                        help="Max concurrent workers (defaults to 8 for vllm, 4 for openai)")
    parser.add_argument("--no-hpc", action="store_true", 
                        help="Use local /Tmp base path instead of HPC cluster paths")
    args = parser.parse_args()

    max_concurrent = args.workers if args.workers is not None else (8 if args.provider == "vllm" else 4)
    main(provider=args.provider, max_workers=max_concurrent)
