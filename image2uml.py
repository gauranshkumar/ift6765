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
from datetime import datetime
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

OPENAI_MODEL_NAME = "gpt-5.4"
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
            {"type": "image_url", "image_url": {"url": image_url}}
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
# OpenAI Batch Logic
# ─────────────────────────────────────────────────────────────────────────────

def submit_openai_vision_batch(df: pd.DataFrame, output_dir: str):
    run_id = os.environ.get("SLURM_JOB_ID", datetime.now().strftime("%Y%m%d_%H%M%S"))
    jsonl_path = f"{output_dir}/vision_batch_requests_{run_id}.jsonl"
    log.info(f"Generating JSONL payload for batch API to {jsonl_path}...")
    
    requests = []
    for idx, row in df.iterrows():
        req = {
            "custom_id": f"req_{idx}",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": OPENAI_MODEL_NAME,
                "messages": build_messages(row["sketch_image"]["bytes"]),
                "temperature": 0.0
            }
        }
        requests.append(req)
        
    log.info("Splitting into chunks by maximum 150MB limit to avoid 413 Payload Too Large...")
    MAX_FILE_SIZE = 150 * 1024 * 1024  # 150 MB
    current_chunk = []
    current_size = 0
    part = 0
    chunk_files = []
    
    def finalize_chunk(chunk_list, part_num):
        jsonl_path = f"{output_dir}/vision_batch_requests_{run_id}_part{part_num}.jsonl"
        with open(jsonl_path, "w", encoding="utf-8") as f:
            f.write("\n".join(chunk_list) + "\n")
        size_mb = os.path.getsize(jsonl_path) / (1024 * 1024)
        chunk_files.append((jsonl_path, size_mb, len(chunk_list)))
        
    for r in requests:
        req_str = json.dumps(r)
        req_size = len(req_str.encode("utf-8")) + 1  # +1 for newline
        
        if current_size + req_size > MAX_FILE_SIZE and current_chunk:
            finalize_chunk(current_chunk, part)
            part += 1
            current_chunk = []
            current_size = 0
            
        current_chunk.append(req_str)
        current_size += req_size
        
    if current_chunk:
        finalize_chunk(current_chunk, part)

    print("\n" + "="*50)
    print("BATCH UPLOAD PLAN")
    print("="*50)
    for path, size_mb, count in chunk_files:
        print(f"File: {os.path.basename(path)}")
        print(f"  Requests : {count}")
        print(f"  Size     : {size_mb:.2f} MB")
    print("="*50)
    
    user_input = input("Proceed with uploading these files to OpenAI and submitting batch jobs? (y/n): ")
    if user_input.lower() not in ['y', 'yes']:
        log.warning("Batch submission aborted by user.")
        return

    batch_ids = []
    for path, size_mb, count in chunk_files:
        log.info(f"Uploading file {os.path.basename(path)} to OpenAI...")
        batch_input_file = openai_client.files.create(
          file=open(path, "rb"),
          purpose="batch"
        )
        
        log.info(f"Uploaded file ID: {batch_input_file.id}. Submitting batch job...")
        batch = openai_client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={
              "description": f"image2uml batch {os.path.basename(path)}"
            }
        )
        log.info(f"Batch submitted! Batch ID: {batch.id}")
        batch_ids.append(batch.id)

    batch_file_path = f"{output_dir}/.vision_batch_job_id"
    with open(batch_file_path, "w") as f:
        f.write(",".join(batch_ids))
    log.info(f"All Batch IDs saved to {batch_file_path} for retrieval.")
    log.info("Run the script again with `--mode retrieve` later to check status.")


def retrieve_openai_vision_batch(df: pd.DataFrame, output_path: str, output_dir: str, max_workers: int):
    batch_file_path = f"{output_dir}/.vision_batch_job_id"
    if not os.path.exists(batch_file_path):
        log.error(f"Batch ID file not found at {batch_file_path}. Did you run --mode submit?")
        return
        
    with open(batch_file_path, "r") as f:
        batch_ids_str = f.read().strip()
        
    batch_ids = [b.strip() for b in batch_ids_str.split(",") if b.strip()]
    results_map = {}
    all_completed = True
    
    for batch_id in batch_ids:
        log.info(f"Retrieving batch status for ID: {batch_id}")
        try:
            batch = openai_client.batches.retrieve(batch_id)
        except Exception as e:
            log.error(f"Failed to retrieve batch: {e}")
            all_completed = False
            continue
        
        if batch.status != "completed":
            log.warning(f"Batch {batch_id} is not completed yet (Status: {batch.status}).")
            all_completed = False
            continue
            
        output_file_id = getattr(batch, "output_file_id", None)
        if not output_file_id:
            error_file_id = getattr(batch, "error_file_id", None)
            if error_file_id:
                log.error(f"Job {batch_id} contains errors! Found error_file_id: {error_file_id}")
                try:
                    err_content = openai_client.files.content(error_file_id).text
                    log.error(f"Errors snippet:\n{err_content[:1000]}")
                except Exception:
                    pass
            else:
                log.error(f"Batch {batch_id} completed but no output_file_id was found!")
            all_completed = False
            continue
            
        log.info(f"Downloading results for Job ID {batch_id} using Output File ID: {output_file_id}...")
        try:
            content = openai_client.files.content(output_file_id).text
        except Exception as e: # fallback strictly using HTTP if SDK misbehaves
            log.warning(f"SDK `files.content` failed ({e}), fetching manually via HTTP...")
            req = urllib.request.Request(
                f"https://api.openai.com/v1/files/{output_file_id}/content",
                headers={"Authorization": f"Bearer {api_key}"}
            )
            with urllib.request.urlopen(req) as response:
                content = response.read().decode('utf-8')
        
        for line in content.strip().split("\n"):
            if not line:
                continue
            obj = json.loads(line)
            idx_str = obj["custom_id"].split("_")[1]
            idx = int(idx_str)
            
            if obj["response"]["status_code"] == 200:
                msg = obj["response"]["body"]["choices"][0]["message"]["content"]
                msg = re.sub(r"<think>.*?</think>", "", msg, flags=re.DOTALL).strip()
                results_map[idx] = msg
            else:
                log.warning(f"Row {idx} failed in batch response.")
                results_map[idx] = None
                
    if not all_completed:
        log.error("Not all batches have completed successfully. Aborting merge process. Please try again later.")
        return
            
    log.info("Validating PlantUML logic via Web API...")
    n_total = len(df)
    results = [None] * n_total
    
    def validate_row(idx, code):
        res = {"uml_code": code if code else "", "uml_valid": False, "uml_error": "", "llm_failed": False}
        if code is None:
            res["llm_failed"] = True
            res["uml_error"] = "LLM failure in batch"
            return res
            
        try:
            val = validator.validate(code)
            res["uml_valid"] = val["valid"]
            if not val["valid"]:
                res["uml_error"] = "; ".join(val.get("errors", []))
        except Exception as e:
            res["uml_error"] = f"Validator exception: {e}"
        return res

    with tqdm(total=n_total, desc="Validating UML") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(validate_row, idx, results_map.get(idx)): idx for idx in range(n_total)}
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                pbar.update(1)
                
    results_df = pd.DataFrame(results, index=df.index)
    df = pd.concat([df, results_df], axis=1)
    
    df.to_parquet(output_path, index=False)
    log.info(f"Saved completed dataset to {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(provider: str, mode: str, max_workers: int):
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

    if provider == "openai" and mode == "submit":
        submit_openai_vision_batch(df, OUTPUT_DIR)
        return
    elif provider == "openai" and mode == "retrieve":
        retrieve_openai_vision_batch(df, OUTPUT_PATH, OUTPUT_DIR, max_workers)
        return

    # ── Resume from checkpoint if available (Interactive mode) ────────────────
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
    parser.add_argument("--mode", choices=["interactive", "submit", "retrieve"], default="interactive",
                        help="Execution mode for openai (interactive processes row by row). Ignored for vllm.")
    parser.add_argument("--no-hpc", action="store_true", 
                        help="Use local /Tmp base path instead of HPC cluster paths")
    args = parser.parse_args()

    max_concurrent = args.workers if args.workers is not None else (8 if args.provider == "vllm" else 4)
    main(provider=args.provider, mode=args.mode, max_workers=max_concurrent)
