"""
benchmark_vlm.py

Runs inference on the held-out test split using a locally-served vLLM server
(or the merged model directly), then evaluates PlantUML syntax validity and
reports per-class metrics.

Usage:
    python benchmark_vlm.py \\
        --test-parquet  /project/.../output/qwen_lora_test_split.parquet \\
        --output-path   /project/.../output/benchmark_results.parquet \\
        --endpoint      http://localhost:8000/v1/chat/completions \\
        --model-name    Qwen/Qwen3-VL-4B-Instruct   # or the merged model path
"""
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pandas", "pyarrow", "tqdm", "python-dotenv"
# ]
# ///

import os
import re
import sys
import json
import base64
import argparse
import logging
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt (must match training prompt for fair evaluation)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert in converting UML diagram sketches to PlantUML code. "
    "Given an image of a UML diagram, output only the equivalent PlantUML source "
    "wrapped in @startuml / @enduml. "
    "Do not include any explanation, markdown fences, or extra text — "
    "only the raw PlantUML source starting with @startuml."
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def bytes_to_b64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def build_messages(image_bytes: bytes) -> list:
    b64 = bytes_to_b64(image_bytes)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Convert this UML sketch into valid PlantUML code."},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            ],
        },
    ]


def call_vllm(image_bytes: bytes, endpoint: str, model_name: str, timeout: int = 120) -> str | None:
    payload = json.dumps({
        "model": model_name,
        "messages": build_messages(image_bytes),
        "temperature": 0.0,
    }).encode("utf-8")

    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
            content = body["choices"][0]["message"]["content"]
            content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL)
            return content.strip()
    except urllib.error.URLError as e:
        log.error(f"vLLM connection error: {e.reason}")
    except Exception as e:
        log.error(f"vLLM call failed: {e}")
    return None


def is_valid_plantuml(code: str) -> tuple[bool, str]:
    """
    Lightweight local validity check:
      - Must start with @startuml and end with @enduml
      - Must contain at least one non-trivial statement
    For production, wire this to the PlantUMLWebValidator instead.
    """
    if not code:
        return False, "Empty response"
    stripped = code.strip()
    if not stripped.startswith("@startuml"):
        return False, "Missing @startuml"
    if not stripped.endswith("@enduml"):
        return False, "Missing @enduml"
    # Must have at least one line between the markers
    inner = stripped[len("@startuml"):stripped.rfind("@enduml")].strip()
    if not inner:
        return False, "Empty diagram body"
    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def _extract_gt_from_messages(messages) -> str:
    """Pull the assistant text out of a messages list (finetune_vlm.py format)."""
    if not isinstance(messages, list):
        return ""
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    return part.get("text", "")
    return ""


def load_test_split(parquet_path: str) -> list[dict]:
    """
    Supports three formats:
      - finetune_vlm.py output: image_bytes + messages columns
      - raw HF dataset:         sketch_image dict with 'bytes' key + uml_code column
      - legacy:                 image_bytes + uml_code columns
    """
    df = pd.read_parquet(parquet_path)
    records = []
    for _, row in df.iterrows():
        if "image_bytes" in df.columns:
            img_bytes = row["image_bytes"]
        elif "sketch_image" in df.columns:
            img_bytes = row["sketch_image"]["bytes"]
        else:
            log.warning("Row has no image column — skipping.")
            continue

        # Ground-truth: direct column takes priority; fall back to messages assistant turn
        uml_gt = row.get("uml_code", "") or ""
        if not uml_gt and "messages" in df.columns:
            uml_gt = _extract_gt_from_messages(row.get("messages"))

        records.append({"image_bytes": img_bytes, "uml_code_gt": uml_gt})
    log.info(f"Loaded {len(records)} test samples from {parquet_path}")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark loop
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(records: list[dict], endpoint: str, model_name: str,
                  max_workers: int, timeout: int) -> list[dict]:
    results = [None] * len(records)

    def process(i, rec):
        uml_pred = call_vllm(rec["image_bytes"], endpoint, model_name, timeout)
        llm_failed = uml_pred is None
        valid, err = (False, "LLM failure") if llm_failed else is_valid_plantuml(uml_pred)
        return {
            "uml_pred":   uml_pred or "",
            "uml_valid":  valid,
            "uml_error":  err,
            "llm_failed": llm_failed,
        }

    with tqdm(total=len(records), desc="Benchmarking") as pbar:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(process, i, rec): i for i, rec in enumerate(records)}
            for fut in as_completed(futures):
                i = futures[fut]
                results[i] = fut.result()
                pbar.update(1)

    return results


def print_summary(records: list[dict], results: list[dict]):
    n = len(results)
    n_valid   = sum(1 for r in results if r["uml_valid"])
    n_failed  = sum(1 for r in results if r["llm_failed"])
    n_invalid = n - n_valid - n_failed

    log.info("=" * 55)
    log.info(f"BENCHMARK RESULTS  ({n} test samples)")
    log.info("=" * 55)
    log.info(f"  LLM failures    : {n_failed:4d}  ({100*n_failed/n:.1f}%)")
    log.info(f"  UML invalid     : {n_invalid:4d}  ({100*n_invalid/n:.1f}%)")
    log.info(f"  UML valid ✓     : {n_valid:4d}  ({100*n_valid/n:.1f}%)")
    log.info("=" * 55)

    # Show a few failure reasons
    errors = [r["uml_error"] for r in results if r["uml_error"] and not r["llm_failed"]]
    if errors:
        from collections import Counter
        top = Counter(errors).most_common(5)
        log.info("Top validation errors:")
        for err, cnt in top:
            log.info(f"  [{cnt:3d}x] {err}")
    log.info("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Benchmark finetuned Qwen-VL on test split")
    parser.add_argument("--test-parquet", required=True, help="Path to test_split.parquet")
    parser.add_argument("--output-path",  required=True, help="Where to save benchmark_results.parquet")
    parser.add_argument("--endpoint",     default="http://localhost:8000/v1/chat/completions")
    parser.add_argument("--model-name",   default="Qwen/Qwen3-VL-4B-Instruct",
                        help="Model name as registered in the vLLM server")
    parser.add_argument("--workers",      type=int, default=8, help="Concurrent request workers")
    parser.add_argument("--timeout",      type=int, default=120, help="Per-request timeout (s)")
    args = parser.parse_args()

    records = load_test_split(args.test_parquet)
    results = run_benchmark(records, args.endpoint, args.model_name,
                            args.workers, args.timeout)
    print_summary(records, results)

    # Merge into a DataFrame and save
    out_df = pd.DataFrame([
        {**{"uml_code_gt": rec["uml_code_gt"]}, **res}
        for rec, res in zip(records, results)
    ])
    out_dir = os.path.dirname(args.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out_df.to_parquet(args.output_path, index=False)
    log.info(f"Results saved → {args.output_path}")


if __name__ == "__main__":
    main()
