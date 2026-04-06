import re
import pandas as pd
import glob
import json
import logging
import urllib.request
import urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

from utils.UML import PlantUMLWebValidator

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

VLLM_ENDPOINT   = "http://localhost:8000/v1/chat/completions"
MODEL_NAME       = "Qwen/Qwen3-Coder-Next"          # ← replace with your vLLM model
OUTPUT_PATH      = "/project/def-syriani/gauransh/ift6765/data/output_with_uml.parquet"
REQUEST_TIMEOUT  = 60                          # seconds per LLM call
PLANTUML_SERVER  = "https://www.plantuml.com/plantuml"
MAX_WORKERS      = 8                           # concurrent vLLM requests per batch
BATCH_SIZE       = 64                          # rows submitted to the server at once

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("/scratch/gauransh/logs/conversion.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert in converting TikZ diagrams to PlantUML code. "
    "Given a TikZ code snippet, output only the equivalent PlantUML source "
    "wrapped in @startuml / @enduml. "
    "Do not include any explanation, markdown fences, or extra text — "
    "only the raw PlantUML source starting with @startuml."
)

def build_user_prompt(tikz_code: str) -> str:
    return (
        f"Convert the following TikZ code to valid PlantUML:\n\n"
        f"{tikz_code.strip()}"
    )

# ─────────────────────────────────────────────────────────────────────────────
# vLLM call
# ─────────────────────────────────────────────────────────────────────────────

def call_vllm(tikz_code: str) -> str | None:
    """
    Sends tikz_code to the vLLM OpenAI-compatible endpoint.
    Returns the generated PlantUML string, or None on failure.
    """
    payload = json.dumps({
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(tikz_code)},
        ],
        "temperature": 0.0,
        "max_tokens":  8192,
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
            # Strip any residual <think>…</think> block (safety net for Qwen3)
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
# Per-row processing
# ─────────────────────────────────────────────────────────────────────────────

validator = PlantUMLWebValidator(server=PLANTUML_SERVER)

def process_row(tikz_code: str) -> dict:
    """
    Returns a dict with keys:
      uml_code        - generated PlantUML string (or empty string on LLM failure)
      uml_valid       - True / False
      uml_error       - error message string, empty if valid
    """
    result = {"uml_code": "", "uml_valid": False, "uml_error": "", "llm_failed": False}

    # ── 1. LLM conversion ────────────────────────────────────────────────────
    uml_code = call_vllm(tikz_code)
    if uml_code is None:
        result["uml_valid"]  = False
        result["llm_failed"] = True
        result["uml_error"]  = "LLM call failed — see conversion.log"
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

def main():
    # ── Load data ─────────────────────────────────────────────────────────────
    train_files = glob.glob("/project/def-syriani/gauransh/ift6765/data/*.parquet")
    if not train_files:
        log.error("No parquet files found in /project/def-syriani/gauransh/ift6765/data/")
        return

    log.info(f"Loading {len(train_files)} parquet file(s)...")
    df = pd.read_parquet(train_files, columns=["tikz_code"])
    log.info(f"Loaded {df.shape[0]} rows.")

    # ── Process in batches to avoid overloading the vLLM server ──────────────
    tikz_codes = df["tikz_code"].tolist()
    results = [None] * len(tikz_codes)
    n_batches = (len(tikz_codes) + BATCH_SIZE - 1) // BATCH_SIZE

    with tqdm(total=len(tikz_codes), desc="Converting TikZ → UML") as pbar:
        for batch_idx in range(n_batches):
            start = batch_idx * BATCH_SIZE
            end   = min(start + BATCH_SIZE, len(tikz_codes))
            batch = tikz_codes[start:end]

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {
                    executor.submit(process_row, str(code)): start + i
                    for i, code in enumerate(batch)
                }
                for future in as_completed(futures):
                    results[futures[future]] = future.result()
                    pbar.update(1)

            log.info(f"Batch {batch_idx + 1}/{n_batches} done ({end}/{len(tikz_codes)} rows)")

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

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_parquet(OUTPUT_PATH, index=False)
    log.info(f"Saved to {OUTPUT_PATH}")

    print(df[["tikz_code", "uml_code", "uml_valid", "llm_failed", "uml_error"]].head())


if __name__ == "__main__":
    main()
