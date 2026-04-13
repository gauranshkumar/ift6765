# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "torch",
#     "transformers",
#     "peft",
#     "trl",
#     "datasets",
#     "accelerate",
#     "bitsandbytes",
#     "pillow",
#     "pandas",
#     "pyarrow",
#     "wandb",
#     "python-dotenv"
# ]
# ///

import os
import argparse
import pandas as pd
import torch
import wandb
from dotenv import load_dotenv
from io import BytesIO
from PIL import Image
from datasets import Dataset, DatasetDict
from transformers import AutoProcessor, AutoModelForCausalLM
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

QWEN_MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"

SYSTEM_PROMPT = (
    "You are an expert in converting UML diagram sketches to PlantUML code. "
    "Convert this UML sketch into valid PlantUML source code wrapped in @startuml / @enduml."
)


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """Convert raw bytes to an RGB PIL Image."""
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def format_qwen_vl_chat(row):
    """
    Format the dataset row into Qwen-VL conversational schema.
    Stores raw image bytes to avoid Arrow serialization issues with PIL Images.
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Convert this UML sketch into valid PlantUML code."}
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": row["uml_code"]}]}
    ]
    return {
        "messages": messages,
        "image_bytes": row["sketch_image"]["bytes"]  # keep as bytes, convert in collator
    }


def load_and_prepare_dataset(parquet_path: str) -> DatasetDict:
    print(f"Loading dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    initial_len = len(df)
    if "uml_valid" in df.columns:
        df = df[df["uml_valid"] == True]
    if "llm_failed" in df.columns:
        df = df[df["llm_failed"] == False]

    print(f"Filtered invalid entries. Keeping {len(df)} / {initial_len} samples.")

    dataset_dict = {}
    if "split" in df.columns:
        for split_name, group_df in df.groupby("split"):
            dataset_dict[split_name] = Dataset.from_pandas(group_df, preserve_index=False)
        ds = DatasetDict(dataset_dict)
    else:
        print("No predefined split found. Creating 80/10/10 train/val/test split...")
        full_ds = Dataset.from_pandas(df, preserve_index=False)
        # First split off 20% for val+test
        train_temp = full_ds.train_test_split(test_size=0.20, seed=42)
        # Split the 20% evenly into val and test
        val_test = train_temp["test"].train_test_split(test_size=0.50, seed=42)
        ds = DatasetDict({
            "train":      train_temp["train"],
            "validation": val_test["train"],
            "test":       val_test["test"],
        })

    print("Sizes:", {k: len(v) for k, v in ds.items()})

    print("Formatting chat templates...")
    for split in ds.keys():
        ds[split] = ds[split].map(
            format_qwen_vl_chat,
            remove_columns=ds[split].column_names,
            desc=f"Formatting {split}",
        )

    return ds


def _find_response_start(ids: list[int], header_ids: list[int]) -> int:
    """Return the first token index of the assistant's actual response text."""
    n = len(header_ids)
    for i in range(len(ids) - n + 1):
        if ids[i:i + n] == header_ids:
            return i + n
    return 0  # fallback: don't mask anything (should not happen)


def collate_fn_vl(batch, processor, max_length: int = 4096):
    """
    Custom collator that:
    1. Applies the Qwen-VL chat template and tokenises the full conversation.
    2. Masks padding tokens AND all prompt tokens in `labels` so loss is
       computed only on the assistant's response.
    """
    texts = [
        processor.apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=False
        )
        for item in batch
    ]
    images = [bytes_to_pil(item["image_bytes"]) for item in batch]

    batch_enc = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )

    input_ids = batch_enc["input_ids"]
    labels = input_ids.clone()

    # Mask padding tokens
    labels[labels == processor.tokenizer.pad_token_id] = -100

    # Mask all prompt tokens — only compute loss on the assistant's response.
    # Qwen chat template wraps the assistant turn with: <|im_start|>assistant\n
    assistant_header_ids = processor.tokenizer.encode(
        "<|im_start|>assistant\n", add_special_tokens=False
    )
    for i in range(input_ids.size(0)):
        response_start = _find_response_start(input_ids[i].tolist(), assistant_header_ids)
        labels[i, :response_start] = -100

    batch_enc["labels"] = labels
    return batch_enc


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="Fine-tune Qwen-VL with QLoRA")
    parser.add_argument("--epochs",     type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Per-device batch size (effective = batch_size × gpus × grad_accum)")
    parser.add_argument("--no-hpc", action="store_true",
                        help="Use local /Tmp/kumargau paths instead of Compute Canada paths")
    args = parser.parse_args()

    BASE_DIR = "/Tmp/kumargau/ift6765" if args.no_hpc else "/project/def-syriani/gauransh/ift6765"
    INPUT_PARQUET = f"{BASE_DIR}/output/image2uml_20260412_190600.parquet"
    OUTPUT_DIR    = f"{BASE_DIR}/output/qwen_lora_finetuned"

    if not os.path.exists(INPUT_PARQUET):
        print(f"[ERROR] Training data not found: {INPUT_PARQUET}")
        return

    # ── Dataset ──────────────────────────────────────────────────────────────
    ds = load_and_prepare_dataset(INPUT_PARQUET)
    train_dataset = ds.get("train", next(iter(ds.values())))
    eval_dataset  = ds.get("validation", ds.get("test", None))

    # ── Processor & Model ────────────────────────────────────────────────────
    print("Loading processor and model in bfloat16...")
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)

    # Do NOT use device_map="auto" here — let the Trainer / accelerate handle
    # device placement so DDP (multi-process) works correctly.
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # ── LoRA ─────────────────────────────────────────────────────────────────
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ── Training args ─────────────────────────────────────────────────────────
    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        optim="adamw_torch_fused",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        fp16=False,
        max_grad_norm=0.3,
        logging_steps=10,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        save_strategy="epoch",
        load_best_model_at_end=eval_dataset is not None,
        metric_for_best_model="eval_loss",
        greater_is_better=False,          # lower loss is better
        max_seq_length=4096,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to="wandb",
        run_name="image2uml-qwen-vl-lora",
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    # NOTE: model is already a PEFT model; do NOT also pass peft_config to
    # SFTTrainer — that would double-wrap the model.
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=lambda batch: collate_fn_vl(batch, processor, max_length=4096),
    )

    # Resume from checkpoint if one exists
    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
        if last_checkpoint:
            print(f"Resuming from checkpoint: {last_checkpoint}")

    print("Starting fine-tuning...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    FINAL_OUT = os.path.join(OUTPUT_DIR, "final")
    print(f"Saving final LoRA adapter to {FINAL_OUT}")
    trainer.save_model(FINAL_OUT)
    processor.save_pretrained(FINAL_OUT)
    print("Done.")


if __name__ == "__main__":
    main()
