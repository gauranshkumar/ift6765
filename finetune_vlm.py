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
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
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
    Format the dataset row into the conversational schema expected by Qwen-VL architecture.
    The processor uses Chat Templates dynamically.
    """
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Convert this UML sketch into valid PlantUML code."}
        ]},
        {"role": "assistant", "content": [{"type": "text", "text": row["uml_code"]}]}
    ]
    return {"messages": messages, "images": [bytes_to_pil(row["sketch_image"]["bytes"])]}

def load_and_prepare_dataset(parquet_path: str) -> DatasetDict:
    print(f"Loading dataset from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    initial_len = len(df)
    if "uml_valid" in df.columns:
        df = df[df["uml_valid"] == True]
    if "llm_failed" in df.columns:
        df = df[df["llm_failed"] == False]
        
    print(f"Filtered out invalid generations. Keeping {len(df)} / {initial_len} samples.")
    
    dataset_dict = {}
    if "split" in df.columns:
        for split_name, group_df in df.groupby("split"):
            hf_ds = Dataset.from_pandas(group_df)
            dataset_dict[split_name] = hf_ds
        ds = DatasetDict(dataset_dict)
    else:
        print("No predefined split found. Creating 85/15 train/test split...")
        full_ds = Dataset.from_pandas(df)
        train_test = full_ds.train_test_split(test_size=0.15, seed=42)
        ds = DatasetDict({
            "train": train_test["train"],
            "validation": train_test["test"]
        })
        
    print(f"Formatting Chat Templates natively...")
    for split in ds.keys():
        ds[split] = ds[split].map(format_qwen_vl_chat, remove_columns=ds[split].column_names)
        
    return ds
    
def collate_fn_vl(batch, processor):
    """
    Custom collator mapping the chat templates through the visual processor.
    `batch` is a list of dicts: [{'messages': [...], 'images': [...]}, ...]
    """
    texts = [processor.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=False) for item in batch]
    image_inputs = [img for item in batch for img in item["images"]]
    
    batch_kwargs = processor(
        text=texts,
        images=image_inputs,
        return_tensors="pt",
        padding=True
    )
    
    labels = batch_kwargs["input_ids"].clone()
    labels[labels == processor.tokenizer.pad_token_id] = -100
    batch_kwargs["labels"] = labels
    return batch_kwargs

def main():
    # Securely load environment configurations to inject WANDB_API_KEY natively
    load_dotenv()

    parser = argparse.ArgumentParser(description="Fine-tune Qwen-VL architecture using QLoRA")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Per device batch size (increased for H100s)")
    parser.add_argument("--no-hpc", action="store_true", help="Use local /Tmp/kumargau paths instead of Compute Canada paths")
    args = parser.parse_args()

    if args.no_hpc:
        BASE_DIR = "/Tmp/kumargau/ift6765"
    else:
        BASE_DIR = "/project/def-syriani/gauransh/ift6765"

    INPUT_PARQUET = f"{BASE_DIR}/output/image2uml_20260412_190600.parquet"
    OUTPUT_DIR    = f"{BASE_DIR}/output/qwen_lora_finetuned"
    
    if not os.path.exists(INPUT_PARQUET):
        print(f"Error: Could not find training data explicitly looking at {INPUT_PARQUET}")
        return

    dataset_dict = load_and_prepare_dataset(INPUT_PARQUET)
    train_dataset = dataset_dict.get("train", next(iter(dataset_dict.values()))) # safety
    eval_dataset  = dataset_dict.get("validation", dataset_dict.get("test", None))

    print("Loading Processor and Model spanning 2x H100s natively (bfloat16)...")
    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        optim="adamw_torch_fused",
        logging_steps=10,
        eval_strategy="epoch" if eval_dataset else "no",
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=True,                             # Harness H100 native performance
        fp16=False, 
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        num_train_epochs=args.epochs,
        max_seq_length=4096,                   
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to="wandb",                     # Enable explicit Weights & Biases logging
        run_name="image2uml-qwen-vl-lora"      # Organized Run label for dashboard
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        data_collator=lambda batch: collate_fn_vl(batch, processor)
    )

    last_checkpoint = None
    if os.path.isdir(OUTPUT_DIR):
        last_checkpoint = get_last_checkpoint(OUTPUT_DIR)
        if last_checkpoint is not None:
            print(f"Detected previous checkpoint at {last_checkpoint}. Resuming natively...")

    print("Starting Fine-Tuning Execution...")
    trainer.train(resume_from_checkpoint=last_checkpoint)

    FINAL_OUT = os.path.join(OUTPUT_DIR, "final")
    print(f"Training completed successfully. Saving final LoRA adapter weights specifically to {FINAL_OUT}")
    trainer.save_model(FINAL_OUT)
    processor.save_pretrained(FINAL_OUT)

if __name__ == "__main__":
    main()
