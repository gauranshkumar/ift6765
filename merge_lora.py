import os
import argparse
import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA weights into base Qwen-VL model")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen3-VL-4B-Instruct", help="Original Model ID")
    parser.add_argument("--lora-path", type=str, required=True, help="Path to the LoRA weights (e.g. output/qwen_lora_finetuned/final)")
    parser.add_argument("--output-dir", type=str, required=True, help="Path to save the fused model")
    args = parser.parse_args()

    print(f"Loading Base Architectures: {args.base_model}...")
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="cpu",  # Loading merging on CPU is safer and avoids multi-GPU VRAM collisions
        trust_remote_code=True
    )
    
    processor = AutoProcessor.from_pretrained(args.lora_path, trust_remote_code=True)

    print(f"Loading LoRA Overlays from {args.lora_path}...")
    peft_model = PeftModel.from_pretrained(base_model, args.lora_path)

    print("Merging and unloading PEFT layers directly into base parameters...")
    # This physically binds the adapters resolving it mathematically back to standard linear layers
    merged_model = peft_model.merge_and_unload()

    print(f"Saving finalized standalone model to {args.output_dir}...")
    os.makedirs(args.output_dir, exist_ok=True)
    merged_model.save_pretrained(args.output_dir, safe_serialization=True)
    processor.save_pretrained(args.output_dir)

    print("[SUCCESS] Merged model completely saved! Ready for vLLM Server Inference.")

if __name__ == "__main__":
    main()
