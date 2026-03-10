#!/usr/bin/env python3
"""Merge a LoRA adapter into base model weights for deployment/export."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge SaulGPT LoRA adapter")
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--adapter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_dir = args.adapter_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter dir not found: {adapter_dir}")

    dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    print("[merge] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)

    print("[merge] Loading adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_dir))

    print("[merge] Merging adapter weights...")
    merged = model.merge_and_unload()

    print("[merge] Saving merged model...")
    merged.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    print(f"[merge] Done. Merged model saved to: {output_dir}")


if __name__ == "__main__":
    main()
