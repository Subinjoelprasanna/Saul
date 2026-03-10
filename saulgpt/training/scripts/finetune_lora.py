#!/usr/bin/env python3
"""Fine-tune an instruction model for SaulGPT using LoRA + SFTTrainer."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for SaulGPT")
    parser.add_argument("--base-model", type=str, default="mistralai/Mistral-7B-Instruct-v0.3")
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("training/outputs/saulgpt-lora"))
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-4bit", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--resume-from-checkpoint", type=str, default="")
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    return parser.parse_args()


def build_quant_config(use_4bit: bool, torch, bitsandbytes_config_cls):
    if not use_4bit:
        return None
    if not torch.cuda.is_available():
        print("[train] 4-bit requested but CUDA not available. Continuing without quantization.")
        return None
    return bitsandbytes_config_cls(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def format_messages(example, tokenizer):
    messages = example["messages"]
    if hasattr(tokenizer, "apply_chat_template"):
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    else:
        chunks = []
        for msg in messages:
            chunks.append(f"{msg['role'].upper()}: {msg['content']}")
        text = "\n".join(chunks)
    return {"text": text}


def main() -> None:
    args = parse_args()

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            TrainingArguments,
        )
        from trl import SFTTrainer
    except ModuleNotFoundError as exc:
        missing_pkg = exc.name or "training dependency"
        raise SystemExit(
            f"Missing dependency '{missing_pkg}'. Install requirements first:\n"
            "pip3 install -r training/requirements-train.txt"
        ) from exc

    train_file = args.train_file.resolve()
    eval_file = args.eval_file.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not train_file.exists():
        raise FileNotFoundError(f"Train file not found: {train_file}")
    if not eval_file.exists():
        raise FileNotFoundError(f"Eval file not found: {eval_file}")

    print("[train] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = build_quant_config(args.use_4bit, torch, BitsAndBytesConfig)
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_kwargs = {
        "torch_dtype": torch_dtype,
        "trust_remote_code": True,
    }
    if quant_config is not None:
        model_kwargs["quantization_config"] = quant_config
        model_kwargs["device_map"] = "auto"

    print("[train] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, **model_kwargs)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    print("[train] Loading dataset...")
    ds = load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "eval": str(eval_file),
        },
    )

    ds = ds.map(
        lambda ex: format_messages(ex, tokenizer),
        remove_columns=ds["train"].column_names,
    )
    train_ds = ds["train"]
    eval_ds = ds["eval"]

    if args.max_train_samples > 0:
        train_ds = train_ds.select(range(min(args.max_train_samples, len(train_ds))))
    if args.max_eval_samples > 0:
        eval_ds = eval_ds.select(range(min(args.max_eval_samples, len(eval_ds))))

    print(f"[train] Train samples: {len(train_ds)} | Eval samples: {len(eval_ds)}")

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    use_bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        bf16=use_bf16,
        fp16=(torch.cuda.is_available() and not use_bf16),
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=True,
    )

    def build_trainer(use_packing: bool):
        return SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            dataset_text_field="text",
            peft_config=peft_config,
            max_seq_length=args.max_seq_len,
            packing=use_packing,
        )

    try:
        trainer = build_trainer(use_packing=True)
    except ValueError as exc:
        if "packing the dataset" not in str(exc).lower():
            raise
        print("[train] Packing failed for this dataset size. Retrying with packing disabled.")
        trainer = build_trainer(use_packing=False)

    print("[train] Starting training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)

    print("[train] Saving LoRA adapter + tokenizer...")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"[train] Done. Adapter saved to: {output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
