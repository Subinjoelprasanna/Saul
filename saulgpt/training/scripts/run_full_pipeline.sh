#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

BASE_MODEL="${BASE_MODEL:-mistralai/Mistral-7B-Instruct-v0.3}"
EPOCHS="${EPOCHS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-0}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-0}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/training/outputs/saulgpt-lora}"
MODEFILE_OUT="${MODEFILE_OUT:-$ROOT_DIR/training/Modelfile.saulgpt}"

TRAIN_ARGS=()
if [[ "$MAX_TRAIN_SAMPLES" -gt 0 ]]; then
  TRAIN_ARGS+=(--max-train-samples "$MAX_TRAIN_SAMPLES")
fi
if [[ "$MAX_EVAL_SAMPLES" -gt 0 ]]; then
  TRAIN_ARGS+=(--max-eval-samples "$MAX_EVAL_SAMPLES")
fi

echo "[1/3] Building SFT dataset..."
python3 training/scripts/build_sft_dataset.py

echo "[2/3] Starting LoRA training..."
python3 training/scripts/finetune_lora.py \
  --base-model "$BASE_MODEL" \
  --train-file training/data/saulgpt_sft_train.jsonl \
  --eval-file training/data/saulgpt_sft_eval.jsonl \
  --output-dir "$OUT_DIR" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --gradient-checkpointing \
  "${TRAIN_ARGS[@]}"

echo "[3/3] Creating Ollama Modelfile..."
python3 training/scripts/create_ollama_modelfile.py \
  --from-model mistral \
  --adapter-path "$OUT_DIR" \
  --output "$MODEFILE_OUT"

echo "Done. Next: ollama create saulgpt-legal-ft -f ${MODEFILE_OUT#$ROOT_DIR/}"
