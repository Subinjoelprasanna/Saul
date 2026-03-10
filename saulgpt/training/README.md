# SaulGPT Fine-Tuning Pipeline (LoRA)

This folder provides a full pipeline to fine-tune SaulGPT for Indian legal intake/reporting behavior.

## 1. Create SFT Dataset

```bash
cd /Users/siddarthguruprasad/Documents/Playground/saulgpt
python3 training/scripts/build_sft_dataset.py
```

Outputs:
- `training/data/saulgpt_sft_train.jsonl`
- `training/data/saulgpt_sft_eval.jsonl`

With the current `laws.txt + laws_expanded.txt`, the builder generates about:
- `846` train examples
- `44` eval examples

## 2. Install Training Dependencies

```bash
pip3 install -r training/requirements-train.txt
# plus torch matching your system
```

If dependencies are missing, `finetune_lora.py` now exits with a clear install message.
Install `bitsandbytes` only on Linux CUDA if you plan to use `--use-4bit`.

## 3. Train LoRA Adapter

```bash
python3 training/scripts/finetune_lora.py \
  --base-model mistralai/Mistral-7B-Instruct-v0.3 \
  --train-file training/data/saulgpt_sft_train.jsonl \
  --eval-file training/data/saulgpt_sft_eval.jsonl \
  --output-dir training/outputs/saulgpt-lora \
  --epochs 2 \
  --batch-size 1 \
  --grad-accum 8 \
  --gradient-checkpointing
```

For CUDA 4-bit training, add `--use-4bit`.

Quick smoke run (small subset, useful to validate pipeline):

```bash
python3 training/scripts/finetune_lora.py \
  --base-model HuggingFaceTB/SmolLM2-135M-Instruct \
  --train-file training/data/saulgpt_sft_train.jsonl \
  --eval-file training/data/saulgpt_sft_eval.jsonl \
  --output-dir training/outputs/saulgpt-smoke-lora \
  --epochs 0.05 \
  --batch-size 1 \
  --grad-accum 1 \
  --max-train-samples 8 \
  --max-eval-samples 4
```

## 4. Merge Adapter (optional)

```bash
python3 training/scripts/merge_lora.py \
  --base-model mistralai/Mistral-7B-Instruct-v0.3 \
  --adapter-dir training/outputs/saulgpt-lora \
  --output-dir training/outputs/saulgpt-merged
```

## 5. Create Ollama Modelfile

Adapter-based (no merge in Ollama if supported by your build):

```bash
python3 training/scripts/create_ollama_modelfile.py \
  --from-model mistral \
  --adapter-path /ABS/PATH/training/outputs/saulgpt-lora \
  --output training/Modelfile.saulgpt
```

GGUF-based (after external HF->GGUF conversion):

```bash
python3 training/scripts/create_ollama_modelfile.py \
  --gguf-path /ABS/PATH/saulgpt-merged.gguf \
  --output training/Modelfile.saulgpt
```

Build Ollama model:

```bash
ollama create saulgpt-legal-ft -f training/Modelfile.saulgpt
```

Run:

```bash
ollama run saulgpt-legal-ft
```

## 6. Hook SaulGPT API to Fine-Tuned Model

Set env before running FastAPI:

```bash
export OLLAMA_MODEL=saulgpt-legal-ft
export USE_OLLAMA_CHAT=1
python3 -m uvicorn saulgpt_api:app --host 127.0.0.1 --port 8000 --reload
```

## Notes

- The dataset builder enforces SaulGPT constraints: concise style, no exact statute/section citations, structured report format.
- If training on Mac CPU, expect very slow runs. GPU training is recommended.
- `training/scripts/run_full_pipeline.sh` supports env overrides (`BASE_MODEL`, `EPOCHS`, `MAX_TRAIN_SAMPLES`, `MAX_EVAL_SAMPLES`, `OUT_DIR`).
