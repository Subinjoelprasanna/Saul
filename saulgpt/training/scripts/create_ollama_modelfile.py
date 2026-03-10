#!/usr/bin/env python3
"""Create an Ollama Modelfile for SaulGPT fine-tuned model."""

from __future__ import annotations

import argparse
from pathlib import Path

SYSTEM_TEXT = (
    "You are SaulGPT, a legal information assistant for Indian law. "
    "Be concise and practical. "
    "Do not provide specific legal advice. "
    "Do not cite exact sections, acts, or statute numbers. "
    "Identify legal category, explain general context, ask only essential missing facts, "
    "and generate structured reports when facts are sufficient."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Modelfile for Ollama")
    parser.add_argument("--from-model", type=str, default="mistral")
    parser.add_argument("--adapter-path", type=str, default="")
    parser.add_argument("--gguf-path", type=str, default="")
    parser.add_argument("--output", type=Path, default=Path("training/Modelfile.saulgpt"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    if args.gguf_path:
        lines.append(f"FROM {args.gguf_path}")
    else:
        lines.append(f"FROM {args.from_model}")

    if args.adapter_path:
        lines.append(f"ADAPTER {args.adapter_path}")

    lines.extend(
        [
            f'SYSTEM """{SYSTEM_TEXT}"""',
            "PARAMETER temperature 0.1",
            "PARAMETER top_p 0.9",
            "PARAMETER num_ctx 4096",
            "",
        ]
    )

    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Modelfile written: {output}")


if __name__ == "__main__":
    main()
