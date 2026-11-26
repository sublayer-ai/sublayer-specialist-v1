#!/usr/bin/env python3
"""Run greedy inference with base or LoRA-adapted Qwen/Qwen3.5-2B."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from prompt import SYSTEM_PROMPT  # noqa: E402

DEFAULT_INPUT = ROOT / "data" / "test.jsonl"
DEFAULT_ADAPTER = ROOT / "outputs" / "sublayer-specialist-v1-lora"
BASE_MODEL = "Qwen/Qwen3.5-2B"


def load_inputs(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if "input" not in obj or not isinstance(obj["input"], str):
                raise ValueError(f"{path}:{line_no}: missing string 'input'")
            rows.append(obj)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Greedy inference for extraction specialist")
    parser.add_argument("--mode", choices=["base", "lora"], required=True)
    parser.add_argument("--adapter-dir", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=None, help="Predictions JSONL path")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    if args.output is None:
        args.output = str(
            ROOT / "outputs" / ("preds_lora.jsonl" if args.mode == "lora" else "preds_base.jsonl")
        )

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        print(
            "error: inference deps missing. Install with: pip install -r requirements-train.txt\n"
            f"detail: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = load_inputs(in_path)

    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    )
    print(f"Loading {args.model} (mode={args.mode}, dtype={dtype})")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.mode == "lora":
        try:
            from peft import PeftModel
        except ImportError as exc:
            print(f"error: peft required for --mode lora ({exc})", file=sys.stderr)
            sys.exit(1)
        adapter = Path(args.adapter_dir)
        if not adapter.exists():
            print(f"error: adapter not found: {adapter}", file=sys.stderr)
            sys.exit(1)
        model = PeftModel.from_pretrained(model, str(adapter))
        print(f"Attached LoRA adapter from {adapter}")

    model.eval()

    with out_path.open("w", encoding="utf-8") as fout:
        for i, row in enumerate(rows, start=1):
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": row["input"]},
            ]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            inputs = tokenizer(prompt, return_tensors="pt")
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                # Deterministic greedy decode — do not pass temperature when do_sample=False.
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            new_tokens = generated[0, inputs["input_ids"].shape[-1] :]
            raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            fout.write(
                json.dumps(
                    {"input": row["input"], "raw_output": raw_output},
                    ensure_ascii=False,
                )
                + "\n"
            )
            if i % 10 == 0 or i == len(rows):
                print(f"  [{i}/{len(rows)}] wrote predictions")

    print(f"Wrote {len(rows)} predictions -> {out_path}")


if __name__ == "__main__":
    main()
