#!/usr/bin/env python3
"""QLoRA / LoRA SFT trainer for Qwen/Qwen3.5-2B (T4-friendly defaults)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "sft_train.jsonl"
DEFAULT_OUT = ROOT / "outputs" / "sublayer-specialist-v1-lora"
BASE_MODEL = "Qwen/Qwen3.5-2B"


def load_messages_dataset(path: Path):
    from datasets import Dataset

    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            obj = json.loads(raw)
            if "messages" not in obj:
                raise ValueError(f"{path}:{line_no}: missing 'messages'")
            rows.append({"messages": obj["messages"]})
    return Dataset.from_list(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen3.5-2B for JSON extraction")
    parser.add_argument("--data", default=str(DEFAULT_DATA), help="Chat-format SFT JSONL")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT), help="Adapter output directory")
    parser.add_argument("--model", default=BASE_MODEL, help="Base HF model id")
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--num-train-epochs", type=float, default=3)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=50)
    args = parser.parse_args()

    try:
        import torch
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        print(
            "error: training deps missing. Install with: pip install -r requirements-train.txt\n"
            f"detail: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"error: data not found: {data_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f"Loading base model: {args.model} (dtype={dtype})")

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False

    def formatting_func(example):
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
            enable_thinking=False,
        )

    dataset = load_messages_dataset(data_path)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_args = SFTConfig(
        output_dir=str(out_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=dtype == torch.bfloat16,
        fp16=dtype == torch.float16,
        gradient_checkpointing=True,
        max_length=args.max_seq_length,
        packing=False,
        report_to="none",
        dataset_text_field=None,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        formatting_func=formatting_func,
    )

    trainer.train()
    trainer.save_model(str(out_dir))
    tokenizer.save_pretrained(str(out_dir))
    print(f"Saved LoRA adapter + tokenizer -> {out_dir}")


if __name__ == "__main__":
    main()
