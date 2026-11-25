# sublayer-specialist-v1

Compact open-source specialist for **messy support text → strict JSON extraction**.

Target base model for LoRA / QLoRA fine-tuning (free T4 / Kaggle): [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B).

## Task

Given noisy unstructured customer messages (typos, ALL CAPS, decoy numbers, prompt-injection noise), emit **only** a raw JSON object matching [`schema.json`](schema.json):

| Field | Type | Notes |
| --- | --- | --- |
| `category` | `string \| null` | `billing`, `bug`, `shipping`, `account`, `other`, or `null` |
| `urgency` | `string` | `low`, `medium`, `high`, `critical` |
| `customer_email` | `string \| null` | Explicit customer email only — never invent; ignore `support@` / `noreply@` |
| `order_id` | `string \| null` | Explicit only — never invent; ignore phones, dates, tracking barcodes |
| `action_required` | `boolean` | `false` for compliments / feedback-only |
| `summary` | `string` | One concise factual sentence (≤20 words) |

`additionalProperties: false`. Unstated identifiers must be `null`.

## Setup

CPU / local (dataset + eval):

```bash
pip install -r requirements.txt
```

GPU / Kaggle (train + inference):

```bash
pip install -r requirements-train.txt
```

> On Windows local machines in this project, prefer `py -3` over `python`.

## End-to-end workflow

### 1) Build + validate dataset (CPU)

```bash
py -3 scripts/build_dataset.py
py -3 scripts/validate.py data/train.jsonl
py -3 scripts/validate.py data/test.jsonl
```

Produces leak-free splits: `data/train.jsonl` (300) and `data/test.jsonl` (100), seeded with `random.seed(42)`.

### 2) Format SFT chat JSONL (CPU)

```bash
py -3 scripts/format_sft.py --input data/train.jsonl --output data/sft_train.jsonl
```

### 3) Train LoRA adapter (GPU)

```bash
py -3 scripts/train_lora.py \
  --data data/sft_train.jsonl \
  --output-dir outputs/sublayer-specialist-v1-lora
```

Defaults are T4-friendly: `batch=4`, `grad_accum=4`, `epochs=3`, `lr=2e-4`, `max_seq_length=512`, `gradient_checkpointing=True`.

### 4) Run inference (GPU)

Base model:

```bash
py -3 scripts/run_inference.py --mode base \
  --input data/test.jsonl \
  --output outputs/preds_base.jsonl
```

Fine-tuned LoRA:

```bash
py -3 scripts/run_inference.py --mode lora \
  --adapter-dir outputs/sublayer-specialist-v1-lora \
  --input data/test.jsonl \
  --output outputs/preds_lora.jsonl
```

### 5) Evaluate (CPU)

```bash
py -3 scripts/eval.py --gold data/test.jsonl --preds outputs/preds_base.jsonl
py -3 scripts/eval.py --gold data/test.jsonl --preds outputs/preds_lora.jsonl
```

Self-check (gold vs gold → 100%):

```bash
py -3 scripts/eval.py --preds data/test.jsonl --gold data/test.jsonl
```

## Benchmark Results

| Metric | Base `Qwen3.5-2B` | Fine-tuned (LoRA) |
| --- | ---: | ---: |
| Raw JSON Parse Rate (%) | — | — |
| Strict Schema Compliance Rate (%) | — | — |
| Exact Match: `category` (%) | — | — |
| Exact Match: `urgency` (%) | — | — |
| Exact Match: `customer_email` (%) | — | — |
| Exact Match: `order_id` (%) | — | — |
| Exact Match: `action_required` (%) | — | — |
| Overall Strict Exact Match (%) | — | — |

## License

Apache-2.0 (planned).
