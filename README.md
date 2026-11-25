# sublayer-specialist-v1

Compact open-source specialist for **messy support text → strict JSON extraction**.

Target base model for QLoRA fine-tuning (free T4 / Kaggle): [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B).

## Task

Given noisy unstructured customer messages (typos, ALL CAPS, decoy numbers, prompt-injection noise), emit **only** a raw JSON object matching [`schema.json`](schema.json):

| Field | Type | Notes |
| --- | --- | --- |
| `category` | `string \| null` | `billing`, `bug`, `shipping`, `account`, `other`, or `null` |
| `urgency` | `string` | `low`, `medium`, `high`, `critical` |
| `customer_email` | `string \| null` | Explicit only — never invent |
| `order_id` | `string \| null` | Explicit only — never invent; ignore phones/dates |
| `action_required` | `boolean` | `false` for compliments / feedback-only |
| `summary` | `string` | One concise factual sentence (≤20 words) |

`additionalProperties: false`. Unstated identifiers must be `null`.

## Setup (CPU / local)

```bash
pip install -r requirements.txt
```

GPU / Kaggle training deps (later):

```bash
pip install -r requirements-train.txt
```

## Local commands

Validate seed data against the schema:

```bash
python scripts/validate.py data/seed.jsonl
```

Format chat-style SFT JSONL (`system` / `user` / `assistant`):

```bash
python scripts/format_sft.py
# -> data/sft_train.jsonl
```

Self-test the eval harness (seed vs. itself → 100%):

```bash
python scripts/eval.py --self-test
```

Score model predictions:

```bash
python scripts/eval.py --gold data/seed.jsonl --pred path/to/predictions.jsonl
```

## Training plan (v0.1 → T4)

1. Freeze schema + grow `data/seed.jsonl` / synthetic augmentations.
2. QLoRA fine-tune `Qwen/Qwen3.5-2B` with `requirements-train.txt` on a free T4.
3. Eval with `scripts/eval.py`; prioritize parse rate + schema compliance + structured-field exact match over `summary` wording.

## Benchmark Results

| Metric | Base `Qwen3.5-2B` | Fine-tuned (QLoRA) |
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
