# sublayer-specialist-v1

Compact open-source specialist for **messy support text → strict JSON extraction**.

**Status:** v2.1 is a **research milestone**, not a production-ready support classifier. JSON parse/schema and ID fields are strong; **urgency** and **category** remain limited (see benchmarks below).

- Base model: [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B)
- LoRA adapter (HF): [`sublayerAI/sublayer-specialist-v2.1-lora`](https://huggingface.co/sublayerAI/sublayer-specialist-v2.1-lora)
- License: [Apache-2.0](LICENSE)

## Task & 6-key schema

Given noisy unstructured customer messages (typos, ALL CAPS, decoy numbers, prompt-injection noise), emit **only** a raw JSON object with **exactly these 6 keys** (see [`schema.json`](schema.json) and the shared system prompt in [`scripts/prompt.py`](scripts/prompt.py)):

| Field | Type | Notes |
| --- | --- | --- |
| `category` | `string \| null` | `billing`, `bug`, `shipping`, `account`, `other`, or `null` |
| `urgency` | `string` | `low`, `medium`, `high`, `critical` |
| `customer_email` | `string \| null` | Explicit customer email only — never invent; ignore `support@` / `noreply@` |
| `order_id` | `string \| null` | Explicit only — never invent; ignore phones, dates, tracking barcodes |
| `action_required` | `boolean` | `false` only for compliments / thanks / feedback-only / casual wishes |
| `summary` | `string` | One concise factual sentence (≤20 words) |

`additionalProperties: false`. Unstated identifiers must be `null`. Do not emit extra keys (`tracking`, `phone`, `ref`, etc.).

## LoRA setup (v2.1)

| Setting | Value |
| --- | --- |
| Base | `Qwen/Qwen3.5-2B` |
| LoRA `r` / `alpha` / dropout | 16 / 32 / 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Epochs / LR | 2 / `1e-4` |
| Batch / grad accum | 4 / 4 |
| `max_seq_length` | 512 |
| Train / test | 600 / 100 (stratified urgency ~25% each, seed `42`) |

## Setup

CPU / local (dataset + eval):

```bash
pip install -r requirements.txt
```

GPU (train + inference):

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

Produces leak-free stratified splits: `data/train.jsonl` (600) and `data/test.jsonl` (100).

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

### 4) Run inference (GPU)

```bash
py -3 scripts/run_inference.py --mode base \
  --input data/test.jsonl \
  --output outputs/preds_base.jsonl

py -3 scripts/run_inference.py --mode lora \
  --adapter-dir outputs/sublayer-specialist-v1-lora \
  --input data/test.jsonl \
  --output outputs/preds_lora.jsonl
```

### 5) Evaluate (CPU)

```bash
py -3 scripts/eval.py --gold data/test.jsonl --preds outputs/preds_lora.jsonl
```

## Benchmark Results (N=100 held-out `data/test.jsonl`)

Research progression on the same eval harness (strict structured-field exact match excludes free-text `summary` wording):

| Metric | v1 | v2 | v2.1 |
| --- | ---: | ---: | ---: |
| Raw JSON Parse Rate (%) | 98.0 | 97.0 | 100.0 |
| Strict Schema Compliance Rate (%) | 95.0 | 90.0 | 97.0 |
| Exact Match: `category` (%) | 87.0 | 81.0 | 82.0 |
| Exact Match: `urgency` (%) | 35.0 | 56.0 | 57.0 |
| Exact Match: `customer_email` (%) | 80.0 | 90.0 | 98.0 |
| Exact Match: `order_id` (%) | 75.0 | 87.0 | 94.0 |
| Exact Match: `action_required` (%) | 85.0 | 88.0 | 89.0 |
| Overall Strict Exact Match (%) | 21.0 | 36.0 | 42.0 |

### Known limitations

- **Not production-ready.** Treat v2.1 as an open research checkpoint.
- **Urgency** remains weak (**57.0%**); ALL CAPS / tone traps still confuse severity.
- **Category** is only moderate (**82.0%**).
- Dataset is **synthetic** (programmatic templates + traps): decoy tracking numbers, multi-email noise, prompt-injection strings, and **fake** contact details — not real customer data.

## Attribution

Built on [`Qwen/Qwen3.5-2B`](https://huggingface.co/Qwen/Qwen3.5-2B) with PEFT LoRA / TRL SFT. Downstream use must also comply with the base model’s license terms.

## License

Apache-2.0. See [LICENSE](LICENSE).
