#!/usr/bin/env python3
"""Side-by-side GOLD vs PRED inspection for a 5-example slice."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD_PATH = ROOT / "outputs" / "test_5.jsonl"
PRED_PATH = ROOT / "outputs" / "preds_5.jsonl"


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            rows.append(json.loads(raw))
    return rows


def main() -> None:
    gold_rows = load_jsonl(GOLD_PATH)
    pred_rows = load_jsonl(PRED_PATH)
    n = min(len(gold_rows), len(pred_rows))
    if len(gold_rows) != len(pred_rows):
        print(
            f"warning: length mismatch gold={len(gold_rows)} pred={len(pred_rows)}; showing {n}",
            file=sys.stderr,
        )

    for i in range(n):
        gold = gold_rows[i]
        pred = pred_rows[i]
        gold_out = gold.get("output", gold)
        pred_out = pred.get("raw_output", pred.get("output", pred))
        inp = gold.get("input") or pred.get("input") or ""

        print("=" * 72)
        print(f"[{i + 1}/{n}] INPUT")
        print(inp)
        print("-" * 72)
        print("GOLD (output):")
        if isinstance(gold_out, (dict, list)):
            print(json.dumps(gold_out, ensure_ascii=False, indent=2))
        else:
            print(gold_out)
        print("-" * 72)
        print("PRED (raw_output):")
        print(pred_out)
        print()


if __name__ == "__main__":
    main()
