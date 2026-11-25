#!/usr/bin/env python3
"""Evaluate prediction JSONL against ground-truth JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

try:
    import jsonschema
except ImportError:
    print("error: jsonschema is required. Install with: pip install jsonschema", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schema.json"
DEFAULT_GOLD = ROOT / "data" / "test.jsonl"
DEFAULT_SEED = ROOT / "data" / "seed.jsonl"

STRUCTURED_FIELDS = (
    "category",
    "urgency",
    "customer_email",
    "order_id",
    "action_required",
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                rows.append(json.loads(raw))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON ({exc})") from exc
    return rows


def extract_prediction(row: dict[str, Any]) -> Any:
    """Accept raw model strings or dataset-style {input, output} objects."""
    if "raw_output" in row:
        return row["raw_output"]
    if "prediction" in row:
        return row["prediction"]
    if "output" in row:
        return row["output"]
    if "assistant" in row:
        return row["assistant"]
    if "messages" in row and isinstance(row["messages"], list):
        for msg in reversed(row["messages"]):
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                return msg.get("content")
    raise KeyError(
        "no prediction field found (expected raw_output|prediction|output|assistant|messages)"
    )


def try_parse_json(value: Any) -> tuple[bool, Any]:
    if isinstance(value, dict):
        return True, value
    if not isinstance(value, str):
        return False, None
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return True, json.loads(text)
    except json.JSONDecodeError:
        return False, None


def pct(numer: int, denom: int) -> float:
    if denom == 0:
        return 0.0
    return 100.0 * numer / denom


def evaluate(gold_path: Path, pred_path: Path) -> int:
    schema = json.load(SCHEMA_PATH.open(encoding="utf-8"))
    validator = jsonschema.Draft7Validator(schema)

    gold_rows = load_jsonl(gold_path)
    pred_rows = load_jsonl(pred_path)

    if len(gold_rows) != len(pred_rows):
        print(
            f"error: length mismatch gold={len(gold_rows)} pred={len(pred_rows)}",
            file=sys.stderr,
        )
        return 1

    n = len(gold_rows)
    parse_ok = 0
    schema_ok = 0
    field_hits = {f: 0 for f in STRUCTURED_FIELDS}
    strict_exact = 0

    for i, (g, p) in enumerate(zip(gold_rows, pred_rows), start=1):
        gold_out = g.get("output")
        if not isinstance(gold_out, dict):
            print(f"error: gold line {i}: missing object 'output'", file=sys.stderr)
            return 1

        try:
            raw_pred = extract_prediction(p)
        except KeyError as exc:
            print(f"error: pred line {i}: {exc}", file=sys.stderr)
            return 1

        ok, parsed = try_parse_json(raw_pred)
        if not ok or not isinstance(parsed, dict):
            continue
        parse_ok += 1

        if validator.is_valid(parsed):
            schema_ok += 1

        all_match = True
        for field in STRUCTURED_FIELDS:
            if parsed.get(field) == gold_out.get(field):
                field_hits[field] += 1
            else:
                all_match = False
        if all_match:
            strict_exact += 1

    rows = [
        ("Raw JSON Parse Rate (%)", f"{pct(parse_ok, n):.1f}"),
        ("Strict Schema Compliance Rate (%)", f"{pct(schema_ok, n):.1f}"),
    ]
    for field in STRUCTURED_FIELDS:
        rows.append((f"Exact Match: {field} (%)", f"{pct(field_hits[field], n):.1f}"))
    rows.append(("Overall Strict Exact Match (%)", f"{pct(strict_exact, n):.1f}"))

    print(f"Gold: {gold_path}")
    print(f"Pred: {pred_path}")
    print(f"N:    {n}")
    print()
    print("| Metric | Score |")
    print("| --- | ---: |")
    for name, score in rows:
        print(f"| {name} | {score} |")

    return 0


def self_test() -> int:
    """Evaluate gold against itself; expect 100% on all metrics."""
    gold = DEFAULT_GOLD if DEFAULT_GOLD.exists() else DEFAULT_SEED
    code = evaluate(gold, gold)
    if code != 0:
        return code

    rows = load_jsonl(gold)
    schema = json.load(SCHEMA_PATH.open(encoding="utf-8"))
    validator = jsonschema.Draft7Validator(schema)
    n = len(rows)
    for row in rows:
        out = row["output"]
        assert validator.is_valid(out), out
        for field in STRUCTURED_FIELDS:
            assert field in out

    print()
    print(f"Self-test OK: {n}/{n} records scored 100% on parse/schema/exact-match.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate extraction predictions")
    parser.add_argument("--gold", default=str(DEFAULT_GOLD), help="Ground-truth JSONL")
    parser.add_argument(
        "--preds",
        "--pred",
        dest="preds",
        default=None,
        help="Predictions JSONL (raw_output strings or output dicts)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Evaluate gold against itself (expect 100%)",
    )
    args = parser.parse_args()

    if args.self_test:
        sys.exit(self_test())

    if not args.preds:
        parser.error("--preds is required unless --self-test is set")

    sys.exit(evaluate(Path(args.gold), Path(args.preds)))


if __name__ == "__main__":
    main()
