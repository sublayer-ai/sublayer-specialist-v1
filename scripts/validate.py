#!/usr/bin/env python3
"""Validate a JSONL dataset against schema.json."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

try:
    import jsonschema
except ImportError:
    print("error: jsonschema is required. Install with: pip install jsonschema", file=sys.stderr)
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schema.json"
DEFAULT_DATA = ROOT / "data" / "seed.jsonl"


def load_schema() -> dict:
    with SCHEMA_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def validate_file(path: Path) -> int:
    schema = load_schema()
    validator = jsonschema.Draft7Validator(schema)

    errors: list[str] = []
    records: list[dict] = []
    inputs_seen: dict[str, int] = {}

    if not path.exists():
        print(f"error: file not found: {path}", file=sys.stderr)
        return 1

    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                errors.append(f"line {line_no}: empty line")
                continue

            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_no}: invalid JSON ({exc})")
                continue

            if not isinstance(obj, dict):
                errors.append(f"line {line_no}: expected object, got {type(obj).__name__}")
                continue

            if "input" not in obj:
                errors.append(f"line {line_no}: missing 'input'")
            elif not isinstance(obj["input"], str) or not obj["input"].strip():
                errors.append(f"line {line_no}: 'input' must be a non-empty string")
            else:
                prev = inputs_seen.get(obj["input"])
                if prev is not None:
                    errors.append(
                        f"line {line_no}: duplicate input (first seen on line {prev})"
                    )
                else:
                    inputs_seen[obj["input"]] = line_no

            if "output" not in obj:
                errors.append(f"line {line_no}: missing 'output'")
            else:
                out = obj["output"]
                if not isinstance(out, dict):
                    errors.append(
                        f"line {line_no}: 'output' must be an object, got {type(out).__name__}"
                    )
                else:
                    schema_errs = sorted(validator.iter_errors(out), key=lambda e: list(e.path))
                    for err in schema_errs:
                        loc = ".".join(str(p) for p in err.path) or "(root)"
                        errors.append(f"line {line_no}: schema ({loc}): {err.message}")

            if isinstance(obj, dict) and "input" in obj and "output" in obj:
                records.append(obj)

    print(f"Validated: {path}")
    print(f"Records:  {len(records)}")
    print(f"Errors:   {len(errors)}")

    if records:
        cat = Counter(r["output"].get("category") for r in records)
        urg = Counter(r["output"].get("urgency") for r in records)
        act = Counter(r["output"].get("action_required") for r in records)

        print("\n## Distribution")
        print("\n### category")
        for k, v in sorted(cat.items(), key=lambda x: (str(x[0]),)):
            print(f"  {k!s:12} {v}")
        print("\n### urgency")
        for k, v in sorted(urg.items(), key=lambda x: str(x[0])):
            print(f"  {k!s:12} {v}")
        print("\n### action_required")
        for k, v in sorted(act.items(), key=lambda x: str(x[0])):
            print(f"  {k!s:12} {v}")

    if errors:
        print("\n## Errors", file=sys.stderr)
        for msg in errors:
            print(f"  - {msg}", file=sys.stderr)
        return 1

    print("\nOK: all records valid.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate extraction JSONL against schema.json")
    parser.add_argument(
        "path",
        nargs="?",
        default=str(DEFAULT_DATA),
        help=f"Path to .jsonl file (default: {DEFAULT_DATA})",
    )
    args = parser.parse_args()
    sys.exit(validate_file(Path(args.path)))


if __name__ == "__main__":
    main()
