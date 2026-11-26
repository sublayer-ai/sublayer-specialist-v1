#!/usr/bin/env python3
"""Convert extraction JSONL into chat-format SFT training data."""

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

DEFAULT_IN = ROOT / "data" / "train.jsonl"
DEFAULT_OUT = ROOT / "data" / "sft_train.jsonl"


def convert(in_path: Path, out_path: Path) -> int:
    if not in_path.exists():
        print(f"error: input not found: {in_path}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0

    with in_path.open(encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line_no, line in enumerate(fin, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"error: line {line_no}: invalid JSON ({exc})", file=sys.stderr)
                return 1

            if "input" not in row or "output" not in row:
                print(f"error: line {line_no}: missing input/output", file=sys.stderr)
                return 1

            record = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["input"]},
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            row["output"], separators=(",", ":"), ensure_ascii=False
                        ),
                    },
                ]
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            n += 1

    print(f"Wrote {n} SFT examples -> {out_path}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Format extraction JSONL into chat SFT JSONL")
    parser.add_argument("--input", default=str(DEFAULT_IN), help="Source JSONL path")
    parser.add_argument("--output", default=str(DEFAULT_OUT), help="Destination JSONL path")
    args = parser.parse_args()
    sys.exit(convert(Path(args.input), Path(args.output)))


if __name__ == "__main__":
    main()
