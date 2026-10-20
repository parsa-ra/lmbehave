#!/usr/bin/env python3
"""Fail before monitored extraction when response parsing is incomplete."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parsed_answer(value: Any) -> str | None:
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, str) and value.upper() in {"A", "B", "C", "D"}:
        return value.upper()
    return None


def sample_files(root: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for path in sorted(root.rglob("*_samples_*.jsonl")):
        task = path.stem.split("_samples_", 1)[-1]
        if task in output:
            raise ValueError(f"Duplicate sample file for {task} under {root}")
        output[task] = path
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--expected-examples", type=int, default=4)
    args = parser.parse_args()

    expected_tasks = set(args.task)
    files = sample_files(args.samples.resolve())
    if set(files) != expected_tasks:
        raise ValueError("Sample files do not match the declared tasks")
    rows = []
    invalid = 0
    missing_raw = 0
    for task in sorted(expected_tasks):
        samples = [json.loads(line) for line in files[task].read_text(encoding="utf-8").splitlines()]
        if len(samples) != args.expected_examples:
            raise ValueError(f"{task}: expected {args.expected_examples} samples, found {len(samples)}")
        task_invalid = sum(parsed_answer(sample.get("filtered_resps")) is None for sample in samples)
        task_missing_raw = sum("resps" not in sample for sample in samples)
        invalid += task_invalid
        missing_raw += task_missing_raw
        rows.append(
            {
                "task": task,
                "total": len(samples),
                "valid": len(samples) - task_invalid,
                "invalid": task_invalid,
                "missing_raw_response": task_missing_raw,
            }
        )
    passed = invalid == 0 and missing_raw == 0
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if passed else "FAIL_PARSER_OR_RAW_RESPONSE",
        "invalid_response_count": invalid,
        "missing_raw_response_count": missing_raw,
        "tasks": rows,
        "claim_limit": "This gate validates response logging and final-answer parsing only; it does not validate representation capture or observer parity.",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
