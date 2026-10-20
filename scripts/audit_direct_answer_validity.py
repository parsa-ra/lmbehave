#!/usr/bin/env python3
"""Audit parse validity and conditional accuracy in direct-answer sample logs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def response_is_valid(value: Any, valid_labels: set[str]) -> bool:
    if isinstance(value, list) and len(value) == 1:
        value = value[0]
    return isinstance(value, str) and value.strip() in valid_labels


def audit(model_roots: list[Path], valid_labels: set[str]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cells: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {"examples": 0, "valid": 0, "correct": 0, "valid_correct": 0}
    )
    sources: list[dict[str, str]] = []
    for model_root in model_roots:
        sample_paths = sorted((model_root / "tasks").rglob("*_samples_*.jsonl"))
        if not sample_paths:
            raise ValueError(f"No sample logs under {model_root}")
        for path in sample_paths:
            sources.append({"path": str(path), "sha256": sha256(path)})
            task = path.name.split("_samples_", 1)[-1].removesuffix(".jsonl")
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
                row = json.loads(line)
                key = (model_root.name, task)
                cell = cells[key]
                cell["examples"] += 1
                valid = response_is_valid(row.get("filtered_resps"), valid_labels)
                correct = float(row.get("exact_match", 0.0)) > 0.5
                cell["valid"] += int(valid)
                cell["correct"] += int(correct)
                cell["valid_correct"] += int(valid and correct)
                if correct and not valid:
                    raise ValueError(f"Invalid response scored correct in {path}:{line_number}")

    rows: list[dict[str, Any]] = []
    for (model, task), cell in sorted(cells.items()):
        examples = int(cell["examples"])
        valid = int(cell["valid"])
        rows.append(
            {
                "model": model,
                "task": task,
                "examples": examples,
                "valid_responses": valid,
                "valid_response_rate": valid / examples,
                "raw_accuracy": cell["correct"] / examples,
                "accuracy_conditional_on_valid_response": (
                    cell["valid_correct"] / valid if valid else None
                ),
            }
        )
    return rows, {
        "schema_version": 1,
        "scope": (
            "Historical logs retain task-filtered responses but not the raw generated text. "
            "Validity means that the stored parser output is one declared answer label."
        ),
        "valid_labels": sorted(valid_labels),
        "sources": sources,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_roots", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--valid-labels", default="A,B,C,D")
    args = parser.parse_args()
    labels = {item.strip() for item in args.valid_labels.split(",") if item.strip()}
    rows, report = audit([path.resolve() for path in args.model_roots], labels)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report.update(
        {
            "status": "PASS",
            "rows": len(rows),
            "models": sorted({row["model"] for row in rows}),
            "output": str(args.output.resolve()),
            "output_sha256": sha256(args.output),
        }
    )
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(args.output), "report": str(report_path), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
