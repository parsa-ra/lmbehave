#!/usr/bin/env python3
"""Audit per-task correct/incorrect support in completed main-track evaluations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_model_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Model roots must use NAME=PATH")
    name, raw_path = value.split("=", 1)
    path = Path(raw_path).expanduser().resolve()
    if not name or not path.is_dir():
        raise argparse.ArgumentTypeError(f"Invalid model root: {value}")
    return name, path


def task_name(path: Path) -> str:
    marker = "_samples_"
    if marker not in path.stem:
        raise ValueError(f"Cannot identify task from {path}")
    return path.stem.split(marker, 1)[1]


def audit_model(name: str, root: Path, expected_tasks: int, expected_examples: int) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    paths = sorted(root.rglob("*_samples_*.jsonl"))
    tasks = [task_name(path) for path in paths]
    if len(paths) != expected_tasks or len(set(tasks)) != expected_tasks:
        raise ValueError(
            f"{name}: expected {expected_tasks} unique task sample files, found {len(paths)}"
        )
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, str]] = []
    for path, task in zip(paths, tasks):
        examples = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(examples) != expected_examples:
            raise ValueError(f"{name}/{task}: expected {expected_examples} examples")
        doc_ids = [int(example["doc_id"]) for example in examples]
        if len(set(doc_ids)) != expected_examples:
            raise ValueError(f"{name}/{task}: duplicate evaluation document IDs")
        scores = [float(example["exact_match"]) for example in examples]
        if any(score not in (0.0, 1.0) for score in scores):
            raise ValueError(f"{name}/{task}: exact_match is not binary")
        correct = int(sum(scores))
        incorrect = expected_examples - correct
        rows.append(
            {
                "model": name,
                "task": task,
                "examples": expected_examples,
                "correct": correct,
                "incorrect": incorrect,
                "accuracy": correct / expected_examples,
                "balanced_per_outcome_cap": min(correct, incorrect),
            }
        )
        provenance.append({"path": str(path), "sha256": sha256_file(path)})
    return rows, provenance


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", action="append", required=True, type=parse_model_root)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--expected-tasks", type=int, default=16)
    parser.add_argument("--expected-examples", type=int, default=100)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    provenance: list[dict[str, str]] = []
    for name, root in args.model_root:
        model_rows, model_provenance = audit_model(
            name,
            root,
            expected_tasks=args.expected_tasks,
            expected_examples=args.expected_examples,
        )
        rows.extend(model_rows)
        provenance.extend(model_provenance)
    rows.sort(key=lambda row: (row["model"], row["task"]))
    csv_path = output_dir / "correctness_support.csv"
    write_csv(csv_path, rows)

    summaries = []
    for model in sorted({str(row["model"]) for row in rows}):
        selected = [row for row in rows if row["model"] == model]
        cap = min(int(row["balanced_per_outcome_cap"]) for row in selected)
        summaries.append(
            {
                "model": model,
                "minimum_correct_per_task": min(int(row["correct"]) for row in selected),
                "minimum_incorrect_per_task": min(int(row["incorrect"]) for row in selected),
                "balanced_per_outcome_cap": cap,
                "balanced_examples_total": 2 * args.expected_tasks * cap,
                "all_tasks_have_at_least_5_per_outcome": cap >= 5,
                "all_tasks_have_at_least_10_per_outcome": cap >= 10,
                "all_tasks_have_at_least_20_per_outcome": cap >= 20,
            }
        )
    report = {
        "schema_version": 1,
        "status": "PASS_SUPPORT_AUDIT_ONLY",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "expected_tasks": args.expected_tasks,
        "expected_examples_per_task": args.expected_examples,
        "models": summaries,
        "claim_limit": "This audit measures available outcome strata. Thresholds 5/10/20 are planning diagnostics, not inferential cutoffs. Subsetting on correctness is post-outcome selection and is not a causal adjustment.",
        "recommended_use": "Keep the all-example 8+8 test primary. Any correctness analysis must preserve all tasks, report cell counts, and avoid interpreting conditioning on a downstream outcome as removal of a nuisance variable.",
        "provenance": {
            "sources": provenance,
            "source_set_sha256": hashlib.sha256(
                "\n".join(f'{item["path"]}\t{item["sha256"]}' for item in provenance).encode()
            ).hexdigest(),
            "correctness_support_csv_sha256": sha256_file(csv_path),
        },
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "models": summaries}, indent=2))


if __name__ == "__main__":
    main()
