#!/usr/bin/env python3
"""Add task-level modality and format indicators to prompt-surface covariates."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

try:
    from .build_representation_nuisance_table import OUTPUT_COLUMNS, build_table
except ImportError:  # Direct script execution.
    from build_representation_nuisance_table import OUTPUT_COLUMNS, build_table


ADDED_COLUMNS = (
    "has_visual_input",
    "format_binary",
    "format_short_answer",
    "format_mixed",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_design(manifest: dict[str, Any], panel_name: str) -> dict[str, dict[str, Any]]:
    panel = manifest["panels"][panel_name]
    design: dict[str, dict[str, Any]] = {}
    for label, field in (("knowledge", "knowledge_tasks"), ("reasoning", "reasoning_tasks")):
        for item in panel[field]:
            task = str(item["task"])
            if task in design:
                raise ValueError(f"Duplicate task in panel: {task}")
            design[task] = {
                "label": label,
                "modality": str(item["modality"]),
                "response_format": str(item["response_format"]),
            }
    return design


def _indicators(task: dict[str, Any]) -> dict[str, int]:
    response_format = str(task["response_format"])
    return {
        "has_visual_input": int(str(task["modality"]) != "text"),
        "format_binary": int(response_format == "binary"),
        "format_short_answer": int(response_format == "short_answer"),
        "format_mixed": int(response_format == "concise_mixed"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_roots", nargs="+", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--panel", default="cross_benchmark_discovery")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    design = _task_design(manifest, args.panel)
    rows, base_report = build_table(args.run_roots)
    observed_tasks = {str(row["task_name"]) for row in rows}
    if observed_tasks != set(design):
        raise ValueError(
            f"Artifact tasks do not match the panel: missing={sorted(set(design) - observed_tasks)}, "
            f"extra={sorted(observed_tasks - set(design))}"
        )
    augmented: list[dict[str, Any]] = []
    for row in rows:
        augmented.append({**row, **_indicators(design[str(row["task_name"])])})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[*OUTPUT_COLUMNS, *ADDED_COLUMNS],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(augmented)

    task_rows = [
        {"task": task, **design[task], **_indicators(design[task])}
        for task in sorted(design)
    ]
    task_path = args.output.with_name("multimodal_task_design.csv")
    with task_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(task_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(task_rows)

    report = {
        **base_report,
        "schema_version": 1,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": _sha256(args.manifest),
        "panel": args.panel,
        "columns": [*OUTPUT_COLUMNS, *ADDED_COLUMNS],
        "task_design": task_rows,
        "visual_indicator_scope": (
            "Task-level presence of a visual input only; image pixels, visual-token count, "
            "and image content are not represented."
        ),
        "format_reference": "Multiple-choice and generation formats not covered by the three indicators.",
        "output_sha256": _sha256(args.output),
        "task_design_sha256": _sha256(task_path),
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(augmented), "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
