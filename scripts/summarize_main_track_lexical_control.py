#!/usr/bin/env python3
"""Validate and summarize the fold-local lexical activation control."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


MODELS = (
    "olmo3-7b-instruct",
    "apertus15-8b",
    "qwen3_5-9b",
    "qwen3_5-27b",
    "gemma4-31b",
)
COMPONENT_COUNTS = (32, 64)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_report(path: Path, *, components: int, expected_head: str) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    implementation = report["configuration"]["implementation"]
    nuisance = report["configuration"]["nuisance_control"]
    lexical = nuisance["lexical"]
    fit = report["lexical_nuisance_fit"]
    if report.get("status") != "PASS":
        raise ValueError(f"Non-PASS report: {path}")
    if implementation.get("git_head") != expected_head or implementation.get("git_dirty"):
        raise ValueError(f"Unbound or dirty implementation: {path}")
    if report.get("assignments") != 12870 or report.get("probe_refits") != 6435:
        raise ValueError(f"Incomplete task-label support: {path}")
    if not report.get("selection_recomputed_for_every_assignment"):
        raise ValueError(f"Fisher selection was not repeated: {path}")
    if not report["macro_f1"]["complement_symmetry"]["verified"]:
        raise ValueError(f"Macro-F1 complement symmetry failed: {path}")
    if lexical.get("requested_components") != components:
        raise ValueError(f"Wrong lexical dimension: {path}")
    if lexical.get("vectorizer_fit_scope") != "training_tasks_per_fold":
        raise ValueError(f"TF-IDF is not fold-local: {path}")
    if lexical.get("svd_fit_scope") != "training_tasks_per_fold":
        raise ValueError(f"SVD is not fold-local: {path}")
    if not fit.get("fit_is_inductive") or fit.get("unique_task_folds") != 229:
        raise ValueError(f"Unexpected lexical fold support: {path}")
    if fit.get("actual_component_counts") != [components]:
        raise ValueError(f"Unexpected fitted component count: {path}")
    distribution = Path(report["outputs"]["distribution"])
    if sha256(distribution) != report["outputs"]["distribution_sha256"]:
        raise ValueError(f"Distribution hash failed: {path}")
    return report


def telemetry_summary(path: Path) -> dict[str, float | int]:
    samples: list[tuple[float, float, float, float]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 5:
                raise ValueError(f"Malformed telemetry row: {row}")
            samples.append(tuple(float(value.strip()) for value in row[1:]))
    if not samples:
        raise ValueError("GPU telemetry is empty")
    utilization, memory, power, power_limit = zip(*samples)
    return {
        "samples": len(samples),
        "mean_gpu_utilization_percent": sum(utilization) / len(utilization),
        "peak_gpu_utilization_percent": max(utilization),
        "mean_memory_mib": sum(memory) / len(memory),
        "peak_memory_mib": max(memory),
        "mean_power_w": sum(power) / len(power),
        "peak_power_w": max(power),
        "power_limit_w": max(power_limit),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    source_paths: list[Path] = []
    for components in COMPONENT_COUNTS:
        for model in MODELS:
            path = source / f"char_tfidf_svd{components}" / model / "report.json"
            report = load_report(
                path,
                components=components,
                expected_head=args.expected_head,
            )
            source_paths.append(path)
            site = report["observed_selected_site"]
            rows.append(
                {
                    "components": components,
                    "model": model,
                    "selected_layer": site["layer"],
                    "selected_stream": site["stream"],
                    "accuracy": report["accuracy"]["observed"],
                    "accuracy_p": report["accuracy"]["randomization_p"],
                    "accuracy_null_q95": report["accuracy"]["null_q95"],
                    "macro_f1": report["macro_f1"]["observed"],
                    "macro_f1_p": report["macro_f1"]["randomization_p"],
                    "macro_f1_null_q95": report["macro_f1"]["null_q95"],
                    "assignments": report["assignments"],
                    "probe_refits": report["probe_refits"],
                    "unique_task_folds": report["lexical_nuisance_fit"]["unique_task_folds"],
                    "analysis_git_head": report["configuration"]["implementation"]["git_head"],
                    "source_sha256": sha256(path),
                }
            )

    telemetry_path = source / "gpu_telemetry.csv"
    telemetry = [telemetry_summary(telemetry_path)]
    source_paths.append(telemetry_path)
    outputs = {
        "lexical_adjusted_probe_summary.csv": rows,
        "gpu_telemetry_summary.csv": telemetry,
    }
    for name, output_rows in outputs.items():
        write_csv(output / name, output_rows)
    report = {
        "schema_version": 1,
        "experiment_id": "representation-dynamics-main-track-lexical-control-v1",
        "status": "PASS",
        "persistent_source": str(source),
        "implementation_revision": args.expected_head,
        "source_sha256": {
            str(path.relative_to(source)): sha256(path) for path in sorted(source_paths)
        },
        "output_sha256": {name: sha256(output / name) for name in sorted(outputs)},
        "models": list(MODELS),
        "component_counts": list(COMPONENT_COUNTS),
        "claim_limit": (
            "Post-hoc sensitivity to a low-rank character-TF-IDF nuisance representation "
            "fit within training-task folds. No condition meets the declared p<0.01 rule, "
            "but nominal predictability remains. The analysis neither exhausts lexical "
            "content nor establishes causal mediation."
        ),
    }
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")
    print(json.dumps({"status": "PASS", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
