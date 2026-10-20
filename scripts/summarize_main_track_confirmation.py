#!/usr/bin/env python3
"""Create a hash-bound, sample-free summary of the 8+8-task confirmation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments" / "representation_dynamics_main_track_v1.json"
PANEL = "matched_mmlu_8x8_confirmatory"
ANALYSIS_NAME = "analysis-cuml-26.06"
MODEL_ORDER = (
    "olmo3-7b-instruct",
    "apertus15-8b",
    "qwen3_5-9b",
    "qwen3_5-27b",
    "gemma4-31b",
)
OUTPUT_COLUMNS = (
    "model",
    "knowledge_accuracy",
    "reasoning_accuracy",
    "fisher_maximum",
    "fisher_maximum_p",
    "fisher_null_median",
    "fisher_null_q95",
    "maximum_layer",
    "maximum_stream",
    "maximum_module",
    "maximum_row_p",
    "maximum_row_q",
    "fisher_bh_rejections",
    "fisher_rejection_count_p",
    "reference_partition_maximum_p",
    "analysis_rows",
    "analysis_failures",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_tasks_sha256(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    encoded = json.dumps(
        payload["tasks"], sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def one(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(f"Expected exactly one {description}, found {len(paths)}")
    return paths[0]


def task_accuracies(model_root: Path, manifest: dict[str, Any]) -> dict[str, float]:
    classes = {
        "knowledge": [
            item["task"] for item in manifest["panels"][PANEL]["knowledge_tasks"]
        ],
        "reasoning": [
            item["task"] for item in manifest["panels"][PANEL]["reasoning_tasks"]
        ],
    }
    output: dict[str, float] = {}
    for class_name, task_names in classes.items():
        task_scores = []
        for task_name in task_names:
            task_root = (
                model_root / "tasks" / f"{class_name}-{task_name}" / "evaluation"
            )
            sample_path = one(
                sorted(task_root.rglob(f"*_samples_{task_name}.jsonl")),
                f"{task_name} sample log",
            )
            rows = [
                json.loads(line)
                for line in sample_path.read_text(encoding="utf-8").splitlines()
            ]
            if len(rows) != int(manifest["extraction"]["sample_count_per_task"]):
                raise RuntimeError(
                    f"Unexpected sample count in {sample_path}: {len(rows)}"
                )
            values = [row.get("exact_match") for row in rows]
            if any(value not in {0, 0.0, 1, 1.0} for value in values):
                raise RuntimeError(
                    f"Non-binary or absent exact_match value in {sample_path}"
                )
            task_scores.append(sum(float(value) for value in values) / len(values))
        output[f"{class_name}_accuracy"] = sum(task_scores) / len(task_scores)
    return output


def summarize_model(
    model: str, run_root: Path, manifest: dict[str, Any]
) -> tuple[dict[str, Any], list[dict[str, str]]]:
    model_root = run_root / PANEL / model
    analysis_root = model_root / ANALYSIS_NAME
    report_path = analysis_root / "report.json"
    rows_path = analysis_root / "layer_separability.csv"
    nulls_path = analysis_root / "permutation_nulls.json.gz"
    analysis_manifest_path = analysis_root / "analysis_manifest.json"
    for path in (report_path, rows_path, nulls_path, analysis_manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("analysis_backend") != "cuml"
        or report.get("analysis_backend_version") != "26.06.00"
    ):
        raise RuntimeError(f"Unexpected estimator family for {model}")
    if report.get("failures"):
        raise RuntimeError(f"Analysis failures are present for {model}")
    if report.get("fdr_group_permutations_requested") != 12869:
        raise RuntimeError(f"Incomplete Fisher assignment support for {model}")
    calibration_by_model = report["selection_aware_randomization_by_model"]
    if len(calibration_by_model) != 1:
        raise RuntimeError(f"Expected one within-model calibration for {model}")
    calibration = next(iter(calibration_by_model.values()))
    if calibration.get("assignments") != 12870:
        raise RuntimeError(f"Unexpected assignment count for {model}")
    fisher = calibration["metrics"]["fdr"]
    if not fisher["maximum_complement_symmetry"]["verified"]:
        raise RuntimeError(f"Fisher complement symmetry failed for {model}")
    with rows_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(report["num_rows"]):
        raise RuntimeError(f"CSV/report row mismatch for {model}")
    maximum = max(rows, key=lambda row: float(row["fdr"]))
    if abs(float(maximum["fdr"]) - float(fisher["observed_maximum"])) > 1e-12:
        raise RuntimeError(f"CSV/report Fisher maximum mismatch for {model}")
    accuracy = task_accuracies(model_root, manifest)
    summary = {
        "model": model,
        **accuracy,
        "fisher_maximum": float(fisher["observed_maximum"]),
        "fisher_maximum_p": float(fisher["maximum_randomization_p"]),
        "fisher_null_median": float(fisher["null_maximum_median"]),
        "fisher_null_q95": float(fisher["null_maximum_q95"]),
        "maximum_layer": int(maximum["layer"]),
        "maximum_stream": maximum["stream"],
        "maximum_module": maximum["module"],
        "maximum_row_p": float(maximum["fdr_p"]),
        "maximum_row_q": float(maximum["fdr_q"]),
        "fisher_bh_rejections": int(fisher["observed_bh_rejections"]),
        "fisher_rejection_count_p": float(fisher["rejection_count_randomization_p"]),
        "reference_partition_maximum_p": float(
            fisher["reference_partition"]["maximum_randomization_p"]
        ),
        "analysis_rows": int(report["num_rows"]),
        "analysis_failures": len(report["failures"]),
    }
    sources = [
        {"path": str(path), "sha256": sha256_file(path)}
        for path in (report_path, rows_path, nulls_path, analysis_manifest_path)
    ]
    return summary, sources


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--standard-root", required=True, type=Path)
    parser.add_argument("--apertus-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tuple(manifest["models"]) != MODEL_ORDER:
        raise RuntimeError("The frozen manifest model order changed")
    standard_root = args.standard_root.resolve()
    apertus_root = args.apertus_root.resolve()
    selection_paths = [
        standard_root / "sample_selection.json",
        apertus_root / "sample_selection.json",
    ]
    task_digests = [canonical_tasks_sha256(path) for path in selection_paths]
    if len(set(task_digests)) != 1:
        raise RuntimeError(
            "The runtime shards do not use identical frozen task/sample identities"
        )
    rows = []
    sources: list[dict[str, str]] = []
    for model in MODEL_ORDER:
        run_root = apertus_root if model == "apertus15-8b" else standard_root
        row, model_sources = summarize_model(model, run_root, manifest)
        rows.append(row)
        sources.extend(model_sources)
    for run_root in (standard_root, apertus_root):
        for name in (
            "run_manifest.json",
            "sample_selection.json",
            "validation_report.json",
            "completed_manifest.json",
        ):
            path = run_root / name
            if not path.is_file():
                raise FileNotFoundError(path)
            sources.append({"path": str(path), "sha256": sha256_file(path)})
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "main_track_model_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "status": "PASS_FIVE_MODEL_SELECTION_AWARE_FISHER_COMPLETE",
        "decision_threshold": float(manifest["confirmatory_inference"]["alpha"]),
        "models_passing_primary_threshold": [
            row["model"]
            for row in rows
            if row["fisher_maximum_p"]
            < float(manifest["confirmatory_inference"]["alpha"])
        ],
        "models_not_passing_primary_threshold": [
            row["model"]
            for row in rows
            if row["fisher_maximum_p"]
            >= float(manifest["confirmatory_inference"]["alpha"])
        ],
        "sample_selection_tasks_sha256": task_digests[0],
        "rows": rows,
        "claim_limit": "The complete task-randomization test establishes whether the named 8+8 MMLU subject partition is unusually separated relative to arbitrary balanced partitions within each model. It does not establish a universal knowledge-reasoning geometry, shared localization, or causal mechanism.",
        "provenance": {
            "manifest": str(manifest_path.relative_to(ROOT)),
            "manifest_sha256": sha256_file(manifest_path),
            "sources": sources,
            "output_csv": csv_path.name,
            "output_csv_sha256": sha256_file(csv_path),
        },
        "contains_activations_prompts_responses_or_per_sample_rows": False,
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "models": len(rows),
                "csv": str(csv_path),
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
