#!/usr/bin/env python3
"""Validate and summarize the appendix-only heterogeneous multimodal analysis."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any


MODELS = ("qwen3_5-9b", "qwen3_5-27b", "gemma4-31b")
CONTROLS = ("raw", "surface_modality", "lexical_64", "combined_64")
SOURCE_ARCHIVE_SHA256 = "589401e880348a103d0cc985ca57b43f34b720fe4818df3693717fb6384d40e4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def telemetry(path: Path) -> dict[str, float | int]:
    rows: list[tuple[float, float, float, float]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) == 5:
                rows.append(tuple(float(value.strip()) for value in row[1:]))
    if not rows:
        raise ValueError("GPU telemetry is empty")
    utilization, memory, power, limit = zip(*rows)
    return {
        "samples": len(rows),
        "mean_gpu_utilization_percent": sum(utilization) / len(utilization),
        "peak_gpu_utilization_percent": max(utilization),
        "mean_memory_mib": sum(memory) / len(memory),
        "peak_memory_mib": max(memory),
        "mean_power_w": sum(power) / len(power),
        "peak_power_w": max(power),
        "power_limit_w": max(limit),
    }


def exact_upper_tail_p(values: list[float], observed: float) -> float:
    return sum(value >= observed - 1e-12 for value in values) / len(values)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--expected-head", required=True)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    source_audit_path = source / "preflight" / "source_artifact_validation.json"
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    if len(source_audit) != 30:
        raise ValueError(f"Expected 30 source artifacts, found {len(source_audit)}")
    for artifact in source_audit:
        if (
            artifact.get("status") != "PASS"
            or artifact.get("records") not in (9600, 18000, 19200)
            or artifact.get("phases") != ["prefill"]
            or artifact.get("storage_dtype") != "float32"
            or list(artifact.get("tasks", {}).values()) != [100]
        ):
            raise ValueError(f"Invalid source artifact audit row: {artifact}")

    nuisance_path = source / "multimodal_nuisance_features.csv"
    nuisance_report_path = nuisance_path.with_suffix(".json")
    task_design_path = source / "multimodal_task_design.csv"
    nuisance_report = json.loads(nuisance_report_path.read_text(encoding="utf-8"))
    if (
        nuisance_report.get("num_samples") != 3000
        or nuisance_report.get("num_record_files") != 30
        or len(nuisance_report.get("task_design", [])) != 10
        or sha256(nuisance_path) != nuisance_report.get("output_sha256")
        or sha256(task_design_path) != nuisance_report.get("task_design_sha256")
    ):
        raise ValueError("Invalid multimodal nuisance-table report")

    prompt_rows: list[dict[str, Any]] = []
    sources: list[Path] = [
        source_audit_path,
        nuisance_path,
        nuisance_report_path,
        task_design_path,
        *(sorted((source / "preflight").glob("*_prompt_audit.json"))),
    ]
    for path in sorted((source / "preflight").glob("*_prompt_audit.json")):
        audit = json.loads(path.read_text(encoding="utf-8"))
        if audit.get("status") != "PASS" or audit.get("failure_count") != 0:
            raise ValueError(f"Invalid prompt audit: {path}")
    for kind in ("word_tfidf", "char_tfidf"):
        path = source / f"prompt_{kind}.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "PASS" or report.get("assignments") != 252:
            raise ValueError(f"Invalid prompt baseline: {path}")
        if len(report.get("macro_f1_distribution", [])) != 252 or abs(
            exact_upper_tail_p(report["macro_f1_distribution"], report["macro_f1_distribution"][0])
            - report["macro_f1"]["randomization_p"]
        ) > 1e-12:
            raise ValueError(f"Prompt baseline p-value mismatch: {path}")
        prompt_rows.append(
            {
                "baseline": kind,
                "accuracy": report["accuracy"]["observed"],
                "accuracy_p": report["accuracy"]["randomization_p"],
                "macro_f1": report["macro_f1"]["observed"],
                "macro_f1_p": report["macro_f1"]["randomization_p"],
                "assignments": report["assignments"],
                "probe_refits": report["probe_refits"],
            }
        )
        sources.append(path)

    model_rows: list[dict[str, Any]] = []
    descriptive_rows: list[dict[str, Any]] = []
    for model in MODELS:
        fisher_path = source / "fisher_nulls" / model / "report.json"
        fisher = json.loads(fisher_path.read_text(encoding="utf-8"))
        if fisher.get("failures") or fisher.get("fdr_group_permutations_requested") != 251:
            raise ValueError(f"Invalid Fisher analysis: {fisher_path}")
        if len(fisher["selection_aware_randomization_by_model"]) != 1:
            raise ValueError(f"Expected one model in {fisher_path}")
        model_name = next(iter(fisher["selection_aware_randomization_by_model"]))
        fisher_inference = fisher["selection_aware_randomization_by_model"][model_name]["metrics"]["fdr"]
        maximum = fisher["primary_maximum_divergence_by_model"][model_name]
        fisher_null_path = fisher_path.with_name("permutation_nulls.json.gz")
        with gzip.open(fisher_null_path, "rt", encoding="utf-8") as handle:
            fisher_nulls = json.load(handle)
        assignment_maxima = [
            max(row["fdr"][assignment] for row in fisher_nulls["rows"])
            for assignment in range(252)
        ]
        if abs(assignment_maxima[0] - maximum["value"]) > 1e-12 or abs(
            exact_upper_tail_p(assignment_maxima, assignment_maxima[0])
            - fisher_inference["maximum_randomization_p"]
        ) > 1e-12:
            raise ValueError(f"Fisher maximum p-value mismatch: {fisher_null_path}")
        row: dict[str, Any] = {
            "model": model,
            "selected_layer": maximum["layer"],
            "selected_stream": maximum["stream"],
            "fisher_maximum": maximum["value"],
            "fisher_maximum_p": fisher_inference["maximum_randomization_p"],
        }
        layer_path = fisher_path.with_name("layer_separability.csv")
        with layer_path.open(encoding="utf-8", newline="") as handle:
            layer_rows = list(csv.DictReader(handle))
        selected_rows = [
            item
            for item in layer_rows
            if int(item["layer"]) == int(maximum["layer"])
            and item["stream"] == maximum["stream"]
            and item["module"] == maximum["module"]
        ]
        if len(selected_rows) != 1 or abs(float(selected_rows[0]["fdr"]) - maximum["value"]) > 1e-12:
            raise ValueError(f"Selected-site row mismatch: {layer_path}")
        selected = selected_rows[0]
        cka = float(selected["cka"])
        cka_pairs = int(selected["cka_pairs"])
        if math.isfinite(cka) or cka_pairs != 0:
            raise ValueError(f"Unpaired panel unexpectedly reports CKA: {layer_path}")
        descriptive_rows.append(
            {
                "model": model,
                "selected_layer": maximum["layer"],
                "selected_stream": maximum["stream"],
                "rbf_probe_f1": selected["rbf_probe_f1"],
                "silhouette_cosine": selected["silhouette_cosine"],
                "covariance_alignment": selected["covariance_alignment"],
                "effective_rank_within_fraction": selected["effective_rank_within_fraction"],
                "stable_rank_within": selected["stable_rank_within"],
                "cka": None,
                "cka_pairs": cka_pairs,
            }
        )
        sources.extend((fisher_path, fisher_null_path, layer_path))
        for control in CONTROLS:
            path = source / "probes" / control / model / "report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            implementation = report["configuration"]["implementation"]
            if (
                report.get("status") != "PASS"
                or report.get("assignments") != 252
                or report.get("probe_refits") != 126
                or implementation.get("git_head") != args.expected_head
                or implementation.get("git_dirty")
                or not report["macro_f1"]["complement_symmetry"]["verified"]
            ):
                raise ValueError(f"Invalid selected-site report: {path}")
            distribution_path = Path(report["outputs"]["distribution"])
            with gzip.open(distribution_path, "rt", encoding="utf-8") as handle:
                distribution = json.load(handle)
            if len(distribution["macro_f1"]) != 252 or len(distribution["accuracy"]) != 252:
                raise ValueError(f"Invalid selected-site distribution: {distribution_path}")
            if abs(distribution["macro_f1"][0] - report["macro_f1"]["observed"]) > 1e-12:
                raise ValueError(f"Observed macro-F1 mismatch: {distribution_path}")
            if abs(
                exact_upper_tail_p(distribution["macro_f1"], distribution["macro_f1"][0])
                - report["macro_f1"]["randomization_p"]
            ) > 1e-12:
                raise ValueError(f"Macro-F1 p-value mismatch: {distribution_path}")
            row[f"{control}_f1"] = report["macro_f1"]["observed"]
            row[f"{control}_f1_p"] = report["macro_f1"]["randomization_p"]
            row[f"{control}_accuracy"] = report["accuracy"]["observed"]
            row[f"{control}_accuracy_p"] = report["accuracy"]["randomization_p"]
            sources.extend((path, distribution_path))
        model_rows.append(row)

    output_files = {
        "model_summary.csv": model_rows,
        "selected_site_descriptive_metrics.csv": descriptive_rows,
        "prompt_baseline_summary.csv": prompt_rows,
    }
    for name, rows in output_files.items():
        write_csv(output / name, rows)
    report = {
        "schema_version": 1,
        "experiment_id": "representation-dynamics-multimodal-side-v1",
        "status": "PASS",
        "role": "appendix_only_side_analysis",
        "implementation_revision": args.expected_head,
        "models": list(MODELS),
        "assignments": 252,
        "complement_equivalence_classes": 126,
        "minimum_attainable_p": 1 / 126,
        "source_archive_sha256": SOURCE_ARCHIVE_SHA256,
        "telemetry_status": (
            "excluded: an idempotence preflight overwrote the original full-run telemetry "
            "with one idle sample; inferential outputs were not modified"
        ),
        "source_sha256": {str(path.relative_to(source)): sha256(path) for path in sorted(set(sources))},
        "output_sha256": {name: sha256(output / name) for name in sorted(output_files)},
        "claim_limit": (
            "The ten named tasks confound modality, domain, benchmark family, and response format. "
            "Text-channel and task-level visual-presence controls do not represent image content. "
            "No result is pooled with or substituted for the primary text-only analysis."
        ),
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
