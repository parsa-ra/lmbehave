#!/usr/bin/env python3
"""Create sample-free, hash-bound summaries of the primary-panel controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


MODELS = (
    "olmo3-7b-instruct",
    "apertus15-8b",
    "qwen3_5-9b",
    "qwen3_5-27b",
    "gemma4-31b",
)
CONTROLS = ("length_only", "six_surface_features")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if payload.get("status") != "PASS":
        raise ValueError(f"Expected PASS report: {path}")
    return payload


def write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    fieldnames = list(fields)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def prompt_rows(source: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    rows: list[dict[str, Any]] = []
    files: list[Path] = []
    for baseline in ("word_tfidf", "char_tfidf"):
        path = source / f"prompt_{baseline}.json"
        report = load_json(path)
        files.append(path)
        rows.append(
            {
                "baseline": baseline,
                "examples": report["num_prompts"],
                "accuracy": report["accuracy"]["observed"],
                "accuracy_p": report["accuracy"]["randomization_p"],
                "accuracy_null_median": report["accuracy"]["null_median"],
                "accuracy_null_q95": report["accuracy"]["null_q95"],
                "macro_f1": report["macro_f1"]["observed"],
                "macro_f1_p": report["macro_f1"]["randomization_p"],
                "macro_f1_null_median": report["macro_f1"]["null_median"],
                "macro_f1_null_q95": report["macro_f1"]["null_q95"],
                "assignments": report["assignments"],
                "probe_refits": report["probe_refits"],
                "source_sha256": sha256(path),
            }
        )
    return rows, files


def adjusted_probe_rows(source: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    rows: list[dict[str, Any]] = []
    files: list[Path] = []
    for control in CONTROLS:
        for model in MODELS:
            path = source / control / model / "report.json"
            report = load_json(path)
            files.append(path)
            nuisance = report["configuration"]["nuisance_control"]
            implementation = report["configuration"]["implementation"]
            if not nuisance.get("enabled"):
                raise ValueError(f"Nuisance control is disabled: {path}")
            if implementation.get("git_dirty"):
                raise ValueError(f"Analysis source was dirty: {path}")
            if not report["macro_f1"]["complement_symmetry"]["verified"]:
                raise ValueError(f"Macro-F1 complement symmetry failed: {path}")
            site = report["observed_selected_site"]
            rows.append(
                {
                    "control": control,
                    "model": model,
                    "nuisance_columns": ";".join(nuisance["columns"]),
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
                    "analysis_git_head": implementation["git_head"],
                    "source_sha256": sha256(path),
                }
            )
    return rows, files


def aggregate_validity(path: Path, group_fields: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], dict[str, float]] = defaultdict(
        lambda: {"examples": 0.0, "valid_responses": 0.0, "correct": 0.0}
    )
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = tuple(row[field] for field in group_fields)
            examples = int(row["examples"])
            grouped[key]["examples"] += examples
            grouped[key]["valid_responses"] += int(row["valid_responses"])
            grouped[key]["correct"] += float(row["raw_accuracy"]) * examples
    rows: list[dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        examples = int(values["examples"])
        valid = int(values["valid_responses"])
        result = dict(zip(group_fields, key))
        result.update(
            {
                "examples": examples,
                "valid_responses": valid,
                "valid_response_rate": valid / examples,
                "raw_accuracy": values["correct"] / examples,
                "accuracy_conditional_on_valid_response": (
                    values["correct"] / valid if valid else None
                ),
            }
        )
        rows.append(result)
    return rows


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def telemetry_rows(path: Path) -> list[dict[str, Any]]:
    samples: list[tuple[float, float, float, float]] = []
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 5:
                raise ValueError(f"Malformed GPU telemetry row in {path}: {row}")
            samples.append(tuple(float(value.strip()) for value in row[1:]))
    if not samples:
        raise ValueError(f"GPU telemetry is empty: {path}")
    utilization, memory, power, power_limit = zip(*samples)
    return [
        {
            "samples": len(samples),
            "mean_gpu_utilization_percent": sum(utilization) / len(utilization),
            "peak_gpu_utilization_percent": max(utilization),
            "mean_memory_mib": sum(memory) / len(memory),
            "peak_memory_mib": max(memory),
            "mean_power_w": sum(power) / len(power),
            "peak_power_w": max(power),
            "power_limit_w": max(power_limit),
        }
    ]


def residual_rows(path: Path) -> list[dict[str, Any]]:
    report = load_json(path)
    rows: list[dict[str, Any]] = []
    for item in report["rows"]:
        artifact = str(item["artifact"])
        model = next((name for name in MODELS if f"/{name}/" in artifact), None)
        if model is None:
            raise ValueError(f"Cannot identify model from artifact path: {artifact}")
        rows.append(
            {
                "model": model,
                "comparisons": item["comparisons"],
                "expected_additivity": item["expected_additivity"],
                "relative_error_median": item["relative_error_median"],
                "relative_error_q95": item["relative_error_q95"],
                "relative_error_maximum": item["relative_error_maximum"],
                "relative_tolerance": item["relative_tolerance"],
                "additivity_verified": item["additivity_verified"],
                "artifact_sha256": item["artifact_sha256"],
            }
        )
    return sorted(rows, key=lambda row: MODELS.index(str(row["model"])))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    prompt, source_files = prompt_rows(source)
    adjusted, adjusted_files = adjusted_probe_rows(source)
    source_files.extend(adjusted_files)

    direct_path = source / "direct_answer_validity.csv"
    longitudinal_path = source / "longitudinal_response_validity.csv"
    additivity_path = source / "residual_additivity.json"
    telemetry_path = source / "gpu_telemetry.csv"
    source_files.extend(
        (
            source / "nuisance_features.csv",
            source / "nuisance_features.json",
            direct_path,
            direct_path.with_suffix(".json"),
            longitudinal_path,
            longitudinal_path.with_suffix(".json"),
            additivity_path,
            telemetry_path,
        )
    )
    direct_by_task = read_csv(direct_path)
    direct = aggregate_validity(direct_path, ("model",))
    longitudinal = aggregate_validity(longitudinal_path, ("family", "checkpoint"))
    additivity = residual_rows(additivity_path)
    telemetry = telemetry_rows(telemetry_path)

    outputs = {
        "prompt_baseline_summary.csv": (
            prompt,
            prompt[0].keys(),
        ),
        "adjusted_probe_summary.csv": (
            adjusted,
            adjusted[0].keys(),
        ),
        "direct_answer_validity_summary.csv": (
            direct,
            direct[0].keys(),
        ),
        "direct_answer_validity_by_task.csv": (
            direct_by_task,
            direct_by_task[0].keys(),
        ),
        "longitudinal_response_validity_summary.csv": (
            longitudinal,
            longitudinal[0].keys(),
        ),
        "residual_additivity_summary.csv": (
            additivity,
            additivity[0].keys(),
        ),
        "gpu_telemetry_summary.csv": (
            telemetry,
            telemetry[0].keys(),
        ),
    }
    for name, (rows, fields) in outputs.items():
        write_csv(output / name, rows, fields)

    report = {
        "schema_version": 1,
        "experiment_id": "representation-dynamics-main-track-controls-v1",
        "status": "PASS",
        "persistent_source": str(source),
        "source_sha256": {
            str(path.relative_to(source)): sha256(path) for path in sorted(source_files)
        },
        "output_sha256": {name: sha256(output / name) for name in sorted(outputs)},
        "models": list(MODELS),
        "prompt_baselines": [row["baseline"] for row in prompt],
        "representation_controls": list(CONTROLS),
        "claim_limit": (
            "Post-hoc sensitivity of the declared 8+8 MMLU partition to prompt-only "
            "prediction and fold-local linear adjustment for measured prompt features. "
            "It does not identify a causal prompt feature or a knowledge--reasoning mechanism."
        ),
    }
    temporary = output / "report.json.tmp"
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output / "report.json")
    print(json.dumps({"status": "PASS", "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
