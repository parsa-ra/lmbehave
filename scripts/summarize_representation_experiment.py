#!/usr/bin/env python3
"""Create hash-bound, sample-free summaries for a completed representation study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Required file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise RuntimeError(f"Required file is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_float(row: dict[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise RuntimeError(f"Expected finite {key}, found {row[key]!r}")
    return value


def maximum_row(rows: Iterable[dict[str, str]], metric: str = "fdr") -> dict[str, str]:
    candidates = [(finite_float(row, metric), row) for row in rows]
    if not candidates:
        raise RuntimeError(f"Cannot select a {metric} maximum from no rows")
    return max(candidates, key=lambda item: item[0])[1]


def pearson_correlation(left: Iterable[float], right: Iterable[float]) -> float:
    x = np.asarray(list(left), dtype=np.float64)
    y = np.asarray(list(right), dtype=np.float64)
    if x.shape != y.shape or x.size < 2:
        return float("nan")
    x = x - x.mean()
    y = y - y.mean()
    denominator = math.sqrt(float(np.dot(x, x)) * float(np.dot(y, y)))
    return float(np.dot(x, y) / denominator) if denominator > 0 else float("nan")


def accuracy_value(metrics: dict[str, Any]) -> tuple[str, float]:
    preferred = ("accuracy,none", "acc,none", "exact_match,none")
    for key in preferred:
        value = metrics.get(key)
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return key, float(value)
    candidates = [
        (key, float(value))
        for key, value in metrics.items()
        if isinstance(value, (int, float))
        and math.isfinite(float(value))
        and "stderr" not in key
        and key != "alias"
    ]
    if len(candidates) != 1:
        raise RuntimeError(f"Could not identify one task accuracy metric in {sorted(metrics)}")
    return candidates[0]


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def resolve_model_selection(
    available_models: Iterable[str], requested_models: Iterable[str] | None
) -> list[str]:
    available = list(available_models)
    selected = list(requested_models) if requested_models is not None else available
    if not selected:
        raise RuntimeError("At least one model must be selected")
    if len(selected) != len(set(selected)):
        raise RuntimeError(f"Duplicate model selections are not allowed: {selected}")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise RuntimeError(f"Unknown model selections: {unknown}")
    return selected


def validate_analysis(
    run_dir: Path,
    panel: str,
    model: str,
    analysis_name: str,
    run_manifest_sha256: str,
    completed_manifest_sha256: str,
    expected_nonlinear_permutations: int = 0,
) -> tuple[dict[str, Any], list[dict[str, str]], dict[str, Any]]:
    analysis_dir = run_dir / panel / model / analysis_name
    report_path = analysis_dir / "report.json"
    csv_path = analysis_dir / "layer_separability.csv"
    manifest = load_json(analysis_dir / "analysis_manifest.json")
    report = load_json(report_path)
    rows = load_csv(csv_path)
    if manifest.get("run_manifest_sha256") != run_manifest_sha256:
        raise RuntimeError(f"Analysis does not bind the run manifest: {panel}/{model}")
    if manifest.get("completed_manifest_sha256") != completed_manifest_sha256:
        raise RuntimeError(f"Analysis does not bind the completed manifest: {panel}/{model}")
    if manifest["outputs"].get("report_json_sha256") != sha256_file(report_path):
        raise RuntimeError(f"Analysis report changed after finalization: {panel}/{model}")
    if manifest["outputs"].get("layer_separability_csv_sha256") != sha256_file(csv_path):
        raise RuntimeError(f"Analysis CSV changed after finalization: {panel}/{model}")
    if report.get("failures") or report.get("num_rows") != len(rows):
        raise RuntimeError(f"Analysis is incomplete or contains failures: {panel}/{model}")
    if (
        report.get("nonlinear_group_permutations_requested")
        != expected_nonlinear_permutations
    ):
        raise RuntimeError(f"Unexpected nonlinear permutation count: {panel}/{model}")
    if (
        manifest["settings"].get("nonlinear_task_label_permutations")
        != expected_nonlinear_permutations
    ):
        raise RuntimeError(
            f"Analysis manifest has an unexpected nonlinear permutation count: {panel}/{model}"
        )
    for repository in ("root_git", "lmms_eval_git"):
        if manifest["analysis_source"][repository]["dirty"]:
            raise RuntimeError(f"Analysis source was dirty: {panel}/{model}/{repository}")
    required = {
        ("cuml", "float64", "training_tasks_per_fold", "251")
        for _row in rows
    }
    observed = {
        (
            row["analysis_backend"],
            row["analysis_feature_dtype"],
            row["probe_pca_fit_scope"],
            row["num_group_permutations"],
        )
        for row in rows
    }
    if observed != required:
        raise RuntimeError(f"Unexpected estimator family or inference settings: {panel}/{model}: {observed}")
    for row in rows:
        if int(row["cka_pairs"]) != 0 or not math.isnan(float(row["cka"])):
            raise RuntimeError(f"Unpaired condition emitted a defined CKA value: {panel}/{model}")
        for field in ("probe_permutation_symmetric", "fdr_permutation_symmetric"):
            if field in row and str(row[field]).lower() not in {"true", "1"}:
                raise RuntimeError(
                    f"Complement-symmetry failure for {field}: {panel}/{model}/"
                    f"layer={row['layer']}/stream={row['stream']}"
                )
        if expected_nonlinear_permutations and "rbf_probe_permutation_symmetric" in row:
            if str(row["rbf_probe_permutation_symmetric"]).lower() not in {"true", "1"}:
                raise RuntimeError(
                    f"Nonlinear complement-symmetry failure: {panel}/{model}/"
                    f"layer={row['layer']}/stream={row['stream']}"
                )
        nonlinear_values = (
            float(row["rbf_probe_accuracy_p"]),
            float(row["rbf_probe_f1_p"]),
            float(row["rbf_probe_accuracy_q"]),
            float(row["rbf_probe_f1_q"]),
        )
        if expected_nonlinear_permutations:
            if not all(math.isfinite(value) for value in nonlinear_values):
                raise RuntimeError(
                    f"Nonlinear inference emitted missing values: {panel}/{model}"
                )
        elif not all(math.isnan(value) for value in nonlinear_values):
            raise RuntimeError(
                f"Descriptive nonlinear analysis emitted inferential values: {panel}/{model}"
            )
    return report, rows, manifest


ROW_KEY = ("model", "phase", "layer", "stream", "module")
NONLINEAR_INFERENCE_FIELDS = {
    "rbf_probe_accuracy_p",
    "rbf_probe_f1_p",
    "rbf_probe_accuracy_q",
    "rbf_probe_f1_q",
}


def row_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[field] for field in ROW_KEY)


def equivalent_value(left: str, right: str, tolerance: float = 1e-12) -> bool:
    if left == right:
        return True
    try:
        left_value = float(left)
        right_value = float(right)
    except ValueError:
        return False
    if math.isnan(left_value) and math.isnan(right_value):
        return True
    return abs(left_value - right_value) <= tolerance


def join_nonlinear_inference(
    baseline_rows: list[dict[str, str]],
    sensitivity_rows: list[dict[str, str]],
) -> tuple[dict[tuple[str, ...], dict[str, str]], dict[str, Any]]:
    """Validate a nonlinear sensitivity rerun and return rows keyed to the baseline."""

    baseline = {row_key(row): row for row in baseline_rows}
    sensitivity = {row_key(row): row for row in sensitivity_rows}
    if len(baseline) != len(baseline_rows) or len(sensitivity) != len(sensitivity_rows):
        raise RuntimeError("Duplicate layer/stream rows prevent a nonlinear sensitivity join")
    if set(baseline) != set(sensitivity):
        raise RuntimeError("Nonlinear sensitivity rows do not match the baseline row family")

    silhouette_differences: list[float] = []
    for key in baseline:
        left = baseline[key]
        right = sensitivity[key]
        fields = set(left) | set(right)
        if set(left) != set(right):
            raise RuntimeError(f"Nonlinear sensitivity schema differs at {key}")
        for field in fields - NONLINEAR_INFERENCE_FIELDS - {"silhouette_cosine"}:
            if not equivalent_value(left[field], right[field]):
                raise RuntimeError(f"Nonlinear sensitivity changed {field} at {key}")
        difference = abs(float(left["silhouette_cosine"]) - float(right["silhouette_cosine"]))
        if difference > 1e-12:
            silhouette_differences.append(difference)
    return sensitivity, {
        "silhouette_difference_rows": len(silhouette_differences),
        "silhouette_max_absolute_difference": max(silhouette_differences, default=0.0),
        "geometry_source": "baseline_analysis",
        "nonlinear_inference_source": "separate_exact_task_permutation_analysis",
    }


def unique_prompt_lengths(records_path: Path) -> list[tuple[str, str, int]]:
    values: dict[str, tuple[str, int]] = {}
    with records_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_id = row["sample_id"]
            candidate = (row["task_name"], int(row["prompt_length"]))
            previous = values.setdefault(sample_id, candidate)
            if previous != candidate:
                raise RuntimeError(f"Inconsistent prompt metadata for {sample_id}")
    return [(sample_id, task, length) for sample_id, (task, length) in sorted(values.items())]


def prompt_length_probe(samples: list[tuple[str, str, int]], class_by_task: dict[str, str], seed: int) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "lmms-eval"))
    from lmms_eval.representations.analysis import grouped_permutation_probe

    features = np.asarray([[length] for _sample_id, _task, length in samples], dtype=np.float64)
    groups = np.asarray([task for _sample_id, task, _length in samples], dtype=object)
    labels = np.asarray([int(class_by_task[task] == "knowledge") for task in groups], dtype=np.int64)
    result = grouped_permutation_probe(
        features,
        labels,
        groups,
        kind="linear",
        requested_splits=5,
        max_permutations=251,
        seed=seed,
        analysis_backend="sklearn",
    )
    knowledge = [length for _sample_id, task, length in samples if class_by_task[task] == "knowledge"]
    reasoning = [length for _sample_id, task, length in samples if class_by_task[task] == "reasoning"]
    return {
        "num_examples": len(samples),
        "knowledge_prompt_tokens_mean": statistics.fmean(knowledge),
        "knowledge_prompt_tokens_median": statistics.median(knowledge),
        "reasoning_prompt_tokens_mean": statistics.fmean(reasoning),
        "reasoning_prompt_tokens_median": statistics.median(reasoning),
        "knowledge_to_reasoning_mean_ratio": statistics.fmean(knowledge) / statistics.fmean(reasoning),
        "probe_accuracy": result["accuracy"],
        "probe_f1": result["f1"],
        "probe_accuracy_p": result["accuracy_p"],
        "probe_f1_p": result["f1_p"],
        "cv_folds": result["cv_folds"],
        "num_group_permutations": result["num_group_permutations"],
        "minimum_attainable_p": result["permutation_design"]["theoretical_minimum_p"],
        "analysis_backend": "sklearn",
        "feature": "scalar_prompt_token_count",
        "status": "post_hoc_nuisance_diagnostic_not_part_of_activation_estimator_family",
    }


def summarize(
    run_dir: Path,
    analysis_name: str,
    output_dir: Path,
    nonlinear_analysis_name: str | None = None,
    models: Iterable[str] | None = None,
) -> dict[str, Any]:
    run_manifest_path = run_dir / "run_manifest.json"
    completed_path = run_dir / "completed_manifest.json"
    run_manifest = load_json(run_manifest_path)
    completed = load_json(completed_path)
    run_hash = sha256_file(run_manifest_path)
    completed_hash = sha256_file(completed_path)
    if completed.get("run_manifest_sha256") != run_hash:
        raise RuntimeError("Completed manifest does not bind the current run manifest")
    manifest = run_manifest["resolved_experiment"]
    artifacts = completed.get("artifacts", [])
    selected_models = resolve_model_selection(manifest["models"], models)
    selected_model_set = set(selected_models)
    selected_artifacts = [
        artifact for artifact in artifacts if artifact.get("model_key") in selected_model_set
    ]
    grouped_artifacts: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for artifact in selected_artifacts:
        if sha256_file(Path(artifact["artifact"])) != artifact["artifact_sha256"]:
            raise RuntimeError(f"Representation artifact changed: {artifact['artifact']}")
        if sha256_file(Path(artifact["result_json"])) != artifact["result_json_sha256"]:
            raise RuntimeError(f"Result artifact changed: {artifact['result_json']}")
        grouped_artifacts[(artifact["panel"], artifact["model_key"])].append(artifact)

    model_panel_rows: list[dict[str, Any]] = []
    task_rows: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    nonlinear_rows: list[dict[str, Any]] = []
    analysis_rows: dict[tuple[str, str], list[dict[str, str]]] = {}
    backends: set[tuple[str, str, str, str]] = set()
    nonlinear_source_commits: set[tuple[str, str]] = set()
    total_analysis_rows = 0
    total_nonlinear_rows = 0
    total_silhouette_difference_rows = 0
    maximum_silhouette_difference = 0.0

    for panel, panel_spec in manifest["panels"].items():
        class_by_task = {
            item["task"]: class_name
            for class_name, key in (("knowledge", "knowledge_tasks"), ("reasoning", "reasoning_tasks"))
            for item in panel_spec[key]
        }
        for model in selected_models:
            condition_artifacts = grouped_artifacts[(panel, model)]
            if len(condition_artifacts) != len(class_by_task):
                raise RuntimeError(f"Expected {len(class_by_task)} artifacts for {panel}/{model}")
            report, rows, _analysis_manifest = validate_analysis(
                run_dir, panel, model, analysis_name, run_hash, completed_hash
            )
            nonlinear_by_key: dict[tuple[str, ...], dict[str, str]] = {}
            nonlinear_diagnostics: dict[str, Any] | None = None
            if nonlinear_analysis_name:
                _nonlinear_report, sensitivity_rows, sensitivity_manifest = validate_analysis(
                    run_dir,
                    panel,
                    model,
                    nonlinear_analysis_name,
                    run_hash,
                    completed_hash,
                    expected_nonlinear_permutations=251,
                )
                nonlinear_by_key, nonlinear_diagnostics = join_nonlinear_inference(
                    rows, sensitivity_rows
                )
                total_nonlinear_rows += len(sensitivity_rows)
                total_silhouette_difference_rows += nonlinear_diagnostics[
                    "silhouette_difference_rows"
                ]
                maximum_silhouette_difference = max(
                    maximum_silhouette_difference,
                    nonlinear_diagnostics["silhouette_max_absolute_difference"],
                )
                nonlinear_source_commits.add(
                    (
                        sensitivity_manifest["analysis_source"]["root_git"]["head"],
                        sensitivity_manifest["analysis_source"]["lmms_eval_git"]["head"],
                    )
                )
            analysis_rows[(panel, model)] = rows
            total_analysis_rows += len(rows)
            backends.add(
                (
                    report["analysis_backend"],
                    report["analysis_backend_version"],
                    report["analysis_feature_dtype"],
                    report["probe_pca_fit_scope"],
                )
            )
            task_accuracies: dict[str, list[float]] = defaultdict(list)
            all_prompt_samples: list[tuple[str, str, int]] = []
            for artifact in condition_artifacts:
                task = artifact["task"]
                result = load_json(Path(artifact["result_json"]))
                metric_name, accuracy = accuracy_value(result["results"][task])
                task_accuracies[artifact["class"]].append(accuracy)
                task_rows.append(
                    {
                        "panel": panel,
                        "model": model,
                        "task_class": artifact["class"],
                        "task": task,
                        "accuracy_metric": metric_name,
                        "accuracy": accuracy,
                        "sample_count": artifact["sample_count"],
                    }
                )
                all_prompt_samples.extend(
                    unique_prompt_lengths(Path(artifact["artifact"]).parent / "records.csv")
                )

            selected = maximum_row(rows)
            selected_nonlinear = nonlinear_by_key.get(row_key(selected))
            branch = maximum_row(row for row in rows if row["stream"] != "residual_state")
            depth_means = {
                stream: statistics.fmean(
                    finite_float(row, "fdr")
                    for row in rows
                    if row["stream"] == stream
                )
                for stream in ("attention_delta", "mlp_delta", "residual_state")
            }
            model_panel_row = {
                "panel": panel,
                "model": model,
                "knowledge_accuracy_mean": statistics.fmean(task_accuracies["knowledge"]),
                "reasoning_accuracy_mean": statistics.fmean(task_accuracies["reasoning"]),
                "overall_accuracy_mean": statistics.fmean(
                    task_accuracies["knowledge"] + task_accuracies["reasoning"]
                ),
                "analysis_rows": len(rows),
                "fdr_max": finite_float(selected, "fdr"),
                "fdr_max_layer": int(selected["layer"]),
                "fdr_max_stream": selected["stream"],
                "fdr_max_rank_fraction": finite_float(
                    selected, "effective_rank_within_fraction"
                ),
                "fdr_max_probe_accuracy": finite_float(selected, "probe_accuracy"),
                "fdr_max_probe_f1": finite_float(selected, "probe_f1"),
                "fdr_max_probe_accuracy_p": finite_float(selected, "probe_accuracy_p"),
                "fdr_max_probe_accuracy_q": finite_float(selected, "probe_accuracy_q"),
                "fdr_max_probe_f1_p": finite_float(selected, "probe_f1_p"),
                "fdr_max_probe_f1_q": finite_float(selected, "probe_f1_q"),
                "branch_fdr_max": finite_float(branch, "fdr"),
                "branch_fdr_max_layer": int(branch["layer"]),
                "branch_fdr_max_stream": branch["stream"],
                "depth_mean_fdr_attention": depth_means["attention_delta"],
                "depth_mean_fdr_mlp": depth_means["mlp_delta"],
                "depth_mean_fdr_residual": depth_means["residual_state"],
                "probe_f1_q_below_0_05_rows": sum(
                    finite_float(row, "probe_f1_q") < 0.05 for row in rows
                ),
                "cka_defined_rows": sum(
                    math.isfinite(float(row["cka"])) for row in rows
                ),
            }
            if selected_nonlinear is not None:
                model_panel_row.update(
                    {
                        "fdr_max_rbf_probe_accuracy": finite_float(
                            selected_nonlinear, "rbf_probe_accuracy"
                        ),
                        "fdr_max_rbf_probe_f1": finite_float(selected_nonlinear, "rbf_probe_f1"),
                        "fdr_max_rbf_probe_accuracy_p": finite_float(
                            selected_nonlinear, "rbf_probe_accuracy_p"
                        ),
                        "fdr_max_rbf_probe_accuracy_q": finite_float(
                            selected_nonlinear, "rbf_probe_accuracy_q"
                        ),
                        "fdr_max_rbf_probe_f1_p": finite_float(
                            selected_nonlinear, "rbf_probe_f1_p"
                        ),
                        "fdr_max_rbf_probe_f1_q": finite_float(
                            selected_nonlinear, "rbf_probe_f1_q"
                        ),
                        "rbf_probe_accuracy_q_below_0_05_rows": sum(
                            finite_float(row, "rbf_probe_accuracy_q") < 0.05
                            for row in nonlinear_by_key.values()
                        ),
                        "rbf_probe_f1_q_below_0_05_rows": sum(
                            finite_float(row, "rbf_probe_f1_q") < 0.05
                            for row in nonlinear_by_key.values()
                        ),
                    }
                )
                best_rbf = maximum_row(nonlinear_by_key.values(), metric="rbf_probe_f1")
                nonlinear_rows.append(
                    {
                        "panel": panel,
                        "model": model,
                        "analysis_rows": len(nonlinear_by_key),
                        "task_label_permutations": 251,
                        "minimum_attainable_p": 1.0 / 126.0,
                        "rbf_probe_accuracy_q_below_0_05_rows": model_panel_row[
                            "rbf_probe_accuracy_q_below_0_05_rows"
                        ],
                        "rbf_probe_f1_q_below_0_05_rows": model_panel_row[
                            "rbf_probe_f1_q_below_0_05_rows"
                        ],
                        "minimum_rbf_probe_accuracy_q": min(
                            finite_float(row, "rbf_probe_accuracy_q")
                            for row in nonlinear_by_key.values()
                        ),
                        "minimum_rbf_probe_f1_q": min(
                            finite_float(row, "rbf_probe_f1_q")
                            for row in nonlinear_by_key.values()
                        ),
                        "maximum_rbf_probe_f1": finite_float(best_rbf, "rbf_probe_f1"),
                        "maximum_rbf_probe_f1_layer": int(best_rbf["layer"]),
                        "maximum_rbf_probe_f1_stream": best_rbf["stream"],
                        "maximum_rbf_probe_f1_p": finite_float(best_rbf, "rbf_probe_f1_p"),
                        "maximum_rbf_probe_f1_q": finite_float(best_rbf, "rbf_probe_f1_q"),
                        **nonlinear_diagnostics,
                    }
                )
            model_panel_rows.append(model_panel_row)
            prompt_rows.append(
                {
                    "panel": panel,
                    "model": model,
                    **prompt_length_probe(all_prompt_samples, class_by_task, int(manifest["analysis"]["seed"])),
                }
            )

    if len(backends) != 1:
        raise RuntimeError(f"Analysis conditions use mixed estimator families: {sorted(backends)}")

    panel_names = list(manifest["panels"])
    if len(panel_names) != 2:
        raise RuntimeError("Cross-panel summary currently requires exactly two panels")
    replication_rows: list[dict[str, Any]] = []
    for model in selected_models:
        left = analysis_rows[(panel_names[0], model)]
        right = analysis_rows[(panel_names[1], model)]
        left_max = maximum_row(left)
        right_max = maximum_row(right)
        for stream in ("attention_delta", "mlp_delta", "residual_state"):
            left_profile = {
                int(row["layer"]): finite_float(row, "fdr")
                for row in left
                if row["stream"] == stream
            }
            right_profile = {
                int(row["layer"]): finite_float(row, "fdr")
                for row in right
                if row["stream"] == stream
            }
            layers = sorted(set(left_profile) & set(right_profile))
            if set(left_profile) != set(right_profile):
                raise RuntimeError(f"Layer sets differ across panels for {model}/{stream}")
            replication_rows.append(
                {
                    "model": model,
                    "stream": stream,
                    "left_panel": panel_names[0],
                    "right_panel": panel_names[1],
                    "layers": len(layers),
                    "fdr_profile_pearson_r": pearson_correlation(
                        [left_profile[layer] for layer in layers],
                        [right_profile[layer] for layer in layers],
                    ),
                    "overall_maximum_site_matches": (
                        left_max["layer"] == right_max["layer"]
                        and left_max["stream"] == right_max["stream"]
                    ),
                    "left_fdr_max_layer": int(left_max["layer"]),
                    "left_fdr_max_stream": left_max["stream"],
                    "right_fdr_max_layer": int(right_max["layer"]),
                    "right_fdr_max_stream": right_max["stream"],
                }
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "model_panel_summary.csv": (model_panel_rows, list(model_panel_rows[0])),
        "cross_panel_replication.csv": (replication_rows, list(replication_rows[0])),
        "task_accuracy_summary.csv": (task_rows, list(task_rows[0])),
        "prompt_length_baseline.csv": (prompt_rows, list(prompt_rows[0])),
    }
    if nonlinear_rows:
        outputs["nonlinear_probe_summary.csv"] = (
            nonlinear_rows,
            list(nonlinear_rows[0]),
        )
    for name, (rows, fields) in outputs.items():
        write_csv(output_dir / name, rows, fields)
    report = {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": manifest["experiment_id"],
        "run_manifest_sha256": run_hash,
        "completed_manifest_sha256": completed_hash,
        "analysis_name": analysis_name,
        "nonlinear_analysis_name": nonlinear_analysis_name,
        "estimator_family": list(next(iter(backends))),
        "panels": {
            name: {"role": spec["role"], "claim_limit": spec["claim_limit"]}
            for name, spec in manifest["panels"].items()
        },
        "models": selected_models,
        "model_scope": "explicit_selection" if models is not None else "all_manifest_models",
        "artifact_count": len(selected_artifacts),
        "completed_manifest_artifact_count": len(artifacts),
        "representation_record_count": sum(
            int(artifact["records"]) for artifact in selected_artifacts
        ),
        "analysis_row_count": total_analysis_rows,
        "nonlinear_analysis_row_count": total_nonlinear_rows,
        "nonlinear_analysis_source_commits": [
            {"root_git": root_commit, "lmms_eval_git": lmms_commit}
            for root_commit, lmms_commit in sorted(nonlinear_source_commits)
        ],
        "nonlinear_join_policy": (
            "Use baseline rows for geometry and observed scores; join exhaustive RBF task-permutation "
            "p/q values by model, phase, layer, stream, and module. Exact inference additionally "
            "requires complement-symmetry and replay checks."
            if nonlinear_analysis_name
            else None
        ),
        "nonlinear_sensitivity_silhouette_difference_rows": total_silhouette_difference_rows,
        "nonlinear_sensitivity_silhouette_max_absolute_difference": maximum_silhouette_difference,
        "analysis_failures": 0,
        "cka_status": "undefined_for_all_rows_because_no_explicit_pairs_exist",
        "prompt_length_baseline_status": (
            "post_hoc_scalar_nuisance_diagnostic_using_sklearn_not_pooled_with_"
            "cuml_activation_rows"
        ),
        "raw_activations_prompts_responses_or_per_sample_rows_in_output": False,
        "outputs": {name: sha256_file(output_dir / name) for name in outputs},
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--analysis-name", default="analysis-cuml-26.06")
    parser.add_argument("--nonlinear-analysis-name")
    parser.add_argument(
        "--model",
        action="append",
        help="Model key to summarize; repeat for multiple models. Defaults to every manifest model.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summarize(
        args.run_dir.resolve(),
        args.analysis_name,
        args.output_dir.resolve(),
        args.nonlinear_analysis_name,
        args.model,
    )


if __name__ == "__main__":
    main()
