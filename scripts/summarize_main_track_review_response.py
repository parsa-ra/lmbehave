#!/usr/bin/env python3
"""Hash-gated summaries for reviewer-requested main-track sensitivity analyses."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


MODELS = (
    "olmo3-7b-instruct",
    "apertus15-8b",
    "qwen3_5-9b",
    "qwen3_5-27b",
    "gemma4-31b",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def effective_null(values: np.ndarray, assignment_ids: list[str]) -> np.ndarray:
    if values.shape != (12870,) or len(assignment_ids) != 12870:
        raise ValueError("Expected 12,870 oriented balanced assignments")
    observed = assignment_ids[0]
    observed_complement = "".join("1" if bit == "0" else "0" for bit in observed)
    seen: set[str] = set()
    selected: list[float] = []
    for index, assignment in enumerate(assignment_ids):
        complement = "".join("1" if bit == "0" else "0" for bit in assignment)
        pair = min(assignment, complement)
        if pair in seen:
            continue
        seen.add(pair)
        if assignment in (observed, observed_complement):
            continue
        selected.append(float(values[index]))
    result = np.asarray(selected, dtype=np.float64)
    if result.shape != (6434,):
        raise ValueError(f"Expected 6,434 non-observed complement pairs, found {result.size}")
    return result


def distribution_fields(values: np.ndarray, assignment_ids: list[str]) -> dict[str, float | int]:
    if not np.all(np.isfinite(values)):
        raise ValueError("Non-finite score distribution")
    null = effective_null(values, assignment_ids)
    return {
        "observed": float(values[0]),
        "exact_oriented_p": float(np.mean(values >= values[0] - 1e-12)),
        "effective_null_pairs": int(null.size),
        "null_mean": float(np.mean(null)),
        "null_sd": float(np.std(null, ddof=1)),
        "null_q025": float(np.quantile(null, 0.025)),
        "null_q05": float(np.quantile(null, 0.05)),
        "null_median": float(np.median(null)),
        "null_q95": float(np.quantile(null, 0.95)),
        "null_q975": float(np.quantile(null, 0.975)),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty summary: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def holm_adjusted(p_values: list[float]) -> list[float]:
    """Return monotone Holm-adjusted values in the original order."""

    if not p_values or any(not 0.0 <= value <= 1.0 for value in p_values):
        raise ValueError("Holm adjustment requires finite probabilities")
    order = sorted(range(len(p_values)), key=lambda index: p_values[index])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(p_values) - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return a two-sided Wilson binomial interval."""

    if trials < 1 or successes < 0 or successes > trials:
        raise ValueError("Wilson interval requires 0 <= successes <= trials")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    center = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return center - half_width, center + half_width


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-probe-root", required=True, type=Path)
    parser.add_argument("--matched-root", required=True, type=Path)
    parser.add_argument("--review-artifact-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    selected_root = args.selected_probe_root.resolve()
    matched_root = args.matched_root.resolve()
    review_artifact_root = args.review_artifact_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = Path(__file__).resolve().parents[1]
    manifest_path = (
        workspace
        / "experiments"
        / "representation_dynamics_main_track_review_response_v1.json"
    )
    manifest = load_json(manifest_path)
    if manifest.get("status") != "reviewer_requested_post_hoc_sensitivity":
        raise ValueError("Unexpected review-response manifest status")

    sources: list[dict[str, str]] = [
        {"path": str(manifest_path), "sha256": sha256(manifest_path)}
    ]
    pca95_rows: list[dict[str, Any]] = []
    null_rows: list[dict[str, Any]] = []
    raw_probe_p: dict[str, float] = {}
    matched_delta_p: dict[str, float] = {}
    for model in MODELS:
        selected_report_path = selected_root / model / "report.json"
        selected_distribution_path = (
            selected_root / model / "selected_site_probe_distribution.json.gz"
        )
        matched_report_path = matched_root / model / "report.json"
        matched_distribution_path = matched_root / model / "matched_excess_distribution.json.gz"
        selected_report = load_json(selected_report_path)
        matched_report = load_json(matched_report_path)
        if selected_report.get("status") != "PASS" or matched_report.get("status") != "PASS":
            raise ValueError(f"Non-passing source report for {model}")
        selected_config = selected_report["configuration"]
        matched_config = matched_report["configuration"]
        if selected_config.get("analysis_backend") != "cuml":
            raise ValueError(f"Selected probe is not cuML for {model}")
        if float(selected_config.get("pca_variance", -1)) != 0.95:
            raise ValueError(f"Selected probe is not fold-local PCA-95 for {model}")
        if int(selected_config.get("requested_splits", -1)) != 5:
            raise ValueError(f"Selected probe does not use five task folds for {model}")
        if int(selected_config.get("seed", -1)) != 0:
            raise ValueError(f"Selected probe seed changed for {model}")
        if int(matched_config.get("components", -1)) != 64:
            raise ValueError(f"Matched prompt arm is not 64-dimensional for {model}")
        if sha256(selected_distribution_path) != selected_report["outputs"]["distribution_sha256"]:
            raise ValueError(f"Selected-probe distribution hash mismatch for {model}")
        if sha256(matched_distribution_path) != matched_report["outputs"]["distribution_sha256"]:
            raise ValueError(f"Matched distribution hash mismatch for {model}")

        selected = load_gzip_json(selected_distribution_path)
        matched = load_gzip_json(matched_distribution_path)
        if selected["assignment_ids"] != matched["assignment_ids"]:
            raise ValueError(f"Assignment order differs between source distributions for {model}")
        if selected["selected_fisher_row_indices"] != matched["selected_fisher_row_indices"]:
            raise ValueError(f"Selected rows differ between source distributions for {model}")
        assignment_ids = [str(value) for value in selected["assignment_ids"]]
        activation = np.asarray(selected["macro_f1"], dtype=np.float64)
        prompt64 = np.asarray(matched["prompt_macro_f1"], dtype=np.float64)
        delta95 = activation - prompt64
        delta64 = np.asarray(matched["macro_f1_delta"], dtype=np.float64)
        pca95_rows.append(
            {
                "model": model,
                "representation_reduction": "fold-local 95%-variance PCA",
                "prompt_reduction": "fold-local character TF-IDF SVD-64",
                **distribution_fields(delta95, assignment_ids),
                "matched64_observed": float(delta64[0]),
                "matched64_exact_p": float(np.mean(delta64 >= delta64[0] - 1e-12)),
            }
        )
        null_rows.append(
            {
                "model": model,
                **distribution_fields(delta64, assignment_ids),
            }
        )
        raw_probe_p[model] = float(selected_report["macro_f1"]["randomization_p"])
        matched_delta_p[model] = float(
            matched_report["macro_f1"]["representation_minus_prompt"]["randomization_p"]
        )
        for path in (
            selected_report_path,
            selected_distribution_path,
            matched_report_path,
            matched_distribution_path,
        ):
            sources.append({"path": str(path), "sha256": sha256(path)})

    split_report_path = (
        review_artifact_root
        / "inference-unit-contrast"
        / "olmo3-7b-instruct"
        / "report.json"
    )
    split_report = load_json(split_report_path)
    if split_report.get("status") != "PASS":
        raise ValueError("Inference-unit contrast report is not passing")
    split_distribution_path = Path(split_report["outputs"]["distribution"])
    if sha256(split_distribution_path) != split_report["outputs"]["distribution_sha256"]:
        raise ValueError("Inference-unit contrast distribution failed its hash gate")
    split_distribution = load_gzip_json(split_distribution_path)
    split_f1 = np.asarray(split_distribution["macro_f1"], dtype=np.float64)
    split_accuracy = np.asarray(split_distribution["accuracy"], dtype=np.float64)
    if split_f1.shape != (10000,) or split_accuracy.shape != (10000,):
        raise ValueError("Inference-unit contrast does not contain one observed and 9,999 null scores")
    recomputed_f1_p = float(np.mean(split_f1 >= split_f1[0] - 1e-12))
    recomputed_accuracy_p = float(np.mean(split_accuracy >= split_accuracy[0] - 1e-12))
    naive = split_report["example_level_naive"]
    valid = split_report["task_level_valid"]
    if not np.isclose(recomputed_f1_p, naive["macro_f1_p"], atol=1e-12):
        raise ValueError("Inference-unit macro-F1 p-value did not reproduce")
    if not np.isclose(recomputed_accuracy_p, naive["accuracy_p"], atol=1e-12):
        raise ValueError("Inference-unit accuracy p-value did not reproduce")
    split_rows = [
        {
            "model": "olmo3-7b-instruct",
            "partition": "alphabetical 8+8",
            "layer": split_report["site"]["layer"],
            "stream": split_report["site"]["stream"],
            "example_level_macro_f1": naive["macro_f1"],
            "example_level_sample_permutation_p": naive["macro_f1_p"],
            "task_heldout_macro_f1": valid["macro_f1"],
            "task_label_exact_p": valid["exact_p"],
            "sample_permutations": naive["sample_permutations"],
            "balanced_task_assignments": valid["balanced_task_assignments"],
        }
    ]
    sources.extend(
        (
            {"path": str(split_report_path), "sha256": sha256(split_report_path)},
            {"path": str(split_distribution_path), "sha256": sha256(split_distribution_path)},
        )
    )

    width_rows: list[dict[str, Any]] = []
    width_reductions = (("fixed16", "fixed:16"), ("fixed256", "fixed:256"), ("raw", "raw"))
    width_reports = [
        review_artifact_root / "width-sensitivity" / slug / model / "report.json"
        for slug, _reduction in width_reductions
        for model in MODELS
    ]
    width_complete = all(path.is_file() for path in width_reports)
    if width_complete:
        for slug, expected_reduction in width_reductions:
            expected_mode, _, expected_components = expected_reduction.partition(":")
            for model in MODELS:
                report_path = review_artifact_root / "width-sensitivity" / slug / model / "report.json"
                report = load_json(report_path)
                if report.get("status") != "PASS" or int(report.get("assignments", -1)) != 12870:
                    raise ValueError(f"Non-passing width sensitivity for {slug}/{model}")
                reduction = report["configuration"]["activation_reduction"]
                expected_value = int(expected_components) if expected_components else None
                if reduction != {"mode": expected_mode, "components": expected_value}:
                    raise ValueError(f"Unexpected width reduction for {slug}/{model}")
                distribution_path = Path(report["outputs"]["distribution"])
                if sha256(distribution_path) != report["outputs"]["distribution_sha256"]:
                    raise ValueError(f"Width distribution hash mismatch for {slug}/{model}")
                distribution = load_gzip_json(distribution_path)
                assignment_ids = [str(value) for value in distribution["assignment_ids"]]
                delta = np.asarray(distribution["macro_f1_delta"], dtype=np.float64)
                width_rows.append(
                    {
                        "model": model,
                        "activation_reduction": expected_reduction,
                        "activation_dimensions": ";".join(
                            str(value) for value in report["activation_dimensions"]
                        ),
                        **distribution_fields(delta, assignment_ids),
                    }
                )
                sources.extend(
                    (
                        {"path": str(report_path), "sha256": sha256(report_path)},
                        {"path": str(distribution_path), "sha256": sha256(distribution_path)},
                    )
                )

    calibration_rows: list[dict[str, Any]] = []
    calibration_reports = [
        review_artifact_root
        / "matched-delta-calibration"
        / "olmo3-7b-instruct"
        / f"difference-{difference:g}"
        / f"seed-{seed}"
        / "report.json"
        for difference in (0.5, 1.0, 2.0, 4.0)
        for seed in range(10)
    ]
    calibration_complete = all(path.is_file() for path in calibration_reports)
    if calibration_complete:
        for difference in (0.5, 1.0, 2.0, 4.0):
            condition_reports: list[dict[str, Any]] = []
            for seed in range(10):
                report_path = (
                    review_artifact_root
                    / "matched-delta-calibration"
                    / "olmo3-7b-instruct"
                    / f"difference-{difference:g}"
                    / f"seed-{seed}"
                    / "report.json"
                )
                report = load_json(report_path)
                if report.get("status") != "PASS":
                    raise ValueError(f"Non-passing matched calibration for d={difference:g}, seed={seed}")
                config = report["configuration"]
                if not np.isclose(float(config["standardized_mean_difference"]), difference):
                    raise ValueError("Matched calibration mean difference changed")
                if int(config["direction_seed"]) != seed:
                    raise ValueError("Matched calibration direction seed changed")
                distribution_path = Path(report["outputs"]["distribution"])
                if sha256(distribution_path) != report["outputs"]["distribution_sha256"]:
                    raise ValueError("Matched calibration distribution hash mismatch")
                distribution = load_gzip_json(distribution_path)
                delta = np.asarray(distribution["macro_f1_delta"], dtype=np.float64)
                index = int(config["calibration_assignment_index"])
                recomputed_p = float(np.mean(delta >= delta[index] - 1e-12))
                if not np.isclose(recomputed_p, float(report["exact_p"]), atol=1e-12):
                    raise ValueError("Matched calibration exact p-value did not reproduce")
                condition_reports.append(report)
                sources.extend(
                    (
                        {"path": str(report_path), "sha256": sha256(report_path)},
                        {"path": str(distribution_path), "sha256": sha256(distribution_path)},
                    )
                )
            detections = sum(bool(report["detected_at_threshold"]) for report in condition_reports)
            lower, upper = wilson_interval(detections, len(condition_reports))
            calibration_rows.append(
                {
                    "standardized_mean_difference": difference,
                    "directions": len(condition_reports),
                    "detections_p_lt_0_01": detections,
                    "detection_fraction": detections / len(condition_reports),
                    "wilson_95_lower": lower,
                    "wilson_95_upper": upper,
                    "delta_mean": float(
                        np.mean([report["calibration_delta_macro_f1"] for report in condition_reports])
                    ),
                    "delta_min": float(
                        np.min([report["calibration_delta_macro_f1"] for report in condition_reports])
                    ),
                    "delta_max": float(
                        np.max([report["calibration_delta_macro_f1"] for report in condition_reports])
                    ),
                    "synthetic_selected_at_target": sum(
                        bool(report["synthetic_selected_at_calibration_assignment"])
                        for report in condition_reports
                    ),
                }
            )

    lexical_dir = workspace / "results" / "representation_dynamics_main_track_lexical_control_v1"
    lexical_report_path = lexical_dir / "report.json"
    lexical_csv_path = lexical_dir / "lexical_adjusted_probe_summary.csv"
    lexical_report = load_json(lexical_report_path)
    if lexical_report.get("status") != "PASS":
        raise ValueError("Lexical-control summary is not passing")
    if sha256(lexical_csv_path) != lexical_report["output_sha256"][lexical_csv_path.name]:
        raise ValueError("Lexical-control CSV failed its hash gate")
    with lexical_csv_path.open(encoding="utf-8", newline="") as handle:
        lexical_rows = list(csv.DictReader(handle))
    lexical64_p = {
        str(row["model"]): float(row["macro_f1_p"])
        for row in lexical_rows
        if int(row["components"]) == 64
    }
    if set(lexical64_p) != set(MODELS):
        raise ValueError("Lexical-64 summary does not contain exactly the five models")
    multiplicity_rows: list[dict[str, Any]] = []
    condition_values = {
        "raw_selected_site_probe": [raw_probe_p[model] for model in MODELS],
        "matched_representation_minus_prompt": [matched_delta_p[model] for model in MODELS],
        "lexical64_adjusted_probe": [lexical64_p[model] for model in MODELS],
    }
    for condition, p_values in condition_values.items():
        adjusted = holm_adjusted(p_values)
        for model, raw_p, holm_p in zip(MODELS, p_values, adjusted, strict=True):
            multiplicity_rows.append(
                {
                    "condition": condition,
                    "model": model,
                    "raw_p": raw_p,
                    "holm_p_across_five_models": holm_p,
                    "reject_0_01": holm_p < 0.01,
                    "reject_0_05": holm_p < 0.05,
                }
            )
    sources.extend(
        (
            {"path": str(lexical_report_path), "sha256": sha256(lexical_report_path)},
            {"path": str(lexical_csv_path), "sha256": sha256(lexical_csv_path)},
        )
    )

    pca95_path = output_dir / "pca95_representation_vs_prompt64.csv"
    null_path = output_dir / "matched_delta_effective_null_summary.csv"
    split_path = output_dir / "inference_unit_contrast.csv"
    multiplicity_path = output_dir / "across_model_multiplicity.csv"
    write_csv(pca95_path, pca95_rows)
    write_csv(null_path, null_rows)
    write_csv(split_path, split_rows)
    write_csv(multiplicity_path, multiplicity_rows)
    width_path = output_dir / "activation_width_sensitivity.csv"
    calibration_path = output_dir / "matched_delta_reselection_calibration.csv"
    if width_complete:
        write_csv(width_path, width_rows)
    elif width_path.exists():
        width_path.unlink()
    if calibration_complete:
        write_csv(calibration_path, calibration_rows)
    elif calibration_path.exists():
        calibration_path.unlink()
    sources.append({"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())})
    output_paths = (pca95_path, null_path, split_path, multiplicity_path)
    optional_output_paths = tuple(
        path
        for complete, path in (
            (width_complete, width_path),
            (calibration_complete, calibration_path),
        )
        if complete
    )
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "experiment_id": manifest["experiment_id"],
        "models": len(MODELS),
        "analyses": {
            "pca95_representation_vs_prompt64": {
                "status": "PASS",
                "selection_recomputed_for_every_assignment": True,
                "assignments": 12870,
                "effective_null_pairs": 6434,
                "claim_limit": (
                    "The unequal-width comparison tests sensitivity to fixed activation width 64; "
                    "it is not a capacity-matched comparison."
                ),
            },
            "matched_delta_null_description": {
                "status": "PASS",
                "effective_null_pairs": 6434,
                "claim_limit": (
                    "The reference distribution is conditional on these 16 tasks and does not "
                    "estimate a population distribution over benchmark suites."
                ),
            },
            "inference_unit_contrast": {
                "status": "PASS",
                "sample_permutations": 9999,
                "balanced_task_assignments": 12870,
                "claim_limit": (
                    "The example-level arm is deliberately invalid for task-category inference; "
                    "the result demonstrates task-identity leakage rather than reproducing another study."
                ),
            },
            "across_model_multiplicity": {
                "status": "PASS",
                "method": "Holm correction across five model-level hypotheses within condition",
                "conditions": list(condition_values),
                "claim_limit": (
                    "This post-hoc family-level sensitivity does not combine distinct "
                    "scientific conditions into one omnibus claim."
                ),
            },
            "representation_width_sensitivity": {
                "status": "PASS" if width_complete else "PENDING_COMPLETE_REPORTS",
                "conditions_expected": 15,
                "claim_limit": (
                    "Activation and prompt widths are unequal; these analyses test "
                    "compression sensitivity rather than matched capacity or equivalence."
                ),
            },
            "matched_delta_reselection_calibration": {
                "status": "PASS" if calibration_complete else "PENDING_COMPLETE_REPORTS",
                "conditions_expected": 40,
                "claim_limit": (
                    "Ten injected directions per scale provide a coarse simulation "
                    "sensitivity estimate in one model, not a model or training-seed result."
                ),
            },
        },
        "sources": sources,
        "outputs": {path.name: sha256(path) for path in output_paths + optional_output_paths},
    }
    report_path = output_dir / "report.json"
    report_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "output": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
