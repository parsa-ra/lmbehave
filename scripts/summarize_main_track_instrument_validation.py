#!/usr/bin/env python3
"""Validate and summarize main-track instrument checks without sample-level data."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
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


def load_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise ValueError(f"Non-passing report: {path}: {report.get('status')}")
    return report


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty summary: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources: list[dict[str, str]] = []
    matched_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for model in MODELS:
        matched_path = source_root / "matched-excess" / model / "report.json"
        diagnostic_path = source_root / "lexical-diagnostics" / model / "report.json"
        matched = load_report(matched_path)
        diagnostic = load_report(diagnostic_path)
        sources.extend(
            (
                {"path": str(matched_path), "sha256": sha256(matched_path)},
                {"path": str(diagnostic_path), "sha256": sha256(diagnostic_path)},
            )
        )
        if matched["assignments"] != 12870 or matched["probe_refits_per_arm"] != 6435:
            raise ValueError(f"Unexpected exact support for {model}")
        if matched["unique_task_folds"] != 229:
            raise ValueError(f"Unexpected task-fold support for {model}")
        site = matched["observed_selected_site"]
        distribution_path = Path(matched["outputs"]["distribution"])
        if sha256(distribution_path) != matched["outputs"]["distribution_sha256"]:
            raise ValueError(f"Matched distribution hash mismatch for {model}")
        sources.append({"path": str(distribution_path), "sha256": sha256(distribution_path)})
        with gzip.open(distribution_path, "rt", encoding="utf-8") as handle:
            distribution = json.load(handle)
        delta_distribution = np.asarray(distribution["macro_f1_delta"], dtype=np.float64)
        if delta_distribution.shape != (12870,) or not np.all(np.isfinite(delta_distribution)):
            raise ValueError(f"Malformed matched-delta distribution for {model}")
        delta_oriented_null = delta_distribution[1:]
        reported_delta = matched["macro_f1"]["representation_minus_prompt"]
        if not np.isclose(delta_distribution[0], reported_delta["observed"], atol=1e-12):
            raise ValueError(f"Observed matched delta disagrees with its distribution for {model}")
        if not np.isclose(np.median(delta_oriented_null), reported_delta["null_median"], atol=1e-12):
            raise ValueError(f"Matched-delta null median disagrees for {model}")
        if not np.isclose(np.quantile(delta_oriented_null, 0.95), reported_delta["null_q95"], atol=1e-12):
            raise ValueError(f"Matched-delta null q95 disagrees for {model}")
        assignment_ids = [str(value) for value in distribution["assignment_ids"]]
        observed_complement = "".join("1" if bit == "0" else "0" for bit in assignment_ids[0])
        effective_null: list[float] = []
        seen_pairs: set[str] = set()
        for index, assignment in enumerate(assignment_ids):
            complement = "".join("1" if bit == "0" else "0" for bit in assignment)
            pair = min(assignment, complement)
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            if assignment == assignment_ids[0] or assignment == observed_complement:
                continue
            effective_null.append(float(delta_distribution[index]))
        delta_effective_null = np.asarray(effective_null, dtype=np.float64)
        if delta_effective_null.shape != (6434,):
            raise ValueError(f"Expected 6,434 non-observed complement pairs for {model}")
        permutation_path = Path(matched["configuration"]["permutation_nulls"]["path"])
        with gzip.open(permutation_path, "rt", encoding="utf-8") as handle:
            task_order = list(json.load(handle)["rows"][0]["task_order"])
        if task_order != sorted(task_order) or len(task_order) != 16:
            raise ValueError(f"Negative-control task order is not alphabetical for {model}")
        negative_assignment = "1" * 8 + "0" * 8
        try:
            negative_index = distribution["assignment_ids"].index(negative_assignment)
        except ValueError as error:
            raise ValueError(f"Alphabetical negative control is absent for {model}") from error

        def negative_score(name: str) -> tuple[float, float]:
            values = [float(value) for value in distribution[name]]
            observed = values[negative_index]
            p_value = sum(value >= observed - 1e-12 for value in values) / len(values)
            return observed, p_value

        negative_representation, negative_representation_p = negative_score(
            "activation_macro_f1"
        )
        negative_prompt, negative_prompt_p = negative_score("prompt_macro_f1")
        negative_delta, negative_delta_p = negative_score("macro_f1_delta")
        matched_rows.append(
            {
                "model": model,
                "layer": site["layer"],
                "stream": site["stream"],
                "representation_macro_f1": matched["macro_f1"]["representation"]["observed"],
                "representation_p": matched["macro_f1"]["representation"]["randomization_p"],
                "prompt_macro_f1": matched["macro_f1"]["prompt"]["observed"],
                "prompt_p": matched["macro_f1"]["prompt"]["randomization_p"],
                "macro_f1_delta": matched["macro_f1"]["representation_minus_prompt"]["observed"],
                "delta_p": matched["macro_f1"]["representation_minus_prompt"]["randomization_p"],
                "delta_null_median": matched["macro_f1"]["representation_minus_prompt"]["null_median"],
                "delta_null_q95": matched["macro_f1"]["representation_minus_prompt"]["null_q95"],
                "delta_effective_null_pairs": int(delta_effective_null.size),
                "delta_effective_null_mean": float(np.mean(delta_effective_null)),
                "delta_effective_null_sd": float(np.std(delta_effective_null, ddof=1)),
                "delta_effective_null_q025": float(np.quantile(delta_effective_null, 0.025)),
                "delta_effective_null_q05": float(np.quantile(delta_effective_null, 0.05)),
                "delta_effective_null_median": float(np.median(delta_effective_null)),
                "delta_effective_null_q95": float(np.quantile(delta_effective_null, 0.95)),
                "delta_effective_null_q975": float(np.quantile(delta_effective_null, 0.975)),
                "negative_control_representation_f1": negative_representation,
                "negative_control_representation_p": negative_representation_p,
                "negative_control_prompt_f1": negative_prompt,
                "negative_control_prompt_p": negative_prompt_p,
                "negative_control_delta": negative_delta,
                "negative_control_delta_p": negative_delta_p,
            }
        )
        variance = diagnostic["variance"]
        task_identity = diagnostic["task_identity_diagnostic"]
        balance = diagnostic["permutation_fold_balance"]
        if balance["maximum_test_task_class_imbalance"] != 0:
            raise ValueError(f"Permuted test folds are imbalanced for {model}")
        if balance["maximum_train_task_class_imbalance"] != 0:
            raise ValueError(f"Permuted training folds are imbalanced for {model}")
        diagnostic_rows.append(
            {
                "model": model,
                "layer": diagnostic["observed_selected_site"]["layer"],
                "stream": diagnostic["observed_selected_site"]["stream"],
                "raw_between_task_fraction": variance["raw"]["between_task_fraction"],
                "adjusted_between_task_fraction": variance["lexical_adjusted_oof"]["between_task_fraction"],
                "retained_total_variance": variance["retained_fraction"]["total"],
                "retained_between_task_variance": variance["retained_fraction"]["between_task"],
                "retained_within_task_variance": variance["retained_fraction"]["within_task"],
                "raw_task_identity_macro_f1": task_identity["raw"]["macro_f1"],
                "adjusted_task_identity_macro_f1": task_identity["lexical_adjusted_oof"]["macro_f1"],
                "task_identity_chance": task_identity["raw"]["chance_accuracy"],
                "maximum_permuted_test_fold_imbalance": balance[
                    "maximum_test_task_class_imbalance"
                ],
            }
        )

    calibration_rows: list[dict[str, Any]] = []
    calibration_conditions: list[dict[str, Any]] = []
    lexical_cache_sources: dict[str, dict[str, str]] = {}
    workspace = Path(__file__).resolve().parents[1]
    shift_dirs = sorted(
        (
            path
            for path in (source_root / "probe-calibration" / "olmo3-7b-instruct").glob(
                "shift-*"
            )
            if any(path.glob("seed-*/report.json"))
        ),
        key=lambda path: float(path.name.removeprefix("shift-")),
    )
    if not shift_dirs:
        raise ValueError("No probe-calibration conditions found")
    observed_extension_shifts = {
        float(path.name.removeprefix("shift-"))
        for path in shift_dirs
        if float(path.name.removeprefix("shift-")) > 2.0
    }
    if observed_extension_shifts:
        extension_manifest_path = (
            workspace
            / "experiments"
            / "representation_dynamics_main_track_calibration_extension_v1.json"
        )
        extension_manifest = json.loads(
            extension_manifest_path.read_text(encoding="utf-8")
        )
        expected_extension_shifts = {
            float(value) for value in extension_manifest["standardized_half_shifts"]
        }
        if observed_extension_shifts != expected_extension_shifts:
            raise ValueError(
                "Calibration extension outputs do not match the frozen extension manifest: "
                f"{sorted(observed_extension_shifts)} versus "
                f"{sorted(expected_extension_shifts)}"
            )
    for shift_dir in shift_dirs:
        shift = shift_dir.name.removeprefix("shift-")
        seeds = (0,) if shift == "0" else tuple(range(10))
        shift_conditions: list[dict[str, Any]] = []
        for direction_seed in seeds:
            calibration_path = (
                source_root
                / "probe-calibration"
                / "olmo3-7b-instruct"
                / shift_dir.name
                / f"seed-{direction_seed}"
                / "report.json"
            )
            calibration = load_report(calibration_path)
            sources.append({"path": str(calibration_path), "sha256": sha256(calibration_path)})
            cache_metadata = calibration["configuration"].get("lexical_cache", {})
            cache_path_value = cache_metadata.get("path")
            if cache_path_value is not None:
                cache_path = Path(cache_path_value)
                if int(cache_metadata.get("folds", -1)) != 229:
                    raise ValueError(f"Unexpected shared lexical-cache support: {calibration_path}")
                cache_source = {"path": str(cache_path), "sha256": sha256(cache_path)}
                existing = lexical_cache_sources.get(str(cache_path))
                if existing is not None and existing != cache_source:
                    raise ValueError(f"Shared lexical cache changed during analysis: {cache_path}")
                lexical_cache_sources[str(cache_path)] = cache_source
            if len(calibration["conditions"]) != 1:
                raise ValueError(f"Calibration condition is not atomic: {calibration_path}")
            condition = calibration["conditions"][0]
            distribution_path = Path(condition["distribution"])
            distribution_sha256 = sha256(distribution_path)
            if distribution_sha256 != condition["distribution_sha256"]:
                raise ValueError(f"Calibration distribution hash mismatch: {distribution_path}")
            with gzip.open(distribution_path, "rt", encoding="utf-8") as handle:
                distribution = json.load(handle)
            if len(distribution["assignment_ids"]) != 12870:
                raise ValueError(f"Unexpected calibration null support: {distribution_path}")
            if condition["probe_refits"] != 6435 or condition["unique_task_folds"] != 229:
                raise ValueError(f"Unexpected calibration fit support: {calibration_path}")
            for score in ("accuracy_complement_symmetry", "macro_f1_complement_symmetry"):
                if not distribution[score]["verified"]:
                    raise ValueError(f"Calibration complement check failed: {distribution_path}")
            sources.append({"path": str(distribution_path), "sha256": distribution_sha256})
            shift_conditions.extend(calibration["conditions"])
        calibration_conditions.extend(shift_conditions)
        detections = sum(bool(row["detected_at_threshold"]) for row in shift_conditions)
        macro_f1 = [float(row["macro_f1"]) for row in shift_conditions]
        calibration_rows.append(
            {
                "standardized_half_shift": float(shift),
                "standardized_mean_difference": 2.0 * float(shift),
                "directions": len(shift_conditions),
                "detections": detections,
                "detection_fraction": detections / len(shift_conditions),
                "macro_f1_mean": sum(macro_f1) / len(macro_f1),
                "macro_f1_min": min(macro_f1),
                "macro_f1_max": max(macro_f1),
            }
        )
    expected_calibration_conditions = sum(
        1 if path.name.removeprefix("shift-") == "0" else 10 for path in shift_dirs
    )
    if len(calibration_conditions) != expected_calibration_conditions:
        raise ValueError(
            f"Expected {expected_calibration_conditions} calibration conditions, "
            f"found {len(calibration_conditions)}"
        )
    calibration_rows.sort(key=lambda row: float(row["standardized_half_shift"]))
    sources.extend(lexical_cache_sources.values())
    manifest_paths = [
        workspace / "experiments" / "representation_dynamics_main_track_instrument_validation_v1.json"
    ]
    extension_manifest_path = (
        workspace
        / "experiments"
        / "representation_dynamics_main_track_calibration_extension_v1.json"
    )
    extension_manifest_payload: dict[str, Any] | None = None
    if extension_manifest_path.is_file():
        extension_manifest_payload = json.loads(
            extension_manifest_path.read_text(encoding="utf-8")
        )
        manifest_paths.append(extension_manifest_path)
    for manifest_path in manifest_paths:
        if not manifest_path.is_file():
            raise ValueError(f"Missing experiment manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if "experiment_id" not in manifest or "status" not in manifest:
            raise ValueError(f"Malformed experiment manifest: {manifest_path}")
        sources.append({"path": str(manifest_path), "sha256": sha256(manifest_path)})
    implementation_paths = [
        workspace / "lmms-eval" / "lmms_eval" / "representations" / "matched_excess.py",
        workspace / "lmms-eval" / "lmms_eval" / "representations" / "lexical_diagnostics.py",
        workspace / "lmms-eval" / "lmms_eval" / "representations" / "probe_calibration.py",
        Path(__file__).resolve(),
    ]
    for implementation_path in implementation_paths:
        if not implementation_path.is_file():
            raise ValueError(f"Missing implementation source: {implementation_path}")
        sources.append(
            {"path": str(implementation_path), "sha256": sha256(implementation_path)}
        )
    write_csv(output_dir / "matched_excess_summary.csv", matched_rows)
    write_csv(output_dir / "lexical_diagnostic_summary.csv", diagnostic_rows)
    write_csv(output_dir / "probe_calibration_summary.csv", calibration_rows)
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "experiment_id": "representation-dynamics-main-track-instrument-validation-v1",
        "models": len(MODELS),
        "matched_assignments_per_model": 12870,
        "matched_refits_per_arm_per_model": 6435,
        "fold_balance_verified_for_every_assignment": True,
        "calibration_standardized_half_shifts": [
            float(path.name.removeprefix("shift-")) for path in shift_dirs
        ],
        "adaptive_calibration_extension_included": any(
            float(path.name.removeprefix("shift-")) > 2.0 for path in shift_dirs
        ),
        "terminated_calibration_extension": (
            extension_manifest_payload["termination"]
            if extension_manifest_payload is not None
            and extension_manifest_payload["status"] == "terminated_without_completed_reports"
            else None
        ),
        "sources": sources,
        "outputs": {
            name: sha256(output_dir / name)
            for name in (
                "matched_excess_summary.csv",
                "lexical_diagnostic_summary.csv",
                "probe_calibration_summary.csv",
            )
        },
        "claim_limit": (
            "Matched excess tests one fixed 64-feature estimator family. Variance and "
            "task-identity results diagnose the lexical sensitivity transform. Synthetic "
            "calibration is conditional on one model and fixed site. None is causal evidence."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "output": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
