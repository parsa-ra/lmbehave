#!/usr/bin/env python3
"""Hash-bind the 20-direction selection-aware matched-delta calibration."""

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


DIFFERENCES = (0.5, 1.0, 2.0, 4.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def wilson_interval(successes: int, trials: int) -> tuple[float, float]:
    z = 1.959963984540054
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


def difference_slug(value: float) -> str:
    return f"{value:g}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", required=True, type=Path)
    parser.add_argument("--extension-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    parent_root = args.parent_root.resolve()
    extension_root = args.extension_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    parent_manifest = workspace / "experiments" / "representation_dynamics_main_track_review_response_v1.json"
    extension_manifest = workspace / "experiments" / "representation_dynamics_main_track_calibration_precision_v1.json"
    if load_json(extension_manifest).get("status") != "post_review_precision_extension_frozen_before_extension_outputs":
        raise ValueError("Unexpected precision-extension manifest status")
    sources = [
        {"path": str(parent_manifest), "sha256": sha256(parent_manifest)},
        {"path": str(extension_manifest), "sha256": sha256(extension_manifest)},
    ]
    condition_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for difference in DIFFERENCES:
        reports: list[dict[str, Any]] = []
        for seed in range(20):
            root = parent_root if seed < 10 else extension_root
            report_path = root / f"difference-{difference_slug(difference)}" / f"seed-{seed}" / "report.json"
            report = load_json(report_path)
            if report.get("status") != "PASS":
                raise ValueError(f"Non-passing calibration report: {report_path}")
            config = report["configuration"]
            if int(config["direction_seed"]) != seed or not np.isclose(
                float(config["standardized_mean_difference"]), difference
            ):
                raise ValueError(f"Calibration identity mismatch: {report_path}")
            distribution_path = Path(report["outputs"]["distribution"])
            if sha256(distribution_path) != report["outputs"]["distribution_sha256"]:
                raise ValueError(f"Calibration distribution hash mismatch: {report_path}")
            with gzip.open(distribution_path, "rt", encoding="utf-8") as handle:
                distribution = json.load(handle)
            delta = np.asarray(distribution["macro_f1_delta"], dtype=np.float64)
            index = int(config["calibration_assignment_index"])
            exact_p = float(np.mean(delta >= delta[index] - 1e-12))
            if not np.isclose(exact_p, float(report["exact_p"]), atol=1e-12):
                raise ValueError(f"Calibration p-value did not reproduce: {report_path}")
            if not all(bool(value["verified"]) for value in report["complement_symmetry"].values()):
                raise ValueError(f"Calibration complement symmetry failed: {report_path}")
            condition_rows.append(
                {
                    "standardized_mean_difference": difference,
                    "direction_seed": seed,
                    "delta_macro_f1": report["calibration_delta_macro_f1"],
                    "exact_p": exact_p,
                    "detected_p_lt_0_01": report["detected_at_threshold"],
                    "synthetic_selected_at_target": report[
                        "synthetic_selected_at_calibration_assignment"
                    ],
                    "source_arm": "parent" if seed < 10 else "precision_extension",
                }
            )
            reports.append(report)
            sources.extend(
                (
                    {"path": str(report_path), "sha256": sha256(report_path)},
                    {"path": str(distribution_path), "sha256": sha256(distribution_path)},
                )
            )
        detections = sum(bool(report["detected_at_threshold"]) for report in reports)
        lower, upper = wilson_interval(detections, len(reports))
        summary_rows.append(
            {
                "standardized_mean_difference": difference,
                "directions": len(reports),
                "detections_p_lt_0_01": detections,
                "detection_fraction": detections / len(reports),
                "wilson_95_lower": lower,
                "wilson_95_upper": upper,
                "delta_mean": float(
                    np.mean([report["calibration_delta_macro_f1"] for report in reports])
                ),
                "delta_min": float(
                    np.min([report["calibration_delta_macro_f1"] for report in reports])
                ),
                "delta_max": float(
                    np.max([report["calibration_delta_macro_f1"] for report in reports])
                ),
                "synthetic_selected_at_target": sum(
                    bool(report["synthetic_selected_at_calibration_assignment"])
                    for report in reports
                ),
            }
        )

    outputs: dict[str, str] = {}
    for name, rows in (
        ("calibration_conditions.csv", condition_rows),
        ("calibration_summary.csv", summary_rows),
    ):
        path = output_dir / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        outputs[name] = sha256(path)
    sources.append({"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())})
    report = {
        "schema_version": 1,
        "status": "PASS",
        "experiment_id": "representation-dynamics-main-track-calibration-precision-v1",
        "directions_per_difference": 20,
        "differences": list(DIFFERENCES),
        "sources": sources,
        "outputs": outputs,
        "claim_limit": (
            "Directions are simulation replicates in one model and panel, not model, data, "
            "or training-seed replicates. This calibrates sensitivity and does not establish a mechanism."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "conditions": len(condition_rows)}, indent=2))


if __name__ == "__main__":
    main()
