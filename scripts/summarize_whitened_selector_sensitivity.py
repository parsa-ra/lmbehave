#!/usr/bin/env python3
"""Hash-bind sample-free summaries of the regularized-whitened selector grid."""

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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def holm_adjust(p_values: list[float]) -> list[float]:
    """Return Holm family-wise adjusted p-values without an optional dependency."""

    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted_sorted: list[float] = []
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(p_values) - rank) * p_values[index])
        adjusted_sorted.append(min(running, 1.0))
    adjusted = [0.0] * len(p_values)
    for index, value in zip(order, adjusted_sorted, strict=True):
        adjusted[index] = value
    return adjusted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace = Path(__file__).resolve().parents[1]
    manifest_path = workspace / "experiments" / "representation_dynamics_selector_sensitivity_v1.json"
    manifest = load_json(manifest_path)
    if manifest.get("status") != "post_review_sensitivity_frozen_before_outputs":
        raise ValueError("Unexpected selector manifest status")

    rows: list[dict[str, Any]] = []
    sources = [{"path": str(manifest_path), "sha256": sha256(manifest_path)}]
    for model in MODELS:
        report_path = source_root / model / "report.json"
        report = load_json(report_path)
        if report.get("status") != "PASS" or report.get("assignments") != 12870:
            raise ValueError(f"Non-passing selector report for {model}")
        if [float(value) for value in report["configuration"]["ridge_alphas"]] != [0.01, 0.1, 1.0]:
            raise ValueError(f"Unexpected ridge grid for {model}")
        for result in report["results"]:
            alpha = float(result["ridge_alpha"])
            source = report["outputs"][f"{alpha:g}"]
            distribution_path = Path(source["path"])
            if sha256(distribution_path) != source["sha256"]:
                raise ValueError(f"Selector distribution hash mismatch for {model}, alpha={alpha:g}")
            with gzip.open(distribution_path, "rt", encoding="utf-8") as handle:
                distribution = json.load(handle)
            maximum = np.asarray(distribution["maximum"], dtype=np.float64)
            if maximum.shape != (12870,) or not np.all(np.isfinite(maximum)):
                raise ValueError(f"Malformed selector maximum for {model}, alpha={alpha:g}")
            p_value = float(np.mean(maximum >= maximum[0] - 1e-12))
            if not np.isclose(p_value, float(result["exact_p"]), atol=1e-12):
                raise ValueError(f"Selector p-value did not reproduce for {model}, alpha={alpha:g}")
            rows.append(
                {
                    "model": model,
                    "ridge_alpha": alpha,
                    "trace_layer": report["trace_observed_site"]["layer"],
                    "trace_stream": report["trace_observed_site"]["stream"],
                    "whitened_layer": result["selected_layer"],
                    "whitened_stream": result["selected_stream"],
                    "agrees_with_trace_site": result["agrees_with_trace_site"],
                    "whitened_maximum": result["maximum"],
                    "exact_p": p_value,
                    "assignments": report["assignments"],
                }
            )
            sources.append({"path": str(distribution_path), "sha256": sha256(distribution_path)})
        sources.append({"path": str(report_path), "sha256": sha256(report_path)})

    adjusted = holm_adjust([float(row["exact_p"]) for row in rows])
    for row, value in zip(rows, adjusted, strict=True):
        row["holm_p_across_grid"] = value

    csv_path = output_dir / "selector_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    sources.append({"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve())})
    summary = {
        "schema_version": 1,
        "status": "PASS",
        "experiment_id": manifest["experiment_id"],
        "models": len(MODELS),
        "ridge_alphas": [0.01, 0.1, 1.0],
        "all_whitened_sites_differ_from_trace_site": all(
            str(row["agrees_with_trace_site"]).lower() == "false" for row in rows
        ),
        "all_grid_maxima_reject_after_holm_at_0.01": all(
            float(row["holm_p_across_grid"]) < 0.01 for row in rows
        ),
        "sources": sources,
        "outputs": {csv_path.name: sha256(csv_path)},
        "claim_limit": (
            "This post-hoc selector grid uses the primary panel only. Site changes show "
            "selector dependence, not causal localization or cross-panel replication."
        ),
    }
    (output_dir / "report.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
