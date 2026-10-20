#!/usr/bin/env python3
"""Create a sample-free table from calibrated matched-panel analysis reports."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis_root", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional sample-free destination; defaults to the analysis root.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or args.analysis_root
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    inputs: list[dict[str, str]] = []
    for report_path in sorted(args.analysis_root.glob("*/report.json")):
        model_key = report_path.parent.name
        report = json.loads(report_path.read_text(encoding="utf-8"))
        failures = report.get("failures", [])
        if failures:
            raise RuntimeError(f"{model_key} has {len(failures)} failed analysis rows")
        calibrations = report["selection_aware_randomization_by_model"]
        if len(calibrations) != 1:
            raise ValueError(f"Expected one model in {report_path}")
        calibration = next(iter(calibrations.values()))
        fisher = calibration["metrics"]["fdr"]
        maximum = report["primary_maximum_divergence"]
        layer_path = report_path.parent / "layer_separability.csv"
        with layer_path.open("r", encoding="utf-8", newline="") as handle:
            layer_rows = list(csv.DictReader(handle))
        reference = fisher.get("reference_partition", {})
        rows.append(
            {
                "model_key": model_key,
                "analysis_rows": len(layer_rows),
                "fdr_max": fisher["observed_maximum"],
                "fdr_max_layer": maximum["layer"],
                "fdr_max_stream": maximum["stream"],
                "fdr_max_randomization_p": fisher["maximum_randomization_p"],
                "fdr_null_max_median": fisher["null_maximum_median"],
                "fdr_null_max_q95": fisher["null_maximum_q95"],
                "fdr_bh_rejections": fisher["observed_bh_rejections"],
                "fdr_rejection_count_randomization_p": fisher[
                    "rejection_count_randomization_p"
                ],
                "minimum_fdr_row_q": min(float(row["fdr_q"]) for row in layer_rows),
                "reference_fdr_max": reference.get("maximum"),
                "reference_fdr_max_randomization_p": reference.get("maximum_randomization_p"),
                "reference_fdr_bh_rejections": reference.get("bh_rejections"),
                "complement_symmetry_verified": fisher["maximum_complement_symmetry"][
                    "verified"
                ],
            }
        )
        inputs.extend(
            {
                "path": str(path),
                "sha256": sha256(path),
            }
            for path in (
                report_path,
                layer_path,
                report_path.parent / "permutation_nulls.json.gz",
                report_path.parent / "analysis_manifest.json",
            )
        )
    if not rows:
        raise ValueError(f"No calibrated reports found under {args.analysis_root}")

    output_path = output_dir / "calibrated_model_summary.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary_path = output_dir / "calibrated_model_summary.json"
    summary = {
        "schema_version": 1,
        "conditions": len(rows),
        "inferential_unit": "whole benchmark task",
        "primary_statistic": "maximum Fisher trace ratio over declared layers and streams",
        "claim_limit": (
            "This is the original five-versus-five matched panel. Its class-symmetric exact "
            "p-value floor is 1/126; the expanded eight-versus-eight confirmatory panel remains required."
        ),
        "analysis_source": {
            "path": str(ROOT / "lmms-eval" / "lmms_eval" / "representations" / "analysis.py"),
            "sha256": sha256(
                ROOT / "lmms-eval" / "lmms_eval" / "representations" / "analysis.py"
            ),
            "lmms_eval_head": subprocess.check_output(
                ["git", "-C", str(ROOT / "lmms-eval"), "rev-parse", "HEAD"], text=True
            ).strip(),
            "lmms_eval_dirty": bool(
                subprocess.check_output(
                    ["git", "-C", str(ROOT / "lmms-eval"), "status", "--porcelain"],
                    text=True,
                ).strip()
            ),
        },
        "inputs": inputs,
        "output": {"path": str(output_path), "sha256": sha256(output_path)},
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "conditions": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
