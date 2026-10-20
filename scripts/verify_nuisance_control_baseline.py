#!/usr/bin/env python3
"""Verify that nuisance-analysis additions preserve every baseline field."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ROW_KEY = ("model", "phase", "layer", "stream", "module")
EXCLUDED_COMMON_FIELDS = {
    # The maintained analyzer replaced a schedule-sensitive cuML convenience
    # calculation with an exact batched implementation after the reference run.
    "silhouette_cosine",
}


def _rows(path: Path) -> dict[tuple[str, ...], dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    indexed = {tuple(row[column] for column in ROW_KEY): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"Duplicate row keys in {path}")
    return indexed


def _equivalent(left: str, right: str) -> tuple[bool, float | None]:
    try:
        left_value = float(left)
        right_value = float(right)
    except ValueError:
        return left == right, None
    if math.isnan(left_value) and math.isnan(right_value):
        return True, 0.0
    difference = abs(left_value - right_value)
    return math.isclose(left_value, right_value, rel_tol=1e-10, abs_tol=1e-12), difference


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("new_root", type=Path)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--panels",
        default="matched_text_mmlu_pro_control",
        help="Comma-separated valid panels; defaults to the matched panel only.",
    )
    args = parser.parse_args()
    selected_panels = {value.strip() for value in args.panels.split(",") if value.strip()}
    if not selected_panels:
        raise ValueError("At least one panel is required")

    comparisons: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for new_path in sorted(args.new_root.glob("*/*/layer_separability.csv")):
        panel_name = new_path.parents[1].name
        if panel_name not in selected_panels:
            continue
        model_key = new_path.parent.name
        references = sorted(
            (args.reference_root / panel_name / model_key).glob(
                "analysis-cuml-26.06-nonlinear-251/layer_separability.csv"
            )
        )
        if len(references) != 1:
            raise ValueError(f"Expected one reference for {panel_name}/{model_key}")
        reference_path = references[0]
        new_rows = _rows(new_path)
        reference_rows = _rows(reference_path)
        if set(new_rows) != set(reference_rows):
            raise ValueError(f"Row-key mismatch for {panel_name}/{model_key}")
        common_fields = (
            set(next(iter(new_rows.values())))
            & set(next(iter(reference_rows.values())))
            - EXCLUDED_COMMON_FIELDS
        )
        maximum_difference = 0.0
        mismatch_count = 0
        for key in sorted(new_rows):
            for field in sorted(common_fields):
                equivalent, difference = _equivalent(new_rows[key][field], reference_rows[key][field])
                if difference is not None:
                    maximum_difference = max(maximum_difference, difference)
                if not equivalent:
                    mismatch_count += 1
                    if len(failures) < 100:
                        failures.append(
                            {
                                "panel": panel_name,
                                "model": model_key,
                                "row": key,
                                "field": field,
                                "new": new_rows[key][field],
                                "reference": reference_rows[key][field],
                            }
                        )
        comparisons.append(
            {
                "panel": panel_name,
                "model": model_key,
                "rows": len(new_rows),
                "common_fields": len(common_fields),
                "mismatches": mismatch_count,
                "maximum_numeric_difference": maximum_difference,
            }
        )

    expected_conditions = 5 * len(selected_panels)
    if len(comparisons) != expected_conditions:
        raise ValueError(
            f"Expected {expected_conditions} completed conditions for {sorted(selected_panels)}, "
            f"found {len(comparisons)}"
        )
    report = {
        "schema_version": 1,
        "conditions": comparisons,
        "included_panels": sorted(selected_panels),
        "excluded_common_fields": sorted(EXCLUDED_COMMON_FIELDS),
        "exclusion_reason": "The reference used a schedule-sensitive silhouette routine; the maintained exact batched calculation is verified separately.",
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"conditions": len(comparisons), "failures": len(failures)}, indent=2))
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())
