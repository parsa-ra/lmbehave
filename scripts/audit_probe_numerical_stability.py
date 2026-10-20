#!/usr/bin/env python3
"""Audit whether repeated GPU probe differences change declared decisions.

This audit does not replace the strict baseline verifier.  It records floating-point
differences separately and passes only when every non-RBF field is exact (apart from the
predeclared silhouette exclusion) and every RBF q-value retains the same side of alpha.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

ROW_KEY = ("model", "phase", "layer", "stream", "module")
EXCLUDED_FIELDS = {"silhouette_cosine"}
RBF_FIELDS = {
    "rbf_probe_accuracy",
    "rbf_probe_f1",
    "rbf_probe_accuracy_p",
    "rbf_probe_f1_p",
    "rbf_probe_accuracy_q",
    "rbf_probe_f1_q",
}
DECISION_FIELDS = ("rbf_probe_accuracy_q", "rbf_probe_f1_q")


def _read(path: Path) -> dict[tuple[str, ...], dict[str, str]]:
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


def audit_condition(
    candidate_path: Path,
    reference_path: Path,
    *,
    alpha: float,
) -> dict[str, Any]:
    candidate = _read(candidate_path)
    reference = _read(reference_path)
    if set(candidate) != set(reference):
        raise ValueError(f"Row-key mismatch for {candidate_path}")

    hard_mismatches: list[dict[str, Any]] = []
    rbf_examples: list[dict[str, Any]] = []
    rbf_counts: Counter[str] = Counter()
    rbf_maximum_difference: dict[str, float] = {field: 0.0 for field in RBF_FIELDS}
    decision_mismatches: Counter[str] = Counter()
    common_fields = set(next(iter(candidate.values()))) & set(next(iter(reference.values())))
    for key in sorted(candidate):
        for field in sorted(common_fields - EXCLUDED_FIELDS):
            equivalent, difference = _equivalent(candidate[key][field], reference[key][field])
            if equivalent:
                continue
            item = {
                "row": key,
                "field": field,
                "candidate": candidate[key][field],
                "reference": reference[key][field],
            }
            if field in RBF_FIELDS:
                rbf_counts[field] += 1
                if difference is not None:
                    rbf_maximum_difference[field] = max(
                        rbf_maximum_difference[field], difference
                    )
                if len(rbf_examples) < 100:
                    rbf_examples.append(item)
            elif len(hard_mismatches) < 100:
                hard_mismatches.append(item)
        for field in DECISION_FIELDS:
            candidate_rejects = float(candidate[key][field]) < alpha
            reference_rejects = float(reference[key][field]) < alpha
            if candidate_rejects != reference_rejects:
                decision_mismatches[field] += 1

    candidate_rows = list(candidate.values())
    reference_rows = list(reference.values())
    adjusted = {}
    if "nuisance_probe_f1_q" in common_fields:
        adjusted = {
            "linear_f1_rejections": sum(
                float(row["nuisance_probe_f1_q"]) < alpha for row in candidate_rows
            ),
            "rbf_f1_rejections": sum(
                float(row["nuisance_rbf_probe_f1_q"]) < alpha for row in candidate_rows
            ),
            "minimum_linear_f1_q": min(
                float(row["nuisance_probe_f1_q"]) for row in candidate_rows
            ),
            "minimum_rbf_f1_q": min(
                float(row["nuisance_rbf_probe_f1_q"]) for row in candidate_rows
            ),
            "interpretation": "Single-run adjusted values; not a numerical replicate.",
        }
    return {
        "rows": len(candidate),
        "hard_mismatch_count": len(hard_mismatches),
        "hard_mismatch_examples": hard_mismatches,
        "rbf_difference_count": sum(rbf_counts.values()),
        "rbf_differences_by_field": dict(sorted(rbf_counts.items())),
        "rbf_maximum_absolute_difference": {
            field: value for field, value in sorted(rbf_maximum_difference.items()) if value
        },
        "rbf_difference_examples": rbf_examples,
        "decision_mismatches": dict(sorted(decision_mismatches.items())),
        "candidate_rejections": {
            field: sum(float(row[field]) < alpha for row in candidate_rows)
            for field in DECISION_FIELDS
        },
        "reference_rejections": {
            field: sum(float(row[field]) < alpha for row in reference_rows)
            for field in DECISION_FIELDS
        },
        "adjusted_candidate": adjusted,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("candidate_root", type=Path)
    parser.add_argument("reference_root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--panel", default="matched_text_mmlu_pro_control")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")

    conditions: list[dict[str, Any]] = []
    for candidate_path in sorted((args.candidate_root / args.panel).glob("*/layer_separability.csv")):
        model = candidate_path.parent.name
        references = sorted(
            (args.reference_root / args.panel / model).glob(
                "analysis-cuml-26.06-nonlinear-251/layer_separability.csv"
            )
        )
        if len(references) != 1:
            raise ValueError(f"Expected one reference for {args.panel}/{model}")
        result = audit_condition(candidate_path, references[0], alpha=args.alpha)
        result.update({"panel": args.panel, "model": model})
        conditions.append(result)
    if not conditions:
        raise ValueError("No completed candidate conditions found")

    hard_mismatches = sum(item["hard_mismatch_count"] for item in conditions)
    decision_mismatches = sum(
        sum(item["decision_mismatches"].values()) for item in conditions
    )
    status = (
        "PASS_DECISIONS_STABLE"
        if hard_mismatches == 0 and decision_mismatches == 0
        else "FAIL"
    )
    report = {
        "schema_version": 1,
        "status": status,
        "alpha": args.alpha,
        "panel": args.panel,
        "conditions": conditions,
        "hard_mismatches": hard_mismatches,
        "rbf_decision_mismatches": decision_mismatches,
        "excluded_fields": sorted(EXCLUDED_FIELDS),
        "claim_limit": (
            "Stable threshold decisions do not make cuML RBF scores bitwise reproducible. "
            "The strict exact verifier remains authoritative for exact replay."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "conditions": len(conditions)}, indent=2))
    return int(status == "FAIL")


if __name__ == "__main__":
    raise SystemExit(main())
