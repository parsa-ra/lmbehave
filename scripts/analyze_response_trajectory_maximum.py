#!/usr/bin/env python3
"""Compute the selection-aware decode-minus-prefill Fisher maximum."""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

EPSILON = 1e-12


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def complement_symmetry(values: np.ndarray, assignment_ids: list[str]) -> dict[str, Any]:
    positions = {assignment: index for index, assignment in enumerate(assignment_ids)}
    differences = []
    for index, assignment in enumerate(assignment_ids):
        complement = "".join("1" if bit == "0" else "0" for bit in assignment)
        if complement not in positions:
            raise ValueError("Task assignments are not closed under label complement")
        differences.append(abs(float(values[index]) - float(values[positions[complement]])))
    maximum = max(differences, default=math.inf)
    return {"verified": bool(maximum <= 1e-10), "maximum_absolute_difference": maximum}


def _position_index(position: str) -> tuple[str, int]:
    for prefix, kind in (("step_", "absolute_step"), ("relative_bin_", "relative_bin")):
        if position.startswith(prefix):
            return kind, int(position.removeprefix(prefix))
    raise ValueError(f"Undefined decode position: {position}")


def analyze(rows: list[dict[str, Any]], *, common_prefix: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if common_prefix < 1:
        raise ValueError("common_prefix must be positive")
    prefill: dict[tuple[Any, ...], dict[str, Any]] = {}
    decode: list[dict[str, Any]] = []
    for row in rows:
        key = (row["model"], int(row["layer"]), row["stream"], row["module"])
        if row["phase"] == "prefill":
            if key in prefill:
                raise ValueError(f"Duplicate prefill row: {key}")
            prefill[key] = row
        elif row["phase"] == "decode":
            position = str(row["trajectory_position"])
            position_kind, step = _position_index(position)
            if step < common_prefix:
                decode.append(
                    {
                        **row,
                        "decode_step": step,
                        "position_kind": position_kind,
                        "_key": key,
                    }
                )
    if not prefill or not decode:
        raise ValueError("Both prefill and eligible decode rows are required")

    first = rows[0]
    assignment_ids = list(first["assignment_ids"])
    task_order = list(first["task_order"])
    if not assignment_ids or len(assignment_ids) != len(set(assignment_ids)):
        raise ValueError("Assignments must be non-empty and unique")
    expected_steps = set(range(common_prefix))
    position_kinds = {str(row["position_kind"]) for row in decode}
    if len(position_kinds) != 1:
        raise ValueError(f"Mixed decode position types are not supported: {sorted(position_kinds)}")
    position_kind = position_kinds.pop()
    steps_by_key: dict[tuple[Any, ...], set[int]] = {key: set() for key in prefill}
    site_rows = []
    delta_vectors = []
    decode_vectors = []
    for row in decode:
        key = row.pop("_key")
        base = prefill.get(key)
        if base is None:
            raise ValueError(f"Decode row lacks a matched prefill row: {key}")
        for candidate in (base, row):
            if candidate["assignment_ids"] != assignment_ids or candidate["task_order"] != task_order:
                raise ValueError("Assignment support differs across trajectory rows")
        steps_by_key.setdefault(key, set()).add(int(row["decode_step"]))
        base_fdr = np.asarray(base["fdr"], dtype=np.float64)
        decode_fdr = np.asarray(row["fdr"], dtype=np.float64)
        if base_fdr.shape != (len(assignment_ids),) or decode_fdr.shape != base_fdr.shape:
            raise ValueError("Fisher null vectors have inconsistent lengths")
        if not np.isfinite(base_fdr).all() or not np.isfinite(decode_fdr).all():
            raise ValueError("Fisher null vectors contain non-finite values")
        delta = decode_fdr - base_fdr
        delta_vectors.append(delta)
        decode_vectors.append(decode_fdr)
        site_rows.append(
            {
                "model": row["model"],
                "decode_step": int(row["decode_step"]),
                "trajectory_position": row["trajectory_position"],
                "position_kind": position_kind,
                "layer": int(row["layer"]),
                "stream": row["stream"],
                "module": row["module"],
                "prefill_fdr": float(base_fdr[0]),
                "decode_fdr": float(decode_fdr[0]),
                "decode_minus_prefill_fdr": float(delta[0]),
            }
        )
    incomplete = {key: sorted(expected_steps - steps) for key, steps in steps_by_key.items() if steps != expected_steps}
    if incomplete:
        preview = list(incomplete.items())[:3]
        raise ValueError(f"Incomplete eligible response positions: {preview}")

    delta_matrix = np.stack(delta_vectors)
    decode_matrix = np.stack(decode_vectors)
    maximum_delta = delta_matrix.max(axis=0)
    maximum_decode = decode_matrix.max(axis=0)
    selected_delta = int(np.argmax(delta_matrix[:, 0]))
    selected_decode = int(np.argmax(decode_matrix[:, 0]))

    def inference(values: np.ndarray) -> dict[str, Any]:
        observed = float(values[0])
        symmetry = complement_symmetry(values, assignment_ids)
        return {
            "observed": observed,
            "randomization_p": (
                float(np.count_nonzero(values >= observed - EPSILON) / len(values))
                if symmetry["verified"]
                else None
            ),
            "null_median": float(np.median(values[1:])),
            "null_q95": float(np.quantile(values[1:], 0.95)),
            "complement_symmetry": symmetry,
        }

    maximum_delta_inference = inference(maximum_delta)
    maximum_decode_inference = inference(maximum_decode)
    inference_valid = bool(
        maximum_delta_inference["complement_symmetry"]["verified"]
        and maximum_decode_inference["complement_symmetry"]["verified"]
    )

    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "PASS" if inference_valid else "INVALID_COMPLEMENT_ASYMMETRY",
        "objective": "Selection-aware maximum Fisher change from prefill across layer, stream, and eligible generated-token position.",
        "tasks": task_order,
        "assignments": len(assignment_ids),
        "common_decode_prefix": common_prefix,
        "position_kind": position_kind,
        "candidate_sites": len(site_rows),
        "theoretical_class_symmetric_minimum_p": 2.0 / len(assignment_ids),
        "maximum_decode_minus_prefill_fisher": {
            **maximum_delta_inference,
            "selected_site": site_rows[selected_delta],
        },
        "maximum_decode_fisher": {
            **maximum_decode_inference,
            "selected_site": site_rows[selected_decode],
        },
        "selection_recomputed_for_every_assignment": True,
        "interpretation": "A positive result identifies an unusual increase for the declared task partition under this response protocol. Neither absolute steps nor relative bins are semantic reasoning steps, and the result does not establish a causal mechanism.",
    }
    distribution = {
        "schema_version": 1,
        "assignment_ids": assignment_ids,
        "maximum_decode_minus_prefill_fisher": maximum_delta.tolist(),
        "maximum_decode_fisher": maximum_decode.tolist(),
    }
    return site_rows, {"report": report, "distribution": distribution}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--permutation-nulls", required=True, type=Path)
    parser.add_argument("--coverage-report", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    null_path = args.permutation_nulls.resolve()
    coverage_path = args.coverage_report.resolve()
    with gzip.open(null_path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
    if coverage.get("status") != "PASS":
        raise ValueError("Coverage report has not passed")
    site_rows, outputs = analyze(
        list(payload["rows"]),
        common_prefix=int(coverage["common_decode_prefix"]),
    )
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    site_path = output_dir / "observed_trajectory_sites.csv"
    with site_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(site_rows[0]))
        writer.writeheader()
        writer.writerows(site_rows)
    distribution_path = output_dir / "trajectory_maximum_null.json.gz"
    with gzip.open(distribution_path, "wt", encoding="utf-8") as handle:
        json.dump(outputs["distribution"], handle, separators=(",", ":"))
    report = outputs["report"]
    report["provenance"] = {
        "permutation_nulls_sha256": sha256_file(null_path),
        "coverage_report_sha256": sha256_file(coverage_path),
        "observed_sites_sha256": sha256_file(site_path),
        "maximum_null_sha256": sha256_file(distribution_path),
    }
    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
