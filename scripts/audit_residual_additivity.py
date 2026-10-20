#!/usr/bin/env python3
"""Check whether stored branch updates reconstruct adjacent residual states."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_additivity(payload: dict[str, Any]) -> tuple[bool, str]:
    specs = list(payload.get("layer_specs") or [])
    if not specs:
        raise ValueError("Artifact has no layer specifications")
    explicit = [spec.get("residual_change_equals_reported_branch_sum") for spec in specs]
    if all(value is not None for value in explicit):
        values = {bool(value) for value in explicit}
        if len(values) != 1:
            return False, "layer specifications differ"
        return values.pop(), "explicit artifact metadata"
    classes = {str(spec.get("layer_class", "")).lower() for spec in specs}
    if any("gemma4textdecoderlayer" in name for name in classes):
        return False, "legacy Gemma 4 artifact; source has a post-residual layer scalar"
    complete = all(
        bool(spec.get("attention_delta_exact"))
        and bool(spec.get("mlp_delta_exact"))
        and spec.get("additional_residual_branch") is None
        for spec in specs
    )
    return complete, "legacy metadata inferred from exactness and additional-branch fields"


def audit_artifact(path: Path, relative_tolerance: float) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    records = [record for record in payload["records"] if record.get("phase") == "prefill"]
    lookup = {
        (str(record["sample_id"]), int(record["layer_index"]), str(record["stream"])): record["vector"].float()
        for record in records
    }
    sample_ids = sorted({key[0] for key in lookup})
    layers = sorted({key[1] for key in lookup})
    errors: list[float] = []
    for sample_id in sample_ids:
        for layer in layers[1:]:
            required = (
                (sample_id, layer - 1, "residual_state"),
                (sample_id, layer, "residual_state"),
                (sample_id, layer, "attention_delta"),
                (sample_id, layer, "mlp_delta"),
            )
            if any(key not in lookup for key in required):
                raise ValueError(f"Incomplete layer reconstruction tuple: {required}")
            previous, current, attention, mlp = (lookup[key] for key in required)
            residual_change = current - previous
            error = torch.linalg.vector_norm(residual_change - attention - mlp)
            scale = torch.linalg.vector_norm(residual_change).clamp_min(1e-12)
            errors.append(float((error / scale).item()))
    if not errors:
        raise ValueError("No adjacent-layer reconstruction comparisons")
    expected, expectation_source = expected_additivity(payload)
    array = np.asarray(errors, dtype=np.float64)
    maximum = float(array.max())
    verified = bool(maximum <= relative_tolerance) if expected else None
    return {
        "artifact": str(path.resolve()),
        "artifact_sha256": sha256(path),
        "selected_stack": payload.get("selected_stack"),
        "comparisons": len(errors),
        "expected_additivity": expected,
        "expectation_source": expectation_source,
        "relative_error_median": float(np.median(array)),
        "relative_error_q95": float(np.quantile(array, 0.95)),
        "relative_error_maximum": maximum,
        "relative_tolerance": relative_tolerance,
        "additivity_verified": verified,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--relative-tolerance", type=float, default=0.05)
    args = parser.parse_args()
    rows = [audit_artifact(path.resolve(), args.relative_tolerance) for path in args.artifacts]
    failed = [row for row in rows if row["additivity_verified"] is False]
    report = {
        "schema_version": 1,
        "status": "FAIL" if failed else "PASS",
        "scope": (
            "Post-hoc software validation at the pooled final prompt token. The tolerance "
            "accounts for low-precision model arithmetic followed by float32 storage; it is not "
            "a scientific effect threshold. Layer 0 is excluded because the embedding input was not stored."
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "output": str(args.output), "artifacts": len(rows)}, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
