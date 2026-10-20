"""Selection-aware synthetic calibration of matched representation excess.

This post-hoc sensitivity analysis appends one synthetic candidate site derived from a
predeclared real activation site.  A seeded direction is shifted according to an arbitrary
balanced task partition, the mean-shift selector is recomputed over the real and synthetic
sites for every balanced task assignment, and the selected activation probe is compared
with the stored prompt-64 probe.  It measures detection sensitivity of the matched-excess
instrument; it is not evidence for a model mechanism or the knowledge/reasoning labels.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from lmms_eval.representations.analysis import (
    EPSILON,
    _complement_symmetry,
    _fisher_from_projection,
    _fisher_permutation_distribution,
    _pca_projection,
)
from lmms_eval.representations.probe_calibration import (
    _calibration_assignment,
    _injected_features,
    fixed_site_probe_distribution,
)
from lmms_eval.representations.selected_site_probe import (
    _configuration_digest,
    _features_and_groups,
    _implementation_provenance,
    _load_selected_records,
    _row_key,
    _sha256,
    selection_plan,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_gzip_json(path: Path) -> dict[str, Any]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _assignments_from_ids(
    assignment_ids: list[str], task_order: list[str]
) -> list[dict[str, int]]:
    if any(len(value) != len(task_order) for value in assignment_ids):
        raise ValueError("Assignment identifiers do not match the task order")
    return [
        {task: int(bit) for task, bit in zip(task_order, assignment, strict=True)}
        for assignment in assignment_ids[1:]
    ]


def _augmented_selected_scores(
    *,
    real_fisher_matrix: np.ndarray,
    real_selected_macro_f1: np.ndarray,
    synthetic_fisher: np.ndarray,
    synthetic_macro_f1: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reselect over real sites plus one synthetic site for every assignment."""

    if real_fisher_matrix.ndim != 2:
        raise ValueError("Real Fisher scores must be a row-by-assignment matrix")
    assignments = real_fisher_matrix.shape[1]
    for name, values in (
        ("real selected F1", real_selected_macro_f1),
        ("synthetic Fisher", synthetic_fisher),
        ("synthetic F1", synthetic_macro_f1),
    ):
        if np.asarray(values).shape != (assignments,):
            raise ValueError(f"{name} does not align to the assignment support")
    if not all(
        np.all(np.isfinite(values))
        for values in (
            real_fisher_matrix,
            real_selected_macro_f1,
            synthetic_fisher,
            synthetic_macro_f1,
        )
    ):
        raise ValueError("Augmented selector inputs must be finite")
    real_maximum = np.max(real_fisher_matrix, axis=0)
    synthetic_selected = synthetic_fisher > real_maximum
    # Exact ties retain the pre-existing real site.  This makes the zero-shift
    # duplicate a no-op and avoids changing results through tie ordering alone.
    selected_macro_f1 = np.where(
        synthetic_selected,
        synthetic_macro_f1,
        real_selected_macro_f1,
    )
    augmented_maximum = np.maximum(real_maximum, synthetic_fisher)
    return selected_macro_f1, augmented_maximum, synthetic_selected


def run_matched_excess_calibration(
    artifact_paths: list[Path],
    *,
    permutation_path: Path,
    matched_report_path: Path,
    matched_distribution_path: Path,
    output_dir: Path,
    direction_seed: int,
    standardized_mean_difference: float,
    requested_splits: int = 5,
    components: int = 64,
    retained_variance: float = 0.95,
    threshold: float = 0.01,
    seed: int = 0,
    analysis_backend: str = "cuml",
) -> dict[str, Any]:
    if standardized_mean_difference < 0.0:
        raise ValueError("standardized_mean_difference must be nonnegative")
    if components < 1 or requested_splits < 2:
        raise ValueError("components and requested_splits must be positive")
    if not 0.0 < retained_variance <= 1.0:
        raise ValueError("retained_variance must lie in (0, 1]")
    started = time.monotonic()
    with gzip.open(permutation_path, "rt", encoding="utf-8") as handle:
        permutation_rows = list(json.load(handle)["rows"])
    assignment_ids, task_order, selected_rows, _selected_by_key = selection_plan(
        permutation_rows
    )
    real_fisher_matrix = np.asarray(
        [row["fdr"] for row in permutation_rows], dtype=np.float64
    )
    calibration_assignment = _calibration_assignment(task_order)
    try:
        calibration_index = assignment_ids.index(calibration_assignment)
    except ValueError as error:
        raise ValueError("Calibration partition is absent from the null support") from error

    matched_report = json.loads(matched_report_path.read_text(encoding="utf-8"))
    if matched_report.get("status") != "PASS":
        raise ValueError("Matched-excess source report is not passing")
    if _sha256(matched_distribution_path) != matched_report["outputs"]["distribution_sha256"]:
        raise ValueError("Matched-excess source distribution failed its hash gate")
    matched = _load_gzip_json(matched_distribution_path)
    if matched["assignment_ids"] != assignment_ids:
        raise ValueError("Matched-excess and Fisher assignment orders differ")
    if matched["selected_fisher_row_indices"] != selected_rows.tolist():
        raise ValueError("Matched-excess source uses a different real-site selector")
    if int(matched_report["configuration"].get("components", -1)) != components:
        raise ValueError("Matched-excess prompt arm has an unexpected width")
    real_selected_f1 = np.asarray(matched["activation_macro_f1"], dtype=np.float64)
    prompt_f1 = np.asarray(matched["prompt_macro_f1"], dtype=np.float64)

    base_row_index = int(selected_rows[0])
    base_row = permutation_rows[base_row_index]
    base_key = _row_key(base_row)
    grouped_records, artifacts = _load_selected_records(
        artifact_paths,
        selected_keys={base_key},
        task_order=task_order,
    )
    models = {str(item["model"]) for item in artifacts}
    if len(models) != 1:
        raise ValueError("Calibration requires one model")
    model_name = next(iter(models))
    features, groups, _nuisance, _prompts, _identities = _features_and_groups(
        grouped_records[base_key],
        task_order,
        analysis_backend,
        model_name=model_name,
        nuisance_lookup=None,
        lexical_prompt_lookup=None,
    )
    task_index = {task: index for index, task in enumerate(task_order)}
    group_task_indices = np.asarray(
        [task_index[str(group)] for group in groups], dtype=np.int64
    )
    calibration_labels = np.asarray(
        [int(bit) for bit in calibration_assignment], dtype=np.int64
    )[group_task_indices]
    observed_labels = np.asarray(
        [int(bit) for bit in assignment_ids[0]], dtype=np.int64
    )[group_task_indices]
    injected, projection_scale = _injected_features(
        features,
        calibration_labels,
        direction_seed=direction_seed,
        standardized_half_shift=standardized_mean_difference / 2.0,
        analysis_backend=analysis_backend,
    )
    projected, explained_variance = _pca_projection(
        injected,
        retained_variance=retained_variance,
        analysis_backend=analysis_backend,
    )
    observed_trace = _fisher_from_projection(
        projected,
        observed_labels,
        analysis_backend=analysis_backend,
        explained_variance=explained_variance,
    )
    fisher_distribution = _fisher_permutation_distribution(
        projected,
        observed_labels,
        groups,
        assignments=_assignments_from_ids(assignment_ids, task_order),
        analysis_backend=analysis_backend,
        explained_variance=explained_variance,
        observed_fdr=float(observed_trace["fdr"]),
    )
    if fisher_distribution["permutation_design"]["assignment_ids"] != assignment_ids:
        raise ValueError("Synthetic Fisher distribution changed assignment order")
    synthetic_fisher = np.asarray(fisher_distribution["distribution"], dtype=np.float64)
    synthetic_probe = fixed_site_probe_distribution(
        injected,
        groups,
        assignment_ids=assignment_ids,
        task_order=task_order,
        requested_splits=requested_splits,
        components=components,
        seed=seed,
        analysis_backend=analysis_backend,
    )
    synthetic_f1 = np.asarray(synthetic_probe["macro_f1"], dtype=np.float64)
    selected_f1, augmented_maximum, synthetic_selected = _augmented_selected_scores(
        real_fisher_matrix=real_fisher_matrix,
        real_selected_macro_f1=real_selected_f1,
        synthetic_fisher=synthetic_fisher,
        synthetic_macro_f1=synthetic_f1,
    )
    delta = selected_f1 - prompt_f1
    observed_delta = float(delta[calibration_index])
    exact_p = float(np.mean(delta >= observed_delta - EPSILON))
    delta_symmetry = _complement_symmetry(delta.tolist(), assignment_ids)
    maximum_symmetry = _complement_symmetry(augmented_maximum.tolist(), assignment_ids)
    selection_symmetry = _complement_symmetry(
        synthetic_selected.astype(float).tolist(), assignment_ids
    )
    valid = bool(
        fisher_distribution["inference_valid"]
        and synthetic_probe["macro_f1_complement_symmetry"]["verified"]
        and delta_symmetry["verified"]
        and maximum_symmetry["verified"]
        and selection_symmetry["verified"]
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    distribution_path = output_dir / "matched_excess_calibration_distribution.json.gz"
    with gzip.open(distribution_path, "wt", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "assignment_ids": assignment_ids,
                "synthetic_fisher": synthetic_fisher.tolist(),
                "synthetic_macro_f1": synthetic_f1.tolist(),
                "augmented_selected_macro_f1": selected_f1.tolist(),
                "prompt_macro_f1": prompt_f1.tolist(),
                "macro_f1_delta": delta.tolist(),
                "synthetic_selected": synthetic_selected.astype(int).tolist(),
            },
            handle,
            separators=(",", ":"),
        )
    configuration = {
        "analysis_module": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "implementation": _implementation_provenance(analysis_backend),
        "artifacts": artifacts,
        "permutation_nulls": {"path": str(permutation_path), "sha256": _sha256(permutation_path)},
        "matched_report": {"path": str(matched_report_path), "sha256": _sha256(matched_report_path)},
        "matched_distribution": {
            "path": str(matched_distribution_path),
            "sha256": _sha256(matched_distribution_path),
        },
        "assignment_ids_sha256": hashlib.sha256(
            "\n".join(assignment_ids).encode("utf-8")
        ).hexdigest(),
        "base_site_policy": "observed real mean-shift-selected site",
        "synthetic_site_policy": "append candidate and reselect under every assignment",
        "calibration_assignment": calibration_assignment,
        "calibration_assignment_index": int(calibration_index),
        "direction_seed": int(direction_seed),
        "standardized_mean_difference": float(standardized_mean_difference),
        "components": int(components),
        "retained_variance": float(retained_variance),
        "requested_splits": int(requested_splits),
        "threshold": float(threshold),
        "seed": int(seed),
        "analysis_backend": analysis_backend,
    }
    report = {
        "schema_version": 1,
        "status": "PASS" if valid else "INVALID_COMPLEMENT_ASYMMETRY",
        "objective": "Calibrate selection-aware matched representation-minus-prompt inference.",
        "configuration": configuration,
        "configuration_sha256": _configuration_digest(configuration),
        "model": model_name,
        "base_site": {
            "row_index": base_row_index,
            "layer": int(base_row["layer"]),
            "stream": str(base_row["stream"]),
            "module": str(base_row["module"]),
        },
        "projection_scale": float(projection_scale),
        "synthetic_pca_components": int(projected.shape[1]),
        "synthetic_pca_explained_variance": float(explained_variance),
        "synthetic_selected_assignments": int(np.sum(synthetic_selected)),
        "synthetic_selected_at_calibration_assignment": bool(
            synthetic_selected[calibration_index]
        ),
        "calibration_macro_f1": float(selected_f1[calibration_index]),
        "calibration_prompt_macro_f1": float(prompt_f1[calibration_index]),
        "calibration_delta_macro_f1": observed_delta,
        "exact_p": exact_p if valid else float("nan"),
        "detected_at_threshold": bool(valid and exact_p < threshold),
        "probe_refits": int(synthetic_probe["probe_refits"]),
        "complement_symmetry": {
            "synthetic_fisher": fisher_distribution["complement_symmetry"],
            "synthetic_probe_f1": synthetic_probe["macro_f1_complement_symmetry"],
            "augmented_maximum": maximum_symmetry,
            "synthetic_selection": selection_symmetry,
            "delta": delta_symmetry,
        },
        "elapsed_seconds": time.monotonic() - started,
        "outputs": {
            "distribution": str(distribution_path),
            "distribution_sha256": _sha256(distribution_path),
        },
        "claim_limit": (
            "This post-hoc synthetic calibration is conditional on one model, base site, "
            "task panel, direction, and signal scale. It measures instrument sensitivity, "
            "not a model mechanism or a training-seed effect."
        ),
    }
    _atomic_json(output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--permutation-nulls", required=True, type=Path)
    parser.add_argument("--matched-report", required=True, type=Path)
    parser.add_argument("--matched-distribution", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--direction-seed", required=True, type=int)
    parser.add_argument("--standardized-mean-difference", required=True, type=float)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--analysis-backend", choices=("sklearn", "cuml"), default="cuml")
    args = parser.parse_args()
    report = run_matched_excess_calibration(
        [path.resolve() for path in args.artifacts],
        permutation_path=args.permutation_nulls.resolve(),
        matched_report_path=args.matched_report.resolve(),
        matched_distribution_path=args.matched_distribution.resolve(),
        output_dir=args.output_dir.resolve(),
        direction_seed=args.direction_seed,
        standardized_mean_difference=args.standardized_mean_difference,
        requested_splits=args.cv_folds,
        components=args.components,
        retained_variance=args.pca_variance,
        threshold=args.threshold,
        seed=args.seed,
        analysis_backend=args.analysis_backend,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
