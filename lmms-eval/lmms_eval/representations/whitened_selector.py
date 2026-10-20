"""Post-hoc regularized-whitened localization sensitivity.

For each representation site, this module uses one label-blind PCA projection and computes
the regularized pooled-within Mahalanobis distance between the two assigned class means for
every balanced task assignment.  The maximum site is reselected under every assignment.
It is a selector sensitivity, not a causal localization analysis.
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
    _pca_projection,
    _require_cuml,
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


def regularized_whitened_distribution(
    projected: Any,
    groups: np.ndarray,
    *,
    task_order: list[str],
    assignment_ids: list[str],
    ridge_alpha: float,
    analysis_backend: str,
) -> tuple[np.ndarray, float, float]:
    """Return assignment-wise regularized pooled-within mean distances.

    The pooled within-class covariance is the label-blind total scatter minus a rank-one
    between-class term.  One inverse of the ridge-regularized total scatter is therefore
    sufficient; Sherman--Morrison evaluates all assignments without one inverse per label.
    """

    if ridge_alpha <= 0.0:
        raise ValueError("ridge_alpha must be positive")
    if analysis_backend == "cuml":
        xp, *_ = _require_cuml()
    elif analysis_backend == "sklearn":
        xp = np
    else:
        raise ValueError(f"Unsupported analysis backend: {analysis_backend}")
    values = xp.asarray(projected, dtype=xp.float64)
    if values.ndim != 2 or values.shape[0] <= 2 or values.shape[1] < 1:
        raise ValueError("Whitened localization requires a nonempty sample-by-feature matrix")
    if len(assignment_ids) < 2 or any(len(value) != len(task_order) for value in assignment_ids):
        raise ValueError("Assignments do not align with the task order")
    string_groups = groups.astype(str)
    if set(string_groups) != set(task_order):
        raise ValueError("Projected rows do not contain exactly the declared tasks")

    task_counts: list[int] = []
    task_sums: list[Any] = []
    for task in task_order:
        mask = xp.asarray(string_groups == task)
        task_values = values[mask]
        task_counts.append(int(task_values.shape[0]))
        task_sums.append(task_values.sum(axis=0))
    counts = xp.asarray(task_counts, dtype=xp.float64)
    sums = xp.stack(task_sums, axis=0)
    assignment = xp.asarray(
        [[float(bit) for bit in value] for value in assignment_ids], dtype=xp.float64
    )
    complement = 1.0 - assignment
    positive_count = assignment @ counts
    negative_count = complement @ counts
    if bool(xp.any(positive_count <= 1.0)) or bool(xp.any(negative_count <= 1.0)):
        raise ValueError("Every assignment requires at least two examples per class")
    positive_mean = (assignment @ sums) / positive_count[:, None]
    negative_mean = (complement @ sums) / negative_count[:, None]
    delta = positive_mean - negative_mean

    sample_count = int(values.shape[0])
    centered = values - values.mean(axis=0, keepdims=True)
    total_covariance_for_pooling = (centered.T @ centered) / float(sample_count - 2)
    mean_eigenvalue = float(
        (xp.trace(total_covariance_for_pooling) / values.shape[1]).item()
    )
    ridge = float(ridge_alpha) * mean_eigenvalue
    if not np.isfinite(ridge) or ridge <= EPSILON:
        raise ValueError("Label-blind covariance gives a degenerate ridge scale")
    regularized_total = total_covariance_for_pooling + ridge * xp.eye(
        values.shape[1], dtype=xp.float64
    )
    solved = xp.linalg.solve(regularized_total, delta.T).T
    quadratic = xp.sum(delta * solved, axis=1)
    between_coefficient = (
        positive_count * negative_count / float(sample_count * (sample_count - 2))
    )
    denominator = 1.0 - between_coefficient * quadratic
    if bool(xp.any(denominator <= 0.0)):
        raise ValueError("Sherman--Morrison denominator is nonpositive")
    scores = quadratic / denominator
    if analysis_backend == "cuml":
        output = xp.asnumpy(scores)
    else:
        output = np.asarray(scores)
    if not np.all(np.isfinite(output)):
        raise ValueError("Whitened score distribution is non-finite")
    return output.astype(np.float64, copy=False), ridge, mean_eigenvalue


def direct_regularized_whitened_score(
    projected: np.ndarray,
    labels: np.ndarray,
    *,
    ridge: float,
) -> float:
    """Reference implementation used to validate the vectorized identity."""

    values = np.asarray(projected, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    positive = values[labels == 1]
    negative = values[labels == 0]
    delta = positive.mean(axis=0) - negative.mean(axis=0)
    positive_scatter = (positive - positive.mean(axis=0)).T @ (
        positive - positive.mean(axis=0)
    )
    negative_scatter = (negative - negative.mean(axis=0)).T @ (
        negative - negative.mean(axis=0)
    )
    pooled = (positive_scatter + negative_scatter) / float(values.shape[0] - 2)
    return float(delta @ np.linalg.solve(pooled + ridge * np.eye(values.shape[1]), delta))


def run_whitened_selector(
    artifact_paths: list[Path],
    *,
    permutation_path: Path,
    output_dir: Path,
    ridge_alphas: tuple[float, ...] = (0.01, 0.1, 1.0),
    retained_variance: float = 0.95,
    analysis_backend: str = "cuml",
) -> dict[str, Any]:
    if not ridge_alphas or any(value <= 0.0 for value in ridge_alphas):
        raise ValueError("At least one positive ridge alpha is required")
    started = time.monotonic()
    with gzip.open(permutation_path, "rt", encoding="utf-8") as handle:
        permutation_rows = list(json.load(handle)["rows"])
    assignment_ids, task_order, trace_selected_rows, _selected_by_key = selection_plan(
        permutation_rows
    )
    keys = {_row_key(row) for row in permutation_rows}
    grouped_records, artifacts = _load_selected_records(
        artifact_paths,
        selected_keys=keys,
        task_order=task_order,
    )
    models = {str(item["model"]) for item in artifacts}
    if len(models) != 1:
        raise ValueError("Whitened selector requires one model")
    model_name = next(iter(models))
    row_by_key = {_row_key(row): index for index, row in enumerate(permutation_rows)}
    if set(row_by_key) != keys or len(row_by_key) != len(permutation_rows):
        raise ValueError("Permutation rows do not have unique representation keys")

    matrices = {
        alpha: np.empty((len(permutation_rows), len(assignment_ids)), dtype=np.float64)
        for alpha in ridge_alphas
    }
    ridge_scales = {alpha: [] for alpha in ridge_alphas}
    first_groups: np.ndarray | None = None
    first_identities: tuple[tuple[str, str], ...] | None = None
    component_counts: list[int] = []
    for key in sorted(keys):
        features, groups, _nuisance, _prompts, identities = _features_and_groups(
            grouped_records[key],
            task_order,
            analysis_backend,
            model_name=model_name,
            nuisance_lookup=None,
            lexical_prompt_lookup=None,
        )
        if first_groups is None:
            first_groups = groups
            first_identities = identities
        elif not np.array_equal(groups, first_groups) or identities != first_identities:
            raise ValueError("Representation sites do not share one sample identity order")
        projected, _explained = _pca_projection(
            features,
            retained_variance=retained_variance,
            analysis_backend=analysis_backend,
        )
        component_counts.append(int(projected.shape[1]))
        row_index = row_by_key[key]
        for alpha in ridge_alphas:
            scores, ridge, mean_eigenvalue = regularized_whitened_distribution(
                projected,
                groups,
                task_order=task_order,
                assignment_ids=assignment_ids,
                ridge_alpha=alpha,
                analysis_backend=analysis_backend,
            )
            matrices[alpha][row_index] = scores
            ridge_scales[alpha].append((ridge, mean_eigenvalue))
        del projected, features
        if analysis_backend == "cuml":
            cp, *_ = _require_cuml()
            cp.get_default_memory_pool().free_all_blocks()

    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    distribution_paths: dict[str, dict[str, str]] = {}
    trace_observed = int(trace_selected_rows[0])
    for alpha in ridge_alphas:
        matrix = matrices[alpha]
        selected = np.argmax(matrix, axis=0).astype(np.int64)
        maximum = np.max(matrix, axis=0)
        symmetry = _complement_symmetry(maximum.tolist(), assignment_ids)
        selected_symmetry = _complement_symmetry(selected.astype(float).tolist(), assignment_ids)
        exact_p = float(np.mean(maximum >= maximum[0] - EPSILON))
        observed_row = permutation_rows[int(selected[0])]
        distribution_path = output_dir / f"whitened_alpha_{alpha:g}.json.gz"
        with gzip.open(distribution_path, "wt", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "assignment_ids": assignment_ids,
                    "selected_row_indices": selected.tolist(),
                    "maximum": maximum.tolist(),
                    "row_scores": matrix.tolist(),
                },
                handle,
                separators=(",", ":"),
            )
        distribution_paths[f"{alpha:g}"] = {
            "path": str(distribution_path),
            "sha256": _sha256(distribution_path),
        }
        results.append(
            {
                "ridge_alpha": float(alpha),
                "maximum": float(maximum[0]),
                "exact_p": exact_p if symmetry["verified"] else float("nan"),
                "selected_row_index": int(selected[0]),
                "selected_layer": int(observed_row["layer"]),
                "selected_stream": str(observed_row["stream"]),
                "selected_module": str(observed_row["module"]),
                "agrees_with_trace_site": bool(int(selected[0]) == trace_observed),
                "maximum_complement_symmetry": symmetry,
                "selection_complement_symmetry": selected_symmetry,
                "ridge_min": float(min(value[0] for value in ridge_scales[alpha])),
                "ridge_max": float(max(value[0] for value in ridge_scales[alpha])),
            }
        )
    valid = all(
        row["maximum_complement_symmetry"]["verified"]
        and row["selection_complement_symmetry"]["verified"]
        for row in results
    )
    configuration = {
        "analysis_module": {"path": str(Path(__file__).resolve()), "sha256": _sha256(Path(__file__).resolve())},
        "implementation": _implementation_provenance(analysis_backend),
        "artifacts": artifacts,
        "permutation_nulls": {"path": str(permutation_path), "sha256": _sha256(permutation_path)},
        "assignment_ids_sha256": hashlib.sha256("\n".join(assignment_ids).encode()).hexdigest(),
        "retained_variance": float(retained_variance),
        "ridge_alphas": [float(value) for value in ridge_alphas],
        "ridge_policy": "alpha times mean eigenvalue of label-blind total covariance",
        "analysis_backend": analysis_backend,
    }
    report = {
        "schema_version": 1,
        "status": "PASS" if valid else "INVALID_COMPLEMENT_ASYMMETRY",
        "objective": "Sensitivity of localization to a regularized covariance-whitened selector.",
        "configuration": configuration,
        "configuration_sha256": _configuration_digest(configuration),
        "model": model_name,
        "rows": len(permutation_rows),
        "assignments": len(assignment_ids),
        "pca_components_min": min(component_counts),
        "pca_components_max": max(component_counts),
        "trace_observed_site": {
            "row_index": trace_observed,
            "layer": int(permutation_rows[trace_observed]["layer"]),
            "stream": str(permutation_rows[trace_observed]["stream"]),
            "module": str(permutation_rows[trace_observed]["module"]),
        },
        "results": results,
        "elapsed_seconds": time.monotonic() - started,
        "outputs": distribution_paths,
        "claim_limit": (
            "This post-hoc ridge grid tests selector sensitivity on one task panel. It does "
            "not establish causal localization or cross-panel transfer."
        ),
    }
    _atomic_json(output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--permutation-nulls", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ridge-alphas", default="0.01,0.1,1")
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--analysis-backend", choices=("sklearn", "cuml"), default="cuml")
    args = parser.parse_args()
    report = run_whitened_selector(
        [path.resolve() for path in args.artifacts],
        permutation_path=args.permutation_nulls.resolve(),
        output_dir=args.output_dir.resolve(),
        ridge_alphas=tuple(float(value) for value in args.ridge_alphas.split(",") if value),
        retained_variance=args.pca_variance,
        analysis_backend=args.analysis_backend,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
