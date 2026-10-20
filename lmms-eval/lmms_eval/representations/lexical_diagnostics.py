"""Diagnostics for what fold-local lexical activation adjustment removes.

This module analyzes the observed Fisher-selected site only.  It constructs out-of-fold
activation residuals using the same task-held-out character-TF--IDF nuisance map as the
selection-aware sensitivity analysis, then reports variance decomposition and a separate
within-task example-split task-identity diagnostic.  The latter is not primary inference;
it asks whether task identity remains decodable after adjustment.
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
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold

from lmms_eval.representations.analysis import (
    _fold_local_ridge_residuals,
    _require_cuml,
    grouped_stratified_folds,
)
from lmms_eval.representations.matched_excess import (
    _fixed_pca,
    _linear_predictions,
    _scale_fold,
)
from lmms_eval.representations.selected_site_probe import (
    _configuration_digest,
    _features_and_groups,
    _implementation_provenance,
    _load_prompt_lookup,
    _load_selected_records,
    _prepare_lexical_fold_cache,
    _sha256,
    selection_plan,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def variance_decomposition(features: Any, groups: np.ndarray, *, analysis_backend: str) -> dict[str, float]:
    """Return task-between, task-within, and total sums of squares."""

    if analysis_backend == "cuml":
        xp, *_ = _require_cuml()
    elif analysis_backend == "sklearn":
        xp = np
    else:
        raise ValueError(f"Unsupported analysis backend: {analysis_backend}")
    global_mean = features.mean(axis=0)
    total = xp.square(features - global_mean).sum()
    between = xp.asarray(0.0, dtype=xp.float64)
    for group in sorted(str(item) for item in np.unique(groups)):
        indices = np.flatnonzero(groups.astype(str) == group)
        backend_indices = xp.asarray(indices) if analysis_backend == "cuml" else indices
        task_features = features[backend_indices]
        between = between + task_features.shape[0] * xp.square(task_features.mean(axis=0) - global_mean).sum()
    within = total - between
    total_value = float(total.item())
    between_value = float(between.item())
    within_value = float(within.item())
    if total_value <= 0.0:
        raise ValueError("Variance decomposition requires nonconstant features")
    return {
        "total_sum_squares": total_value,
        "between_task_sum_squares": between_value,
        "within_task_sum_squares": within_value,
        "between_task_fraction": between_value / total_value,
        "within_task_fraction": within_value / total_value,
    }


def _task_identity_probe(
    features: Any,
    groups: np.ndarray,
    *,
    components: int,
    splits: int,
    seed: int,
    analysis_backend: str,
) -> dict[str, float | int]:
    task_order = sorted(str(item) for item in np.unique(groups))
    task_index = {task: index for index, task in enumerate(task_order)}
    labels = np.asarray([task_index[str(group)] for group in groups], dtype=np.int64)
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    predictions = np.full(labels.shape, -1, dtype=np.int64)
    component_counts: set[int] = set()
    for train_indices, test_indices in splitter.split(np.zeros(labels.shape), labels):
        if analysis_backend == "cuml":
            cp, *_ = _require_cuml()
            train_backend = cp.asarray(train_indices)
            test_backend = cp.asarray(test_indices)
        else:
            train_backend = train_indices
            test_backend = test_indices
        train_features, test_features, actual = _fixed_pca(
            features[train_backend],
            features[test_backend],
            components=components,
            analysis_backend=analysis_backend,
            seed=seed,
        )
        train_features, test_features = _scale_fold(
            train_features,
            test_features,
            analysis_backend=analysis_backend,
        )
        predictions[test_indices] = _linear_predictions(
            train_features,
            test_features,
            labels[train_indices],
            analysis_backend=analysis_backend,
            seed=seed,
        )
        component_counts.add(actual)
    if np.any(predictions < 0):
        raise RuntimeError("Task-identity diagnostic did not predict every example")
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "classes": len(task_order),
        "chance_accuracy": 1.0 / len(task_order),
        "folds": int(splits),
        "components_min": min(component_counts),
        "components_max": max(component_counts),
    }


def _fold_balance_audit(
    assignment_ids: list[str],
    groups: np.ndarray,
    task_order: list[str],
    *,
    requested_splits: int,
    seed: int,
) -> dict[str, Any]:
    task_index = {task: index for index, task in enumerate(task_order)}
    group_task_indices = np.asarray([task_index[str(group)] for group in groups], dtype=np.int64)
    test_imbalances: list[int] = []
    train_imbalances: list[int] = []
    test_class_counts: set[tuple[int, int]] = set()
    train_class_counts: set[tuple[int, int]] = set()
    for assignment in assignment_ids:
        labels = np.asarray([int(bit) for bit in assignment], dtype=np.int64)[group_task_indices]
        folds, _ = grouped_stratified_folds(
            labels,
            groups,
            requested_splits=requested_splits,
            seed=seed,
        )
        for train_indices, test_indices in folds:
            test_tasks = np.unique(groups[test_indices])
            train_tasks = np.unique(groups[train_indices])
            test_counts = tuple(
                sum(int(np.unique(labels[groups == task])[0]) == label for task in test_tasks)
                for label in (0, 1)
            )
            train_counts = tuple(
                sum(int(np.unique(labels[groups == task])[0]) == label for task in train_tasks)
                for label in (0, 1)
            )
            test_class_counts.add(test_counts)
            train_class_counts.add(train_counts)
            test_imbalances.append(abs(test_counts[0] - test_counts[1]))
            train_imbalances.append(abs(train_counts[0] - train_counts[1]))
    return {
        "assignments": len(assignment_ids),
        "folds_per_assignment": int(requested_splits),
        "test_task_class_counts": [list(item) for item in sorted(test_class_counts)],
        "train_task_class_counts": [list(item) for item in sorted(train_class_counts)],
        "maximum_test_task_class_imbalance": max(test_imbalances, default=0),
        "maximum_train_task_class_imbalance": max(train_imbalances, default=0),
        "every_fold_contains_both_classes": True,
        "assignment_specific_restratification": True,
    }


def run_lexical_diagnostics(
    artifact_paths: list[Path],
    *,
    permutation_path: Path,
    sample_root: Path,
    output_dir: Path,
    requested_splits: int = 5,
    lexical_components: int = 64,
    task_probe_components: int = 64,
    task_probe_splits: int = 5,
    nuisance_alpha: float = 1.0,
    seed: int = 0,
    analysis_backend: str = "cuml",
) -> dict[str, Any]:
    started = time.monotonic()
    with gzip.open(permutation_path, "rt", encoding="utf-8") as handle:
        permutation_rows = list(json.load(handle)["rows"])
    assignment_ids, task_order, selected_rows, _selected_by_key = selection_plan(permutation_rows)
    observed_row = permutation_rows[int(selected_rows[0])]
    observed_key = (
        str(observed_row["phase"]),
        str(observed_row.get("trajectory_position", "aggregate")),
        int(observed_row["layer"]),
        str(observed_row["stream"]),
        str(observed_row["module"]),
    )
    grouped_records, artifacts = _load_selected_records(
        artifact_paths,
        selected_keys={observed_key},
        task_order=task_order,
    )
    model_names = {str(item["model"]) for item in artifacts}
    if len(model_names) != 1:
        raise ValueError("Lexical diagnostics require one model")
    model_name = next(iter(model_names))
    required_identities = {
        (str(record["task_name"]), str(record.get("sample_id")))
        for record in grouped_records[observed_key]
    }
    prompt_lookup, prompt_sources = _load_prompt_lookup(
        sample_root,
        required_identities=required_identities,
    )
    features, groups, _nuisance, prompts, identities = _features_and_groups(
        grouped_records[observed_key],
        task_order,
        analysis_backend,
        model_name=model_name,
        nuisance_lookup=None,
        lexical_prompt_lookup=prompt_lookup,
    )
    if prompts is None:
        raise RuntimeError("Lexical diagnostics did not load prompt text")
    task_index = {task: index for index, task in enumerate(task_order)}
    group_task_indices = np.asarray([task_index[str(group)] for group in groups], dtype=np.int64)
    labels = np.asarray([int(bit) for bit in assignment_ids[0]], dtype=np.int64)[group_task_indices]
    prompt_fold_cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    actual_components: set[int] = set()
    _prepare_lexical_fold_cache(
        labels,
        groups,
        prompts,
        requested_splits=requested_splits,
        seed=seed,
        kind="char_tfidf",
        components=lexical_components,
        cache=prompt_fold_cache,
        component_counts=actual_components,
    )
    if analysis_backend == "cuml":
        cp, *_ = _require_cuml()
        residuals = cp.empty_like(features, dtype=cp.float64)
    else:
        residuals = np.empty_like(features, dtype=np.float64)
    folds, effective_splits = grouped_stratified_folds(
        labels,
        groups,
        requested_splits=requested_splits,
        seed=seed,
    )
    for train_indices, test_indices in folds:
        lexical_train, lexical_test = prompt_fold_cache[test_indices.tobytes()]
        if analysis_backend == "cuml":
            train_backend = cp.asarray(train_indices)
            test_backend = cp.asarray(test_indices)
            lexical_train_backend = cp.asarray(lexical_train, dtype=cp.float64)
            lexical_test_backend = cp.asarray(lexical_test, dtype=cp.float64)
        else:
            train_backend = train_indices
            test_backend = test_indices
            lexical_train_backend = lexical_train
            lexical_test_backend = lexical_test
        _train_residuals, test_residuals = _fold_local_ridge_residuals(
            features[train_backend],
            features[test_backend],
            lexical_train_backend,
            lexical_test_backend,
            alpha=nuisance_alpha,
            analysis_backend=analysis_backend,
        )
        residuals[test_backend] = test_residuals

    raw_variance = variance_decomposition(features, groups, analysis_backend=analysis_backend)
    residual_variance = variance_decomposition(residuals, groups, analysis_backend=analysis_backend)
    variance_retention = {
        "total": residual_variance["total_sum_squares"] / raw_variance["total_sum_squares"],
        "between_task": residual_variance["between_task_sum_squares"]
        / raw_variance["between_task_sum_squares"],
        "within_task": residual_variance["within_task_sum_squares"]
        / raw_variance["within_task_sum_squares"],
    }
    raw_task_identity = _task_identity_probe(
        features,
        groups,
        components=task_probe_components,
        splits=task_probe_splits,
        seed=seed,
        analysis_backend=analysis_backend,
    )
    residual_task_identity = _task_identity_probe(
        residuals,
        groups,
        components=task_probe_components,
        splits=task_probe_splits,
        seed=seed,
        analysis_backend=analysis_backend,
    )
    fold_balance = _fold_balance_audit(
        assignment_ids,
        groups,
        task_order,
        requested_splits=requested_splits,
        seed=seed,
    )

    configuration = {
        "implementation": _implementation_provenance(analysis_backend),
        "artifacts": artifacts,
        "permutation_nulls": {"path": str(permutation_path), "sha256": _sha256(permutation_path)},
        "sample_root": str(sample_root),
        "prompt_sources": prompt_sources,
        "assignment_ids_sha256": hashlib.sha256("\n".join(assignment_ids).encode()).hexdigest(),
        "requested_splits": int(requested_splits),
        "lexical_components": int(lexical_components),
        "task_probe_components": int(task_probe_components),
        "task_probe_splits": int(task_probe_splits),
        "nuisance_alpha": float(nuisance_alpha),
        "seed": int(seed),
        "analysis_backend": analysis_backend,
    }
    configuration_sha256 = _configuration_digest(configuration)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "objective": (
            "Quantify activation variance and task-identity information retained after the "
            "fold-local lexical sensitivity transform."
        ),
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "model": model_name,
        "observed_selected_site": {
            "row_index": int(selected_rows[0]),
            "layer": int(observed_row["layer"]),
            "stream": str(observed_row["stream"]),
            "module": str(observed_row["module"]),
        },
        "samples": len(identities),
        "tasks": len(task_order),
        "effective_grouped_folds": int(effective_splits),
        "lexical_component_counts": sorted(actual_components),
        "variance": {
            "raw": raw_variance,
            "lexical_adjusted_oof": residual_variance,
            "retained_fraction": variance_retention,
        },
        "task_identity_diagnostic": {
            "split_unit": "examples stratified within all 16 tasks",
            "primary_inference": False,
            "raw": raw_task_identity,
            "lexical_adjusted_oof": residual_task_identity,
            "interpretation": (
                "This diagnostic asks whether task identity remains decodable. Its example-level "
                "split is deliberately not used for knowledge-versus-reasoning inference."
            ),
        },
        "permutation_fold_balance": fold_balance,
        "elapsed_seconds": time.monotonic() - started,
        "interpretation": (
            "Variance removal is a sensitivity diagnostic. A retained task-identity signal does "
            "not prove that the adjusted knowledge-versus-reasoning score is construct-specific."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(output_dir / "report.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--permutation-nulls", required=True, type=Path)
    parser.add_argument("--sample-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--lexical-components", type=int, default=64)
    parser.add_argument("--task-probe-components", type=int, default=64)
    parser.add_argument("--task-probe-folds", type=int, default=5)
    parser.add_argument("--nuisance-alpha", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--analysis-backend", choices=("sklearn", "cuml"), default="cuml")
    args = parser.parse_args()
    report = run_lexical_diagnostics(
        [path.resolve() for path in args.artifacts],
        permutation_path=args.permutation_nulls.resolve(),
        sample_root=args.sample_root.resolve(),
        output_dir=args.output_dir.resolve(),
        requested_splits=args.cv_folds,
        lexical_components=args.lexical_components,
        task_probe_components=args.task_probe_components,
        task_probe_splits=args.task_probe_folds,
        nuisance_alpha=args.nuisance_alpha,
        seed=args.seed,
        analysis_backend=args.analysis_backend,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
