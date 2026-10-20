from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import itertools
import json
import math
import multiprocessing
import os
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

EPSILON = 1e-12
_PROCESS_WORK_ITEMS: list[tuple[tuple[str, str, str, int, str, str], list[dict[str, Any]]]] = []
_PROCESS_ANALYSIS_KWARGS: dict[str, Any] = {}
_CUML_MODULES: tuple[Any, ...] | None = None


def _require_cuml() -> tuple[Any, ...]:
    """Import the optional GPU analysis stack only when explicitly requested."""

    global _CUML_MODULES
    if _CUML_MODULES is not None:
        return _CUML_MODULES
    try:
        import cuml
        import cupy as cp
        from cuml.decomposition import PCA as CumlPCA
        from cuml.linear_model import LogisticRegression as CumlLogisticRegression
        from cuml.metrics.cluster import silhouette_score as cuml_silhouette_score
        from cuml.preprocessing import StandardScaler as CumlStandardScaler
        from cuml.svm import SVC as CumlSVC
    except ImportError as error:
        raise RuntimeError(
            "analysis_backend='cuml' requires CuPy and RAPIDS cuML in the selected Python environment"
        ) from error
    _CUML_MODULES = (
        cp,
        cuml,
        CumlPCA,
        CumlLogisticRegression,
        cuml_silhouette_score,
        CumlStandardScaler,
        CumlSVC,
    )
    return _CUML_MODULES


def _json_value(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def _task_set(value: str) -> set[str]:
    return {item.strip() for item in value.split(",") if item.strip()}


def benjamini_hochberg(
    p_values: Iterable[float],
    *,
    family_size: int | None = None,
) -> list[float]:
    """Return Benjamini--Hochberg adjusted p-values, preserving missing rows.

    Missing or failed hypotheses remain ``NaN`` but still count toward the
    declared family size.  Silently shrinking the family to finite p-values
    would make the correction anti-conservative.
    """

    values = np.asarray(list(p_values), dtype=np.float64)
    adjusted = np.full(values.shape, np.nan, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if finite_indices.size == 0:
        return adjusted.tolist()
    finite_values = values[finite_indices]
    order = np.argsort(finite_values)
    ranked = finite_values[order]
    declared_count = values.size if family_size is None else int(family_size)
    if declared_count < values.size or declared_count < finite_indices.size:
        raise ValueError("family_size cannot be smaller than the supplied hypothesis family")
    ranked_adjusted = ranked * declared_count / np.arange(1, ranked.size + 1)
    ranked_adjusted = np.minimum.accumulate(ranked_adjusted[::-1])[::-1]
    ranked_adjusted = np.clip(ranked_adjusted, 0.0, 1.0)
    restored = np.empty_like(ranked_adjusted)
    restored[order] = ranked_adjusted
    adjusted[finite_indices] = restored
    return adjusted.tolist()


def _load_artifact(path: Path) -> dict[str, Any]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(artifact, dict) or not isinstance(artifact.get("records"), list):
        raise ValueError(f"{path} is not a representation-dynamics artifact")
    return artifact


def _load_nuisance_table(
    path: Path,
    columns: tuple[str, ...],
) -> tuple[dict[tuple[str, str, str], tuple[float, ...]], str]:
    """Load one finite nuisance vector per model/task/sample identity."""

    if not columns:
        raise ValueError("At least one nuisance column is required")
    if len(columns) != len(set(columns)):
        raise ValueError("Nuisance column names must be unique")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    table: dict[tuple[str, str, str], tuple[float, ...]] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"model", "task_name", "sample_id", *columns}
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Nuisance table is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            key = (str(row["model"]), str(row["task_name"]), str(row["sample_id"]))
            if key in table:
                raise ValueError(f"Duplicate nuisance identity at line {line_number}: {key}")
            try:
                values = tuple(float(row[column]) for column in columns)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid nuisance value at line {line_number}") from error
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Non-finite nuisance value at line {line_number}")
            table[key] = values
    if not table:
        raise ValueError("Nuisance table is empty")
    return table, digest


def grouped_stratified_folds(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    requested_splits: int,
    seed: int,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], int]:
    """Build folds that hold out complete tasks and contain both classes."""

    group_labels: dict[str, int] = {}
    for group in np.unique(groups):
        values = np.unique(labels[groups == group])
        if values.size != 1:
            raise ValueError(f"Task group {group!r} has mixed class labels")
        group_labels[str(group)] = int(values[0])
    # Establish one label-independent task ordering. Filtering this fixed order
    # by the candidate labels makes the fold builder a deterministic, symmetric
    # part of the statistic evaluated for every task-label assignment.
    ordered_groups = sorted(group_labels)
    rng = np.random.default_rng(seed)
    rng.shuffle(ordered_groups)
    class_groups = {
        label: [group for group in ordered_groups if group_labels[group] == label]
        for label in (0, 1)
    }
    if min(len(class_groups[0]), len(class_groups[1])) < 2:
        raise ValueError("Grouped probing requires at least two distinct tasks in each class")

    effective_splits = min(int(requested_splits), len(class_groups[0]), len(class_groups[1]))
    if effective_splits < 2:
        raise ValueError("At least two grouped cross-validation folds are required")
    chunks = {
        label: np.array_split(np.asarray(class_groups[label], dtype=object), effective_splits)
        for label in (0, 1)
    }

    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_index in range(effective_splits):
        test_groups = set(chunks[0][fold_index].tolist()) | set(chunks[1][fold_index].tolist())
        test_mask = np.asarray([str(group) in test_groups for group in groups], dtype=bool)
        train_indices = np.flatnonzero(~test_mask)
        test_indices = np.flatnonzero(test_mask)
        if np.unique(labels[train_indices]).size != 2 or np.unique(labels[test_indices]).size != 2:
            raise ValueError("A grouped fold does not contain both classes in train and test")
        folds.append((train_indices, test_indices))
    return folds, effective_splits


def _probe_predictions(
    features: Any,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    kind: str,
    requested_splits: int,
    seed: int,
    analysis_backend: str = "sklearn",
    fold_pca_variance: float | None = None,
    nuisance_features: Any | None = None,
    nuisance_alpha: float = 1.0,
    fold_cache: dict[bytes, tuple[Any, Any, Any, Any]] | None = None,
    fold_nuisance_cache: dict[bytes, tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[float, float, int]:
    folds, effective_splits = grouped_stratified_folds(
        labels,
        groups,
        requested_splits=requested_splits,
        seed=seed,
    )
    predictions = np.full(labels.shape, -1, dtype=np.int64)
    for train_indices, test_indices in folds:
        if analysis_backend == "sklearn":
            if kind == "linear":
                classifier = LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=3000,
                    random_state=seed,
                    solver="liblinear",
                )
            elif kind == "rbf":
                classifier = SVC(
                    C=1.0,
                    gamma="scale",
                    kernel="rbf",
                    class_weight="balanced",
                )
            else:
                raise ValueError(f"Unsupported probe kind: {kind}")
            if (
                fold_pca_variance is None
                and nuisance_features is None
                and fold_nuisance_cache is None
            ):
                estimator = make_pipeline(StandardScaler(), classifier)
                estimator.fit(features[train_indices], labels[train_indices])
                predictions[test_indices] = estimator.predict(features[test_indices])
            else:
                cache_key = test_indices.tobytes()
                cached = fold_cache.get(cache_key) if fold_cache is not None else None
                if cached is None:
                    train_features = features[train_indices]
                    test_features = features[test_indices]
                    train_nuisance = (
                        nuisance_features[train_indices]
                        if nuisance_features is not None
                        else None
                    )
                    test_nuisance = (
                        nuisance_features[test_indices]
                        if nuisance_features is not None
                        else None
                    )
                    if fold_nuisance_cache is not None:
                        lexical = fold_nuisance_cache.get(cache_key)
                        if lexical is None:
                            raise ValueError("No fold-local nuisance features for a grouped probe fold")
                        lexical_train, lexical_test = lexical
                        train_nuisance = (
                            lexical_train
                            if train_nuisance is None
                            else np.concatenate((train_nuisance, lexical_train), axis=1)
                        )
                        test_nuisance = (
                            lexical_test
                            if test_nuisance is None
                            else np.concatenate((test_nuisance, lexical_test), axis=1)
                        )
                    if train_nuisance is not None:
                        train_features, test_features = _fold_local_ridge_residuals(
                            train_features,
                            test_features,
                            train_nuisance,
                            test_nuisance,
                            alpha=nuisance_alpha,
                            analysis_backend=analysis_backend,
                        )
                    if fold_pca_variance is not None:
                        train_features, test_features = _fold_local_pca(
                            train_features,
                            test_features,
                            retained_variance=fold_pca_variance,
                            analysis_backend=analysis_backend,
                        )
                    scaler = StandardScaler()
                    train_features = scaler.fit_transform(train_features)
                    test_features = scaler.transform(test_features)
                    cached = (train_indices, test_indices, train_features, test_features)
                    if fold_cache is not None:
                        fold_cache[cache_key] = cached
                _train_indices, _test_indices, train_features, test_features = cached
                classifier.fit(train_features, labels[train_indices])
                predictions[test_indices] = classifier.predict(test_features)
        elif analysis_backend == "cuml":
            cp, _cuml, _pca, CumlLogisticRegression, _silhouette, CumlStandardScaler, CumlSVC = (
                _require_cuml()
            )
            cache_key = test_indices.tobytes()
            cached = fold_cache.get(cache_key) if fold_cache is not None else None
            if cached is None:
                train_gpu = cp.asarray(train_indices)
                test_gpu = cp.asarray(test_indices)
                train_features = features[train_gpu]
                test_features = features[test_gpu]
                train_nuisance = (
                    nuisance_features[train_gpu]
                    if nuisance_features is not None
                    else None
                )
                test_nuisance = (
                    nuisance_features[test_gpu]
                    if nuisance_features is not None
                    else None
                )
                if fold_nuisance_cache is not None:
                    lexical = fold_nuisance_cache.get(cache_key)
                    if lexical is None:
                        raise ValueError("No fold-local nuisance features for a grouped probe fold")
                    lexical_train, lexical_test = lexical
                    lexical_train_gpu = cp.asarray(lexical_train, dtype=cp.float64)
                    lexical_test_gpu = cp.asarray(lexical_test, dtype=cp.float64)
                    train_nuisance = (
                        lexical_train_gpu
                        if train_nuisance is None
                        else cp.concatenate((train_nuisance, lexical_train_gpu), axis=1)
                    )
                    test_nuisance = (
                        lexical_test_gpu
                        if test_nuisance is None
                        else cp.concatenate((test_nuisance, lexical_test_gpu), axis=1)
                    )
                if train_nuisance is not None:
                    train_features, test_features = _fold_local_ridge_residuals(
                        train_features,
                        test_features,
                        train_nuisance,
                        test_nuisance,
                        alpha=nuisance_alpha,
                        analysis_backend=analysis_backend,
                    )
                if fold_pca_variance is not None:
                    train_features, test_features = _fold_local_pca(
                        train_features,
                        test_features,
                        retained_variance=fold_pca_variance,
                        analysis_backend=analysis_backend,
                    )
                scaler = CumlStandardScaler(output_type="cupy")
                train_features = scaler.fit_transform(train_features)
                test_features = scaler.transform(test_features)
                if fold_cache is not None:
                    fold_cache[cache_key] = (train_gpu, test_gpu, train_features, test_features)
            else:
                train_gpu, test_gpu, train_features, test_features = cached
            train_labels_gpu = cp.asarray(labels[train_indices], dtype=cp.int32)
            if kind == "linear":
                classifier = CumlLogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=3000,
                    penalty="l2",
                    penalty_normalized=True,
                    solver="qn",
                    output_type="cupy",
                )
            elif kind == "rbf":
                classifier = CumlSVC(
                    C=1.0,
                    gamma="scale",
                    kernel="rbf",
                    class_weight="balanced",
                    random_state=seed,
                    output_type="cupy",
                )
            else:
                raise ValueError(f"Unsupported probe kind: {kind}")
            classifier.fit(train_features, train_labels_gpu)
            predictions[test_indices] = cp.asnumpy(classifier.predict(test_features)).astype(np.int64)
        else:
            raise ValueError(f"Unsupported analysis backend: {analysis_backend}")
    if np.any(predictions < 0):
        raise RuntimeError("Grouped cross-validation did not produce one prediction per example")
    return (
        float(accuracy_score(labels, predictions)),
        float(f1_score(labels, predictions, average="macro", zero_division=0)),
        effective_splits,
    )


def _permuted_group_assignments(
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    max_permutations: int,
    seed: int,
) -> list[dict[str, int]]:
    if max_permutations <= 0:
        return []
    unique_groups = sorted(str(group) for group in np.unique(groups))
    group_to_label = {group: int(np.unique(labels[groups.astype(str) == group])[0]) for group in unique_groups}
    positive_count = sum(group_to_label[group] == 1 for group in unique_groups)
    observed_positive = frozenset(group for group in unique_groups if group_to_label[group] == 1)
    total_assignments = math.comb(len(unique_groups), positive_count)

    positive_sets: list[frozenset[str]] = []
    if total_assignments - 1 <= max_permutations:
        for combination in itertools.combinations(unique_groups, positive_count):
            assignment = frozenset(combination)
            if assignment != observed_positive:
                positive_sets.append(assignment)
    else:
        rng = np.random.default_rng(seed)
        seen = {observed_positive}
        while len(positive_sets) < max_permutations:
            chosen = frozenset(rng.choice(unique_groups, size=positive_count, replace=False).tolist())
            if chosen in seen:
                continue
            seen.add(chosen)
            positive_sets.append(chosen)

    return [
        {group: int(group in positive_groups) for group in unique_groups}
        for positive_groups in positive_sets[:max_permutations]
    ]


def _assignment_id(assignment: dict[str, int], ordered_groups: list[str]) -> str:
    """Return a compact label vector in one fixed task order."""

    return "".join(str(int(assignment[group])) for group in ordered_groups)


def _permutation_design(
    labels: np.ndarray,
    groups: np.ndarray,
    assignments: list[dict[str, int]],
) -> dict[str, Any]:
    """Describe the exact task-label randomization support.

    Accuracy, macro-F1, and Fisher's trace ratio are invariant to exchanging the
    two class names.  When the task design is balanced and every oriented label
    assignment is enumerated, the support therefore consists of complementary
    pairs.  The nominal 1 / choose(2g, g) grid must not be advertised as the
    attainable minimum for these statistics.
    """

    ordered_groups = sorted(str(group) for group in np.unique(groups))
    group_labels = {
        group: int(np.unique(labels[groups.astype(str) == group])[0])
        for group in ordered_groups
    }
    positive_count = sum(group_labels.values())
    total_assignments = math.comb(len(ordered_groups), positive_count)
    exhaustive = len(assignments) == total_assignments - 1
    balanced = positive_count * 2 == len(ordered_groups)
    observed_id = _assignment_id(group_labels, ordered_groups)
    assignment_ids = [observed_id] + [
        _assignment_id(assignment, ordered_groups) for assignment in assignments
    ]
    expected_ids = len(set(assignment_ids))
    if expected_ids != len(assignment_ids):
        raise RuntimeError("Task-label permutation assignments are not unique")
    assignment_id_set = set(assignment_ids)
    complement_paired = exhaustive and balanced and all(
        "".join("1" if bit == "0" else "0" for bit in assignment_id) in assignment_id_set
        for assignment_id in assignment_ids
    )
    return {
        "task_order": ordered_groups,
        "assignment_ids": assignment_ids,
        "total_oriented_assignments": total_assignments,
        "exhaustive": exhaustive,
        "balanced": balanced,
        "complement_paired": complement_paired,
        "effective_unoriented_assignments": total_assignments // 2 if complement_paired else total_assignments,
        "theoretical_minimum_p": (
            2.0 / total_assignments
            if complement_paired
            else (1.0 / (len(assignments) + 1) if assignments else float("nan"))
        ),
    }


def _complement_symmetry(values: list[float], assignment_ids: list[str]) -> dict[str, Any]:
    """Check the numerical statistic against its label-complement invariant."""

    if not values or len(values) != len(assignment_ids):
        return {"verified": False, "maximum_absolute_difference": float("nan")}
    index = {assignment_id: position for position, assignment_id in enumerate(assignment_ids)}
    differences: list[float] = []
    for position, assignment_id in enumerate(assignment_ids):
        complement = "".join("1" if bit == "0" else "0" for bit in assignment_id)
        if complement not in index:
            return {"verified": False, "maximum_absolute_difference": float("nan")}
        differences.append(abs(float(values[position]) - float(values[index[complement]])))
    maximum = max(differences, default=0.0)
    return {
        "verified": bool(maximum <= EPSILON),
        "maximum_absolute_difference": float(maximum),
    }


def grouped_permutation_probe(
    features: Any,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    kind: str,
    requested_splits: int,
    max_permutations: int,
    seed: int,
    analysis_backend: str = "sklearn",
    fold_pca_variance: float | None = None,
    nuisance_features: Any | None = None,
    nuisance_alpha: float = 1.0,
    fold_cache: dict[bytes, tuple[Any, Any, Any, Any]] | None = None,
) -> dict[str, Any]:
    if (fold_pca_variance is not None or nuisance_features is not None) and fold_cache is None:
        fold_cache = {}
    observed_accuracy, observed_f1, effective_splits = _probe_predictions(
        features,
        labels,
        groups,
        kind=kind,
        requested_splits=requested_splits,
        seed=seed,
        analysis_backend=analysis_backend,
        fold_pca_variance=fold_pca_variance,
        nuisance_features=nuisance_features,
        nuisance_alpha=nuisance_alpha,
        fold_cache=fold_cache,
    )
    assignments = _permuted_group_assignments(
        labels,
        groups,
        max_permutations=max_permutations,
        seed=seed,
    )
    design = _permutation_design(labels, groups, assignments)
    accuracy_distribution: list[float | None] = [None] * (len(assignments) + 1)
    f1_distribution: list[float | None] = [None] * (len(assignments) + 1)
    accuracy_distribution[0] = float(observed_accuracy)
    f1_distribution[0] = float(observed_f1)
    assignment_index = {
        assignment_id: index for index, assignment_id in enumerate(design["assignment_ids"])
    }
    reuse_complements = bool(kind == "linear" and design["complement_paired"])
    string_groups = groups.astype(str)
    probe_refits = 1
    for index, assignment in enumerate(assignments, start=1):
        if accuracy_distribution[index] is not None and f1_distribution[index] is not None:
            continue
        if reuse_complements:
            complement_id = "".join(
                "1" if bit == "0" else "0" for bit in design["assignment_ids"][index]
            )
            complement_index = assignment_index[complement_id]
            if complement_index == 0:
                # The observed fit is the identical statistic after exchanging
                # the two class names.  Copy it rather than fitting its oriented
                # complement a second time.
                accuracy_distribution[index] = float(observed_accuracy)
                f1_distribution[index] = float(observed_f1)
                continue
        else:
            complement_index = index
        permuted_labels = np.asarray([assignment[group] for group in string_groups], dtype=np.int64)
        accuracy, f1, _ = _probe_predictions(
            features,
            permuted_labels,
            groups,
            kind=kind,
            requested_splits=requested_splits,
            # Use the identical deterministic statistic for the observed and
            # every permuted task-label assignment. Changing this seed across
            # assignments would mix fold randomness into the randomization null.
            seed=seed,
            analysis_backend=analysis_backend,
            fold_pca_variance=fold_pca_variance,
            nuisance_features=nuisance_features,
            nuisance_alpha=nuisance_alpha,
            fold_cache=fold_cache,
        )
        probe_refits += 1
        accuracy_distribution[index] = float(accuracy)
        f1_distribution[index] = float(f1)
        if reuse_complements:
            accuracy_distribution[complement_index] = float(accuracy)
            f1_distribution[complement_index] = float(f1)

    num_permutations = len(assignments)
    if any(value is None for value in accuracy_distribution + f1_distribution):
        raise RuntimeError("Grouped probe did not evaluate the complete randomization support")
    accuracy_values = [float(value) for value in accuracy_distribution]
    f1_values = [float(value) for value in f1_distribution]
    null_accuracy = accuracy_values[1:]
    null_f1 = f1_values[1:]
    accuracy_symmetry = _complement_symmetry(accuracy_values, design["assignment_ids"])
    f1_symmetry = _complement_symmetry(f1_values, design["assignment_ids"])
    accuracy_p = (
        (1 + sum(score >= observed_accuracy - EPSILON for score in null_accuracy)) / (1 + num_permutations)
        if num_permutations
        else float("nan")
    )
    f1_p = (
        (1 + sum(score >= observed_f1 - EPSILON for score in null_f1)) / (1 + num_permutations)
        if num_permutations
        else float("nan")
    )
    accuracy_inference_valid = bool(not design["complement_paired"] or accuracy_symmetry["verified"])
    f1_inference_valid = bool(not design["complement_paired"] or f1_symmetry["verified"])
    return {
        "accuracy": observed_accuracy,
        "f1": observed_f1,
        "accuracy_p": float(accuracy_p) if accuracy_inference_valid else float("nan"),
        "f1_p": float(f1_p) if f1_inference_valid else float("nan"),
        "cv_folds": effective_splits,
        "num_group_permutations": num_permutations,
        "fold_seed": int(seed),
        "probe_refits": int(probe_refits),
        "class_label_complements_share_one_probe_refit": reuse_complements,
        "complement_score_symmetry_method": (
            "imposed_by_reusing_one_fit_per_label_complement_pair"
            if reuse_complements
            else "independently_computed"
        ),
        "permutation_design": design,
        "accuracy_distribution": accuracy_values,
        "f1_distribution": f1_values,
        "accuracy_complement_symmetry": accuracy_symmetry,
        "f1_complement_symmetry": f1_symmetry,
        "permutation_symmetric": bool(accuracy_symmetry["verified"] and f1_symmetry["verified"]),
        "accuracy_inference_valid": accuracy_inference_valid,
        "f1_inference_valid": f1_inference_valid,
    }


def _pca_projection(
    features: Any,
    *,
    retained_variance: float,
    analysis_backend: str = "sklearn",
) -> tuple[Any, float]:
    max_components = min(features.shape[0] - 1, features.shape[1])
    if max_components < 1:
        raise ValueError("FDR requires at least two examples and one feature")
    if not 0.0 < retained_variance <= 1.0:
        raise ValueError("retained_variance must lie in (0, 1]")
    if analysis_backend == "sklearn":
        n_components: float | int = retained_variance if retained_variance < 1.0 else max_components
        pca = PCA(n_components=n_components, svd_solver="full")
        projected = pca.fit_transform(features)
        explained_variance = float(pca.explained_variance_ratio_.sum())
    elif analysis_backend == "cuml":
        cp, _cuml, CumlPCA, _logistic, _silhouette, _scaler, _svc = _require_cuml()
        # cuML requires an integer component count. Fit the complete sample-rank
        # basis, then apply the same cumulative explained-variance rule explicitly.
        pca = CumlPCA(n_components=max_components, svd_solver="full", output_type="cupy")
        complete_projection = pca.fit_transform(features)
        cumulative = cp.cumsum(pca.explained_variance_ratio_)
        component_count = (
            max_components
            if retained_variance == 1.0
            else min(max_components, int(cp.searchsorted(cumulative, retained_variance).item()) + 1)
        )
        projected = complete_projection[:, :component_count]
        explained_variance = float(cumulative[component_count - 1].item())
    else:
        raise ValueError(f"Unsupported analysis backend: {analysis_backend}")
    return projected, explained_variance


def _fold_local_pca(
    train_features: Any,
    test_features: Any,
    *,
    retained_variance: float,
    analysis_backend: str,
) -> tuple[Any, Any]:
    """Fit PCA only on a probe fold's training tasks and transform its test tasks."""

    max_components = min(train_features.shape[0] - 1, train_features.shape[1])
    if max_components < 1:
        raise ValueError("Fold-local PCA requires at least two training examples")
    if not 0.0 < retained_variance <= 1.0:
        raise ValueError("retained_variance must lie in (0, 1]")
    if analysis_backend == "sklearn":
        n_components: float | int = retained_variance if retained_variance < 1.0 else max_components
        pca = PCA(n_components=n_components, svd_solver="full")
        return pca.fit_transform(train_features), pca.transform(test_features)
    if analysis_backend == "cuml":
        cp, _cuml, CumlPCA, _logistic, _silhouette, _scaler, _svc = _require_cuml()
        pca = CumlPCA(n_components=max_components, svd_solver="full", output_type="cupy")
        complete_train = pca.fit_transform(train_features)
        cumulative = cp.cumsum(pca.explained_variance_ratio_)
        component_count = (
            max_components
            if retained_variance == 1.0
            else min(max_components, int(cp.searchsorted(cumulative, retained_variance).item()) + 1)
        )
        complete_test = pca.transform(test_features)
        return complete_train[:, :component_count], complete_test[:, :component_count]
    raise ValueError(f"Unsupported analysis backend: {analysis_backend}")


def _fold_local_ridge_residuals(
    train_features: Any,
    test_features: Any,
    train_nuisance: Any,
    test_nuisance: Any,
    *,
    alpha: float,
    analysis_backend: str,
) -> tuple[Any, Any]:
    """Remove nuisance-predictable activation variance without seeing held-out tasks.

    Nuisance columns are standardized on the training tasks. A multi-output Ridge
    model then predicts every activation dimension jointly; its intercept is not
    penalized. The same fitted transform is applied to the held-out tasks. This
    operation is label-blind but must still be repeated for each permutation fold
    because task-label permutations can change which tasks are held out.
    """

    if alpha <= 0.0:
        raise ValueError("nuisance_alpha must be positive")
    if train_nuisance.ndim != 2 or test_nuisance.ndim != 2:
        raise ValueError("Nuisance features must be two-dimensional matrices")
    if train_nuisance.shape[1] < 1 or train_nuisance.shape[1] != test_nuisance.shape[1]:
        raise ValueError("Train and test nuisance matrices need the same nonzero width")
    if train_features.shape[0] != train_nuisance.shape[0]:
        raise ValueError("Training activation and nuisance row counts differ")
    if test_features.shape[0] != test_nuisance.shape[0]:
        raise ValueError("Test activation and nuisance row counts differ")

    if analysis_backend == "sklearn":
        xp = np
    elif analysis_backend == "cuml":
        xp, *_ = _require_cuml()
    else:
        raise ValueError(f"Unsupported analysis backend: {analysis_backend}")

    train_nuisance = xp.asarray(train_nuisance, dtype=xp.float64)
    test_nuisance = xp.asarray(test_nuisance, dtype=xp.float64)
    if not bool(xp.all(xp.isfinite(train_nuisance))) or not bool(xp.all(xp.isfinite(test_nuisance))):
        raise ValueError("Nuisance matrices contain non-finite values")

    nuisance_mean = xp.mean(train_nuisance, axis=0)
    nuisance_scale = xp.std(train_nuisance, axis=0)
    nuisance_scale = xp.where(nuisance_scale > EPSILON, nuisance_scale, 1.0)
    train_standardized = (train_nuisance - nuisance_mean) / nuisance_scale
    test_standardized = (test_nuisance - nuisance_mean) / nuisance_scale
    train_design = xp.concatenate(
        (xp.ones((train_standardized.shape[0], 1), dtype=xp.float64), train_standardized),
        axis=1,
    )
    test_design = xp.concatenate(
        (xp.ones((test_standardized.shape[0], 1), dtype=xp.float64), test_standardized),
        axis=1,
    )
    penalty = xp.eye(train_design.shape[1], dtype=xp.float64) * float(alpha)
    penalty[0, 0] = 0.0
    coefficients = xp.linalg.solve(
        train_design.T @ train_design + penalty,
        train_design.T @ train_features,
    )
    return (
        train_features - train_design @ coefficients,
        test_features - test_design @ coefficients,
    )


def _fisher_from_projection(
    projected: Any,
    labels: np.ndarray,
    *,
    analysis_backend: str,
    explained_variance: float,
) -> dict[str, float | int]:
    if analysis_backend == "cuml":
        cp, *_ = _require_cuml()
        xp = cp
        backend_labels = cp.asarray(labels)
    else:
        xp = np
        backend_labels = labels
    knowledge = projected[backend_labels == 1]
    reasoning = projected[backend_labels == 0]
    mean_difference_sq = xp.square(knowledge.mean(axis=0) - reasoning.mean(axis=0))
    within_variance = knowledge.var(axis=0, ddof=1) + reasoning.var(axis=0, ddof=1)
    per_component = mean_difference_sq / xp.maximum(within_variance, EPSILON)
    return {
        "fdr": float((mean_difference_sq.sum() / xp.maximum(within_variance.sum(), EPSILON)).item()),
        "fdr_between_sum": float(mean_difference_sq.sum().item()),
        "fdr_within_sum": float(within_variance.sum().item()),
        "fdr_pc_mean": float(per_component.mean().item()),
        "fdr_pc_max": float(per_component.max().item()),
        "pca_components": int(projected.shape[1]),
        "pca_explained_variance": explained_variance,
    }


def _fisher_permutation_distribution(
    projected: Any,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    assignments: list[dict[str, int]],
    analysis_backend: str,
    explained_variance: float,
    observed_fdr: float,
) -> dict[str, Any]:
    """Evaluate Fisher's trace ratio on the same task-label randomization support."""

    design = _permutation_design(labels, groups, assignments)
    if analysis_backend == "cuml":
        xp, *_ = _require_cuml()
    else:
        xp = np

    # Recomputing a two-class variance with one GPU kernel launch per assignment
    # leaves the accelerator mostly idle. Fisher's trace ratio depends only on the
    # count, sum, and squared sum of every task. Compute these sufficient statistics
    # once, then evaluate the complete assignment support with matrix operations.
    string_groups = groups.astype(str)
    task_sums = []
    task_squared_sums = []
    task_counts = []
    for task_name in design["task_order"]:
        mask = xp.asarray(string_groups == task_name)
        task_values = projected[mask]
        task_counts.append(int(task_values.shape[0]))
        task_sums.append(task_values.sum(axis=0))
        task_squared_sums.append(xp.square(task_values).sum(axis=0))
    assignment_matrix = xp.asarray(
        [[float(bit) for bit in assignment_id] for assignment_id in design["assignment_ids"]],
        dtype=xp.float64,
    )
    complement_matrix = 1.0 - assignment_matrix
    counts = xp.asarray(task_counts, dtype=xp.float64)
    sums = xp.stack(task_sums, axis=0)
    squared_sums = xp.stack(task_squared_sums, axis=0)

    knowledge_count = assignment_matrix @ counts
    reasoning_count = complement_matrix @ counts
    if bool(xp.any(knowledge_count <= 1.0)) or bool(xp.any(reasoning_count <= 1.0)):
        raise ValueError("Every Fisher assignment requires at least two examples per class")
    knowledge_sum = assignment_matrix @ sums
    reasoning_sum = complement_matrix @ sums
    knowledge_squared_sum = assignment_matrix @ squared_sums
    reasoning_squared_sum = complement_matrix @ squared_sums
    knowledge_count_column = knowledge_count[:, None]
    reasoning_count_column = reasoning_count[:, None]
    knowledge_mean = knowledge_sum / knowledge_count_column
    reasoning_mean = reasoning_sum / reasoning_count_column
    between = xp.square(knowledge_mean - reasoning_mean).sum(axis=1)
    knowledge_variance = (
        knowledge_squared_sum - xp.square(knowledge_sum) / knowledge_count_column
    ) / (knowledge_count_column - 1.0)
    reasoning_variance = (
        reasoning_squared_sum - xp.square(reasoning_sum) / reasoning_count_column
    ) / (reasoning_count_column - 1.0)
    # PCA scores are globally centered and all calculations use float64. The clamp
    # only removes sub-ulp negative values from the sufficient-statistic identity.
    within = (
        xp.maximum(knowledge_variance, 0.0) + xp.maximum(reasoning_variance, 0.0)
    ).sum(axis=1)
    ratios = between / xp.maximum(within, EPSILON)
    if analysis_backend == "cuml":
        distribution = [float(value) for value in xp.asnumpy(ratios).tolist()]
    else:
        distribution = [float(value) for value in np.asarray(ratios).tolist()]
    # Retain the observed value from the already reported two-pass calculation.
    # This prevents an implementation-only rounding difference between the row and
    # its randomization payload. Set its label complement to the same value because
    # Fisher's trace ratio is exactly invariant to swapping the two class names.
    distribution[0] = float(observed_fdr)
    observed_complement = "".join(
        "1" if bit == "0" else "0" for bit in design["assignment_ids"][0]
    )
    if observed_complement in design["assignment_ids"]:
        distribution[design["assignment_ids"].index(observed_complement)] = float(observed_fdr)
    p_value = (
        sum(value >= float(observed_fdr) - EPSILON for value in distribution) / len(distribution)
        if distribution
        else float("nan")
    )
    symmetry = _complement_symmetry(distribution, design["assignment_ids"])
    inference_valid = bool(not design["complement_paired"] or symmetry["verified"])
    return {
        "p": float(p_value) if inference_valid else float("nan"),
        "distribution": distribution,
        "permutation_design": design,
        "complement_symmetry": symmetry,
        "inference_valid": inference_valid,
        "calculation": "vectorized_task_sufficient_statistics",
    }


def _randomization_p(distribution: np.ndarray, observed_index: int = 0) -> float:
    """Return the finite-support randomization p-value including the observed assignment."""

    values = np.asarray(distribution, dtype=np.float64)
    if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
        return float("nan")
    return float(np.mean(values >= values[observed_index] - EPSILON))


def _assignmentwise_p_values(score_matrix: np.ndarray) -> np.ndarray:
    """Compute each assignment's exact row-wise p-value on a shared null support."""

    scores = np.asarray(score_matrix, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] < 1 or not np.all(np.isfinite(scores)):
        raise ValueError("Permutation score matrix must be finite and two-dimensional")
    output = np.empty_like(scores)
    for row_index, values in enumerate(scores):
        # Columns are candidate observed assignments; rows in the comparison
        # are the complete finite randomization support. Ties are included.
        output[row_index] = np.mean(
            values[None, :] >= values[:, None] - EPSILON,
            axis=1,
        )
    return output


def _model_randomization_summary(
    model_rows: list[dict[str, Any]],
    permutation_payloads: list[dict[str, Any]],
    *,
    alpha: float = 0.05,
    reference_positive_tasks: set[str] | None = None,
) -> dict[str, Any]:
    """Calibrate maxima and BH rejection counts over one shared task-label null."""

    if len(model_rows) != len(permutation_payloads) or not model_rows:
        raise ValueError("Randomization payloads must align one-to-one with model rows")
    assignment_ids = permutation_payloads[0]["assignment_ids"]
    task_order = permutation_payloads[0]["task_order"]
    for payload in permutation_payloads[1:]:
        if payload["assignment_ids"] != assignment_ids or payload["task_order"] != task_order:
            raise ValueError("Layer/stream rows do not share one aligned task-label null")

    result: dict[str, Any] = {
        "task_order": task_order,
        "assignment_ids": assignment_ids,
        "assignments": len(assignment_ids),
        "alpha": float(alpha),
        "metrics": {},
    }
    reference_index = None
    reference_id = None
    if reference_positive_tasks is not None:
        unknown = reference_positive_tasks - set(task_order)
        if unknown:
            raise ValueError(f"Reference partition contains unknown tasks: {sorted(unknown)}")
        if len(reference_positive_tasks) * 2 != len(task_order):
            raise ValueError("Reference partition must assign exactly half of tasks to its positive group")
        reference_id = "".join("1" if task in reference_positive_tasks else "0" for task in task_order)
        try:
            reference_index = assignment_ids.index(reference_id)
        except ValueError as error:
            raise ValueError("Reference partition is absent from the task-label randomization support") from error
        result["reference_partition"] = {
            "positive_tasks": sorted(reference_positive_tasks),
            "assignment_id": reference_id,
            "assignment_index": int(reference_index),
        }
    available_matrices: dict[str, np.ndarray] = {}
    for metric, payload_key in (
        ("fdr", "fdr"),
        ("linear_accuracy", "linear_accuracy"),
        ("linear_f1", "linear_f1"),
        ("rbf_accuracy", "rbf_accuracy"),
        ("rbf_f1", "rbf_f1"),
        ("nuisance_linear_accuracy", "nuisance_linear_accuracy"),
        ("nuisance_linear_f1", "nuisance_linear_f1"),
        ("nuisance_rbf_accuracy", "nuisance_rbf_accuracy"),
        ("nuisance_rbf_f1", "nuisance_rbf_f1"),
    ):
        if any(payload.get(payload_key) is None for payload in permutation_payloads):
            continue
        matrix = np.asarray([payload[payload_key] for payload in permutation_payloads], dtype=np.float64)
        if matrix.shape != (len(model_rows), len(assignment_ids)) or not np.all(np.isfinite(matrix)):
            continue
        available_matrices[metric] = matrix
        maximum_distribution = np.max(matrix, axis=0)
        assignmentwise_p = _assignmentwise_p_values(matrix)
        rejection_counts: list[int] = []
        for assignment_index in range(matrix.shape[1]):
            q_values = benjamini_hochberg(assignmentwise_p[:, assignment_index])
            rejection_counts.append(sum(float(value) < alpha for value in q_values if np.isfinite(value)))
        rejection_array = np.asarray(rejection_counts, dtype=np.int64)
        maximum_symmetry = _complement_symmetry(
            maximum_distribution.tolist(),
            assignment_ids,
        )
        count_symmetry = _complement_symmetry(
            rejection_array.astype(np.float64).tolist(),
            assignment_ids,
        )
        theoretical_minimum = float(permutation_payloads[0]["theoretical_minimum_p"])
        has_randomization_null = len(assignment_ids) > 1
        minimum_bh_rejections = (
            int(math.ceil(len(model_rows) * theoretical_minimum / alpha))
            if has_randomization_null and math.isfinite(theoretical_minimum)
            else None
        )
        null_maxima = maximum_distribution[1:]
        null_rejection_counts = rejection_array[1:]
        metric_summary = {
            "observed_maximum": float(maximum_distribution[0]),
            "maximum_randomization_p": (
                _randomization_p(maximum_distribution)
                if maximum_symmetry["verified"]
                else float("nan")
            ),
            "maximum_complement_symmetry": maximum_symmetry,
            "null_maximum_median": (
                float(np.median(null_maxima)) if has_randomization_null else None
            ),
            "null_maximum_q95": (
                float(np.quantile(null_maxima, 0.95)) if has_randomization_null else None
            ),
            "observed_bh_rejections": int(rejection_array[0]),
            "rejection_count_randomization_p": (
                _randomization_p(rejection_array.astype(np.float64))
                if count_symmetry["verified"]
                else float("nan")
            ),
            "rejection_count_complement_symmetry": count_symmetry,
            "null_rejection_count_median": (
                float(np.median(null_rejection_counts)) if has_randomization_null else None
            ),
            "null_rejection_count_q95": (
                float(np.quantile(null_rejection_counts, 0.95))
                if has_randomization_null
                else None
            ),
            "minimum_bh_rejections_from_discrete_floor": (
                minimum_bh_rejections
                if minimum_bh_rejections is not None
                and minimum_bh_rejections <= len(model_rows)
                else None
            ),
            "bh_rejection_possible_at_discrete_floor": (
                minimum_bh_rejections <= len(model_rows)
                if minimum_bh_rejections is not None
                else None
            ),
        }
        if reference_index is not None:
            metric_summary["reference_partition"] = {
                "maximum": float(maximum_distribution[reference_index]),
                "maximum_randomization_p": _randomization_p(maximum_distribution, reference_index),
                "bh_rejections": int(rejection_array[reference_index]),
                "rejection_count_randomization_p": _randomization_p(
                    rejection_array.astype(np.float64), reference_index
                ),
            }
        result["metrics"][metric] = metric_summary
    if "fdr" in available_matrices:
        fdr_matrix = available_matrices["fdr"]
        selected_rows = np.argmax(fdr_matrix, axis=0)
        selection_counts = Counter(int(index) for index in selected_rows.tolist())
        result["fdr_selected_site_frequency"] = [
            {
                "layer": int(model_rows[row_index]["layer"]),
                "stream": str(model_rows[row_index]["stream"]),
                "trajectory_position": str(model_rows[row_index].get("trajectory_position", "aggregate")),
                "assignments_selected": int(count),
            }
            for row_index, count in selection_counts.most_common()
        ]
        assignment_indices = np.arange(len(assignment_ids), dtype=np.int64)
        for metric in (
            "linear_accuracy",
            "linear_f1",
            "rbf_accuracy",
            "rbf_f1",
            "nuisance_linear_accuracy",
            "nuisance_linear_f1",
            "nuisance_rbf_accuracy",
            "nuisance_rbf_f1",
        ):
            matrix = available_matrices.get(metric)
            if matrix is None:
                continue
            selected_distribution = matrix[selected_rows, assignment_indices]
            selected_symmetry = _complement_symmetry(
                selected_distribution.tolist(),
                assignment_ids,
            )
            result["metrics"][f"{metric}_at_fdr_selected_row"] = {
                "observed": float(selected_distribution[0]),
                "randomization_p": (
                    _randomization_p(selected_distribution)
                    if selected_symmetry["verified"]
                    else float("nan")
                ),
                "complement_symmetry": selected_symmetry,
                "null_median": (
                    float(np.median(selected_distribution[1:]))
                    if len(assignment_ids) > 1
                    else None
                ),
                "null_q95": (
                    float(np.quantile(selected_distribution[1:], 0.95))
                    if len(assignment_ids) > 1
                    else None
                ),
                "selection_recomputed_for_every_assignment": True,
            }
            if reference_index is not None:
                result["metrics"][f"{metric}_at_fdr_selected_row"]["reference_partition"] = {
                    "value": float(selected_distribution[reference_index]),
                    "randomization_p": _randomization_p(selected_distribution, reference_index),
                }
    return result


def _effective_rank(values: Any, *, analysis_backend: str) -> dict[str, float | int]:
    """Return entropy and stable ranks from the singular values of a centered matrix."""

    if analysis_backend == "cuml":
        cp, *_ = _require_cuml()
        xp = cp
    else:
        xp = np
    centered = values - values.mean(axis=0, keepdims=True)
    gram = centered @ centered.T
    eigenvalues = xp.maximum(xp.linalg.eigvalsh(gram), 0.0)
    largest_eigenvalue = eigenvalues.max()
    tolerance = largest_eigenvalue * max(values.shape) * xp.finfo(eigenvalues.dtype).eps
    eigenvalues = xp.where(eigenvalues > tolerance, eigenvalues, 0.0)
    total_squared = eigenvalues.sum()
    largest_squared = eigenvalues.max()
    singular_values = xp.sqrt(eigenvalues)
    singular_total = singular_values.sum()
    capacity = min(max(int(values.shape[0]) - 1, 0), int(values.shape[1]))
    if float(singular_total.item()) <= EPSILON:
        return {
            "effective_rank": float("nan"),
            "effective_rank_fraction": float("nan"),
            "stable_rank": float("nan"),
            "rank_capacity": capacity,
        }
    probabilities = singular_values / singular_total
    positive = probabilities > 0
    entropy = -(probabilities[positive] * xp.log(probabilities[positive])).sum()
    effective_rank = float(xp.exp(entropy).item())
    stable_rank = (
        float((total_squared / largest_squared).item())
        if float(largest_squared.item()) > EPSILON
        else float("nan")
    )
    return {
        "effective_rank": effective_rank,
        "effective_rank_fraction": effective_rank / capacity if capacity > 0 else float("nan"),
        "stable_rank": stable_rank,
        "rank_capacity": capacity,
    }


def rank_diagnostics(features: Any, labels: np.ndarray, *, analysis_backend: str = "sklearn") -> dict[str, float | int]:
    """Measure total and within-class dimensionality without changing the primary FDR."""

    if analysis_backend == "cuml":
        cp, *_ = _require_cuml()
        backend_labels = cp.asarray(labels)
        xp = cp
    else:
        backend_labels = labels
        xp = np
    knowledge = features[backend_labels == 1]
    reasoning = features[backend_labels == 0]
    within = features.copy()
    within[backend_labels == 1] -= knowledge.mean(axis=0, keepdims=True)
    within[backend_labels == 0] -= reasoning.mean(axis=0, keepdims=True)

    all_rank = _effective_rank(features, analysis_backend=analysis_backend)
    within_rank = _effective_rank(within, analysis_backend=analysis_backend)
    within_capacity = min(max(int(features.shape[0]) - 2, 0), int(features.shape[1]))
    within_rank["rank_capacity"] = within_capacity
    within_rank["effective_rank_fraction"] = (
        float(within_rank["effective_rank"]) / within_capacity
        if within_capacity > 0 and np.isfinite(float(within_rank["effective_rank"]))
        else float("nan")
    )
    knowledge_rank = _effective_rank(knowledge, analysis_backend=analysis_backend)
    reasoning_rank = _effective_rank(reasoning, analysis_backend=analysis_backend)
    mean_difference = knowledge.mean(axis=0) - reasoning.mean(axis=0)
    within_variance = knowledge.var(axis=0, ddof=1).sum() + reasoning.var(axis=0, ddof=1).sum()
    return {
        "effective_rank_all": all_rank["effective_rank"],
        "effective_rank_all_fraction": all_rank["effective_rank_fraction"],
        "stable_rank_all": all_rank["stable_rank"],
        "rank_capacity_all": all_rank["rank_capacity"],
        "effective_rank_within": within_rank["effective_rank"],
        "effective_rank_within_fraction": within_rank["effective_rank_fraction"],
        "stable_rank_within": within_rank["stable_rank"],
        "rank_capacity_within": within_rank["rank_capacity"],
        "effective_rank_knowledge": knowledge_rank["effective_rank"],
        "effective_rank_reasoning": reasoning_rank["effective_rank"],
        "between_mean_squared_norm_raw": float(xp.square(mean_difference).sum().item()),
        "within_variance_sum_raw": float(within_variance.item()),
    }


def pca_fisher_ratio(
    features: Any,
    labels: np.ndarray,
    *,
    retained_variance: float,
    analysis_backend: str = "sklearn",
) -> dict[str, float | int]:
    projected, explained_variance = _pca_projection(
        features,
        retained_variance=retained_variance,
        analysis_backend=analysis_backend,
    )
    return _fisher_from_projection(
        projected,
        labels,
        analysis_backend=analysis_backend,
        explained_variance=explained_variance,
    )


def covariance_alignment(knowledge: np.ndarray, reasoning: np.ndarray) -> float:
    """Order-invariant normalized inner product between feature covariances."""

    x = knowledge - knowledge.mean(axis=0, keepdims=True)
    y = reasoning - reasoning.mean(axis=0, keepdims=True)
    cross = x @ y.T
    x_gram = x @ x.T
    y_gram = y @ y.T
    numerator = float(np.square(cross).sum())
    denominator = math.sqrt(float(np.square(x_gram).sum()) * float(np.square(y_gram).sum()))
    return numerator / denominator if denominator > EPSILON else float("nan")


def paired_linear_cka(knowledge: np.ndarray, reasoning: np.ndarray) -> float:
    """Linear CKA for row-aligned observations only."""

    if knowledge.shape[0] != reasoning.shape[0] or knowledge.shape[0] < 2:
        return float("nan")
    x = knowledge - knowledge.mean(axis=0, keepdims=True)
    y = reasoning - reasoning.mean(axis=0, keepdims=True)
    cross = x.T @ y
    x_cov = x.T @ x
    y_cov = y.T @ y
    numerator = float(np.square(cross).sum())
    denominator = math.sqrt(float(np.square(x_cov).sum()) * float(np.square(y_cov).sum()))
    return numerator / denominator if denominator > EPSILON else float("nan")


def _paired_matrices(features: np.ndarray, labels: np.ndarray, pair_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    paired: dict[str, dict[int, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for vector, label, pair_id in zip(features, labels, pair_ids):
        if pair_id in {None, "", "None"}:
            continue
        paired[str(pair_id)][int(label)].append(vector)
    complete = [
        (values[1][0], values[0][0])
        for _, values in sorted(paired.items())
        if len(values.get(0, [])) == 1 and len(values.get(1, [])) == 1
    ]
    if len(complete) < 2:
        return np.empty((0, features.shape[1])), np.empty((0, features.shape[1]))
    knowledge, reasoning = zip(*complete)
    return np.stack(knowledge), np.stack(reasoning)


def cosine_silhouette(features: np.ndarray, labels: np.ndarray, *, max_samples: int, seed: int) -> tuple[float, int]:
    nonzero = np.linalg.norm(features, axis=1) > EPSILON
    values = features[nonzero]
    value_labels = labels[nonzero]
    if np.unique(value_labels).size != 2:
        return float("nan"), 0
    rng = np.random.default_rng(seed)
    per_class_cap = max(2, max_samples // 2)
    selected: list[int] = []
    for label in (0, 1):
        indices = np.flatnonzero(value_labels == label)
        if indices.size > per_class_cap:
            indices = rng.choice(indices, size=per_class_cap, replace=False)
        selected.extend(indices.tolist())
    selected_array = np.asarray(selected, dtype=np.int64)
    if selected_array.size < 4:
        return float("nan"), int(selected_array.size)
    selected_features = values[selected_array]
    selected_labels = value_labels[selected_array]
    score = _batched_two_class_cosine_silhouette(
        np,
        selected_features,
        selected_labels,
    )
    return score, int(selected_array.size)


def _cuml_covariance_alignment(features: Any, labels: np.ndarray) -> float:
    cp, *_ = _require_cuml()
    backend_labels = cp.asarray(labels)
    knowledge = features[backend_labels == 1]
    reasoning = features[backend_labels == 0]
    x = knowledge - knowledge.mean(axis=0, keepdims=True)
    y = reasoning - reasoning.mean(axis=0, keepdims=True)
    cross = x @ y.T
    x_gram = x @ x.T
    y_gram = y @ y.T
    numerator = cp.square(cross).sum()
    denominator = cp.sqrt(cp.square(x_gram).sum() * cp.square(y_gram).sum())
    return float((numerator / denominator).item()) if float(denominator.item()) > EPSILON else float("nan")


def _batched_two_class_cosine_silhouette(
    array_module: Any,
    features: Any,
    labels: Any,
    *,
    batch_size: int = 128,
) -> float:
    """Compute exact two-class cosine silhouette without a full distance matrix."""

    norms = array_module.linalg.norm(features, axis=1, keepdims=True)
    normalized = features / norms
    scores = []
    num_samples = int(features.shape[0])
    for start in range(0, num_samples, batch_size):
        stop = min(start + batch_size, num_samples)
        distances = 1.0 - normalized[start:stop] @ normalized.T
        distances = array_module.clip(distances, 0.0, 2.0)
        batch_indices = array_module.arange(stop - start)
        source_rows = array_module.arange(start, stop)
        distances[batch_indices, source_rows] = 0.0

        batch_labels = labels[start:stop]
        same_class = batch_labels[:, None] == labels[None, :]
        same_count = array_module.sum(same_class, axis=1) - 1
        other_count = num_samples - array_module.sum(same_class, axis=1)
        within = array_module.sum(array_module.where(same_class, distances, 0.0), axis=1) / same_count
        nearest_other = (
            array_module.sum(array_module.where(~same_class, distances, 0.0), axis=1)
            / other_count
        )
        denominator = array_module.maximum(within, nearest_other)
        scores.append(
            array_module.where(
                denominator > EPSILON,
                (nearest_other - within) / denominator,
                0.0,
            )
        )
    return float(array_module.concatenate(scores).mean().item())


def _cuml_cosine_silhouette(
    features: Any,
    labels: np.ndarray,
    *,
    max_samples: int,
    seed: int,
) -> tuple[float, int]:
    cp, *_ = _require_cuml()
    nonzero = cp.asnumpy(cp.linalg.norm(features, axis=1) > EPSILON)
    valid_indices = np.flatnonzero(nonzero)
    value_labels = labels[valid_indices]
    if np.unique(value_labels).size != 2:
        return float("nan"), 0
    rng = np.random.default_rng(seed)
    per_class_cap = max(2, max_samples // 2)
    selected: list[int] = []
    for label in (0, 1):
        indices = np.flatnonzero(value_labels == label)
        if indices.size > per_class_cap:
            indices = rng.choice(indices, size=per_class_cap, replace=False)
        selected.extend(indices.tolist())
    selected_array = np.asarray(selected, dtype=np.int64)
    if selected_array.size < 4:
        return float("nan"), int(selected_array.size)
    source_indices = cp.asarray(valid_indices[selected_array])
    selected_labels = cp.asarray(value_labels[selected_array], dtype=cp.int32)
    selected_features = features[source_indices]
    # cuML 26.06's convenience implementation produced schedule-dependent
    # values for a small number of real rows under concurrent MPS processes.
    # Compute the exact two-class cosine silhouette explicitly in fixed-size
    # batches. This matches sklearn's float64 result without materializing the
    # complete N x N distance matrix.
    num_samples = int(selected_array.size)
    score = _batched_two_class_cosine_silhouette(cp, selected_features, selected_labels)
    return score, num_samples


def _metric_extrema(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_directions = {
        "fdr": "max",
        "probe_accuracy": "max",
        "probe_f1": "max",
        "rbf_probe_accuracy": "max",
        "rbf_probe_f1": "max",
        "silhouette_cosine": "max",
        "cka": "min",
        "covariance_alignment": "min",
    }
    extrema: dict[str, Any] = {}
    for metric, direction in metric_directions.items():
        candidates = [row for row in rows if np.isfinite(float(row[metric]))]
        if not candidates:
            extrema[metric] = None
            continue
        selected = (max if direction == "max" else min)(candidates, key=lambda row: float(row[metric]))
        extrema[metric] = {
            "direction": direction,
            "value": selected[metric],
            "model": selected["model"],
            "phase": selected["phase"],
            "trajectory_position": selected.get("trajectory_position", "aggregate"),
            "layer": selected["layer"],
            "stream": selected["stream"],
            "module": selected["module"],
        }
    return extrema


def _analyze_record_group(
    item: tuple[tuple[str, str, str, int, str, str], list[dict[str, Any]]],
    *,
    cv_folds: int,
    permutations: int,
    fdr_permutations: int,
    nonlinear_permutations: int,
    pca_variance: float,
    silhouette_max_samples: int,
    seed: int,
    analysis_backend: str,
    nuisance_columns: tuple[str, ...],
    nuisance_alpha: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Analyze one independent model/layer/stream group.

    The caller may execute this function concurrently. All random seeds are
    derived solely from the analysis configuration, so scheduling cannot alter
    folds, permutations, or reported values.
    """

    key, records = item
    model_name, row_phase, trajectory_position, layer_index, stream, module_name = key
    try:
        vectors = [record["vector"].detach().float().cpu().numpy().reshape(-1) for record in records]
        sample_ids = [str(record.get("sample_id")) for record in records]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Duplicate sample_id values detected; repeated examples cannot be treated as independent")
        hidden_sizes = {vector.shape[0] for vector in vectors}
        if len(hidden_sizes) != 1:
            raise ValueError(f"Mixed hidden sizes: {sorted(hidden_sizes)}")
        # Promote stored activation vectors to float64 for both estimator
        # families. Real float32 cuML probes showed line-search stagnation;
        # float64 PCA-space probes were stable in the predeclared smoke row.
        features = np.stack(vectors).astype(np.float64, copy=False)
        labels = np.asarray([record["label"] for record in records], dtype=np.int64)
        groups = np.asarray([record["task_name"] for record in records], dtype=object)
        pair_ids = np.asarray([record.get("pair_id") for record in records], dtype=object)
        if min(np.sum(labels == 0), np.sum(labels == 1)) < 2:
            raise ValueError("Each class needs at least two examples")

        if analysis_backend == "cuml":
            cp, cuml, *_ = _require_cuml()
            analysis_features = cp.asarray(features)
            analysis_nuisance = (
                cp.asarray([record["nuisance_features"] for record in records], dtype=cp.float64)
                if nuisance_columns
                else None
            )
            backend_version = str(cuml.__version__)
        else:
            analysis_features = features
            analysis_nuisance = (
                np.asarray([record["nuisance_features"] for record in records], dtype=np.float64)
                if nuisance_columns
                else None
            )
            backend_version = None

        projected_features, explained_variance = _pca_projection(
            analysis_features,
            retained_variance=pca_variance,
            analysis_backend=analysis_backend,
        )
        fdr = _fisher_from_projection(
            projected_features,
            labels,
            analysis_backend=analysis_backend,
            explained_variance=explained_variance,
        )
        fdr_assignments = _permuted_group_assignments(
            labels,
            groups,
            max_permutations=fdr_permutations,
            seed=seed,
        )
        fdr_permutation = _fisher_permutation_distribution(
            projected_features,
            labels,
            groups,
            assignments=fdr_assignments,
            analysis_backend=analysis_backend,
            explained_variance=explained_variance,
            observed_fdr=float(fdr["fdr"]),
        )
        ranks = rank_diagnostics(analysis_features, labels, analysis_backend=analysis_backend)
        probe_features = analysis_features
        fold_pca_variance = pca_variance if analysis_backend == "cuml" else None
        probe_feature_space = (
            f"fold_local_pca_{pca_variance:g}" if analysis_backend == "cuml" else "raw"
        )
        shared_fold_cache: dict[bytes, tuple[Any, Any, Any, Any]] | None = (
            {} if fold_pca_variance is not None else None
        )
        linear_probe = grouped_permutation_probe(
            probe_features,
            labels,
            groups,
            kind="linear",
            requested_splits=cv_folds,
            max_permutations=permutations,
            seed=seed,
            analysis_backend=analysis_backend,
            fold_pca_variance=fold_pca_variance,
            fold_cache=shared_fold_cache,
        )
        nonlinear_probe = grouped_permutation_probe(
            probe_features,
            labels,
            groups,
            kind="rbf",
            requested_splits=cv_folds,
            max_permutations=nonlinear_permutations,
            seed=seed,
            analysis_backend=analysis_backend,
            fold_pca_variance=fold_pca_variance,
            fold_cache=shared_fold_cache,
        )
        nuisance_probe = None
        nuisance_nonlinear_probe = None
        if analysis_nuisance is not None:
            nuisance_fold_cache: dict[bytes, tuple[Any, Any, Any, Any]] = {}
            nuisance_probe = grouped_permutation_probe(
                probe_features,
                labels,
                groups,
                kind="linear",
                requested_splits=cv_folds,
                max_permutations=permutations,
                seed=seed,
                analysis_backend=analysis_backend,
                fold_pca_variance=fold_pca_variance,
                nuisance_features=analysis_nuisance,
                nuisance_alpha=nuisance_alpha,
                fold_cache=nuisance_fold_cache,
            )
            nuisance_nonlinear_probe = grouped_permutation_probe(
                probe_features,
                labels,
                groups,
                kind="rbf",
                requested_splits=cv_folds,
                max_permutations=nonlinear_permutations,
                seed=seed,
                analysis_backend=analysis_backend,
                fold_pca_variance=fold_pca_variance,
                nuisance_features=analysis_nuisance,
                nuisance_alpha=nuisance_alpha,
                fold_cache=nuisance_fold_cache,
            )
        knowledge = features[labels == 1]
        reasoning = features[labels == 0]
        paired_knowledge, paired_reasoning = _paired_matrices(features, labels, pair_ids)
        cka = paired_linear_cka(paired_knowledge, paired_reasoning)
        if analysis_backend == "cuml":
            alignment = _cuml_covariance_alignment(analysis_features, labels)
            silhouette, silhouette_samples = _cuml_cosine_silhouette(
                analysis_features,
                labels,
                max_samples=silhouette_max_samples,
                seed=seed,
            )
        else:
            alignment = covariance_alignment(knowledge, reasoning)
            silhouette, silhouette_samples = cosine_silhouette(
                features,
                labels,
                max_samples=silhouette_max_samples,
                seed=seed,
            )
        row = {
                "model": model_name,
                "phase": row_phase,
                "trajectory_position": trajectory_position,
                "layer": layer_index,
                "stream": stream,
                "module": module_name,
                "analysis_backend": analysis_backend,
                "analysis_backend_version": backend_version,
                "analysis_feature_dtype": str(features.dtype),
                "probe_feature_space": probe_feature_space,
                "probe_pca_fit_scope": "training_tasks_per_fold" if fold_pca_variance is not None else "none",
                "residual_delta_exact": all(bool(record.get("residual_delta_exact")) for record in records),
                "num_examples": int(features.shape[0]),
                "num_knowledge": int(np.sum(labels == 1)),
                "num_reasoning": int(np.sum(labels == 0)),
                "num_knowledge_tasks": int(np.unique(groups[labels == 1]).size),
                "num_reasoning_tasks": int(np.unique(groups[labels == 0]).size),
                "hidden_size": int(features.shape[1]),
                **fdr,
                **ranks,
                "probe_accuracy": linear_probe["accuracy"],
                "probe_f1": linear_probe["f1"],
                "probe_accuracy_p": linear_probe["accuracy_p"],
                "probe_f1_p": linear_probe["f1_p"],
                "fdr_p": fdr_permutation["p"],
                "fdr_num_group_permutations": len(fdr_assignments),
                "rbf_probe_accuracy": nonlinear_probe["accuracy"],
                "rbf_probe_f1": nonlinear_probe["f1"],
                "rbf_probe_accuracy_p": nonlinear_probe["accuracy_p"],
                "rbf_probe_f1_p": nonlinear_probe["f1_p"],
                "cv_folds": linear_probe["cv_folds"],
                "num_group_permutations": linear_probe["num_group_permutations"],
                "probe_fold_seed": linear_probe["fold_seed"],
                "probe_permutation_symmetric": bool(
                    linear_probe["accuracy_complement_symmetry"]["verified"]
                    and linear_probe["f1_complement_symmetry"]["verified"]
                ),
                "probe_accuracy_inference_valid": bool(linear_probe["accuracy_inference_valid"]),
                "probe_f1_inference_valid": bool(linear_probe["f1_inference_valid"]),
                "probe_complement_max_absolute_difference": max(
                    float(linear_probe["accuracy_complement_symmetry"]["maximum_absolute_difference"]),
                    float(linear_probe["f1_complement_symmetry"]["maximum_absolute_difference"]),
                ),
                "minimum_attainable_probe_p": float(
                    linear_probe["permutation_design"]["theoretical_minimum_p"]
                ),
                "minimum_attainable_fdr_p": float(
                    fdr_permutation["permutation_design"]["theoretical_minimum_p"]
                ),
                "minimum_attainable_rbf_probe_p": float(
                    nonlinear_probe["permutation_design"]["theoretical_minimum_p"]
                ),
                "rbf_probe_permutation_symmetric": bool(
                    nonlinear_probe["accuracy_complement_symmetry"]["verified"]
                    and nonlinear_probe["f1_complement_symmetry"]["verified"]
                ),
                "rbf_probe_accuracy_inference_valid": bool(nonlinear_probe["accuracy_inference_valid"]),
                "rbf_probe_f1_inference_valid": bool(nonlinear_probe["f1_inference_valid"]),
                "rbf_probe_complement_max_absolute_difference": max(
                    float(nonlinear_probe["accuracy_complement_symmetry"]["maximum_absolute_difference"]),
                    float(nonlinear_probe["f1_complement_symmetry"]["maximum_absolute_difference"]),
                ),
                "fdr_permutation_symmetric": bool(fdr_permutation["complement_symmetry"]["verified"]),
                "fdr_inference_valid": bool(fdr_permutation["inference_valid"]),
                "fdr_complement_max_absolute_difference": float(
                    fdr_permutation["complement_symmetry"]["maximum_absolute_difference"]
                ),
                "permutation_oriented_assignments": int(
                    fdr_permutation["permutation_design"]["total_oriented_assignments"]
                ),
                "permutation_effective_unoriented_assignments": int(
                    fdr_permutation["permutation_design"]["effective_unoriented_assignments"]
                ),
                "cka": cka,
                "cka_pairs": int(paired_knowledge.shape[0]),
                "covariance_alignment": alignment,
                "silhouette_cosine": silhouette,
                "silhouette_samples": silhouette_samples,
            }
        if nuisance_probe is not None and nuisance_nonlinear_probe is not None:
            row.update(
                {
                    "nuisance_columns": ",".join(nuisance_columns),
                    "nuisance_alpha": float(nuisance_alpha),
                    "nuisance_fit_scope": "training_tasks_per_fold",
                    "nuisance_probe_accuracy": nuisance_probe["accuracy"],
                    "nuisance_probe_f1": nuisance_probe["f1"],
                    "nuisance_probe_accuracy_p": nuisance_probe["accuracy_p"],
                    "nuisance_probe_f1_p": nuisance_probe["f1_p"],
                    "nuisance_rbf_probe_accuracy": nuisance_nonlinear_probe["accuracy"],
                    "nuisance_rbf_probe_f1": nuisance_nonlinear_probe["f1"],
                    "nuisance_rbf_probe_accuracy_p": nuisance_nonlinear_probe["accuracy_p"],
                    "nuisance_rbf_probe_f1_p": nuisance_nonlinear_probe["f1_p"],
                    "nuisance_num_group_permutations": nuisance_probe["num_group_permutations"],
                    "minimum_attainable_nuisance_probe_p": float(
                        nuisance_probe["permutation_design"]["theoretical_minimum_p"]
                    ),
                    "nuisance_probe_permutation_symmetric": bool(
                        nuisance_probe["accuracy_complement_symmetry"]["verified"]
                        and nuisance_probe["f1_complement_symmetry"]["verified"]
                    ),
                    "nuisance_probe_accuracy_inference_valid": bool(
                        nuisance_probe["accuracy_inference_valid"]
                    ),
                    "nuisance_probe_f1_inference_valid": bool(nuisance_probe["f1_inference_valid"]),
                    "nuisance_rbf_probe_permutation_symmetric": bool(
                        nuisance_nonlinear_probe["accuracy_complement_symmetry"]["verified"]
                        and nuisance_nonlinear_probe["f1_complement_symmetry"]["verified"]
                    ),
                    "nuisance_rbf_probe_accuracy_inference_valid": bool(
                        nuisance_nonlinear_probe["accuracy_inference_valid"]
                    ),
                    "nuisance_rbf_probe_f1_inference_valid": bool(
                        nuisance_nonlinear_probe["f1_inference_valid"]
                    ),
                }
            )
        row["_permutation_payload"] = {
            "task_order": fdr_permutation["permutation_design"]["task_order"],
            "assignment_ids": fdr_permutation["permutation_design"]["assignment_ids"],
            "theoretical_minimum_p": fdr_permutation["permutation_design"]["theoretical_minimum_p"],
            "fdr": fdr_permutation["distribution"] if fdr_permutation["inference_valid"] else None,
            "linear_accuracy": (
                linear_probe["accuracy_distribution"]
                if linear_probe["num_group_permutations"] == len(fdr_assignments)
                and linear_probe["accuracy_inference_valid"]
                else None
            ),
            "linear_f1": (
                linear_probe["f1_distribution"]
                if linear_probe["num_group_permutations"] == len(fdr_assignments)
                and linear_probe["f1_inference_valid"]
                else None
            ),
            "rbf_accuracy": (
                nonlinear_probe["accuracy_distribution"]
                if nonlinear_probe["num_group_permutations"] == len(fdr_assignments)
                and nonlinear_probe["accuracy_inference_valid"]
                else None
            ),
            "rbf_f1": (
                nonlinear_probe["f1_distribution"]
                if nonlinear_probe["num_group_permutations"] == len(fdr_assignments)
                and nonlinear_probe["f1_inference_valid"]
                else None
            ),
            "nuisance_linear_accuracy": (
                nuisance_probe["accuracy_distribution"]
                if nuisance_probe is not None
                and nuisance_probe["num_group_permutations"] == len(fdr_assignments)
                and nuisance_probe["accuracy_inference_valid"]
                else None
            ),
            "nuisance_linear_f1": (
                nuisance_probe["f1_distribution"]
                if nuisance_probe is not None
                and nuisance_probe["num_group_permutations"] == len(fdr_assignments)
                and nuisance_probe["f1_inference_valid"]
                else None
            ),
            "nuisance_rbf_accuracy": (
                nuisance_nonlinear_probe["accuracy_distribution"]
                if nuisance_nonlinear_probe is not None
                and nuisance_nonlinear_probe["num_group_permutations"]
                == len(fdr_assignments)
                and nuisance_nonlinear_probe["accuracy_inference_valid"]
                else None
            ),
            "nuisance_rbf_f1": (
                nuisance_nonlinear_probe["f1_distribution"]
                if nuisance_nonlinear_probe is not None
                and nuisance_nonlinear_probe["num_group_permutations"]
                == len(fdr_assignments)
                and nuisance_nonlinear_probe["f1_inference_valid"]
                else None
            ),
        }
        return row, None
    except Exception as error:
        return (
            None,
            {
                "model": model_name,
                "phase": row_phase,
                "trajectory_position": trajectory_position,
                "layer": layer_index,
                "stream": stream,
                "module": module_name,
                "error": f"{type(error).__name__}: {error}",
            },
        )


def _analyze_process_index(index: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Process-pool entry point using fork-shared, read-only artifact records."""

    with threadpool_limits(limits=1):
        return _analyze_record_group(_PROCESS_WORK_ITEMS[index], **_PROCESS_ANALYSIS_KWARGS)


def analyze_artifacts(
    artifact_paths: Iterable[Path],
    *,
    knowledge_tasks: set[str],
    reasoning_tasks: set[str],
    phase: str = "prefill",
    cv_folds: int = 5,
    permutations: int = 251,
    fdr_permutations: int | None = None,
    nonlinear_permutations: int = 0,
    pca_variance: float = 0.95,
    silhouette_max_samples: int = 2000,
    seed: int = 0,
    jobs: int = 1,
    parallel_backend: str = "process",
    analysis_backend: str = "sklearn",
    nuisance_table: Path | None = None,
    nuisance_columns: tuple[str, ...] = (),
    nuisance_alpha: float = 1.0,
    reference_positive_tasks: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    overlap = knowledge_tasks & reasoning_tasks
    if overlap:
        raise ValueError(f"Tasks cannot belong to both classes: {sorted(overlap)}")
    if not knowledge_tasks or not reasoning_tasks:
        raise ValueError("Both knowledge_tasks and reasoning_tasks must be non-empty")
    if phase not in {"prefill", "decode", "both"}:
        raise ValueError("phase must be 'prefill', 'decode', or 'both'")
    if jobs < 1:
        raise ValueError("jobs must be at least 1")
    if parallel_backend not in {"process", "thread"}:
        raise ValueError("parallel_backend must be 'process' or 'thread'")
    if analysis_backend not in {"sklearn", "cuml"}:
        raise ValueError("analysis_backend must be 'sklearn' or 'cuml'")
    if (nuisance_table is None) != (not nuisance_columns):
        raise ValueError("nuisance_table and nuisance_columns must be provided together")
    if nuisance_alpha <= 0.0:
        raise ValueError("nuisance_alpha must be positive")
    if fdr_permutations is None:
        fdr_permutations = permutations
    if permutations < 0 or fdr_permutations < 0 or nonlinear_permutations < 0:
        raise ValueError("permutation counts must be non-negative")
    parallel_cuml = analysis_backend == "cuml" and jobs != 1
    if parallel_cuml:
        if parallel_backend != "process":
            raise ValueError("parallel cuML analysis requires parallel_backend='process'")
        if "fork" not in multiprocessing.get_all_start_methods():
            raise ValueError("parallel cuML analysis requires the multiprocessing 'fork' start method")
        mps_pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
        if not mps_pipe or not Path(mps_pipe).is_dir():
            raise ValueError(
                "parallel cuML analysis requires an active NVIDIA MPS pipe directory "
                "in CUDA_MPS_PIPE_DIRECTORY"
            )
    if analysis_backend == "cuml" and not parallel_cuml:
        _require_cuml()

    grouped_records: dict[tuple[str, str, str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    artifact_metadata: list[dict[str, Any]] = []
    ignored_tasks: set[str] = set()
    nuisance_lookup: dict[tuple[str, str, str], tuple[float, ...]] = {}
    nuisance_table_sha256 = None
    if nuisance_table is not None:
        nuisance_lookup, nuisance_table_sha256 = _load_nuisance_table(nuisance_table, nuisance_columns)
    used_nuisance_keys: set[tuple[str, str, str]] = set()
    for artifact_index, path in enumerate(artifact_paths):
        artifact = _load_artifact(path)
        metadata = dict(artifact.get("metadata") or {})
        model_name = str(metadata.get("pretrained") or metadata.get("model") or f"artifact_{artifact_index}")
        artifact_metadata.append({"path": str(path), **metadata})
        for record in artifact["records"]:
            task_name = str(record.get("task_name", "unknown"))
            if task_name in knowledge_tasks:
                label = 1
            elif task_name in reasoning_tasks:
                label = 0
            else:
                ignored_tasks.add(task_name)
                continue
            record_phase = str(record.get("phase", "unknown"))
            if phase != "both" and record_phase != phase:
                continue
            enriched = dict(record)
            enriched["label"] = label
            enriched["model_name"] = model_name
            if nuisance_columns:
                nuisance_key = (model_name, task_name, str(record.get("sample_id")))
                if nuisance_key not in nuisance_lookup:
                    raise ValueError(f"No nuisance row for representation identity: {nuisance_key}")
                enriched["nuisance_features"] = nuisance_lookup[nuisance_key]
                used_nuisance_keys.add(nuisance_key)
            key = (
                model_name,
                record_phase,
                (
                    str(record["trajectory_position"])
                    if record_phase == "decode" and record.get("trajectory_position") is not None
                    else f"step_{int(record['decode_step']):04d}"
                    if record_phase == "decode" and record.get("decode_step") is not None
                    else "aggregate"
                ),
                int(record["layer_index"]),
                str(record["stream"]),
                str(record["module_name"]),
            )
            grouped_records[key].append(enriched)

    work_items = sorted(grouped_records.items())
    if not work_items:
        raise ValueError("No representation records matched the requested tasks and phase")
    analysis_kwargs = {
        "cv_folds": cv_folds,
        "permutations": permutations,
        "fdr_permutations": fdr_permutations,
        "nonlinear_permutations": nonlinear_permutations,
        "pca_variance": pca_variance,
        "silhouette_max_samples": silhouette_max_samples,
        "seed": seed,
        "analysis_backend": analysis_backend,
        "nuisance_columns": nuisance_columns,
        "nuisance_alpha": nuisance_alpha,
    }
    analyze_one = lambda item: _analyze_record_group(item, **analysis_kwargs)
    effective_backend = "serial"
    if jobs == 1:
        results = map(analyze_one, work_items)
    elif parallel_backend == "process" and "fork" in multiprocessing.get_all_start_methods():
        effective_backend = "process-fork"
        global _PROCESS_WORK_ITEMS, _PROCESS_ANALYSIS_KWARGS
        _PROCESS_WORK_ITEMS = work_items
        _PROCESS_ANALYSIS_KWARGS = analysis_kwargs
        executor = ProcessPoolExecutor(
            max_workers=min(jobs, len(work_items)),
            mp_context=multiprocessing.get_context("fork"),
        )
        results = executor.map(_analyze_process_index, range(len(work_items)))
    else:
        effective_backend = "thread"
        executor = ThreadPoolExecutor(max_workers=min(jobs, len(work_items)))
        results = executor.map(analyze_one, work_items)
    try:
        analyzed = list(results)
    finally:
        if jobs != 1:
            executor.shutdown(wait=True)
        _PROCESS_WORK_ITEMS = []
        _PROCESS_ANALYSIS_KWARGS = {}
    rows = [row for row, _failure in analyzed if row is not None]
    failures = [failure for _row, failure in analyzed if failure is not None]

    permutation_payloads_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    permutation_null_rows: list[dict[str, Any]] = []
    for row in rows:
        payload = row.pop("_permutation_payload")
        model_name = str(row["model"])
        permutation_payloads_by_model[model_name].append(payload)
        permutation_null_rows.append(
            {
                "model": model_name,
                "phase": row["phase"],
                "trajectory_position": row["trajectory_position"],
                "layer": row["layer"],
                "stream": row["stream"],
                "module": row["module"],
                **payload,
            }
        )

    rows_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        rows_by_model[str(row["model"])].append(row)
    failed_hypotheses_by_model = Counter(str(failure["model"]) for failure in failures)

    # Screening every layer/module creates a multiple-testing family. Preserve
    # the raw permutation p-values and add false-discovery-rate correction
    # within each model. Different architectures are never one test family.
    for p_column, q_column in (
        ("fdr_p", "fdr_q"),
        ("probe_accuracy_p", "probe_accuracy_q"),
        ("probe_f1_p", "probe_f1_q"),
        ("rbf_probe_accuracy_p", "rbf_probe_accuracy_q"),
        ("rbf_probe_f1_p", "rbf_probe_f1_q"),
        ("nuisance_probe_accuracy_p", "nuisance_probe_accuracy_q"),
        ("nuisance_probe_f1_p", "nuisance_probe_f1_q"),
        ("nuisance_rbf_probe_accuracy_p", "nuisance_rbf_probe_accuracy_q"),
        ("nuisance_rbf_probe_f1_p", "nuisance_rbf_probe_f1_q"),
    ):
        for model_rows in rows_by_model.values():
            if not model_rows or p_column not in model_rows[0]:
                continue
            model_name = str(model_rows[0]["model"])
            declared_family_size = len(model_rows) + int(
                failed_hypotheses_by_model.get(model_name, 0)
            )
            adjusted = benjamini_hochberg(
                (row[p_column] for row in model_rows),
                family_size=declared_family_size,
            )
            for row, q_value in zip(model_rows, adjusted):
                row[q_column] = q_value

    maxima = _metric_extrema(rows)
    maxima_by_model = {model_name: _metric_extrema(model_rows) for model_name, model_rows in rows_by_model.items()}
    randomization_by_model = {
        model_name: _model_randomization_summary(
            model_rows,
            permutation_payloads_by_model[model_name],
            reference_positive_tasks=reference_positive_tasks,
        )
        for model_name, model_rows in rows_by_model.items()
    }
    branch_rows = [row for row in rows if row["stream"] in {"attention_delta", "mlp_delta"}]
    branch_maxima = _metric_extrema(branch_rows)
    branch_maxima_by_model = {
        model_name: _metric_extrema(
            [row for row in model_rows if row["stream"] in {"attention_delta", "mlp_delta"}]
        )
        for model_name, model_rows in rows_by_model.items()
    }
    branch_fdr_by_stream = {
        stream: _metric_extrema([row for row in rows if row["stream"] == stream]).get("fdr")
        for stream in ("attention_delta", "mlp_delta")
    }
    branch_fdr_by_model_and_stream = {
        model_name: {
            stream: _metric_extrema([row for row in model_rows if row["stream"] == stream]).get("fdr")
            for stream in ("attention_delta", "mlp_delta")
        }
        for model_name, model_rows in rows_by_model.items()
    }

    report = {
        "schema_version": 3,
        "objective": "Quantitative task-group separation of pooled intermediate model states",
        "class_labels": {"0": "capability_first_reasoning", "1": "knowledge_heavy"},
        "knowledge_tasks": sorted(knowledge_tasks),
        "reasoning_tasks": sorted(reasoning_tasks),
        "reference_positive_tasks": sorted(reference_positive_tasks) if reference_positive_tasks else None,
        "phase": phase,
        "cv_folds_requested": cv_folds,
        "group_permutations_requested": permutations,
        "fdr_group_permutations_requested": fdr_permutations,
        "nonlinear_group_permutations_requested": nonlinear_permutations,
        "pca_retained_variance": pca_variance,
        "seed": seed,
        "grouped_fold_policy": "single seeded global task order, filtered by each candidate task-label assignment",
        "permutation_statistic_deterministic": None,
        "permutation_determinism_note": (
            "Determinism requires a separate replay. Fisher and RBF complement symmetry are checked "
            "from independently evaluated orientations. Linear-probe complements reuse one fitted "
            "orientation because balanced accuracy and macro-F1 are invariant to exchanging binary "
            "class names; their equality is therefore imposed rather than an independent numerical check."
        ),
        "parallel_jobs": jobs,
        "parallel_backend_requested": parallel_backend,
        "parallel_backend_effective": effective_backend,
        "analysis_backend": analysis_backend,
        "analysis_backend_version": rows[0]["analysis_backend_version"] if rows else None,
        "analysis_feature_dtype": rows[0]["analysis_feature_dtype"] if rows else None,
        "probe_feature_space": rows[0]["probe_feature_space"] if rows else None,
        "probe_pca_fit_scope": rows[0]["probe_pca_fit_scope"] if rows else None,
        "nuisance_control": {
            "enabled": bool(nuisance_columns),
            "table": str(nuisance_table) if nuisance_table is not None else None,
            "table_sha256": nuisance_table_sha256,
            "columns": list(nuisance_columns),
            "alpha": float(nuisance_alpha) if nuisance_columns else None,
            "fit_scope": "training_tasks_per_fold" if nuisance_columns else None,
            "matched_sample_rows": len(used_nuisance_keys),
            "unused_table_rows": len(nuisance_lookup) - len(used_nuisance_keys),
        },
        "artifacts": artifact_metadata,
        "ignored_tasks": sorted(ignored_tasks),
        "num_rows": len(rows),
        "failures": failures,
        "multiplicity_family_size_by_model": {
            model_name: len(model_rows) + int(failed_hypotheses_by_model.get(model_name, 0))
            for model_name, model_rows in rows_by_model.items()
        },
        "maxima": maxima,
        "maxima_by_model": maxima_by_model,
        "primary_maximum_divergence": maxima.get("fdr"),
        "primary_maximum_divergence_by_model": {
            model_name: model_maxima.get("fdr") for model_name, model_maxima in maxima_by_model.items()
        },
        "primary_branch_maximum_divergence": branch_maxima.get("fdr"),
        "primary_branch_maximum_divergence_by_model": {
            model_name: model_maxima.get("fdr") for model_name, model_maxima in branch_maxima_by_model.items()
        },
        "primary_branch_maximum_divergence_by_stream": branch_fdr_by_stream,
        "primary_branch_maximum_divergence_by_model_and_stream": branch_fdr_by_model_and_stream,
        "selection_aware_randomization_by_model": randomization_by_model,
        "permutation_null_artifact": {
            "filename": "permutation_nulls.json.gz",
            "schema_version": 1,
            "rows": len(permutation_null_rows),
            "contains_prompts_or_activations": False,
        },
        "interpretation": {
            "probe_p_values": "Task-label permutation p-values; p < 0.01 is evidence against this null, not proof of a universal separation law.",
            "probe_q_values": "Benjamini-Hochberg adjusted values use the complete declared layer/module family within each model. Failed rows remain missing but still count toward the family size.",
            "linear_probe_regularization": "cuML logistic regression uses C=1, L2, and penalty_normalized=true. This frozen cuML objective is not numerically equivalent to sklearn C=1; objective scaling and effective regularization are library- and sample-size-dependent.",
            "fdr_p_values": "Fisher task-label permutation values compare the declared partition with balanced partitions of the same complete tasks. The model-level maximum test includes layer/stream selection.",
            "cka": "Ordinary linear CKA is reported only for explicitly paired rows sharing pair_id; it is NaN for unpaired benchmark sets.",
            "covariance_alignment": "Order-invariant normalized feature-covariance alignment for unpaired task sets; lower means more divergent second-order geometry.",
            "nonlinear": "The RBF probe is the direct nonlinear-separability test. CKA and silhouette are geometry descriptors, not classifiers.",
            "rank": "Effective and stable rank are reported beside FDR to show whether depth-wise separation changes accompany a loss of active dimensions. FDR is not divided by rank.",
            "visualization": "No t-SNE or UMAP value is computed or used as evidence.",
            "nuisance_control": "Ridge residualization is fit on training tasks inside every observed and permuted probe fold. It is a surface-feature sensitivity analysis, not proof that residual signal is reasoning-specific.",
        },
        "_permutation_null_payload": {
            "schema_version": 1,
            "rows": permutation_null_rows,
        },
    }
    return rows, report


def write_analysis(rows: list[dict[str, Any]], report: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "layer_separability.csv"
    report_path = output_dir / "report.json"
    permutation_path = output_dir / "permutation_nulls.json.gz"
    if rows:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        csv_path.write_text("", encoding="utf-8")
    permutation_payload = report.get("_permutation_null_payload")
    public_report = {key: value for key, value in report.items() if not key.startswith("_")}
    outputs = {"csv": csv_path, "report": report_path}
    if permutation_payload is not None:
        with gzip.open(permutation_path, "wt", encoding="utf-8") as handle:
            json.dump(permutation_payload, handle, separators=(",", ":"), default=_json_value)
        digest = hashlib.sha256(permutation_path.read_bytes()).hexdigest()
        public_report["permutation_null_artifact"]["sha256"] = digest
        outputs["permutation_nulls"] = permutation_path
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(public_report, handle, indent=2, default=_json_value)
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Quantify knowledge-versus-reasoning representation separability.")
    parser.add_argument("artifacts", nargs="+", type=Path, help="One or more representations.pt artifacts.")
    parser.add_argument("--knowledge-tasks", required=True, type=str, help="Comma-separated knowledge-heavy task names.")
    parser.add_argument("--reasoning-tasks", required=True, type=str, help="Comma-separated capability-first/reasoning task names.")
    parser.add_argument(
        "--reference-positive-tasks",
        type=str,
        default="",
        help=(
            "Optional comma-separated half of the same tasks defining one illustrative reference partition. "
            "Inference still uses the complete balanced-partition distribution."
        ),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phase", choices=["prefill", "decode", "both"], default="prefill")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--permutations", type=int, default=251)
    parser.add_argument(
        "--fdr-permutations",
        type=int,
        help=(
            "Grouped label permutations for Fisher inference. Defaults to --permutations; "
            "set independently for an exhaustive Fisher null without refitting a probe at every row."
        ),
    )
    parser.add_argument(
        "--nonlinear-permutations",
        type=int,
        default=0,
        help="Optional grouped label permutations for the RBF probe. Default 0 avoids a second expensive refitting loop.",
    )
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--silhouette-max-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--nuisance-table",
        type=Path,
        help="Optional CSV keyed by model, task_name, and sample_id for fold-local Ridge residualization.",
    )
    parser.add_argument(
        "--nuisance-columns",
        type=str,
        default="",
        help="Comma-separated finite numeric columns from --nuisance-table. Both options are required together.",
    )
    parser.add_argument(
        "--nuisance-alpha",
        type=float,
        default=1.0,
        help="Fixed Ridge penalty for the fold-local nuisance model (default: 1.0).",
    )
    parser.add_argument(
        "--analysis-backend",
        choices=("sklearn", "cuml"),
        default="sklearn",
        help="Estimator family. cuML is explicit, GPU-only, float64, and not numerically interchangeable.",
    )
    parser.add_argument("--jobs", type=int, default=1, help="Concurrent layer/stream analysis workers.")
    parser.add_argument(
        "--parallel-backend",
        choices=("process", "thread"),
        default="process",
        help="Use fork-shared processes when available, or shared-memory threads.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows, report = analyze_artifacts(
        args.artifacts,
        knowledge_tasks=_task_set(args.knowledge_tasks),
        reasoning_tasks=_task_set(args.reasoning_tasks),
        phase=args.phase,
        cv_folds=args.cv_folds,
        permutations=args.permutations,
        fdr_permutations=args.fdr_permutations,
        nonlinear_permutations=args.nonlinear_permutations,
        pca_variance=args.pca_variance,
        silhouette_max_samples=args.silhouette_max_samples,
        seed=args.seed,
        jobs=args.jobs,
        parallel_backend=args.parallel_backend,
        analysis_backend=args.analysis_backend,
        nuisance_table=args.nuisance_table,
        nuisance_columns=tuple(item.strip() for item in args.nuisance_columns.split(",") if item.strip()),
        nuisance_alpha=args.nuisance_alpha,
        reference_positive_tasks=(
            _task_set(args.reference_positive_tasks) if args.reference_positive_tasks.strip() else None
        ),
    )
    outputs = write_analysis(rows, report, args.output_dir)
    print(json.dumps({key: str(value) for key, value in outputs.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
