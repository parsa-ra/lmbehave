"""Synthetic detection calibration for the lexical-adjusted task-held-out probe.

The calibration uses an arbitrary alphabetical 8/8 task partition rather than the study's
knowledge/reasoning labels.  At one fixed representation site, it adds a seeded unit-vector
shift aligned with that calibration partition, then evaluates the complete 12,870-assignment
task-level null with assignment-specific balanced folds.  Character-TF--IDF features and
the activation nuisance map are fitted on training tasks only for every distinct fold.  This
measures what standardized mean shift the adjusted fixed-site probe can detect; it is not
evidence about a model mechanism and it does not calibrate layer selection.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from lmms_eval.representations.analysis import (
    EPSILON,
    _complement_symmetry,
    _probe_predictions,
    _require_cuml,
    grouped_stratified_folds,
)
from lmms_eval.representations.matched_excess import (
    _fixed_pca,
    _linear_predictions,
    _scale_fold,
)
from lmms_eval.representations.selected_site_probe import (
    _complement_representatives,
    _configuration_digest,
    _features_and_groups,
    _fit_fold_local_lexical_nuisance,
    _implementation_provenance,
    _load_prompt_lookup,
    _load_selected_records,
    _prepare_lexical_fold_cache,
    _sha256,
    selection_plan,
)


_LEXICAL_WORKER_PROMPTS: tuple[str, ...] | None = None
_LEXICAL_WORKER_COMPONENTS: int | None = None
_LEXICAL_WORKER_SEED: int | None = None


def _initialize_lexical_worker(
    prompts: tuple[str, ...],
    components: int,
    seed: int,
) -> None:
    global _LEXICAL_WORKER_PROMPTS, _LEXICAL_WORKER_COMPONENTS, _LEXICAL_WORKER_SEED
    _LEXICAL_WORKER_PROMPTS = prompts
    _LEXICAL_WORKER_COMPONENTS = int(components)
    _LEXICAL_WORKER_SEED = int(seed)


def _fit_lexical_fold_worker(
    item: tuple[bytes, tuple[np.ndarray, np.ndarray]],
) -> tuple[bytes, tuple[np.ndarray, np.ndarray], int]:
    if (
        _LEXICAL_WORKER_PROMPTS is None
        or _LEXICAL_WORKER_COMPONENTS is None
        or _LEXICAL_WORKER_SEED is None
    ):
        raise RuntimeError("Lexical-cache worker was not initialized")
    key, (train_indices, test_indices) = item
    train_features, test_features, actual = _fit_fold_local_lexical_nuisance(
        [_LEXICAL_WORKER_PROMPTS[index] for index in train_indices],
        [_LEXICAL_WORKER_PROMPTS[index] for index in test_indices],
        kind="char_tfidf",
        components=_LEXICAL_WORKER_COMPONENTS,
        seed=_LEXICAL_WORKER_SEED,
    )
    return key, (train_features, test_features), actual


def _lexical_cache_digest(
    *,
    assignment_ids: list[str],
    task_order: list[str],
    groups: np.ndarray,
    prompts: np.ndarray,
    requested_splits: int,
    components: int,
    seed: int,
) -> str:
    """Bind a shared lexical cache to every input that determines its folds."""

    prompt_hash = hashlib.sha256()
    for group, prompt in zip(groups, prompts, strict=True):
        prompt_hash.update(str(group).encode("utf-8"))
        prompt_hash.update(b"\0")
        prompt_hash.update(str(prompt).encode("utf-8"))
        prompt_hash.update(b"\n")
    return _configuration_digest(
        {
            "assignment_ids_sha256": hashlib.sha256(
                "\n".join(assignment_ids).encode("utf-8")
            ).hexdigest(),
            "task_order": task_order,
            "prompt_rows_sha256": prompt_hash.hexdigest(),
            "requested_splits": int(requested_splits),
            "kind": "char_tfidf",
            "components": int(components),
            "seed": int(seed),
        }
    )


def _required_lexical_folds(
    *,
    assignment_ids: list[str],
    task_order: list[str],
    groups: np.ndarray,
    requested_splits: int,
    seed: int,
) -> dict[bytes, tuple[np.ndarray, np.ndarray]]:
    """Enumerate each distinct grouped fold once over the exact assignment support."""

    representatives, _ = _complement_representatives(assignment_ids)
    task_index = {task: index for index, task in enumerate(task_order)}
    group_task_indices = np.asarray([task_index[str(group)] for group in groups], dtype=np.int64)
    folds_by_key: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    for representative in representatives:
        labels = np.asarray(
            [int(bit) for bit in assignment_ids[representative]], dtype=np.int64
        )[group_task_indices]
        folds, _ = grouped_stratified_folds(
            labels,
            groups,
            requested_splits=requested_splits,
            seed=seed,
        )
        for train_indices, test_indices in folds:
            folds_by_key.setdefault(test_indices.tobytes(), (train_indices, test_indices))
    return folds_by_key


def _write_lexical_cache(
    path: Path,
    *,
    cache_digest: str,
    cache: dict[bytes, tuple[np.ndarray, np.ndarray]],
) -> None:
    ordered = sorted(cache.items(), key=lambda item: item[0])
    metadata = {
        "schema_version": 1,
        "cache_digest": cache_digest,
        "folds": len(ordered),
        "keys_hex": [key.hex() for key, _ in ordered],
    }
    arrays: dict[str, np.ndarray] = {
        "metadata": np.asarray(json.dumps(metadata, sort_keys=True)),
    }
    for index, (_key, (train_features, test_features)) in enumerate(ordered):
        arrays[f"train_{index:04d}"] = np.asarray(train_features, dtype=np.float64)
        arrays[f"test_{index:04d}"] = np.asarray(test_features, dtype=np.float64)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
    os.replace(temporary, path)


def _read_lexical_cache(
    path: Path,
    *,
    expected_digest: str,
) -> dict[bytes, tuple[np.ndarray, np.ndarray]]:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        if metadata.get("schema_version") != 1:
            raise ValueError(f"Unsupported lexical-cache schema in {path}")
        if metadata.get("cache_digest") != expected_digest:
            raise ValueError(f"Shared lexical cache does not match this analysis: {path}")
        keys = [bytes.fromhex(value) for value in metadata["keys_hex"]]
        if len(keys) != int(metadata["folds"]):
            raise ValueError(f"Malformed lexical-cache metadata in {path}")
        cache = {
            key: (
                np.asarray(payload[f"train_{index:04d}"], dtype=np.float64),
                np.asarray(payload[f"test_{index:04d}"], dtype=np.float64),
            )
            for index, key in enumerate(keys)
        }
    return cache


def _load_or_build_shared_lexical_cache(
    path: Path,
    *,
    cache_digest: str,
    assignment_ids: list[str],
    task_order: list[str],
    groups: np.ndarray,
    prompts: np.ndarray,
    requested_splits: int,
    components: int,
    seed: int,
    workers: int,
) -> dict[bytes, tuple[np.ndarray, np.ndarray]]:
    """Build one cache under a file lock, then share it across calibration workers."""

    if workers < 1:
        raise ValueError("lexical_cache_workers must be positive")
    lock_path = path.with_suffix(path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if path.exists():
            return _read_lexical_cache(path, expected_digest=cache_digest)
        folds_by_key = _required_lexical_folds(
            assignment_ids=assignment_ids,
            task_order=task_order,
            groups=groups,
            requested_splits=requested_splits,
            seed=seed,
        )
        ordered = sorted(folds_by_key.items(), key=lambda item: item[0])
        cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
        component_counts: set[int] = set()
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialize_lexical_worker,
            initargs=(tuple(str(prompt) for prompt in prompts), components, seed),
        ) as executor:
            for key, value, actual in executor.map(_fit_lexical_fold_worker, ordered):
                cache[key] = value
                component_counts.add(int(actual))
        if component_counts != {int(components)}:
            raise ValueError(
                "The requested lexical width is unavailable in every calibration fold: "
                f"{component_counts}"
            )
        _write_lexical_cache(path, cache_digest=cache_digest, cache=cache)
        return cache


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def fixed_site_probe_distribution(
    features: Any,
    groups: np.ndarray,
    *,
    assignment_ids: list[str],
    task_order: list[str],
    requested_splits: int,
    components: int,
    seed: int,
    analysis_backend: str,
) -> dict[str, Any]:
    """Evaluate one fixed activation site on the complete class-symmetric support."""

    representatives, representative_for = _complement_representatives(assignment_ids)
    assignment_index = {assignment: index for index, assignment in enumerate(assignment_ids)}
    complement_of = {
        index: assignment_index["".join("1" if bit == "0" else "0" for bit in assignment)]
        for index, assignment in enumerate(assignment_ids)
    }
    task_index = {task: index for index, task in enumerate(task_order)}
    group_task_indices = np.asarray([task_index[str(group)] for group in groups], dtype=np.int64)
    accuracy: list[float | None] = [None] * len(assignment_ids)
    macro_f1: list[float | None] = [None] * len(assignment_ids)
    fold_cache: dict[bytes, tuple[Any, Any, Any, Any, int]] = {}
    for representative in representatives:
        labels = np.asarray([int(bit) for bit in assignment_ids[representative]], dtype=np.int64)[
            group_task_indices
        ]
        folds, _ = grouped_stratified_folds(
            labels,
            groups,
            requested_splits=requested_splits,
            seed=seed,
        )
        predictions = np.full(labels.shape, -1, dtype=np.int64)
        for train_indices, test_indices in folds:
            cache_key = test_indices.tobytes()
            cached = fold_cache.get(cache_key)
            if cached is None:
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
                cached = (train_backend, test_backend, train_features, test_features, actual)
                fold_cache[cache_key] = cached
            _train_backend, _test_backend, train_features, test_features, actual = cached
            if int(actual) != int(components):
                raise ValueError("Calibration feature width is not constant across folds")
            predictions[test_indices] = _linear_predictions(
                train_features,
                test_features,
                labels[train_indices],
                analysis_backend=analysis_backend,
                seed=seed,
            )
        if np.any(predictions < 0):
            raise RuntimeError("Calibration probe did not predict every example")
        score_accuracy = float(accuracy_score(labels, predictions))
        score_f1 = float(f1_score(labels, predictions, average="macro", zero_division=0))
        complement = complement_of[representative]
        accuracy[representative] = score_accuracy
        accuracy[complement] = score_accuracy
        macro_f1[representative] = score_f1
        macro_f1[complement] = score_f1
    if any(value is None for value in accuracy + macro_f1):
        raise RuntimeError("Calibration distribution is incomplete")
    accuracy_values = [float(value) for value in accuracy]
    f1_values = [float(value) for value in macro_f1]
    return {
        "accuracy": accuracy_values,
        "macro_f1": f1_values,
        "accuracy_complement_symmetry": _complement_symmetry(accuracy_values, assignment_ids),
        "macro_f1_complement_symmetry": _complement_symmetry(f1_values, assignment_ids),
        "unique_task_folds": len(fold_cache),
        "probe_refits": len(representatives),
    }


def lexical_adjusted_probe_distribution(
    features: Any,
    groups: np.ndarray,
    *,
    assignment_ids: list[str],
    task_order: list[str],
    requested_splits: int,
    seed: int,
    analysis_backend: str,
    lexical_fold_cache: dict[bytes, tuple[np.ndarray, np.ndarray]],
    nuisance_alpha: float,
    pca_variance: float,
) -> dict[str, Any]:
    """Evaluate the full fold-local lexical-adjusted fixed-site probe."""

    representatives, _representative_for = _complement_representatives(assignment_ids)
    assignment_index = {assignment: index for index, assignment in enumerate(assignment_ids)}
    complement_of = {
        index: assignment_index["".join("1" if bit == "0" else "0" for bit in assignment)]
        for index, assignment in enumerate(assignment_ids)
    }
    task_index = {task: index for index, task in enumerate(task_order)}
    group_task_indices = np.asarray([task_index[str(group)] for group in groups], dtype=np.int64)
    accuracy: list[float | None] = [None] * len(assignment_ids)
    macro_f1: list[float | None] = [None] * len(assignment_ids)
    fold_cache: dict[bytes, tuple[Any, Any, Any, Any]] = {}
    for representative in representatives:
        labels = np.asarray([int(bit) for bit in assignment_ids[representative]], dtype=np.int64)[
            group_task_indices
        ]
        score_accuracy, score_f1, _effective_splits = _probe_predictions(
            features,
            labels,
            groups,
            kind="linear",
            requested_splits=requested_splits,
            seed=seed,
            analysis_backend=analysis_backend,
            fold_pca_variance=pca_variance,
            nuisance_alpha=nuisance_alpha,
            fold_cache=fold_cache,
            fold_nuisance_cache=lexical_fold_cache,
        )
        complement = complement_of[representative]
        accuracy[representative] = float(score_accuracy)
        accuracy[complement] = float(score_accuracy)
        macro_f1[representative] = float(score_f1)
        macro_f1[complement] = float(score_f1)
    if any(value is None for value in accuracy + macro_f1):
        raise RuntimeError("Adjusted calibration distribution is incomplete")
    accuracy_values = [float(value) for value in accuracy]
    f1_values = [float(value) for value in macro_f1]
    return {
        "accuracy": accuracy_values,
        "macro_f1": f1_values,
        "accuracy_complement_symmetry": _complement_symmetry(accuracy_values, assignment_ids),
        "macro_f1_complement_symmetry": _complement_symmetry(f1_values, assignment_ids),
        "unique_task_folds": len(fold_cache),
        "probe_refits": len(representatives),
    }


def _calibration_assignment(task_order: list[str]) -> str:
    midpoint = len(task_order) // 2
    if midpoint * 2 != len(task_order):
        raise ValueError("Alphabetical calibration requires an even number of tasks")
    return "".join("1" if index < midpoint else "0" for index in range(len(task_order)))


def _injected_features(
    features: Any,
    sample_labels: np.ndarray,
    *,
    direction_seed: int,
    standardized_half_shift: float,
    analysis_backend: str,
) -> tuple[Any, float]:
    width = int(features.shape[1])
    rng = np.random.default_rng(direction_seed)
    direction_cpu = rng.normal(size=width).astype(np.float64)
    direction_cpu /= np.linalg.norm(direction_cpu)
    if analysis_backend == "cuml":
        cp, *_ = _require_cuml()
        direction = cp.asarray(direction_cpu)
        signed_labels = cp.asarray(2 * sample_labels - 1, dtype=cp.float64)
        projection_scale = float(cp.std(features @ direction, ddof=1).item())
        injected = features + (
            signed_labels[:, None]
            * float(standardized_half_shift)
            * projection_scale
            * direction[None, :]
        )
    else:
        direction = direction_cpu
        signed_labels = (2 * sample_labels - 1).astype(np.float64)
        projection_scale = float(np.std(features @ direction, ddof=1))
        injected = features + (
            signed_labels[:, None]
            * float(standardized_half_shift)
            * projection_scale
            * direction[None, :]
        )
    if not np.isfinite(projection_scale) or projection_scale <= EPSILON:
        raise ValueError("Calibration direction has degenerate activation variance")
    return injected, projection_scale


def run_probe_calibration(
    artifact_paths: list[Path],
    *,
    permutation_path: Path,
    sample_root: Path,
    output_dir: Path,
    requested_splits: int = 5,
    lexical_components: int = 64,
    nuisance_alpha: float = 1.0,
    pca_variance: float = 0.95,
    direction_seeds: tuple[int, ...] = tuple(range(10)),
    standardized_half_shifts: tuple[float, ...] = (0.0, 0.25, 0.5, 1.0, 2.0),
    threshold: float = 0.01,
    seed: int = 0,
    analysis_backend: str = "cuml",
    lexical_cache_path: Path | None = None,
    lexical_cache_workers: int = 1,
) -> dict[str, Any]:
    if not direction_seeds or not standardized_half_shifts:
        raise ValueError("Calibration requires at least one direction and shift")
    if any(value < 0.0 for value in standardized_half_shifts):
        raise ValueError("Calibration shifts must be nonnegative")
    if lexical_components < 1 or nuisance_alpha <= 0.0:
        raise ValueError("Lexical components and nuisance alpha must be positive")
    if not 0.0 < pca_variance <= 1.0:
        raise ValueError("pca_variance must be in (0, 1]")
    started = time.monotonic()
    with gzip.open(permutation_path, "rt", encoding="utf-8") as handle:
        permutation_rows = list(json.load(handle)["rows"])
    assignment_ids, task_order, selected_rows, _selected_by_key = selection_plan(permutation_rows)
    calibration_assignment = _calibration_assignment(task_order)
    try:
        calibration_index = assignment_ids.index(calibration_assignment)
    except ValueError as error:
        raise ValueError("Alphabetical calibration assignment is absent from the null support") from error
    selected_row = permutation_rows[int(selected_rows[0])]
    selected_key = (
        str(selected_row["phase"]),
        str(selected_row.get("trajectory_position", "aggregate")),
        int(selected_row["layer"]),
        str(selected_row["stream"]),
        str(selected_row["module"]),
    )
    grouped_records, artifacts = _load_selected_records(
        artifact_paths,
        selected_keys={selected_key},
        task_order=task_order,
    )
    model_names = {str(item["model"]) for item in artifacts}
    if len(model_names) != 1:
        raise ValueError("Probe calibration requires one model")
    model_name = next(iter(model_names))
    required_identities = {
        (str(record["task_name"]), str(record.get("sample_id")))
        for record in grouped_records[selected_key]
    }
    prompt_lookup, prompt_sources = _load_prompt_lookup(
        sample_root,
        required_identities=required_identities,
    )
    features, groups, _nuisance, prompts, _identities = _features_and_groups(
        grouped_records[selected_key],
        task_order,
        analysis_backend,
        model_name=model_name,
        nuisance_lookup=None,
        lexical_prompt_lookup=prompt_lookup,
    )
    if prompts is None:
        raise RuntimeError("Probe calibration did not load prompt text")
    task_index = {task: index for index, task in enumerate(task_order)}
    group_task_indices = np.asarray([task_index[str(group)] for group in groups], dtype=np.int64)
    calibration_labels = np.asarray(
        [int(bit) for bit in calibration_assignment], dtype=np.int64
    )[group_task_indices]
    lexical_cache_digest = _lexical_cache_digest(
        assignment_ids=assignment_ids,
        task_order=task_order,
        groups=groups,
        prompts=prompts,
        requested_splits=requested_splits,
        components=lexical_components,
        seed=seed,
    )
    if lexical_cache_path is not None:
        lexical_fold_cache = _load_or_build_shared_lexical_cache(
            lexical_cache_path,
            cache_digest=lexical_cache_digest,
            assignment_ids=assignment_ids,
            task_order=task_order,
            groups=groups,
            prompts=prompts,
            requested_splits=requested_splits,
            components=lexical_components,
            seed=seed,
            workers=lexical_cache_workers,
        )
    else:
        representatives, _representative_for = _complement_representatives(assignment_ids)
        lexical_fold_cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
        lexical_component_counts: set[int] = set()
        for representative in representatives:
            labels = np.asarray(
                [int(bit) for bit in assignment_ids[representative]], dtype=np.int64
            )[group_task_indices]
            _prepare_lexical_fold_cache(
                labels,
                groups,
                prompts,
                requested_splits=requested_splits,
                seed=seed,
                kind="char_tfidf",
                components=lexical_components,
                cache=lexical_fold_cache,
                component_counts=lexical_component_counts,
            )
        if lexical_component_counts != {int(lexical_components)}:
            raise ValueError(
                "The requested lexical width is unavailable in every calibration fold: "
                f"{lexical_component_counts}"
            )
    del prompts

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.json"
    distribution_dir = output_dir / "distributions"
    distribution_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "report.json"
    conditions = [
        (direction_seed, shift)
        for shift in standardized_half_shifts
        for direction_seed in ((direction_seeds[0],) if shift == 0.0 else direction_seeds)
    ]
    configuration = {
        "implementation": _implementation_provenance(analysis_backend),
        "artifacts": artifacts,
        "permutation_nulls": {"path": str(permutation_path), "sha256": _sha256(permutation_path)},
        "sample_root": str(sample_root),
        "prompt_sources": prompt_sources,
        "assignment_ids_sha256": hashlib.sha256("\n".join(assignment_ids).encode()).hexdigest(),
        "calibration_assignment": calibration_assignment,
        "calibration_assignment_index": calibration_index,
        "calibration_policy": "alphabetically first half of fixed task order",
        "requested_splits": int(requested_splits),
        "lexical_components": int(lexical_components),
        "nuisance_alpha": float(nuisance_alpha),
        "pca_variance": float(pca_variance),
        "direction_seeds": list(direction_seeds),
        "standardized_half_shifts": list(standardized_half_shifts),
        "threshold": float(threshold),
        "seed": int(seed),
        "analysis_backend": analysis_backend,
        "lexical_cache": {
            "path": str(lexical_cache_path) if lexical_cache_path is not None else None,
            "cache_digest": lexical_cache_digest,
            "folds": len(lexical_fold_cache),
            "workers": int(lexical_cache_workers),
        },
        "injection_scale": (
            "global sample standard deviation of the unadjusted activation projected on "
            "the seeded unit direction"
        ),
    }
    configuration_sha256 = _configuration_digest(configuration)
    completed: dict[str, dict[str, Any]] = {}
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("configuration_sha256") != configuration_sha256:
            raise ValueError("Existing calibration progress belongs to another configuration")
        completed = dict(progress.get("conditions", {}))

    for direction_seed, shift in conditions:
        key = f"seed_{direction_seed:04d}_shift_{shift:.6f}"
        if key in completed:
            continue
        injected, projection_scale = _injected_features(
            features,
            calibration_labels,
            direction_seed=direction_seed,
            standardized_half_shift=shift,
            analysis_backend=analysis_backend,
        )
        distribution = lexical_adjusted_probe_distribution(
            injected,
            groups,
            assignment_ids=assignment_ids,
            task_order=task_order,
            requested_splits=requested_splits,
            seed=seed,
            analysis_backend=analysis_backend,
            lexical_fold_cache=lexical_fold_cache,
            nuisance_alpha=nuisance_alpha,
            pca_variance=pca_variance,
        )
        f1_values = np.asarray(distribution["macro_f1"], dtype=np.float64)
        accuracy_values = np.asarray(distribution["accuracy"], dtype=np.float64)
        f1_p = float(np.mean(f1_values >= f1_values[calibration_index] - EPSILON))
        accuracy_p = float(
            np.mean(accuracy_values >= accuracy_values[calibration_index] - EPSILON)
        )
        distribution_path = distribution_dir / f"{key}.json.gz"
        with gzip.open(distribution_path, "wt", encoding="utf-8") as handle:
            json.dump(
                {
                    "schema_version": 1,
                    "assignment_ids": assignment_ids,
                    **distribution,
                },
                handle,
                separators=(",", ":"),
            )
        completed[key] = {
            "direction_seed": int(direction_seed),
            "standardized_half_shift": float(shift),
            "standardized_mean_difference": 2.0 * float(shift),
            "projection_scale": projection_scale,
            "macro_f1": float(f1_values[calibration_index]),
            "macro_f1_p": f1_p,
            "accuracy": float(accuracy_values[calibration_index]),
            "accuracy_p": accuracy_p,
            "detected_at_threshold": bool(f1_p < threshold),
            "distribution": str(distribution_path),
            "distribution_sha256": _sha256(distribution_path),
            "unique_task_folds": int(distribution["unique_task_folds"]),
            "probe_refits": int(distribution["probe_refits"]),
        }
        _atomic_json(
            progress_path,
            {
                "schema_version": 1,
                "configuration_sha256": configuration_sha256,
                "conditions": completed,
            },
        )
        del injected
        if analysis_backend == "cuml":
            cp, *_ = _require_cuml()
            cp.get_default_memory_pool().free_all_blocks()

    rows = [completed[key] for key in sorted(completed)]
    detection_curve = []
    for shift in standardized_half_shifts:
        shift_rows = [row for row in rows if row["standardized_half_shift"] == shift]
        detection_curve.append(
            {
                "standardized_half_shift": float(shift),
                "standardized_mean_difference": 2.0 * float(shift),
                "directions": len(shift_rows),
                "detections": sum(bool(row["detected_at_threshold"]) for row in shift_rows),
                "detection_fraction": float(
                    np.mean([bool(row["detected_at_threshold"]) for row in shift_rows])
                ),
                "macro_f1_mean": float(np.mean([row["macro_f1"] for row in shift_rows])),
                "macro_f1_std": float(
                    np.std([row["macro_f1"] for row in shift_rows], ddof=1)
                    if len(shift_rows) > 1
                    else 0.0
                ),
            }
        )
    report = {
        "schema_version": 1,
        "status": "PASS",
        "objective": (
            "Calibrate lexical-adjusted fixed-site task-held-out probe detection with "
            "injected mean shifts."
        ),
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "model": model_name,
        "fixed_site": {
            "row_index": int(selected_rows[0]),
            "layer": int(selected_row["layer"]),
            "stream": str(selected_row["stream"]),
            "module": str(selected_row["module"]),
        },
        "conditions": rows,
        "detection_curve": detection_curve,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "interpretation": (
            "This is a synthetic calibration of the fold-local lexical-adjusted fixed-site "
            "probe. It is conditional on one model and site, does not include layer selection, "
            "and does not constitute evidence for a knowledge-versus-reasoning mechanism."
        ),
    }
    _atomic_json(report_path, report)
    _atomic_json(
        progress_path,
        {
            "schema_version": 1,
            "configuration_sha256": configuration_sha256,
            "conditions": completed,
            "complete": True,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--permutation-nulls", required=True, type=Path)
    parser.add_argument("--sample-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--lexical-components", type=int, default=64)
    parser.add_argument("--nuisance-alpha", type=float, default=1.0)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--direction-seeds", default="0,1,2,3,4,5,6,7,8,9")
    parser.add_argument("--standardized-half-shifts", default="0,0.25,0.5,1,2")
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--analysis-backend", choices=("sklearn", "cuml"), default="cuml")
    parser.add_argument("--lexical-cache", type=Path)
    parser.add_argument("--lexical-cache-workers", type=int, default=1)
    args = parser.parse_args()
    report = run_probe_calibration(
        [path.resolve() for path in args.artifacts],
        permutation_path=args.permutation_nulls.resolve(),
        sample_root=args.sample_root.resolve(),
        output_dir=args.output_dir.resolve(),
        requested_splits=args.cv_folds,
        lexical_components=args.lexical_components,
        nuisance_alpha=args.nuisance_alpha,
        pca_variance=args.pca_variance,
        direction_seeds=tuple(int(item) for item in args.direction_seeds.split(",") if item),
        standardized_half_shifts=tuple(
            float(item) for item in args.standardized_half_shifts.split(",") if item
        ),
        threshold=args.threshold,
        seed=args.seed,
        analysis_backend=args.analysis_backend,
        lexical_cache_path=args.lexical_cache.resolve() if args.lexical_cache else None,
        lexical_cache_workers=args.lexical_cache_workers,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
