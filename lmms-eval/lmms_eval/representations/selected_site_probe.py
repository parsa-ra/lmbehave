"""Selection-aware grouped linear probing at Fisher-selected representation sites.

The Fisher site must be reselected for every task-label assignment.  This module
then evaluates exactly one task-held-out probe per assignment at that assignment's
selected site.  It avoids the much larger and unnecessary Cartesian product of
every probe site with every task-label assignment.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from lmms_eval.representations.analysis import (
    _complement_symmetry,
    _load_artifact,
    _load_nuisance_table,
    _probe_predictions,
    _randomization_p,
    _require_cuml,
    grouped_stratified_folds,
)

RowKey = tuple[str, str, int, str, str]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _trajectory_position(record: dict[str, Any]) -> str:
    if record.get("phase") == "decode" and record.get("trajectory_position") is not None:
        return str(record["trajectory_position"])
    if record.get("phase") == "decode" and record.get("decode_step") is not None:
        return f"step_{int(record['decode_step']):04d}"
    return "aggregate"


def _row_key(row: dict[str, Any]) -> RowKey:
    return (
        str(row["phase"]),
        str(row.get("trajectory_position", "aggregate")),
        int(row["layer"]),
        str(row["stream"]),
        str(row["module"]),
    )


def _record_key(record: dict[str, Any]) -> RowKey:
    return (
        str(record["phase"]),
        _trajectory_position(record),
        int(record["layer_index"]),
        str(record["stream"]),
        str(record["module_name"]),
    )


def selection_plan(
    permutation_rows: list[dict[str, Any]],
) -> tuple[list[str], list[str], np.ndarray, dict[RowKey, list[int]]]:
    """Return aligned assignments and Fisher-selected row indices.

    The permutation artifact is authoritative for the finite randomization support.
    Every row must share the same ordered task labels and assignment identifiers.
    """

    if not permutation_rows:
        raise ValueError("The permutation artifact contains no rows")
    assignment_ids = list(permutation_rows[0]["assignment_ids"])
    task_order = list(permutation_rows[0]["task_order"])
    if not assignment_ids or not task_order:
        raise ValueError("The permutation artifact has an empty task design")
    for row in permutation_rows:
        if row["assignment_ids"] != assignment_ids or row["task_order"] != task_order:
            raise ValueError("Permutation rows do not share one aligned task-label support")
    matrix = np.asarray([row["fdr"] for row in permutation_rows], dtype=np.float64)
    expected = (len(permutation_rows), len(assignment_ids))
    if matrix.shape != expected or not np.all(np.isfinite(matrix)):
        raise ValueError("The aligned Fisher permutation matrix is incomplete or non-finite")
    selected_rows = np.argmax(matrix, axis=0).astype(np.int64)
    selected_by_key: dict[RowKey, list[int]] = defaultdict(list)
    for assignment_index, row_index in enumerate(selected_rows.tolist()):
        selected_by_key[_row_key(permutation_rows[row_index])].append(assignment_index)
    return assignment_ids, task_order, selected_rows, dict(selected_by_key)


def _validate_complement_selection(assignment_ids: list[str], selected_rows: np.ndarray) -> None:
    index = {assignment_id: position for position, assignment_id in enumerate(assignment_ids)}
    for position, assignment_id in enumerate(assignment_ids):
        complement = "".join("1" if bit == "0" else "0" for bit in assignment_id)
        if complement not in index:
            raise ValueError("The task-label support is not complement-paired")
        if int(selected_rows[position]) != int(selected_rows[index[complement]]):
            raise ValueError("Fisher site selection is not invariant under label complement")


def _complement_representatives(assignment_ids: list[str]) -> tuple[list[int], dict[int, int]]:
    """Return one fitted orientation per class-label-complement pair.

    Balanced accuracy and macro-F1 are invariant to exchanging the two class names.
    The observed assignment (index zero) is retained as its pair's fitted orientation so
    the reported observed score always comes from a direct fit rather than a copied null.
    """

    index = {assignment_id: position for position, assignment_id in enumerate(assignment_ids)}
    if len(index) != len(assignment_ids):
        raise ValueError("The task-label support contains duplicate assignments")
    representatives: list[int] = []
    representative_for: dict[int, int] = {}
    visited: set[int] = set()
    for position, assignment_id in enumerate(assignment_ids):
        if position in visited:
            continue
        complement_id = "".join("1" if bit == "0" else "0" for bit in assignment_id)
        complement_position = index.get(complement_id)
        if complement_position is None:
            raise ValueError("The task-label support is not complement-paired")
        representative = 0 if position == 0 or complement_position == 0 else min(position, complement_position)
        representatives.append(representative)
        representative_for[position] = representative
        representative_for[complement_position] = representative
        visited.update((position, complement_position))
    return representatives, representative_for


def _load_selected_records(
    artifact_paths: Iterable[Path],
    *,
    selected_keys: set[RowKey],
    task_order: list[str],
) -> tuple[dict[RowKey, list[dict[str, Any]]], list[dict[str, Any]]]:
    task_set = set(task_order)
    grouped: dict[RowKey, list[dict[str, Any]]] = defaultdict(list)
    provenance: list[dict[str, Any]] = []
    model_names: set[str] = set()
    for path in artifact_paths:
        artifact = _load_artifact(path)
        metadata = dict(artifact.get("metadata") or {})
        model_name = str(metadata.get("pretrained") or metadata.get("model"))
        model_names.add(model_name)
        provenance.append(
            {
                "path": str(path),
                "sha256": _sha256(path),
                "model": model_name,
                "lmms_eval_git_head": metadata.get("lmms_eval_git_head"),
                "lmms_eval_git_dirty": metadata.get("lmms_eval_git_dirty"),
                "transformers_version": metadata.get("transformers_version"),
            }
        )
        for record in artifact["records"]:
            task = str(record.get("task_name"))
            if task not in task_set:
                continue
            key = _record_key(record)
            if key in selected_keys:
                grouped[key].append(record)
    if len(model_names) != 1:
        raise ValueError(f"Selected-site probing is within-model; found {sorted(model_names)}")
    missing = selected_keys - set(grouped)
    if missing:
        raise ValueError(f"No activation records for selected Fisher rows: {sorted(missing)}")
    return dict(grouped), provenance


def _features_and_groups(
    records: list[dict[str, Any]],
    task_order: list[str],
    analysis_backend: str,
    *,
    model_name: str,
    nuisance_lookup: dict[tuple[str, str, str], tuple[float, ...]] | None,
    lexical_prompt_lookup: dict[tuple[str, str], str] | None,
) -> tuple[Any, np.ndarray, Any | None, np.ndarray | None, tuple[tuple[str, str], ...]]:
    records = sorted(
        records,
        key=lambda row: (str(row["task_name"]), str(row.get("sample_id"))),
    )
    identities = [(str(row["task_name"]), str(row.get("sample_id"))) for row in records]
    if len(identities) != len(set(identities)):
        raise ValueError("A selected row contains duplicate task/sample identities")
    observed_tasks = {task for task, _sample in identities}
    if observed_tasks != set(task_order):
        raise ValueError("A selected row does not contain the complete task set")
    groups = np.asarray([task for task, _sample in identities], dtype=object)
    vectors = np.stack([row["vector"].detach().cpu().numpy() for row in records]).astype(np.float64, copy=False)
    if not np.all(np.isfinite(vectors)):
        raise ValueError("A selected row contains non-finite activations")
    nuisance = None
    if nuisance_lookup is not None:
        missing = [
            (model_name, task, sample_id)
            for task, sample_id in identities
            if (model_name, task, sample_id) not in nuisance_lookup
        ]
        if missing:
            raise ValueError(f"No nuisance values for selected identities: {missing[:3]}")
        nuisance = np.asarray(
            [nuisance_lookup[(model_name, task, sample_id)] for task, sample_id in identities],
            dtype=np.float64,
        )
    lexical_prompts = None
    if lexical_prompt_lookup is not None:
        missing_prompts = [
            (task, sample_id)
            for task, sample_id in identities
            if (task, sample_id) not in lexical_prompt_lookup
        ]
        if missing_prompts:
            raise ValueError(f"No logged prompt for selected identities: {missing_prompts[:3]}")
        lexical_prompts = np.asarray(
            [lexical_prompt_lookup[(task, sample_id)] for task, sample_id in identities],
            dtype=object,
        )
    if analysis_backend == "cuml":
        cp, *_ = _require_cuml()
        return (
            cp.asarray(vectors),
            groups,
            None if nuisance is None else cp.asarray(nuisance),
            lexical_prompts,
            tuple(identities),
        )
    if analysis_backend == "sklearn":
        return vectors, groups, nuisance, lexical_prompts, tuple(identities)
    raise ValueError(f"Unsupported analysis backend: {analysis_backend}")


def _load_prompt_lookup(
    sample_root: Path,
    *,
    required_identities: set[tuple[str, str]],
) -> tuple[dict[tuple[str, str], str], list[dict[str, str]]]:
    """Bind logged pre-template prompts to exact activation sample identities."""

    lookup: dict[tuple[str, str], str] = {}
    provenance: list[dict[str, str]] = []
    tasks = sorted({task for task, _sample_id in required_identities})
    for task in tasks:
        task_candidates = [sample_root / "tasks" / task]
        task_candidates.extend(sorted((sample_root / "tasks").glob(f"*-{task}")))
        paths = sorted(
            path
            for task_root in task_candidates
            if task_root.is_dir()
            for path in (task_root / "evaluation").rglob("*_samples_*.jsonl")
        )
        if len(paths) != 1:
            raise ValueError(f"Expected one sample log for {task}, found {len(paths)}")
        path = paths[0]
        provenance.append({"path": str(path), "sha256": _sha256(path)})
        task_identities = sorted(
            sample_id for identity_task, sample_id in required_identities if identity_task == task
        )
        suffix_index: dict[str, list[str]] = defaultdict(list)
        for sample_id in task_identities:
            suffix_index[sample_id.rsplit(":", 1)[-1]].append(sample_id)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            row = json.loads(line)
            doc_id = str(row.get("doc_id"))
            prompt = row.get("input")
            candidates = suffix_index.get(doc_id, [])
            if not candidates:
                continue
            if len(candidates) != 1 or not isinstance(prompt, str):
                raise ValueError(
                    f"Ambiguous or invalid prompt identity in {path}:{line_number}: {doc_id!r}"
                )
            key = (task, candidates[0])
            if key in lookup:
                raise ValueError(f"Duplicate prompt identity in {path}:{line_number}: {key}")
            lookup[key] = prompt
    missing = required_identities - set(lookup)
    if missing:
        raise ValueError(f"Prompt logs do not cover activation identities: {sorted(missing)[:3]}")
    return lookup, provenance


def _fit_fold_local_lexical_nuisance(
    train_prompts: list[str],
    test_prompts: list[str],
    *,
    kind: str,
    components: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Fit TF--IDF and SVD on training-task prompts only."""

    if kind != "char_tfidf":
        raise ValueError(f"Unsupported lexical nuisance kind: {kind}")
    if components < 1:
        raise ValueError("lexical_nuisance_components must be positive")
    vectorizer = TfidfVectorizer(
        lowercase=True,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
        sublinear_tf=True,
        max_features=100_000,
        dtype=np.float64,
    )
    train_tfidf = vectorizer.fit_transform(train_prompts)
    test_tfidf = vectorizer.transform(test_prompts)
    actual_components = min(
        int(components),
        int(train_tfidf.shape[0]) - 1,
        int(train_tfidf.shape[1]) - 1,
    )
    if actual_components < 1:
        raise ValueError("Fold-local TF--IDF matrix has insufficient rank for truncated SVD")
    svd = TruncatedSVD(
        n_components=actual_components,
        algorithm="randomized",
        n_iter=7,
        random_state=seed,
    )
    train_features = svd.fit_transform(train_tfidf).astype(np.float64, copy=False)
    test_features = svd.transform(test_tfidf).astype(np.float64, copy=False)
    if not np.all(np.isfinite(train_features)) or not np.all(np.isfinite(test_features)):
        raise ValueError("Fold-local lexical nuisance features are non-finite")
    return train_features, test_features, actual_components


def _prepare_lexical_fold_cache(
    labels: np.ndarray,
    groups: np.ndarray,
    prompts: np.ndarray,
    *,
    requested_splits: int,
    seed: int,
    kind: str,
    components: int,
    cache: dict[bytes, tuple[np.ndarray, np.ndarray]],
    component_counts: set[int],
) -> None:
    """Populate only the task folds needed by one label assignment."""

    folds, _ = grouped_stratified_folds(
        labels,
        groups,
        requested_splits=requested_splits,
        seed=seed,
    )
    for train_indices, test_indices in folds:
        cache_key = test_indices.tobytes()
        if cache_key in cache:
            continue
        train_features, test_features, actual_components = (
            _fit_fold_local_lexical_nuisance(
                [str(prompts[index]) for index in train_indices],
                [str(prompts[index]) for index in test_indices],
                kind=kind,
                components=components,
                seed=seed,
            )
        )
        cache[cache_key] = (train_features, test_features)
        component_counts.add(actual_components)


def _configuration_digest(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _implementation_provenance(analysis_backend: str) -> dict[str, Any]:
    module_path = Path(__file__).resolve()
    repository = module_path.parents[2]
    git_head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repository,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=repository,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    ).stdout.strip()
    if analysis_backend == "cuml":
        _cp, cuml, *_ = _require_cuml()
        backend_version = str(cuml.__version__)
    else:
        import sklearn

        backend_version = str(sklearn.__version__)
    return {
        "module": str(module_path),
        "module_sha256": _sha256(module_path),
        "git_head": git_head or None,
        "git_dirty": bool(dirty),
        "python": sys.version,
        "analysis_backend": analysis_backend,
        "analysis_backend_version": backend_version,
    }


def run_selection_aware_probe(
    artifact_paths: list[Path],
    *,
    permutation_path: Path,
    output_dir: Path,
    requested_splits: int,
    pca_variance: float,
    seed: int,
    analysis_backend: str,
    checkpoint_every: int = 25,
    nuisance_table: Path | None = None,
    nuisance_columns: tuple[str, ...] = (),
    nuisance_alpha: float = 1.0,
    lexical_nuisance_sample_root: Path | None = None,
    lexical_nuisance_kind: str = "char_tfidf",
    lexical_nuisance_components: int = 64,
) -> dict[str, Any]:
    if checkpoint_every < 1:
        raise ValueError("checkpoint_every must be positive")
    with gzip.open(permutation_path, "rt", encoding="utf-8") as handle:
        permutation_payload = json.load(handle)
    permutation_rows = list(permutation_payload["rows"])
    permutation_models = {str(row.get("model")) for row in permutation_rows}
    if len(permutation_models) != 1:
        raise ValueError("Selected-site probing requires one within-model permutation artifact; " f"found {sorted(permutation_models)}")
    assignment_ids, task_order, selected_rows, selected_by_key = selection_plan(permutation_rows)
    _validate_complement_selection(assignment_ids, selected_rows)
    complement_representatives, representative_for = _complement_representatives(assignment_ids)
    assignment_index = {assignment_id: position for position, assignment_id in enumerate(assignment_ids)}
    complement_of = {
        position: assignment_index["".join("1" if bit == "0" else "0" for bit in assignment_id)]
        for position, assignment_id in enumerate(assignment_ids)
    }

    selected_keys = set(selected_by_key)
    grouped_records, artifacts = _load_selected_records(artifact_paths, selected_keys=selected_keys, task_order=task_order)
    artifact_models = {str(item["model"]) for item in artifacts}
    if artifact_models != permutation_models:
        raise ValueError("Artifact model does not match the Fisher permutation artifact: " f"{sorted(artifact_models)} versus {sorted(permutation_models)}")
    model_name = next(iter(artifact_models))
    if (nuisance_table is None) != (not nuisance_columns):
        raise ValueError("nuisance_table and nuisance_columns must be provided together")
    if nuisance_alpha <= 0.0:
        raise ValueError("nuisance_alpha must be positive")
    if lexical_nuisance_sample_root is None and lexical_nuisance_components != 64:
        raise ValueError(
            "lexical_nuisance_components requires lexical_nuisance_sample_root"
        )
    if lexical_nuisance_sample_root is not None and lexical_nuisance_components < 1:
        raise ValueError("lexical_nuisance_components must be positive")
    nuisance_lookup = None
    nuisance_table_sha256 = None
    if nuisance_table is not None:
        nuisance_lookup, nuisance_table_sha256 = _load_nuisance_table(
            nuisance_table,
            nuisance_columns,
        )
    lexical_prompt_lookup = None
    lexical_prompt_sources: list[dict[str, str]] = []
    if lexical_nuisance_sample_root is not None:
        required_identities = {
            (str(record["task_name"]), str(record.get("sample_id")))
            for records in grouped_records.values()
            for record in records
        }
        lexical_prompt_lookup, lexical_prompt_sources = _load_prompt_lookup(
            lexical_nuisance_sample_root,
            required_identities=required_identities,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "selected_site_probe_progress.json"
    distribution_path = output_dir / "selected_site_probe_distribution.json.gz"
    report_path = output_dir / "report.json"
    configuration = {
        "implementation": _implementation_provenance(analysis_backend),
        "artifacts": artifacts,
        "permutation_nulls": {"path": str(permutation_path), "sha256": _sha256(permutation_path)},
        "assignment_ids_sha256": hashlib.sha256("\n".join(assignment_ids).encode()).hexdigest(),
        "requested_splits": int(requested_splits),
        "pca_variance": float(pca_variance),
        "seed": int(seed),
        "analysis_backend": analysis_backend,
        "nuisance_control": {
            "enabled": nuisance_lookup is not None or lexical_prompt_lookup is not None,
            "table": str(nuisance_table) if nuisance_table is not None else None,
            "table_sha256": nuisance_table_sha256,
            "columns": list(nuisance_columns),
            "alpha": (
                float(nuisance_alpha)
                if nuisance_lookup is not None or lexical_prompt_lookup is not None
                else None
            ),
            "fit_scope": (
                "training_tasks_per_fold"
                if nuisance_lookup is not None or lexical_prompt_lookup is not None
                else None
            ),
            "lexical": {
                "enabled": lexical_prompt_lookup is not None,
                "sample_root": (
                    str(lexical_nuisance_sample_root)
                    if lexical_nuisance_sample_root is not None
                    else None
                ),
                "kind": lexical_nuisance_kind if lexical_prompt_lookup is not None else None,
                "requested_components": (
                    int(lexical_nuisance_components)
                    if lexical_prompt_lookup is not None
                    else None
                ),
                "vectorizer_fit_scope": (
                    "training_tasks_per_fold" if lexical_prompt_lookup is not None else None
                ),
                "svd_fit_scope": (
                    "training_tasks_per_fold" if lexical_prompt_lookup is not None else None
                ),
                "sources": lexical_prompt_sources,
            },
        },
    }
    configuration_sha256 = _configuration_digest(configuration)
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("configuration_sha256") != configuration_sha256:
            raise ValueError("Existing progress belongs to a different selected-site analysis")
        if progress.get("complete") and report_path.exists() and distribution_path.exists():
            existing_report = json.loads(report_path.read_text(encoding="utf-8"))
            if existing_report.get("configuration_sha256") != configuration_sha256:
                raise ValueError("Existing report belongs to a different selected-site analysis")
            expected_distribution_hash = (
                existing_report.get("outputs", {}).get("distribution_sha256")
            )
            if expected_distribution_hash != _sha256(distribution_path):
                raise ValueError("Existing selected-site distribution failed its hash gate")
            return existing_report
        accuracy = list(progress["accuracy"])
        f1 = list(progress["f1"])
    else:
        accuracy = [None] * len(assignment_ids)
        f1 = [None] * len(assignment_ids)

    started = time.monotonic()
    completed_since_checkpoint = 0
    task_index = {task: index for index, task in enumerate(task_order)}
    lexical_fold_cache: dict[bytes, tuple[np.ndarray, np.ndarray]] | None = (
        {} if lexical_prompt_lookup is not None else None
    )
    lexical_component_counts: set[int] = set()
    lexical_identity_order: tuple[tuple[str, str], ...] | None = None
    for key, assignment_indices in sorted(selected_by_key.items()):
        representatives_at_site = sorted(
            {representative_for[index] for index in assignment_indices}
        )
        pending = [
            index
            for index in representatives_at_site
            if accuracy[index] is None
            or f1[index] is None
            or accuracy[complement_of[index]] is None
            or f1[complement_of[index]] is None
        ]
        if not pending:
            continue
        features, groups, nuisance_features, lexical_prompts, identities = _features_and_groups(
            grouped_records[key],
            task_order,
            analysis_backend,
            model_name=model_name,
            nuisance_lookup=nuisance_lookup,
            lexical_prompt_lookup=lexical_prompt_lookup,
        )
        if lexical_prompts is not None:
            if lexical_identity_order is None:
                lexical_identity_order = identities
            elif identities != lexical_identity_order:
                raise ValueError("Selected Fisher rows do not share one prompt identity order")
        group_task_indices = np.asarray([task_index[str(group)] for group in groups], dtype=np.int64)
        if lexical_fold_cache is not None and not lexical_fold_cache:
            assert lexical_prompts is not None
            for representative_index in complement_representatives:
                representative_labels = np.asarray(
                    [int(bit) for bit in assignment_ids[representative_index]],
                    dtype=np.int64,
                )[group_task_indices]
                _prepare_lexical_fold_cache(
                    representative_labels,
                    groups,
                    lexical_prompts,
                    requested_splits=requested_splits,
                    seed=seed,
                    kind=lexical_nuisance_kind,
                    components=lexical_nuisance_components,
                    cache=lexical_fold_cache,
                    component_counts=lexical_component_counts,
                )
        fold_cache: dict[bytes, tuple[Any, Any, Any, Any]] = {}
        for assignment_index in pending:
            assignment = assignment_ids[assignment_index]
            task_labels = np.asarray([int(bit) for bit in assignment], dtype=np.int64)
            labels = task_labels[group_task_indices]
            score_accuracy, score_f1, _effective_splits = _probe_predictions(
                features,
                labels,
                groups,
                kind="linear",
                requested_splits=requested_splits,
                seed=seed,
                analysis_backend=analysis_backend,
                fold_pca_variance=pca_variance,
                nuisance_features=nuisance_features,
                nuisance_alpha=nuisance_alpha,
                fold_cache=fold_cache,
                fold_nuisance_cache=lexical_fold_cache,
            )
            complement_index = complement_of[assignment_index]
            accuracy[assignment_index] = float(score_accuracy)
            accuracy[complement_index] = float(score_accuracy)
            f1[assignment_index] = float(score_f1)
            f1[complement_index] = float(score_f1)
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= checkpoint_every:
                _atomic_json(
                    progress_path,
                    {
                        "schema_version": 1,
                        "configuration_sha256": configuration_sha256,
                        "accuracy": accuracy,
                        "f1": f1,
                    },
                )
                completed_since_checkpoint = 0
        del fold_cache, features, nuisance_features, lexical_prompts
        if analysis_backend == "cuml":
            cp, *_ = _require_cuml()
            cp.get_default_memory_pool().free_all_blocks()

    if any(value is None for value in accuracy) or any(value is None for value in f1):
        raise RuntimeError("Selected-site probing ended with incomplete assignments")
    accuracy_values = [float(value) for value in accuracy]
    f1_values = [float(value) for value in f1]
    accuracy_symmetry = _complement_symmetry(accuracy_values, assignment_ids)
    f1_symmetry = _complement_symmetry(f1_values, assignment_ids)
    inference_valid = bool(accuracy_symmetry["verified"] and f1_symmetry["verified"])
    distribution = {
        "schema_version": 1,
        "assignment_ids": assignment_ids,
        "selected_fisher_row_indices": selected_rows.tolist(),
        "accuracy": accuracy_values,
        "macro_f1": f1_values,
    }
    with gzip.open(distribution_path, "wt", encoding="utf-8") as handle:
        json.dump(distribution, handle, separators=(",", ":"))
    selection_counts = Counter(int(index) for index in selected_rows.tolist())
    report = {
        "schema_version": 1,
        "status": "PASS" if inference_valid else "INVALID_COMPLEMENT_ASYMMETRY",
        "objective": (
            "Task-held-out linear probe at the Fisher-selected row, with Fisher selection "
            "repeated for every balanced task assignment and optional fold-local prompt residualization."
        ),
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "assignments": len(assignment_ids),
        "probe_refits": len(complement_representatives),
        "selected_sites": len(selection_counts),
        "observed_selected_site": {
            "row_index": int(selected_rows[0]),
            **{name: permutation_rows[int(selected_rows[0])][name] for name in ("layer", "stream", "module")},
        },
        "accuracy": {
            "observed": accuracy_values[0],
            "randomization_p": _randomization_p(np.asarray(accuracy_values)) if inference_valid else None,
            "null_median": float(np.median(accuracy_values[1:])),
            "null_q95": float(np.quantile(accuracy_values[1:], 0.95)),
            "complement_symmetry": accuracy_symmetry,
        },
        "macro_f1": {
            "observed": f1_values[0],
            "randomization_p": _randomization_p(np.asarray(f1_values)) if inference_valid else None,
            "null_median": float(np.median(f1_values[1:])),
            "null_q95": float(np.quantile(f1_values[1:], 0.95)),
            "complement_symmetry": f1_symmetry,
        },
        "selection_recomputed_for_every_assignment": True,
        "class_label_complements_share_one_probe_refit": True,
        "selection_complement_symmetry": {
            "verified": True,
            "method": "independent_comparison_of_fisher_selected_row_indices",
        },
        "score_complement_symmetry_method": (
            "imposed_by_reusing_one_fit_per_label_complement_pair; balanced accuracy and "
            "macro-F1 are invariant to exchanging the two class names"
        ),
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "outputs": {"distribution": str(distribution_path), "distribution_sha256": _sha256(distribution_path)},
        "interpretation": "This is an exact finite-support test of the selected task partition under the declared grouped linear-probe statistic. Fisher-site complement symmetry is checked independently; probe-score symmetry follows from one shared fit per label-complement pair and is not an independent reproducibility check. Fold-local prompt residualization, when enabled, is a sensitivity analysis rather than proof of construct-specific information. This is not evidence of a causal mechanism.",
    }
    if lexical_fold_cache is not None:
        report["lexical_nuisance_fit"] = {
            "unique_task_folds": len(lexical_fold_cache),
            "actual_component_counts": sorted(lexical_component_counts),
            "fit_is_inductive": True,
        }
    _atomic_json(report_path, report)
    _atomic_json(
        progress_path,
        {
            "schema_version": 1,
            "configuration_sha256": configuration_sha256,
            "accuracy": accuracy_values,
            "f1": f1_values,
            "complete": True,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--permutation-nulls", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--pca-variance", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--analysis-backend", choices=("sklearn", "cuml"), default="sklearn")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--nuisance-table", type=Path)
    parser.add_argument(
        "--nuisance-columns",
        default="",
        help="Comma-separated finite prompt-only columns; requires --nuisance-table.",
    )
    parser.add_argument("--nuisance-alpha", type=float, default=1.0)
    parser.add_argument(
        "--lexical-nuisance-sample-root",
        type=Path,
        help=(
            "Model-condition root containing logged prompts. TF--IDF and truncated SVD "
            "are fit on training tasks inside every grouped fold."
        ),
    )
    parser.add_argument(
        "--lexical-nuisance-kind",
        choices=("char_tfidf",),
        default="char_tfidf",
    )
    parser.add_argument("--lexical-nuisance-components", type=int, default=64)
    args = parser.parse_args()
    report = run_selection_aware_probe(
        [path.resolve() for path in args.artifacts],
        permutation_path=args.permutation_nulls.resolve(),
        output_dir=args.output_dir.resolve(),
        requested_splits=args.cv_folds,
        pca_variance=args.pca_variance,
        seed=args.seed,
        analysis_backend=args.analysis_backend,
        checkpoint_every=args.checkpoint_every,
        nuisance_table=(
            args.nuisance_table.resolve() if args.nuisance_table is not None else None
        ),
        nuisance_columns=tuple(
            column.strip() for column in args.nuisance_columns.split(",") if column.strip()
        ),
        nuisance_alpha=args.nuisance_alpha,
        lexical_nuisance_sample_root=(
            args.lexical_nuisance_sample_root.resolve()
            if args.lexical_nuisance_sample_root is not None
            else None
        ),
        lexical_nuisance_kind=args.lexical_nuisance_kind,
        lexical_nuisance_components=args.lexical_nuisance_components,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
