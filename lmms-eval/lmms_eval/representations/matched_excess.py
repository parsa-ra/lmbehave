"""Exact matched representation-versus-prompt excess-separation inference.

For every balanced task-label assignment, this module reuses the assignment-specific
Fisher-selected representation site, fits task-held-out probes to both activations and
prompt text, and records their paired score difference.  The two arms use the same folds,
feature count, scaler, classifier implementation, and regularization.  Prompt vocabulary,
TF--IDF weights, and truncated SVD are fit on training tasks only.

The resulting randomization test asks whether the declared task partition has unusually
large representation-over-prompt probe performance among all balanced partitions of the
same tasks.  It is not a causal test and it does not make the two feature spaces identical.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from lmms_eval.representations.analysis import (
    EPSILON,
    _complement_symmetry,
    _randomization_p,
    _require_cuml,
    grouped_stratified_folds,
)
from lmms_eval.representations.selected_site_probe import (
    _complement_representatives,
    _configuration_digest,
    _features_and_groups,
    _implementation_provenance,
    _load_prompt_lookup,
    _load_selected_records,
    _prepare_lexical_fold_cache,
    _sha256,
    _validate_complement_selection,
    selection_plan,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _fixed_pca(
    train_features: Any,
    test_features: Any,
    *,
    components: int,
    analysis_backend: str,
    seed: int,
) -> tuple[Any, Any, int]:
    """Fit a fixed-width activation PCA on training tasks only."""

    actual = min(int(components), int(train_features.shape[0]) - 1, int(train_features.shape[1]))
    if actual < 1:
        raise ValueError("Matched probing requires at least one PCA component")
    if analysis_backend == "sklearn":
        solver = "full" if actual == min(train_features.shape[0] - 1, train_features.shape[1]) else "randomized"
        pca = PCA(n_components=actual, svd_solver=solver, random_state=seed)
        return pca.fit_transform(train_features), pca.transform(test_features), actual
    if analysis_backend == "cuml":
        _cp, _cuml, CumlPCA, *_rest = _require_cuml()
        pca = CumlPCA(n_components=actual, svd_solver="full", output_type="cupy")
        return pca.fit_transform(train_features), pca.transform(test_features), actual
    raise ValueError(f"Unsupported analysis backend: {analysis_backend}")


def _scale_fold(
    train_features: Any,
    test_features: Any,
    *,
    analysis_backend: str,
) -> tuple[Any, Any]:
    if analysis_backend == "sklearn":
        scaler = StandardScaler()
    elif analysis_backend == "cuml":
        _cp, _cuml, _pca, _logistic, _silhouette, CumlStandardScaler, _svc = _require_cuml()
        scaler = CumlStandardScaler(output_type="cupy")
    else:
        raise ValueError(f"Unsupported analysis backend: {analysis_backend}")
    return scaler.fit_transform(train_features), scaler.transform(test_features)


def _linear_predictions(
    train_features: Any,
    test_features: Any,
    train_labels: np.ndarray,
    *,
    analysis_backend: str,
    seed: int,
) -> np.ndarray:
    """Fit the same declared classifier for either matched feature arm."""

    if analysis_backend == "sklearn":
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            random_state=seed,
            solver="liblinear",
        )
        classifier.fit(train_features, train_labels)
        return np.asarray(classifier.predict(test_features), dtype=np.int64)
    if analysis_backend == "cuml":
        cp, _cuml, _pca, CumlLogisticRegression, *_rest = _require_cuml()
        classifier = CumlLogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=3000,
            penalty="l2",
            penalty_normalized=True,
            solver="qn",
            output_type="cupy",
        )
        classifier.fit(train_features, cp.asarray(train_labels, dtype=cp.int32))
        return cp.asnumpy(classifier.predict(test_features)).astype(np.int64)
    raise ValueError(f"Unsupported analysis backend: {analysis_backend}")


def matched_probe_predictions(
    activation_features: Any,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    requested_splits: int,
    components: int,
    seed: int,
    analysis_backend: str,
    prompt_fold_cache: dict[bytes, tuple[np.ndarray, np.ndarray]],
    activation_fold_cache: dict[bytes, tuple[Any, Any, Any, Any, int]],
    scaled_prompt_fold_cache: dict[bytes, tuple[Any, Any, Any, Any, int]],
) -> dict[str, float | int]:
    """Return paired out-of-fold scores under one task-label assignment."""

    folds, effective_splits = grouped_stratified_folds(
        labels,
        groups,
        requested_splits=requested_splits,
        seed=seed,
    )
    activation_predictions = np.full(labels.shape, -1, dtype=np.int64)
    prompt_predictions = np.full(labels.shape, -1, dtype=np.int64)
    component_counts: set[int] = set()
    for train_indices, test_indices in folds:
        key = test_indices.tobytes()
        activation_cached = activation_fold_cache.get(key)
        if activation_cached is None:
            if analysis_backend == "cuml":
                cp, *_ = _require_cuml()
                train_backend = cp.asarray(train_indices)
                test_backend = cp.asarray(test_indices)
            else:
                train_backend = train_indices
                test_backend = test_indices
            train_activation, test_activation, actual = _fixed_pca(
                activation_features[train_backend],
                activation_features[test_backend],
                components=components,
                analysis_backend=analysis_backend,
                seed=seed,
            )
            train_activation, test_activation = _scale_fold(
                train_activation,
                test_activation,
                analysis_backend=analysis_backend,
            )
            activation_cached = (
                train_backend,
                test_backend,
                train_activation,
                test_activation,
                actual,
            )
            activation_fold_cache[key] = activation_cached
        _train_backend, _test_backend, train_activation, test_activation, activation_components = activation_cached

        prompt_cached = scaled_prompt_fold_cache.get(key)
        if prompt_cached is None:
            prompt_fold = prompt_fold_cache.get(key)
            if prompt_fold is None:
                raise ValueError("No fold-local prompt representation for a grouped fold")
            train_prompt_cpu, test_prompt_cpu = prompt_fold
            prompt_components = int(train_prompt_cpu.shape[1])
            if prompt_components != activation_components:
                raise ValueError(
                    "Matched arms have different feature counts: "
                    f"activation={activation_components}, prompt={prompt_components}"
                )
            if analysis_backend == "cuml":
                cp, *_ = _require_cuml()
                train_prompt = cp.asarray(train_prompt_cpu, dtype=cp.float64)
                test_prompt = cp.asarray(test_prompt_cpu, dtype=cp.float64)
                train_backend = cp.asarray(train_indices)
                test_backend = cp.asarray(test_indices)
            else:
                train_prompt = train_prompt_cpu
                test_prompt = test_prompt_cpu
                train_backend = train_indices
                test_backend = test_indices
            train_prompt, test_prompt = _scale_fold(
                train_prompt,
                test_prompt,
                analysis_backend=analysis_backend,
            )
            prompt_cached = (
                train_backend,
                test_backend,
                train_prompt,
                test_prompt,
                prompt_components,
            )
            scaled_prompt_fold_cache[key] = prompt_cached
        _prompt_train_backend, _prompt_test_backend, train_prompt, test_prompt, prompt_components = prompt_cached
        if activation_components != prompt_components:
            raise ValueError("Matched feature counts changed across cached folds")
        component_counts.add(int(activation_components))

        train_labels = labels[train_indices]
        activation_predictions[test_indices] = _linear_predictions(
            train_activation,
            test_activation,
            train_labels,
            analysis_backend=analysis_backend,
            seed=seed,
        )
        prompt_predictions[test_indices] = _linear_predictions(
            train_prompt,
            test_prompt,
            train_labels,
            analysis_backend=analysis_backend,
            seed=seed,
        )
    if np.any(activation_predictions < 0) or np.any(prompt_predictions < 0):
        raise RuntimeError("Matched grouped probing did not predict every example")
    if component_counts != {int(components)}:
        raise ValueError(f"Requested matched width was not attainable in every fold: {component_counts}")
    activation_accuracy = float(accuracy_score(labels, activation_predictions))
    prompt_accuracy = float(accuracy_score(labels, prompt_predictions))
    activation_f1 = float(f1_score(labels, activation_predictions, average="macro", zero_division=0))
    prompt_f1 = float(f1_score(labels, prompt_predictions, average="macro", zero_division=0))
    return {
        "activation_accuracy": activation_accuracy,
        "prompt_accuracy": prompt_accuracy,
        "accuracy_delta": activation_accuracy - prompt_accuracy,
        "activation_macro_f1": activation_f1,
        "prompt_macro_f1": prompt_f1,
        "macro_f1_delta": activation_f1 - prompt_f1,
        "effective_splits": int(effective_splits),
        "components": int(components),
    }


def _score_summary(values: list[float], *, greater_is_more_extreme: bool = True) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise ValueError("Matched score distribution is non-finite")
    if greater_is_more_extreme:
        p_value = _randomization_p(array)
    else:
        p_value = float(np.mean(array <= array[0] + EPSILON))
    return {
        "observed": float(array[0]),
        "randomization_p": float(p_value),
        "null_median": float(np.median(array[1:])),
        "null_q05": float(np.quantile(array[1:], 0.05)),
        "null_q95": float(np.quantile(array[1:], 0.95)),
    }


def run_matched_excess_probe(
    artifact_paths: list[Path],
    *,
    permutation_path: Path,
    sample_root: Path,
    output_dir: Path,
    requested_splits: int = 5,
    components: int = 64,
    seed: int = 0,
    analysis_backend: str = "cuml",
    checkpoint_every: int = 25,
) -> dict[str, Any]:
    if components < 1 or checkpoint_every < 1:
        raise ValueError("components and checkpoint_every must be positive")
    with gzip.open(permutation_path, "rt", encoding="utf-8") as handle:
        permutation_payload = json.load(handle)
    permutation_rows = list(permutation_payload["rows"])
    permutation_models = {str(row.get("model")) for row in permutation_rows}
    if len(permutation_models) != 1:
        raise ValueError("Matched excess inference requires one within-model Fisher artifact")
    assignment_ids, task_order, selected_rows, selected_by_key = selection_plan(permutation_rows)
    _validate_complement_selection(assignment_ids, selected_rows)
    representatives, representative_for = _complement_representatives(assignment_ids)
    assignment_index = {assignment: index for index, assignment in enumerate(assignment_ids)}
    complement_of = {
        index: assignment_index["".join("1" if bit == "0" else "0" for bit in assignment)]
        for index, assignment in enumerate(assignment_ids)
    }

    grouped_records, artifacts = _load_selected_records(
        artifact_paths,
        selected_keys=set(selected_by_key),
        task_order=task_order,
    )
    artifact_models = {str(item["model"]) for item in artifacts}
    if artifact_models != permutation_models:
        raise ValueError("Activation artifacts and Fisher nulls identify different models")
    model_name = next(iter(artifact_models))
    required_identities = {
        (str(record["task_name"]), str(record.get("sample_id")))
        for records in grouped_records.values()
        for record in records
    }
    prompt_lookup, prompt_sources = _load_prompt_lookup(
        sample_root,
        required_identities=required_identities,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "matched_excess_progress.json"
    distribution_path = output_dir / "matched_excess_distribution.json.gz"
    report_path = output_dir / "report.json"
    configuration = {
        "implementation": _implementation_provenance(analysis_backend),
        "artifacts": artifacts,
        "permutation_nulls": {"path": str(permutation_path), "sha256": _sha256(permutation_path)},
        "sample_root": str(sample_root),
        "prompt_sources": prompt_sources,
        "assignment_ids_sha256": hashlib.sha256("\n".join(assignment_ids).encode()).hexdigest(),
        "requested_splits": int(requested_splits),
        "components": int(components),
        "seed": int(seed),
        "analysis_backend": analysis_backend,
        "prompt_features": {
            "kind": "lowercase character-within-word TF--IDF 3--5 grams",
            "min_df": 2,
            "sublinear_tf": True,
            "max_features": 100000,
            "reduction": "fold-local randomized TruncatedSVD",
        },
        "matched_fields": [
            "task folds",
            "feature count",
            "standardization",
            "classifier implementation",
            "C",
            "class weighting",
            "maximum iterations",
        ],
    }
    configuration_sha256 = _configuration_digest(configuration)
    score_names = (
        "activation_accuracy",
        "prompt_accuracy",
        "accuracy_delta",
        "activation_macro_f1",
        "prompt_macro_f1",
        "macro_f1_delta",
    )
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("configuration_sha256") != configuration_sha256:
            raise ValueError("Existing matched-excess progress belongs to another configuration")
        if progress.get("complete") and report_path.exists() and distribution_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("outputs", {}).get("distribution_sha256") != _sha256(distribution_path):
                raise ValueError("Existing matched-excess distribution failed its hash gate")
            return report
        distributions = {name: list(progress[name]) for name in score_names}
    else:
        distributions = {name: [None] * len(assignment_ids) for name in score_names}

    first_key = sorted(grouped_records)[0]
    _first_features, groups, _nuisance, prompts, identities = _features_and_groups(
        grouped_records[first_key],
        task_order,
        analysis_backend,
        model_name=model_name,
        nuisance_lookup=None,
        lexical_prompt_lookup=prompt_lookup,
    )
    if prompts is None:
        raise RuntimeError("Matched excess inference did not load prompt text")
    prompt_fold_cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
    actual_prompt_components: set[int] = set()
    task_index = {task: index for index, task in enumerate(task_order)}
    group_task_indices = np.asarray([task_index[str(group)] for group in groups], dtype=np.int64)
    for representative in representatives:
        labels = np.asarray([int(bit) for bit in assignment_ids[representative]], dtype=np.int64)[
            group_task_indices
        ]
        _prepare_lexical_fold_cache(
            labels,
            groups,
            prompts,
            requested_splits=requested_splits,
            seed=seed,
            kind="char_tfidf",
            components=components,
            cache=prompt_fold_cache,
            component_counts=actual_prompt_components,
        )
    if actual_prompt_components != {int(components)}:
        raise ValueError(
            f"The fixed matched width is not supported by every prompt fold: {actual_prompt_components}"
        )
    del _first_features, _nuisance, prompts

    scaled_prompt_fold_cache: dict[bytes, tuple[Any, Any, Any, Any, int]] = {}
    completed_since_checkpoint = 0
    started = time.monotonic()
    for key, assignment_indices in sorted(selected_by_key.items()):
        representatives_at_site = sorted({representative_for[index] for index in assignment_indices})
        pending = [
            index
            for index in representatives_at_site
            if any(
                distributions[name][index] is None
                or distributions[name][complement_of[index]] is None
                for name in score_names
            )
        ]
        if not pending:
            continue
        features, site_groups, _nuisance, _prompts, site_identities = _features_and_groups(
            grouped_records[key],
            task_order,
            analysis_backend,
            model_name=model_name,
            nuisance_lookup=None,
            lexical_prompt_lookup=None,
        )
        if site_identities != identities or not np.array_equal(site_groups, groups):
            raise ValueError("Fisher-selected rows do not share one sample identity order")
        activation_fold_cache: dict[bytes, tuple[Any, Any, Any, Any, int]] = {}
        for index in pending:
            labels = np.asarray([int(bit) for bit in assignment_ids[index]], dtype=np.int64)[
                group_task_indices
            ]
            scores = matched_probe_predictions(
                features,
                labels,
                groups,
                requested_splits=requested_splits,
                components=components,
                seed=seed,
                analysis_backend=analysis_backend,
                prompt_fold_cache=prompt_fold_cache,
                activation_fold_cache=activation_fold_cache,
                scaled_prompt_fold_cache=scaled_prompt_fold_cache,
            )
            complement = complement_of[index]
            for name in score_names:
                value = float(scores[name])
                distributions[name][index] = value
                distributions[name][complement] = value
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= checkpoint_every:
                _atomic_json(
                    progress_path,
                    {
                        "schema_version": 1,
                        "configuration_sha256": configuration_sha256,
                        **distributions,
                    },
                )
                completed_since_checkpoint = 0
        del activation_fold_cache, features
        if analysis_backend == "cuml":
            cp, *_ = _require_cuml()
            cp.get_default_memory_pool().free_all_blocks()

    if any(value is None for values in distributions.values() for value in values):
        raise RuntimeError("Matched excess inference ended with incomplete assignments")
    completed = {name: [float(value) for value in values] for name, values in distributions.items()}
    symmetries = {
        name: _complement_symmetry(values, assignment_ids) for name, values in completed.items()
    }
    inference_valid = all(item["verified"] for item in symmetries.values())
    distribution = {
        "schema_version": 1,
        "assignment_ids": assignment_ids,
        "selected_fisher_row_indices": selected_rows.tolist(),
        **completed,
    }
    with gzip.open(distribution_path, "wt", encoding="utf-8") as handle:
        json.dump(distribution, handle, separators=(",", ":"))
    report = {
        "schema_version": 1,
        "status": "PASS" if inference_valid else "INVALID_COMPLEMENT_ASYMMETRY",
        "objective": (
            "Exact paired representation-minus-prompt task-label inference using matched "
            "folds, feature count, scaler, classifier, and regularization."
        ),
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "assignments": len(assignment_ids),
        "probe_refits_per_arm": len(representatives),
        "unique_task_folds": len(prompt_fold_cache),
        "selected_sites": len(Counter(int(index) for index in selected_rows.tolist())),
        "observed_selected_site": {
            "row_index": int(selected_rows[0]),
            **{
                name: permutation_rows[int(selected_rows[0])][name]
                for name in ("layer", "stream", "module")
            },
        },
        "accuracy": {
            "representation": _score_summary(completed["activation_accuracy"]),
            "prompt": _score_summary(completed["prompt_accuracy"]),
            "representation_minus_prompt": _score_summary(completed["accuracy_delta"]),
        },
        "macro_f1": {
            "representation": _score_summary(completed["activation_macro_f1"]),
            "prompt": _score_summary(completed["prompt_macro_f1"]),
            "representation_minus_prompt": _score_summary(completed["macro_f1_delta"]),
        },
        "complement_symmetry": symmetries,
        "selection_recomputed_for_every_assignment": True,
        "class_label_complements_share_one_probe_refit": True,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "outputs": {
            "distribution": str(distribution_path),
            "distribution_sha256": _sha256(distribution_path),
        },
        "interpretation": (
            "The excess statistic tests whether the declared partition gives an unusually "
            "large representation-over-prompt probe gap among balanced partitions of these "
            "tasks. It does not prove prompt-independent computation, causal relevance, or "
            "equivalence of the two feature spaces."
        ),
    }
    _atomic_json(report_path, report)
    _atomic_json(
        progress_path,
        {
            "schema_version": 1,
            "configuration_sha256": configuration_sha256,
            **completed,
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
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--analysis-backend", choices=("sklearn", "cuml"), default="cuml")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()
    report = run_matched_excess_probe(
        [path.resolve() for path in args.artifacts],
        permutation_path=args.permutation_nulls.resolve(),
        sample_root=args.sample_root.resolve(),
        output_dir=args.output_dir.resolve(),
        requested_splits=args.cv_folds,
        components=args.components,
        seed=args.seed,
        analysis_backend=args.analysis_backend,
        checkpoint_every=args.checkpoint_every,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
