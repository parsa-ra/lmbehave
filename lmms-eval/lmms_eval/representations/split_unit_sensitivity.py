"""Contrast deliberately naive example-level and valid task-level inference.

The analysis uses one arbitrary alphabetical 8/8 task partition.  The valid reference is
the existing task-held-out, task-label randomization result.  The deliberately naive arm
mixes examples from every task across folds and permutes labels over examples.  A difference
between the two demonstrates task-identity leakage; it is not evidence about a scientific
knowledge--reasoning partition and is not a reproduction of another paper's datasets.
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

from lmms_eval.representations.matched_excess import (
    _fixed_pca,
    _linear_predictions,
    _scale_fold,
)
from lmms_eval.representations.selected_site_probe import (
    _configuration_digest,
    _features_and_groups,
    _implementation_provenance,
    _load_selected_records,
    _sha256,
    selection_plan,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _row_key(row: dict[str, Any]) -> tuple[str, str, int, str, str]:
    return (
        str(row["phase"]),
        str(row.get("trajectory_position", "aggregate")),
        int(row["layer"]),
        str(row["stream"]),
        str(row["module"]),
    )


def _prepare_example_folds(
    features: Any,
    labels: np.ndarray,
    *,
    components: int,
    splits: int,
    seed: int,
    analysis_backend: str,
) -> list[tuple[np.ndarray, np.ndarray, Any, Any]]:
    splitter = StratifiedKFold(n_splits=splits, shuffle=True, random_state=seed)
    prepared: list[tuple[np.ndarray, np.ndarray, Any, Any]] = []
    for train_indices, test_indices in splitter.split(np.zeros(labels.size), labels):
        if analysis_backend == "cuml":
            from lmms_eval.representations.analysis import _require_cuml

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
        if actual != components:
            raise ValueError(f"Requested {components} components but fold supports {actual}")
        train_features, test_features = _scale_fold(
            train_features,
            test_features,
            analysis_backend=analysis_backend,
        )
        prepared.append((train_indices, test_indices, train_features, test_features))
    return prepared


def example_level_scores(
    folds: list[tuple[np.ndarray, np.ndarray, Any, Any]],
    labels: np.ndarray,
    *,
    seed: int,
    analysis_backend: str,
) -> tuple[float, float, int]:
    predictions = np.full(labels.shape, -1, dtype=np.int64)
    minimum_training_class_count = labels.size
    for train_indices, test_indices, train_features, test_features in folds:
        train_labels = labels[train_indices]
        counts = np.bincount(train_labels, minlength=2)
        minimum_training_class_count = min(minimum_training_class_count, int(counts.min()))
        if np.any(counts == 0):
            raise ValueError("An example-level fold has a single-class training split")
        predictions[test_indices] = _linear_predictions(
            train_features,
            test_features,
            train_labels,
            analysis_backend=analysis_backend,
            seed=seed,
        )
    if np.any(predictions < 0):
        raise RuntimeError("Example-level probing did not predict every example")
    return (
        float(accuracy_score(labels, predictions)),
        float(f1_score(labels, predictions, average="macro", zero_division=0)),
        minimum_training_class_count,
    )


def run_split_unit_sensitivity(
    artifact_paths: list[Path],
    *,
    permutation_path: Path,
    matched_report_path: Path,
    matched_distribution_path: Path,
    output_dir: Path,
    components: int = 64,
    splits: int = 5,
    sample_permutations: int = 9999,
    seed: int = 0,
    analysis_backend: str = "cuml",
    checkpoint_every: int = 100,
) -> dict[str, Any]:
    if min(components, splits, sample_permutations, checkpoint_every) < 1:
        raise ValueError("Components, folds, permutations, and checkpoint interval must be positive")
    with gzip.open(permutation_path, "rt", encoding="utf-8") as handle:
        permutation_payload = json.load(handle)
    permutation_rows = list(permutation_payload["rows"])
    assignment_ids, task_order, selected_rows, _selected_by_key = selection_plan(permutation_rows)
    if task_order != sorted(task_order) or len(task_order) != 16:
        raise ValueError("The inference-unit contrast requires the alphabetical 16-task design")
    alphabetical_assignment = "1" * 8 + "0" * 8
    try:
        assignment_index = assignment_ids.index(alphabetical_assignment)
    except ValueError as error:
        raise ValueError("Alphabetical 8/8 assignment is absent") from error
    selected_row_index = int(selected_rows[assignment_index])
    selected_row = permutation_rows[selected_row_index]
    selected_key = _row_key(selected_row)

    grouped_records, artifacts = _load_selected_records(
        artifact_paths,
        selected_keys={selected_key},
        task_order=task_order,
    )
    model_names = {str(item["model"]) for item in artifacts}
    if len(model_names) != 1:
        raise ValueError("Inference-unit sensitivity must be within one model")
    model_name = next(iter(model_names))
    features, groups, _nuisance, _prompts, _identities = _features_and_groups(
        grouped_records[selected_key],
        task_order,
        analysis_backend,
        model_name=model_name,
        nuisance_lookup=None,
        lexical_prompt_lookup=None,
    )
    task_index = {task: index for index, task in enumerate(task_order)}
    labels = np.asarray(
        [int(alphabetical_assignment[task_index[str(group)]]) for group in groups],
        dtype=np.int64,
    )
    if sorted(np.bincount(labels, minlength=2).tolist()) != [800, 800]:
        raise ValueError("Expected 800 examples in each arbitrary class")

    matched_report = json.loads(matched_report_path.read_text(encoding="utf-8"))
    if matched_report.get("status") != "PASS":
        raise ValueError("Matched task-level report is not passing")
    if _sha256(matched_distribution_path) != matched_report["outputs"]["distribution_sha256"]:
        raise ValueError("Matched task-level distribution failed its hash gate")
    with gzip.open(matched_distribution_path, "rt", encoding="utf-8") as handle:
        matched_distribution = json.load(handle)
    if matched_distribution["assignment_ids"] != assignment_ids:
        raise ValueError("Task-level and example-level assignment supports differ")
    if matched_distribution["selected_fisher_row_indices"] != selected_rows.tolist():
        raise ValueError("Task-level and example-level selected sites differ")
    task_scores = np.asarray(matched_distribution["activation_macro_f1"], dtype=np.float64)
    task_observed = float(task_scores[assignment_index])
    task_p = float(np.mean(task_scores >= task_observed - 1e-12))

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "split_unit_progress.json"
    distribution_path = output_dir / "split_unit_distribution.json.gz"
    report_path = output_dir / "report.json"
    configuration = {
        "implementation": _implementation_provenance(analysis_backend),
        "analysis_module": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "artifacts": artifacts,
        "permutation_nulls": {"path": str(permutation_path), "sha256": _sha256(permutation_path)},
        "matched_task_report": {"path": str(matched_report_path), "sha256": _sha256(matched_report_path)},
        "matched_task_distribution": {
            "path": str(matched_distribution_path),
            "sha256": _sha256(matched_distribution_path),
        },
        "assignment": alphabetical_assignment,
        "assignment_index": assignment_index,
        "components": components,
        "splits": splits,
        "sample_permutations": sample_permutations,
        "seed": seed,
        "analysis_backend": analysis_backend,
        "fold_policy": (
            "five example-stratified folds fixed from the observed arbitrary labels; "
            "labels are then permuted over examples while the feature transforms and folds stay fixed"
        ),
    }
    configuration_sha256 = _configuration_digest(configuration)
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("configuration_sha256") != configuration_sha256:
            raise ValueError("Existing split-unit progress belongs to another configuration")
        accuracy_values = [float(value) for value in progress.get("accuracy", [])]
        f1_values = [float(value) for value in progress.get("macro_f1", [])]
        minimum_counts = [int(value) for value in progress.get("minimum_training_class_count", [])]
    else:
        accuracy_values = []
        f1_values = []
        minimum_counts = []

    folds = _prepare_example_folds(
        features,
        labels,
        components=components,
        splits=splits,
        seed=seed,
        analysis_backend=analysis_backend,
    )
    rng = np.random.default_rng(seed)
    if not accuracy_values:
        accuracy, macro_f1, minimum = example_level_scores(
            folds,
            labels,
            seed=seed,
            analysis_backend=analysis_backend,
        )
        accuracy_values.append(accuracy)
        f1_values.append(macro_f1)
        minimum_counts.append(minimum)
    completed_nulls = len(accuracy_values) - 1
    for _ in range(completed_nulls):
        rng.permutation(labels)

    started = time.monotonic()
    for permutation_index in range(completed_nulls, sample_permutations):
        permuted_labels = rng.permutation(labels)
        accuracy, macro_f1, minimum = example_level_scores(
            folds,
            permuted_labels,
            seed=seed,
            analysis_backend=analysis_backend,
        )
        accuracy_values.append(accuracy)
        f1_values.append(macro_f1)
        minimum_counts.append(minimum)
        if (permutation_index + 1) % checkpoint_every == 0:
            _atomic_json(
                progress_path,
                {
                    "schema_version": 1,
                    "configuration_sha256": configuration_sha256,
                    "accuracy": accuracy_values,
                    "macro_f1": f1_values,
                    "minimum_training_class_count": minimum_counts,
                },
            )

    if len(f1_values) != sample_permutations + 1:
        raise RuntimeError("Example-level permutation distribution is incomplete")
    accuracy_array = np.asarray(accuracy_values, dtype=np.float64)
    f1_array = np.asarray(f1_values, dtype=np.float64)
    distribution = {
        "schema_version": 1,
        "accuracy": accuracy_values,
        "macro_f1": f1_values,
        "minimum_training_class_count": minimum_counts,
    }
    with gzip.open(distribution_path, "wt", encoding="utf-8") as handle:
        json.dump(distribution, handle, separators=(",", ":"))
    report = {
        "schema_version": 1,
        "status": "PASS",
        "objective": "Demonstrate the consequence of changing the inferential unit on one arbitrary task partition.",
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "model": model_name,
        "site": {
            "row_index": selected_row_index,
            "layer": selected_row["layer"],
            "stream": selected_row["stream"],
            "module": selected_row["module"],
        },
        "example_level_naive": {
            "macro_f1": float(f1_array[0]),
            "macro_f1_p": float(np.mean(f1_array >= f1_array[0] - 1e-12)),
            "accuracy": float(accuracy_array[0]),
            "accuracy_p": float(np.mean(accuracy_array >= accuracy_array[0] - 1e-12)),
            "sample_permutations": sample_permutations,
            "minimum_permuted_training_class_count": min(minimum_counts[1:]),
        },
        "task_level_valid": {
            "macro_f1": task_observed,
            "exact_p": task_p,
            "balanced_task_assignments": len(assignment_ids),
        },
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "outputs": {
            "distribution": str(distribution_path),
            "distribution_sha256": _sha256(distribution_path),
        },
        "claim_limit": (
            "The example-level arm is deliberately invalid for task-category inference because "
            "each task appears in both train and test. This is not a reproduction of Sahoo et al."
        ),
    }
    _atomic_json(report_path, report)
    _atomic_json(
        progress_path,
        {
            "schema_version": 1,
            "configuration_sha256": configuration_sha256,
            "accuracy": accuracy_values,
            "macro_f1": f1_values,
            "minimum_training_class_count": minimum_counts,
            "complete": True,
        },
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--permutation-nulls", required=True, type=Path)
    parser.add_argument("--matched-report", required=True, type=Path)
    parser.add_argument("--matched-distribution", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--components", type=int, default=64)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--sample-permutations", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--analysis-backend", choices=("sklearn", "cuml"), default="cuml")
    parser.add_argument("--checkpoint-every", type=int, default=100)
    args = parser.parse_args()
    report = run_split_unit_sensitivity(
        [path.resolve() for path in args.artifacts],
        permutation_path=args.permutation_nulls.resolve(),
        matched_report_path=args.matched_report.resolve(),
        matched_distribution_path=args.matched_distribution.resolve(),
        output_dir=args.output_dir.resolve(),
        components=args.components,
        splits=args.cv_folds,
        sample_permutations=args.sample_permutations,
        seed=args.seed,
        analysis_backend=args.analysis_backend,
        checkpoint_every=args.checkpoint_every,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
