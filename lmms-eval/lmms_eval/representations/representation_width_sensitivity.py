"""Selection-aware activation-width sensitivity against a stored prompt-64 arm.

This reviewer-requested analysis preserves the exact task assignments, task-held-out folds,
mean-shift site reselection, scaler, classifier, and prompt-64 scores from the matched
analysis while changing only activation reduction.  Unequal-width comparisons are
sensitivity analyses, not capacity-matched tests.
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
from sklearn.metrics import accuracy_score, f1_score

from lmms_eval.representations.analysis import (
    _complement_symmetry,
    _require_cuml,
    grouped_stratified_folds,
)
from lmms_eval.representations.matched_excess import (
    _fixed_pca,
    _linear_predictions,
    _scale_fold,
    _score_summary,
)
from lmms_eval.representations.selected_site_probe import (
    _complement_representatives,
    _configuration_digest,
    _features_and_groups,
    _implementation_provenance,
    _load_selected_records,
    _sha256,
    _validate_complement_selection,
    selection_plan,
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _parse_reduction(value: str) -> tuple[str, int | None]:
    if value == "raw":
        return "raw", None
    if value.startswith("fixed:"):
        try:
            components = int(value.split(":", 1)[1])
        except ValueError as error:
            raise argparse.ArgumentTypeError("fixed reduction must be fixed:POSITIVE_INT") from error
        if components < 1:
            raise argparse.ArgumentTypeError("fixed reduction must use a positive width")
        return "fixed", components
    raise argparse.ArgumentTypeError("reduction must be raw or fixed:POSITIVE_INT")


def activation_probe_predictions(
    activation_features: Any,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    requested_splits: int,
    reduction: tuple[str, int | None],
    seed: int,
    analysis_backend: str,
    activation_fold_cache: dict[bytes, tuple[Any, Any, int]],
) -> dict[str, float | int]:
    folds, effective_splits = grouped_stratified_folds(
        labels,
        groups,
        requested_splits=requested_splits,
        seed=seed,
    )
    predictions = np.full(labels.shape, -1, dtype=np.int64)
    dimensions: set[int] = set()
    mode, requested_components = reduction
    for train_indices, test_indices in folds:
        key = test_indices.tobytes()
        cached = activation_fold_cache.get(key)
        if cached is None:
            if analysis_backend == "cuml":
                cp, *_ = _require_cuml()
                train_backend = cp.asarray(train_indices)
                test_backend = cp.asarray(test_indices)
            else:
                train_backend = train_indices
                test_backend = test_indices
            train_features = activation_features[train_backend]
            test_features = activation_features[test_backend]
            if mode == "fixed":
                if requested_components is None:
                    raise RuntimeError("Fixed reduction is missing its width")
                train_features, test_features, dimension = _fixed_pca(
                    train_features,
                    test_features,
                    components=requested_components,
                    analysis_backend=analysis_backend,
                    seed=seed,
                )
                if dimension != requested_components:
                    raise ValueError(
                        f"Requested activation width {requested_components}, fold supports {dimension}"
                    )
            elif mode == "raw":
                dimension = int(train_features.shape[1])
            else:
                raise ValueError(f"Unsupported reduction mode: {mode}")
            train_features, test_features = _scale_fold(
                train_features,
                test_features,
                analysis_backend=analysis_backend,
            )
            cached = (train_features, test_features, dimension)
            activation_fold_cache[key] = cached
        train_features, test_features, dimension = cached
        dimensions.add(int(dimension))
        predictions[test_indices] = _linear_predictions(
            train_features,
            test_features,
            labels[train_indices],
            analysis_backend=analysis_backend,
            seed=seed,
        )
    if np.any(predictions < 0):
        raise RuntimeError("Activation-width probing did not predict every example")
    if len(dimensions) != 1:
        raise ValueError(f"Activation dimension changed across folds: {sorted(dimensions)}")
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, average="macro", zero_division=0)),
        "effective_splits": int(effective_splits),
        "activation_dimension": int(next(iter(dimensions))),
    }


def run_representation_width_sensitivity(
    artifact_paths: list[Path],
    *,
    permutation_path: Path,
    matched_report_path: Path,
    matched_distribution_path: Path,
    output_dir: Path,
    reduction: tuple[str, int | None],
    requested_splits: int = 5,
    seed: int = 0,
    analysis_backend: str = "cuml",
    checkpoint_every: int = 25,
    maximum_refits: int | None = None,
) -> dict[str, Any]:
    if requested_splits < 2 or checkpoint_every < 1:
        raise ValueError("requested_splits must exceed one and checkpoint_every must be positive")
    if maximum_refits is not None and maximum_refits < 1:
        raise ValueError("maximum_refits must be positive when provided")
    with gzip.open(permutation_path, "rt", encoding="utf-8") as handle:
        permutation_payload = json.load(handle)
    permutation_rows = list(permutation_payload["rows"])
    assignment_ids, task_order, selected_rows, selected_by_key = selection_plan(permutation_rows)
    _validate_complement_selection(assignment_ids, selected_rows)
    representatives, representative_for = _complement_representatives(assignment_ids)
    assignment_index = {assignment: index for index, assignment in enumerate(assignment_ids)}
    complement_of = {
        index: assignment_index["".join("1" if bit == "0" else "0" for bit in assignment)]
        for index, assignment in enumerate(assignment_ids)
    }

    matched_report = json.loads(matched_report_path.read_text(encoding="utf-8"))
    if matched_report.get("status") != "PASS":
        raise ValueError("Matched prompt-64 report is not passing")
    if int(matched_report["configuration"].get("components", -1)) != 64:
        raise ValueError("Stored matched prompt arm is not 64-dimensional")
    if _sha256(matched_distribution_path) != matched_report["outputs"]["distribution_sha256"]:
        raise ValueError("Stored matched distribution failed its hash gate")
    with gzip.open(matched_distribution_path, "rt", encoding="utf-8") as handle:
        matched_distribution = json.load(handle)
    if matched_distribution["assignment_ids"] != assignment_ids:
        raise ValueError("Stored prompt and activation assignment supports differ")
    if matched_distribution["selected_fisher_row_indices"] != selected_rows.tolist():
        raise ValueError("Stored prompt and activation selected sites differ")
    prompt_accuracy = np.asarray(matched_distribution["prompt_accuracy"], dtype=np.float64)
    prompt_macro_f1 = np.asarray(matched_distribution["prompt_macro_f1"], dtype=np.float64)
    if not np.all(np.isfinite(prompt_accuracy)) or not np.all(np.isfinite(prompt_macro_f1)):
        raise ValueError("Stored prompt scores are non-finite")

    grouped_records, artifacts = _load_selected_records(
        artifact_paths,
        selected_keys=set(selected_by_key),
        task_order=task_order,
    )
    permutation_models = {str(row.get("model")) for row in permutation_rows}
    artifact_models = {str(item["model"]) for item in artifacts}
    if len(permutation_models) != 1 or artifact_models != permutation_models:
        raise ValueError("Activation artifacts and mean-shift nulls identify different models")
    model_name = next(iter(artifact_models))

    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "width_sensitivity_progress.json"
    distribution_path = output_dir / "width_sensitivity_distribution.json.gz"
    report_path = output_dir / "report.json"
    mode, components = reduction
    configuration = {
        "implementation": _implementation_provenance(analysis_backend),
        "analysis_module": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
        },
        "artifacts": artifacts,
        "permutation_nulls": {"path": str(permutation_path), "sha256": _sha256(permutation_path)},
        "matched_prompt64_report": {
            "path": str(matched_report_path),
            "sha256": _sha256(matched_report_path),
        },
        "matched_prompt64_distribution": {
            "path": str(matched_distribution_path),
            "sha256": _sha256(matched_distribution_path),
        },
        "assignment_ids_sha256": hashlib.sha256("\n".join(assignment_ids).encode()).hexdigest(),
        "requested_splits": requested_splits,
        "activation_reduction": {"mode": mode, "components": components},
        "prompt_components": 64,
        "seed": seed,
        "analysis_backend": analysis_backend,
        "classifier": "same class-balanced C=1 QN logistic regression as matched prompt-64",
    }
    configuration_sha256 = _configuration_digest(configuration)
    score_names = ("activation_accuracy", "activation_macro_f1")
    if progress_path.exists():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        if progress.get("configuration_sha256") != configuration_sha256:
            raise ValueError("Existing width-sensitivity progress belongs to another configuration")
        if progress.get("complete") and report_path.exists() and distribution_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if report.get("outputs", {}).get("distribution_sha256") != _sha256(distribution_path):
                raise ValueError("Existing width-sensitivity distribution failed its hash gate")
            return report
        distributions = {name: list(progress[name]) for name in score_names}
        observed_dimensions = {int(value) for value in progress.get("activation_dimensions", [])}
    else:
        distributions = {name: [None] * len(assignment_ids) for name in score_names}
        observed_dimensions: set[int] = set()

    first_key = sorted(grouped_records)[0]
    first_features, groups, _nuisance, _prompts, identities = _features_and_groups(
        grouped_records[first_key],
        task_order,
        analysis_backend,
        model_name=model_name,
        nuisance_lookup=None,
        lexical_prompt_lookup=None,
    )
    del first_features
    task_index = {task: index for index, task in enumerate(task_order)}
    group_task_indices = np.asarray([task_index[str(group)] for group in groups], dtype=np.int64)
    completed_since_checkpoint = 0
    total_new_refits = 0
    started = time.monotonic()
    stopped_early = False
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
            raise ValueError("Mean-shift-selected rows do not share one sample identity order")
        fold_cache: dict[bytes, tuple[Any, Any, int]] = {}
        for index in pending:
            if maximum_refits is not None and total_new_refits >= maximum_refits:
                stopped_early = True
                break
            labels = np.asarray([int(bit) for bit in assignment_ids[index]], dtype=np.int64)[
                group_task_indices
            ]
            scores = activation_probe_predictions(
                features,
                labels,
                groups,
                requested_splits=requested_splits,
                reduction=reduction,
                seed=seed,
                analysis_backend=analysis_backend,
                activation_fold_cache=fold_cache,
            )
            observed_dimensions.add(int(scores["activation_dimension"]))
            complement = complement_of[index]
            for target, source in (
                ("activation_accuracy", "accuracy"),
                ("activation_macro_f1", "macro_f1"),
            ):
                value = float(scores[source])
                distributions[target][index] = value
                distributions[target][complement] = value
            total_new_refits += 1
            completed_since_checkpoint += 1
            if completed_since_checkpoint >= checkpoint_every:
                _atomic_json(
                    progress_path,
                    {
                        "schema_version": 1,
                        "configuration_sha256": configuration_sha256,
                        **distributions,
                        "activation_dimensions": sorted(observed_dimensions),
                    },
                )
                completed_since_checkpoint = 0
        del fold_cache, features
        if analysis_backend == "cuml":
            cp, *_ = _require_cuml()
            cp.get_default_memory_pool().free_all_blocks()
        if stopped_early:
            break

    if stopped_early:
        _atomic_json(
            progress_path,
            {
                "schema_version": 1,
                "configuration_sha256": configuration_sha256,
                **distributions,
                "activation_dimensions": sorted(observed_dimensions),
                "bounded_smoke_complete": True,
            },
        )
        return {
            "schema_version": 1,
            "status": "BOUNDED_SMOKE",
            "configuration": configuration,
            "new_refits": total_new_refits,
            "activation_dimensions": sorted(observed_dimensions),
            "elapsed_seconds_this_invocation": time.monotonic() - started,
        }

    if any(value is None for values in distributions.values() for value in values):
        raise RuntimeError("Width-sensitivity inference ended with incomplete assignments")
    completed = {name: np.asarray(values, dtype=np.float64) for name, values in distributions.items()}
    accuracy_delta = completed["activation_accuracy"] - prompt_accuracy
    macro_f1_delta = completed["activation_macro_f1"] - prompt_macro_f1
    completed_lists = {name: values.tolist() for name, values in completed.items()}
    symmetries = {
        name: _complement_symmetry(values, assignment_ids)
        for name, values in {
            **completed_lists,
            "accuracy_delta": accuracy_delta.tolist(),
            "macro_f1_delta": macro_f1_delta.tolist(),
        }.items()
    }
    inference_valid = all(item["verified"] for item in symmetries.values())
    distribution = {
        "schema_version": 1,
        "assignment_ids": assignment_ids,
        "selected_fisher_row_indices": selected_rows.tolist(),
        **completed_lists,
        "prompt64_accuracy": prompt_accuracy.tolist(),
        "prompt64_macro_f1": prompt_macro_f1.tolist(),
        "accuracy_delta": accuracy_delta.tolist(),
        "macro_f1_delta": macro_f1_delta.tolist(),
    }
    with gzip.open(distribution_path, "wt", encoding="utf-8") as handle:
        json.dump(distribution, handle, separators=(",", ":"))
    report = {
        "schema_version": 1,
        "status": "PASS" if inference_valid else "INVALID_COMPLEMENT_ASYMMETRY",
        "objective": "Activation-width sensitivity for exact representation-minus-prompt64 inference.",
        "configuration": configuration,
        "configuration_sha256": configuration_sha256,
        "assignments": len(assignment_ids),
        "probe_refits": len(representatives),
        "selected_sites": len(Counter(int(value) for value in selected_rows.tolist())),
        "activation_dimensions": sorted(observed_dimensions),
        "observed_selected_site": {
            "row_index": int(selected_rows[0]),
            **{
                name: permutation_rows[int(selected_rows[0])][name]
                for name in ("layer", "stream", "module")
            },
        },
        "accuracy": {
            "representation": _score_summary(completed["activation_accuracy"].tolist()),
            "prompt64": _score_summary(prompt_accuracy.tolist()),
            "representation_minus_prompt64": _score_summary(accuracy_delta.tolist()),
        },
        "macro_f1": {
            "representation": _score_summary(completed["activation_macro_f1"].tolist()),
            "prompt64": _score_summary(prompt_macro_f1.tolist()),
            "representation_minus_prompt64": _score_summary(macro_f1_delta.tolist()),
        },
        "complement_symmetry": symmetries,
        "selection_recomputed_for_every_assignment": True,
        "elapsed_seconds_this_invocation": time.monotonic() - started,
        "outputs": {
            "distribution": str(distribution_path),
            "distribution_sha256": _sha256(distribution_path),
        },
        "claim_limit": (
            "Activation and prompt dimensions are intentionally unequal. This sensitivity "
            "analysis does not establish matched capacity or equivalence."
        ),
    }
    _atomic_json(report_path, report)
    _atomic_json(
        progress_path,
        {
            "schema_version": 1,
            "configuration_sha256": configuration_sha256,
            **completed_lists,
            "activation_dimensions": sorted(observed_dimensions),
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
    parser.add_argument("--activation-reduction", required=True, type=_parse_reduction)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--analysis-backend", choices=("sklearn", "cuml"), default="cuml")
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--maximum-refits", type=int)
    args = parser.parse_args()
    report = run_representation_width_sensitivity(
        [path.resolve() for path in args.artifacts],
        permutation_path=args.permutation_nulls.resolve(),
        matched_report_path=args.matched_report.resolve(),
        matched_distribution_path=args.matched_distribution.resolve(),
        output_dir=args.output_dir.resolve(),
        reduction=args.activation_reduction,
        requested_splits=args.cv_folds,
        seed=args.seed,
        analysis_backend=args.analysis_backend,
        checkpoint_every=args.checkpoint_every,
        maximum_refits=args.maximum_refits,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
