#!/usr/bin/env python3
"""Validate Sherman--Morrison selector scores against direct real-data inversions."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from lmms_eval.representations.analysis import _pca_projection, _require_cuml
from lmms_eval.representations.selected_site_probe import (
    _features_and_groups,
    _load_selected_records,
    _row_key,
    selection_plan,
)
from lmms_eval.representations.whitened_selector import (
    direct_regularized_whitened_score,
    regularized_whitened_distribution,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument("--permutation-nulls", required=True, type=Path)
    parser.add_argument("--selector-report", required=True, type=Path)
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()

    with gzip.open(args.permutation_nulls, "rt", encoding="utf-8") as handle:
        permutation_rows = list(json.load(handle)["rows"])
    assignment_ids, task_order, _selected, _by_key = selection_plan(permutation_rows)
    report = json.loads(args.selector_report.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        raise ValueError("Selector report is not passing")
    row_indices = {int(result["selected_row_index"]) for result in report["results"]}
    keys = {_row_key(permutation_rows[index]) for index in row_indices}
    grouped, artifacts = _load_selected_records(
        [path.resolve() for path in args.artifacts],
        selected_keys=keys,
        task_order=task_order,
    )
    model_names = {str(item["model"]) for item in artifacts}
    if len(model_names) != 1:
        raise ValueError("Validation requires one model")
    model_name = next(iter(model_names))
    sampled_indices = (0, 1, 17, 101, 1000, 5000, 10000, 12869)
    validations = []
    for result in report["results"]:
        alpha = float(result["ridge_alpha"])
        row_index = int(result["selected_row_index"])
        key = _row_key(permutation_rows[row_index])
        features, groups, _nuisance, _prompts, _identities = _features_and_groups(
            grouped[key],
            task_order,
            "cuml",
            model_name=model_name,
            nuisance_lookup=None,
            lexical_prompt_lookup=None,
        )
        projected, _explained = _pca_projection(
            features,
            retained_variance=0.95,
            analysis_backend="cuml",
        )
        scores, ridge, _mean_eigenvalue = regularized_whitened_distribution(
            projected,
            groups,
            task_order=task_order,
            assignment_ids=assignment_ids,
            ridge_alpha=alpha,
            analysis_backend="cuml",
        )
        cp, *_ = _require_cuml()
        projected_cpu = cp.asnumpy(projected)
        source = report["outputs"][f"{alpha:g}"]
        with gzip.open(source["path"], "rt", encoding="utf-8") as handle:
            stored = json.load(handle)
        stored_row = np.asarray(stored["row_scores"][row_index], dtype=np.float64)
        stored_error = float(np.max(np.abs(stored_row - scores)))
        direct_errors = []
        for assignment_index in sampled_indices:
            task_labels = {
                task: int(bit)
                for task, bit in zip(
                    task_order, assignment_ids[assignment_index], strict=True
                )
            }
            labels = np.asarray(
                [task_labels[str(group)] for group in groups], dtype=np.int64
            )
            direct = direct_regularized_whitened_score(
                projected_cpu,
                labels,
                ridge=ridge,
            )
            direct_errors.append(abs(direct - float(scores[assignment_index])))
        maximum_direct_error = float(max(direct_errors))
        if stored_error > args.tolerance or maximum_direct_error > args.tolerance:
            raise ValueError(
                f"Whitened identity failed at alpha={alpha:g}: "
                f"stored={stored_error}, direct={maximum_direct_error}"
            )
        validations.append(
            {
                "ridge_alpha": alpha,
                "row_index": row_index,
                "sampled_assignments": len(sampled_indices),
                "maximum_stored_error": stored_error,
                "maximum_direct_inverse_error": maximum_direct_error,
            }
        )
        del features, projected
        cp.get_default_memory_pool().free_all_blocks()
    print(json.dumps({"status": "PASS", "model": model_name, "validations": validations}, indent=2))


if __name__ == "__main__":
    main()
