"""Exact task-level prompt-text baselines for representation studies."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits

from lmms_eval.representations.analysis import EPSILON, grouped_stratified_folds


_THREADPOOL_LIMIT = None


def _initialize_worker() -> None:
    """Prevent process-level parallelism from nesting threaded BLAS pools."""

    global _THREADPOOL_LIMIT
    _THREADPOOL_LIMIT = threadpool_limits(limits=1)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prompts(
    sample_root: Path,
    *,
    tasks: set[str],
) -> tuple[list[str], np.ndarray, list[dict[str, str]]]:
    """Load one logged pre-template prompt per task/document identity."""

    prompts: list[str] = []
    groups: list[str] = []
    provenance: list[dict[str, str]] = []
    for task in sorted(tasks):
        task_candidates = [sample_root / "tasks" / task]
        task_candidates.extend(sorted((sample_root / "tasks").glob(f"*-{task}")))
        task_roots = [candidate for candidate in task_candidates if candidate.is_dir()]
        paths = sorted(
            path
            for task_root in task_roots
            for path in (task_root / "evaluation").rglob("*_samples_*.jsonl")
        )
        if len(paths) != 1:
            raise ValueError(f"Expected one sample log for {task}, found {len(paths)}")
        path = paths[0]
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if not rows:
            raise ValueError(f"Empty sample log: {path}")
        seen_doc_ids: set[str] = set()
        for row in rows:
            doc_id = str(row.get("doc_id"))
            prompt = row.get("input")
            if doc_id in seen_doc_ids or not isinstance(prompt, str):
                raise ValueError(f"Invalid prompt identity in {path}: {doc_id!r}")
            seen_doc_ids.add(doc_id)
            prompts.append(prompt)
            groups.append(task)
        provenance.append({"path": str(path), "sha256": _sha256(path)})
    counts = {task: groups.count(task) for task in sorted(tasks)}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Prompt counts are not task-balanced: {counts}")
    return prompts, np.asarray(groups, dtype=object), provenance


def _assignment_ids(tasks: list[str], positive_tasks: set[str]) -> list[str]:
    positive_count = len(positive_tasks)
    observed = "".join("1" if task in positive_tasks else "0" for task in tasks)
    ids = [observed]
    for combination in itertools.combinations(tasks, positive_count):
        candidate = "".join("1" if task in combination else "0" for task in tasks)
        if candidate != observed:
            ids.append(candidate)
    if len(ids) != math.comb(len(tasks), positive_count) or len(ids) != len(set(ids)):
        raise RuntimeError("Failed to enumerate the complete balanced task-label support")
    return ids


def _representatives(assignment_ids: list[str]) -> tuple[list[int], dict[int, int]]:
    index = {assignment: position for position, assignment in enumerate(assignment_ids)}
    representatives: list[int] = []
    representative_for: dict[int, int] = {}
    visited: set[int] = set()
    for position, assignment in enumerate(assignment_ids):
        if position in visited:
            continue
        complement = "".join("1" if bit == "0" else "0" for bit in assignment)
        complement_position = index.get(complement)
        if complement_position is None:
            raise ValueError("Assignments are not closed under label complement")
        representative = 0 if 0 in (position, complement_position) else min(position, complement_position)
        representatives.append(representative)
        representative_for[position] = representative
        representative_for[complement_position] = representative
        visited.update((position, complement_position))
    return representatives, representative_for


def _vectorizer(kind: str) -> TfidfVectorizer:
    if kind == "word_tfidf":
        return TfidfVectorizer(
            lowercase=True,
            analyzer="word",
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            dtype=np.float64,
        )
    if kind == "char_tfidf":
        return TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            max_features=100_000,
            dtype=np.float64,
        )
    raise ValueError(f"Unknown prompt baseline: {kind}")


def _score_assignment(
    payload: tuple[int, str, list[str], np.ndarray, list[str], str, int, int],
) -> tuple[int, float, float]:
    index, assignment, task_order, groups, prompts, kind, requested_splits, seed = payload
    task_labels = {task: int(bit) for task, bit in zip(task_order, assignment)}
    labels = np.asarray([task_labels[str(group)] for group in groups], dtype=np.int64)
    folds, _ = grouped_stratified_folds(
        labels,
        groups,
        requested_splits=requested_splits,
        seed=seed,
    )
    predictions = np.full(labels.shape, -1, dtype=np.int64)
    for train_indices, test_indices in folds:
        vectorizer = _vectorizer(kind)
        train = vectorizer.fit_transform([prompts[position] for position in train_indices])
        test = vectorizer.transform([prompts[position] for position in test_indices])
        classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="liblinear",
            max_iter=3000,
            random_state=seed,
        )
        classifier.fit(train, labels[train_indices])
        predictions[test_indices] = classifier.predict(test)
    if np.any(predictions < 0):
        raise RuntimeError("Prompt baseline did not predict every example")
    return (
        index,
        float(accuracy_score(labels, predictions)),
        float(f1_score(labels, predictions, average="macro", zero_division=0)),
    )


def run_prompt_baseline(
    prompts: list[str],
    groups: np.ndarray,
    *,
    knowledge_tasks: set[str],
    kind: str,
    requested_splits: int,
    seed: int,
    jobs: int,
) -> dict[str, Any]:
    task_order = sorted(str(task) for task in np.unique(groups))
    assignment_ids = _assignment_ids(task_order, knowledge_tasks)
    representatives, representative_for = _representatives(assignment_ids)
    payloads = [
        (index, assignment_ids[index], task_order, groups, prompts, kind, requested_splits, seed)
        for index in representatives
    ]
    accuracy: list[float | None] = [None] * len(assignment_ids)
    macro_f1: list[float | None] = [None] * len(assignment_ids)
    iterator = map(_score_assignment, payloads)
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs, initializer=_initialize_worker) as executor:
            iterator = executor.map(_score_assignment, payloads, chunksize=4)
            results = list(iterator)
    else:
        _initialize_worker()
        results = list(iterator)
    for index, score_accuracy, score_f1 in results:
        accuracy[index] = score_accuracy
        macro_f1[index] = score_f1
    for index in range(len(assignment_ids)):
        representative = representative_for[index]
        accuracy[index] = accuracy[representative]
        macro_f1[index] = macro_f1[representative]
    if any(value is None for value in accuracy + macro_f1):
        raise RuntimeError("Prompt baseline randomization support is incomplete")

    def summary(values: list[float | None]) -> dict[str, float]:
        array = np.asarray(values, dtype=np.float64)
        observed = float(array[0])
        return {
            "observed": observed,
            "null_median": float(np.median(array[1:])),
            "null_q95": float(np.quantile(array[1:], 0.95)),
            "randomization_p": float(np.mean(array >= observed - EPSILON)),
        }

    return {
        "schema_version": 1,
        "baseline": kind,
        "task_order": task_order,
        "assignments": len(assignment_ids),
        "probe_refits": len(representatives),
        "cv_folds": requested_splits,
        "seed": seed,
        "accuracy": summary(accuracy),
        "macro_f1": summary(macro_f1),
        "class_label_complements_share_one_probe_refit": True,
        "score_complement_symmetry_method": "imposed_by_reusing_one_fit_per_label_complement_pair",
        "assignment_ids": assignment_ids,
        "accuracy_distribution": [float(value) for value in accuracy],
        "macro_f1_distribution": [float(value) for value in macro_f1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-root", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--panel", default="matched_mmlu_8x8_confirmatory")
    parser.add_argument("--baseline", choices=("word_tfidf", "char_tfidf"), required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--jobs", type=int, default=1)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    panel = manifest["panels"][args.panel]
    knowledge = {str(item["task"]) for item in panel["knowledge_tasks"]}
    reasoning = {str(item["task"]) for item in panel["reasoning_tasks"]}
    started = time.monotonic()
    prompts, groups, sources = load_prompts(
        args.sample_root,
        tasks=knowledge | reasoning,
    )
    report = run_prompt_baseline(
        prompts,
        groups,
        knowledge_tasks=knowledge,
        kind=args.baseline,
        requested_splits=args.cv_folds,
        seed=args.seed,
        jobs=args.jobs,
    )
    report.update(
        {
            "status": "PASS",
            "scope": "logged prompt text before model-specific chat-template rendering",
            "sample_root": str(args.sample_root.resolve()),
            "manifest": str(args.manifest.resolve()),
            "manifest_sha256": _sha256(args.manifest),
            "sources": sources,
            "num_prompts": len(prompts),
            "elapsed_seconds": time.monotonic() - started,
            "interpretation": "Prompt-only task-set predictability under the same grouped folds and complete balanced task-label support. It is a surface-form control, not a model-representation result.",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)
    print(json.dumps({key: report[key] for key in ("status", "baseline", "accuracy", "macro_f1", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
