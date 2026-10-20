#!/usr/bin/env python3
"""Validate and summarize Apertus 1.0 observer and benchmark preflight artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments" / "apertus1_longitudinal_v1.json"
DEFAULT_PARITY_ROOT = ROOT / "logs" / "apertus1-observer-parity-v1"
DEFAULT_BENCHMARK_ROOT = ROOT / "logs" / "apertus1-benchmark-anchor-main-ai2-arc-v1"
DEFAULT_OUTPUT = ROOT / "results" / "apertus1_preflight_v1" / "report.json"
PARITY_CHECKPOINTS = (
    "pretrain-stage1-210b",
    "pretrain-stage3-7047b",
    "pretrain-stage5-15000b",
)
PUBLISHED_ARC = 0.727
RELEASED_MAIN_REVISION = "3162c99675aa588097cecd4a24b9aa1f712af477"
REPORT_TIME_HARNESS_REFERENCE = "de496b80d60c267a2d7eea3b3c1dc40f693daee7"
ARC_CONFIG_SHA256 = {
    "arc_easy": "96da1d9efe1df88659481cedc13985cb347dc068554a2eae561303fa39bfab3f",
    "arc_challenge": "682621c26ae24b1065a2d732460a55a8ef2b9f2544574431666c9309a9efd7bd",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one(paths: list[Path], description: str) -> Path:
    if len(paths) != 1:
        raise RuntimeError(f"Expected exactly one {description}, found {len(paths)}")
    return paths[0]


def parse_model_revision(model_args: str) -> str:
    fields = dict(item.split("=", 1) for item in model_args.split(",") if "=" in item)
    return fields.get("revision", "")


def load_parity_reports(root: Path, manifest_sha256: str) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for checkpoint_id in PARITY_CHECKPOINTS:
        path = root / checkpoint_id / "observer_parity.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        if report.get("status") != "PASS":
            raise RuntimeError(f"Observer parity did not pass: {path}")
        if report.get("manifest_sha256") != manifest_sha256:
            raise RuntimeError(f"Observer parity uses a different manifest: {path}")
        if report["checkpoint"]["id"] != checkpoint_id:
            raise RuntimeError(f"Observer parity checkpoint mismatch: {path}")
        parity = report["parity"]
        capture = report["capture"]
        required = (
            parity["logits_equal_observer_on"],
            parity["logits_equal_after_removal"],
            parity["greedy_tokens_equal_observer_on"],
            parity["greedy_tokens_equal_after_removal"],
            capture["finite"],
            capture["exact"],
            capture["observed_records"] == capture["expected_records"],
            not capture["skipped"],
            report["cleanup"]["hooks_after_removal"] == 0,
        )
        if not all(required):
            raise RuntimeError(f"Observer parity assertions are incomplete: {path}")
        summaries.append(
            {
                "checkpoint_id": checkpoint_id,
                "revision": report["checkpoint"]["revision"],
                "report_sha256": sha256_file(path),
                "records": capture["observed_records"],
                "layers": report["stack"]["layers"],
                "hooks": report["stack"]["hooks"],
                "selected_stack": report["stack"]["selected_stack"],
                "xielu": report["runtime"]["xielu"],
                "lmms_eval_git_head": report["runtime"]["lmms_eval_git_head"],
                "lmms_eval_git_dirty": report["runtime"].get("lmms_eval_git_dirty"),
                "observer_script_sha256": report["runtime"].get("observer_script_sha256"),
            }
        )
    return summaries


def load_benchmark(root: Path) -> dict[str, Any]:
    path = one(sorted(root.glob("*/*_results.json")), "benchmark results JSON")
    report = json.loads(path.read_text(encoding="utf-8"))
    results = report["results"]
    sample_counts = report["n-samples"]
    if set(results) != {"arc_easy", "arc_challenge"}:
        raise RuntimeError(f"Expected ARC-Easy and ARC-Challenge, found {sorted(results)}")
    revision = parse_model_revision(report["config"]["model_args"])
    if revision != RELEASED_MAIN_REVISION:
        raise RuntimeError(f"Benchmark does not use the frozen released-main revision: {revision}")

    correct = 0.0
    total = 0
    tasks: dict[str, Any] = {}
    for task in ("arc_easy", "arc_challenge"):
        count = int(sample_counts[task]["effective"])
        accuracy = float(results[task]["acc,none"])
        normalized = float(results[task]["acc_norm,none"])
        correct += count * accuracy
        total += count
        tasks[task] = {
            "samples": count,
            "accuracy": accuracy,
            "accuracy_normalized": normalized,
            "task_hash": report["task_hashes"][task],
        }
    pooled_accuracy = correct / total
    difference = pooled_accuracy - PUBLISHED_ARC
    return {
        "status": "NEAR_MATCH_NOT_EXACT_REPRODUCTION",
        "interpretation": (
            "The pooled raw accuracy is close to the published aggregate, but the technical "
            "report does not state its ARC pooling rule and the values are not identical. This "
            "validates model loading and probabilistic scoring only; it is not an exact benchmark reproduction."
        ),
        "repository": report["model_name"],
        "revision": revision,
        "lmms_eval_git_head": report["git_hash"],
        "result_sha256": sha256_file(path),
        "tasks": tasks,
        "pooled_raw_accuracy": pooled_accuracy,
        "published_arc": PUBLISHED_ARC,
        "difference": difference,
        "absolute_difference_percentage_points": abs(difference) * 100.0,
        "pooling_used_here": "sample-count-weighted mean of raw accuracy across ARC-Easy and ARC-Challenge",
        "paper_protocol": "lm-evaluation-harness default task configuration with probabilistic scoring",
        "report_time_harness_reference": REPORT_TIME_HARNESS_REFERENCE,
        "task_config_sha256": ARC_CONFIG_SHA256,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--parity-root", type=Path, default=DEFAULT_PARITY_ROOT)
    parser.add_argument("--benchmark-root", type=Path, default=DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest_sha256 = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    parity = load_parity_reports(args.parity_root.resolve(), manifest_sha256)
    benchmark = load_benchmark(args.benchmark_root.resolve())
    observer_provenance_complete = all(
        item["lmms_eval_git_dirty"] is False and item["observer_script_sha256"]
        for item in parity
    )
    output = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "PREFLIGHT_COMPLETE_WITH_NONEXACT_BENCHMARK_ANCHOR"
            if observer_provenance_complete
            else "ENGINEERING_PASS_OBSERVER_PROVENANCE_INCOMPLETE"
        ),
        "experiment_id": manifest["experiment_id"],
        "manifest": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "observer_parity": parity,
        "observer_provenance_complete": observer_provenance_complete,
        "benchmark_anchor": benchmark,
        "claim_limit": (
            "The three observer checks establish behavior-preserving exact extraction at sampled "
            "pretraining checkpoints. The ARC check is a released-main engineering anchor and is "
            "not part of the pretraining trajectory or evidence for its scientific hypothesis."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
