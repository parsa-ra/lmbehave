#!/usr/bin/env python3
"""Create hash-bound, sample-free aggregates for the relative-response pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def sample_files(root: Path) -> dict[str, Path]:
    output = {}
    for path in sorted(root.rglob("*_samples_*.jsonl")):
        task = path.stem.split("_samples_", 1)[-1]
        if task in output:
            raise ValueError(f"Duplicate sample file for {task}")
        output[task] = path
    return output


def behavioral_summary(monitored: Path, knowledge: set[str], reasoning: set[str]) -> list[dict[str, Any]]:
    groups = {"knowledge": knowledge, "reasoning": reasoning}
    files = sample_files(monitored)
    rows = []
    for label, tasks in groups.items():
        values = []
        for task in sorted(tasks):
            if task not in files:
                raise ValueError(f"Missing monitored samples for {task}")
            task_rows = [json.loads(line) for line in files[task].read_text(encoding="utf-8").splitlines()]
            task_values = [float(row["exact_match"]) for row in task_rows]
            values.extend(task_values)
            rows.append(
                {
                    "scope": "task",
                    "group": label,
                    "task": task,
                    "examples": len(task_values),
                    "accuracy": statistics.fmean(task_values),
                }
            )
        rows.append(
            {
                "scope": "group",
                "group": label,
                "task": "ALL",
                "examples": len(values),
                "accuracy": statistics.fmean(values),
            }
        )
    return rows


def response_length_summary(monitored: Path) -> dict[str, Any]:
    words = []
    for path in sample_files(monitored).values():
        for line in path.read_text(encoding="utf-8").splitlines():
            response = json.loads(line)["resps"][0]
            words.append(len(str(response).split()))
    return {
        "minimum_whitespace_words": min(words),
        "median_whitespace_words": float(statistics.median(words)),
        "maximum_whitespace_words": max(words),
        "at_most_60_words": sum(value <= 60 for value in words),
        "total": len(words),
    }


def trajectory_summary(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["trajectory_position"], row["stream"])].append(row)
    output = []
    order = {"aggregate": -1, **{f"relative_bin_{index:04d}": index for index in range(8)}}
    for (position, stream), values in grouped.items():
        fdr = [float(row["fdr"]) for row in values]
        erank = [float(row["effective_rank_all_fraction"]) for row in values]
        between = [float(row["fdr_between_sum"]) for row in values]
        within = [float(row["fdr_within_sum"]) for row in values]
        output.append(
            {
                "position": position,
                "position_index": order[position],
                "stream": stream,
                "layers": len(values),
                "fdr_depth_mean": statistics.fmean(fdr),
                "fdr_depth_std": statistics.stdev(fdr) if len(fdr) > 1 else 0.0,
                "fdr_depth_max": max(fdr),
                "effective_rank_fraction_depth_mean": statistics.fmean(erank),
                "between_sum_depth_mean": statistics.fmean(between),
                "within_sum_depth_mean": statistics.fmean(within),
            }
        )
    return sorted(output, key=lambda row: (row["position_index"], row["stream"]))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()
    paths = {
        "manifest": Path(__file__).resolve().parents[1]
        / "experiments"
        / "representation_dynamics_relative_response_pilot_v2.json",
        "parity": run_dir / "parity_report.json",
        "binning": run_dir / "relative_bins" / "report.json",
        "analysis_report": run_dir / "analysis-cuml" / "report.json",
        "analysis_csv": run_dir / "analysis-cuml" / "layer_separability.csv",
        "inference": run_dir / "trajectory-inference" / "report.json",
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
    parity = read_json(paths["parity"])
    binning = read_json(paths["binning"])
    analysis = read_json(paths["analysis_report"])
    inference = read_json(paths["inference"])
    if any(payload.get("status") != "PASS" for payload in (parity, binning, inference)):
        raise ValueError("Parity, binning, and inference reports must all pass")
    if analysis.get("analysis_backend") != "cuml" or analysis.get("phase") != "both":
        raise ValueError("Pilot summary requires the frozen joint cuML analysis")
    if int(inference.get("assignments", 0)) != 70 or inference.get("position_kind") != "relative_bin":
        raise ValueError("Pilot inference does not match the frozen 4+4 relative-bin design")

    manifest = read_json(paths["manifest"])
    knowledge = set(manifest["tasks"]["knowledge"])
    reasoning = set(manifest["tasks"]["reasoning"])
    trajectory_rows = trajectory_summary(read_csv(paths["analysis_csv"]))
    behavior_rows = behavioral_summary(run_dir / "monitored", knowledge, reasoning)
    output_dir.mkdir(parents=True, exist_ok=True)
    trajectory_path = output_dir / "relative_trajectory_summary.csv"
    behavior_path = output_dir / "task_accuracy_summary.csv"
    write_csv(trajectory_path, trajectory_rows)
    write_csv(behavior_path, behavior_rows)

    report = {
        "schema_version": 1,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": manifest["experiment_id"],
        "model": manifest["model"],
        "tasks_per_group": 4,
        "examples_per_task": 4,
        "relative_bins": 8,
        "observer_off_on_sample_parity": parity["observer_off_on_sample_parity"],
        "response_update_count": {
            key: binning["response_update_count"][key]
            for key in ("minimum", "median", "maximum")
        },
        "accuracy": {
            row["group"]: row["accuracy"] for row in behavior_rows if row["scope"] == "group"
        },
        "response_length": response_length_summary(run_dir / "monitored"),
        "maximum_decode_minus_prefill_fisher": inference["maximum_decode_minus_prefill_fisher"],
        "maximum_decode_fisher": inference["maximum_decode_fisher"],
        "minimum_attainable_class_symmetric_p": 2 / 70,
        "confirmatory_threshold_attainable": False,
        "provenance": {
            name: {"path": str(path), "sha256": sha256_file(path)} for name, path in paths.items()
        },
        "outputs": {
            "relative_trajectory_summary": {
                "path": str(trajectory_path),
                "sha256": sha256_file(trajectory_path),
            },
            "task_accuracy_summary": {"path": str(behavior_path), "sha256": sha256_file(behavior_path)},
        },
        "claim_limit": manifest["claim_limit"],
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
