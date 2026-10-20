#!/usr/bin/env python3
"""Run, validate, analyze, and summarize a frozen longitudinal representation study."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from run_representation_experiment import environment_metadata, git_state, sha256_file


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments" / "olmo3_longitudinal_v1.json"
DEFAULT_PYTHON = ROOT / ".venv" / "bin" / "python"
ANALYSIS_METRICS = (
    "fdr",
    "fdr_between_sum",
    "fdr_within_sum",
    "effective_rank_all_fraction",
    "effective_rank_within_fraction",
    "probe_accuracy",
    "probe_f1",
    "rbf_probe_accuracy",
    "rbf_probe_f1",
    "covariance_alignment",
    "silhouette_cosine",
)


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Only longitudinal schema_version=1 is supported")
    task_groups = manifest.get("tasks", {})
    knowledge = task_groups.get("knowledge", [])
    reasoning = task_groups.get("reasoning", [])
    if len(knowledge) < 2 or len(knowledge) != len(reasoning):
        raise ValueError("Tasks must contain equal knowledge and reasoning groups with at least two tasks each")
    if len(set(knowledge + reasoning)) != len(knowledge + reasoning):
        raise ValueError("Task names must be unique")
    assignments = math.comb(len(knowledge) + len(reasoning), len(knowledge))
    requested = int(manifest["analysis"]["linear_task_label_permutations"])
    if requested > assignments - 1:
        raise ValueError(f"Only {assignments - 1} non-observed balanced task assignments exist")

    replicate_ids = [item["id"] for item in manifest.get("replicates", [])]
    if len(replicate_ids) < 2 or len(replicate_ids) != len(set(replicate_ids)):
        raise ValueError("At least two uniquely named replicate blocks are required")
    intervals = []
    for replicate in manifest["replicates"]:
        start = int(replicate["offset"])
        count = int(replicate["sample_count_per_task"])
        if start < 0 or count < 1:
            raise ValueError("Replicate offsets must be non-negative and counts must be positive")
        intervals.append((start, start + count, replicate["id"]))
    for index, (start, end, name) in enumerate(intervals):
        for other_start, other_end, other_name in intervals[index + 1 :]:
            if max(start, other_start) < min(end, other_end):
                raise ValueError(f"Replicate blocks {name!r} and {other_name!r} overlap")

    checkpoints = manifest.get("checkpoints", [])
    checkpoint_ids = [item["id"] for item in checkpoints]
    if not checkpoints or len(checkpoint_ids) != len(set(checkpoint_ids)):
        raise ValueError("Checkpoint IDs must be present and unique")
    for checkpoint in checkpoints:
        revision = str(checkpoint["revision"])
        if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
            raise ValueError(f"Checkpoint {checkpoint['id']!r} does not use an immutable 40-character revision")
        if not isinstance(checkpoint.get("position"), int) or checkpoint["position"] < 0:
            raise ValueError(f"Checkpoint {checkpoint['id']!r} needs a non-negative integer position")
    return manifest


def select_checkpoints(
    manifest: dict[str, Any], requested: list[str] | None, tiers: list[str] | None
) -> list[dict[str, Any]]:
    checkpoints = manifest["checkpoints"]
    by_id = {item["id"]: item for item in checkpoints}
    if requested:
        unknown = sorted(set(requested) - set(by_id))
        if unknown:
            raise ValueError(f"Unknown checkpoints: {', '.join(unknown)}")
        return [item for item in checkpoints if item["id"] in requested]
    selected_tiers = set(tiers or manifest["default_tiers"])
    known_tiers = {item["tier"] for item in checkpoints}
    unknown_tiers = sorted(selected_tiers - known_tiers)
    if unknown_tiers:
        raise ValueError(f"Unknown checkpoint tiers: {', '.join(unknown_tiers)}")
    return [item for item in checkpoints if item["tier"] in selected_tiers]


def select_replicates(manifest: dict[str, Any], requested: list[str] | None) -> list[dict[str, Any]]:
    replicates = manifest["replicates"]
    if not requested:
        return replicates
    known = {item["id"] for item in replicates}
    unknown = sorted(set(requested) - known)
    if unknown:
        raise ValueError(f"Unknown replicate blocks: {', '.join(unknown)}")
    return [item for item in replicates if item["id"] in requested]


def condition_root(run_dir: Path, checkpoint_id: str, replicate_id: str) -> Path:
    return run_dir / "checkpoints" / checkpoint_id / replicate_id


def one_path(paths: Iterable[Path], label: str, root: Path) -> Path:
    candidates = list(paths)
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one {label} under {root}, found {len(candidates)}")
    return candidates[0]


def condition_outputs(root: Path) -> tuple[Path, Path] | None:
    artifacts = sorted((root / "representations").glob("*/representations.pt"))
    results = sorted((root / "evaluation").glob("*/*_results.json"))
    if not artifacts and not results:
        return None
    if len(artifacts) != 1 or len(results) != 1:
        raise RuntimeError(
            f"Incomplete or ambiguous output under {root}: representations={len(artifacts)}, results={len(results)}"
        )
    return artifacts[0], results[0]


def scalar(value: Any) -> str:
    return str(value).lower() if isinstance(value, bool) else str(value)


def model_args(manifest: dict[str, Any], checkpoint: dict[str, Any]) -> str:
    values = {
        "pretrained": checkpoint["repository"],
        "revision": checkpoint["revision"],
        **manifest["model"].get("model_args", {}),
    }
    return ",".join(f"{key}={scalar(value)}" for key, value in values.items())


def build_eval_command(
    python: Path,
    manifest: dict[str, Any],
    checkpoint: dict[str, Any],
    replicate: dict[str, Any],
    root: Path,
) -> list[str]:
    extraction = manifest["extraction"]
    tasks = manifest["tasks"]["knowledge"] + manifest["tasks"]["reasoning"]
    return [
        str(python),
        "-m",
        "lmms_eval",
        "eval",
        "--model",
        manifest["model"]["lmms_model"],
        "--force_simple",
        "--model_args",
        model_args(manifest, checkpoint),
        "--tasks",
        ",".join(tasks),
        "--log_samples",
        "--output_path",
        str(root / "evaluation"),
        "--batch_size",
        str(manifest["model"]["batch_size"]),
        "--limit",
        str(replicate["sample_count_per_task"]),
        "--offset",
        str(replicate["offset"]),
        "--seed",
        str(extraction["seed"]),
        "--representation_dynamics",
        "--representation_dynamics_output",
        str(root / "representations"),
        "--representation_dynamics_pooling",
        extraction["pooling"],
        "--representation_dynamics_phases",
        extraction["phases"],
        "--representation_dynamics_storage_dtype",
        extraction["storage_dtype"],
    ]


def resolved_run_manifest(
    manifest_path: Path, manifest: dict[str, Any], python: Path
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment_manifest": str(manifest_path),
        "experiment_manifest_sha256": sha256_file(manifest_path),
        "experiment_id": manifest["experiment_id"],
        "root_git": git_state(ROOT),
        "lmms_eval_git": git_state(ROOT / "lmms-eval"),
        "environment": environment_metadata(python),
        "resolved_experiment": manifest,
    }


def ensure_run_manifest(
    run_dir: Path,
    manifest_path: Path,
    manifest: dict[str, Any],
    python: Path,
    allow_dirty: bool,
) -> dict[str, Any]:
    candidate = resolved_run_manifest(manifest_path, manifest, python)
    if not allow_dirty and (candidate["root_git"]["dirty"] or candidate["lmms_eval_git"]["dirty"]):
        raise RuntimeError("Final runs require clean root and lmms-eval trees; use --allow-dirty only for debugging")
    path = run_dir / "run_manifest.json"
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        for key in (
            "experiment_manifest_sha256",
            "root_git",
            "lmms_eval_git",
            "environment",
            "resolved_experiment",
        ):
            if current.get(key) != candidate.get(key):
                raise RuntimeError(f"Existing {path} does not match the current source, environment, or experiment")
        return current
    run_dir.mkdir(parents=True, exist_ok=True)
    candidate["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(candidate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return candidate


def command_plan(
    manifest: dict[str, Any],
    checkpoints: list[dict[str, Any]],
    replicates: list[dict[str, Any]],
) -> None:
    layers = int(manifest.get("model", {}).get("layers", 32))
    streams = 3
    examples = sum(int(item["sample_count_per_task"]) for item in replicates)
    task_count = sum(len(group) for group in manifest["tasks"].values())
    print(
        json.dumps(
            {
                "experiment_id": manifest["experiment_id"],
                "checkpoints": [item["id"] for item in checkpoints],
                "replicate_blocks": replicates,
                "model_loads": len(checkpoints) * len(replicates),
                "examples_per_checkpoint": examples * task_count,
                "expected_vectors": len(checkpoints) * examples * task_count * layers * streams,
                "repeat_interpretation": manifest["repeat_interpretation"],
                "claim_limit": manifest["claim_limit"],
            },
            indent=2,
        )
    )


def command_run(
    manifest_path: Path,
    manifest: dict[str, Any],
    python: Path,
    run_dir: Path,
    checkpoints: list[dict[str, Any]],
    replicates: list[dict[str, Any]],
    *,
    allow_dirty: bool,
    dry_run: bool,
    skip_existing: bool,
) -> None:
    if not dry_run:
        ensure_run_manifest(run_dir, manifest_path, manifest, python, allow_dirty)
    commands = []
    for checkpoint in checkpoints:
        for replicate in replicates:
            root = condition_root(run_dir, checkpoint["id"], replicate["id"])
            existing = condition_outputs(root)
            if existing:
                if skip_existing:
                    continue
                raise RuntimeError(f"Refusing to overwrite completed output under {root}")
            commands.append((root, build_eval_command(python, manifest, checkpoint, replicate, root)))
    if dry_run:
        for _root, command in commands:
            print(subprocess.list2cmdline(command))
        return
    env = dict(os.environ)
    env["WANDB_MODE"] = "disabled"
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    for index, (root, command) in enumerate(commands, start=1):
        print(f"[{index}/{len(commands)}] {subprocess.list2cmdline(command)}", flush=True)
        subprocess.run(command, cwd=ROOT, env=env, check=True)
        if condition_outputs(root) is None:
            raise RuntimeError(f"Evaluation returned success without complete outputs under {root}")


def sample_ids(artifact: Path) -> dict[str, set[str]]:
    by_task: dict[str, set[str]] = defaultdict(set)
    with (artifact.parent / "records.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            by_task[row["task_name"]].add(row["sample_id"])
    return by_task


def command_validate(
    manifest: dict[str, Any],
    python: Path,
    run_dir: Path,
    checkpoints: list[dict[str, Any]],
    replicates: list[dict[str, Any]],
) -> None:
    run_manifest_path = run_dir / "run_manifest.json"
    if not run_manifest_path.is_file():
        raise RuntimeError(f"Missing {run_manifest_path}")
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    if run_manifest.get("resolved_experiment") != manifest:
        raise RuntimeError("The current experiment manifest differs from the run-locked manifest")
    expected_lmms = run_manifest["lmms_eval_git"]
    tasks = manifest["tasks"]["knowledge"] + manifest["tasks"]["reasoning"]
    condition_records = []
    ids_by_replicate: dict[str, dict[str, set[str]]] = {}
    for replicate in replicates:
        artifacts = []
        for checkpoint in checkpoints:
            outputs = condition_outputs(condition_root(run_dir, checkpoint["id"], replicate["id"]))
            if outputs is None:
                raise RuntimeError(f"Missing output for {checkpoint['id']} / {replicate['id']}")
            artifacts.append(outputs[0])
        command = [
            str(python),
            str(ROOT / "scripts" / "validate_representation_artifact.py"),
            *(str(path) for path in artifacts),
        ]
        for task in tasks:
            command.extend(("--expected-task", f"{task}={replicate['sample_count_per_task']}"))
        command.extend(("--require-exact", "--require-matching-sample-ids"))
        result = subprocess.run(command, cwd=ROOT, check=True, stdout=subprocess.PIPE, text=True)
        summaries = json.loads(result.stdout)
        reference_ids = sample_ids(artifacts[0])
        ids_by_replicate[replicate["id"]] = reference_ids
        for checkpoint, artifact, summary in zip(checkpoints, artifacts, summaries):
            metadata = json.loads((artifact.parent / "metadata.json").read_text(encoding="utf-8"))[
                "metadata"
            ]
            if metadata.get("pretrained") != checkpoint["repository"]:
                raise RuntimeError(f"Artifact {artifact} records an unexpected model repository")
            if f"revision={checkpoint['revision']}" not in str(metadata.get("model_args", "")):
                raise RuntimeError(f"Artifact {artifact} does not record the immutable checkpoint revision")
            if int(float(metadata.get("offset", -1))) != int(replicate["offset"]):
                raise RuntimeError(f"Artifact {artifact} records an unexpected dataset offset")
            if int(float(metadata.get("limit", -1))) != int(replicate["sample_count_per_task"]):
                raise RuntimeError(f"Artifact {artifact} records an unexpected sample count")
            if metadata.get("lmms_eval_git_head") != expected_lmms["head"]:
                raise RuntimeError(f"Artifact {artifact} was extracted from an unexpected lmms-eval commit")
            if bool(metadata.get("lmms_eval_git_dirty")) != bool(expected_lmms["dirty"]):
                raise RuntimeError(f"Artifact {artifact} records an unexpected lmms-eval dirty state")
            if (
                metadata.get("lmms_eval_source_fingerprint_sha256")
                != expected_lmms["source_fingerprint_sha256"]
            ):
                raise RuntimeError(f"Artifact {artifact} records an unexpected lmms-eval source fingerprint")
            condition_records.append(
                {
                    "checkpoint": checkpoint["id"],
                    "replicate": replicate["id"],
                    "artifact": str(artifact),
                    "artifact_sha256": sha256_file(artifact),
                    "status": summary["status"],
                    "records": summary["records"],
                    "layers": summary["layers"],
                    "tasks": summary["tasks"],
                    "sample_ids_sha256": hashlib.sha256(
                        "\n".join(
                            f"{task}:{sample_id}"
                            for task in sorted(reference_ids)
                            for sample_id in sorted(reference_ids[task])
                        ).encode()
                    ).hexdigest(),
                }
            )
    for index, left in enumerate(replicates):
        for right in replicates[index + 1 :]:
            for task in tasks:
                overlap = ids_by_replicate[left["id"]][task] & ids_by_replicate[right["id"]][task]
                if overlap:
                    raise RuntimeError(
                        f"Replicate blocks {left['id']} and {right['id']} overlap for {task}: {len(overlap)} IDs"
                    )
    report = {
        "schema_version": 1,
        "status": "PASS",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_manifest_sha256": sha256_file(run_manifest_path),
        "conditions": condition_records,
        "replicate_blocks_are_disjoint": True,
        "sample_ids_match_across_checkpoints_within_each_block": True,
    }
    (run_dir / "validation_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "PASS", "conditions": len(condition_records)}, indent=2))


def command_analyze(
    manifest: dict[str, Any],
    python: Path,
    run_dir: Path,
    checkpoints: list[dict[str, Any]],
    replicates: list[dict[str, Any]],
    *,
    backend: str,
    jobs: int,
    condition_jobs: int,
) -> None:
    validation_path = run_dir / "validation_report.json"
    if not validation_path.is_file():
        raise RuntimeError("Run validation before analysis")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validated = {
        (item["checkpoint"], item["replicate"]): item for item in validation.get("conditions", [])
    }
    analysis = manifest["analysis"]
    def analyze_condition(checkpoint: dict[str, Any], replicate: dict[str, Any]) -> None:
        root = condition_root(run_dir, checkpoint["id"], replicate["id"])
        outputs = condition_outputs(root)
        if outputs is None:
            raise RuntimeError(f"Missing output for {checkpoint['id']} / {replicate['id']}")
        artifact = outputs[0]
        record = validated.get((checkpoint["id"], replicate["id"]))
        if not record or record["artifact_sha256"] != sha256_file(artifact):
            raise RuntimeError(f"Artifact was not validated or changed after validation: {artifact}")
        output_dir = root / f"analysis-{backend}"
        if (output_dir / "report.json").is_file():
            print(f"Skipping existing analysis: {output_dir}", flush=True)
            return
        command = [
            str(python),
            "-m",
            "lmms_eval.representations.analysis",
            str(artifact),
            "--knowledge-tasks",
            ",".join(manifest["tasks"]["knowledge"]),
            "--reasoning-tasks",
            ",".join(manifest["tasks"]["reasoning"]),
            "--output-dir",
            str(output_dir),
            "--cv-folds",
            str(analysis["cv_folds"]),
            "--permutations",
            str(analysis["linear_task_label_permutations"]),
            "--nonlinear-permutations",
            str(analysis["nonlinear_task_label_permutations"]),
            "--pca-variance",
            str(analysis["pca_retained_variance"]),
            "--silhouette-max-samples",
            str(analysis["silhouette_max_samples"]),
            "--seed",
            str(analysis["seed"]),
            "--jobs",
            str(jobs),
            "--parallel-backend",
            "process",
            "--analysis-backend",
            backend,
        ]
        print(subprocess.list2cmdline(command), flush=True)
        subprocess.run(command, cwd=ROOT, check=True)
        report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
        expected_rows = int(manifest.get("model", {}).get("layers", 32)) * 3
        if report.get("failures") or int(report.get("num_rows", 0)) != expected_rows:
            raise RuntimeError(
                f"Analysis under {output_dir} has {len(report.get('failures', []))} failures "
                f"and {report.get('num_rows')} rows; expected 0 and {expected_rows}"
            )

    conditions = [(checkpoint, replicate) for checkpoint in checkpoints for replicate in replicates]
    if condition_jobs == 1:
        for checkpoint, replicate in conditions:
            analyze_condition(checkpoint, replicate)
        return
    with ThreadPoolExecutor(max_workers=condition_jobs) as executor:
        futures = {
            executor.submit(analyze_condition, checkpoint, replicate): (checkpoint["id"], replicate["id"])
            for checkpoint, replicate in conditions
        }
        for future in as_completed(futures):
            checkpoint_id, replicate_id = futures[future]
            future.result()
            print(f"Completed analysis: {checkpoint_id} / {replicate_id}", flush=True)


def finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def mean_std(values: Iterable[float | None]) -> tuple[float | None, float | None, int]:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None, None, 0
    return statistics.mean(finite), statistics.stdev(finite) if len(finite) > 1 else None, len(finite)


def task_accuracy(results_path: Path, tasks: list[str]) -> float | None:
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    values = []
    for task in tasks:
        metrics = payload.get("results", {}).get(task, {})
        value = metrics.get("exact_match,letter")
        if value is None:
            candidates = [
                metric_value
                for metric_name, metric_value in metrics.items()
                if metric_name.startswith("exact_match,") and "stderr" not in metric_name
            ]
            value = candidates[0] if len(candidates) == 1 else None
        parsed = finite_float(value)
        if parsed is not None:
            values.append(parsed)
    return statistics.mean(values) if len(values) == len(tasks) else None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def command_summarize(
    manifest: dict[str, Any],
    run_dir: Path,
    checkpoints: list[dict[str, Any]],
    replicates: list[dict[str, Any]],
    *,
    backend: str,
) -> None:
    checkpoint_by_id = {item["id"]: item for item in checkpoints}
    raw_rows: dict[tuple[str, str], list[dict[str, str]]] = {}
    accuracies: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    for checkpoint in checkpoints:
        for replicate in replicates:
            root = condition_root(run_dir, checkpoint["id"], replicate["id"])
            output = root / f"analysis-{backend}"
            report = json.loads((output / "report.json").read_text(encoding="utf-8"))
            expected_rows = int(manifest.get("model", {}).get("layers", 32)) * 3
            if report.get("failures") or int(report.get("num_rows", 0)) != expected_rows:
                raise RuntimeError(f"Analysis is incomplete under {output}")
            with (output / "layer_separability.csv").open(newline="", encoding="utf-8") as handle:
                raw_rows[(checkpoint["id"], replicate["id"])] = list(csv.DictReader(handle))
            outputs = condition_outputs(root)
            assert outputs is not None
            accuracies[(checkpoint["id"], replicate["id"])] = (
                task_accuracy(outputs[1], manifest["tasks"]["knowledge"]),
                task_accuracy(outputs[1], manifest["tasks"]["reasoning"]),
            )

    layer_groups: dict[tuple[str, int, str], list[dict[str, str]]] = defaultdict(list)
    for (checkpoint_id, _replicate_id), rows in raw_rows.items():
        for row in rows:
            layer_groups[(checkpoint_id, int(row["layer"]), row["stream"])].append(row)
    layer_output = []
    for (checkpoint_id, layer, stream), rows in layer_groups.items():
        checkpoint = checkpoint_by_id[checkpoint_id]
        output_row: dict[str, Any] = {
            "checkpoint": checkpoint_id,
            "source_ref": checkpoint["source_ref"],
            "revision": checkpoint["revision"],
            "tier": checkpoint["tier"],
            "trajectory": checkpoint["trajectory"],
            "stage": checkpoint["stage"],
            "position": checkpoint["position"],
            "position_unit": checkpoint["position_unit"],
            "layer": layer,
            "stream": stream,
            "replicates": len(rows),
        }
        for metric in ANALYSIS_METRICS:
            mean, std, count = mean_std(finite_float(row.get(metric)) for row in rows)
            output_row[f"{metric}_mean"] = mean
            output_row[f"{metric}_std"] = std
            output_row[f"{metric}_n"] = count
        layer_output.append(output_row)
    layer_output.sort(key=lambda row: (list(checkpoint_by_id).index(row["checkpoint"]), row["layer"], row["stream"]))

    per_replicate = []
    for checkpoint in checkpoints:
        for replicate in replicates:
            rows = raw_rows[(checkpoint["id"], replicate["id"])]
            by_stream: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in rows:
                by_stream[row["stream"]].append(row)
            knowledge_accuracy, reasoning_accuracy = accuracies[(checkpoint["id"], replicate["id"])]
            for stream, stream_rows in by_stream.items():
                item: dict[str, Any] = {
                    "checkpoint": checkpoint["id"],
                    "replicate": replicate["id"],
                    "stream": stream,
                    "knowledge_accuracy": knowledge_accuracy,
                    "reasoning_accuracy": reasoning_accuracy,
                }
                for metric in ANALYSIS_METRICS:
                    item[f"depth_mean_{metric}"] = mean_std(
                        finite_float(row.get(metric)) for row in stream_rows
                    )[0]
                fdr_rows = [
                    (finite_float(row.get("fdr")), int(row["layer"]))
                    for row in stream_rows
                    if finite_float(row.get("fdr")) is not None
                ]
                if fdr_rows:
                    item["maximum_fdr"], item["maximum_fdr_layer"] = max(fdr_rows)
                per_replicate.append(item)

    replicate_lookup = {
        (row["checkpoint"], row["replicate"], row["stream"]): row for row in per_replicate
    }
    trajectories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for checkpoint in checkpoints:
        trajectories[checkpoint["trajectory"]].append(checkpoint)
    for trajectory_checkpoints in trajectories.values():
        positions = [checkpoint["position"] for checkpoint in trajectory_checkpoints]
        units = {checkpoint["position_unit"] for checkpoint in trajectory_checkpoints}
        if len(units) != 1 or len(set(positions)) != len(positions):
            continue
        ordered = sorted(trajectory_checkpoints, key=lambda item: item["position"])
        for previous, current in zip(ordered, ordered[1:]):
            delta_position = current["position"] - previous["position"]
            for replicate in replicates:
                for stream in ("attention_delta", "mlp_delta", "residual_state"):
                    before = replicate_lookup.get((previous["id"], replicate["id"], stream))
                    after = replicate_lookup.get((current["id"], replicate["id"], stream))
                    if not before or not after:
                        continue
                    before_fdr = before.get("depth_mean_fdr")
                    after_fdr = after.get("depth_mean_fdr")
                    if before_fdr is not None and after_fdr is not None:
                        after["fdr_velocity"] = (after_fdr - before_fdr) / delta_position
                        after["velocity_from_checkpoint"] = previous["id"]

    summary_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_replicate:
        summary_groups[(row["checkpoint"], row["stream"])].append(row)
    checkpoint_output = []
    summary_metrics = (
        "knowledge_accuracy",
        "reasoning_accuracy",
        *(f"depth_mean_{metric}" for metric in ANALYSIS_METRICS),
        "maximum_fdr",
        "maximum_fdr_layer",
        "fdr_velocity",
    )
    for (checkpoint_id, stream), rows in summary_groups.items():
        checkpoint = checkpoint_by_id[checkpoint_id]
        item: dict[str, Any] = {
            "checkpoint": checkpoint_id,
            "source_ref": checkpoint["source_ref"],
            "revision": checkpoint["revision"],
            "tier": checkpoint["tier"],
            "training_path": checkpoint["training_path"],
            "trajectory": checkpoint["trajectory"],
            "stage": checkpoint["stage"],
            "position": checkpoint["position"],
            "position_unit": checkpoint["position_unit"],
            "stream": stream,
            "replicates": len(rows),
            "velocity_from_checkpoint": next(
                (row["velocity_from_checkpoint"] for row in rows if row.get("velocity_from_checkpoint")), None
            ),
        }
        for metric in summary_metrics:
            mean, std, count = mean_std(finite_float(row.get(metric)) for row in rows)
            item[f"{metric}_mean"] = mean
            item[f"{metric}_std"] = std
            item[f"{metric}_n"] = count
        checkpoint_output.append(item)
    checkpoint_output.sort(
        key=lambda row: (list(checkpoint_by_id).index(row["checkpoint"]), row["stream"])
    )

    output_dir = run_dir / f"longitudinal-summary-{backend}"
    layer_fields = list(layer_output[0]) if layer_output else []
    checkpoint_fields = list(checkpoint_output[0]) if checkpoint_output else []
    write_csv(output_dir / "longitudinal_layer_summary.csv", layer_output, layer_fields)
    write_csv(output_dir / "longitudinal_checkpoint_summary.csv", checkpoint_output, checkpoint_fields)
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": manifest["experiment_id"],
        "analysis_backend": backend,
        "checkpoints": len(checkpoints),
        "replicate_blocks": len(replicates),
        "layer_rows": len(layer_output),
        "checkpoint_stream_rows": len(checkpoint_output),
        "repeat_interpretation": manifest["repeat_interpretation"],
        "claim_limit": manifest["claim_limit"],
        "velocity_definition": "Change in the replicate-specific depth-mean FDR divided by checkpoint-position change, only within trajectories with a shared unit and distinct positions.",
        "outputs": {
            "layer_csv_sha256": sha256_file(output_dir / "longitudinal_layer_summary.csv"),
            "checkpoint_csv_sha256": sha256_file(output_dir / "longitudinal_checkpoint_summary.csv"),
        },
    }
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def add_selectors(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--checkpoint", action="append", help="Checkpoint ID; repeat to select multiple.")
    parser.add_argument("--tier", action="append", help="Checkpoint tier; ignored when --checkpoint is used.")
    parser.add_argument("--replicate", action="append", help="Replicate block ID; repeat to select multiple.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "validate"):
        subparser = commands.add_parser(name)
        add_selectors(subparser)
        if name == "validate":
            subparser.add_argument("--run-dir", type=Path, required=True)
    run_parser = commands.add_parser("run")
    add_selectors(run_parser)
    run_parser.add_argument("--run-dir", type=Path, required=True)
    run_parser.add_argument("--allow-dirty", action="store_true")
    run_parser.add_argument("--dry-run", action="store_true")
    run_parser.add_argument("--skip-existing", action="store_true")
    analyze_parser = commands.add_parser("analyze")
    add_selectors(analyze_parser)
    analyze_parser.add_argument("--run-dir", type=Path, required=True)
    analyze_parser.add_argument("--analysis-backend", choices=("sklearn", "cuml"), default="cuml")
    analyze_parser.add_argument("--jobs", type=int, default=1)
    analyze_parser.add_argument(
        "--condition-jobs",
        type=int,
        default=1,
        help="Number of checkpoint/block conditions analyzed concurrently.",
    )
    summarize_parser = commands.add_parser("summarize")
    add_selectors(summarize_parser)
    summarize_parser.add_argument("--run-dir", type=Path, required=True)
    summarize_parser.add_argument("--analysis-backend", choices=("sklearn", "cuml"), default="cuml")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    manifest = load_manifest(manifest_path)
    python = args.python.expanduser()
    if not python.is_absolute():
        python = (Path.cwd() / python).absolute()
    if not python.is_file():
        raise ValueError(f"Python executable does not exist: {python}")
    checkpoints = select_checkpoints(manifest, args.checkpoint, args.tier)
    replicates = select_replicates(manifest, args.replicate)
    run_dir = getattr(args, "run_dir", None)
    if run_dir is not None:
        run_dir = run_dir.resolve()

    if args.command == "plan":
        command_plan(manifest, checkpoints, replicates)
    elif args.command == "run":
        command_run(
            manifest_path,
            manifest,
            python,
            run_dir,
            checkpoints,
            replicates,
            allow_dirty=args.allow_dirty,
            dry_run=args.dry_run,
            skip_existing=args.skip_existing,
        )
    elif args.command == "validate":
        command_validate(manifest, python, run_dir, checkpoints, replicates)
    elif args.command == "analyze":
        if args.jobs < 1 or args.condition_jobs < 1:
            raise ValueError("--jobs and --condition-jobs must each be at least one")
        command_analyze(
            manifest,
            python,
            run_dir,
            checkpoints,
            replicates,
            backend=args.analysis_backend,
            jobs=args.jobs,
            condition_jobs=args.condition_jobs,
        )
    elif args.command == "summarize":
        command_summarize(
            manifest,
            run_dir,
            checkpoints,
            replicates,
            backend=args.analysis_backend,
        )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
