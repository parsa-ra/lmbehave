#!/usr/bin/env python3
"""Validate representation artifacts before quantitative analysis."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch


REQUIRED_STREAMS = {"attention_delta", "mlp_delta", "residual_state"}


def parse_expected_tasks(values: list[str]) -> dict[str, int]:
    expected: dict[str, int] = {}
    for value in values:
        task, separator, count = value.partition("=")
        if not separator or not task or not count.isdigit():
            raise argparse.ArgumentTypeError(f"expected TASK=COUNT, got {value!r}")
        expected[task] = int(count)
    return expected


def validate(path: Path, expected_tasks: dict[str, int], require_exact: bool) -> dict[str, object]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    records = payload.get("records", [])
    errors: list[str] = []
    if not records:
        errors.append("no records")
        return {"artifact": str(path), "status": "FAIL", "errors": errors}

    layers = {int(record["layer_index"]) for record in records}
    streams = {str(record["stream"]) for record in records}
    phases = {str(record["phase"]) for record in records}
    task_sample_counts = Counter((str(record["task_name"]), str(record["sample_id"])) for record in records)
    unique_coordinates = {
        (
            str(record["task_name"]),
            str(record["sample_id"]),
            str(record["phase"]),
            record.get("decode_step"),
            int(record["layer_index"]),
            str(record["stream"]),
        )
        for record in records
    }
    task_counts = Counter(task for task, _sample_id in task_sample_counts)
    sample_ids_by_task = {
        task: {sample_id for record_task, sample_id in task_sample_counts if record_task == task}
        for task in task_counts
    }

    if int(payload.get("num_records", -1)) != len(records):
        errors.append("num_records metadata mismatch")
    if int(payload.get("num_contexts", 0)) <= 0:
        errors.append("no tracking contexts")
    if payload.get("skipped"):
        errors.append(f"skipped captures: {payload['skipped']}")
    if streams != REQUIRED_STREAMS:
        errors.append(f"streams are {sorted(streams)}, expected {sorted(REQUIRED_STREAMS)}")
    if layers != set(range(len(payload.get("layer_specs", [])))):
        errors.append("layer coverage does not match layer_specs")
    if len(unique_coordinates) != len(records):
        errors.append("duplicate sample/layer/stream/phase coordinates")

    records_by_sample: dict[tuple[str, str], list[dict[str, object]]] = {}
    for record in records:
        records_by_sample.setdefault(
            (str(record["task_name"]), str(record["sample_id"])),
            [],
        ).append(record)
    decode_aggregation = str(payload.get("decode_aggregation", "mean_updates"))
    decode_coverage: Counter[int] = Counter()
    bad_sample_counts: list[tuple[str, str]] = []
    expected_layer_streams = {
        (layer, stream) for layer in layers for stream in REQUIRED_STREAMS
    }
    for sample_key, sample_records in records_by_sample.items():
        prefill_records = [record for record in sample_records if record["phase"] == "prefill"]
        if "prefill" in phases and {
            (int(record["layer_index"]), str(record["stream"])) for record in prefill_records
        } != expected_layer_streams:
            bad_sample_counts.append(sample_key)
            continue
        decode_records = [record for record in sample_records if record["phase"] == "decode"]
        if "decode" not in phases:
            continue
        if decode_aggregation == "all_updates":
            if any(record.get("decode_step") is None for record in decode_records):
                bad_sample_counts.append(sample_key)
                continue
            steps = sorted({int(record["decode_step"]) for record in decode_records if record.get("decode_step") is not None})
            if not steps or steps != list(range(steps[-1] + 1)):
                bad_sample_counts.append(sample_key)
                continue
            complete = True
            for step in steps:
                step_coordinates = {
                    (int(record["layer_index"]), str(record["stream"]))
                    for record in decode_records
                    if int(record.get("decode_step", -1)) == step
                }
                if step_coordinates != expected_layer_streams:
                    complete = False
                    break
                decode_coverage[step] += 1
            if not complete:
                bad_sample_counts.append(sample_key)
        elif {
            (int(record["layer_index"]), str(record["stream"])) for record in decode_records
        } != expected_layer_streams:
            bad_sample_counts.append(sample_key)
    if bad_sample_counts:
        errors.append(f"{len(bad_sample_counts)} samples have incomplete record coordinates")

    hidden_sizes = {int(record["hidden_size"]) for record in records}
    vector_dtypes = {str(record["vector"].dtype) for record in records}
    if len(hidden_sizes) != 1:
        errors.append(f"mixed hidden sizes: {sorted(hidden_sizes)}")
    for record in records:
        vector = record["vector"]
        if vector.ndim != 1 or vector.numel() != int(record["hidden_size"]):
            errors.append("vector shape/hidden_size mismatch")
            break
        if not bool(torch.isfinite(vector).all().item()):
            errors.append("non-finite vector values")
            break
    if require_exact and not all(bool(record.get("residual_delta_exact")) for record in records):
        errors.append("one or more branch deltas are not exact")
    if expected_tasks and dict(task_counts) != expected_tasks:
        errors.append(f"task sample counts are {dict(task_counts)}, expected {expected_tasks}")

    return {
        "artifact": str(path),
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "model": payload.get("metadata", {}).get("pretrained"),
        "tasks": dict(sorted(task_counts.items())),
        "layers": len(layers),
        "streams": sorted(streams),
        "phases": sorted(phases),
        "hidden_sizes": sorted(hidden_sizes),
        "vector_dtypes": sorted(vector_dtypes),
        "records": len(records),
        "contexts": payload.get("num_contexts"),
        "storage_dtype": payload.get("storage_dtype"),
        "decode_aggregation": decode_aggregation,
        "decode_max_updates": payload.get("decode_max_updates"),
        "decode_step_sample_coverage": dict(sorted(decode_coverage.items())),
        "truncated": payload.get("truncated", {}),
        "_sample_ids_by_task": sample_ids_by_task,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", nargs="+", type=Path)
    parser.add_argument(
        "--expected-task",
        action="append",
        default=[],
        metavar="TASK=COUNT",
        help="Require an exact unique-sample count for a task; repeat for each task.",
    )
    parser.add_argument("--require-exact", action="store_true", help="Fail if any branch delta is approximate.")
    parser.add_argument(
        "--require-matching-sample-ids",
        action="store_true",
        help="Require every artifact to contain identical sample IDs for each shared task.",
    )
    args = parser.parse_args()
    expected_tasks = parse_expected_tasks(args.expected_task)
    summaries = [validate(path, expected_tasks, args.require_exact) for path in args.artifacts]
    if args.require_matching_sample_ids and summaries:
        task_names = sorted(set().union(*(summary.get("_sample_ids_by_task", {}) for summary in summaries)))
        for task_name in task_names:
            reference = None
            for summary in summaries:
                sample_ids = summary.get("_sample_ids_by_task", {}).get(task_name)
                if sample_ids is None:
                    summary["errors"].append(f"missing task {task_name!r} required for sample-ID matching")
                    summary["status"] = "FAIL"
                    continue
                if reference is None:
                    reference = sample_ids
                elif sample_ids != reference:
                    summary["errors"].append(f"sample IDs differ for task {task_name!r}")
                    summary["status"] = "FAIL"
    for summary in summaries:
        summary.pop("_sample_ids_by_task", None)
    print(json.dumps(summaries, indent=2, sort_keys=True))
    if any(summary["status"] != "PASS" for summary in summaries):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
