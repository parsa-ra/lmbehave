#!/usr/bin/env python3
"""Verify observer parity and decode coverage for a long-response smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_files(root: Path) -> dict[str, Path]:
    files = sorted(root.rglob("*_samples_*.jsonl"))
    output = {}
    for path in files:
        task = path.stem.split("_samples_", 1)[-1]
        if task in output:
            raise ValueError(f"Duplicate sample file for {task} under {root}")
        output[task] = path
    return output


def read_samples(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def parsed_answer(value: Any) -> str | None:
    """Return one valid multiple-choice answer from lmms-eval's logged filter output."""
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    if isinstance(value, str) and value.upper() in {"A", "B", "C", "D"}:
        return value.upper()
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unmonitored", required=True, type=Path)
    parser.add_argument("--monitored", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--task", action="append", required=True)
    parser.add_argument("--expected-examples", type=int, default=4)
    parser.add_argument("--coverage-threshold", type=float, default=0.95)
    parser.add_argument("--minimum-prefix", type=int, default=8)
    args = parser.parse_args()

    task_set = set(args.task)
    if len(task_set) != len(args.task):
        raise ValueError("Tasks must be unique")
    off_files = sample_files(args.unmonitored.resolve())
    on_files = sample_files(args.monitored.resolve())
    if set(off_files) != task_set or set(on_files) != task_set:
        raise ValueError("Observer-off/on sample files do not match the declared tasks")

    parity_fields = (
        "doc_id",
        "doc_hash",
        "target",
        "resps",
        "filtered_resps",
        "exact_match",
        "input",
    )
    parity = True
    sample_provenance = []
    parser_rows = []
    invalid_response_count = 0
    for task in sorted(task_set):
        off_rows = read_samples(off_files[task])
        on_rows = read_samples(on_files[task])
        if len(off_rows) != args.expected_examples or len(on_rows) != args.expected_examples:
            raise ValueError(f"{task}: unexpected sample count")
        off_values = [{field: row.get(field) for field in parity_fields} for row in off_rows]
        on_values = [{field: row.get(field) for field in parity_fields} for row in on_rows]
        parity = parity and off_values == on_values
        task_invalid = sum(parsed_answer(row.get("filtered_resps")) is None for row in on_rows)
        invalid_response_count += task_invalid
        parser_rows.append(
            {
                "task": task,
                "valid": len(on_rows) - task_invalid,
                "invalid": task_invalid,
                "total": len(on_rows),
            }
        )
        sample_provenance.extend(
            [
                {"path": str(off_files[task]), "sha256": sha256_file(off_files[task])},
                {"path": str(on_files[task]), "sha256": sha256_file(on_files[task])},
            ]
        )

    artifact_path = args.artifact.resolve()
    artifact = torch.load(artifact_path, map_location="cpu", weights_only=False)
    records = list(artifact["records"])
    metadata = dict(artifact.get("metadata") or {})
    layers = sorted({int(record["layer_index"]) for record in records})
    streams = sorted({str(record["stream"]) for record in records})
    expected_streams = ["attention_delta", "mlp_delta", "residual_state"]
    if len(layers) != 32 or streams != expected_streams:
        raise ValueError(f"Unexpected observer layout: {len(layers)} layers, streams={streams}")
    if any(not bool(record.get("residual_delta_exact")) for record in records):
        raise ValueError("The smoke contains an inexact branch record")
    if any(not np.isfinite(record["vector"].detach().cpu().numpy()).all() for record in records):
        raise ValueError("The smoke contains a non-finite vector")

    position_samples: dict[tuple[str, str], set[str]] = defaultdict(set)
    position_records: dict[tuple[str, str], int] = defaultdict(int)
    for record in records:
        task = str(record["task_name"])
        if task not in task_set:
            raise ValueError(f"Unexpected task in artifact: {task}")
        if record["phase"] == "prefill":
            position = "prefill"
        elif record["phase"] == "decode" and record.get("decode_step") is not None:
            position = f'decode_step_{int(record["decode_step"])}'
        else:
            raise ValueError("Artifact has an undefined trajectory position")
        key = (task, position)
        position_samples[key].add(str(record["sample_id"]))
        position_records[key] += 1

    expected_per_position = len(layers) * len(streams) * args.expected_examples
    for task in task_set:
        key = (task, "prefill")
        if len(position_samples[key]) != args.expected_examples:
            raise ValueError(f"{task}: incomplete prefill samples")
        if position_records[key] != expected_per_position:
            raise ValueError(f"{task}: incomplete prefill layer-stream records")

    coverage_rows = []
    common_prefix = 0
    for step in range(32):
        coverages = []
        complete_records = True
        for task in sorted(task_set):
            key = (task, f"decode_step_{step}")
            samples = len(position_samples[key])
            coverage = samples / args.expected_examples
            records_at_position = position_records[key]
            coverages.append(coverage)
            if records_at_position != samples * len(layers) * len(streams):
                complete_records = False
            coverage_rows.append(
                {
                    "task": task,
                    "decode_step": step,
                    "samples": samples,
                    "coverage": coverage,
                    "records": records_at_position,
                }
            )
        if min(coverages) >= args.coverage_threshold and complete_records:
            common_prefix = step + 1
        else:
            break

    parser_valid = invalid_response_count == 0
    passed = parity and parser_valid and common_prefix >= args.minimum_prefix
    report = {
        "schema_version": 2,
        "status": "PASS" if passed else "FAIL_VALIDATION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "observer_off_on_sample_parity": parity,
        "parsed_responses_valid": parser_valid,
        "invalid_response_count": invalid_response_count,
        "parser_summary": parser_rows,
        "parity_fields": list(parity_fields),
        "tasks": sorted(task_set),
        "examples_per_task": args.expected_examples,
        "layers": len(layers),
        "streams": streams,
        "records": len(records),
        "coverage_threshold": args.coverage_threshold,
        "minimum_required_prefix": args.minimum_prefix,
        "common_decode_prefix": common_prefix,
        "coverage": coverage_rows,
        "metadata": {
            "model": metadata.get("pretrained") or metadata.get("model"),
            "phases": metadata.get("phases"),
            "decode_aggregation": metadata.get("decode_aggregation"),
            "decode_max_updates": metadata.get("decode_max_updates"),
            "storage_dtype": metadata.get("storage_dtype"),
        },
        "provenance": {
            "artifact": {"path": str(artifact_path), "sha256": sha256_file(artifact_path)},
            "sample_files": sample_provenance,
        },
        "claim_limit": "Passing establishes observer parity, valid final-answer parsing, exact capture, and a usable absolute-token prefix for this smoke. It does not establish semantic reasoning-step alignment or task-set separation.",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "status": report["status"],
                "common_decode_prefix": common_prefix,
                "invalid_response_count": invalid_response_count,
            },
            indent=2,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
