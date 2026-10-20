#!/usr/bin/env python3
"""Pool uncapped, batch-one decode states into equal relative-response bins."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _site_key(record: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(record["sample_id"]),
        int(record["layer_index"]),
        str(record["stream"]),
        str(record["module_name"]),
        bool(record["residual_delta_exact"]),
    )


def bin_artifact(source: Path, output: Path, report_path: Path, *, bins: int) -> dict[str, Any]:
    if bins < 2:
        raise ValueError("bins must be at least two")
    artifact = torch.load(source, map_location="cpu", weights_only=False)
    metadata = dict(artifact.get("metadata") or {})
    if metadata.get("decode_aggregation") != "all_updates":
        raise ValueError("Source artifact must use all_updates decode capture")
    if metadata.get("decode_max_updates") is not None:
        raise ValueError("Relative binning requires uncapped decode capture")
    if artifact.get("truncated"):
        raise ValueError(f"Source artifact reports truncated captures: {artifact['truncated']}")

    prefill = [dict(record) for record in artifact["records"] if record.get("phase") == "prefill"]
    decode = [record for record in artifact["records"] if record.get("phase") == "decode"]
    if not prefill or not decode:
        raise ValueError("Source artifact must contain both prefill and decode records")
    if any(record.get("decode_step") is None for record in decode):
        raise ValueError("Every source decode record must have an explicit decode_step")
    if any(int(record.get("sample_index", -1)) != 0 for record in artifact["records"]):
        raise ValueError("Pilot relative binning requires batch size one (sample_index=0)")

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    sample_sites: dict[str, set[tuple[Any, ...]]] = defaultdict(set)
    for record in decode:
        key = _site_key(record)
        grouped[key].append(record)
        sample_sites[key[0]].add(key[1:])

    expected_sites = None
    response_updates: dict[str, int] = {}
    for sample_id, sites in sample_sites.items():
        if expected_sites is None:
            expected_sites = sites
        elif sites != expected_sites:
            raise ValueError(f"Layer-stream site support differs for {sample_id}")
        counts = set()
        for site in sites:
            rows = grouped[(sample_id, *site)]
            steps = sorted(int(row["decode_step"]) for row in rows)
            if steps != list(range(len(steps))):
                raise ValueError(f"Decode steps are not contiguous for {sample_id}, site={site}")
            counts.add(len(steps))
        if len(counts) != 1:
            raise ValueError(f"Decode update counts differ across sites for {sample_id}: {sorted(counts)}")
        count = counts.pop()
        if count < bins:
            raise ValueError(f"{sample_id} has {count} response updates, fewer than {bins} bins")
        response_updates[sample_id] = count

    binned_records: list[dict[str, Any]] = []
    bin_sizes: dict[str, list[int]] = {}
    for key, rows in grouped.items():
        sample_id = key[0]
        total = response_updates[sample_id]
        rows = sorted(rows, key=lambda row: int(row["decode_step"]))
        buckets: list[list[dict[str, Any]]] = [[] for _ in range(bins)]
        for row in rows:
            step = int(row["decode_step"])
            bin_index = min(bins - 1, (step * bins) // total)
            buckets[bin_index].append(row)
        sizes = [len(bucket) for bucket in buckets]
        if any(size == 0 for size in sizes):
            raise ValueError(f"Empty relative bin for {sample_id}: {sizes}")
        previous = bin_sizes.setdefault(sample_id, sizes)
        if previous != sizes:
            raise ValueError(f"Relative bin sizes differ across sites for {sample_id}")
        for bin_index, bucket in enumerate(buckets):
            template = dict(bucket[0])
            vectors = torch.stack([row["vector"].float() for row in bucket])
            vector = vectors.mean(dim=0).to(dtype=bucket[0]["vector"].dtype)
            if not bool(torch.isfinite(vector).all().item()):
                raise ValueError(f"Non-finite pooled vector for {sample_id}, bin={bin_index}")
            template.update(
                {
                    "decode_aggregation": "relative_bin_mean",
                    "decode_step": None,
                    "trajectory_position": f"relative_bin_{bin_index:04d}",
                    "relative_bin": bin_index,
                    "relative_bin_start": bin_index / bins,
                    "relative_bin_end": (bin_index + 1) / bins,
                    "response_update_count": total,
                    "bin_first_decode_step": int(bucket[0]["decode_step"]),
                    "bin_last_decode_step": int(bucket[-1]["decode_step"]),
                    "num_updates": len(bucket),
                    "vector": vector,
                }
            )
            binned_records.append(template)

    for record in prefill:
        record["trajectory_position"] = "aggregate"
    output_artifact = {
        **artifact,
        "metadata": {
            **metadata,
            "decode_aggregation": "relative_bin_mean",
            "decode_max_updates": None,
            "relative_response_bins": bins,
            "relative_response_definition": "equal-width bins over all response-valid cached decode updates",
            "source_artifact_sha256": sha256_file(source),
            "num_records": len(prefill) + len(binned_records),
        },
        "decode_aggregation": "relative_bin_mean",
        "decode_max_updates": None,
        "relative_response_bins": bins,
        "records": prefill + binned_records,
        "num_records": len(prefill) + len(binned_records),
        "truncated": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_artifact, output)

    counts = sorted(response_updates.values())
    report = {
        "schema_version": 1,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "bins": bins,
        "common_decode_prefix": bins,
        "position_labels": [f"relative_bin_{index:04d}" for index in range(bins)],
        "batch_size_requirement": 1,
        "samples": len(response_updates),
        "sites_per_sample": len(expected_sites or ()),
        "source_decode_records": len(decode),
        "binned_decode_records": len(binned_records),
        "prefill_records": len(prefill),
        "response_update_count": {
            "minimum": min(counts),
            "median": float(statistics.median(counts)),
            "maximum": max(counts),
            "per_sample": dict(sorted(response_updates.items())),
        },
        "bin_sizes_per_sample": dict(sorted(bin_sizes.items())),
        "provenance": {
            "source_artifact": str(source),
            "source_artifact_sha256": sha256_file(source),
            "binned_artifact": str(output),
            "binned_artifact_sha256": sha256_file(output),
        },
        "claim_limit": (
            "Bins align relative response progress, not semantic reasoning steps. The answer-first protocol makes "
            "this a post-answer response trajectory, and the first bin includes the fixed answer marker."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--bins", type=int, default=8)
    args = parser.parse_args()
    report = bin_artifact(args.source.resolve(), args.output.resolve(), args.report.resolve(), bins=args.bins)
    print(json.dumps({key: report[key] for key in ("status", "bins", "samples", "response_update_count")}, indent=2))


if __name__ == "__main__":
    main()
