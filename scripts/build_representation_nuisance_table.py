#!/usr/bin/env python3
"""Build a sample-level, prompt-only nuisance table from an extracted study run.

The output deliberately excludes prompts, targets, and responses. It combines the
exact model-tokenizer prompt length recorded by the representation tracker with
deterministic surface features derived from the logged prompt string.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

OPTION_PATTERN = re.compile(r"^[A-Z]\.\s", flags=re.MULTILINE)
TOKEN_PATTERN = re.compile(r"\S+")
OUTPUT_COLUMNS = (
    "model",
    "task_name",
    "sample_id",
    "prompt_token_count",
    "prompt_char_count",
    "prompt_line_count",
    "option_count",
    "whitespace_token_count",
    "whitespace_token_entropy",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _whitespace_entropy(prompt: str) -> tuple[int, float]:
    tokens = TOKEN_PATTERN.findall(prompt)
    if not tokens:
        return 0, 0.0
    counts = Counter(tokens)
    total = float(len(tokens))
    entropy = -sum((count / total) * math.log(count / total) for count in counts.values())
    return len(tokens), float(entropy)


def _sample_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            doc_id = str(row.get("doc_id"))
            if doc_id in rows:
                raise ValueError(f"Duplicate doc_id {doc_id!r} in {path}:{line_number}")
            if not isinstance(row.get("input"), str):
                raise TypeError(f"Missing prompt input in {path}:{line_number}")
            rows[doc_id] = row
    return rows


def build_table(run_roots: Iterable[Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: dict[tuple[str, str, str], dict[str, Any]] = {}
    source_hashes: list[dict[str, str]] = []
    roots = [root.resolve() for root in run_roots]
    if not roots:
        raise ValueError("At least one extracted run root is required")
    records_paths = sorted(
        (root, path)
        for root in roots
        for path in root.rglob("records.csv")
    )
    if not records_paths:
        raise ValueError(f"No records.csv files found under {roots}")

    for run_root, records_path in records_paths:
        representation_dir = records_path.parent
        task_dir = representation_dir.parent.parent
        metadata_path = representation_dir / "metadata.json"
        sample_paths = sorted((task_dir / "evaluation").rglob("*_samples_*.jsonl"))
        if not metadata_path.is_file() or len(sample_paths) != 1:
            raise ValueError(
                f"Expected one metadata file and one sample log for {records_path}; "
                f"found metadata={metadata_path.is_file()}, sample_logs={len(sample_paths)}"
            )
        metadata_payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata = metadata_payload.get("metadata", metadata_payload)
        model = str(metadata.get("pretrained") or metadata.get("model"))
        samples = _sample_rows(sample_paths[0])
        source_hashes.extend(
            (
                {"path": str(records_path.relative_to(run_root)), "sha256": _sha256(records_path)},
                {"path": str(sample_paths[0].relative_to(run_root)), "sha256": _sha256(sample_paths[0])},
                {"path": str(metadata_path.relative_to(run_root)), "sha256": _sha256(metadata_path)},
            )
        )

        with records_path.open("r", encoding="utf-8", newline="") as handle:
            for line_number, record in enumerate(csv.DictReader(handle), start=2):
                sample_id = str(record["sample_id"])
                task_name = str(record["task_name"])
                key = (model, task_name, sample_id)
                doc_id = sample_id.rsplit(":", 1)[-1]
                if doc_id not in samples:
                    raise ValueError(f"No doc_id {doc_id!r} for {records_path}:{line_number}")
                prompt = str(samples[doc_id]["input"])
                whitespace_count, whitespace_entropy = _whitespace_entropy(prompt)
                row = {
                    "model": model,
                    "task_name": task_name,
                    "sample_id": sample_id,
                    "prompt_token_count": int(record["prompt_length"]),
                    "prompt_char_count": len(prompt),
                    "prompt_line_count": prompt.count("\n") + 1,
                    "option_count": len(OPTION_PATTERN.findall(prompt)),
                    "whitespace_token_count": whitespace_count,
                    "whitespace_token_entropy": f"{whitespace_entropy:.17g}",
                }
                previous = output.get(key)
                if previous is not None and previous != row:
                    raise ValueError(f"Inconsistent nuisance values across activation rows for {key}")
                output[key] = row

    rows = [output[key] for key in sorted(output)]
    report = {
        "schema_version": 1,
        "run_roots": [str(root) for root in roots],
        "num_samples": len(rows),
        "num_record_files": len(records_paths),
        "models": sorted({row["model"] for row in rows}),
        "tasks": sorted({row["task_name"] for row in rows}),
        "columns": list(OUTPUT_COLUMNS),
        "feature_scope": "prompt-only surface features; no targets or generated responses",
        "entropy_definition": "natural-log Shannon entropy of case-sensitive whitespace-delimited prompt tokens",
        "source_hashes": source_hashes,
    }
    return rows, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_roots",
        nargs="+",
        type=Path,
        help="One or more extracted representation experiment roots.",
    )
    parser.add_argument("--output", required=True, type=Path, help="Destination CSV.")
    args = parser.parse_args()

    rows, report = build_table(args.run_roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report_path = args.output.with_suffix(".json")
    report["output_sha256"] = _sha256(args.output)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(args.output), "report": str(report_path), "samples": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
