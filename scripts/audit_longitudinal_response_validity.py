#!/usr/bin/env python3
"""Audit parser-validity rates for the two released training trajectories."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path, PurePosixPath
from typing import Iterable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows_from_lines(lines: Iterable[str]) -> tuple[int, int, int]:
    examples = valid = correct = 0
    for line in lines:
        row = json.loads(line)
        examples += 1
        parsed = row.get("filtered_resps")
        if isinstance(parsed, list) and len(parsed) == 1:
            parsed = parsed[0]
        is_valid = isinstance(parsed, str) and parsed.strip() in {"A", "B", "C", "D"}
        is_correct = float(row.get("exact_match", 0.0)) > 0.5
        if is_correct and not is_valid:
            raise ValueError("An invalid parser output was scored correct")
        valid += int(is_valid)
        correct += int(is_correct)
    return examples, valid, correct


def add_cell(cells: dict, key: tuple[str, str, str], counts: tuple[int, int, int]) -> None:
    target = cells[key]
    for index, value in enumerate(counts):
        target[index] += value


def audit(olmo_archive: Path, apertus_root: Path) -> tuple[list[dict], dict]:
    cells: dict[tuple[str, str, str], list[int]] = defaultdict(lambda: [0, 0, 0])
    with tarfile.open(olmo_archive, "r:gz") as archive:
        for member in archive:
            if not member.isfile() or "_samples_" not in member.name or not member.name.endswith(".jsonl"):
                continue
            parts = PurePosixPath(member.name).parts
            checkpoint_index = parts.index("checkpoints") + 1
            checkpoint, block = parts[checkpoint_index], parts[checkpoint_index + 1]
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"Could not read {member.name}")
            lines = io.TextIOWrapper(extracted, encoding="utf-8")
            add_cell(cells, ("OLMo 3 7B", checkpoint, block), rows_from_lines(lines))

    for path in sorted(apertus_root.rglob("*_samples_*.jsonl")):
        relative = path.relative_to(apertus_root)
        checkpoint_index = relative.parts.index("checkpoints") + 1
        checkpoint, block = relative.parts[checkpoint_index], relative.parts[checkpoint_index + 1]
        add_cell(
            cells,
            ("Apertus 1.0 8B", checkpoint, block),
            rows_from_lines(path.read_text(encoding="utf-8").splitlines()),
        )

    output = []
    for (family, checkpoint, block), (examples, valid, correct) in sorted(cells.items()):
        output.append(
            {
                "family": family,
                "checkpoint": checkpoint,
                "block": block,
                "examples": examples,
                "valid_responses": valid,
                "valid_response_rate": valid / examples,
                "raw_accuracy": correct / examples,
                "accuracy_conditional_on_valid_response": correct / valid if valid else None,
            }
        )
    return output, {
        "schema_version": 1,
        "scope": (
            "Historical logs retain task-filtered responses but not raw generated text; "
            "validity means that the stored parser output is A, B, C, or D."
        ),
        "sources": {
            "olmo_archive": str(olmo_archive.resolve()),
            "olmo_archive_sha256": sha256(olmo_archive),
            "apertus_root": str(apertus_root.resolve()),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--olmo-archive", required=True, type=Path)
    parser.add_argument("--apertus-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    rows, report = audit(args.olmo_archive.resolve(), args.apertus_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report.update(
        {
            "status": "PASS",
            "rows": len(rows),
            "output": str(args.output.resolve()),
            "output_sha256": sha256(args.output),
        }
    )
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"csv": str(args.output), "report": str(report_path), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
