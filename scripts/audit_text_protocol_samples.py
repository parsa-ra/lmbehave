#!/usr/bin/env python3
"""Audit logged direct-answer prompts before using representation artifacts."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

CHOICE_TASK_RULES: dict[str, tuple[int, int]] = {
    "repdyn_arc_challenge_direct": (4, 4),
    "repdyn_hellaswag_direct": (4, 4),
    "repdyn_musr_murder_mysteries": (2, 2),
    "repdyn_super_gpqa_direct": (2, 26),
}
CHOICE_PREFIXES = ("repdyn_mmlu_pro_",)
HEADER_PATTERN = re.compile(r"Return only one letter: ([A-Z](?:, [A-Z])*)\.$")
ROW_PATTERN = re.compile(r"^([A-Z])\. (.*)$")


def _task_name(path: Path) -> str:
    marker = "_samples_"
    if marker not in path.stem:
        raise ValueError(f"Cannot recover task name from {path}")
    return path.stem.split(marker, 1)[1]


def _choice_rule(task: str) -> tuple[int, int] | None:
    if task in CHOICE_TASK_RULES:
        return CHOICE_TASK_RULES[task]
    if task.startswith(CHOICE_PREFIXES):
        return (2, 26)
    return None


def _audit_choice_prompt(task: str, sample: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    prompt = str(sample.get("input", ""))
    lines = prompt.splitlines()
    header = HEADER_PATTERN.search(lines[0]) if lines else None
    if header is None:
        return ["missing or malformed choice header"]
    declared = header.group(1).split(", ")
    expected_labels = [chr(ord("A") + index) for index in range(len(declared))]
    # A question stem can itself contain a quoted line such as "A. J. P. Taylor".
    # The adapter appends the actual choices immediately before the final Answer line,
    # so audit the final declared-size sequence rather than every label-like line.
    matches = [ROW_PATTERN.fullmatch(line) for line in lines]
    matches = [match for match in matches if match is not None]
    rows = matches[-len(declared) :]
    row_labels = [match.group(1) for match in rows]
    row_values = [match.group(2).strip() for match in rows]
    minimum, maximum = _choice_rule(task) or (2, 26)
    if not minimum <= len(declared) <= maximum:
        errors.append(f"declared choice count {len(declared)} outside [{minimum}, {maximum}]")
    if declared != expected_labels:
        errors.append("choice header is not a contiguous A-based sequence")
    if row_labels != declared:
        errors.append("rendered choice rows do not match the declared labels")
    if any(not value for value in row_values):
        errors.append("one or more rendered choices are empty")
    target = str(sample.get("target", "")).strip().upper()
    if target not in declared:
        errors.append(f"target {target!r} is outside the declared choices")
    return errors


def audit(root: Path) -> dict[str, Any]:
    files = sorted(root.rglob("*_samples_*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No logged sample files found below {root}")
    failures: list[dict[str, Any]] = []
    file_counts: Counter[str] = Counter()
    sample_counts: Counter[str] = Counter()
    option_counts: dict[str, Counter[int]] = defaultdict(Counter)
    for path in files:
        task = _task_name(path)
        file_counts[task] += 1
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                sample = json.loads(line)
                sample_counts[task] += 1
                rule = _choice_rule(task)
                prompt_lines = str(sample.get("input", "")).splitlines()
                has_choice_header = bool(prompt_lines and HEADER_PATTERN.search(prompt_lines[0]))
                if rule is None and not has_choice_header:
                    continue
                errors = _audit_choice_prompt(task, sample)
                if not errors:
                    header = HEADER_PATTERN.search(str(sample["input"]).splitlines()[0])
                    assert header is not None
                    option_counts[task][len(header.group(1).split(", "))] += 1
                else:
                    failures.append(
                        {
                            "file": str(path),
                            "line": line_number,
                            "task": task,
                            "doc_id": sample.get("doc_id"),
                            "errors": errors,
                        }
                    )
    return {
        "schema_version": 1,
        "status": "PASS" if not failures else "FAIL",
        "root": str(root.resolve()),
        "sample_files": len(files),
        "files_per_task": dict(sorted(file_counts.items())),
        "samples_per_task": dict(sorted(sample_counts.items())),
        "valid_option_counts": {
            task: {str(count): frequency for count, frequency in sorted(counts.items())}
            for task, counts in sorted(option_counts.items())
        },
        "failure_count": len(failures),
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Run root containing lmms-eval logged sample JSONL files")
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--allow-failures", action="store_true", help="Report failures without a nonzero exit")
    args = parser.parse_args()
    report = audit(args.root)
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["status"] == "PASS" or args.allow_failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
