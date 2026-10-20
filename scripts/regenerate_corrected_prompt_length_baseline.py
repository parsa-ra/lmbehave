#!/usr/bin/env python3
"""Regenerate the scalar prompt-length diagnostic for the corrected current panel."""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lmms-eval"))
sys.path.insert(0, str(ROOT / "scripts"))

from summarize_representation_experiment import prompt_length_probe, unique_prompt_lengths  # noqa: E402


PANEL = "matched_text_mmlu_pro_control"
SOURCES = {
    "olmo3-7b-instruct": "base",
    "apertus15-8b": "apertus15",
    "qwen3_5-9b": "base",
    "qwen3_5-27b": "base",
    "gemma4-31b": "base",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-run-root",
        type=Path,
        default=ROOT / "logs" / "representation-dynamics-text-expansion-v1-v2",
    )
    parser.add_argument(
        "--apertus15-run-root",
        type=Path,
        default=ROOT / "logs" / "apertus15-current-panel-v1",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiments" / "representation_dynamics_text_expansion_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "representation_dynamics_text_expansion_corrected_v1",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    panel = manifest["panels"][PANEL]
    class_by_task = {
        str(item["task"]): label
        for label in ("knowledge", "reasoning")
        for item in panel[f"{label}_tasks"]
    }
    roots = {"base": args.base_run_root, "apertus15": args.apertus15_run_root}
    rows: list[dict[str, object]] = []
    sources: list[dict[str, object]] = []
    for model, source in SOURCES.items():
        artifact_paths = sorted((roots[source] / PANEL / model / "tasks").rglob("representations.pt"))
        if len(artifact_paths) != 10:
            raise ValueError(f"Expected 10 corrected matched-panel artifacts for {model}")
        samples: list[tuple[str, str, int]] = []
        for artifact_path in artifact_paths:
            records_path = artifact_path.parent / "records.csv"
            if not records_path.is_file():
                raise FileNotFoundError(records_path)
            samples.extend(unique_prompt_lengths(records_path))
            sources.append(
                {
                    "model": model,
                    "artifact": str(artifact_path),
                    "artifact_sha256": sha256(artifact_path),
                    "records": str(records_path),
                    "records_sha256": sha256(records_path),
                }
            )
        if len(samples) != 1000 or len({sample_id for sample_id, _task, _length in samples}) != 1000:
            raise ValueError(f"Expected 1,000 unique examples for {model}")
        rows.append(
            {
                "panel": PANEL,
                "model": model,
                **prompt_length_probe(samples, class_by_task, int(manifest["analysis"]["seed"])),
            }
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "prompt_length_baseline.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    report_path = args.output_dir / "prompt_length_baseline_report.json"
    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "corrected_current_model_matched_panel",
        "models": list(SOURCES),
        "inferential_unit": "whole benchmark task",
        "feature": "scalar prompt token count",
        "claim_limit": (
            "This post-hoc scalar diagnostic tests only linear predictability from prompt "
            "token count. It neither removes prompt-surface confounding nor identifies a mechanism."
        ),
        "manifest": str(args.manifest),
        "manifest_sha256": sha256(args.manifest),
        "sources": sources,
        "output": {"path": str(output_path), "sha256": sha256(output_path)},
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output_path), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
