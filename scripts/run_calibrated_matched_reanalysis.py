#!/usr/bin/env python3
"""Reanalyze the corrected five-model matched panel with an exact Fisher null.

The script consumes the immutable extraction artifacts already archived on persistent
storage.  It never reloads a model or changes an evaluation sample.  Apertus 1.5 is read
from its corrected extension run; the superseded Apertus 1.0 directory is never used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LMMS_ROOT = ROOT / "lmms-eval"
if str(LMMS_ROOT) not in sys.path:
    sys.path.insert(0, str(LMMS_ROOT))

from lmms_eval.representations.analysis import analyze_artifacts, write_analysis  # noqa: E402


PANEL = "matched_text_mmlu_pro_control"
MODEL_SOURCES = {
    "olmo3-7b-instruct": "base",
    "apertus15-8b": "apertus15",
    "qwen3_5-9b": "base",
    "qwen3_5-27b": "base",
    "gemma4-31b": "base",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_state(path: Path) -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"], text=True
        ).strip()
    )
    tracked_diff = subprocess.check_output(["git", "-C", str(path), "diff", "--binary", "HEAD"])
    return {
        "head": head,
        "dirty": dirty,
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
    }


def _versions(backend: str) -> dict[str, str]:
    import numpy
    import sklearn
    import torch

    versions = {
        "python": platform.python_version(),
        "numpy": str(numpy.__version__),
        "scikit_learn": str(sklearn.__version__),
        "torch": str(torch.__version__),
    }
    if backend == "cuml":
        import cuml
        import cupy

        versions.update({"cuml": str(cuml.__version__), "cupy": str(cupy.__version__)})
    return versions


def _expected_hashes(completion_path: Path, model_key: str) -> dict[str, str]:
    payload = json.loads(completion_path.read_text(encoding="utf-8"))
    entries = payload.get("artifacts", payload.get("conditions", []))
    expected: dict[str, str] = {}
    for item in entries:
        if str(item.get("model_key", item.get("model", ""))) != model_key:
            continue
        if str(item.get("panel", PANEL)) != PANEL:
            continue
        artifact = item.get("artifact") or item.get("artifact_path")
        digest = item.get("artifact_sha256") or item.get("sha256")
        if artifact and digest:
            expected[Path(str(artifact)).name + ":" + str(item.get("task", ""))] = str(digest)
    return expected


def _artifacts(run_root: Path, model_key: str) -> list[Path]:
    paths = sorted((run_root / PANEL / model_key / "tasks").rglob("representations.pt"))
    if len(paths) != 10:
        raise ValueError(f"Expected 10 artifacts for {model_key}, found {len(paths)} in {run_root}")
    return paths


def _task_set(panel: dict[str, Any], label: str) -> set[str]:
    return {str(item["task"]) for item in panel[f"{label}_tasks"]}


def main() -> int:
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
        "--output-root",
        type=Path,
        default=ROOT / "logs" / "main-track-fixes-v1" / "calibrated-matched-panel",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiments" / "representation_dynamics_text_expansion_v1.json",
    )
    parser.add_argument(
        "--models", default=",".join(MODEL_SOURCES), help="Comma-separated model keys."
    )
    parser.add_argument("--backend", choices=("cuml", "sklearn"), default="cuml")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--linear-permutations", type=int, default=0)
    parser.add_argument("--fisher-permutations", type=int, default=251)
    parser.add_argument("--nonlinear-permutations", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    selected_models = [item.strip() for item in args.models.split(",") if item.strip()]
    unknown = set(selected_models) - set(MODEL_SOURCES)
    if not selected_models or unknown:
        raise ValueError(f"Invalid model selection; unknown={sorted(unknown)}")

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    panel = manifest["panels"][PANEL]
    analysis = manifest["analysis"]
    knowledge_tasks = _task_set(panel, "knowledge")
    reasoning_tasks = _task_set(panel, "reasoning")
    reference_positive_tasks = set(sorted(knowledge_tasks | reasoning_tasks)[:5])
    roots = {"base": args.base_run_root, "apertus15": args.apertus15_run_root}
    completion_files = {
        "base": args.base_run_root / "completed_manifest.json",
        "apertus15": args.apertus15_run_root / "completed_manifest.json",
    }
    for path in completion_files.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output_root.mkdir(parents=True, exist_ok=True)
    common_provenance = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Exact Fisher calibration of the corrected five-model matched panel.",
        "claim_limit": (
            "This reuses the original five-versus-five tasks and examples. It calibrates "
            "Fisher rows and the selected maximum but does not add inferential units or "
            "repair the limited power of the ten-task design."
        ),
        "manifest": str(args.manifest),
        "manifest_sha256": _sha256(args.manifest),
        "root_git": _git_state(ROOT),
        "lmms_eval_git": _git_state(LMMS_ROOT),
        "analysis_source_sha256": _sha256(
            LMMS_ROOT / "lmms_eval" / "representations" / "analysis.py"
        ),
        "runner_source_sha256": _sha256(Path(__file__).resolve()),
        "environment": _versions(args.backend),
        "settings": {
            "backend": args.backend,
            "jobs": args.jobs,
            "linear_permutations": args.linear_permutations,
            "fisher_permutations": args.fisher_permutations,
            "nonlinear_permutations": args.nonlinear_permutations,
            "seed": int(analysis["seed"]),
        },
    }
    completed: list[dict[str, Any]] = []
    for model_key in selected_models:
        source = MODEL_SOURCES[model_key]
        root = roots[source]
        output_dir = args.output_root / model_key
        artifact_paths = _artifacts(root, model_key)
        artifact_records = [
            {"path": str(path), "sha256": _sha256(path)} for path in artifact_paths
        ]
        # Completion manifests are independently checked even though their old paths refer
        # to the original machine location. Match by task and digest, not absolute path.
        expected = set(_expected_hashes(completion_files[source], model_key).values())
        actual = {item["sha256"] for item in artifact_records}
        if len(expected) != 10 or actual != expected:
            raise ValueError(f"Completion-manifest hash mismatch for {model_key}")

        provenance_path = output_dir / "analysis_manifest.json"
        if args.resume and (output_dir / "report.json").is_file():
            if not provenance_path.is_file():
                raise ValueError(f"Cannot resume without {provenance_path}")
            old = json.loads(provenance_path.read_text(encoding="utf-8"))
            if old.get("artifacts") != artifact_records or old.get("settings") != common_provenance["settings"]:
                raise ValueError(f"Refusing stale resume for {model_key}")
            completed.append({"model": model_key, "status": "existing"})
            continue

        rows, report = analyze_artifacts(
            artifact_paths,
            knowledge_tasks=knowledge_tasks,
            reasoning_tasks=reasoning_tasks,
            phase="prefill",
            cv_folds=int(analysis["cv_folds"]),
            permutations=args.linear_permutations,
            fdr_permutations=args.fisher_permutations,
            nonlinear_permutations=args.nonlinear_permutations,
            pca_variance=float(analysis["pca_retained_variance"]),
            silhouette_max_samples=int(analysis["silhouette_max_samples"]),
            seed=int(analysis["seed"]),
            jobs=args.jobs,
            parallel_backend="process",
            analysis_backend=args.backend,
            reference_positive_tasks=reference_positive_tasks,
        )
        outputs = write_analysis(rows, report, output_dir)
        if report["failures"]:
            raise RuntimeError(f"{model_key} produced {len(report['failures'])} failed rows")
        condition_provenance = {
            **common_provenance,
            "model_key": model_key,
            "source_run_root": str(root),
            "source_completion_manifest": str(completion_files[source]),
            "source_completion_manifest_sha256": _sha256(completion_files[source]),
            "artifacts": artifact_records,
            "rows": len(rows),
            "outputs": {name: {"path": str(path), "sha256": _sha256(path)} for name, path in outputs.items()},
            "settings": common_provenance["settings"],
        }
        provenance_path.write_text(
            json.dumps(condition_provenance, indent=2) + "\n", encoding="utf-8"
        )
        completed.append({"model": model_key, "status": "completed", "rows": len(rows)})
        if args.backend == "cuml":
            import cupy as cp

            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()

    summary = {**common_provenance, "conditions": completed}
    summary_path = args.output_root / "run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output_root), "conditions": completed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
