#!/usr/bin/env python3
"""Run the fold-local prompt-surface sensitivity analysis on frozen artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LMMS_ROOT = ROOT / "lmms-eval"
if str(LMMS_ROOT) not in sys.path:
    sys.path.insert(0, str(LMMS_ROOT))

from lmms_eval.representations.analysis import (  # noqa: E402
    analyze_artifacts,
    write_analysis,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_names(panel: dict[str, Any], label: str) -> set[str]:
    return {str(item["task"]) for item in panel[f"{label}_tasks"]}


def _artifacts(run_root: Path, panel_name: str, model_key: str) -> list[Path]:
    paths = sorted((run_root / panel_name / model_key / "tasks").rglob("representations.pt"))
    if len(paths) != 10:
        raise ValueError(
            f"Expected 10 task artifacts for {panel_name}/{model_key}, found {len(paths)}"
        )
    return paths


def _environment_versions(backend: str) -> dict[str, str]:
    import numpy
    import sklearn
    import torch

    versions = {
        "numpy": str(numpy.__version__),
        "scikit_learn": str(sklearn.__version__),
        "torch": str(torch.__version__),
    }
    if backend == "cuml":
        import cuml
        import cupy

        versions.update({"cupy": str(cupy.__version__), "cuml": str(cuml.__version__)})
    return versions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path, help="Extracted current-study run root.")
    parser.add_argument("--nuisance-table", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "experiments" / "representation_dynamics_text_expansion_v1.json",
    )
    parser.add_argument(
        "--sensitivity-manifest",
        type=Path,
        default=ROOT / "experiments" / "representation_dynamics_nuisance_control_v1.json",
    )
    parser.add_argument("--models", default="", help="Optional comma-separated model keys.")
    parser.add_argument(
        "--panels",
        default="matched_text_mmlu_pro_control",
        help="Comma-separated valid panels; defaults to the matched panel until MuSR is corrected.",
    )
    parser.add_argument("--backend", choices=("cuml", "sklearn"), default="cuml")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Concurrent layer/stream workers. cuML jobs >1 require a caller-managed MPS service.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    sensitivity = json.loads(args.sensitivity_manifest.read_text(encoding="utf-8"))
    selected_models = (
        [item.strip() for item in args.models.split(",") if item.strip()]
        if args.models
        else list(manifest["models"])
    )
    unknown = set(selected_models) - set(manifest["models"])
    if unknown:
        raise ValueError(f"Unknown model keys: {sorted(unknown)}")
    selected_panels = [item.strip() for item in args.panels.split(",") if item.strip()]
    unknown_panels = set(selected_panels) - set(manifest["panels"])
    if not selected_panels or unknown_panels:
        raise ValueError(f"Invalid panel selection; unknown={sorted(unknown_panels)}")

    analysis = manifest["analysis"]
    nuisance_columns = tuple(sensitivity["nuisance_features"])
    nuisance_alpha = float(sensitivity["residualizer"]["alpha"])
    parent_manifest_sha256 = _sha256(args.manifest)
    sensitivity_manifest_sha256 = _sha256(args.sensitivity_manifest)
    nuisance_table_sha256 = _sha256(args.nuisance_table)
    environment_versions = _environment_versions(args.backend)
    completed: list[dict[str, Any]] = []
    for panel_name in selected_panels:
        panel = manifest["panels"][panel_name]
        knowledge_tasks = _task_names(panel, "knowledge")
        reasoning_tasks = _task_names(panel, "reasoning")
        for model_key in selected_models:
            output_dir = args.output_root / panel_name / model_key
            artifact_paths = _artifacts(args.run_root, panel_name, model_key)
            artifact_hashes = [
                {"path": str(path), "sha256": _sha256(path)} for path in artifact_paths
            ]
            if args.resume and (output_dir / "report.json").is_file():
                analysis_manifest_path = output_dir / "analysis_manifest.json"
                if not analysis_manifest_path.is_file():
                    raise ValueError(f"Cannot resume without {analysis_manifest_path}")
                previous = json.loads(analysis_manifest_path.read_text(encoding="utf-8"))
                expected = {
                    "backend": args.backend,
                    "jobs": args.jobs,
                    "parent_manifest_sha256": parent_manifest_sha256,
                    "sensitivity_manifest_sha256": sensitivity_manifest_sha256,
                    "nuisance_table_sha256": nuisance_table_sha256,
                    "artifacts": artifact_hashes,
                }
                mismatched = [key for key, value in expected.items() if previous.get(key) != value]
                if mismatched:
                    raise ValueError(
                        f"Refusing stale resume for {panel_name}/{model_key}; mismatched {mismatched}"
                    )
                previous_report = json.loads((output_dir / "report.json").read_text(encoding="utf-8"))
                if previous_report.get("failures"):
                    raise ValueError(f"Cannot resume failed condition {panel_name}/{model_key}")
                completed.append(
                    {"panel": panel_name, "model": model_key, "status": "existing"}
                )
                continue
            rows, report = analyze_artifacts(
                artifact_paths,
                knowledge_tasks=knowledge_tasks,
                reasoning_tasks=reasoning_tasks,
                phase="prefill",
                cv_folds=int(analysis["cv_folds"]),
                permutations=int(sensitivity["inference"]["linear_task_label_permutations"]),
                nonlinear_permutations=int(
                    sensitivity["inference"]["nonlinear_task_label_permutations"]
                ),
                pca_variance=float(analysis["pca_retained_variance"]),
                silhouette_max_samples=int(analysis["silhouette_max_samples"]),
                seed=int(sensitivity["inference"]["seed"]),
                jobs=args.jobs,
                parallel_backend="process",
                analysis_backend=args.backend,
                nuisance_table=args.nuisance_table,
                nuisance_columns=nuisance_columns,
                nuisance_alpha=nuisance_alpha,
            )
            outputs = write_analysis(rows, report, output_dir)
            analysis_manifest = {
                "schema_version": 1,
                "panel": panel_name,
                "model_key": model_key,
                "backend": args.backend,
                "jobs": args.jobs,
                "environment_versions": environment_versions,
                "parent_manifest": str(args.manifest),
                "parent_manifest_sha256": parent_manifest_sha256,
                "sensitivity_manifest": str(args.sensitivity_manifest),
                "sensitivity_manifest_sha256": sensitivity_manifest_sha256,
                "nuisance_table": str(args.nuisance_table),
                "nuisance_table_sha256": nuisance_table_sha256,
                "artifacts": artifact_hashes,
                "outputs": {name: str(path) for name, path in outputs.items()},
                "num_rows": len(rows),
                "num_failures": len(report["failures"]),
            }
            if report["failures"]:
                raise RuntimeError(
                    f"Analysis produced {len(report['failures'])} failures for "
                    f"{panel_name}/{model_key}"
                )
            (output_dir / "analysis_manifest.json").write_text(
                json.dumps(analysis_manifest, indent=2) + "\n", encoding="utf-8"
            )
            completed.append(
                {
                    "panel": panel_name,
                    "model": model_key,
                    "status": "completed",
                    "rows": len(rows),
                }
            )
            if args.backend == "cuml":
                import cupy as cp

                cp.get_default_memory_pool().free_all_blocks()
                cp.get_default_pinned_memory_pool().free_all_blocks()

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_suffix = "-".join(selected_models)
    (args.output_root / f"run_summary-{summary_suffix}.json").write_text(
        json.dumps({"conditions": completed}, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"conditions": completed}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
