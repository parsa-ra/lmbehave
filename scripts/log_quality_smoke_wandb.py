#!/usr/bin/env python3
"""Log one aggregate-only W&B summary after a validated smoke or full run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import wandb


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Required file is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_values(path: Path, column: str) -> list[float]:
    with path.open(newline="", encoding="utf-8") as handle:
        values = []
        for row in csv.DictReader(handle):
            try:
                value = float(row[column])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)
        return values


def single_value(values: set, label: str, prefix: str):
    if len(values) != 1:
        raise RuntimeError(
            f"Expected one {label} for {prefix}, found {sorted(values, key=str)}"
        )
    return next(iter(values))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    parser.add_argument(
        "--project",
        default=os.environ.get("WANDB_PROJECT"),
        help="Required for upload; defaults only to the explicit WANDB_PROJECT environment variable.",
    )
    parser.add_argument("--entity")
    parser.add_argument("--name", default="quality-smoke-aggregate")
    parser.add_argument(
        "--kind",
        choices=("quality-smoke", "full-experiment"),
        default="quality-smoke",
    )
    parser.add_argument("--minimum-samples-per-task", type=int, default=32)
    parser.add_argument(
        "--analysis-name",
        default="analysis",
        help="Per-model analysis directory basename.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the exact payload without contacting W&B",
    )
    args = parser.parse_args()
    if Path(args.analysis_name).name != args.analysis_name or args.analysis_name in {"", ".", ".."}:
        raise RuntimeError("--analysis-name must be one non-empty path basename")
    run_root = args.run_root.resolve()

    run_manifest = load_json(run_root / "run_manifest.json")
    completed = load_json(run_root / "completed_manifest.json")
    validation = load_json(run_root / "validation_report.json")
    validations = validation.get("validations", [])
    if validation.get("status") != "PASS" or any(
        row.get("status") != "PASS" for row in validations
    ):
        raise RuntimeError("Artifact validation did not pass; refusing W&B upload")
    sample_count = int(run_manifest.get("effective_sample_count_per_task", 0))
    if sample_count < args.minimum_samples_per_task:
        raise RuntimeError(
            f"Run has {sample_count} samples/task; minimum quality-smoke threshold is "
            f"{args.minimum_samples_per_task}"
        )
    manifest_sample_count = int(
        run_manifest["resolved_experiment"]["extraction"]["sample_count_per_task"]
    )
    if args.kind == "full-experiment" and sample_count != manifest_sample_count:
        raise RuntimeError(
            f"Full experiment requires the manifest sample count {manifest_sample_count}; "
            f"run contains {sample_count}"
        )
    for repository in ("root_git", "lmms_eval_git"):
        if run_manifest[repository]["dirty"]:
            raise RuntimeError(
                f"Extraction source {repository} was dirty; refusing W&B upload"
            )
    run_manifest_sha256 = sha256_file(run_root / "run_manifest.json")
    completed_manifest_sha256 = sha256_file(run_root / "completed_manifest.json")
    if completed.get("run_manifest_sha256") != run_manifest_sha256:
        raise RuntimeError("Completed manifest does not bind the current run manifest")

    artifacts = completed.get("artifacts", [])
    if not artifacts or len(artifacts) != len(validations):
        raise RuntimeError(
            f"Completed/validated artifact count mismatch: {len(artifacts)} versus {len(validations)}"
        )
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for artifact in artifacts:
        grouped[(artifact["panel"], artifact["model_key"])].append(artifact)

    summary: dict[str, float | int | str | bool] = {}
    group_config: dict[str, object] = {}
    probe_validation_scopes: set[str] = set()
    task_label_warnings: set[str] = set()
    for (panel, model_key), rows in sorted(grouped.items()):
        model_dir = run_root / panel / model_key
        analysis_dir = model_dir / args.analysis_name
        report = load_json(analysis_dir / "report.json")
        probe_validation_scopes.add(
            "inductive_fold_local_preprocessing"
            if report.get("probe_pca_fit_scope") == "training_tasks_per_fold"
            else "shared_unlabeled_pca_or_raw"
        )
        panel_spec = run_manifest["resolved_experiment"]["panels"][panel]
        task_label_warnings.add(panel_spec["claim_limit"])
        analysis_manifest = load_json(analysis_dir / "analysis_manifest.json")
        failures = report.get("failures", [])
        if failures:
            raise RuntimeError(
                f"Analysis has {len(failures)} failed rows for {panel}/{model_key}"
            )
        if analysis_manifest.get("run_manifest_sha256") != run_manifest_sha256:
            raise RuntimeError(
                f"Analysis does not bind the run manifest for {panel}/{model_key}"
            )
        if (
            analysis_manifest.get("completed_manifest_sha256")
            != completed_manifest_sha256
        ):
            raise RuntimeError(
                f"Analysis does not bind the completed manifest for {panel}/{model_key}"
            )
        for repository in ("root_git", "lmms_eval_git"):
            if analysis_manifest["analysis_source"][repository]["dirty"]:
                raise RuntimeError(
                    f"Analysis source {repository} was dirty for {panel}/{model_key}"
                )

        prefix = f"{panel}/{model_key}"
        task_metrics: dict[str, dict] = {}
        metadata_rows = []
        for artifact in rows:
            result_path = Path(artifact["result_json"])
            representation_path = Path(artifact["artifact"])
            if not result_path.is_file() or not representation_path.is_file():
                raise RuntimeError(
                    f"Missing validated task artifact for {prefix}/{artifact['task']}"
                )
            result = load_json(result_path)
            task = artifact["task"]
            if task not in result.get("results", {}):
                raise RuntimeError(f"Task {task!r} absent from {result_path}")
            task_metrics[task] = result["results"][task]
            metadata_rows.append(
                load_json(representation_path.parent / "metadata.json")["metadata"]
            )

        for task, metrics in sorted(task_metrics.items()):
            for metric, value in metrics.items():
                if (
                    metric == "alias"
                    or value is None
                    or not isinstance(value, (int, float))
                ):
                    continue
                summary[f"{prefix}/task/{task}/{metric.replace(',', '/')}"] = value

        for metric in (
            "fdr",
            "probe_accuracy",
            "probe_f1",
            "rbf_probe_accuracy",
            "rbf_probe_f1",
            "silhouette_cosine",
            "covariance_alignment",
        ):
            maximum = report["maxima"].get(metric)
            if maximum:
                summary[f"{prefix}/analysis/{metric}/value"] = maximum["value"]
                summary[f"{prefix}/analysis/{metric}/layer"] = maximum["layer"]
                summary[f"{prefix}/analysis/{metric}/stream"] = maximum["stream"]
        for stream, maximum in report[
            "primary_branch_maximum_divergence_by_stream"
        ].items():
            if maximum:
                summary[f"{prefix}/analysis/fdr_{stream}/value"] = maximum["value"]
                summary[f"{prefix}/analysis/fdr_{stream}/layer"] = maximum["layer"]

        csv_path = analysis_dir / "layer_separability.csv"
        expected_outputs = analysis_manifest["outputs"]
        if expected_outputs["layer_separability_csv_sha256"] != sha256_file(csv_path):
            raise RuntimeError(f"Analysis CSV changed after finalization for {prefix}")
        if expected_outputs["report_json_sha256"] != sha256_file(analysis_dir / "report.json"):
            raise RuntimeError(
                f"Analysis report changed after finalization for {prefix}"
            )
        raw_p = finite_values(csv_path, "probe_accuracy_p")
        adjusted_q = finite_values(csv_path, "probe_accuracy_q")
        attainable = finite_values(csv_path, "minimum_attainable_probe_p")
        effective_rank = finite_values(csv_path, "effective_rank_all_fraction")
        within_effective_rank = finite_values(csv_path, "effective_rank_within_fraction")
        if not raw_p or not adjusted_q or not attainable:
            raise RuntimeError(
                f"Missing finite linear-probe inference values for {prefix}"
            )
        summary[f"{prefix}/inference/min_raw_probe_accuracy_p"] = min(raw_p)
        summary[f"{prefix}/inference/min_bh_probe_accuracy_q"] = min(adjusted_q)
        summary[f"{prefix}/inference/min_attainable_p"] = min(attainable)
        if effective_rank:
            summary[f"{prefix}/rank/min_effective_rank_fraction"] = min(effective_rank)
            summary[f"{prefix}/rank/max_effective_rank_fraction"] = max(effective_rank)
        if within_effective_rank:
            summary[f"{prefix}/rank/min_within_effective_rank_fraction"] = min(
                within_effective_rank
            )
            summary[f"{prefix}/rank/max_within_effective_rank_fraction"] = max(
                within_effective_rank
            )
        summary[f"{prefix}/artifact/records"] = sum(int(row["records"]) for row in rows)
        summary[f"{prefix}/artifact/examples"] = sum(
            int(row["sample_count"]) for row in rows
        )
        summary[f"{prefix}/artifact/tasks"] = len(rows)
        summary[f"{prefix}/artifact/layers"] = int(
            single_value({row["layers"] for row in rows}, "layer count", prefix)
        )
        summary[f"{prefix}/artifact/analysis_rows"] = int(report["num_rows"])
        summary[f"{prefix}/artifact/analysis_failures"] = 0
        summary[f"{prefix}/artifact/cka_defined"] = (
            report["maxima"].get("cka") is not None
        )

        group_config[prefix] = {
            "pretrained": single_value(
                {row.get("pretrained") for row in metadata_rows}, "model", prefix
            ),
            "selected_stack": single_value(
                {row["selected_stack"] for row in rows}, "decoder stack", prefix
            ),
            "storage_dtype": single_value(
                {row["storage_dtype"] for row in rows}, "storage dtype", prefix
            ),
            "lmms_eval_git_head": single_value(
                {row["lmms_eval_git_head"] for row in rows},
                "lmms-eval revision",
                prefix,
            ),
            "analysis_source": analysis_manifest["analysis_source"],
            "knowledge_tasks": report["knowledge_tasks"],
            "reasoning_tasks": report["reasoning_tasks"],
            "analysis_settings": analysis_manifest["settings"],
        }

    probe_validation_scope = single_value(
        probe_validation_scopes, "probe validation scope", "aggregate"
    )
    inductive_claim_supported = probe_validation_scope == "inductive_fold_local_preprocessing"
    config = {
        "kind": f"{args.kind.replace('-', '_')}_not_confirmatory_evidence",
        "run_id": run_root.name,
        "experiment_id": run_manifest.get("experiment_id"),
        "effective_sample_count_per_task": sample_count,
        "artifact_validation_passed": True,
        "validated_artifact_count": len(validations),
        "analysis_group_count": len(grouped),
        "raw_samples_prompts_or_activations_uploaded": False,
        "task_label_warnings": sorted(task_label_warnings),
        "inference_warning": "Maxima are exploratory and require controlled, independent replication.",
        "probe_validation_scope": probe_validation_scope,
        "inductive_unseen_task_claim_supported": inductive_claim_supported,
        "required_follow_up": (
            None
            if inductive_claim_supported
            else "Fit PCA within each training fold and rerun all model conditions uniformly."
        ),
        "package_inventory_sha256": run_manifest["environment"][
            "package_inventory_sha256"
        ],
        "root_git": run_manifest["root_git"],
        "lmms_eval_git": run_manifest["lmms_eval_git"],
        "groups": group_config,
    }
    if args.dry_run:
        print(
            json.dumps({"config": config, "summary": summary}, indent=2, sort_keys=True)
        )
        return
    if not args.project:
        raise RuntimeError(
            "Set --project or export WANDB_PROJECT; no implicit W&B project is used"
        )
    run = wandb.init(
        project=args.project,
        entity=args.entity,
        name=args.name,
        job_type=args.kind,
        config=config,
        save_code=False,
    )
    run.log(summary)
    run.summary.update(summary)
    run.finish()
    print(run.url)


if __name__ == "__main__":
    main()
