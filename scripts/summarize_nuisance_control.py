#!/usr/bin/env python3
"""Create sample-free summaries of the completed nuisance-control analysis."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments" / "representation_dynamics_text_expansion_v1.json"
NUISANCE_COLUMNS = (
    "prompt_token_count",
    "prompt_char_count",
    "prompt_line_count",
    "option_count",
    "whitespace_token_count",
    "whitespace_token_entropy",
)


def _environment_versions() -> dict[str, str]:
    import cuml
    import cupy
    import numpy
    import sklearn
    import torch

    return {
        "cupy": str(cupy.__version__),
        "cuml": str(cuml.__version__),
        "numpy": str(numpy.__version__),
        "scikit_learn": str(sklearn.__version__),
        "torch": str(torch.__version__),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty summary: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def select_models(manifest: dict[str, Any], requested: list[str] | None) -> list[str]:
    available = list(manifest["models"])
    selected = list(requested) if requested is not None else available
    if not selected:
        raise ValueError("At least one model must be selected")
    if len(selected) != len(set(selected)):
        raise ValueError(f"Duplicate model selections are not allowed: {selected}")
    unknown = sorted(set(selected) - set(available))
    if unknown:
        raise ValueError(f"Unknown model selections: {unknown}")
    return selected


def _probe_counts(rows: list[dict[str, str]], prefix: str, adjusted_prefix: str) -> dict[str, Any]:
    base_q = [float(row[f"{prefix}_f1_q"]) for row in rows]
    adjusted_q = [float(row[f"{adjusted_prefix}_f1_q"]) for row in rows]
    base_f1 = [float(row[f"{prefix}_f1"]) for row in rows]
    adjusted_f1 = [float(row[f"{adjusted_prefix}_f1"]) for row in rows]
    base_significant = [value < 0.05 for value in base_q]
    adjusted_significant = [value < 0.05 for value in adjusted_q]
    deltas = [adjusted - base for adjusted, base in zip(adjusted_f1, base_f1)]
    return {
        "base_significant": sum(base_significant),
        "adjusted_significant": sum(adjusted_significant),
        "base_minimum_q": min(base_q),
        "adjusted_minimum_q": min(adjusted_q),
        "persist_significant": sum(a and b for a, b in zip(base_significant, adjusted_significant)),
        "lost_significant": sum(a and not b for a, b in zip(base_significant, adjusted_significant)),
        "gained_significant": sum(not a and b for a, b in zip(base_significant, adjusted_significant)),
        "mean_f1_delta": statistics.fmean(deltas),
        "median_f1_delta": statistics.median(deltas),
    }


def _pearson(left: list[float], right: list[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("Pearson correlation requires equal vectors with at least two values")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_centered = [value - left_mean for value in left]
    right_centered = [value - right_mean for value in right]
    denominator = math.sqrt(
        sum(value * value for value in left_centered)
        * sum(value * value for value in right_centered)
    )
    if denominator == 0.0:
        return math.nan
    return sum(a * b for a, b in zip(left_centered, right_centered)) / denominator


def profile_replication_rows(
    condition_rows: dict[tuple[str, str], list[dict[str, str]]],
    panel_names: list[str],
    model_keys: list[str],
) -> list[dict[str, Any]]:
    if len(panel_names) != 2:
        return []
    left_panel, right_panel = panel_names
    output: list[dict[str, Any]] = []
    for model_key in model_keys:
        left_rows = condition_rows[(model_key, left_panel)]
        right_rows = condition_rows[(model_key, right_panel)]
        left_max = max(left_rows, key=lambda row: float(row["fdr"]))
        right_max = max(right_rows, key=lambda row: float(row["fdr"]))
        for stream in ("attention_delta", "mlp_delta", "residual_state"):
            left = sorted(
                (row for row in left_rows if row["stream"] == stream),
                key=lambda row: int(row["layer"]),
            )
            right = sorted(
                (row for row in right_rows if row["stream"] == stream),
                key=lambda row: int(row["layer"]),
            )
            left_layers = [int(row["layer"]) for row in left]
            right_layers = [int(row["layer"]) for row in right]
            if left_layers != right_layers or len(left_layers) != len(set(left_layers)):
                raise ValueError(
                    f"Incompatible FDR profiles for {model_key}/{stream}: "
                    f"{left_layers} versus {right_layers}"
                )
            output.append(
                {
                    "model_key": model_key,
                    "stream": stream,
                    "left_panel": left_panel,
                    "right_panel": right_panel,
                    "layers": len(left_layers),
                    "fdr_profile_pearson_r": _pearson(
                        [float(row["fdr"]) for row in left],
                        [float(row["fdr"]) for row in right],
                    ),
                    "overall_maximum_site_matches": (
                        left_max["layer"] == right_max["layer"]
                        and left_max["stream"] == right_max["stream"]
                    ),
                    "left_fdr_max_layer": int(left_max["layer"]),
                    "left_fdr_max_stream": left_max["stream"],
                    "right_fdr_max_layer": int(right_max["layer"]),
                    "right_fdr_max_stream": right_max["stream"],
                }
            )
    return output


def summarize_condition(
    output_root: Path,
    panel_name: str,
    model_key: str,
) -> tuple[dict[str, Any], list[dict[str, str]], list[dict[str, str]]]:
    condition_dir = output_root / panel_name / model_key
    csv_path = condition_dir / "layer_separability.csv"
    report_path = condition_dir / "report.json"
    analysis_manifest_path = condition_dir / "analysis_manifest.json"
    rows = _read_csv(csv_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    analysis_manifest = json.loads(analysis_manifest_path.read_text(encoding="utf-8"))
    if report.get("failures") or int(report.get("num_rows", -1)) != len(rows):
        raise ValueError(f"Incomplete analysis condition: {panel_name}/{model_key}")
    if int(analysis_manifest.get("num_failures", -1)) != 0:
        raise ValueError(f"Manifest records failures: {panel_name}/{model_key}")
    for row in rows:
        if int(row["num_group_permutations"]) != 251:
            raise ValueError(f"Non-exact base inference in {panel_name}/{model_key}")
        if int(row["nuisance_num_group_permutations"]) != 251:
            raise ValueError(f"Non-exact adjusted inference in {panel_name}/{model_key}")
        for field in (
            "probe_permutation_symmetric",
            "rbf_probe_permutation_symmetric",
            "nuisance_probe_permutation_symmetric",
            "nuisance_rbf_probe_permutation_symmetric",
            "fdr_permutation_symmetric",
        ):
            if field in row and str(row[field]).lower() not in {"true", "1"}:
                raise ValueError(
                    f"Complement-symmetry failure for {field}: {panel_name}/{model_key}/"
                    f"layer={row['layer']}/stream={row['stream']}"
                )

    linear = _probe_counts(rows, "probe", "nuisance_probe")
    rbf = _probe_counts(rows, "rbf_probe", "nuisance_rbf_probe")
    selected = max(rows, key=lambda row: float(row["fdr"]))
    summary = {
        "model_key": model_key,
        "panel": panel_name,
        "analysis_backend": rows[0]["analysis_backend"],
        "analysis_backend_version": rows[0]["analysis_backend_version"],
        "rows": len(rows),
        "selected_layer": int(selected["layer"]),
        "selected_stream": selected["stream"],
        "selected_fdr": float(selected["fdr"]),
        "selected_rank_fraction": float(selected["effective_rank_within_fraction"]),
        "selected_linear_accuracy": float(selected["probe_accuracy"]),
        "selected_linear_accuracy_p": float(selected["probe_accuracy_p"]),
        "selected_linear_accuracy_q": float(selected["probe_accuracy_q"]),
        "selected_linear_f1": float(selected["probe_f1"]),
        "selected_adjusted_linear_f1": float(selected["nuisance_probe_f1"]),
        "selected_adjusted_linear_q": float(selected["nuisance_probe_f1_q"]),
        "selected_rbf_f1": float(selected["rbf_probe_f1"]),
        "selected_rbf_accuracy": float(selected["rbf_probe_accuracy"]),
        "selected_rbf_accuracy_p": float(selected["rbf_probe_accuracy_p"]),
        "selected_rbf_accuracy_q": float(selected["rbf_probe_accuracy_q"]),
        "selected_adjusted_rbf_f1": float(selected["nuisance_rbf_probe_f1"]),
        "selected_adjusted_rbf_q": float(selected["nuisance_rbf_probe_f1_q"]),
        **{
            f"depth_mean_fdr_{stream}": statistics.fmean(
                float(row["fdr"]) for row in rows if row["stream"] == stream_name
            )
            for stream, stream_name in (
                ("attention", "attention_delta"),
                ("mlp", "mlp_delta"),
                ("residual", "residual_state"),
            )
        },
        **{f"linear_{key}": value for key, value in linear.items()},
        **{f"rbf_{key}": value for key, value in rbf.items()},
    }
    inputs = [
        {"path": str(path), "sha256": _sha256(path)}
        for path in (csv_path, report_path, analysis_manifest_path)
    ]
    return summary, rows, inputs


def feature_summaries(
    nuisance_table: Path,
    manifest: dict[str, Any],
    panel_names: list[str],
    model_keys: list[str],
) -> list[dict[str, Any]]:
    nuisance_rows = _read_csv(nuisance_table)
    model_by_pretrained = {
        str(manifest["models"][model_key]["pretrained"]): model_key
        for model_key in model_keys
    }
    summaries: list[dict[str, Any]] = []
    for panel_name in panel_names:
        panel = manifest["panels"][panel_name]
        class_by_task = {
            str(item["task"]): label
            for label in ("knowledge", "reasoning")
            for item in panel[f"{label}_tasks"]
        }
        for pretrained, model_key in model_by_pretrained.items():
            for label in ("knowledge", "reasoning"):
                selected = [
                    row
                    for row in nuisance_rows
                    if row["model"] == pretrained and class_by_task.get(row["task_name"]) == label
                ]
                if len(selected) != 500:
                    raise ValueError(
                        f"Expected 500 nuisance rows for {panel_name}/{model_key}/{label}, "
                        f"found {len(selected)}"
                    )
                summary: dict[str, Any] = {
                    "model_key": model_key,
                    "panel": panel_name,
                    "task_set": label,
                    "samples": len(selected),
                }
                for column in NUISANCE_COLUMNS:
                    values = [float(row[column]) for row in selected]
                    if not all(math.isfinite(value) for value in values):
                        raise ValueError(f"Non-finite {column} in nuisance table")
                    summary[f"{column}_mean"] = statistics.fmean(values)
                    summary[f"{column}_std"] = statistics.stdev(values)
                summaries.append(summary)
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("--nuisance-table", required=True, type=Path)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--model",
        action="append",
        help="Model key to summarize; repeat for multiple models. Defaults to every manifest model.",
    )
    parser.add_argument(
        "--panels",
        default="matched_text_mmlu_pro_control",
        help="Comma-separated valid panels to summarize; defaults to the matched panel only.",
    )
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    panel_names = [value.strip() for value in args.panels.split(",") if value.strip()]
    unknown_panels = set(panel_names) - set(manifest["panels"])
    if not panel_names or unknown_panels:
        raise ValueError(f"Invalid panel selection; unknown={sorted(unknown_panels)}")
    model_keys = select_models(manifest, args.model)
    summaries: list[dict[str, Any]] = []
    rows_by_condition: dict[tuple[str, str], list[dict[str, str]]] = {}
    input_hashes: list[dict[str, str]] = []
    for panel_name in panel_names:
        for model_key in model_keys:
            summary, condition_rows, inputs = summarize_condition(
                args.output_root, panel_name, model_key
            )
            summaries.append(summary)
            rows_by_condition[(model_key, panel_name)] = condition_rows
            input_hashes.extend(inputs)
    features = feature_summaries(args.nuisance_table, manifest, panel_names, model_keys)
    replication = profile_replication_rows(rows_by_condition, panel_names, model_keys)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "probe_persistence_summary.csv"
    feature_path = args.output_dir / "nuisance_feature_summary.csv"
    replication_path = args.output_dir / "fdr_profile_replication.csv"
    _write_csv(summary_path, summaries)
    _write_csv(feature_path, features)
    if replication:
        _write_csv(replication_path, replication)
    report = {
        "schema_version": 1,
        "conditions": len(summaries),
        "models": model_keys,
        "model_scope": "explicit_selection" if args.model is not None else "all_manifest_models",
        "included_panels": panel_names,
        "excluded_invalid_panels": [
            name for name in manifest["panels"] if name not in panel_names
        ],
        "analysis_rows": sum(int(row["rows"]) for row in summaries),
        "failures": [],
        "summary_environment_versions": _environment_versions(),
        "manifest": {"path": str(args.manifest), "sha256": _sha256(args.manifest)},
        "nuisance_table": {
            "path": str(args.nuisance_table),
            "sha256": _sha256(args.nuisance_table),
        },
        "inputs": input_hashes,
        "outputs": {
            "probe_persistence_summary.csv": _sha256(summary_path),
            "nuisance_feature_summary.csv": _sha256(feature_path),
            **(
                {"fdr_profile_replication.csv": _sha256(replication_path)}
                if replication
                else {}
            ),
        },
        "claim_limit": "Adjusted predictability is robustness to six measured prompt-surface variables, not evidence that the residual representation is specific to reasoning or knowledge.",
    }
    report_path = args.output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": str(summary_path),
                "features": str(feature_path),
                "replication": str(replication_path) if replication else None,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
