from __future__ import annotations

import sys

# CLI dispatcher which handles subcommands (tasks, models, eval, ui, ...)
# as well as the legacy flat-args form (--model X --tasks Y).
if __name__ == "__main__":
    from lmms_eval.cli.dispatch import main

    main()
    sys.exit(0)

import argparse
import csv
import datetime
import hashlib
import importlib
import json
import os
import subprocess
import traceback
import warnings
from functools import partial
from pathlib import Path

import numpy as np
import torch
import yaml

warnings.simplefilter("ignore", category=DeprecationWarning)

from typing import Union

from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from loguru import logger as eval_logger

from lmms_eval import evaluator, utils
from lmms_eval.api.metrics import power_analysis
from lmms_eval.api.registry import ALL_TASKS
from lmms_eval.cli.power_utils import collect_task_sizes
from lmms_eval.evaluator import request_caching_arg_to_dict
from lmms_eval.loggers import EvaluationTracker, WandbLogger
from lmms_eval.pruning import summarize_pruning_payload
from lmms_eval.tasks import TaskManager
from lmms_eval.utils import (
    get_eval_banner,
    make_table,
    simple_parse_args_string,
)


def _build_evaluation_tracker_args(args: argparse.Namespace) -> dict:
    """Build runtime tracker arguments without mutating serializable CLI state."""

    tracker_args = simple_parse_args_string(args.hf_hub_log_args)
    if args.output_path:
        tracker_args["output_path"] = args.output_path
    if os.environ.get("HF_TOKEN", None):
        tracker_args["token"] = os.environ["HF_TOKEN"]
    return tracker_args


def _int_or_none_list_arg_type(min_len: int, max_len: int, defaults: str, value: str, split_char: str = ","):
    def parse_value(item):
        item = item.strip().lower()
        if item == "none":
            return None
        try:
            return int(item)
        except ValueError:
            raise argparse.ArgumentTypeError(f"{item} is not an integer or None")

    items = [parse_value(v) for v in value.split(split_char)]
    num_items = len(items)

    if num_items == 1:
        # Makes downstream handling the same for single and multiple values
        items = items * max_len
    elif num_items < min_len or num_items > max_len:
        raise argparse.ArgumentTypeError(f"Argument requires {max_len} integers or None, separated by '{split_char}'")
    elif num_items != max_len:
        eval_logger.warning(f"Argument requires {max_len} integers or None, separated by '{split_char}'. " "Missing values will be filled with defaults.")
        default_items = [parse_value(v) for v in defaults.split(split_char)]
        items.extend(default_items[num_items:])  # extend items list with missing defaults

    return items


def check_argument_types(parser: argparse.ArgumentParser):
    """
    Check to make sure all CLI args are typed, raises error if not
    """
    for action in parser._actions:
        if action.dest != "help" and not action.const:
            if action.type is None:
                raise ValueError(f"Argument '{action.dest}' doesn't have a type specified.")
            else:
                continue


def _handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def _default_pruning_tracker_dir(output_path: str | None) -> Path:
    if output_path:
        base = Path(output_path)
        if base.suffix:
            base = base.parent
    else:
        base = Path(".")
    return base / "pruning_stats"


def _pruning_run_name(args: argparse.Namespace, datetime_str: str) -> str:
    model_name = utils.sanitize_long_string(str(args.model))
    return f"pruning_stats_{model_name}_{datetime_str}"


def _resolve_pruning_tracker_paths(args: argparse.Namespace, datetime_str: str) -> dict[str, Path]:
    default_run_name = _pruning_run_name(args, datetime_str)
    if args.pruning_tracker_output:
        candidate = Path(args.pruning_tracker_output)
        if candidate.suffix:
            run_dir = candidate.parent / utils.sanitize_long_string(candidate.stem)
            payload_path = run_dir / candidate.name
        else:
            run_dir = candidate / default_run_name
            payload_path = run_dir / "pruning_stats.pt"
    else:
        run_dir = _default_pruning_tracker_dir(args.output_path) / default_run_name
        payload_path = run_dir / "pruning_stats.pt"

    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "payload_path": payload_path,
        "summary_path": run_dir / "summary.json",
        "histograms_path": run_dir / "histograms.json",
        "visualizations_path": run_dir / "visualizations.json",
        "plots_dir": run_dir / "plots",
        "plots_manifest_path": run_dir / "plots_manifest.json",
        "mlp_modules_path": run_dir / "mlp_modules.csv",
        "attention_modules_path": run_dir / "attention_modules.csv",
        "linear_state_modules_path": run_dir / "linear_state_modules.csv",
        "linear_state_components_path": run_dir / "linear_state_components.csv",
        "token_group_modules_path": run_dir / "token_group_modules.csv",
        "attention_routing_modules_path": run_dir / "attention_routing_modules.csv",
        "network_sequence_path": run_dir / "network_sequence.csv",
    }


def _write_pruning_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=_handle_non_serializable)


def _write_pruning_csv(path: Path, rows: list[dict]) -> Path | None:
    if not rows:
        return None

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path


def _resolve_representation_dynamics_paths(args: argparse.Namespace, datetime_str: str) -> dict[str, Path]:
    model_name = utils.sanitize_long_string(str(args.model))
    run_name = f"representation_dynamics_{model_name}_{datetime_str}"
    if args.representation_dynamics_output:
        candidate = Path(args.representation_dynamics_output)
        if candidate.suffix:
            run_dir = candidate.parent / utils.sanitize_long_string(candidate.stem)
            payload_path = run_dir / candidate.name
        else:
            run_dir = candidate / run_name
            payload_path = run_dir / "representations.pt"
    else:
        if args.output_path:
            base = Path(args.output_path)
            if base.suffix:
                base = base.parent
        else:
            base = Path(".")
        run_dir = base / "representation_dynamics" / run_name
        payload_path = run_dir / "representations.pt"
    run_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "payload_path": payload_path,
        "metadata_path": run_dir / "metadata.json",
        "records_path": run_dir / "records.csv",
    }


def _lmms_eval_source_state() -> dict[str, object]:
    """Fingerprint tracked and untracked source state without storing source content."""

    repository = Path(__file__).resolve().parents[1]
    try:
        head = subprocess.run(
            ("git", "rev-parse", "HEAD"),
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ("git", "status", "--porcelain=v1", "--untracked-files=all"),
            cwd=repository,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        if not status:
            return {
                "lmms_eval_git_head": head,
                "lmms_eval_git_dirty": False,
                "lmms_eval_source_fingerprint_sha256": None,
            }
        digest = hashlib.sha256(status)
        digest.update(
            subprocess.run(
                ("git", "diff", "--binary", "HEAD", "--"),
                cwd=repository,
                check=True,
                stdout=subprocess.PIPE,
            ).stdout
        )
        for line in status.decode("utf-8", errors="surrogateescape").splitlines():
            if not line.startswith("?? "):
                continue
            path = repository / line[3:]
            if path.is_file():
                digest.update(line[3:].encode("utf-8", errors="surrogateescape"))
                digest.update(path.read_bytes())
        return {
            "lmms_eval_git_head": head,
            "lmms_eval_git_dirty": True,
            "lmms_eval_source_fingerprint_sha256": digest.hexdigest(),
        }
    except (OSError, subprocess.CalledProcessError):
        return {
            "lmms_eval_git_head": None,
            "lmms_eval_git_dirty": None,
            "lmms_eval_source_fingerprint_sha256": None,
        }


def _save_representation_dynamics_payload(args: argparse.Namespace, datetime_str: str, payload: dict) -> dict[str, Path]:
    output_paths = _resolve_representation_dynamics_paths(args, datetime_str)
    try:
        import transformers

        transformers_version = transformers.__version__
    except Exception:
        transformers_version = None
    metadata = {
        "schema_version": payload.get("schema_version"),
        "model": args.model,
        "model_args": evaluator.redact_sensitive_data(args.model_args),
        "pretrained": simple_parse_args_string(args.model_args or "").get("pretrained"),
        "tasks": args.tasks,
        "datetime": datetime_str,
        "limit": args.limit,
        "offset": args.offset,
        "batch_size": args.batch_size,
        "gen_kwargs": evaluator.redact_sensitive_data(args.gen_kwargs),
        "seeds": list(args.seed),
        "lmms_eval_git_hash": utils.get_git_commit_hash(),
        "lmms_eval_git_branch": utils.get_git_branch_name(),
        "lmms_eval_version": utils.get_lmms_eval_version_string(),
        **_lmms_eval_source_state(),
        "transformers_version": transformers_version,
        "selected_stack": payload.get("selected_stack"),
        "pooling": payload.get("pooling"),
        "decode_aggregation": payload.get("decode_aggregation"),
        "decode_max_updates": payload.get("decode_max_updates"),
        "phases": payload.get("phases"),
        "storage_dtype": payload.get("storage_dtype"),
        "num_layers": len(payload.get("layer_specs", [])),
        "num_hooks": payload.get("num_hooks", 0),
        "num_contexts": payload.get("num_contexts", 0),
        "num_records": payload.get("num_records", 0),
        "skipped": payload.get("skipped", {}),
        "truncated": payload.get("truncated", {}),
    }
    artifact = {"metadata": metadata, **payload}
    torch.save(artifact, output_paths["payload_path"])
    with output_paths["metadata_path"].open("w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "layer_specs": payload.get("layer_specs", [])}, handle, indent=2, default=_handle_non_serializable)

    manifest_rows: list[dict] = []
    for record in payload.get("records", []):
        vector = record.get("vector")
        row = {key: value for key, value in record.items() if key != "vector"}
        if torch.is_tensor(vector):
            values = vector.detach().float()
            row.update(
                {
                    "vector_abs_mean": float(values.abs().mean().item()),
                    "vector_rms": float(values.square().mean().sqrt().item()),
                    "vector_l2": float(values.norm().item()),
                }
            )
        manifest_rows.append(row)
    _write_pruning_csv(output_paths["records_path"], manifest_rows)
    return output_paths


def _histogram_plot_payload(histogram: dict, title: str, x_label: str) -> dict:
    bin_edges = histogram.get("bin_edges", []) or []
    counts = histogram.get("counts", []) or []
    bins = []

    if len(bin_edges) == len(counts) + 1:
        for idx, count in enumerate(counts):
            left = float(bin_edges[idx])
            right = float(bin_edges[idx + 1])
            bins.append(
                {
                    "left": left,
                    "right": right,
                    "center": (left + right) / 2.0,
                    "count": int(count),
                    "label": f"[{left:g}, {right:g})",
                }
            )

    return {
        "chart_type": "histogram",
        "title": title,
        "x_label": x_label,
        "y_label": "count",
        "num_samples": int(histogram.get("num_samples", 0) or 0),
        "num_source_values": int(histogram.get("num_source_values", 0) or 0),
        "is_sampled": bool(histogram.get("is_sampled", False)),
        "sampling_ratio": histogram.get("sampling_ratio"),
        "bins": bins,
    }


def _bar_plot_payload(rows: list[dict], title: str, metric_key: str, top_k: int = 20) -> dict:
    filtered = [row for row in rows if row.get(metric_key) is not None]
    sorted_rows = sorted(filtered, key=lambda row: float(row.get(metric_key, 0.0)), reverse=True)[:top_k]
    return {
        "chart_type": "bar",
        "title": title,
        "x_key": "module",
        "y_key": metric_key,
        "data": [
            {
                "module": row["module"],
                metric_key: row[metric_key],
            }
            for row in sorted_rows
        ],
    }


def _sequence_plot_payload(sequence_rows: list[dict], title: str, metric_key: str, y_label: str) -> dict:
    data = []
    for row in sequence_rows:
        value = row.get(metric_key)
        if value is None:
            continue
        data.append(
            {
                "sequence_index": int(row.get("sequence_index", len(data))),
                "module": row.get("module"),
                "stage_pattern": row.get("stage_pattern"),
                "kind": row.get("kind"),
                "primary_layer_index": row.get("primary_layer_index"),
                "value": float(value),
            }
        )

    return {
        "chart_type": "line",
        "title": title,
        "x_key": "sequence_index",
        "x_label": "network order",
        "y_key": "value",
        "y_label": y_label,
        "data": data,
    }


def _stage_pattern_plot_payload(stage_patterns: list[dict], title: str, metric_key: str, y_label: str) -> dict:
    data = []
    for row in stage_patterns:
        value = row.get(metric_key)
        if value is None:
            continue
        data.append(
            {
                "stage_pattern": row.get("stage_pattern"),
                "kind": row.get("kind"),
                "count": int(row.get("count", 0) or 0),
                "value": float(value),
            }
        )

    return {
        "chart_type": "bar",
        "title": title,
        "x_key": "stage_pattern",
        "y_key": "value",
        "y_label": y_label,
        "data": data,
    }


def _normalize_vector(tensor: torch.Tensor, count: int) -> list[float]:
    values = tensor.detach().float().reshape(-1).cpu()
    if count > 0:
        values = values / count
    return [float(value) for value in values.tolist()]


def _heatmap_plot_payload(
    sequence_rows: list[dict],
    stats: dict[str, torch.Tensor],
    counts: dict[str, int],
    title: str,
    value_label: str,
    source_lookup: dict[str, str] | None = None,
) -> dict:
    rows = []
    max_width = 0

    for row in sequence_rows:
        module_name = row.get("module")
        tensor = stats.get(module_name)
        if tensor is None:
            continue

        values = _normalize_vector(tensor, int(counts.get(module_name, 0)))
        max_width = max(max_width, len(values))
        rows.append(
            {
                "sequence_index": int(row.get("sequence_index", len(rows))),
                "module": module_name,
                "stage_pattern": row.get("stage_pattern"),
                "kind": row.get("kind"),
                "primary_layer_index": row.get("primary_layer_index"),
                "source": (source_lookup or {}).get(module_name),
                "values": values,
            }
        )

    if not rows:
        return {
            "chart_type": "heatmap",
            "title": title,
            "value_label": value_label,
            "available": False,
            "x_labels": [],
            "y_labels": [],
            "rows": [],
        }

    for row in rows:
        if len(row["values"]) < max_width:
            row["values"] = row["values"] + [None] * (max_width - len(row["values"]))

    return {
        "chart_type": "heatmap",
        "title": title,
        "value_label": value_label,
        "available": True,
        "x_labels": [f"head_{index}" for index in range(max_width)],
        "y_labels": [f"{row['sequence_index']}:{row['module']}" for row in rows],
        "rows": rows,
    }


def _build_pruning_visualizations(summary: dict, payload: dict) -> dict:
    mlp_summary = summary.get("mlp", {})
    attn_summary = summary.get("attention", {})
    linear_state_summary = summary.get("linear_state", {})
    linear_state_component_summary = summary.get("linear_state_components", {})
    network_summary = summary.get("network", {})
    mlp_rows = mlp_summary.get("module_rows", []) or []
    attn_rows = attn_summary.get("module_rows", []) or []
    linear_state_rows = linear_state_summary.get("module_rows", []) or []
    linear_state_component_rows = linear_state_component_summary.get("module_rows", []) or []
    sequence_rows = network_summary.get("module_sequence", []) or []
    stage_patterns = network_summary.get("stage_patterns", []) or []

    return {
        "charts": {
            "mlp_abs_activation_histogram": _histogram_plot_payload(
                mlp_summary.get("histogram", {}),
                title="MLP Absolute Activation Histogram",
                x_label="absolute activation",
            ),
            "attention_weight_histogram": _histogram_plot_payload(
                attn_summary.get("histogram", {}),
                title="Attention Weight Histogram",
                x_label="attention weight",
            ),
            "linear_state_output_abs_histogram": _histogram_plot_payload(
                linear_state_summary.get("histograms", {}).get("output_abs", {}),
                title="Linear-State Output Magnitude Histogram",
                x_label="absolute output activation",
            ),
            "linear_state_input_output_delta_abs_histogram": _histogram_plot_payload(
                linear_state_summary.get("histograms", {}).get("input_output_delta_abs", {}),
                title="Linear-State Input-Output Delta Magnitude Histogram",
                x_label="absolute input-output delta",
            ),
            "linear_state_component_abs_histogram": _histogram_plot_payload(
                linear_state_component_summary.get("histogram", {}),
                title="Linear-State Component Activation Histogram",
                x_label="absolute component activation",
            ),
            "top_mlp_modules_by_mean_activation": _bar_plot_payload(
                mlp_rows,
                title="Top MLP Modules by Mean Activation",
                metric_key="mean",
            ),
            "top_mlp_modules_by_active_fraction": _bar_plot_payload(
                mlp_rows,
                title="Top MLP Modules by Active Fraction",
                metric_key="mean_active_fraction",
            ),
            "top_attention_modules_by_mean_mass": _bar_plot_payload(
                attn_rows,
                title="Top Attention Modules by Mean Attention Mass",
                metric_key="mean",
            ),
            "top_attention_modules_by_peak_mass": _bar_plot_payload(
                attn_rows,
                title="Top Attention Modules by Peak Attention Mass",
                metric_key="mean_peak_mass",
            ),
            "top_linear_state_modules_by_delta_ratio": _bar_plot_payload(
                linear_state_rows,
                title="Top Linear-State Modules by Output/Input Delta RMS Ratio",
                metric_key="input_output_delta_rms_ratio",
            ),
            "top_linear_state_components_by_mean_activation": _bar_plot_payload(
                linear_state_component_rows,
                title="Top Linear-State Components by Mean Activation",
                metric_key="mean",
            ),
            "mlp_mean_by_network_order": _sequence_plot_payload(
                sequence_rows,
                title="MLP Mean Activation by Network Order",
                metric_key="mlp_mean",
                y_label="mean absolute activation",
            ),
            "mlp_active_fraction_by_network_order": _sequence_plot_payload(
                sequence_rows,
                title="MLP Active Fraction by Network Order",
                metric_key="mlp_active_fraction",
                y_label="active fraction",
            ),
            "attention_mean_by_network_order": _sequence_plot_payload(
                sequence_rows,
                title="Attention Mean Mass by Network Order",
                metric_key="attention_mean",
                y_label="mean attention mass",
            ),
            "attention_entropy_by_network_order": _sequence_plot_payload(
                sequence_rows,
                title="Attention Entropy by Network Order",
                metric_key="attention_entropy",
                y_label="entropy",
            ),
            "attention_peak_mass_by_network_order": _sequence_plot_payload(
                sequence_rows,
                title="Attention Peak Mass by Network Order",
                metric_key="attention_peak_mass",
                y_label="peak attention mass",
            ),
            "attention_valid_fraction_by_network_order": _sequence_plot_payload(
                sequence_rows,
                title="Attention Valid Fraction by Network Order",
                metric_key="attention_valid_fraction",
                y_label="nonzero attention fraction",
            ),
            "linear_state_delta_ratio_by_network_order": _sequence_plot_payload(
                sequence_rows,
                title="Linear-State Input-Output Delta RMS Ratio by Network Order",
                metric_key="linear_state_input_output_delta_rms_ratio",
                y_label="input-output delta RMS / input RMS",
            ),
            "linear_state_output_ratio_by_network_order": _sequence_plot_payload(
                sequence_rows,
                title="Linear-State Output RMS Ratio by Network Order",
                metric_key="linear_state_output_input_rms_ratio",
                y_label="output RMS / input RMS",
            ),
            "linear_state_output_input_cosine_by_network_order": _sequence_plot_payload(
                sequence_rows,
                title="Linear-State Output/Input Cosine by Network Order",
                metric_key="linear_state_output_input_cosine",
                y_label="cos(input, output)",
            ),
            "mlp_mean_by_stage_pattern": _stage_pattern_plot_payload(
                stage_patterns,
                title="MLP Mean Activation by Stage Pattern",
                metric_key="mlp_mean",
                y_label="mean absolute activation",
            ),
            "attention_mean_by_stage_pattern": _stage_pattern_plot_payload(
                stage_patterns,
                title="Attention Mean Mass by Stage Pattern",
                metric_key="attention_mean",
                y_label="mean attention mass",
            ),
            "attention_valid_fraction_by_stage_pattern": _stage_pattern_plot_payload(
                stage_patterns,
                title="Attention Valid Fraction by Stage Pattern",
                metric_key="attention_valid_fraction",
                y_label="nonzero attention fraction",
            ),
            "linear_state_delta_ratio_by_stage_pattern": _stage_pattern_plot_payload(
                stage_patterns,
                title="Linear-State Input-Output Delta RMS Ratio by Stage Pattern",
                metric_key="linear_state_input_output_delta_rms_ratio",
                y_label="input-output delta RMS / input RMS",
            ),
            "linear_state_output_input_cosine_by_stage_pattern": _stage_pattern_plot_payload(
                stage_patterns,
                title="Linear-State Output/Input Cosine by Stage Pattern",
                metric_key="linear_state_output_input_cosine",
                y_label="cos(input, output)",
            ),
        },
        "heatmaps": {
            "attention_head_mean_by_network_order": _heatmap_plot_payload(
                sequence_rows,
                payload.get("attn_stats", {}),
                payload.get("attn_counts", {}),
                title="Attention Head Mean by Network Order",
                value_label="mean attention mass",
            ),
            "attention_head_entropy_by_network_order": _heatmap_plot_payload(
                sequence_rows,
                payload.get("attn_entropy", {}),
                payload.get("attn_counts", {}),
                title="Attention Head Entropy by Network Order",
                value_label="mean entropy",
            ),
            "attention_head_peak_mass_by_network_order": _heatmap_plot_payload(
                sequence_rows,
                payload.get("attn_peak_mass", {}),
                payload.get("attn_counts", {}),
                title="Attention Head Peak Mass by Network Order",
                value_label="mean peak attention mass",
            ),
        },
        "network": {
            "module_sequence": sequence_rows,
            "stage_patterns": stage_patterns,
        },
    }


def _safe_plot_stem(name: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in str(name).strip().lower())
    return safe.strip("_") or "plot"


def _truncate_label(label: object, max_chars: int = 72) -> str:
    text = str(label)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _tick_indices(num_items: int, max_ticks: int = 18) -> list[int]:
    if num_items <= 0:
        return []
    if num_items <= max_ticks:
        return list(range(num_items))
    step = max(1, int(np.ceil(num_items / max_ticks)))
    return list(range(0, num_items, step))


def _render_histogram_plot(ax, chart: dict) -> bool:
    bins = chart.get("bins", []) or []
    if not bins:
        return False

    labels = [_truncate_label(bin_row.get("label", "")) for bin_row in bins]
    counts = [float(bin_row.get("count", 0.0) or 0.0) for bin_row in bins]
    positions = np.arange(len(labels))
    ax.bar(positions, counts, color="#4C6EF5")
    tick_ids = _tick_indices(len(labels), max_ticks=12)
    ax.set_xticks(tick_ids)
    ax.set_xticklabels([labels[idx] for idx in tick_ids], rotation=45, ha="right", fontsize=8)
    return True


def _render_bar_plot(ax, chart: dict) -> bool:
    data = chart.get("data", []) or []
    x_key = chart.get("x_key")
    y_key = chart.get("y_key")
    if not data or not x_key or not y_key:
        return False

    labels = [_truncate_label(row.get(x_key, "")) for row in data]
    values = [float(row.get(y_key, 0.0) or 0.0) for row in data]
    positions = np.arange(len(labels))
    ax.barh(positions, values, color="#1098AD")
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    return True


def _render_line_plot(ax, chart: dict) -> bool:
    data = chart.get("data", []) or []
    if not data:
        return False

    x_values = [float(row.get("sequence_index", idx)) for idx, row in enumerate(data)]
    y_values = [float(row.get("value", 0.0) or 0.0) for row in data]
    ax.plot(x_values, y_values, marker="o", linewidth=1.75, markersize=3, color="#F08C00")
    tick_ids = _tick_indices(len(x_values), max_ticks=12)
    ax.set_xticks([x_values[idx] for idx in tick_ids])
    return True


def _render_heatmap_plot(fig, ax, chart: dict) -> bool:
    rows = chart.get("rows", []) or []
    if not rows:
        return False

    matrix = np.array(
        [
            [np.nan if value is None else float(value) for value in row.get("values", [])]
            for row in rows
        ],
        dtype=float,
    )
    if matrix.size == 0:
        return False

    image = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)

    x_labels = chart.get("x_labels", []) or []
    y_labels = chart.get("y_labels", []) or []
    x_tick_ids = _tick_indices(len(x_labels), max_ticks=16)
    y_tick_ids = _tick_indices(len(y_labels), max_ticks=18)
    ax.set_xticks(x_tick_ids)
    ax.set_xticklabels([_truncate_label(x_labels[idx], max_chars=20) for idx in x_tick_ids], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(y_tick_ids)
    ax.set_yticklabels([_truncate_label(y_labels[idx], max_chars=60) for idx in y_tick_ids], fontsize=7)
    return True


def _write_pruning_plots(run_dir: Path, visualizations: dict) -> dict[str, object]:
    plots_dir = run_dir / "plots"
    manifest_path = run_dir / "plots_manifest.json"
    plots_dir.mkdir(parents=True, exist_ok=True)

    mpl_config_dir = Path(os.environ.get("MPLCONFIGDIR", "/tmp/lmms_eval_matplotlib"))
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)
    xdg_cache_home = Path(os.environ.get("XDG_CACHE_HOME", "/tmp/lmms_eval_cache"))
    xdg_cache_home.mkdir(parents=True, exist_ok=True)
    os.environ["XDG_CACHE_HOME"] = str(xdg_cache_home)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        eval_logger.warning(f"Skipping pruning plot generation because matplotlib is unavailable: {exc}")
        return {
            "plots_dir": None,
            "plots_manifest_path": None,
            "plot_files": {},
        }

    plot_files: dict[str, str] = {}
    render_groups = [
        ("charts", visualizations.get("charts", {}) or {}),
        ("heatmaps", visualizations.get("heatmaps", {}) or {}),
    ]

    for _, charts in render_groups:
        for chart_name, chart in charts.items():
            chart_type = chart.get("chart_type")
            fig, ax = plt.subplots(figsize=(12, 6))
            rendered = False

            try:
                if chart_type == "histogram":
                    rendered = _render_histogram_plot(ax, chart)
                elif chart_type == "bar":
                    rendered = _render_bar_plot(ax, chart)
                elif chart_type == "line":
                    rendered = _render_line_plot(ax, chart)
                elif chart_type == "heatmap":
                    rendered = _render_heatmap_plot(fig, ax, chart)

                if not rendered:
                    plt.close(fig)
                    continue

                ax.set_title(chart.get("title", chart_name))
                if chart.get("x_label"):
                    ax.set_xlabel(chart["x_label"])
                if chart.get("y_label"):
                    ax.set_ylabel(chart["y_label"])
                if chart.get("value_label") and chart_type == "heatmap":
                    ax.set_xlabel(chart.get("x_label", "head index"))
                    ax.set_ylabel(chart.get("y_label", "network order"))

                fig.tight_layout()
                plot_path = plots_dir / f"{_safe_plot_stem(chart_name)}.png"
                fig.savefig(plot_path, dpi=180, bbox_inches="tight")
                plot_files[chart_name] = str(plot_path)
            finally:
                plt.close(fig)

    manifest = {
        "plots_dir": str(plots_dir),
        "plots": plot_files,
    }
    _write_pruning_json(manifest_path, manifest)
    return {
        "plots_dir": plots_dir,
        "plots_manifest_path": manifest_path,
        "plot_files": plot_files,
    }


def _save_pruning_tracker_payload(args: argparse.Namespace, datetime_str: str, payload: dict) -> dict[str, object]:
    output_paths = _resolve_pruning_tracker_paths(args, datetime_str)
    compact_summary = summarize_pruning_payload(payload)
    detailed_summary = summarize_pruning_payload(payload, include_module_rows=True)
    visualizations = _build_pruning_visualizations(detailed_summary, payload)
    artifact = {
        "metadata": {
            "model": args.model,
            "model_args": evaluator.redact_sensitive_data(args.model_args),
            "tasks": args.tasks,
            "datetime": datetime_str,
            "num_hooks": payload.get("num_hooks", 0),
            "num_mlp_modules": len(payload.get("mlp_stats", {})),
            "num_attn_modules": len(payload.get("attn_stats", {})),
            "num_linear_state_modules": len(payload.get("linear_state_counts", {})),
            "num_linear_state_components": len(payload.get("linear_state_component_stats", {})),
            "force_eager_attention": bool(getattr(args, "pruning_tracker_force_eager_attention", True)),
            "force_output_attentions": bool(getattr(args, "pruning_tracker_force_output_attentions", False)),
            "run_dir": str(output_paths["run_dir"]),
        },
        **payload,
        "summary": compact_summary,
    }

    torch.save(artifact, output_paths["payload_path"])
    _write_pruning_json(
        output_paths["summary_path"],
        {
            "metadata": artifact["metadata"],
            "summary": detailed_summary,
        },
    )
    _write_pruning_json(
        output_paths["histograms_path"],
        {
            "metadata": artifact["metadata"],
            "mlp_abs_activation": detailed_summary.get("mlp", {}).get("histogram", {}),
            "attention_weight": detailed_summary.get("attention", {}).get("histogram", {}),
            "linear_state_output_abs": detailed_summary.get("linear_state", {}).get("histograms", {}).get("output_abs", {}),
            "linear_state_input_output_delta_abs": detailed_summary.get("linear_state", {}).get("histograms", {}).get("input_output_delta_abs", {}),
            "linear_state_component_abs_activation": detailed_summary.get("linear_state_components", {}).get("histogram", {}),
        },
    )
    _write_pruning_json(
        output_paths["visualizations_path"],
        {
            "metadata": artifact["metadata"],
            **visualizations,
        },
    )
    plot_outputs = _write_pruning_plots(output_paths["run_dir"], visualizations)

    mlp_csv_path = _write_pruning_csv(output_paths["mlp_modules_path"], detailed_summary.get("mlp", {}).get("module_rows", []))
    attention_csv_path = _write_pruning_csv(
        output_paths["attention_modules_path"],
        detailed_summary.get("attention", {}).get("module_rows", []),
    )
    linear_state_csv_path = _write_pruning_csv(
        output_paths["linear_state_modules_path"],
        detailed_summary.get("linear_state", {}).get("module_rows", []),
    )
    linear_state_component_csv_path = _write_pruning_csv(
        output_paths["linear_state_components_path"],
        detailed_summary.get("linear_state_components", {}).get("module_rows", []),
    )
    token_group_csv_path = _write_pruning_csv(
        output_paths["token_group_modules_path"],
        detailed_summary.get("token_groups", {}).get("module_rows", []),
    )
    attention_routing_csv_path = _write_pruning_csv(
        output_paths["attention_routing_modules_path"],
        detailed_summary.get("attention_routing", {}).get("module_rows", []),
    )
    network_sequence_csv_path = _write_pruning_csv(
        output_paths["network_sequence_path"],
        detailed_summary.get("network", {}).get("module_sequence", []),
    )

    return {
        "run_dir": output_paths["run_dir"],
        "payload_path": output_paths["payload_path"],
        "summary_path": output_paths["summary_path"],
        "histograms_path": output_paths["histograms_path"],
        "visualizations_path": output_paths["visualizations_path"],
        "plots_dir": plot_outputs["plots_dir"],
        "plots_manifest_path": plot_outputs["plots_manifest_path"],
        "plot_files": plot_outputs["plot_files"],
        "mlp_modules_path": mlp_csv_path,
        "attention_modules_path": attention_csv_path,
        "linear_state_modules_path": linear_state_csv_path,
        "linear_state_components_path": linear_state_component_csv_path,
        "token_group_modules_path": token_group_csv_path,
        "attention_routing_modules_path": attention_routing_csv_path,
        "network_sequence_path": network_sequence_csv_path,
        "summary": compact_summary,
    }


def _run_power_analysis(args: argparse.Namespace) -> None:
    """Run power analysis to calculate minimum sample size for detecting a given effect."""
    task_sizes = {}
    if args.tasks and args.tasks not in ["list", "list_groups", "list_tags", "list_subtasks"]:
        task_sizes = collect_task_sizes(args.tasks, verbosity=args.verbosity, include_path=args.include_path)

    result = power_analysis(
        effect_size=args.effect_size,
        std_a=args.std_a,
        std_b=args.std_b,
        alpha=args.alpha,
        power=args.power,
        correlation=args.correlation,
    )

    print("\n" + "=" * 60)
    print("POWER ANALYSIS RESULTS")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  Effect size (delta):     {args.effect_size:.1%}")
    print(f"  Std (model A):           {result['std_a']}")
    print(f"  Std (model B):           {result['std_b']}")
    print(f"  Significance level (α):  {args.alpha}")
    print(f"  Desired power (1-β):     {args.power}")
    print(f"  Correlation (ρ):         {args.correlation}")
    print(f"\nResult:")
    print(f"  Minimum sample size:     n = {result['min_n']}")
    print(f"\nInterpretation:")
    print(f"  To detect a {args.effect_size:.1%} difference with {args.power:.0%} power,")
    print(f"  you need at least {result['min_n']} questions in your benchmark.")

    if task_sizes:
        print(f"\n" + "-" * 60)
        print("TASK ANALYSIS")
        print("-" * 60)
        for task_name, n_samples in task_sizes.items():
            task_result = power_analysis(
                effect_size=args.effect_size,
                std_a=args.std_a,
                std_b=args.std_b,
                alpha=args.alpha,
                power=args.power,
                correlation=args.correlation,
                current_n=n_samples,
            )
            status = "✓ Sufficient" if n_samples >= result["min_n"] else "✗ Insufficient"
            print(f"\n  {task_name}:")
            print(f"    Sample size:         n = {n_samples}")
            print(f"    Current power:       {task_result['current_power']:.1%}")
            print(f"    Min detectable Δ:    {task_result['min_detectable_effect']:.1%}")
            print(f"    Status:              {status}")

    print("\n" + "=" * 60 + "\n")


def parse_eval_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config",
        default="",
        help="Path to a yaml file specifying eval arguments. CLI arguments override YAML values.",
    )
    parser.add_argument("--model", default="hf", help="Name of model e.g. `hf`")
    parser.add_argument(
        "--tasks",
        default=None,
        help="To get full list of tasks, use the command lmms-eval --tasks list",
    )
    parser.add_argument(
        "--model_args",
        default="",
        help="String arguments for model, e.g. `pretrained=EleutherAI/pythia-160m,dtype=float32`",
    )
    parser.add_argument(
        "--launcher_args",
        default=None,
        help="String arguments for launcher for local llm as judge, e.g. `tp=8`, if None then no launcher will be used.",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=None,
        help="Number of examples in few-shot context",
    )
    parser.add_argument(
        "--batch_size",
        "-b",
        type=str,
        default=1,
        metavar="auto|auto:N|N",
        help="Acceptable values are 'auto', 'auto:N' or N, where N is an integer. Default 1.",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=None,
        metavar="N",
        help="Maximal batch size to try with --batch_size auto.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (e.g. cuda, cuda:0, cpu)",
    )
    parser.add_argument(
        "--output_path",
        default=None,
        type=str,
        metavar="= [dir/file.jsonl] [DIR]",
        help="The path to the output file where the result metrics will be saved. If the path is a directory and log_samples is true, the results will be saved in the directory. Else the parent directory will be used.",
    )
    parser.add_argument(
        "--pruning_tracker",
        action="store_true",
        default=False,
        help="Collect forward-hook pruning statistics for MLP and attention modules during evaluation.",
    )
    parser.add_argument(
        "--pruning_tracker_output",
        type=str,
        default=None,
        help="Optional directory or .pt filename for pruning reports. Each run creates a local subdirectory with the raw payload and JSON/CSV summaries. Defaults to <output_path>/pruning_stats/.",
    )
    parser.add_argument(
        "--pruning_tracker_stride",
        type=int,
        default=1,
        help="Track every Nth forward for each hooked module. Use values >1 to reduce pruning-tracker overhead during long runs.",
    )
    parser.add_argument(
        "--pruning_tracker_max_histogram_samples",
        type=int,
        default=65536,
        help="Maximum number of sampled tensor values retained per tracked module for adaptive histogram summaries. Keeps histogram collection lightweight on large tensors.",
    )
    parser.add_argument(
        "--pruning_tracker_force_eager_attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When pruning tracking is enabled, request explicit HF attention maps by injecting attn_implementation=eager when supported and setting eager attention on model configs. Use --no-pruning_tracker_force_eager_attention to disable.",
    )
    parser.add_argument(
        "--pruning_tracker_force_output_attentions",
        action="store_true",
        default=False,
        help="Also set HF config.output_attentions=True/generation_config.output_attentions=True. This can retain full attention maps during generation and is high-memory; keep disabled for large VLM calibration runs.",
    )
    parser.add_argument(
        "--representation_dynamics",
        action="store_true",
        default=False,
        help="Extract pooled per-example decoder attention/mixer deltas, MLP deltas, and final residual states for offline separability analysis.",
    )
    parser.add_argument(
        "--representation_dynamics_output",
        type=str,
        default=None,
        help="Optional directory or .pt filename for representation records. Defaults to <output_path>/representation_dynamics/.",
    )
    parser.add_argument(
        "--representation_dynamics_pooling",
        type=str,
        choices=["last_valid", "mean_valid"],
        default="last_valid",
        help="Pool prefill vectors from the last valid prompt token or the mean over valid prompt tokens.",
    )
    parser.add_argument(
        "--representation_dynamics_phases",
        type=str,
        choices=["prefill", "decode", "both"],
        default="prefill",
        help="Capture prompt prefill, the mean over generated decode steps, or both. Prefill is the confirmatory default.",
    )
    parser.add_argument(
        "--representation_dynamics_decode_aggregation",
        type=str,
        choices=["mean_updates", "first_update", "all_updates"],
        default="mean_updates",
        help="For decode capture, average updates, retain the first update, or emit one record per generated-token update.",
    )
    parser.add_argument(
        "--representation_dynamics_decode_max_updates",
        type=int,
        default=None,
        help="Optional positive cap on per-example decode steps when --representation_dynamics_decode_aggregation=all_updates.",
    )
    parser.add_argument(
        "--representation_dynamics_storage_dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float32",
        help="CPU storage dtype for pooled activation vectors. Float32 avoids silent overflow and is the scientific default; reductions and accumulation remain float32.",
    )
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help=("Limit examples per task: use -1 (or omit) for all samples, " "0 < limit < 1 for a fraction of the dataset, and limit >= 1 " "for an absolute sample count."),
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=0,
        help="Start evaluation from this dataset index for each task.",
    )
    parser.add_argument(
        "--use_cache",
        "-c",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to a response-cache root directory or SQLite .db file. "
        "If PATH is a directory, or an explicit PATH/cache.db root file, lmms-eval keeps the shared root cache at "
        "PATH/cache.db and writes each run into PATH/runs/<run_id>/ before auto-merging back into the root DB. "
        "This avoids write contention across concurrent jobs while preserving cache reuse. "
        "If PATH is a file, legacy single-target behavior is preserved. `None` disables response caching.",
    )
    parser.add_argument(
        "--cache_requests",
        type=str,
        default=None,
        choices=["true", "refresh", "delete"],
        help="Speed up evaluation by caching the building of dataset requests. `None` if not caching.",
    )
    parser.add_argument(
        "--check_integrity",
        action="store_true",
        help="Whether to run the relevant part of the test suite for the tasks",
    )
    parser.add_argument(
        "--write_out",
        "-w",
        action="store_true",
        default=False,
        help="DEPRECATED: This flag is deprecated and will be removed in a future version. "
        "For debugging, use --log_samples to save all outputs to files. "
        "This flag prints prompts for the first few documents to console, impacting performance.",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis",
    )
    parser.add_argument(
        "--wandb_log_samples",
        action="store_true",
        default=False,
        help="If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis to Weights and Biases",
    )
    parser.add_argument(
        "--log_samples_suffix",
        type=str,
        default="model_outputs",
        help="Specify a suffix for the log_samples file name.",
    )
    parser.add_argument(
        "--system_instruction",
        type=str,
        default=None,
        help="System instruction to be used in the prompt",
    )
    parser.add_argument(
        "--apply_chat_template",
        action="store_true",
        default=False,
        help="If True, applies the chat template to the prompt",
    )
    parser.add_argument(
        "--fewshot_as_multiturn",
        action="store_true",
        default=False,
        help="If True, uses the fewshot as a multi-turn conversation",
    )
    parser.add_argument(
        "--show_config",
        action="store_true",
        default=False,
        help="If True, shows the the full config of all tasks at the end of the evaluation.",
    )
    parser.add_argument(
        "--include_path",
        type=str,
        default=None,
        help="Additional path to include if there are external tasks to include.",
    )
    parser.add_argument(
        "--gen_kwargs",
        default="",
        help=("String arguments for model generation on greedy_until tasks," " e.g. `temperature=0,top_k=0,top_p=0`"),
    )
    parser.add_argument(
        "--reasoning_tags",
        type=str,
        default='[["<think>", "</think>"], ["<analysis>", "</analysis>"]]',
        help="JSON string list of [start_tag, end_tag] pairs used for reasoning extraction.",
    )
    parser.add_argument(
        "--verbosity",
        type=str,
        default="INFO",
        help="Log error when tasks are not registered.",
    )
    parser.add_argument(
        "--wandb_args",
        default="",
        help="Comma separated string arguments passed to wandb.init, e.g. `project=lmms-eval,job_type=eval",
    )
    parser.add_argument(
        "--timezone",
        default="Asia/Singapore",
        help="Timezone for datetime string, e.g. Asia/Singapore, America/New_York, America/Los_Angeles. You can check the full list via `import pytz; print(pytz.common_timezones)`",
    )
    parser.add_argument(
        "--hf_hub_log_args",
        type=str,
        default="",
        help="Comma separated string arguments passed to Hugging Face Hub's log function, e.g. `hub_results_org=EleutherAI,hub_repo_name=lm-eval-results`",
    )
    parser.add_argument(
        "--predict_only",
        "-x",
        action="store_true",
        default=False,
        help="Use with --log_samples. Only model outputs will be saved and metrics will not be evaluated.",
    )
    default_seed_string = "0,1234,1234,1234"
    parser.add_argument(
        "--seed",
        type=partial(_int_or_none_list_arg_type, 3, 4, default_seed_string),
        default=default_seed_string,  # for backward compatibility
        help=(
            "Set seed for python's random, numpy, torch, and fewshot sampling.\n"
            "Accepts a comma-separated list of 4 values for python's random, numpy, torch, and fewshot sampling seeds, "
            "respectively, or a single integer to set the same seed for all four.\n"
            f"The values are either an integer or 'None' to not set the seed. Default is `{default_seed_string}` "
            "(for backward compatibility).\n"
            "E.g. `--seed 0,None,8,52` sets `random.seed(0)`, `torch.manual_seed(8)`, and fewshot sampling seed to 52. "
            "Here numpy's seed is not set since the second value is `None`.\n"
            "E.g, `--seed 42` sets all four seeds to 42."
        ),
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Sets trust_remote_code to True to execute code to create HF Datasets from the Hub",
    )
    parser.add_argument(
        "--process_with_media",
        action="store_true",
        help="Whether you will process you dataset with audio, image. By default set to False" "In case some benchmarks need to be processed with media, set this flag to True.",
    )
    parser.add_argument(
        "--agentic_trace_mode",
        type=str,
        default="basic",
        choices=["basic", "full"],
        help="Controls agentic trace logging level. 'basic' logs compact final trace payload, 'full' logs per-round input/output/state snapshots.",
    )
    parser.add_argument(
        "--force_simple",
        action="store_true",
        help="Force the evaluation to use the simple mode of the models",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Launch interactive TUI mode for configuration",
    )
    parser.add_argument(
        "-n",
        "--repeats",
        "--num_samples",
        dest="repeats",
        type=int,
        default=1,
        help=("Number of repeated generations per question for model stability " "measurement. Backward-compatible alias: --num_samples. " "When n > 1, enables k-samples " "mode and computes EA, CA, IV, CR metrics."),
    )
    parser.add_argument("--baseline", type=str, default=None, help="Baseline for paired t-test comparison. Accepts: local JSONL path, hf://user/repo, or preset name (e.g., qwen25vl).")

    # Cost & Token Tracking
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=None,
        help="Maximum total tokens (input+output+reasoning). Evaluation stops gracefully when exceeded. Disabled by default.",
    )

    # Power Analysis arguments
    parser.add_argument(
        "--power-analysis",
        action="store_true",
        default=False,
        help="Enable power analysis to calculate minimum sample size for detecting a given effect size.",
    )
    parser.add_argument(
        "--effect-size",
        type=float,
        default=0.03,
        help="Minimum effect size to detect (default: 0.03 = 3%%). Used with --power-analysis.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for power analysis (default: 0.05). Used with --power-analysis.",
    )
    parser.add_argument(
        "--power",
        type=float,
        default=0.80,
        help="Desired statistical power (default: 0.80). Used with --power-analysis.",
    )
    parser.add_argument(
        "--correlation",
        type=float,
        default=0.5,
        help="Expected correlation between paired samples (default: 0.5). Used with --power-analysis.",
    )
    parser.add_argument(
        "--std-a",
        type=float,
        default=None,
        help="Std deviation of model A scores (estimate from previous eval). Default: 0.5 for binary. Used with --power-analysis.",
    )
    parser.add_argument(
        "--std-b",
        type=float,
        default=None,
        help="Std deviation of model B scores (estimate from previous eval). If not set, assumes equal to --std-a. Used with --power-analysis.",
    )

    args = parser.parse_args()
    return parser, args


def cli_evaluate(args: Union[argparse.Namespace, None] = None) -> None:
    parser, default_args = parse_eval_args()

    # If args were provided, override the defaults
    if args:
        for key, value in vars(args).items():
            setattr(default_args, key, value)

    args = default_args

    # Handle power analysis mode (pre-evaluation planning)
    if getattr(args, "power_analysis", False):
        _run_power_analysis(args)
        sys.exit(0)

    if args.wandb_args:
        if "name" not in args.wandb_args:
            safe_model_args = evaluator.redact_sensitive_data(args.model_args)
            name = f"{args.model}_{safe_model_args}_{utils.get_datetime_str(timezone=args.timezone)}"
            name = utils.sanitize_long_string(name)
            args.wandb_args += f",name={name}"
        wandb_logger = WandbLogger(**simple_parse_args_string(args.wandb_args))

    # reset logger
    eval_logger.remove()
    # Configure logger with detailed format including file path, function name, and line number
    log_format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | " "<level>{level: <8}</level> | " "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - " "<level>{message}</level>"
    eval_logger.add(sys.stdout, colorize=True, level=args.verbosity, format=log_format)
    eval_logger.info(f"Verbosity set to {args.verbosity}")
    os.environ["VERBOSITY"] = args.verbosity

    args_list = []
    results_list = []
    if args.config:
        if not os.path.exists(args.config):
            raise ValueError(f"Config file does not exist: {args.config}")

        with open(args.config, "r") as file:
            config_args = yaml.safe_load(file)
        config_args = [config_args] if type(config_args) != list else config_args

        # Extract and apply env vars before validation (env is not a CLI arg)
        for config in config_args:
            env_config = config.pop("env", None)
            if env_config:
                if not isinstance(env_config, dict):
                    raise ValueError(f"'env' in config must be a dict, got {type(env_config).__name__}")
                for env_key, env_value in env_config.items():
                    resolved = os.path.expandvars(str(env_value))
                    os.environ[env_key] = resolved
                    eval_logger.info(f"Config env: {env_key}={'*' * min(len(resolved), 8) if any(s in env_key.upper() for s in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD')) else resolved}")

        # Validate config keys
        valid_keys = {action.dest for action in parser._actions}
        for config in config_args:
            unknown_keys = set(config.keys()) - valid_keys
            if unknown_keys:
                raise ValueError(f"Unknown keys in config file: {sorted(unknown_keys)}. " f"Valid keys are: {sorted(valid_keys - {'help'})}")

        # Determine which CLI args were explicitly provided by the user.
        default_config_args = parser.parse_args([])
        cli_explicit = {}
        for key, value in vars(args).items():
            default_value = getattr(default_config_args, key, None)
            if value != default_value:
                cli_explicit[key] = value

        # multiple configs, create args list first
        for config in config_args:
            args_copy = argparse.Namespace(**vars(default_config_args))
            for key, value in config.items():
                setattr(args_copy, key, value)
            for key, value in cli_explicit.items():
                setattr(args_copy, key, value)
            args_list.append(args_copy)
    else:
        args_list.append(args)

    # initialize Accelerator only if not already in a distributed context
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        accelerator = None
        is_main_process = torch.distributed.get_rank() == 0
    else:
        kwargs_handler = InitProcessGroupKwargs(timeout=datetime.timedelta(seconds=60000))
        accelerator = Accelerator(kwargs_handlers=[kwargs_handler])
        if accelerator.is_main_process:
            is_main_process = True
        else:
            is_main_process = False

    for args in args_list:
        try:
            # if is_main_process and args.wandb_args:  # thoughtfully we should only init wandb once, instead of multiple ranks to avoid network traffics and unwanted behaviors.
            #     wandb_logger = WandbLogger()

            results, samples = cli_evaluate_single(args)
            results_list.append(results)

            if accelerator:
                accelerator.wait_for_everyone()
            elif torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.barrier()
            if is_main_process and args.wandb_args:
                try:
                    wandb_logger.post_init(results)
                    wandb_logger.log_eval_result()
                    if args.wandb_log_samples and samples is not None:
                        wandb_logger.log_eval_samples(samples)
                except Exception as e:
                    eval_logger.info(f"Logging to Weights and Biases failed due to {e}")
                # wandb_logger.finish()

        except Exception as e:
            if args.verbosity == "DEBUG":
                raise e
            else:
                traceback.print_exc()
                eval_logger.error(f"Error during evaluation: {e}. Please set `--verbosity=DEBUG` to get more information.")
                results_list.append(None)

    for args, results in zip(args_list, results_list):
        # cli_evaluate will return none if the process is not the main process (rank 0)
        if results is not None:
            safe_model_args = evaluator.redact_sensitive_data(args.model_args)
            safe_gen_kwargs = evaluator.redact_sensitive_data(args.gen_kwargs)
            print(f"{args.model} ({safe_model_args}), gen_kwargs: ({safe_gen_kwargs}), " f"limit: {args.limit}, offset: {args.offset}, num_fewshot: {args.num_fewshot}, " f"batch_size: {args.batch_size}")
            print(get_eval_banner(branch=results.get("git_branch"), commit=results.get("git_hash")))
            print(make_table(results))
            if "groups" in results:
                print(make_table(results, "groups"))

    if args.wandb_args:
        wandb_logger.run.finish()


def cli_evaluate_single(args: Union[argparse.Namespace, None] = None) -> None:
    selected_task_list = args.tasks.split(",") if args.tasks else None

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")
    task_manager = TaskManager(args.verbosity, include_path=args.include_path, model_name=args.model)

    # Build runtime-only tracker arguments. Credentials must not be appended to
    # args.hf_hub_log_args because the resolved CLI namespace is serialized.
    evaluation_tracker_args = _build_evaluation_tracker_args(args)

    eval_logger.info(f"Evaluation tracker args: {evaluator.redact_sensitive_data(evaluation_tracker_args)}")

    evaluation_tracker = EvaluationTracker(**evaluation_tracker_args)

    if args.write_out:
        eval_logger.warning(
            "DEPRECATION WARNING: --write_out is deprecated and will be removed in v0.5.0. "
            "For debugging and analysis, use --log_samples instead, which saves all model "
            "outputs to files without impacting performance. The --write_out flag only prints "
            "the first few documents to console and provides limited debugging value."
        )

    if args.predict_only:
        args.log_samples = True
    if (args.log_samples or args.predict_only) and not args.output_path:
        raise ValueError("Specify --output_path if providing --log_samples or --predict_only")

    if args.fewshot_as_multiturn and args.apply_chat_template is False:
        raise ValueError("If fewshot_as_multiturn is set, apply_chat_template must be set to True.")

    if (args.num_fewshot is None or args.num_fewshot == 0) and args.fewshot_as_multiturn:
        raise ValueError("If fewshot_as_multiturn is set, num_fewshot must be greater than 0.")

    if args.include_path is not None:
        eval_logger.info(f"Including path: {args.include_path}")

    if "push_samples_to_hub" in evaluation_tracker_args and not args.log_samples:
        eval_logger.warning("Pushing samples to the Hub requires --log_samples to be set. Samples will not be pushed to the Hub.")

    if args.limit is not None and args.limit != -1:
        eval_logger.warning(" --limit SHOULD ONLY BE USED FOR TESTING." "REAL METRICS SHOULD NOT BE COMPUTED USING LIMIT.")
    if args.limit is not None and args.limit < 0 and args.limit != -1:
        raise ValueError("--limit must be -1 or non-negative")
    if args.offset < 0:
        raise ValueError("--offset must be >= 0")

    if os.environ.get("LMMS_EVAL_PLUGINS", None):
        args.include_path = [args.include_path] if args.include_path else []
        for plugin in os.environ["LMMS_EVAL_PLUGINS"].split(","):
            package_tasks_location = importlib.util.find_spec(f"{plugin}.tasks").submodule_search_locations[0]
            args.include_path.append(package_tasks_location)

    if args.tasks is None:
        eval_logger.error("Need to specify task to evaluate.")
        sys.exit()
    elif args.tasks == "list":
        eval_logger.info("Available Tasks:\n - {}".format("\n - ".join(sorted(task_manager.all_tasks))))
        sys.exit()
    elif args.tasks == "list_groups":
        eval_logger.info(task_manager.list_all_tasks(list_subtasks=False, list_tags=False))
        sys.exit()
    elif args.tasks == "list_tags":
        eval_logger.info(task_manager.list_all_tasks(list_groups=False, list_subtasks=False))
        sys.exit()
    elif args.tasks == "list_subtasks":
        eval_logger.info(task_manager.list_all_tasks(list_groups=False, list_tags=False))
        sys.exit()
    else:
        if os.path.isdir(args.tasks):
            import glob

            task_names = []
            yaml_path = os.path.join(args.tasks, "*.yaml")
            for yaml_file in glob.glob(yaml_path):
                config = utils.load_yaml_config(yaml_file)
                task_names.append(config)
        else:
            task_list = args.tasks.split(",")
            task_names = task_manager.match_tasks(task_list)
            for task in [task for task in task_list if task not in task_names]:
                if os.path.isfile(task):
                    config = utils.load_yaml_config(task)
                    task_names.append(config)
            task_missing = [task for task in task_list if task not in task_names and "*" not in task]  # we don't want errors if a wildcard ("*") task name was used

            if task_missing:
                missing = ", ".join(task_missing)
                eval_logger.error(
                    f"Tasks were not found: {missing}\n" f"{utils.SPACING}Try `lmms-eval --tasks list` for list of available tasks",
                )
                raise ValueError(
                    f"Tasks not found: {missing}. Try `lmms-eval --tasks {{list_groups,list_subtasks,list_tags,list}}` to list out all available names for task groupings; only (sub)tasks; tags; or all of the above, or pass '--verbosity DEBUG' to troubleshoot task registration issues."
                )

    eval_logger.info(f"Selected Tasks: {task_names}")
    request_caching_args = request_caching_arg_to_dict(cache_requests=args.cache_requests)
    datetime_str = utils.get_datetime_str(timezone=args.timezone)

    results = evaluator.simple_evaluate(
        model=args.model,
        model_args=args.model_args,
        tasks=task_names,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        max_batch_size=args.max_batch_size,
        device=args.device,
        use_cache=args.use_cache,
        limit=args.limit,
        offset=args.offset,
        check_integrity=args.check_integrity,
        write_out=args.write_out,
        log_samples=args.log_samples,
        evaluation_tracker=evaluation_tracker,
        system_instruction=args.system_instruction,
        apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=args.fewshot_as_multiturn,
        gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
        verbosity=args.verbosity,
        predict_only=args.predict_only,
        random_seed=args.seed[0],
        numpy_random_seed=args.seed[1],
        torch_random_seed=args.seed[2],
        fewshot_random_seed=args.seed[3],
        cli_args=args,
        datetime_str=datetime_str,
        distributed_executor_backend="torchrun" if (torch.distributed.is_available() and torch.distributed.is_initialized()) else "accelerate",
        force_simple=args.force_simple,
        launcher_args=args.launcher_args,
        repeats=args.repeats,
        baseline=args.baseline,
        max_tokens=args.max_tokens,
        **request_caching_args,
    )

    if results is not None:
        if args.log_samples:
            samples = results.pop("samples")
        else:
            samples = None

        pruning_payload = results.pop("_pruning_tracker_data", None)
        if pruning_payload is not None:
            pruning_outputs = _save_pruning_tracker_payload(args, datetime_str, pruning_payload)
            pruning_summary = pruning_outputs["summary"]
            results["pruning_tracker"] = {
                "output_path": str(pruning_outputs["payload_path"]),
                "output_dir": str(pruning_outputs["run_dir"]),
                "summary_path": str(pruning_outputs["summary_path"]),
                "histograms_path": str(pruning_outputs["histograms_path"]),
                "visualizations_path": str(pruning_outputs["visualizations_path"]),
                "plots_dir": str(pruning_outputs["plots_dir"]) if pruning_outputs["plots_dir"] else None,
                "plots_manifest_path": str(pruning_outputs["plots_manifest_path"]) if pruning_outputs["plots_manifest_path"] else None,
                "mlp_modules_path": str(pruning_outputs["mlp_modules_path"]) if pruning_outputs["mlp_modules_path"] else None,
                "attention_modules_path": str(pruning_outputs["attention_modules_path"]) if pruning_outputs["attention_modules_path"] else None,
                "linear_state_modules_path": str(pruning_outputs["linear_state_modules_path"]) if pruning_outputs["linear_state_modules_path"] else None,
                "linear_state_components_path": str(pruning_outputs["linear_state_components_path"]) if pruning_outputs["linear_state_components_path"] else None,
                "token_group_modules_path": str(pruning_outputs["token_group_modules_path"]) if pruning_outputs["token_group_modules_path"] else None,
                "attention_routing_modules_path": str(pruning_outputs["attention_routing_modules_path"]) if pruning_outputs["attention_routing_modules_path"] else None,
                "network_sequence_path": str(pruning_outputs["network_sequence_path"]) if pruning_outputs["network_sequence_path"] else None,
                "num_mlp_modules": len(pruning_payload.get("mlp_stats", {})),
                "num_attn_modules": len(pruning_payload.get("attn_stats", {})),
                "num_linear_state_modules": len(pruning_payload.get("linear_state_counts", {})),
                "num_linear_state_components": len(pruning_payload.get("linear_state_component_stats", {})),
                "mlp_updates": int(sum(pruning_payload.get("mlp_counts", {}).values())),
                "attn_updates": int(sum(pruning_payload.get("attn_counts", {}).values())),
                "linear_state_updates": int(sum(pruning_payload.get("linear_state_counts", {}).values())),
                "linear_state_component_updates": int(sum(pruning_payload.get("linear_state_component_counts", {}).values())),
                "summary": pruning_summary,
            }
            if "config" in results and "pruning_tracker" in results["config"]:
                results["config"]["pruning_tracker"]["output_path"] = str(pruning_outputs["payload_path"])
                results["config"]["pruning_tracker"]["output_dir"] = str(pruning_outputs["run_dir"])
            eval_logger.info(f"Saved pruning statistics to {pruning_outputs['run_dir']}")

        representation_payload = results.pop("_representation_dynamics_data", None)
        if representation_payload is not None:
            representation_outputs = _save_representation_dynamics_payload(args, datetime_str, representation_payload)
            results["representation_dynamics"] = {
                "output_path": str(representation_outputs["payload_path"]),
                "output_dir": str(representation_outputs["run_dir"]),
                "metadata_path": str(representation_outputs["metadata_path"]),
                "records_path": str(representation_outputs["records_path"]),
                "num_records": representation_payload.get("num_records", 0),
                "num_contexts": representation_payload.get("num_contexts", 0),
                "num_layers": len(representation_payload.get("layer_specs", [])),
                "selected_stack": representation_payload.get("selected_stack"),
            }
            if "config" in results and "representation_dynamics" in results["config"]:
                results["config"]["representation_dynamics"]["output_path"] = str(representation_outputs["payload_path"])
                results["config"]["representation_dynamics"]["output_dir"] = str(representation_outputs["run_dir"])
            eval_logger.info(f"Saved representation dynamics to {representation_outputs['run_dir']}")

        # Print token usage summary if available
        if results.get("usage") and "total" in results["usage"]:
            u = results["usage"]["total"]
            eval_logger.info(
                "Token Usage - Input: {} | Output: {} | Reasoning: {} | Total: {} | API Calls: {}",
                f"{u['input_tokens']:,}",
                f"{u['output_tokens']:,}",
                f"{u['reasoning_tokens']:,}",
                f"{u['total_tokens']:,}",
                f"{u['n_api_calls']:,}",
            )
            if results["usage"].get("budget_exceeded"):
                eval_logger.warning("Evaluation stopped early: token budget exceeded. Results are partial.")

        dumped = json.dumps(results, indent=4, default=_handle_non_serializable)
        if args.show_config:
            print(dumped)

        batch_sizes = ",".join(map(str, results["config"]["batch_sizes"]))

        evaluation_tracker.save_results_aggregated(
            results=results,
            samples=samples if args.log_samples else None,
            datetime_str=datetime_str,
        )

        if args.log_samples:
            for task_name, config in results["configs"].items():
                evaluation_tracker.save_results_samples(task_name=task_name, samples=samples[task_name])

        if evaluation_tracker.push_results_to_hub or evaluation_tracker.push_samples_to_hub:
            evaluation_tracker.recreate_metadata_card()

        return results, samples
    return None, None


def print_results(args, results):
    safe_model_args = evaluator.redact_sensitive_data(args.model_args)
    safe_gen_kwargs = evaluator.redact_sensitive_data(args.gen_kwargs)
    print(f"{args.model} ({safe_model_args}),\n" f"gen_kwargs: ({safe_gen_kwargs}),\n" f"limit: {args.limit},\n" f"offset: {args.offset},\n" f"num_fewshot: {args.num_fewshot},\n" f"batch_size: {args.batch_size}")
    print(get_eval_banner(branch=results.get("git_branch"), commit=results.get("git_hash")))
    print(evaluator.make_table(results))
    if "groups" in results:
        print(evaluator.make_table(results, "groups"))


if __name__ == "__main__":
    cli_evaluate()
