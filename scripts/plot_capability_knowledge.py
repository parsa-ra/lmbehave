#!/usr/bin/env python3
"""Compare pruning-tracker traces for capability and knowledge tasks.

The script scans analyzer outputs such as:

    logs/analyzer-google/gemma-3-27B-it/capability-mme/pruning_stats_*/visualizations.json
    logs/analyzer-google/gemma-3-27B-it/knowledge-medqa/pruning_stats_*/visualizations.json

For each model it creates side-by-side plots: capability tasks on the left,
knowledge tasks on the right. It uses tracker JSON/CSV artifacts, not the
already-rendered per-task PNGs, so all tasks share the same plotting style.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CAPABILITY_TASKS = (
    "pope_random",
    "mme",
    "textvqa_val_lite",
    "ocrbench_v2",
    "docvqa_val_lite",
    "chartqa_lite",
    "refcoco_bbox_val_lite",
)

DEFAULT_KNOWLEDGE_TASKS = (
    "ok_vqa_val2014_lite",
    "medqa",
    "mmmu_val",
    "ai2d_lite",
    "gpqa_main_zeroshot",
)

DEFAULT_LINE_METRICS = (
    "mlp_mean_by_network_order",
    "mlp_active_fraction_by_network_order",
    "attention_peak_mass_by_network_order",
    "attention_entropy_by_network_order",
    "attention_valid_fraction_by_network_order",
    "linear_state_delta_ratio_by_network_order",
    "linear_state_output_ratio_by_network_order",
    "linear_state_output_input_cosine_by_network_order",
)

SUMMARY_METRICS = (
    ("mlp.mean", "MLP mean abs activation"),
    ("mlp.mean_active_fraction", "MLP active fraction"),
    ("attention.mean", "Attention mean mass"),
    ("attention.mean_entropy", "Attention entropy"),
    ("attention.mean_peak_mass", "Attention peak mass"),
    ("attention.attention_valid_fraction", "Attention valid fraction"),
    ("linear_state.mean_output_input_rms_ratio", "Linear-state output/input RMS"),
    ("linear_state.mean_input_output_delta_rms_ratio", "Linear-state delta/input RMS"),
    ("linear_state.mean_output_input_cosine", "Linear-state input/output cosine"),
)

TOKEN_GROUP_METRICS = (
    ("output_rms", "Output RMS"),
    ("input_output_delta_rms_ratio", "Delta/input RMS"),
    ("output_input_cosine", "Input/output cosine"),
)

TOKEN_GROUP_STREAMS = ("mlp", "attention", "linear_state")
TOKEN_GROUP_PHASES = {
    "vision": "prefill",
    "text": "prefill",
    "sink_first_4": "prefill",
    "generated": "decode",
}

ATTENTION_ROUTE_SPECS = (
    ("text", "vision"),
    ("vision", "text"),
    ("text", "text"),
    ("vision", "vision"),
    ("text", "sink_first_4"),
    ("vision", "sink_first_4"),
)

ATTENTION_ROUTE_DERIVED_METRICS = (
    ("cross_modal_mean", "Cross-modal mean"),
    ("intra_modal_mean", "Intra-modal mean"),
    ("cross_intra_ratio", "Cross/intra ratio"),
    ("sink_reliance", "Sink reliance"),
)


@dataclass(frozen=True)
class RunRecord:
    model_id: str
    category: str
    folder_category: str | None
    task: str
    task_dir: Path
    run_dir: Path
    visualizations_path: Path
    summary_path: Path
    datetime: str

    @property
    def label(self) -> str:
        return self.task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot capability-vs-knowledge pruning tracker traces side by side "
            "for every model under logs/analyzer-*."
        )
    )
    parser.add_argument("--log-root", type=Path, default=Path("logs"), help="Root directory containing analyzer logs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/capability_knowledge_comparison"),
        help="Directory where comparison plots and manifests are written.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_LINE_METRICS),
        help="Line-chart metric keys from visualizations.json to plot.",
    )
    parser.add_argument(
        "--capability-tasks",
        nargs="+",
        default=list(DEFAULT_CAPABILITY_TASKS),
        help="Task names treated as capability tasks, even if older folders are unprefixed.",
    )
    parser.add_argument(
        "--knowledge-tasks",
        nargs="+",
        default=list(DEFAULT_KNOWLEDGE_TASKS),
        help="Task names treated as knowledge tasks, even if older folders are unprefixed.",
    )
    parser.add_argument(
        "--model",
        action="append",
        dest="model_filters",
        default=[],
        help="Only include model labels containing this substring. Can be passed multiple times.",
    )
    parser.add_argument(
        "--prefer-folder-prefix",
        action="store_true",
        help=(
            "Trust capability-/knowledge- folder prefixes over task-name defaults. "
            "By default, known task names fix older mis-prefixed folders."
        ),
    )
    parser.add_argument(
        "--no-category-mean",
        action="store_true",
        help="Do not draw the thick per-category mean line over task traces.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="PNG resolution.")
    parser.add_argument(
        "--list-metrics",
        action="store_true",
        help="List available chart keys discovered in visualizations.json and exit.",
    )
    return parser.parse_args()


def sanitize_stem(text: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", text.strip())
    return safe.strip("._-") or "model"


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    except OSError as exc:
        print(f"warning: failed to read {path}: {exc}", file=sys.stderr)
        return []


def parse_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def mean_finite(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not finite:
        return None
    return sum(finite) / len(finite)


def weighted_mean_finite(values_and_weights: Iterable[tuple[float | None, float | None]]) -> float | None:
    numerator = 0.0
    denominator = 0.0
    for value, weight in values_and_weights:
        if value is None or weight is None:
            continue
        if not math.isfinite(value) or not math.isfinite(weight) or weight <= 0:
            continue
        numerator += value * weight
        denominator += weight
    if denominator <= 0:
        return None
    return numerator / denominator


def infer_datetime(run_dir: Path, visualizations_path: Path) -> str:
    match = re.search(r"(\d{8}_\d{6})", run_dir.name)
    if match:
        return match.group(1)
    try:
        metadata = read_json(visualizations_path).get("metadata", {})
        value = metadata.get("datetime")
        if isinstance(value, str):
            return value
    except (OSError, json.JSONDecodeError):
        pass
    return ""


def analyzer_model_id(log_root: Path, visualizations_path: Path) -> str:
    rel_parts = visualizations_path.relative_to(log_root).parts
    analyzer_index = next((i for i, part in enumerate(rel_parts) if part.startswith("analyzer-")), None)
    if analyzer_index is None:
        return "unknown"

    analyzer_name = rel_parts[analyzer_index]
    provider = analyzer_name.removeprefix("analyzer-")
    model_parts = rel_parts[analyzer_index + 1 : -3]
    if not model_parts:
        return provider or analyzer_name
    return "/".join((provider, *model_parts))


def split_task_dir_name(task_dir_name: str) -> tuple[str | None, str]:
    for category in ("capability", "knowledge"):
        prefix = f"{category}-"
        if task_dir_name.startswith(prefix):
            return category, task_dir_name[len(prefix) :]
    return None, task_dir_name


def classify_task(
    task_dir_name: str,
    capability_tasks: set[str],
    knowledge_tasks: set[str],
    prefer_folder_prefix: bool,
) -> tuple[str | None, str | None, str]:
    folder_category, task = split_task_dir_name(task_dir_name)
    semantic_category = None
    if task in capability_tasks:
        semantic_category = "capability"
    elif task in knowledge_tasks:
        semantic_category = "knowledge"

    if prefer_folder_prefix and folder_category is not None:
        return folder_category, folder_category, task
    return semantic_category or folder_category, folder_category, task


def discover_runs(
    log_root: Path,
    capability_tasks: set[str],
    knowledge_tasks: set[str],
    prefer_folder_prefix: bool,
    model_filters: Iterable[str],
) -> list[RunRecord]:
    filters = tuple(model_filters)
    records: list[RunRecord] = []

    for visualizations_path in sorted(log_root.glob("analyzer-*/**/visualizations.json")):
        run_dir = visualizations_path.parent
        if not run_dir.name.startswith("pruning_stats_"):
            continue
        task_dir = run_dir.parent
        category, folder_category, task = classify_task(
            task_dir.name,
            capability_tasks,
            knowledge_tasks,
            prefer_folder_prefix,
        )
        if category not in {"capability", "knowledge"}:
            continue

        model_id = analyzer_model_id(log_root, visualizations_path)
        if filters and not any(token in model_id for token in filters):
            continue

        records.append(
            RunRecord(
                model_id=model_id,
                category=category,
                folder_category=folder_category,
                task=task,
                task_dir=task_dir,
                run_dir=run_dir,
                visualizations_path=visualizations_path,
                summary_path=run_dir / "summary.json",
                datetime=infer_datetime(run_dir, visualizations_path),
            )
        )

    grouped: dict[tuple[str, str, str], list[RunRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.model_id, record.category, record.task)].append(record)

    selected = [max(items, key=run_sort_key) for items in grouped.values()]
    return sorted(selected, key=lambda item: (item.model_id, item.category, item.task))


def run_sort_key(record: RunRecord) -> tuple[str, float]:
    try:
        mtime = record.run_dir.stat().st_mtime
    except OSError:
        mtime = 0.0
    return record.datetime, mtime


def configure_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/vlm_pruning_matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/vlm_pruning_cache")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for plotting. Run with the project environment, "
            "for example `.venv/bin/python scripts/plot_capability_knowledge.py ...`, "
            "or install matplotlib in the active Python environment."
        ) from exc

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def extract_line(chart: dict[str, Any]) -> tuple[list[float], list[float], str, str]:
    x_key = chart.get("x_key", "sequence_index")
    y_key = chart.get("y_key", "value")
    x_values: list[float] = []
    y_values: list[float] = []

    for point in chart.get("data", []):
        if not isinstance(point, dict):
            continue
        x_value = point.get(x_key)
        y_value = point.get(y_key)
        if is_finite_number(x_value) and is_finite_number(y_value):
            x_values.append(float(x_value))
            y_values.append(float(y_value))

    return x_values, y_values, str(chart.get("x_label", x_key)), str(chart.get("y_label", y_key))


def mean_by_x(series: list[tuple[list[float], list[float]]]) -> tuple[list[float], list[float]]:
    grouped: dict[float, list[float]] = defaultdict(list)
    for x_values, y_values in series:
        for x_value, y_value in zip(x_values, y_values):
            grouped[x_value].append(y_value)
    x_mean = sorted(grouped)
    y_mean = [sum(grouped[x_value]) / len(grouped[x_value]) for x_value in x_mean]
    return x_mean, y_mean


def load_visualizations(records: list[RunRecord]) -> dict[Path, dict[str, Any]]:
    loaded: dict[Path, dict[str, Any]] = {}
    for record in records:
        try:
            loaded[record.visualizations_path] = read_json(record.visualizations_path)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: failed to read {record.visualizations_path}: {exc}", file=sys.stderr)
    return loaded


def plot_line_metric_for_model(
    plt: Any,
    model_id: str,
    records: list[RunRecord],
    visualizations: dict[Path, dict[str, Any]],
    metric: str,
    output_dir: Path,
    draw_category_mean: bool,
    dpi: int,
) -> Path | None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
    categories = ("capability", "knowledge")
    chart_title = metric
    any_plotted = False

    for axis, category in zip(axes, categories):
        axis_records = [record for record in records if record.category == category]
        axis_records.sort(key=lambda record: record.task)
        category_series: list[tuple[list[float], list[float]]] = []
        plotted_count = 0
        x_label = "sequence index"
        y_label = metric

        for record in axis_records:
            chart = visualizations.get(record.visualizations_path, {}).get("charts", {}).get(metric)
            if not isinstance(chart, dict) or chart.get("chart_type") != "line":
                continue
            x_values, y_values, x_label, y_label = extract_line(chart)
            if not x_values:
                continue
            chart_title = str(chart.get("title") or chart_title)
            axis.plot(x_values, y_values, linewidth=1.1, alpha=0.65, label=record.label)
            category_series.append((x_values, y_values))
            plotted_count += 1
            any_plotted = True

        if draw_category_mean and len(category_series) > 1:
            x_mean, y_mean = mean_by_x(category_series)
            axis.plot(x_mean, y_mean, color="black", linewidth=2.4, label=f"{category} mean")

        axis.set_title(f"{category.capitalize()} ({plotted_count} tasks)")
        axis.set_xlabel(x_label)
        if axis is axes[0]:
            axis.set_ylabel(y_label)
        axis.grid(True, alpha=0.25, linewidth=0.7)
        if plotted_count:
            axis.legend(fontsize=7, loc="best", frameon=False)
        else:
            axis.text(
                0.5,
                0.5,
                f"No {category} traces",
                ha="center",
                va="center",
                transform=axis.transAxes,
                fontsize=11,
            )

    if not any_plotted:
        plt.close(fig)
        return None

    fig.suptitle(f"{model_id} - {chart_title}", fontsize=13)
    model_dir = output_dir / sanitize_stem(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_path = model_dir / f"{sanitize_stem(metric)}.png"
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def nested_get(data: dict[str, Any], dotted_key: str) -> float | None:
    current: Any = data
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    if is_finite_number(current):
        return float(current)
    return None


def load_summary_values(record: RunRecord) -> dict[str, float | None]:
    if not record.summary_path.is_file():
        return {key: None for key, _ in SUMMARY_METRICS}
    try:
        summary = read_json(record.summary_path).get("summary", {})
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: failed to read {record.summary_path}: {exc}", file=sys.stderr)
        return {key: None for key, _ in SUMMARY_METRICS}
    return {key: nested_get(summary, key) for key, _ in SUMMARY_METRICS}


def write_run_summary_csv(records: list[RunRecord], output_dir: Path) -> Path:
    csv_path = output_dir / "run_summary.csv"
    fieldnames = [
        "model",
        "category",
        "folder_category",
        "task",
        "datetime",
        "run_dir",
        *[key for key, _ in SUMMARY_METRICS],
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row: dict[str, Any] = {
                "model": record.model_id,
                "category": record.category,
                "folder_category": record.folder_category or "",
                "task": record.task,
                "datetime": record.datetime,
                "run_dir": str(record.run_dir),
            }
            row.update(load_summary_values(record))
            writer.writerow(row)
    return csv_path


def token_group_column(stream: str, phase: str, group: str, metric: str) -> str:
    return f"token_group.{stream}.{phase}.{group}.{metric}"


def route_column(query_group: str, key_group: str) -> str:
    return f"attention_route.prefill.{query_group}_to_{key_group}"


def token_group_label(stream: str, group: str) -> str:
    return f"{stream}:{group}"


def route_label(query_group: str, key_group: str) -> str:
    return f"{query_group}->{key_group}"


def load_token_group_values(record: RunRecord) -> dict[str, float | None]:
    rows = read_csv_rows(record.run_dir / "token_group_modules.csv")
    values: dict[str, float | None] = {}
    for stream in TOKEN_GROUP_STREAMS:
        for group, phase in TOKEN_GROUP_PHASES.items():
            filtered = [
                row
                for row in rows
                if row.get("stream") == stream and row.get("phase") == phase and row.get("token_group") == group
            ]
            for metric, _ in TOKEN_GROUP_METRICS:
                values[token_group_column(stream, phase, group, metric)] = mean_finite(
                    parse_float(row.get(metric)) for row in filtered
                )
    return values


def load_attention_route_values(record: RunRecord) -> dict[str, float | None]:
    rows = read_csv_rows(record.run_dir / "attention_routing_modules.csv")
    values: dict[str, float | None] = {}
    route_values: dict[tuple[str, str], float | None] = {}

    for query_group, key_group in ATTENTION_ROUTE_SPECS:
        filtered = [
            row
            for row in rows
            if row.get("phase") == "prefill"
            and row.get("query_group") == query_group
            and row.get("key_group") == key_group
        ]
        value = weighted_mean_finite(
            (
                parse_float(row.get("mean_mass")),
                (parse_float(row.get("query_count")) or 0.0) * (parse_float(row.get("num_heads")) or 0.0),
            )
            for row in filtered
        )
        route_values[(query_group, key_group)] = value
        values[route_column(query_group, key_group)] = value

    cross_modal = mean_finite([route_values.get(("text", "vision")), route_values.get(("vision", "text"))])
    intra_modal = mean_finite([route_values.get(("text", "text")), route_values.get(("vision", "vision"))])
    sink_reliance = mean_finite([route_values.get(("text", "sink_first_4")), route_values.get(("vision", "sink_first_4"))])
    cross_intra_ratio = None
    if cross_modal is not None and intra_modal is not None and abs(intra_modal) > 1e-12:
        cross_intra_ratio = cross_modal / intra_modal

    values["attention_route.cross_modal_mean"] = cross_modal
    values["attention_route.intra_modal_mean"] = intra_modal
    values["attention_route.cross_intra_ratio"] = cross_intra_ratio
    values["attention_route.sink_reliance"] = sink_reliance
    return values


def write_token_group_summary_csv(records: list[RunRecord], output_dir: Path) -> Path:
    csv_path = output_dir / "token_group_behavior_summary.csv"
    metric_columns = [
        token_group_column(stream, phase, group, metric)
        for stream in TOKEN_GROUP_STREAMS
        for group, phase in TOKEN_GROUP_PHASES.items()
        for metric, _ in TOKEN_GROUP_METRICS
    ]
    fieldnames = ["model", "category", "folder_category", "task", "datetime", "run_dir", *metric_columns]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row: dict[str, Any] = {
                "model": record.model_id,
                "category": record.category,
                "folder_category": record.folder_category or "",
                "task": record.task,
                "datetime": record.datetime,
                "run_dir": str(record.run_dir),
            }
            row.update(load_token_group_values(record))
            writer.writerow(row)
    return csv_path


def write_attention_routing_summary_csv(records: list[RunRecord], output_dir: Path) -> Path:
    csv_path = output_dir / "attention_routing_summary.csv"
    route_columns = [route_column(query_group, key_group) for query_group, key_group in ATTENTION_ROUTE_SPECS]
    derived_columns = [f"attention_route.{key}" for key, _ in ATTENTION_ROUTE_DERIVED_METRICS]
    fieldnames = ["model", "category", "folder_category", "task", "datetime", "run_dir", *route_columns, *derived_columns]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row: dict[str, Any] = {
                "model": record.model_id,
                "category": record.category,
                "folder_category": record.folder_category or "",
                "task": record.task,
                "datetime": record.datetime,
                "run_dir": str(record.run_dir),
            }
            row.update(load_attention_route_values(record))
            writer.writerow(row)
    return csv_path


def plot_summary_bars_for_model(
    plt: Any,
    model_id: str,
    records: list[RunRecord],
    output_dir: Path,
    dpi: int,
) -> Path | None:
    values_by_record = {record: load_summary_values(record) for record in records}
    available_metrics = [
        (key, label)
        for key, label in SUMMARY_METRICS
        if any(values.get(key) is not None for values in values_by_record.values())
    ]
    if not available_metrics:
        return None

    cols = 3
    rows = math.ceil(len(available_metrics) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, max(3.4, rows * 3.2)))
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]

    for axis, (metric_key, label) in zip(axes_flat, available_metrics):
        category_values: dict[str, list[float]] = {"capability": [], "knowledge": []}
        for record, metric_values in values_by_record.items():
            value = metric_values.get(metric_key)
            if value is not None:
                category_values[record.category].append(value)

        means: list[float] = []
        for category in ("capability", "knowledge"):
            values = category_values[category]
            means.append(sum(values) / len(values) if values else math.nan)

        x_positions = [0, 1]
        bar_colors = ["#4C78A8", "#F58518"]
        axis.bar(x_positions, means, color=bar_colors, alpha=0.85)
        for x_position, category in zip(x_positions, ("capability", "knowledge")):
            values = category_values[category]
            if not values:
                continue
            axis.scatter([x_position] * len(values), values, color="black", s=12, alpha=0.65, zorder=3)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(["Capability", "Knowledge"], rotation=20, ha="right")
        axis.set_title(label, fontsize=10)
        axis.grid(True, axis="y", alpha=0.25, linewidth=0.7)

    for axis in axes_flat[len(available_metrics) :]:
        axis.axis("off")

    fig.suptitle(f"{model_id} - category scalar summaries", fontsize=13)
    model_dir = output_dir / sanitize_stem(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_path = model_dir / "category_summary_bars.png"
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def category_values(
    records: list[RunRecord],
    values_by_record: dict[RunRecord, dict[str, float | None]],
    column: str,
) -> dict[str, list[float]]:
    grouped: dict[str, list[float]] = {"capability": [], "knowledge": []}
    for record in records:
        value = values_by_record.get(record, {}).get(column)
        if value is not None and math.isfinite(value):
            grouped[record.category].append(float(value))
    return grouped


def plot_category_scatter_bars(
    axis: Any,
    x_positions: list[int],
    labels: list[str],
    records: list[RunRecord],
    values_by_record: dict[RunRecord, dict[str, float | None]],
    columns: list[str],
    y_label: str,
) -> bool:
    colors = {"capability": "#4C78A8", "knowledge": "#F58518"}
    offsets = {"capability": -0.18, "knowledge": 0.18}
    width = 0.32
    any_plotted = False

    for category in ("capability", "knowledge"):
        means: list[float] = []
        for column in columns:
            values = category_values(records, values_by_record, column)[category]
            means.append(sum(values) / len(values) if values else math.nan)
        axis.bar(
            [x + offsets[category] for x in x_positions],
            means,
            width=width,
            color=colors[category],
            alpha=0.85,
            label=category.capitalize(),
        )
        for item_index, column in enumerate(columns):
            category_records = [record for record in records if record.category == category]
            finite_values = [
                values_by_record.get(record, {}).get(column)
                for record in category_records
                if values_by_record.get(record, {}).get(column) is not None
            ]
            finite_values = [float(value) for value in finite_values if math.isfinite(float(value))]
            if not finite_values:
                continue
            jitter_step = min(0.06, 0.14 / max(len(finite_values), 1))
            jitters = [
                (point_index - (len(finite_values) - 1) / 2.0) * jitter_step
                for point_index in range(len(finite_values))
            ]
            axis.scatter(
                [x_positions[item_index] + offsets[category] + jitter for jitter in jitters],
                finite_values,
                color="black",
                s=12,
                alpha=0.62,
                zorder=3,
            )
            any_plotted = True

    axis.set_ylabel(y_label)
    axis.set_xticks(x_positions)
    axis.set_xticklabels(labels, rotation=35, ha="right")
    axis.grid(True, axis="y", alpha=0.25, linewidth=0.7)
    return any_plotted


def plot_token_group_bars_for_model(
    plt: Any,
    model_id: str,
    records: list[RunRecord],
    output_dir: Path,
    dpi: int,
) -> Path | None:
    values_by_record = {record: load_token_group_values(record) for record in records}
    specs = [
        (stream, phase, group)
        for stream in TOKEN_GROUP_STREAMS
        for group, phase in TOKEN_GROUP_PHASES.items()
        if any(
            values.get(token_group_column(stream, phase, group, metric)) is not None
            for values in values_by_record.values()
            for metric, _ in TOKEN_GROUP_METRICS
        )
    ]
    if not specs:
        return None

    fig, axes = plt.subplots(len(TOKEN_GROUP_METRICS), 1, figsize=(max(11, len(specs) * 0.9), 9.5), sharex=True)
    axes_flat = list(axes.flat) if hasattr(axes, "flat") else [axes]
    labels = [token_group_label(stream, group) for stream, _, group in specs]
    x_positions = list(range(len(specs)))
    any_plotted = False

    for axis, (metric, metric_label) in zip(axes_flat, TOKEN_GROUP_METRICS):
        columns = [token_group_column(stream, phase, group, metric) for stream, phase, group in specs]
        any_plotted = (
            plot_category_scatter_bars(axis, x_positions, labels, records, values_by_record, columns, metric_label)
            or any_plotted
        )
        axis.set_title(metric_label, fontsize=10)

    if not any_plotted:
        plt.close(fig)
        return None

    handles, labels_for_legend = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles[:2], labels_for_legend[:2], loc="upper right", frameon=False)
    fig.suptitle(f"{model_id} - token-group behavior by task category", fontsize=13)
    model_dir = output_dir / sanitize_stem(model_id)
    model_dir.mkdir(parents=True, exist_ok=True)
    plot_path = model_dir / "token_group_behavior_bars.png"
    fig.savefig(plot_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def plot_attention_route_bars_for_model(
    plt: Any,
    model_id: str,
    records: list[RunRecord],
    output_dir: Path,
    dpi: int,
) -> tuple[Path | None, Path | None]:
    values_by_record = {record: load_attention_route_values(record) for record in records}

    route_specs = [
        (query_group, key_group)
        for query_group, key_group in ATTENTION_ROUTE_SPECS
        if any(values.get(route_column(query_group, key_group)) is not None for values in values_by_record.values())
    ]
    route_plot_path = None
    if route_specs:
        fig, axis = plt.subplots(1, 1, figsize=(max(9, len(route_specs) * 1.15), 4.8))
        labels = [route_label(query_group, key_group) for query_group, key_group in route_specs]
        columns = [route_column(query_group, key_group) for query_group, key_group in route_specs]
        any_plotted = plot_category_scatter_bars(
            axis,
            list(range(len(route_specs))),
            labels,
            records,
            values_by_record,
            columns,
            "Mean attention mass",
        )
        if any_plotted:
            axis.set_title("Prefill attention routing", fontsize=10)
            axis.legend(frameon=False)
            fig.suptitle(f"{model_id} - attention routing by task category", fontsize=13)
            model_dir = output_dir / sanitize_stem(model_id)
            model_dir.mkdir(parents=True, exist_ok=True)
            route_plot_path = model_dir / "attention_routing_bars.png"
            fig.savefig(route_plot_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    derived_specs = [
        (key, label)
        for key, label in ATTENTION_ROUTE_DERIVED_METRICS
        if any(values.get(f"attention_route.{key}") is not None for values in values_by_record.values())
    ]
    derived_plot_path = None
    if derived_specs:
        fig, axis = plt.subplots(1, 1, figsize=(max(8, len(derived_specs) * 1.4), 4.8))
        labels = [label for _, label in derived_specs]
        columns = [f"attention_route.{key}" for key, _ in derived_specs]
        any_plotted = plot_category_scatter_bars(
            axis,
            list(range(len(derived_specs))),
            labels,
            records,
            values_by_record,
            columns,
            "Route-derived value",
        )
        if any_plotted:
            axis.set_title("Derived routing contrasts", fontsize=10)
            axis.legend(frameon=False)
            fig.suptitle(f"{model_id} - routing contrasts by task category", fontsize=13)
            model_dir = output_dir / sanitize_stem(model_id)
            model_dir.mkdir(parents=True, exist_ok=True)
            derived_plot_path = model_dir / "attention_routing_contrasts.png"
            fig.savefig(derived_plot_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    return route_plot_path, derived_plot_path


def write_plots_manifest(output_dir: Path, payload: dict[str, Any]) -> Path:
    manifest_path = output_dir / "plots_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return manifest_path


def list_available_metrics(records: list[RunRecord]) -> None:
    metric_types: dict[str, set[str]] = defaultdict(set)
    for record in records:
        try:
            charts = read_json(record.visualizations_path).get("charts", {})
        except (OSError, json.JSONDecodeError) as exc:
            print(f"warning: failed to read {record.visualizations_path}: {exc}", file=sys.stderr)
            continue
        for name, chart in charts.items():
            if isinstance(chart, dict):
                metric_types[name].add(str(chart.get("chart_type", "unknown")))
    for metric in sorted(metric_types):
        print(f"{metric}\t{','.join(sorted(metric_types[metric]))}")


def main() -> int:
    args = parse_args()
    capability_tasks = set(args.capability_tasks)
    knowledge_tasks = set(args.knowledge_tasks)
    records = discover_runs(
        args.log_root,
        capability_tasks,
        knowledge_tasks,
        args.prefer_folder_prefix,
        args.model_filters,
    )

    if not records:
        print(f"No capability/knowledge analyzer runs found under {args.log_root}", file=sys.stderr)
        return 1

    if args.list_metrics:
        list_available_metrics(records)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plt = configure_matplotlib()
    visualizations = load_visualizations(records)
    records_by_model: dict[str, list[RunRecord]] = defaultdict(list)
    for record in records:
        records_by_model[record.model_id].append(record)

    plots_payload: dict[str, Any] = {
        "log_root": str(args.log_root),
        "output_dir": str(args.output_dir),
        "metrics": list(args.metrics),
        "models": {},
    }

    for model_id in sorted(records_by_model):
        model_records = records_by_model[model_id]
        model_payload: dict[str, Any] = {
            "line_plots": {},
            "summary_plot": None,
            "token_group_plot": None,
            "attention_routing_plot": None,
            "attention_routing_contrast_plot": None,
        }
        for metric in args.metrics:
            plot_path = plot_line_metric_for_model(
                plt,
                model_id,
                model_records,
                visualizations,
                metric,
                args.output_dir,
                draw_category_mean=not args.no_category_mean,
                dpi=args.dpi,
            )
            if plot_path is not None:
                model_payload["line_plots"][metric] = str(plot_path)
        summary_plot = plot_summary_bars_for_model(plt, model_id, model_records, args.output_dir, args.dpi)
        if summary_plot is not None:
            model_payload["summary_plot"] = str(summary_plot)
        token_group_plot = plot_token_group_bars_for_model(plt, model_id, model_records, args.output_dir, args.dpi)
        if token_group_plot is not None:
            model_payload["token_group_plot"] = str(token_group_plot)
        routing_plot, routing_contrast_plot = plot_attention_route_bars_for_model(
            plt,
            model_id,
            model_records,
            args.output_dir,
            args.dpi,
        )
        if routing_plot is not None:
            model_payload["attention_routing_plot"] = str(routing_plot)
        if routing_contrast_plot is not None:
            model_payload["attention_routing_contrast_plot"] = str(routing_contrast_plot)
        plots_payload["models"][model_id] = model_payload

    summary_csv = write_run_summary_csv(records, args.output_dir)
    token_group_csv = write_token_group_summary_csv(records, args.output_dir)
    attention_routing_csv = write_attention_routing_summary_csv(records, args.output_dir)
    plots_payload["summary_csvs"] = {
        "scalar": str(summary_csv),
        "token_group_behavior": str(token_group_csv),
        "attention_routing": str(attention_routing_csv),
    }
    manifest_path = write_plots_manifest(args.output_dir, plots_payload)

    num_line_plots = sum(len(model["line_plots"]) for model in plots_payload["models"].values())
    num_token_group_plots = sum(1 for model in plots_payload["models"].values() if model.get("token_group_plot"))
    num_routing_plots = sum(1 for model in plots_payload["models"].values() if model.get("attention_routing_plot"))
    num_routing_contrast_plots = sum(
        1 for model in plots_payload["models"].values() if model.get("attention_routing_contrast_plot")
    )
    print(f"Selected {len(records)} latest task runs across {len(records_by_model)} models.")
    print(f"Wrote {num_line_plots} line comparison plots.")
    print(f"Wrote {num_token_group_plots} token-group comparison plots.")
    print(f"Wrote {num_routing_plots} attention-routing comparison plots.")
    print(f"Wrote {num_routing_contrast_plots} attention-routing contrast plots.")
    print(f"Wrote scalar summary CSV: {summary_csv}")
    print(f"Wrote token-group summary CSV: {token_group_csv}")
    print(f"Wrote attention-routing summary CSV: {attention_routing_csv}")
    print(f"Wrote plots manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
