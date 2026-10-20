import csv
import importlib.util
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    path = ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


direct = load_script("audit_direct_answer_validity")
longitudinal = load_script("audit_longitudinal_response_validity")
additivity = load_script("audit_residual_additivity")
summary = load_script("summarize_main_track_controls")


def test_parser_validity_accepts_only_one_declared_label():
    labels = {"A", "B", "C", "D"}
    assert direct.response_is_valid(" A ", labels)
    assert direct.response_is_valid(["D"], labels)
    assert not direct.response_is_valid([], labels)
    assert not direct.response_is_valid(["A", "B"], labels)
    assert not direct.response_is_valid("The answer is A", labels)


def test_longitudinal_audit_rejects_invalid_response_scored_correct():
    line = json.dumps({"filtered_resps": ["invalid"], "exact_match": 1.0})
    with pytest.raises(ValueError, match="invalid parser output"):
        longitudinal.rows_from_lines([line])


def test_explicit_additivity_metadata_takes_precedence():
    payload = {
        "layer_specs": [
            {"residual_change_equals_reported_branch_sum": False},
            {"residual_change_equals_reported_branch_sum": False},
        ]
    }
    expected, source = additivity.expected_additivity(payload)
    assert expected is False
    assert source == "explicit artifact metadata"


def test_validity_summary_aggregates_by_sample_count(tmp_path):
    path = tmp_path / "validity.csv"
    rows = [
        {"model": "m", "task": "a", "examples": 100, "valid_responses": 50, "raw_accuracy": 0.2},
        {"model": "m", "task": "b", "examples": 20, "valid_responses": 20, "raw_accuracy": 0.5},
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    aggregated = summary.aggregate_validity(path, ("model",))
    assert len(aggregated) == 1
    assert aggregated[0]["examples"] == 120
    assert aggregated[0]["valid_responses"] == 70
    assert aggregated[0]["valid_response_rate"] == pytest.approx(70 / 120)
    assert aggregated[0]["raw_accuracy"] == pytest.approx(30 / 120)
    assert aggregated[0]["accuracy_conditional_on_valid_response"] == pytest.approx(30 / 70)


def test_telemetry_summary_uses_all_samples(tmp_path):
    path = tmp_path / "gpu.csv"
    path.write_text(
        "2026/08/15 00:00:00.000, 0, 10, 100, 900\n"
        "2026/08/15 00:00:10.000, 100, 20, 500, 900\n",
        encoding="utf-8",
    )
    row = summary.telemetry_rows(path)[0]
    assert row["samples"] == 2
    assert row["mean_gpu_utilization_percent"] == 50
    assert row["peak_memory_mib"] == 20
    assert row["mean_power_w"] == 300
    assert row["power_limit_w"] == 900
