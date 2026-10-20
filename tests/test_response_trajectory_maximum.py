import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "analyze_response_trajectory_maximum.py"
SPEC = importlib.util.spec_from_file_location("trajectory_maximum", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def row(phase, position, layer, values):
    return {
        "model": "model",
        "phase": phase,
        "trajectory_position": position,
        "layer": layer,
        "stream": "residual_state",
        "module": f"layers.{layer}",
        "task_order": ["k1", "k2", "r1", "r2"],
        "assignment_ids": ["0011", "1100", "0101", "1010"],
        "fdr": values,
    }


def test_selection_is_repeated_for_every_assignment():
    rows = [
        row("prefill", "aggregate", 0, [1.0, 1.0, 1.0, 1.0]),
        row("decode", "step_0000", 0, [4.0, 4.0, 2.0, 2.0]),
        row("prefill", "aggregate", 1, [1.0, 1.0, 1.0, 1.0]),
        row("decode", "step_0000", 1, [2.0, 2.0, 3.0, 3.0]),
    ]
    sites, output = MODULE.analyze(rows, common_prefix=1)
    result = output["report"]["maximum_decode_minus_prefill_fisher"]
    assert len(sites) == 2
    assert result["observed"] == 3.0
    assert result["randomization_p"] == 0.5
    assert result["selected_site"]["layer"] == 0
    assert result["complement_symmetry"]["verified"]


def test_missing_common_prefix_position_is_rejected():
    rows = [
        row("prefill", "aggregate", 0, [1.0, 1.0, 1.0, 1.0]),
        row("decode", "step_0000", 0, [2.0, 2.0, 2.0, 2.0]),
    ]
    with pytest.raises(ValueError, match="Incomplete eligible response positions"):
        MODULE.analyze(rows, common_prefix=2)


def test_relative_bins_are_supported_without_calling_them_token_steps():
    rows = [
        row("prefill", "aggregate", 0, [1.0, 1.0, 1.0, 1.0]),
        row("decode", "relative_bin_0000", 0, [2.0, 2.0, 1.5, 1.5]),
        row("decode", "relative_bin_0001", 0, [3.0, 3.0, 2.0, 2.0]),
    ]
    sites, output = MODULE.analyze(rows, common_prefix=2)
    assert output["report"]["position_kind"] == "relative_bin"
    assert {site["trajectory_position"] for site in sites} == {
        "relative_bin_0000",
        "relative_bin_0001",
    }
