import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_parser_gate_and_final_summary_accept_only_one_answer_label():
    gate = load_script("validate_response_trajectory_samples.py")
    summary = load_script("summarize_response_trajectory_smoke.py")
    accepted = ["A", "b", ["C"], [["d"]]]
    rejected = ["[invalid]", "Final answer: A", ["A", "B"], None]
    for value in accepted:
        assert gate.parsed_answer(value) is not None
        assert summary.parsed_answer(value) is not None
    for value in rejected:
        assert gate.parsed_answer(value) is None
        assert summary.parsed_answer(value) is None
