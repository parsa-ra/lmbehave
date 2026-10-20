from pathlib import Path

import yaml

from lmms_eval.filters.extraction import RegexFilter


TASK_ROOT = (
    Path(__file__).resolve().parents[2]
    / "lmms_eval"
    / "tasks"
    / "representation_dynamics_response_trajectory"
)


def test_response_trajectory_task_family_is_complete_and_uniform():
    task_files = sorted(TASK_ROOT.glob("repdyn_response_mmlu_*.yaml"))
    assert len(task_files) == 16
    dataset_names = set()
    for path in task_files:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        assert payload["include"] == "_default_template_yaml"
        assert payload["task"] == path.stem
        dataset_names.add(payload["dataset_name"])
    assert len(dataset_names) == 16


def test_response_trajectory_template_has_one_matched_generation_contract():
    payload = yaml.safe_load((TASK_ROOT / "_default_template_yaml").read_text(encoding="utf-8"))
    prompt = payload["doc_to_text"]
    assert "exactly two short explanatory sentences" in prompt
    assert "at most 60 words" in prompt
    assert "replacing X with one of A, B, C, or D even if uncertain" in prompt
    assert "Begin with exactly `Final answer: X`" in prompt
    assert "Final answer: X" in prompt
    assert payload["generation_kwargs"] == {
        "max_new_tokens": 256,
        "temperature": 0.0,
        "do_sample": False,
        "until": [],
    }
    regex = payload["filter_list"][0]["filter"][0]
    assert regex["group_select"] == -1


def test_response_trajectory_parser_uses_last_explicit_final_answer():
    parser = RegexFilter(
        regex_pattern=r"\b[Ff]inal answer\s*:?\s*([A-Da-d])\b",
        group_select=-1,
    )
    responses = [
        ["A is initially plausible. Final answer: B"],
        ["The model omitted punctuation. Final answer c"],
        ["Final answer: A\nCorrection after checking the equation. Final answer: d"],
        ["The explanation never emits the required marker."],
    ]
    assert parser.apply(responses, docs=[{}, {}, {}, {}]) == [["B"], ["c"], ["d"], ["[invalid]"]]


def test_response_trajectory_resolved_config_disables_implicit_paragraph_stop():
    from lmms_eval.tasks import TaskManager, get_task_dict

    task = get_task_dict(
        ["repdyn_response_mmlu_abstract_algebra"],
        TaskManager("ERROR"),
        "simple",
    )["repdyn_response_mmlu_abstract_algebra"]
    assert task.config.generation_kwargs["until"] == []
    assert task.config.generation_kwargs["max_new_tokens"] == 256
