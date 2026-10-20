import numpy as np

from lmms_eval.representations.prompt_baseline import run_prompt_baseline


def test_prompt_baseline_enumerates_complete_task_support():
    tasks = ["knowledge_a", "knowledge_b", "reasoning_a", "reasoning_b"]
    prompts = []
    groups = []
    for task in tasks:
        token = "archive fact" if task.startswith("knowledge") else "derive equation"
        for index in range(6):
            prompts.append(f"{token} shared prompt {index}")
            groups.append(task)

    report = run_prompt_baseline(
        prompts,
        np.asarray(groups, dtype=object),
        knowledge_tasks={"knowledge_a", "knowledge_b"},
        kind="word_tfidf",
        requested_splits=2,
        seed=0,
        jobs=1,
    )

    assert report["assignments"] == 6
    assert report["probe_refits"] == 3
    assert report["macro_f1"]["observed"] == 1.0
    assert report["score_complement_symmetry_method"].startswith("imposed_by_reusing")
