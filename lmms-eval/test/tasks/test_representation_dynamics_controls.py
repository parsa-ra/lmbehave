import os
import subprocess
import sys

from lmms_eval.tasks.representation_dynamics_controls import utils


def test_medqa_invalid_generation_is_scored_incorrect_not_random():
    doc = {"answer_idx": "C", "options": {letter: letter for letter in "ABCDE"}}
    assert utils.medqa_process_results(doc, ["No parseable answer"])["accuracy"] == 0.0


def test_mmmu_standard_utils_import_without_openai_key():
    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    result = subprocess.run(
        [sys.executable, "-c", "import lmms_eval.tasks.mmmu.utils"],
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr
