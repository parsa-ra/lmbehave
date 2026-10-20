import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_full_manifest_freezes_common_support_and_calibrated_batches():
    manifest = json.loads(
        (ROOT / "experiments" / "representation_dynamics_v1.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["experiment_id"] == "representation-dynamics-v1-common-support"
    assert manifest["extraction"]["sample_count_per_task"] == 100
    assert manifest["models"]["gemma3-4b"]["batch_size"] == 100
    assert manifest["models"]["qwen3_5-9b"]["batch_size"] == 20

    for panel in manifest["panels"].values():
        assert len(panel["knowledge_tasks"]) == 5
        assert len(panel["reasoning_tasks"]) == 5
