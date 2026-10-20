import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_representation_experiment.py"
SPEC = importlib.util.spec_from_file_location("run_representation_experiment", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CompletedTaskOutputTests(unittest.TestCase):
    def test_empty_output_is_not_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(MODULE.completed_task_output(Path(directory)))

    def test_requires_both_artifact_and_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "representations" / "run" / "representations.pt"
            artifact.parent.mkdir(parents=True)
            artifact.touch()
            with self.assertRaisesRegex(RuntimeError, "Incomplete or ambiguous"):
                MODULE.completed_task_output(root)

    def test_accepts_exactly_one_complete_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "representations" / "run" / "representations.pt"
            result = root / "evaluation" / "model" / "timestamp_results.json"
            artifact.parent.mkdir(parents=True)
            result.parent.mkdir(parents=True)
            artifact.touch()
            result.touch()
            self.assertEqual(MODULE.completed_task_output(root), (artifact, result))


class FrozenSampleSelectionTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {
            "require_frozen_sample_ids": True,
            "panels": {
                "first": {
                    "knowledge_tasks": [{"task": "task_a"}],
                    "reasoning_tasks": [{"task": "task_b"}],
                },
                "second": {
                    "knowledge_tasks": [{"task": "task_b"}],
                    "reasoning_tasks": [{"task": "task_c"}],
                },
            },
        }

    def test_selected_task_names_are_stable_and_unique(self):
        self.assertEqual(
            MODULE.selected_task_names(self.manifest, ["first", "second"]),
            ["task_a", "task_b", "task_c"],
        )

    def test_required_selection_must_exist(self):
        candidate = {
            "experiment_id": "experiment",
            "experiment_manifest_sha256": "manifest",
            "effective_sample_count_per_task": 100,
            "root_git": {"head": "root"},
            "lmms_eval_git": {"head": "child"},
        }
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "freeze-samples"):
                MODULE.bind_sample_selection(candidate, Path(directory), self.manifest)

    def test_matching_selection_is_hash_bound(self):
        candidate = {
            "experiment_id": "experiment",
            "experiment_manifest_sha256": "manifest",
            "effective_sample_count_per_task": 100,
            "root_git": {"head": "root"},
            "lmms_eval_git": {"head": "child"},
        }
        sample_ids = [f"task:test:{index}" for index in range(100)]
        task_selection = {
            "sample_ids": sample_ids,
            "sample_ids_sha256": MODULE.hashlib.sha256("\n".join(sample_ids).encode()).hexdigest(),
            "selected_docs_fingerprint": "documents",
        }
        selection = {
            **candidate,
            "panels": ["first", "second"],
            "tasks": {name: task_selection for name in ("task_a", "task_b", "task_c")},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / MODULE.SAMPLE_SELECTION_FILENAME
            path.write_text(json.dumps(selection), encoding="utf-8")
            MODULE.bind_sample_selection(candidate, Path(directory), self.manifest)
            self.assertEqual(candidate["sample_selection"]["sha256"], MODULE.sha256_file(path))
            self.assertEqual(
                MODULE.load_bound_sample_selection(Path(directory), candidate),
                selection,
            )
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                MODULE.load_bound_sample_selection(Path(directory), candidate)

    def test_legacy_manifest_does_not_require_selection(self):
        candidate = {}
        MODULE.bind_sample_selection(candidate, Path("/does/not/exist"), {"panels": {}})
        self.assertNotIn("sample_selection", candidate)

    def test_sample_identity_digest_ignores_traversal_order(self):
        ordered = ["task:test:0", "task:test:1", "task:test:10", "task:test:2"]
        shuffled = ["task:test:10", "task:test:2", "task:test:0", "task:test:1"]
        self.assertEqual(
            MODULE.sample_id_set_sha256(ordered),
            MODULE.sample_id_set_sha256(shuffled),
        )


class ModelRevisionCheckTests(unittest.TestCase):
    @staticmethod
    def _manifest():
        return {
            "models": {
                "current": {
                    "pretrained": "organization/current-model",
                    "revision": "a" * 40,
                }
            }
        }

    def _run_with_info(self, info):
        api = types.SimpleNamespace(model_info=lambda *args, **kwargs: info)
        module = types.SimpleNamespace(HfApi=lambda token=None: api)
        with patch.dict(sys.modules, {"huggingface_hub": module}):
            MODULE.check_models(self._manifest(), ["current"])

    def test_model_check_requires_exact_revision_and_weights(self):
        info = types.SimpleNamespace(
            sha="a" * 40,
            siblings=[types.SimpleNamespace(rfilename="model.safetensors")],
        )
        self._run_with_info(info)

    def test_model_check_rejects_revision_mismatch(self):
        info = types.SimpleNamespace(
            sha="b" * 40,
            siblings=[types.SimpleNamespace(rfilename="model.safetensors")],
        )
        with self.assertRaisesRegex(RuntimeError, "unexpected commit"):
            self._run_with_info(info)

    def test_model_check_rejects_missing_weights(self):
        info = types.SimpleNamespace(
            sha="a" * 40,
            siblings=[types.SimpleNamespace(rfilename="config.json")],
        )
        with self.assertRaisesRegex(RuntimeError, "no model weight files"):
            self._run_with_info(info)


class AnalysisSensitivitySettingsTests(unittest.TestCase):
    def test_nonlinear_override_preserves_locked_settings(self):
        locked = {
            "cv_folds": 5,
            "linear_task_label_permutations": 251,
            "nonlinear_task_label_permutations": 0,
            "seed": 0,
        }
        effective = MODULE.effective_analysis_settings(locked, 251)
        self.assertEqual(effective["nonlinear_task_label_permutations"], 251)
        self.assertEqual(locked["nonlinear_task_label_permutations"], 0)
        self.assertEqual(effective["linear_task_label_permutations"], 251)

    def test_nonlinear_override_rejects_negative_counts(self):
        with self.assertRaisesRegex(ValueError, "non-negative"):
            MODULE.effective_analysis_settings({"nonlinear_task_label_permutations": 0}, -1)

if __name__ == "__main__":
    unittest.main()
