import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "check_apertus15_release",
    ROOT / "scripts" / "check_apertus15_release.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TestApertus15ReleaseCheck(unittest.TestCase):
    def setUp(self):
        self.manifest = json.loads(
            (ROOT / "experiments" / "apertus15_longitudinal_v1.json").read_text(
                encoding="utf-8"
            )
        )

    def test_current_single_main_ref_is_not_longitudinal(self):
        revision = self.manifest["released_endpoint"]["revision"]
        refs = {"branches": [{"name": "main", "targetCommit": revision}], "tags": []}
        related = [{"id": "swiss-ai/Apertus-v1.5-8B", "sha": revision}]
        report = MODULE.evaluate_release(self.manifest, refs, related)
        self.assertTrue(report["main_revision_matches_manifest"])
        self.assertEqual(report["candidate_count"], 0)
        self.assertFalse(report["longitudinal_ready"])

    def test_candidate_refs_require_immutable_revisions(self):
        revision = self.manifest["released_endpoint"]["revision"]
        refs = {
            "branches": [{"name": "main", "targetCommit": revision}],
            "tags": [
                {"name": "stage1-step1000", "targetCommit": "1" * 40},
                {"name": "stage1-step2000", "targetCommit": "2" * 40},
                {"name": "stage1-step3000", "targetCommit": "not-immutable"},
            ],
        }
        report = MODULE.evaluate_release(self.manifest, refs, [])
        self.assertEqual(report["candidate_count"], 2)
        self.assertTrue(report["longitudinal_ready"])
        self.assertEqual(report["status"], "READY_FOR_MANUAL_TRAJECTORY_AUDIT")


if __name__ == "__main__":
    unittest.main()
