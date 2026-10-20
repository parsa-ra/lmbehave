import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_text_protocol_samples.py"
SPEC = importlib.util.spec_from_file_location("audit_text_protocol_samples", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TextProtocolAuditTests(unittest.TestCase):
    @staticmethod
    def _write_sample(root: Path, task: str, prompt: str, target: str = "A") -> None:
        path = root / f"20260810_samples_{task}.jsonl"
        path.write_text(
            json.dumps({"doc_id": 0, "target": target, "input": prompt}) + "\n",
            encoding="utf-8",
        )

    def test_valid_musr_prompt_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_sample(
                root,
                "repdyn_musr_murder_mysteries",
                "Answer the following multiple-choice question. Return only one letter: A, B.\n"
                "Story\nQuestion\nA. Madison\nB. Christine\nAnswer:",
            )
            report = MODULE.audit(root)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["valid_option_counts"]["repdyn_musr_murder_mysteries"], {"2": 1})

    def test_character_split_musr_prompt_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_sample(
                root,
                "repdyn_musr_murder_mysteries",
                "Answer the following multiple-choice question. Return only one letter: A, B, C.\n"
                "Story\nQuestion\nA. [\nB. '\nC. M\nAnswer:",
            )
            report = MODULE.audit(root)
            self.assertEqual(report["status"], "FAIL")
            self.assertEqual(report["failure_count"], 1)
            self.assertIn("outside [2, 2]", report["failures"][0]["errors"][0])


if __name__ == "__main__":
    unittest.main()
