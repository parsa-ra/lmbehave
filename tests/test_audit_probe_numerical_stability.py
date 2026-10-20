import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_probe_numerical_stability.py"
SPEC = importlib.util.spec_from_file_location("audit_probe_numerical_stability", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ProbeNumericalStabilityTests(unittest.TestCase):
    def _write(self, path: Path, *, linear: float, rbf: float) -> None:
        row = {
            "model": "m",
            "phase": "prefill",
            "layer": "0",
            "stream": "residual_state",
            "module": "layer.0",
            "probe_f1_q": str(linear),
            "rbf_probe_accuracy": "0.7",
            "rbf_probe_f1": "0.7",
            "rbf_probe_accuracy_p": "0.04",
            "rbf_probe_f1_p": "0.04",
            "rbf_probe_accuracy_q": str(rbf),
            "rbf_probe_f1_q": str(rbf),
            "silhouette_cosine": "0.1",
        }
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)

    def test_rbf_numeric_change_is_separate_when_decision_is_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.csv"
            reference = root / "reference.csv"
            self._write(candidate, linear=0.2, rbf=0.041)
            self._write(reference, linear=0.2, rbf=0.042)
            result = MODULE.audit_condition(candidate, reference, alpha=0.05)
            self.assertEqual(result["hard_mismatch_count"], 0)
            self.assertGreater(result["rbf_difference_count"], 0)
            self.assertEqual(result["decision_mismatches"], {})

    def test_non_rbf_and_threshold_changes_are_detected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            candidate = root / "candidate.csv"
            reference = root / "reference.csv"
            self._write(candidate, linear=0.3, rbf=0.06)
            self._write(reference, linear=0.2, rbf=0.04)
            result = MODULE.audit_condition(candidate, reference, alpha=0.05)
            self.assertEqual(result["hard_mismatch_count"], 1)
            self.assertEqual(result["decision_mismatches"]["rbf_probe_f1_q"], 1)


if __name__ == "__main__":
    unittest.main()
