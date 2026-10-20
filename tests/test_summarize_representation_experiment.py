import importlib.util
import math
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_representation_experiment.py"
SPEC = importlib.util.spec_from_file_location("summarize_representation_experiment", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SummaryHelperTests(unittest.TestCase):
    def test_maximum_row_uses_requested_metric(self):
        rows = [{"fdr": "0.2", "layer": "1"}, {"fdr": "0.8", "layer": "3"}]
        self.assertEqual(MODULE.maximum_row(rows)["layer"], "3")

    def test_pearson_correlation(self):
        self.assertAlmostEqual(MODULE.pearson_correlation([1, 2, 3], [2, 4, 6]), 1.0)
        self.assertAlmostEqual(MODULE.pearson_correlation([1, 2, 3], [6, 4, 2]), -1.0)
        self.assertTrue(math.isnan(MODULE.pearson_correlation([1], [1])))

    def test_accuracy_value_prefers_declared_accuracy(self):
        name, value = MODULE.accuracy_value(
            {"accuracy,none": 0.7, "accuracy_stderr,none": 0.1, "other": 0.3}
        )
        self.assertEqual(name, "accuracy,none")
        self.assertEqual(value, 0.7)

    def test_model_selection_defaults_to_manifest_order(self):
        available = ["one", "two", "three"]
        self.assertEqual(MODULE.resolve_model_selection(available, None), available)

    def test_model_selection_accepts_explicit_subset(self):
        self.assertEqual(
            MODULE.resolve_model_selection(["one", "two", "three"], ["three", "one"]),
            ["three", "one"],
        )

    def test_model_selection_rejects_unknown_or_duplicate_models(self):
        with self.assertRaisesRegex(RuntimeError, "Unknown model"):
            MODULE.resolve_model_selection(["one", "two"], ["three"])
        with self.assertRaisesRegex(RuntimeError, "Duplicate model"):
            MODULE.resolve_model_selection(["one", "two"], ["one", "one"])

    def test_nonlinear_join_allows_only_inference_and_silhouette_changes(self):
        baseline = {
            "model": "model",
            "phase": "prefill",
            "layer": "1",
            "stream": "residual_state",
            "module": "model.layers.1",
            "fdr": "0.5",
            "rbf_probe_f1": "0.7",
            "rbf_probe_accuracy_p": "nan",
            "rbf_probe_f1_p": "nan",
            "rbf_probe_accuracy_q": "nan",
            "rbf_probe_f1_q": "nan",
            "silhouette_cosine": "0.2",
        }
        sensitivity = {
            **baseline,
            "rbf_probe_accuracy_p": "0.01",
            "rbf_probe_f1_p": "0.02",
            "rbf_probe_accuracy_q": "0.03",
            "rbf_probe_f1_q": "0.04",
            "silhouette_cosine": "0.19",
        }
        joined, diagnostics = MODULE.join_nonlinear_inference([baseline], [sensitivity])
        self.assertEqual(joined[MODULE.row_key(baseline)]["rbf_probe_f1_q"], "0.04")
        self.assertEqual(diagnostics["silhouette_difference_rows"], 1)
        self.assertAlmostEqual(diagnostics["silhouette_max_absolute_difference"], 0.01)

    def test_nonlinear_join_rejects_changed_observed_metric(self):
        baseline = {
            "model": "model",
            "phase": "prefill",
            "layer": "1",
            "stream": "mlp_delta",
            "module": "model.layers.1.mlp",
            "fdr": "0.5",
            "rbf_probe_accuracy_p": "nan",
            "rbf_probe_f1_p": "nan",
            "rbf_probe_accuracy_q": "nan",
            "rbf_probe_f1_q": "nan",
            "silhouette_cosine": "0.2",
        }
        sensitivity = {**baseline, "fdr": "0.6"}
        with self.assertRaisesRegex(RuntimeError, "changed fdr"):
            MODULE.join_nonlinear_inference([baseline], [sensitivity])


if __name__ == "__main__":
    unittest.main()
