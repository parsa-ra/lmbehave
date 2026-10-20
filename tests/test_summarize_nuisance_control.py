import importlib.util
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "summarize_nuisance_control.py"
SPEC = importlib.util.spec_from_file_location("summarize_nuisance_control", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class NuisanceSummaryHelperTests(unittest.TestCase):
    def setUp(self):
        self.manifest = {"models": {"one": {}, "two": {}, "three": {}}}

    def test_selection_defaults_to_manifest_order(self):
        self.assertEqual(MODULE.select_models(self.manifest, None), ["one", "two", "three"])

    def test_selection_accepts_subset_and_rejects_invalid_requests(self):
        self.assertEqual(MODULE.select_models(self.manifest, ["three", "one"]), ["three", "one"])
        with self.assertRaisesRegex(ValueError, "Unknown model"):
            MODULE.select_models(self.manifest, ["missing"])
        with self.assertRaisesRegex(ValueError, "Duplicate model"):
            MODULE.select_models(self.manifest, ["one", "one"])

    def test_profile_replication_uses_same_stream_and_depth(self):
        def panel_rows(values, maximum_stream="mlp_delta"):
            rows = []
            for stream in ("attention_delta", "mlp_delta", "residual_state"):
                for layer, value in enumerate(values):
                    rows.append(
                        {
                            "layer": str(layer),
                            "stream": stream,
                            "fdr": str(value + (1.0 if stream == maximum_stream else 0.0)),
                        }
                    )
            return rows

        conditions = {
            ("one", "left"): panel_rows([1.0, 2.0, 3.0]),
            ("one", "right"): panel_rows([2.0, 4.0, 6.0]),
        }
        rows = MODULE.profile_replication_rows(conditions, ["left", "right"], ["one"])
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(abs(row["fdr_profile_pearson_r"] - 1.0) < 1e-12 for row in rows))
        self.assertTrue(all(row["overall_maximum_site_matches"] for row in rows))

    def test_profile_replication_requires_matching_unique_layers(self):
        conditions = {
            ("one", "left"): [
                {"layer": "0", "stream": stream, "fdr": "1"}
                for stream in ("attention_delta", "mlp_delta", "residual_state")
            ],
            ("one", "right"): [
                {"layer": "1", "stream": stream, "fdr": "1"}
                for stream in ("attention_delta", "mlp_delta", "residual_state")
            ],
        }
        with self.assertRaisesRegex(ValueError, "Incompatible FDR profiles"):
            MODULE.profile_replication_rows(conditions, ["left", "right"], ["one"])


if __name__ == "__main__":
    unittest.main()
