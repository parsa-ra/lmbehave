import csv
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "representation_dynamics_text_expansion_corrected_v1"
PLOT_DATA = ROOT / "paper" / "figures" / "paper_plot_data"

MODEL_NAMES = {
    "olmo3-7b-instruct": "OLMo 3 7B",
    "apertus15-8b": "Apertus 1.5 8B",
    "qwen3_5-9b": "Qwen 3.5 9B",
    "qwen3_5-27b": "Qwen 3.5 27B",
    "gemma4-31b": "Gemma 4 31B",
}
STREAM_NAMES = {
    "attention_delta": "Mixer",
    "mlp_delta": "MLP",
    "residual_state": "Residual",
}


def rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class CurrentPaperPlotDataTests(unittest.TestCase):
    def test_corrected_cross_panel_has_no_unadjusted_probe_rejections(self):
        selected = [
            row for row in rows(RESULTS / "probe_persistence_summary.csv")
            if row["panel"] == "text_cross_benchmark_stress_test"
        ]
        self.assertEqual(len(selected), 5)
        for row in selected:
            self.assertEqual(row["linear_base_significant"], "0")
            self.assertEqual(row["rbf_base_significant"], "0")

    def test_profile_correlations_match_hash_bound_summary(self):
        expected = {
            (MODEL_NAMES[row["model_key"]], STREAM_NAMES[row["stream"]]): float(row["fdr_profile_pearson_r"])
            for row in rows(RESULTS / "fdr_profile_replication.csv")
        }
        observed = {
            (row["model"], row["stream"]): float(row["correlation"])
            for row in rows(PLOT_DATA / "profile_correlation.csv")
        }
        self.assertEqual(set(observed), set(expected))
        for key in expected:
            self.assertAlmostEqual(observed[key], expected[key], places=10)

    def test_localization_matches_hash_bound_summary(self):
        panel_names = {
            "matched_text_mmlu_pro_control": "Matched control",
            "text_cross_benchmark_stress_test": "Cross-benchmark",
        }
        expected = {}
        for row in rows(RESULTS / "probe_persistence_summary.csv"):
            key = (MODEL_NAMES[row["model_key"]], panel_names[row["panel"]])
            expected[key] = (
                int(row["selected_layer"]),
                STREAM_NAMES[row["selected_stream"]],
            )
        layers = {"OLMo 3 7B": 32, "Apertus 1.5 8B": 32, "Qwen 3.5 9B": 32, "Qwen 3.5 27B": 64, "Gemma 4 31B": 60}
        observed = rows(PLOT_DATA / "localization.csv")
        self.assertEqual({(row["model"], row["panel"]) for row in observed}, set(expected))
        for row in observed:
            key = (row["model"], row["panel"])
            layer, stream = expected[key]
            self.assertEqual(row["stream"], stream)
            self.assertAlmostEqual(float(row["normalized_depth"]), layer / (layers[row["model"]] - 1), places=6)


if __name__ == "__main__":
    unittest.main()
