import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "apertus15_endpoint_anchor_v1"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Apertus15EndpointResultTests(unittest.TestCase):
    def setUp(self):
        self.report = json.loads((RESULTS / "report.json").read_text(encoding="utf-8"))

    def test_outputs_are_hash_bound_and_sample_free(self):
        for name, expected in self.report["outputs"].items():
            self.assertEqual(sha256(RESULTS / name), expected)
        self.assertFalse(
            self.report["raw_activations_prompts_responses_or_per_sample_rows_in_output"]
        )
        forbidden = {"sample_id", "prompt", "response", "target", "vector"}
        for path in RESULTS.glob("*.csv"):
            with path.open(newline="", encoding="utf-8") as handle:
                fields = set(csv.DictReader(handle).fieldnames or ())
            self.assertTrue(forbidden.isdisjoint(fields), path)

    def test_checkpoint_summary_matches_declared_endpoint_aggregate(self):
        with (RESULTS / "longitudinal_checkpoint_summary.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        self.assertEqual({row["stream"] for row in rows}, {
            "attention_delta", "mlp_delta", "residual_state"
        })
        for row in rows:
            self.assertEqual(row["checkpoint"], "apertus15-final")
            self.assertEqual(int(row["replicates"]), 3)
            self.assertAlmostEqual(
                float(row["knowledge_accuracy_mean"]),
                self.report["aggregate_results"]["knowledge_accuracy_mean"],
            )
            self.assertAlmostEqual(
                float(row["reasoning_accuracy_mean"]),
                self.report["aggregate_results"]["reasoning_accuracy_mean"],
            )
            declared = self.report["aggregate_results"]["depth_mean_fdr"][row["stream"]]
            self.assertAlmostEqual(float(row["depth_mean_fdr_mean"]), declared["mean"])
            self.assertAlmostEqual(float(row["depth_mean_fdr_std"]), declared["std"])

    def test_repeatability_counts_bind_the_claim(self):
        with (RESULTS / "probe_repeatability_summary.csv").open(
            newline="", encoding="utf-8"
        ) as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 3)
        self.assertEqual(
            sum(int(row["significant_all_3_blocks"]) for row in rows),
            self.report["aggregate_results"][
                "linear_probe_f1_q_below_0_05_rows_in_all_blocks"
            ],
        )


if __name__ == "__main__":
    unittest.main()
