import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts.build_multimodal_nuisance_table import _task_design
from scripts.summarize_multimodal_side_analysis import exact_upper_tail_p


ROOT = Path(__file__).resolve().parents[1]


def test_multimodal_panel_records_the_modality_imbalance():
    manifest = json.loads(
        (ROOT / "experiments" / "representation_dynamics_v1.json").read_text(encoding="utf-8")
    )
    design = _task_design(manifest, "cross_benchmark_discovery")
    knowledge = [row for row in design.values() if row["label"] == "knowledge"]
    reasoning = [row for row in design.values() if row["label"] == "reasoning"]
    assert len(knowledge) == len(reasoning) == 5
    assert sum(row["modality"] != "text" for row in knowledge) == 3
    assert sum(row["modality"] != "text" for row in reasoning) == 5


def test_exact_upper_tail_includes_ties_and_observed_assignment():
    assert exact_upper_tail_p([0.8, 0.8, 0.5, 0.2], 0.8) == pytest.approx(0.5)


def test_committed_multimodal_summary_matches_claim_boundary():
    result_root = ROOT / "results" / "representation_dynamics_multimodal_side_v1"
    report = json.loads((result_root / "report.json").read_text(encoding="utf-8"))
    assert report["role"] == "appendix_only_side_analysis"
    assert report["assignments"] == 252
    assert report["complement_equivalence_classes"] == 126
    assert report["minimum_attainable_p"] == pytest.approx(1 / 126)

    with (result_root / "model_summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["model"] for row in rows] == ["qwen3_5-9b", "qwen3_5-27b", "gemma4-31b"]
    assert all(float(row["fisher_maximum_p"]) >= 0.01 for row in rows)
    assert all(float(row["raw_f1_p"]) >= 0.01 for row in rows)
    assert min(float(row["lexical_64_f1_p"]) for row in rows) == pytest.approx(8 / 126)
    assert min(float(row["combined_64_f1_p"]) for row in rows) > 0.9

    with (result_root / "selected_site_descriptive_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        descriptive = list(csv.DictReader(handle))
    assert len(descriptive) == 3
    assert all(row["cka"] == "" and int(row["cka_pairs"]) == 0 for row in descriptive)
    assert [float(row["silhouette_cosine"]) for row in descriptive] == pytest.approx(
        [0.2819548058338387, 0.25643824970634765, 0.43360359753436034]
    )

    for filename, expected in report["output_sha256"].items():
        observed = hashlib.sha256((result_root / filename).read_bytes()).hexdigest()
        assert observed == expected
