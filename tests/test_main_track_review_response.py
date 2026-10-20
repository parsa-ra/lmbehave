import csv
import hashlib
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _rows(path: Path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_review_response_summary_is_hash_bound_and_matches_claims():
    result_root = ROOT / "results" / "representation_dynamics_main_track_review_response_v1"
    report = json.loads((result_root / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["models"] == 5
    assert report["analyses"]["pca95_representation_vs_prompt64"]["effective_null_pairs"] == 6434
    assert report["analyses"]["representation_width_sensitivity"]["status"] in {
        "PASS",
        "PENDING_COMPLETE_REPORTS",
    }
    assert report["analyses"]["matched_delta_reselection_calibration"]["status"] in {
        "PASS",
        "PENDING_COMPLETE_REPORTS",
    }

    pca95 = _rows(result_root / "pca95_representation_vs_prompt64.csv")
    assert len(pca95) == 5
    assert [float(row["observed"]) for row in pca95] == pytest.approx(
        [0.0251417469, 0.0488509166, 0.0499808190, 0.0409513858, 0.0497206542]
    )
    assert min(float(row["exact_oriented_p"]) for row in pca95) > 0.25

    split = _rows(result_root / "inference_unit_contrast.csv")
    assert len(split) == 1
    assert float(split[0]["example_level_macro_f1"]) == pytest.approx(0.8022401312)
    assert float(split[0]["example_level_sample_permutation_p"]) == pytest.approx(0.0001)
    assert float(split[0]["task_heldout_macro_f1"]) == pytest.approx(0.5737440058)
    assert float(split[0]["task_label_exact_p"]) == pytest.approx(0.2276612277)

    multiplicity = _rows(result_root / "across_model_multiplicity.csv")
    assert len(multiplicity) == 15
    by_condition = {}
    for row in multiplicity:
        by_condition.setdefault(row["condition"], []).append(row)
    assert all(row["reject_0_01"] == "True" for row in by_condition["raw_selected_site_probe"])
    assert all(
        row["reject_0_05"] == "False"
        for condition in (
            "matched_representation_minus_prompt",
            "lexical64_adjusted_probe",
        )
        for row in by_condition[condition]
    )

    for filename, expected in report["outputs"].items():
        observed = hashlib.sha256((result_root / filename).read_bytes()).hexdigest()
        assert observed == expected
