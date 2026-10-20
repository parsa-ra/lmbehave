import csv
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_whitened_selector_summary_is_complete_and_hash_bound():
    result_root = ROOT / "results" / "representation_dynamics_selector_sensitivity_v1"
    report = json.loads((result_root / "report.json").read_text(encoding="utf-8"))
    assert report["status"] == "PASS"
    assert report["models"] == 5
    assert report["ridge_alphas"] == [0.01, 0.1, 1.0]
    assert report["all_whitened_sites_differ_from_trace_site"] is True
    assert report["all_grid_maxima_reject_after_holm_at_0.01"] is True
    with (result_root / "selector_summary.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 15
    assert all(row["agrees_with_trace_site"] == "False" for row in rows)
    assert max(float(row["exact_p"]) for row in rows) < 0.01
    assert max(float(row["holm_p_across_grid"]) for row in rows) < 0.01
    for filename, expected in report["outputs"].items():
        observed = hashlib.sha256((result_root / filename).read_bytes()).hexdigest()
        assert observed == expected
