import hashlib
import json

from scripts.summarize_main_track_lexical_control import load_report, telemetry_summary


def test_lexical_report_requires_inductive_complete_support(tmp_path):
    distribution = tmp_path / "distribution.json.gz"
    distribution.write_bytes(b"distribution")
    digest = hashlib.sha256(distribution.read_bytes()).hexdigest()
    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps(
            {
                "status": "PASS",
                "assignments": 12870,
                "probe_refits": 6435,
                "selection_recomputed_for_every_assignment": True,
                "configuration": {
                    "implementation": {"git_head": "abc", "git_dirty": False},
                    "nuisance_control": {
                        "lexical": {
                            "requested_components": 64,
                            "vectorizer_fit_scope": "training_tasks_per_fold",
                            "svd_fit_scope": "training_tasks_per_fold",
                        }
                    },
                },
                "macro_f1": {"complement_symmetry": {"verified": True}},
                "lexical_nuisance_fit": {
                    "fit_is_inductive": True,
                    "unique_task_folds": 229,
                    "actual_component_counts": [64],
                },
                "outputs": {
                    "distribution": str(distribution),
                    "distribution_sha256": digest,
                },
            }
        ),
        encoding="utf-8",
    )
    report = load_report(report_path, components=64, expected_head="abc")
    assert report["assignments"] == 12870


def test_lexical_telemetry_summary(tmp_path):
    path = tmp_path / "telemetry.csv"
    path.write_text(
        "2026-08-16, 10, 1000, 200, 900\n"
        "2026-08-16, 90, 3000, 500, 900\n",
        encoding="utf-8",
    )
    summary = telemetry_summary(path)
    assert summary["samples"] == 2
    assert summary["mean_gpu_utilization_percent"] == 50
    assert summary["peak_memory_mib"] == 3000
    assert summary["peak_power_w"] == 500
