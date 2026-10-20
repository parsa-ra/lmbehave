import csv
import gzip
import json
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from lmms_eval.representations.selected_site_probe import (
    _fit_fold_local_lexical_nuisance,
    run_selection_aware_probe,
    selection_plan,
)


class SelectionAwareProbeTest(unittest.TestCase):
    def test_lexical_nuisance_fit_does_not_learn_from_held_out_prompts(self):
        train_prompts = [
            "shared archive alpha",
            "shared archive beta",
            "shared derive gamma",
            "shared derive delta",
        ]
        first_train, first_test, components = _fit_fold_local_lexical_nuisance(
            train_prompts,
            ["shared archive first"],
            kind="char_tfidf",
            components=2,
            seed=0,
        )
        second_train, second_test, second_components = _fit_fold_local_lexical_nuisance(
            train_prompts,
            ["shared derive second"],
            kind="char_tfidf",
            components=2,
            seed=0,
        )
        np.testing.assert_array_equal(first_train, second_train)
        self.assertFalse(np.array_equal(first_test, second_test))
        self.assertEqual(components, second_components)

    def _permutation_rows(self):
        task_order = ["task_a", "task_b", "task_c", "task_d"]
        observed = "1100"
        assignment_ids = [observed]
        assignment_ids.extend(assignment for assignment in ("".join("1" if index in chosen else "0" for index in range(4)) for chosen in combinations(range(4), 2)) if assignment != observed)
        first = [5.0 if value in {"1100", "0011"} else 1.0 for value in assignment_ids]
        second = [5.0 if value in {"1010", "0101"} else 4.0 if value in {"1001", "0110"} else 1.0 for value in assignment_ids]
        common = {"phase": "prefill", "trajectory_position": "aggregate"}
        return [
            {
                **common,
                "model": "toy/model",
                "layer": 0,
                "stream": "attention_delta",
                "module": "layers.0.attention",
                "task_order": task_order,
                "assignment_ids": assignment_ids,
                "fdr": first,
            },
            {
                **common,
                "model": "toy/model",
                "layer": 1,
                "stream": "mlp_delta",
                "module": "layers.1.mlp",
                "task_order": task_order,
                "assignment_ids": assignment_ids,
                "fdr": second,
            },
        ]

    def test_selection_plan_repeats_site_selection_for_each_assignment(self):
        rows = self._permutation_rows()
        assignment_ids, task_order, selected, by_key = selection_plan(rows)
        self.assertEqual(task_order, ["task_a", "task_b", "task_c", "task_d"])
        self.assertEqual(len(assignment_ids), 6)
        self.assertEqual(selected.tolist().count(0), 2)
        self.assertEqual(selected.tolist().count(1), 4)
        self.assertEqual(sum(map(len, by_key.values())), 6)

    def test_end_to_end_exact_selected_site_probe(self):
        rng = np.random.default_rng(7)
        permutation_rows = self._permutation_rows()
        task_order = permutation_rows[0]["task_order"]
        records = []
        for task_index, task in enumerate(task_order):
            for sample_index in range(8):
                noise = rng.normal(scale=0.1, size=6)
                row0 = noise.copy()
                row0[0] += 2.0 if task_index < 2 else -2.0
                row1 = noise.copy()
                row1[1] += 2.0 if task_index in {0, 2} else -2.0
                for layer, stream, module, vector in (
                    (0, "attention_delta", "layers.0.attention", row0),
                    (1, "mlp_delta", "layers.1.mlp", row1),
                ):
                    records.append(
                        {
                            "phase": "prefill",
                            "layer_index": layer,
                            "stream": stream,
                            "module_name": module,
                            "task_name": task,
                            "sample_id": f"{task}:{sample_index}",
                            "vector": torch.tensor(vector, dtype=torch.float32),
                        }
                    )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "representations.pt"
            torch.save(
                {
                    "metadata": {
                        "pretrained": "toy/model",
                        "lmms_eval_git_head": "deadbeef",
                        "lmms_eval_git_dirty": False,
                        "transformers_version": "test",
                    },
                    "records": records,
                },
                artifact,
            )
            permutation_path = root / "permutation_nulls.json.gz"
            with gzip.open(permutation_path, "wt", encoding="utf-8") as handle:
                json.dump({"schema_version": 1, "rows": permutation_rows}, handle)
            from lmms_eval.representations import selected_site_probe

            original_probe = selected_site_probe._probe_predictions
            with patch(
                "lmms_eval.representations.selected_site_probe._probe_predictions",
                wraps=original_probe,
            ) as probe:
                report = run_selection_aware_probe(
                    [artifact],
                    permutation_path=permutation_path,
                    output_dir=root / "selected",
                    requested_splits=2,
                    pca_variance=0.95,
                    seed=0,
                    analysis_backend="sklearn",
                    checkpoint_every=2,
                )
                self.assertEqual(probe.call_count, 3)
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["assignments"], 6)
            self.assertEqual(report["probe_refits"], 3)
            self.assertEqual(report["selected_sites"], 2)
            self.assertTrue(report["selection_recomputed_for_every_assignment"])
            self.assertTrue(report["class_label_complements_share_one_probe_refit"])
            self.assertEqual(report["configuration"]["implementation"]["analysis_backend"], "sklearn")
            self.assertTrue(report["configuration"]["implementation"]["module_sha256"])
            self.assertTrue(report["accuracy"]["complement_symmetry"]["verified"])
            self.assertTrue(report["macro_f1"]["complement_symmetry"]["verified"])
            self.assertTrue((root / "selected" / "selected_site_probe_distribution.json.gz").is_file())
            resumed = run_selection_aware_probe(
                [artifact],
                permutation_path=permutation_path,
                output_dir=root / "selected",
                requested_splits=2,
                pca_variance=0.95,
                seed=0,
                analysis_backend="sklearn",
                checkpoint_every=2,
            )
            self.assertEqual(resumed["status"], "PASS")
            self.assertEqual(resumed["macro_f1"]["observed"], report["macro_f1"]["observed"])

            nuisance_path = root / "nuisance.csv"
            with nuisance_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("model", "task_name", "sample_id", "prompt_length"),
                )
                writer.writeheader()
                for task_index, task in enumerate(task_order):
                    for sample_index in range(8):
                        writer.writerow(
                            {
                                "model": "toy/model",
                                "task_name": task,
                                "sample_id": f"{task}:{sample_index}",
                                "prompt_length": task_index + sample_index / 10,
                            }
                        )
            adjusted = run_selection_aware_probe(
                [artifact],
                permutation_path=permutation_path,
                output_dir=root / "selected-adjusted",
                requested_splits=2,
                pca_variance=0.95,
                seed=0,
                analysis_backend="sklearn",
                checkpoint_every=2,
                nuisance_table=nuisance_path,
                nuisance_columns=("prompt_length",),
                nuisance_alpha=1.0,
            )
            self.assertEqual(adjusted["status"], "PASS")
            self.assertTrue(adjusted["configuration"]["nuisance_control"]["enabled"])
            self.assertEqual(
                adjusted["configuration"]["nuisance_control"]["columns"],
                ["prompt_length"],
            )

            prompt_root = root / "prompt-root"
            for task_index, task in enumerate(task_order):
                log = (
                    prompt_root
                    / "tasks"
                    / task
                    / "evaluation"
                    / "toy"
                    / f"20260816_samples_{task}.jsonl"
                )
                log.parent.mkdir(parents=True, exist_ok=True)
                token = "archive fact" if task_index < 2 else "derive equation"
                log.write_text(
                    "".join(
                        json.dumps(
                            {
                                "doc_id": sample_index,
                                "input": f"{token} shared prompt {sample_index}",
                            }
                        )
                        + "\n"
                        for sample_index in range(8)
                    ),
                    encoding="utf-8",
                )
            lexical = run_selection_aware_probe(
                [artifact],
                permutation_path=permutation_path,
                output_dir=root / "selected-lexical",
                requested_splits=2,
                pca_variance=0.95,
                seed=0,
                analysis_backend="sklearn",
                checkpoint_every=2,
                lexical_nuisance_sample_root=prompt_root,
                lexical_nuisance_components=4,
            )
            lexical_config = lexical["configuration"]["nuisance_control"]["lexical"]
            self.assertEqual(lexical["status"], "PASS")
            self.assertTrue(lexical_config["enabled"])
            self.assertEqual(lexical_config["requested_components"], 4)
            self.assertEqual(lexical_config["vectorizer_fit_scope"], "training_tasks_per_fold")
            self.assertTrue(lexical["lexical_nuisance_fit"]["fit_is_inductive"])
            self.assertGreater(lexical["lexical_nuisance_fit"]["unique_task_folds"], 0)


if __name__ == "__main__":
    unittest.main()
