import argparse
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from lmms_eval import __main__ as main_module


class TestCliParseArgs(unittest.TestCase):
    def test_git_metadata_is_resolved_from_lmms_eval_source(self):
        expected_cwd = Path(main_module.utils.__file__).resolve().parent
        with patch.object(main_module.utils.subprocess, "check_output", return_value=b"abc123\n") as check_output:
            self.assertEqual(main_module.utils.get_git_commit_hash(), "abc123")
        self.assertEqual(check_output.call_args.kwargs["cwd"], expected_cwd)
        self.assertEqual(check_output.call_args.kwargs["stderr"], main_module.utils.subprocess.DEVNULL)

        with patch.object(main_module.utils.subprocess, "check_output", return_value=b"research\n") as check_output:
            self.assertEqual(main_module.utils.get_git_branch_name(), "research")
        self.assertEqual(check_output.call_args.kwargs["cwd"], expected_cwd)
        self.assertEqual(check_output.call_args.kwargs["stderr"], main_module.utils.subprocess.DEVNULL)

    def test_evaluation_tracker_args_keep_environment_token_runtime_only(self):
        fake_hf_token = "hf_" + ("a" * 32)
        args = argparse.Namespace(
            hf_hub_log_args="repo_name=private-results",
            output_path="./results",
        )

        with patch.dict(os.environ, {"HF_TOKEN": fake_hf_token}, clear=False):
            tracker_args = main_module._build_evaluation_tracker_args(args)

        self.assertEqual(args.hf_hub_log_args, "repo_name=private-results")
        self.assertEqual(tracker_args["output_path"], "./results")
        self.assertEqual(tracker_args["token"], fake_hf_token)

    def test_resolved_cli_args_redact_credentials_without_mutating_inputs(self):
        raw_hub_args = "repo_name=private-results,token=" + "hf_" + ("a" * 32)
        args = argparse.Namespace(
            hf_hub_log_args=raw_hub_args,
            api_key="sk-" + ("b" * 32),
            model_args={"pretrained": "private/model", "token": "model-secret"},
            max_tokens=128,
            nested={"password": "do-not-store", "label": "safe"},
        )

        resolved = main_module.evaluator.resolved_cli_args_for_logging(args)

        self.assertEqual(args.hf_hub_log_args, raw_hub_args)
        self.assertEqual(resolved["hf_hub_log_args"], "repo_name=private-results,token=<redacted>")
        self.assertEqual(resolved["api_key"], "<redacted>")
        self.assertEqual(resolved["model_args"], {"pretrained": "private/model", "token": "<redacted>"})
        self.assertEqual(resolved["max_tokens"], 128)
        self.assertEqual(resolved["nested"], {"password": "<redacted>", "label": "safe"})

    def test_max_tokens_arg_is_parsed(self):
        argv = [
            "lmms-eval",
            "--model",
            "openai",
            "--tasks",
            "mme",
            "--max_tokens",
            "12345",
        ]
        with patch.object(sys, "argv", argv):
            _, args = main_module.parse_eval_args()
        self.assertEqual(args.max_tokens, 12345)

    def test_max_tokens_defaults_to_none(self):
        argv = [
            "lmms-eval",
            "--model",
            "openai",
            "--tasks",
            "mme",
        ]
        with patch.object(sys, "argv", argv):
            _, args = main_module.parse_eval_args()
        self.assertIsNone(args.max_tokens)

    def test_pruning_tracker_flags_are_parsed(self):
        argv = [
            "lmms-eval",
            "--model",
            "openai",
            "--tasks",
            "mme",
            "--pruning_tracker",
            "--pruning_tracker_output",
            "./logs/pruning",
            "--pruning_tracker_stride",
            "4",
            "--pruning_tracker_max_histogram_samples",
            "2048",
            "--no-pruning_tracker_force_eager_attention",
            "--pruning_tracker_force_output_attentions",
        ]
        with patch.object(sys, "argv", argv):
            _, args = main_module.parse_eval_args()
        self.assertTrue(args.pruning_tracker)
        self.assertEqual(args.pruning_tracker_output, "./logs/pruning")
        self.assertEqual(args.pruning_tracker_stride, 4)
        self.assertEqual(args.pruning_tracker_max_histogram_samples, 2048)
        self.assertFalse(args.pruning_tracker_force_eager_attention)
        self.assertTrue(args.pruning_tracker_force_output_attentions)

    def test_pruning_tracker_attention_defaults(self):
        argv = [
            "lmms-eval",
            "--model",
            "openai",
            "--tasks",
            "mme",
            "--pruning_tracker",
        ]
        with patch.object(sys, "argv", argv):
            _, args = main_module.parse_eval_args()
        self.assertTrue(args.pruning_tracker_force_eager_attention)
        self.assertFalse(args.pruning_tracker_force_output_attentions)

    def test_representation_dynamics_flags_are_parsed(self):
        argv = [
            "lmms-eval",
            "--model",
            "openai",
            "--tasks",
            "mme",
            "--representation_dynamics",
            "--representation_dynamics_output",
            "./logs/representations",
            "--representation_dynamics_pooling",
            "mean_valid",
            "--representation_dynamics_phases",
            "both",
            "--representation_dynamics_decode_aggregation",
            "all_updates",
            "--representation_dynamics_decode_max_updates",
            "8",
            "--representation_dynamics_storage_dtype",
            "float32",
        ]
        with patch.object(sys, "argv", argv):
            _, args = main_module.parse_eval_args()
        self.assertTrue(args.representation_dynamics)
        self.assertEqual(args.representation_dynamics_output, "./logs/representations")
        self.assertEqual(args.representation_dynamics_pooling, "mean_valid")
        self.assertEqual(args.representation_dynamics_phases, "both")
        self.assertEqual(args.representation_dynamics_decode_aggregation, "all_updates")
        self.assertEqual(args.representation_dynamics_decode_max_updates, 8)
        self.assertEqual(args.representation_dynamics_storage_dtype, "float32")

    def test_representation_dynamics_storage_defaults_to_float32(self):
        argv = ["lmms-eval", "--model", "openai", "--tasks", "mme", "--representation_dynamics"]
        with patch.object(sys, "argv", argv):
            _, args = main_module.parse_eval_args()
        self.assertEqual(args.representation_dynamics_storage_dtype, "float32")
        self.assertEqual(args.representation_dynamics_decode_aggregation, "mean_updates")
        self.assertIsNone(args.representation_dynamics_decode_max_updates)

    def test_representation_dynamics_save_writes_payload_metadata_and_manifest(self):
        payload = {
            "schema_version": 1,
            "selected_stack": "language_model.layers",
            "layer_specs": [{"layer_index": 0, "layer_name": "language_model.layers.0"}],
            "pooling": "last_valid",
            "phases": ["prefill"],
            "decode_aggregation": "mean_updates",
            "decode_max_updates": None,
            "storage_dtype": "float16",
            "num_hooks": 3,
            "num_contexts": 1,
            "num_records": 1,
            "skipped": {},
            "records": [
                {
                    "sample_id": "mme:test:0",
                    "task_name": "mme",
                    "split_name": "test",
                    "layer_index": 0,
                    "stream": "residual_state",
                    "phase": "prefill",
                    "hidden_size": 3,
                    "vector": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float16),
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                model="gemma3",
                model_args="pretrained=google/gemma-3-4b-it",
                tasks="mme",
                output_path=tmpdir,
                representation_dynamics_output=None,
                limit=2,
                offset=0,
                batch_size="1",
                gen_kwargs="max_new_tokens=8",
                seed=[0, 1234, 1234, 1234],
            )
            outputs = main_module._save_representation_dynamics_payload(args, "20260720_010101", payload)

            self.assertTrue(outputs["payload_path"].is_file())
            self.assertTrue(outputs["metadata_path"].is_file())
            self.assertTrue(outputs["records_path"].is_file())
            artifact = torch.load(outputs["payload_path"], map_location="cpu", weights_only=False)
            self.assertEqual(artifact["metadata"]["pretrained"], "google/gemma-3-4b-it")
            self.assertEqual(artifact["metadata"]["num_records"], 1)
            self.assertEqual(artifact["metadata"]["decode_aggregation"], "mean_updates")
            self.assertIn("lmms_eval_git_head", artifact["metadata"])
            self.assertIn("lmms_eval_git_dirty", artifact["metadata"])
            self.assertIn("lmms_eval_source_fingerprint_sha256", artifact["metadata"])
            self.assertTrue(torch.equal(artifact["records"][0]["vector"], payload["records"][0]["vector"]))

    def test_pruning_tracker_save_creates_per_run_directory(self):
        payload = {
            "mlp_stats": {"model.layers.0.mlp": torch.tensor([1.0, 2.0])},
            "attn_stats": {"model.layers.0.self_attn": torch.tensor([0.25, 0.75])},
            "mlp_counts": {"model.layers.0.mlp": 1},
            "attn_counts": {"model.layers.0.self_attn": 1},
            "linear_state_stats": {
                "input_output_delta_rms_ratio": {"model.layers.1.linear_attn": torch.tensor(0.25)},
                "output_input_rms_ratio": {"model.layers.1.linear_attn": torch.tensor(1.0)},
                "output_input_cosine": {"model.layers.1.linear_attn": torch.tensor(0.95)},
            },
            "linear_state_counts": {"model.layers.1.linear_attn": 1},
            "linear_state_component_stats": {"model.layers.1.linear_attn.in_proj_qkv": torch.tensor([0.5, 1.0])},
            "linear_state_component_counts": {"model.layers.1.linear_attn.in_proj_qkv": 1},
            "mlp_hist_counts": {"model.layers.0.mlp": torch.tensor([1.0, 1.0])},
            "attn_hist_counts": {"model.layers.0.self_attn": torch.tensor([1.0, 1.0])},
            "linear_state_output_hist_counts": {"model.layers.1.linear_attn": torch.tensor([1.0, 1.0])},
            "linear_state_input_output_delta_hist_counts": {"model.layers.1.linear_attn": torch.tensor([1.0, 1.0])},
            "linear_state_component_hist_counts": {"model.layers.1.linear_attn.in_proj_qkv": torch.tensor([1.0, 1.0])},
            "mlp_hist_source_elements": {"model.layers.0.mlp": 2},
            "attn_hist_source_elements": {"model.layers.0.self_attn": 2},
            "linear_state_output_hist_source_elements": {"model.layers.1.linear_attn": 2},
            "linear_state_input_output_delta_hist_source_elements": {"model.layers.1.linear_attn": 2},
            "linear_state_component_hist_source_elements": {"model.layers.1.linear_attn.in_proj_qkv": 2},
            "mlp_hist_bin_edges": [0.0, 1.0, 2.0],
            "attn_hist_bin_edges": [0.0, 0.5, 1.0],
            "linear_state_output_hist_bin_edges": [0.0, 0.5, 1.0],
            "linear_state_input_output_delta_hist_bin_edges": [0.0, 0.5, 1.0],
            "linear_state_component_hist_bin_edges": [0.0, 0.5, 1.0],
            "mlp_active_elements": {"model.layers.0.mlp": 2},
            "mlp_total_elements": {"model.layers.0.mlp": 2},
            "mlp_position_counts": {"model.layers.0.mlp": 1},
            "mlp_channel_fire_counts": {"model.layers.0.mlp": torch.tensor([1.0, 1.0])},
            "mlp_channel_fire_denoms": {"model.layers.0.mlp": 1},
            "linear_state_component_active_elements": {"model.layers.1.linear_attn.in_proj_qkv": 2},
            "linear_state_component_total_elements": {"model.layers.1.linear_attn.in_proj_qkv": 2},
            "linear_state_component_position_counts": {"model.layers.1.linear_attn.in_proj_qkv": 1},
            "linear_state_component_channel_fire_counts": {"model.layers.1.linear_attn.in_proj_qkv": torch.tensor([1.0, 1.0])},
            "linear_state_component_channel_fire_denoms": {"model.layers.1.linear_attn.in_proj_qkv": 1},
            "attn_entropy": {"model.layers.0.self_attn": torch.tensor([1.0, 1.5])},
            "attn_peak_mass": {"model.layers.0.self_attn": torch.tensor([0.4, 0.6])},
            "linear_state_component_parents": {"model.layers.1.linear_attn.in_proj_qkv": "model.layers.1.linear_attn"},
            "linear_state_component_roles": {"model.layers.1.linear_attn.in_proj_qkv": "write"},
            "module_registration_order": [
                "model.layers.0.self_attn",
                "model.layers.0.mlp",
                "model.layers.1.linear_attn",
                "model.layers.1.linear_attn.in_proj_qkv",
            ],
            "tracking_config": {"mlp_active_threshold_abs": 1e-6},
            "num_hooks": 3,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                model="openai",
                model_args="",
                tasks="mme",
                output_path=tmpdir,
                pruning_tracker_output=None,
            )
            outputs = main_module._save_pruning_tracker_payload(args, "20260416_010101", payload)

            self.assertTrue(Path(outputs["run_dir"]).is_dir())
            self.assertTrue(Path(outputs["payload_path"]).is_file())
            self.assertTrue(Path(outputs["summary_path"]).is_file())
            self.assertTrue(Path(outputs["histograms_path"]).is_file())
            self.assertTrue(Path(outputs["visualizations_path"]).is_file())
            self.assertTrue(Path(outputs["mlp_modules_path"]).is_file())
            self.assertTrue(Path(outputs["attention_modules_path"]).is_file())
            self.assertTrue(Path(outputs["linear_state_modules_path"]).is_file())
            self.assertTrue(Path(outputs["linear_state_components_path"]).is_file())
            self.assertTrue(Path(outputs["network_sequence_path"]).is_file())
            if outputs["plots_dir"] is not None:
                self.assertTrue(Path(outputs["plots_dir"]).is_dir())
            if outputs["plots_manifest_path"] is not None:
                self.assertTrue(Path(outputs["plots_manifest_path"]).is_file())

            with Path(outputs["visualizations_path"]).open("r", encoding="utf-8") as handle:
                visualizations = json.load(handle)
            self.assertIn("mlp_abs_activation_histogram", visualizations["charts"])
            self.assertIn("mlp_mean_by_network_order", visualizations["charts"])
            self.assertIn("linear_state_delta_ratio_by_network_order", visualizations["charts"])
            self.assertTrue(visualizations["heatmaps"]["attention_head_mean_by_network_order"]["available"])
            self.assertEqual(visualizations["network"]["module_sequence"][0]["module"], "model.layers.0.self_attn")
            if outputs["plots_manifest_path"] is not None:
                with Path(outputs["plots_manifest_path"]).open("r", encoding="utf-8") as handle:
                    plots_manifest = json.load(handle)
                self.assertIn("mlp_mean_by_network_order", plots_manifest["plots"])


if __name__ == "__main__":
    unittest.main()
