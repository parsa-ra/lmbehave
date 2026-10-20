import json
import itertools
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from lmms_eval.representations.analysis import (
    _batched_two_class_cosine_silhouette,
    _fisher_from_projection,
    _fisher_permutation_distribution,
    _fold_local_pca,
    _fold_local_ridge_residuals,
    _pca_projection,
    _permuted_group_assignments,
    analyze_artifacts,
    benjamini_hochberg,
    covariance_alignment,
    grouped_permutation_probe,
    grouped_stratified_folds,
    paired_linear_cka,
    rank_diagnostics,
    write_analysis,
)
from lmms_eval.representations.matched_excess import matched_probe_predictions
from lmms_eval.representations.matched_excess_calibration import (
    _augmented_selected_scores,
)
from lmms_eval.representations.lexical_diagnostics import variance_decomposition
from lmms_eval.representations.probe_calibration import (
    _lexical_cache_digest,
    _load_or_build_shared_lexical_cache,
    _read_lexical_cache,
    _write_lexical_cache,
    fixed_site_probe_distribution,
    lexical_adjusted_probe_distribution,
)
from lmms_eval.representations.representation_width_sensitivity import (
    activation_probe_predictions,
)
from lmms_eval.representations.selected_site_probe import _prepare_lexical_fold_cache
from lmms_eval.representations.split_unit_sensitivity import (
    _prepare_example_folds,
    example_level_scores,
)
from lmms_eval.representations.tracker import (
    RepresentationDynamicsTracker,
    build_representation_context,
)
from lmms_eval.representations.whitened_selector import (
    direct_regularized_whitened_score,
    regularized_whitened_distribution,
)
from torch import nn


class ToyAttention(nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 0.25


class ToyMLP(nn.Module):
    def forward(self, hidden_states):
        return hidden_states * 0.5


class Qwen3_5DecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = ToyAttention()
        self.mlp = ToyMLP()

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.self_attn(hidden_states)
        hidden_states = hidden_states + self.mlp(hidden_states)
        return hidden_states


class ToyLanguageModel(nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([Qwen3_5DecoderLayer() for _ in range(num_layers)])

    def forward(self, hidden_states):
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return hidden_states


class ToyVLM(nn.Module):
    def __init__(self, num_layers=2):
        super().__init__()
        self.language_model = ToyLanguageModel(num_layers=num_layers)

    def forward(self, hidden_states):
        return self.language_model(hidden_states)


class Qwen3_5LinearDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_attn = ToyAttention()
        self.mlp = ToyMLP()

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.linear_attn(hidden_states)
        return hidden_states + self.mlp(hidden_states)


class ToyLinearLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([Qwen3_5LinearDecoderLayer()])

    def forward(self, hidden_states):
        return self.layers[0](hidden_states)


class Gemma4TextDecoderLayer(nn.Module):
    def __init__(self, *, per_layer_input=False):
        super().__init__()
        self.self_attn = ToyAttention()
        self.mlp = ToyMLP()
        self.post_attention_layernorm = nn.Identity()
        self.post_feedforward_layernorm = nn.Identity()
        self.hidden_size_per_layer_input = 4 if per_layer_input else 0
        self.layer_scalar = 1.0

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.post_attention_layernorm(self.self_attn(hidden_states))
        hidden_states = hidden_states + self.post_feedforward_layernorm(self.mlp(hidden_states))
        return hidden_states


class Apertus1p5TextDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = ToyAttention()
        self.mlp = ToyMLP()

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.self_attn(hidden_states)
        return hidden_states + self.mlp(hidden_states)


class ToyApertus1p5(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.layers = nn.ModuleList([Apertus1p5TextDecoderLayer()])

    def forward(self, hidden_states):
        return self.model.language_model.layers[0](hidden_states)


class ApertusDecoderLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = ToyAttention()
        self.mlp = ToyMLP()

    def forward(self, hidden_states):
        hidden_states = hidden_states + self.self_attn(hidden_states)
        return hidden_states + self.mlp(hidden_states)


class ToyApertusBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([ApertusDecoderLayer()])

    def forward(self, hidden_states):
        return self.model.layers[0](hidden_states)


class ToyGemma4(nn.Module):
    def __init__(self, *, per_layer_input=False):
        super().__init__()
        self.language_model = nn.Module()
        self.language_model.layers = nn.ModuleList(
            [Gemma4TextDecoderLayer(per_layer_input=per_layer_input)]
        )

    def forward(self, hidden_states):
        return self.language_model.layers[0](hidden_states)


class ToyAmbiguousStacks(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Module()
        self.alpha.layers = nn.ModuleList([Qwen3_5DecoderLayer()])
        self.beta = nn.Module()
        self.beta.layers = nn.ModuleList([Qwen3_5DecoderLayer()])


class ToyAmbiguousUnequalStacks(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha = nn.Module()
        self.alpha.layers = nn.ModuleList([Qwen3_5DecoderLayer()])
        self.beta = nn.Module()
        self.beta.layers = nn.ModuleList([Qwen3_5DecoderLayer(), Qwen3_5DecoderLayer()])


class TestRepresentationDynamicsTracker(unittest.TestCase):
    def test_float32_is_default_and_float16_overflow_fails_loudly(self):
        model = ToyVLM()
        default_tracker = RepresentationDynamicsTracker(model)
        self.assertEqual(default_tracker.storage_dtype_name, "float32")
        default_tracker.remove_hooks()

        tracker = RepresentationDynamicsTracker(model, storage_dtype="float16")
        context = build_representation_context(
            torch.tensor([[1, 2]]),
            torch.ones(1, 2, dtype=torch.long),
            task_names="knowledge_a",
            sample_ids=[0],
        )
        with self.assertRaisesRegex(RuntimeError, "storage conversion produced non-finite"):
            with tracker.track_context(context):
                model(torch.full((1, 2, 4), 100_000.0))
        tracker.remove_hooks()

    def test_extracts_per_example_prefill_and_mean_decode_vectors(self):
        model = ToyVLM()
        tracker = RepresentationDynamicsTracker(
            model,
            pooling="last_valid",
            phases=("prefill", "decode"),
            storage_dtype="float32",
        )
        input_ids = torch.tensor([[0, 10, 11], [20, 21, 22]])
        attention_mask = torch.tensor([[0, 1, 1], [1, 1, 1]])
        context = build_representation_context(
            input_ids,
            attention_mask,
            task_names=["knowledge_a", "reasoning_a"],
            split_names=["test", "test"],
            sample_ids=[7, 9],
        )
        prefill = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        decode_one = torch.ones(2, 1, 4)
        decode_two = torch.full((2, 1, 4), 3.0)

        with tracker.track_context(context):
            model(prefill)
            model(decode_one)
            model(decode_two)
            tracker.record_generated_tokens(
                torch.tensor([[30, 31, 32], [40, 41, 42]]),
                eos_token_id=99,
                pad_token_id=0,
            )

        payload = tracker.export_state()
        self.assertEqual(payload["selected_stack"], "language_model.layers")
        self.assertEqual(len(payload["layer_specs"]), 2)
        self.assertEqual(payload["num_hooks"], 6)
        self.assertEqual(payload["num_records"], 2 * 2 * 3 * 2)
        self.assertEqual(payload["skipped"], {})

        first_prefill_attention = next(
            record
            for record in payload["records"]
            if record["sample_index"] == 0
            and record["layer_index"] == 0
            and record["stream"] == "attention_delta"
            and record["phase"] == "prefill"
        )
        self.assertTrue(torch.allclose(first_prefill_attention["vector"], prefill[0, 2] * 0.25))
        self.assertTrue(first_prefill_attention["residual_delta_exact"])
        self.assertEqual(first_prefill_attention["rank"], 0)
        self.assertEqual(first_prefill_attention["sample_id"], "knowledge_a:test:7")

        first_decode_attention = next(
            record
            for record in payload["records"]
            if record["sample_index"] == 0
            and record["layer_index"] == 0
            and record["stream"] == "attention_delta"
            and record["phase"] == "decode"
        )
        self.assertEqual(first_decode_attention["num_updates"], 2)
        self.assertEqual(first_decode_attention["decode_aggregation"], "mean_updates")
        expected_decode = ((decode_one[0, 0] * 0.25) + (decode_two[0, 0] * 0.25)) / 2
        self.assertTrue(torch.allclose(first_decode_attention["vector"], expected_decode))

        tracker.remove_hooks()
        self.assertEqual(tracker.hook_handles, [])

    def test_first_decode_update_is_retained_without_averaging(self):
        model = ToyVLM()
        tracker = RepresentationDynamicsTracker(
            model,
            phases=("decode",),
            decode_aggregation="first_update",
            storage_dtype="float32",
        )
        context = build_representation_context(
            torch.tensor([[10, 11]]),
            torch.ones(1, 2, dtype=torch.long),
            task_names="reasoning_a",
            sample_ids=[0],
        )
        with tracker.track_context(context):
            model(torch.zeros(1, 2, 4))
            model(torch.ones(1, 1, 4))
            model(torch.full((1, 1, 4), 9.0))
            tracker.record_generated_tokens(
                torch.tensor([[30, 31, 32]]),
                eos_token_id=99,
                pad_token_id=0,
            )

        record = next(
            record
            for record in tracker.records
            if record["phase"] == "decode" and record["stream"] == "attention_delta"
        )
        self.assertEqual(record["num_updates"], 1)
        self.assertEqual(record["decode_aggregation"], "first_update")
        self.assertTrue(torch.allclose(record["vector"], torch.full((4,), 0.25)))

    def test_all_decode_updates_are_emitted_with_explicit_steps_and_cap(self):
        model = ToyVLM()
        tracker = RepresentationDynamicsTracker(
            model,
            phases=("decode",),
            decode_aggregation="all_updates",
            decode_max_updates=2,
            storage_dtype="float32",
        )
        context = build_representation_context(
            torch.tensor([[10, 11]]),
            torch.ones(1, 2, dtype=torch.long),
            task_names="reasoning_a",
            sample_ids=[0],
        )
        with tracker.track_context(context):
            model(torch.zeros(1, 2, 4))
            model(torch.ones(1, 1, 4))
            model(torch.full((1, 1, 4), 3.0))
            model(torch.full((1, 1, 4), 9.0))
            tracker.record_generated_tokens(
                torch.tensor([[30, 31, 32, 33]]),
                eos_token_id=99,
                pad_token_id=0,
            )

        attention = [
            record
            for record in tracker.records
            if record["phase"] == "decode" and record["stream"] == "attention_delta"
        ]
        layer_zero = [record for record in attention if record["layer_index"] == 0]
        self.assertEqual([record["decode_step"] for record in layer_zero], [0, 1])
        self.assertEqual([record["num_updates"] for record in layer_zero], [1, 1])
        self.assertTrue(torch.allclose(layer_zero[0]["vector"], torch.full((4,), 0.25)))
        self.assertTrue(torch.allclose(layer_zero[1]["vector"], torch.full((4,), 0.75)))
        self.assertEqual(tracker.truncated["decode_updates_after_cap"], 6)
        self.assertEqual(tracker.skipped, {})
        self.assertEqual(tracker.export_state()["decode_max_updates"], 2)

    def test_decode_updates_after_eos_are_excluded_per_sample(self):
        model = ToyVLM(num_layers=1)
        tracker = RepresentationDynamicsTracker(
            model,
            phases=("decode",),
            decode_aggregation="all_updates",
            storage_dtype="float32",
        )
        context = build_representation_context(
            torch.tensor([[10, 11], [20, 21]]),
            torch.ones(2, 2, dtype=torch.long),
            task_names=["knowledge_a", "reasoning_a"],
            sample_ids=[0, 1],
        )
        with tracker.track_context(context):
            model(torch.zeros(2, 2, 4))
            model(torch.ones(2, 1, 4))
            model(torch.full((2, 1, 4), 3.0))
            model(torch.full((2, 1, 4), 9.0))
            tracker.record_generated_tokens(
                torch.tensor([[30, 99, 0, 0], [40, 41, 42, 43]]),
                eos_token_id=99,
                pad_token_id=0,
            )

        layer_zero_attention = [
            record
            for record in tracker.records
            if record["layer_index"] == 0 and record["stream"] == "attention_delta"
        ]
        steps_by_sample = {
            sample_index: [
                record["decode_step"]
                for record in layer_zero_attention
                if record["sample_index"] == sample_index
            ]
            for sample_index in (0, 1)
        }
        self.assertEqual(steps_by_sample, {0: [0], 1: [0, 1, 2]})
        self.assertEqual(tracker.skipped["decode_after_eos_or_padding"], 6)

    def test_mean_valid_pooling_excludes_padding(self):
        model = ToyVLM()
        tracker = RepresentationDynamicsTracker(model, pooling="mean_valid", storage_dtype="float32")
        values = torch.tensor([[[100.0, 100.0], [2.0, 4.0], [6.0, 8.0]]])
        context = build_representation_context(
            torch.tensor([[0, 1, 2]]),
            torch.tensor([[0, 1, 1]]),
            task_names="knowledge_a",
            sample_ids=[0],
        )
        with tracker.track_context(context):
            model(values)
        record = next(
            record
            for record in tracker.records
            if record["layer_index"] == 0 and record["stream"] == "attention_delta"
        )
        self.assertTrue(torch.allclose(record["vector"], torch.tensor([1.0, 1.5])))

    def test_context_rejects_mask_shape_mismatch(self):
        with self.assertRaisesRegex(ValueError, "does not match"):
            build_representation_context(
                torch.ones(2, 3, dtype=torch.long),
                torch.ones(2, 2, dtype=torch.long),
                task_names=["a", "b"],
                sample_ids=[0, 1],
            )

    def test_linear_attention_is_exported_as_attention_delta(self):
        model = ToyLinearLanguageModel()
        tracker = RepresentationDynamicsTracker(model, storage_dtype="float32")
        context = build_representation_context(
            torch.tensor([[1, 2]]),
            torch.ones(1, 2, dtype=torch.long),
            task_names="reasoning_a",
            sample_ids=[0],
        )
        with tracker.track_context(context):
            model(torch.ones(1, 2, 4))
        streams = {record["stream"] for record in tracker.records}
        self.assertEqual(streams, {"attention_delta", "mlp_delta", "residual_state"})

    def test_gemma4_uses_exact_post_branch_updates_and_marks_extra_ple_branch(self):
        model = ToyGemma4(per_layer_input=True)
        tracker = RepresentationDynamicsTracker(model, storage_dtype="float32")
        spec = tracker.export_state()["layer_specs"][0]
        self.assertEqual(spec["mixer_name"], "language_model.layers.0.post_attention_layernorm")
        self.assertEqual(spec["mlp_name"], "language_model.layers.0.post_feedforward_layernorm")
        self.assertTrue(spec["attention_delta_exact"])
        self.assertTrue(spec["mlp_delta_exact"])
        self.assertEqual(spec["additional_residual_branch"], "per_layer_input")
        self.assertEqual(spec["post_residual_transform"], "layer_scalar")
        self.assertFalse(spec["residual_change_equals_reported_branch_sum"])

    def test_ambiguous_decoder_stacks_fail_instead_of_using_name_tiebreak(self):
        with self.assertRaisesRegex(ValueError, "ambiguous decoder stacks"):
            RepresentationDynamicsTracker(ToyAmbiguousStacks(), storage_dtype="float32")

    def test_ambiguous_decoder_stacks_fail_instead_of_using_depth_tiebreak(self):
        with self.assertRaisesRegex(ValueError, "ambiguous decoder stacks"):
            RepresentationDynamicsTracker(ToyAmbiguousUnequalStacks(), storage_dtype="float32")

    def test_apertus1p5_selects_text_stack_and_exact_direct_branch_updates(self):
        model = ToyApertus1p5()
        tracker = RepresentationDynamicsTracker(model, storage_dtype="float32")
        try:
            payload = tracker.export_state()
            self.assertEqual(payload["selected_stack"], "model.language_model.layers")
            self.assertEqual(len(payload["layer_specs"]), 1)
            spec = payload["layer_specs"][0]
            self.assertTrue(spec["attention_delta_exact"])
            self.assertTrue(spec["mlp_delta_exact"])
            self.assertTrue(spec["mixer_name"].endswith("self_attn"))
            self.assertTrue(spec["mlp_name"].endswith("mlp"))
        finally:
            tracker.remove_hooks()

    def test_apertus_base_selects_exact_direct_branch_updates(self):
        model = ToyApertusBase()
        tracker = RepresentationDynamicsTracker(model, storage_dtype="float32")
        try:
            payload = tracker.export_state()
            self.assertEqual(payload["selected_stack"], "model.layers")
            self.assertEqual(len(payload["layer_specs"]), 1)
            spec = payload["layer_specs"][0]
            self.assertTrue(spec["attention_delta_exact"])
            self.assertTrue(spec["mlp_delta_exact"])
            self.assertTrue(spec["mixer_name"].endswith("self_attn"))
            self.assertTrue(spec["mlp_name"].endswith("mlp"))
        finally:
            tracker.remove_hooks()


class TestRepresentationAnalysis(unittest.TestCase):
    def test_batched_cosine_silhouette_matches_sklearn_across_batch_sizes(self):
        rng = np.random.default_rng(19)
        labels = np.repeat(np.asarray([0, 1], dtype=np.int64), 20)
        features = rng.normal(size=(40, 17)).astype(np.float64)
        features[labels == 1, :3] += 1.5

        from sklearn.metrics import silhouette_score

        expected = float(silhouette_score(features, labels, metric="cosine"))
        for batch_size in (1, 7, 128):
            observed = _batched_two_class_cosine_silhouette(
                np,
                features,
                labels,
                batch_size=batch_size,
            )
            self.assertAlmostEqual(observed, expected, places=12)

    def test_effective_rank_is_scale_invariant_and_tracks_active_dimensions(self):
        rank_one = np.asarray([[1.0, 0.0], [-1.0, 0.0], [2.0, 0.0], [-2.0, 0.0]])
        rank_two = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)

        one = rank_diagnostics(rank_one, labels)
        scaled = rank_diagnostics(19.0 * rank_one, labels)
        two = rank_diagnostics(rank_two, labels)

        self.assertAlmostEqual(one["effective_rank_all"], 1.0, places=12)
        self.assertAlmostEqual(one["effective_rank_all"], scaled["effective_rank_all"], places=12)
        self.assertAlmostEqual(two["effective_rank_all"], 2.0, places=12)
        self.assertEqual(two["rank_capacity_all"], 2)

    def test_fold_local_pca_fits_only_training_rows(self):
        train = np.asarray([[1.0, 0.0], [-1.0, 0.0], [0.5, 0.0], [-0.5, 0.0]])
        test = np.asarray([[0.0, 100.0], [0.0, -100.0]])
        projected_train, projected_test = _fold_local_pca(
            train,
            test,
            retained_variance=0.95,
            analysis_backend="sklearn",
        )
        self.assertEqual(projected_train.shape, (4, 1))
        self.assertEqual(projected_test.shape, (2, 1))
        self.assertTrue(np.allclose(projected_test, 0.0))

    def test_fold_local_ridge_residualization_uses_training_fit(self):
        train_nuisance = np.asarray([[0.0], [1.0], [2.0], [3.0]])
        test_nuisance = np.asarray([[4.0], [5.0]])
        train = np.column_stack((2.0 + 3.0 * train_nuisance[:, 0], -1.0 * train_nuisance[:, 0]))
        test = np.column_stack((2.0 + 3.0 * test_nuisance[:, 0], -1.0 * test_nuisance[:, 0]))

        train_residual, test_residual = _fold_local_ridge_residuals(
            train,
            test,
            train_nuisance,
            test_nuisance,
            alpha=1e-9,
            analysis_backend="sklearn",
        )

        self.assertTrue(np.allclose(train_residual, 0.0, atol=1e-8))
        self.assertTrue(np.allclose(test_residual, 0.0, atol=1e-8))

    def test_benjamini_hochberg_adjustment_preserves_order_and_nan(self):
        adjusted = benjamini_hochberg([0.01, 0.04, float("nan"), 0.03])
        self.assertTrue(
            np.allclose(
                [adjusted[0], adjusted[1], adjusted[3]],
                [0.04, 0.05333333333333334, 0.05333333333333334],
            )
        )
        self.assertTrue(math.isnan(adjusted[2]))

    def test_benjamini_hochberg_retains_failed_declared_hypotheses(self):
        adjusted = benjamini_hochberg([0.01, 0.03], family_size=4)
        self.assertTrue(np.allclose(adjusted, [0.04, 0.06]))

    def test_alignment_is_missing_for_degenerate_geometry(self):
        constant = np.ones((4, 3))
        self.assertTrue(math.isnan(covariance_alignment(constant, constant)))
        self.assertTrue(math.isnan(paired_linear_cka(constant, constant)))

    def test_grouped_folds_are_deterministic_under_global_label_swap(self):
        groups = np.repeat(np.asarray(["a", "b", "c", "d", "e", "f"], dtype=object), 2)
        labels = np.repeat(np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64), 2)
        folds, effective = grouped_stratified_folds(labels, groups, requested_splits=3, seed=17)
        swapped, swapped_effective = grouped_stratified_folds(1 - labels, groups, requested_splits=3, seed=17)

        self.assertEqual(effective, 3)
        self.assertEqual(swapped_effective, 3)
        for (_, test_indices), (_, swapped_test_indices) in zip(folds, swapped):
            self.assertEqual(set(groups[test_indices]), set(groups[swapped_test_indices]))

    def test_grouped_folds_restratify_every_balanced_assignment(self):
        task_names = [f"task_{index}" for index in range(8)]
        groups = np.repeat(np.asarray(task_names, dtype=object), 3)
        for positive_indices in itertools.combinations(range(8), 4):
            positive = set(positive_indices)
            task_labels = np.asarray(
                [int(index in positive) for index in range(8)],
                dtype=np.int64,
            )
            labels = np.repeat(task_labels, 3)
            folds, effective = grouped_stratified_folds(
                labels,
                groups,
                requested_splits=4,
                seed=17,
            )
            self.assertEqual(effective, 4)
            for train_indices, test_indices in folds:
                self.assertEqual(set(labels[train_indices]), {0, 1})
                self.assertEqual(set(labels[test_indices]), {0, 1})
                test_task_labels = [
                    task_labels[task_names.index(str(task))]
                    for task in np.unique(groups[test_indices])
                ]
                self.assertEqual(test_task_labels.count(0), test_task_labels.count(1))

    def test_matched_excess_uses_equal_width_and_is_complement_symmetric(self):
        rng = np.random.default_rng(29)
        task_names = [f"task_{index}" for index in range(8)]
        groups = np.repeat(np.asarray(task_names, dtype=object), 5)
        labels = np.repeat(np.asarray([1] * 4 + [0] * 4, dtype=np.int64), 5)
        activations = rng.normal(scale=0.05, size=(40, 5))
        activations[:, 0] += np.where(labels == 1, 2.0, -2.0)
        prompt_features = rng.normal(size=(40, 2))

        prompt_fold_cache = {}
        folds, _ = grouped_stratified_folds(labels, groups, requested_splits=4, seed=13)
        for train_indices, test_indices in folds:
            prompt_fold_cache[test_indices.tobytes()] = (
                prompt_features[train_indices],
                prompt_features[test_indices],
            )
        observed = matched_probe_predictions(
            activations,
            labels,
            groups,
            requested_splits=4,
            components=2,
            seed=13,
            analysis_backend="sklearn",
            prompt_fold_cache=prompt_fold_cache,
            activation_fold_cache={},
            scaled_prompt_fold_cache={},
        )
        complemented = matched_probe_predictions(
            activations,
            1 - labels,
            groups,
            requested_splits=4,
            components=2,
            seed=13,
            analysis_backend="sklearn",
            prompt_fold_cache=prompt_fold_cache,
            activation_fold_cache={},
            scaled_prompt_fold_cache={},
        )
        self.assertEqual(observed["components"], 2)
        self.assertGreater(observed["activation_macro_f1"], observed["prompt_macro_f1"])
        self.assertAlmostEqual(
            observed["macro_f1_delta"],
            complemented["macro_f1_delta"],
        )

    def test_example_level_probe_reuses_fixed_label_blind_folds(self):
        rng = np.random.default_rng(41)
        labels = np.asarray([0] * 50 + [1] * 50, dtype=np.int64)
        features = rng.normal(scale=0.1, size=(100, 4))
        features[:, 0] += np.where(labels == 1, 2.0, -2.0)
        folds = _prepare_example_folds(
            features,
            labels,
            components=2,
            splits=5,
            seed=7,
            analysis_backend="sklearn",
        )
        accuracy, macro_f1, minimum = example_level_scores(
            folds,
            labels,
            seed=7,
            analysis_backend="sklearn",
        )
        self.assertEqual(len(folds), 5)
        self.assertGreater(accuracy, 0.95)
        self.assertGreater(macro_f1, 0.95)
        self.assertEqual(minimum, 40)

    def test_activation_width_probe_supports_fixed_and_raw_reductions(self):
        rng = np.random.default_rng(43)
        task_names = [f"task_{index}" for index in range(8)]
        groups = np.repeat(np.asarray(task_names, dtype=object), 5)
        labels = np.repeat(np.asarray([1] * 4 + [0] * 4, dtype=np.int64), 5)
        features = rng.normal(scale=0.1, size=(40, 5))
        features[:, 0] += np.where(labels == 1, 2.0, -2.0)
        fixed = activation_probe_predictions(
            features,
            labels,
            groups,
            requested_splits=4,
            reduction=("fixed", 2),
            seed=11,
            analysis_backend="sklearn",
            activation_fold_cache={},
        )
        raw = activation_probe_predictions(
            features,
            labels,
            groups,
            requested_splits=4,
            reduction=("raw", None),
            seed=11,
            analysis_backend="sklearn",
            activation_fold_cache={},
        )
        complemented = activation_probe_predictions(
            features,
            1 - labels,
            groups,
            requested_splits=4,
            reduction=("raw", None),
            seed=11,
            analysis_backend="sklearn",
            activation_fold_cache={},
        )
        self.assertEqual(fixed["activation_dimension"], 2)
        self.assertEqual(raw["activation_dimension"], 5)
        self.assertAlmostEqual(raw["macro_f1"], complemented["macro_f1"])

    def test_matched_calibration_appends_site_without_overwriting_real_selector(self):
        real_fisher = np.asarray(
            [[0.7, 0.2, 0.4], [0.3, 0.8, 0.5]], dtype=np.float64
        )
        real_f1 = np.asarray([0.61, 0.62, 0.63], dtype=np.float64)
        synthetic_fisher = np.asarray([0.7, 0.9, 0.1], dtype=np.float64)
        synthetic_f1 = np.asarray([0.91, 0.92, 0.93], dtype=np.float64)
        selected_f1, maxima, selected = _augmented_selected_scores(
            real_fisher_matrix=real_fisher,
            real_selected_macro_f1=real_f1,
            synthetic_fisher=synthetic_fisher,
            synthetic_macro_f1=synthetic_f1,
        )
        # A tie retains the pre-existing real site; a strict improvement selects
        # the synthetic site; a weaker candidate leaves the real result unchanged.
        np.testing.assert_array_equal(selected, [False, True, False])
        np.testing.assert_allclose(selected_f1, [0.61, 0.92, 0.63])
        np.testing.assert_allclose(maxima, [0.7, 0.9, 0.5])

    def test_whitened_selector_matches_direct_pooled_covariance_inverse(self):
        rng = np.random.default_rng(47)
        task_order = ["a", "b", "c", "d"]
        groups = np.repeat(np.asarray(task_order, dtype=object), 6)
        projected = rng.normal(size=(24, 5))
        assignment_ids = ["1100", "1010", "1001", "0110", "0101", "0011"]
        scores, ridge, _mean_eigenvalue = regularized_whitened_distribution(
            projected,
            groups,
            task_order=task_order,
            assignment_ids=assignment_ids,
            ridge_alpha=0.1,
            analysis_backend="sklearn",
        )
        for assignment, score in zip(assignment_ids, scores, strict=True):
            task_labels = {task: int(bit) for task, bit in zip(task_order, assignment, strict=True)}
            labels = np.asarray([task_labels[str(group)] for group in groups], dtype=np.int64)
            direct = direct_regularized_whitened_score(projected, labels, ridge=ridge)
            self.assertAlmostEqual(float(score), direct, places=10)
        np.testing.assert_allclose(scores[:3], scores[:2:-1])

    def test_task_variance_decomposition_separates_between_and_within_terms(self):
        groups = np.asarray(["a", "a", "b", "b"], dtype=object)
        features = np.asarray([[0.0], [2.0], [4.0], [6.0]], dtype=np.float64)
        result = variance_decomposition(features, groups, analysis_backend="sklearn")
        self.assertAlmostEqual(result["total_sum_squares"], 20.0)
        self.assertAlmostEqual(result["between_task_sum_squares"], 16.0)
        self.assertAlmostEqual(result["within_task_sum_squares"], 4.0)
        self.assertAlmostEqual(result["between_task_fraction"], 0.8)

    def test_fixed_site_calibration_retains_complete_complement_support(self):
        rng = np.random.default_rng(37)
        task_order = ["a", "b", "c", "d"]
        assignment_ids = [
            "".join("1" if index in positive else "0" for index in range(4))
            for positive in itertools.combinations(range(4), 2)
        ]
        groups = np.repeat(np.asarray(task_order, dtype=object), 6)
        observed_labels = np.repeat(np.asarray([1, 1, 0, 0], dtype=np.int64), 6)
        features = rng.normal(scale=0.05, size=(24, 4))
        features[:, 0] += np.where(observed_labels == 1, 1.0, -1.0)
        result = fixed_site_probe_distribution(
            features,
            groups,
            assignment_ids=assignment_ids,
            task_order=task_order,
            requested_splits=2,
            components=2,
            seed=5,
            analysis_backend="sklearn",
        )
        self.assertEqual(len(result["macro_f1"]), 6)
        self.assertEqual(result["probe_refits"], 3)
        self.assertTrue(result["macro_f1_complement_symmetry"]["verified"])

    def test_lexical_adjusted_calibration_retains_complete_complement_support(self):
        rng = np.random.default_rng(41)
        task_order = ["a", "b", "c", "d"]
        assignment_ids = [
            "".join("1" if index in positive else "0" for index in range(4))
            for positive in itertools.combinations(range(4), 2)
        ]
        groups = np.repeat(np.asarray(task_order, dtype=object), 6)
        observed_labels = np.repeat(np.asarray([1, 1, 0, 0], dtype=np.int64), 6)
        features = rng.normal(scale=0.05, size=(24, 4))
        features[:, 0] += np.where(observed_labels == 1, 1.0, -1.0)
        nuisance = rng.normal(size=(24, 2))
        lexical_fold_cache = {}
        for assignment in assignment_ids[:3]:
            labels = np.asarray(
                [int(assignment[task_order.index(str(group))]) for group in groups],
                dtype=np.int64,
            )
            folds, _ = grouped_stratified_folds(labels, groups, requested_splits=2, seed=5)
            for train_indices, test_indices in folds:
                lexical_fold_cache[test_indices.tobytes()] = (
                    nuisance[train_indices],
                    nuisance[test_indices],
                )
        result = lexical_adjusted_probe_distribution(
            features,
            groups,
            assignment_ids=assignment_ids,
            task_order=task_order,
            requested_splits=2,
            seed=5,
            analysis_backend="sklearn",
            lexical_fold_cache=lexical_fold_cache,
            nuisance_alpha=1.0,
            pca_variance=0.95,
        )
        self.assertEqual(len(result["macro_f1"]), 6)
        self.assertEqual(result["probe_refits"], 3)
        self.assertTrue(result["macro_f1_complement_symmetry"]["verified"])

    def test_shared_lexical_cache_round_trip_is_digest_bound(self):
        cache = {
            np.asarray([1, 3], dtype=np.int64).tobytes(): (
                np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64),
                np.asarray([[5.0, 6.0]], dtype=np.float64),
            )
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lexical.npz"
            _write_lexical_cache(path, cache_digest="expected", cache=cache)
            restored = _read_lexical_cache(path, expected_digest="expected")
            self.assertEqual(set(restored), set(cache))
            key = next(iter(cache))
            np.testing.assert_array_equal(restored[key][0], cache[key][0])
            np.testing.assert_array_equal(restored[key][1], cache[key][1])
            with self.assertRaisesRegex(ValueError, "does not match"):
                _read_lexical_cache(path, expected_digest="different")

    def test_shared_lexical_cache_build_is_reusable(self):
        task_order = ["a", "b", "c", "d"]
        assignments = [
            "".join("1" if index in positive else "0" for index in range(4))
            for positive in itertools.combinations(range(4), 2)
        ]
        groups = np.repeat(np.asarray(task_order, dtype=object), 4)
        prompts = np.asarray(
            [f"task {group} sample {index} alpha beta gamma" for index, group in enumerate(groups)],
            dtype=object,
        )
        digest = _lexical_cache_digest(
            assignment_ids=assignments,
            task_order=task_order,
            groups=groups,
            prompts=prompts,
            requested_splits=2,
            components=2,
            seed=7,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shared.npz"
            first = _load_or_build_shared_lexical_cache(
                path,
                cache_digest=digest,
                assignment_ids=assignments,
                task_order=task_order,
                groups=groups,
                prompts=prompts,
                requested_splits=2,
                components=2,
                seed=7,
                workers=2,
            )
            second = _load_or_build_shared_lexical_cache(
                path,
                cache_digest=digest,
                assignment_ids=assignments,
                task_order=task_order,
                groups=groups,
                prompts=prompts,
                requested_splits=2,
                components=2,
                seed=7,
                workers=2,
            )
            self.assertEqual(set(first), set(second))
            self.assertGreater(len(first), 0)
            sequential = {}
            component_counts = set()
            for assignment in assignments[:3]:
                labels = np.asarray(
                    [int(assignment[task_order.index(str(group))]) for group in groups],
                    dtype=np.int64,
                )
                _prepare_lexical_fold_cache(
                    labels,
                    groups,
                    prompts,
                    requested_splits=2,
                    seed=7,
                    kind="char_tfidf",
                    components=2,
                    cache=sequential,
                    component_counts=component_counts,
                )
            self.assertEqual(set(first), set(sequential))
            for key in first:
                np.testing.assert_allclose(first[key][0], sequential[key][0], rtol=0, atol=0)
                np.testing.assert_allclose(first[key][1], sequential[key][1], rtol=0, atol=0)

    def test_permutations_reuse_the_observed_fold_seed(self):
        groups = np.repeat(np.asarray(["ka", "kb", "ra", "rb"], dtype=object), 2)
        labels = np.repeat(np.asarray([1, 1, 0, 0], dtype=np.int64), 2)
        features = np.arange(24, dtype=np.float64).reshape(8, 3)

        with patch(
            "lmms_eval.representations.analysis._probe_predictions",
            return_value=(0.5, 0.5, 2),
        ) as predictions:
            result = grouped_permutation_probe(
                features,
                labels,
                groups,
                kind="linear",
                requested_splits=2,
                max_permutations=5,
                seed=23,
            )

        # Six oriented assignments form three complementary pairs.  A balanced
        # linear probe has the same score after exchanging class names, so only
        # one fit per pair is required.
        self.assertEqual(predictions.call_count, 3)
        self.assertEqual({call.kwargs["seed"] for call in predictions.call_args_list}, {23})
        self.assertEqual(result["fold_seed"], 23)
        self.assertEqual(result["probe_refits"], 3)
        self.assertTrue(result["class_label_complements_share_one_probe_refit"])
        self.assertTrue(result["permutation_symmetric"])
        self.assertEqual(result["permutation_design"]["effective_unoriented_assignments"], 3)
        self.assertAlmostEqual(result["permutation_design"]["theoretical_minimum_p"], 1.0 / 3.0)

    def test_exhaustive_balanced_probe_has_complement_aware_p_floor(self):
        task_names = [f"task_{index}" for index in range(10)]
        groups = np.repeat(np.asarray(task_names, dtype=object), 2)
        labels = np.repeat(np.asarray([1] * 5 + [0] * 5, dtype=np.int64), 2)
        rng = np.random.default_rng(9)
        task_centers = rng.normal(size=(10, 4))
        features = np.vstack(
            [task_centers[index] + rng.normal(scale=0.01, size=(2, 4)) for index in range(10)]
        )

        result = grouped_permutation_probe(
            features,
            labels,
            groups,
            kind="linear",
            requested_splits=5,
            max_permutations=251,
            seed=4,
        )

        self.assertEqual(result["num_group_permutations"], 251)
        self.assertTrue(result["permutation_design"]["complement_paired"])
        self.assertEqual(result["permutation_design"]["effective_unoriented_assignments"], 126)
        self.assertAlmostEqual(result["permutation_design"]["theoretical_minimum_p"], 1.0 / 126.0)
        self.assertGreaterEqual(result["accuracy_p"], 1.0 / 126.0)
        self.assertGreaterEqual(result["f1_p"], 1.0 / 126.0)

    def test_vectorized_fisher_null_matches_direct_recomputation(self):
        rng = np.random.default_rng(71)
        task_names = [f"task_{index}" for index in range(10)]
        groups = np.repeat(np.asarray(task_names, dtype=object), 7)
        labels = np.repeat(np.asarray([1] * 5 + [0] * 5, dtype=np.int64), 7)
        features = rng.normal(size=(len(groups), 13))
        features[labels == 1, :2] += 0.4
        projected, explained = _pca_projection(
            features, retained_variance=0.95, analysis_backend="sklearn"
        )
        observed = _fisher_from_projection(
            projected,
            labels,
            analysis_backend="sklearn",
            explained_variance=explained,
        )
        assignments = _permuted_group_assignments(
            labels, groups, max_permutations=251, seed=0
        )
        vectorized = _fisher_permutation_distribution(
            projected,
            labels,
            groups,
            assignments=assignments,
            analysis_backend="sklearn",
            explained_variance=explained,
            observed_fdr=float(observed["fdr"]),
        )
        direct = [float(observed["fdr"])]
        string_groups = groups.astype(str)
        for assignment in assignments:
            permuted = np.asarray([assignment[group] for group in string_groups], dtype=np.int64)
            direct.append(
                float(
                    _fisher_from_projection(
                        projected,
                        permuted,
                        analysis_backend="sklearn",
                        explained_variance=explained,
                    )["fdr"]
                )
            )

        np.testing.assert_allclose(vectorized["distribution"], direct, rtol=1e-12, atol=1e-12)
        self.assertEqual(vectorized["calculation"], "vectorized_task_sufficient_statistics")
        self.assertTrue(vectorized["complement_symmetry"]["verified"])

    def test_complement_asymmetry_invalidates_exact_probe_inference(self):
        groups = np.repeat(np.asarray(["ka", "kb", "ra", "rb"], dtype=object), 2)
        labels = np.repeat(np.asarray([1, 1, 0, 0], dtype=np.int64), 2)
        features = np.arange(24, dtype=np.float64).reshape(8, 3)
        # Six values cover the observed assignment plus all five alternatives.
        # They deliberately violate equality for at least one complement pair.
        outputs = [(0.8, 0.8, 2), (0.7, 0.7, 2), (0.6, 0.6, 2),
                   (0.5, 0.5, 2), (0.4, 0.4, 2), (0.3, 0.3, 2)]
        with patch(
            "lmms_eval.representations.analysis._probe_predictions",
            side_effect=outputs,
        ):
            result = grouped_permutation_probe(
                features,
                labels,
                groups,
                kind="rbf",
                requested_splits=2,
                max_permutations=5,
                seed=23,
            )

        self.assertFalse(result["permutation_symmetric"])
        self.assertFalse(result["accuracy_inference_valid"])
        self.assertFalse(result["f1_inference_valid"])
        self.assertTrue(math.isnan(result["accuracy_p"]))
        self.assertTrue(math.isnan(result["f1_p"]))

    @staticmethod
    def _artifact(path: Path) -> None:
        rng = np.random.default_rng(3)
        records = []
        tasks = [("knowledge_a", 1), ("knowledge_b", 1), ("reasoning_a", 0), ("reasoning_b", 0)]
        for layer in (0, 1):
            for stream in ("attention_delta", "mlp_delta", "residual_state"):
                for task_name, label in tasks:
                    for example_index in range(8):
                        vector = rng.normal(0.0, 0.5, size=6)
                        if layer == 1 and stream == "attention_delta":
                            vector[0] += 5.0 if label == 1 else -5.0
                        pair_group = task_name.rsplit("_", 1)[-1]
                        records.append(
                            {
                                "context_id": f"{task_name}:{example_index}",
                                "sample_index": 0,
                                "sample_id": f"{task_name}:test:{example_index}",
                                "pair_id": f"pair:{pair_group}:{example_index}",
                                "task_name": task_name,
                                "split_name": "test",
                                "prompt_length": 5,
                                "layer_index": layer,
                                "stream": stream,
                                "module_name": f"language_model.layers.{layer}.{stream}",
                                "phase": "prefill",
                                "pooling": "last_valid",
                                "num_updates": 1,
                                "hidden_size": 6,
                                "residual_delta_exact": True,
                                "vector": torch.tensor(vector, dtype=torch.float32),
                            }
                        )
        torch.save(
            {
                "metadata": {"model": "toy", "pretrained": "toy/model"},
                "schema_version": 1,
                "records": records,
            },
            path,
        )

    def test_analysis_reports_all_quantitative_metrics_and_maximum(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "representations.pt"
            self._artifact(artifact_path)
            rows, report = analyze_artifacts(
                [artifact_path],
                knowledge_tasks={"knowledge_a", "knowledge_b"},
                reasoning_tasks={"reasoning_a", "reasoning_b"},
                cv_folds=5,
                permutations=5,
                seed=4,
            )
            self.assertEqual(len(rows), 6)
            self.assertIsNone(report["permutation_statistic_deterministic"])
            self.assertIn("separate replay", report["permutation_determinism_note"])
            self.assertIn("global task order", report["grouped_fold_policy"])
            self.assertEqual(report["primary_maximum_divergence"]["layer"], 1)
            self.assertEqual(report["primary_maximum_divergence"]["stream"], "attention_delta")
            self.assertEqual(report["primary_maximum_divergence_by_model"]["toy/model"]["layer"], 1)
            self.assertEqual(
                report["primary_branch_maximum_divergence_by_model"]["toy/model"]["stream"],
                "attention_delta",
            )
            self.assertEqual(
                set(report["primary_branch_maximum_divergence_by_stream"]),
                {"attention_delta", "mlp_delta"},
            )
            self.assertEqual(
                report["primary_branch_maximum_divergence_by_model_and_stream"]["toy/model"][
                    "attention_delta"
                ]["layer"],
                1,
            )
            self.assertEqual(
                report["primary_branch_maximum_divergence_by_model_and_stream"]["toy/model"]["mlp_delta"][
                    "stream"
                ],
                "mlp_delta",
            )
            target = next(row for row in rows if row["layer"] == 1 and row["stream"] == "attention_delta")
            self.assertGreater(target["fdr"], 1.0)
            self.assertTrue(0.0 <= target["fdr_p"] <= 1.0)
            self.assertTrue(0.0 <= target["fdr_q"] <= 1.0)
            self.assertGreater(target["fdr_between_sum"], 0.0)
            self.assertGreater(target["fdr_within_sum"], 0.0)
            self.assertGreater(target["effective_rank_all"], 0.0)
            self.assertGreater(target["effective_rank_within"], 0.0)
            self.assertGreater(target["stable_rank_all"], 0.0)
            self.assertGreater(target["within_variance_sum_raw"], 0.0)
            self.assertEqual(target["probe_pca_fit_scope"], "none")
            self.assertGreaterEqual(target["probe_accuracy"], 0.9)
            self.assertGreaterEqual(target["rbf_probe_accuracy"], 0.8)
            self.assertEqual(target["cv_folds"], 2)
            self.assertTrue(0.0 <= target["probe_accuracy_q"] <= 1.0)
            self.assertTrue(math.isnan(target["rbf_probe_accuracy_q"]))
            self.assertTrue(0.0 <= target["cka"] <= 1.0)
            self.assertTrue(0.0 <= target["covariance_alignment"] <= 1.0)
            self.assertTrue(math.isfinite(target["silhouette_cosine"]))
            calibration = report["selection_aware_randomization_by_model"]["toy/model"]["metrics"]
            self.assertIn("fdr", calibration)
            self.assertIn("linear_f1", calibration)
            self.assertIn("linear_f1_at_fdr_selected_row", calibration)
            self.assertTrue(
                calibration["linear_f1_at_fdr_selected_row"][
                    "selection_recomputed_for_every_assignment"
                ]
            )
            self.assertTrue(0.0 <= calibration["fdr"]["maximum_randomization_p"] <= 1.0)

            outputs = write_analysis(rows, report, Path(tmpdir) / "analysis")
            self.assertTrue(outputs["csv"].is_file())
            self.assertTrue(outputs["report"].is_file())
            self.assertTrue(outputs["permutation_nulls"].is_file())
            saved_report = json.loads(outputs["report"].read_text(encoding="utf-8"))
            self.assertEqual(saved_report["num_rows"], 6)
            self.assertNotIn("_permutation_null_payload", saved_report)
            self.assertEqual(len(saved_report["permutation_null_artifact"]["sha256"]), 64)

            parallel_rows, parallel_report = analyze_artifacts(
                [artifact_path],
                knowledge_tasks={"knowledge_a", "knowledge_b"},
                reasoning_tasks={"reasoning_a", "reasoning_b"},
                cv_folds=5,
                permutations=5,
                seed=4,
                jobs=2,
            )
            self.assertEqual(parallel_report["parallel_jobs"], 2)
            self.assertEqual(len(parallel_rows), len(rows))
            for serial, parallel in zip(rows, parallel_rows):
                self.assertEqual(serial.keys(), parallel.keys())
                for key in serial:
                    if isinstance(serial[key], float) and math.isnan(serial[key]):
                        self.assertTrue(math.isnan(parallel[key]))
                    elif isinstance(serial[key], float):
                        self.assertAlmostEqual(serial[key], parallel[key], places=12)
                    else:
                        self.assertEqual(serial[key], parallel[key])

    def test_analysis_reports_fold_local_nuisance_probe(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "representations.pt"
            nuisance_path = Path(tmpdir) / "nuisance.csv"
            self._artifact(artifact_path)
            with nuisance_path.open("w", encoding="utf-8", newline="") as handle:
                handle.write("model,task_name,sample_id,prompt_char_count,option_count\n")
                for task_name in ("knowledge_a", "knowledge_b", "reasoning_a", "reasoning_b"):
                    for example_index in range(8):
                        handle.write(
                            f"toy/model,{task_name},{task_name}:test:{example_index},"
                            f"{20 + example_index},{4 if task_name.startswith('knowledge') else 0}\n"
                        )

            rows, report = analyze_artifacts(
                [artifact_path],
                knowledge_tasks={"knowledge_a", "knowledge_b"},
                reasoning_tasks={"reasoning_a", "reasoning_b"},
                cv_folds=2,
                permutations=5,
                seed=4,
                nuisance_table=nuisance_path,
                nuisance_columns=("prompt_char_count", "option_count"),
                nuisance_alpha=1.0,
            )

            self.assertEqual(len(rows), 6)
            self.assertTrue(report["nuisance_control"]["enabled"])
            self.assertEqual(report["nuisance_control"]["matched_sample_rows"], 32)
            self.assertEqual(report["nuisance_control"]["fit_scope"], "training_tasks_per_fold")
            self.assertEqual(len(report["nuisance_control"]["table_sha256"]), 64)
            for row in rows:
                self.assertEqual(row["nuisance_fit_scope"], "training_tasks_per_fold")
                self.assertTrue(0.0 <= row["nuisance_probe_accuracy"] <= 1.0)
                self.assertTrue(0.0 <= row["nuisance_probe_f1_q"] <= 1.0)

    def test_fisher_null_can_be_exhaustive_without_exhaustive_rowwise_probes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "representations.pt"
            self._artifact(artifact_path)
            rows, report = analyze_artifacts(
                [artifact_path],
                knowledge_tasks={"knowledge_a", "knowledge_b"},
                reasoning_tasks={"reasoning_a", "reasoning_b"},
                cv_folds=2,
                permutations=1,
                fdr_permutations=5,
                reference_positive_tasks={"knowledge_a", "reasoning_a"},
                seed=4,
            )

            self.assertEqual(report["group_permutations_requested"], 1)
            self.assertEqual(report["fdr_group_permutations_requested"], 5)
            self.assertTrue(all(row["num_group_permutations"] == 1 for row in rows))
            self.assertTrue(all(row["fdr_num_group_permutations"] == 5 for row in rows))
            self.assertTrue(all(math.isclose(row["minimum_attainable_fdr_p"], 1.0 / 3.0) for row in rows))
            metrics = report["selection_aware_randomization_by_model"]["toy/model"]["metrics"]
            self.assertIn("fdr", metrics)
            self.assertNotIn("linear_f1", metrics)
            self.assertNotIn("linear_f1_at_fdr_selected_row", metrics)
            reference = report["selection_aware_randomization_by_model"]["toy/model"][
                "reference_partition"
            ]
            self.assertEqual(reference["positive_tasks"], ["knowledge_a", "reasoning_a"])
            self.assertTrue(0.0 <= metrics["fdr"]["reference_partition"]["maximum_randomization_p"] <= 1.0)

    def test_zero_permutation_analysis_preserves_undefined_null_fields(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_path = Path(tmpdir) / "representations.pt"
            self._artifact(artifact_path)
            rows, report = analyze_artifacts(
                [artifact_path],
                knowledge_tasks={"knowledge_a", "knowledge_b"},
                reasoning_tasks={"reasoning_a", "reasoning_b"},
                cv_folds=2,
                permutations=0,
                fdr_permutations=0,
                nonlinear_permutations=0,
                seed=4,
            )

            self.assertEqual(len(rows), 6)
            metric = report["selection_aware_randomization_by_model"]["toy/model"]["metrics"][
                "fdr"
            ]
            self.assertTrue(math.isnan(metric["maximum_randomization_p"]))
            self.assertIsNone(metric["null_maximum_median"])
            self.assertIsNone(metric["null_maximum_q95"])
            self.assertIsNone(metric["minimum_bh_rejections_from_discrete_floor"])
            self.assertIsNone(metric["bh_rejection_possible_at_discrete_floor"])

    def test_analysis_keeps_decode_steps_as_separate_selection_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.pt"
            trajectory = Path(tmpdir) / "trajectory.pt"
            self._artifact(source)
            artifact = torch.load(source, map_location="cpu", weights_only=False)
            records = []
            for record in artifact["records"]:
                for step in (0, 1):
                    updated = dict(record)
                    updated["phase"] = "decode"
                    updated["decode_aggregation"] = "all_updates"
                    updated["decode_step"] = step
                    updated["num_updates"] = 1
                    updated["vector"] = record["vector"] + step
                    records.append(updated)
            artifact["records"] = records
            torch.save(artifact, trajectory)

            rows, report = analyze_artifacts(
                [trajectory],
                knowledge_tasks={"knowledge_a", "knowledge_b"},
                reasoning_tasks={"reasoning_a", "reasoning_b"},
                phase="decode",
                cv_folds=2,
                permutations=5,
                seed=4,
            )

            self.assertEqual(len(rows), 12)
            self.assertEqual({row["trajectory_position"] for row in rows}, {"step_0000", "step_0001"})
            calibration = report["selection_aware_randomization_by_model"]["toy/model"]
            self.assertIsInstance(calibration["metrics"]["fdr"]["observed_bh_rejections"], int)
            self.assertIn(
                report["primary_maximum_divergence"]["trajectory_position"],
                {"step_0000", "step_0001"},
            )

    def test_analysis_accepts_explicit_relative_positions_and_both_phases(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source.pt"
            trajectory = Path(tmpdir) / "trajectory.pt"
            self._artifact(source)
            artifact = torch.load(source, map_location="cpu", weights_only=False)
            records = list(artifact["records"])
            for record in artifact["records"]:
                updated = dict(record)
                updated["phase"] = "decode"
                updated["decode_aggregation"] = "relative_bin_mean"
                updated["decode_step"] = None
                updated["trajectory_position"] = "relative_bin_0000"
                records.append(updated)
            artifact["records"] = records
            torch.save(artifact, trajectory)

            rows, _report = analyze_artifacts(
                [trajectory],
                knowledge_tasks={"knowledge_a", "knowledge_b"},
                reasoning_tasks={"reasoning_a", "reasoning_b"},
                phase="both",
                cv_folds=2,
                permutations=0,
                fdr_permutations=0,
                nonlinear_permutations=0,
                seed=4,
            )

            self.assertEqual(len(rows), 12)
            self.assertEqual({row["phase"] for row in rows}, {"prefill", "decode"})
            self.assertEqual(
                {row["trajectory_position"] for row in rows if row["phase"] == "decode"},
                {"relative_bin_0000"},
            )

    def test_parallel_cuml_backend_requires_mps_before_loading_optional_dependency(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "requires an active NVIDIA MPS"):
                analyze_artifacts(
                    [],
                    knowledge_tasks={"knowledge"},
                    reasoning_tasks={"reasoning"},
                    analysis_backend="cuml",
                    jobs=2,
                )

    def test_parallel_cuml_backend_rejects_threads(self):
        with self.assertRaisesRegex(ValueError, "parallel_backend='process'"):
            analyze_artifacts(
                [],
                knowledge_tasks={"knowledge"},
                reasoning_tasks={"reasoning"},
                analysis_backend="cuml",
                jobs=2,
                parallel_backend="thread",
            )


if __name__ == "__main__":
    unittest.main()
