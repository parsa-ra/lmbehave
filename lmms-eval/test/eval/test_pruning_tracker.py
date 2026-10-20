import torch
import torch.nn as nn

from lmms_eval.pruning import (
    PruningTracker,
    configure_model_for_explicit_attention,
    infer_token_group_context,
    merge_pruning_payloads,
    resolve_pruning_target,
    summarize_pruning_payload,
)
from lmms_eval.pruning.tracker import _take_evenly_spaced_samples


class ToyMLP(nn.Module):
    def forward(self, x):
        return x * 2.0


class ToyAttention(nn.Module):
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        attn = torch.arange(
            batch_size * 2 * seq_len * seq_len,
            dtype=torch.float32,
        ).reshape(batch_size, 2, seq_len, seq_len)
        return x + 1.0, attn


class ToyAttentionWithNone(nn.Module):
    def forward(self, x):
        return x, None


class ToyAttentionLikeInternalNorm(nn.Module):
    def forward(self, x):
        batch_size, seq_len, channels = x.shape
        head_dim = 2
        num_heads = channels // head_dim
        return x.view(batch_size, seq_len, num_heads, head_dim)


class ToyAttentionBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_norm = ToyAttentionLikeInternalNorm()

    def forward(self, x):
        _ = self.q_norm(x)
        batch_size, seq_len, _ = x.shape
        attn = torch.arange(
            batch_size * 2 * seq_len * seq_len,
            dtype=torch.float32,
        ).reshape(batch_size, 2, seq_len, seq_len)
        return x, attn


class ToyLinearAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_k_heads = 2
        self.head_k_dim = 3
        self.key_dim = self.num_k_heads * self.head_k_dim
        self.value_dim = 4
        self.in_proj_qkv = nn.Linear(5, self.key_dim * 2 + self.value_dim, bias=False)

    def forward(self, x):
        _ = self.in_proj_qkv(x)
        return x


class ToyAttentionDenseProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.dense(x)


class ToyFeedForwardDenseProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(4, 4, bias=False)

    def forward(self, x):
        return self.dense(x)


class ToyModel(nn.Module):
    def __init__(self, use_none_attention: bool = False):
        super().__init__()
        self.mlp = ToyMLP()
        self.attn = ToyAttentionWithNone() if use_none_attention else ToyAttention()

    def forward(self, x):
        x = self.mlp(x)
        x, _ = self.attn(x)
        return x


class WrappedToyModel:
    def __init__(self):
        self._model = ToyModel()

    @property
    def model(self):
        return self._model


def test_pruning_tracker_collects_mlp_and_attention_statistics():
    model = ToyModel()
    tracker = PruningTracker(model)
    inputs = torch.tensor(
        [
            [[1.0, -2.0, 3.0], [4.0, -5.0, 6.0]],
            [[-1.0, 2.0, -3.0], [-4.0, 5.0, -6.0]],
        ]
    )

    model(inputs)
    mlp_stats, attn_stats = tracker.aggregate_statistics()

    expected_mlp = (inputs * 2.0).abs().mean(dim=(0, 1))
    expected_attn_tensor = torch.arange(2 * 2 * 2 * 2, dtype=torch.float32).reshape(2, 2, 2, 2).abs()
    expected_attn = expected_attn_tensor.sum(dim=(0, 2, 3)) / torch.count_nonzero(expected_attn_tensor, dim=(0, 2, 3)).clamp_min(1)

    assert torch.allclose(mlp_stats["mlp"], expected_mlp)
    assert torch.allclose(attn_stats["attn"], expected_attn)

    payload = tracker.export_state()
    summary = summarize_pruning_payload(payload, include_module_rows=True)
    assert payload["mlp_hist_counts"]["mlp"].sum().item() == inputs.numel()
    assert payload["mlp_hist_source_elements"]["mlp"] == inputs.numel()
    assert payload["mlp_hist_bin_edges"][0] == 0.0
    assert abs(payload["mlp_hist_bin_edges"][-1] - float((inputs * 2.0).abs().max().item())) < 1e-5
    assert payload["attn_hist_bin_edges"][0] == 0.0
    assert abs(payload["attn_hist_bin_edges"][-1] - float(torch.arange(2 * 2 * 2 * 2, dtype=torch.float32).max().item())) < 1e-5
    assert summary["mlp"]["mean_active_neurons"] == inputs.shape[-1]
    assert summary["mlp"]["mean_active_fraction"] == 1.0
    assert summary["attention"]["histogram"]["num_samples"] == (2 * 2 * 2 * 2) - 1
    assert summary["attention"]["attention_valid_fraction"] == ((2 * 2 * 2 * 2) - 1) / (2 * 2 * 2 * 2)


def test_pruning_tracker_reduces_histogram_samples_and_supports_stride():
    model = ToyModel()
    tracker = PruningTracker(model, update_stride=2, max_histogram_samples=4)
    inputs = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)

    model(inputs)
    model(inputs + 1.0)

    payload = tracker.export_state()
    assert payload["mlp_counts"]["mlp"] == 1
    assert payload["attn_counts"]["attn"] == 1
    assert int(payload["mlp_hist_counts"]["mlp"].sum().item()) == 4
    assert payload["mlp_hist_samples"]["mlp"].numel() == 4
    assert payload["mlp_hist_source_elements"]["mlp"] == inputs.numel()
    assert payload["tracking_config"]["update_stride"] == 2
    assert payload["tracking_config"]["max_histogram_samples"] == 4

    summary = summarize_pruning_payload(payload)
    assert summary["mlp"]["histogram"]["is_sampled"] is True
    assert summary["mlp"]["histogram"]["num_source_values"] == inputs.numel()


def test_histogram_sampler_handles_large_tensors_without_out_of_bounds():
    large_numel = (1 << 24) + 17
    values = torch.zeros(large_numel, dtype=torch.float32)
    values[-1] = 1.0

    sampled = _take_evenly_spaced_samples(values, max_samples=32)

    assert sampled.numel() == 32
    assert sampled[0].item() == 0.0
    assert sampled[-1].item() == 1.0


def test_pruning_tracker_skips_attention_outputs_without_tensor():
    model = ToyModel(use_none_attention=True)
    tracker = PruningTracker(model)
    model(torch.ones(1, 2, 3))

    _, attn_stats = tracker.aggregate_statistics()
    assert attn_stats == {}


def test_pruning_tracker_ignores_zero_masked_attention_entries():
    class ToyCausalAttention(nn.Module):
        def forward(self, x):
            seq_len = x.shape[1]
            attn = torch.tril(torch.ones(1, 1, seq_len, seq_len, dtype=torch.float32))
            attn = attn / attn.sum(dim=-1, keepdim=True)
            return x, attn

    class ToyContainer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = ToyCausalAttention()

        def forward(self, x):
            return self.attn(x)

    model = ToyContainer()
    tracker = PruningTracker(model)
    model(torch.ones(1, 4, 3))

    payload = tracker.export_state()
    summary = summarize_pruning_payload(payload, include_module_rows=True)

    assert torch.allclose(payload["attn_stats"]["attn"], torch.tensor([0.4]))
    assert payload["attn_valid_elements"]["attn"] == 10
    assert payload["attn_total_elements"]["attn"] == 16
    assert summary["attention"]["attention_valid_fraction"] == 10 / 16
    assert summary["attention"]["module_rows"][0]["attention_valid_fraction"] == 10 / 16
    assert summary["attention"]["histogram"]["num_source_values"] == 10


def test_pruning_tracker_collects_token_group_and_attention_routing_statistics():
    class ToyTokenizer:
        pad_token_id = 0
        unk_token_id = 99
        all_special_ids = [0, 1, 2]

        def convert_tokens_to_ids(self, token):
            return {
                "<|image_pad|>": 42,
                "<|vision_start|>": 1,
                "<|vision_end|>": 2,
            }.get(token, self.unk_token_id)

    class ToyConfig:
        image_token_id = 42

    class ToyGroupedAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = ToyConfig()

        def forward(self, x):
            attn = torch.zeros(1, 1, 4, 4, dtype=torch.float32)
            attn[:, :, :, 1:3] = 0.5
            return x * 3.0, attn

    class ToyContainer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = ToyGroupedAttention()

        def forward(self, x):
            return self.attn(x)

    model = ToyContainer()
    tracker = PruningTracker(model)
    input_ids = torch.tensor([[10, 42, 42, 11]])
    context = infer_token_group_context(
        input_ids,
        attention_mask=torch.ones_like(input_ids),
        tokenizer=ToyTokenizer(),
        model=model.attn,
    )

    with tracker.track_context(context):
        model(torch.ones(1, 4, 3))

    payload = tracker.export_state()
    summary = summarize_pruning_payload(payload, include_module_rows=True)

    assert summary["token_groups"]["available"] is True
    assert summary["attention_routing"]["available"] is True
    route_rows = {
        (row["query_group"], row["key_group"]): row
        for row in summary["attention_routing"]["module_rows"]
        if row["module"] == "attn"
    }
    assert route_rows[("text", "vision")]["mean_mass"] == 1.0
    assert route_rows[("vision", "vision")]["mean_mass"] == 1.0


def test_configure_model_for_explicit_attention_keeps_output_attentions_disabled_by_default():
    class ToyConfig:
        def __init__(self, output_attentions=False):
            self.output_attentions = output_attentions
            self._attn_implementation = "sdpa"

    class ToyHFModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = ToyConfig()
            self.generation_config = ToyConfig(output_attentions=True)

    model = ToyHFModule()
    result = configure_model_for_explicit_attention(model)

    assert model.config.output_attentions is False
    assert model.generation_config.output_attentions is False
    assert model.config._attn_implementation == "eager"
    assert model.generation_config._attn_implementation == "eager"
    assert result["configs_seen"] == 2
    assert result["output_attentions_updates"] == 1
    assert result["output_attentions_disabled"] == 1
    assert result["attn_implementation_updates"] == 2
    assert result["force_output_attentions"] is False


def test_configure_model_for_explicit_attention_can_force_output_attentions():
    class ToyConfig:
        def __init__(self):
            self.output_attentions = False
            self._attn_implementation = "sdpa"

    class ToyHFModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = ToyConfig()
            self.generation_config = ToyConfig()

    model = ToyHFModule()
    result = configure_model_for_explicit_attention(model, force_output_attentions=True)

    assert model.config.output_attentions is True
    assert model.generation_config.output_attentions is True
    assert model.config._attn_implementation == "eager"
    assert model.generation_config._attn_implementation == "eager"
    assert result["output_attentions_updates"] == 2
    assert result["output_attentions_disabled"] == 0
    assert result["force_output_attentions"] is True


def test_pruning_tracker_skips_non_square_attention_like_outputs():
    class ToyContainer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn_norm = ToyAttentionLikeInternalNorm()

        def forward(self, x):
            return self.attn_norm(x)

    model = ToyContainer()
    tracker = PruningTracker(model)
    model(torch.ones(1, 5, 8))

    _, attn_stats = tracker.aggregate_statistics()
    assert attn_stats == {}


def test_pruning_tracker_monitors_linear_attention_without_explicit_attention_map():
    class ToyContainer(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear_attn = ToyLinearAttention()

        def forward(self, x):
            return self.linear_attn(x)

    model = ToyContainer()
    tracker = PruningTracker(model)
    model(torch.ones(1, 4, 5))

    payload = tracker.export_state()
    summary = summarize_pruning_payload(payload, include_module_rows=True)
    assert payload["attn_stats"] == {}
    assert "linear_attn" in payload["module_registration_order"]
    assert payload["linear_state_counts"]["linear_attn"] == 1
    assert torch.allclose(payload["linear_state_stats"]["input_output_delta_rms_ratio"]["linear_attn"], torch.tensor(0.0))
    assert torch.allclose(payload["linear_state_stats"]["output_input_cosine"]["linear_attn"], torch.tensor(1.0))
    assert "linear_attn.in_proj_qkv" in payload["linear_state_component_stats"]
    assert payload["linear_state_component_counts"]["linear_attn.in_proj_qkv"] == 1
    assert summary["linear_state"]["available"] is True
    assert summary["linear_state_components"]["available"] is True


def test_pruning_tracker_prefers_attention_block_over_internal_attention_named_submodule():
    class ToyContainer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = ToyAttentionBlock()

        def forward(self, x):
            return self.attn(x)

    model = ToyContainer()
    tracker = PruningTracker(model)
    model(torch.ones(1, 3, 8))

    _, attn_stats = tracker.aggregate_statistics()
    assert "attn" in attn_stats
    assert "attn.q_norm" not in attn_stats


def test_pruning_tracker_does_not_treat_attention_dense_projections_as_mlp_modules():
    class ToyContainer(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = ToyAttentionDenseProjection()
            self.ffn = ToyFeedForwardDenseProjection()

        def forward(self, x):
            x = self.attention(x)
            x = self.ffn(x)
            return x

    model = ToyContainer()
    tracker = PruningTracker(model)
    model(torch.ones(1, 2, 4))

    mlp_stats, _ = tracker.aggregate_statistics()
    assert "attention" not in mlp_stats
    assert "attention.dense" not in mlp_stats
    assert "ffn" in mlp_stats or "ffn.dense" in mlp_stats


def test_pruning_tracker_remove_hooks_stops_future_updates():
    model = ToyModel()
    tracker = PruningTracker(model)
    inputs = torch.ones(1, 2, 3)

    model(inputs)
    first_mlp, _ = tracker.aggregate_statistics()
    tracker.remove_hooks()
    model(inputs)
    second_mlp, _ = tracker.aggregate_statistics()

    assert torch.equal(first_mlp["mlp"], second_mlp["mlp"])


def test_resolve_pruning_target_and_merge_payloads():
    wrapped = WrappedToyModel()
    resolved = resolve_pruning_target(wrapped)

    assert resolved is wrapped._model

    payload_a = {
        "mlp_stats": {"mlp": torch.tensor([1.0, 2.0])},
        "attn_stats": {"attn": torch.tensor([3.0])},
        "mlp_counts": {"mlp": 1},
        "attn_counts": {"attn": 2},
        "mlp_hist_counts": {"mlp": torch.tensor([1.0, 2.0])},
        "attn_hist_counts": {"attn": torch.tensor([3.0, 4.0])},
        "mlp_hist_source_elements": {"mlp": 3},
        "attn_hist_source_elements": {"attn": 7},
        "mlp_hist_bin_edges": [0.0, 1.0, 2.0],
        "attn_hist_bin_edges": [0.0, 0.5, 1.0],
        "mlp_active_elements": {"mlp": 5},
        "mlp_total_elements": {"mlp": 8},
        "mlp_position_counts": {"mlp": 4},
        "mlp_channel_fire_counts": {"mlp": torch.tensor([2.0, 3.0])},
        "mlp_channel_fire_denoms": {"mlp": 4},
        "attn_entropy": {"attn": torch.tensor([1.0])},
        "attn_peak_mass": {"attn": torch.tensor([0.5])},
        "tracking_config": {"mlp_active_threshold_abs": 1e-6},
        "num_hooks": 4,
    }
    payload_b = {
        "mlp_stats": {"mlp": torch.tensor([0.5, 1.5])},
        "attn_stats": {"attn": torch.tensor([2.0])},
        "mlp_counts": {"mlp": 3},
        "attn_counts": {"attn": 1},
        "mlp_hist_counts": {"mlp": torch.tensor([0.0, 1.0])},
        "attn_hist_counts": {"attn": torch.tensor([1.0, 0.0])},
        "mlp_hist_source_elements": {"mlp": 1},
        "attn_hist_source_elements": {"attn": 1},
        "mlp_hist_bin_edges": [0.0, 1.0, 2.0],
        "attn_hist_bin_edges": [0.0, 0.5, 1.0],
        "mlp_active_elements": {"mlp": 1},
        "mlp_total_elements": {"mlp": 4},
        "mlp_position_counts": {"mlp": 2},
        "mlp_channel_fire_counts": {"mlp": torch.tensor([1.0, 0.0])},
        "mlp_channel_fire_denoms": {"mlp": 2},
        "attn_entropy": {"attn": torch.tensor([0.5])},
        "attn_peak_mass": {"attn": torch.tensor([0.25])},
        "tracking_config": {"mlp_active_threshold_abs": 1e-6},
        "num_hooks": 4,
    }

    merged = merge_pruning_payloads([payload_a, payload_b])
    assert torch.allclose(merged["mlp_stats"]["mlp"], torch.tensor([1.5, 3.5]))
    assert torch.allclose(merged["attn_stats"]["attn"], torch.tensor([5.0]))
    assert merged["mlp_counts"]["mlp"] == 4
    assert merged["attn_counts"]["attn"] == 3
    assert torch.allclose(merged["mlp_hist_counts"]["mlp"], torch.tensor([1.0, 3.0]))
    assert merged["mlp_hist_source_elements"]["mlp"] == 4
    assert merged["mlp_active_elements"]["mlp"] == 6
    assert torch.allclose(merged["attn_peak_mass"]["attn"], torch.tensor([0.75]))


def test_summarize_pruning_payload_returns_compact_summary():
    payload = {
        "mlp_stats": {
            "mlp_a": torch.tensor([1.0, 3.0]),
            "mlp_b": torch.tensor([2.0]),
        },
        "attn_stats": {
            "attn_a": torch.tensor([0.5, 1.5]),
        },
        "mlp_counts": {
            "mlp_a": 2,
            "mlp_b": 1,
        },
        "attn_counts": {
            "attn_a": 4,
        },
        "mlp_hist_counts": {
            "mlp_a": torch.tensor([1.0, 2.0]),
            "mlp_b": torch.tensor([0.0, 1.0]),
        },
        "attn_hist_counts": {
            "attn_a": torch.tensor([3.0, 1.0]),
        },
        "mlp_hist_source_elements": {
            "mlp_a": 3,
            "mlp_b": 1,
        },
        "attn_hist_source_elements": {
            "attn_a": 4,
        },
        "mlp_hist_bin_edges": [0.0, 1.0, 2.0],
        "attn_hist_bin_edges": [0.0, 0.5, 1.0],
        "mlp_active_elements": {
            "mlp_a": 4,
            "mlp_b": 2,
        },
        "mlp_total_elements": {
            "mlp_a": 8,
            "mlp_b": 2,
        },
        "mlp_position_counts": {
            "mlp_a": 4,
            "mlp_b": 1,
        },
        "mlp_channel_fire_counts": {
            "mlp_a": torch.tensor([1.0, 3.0]),
            "mlp_b": torch.tensor([1.0]),
        },
        "mlp_channel_fire_denoms": {
            "mlp_a": 4,
            "mlp_b": 1,
        },
        "attn_entropy": {
            "attn_a": torch.tensor([2.0, 4.0]),
        },
        "attn_peak_mass": {
            "attn_a": torch.tensor([1.0, 2.0]),
        },
        "tracking_config": {
            "mlp_active_threshold_abs": 1e-6,
        },
    }

    summary = summarize_pruning_payload(payload, top_k=1, include_module_rows=True)

    assert summary["mlp"]["available"] is True
    assert summary["mlp"]["num_modules"] == 2
    assert summary["mlp"]["num_updates"] == 3
    assert summary["mlp"]["top_modules"][0]["module"] == "mlp_b"
    assert summary["mlp"]["mean_active_neurons"] == 1.2
    assert summary["mlp"]["mean_active_fraction"] == 0.6
    assert round(summary["mlp"]["mean_channel_firing_rate"], 6) == round(2.0 / 3.0, 6)
    assert summary["mlp"]["histogram"]["num_samples"] == 4
    assert summary["mlp"]["histogram"]["num_source_values"] == 4
    assert summary["mlp"]["histogram"]["is_sampled"] is False
    assert summary["attention"]["mean"] == 0.25
    assert summary["attention"]["mean_entropy"] == 0.75
    assert summary["attention"]["mean_peak_mass"] == 0.375
    assert summary["attention"]["module_rows"][0]["updates"] == 4
    assert summary["cross_modal_attention"]["available"] is False


def test_summarize_pruning_payload_builds_network_order_for_mixed_attention_stacks():
    payload = {
        "mlp_stats": {
            "model.layers.0.mlp": torch.tensor([4.0]),
            "model.layers.1.mlp": torch.tensor([6.0]),
        },
        "attn_stats": {
            "model.layers.0.self_attn": torch.tensor([1.0, 3.0]),
        },
        "mlp_counts": {
            "model.layers.0.mlp": 2,
            "model.layers.1.mlp": 2,
        },
        "attn_counts": {
            "model.layers.0.self_attn": 2,
        },
        "linear_state_stats": {
            "input_output_delta_rms_ratio": {
                "model.layers.1.linear_attn": torch.tensor(3.0),
            },
            "output_input_rms_ratio": {
                "model.layers.1.linear_attn": torch.tensor(4.0),
            },
            "output_input_cosine": {
                "model.layers.1.linear_attn": torch.tensor(1.0),
            },
        },
        "linear_state_counts": {
            "model.layers.1.linear_attn": 2,
        },
        "mlp_hist_counts": {},
        "attn_hist_counts": {},
        "mlp_hist_source_elements": {},
        "attn_hist_source_elements": {},
        "mlp_hist_bin_edges": [],
        "attn_hist_bin_edges": [],
        "mlp_active_elements": {
            "model.layers.0.mlp": 6,
            "model.layers.1.mlp": 8,
        },
        "mlp_total_elements": {
            "model.layers.0.mlp": 8,
            "model.layers.1.mlp": 10,
        },
        "mlp_position_counts": {
            "model.layers.0.mlp": 2,
            "model.layers.1.mlp": 2,
        },
        "mlp_channel_fire_counts": {
            "model.layers.0.mlp": torch.tensor([1.0]),
            "model.layers.1.mlp": torch.tensor([2.0]),
        },
        "mlp_channel_fire_denoms": {
            "model.layers.0.mlp": 2,
            "model.layers.1.mlp": 2,
        },
        "attn_entropy": {
            "model.layers.0.self_attn": torch.tensor([1.0, 2.0]),
        },
        "attn_peak_mass": {
            "model.layers.0.self_attn": torch.tensor([0.5, 1.0]),
        },
        "module_registration_order": [
            "model.layers.0.self_attn",
            "model.layers.0.mlp",
            "model.layers.1.linear_attn",
            "model.layers.1.mlp",
        ],
        "tracking_config": {
            "mlp_active_threshold_abs": 1e-6,
        },
    }

    summary = summarize_pruning_payload(payload, include_module_rows=True)
    network = summary["network"]

    assert network["available"] is True
    assert [row["module"] for row in network["module_sequence"]] == [
        "model.layers.0.self_attn",
        "model.layers.0.mlp",
        "model.layers.1.linear_attn",
        "model.layers.1.mlp",
    ]
    assert network["module_sequence"][0]["stage_pattern"] == "model.layers.*.self_attn"
    assert network["module_sequence"][2]["kind"] == "linear_state"
    assert network["module_sequence"][2]["linear_state_input_output_delta_rms_ratio"] == 1.5

    stage_patterns = {row["stage_pattern"]: row for row in network["stage_patterns"]}
    assert stage_patterns["model.layers.*.self_attn"]["count"] == 1
    assert stage_patterns["model.layers.*.self_attn"]["attention_mean"] == 1.0
    assert stage_patterns["model.layers.*.linear_attn"]["linear_state_input_output_delta_rms_ratio"] == 1.5
    assert stage_patterns["model.layers.*.mlp"]["count"] == 2
