from __future__ import annotations

import math
from contextlib import contextmanager
from collections.abc import Iterable
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn


class PruningTracker:
    """Collect forward-pass statistics for pruning via lightweight hooks."""

    MLP_KEYWORDS = ("mlp", "ffn", "dense")
    ATTN_KEYWORDS = ("attention", "attn")
    LINEAR_STATE_KEYWORDS = (
        "linear_attn",
        "linear_attention",
        "gateddeltanet",
        "gated_delta",
        "deltanet",
        "delta_net",
        "retention",
        "retnet",
        "mamba",
        "ssm",
        "rwkv",
    )
    LINEAR_STATE_COMPONENT_NAMES = (
        "in_proj",
        "in_proj_qkv",
        "in_proj_z",
        "in_proj_b",
        "in_proj_a",
        "q_proj",
        "k_proj",
        "v_proj",
        "qkv",
        "to_qkv",
        "c_attn",
        "conv1d",
        "x_proj",
        "dt_proj",
        "out_proj",
        "o_proj",
        "norm",
    )
    LINEAR_STATE_METRIC_NAMES = (
        "input_abs_mean",
        "output_abs_mean",
        "input_output_delta_abs_mean",
        "input_rms",
        "output_rms",
        "input_output_delta_rms",
        "output_input_rms_ratio",
        "input_output_delta_rms_ratio",
        "output_input_cosine",
    )
    ACTIVE_THRESHOLD = 1e-6
    MLP_HIST_NUM_BINS = 15
    ATTN_HIST_NUM_BINS = 20
    LINEAR_STATE_HIST_NUM_BINS = 20

    def __init__(
        self,
        model: nn.Module,
        update_stride: int = 1,
        max_histogram_samples: int = 65536,
    ) -> None:
        if not isinstance(model, nn.Module):
            raise TypeError(f"PruningTracker expects an nn.Module, got {type(model).__name__}")
        if update_stride < 1:
            raise ValueError(f"PruningTracker update_stride must be >= 1, got {update_stride}")
        if max_histogram_samples < 1:
            raise ValueError(f"PruningTracker max_histogram_samples must be >= 1, got {max_histogram_samples}")

        self.model = model
        self.update_stride = int(update_stride)
        self.max_histogram_samples = int(max_histogram_samples)
        self.mlp_stats: dict[str, torch.Tensor] = {}
        self.attn_stats: dict[str, torch.Tensor] = {}
        self.mlp_counts: dict[str, int] = {}
        self.attn_counts: dict[str, int] = {}
        self.linear_state_stats: dict[str, dict[str, torch.Tensor]] = {metric: {} for metric in self.LINEAR_STATE_METRIC_NAMES}
        self.linear_state_counts: dict[str, int] = {}
        self.linear_state_component_stats: dict[str, torch.Tensor] = {}
        self.linear_state_component_counts: dict[str, int] = {}
        self.mlp_hist_samples: dict[str, torch.Tensor] = {}
        self.attn_hist_samples: dict[str, torch.Tensor] = {}
        self.linear_state_output_hist_samples: dict[str, torch.Tensor] = {}
        self.linear_state_input_output_delta_hist_samples: dict[str, torch.Tensor] = {}
        self.linear_state_component_hist_samples: dict[str, torch.Tensor] = {}
        self.mlp_hist_source_elements: dict[str, int] = {}
        self.attn_hist_source_elements: dict[str, int] = {}
        self.linear_state_output_hist_source_elements: dict[str, int] = {}
        self.linear_state_input_output_delta_hist_source_elements: dict[str, int] = {}
        self.linear_state_component_hist_source_elements: dict[str, int] = {}
        self.mlp_active_elements: dict[str, int] = {}
        self.mlp_total_elements: dict[str, int] = {}
        self.mlp_position_counts: dict[str, int] = {}
        self.mlp_channel_fire_counts: dict[str, torch.Tensor] = {}
        self.mlp_channel_fire_denoms: dict[str, int] = {}
        self.linear_state_component_active_elements: dict[str, int] = {}
        self.linear_state_component_total_elements: dict[str, int] = {}
        self.linear_state_component_position_counts: dict[str, int] = {}
        self.linear_state_component_channel_fire_counts: dict[str, torch.Tensor] = {}
        self.linear_state_component_channel_fire_denoms: dict[str, int] = {}
        self.attn_valid_elements: dict[str, int] = {}
        self.attn_total_elements: dict[str, int] = {}
        self.attn_entropy: dict[str, torch.Tensor] = {}
        self.attn_peak_mass: dict[str, torch.Tensor] = {}
        self.token_group_output_abs_sums: dict[str, torch.Tensor] = {}
        self.token_group_output_sq_sums: dict[str, torch.Tensor] = {}
        self.token_group_input_sq_sums: dict[str, torch.Tensor] = {}
        self.token_group_delta_abs_sums: dict[str, torch.Tensor] = {}
        self.token_group_delta_sq_sums: dict[str, torch.Tensor] = {}
        self.token_group_element_counts: dict[str, int] = {}
        self.token_group_token_counts: dict[str, int] = {}
        self.token_group_update_counts: dict[str, int] = {}
        self.token_group_cosine_sums: dict[str, torch.Tensor] = {}
        self.token_group_cosine_counts: dict[str, int] = {}
        self.attn_route_stats: dict[str, torch.Tensor] = {}
        self.attn_route_query_counts: dict[str, int] = {}
        self.attn_route_update_counts: dict[str, int] = {}
        self.linear_state_component_parents: dict[str, str] = {}
        self.linear_state_component_roles: dict[str, str] = {}
        self._context_stack: list[dict[str, Any]] = []
        self._module_seen_counts: dict[str, int] = {}
        self.module_registration_order: list[str] = []
        self.hook_handles: list[torch.utils.hooks.RemovableHandle] = []
        self._register_hooks()

    @staticmethod
    def _module_has_children(module: nn.Module) -> bool:
        return next(module.children(), None) is not None

    @classmethod
    def _should_track_mlp_module(cls, module_name: str, module: nn.Module) -> bool:
        lowered = module_name.lower()
        class_lowered = type(module).__name__.lower()
        basename = lowered.rsplit(".", 1)[-1]
        strong_mlp_hint = any(keyword in lowered for keyword in ("mlp", "ffn")) or any(
            keyword in class_lowered for keyword in ("mlp", "ffn")
        )
        dense_hint = "dense" in lowered or "dense" in class_lowered
        if not strong_mlp_hint and not dense_hint:
            return False

        has_attention_hint = any(keyword in lowered for keyword in cls.ATTN_KEYWORDS) or any(
            keyword in class_lowered for keyword in cls.ATTN_KEYWORDS
        )

        # "dense" is too broad on its own and often appears inside attention output projections.
        dense_only_match = (
            dense_hint
            and not strong_mlp_hint
            and ("dense" in basename or "dense" in class_lowered)
        )
        if dense_only_match and has_attention_hint:
            return False

        return True

    @classmethod
    def _should_track_attention_module(cls, module_name: str, module: nn.Module) -> bool:
        lowered = module_name.lower()
        class_lowered = type(module).__name__.lower()
        basename = lowered.rsplit(".", 1)[-1]

        has_attention_hint = any(keyword in lowered for keyword in cls.ATTN_KEYWORDS) or any(
            keyword in class_lowered for keyword in cls.ATTN_KEYWORDS
        )
        if not has_attention_hint:
            return False

        # Track full attention blocks or modules explicitly named/classed as attention.
        return (
            cls._module_has_children(module)
            or any(keyword in basename for keyword in cls.ATTN_KEYWORDS)
            or any(keyword in class_lowered for keyword in cls.ATTN_KEYWORDS)
        )

    @classmethod
    def _should_track_linear_state_module(cls, module_name: str, module: nn.Module) -> bool:
        lowered = module_name.lower()
        class_lowered = type(module).__name__.lower()
        basename = lowered.rsplit(".", 1)[-1]
        has_linear_state_hint = any(keyword in basename for keyword in cls.LINEAR_STATE_KEYWORDS) or any(
            keyword in class_lowered for keyword in cls.LINEAR_STATE_KEYWORDS
        )
        return has_linear_state_hint and cls._module_has_children(module)

    @classmethod
    def _linear_state_component_role(cls, component_name: str) -> str:
        lowered = component_name.lower()
        if (
            "gate" in lowered
            or lowered.endswith("_z")
            or lowered.endswith("_a")
            or lowered.endswith("_b")
            or lowered in {"z", "a", "b", "dt_proj"}
            or "dt_proj" in lowered
        ):
            return "gate"
        if any(token in lowered for token in ("qkv", "q_proj", "k_proj", "v_proj", "in_proj", "conv", "x_proj")):
            return "write"
        if any(token in lowered for token in ("out", "o_proj", "norm")):
            return "readout"
        return "component"

    @classmethod
    def _should_track_linear_state_component(cls, component_name: str, module: nn.Module) -> bool:
        lowered = component_name.lower()
        return any(component == lowered or component in lowered for component in cls.LINEAR_STATE_COMPONENT_NAMES)

    def _register_hooks(self) -> None:
        for module_name, module in self.model.named_modules():
            if self._should_track_linear_state_module(module_name, module):
                self._record_module_registration(module_name)
                self.hook_handles.append(module.register_forward_hook(self._build_linear_state_hook(module_name)))
                for child_name, child_module in module.named_children():
                    if self._should_track_linear_state_component(child_name, child_module):
                        child_module_name = f"{module_name}.{child_name}" if module_name else child_name
                        self._record_module_registration(child_module_name)
                        self.linear_state_component_parents[child_module_name] = module_name
                        self.linear_state_component_roles[child_module_name] = self._linear_state_component_role(child_name)
                        self.hook_handles.append(
                            child_module.register_forward_hook(
                                self._build_linear_state_component_hook(
                                    parent_name=module_name,
                                    component_name=child_module_name,
                                )
                            )
                        )
            if self._should_track_mlp_module(module_name, module):
                self._record_module_registration(module_name)
                self.hook_handles.append(module.register_forward_hook(self._build_mlp_hook(module_name)))
            if self._should_track_attention_module(module_name, module):
                self._record_module_registration(module_name)
                self.hook_handles.append(module.register_forward_hook(self._build_attention_hook(module_name)))

    def _record_module_registration(self, module_name: str) -> None:
        if module_name and module_name not in self.module_registration_order:
            self.module_registration_order.append(module_name)

    def _build_mlp_hook(self, module_name: str):
        def _hook(module, inputs, output):
            self._track_mlp(module_name, module, inputs, output)

        return _hook

    def _build_attention_hook(self, module_name: str):
        def _hook(module, inputs, output):
            self._track_attention(module_name, module, inputs, output)

        return _hook

    def _build_linear_state_hook(self, module_name: str):
        def _hook(module, inputs, output):
            self._track_linear_state(module_name, module, inputs, output)

        return _hook

    def _build_linear_state_component_hook(self, parent_name: str, component_name: str):
        def _hook(module, inputs, output):
            self._track_linear_state_component(parent_name, component_name, module, inputs, output)

        return _hook

    @staticmethod
    def _iter_tensors(value: Any) -> Iterable[torch.Tensor]:
        if torch.is_tensor(value):
            yield value
            return

        if value is None:
            return

        if isinstance(value, dict):
            for nested in value.values():
                yield from PruningTracker._iter_tensors(nested)
            return

        if hasattr(value, "to_tuple"):
            yield from PruningTracker._iter_tensors(value.to_tuple())
            return

        if isinstance(value, (tuple, list)):
            for nested in value:
                yield from PruningTracker._iter_tensors(nested)

    @classmethod
    def _extract_primary_tensor(cls, output: Any) -> torch.Tensor | None:
        for tensor in cls._iter_tensors(output):
            if tensor.ndim >= 1:
                return tensor
        return None

    @classmethod
    def _is_square_attention_tensor(cls, tensor: torch.Tensor) -> bool:
        return tensor.ndim == 4 and min(tensor.shape) > 0 and tensor.shape[-1] == tensor.shape[-2]

    @classmethod
    def _extract_attention_tensor(cls, output: Any) -> torch.Tensor | None:
        if output is None:
            return None

        if hasattr(output, "to_tuple"):
            output = output.to_tuple()

        # Follow the common HF contract first: attention weights usually live at output[1].
        if isinstance(output, (tuple, list)):
            if len(output) > 1:
                candidate = output[1]
                if torch.is_tensor(candidate) and cls._is_square_attention_tensor(candidate):
                    return candidate
            for candidate in output[1:]:
                if torch.is_tensor(candidate) and cls._is_square_attention_tensor(candidate):
                    return candidate
                if hasattr(candidate, "to_tuple"):
                    nested = cls._extract_attention_tensor(candidate.to_tuple())
                    if nested is not None:
                        return nested
                if isinstance(candidate, (tuple, list, dict)):
                    nested = cls._extract_attention_tensor(candidate)
                    if nested is not None:
                        return nested
            return None

        if isinstance(output, dict):
            for key in ("attentions", "attention", "attn_weights", "attn_probs"):
                candidate = output.get(key)
                if torch.is_tensor(candidate) and cls._is_square_attention_tensor(candidate):
                    return candidate
                if hasattr(candidate, "to_tuple") or isinstance(candidate, (tuple, list, dict)):
                    nested = cls._extract_attention_tensor(candidate)
                    if nested is not None:
                        return nested
            return None

        return None

    @staticmethod
    def _accumulate_tensor(target: dict[str, torch.Tensor], module_name: str, value: torch.Tensor) -> bool:
        if module_name in target:
            if target[module_name].shape != value.shape:
                return False
            target[module_name] = target[module_name] + value
        else:
            target[module_name] = value
        return True

    @classmethod
    def _accumulate(cls, target: dict[str, torch.Tensor], counts: dict[str, int], module_name: str, value: torch.Tensor) -> None:
        if not cls._accumulate_tensor(target, module_name, value):
            return
        counts[module_name] = counts.get(module_name, 0) + 1

    @staticmethod
    def _accumulate_int(target: dict[str, int], module_name: str, value: int) -> None:
        target[module_name] = target.get(module_name, 0) + int(value)

    @staticmethod
    def _rms(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.numel() == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=tensor.device)
        return torch.sqrt(torch.mean(tensor.float() * tensor.float()).clamp_min(0.0))

    @staticmethod
    def _cosine_similarity(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        left_flat = left.float().reshape(-1)
        right_flat = right.float().reshape(-1)
        if left_flat.numel() == 0 or right_flat.numel() == 0 or left_flat.numel() != right_flat.numel():
            return torch.tensor(0.0, dtype=torch.float32, device=left.device)
        denom = left_flat.norm() * right_flat.norm()
        if denom <= 0:
            return torch.tensor(0.0, dtype=torch.float32, device=left.device)
        return torch.dot(left_flat, right_flat) / denom.clamp_min(1e-12)

    @staticmethod
    def _channel_reduce_dims(tensor: torch.Tensor, module: nn.Module) -> tuple[int, tuple[int, ...]]:
        if isinstance(module, nn.Conv1d) and tensor.ndim >= 2:
            channel_dim = 1
        else:
            channel_dim = tensor.ndim - 1
        reduce_dims = tuple(dim for dim in range(tensor.ndim) if dim != channel_dim)
        return channel_dim, reduce_dims

    def _should_update_module(self, module_name: str, stream: str) -> bool:
        seen_key = f"{stream}:{module_name}"
        seen_count = self._module_seen_counts.get(seen_key, 0) + 1
        self._module_seen_counts[seen_key] = seen_count
        return (seen_count - 1) % self.update_stride == 0

    @contextmanager
    def track_context(self, context: dict[str, Any] | None):
        if not context:
            yield
            return
        self._context_stack.append(context)
        try:
            yield
        finally:
            self._context_stack.pop()

    def set_context(self, context: dict[str, Any] | None) -> None:
        self._context_stack = [context] if context else []

    def clear_context(self) -> None:
        self._context_stack.clear()

    def _current_context(self) -> dict[str, Any] | None:
        if not self._context_stack:
            return None
        return self._context_stack[-1]

    @staticmethod
    def _behavior_key(stream: str, module_name: str, phase: str, group_name: str) -> str:
        return "\t".join((str(stream), str(module_name), str(phase), str(group_name)))

    @staticmethod
    def _route_key(module_name: str, phase: str, query_group: str, key_group: str) -> str:
        return "\t".join((str(module_name), str(phase), str(query_group), str(key_group)))

    @staticmethod
    def _split_behavior_key(key: str) -> tuple[str, str, str, str]:
        parts = str(key).split("\t", 3)
        if len(parts) == 4:
            return parts[0], parts[1], parts[2], parts[3]
        return "unknown", str(key), "unknown", "unknown"

    @staticmethod
    def _split_route_key(key: str) -> tuple[str, str, str, str]:
        parts = str(key).split("\t", 3)
        if len(parts) == 4:
            return parts[0], parts[1], parts[2], parts[3]
        return str(key), "unknown", "unknown", "unknown"

    @staticmethod
    def _as_bool_mask(mask: Any, device: torch.device) -> torch.Tensor | None:
        if mask is None:
            return None
        if torch.is_tensor(mask):
            return mask.detach().to(device=device, dtype=torch.bool)
        try:
            return torch.as_tensor(mask, device=device, dtype=torch.bool)
        except Exception:
            return None

    @staticmethod
    def _broadcast_mask(mask: torch.Tensor, batch_size: int, seq_len: int) -> torch.Tensor | None:
        if mask.ndim == 1:
            if mask.numel() != seq_len:
                return None
            mask = mask.unsqueeze(0)
        elif mask.ndim != 2:
            return None

        if mask.shape[-1] != seq_len:
            return None
        if mask.shape[0] == batch_size:
            return mask
        if mask.shape[0] == 1:
            return mask.expand(batch_size, seq_len)
        return None

    def _token_group_masks_for_tensor(
        self,
        tensor: torch.Tensor,
        channel_dim: int,
    ) -> tuple[str, dict[str, torch.Tensor], int] | None:
        context = self._current_context()
        if not context or tensor.ndim < 2:
            return None

        normalized_channel_dim = channel_dim if channel_dim >= 0 else tensor.ndim + channel_dim
        candidate_seq_dims = [
            dim
            for dim in range(tensor.ndim)
            if dim != normalized_channel_dim and not (tensor.ndim >= 3 and dim == 0)
        ]
        if not candidate_seq_dims:
            return None

        expected_seq_len = int(context.get("sequence_length") or 0)
        seq_dim = None
        if expected_seq_len > 0:
            for dim in candidate_seq_dims:
                if int(tensor.shape[dim]) == expected_seq_len:
                    seq_dim = dim
                    break
        if seq_dim is None:
            fallback_dim = tensor.ndim - 2 if normalized_channel_dim == tensor.ndim - 1 and tensor.ndim >= 2 else candidate_seq_dims[-1]
            if fallback_dim in candidate_seq_dims:
                seq_dim = fallback_dim
        if seq_dim is None:
            return None

        seq_len = int(tensor.shape[seq_dim])
        batch_dim = 0 if tensor.ndim >= 3 and seq_dim != 0 else None
        batch_size = int(tensor.shape[batch_dim]) if batch_dim is not None else 1
        device = tensor.device

        token_groups = context.get("token_groups") or {}
        masks: dict[str, torch.Tensor] = {}
        phase = "prefill" if expected_seq_len > 0 and seq_len == expected_seq_len else "unknown"

        if expected_seq_len > 0 and seq_len == expected_seq_len:
            for group_name, raw_mask in token_groups.items():
                mask = self._as_bool_mask(raw_mask, device)
                if mask is None:
                    continue
                mask = self._broadcast_mask(mask, batch_size=batch_size, seq_len=seq_len)
                if mask is not None and bool(mask.any().item()):
                    masks[str(group_name)] = mask
        elif seq_len == 1 and expected_seq_len != 1:
            phase = "decode"
            masks["generated"] = torch.ones((batch_size, seq_len), dtype=torch.bool, device=device)

        if not masks:
            return None
        return phase, masks, seq_dim

    @staticmethod
    def _select_token_group(tensor: torch.Tensor, mask: torch.Tensor, seq_dim: int, channel_dim: int) -> torch.Tensor:
        if tensor.ndim == 2:
            if seq_dim == 0:
                return tensor[mask.reshape(-1)]
            if seq_dim == 1:
                return tensor[:, mask.reshape(-1)].transpose(0, 1)

        if tensor.ndim >= 3 and seq_dim != 0:
            dim_order = [0, seq_dim] + [dim for dim in range(tensor.ndim) if dim not in {0, seq_dim}]
            moved = tensor.permute(dim_order).contiguous().reshape(tensor.shape[0], tensor.shape[seq_dim], -1)
            return moved[mask]

        return tensor.reshape(-1, tensor.shape[-1])

    def _track_token_group_behavior(
        self,
        stream: str,
        module_name: str,
        inputs: Any,
        output_tensor: torch.Tensor,
        channel_dim: int = -1,
    ) -> None:
        context_result = self._token_group_masks_for_tensor(output_tensor, channel_dim=channel_dim)
        if context_result is None:
            return
        phase, masks, seq_dim = context_result

        input_tensor = self._extract_primary_tensor(inputs)
        has_matching_input = input_tensor is not None and input_tensor.shape == output_tensor.shape
        input_tensor = input_tensor.detach().float() if has_matching_input else None
        output_tensor = output_tensor.detach().float()

        for group_name, mask in masks.items():
            selected_output = self._select_token_group(output_tensor, mask, seq_dim=seq_dim, channel_dim=channel_dim)
            if selected_output.numel() == 0:
                continue
            key = self._behavior_key(stream, module_name, phase, group_name)
            selected_output = selected_output.float()
            output_abs = selected_output.abs()
            self._accumulate_tensor(
                self.token_group_output_abs_sums,
                key,
                output_abs.sum().detach().reshape(()).cpu(),
            )
            self._accumulate_tensor(
                self.token_group_output_sq_sums,
                key,
                (selected_output * selected_output).sum().detach().reshape(()).cpu(),
            )
            self._accumulate_int(self.token_group_element_counts, key, int(selected_output.numel()))
            token_count = int(mask.sum().item())
            self._accumulate_int(self.token_group_token_counts, key, token_count)
            self._accumulate_int(self.token_group_update_counts, key, 1)

            if input_tensor is None:
                continue

            selected_input = self._select_token_group(input_tensor, mask, seq_dim=seq_dim, channel_dim=channel_dim).float()
            if selected_input.shape != selected_output.shape or selected_input.numel() == 0:
                continue
            delta = selected_output - selected_input
            self._accumulate_tensor(
                self.token_group_input_sq_sums,
                key,
                (selected_input * selected_input).sum().detach().reshape(()).cpu(),
            )
            self._accumulate_tensor(
                self.token_group_delta_abs_sums,
                key,
                delta.abs().sum().detach().reshape(()).cpu(),
            )
            self._accumulate_tensor(
                self.token_group_delta_sq_sums,
                key,
                (delta * delta).sum().detach().reshape(()).cpu(),
            )
            cosine = self._cosine_similarity(selected_input, selected_output)
            self._accumulate_tensor(self.token_group_cosine_sums, key, cosine.detach().reshape(()).cpu())
            self._accumulate_int(self.token_group_cosine_counts, key, 1)

    def _track_attention_routing(self, module_name: str, attn_tensor: torch.Tensor) -> None:
        context_result = self._token_group_masks_for_tensor(attn_tensor, channel_dim=1)
        if context_result is None:
            return
        phase, masks, _ = context_result
        if attn_tensor.ndim != 4:
            return

        attn = attn_tensor.detach().float().abs()
        denom = attn.sum(dim=-1, keepdim=True)
        normalized = torch.where(denom > 0, attn / denom.clamp_min(1e-12), torch.zeros_like(attn))
        batch_size = int(normalized.shape[0])
        num_heads = int(normalized.shape[1])

        for query_group, query_mask in masks.items():
            query_mask = self._broadcast_mask(query_mask, batch_size=batch_size, seq_len=int(normalized.shape[2]))
            if query_mask is None:
                continue
            for key_group, key_mask in masks.items():
                key_mask = self._broadcast_mask(key_mask, batch_size=batch_size, seq_len=int(normalized.shape[3]))
                if key_mask is None:
                    continue
                route_sum = torch.zeros(num_heads, dtype=torch.float32, device=normalized.device)
                query_count = 0
                for batch_index in range(batch_size):
                    q_mask = query_mask[batch_index]
                    k_mask = key_mask[batch_index]
                    if not bool(q_mask.any().item()) or not bool(k_mask.any().item()):
                        continue
                    rows = normalized[batch_index, :, q_mask, :]
                    route_sum = route_sum + rows[:, :, k_mask].sum(dim=-1).sum(dim=-1)
                    query_count += int(q_mask.sum().item())
                if query_count <= 0:
                    continue
                key = self._route_key(module_name, phase, query_group, key_group)
                self._accumulate_tensor(self.attn_route_stats, key, route_sum.detach().cpu())
                self._accumulate_int(self.attn_route_query_counts, key, query_count)
                self._accumulate_int(self.attn_route_update_counts, key, 1)

    def _histogram_samples(self, tensor: torch.Tensor, ignore_zeros: bool = False) -> tuple[torch.Tensor, int]:
        values = tensor.detach().float().reshape(-1)
        source_numel = int(torch.count_nonzero(values).item()) if ignore_zeros else int(values.numel())
        if values.numel() == 0:
            return torch.zeros(0, dtype=torch.float32), source_numel
        if ignore_zeros and source_numel == 0:
            return torch.zeros(0, dtype=torch.float32), source_numel

        sample_budget = self.max_histogram_samples
        if ignore_zeros:
            sample_budget = min(int(values.numel()), max(self.max_histogram_samples, self.max_histogram_samples * 4))
        values = _take_evenly_spaced_samples(values, sample_budget)
        if ignore_zeros:
            values = values[values > 0]
            values = _take_evenly_spaced_samples(values, self.max_histogram_samples)
        return values.detach().float().cpu(), source_numel

    def _update_histogram_samples(self, target: dict[str, torch.Tensor], module_name: str, values: torch.Tensor) -> None:
        if values.numel() == 0:
            return

        current = target.get(module_name)
        if current is None:
            target[module_name] = _take_evenly_spaced_samples(values.detach().float().cpu(), self.max_histogram_samples)
            return

        combined = torch.cat([current.detach().float().cpu(), values.detach().float().cpu()])
        target[module_name] = _take_evenly_spaced_samples(combined, self.max_histogram_samples)

    def _track_mlp(self, module_name: str, module, inputs, output) -> None:
        with torch.no_grad():
            if not self._should_update_module(module_name, "mlp"):
                return

            tensor = self._extract_primary_tensor(output)
            if tensor is None:
                return

            tensor = tensor.detach()
            if tensor.ndim == 0:
                return

            tensor = tensor.float()
            reduce_dims = tuple(range(tensor.ndim - 1))
            abs_tensor = tensor.abs()
            stats = abs_tensor
            if reduce_dims:
                stats = stats.mean(dim=reduce_dims)
            self._accumulate(self.mlp_stats, self.mlp_counts, module_name, stats.cpu())

            hist_samples, hist_source_numel = self._histogram_samples(abs_tensor)
            self._update_histogram_samples(self.mlp_hist_samples, module_name, hist_samples)
            self._accumulate_int(self.mlp_hist_source_elements, module_name, hist_source_numel)

            active_mask = abs_tensor > self.ACTIVE_THRESHOLD
            positions = int(active_mask.numel() / active_mask.shape[-1]) if active_mask.ndim >= 1 else 1
            self._accumulate_int(self.mlp_active_elements, module_name, int(active_mask.sum().item()))
            self._accumulate_int(self.mlp_total_elements, module_name, int(active_mask.numel()))
            self._accumulate_int(self.mlp_position_counts, module_name, positions)

            channel_fire_counts = active_mask.float()
            if reduce_dims:
                channel_fire_counts = channel_fire_counts.sum(dim=reduce_dims)
            if self._accumulate_tensor(self.mlp_channel_fire_counts, module_name, channel_fire_counts.cpu()):
                self._accumulate_int(self.mlp_channel_fire_denoms, module_name, positions)
            self._track_token_group_behavior("mlp", module_name, inputs, tensor, channel_dim=-1)

    def _track_attention(self, module_name: str, module, inputs, output) -> None:
        with torch.no_grad():
            if not self._should_update_module(module_name, "attention"):
                return

            attn_tensor = self._extract_attention_tensor(output)
            if attn_tensor is None:
                return

            attn_tensor = attn_tensor.detach()
            if not self._is_square_attention_tensor(attn_tensor):
                return

            attn_tensor = attn_tensor.float().abs()
            primary_output = self._extract_primary_tensor(output)
            if primary_output is not None and primary_output.ndim >= 2:
                self._track_token_group_behavior("attention", module_name, inputs, primary_output, channel_dim=-1)
            valid_counts = torch.count_nonzero(attn_tensor, dim=(0, 2, 3))
            stats = attn_tensor.sum(dim=(0, 2, 3)) / valid_counts.clamp_min(1).to(attn_tensor.dtype)
            self._accumulate(self.attn_stats, self.attn_counts, module_name, stats.cpu())
            self._accumulate_int(self.attn_valid_elements, module_name, int(valid_counts.sum().item()))
            self._accumulate_int(self.attn_total_elements, module_name, int(attn_tensor.numel()))

            hist_samples, hist_source_numel = self._histogram_samples(attn_tensor, ignore_zeros=True)
            self._update_histogram_samples(self.attn_hist_samples, module_name, hist_samples)
            self._accumulate_int(self.attn_hist_source_elements, module_name, hist_source_numel)

            attn_denom = attn_tensor.sum(dim=-1, keepdim=True)
            normalized_attn = torch.where(attn_denom > 0, attn_tensor / attn_denom.clamp_min(1e-12), torch.zeros_like(attn_tensor))
            entropy = -(normalized_attn.clamp_min(1e-12) * normalized_attn.clamp_min(1e-12).log()).sum(dim=-1).mean(dim=(0, 2))
            peak_mass = normalized_attn.max(dim=-1).values.mean(dim=(0, 2))
            self._accumulate_tensor(self.attn_entropy, module_name, entropy.cpu())
            self._accumulate_tensor(self.attn_peak_mass, module_name, peak_mass.cpu())
            self._track_attention_routing(module_name, attn_tensor)

    def _track_linear_state(self, module_name: str, module, inputs, output) -> None:
        with torch.no_grad():
            if not self._should_update_module(module_name, "linear_state"):
                return

            output_tensor = self._extract_primary_tensor(output)
            if output_tensor is None or output_tensor.ndim == 0:
                return

            output_tensor = output_tensor.detach().float()
            input_tensor = self._extract_primary_tensor(inputs)
            input_tensor = input_tensor.detach().float() if input_tensor is not None and input_tensor.ndim > 0 else None

            metrics: dict[str, torch.Tensor] = {
                "output_abs_mean": output_tensor.abs().mean(),
                "output_rms": self._rms(output_tensor),
            }

            hist_samples, hist_source_numel = self._histogram_samples(output_tensor.abs())
            self._update_histogram_samples(self.linear_state_output_hist_samples, module_name, hist_samples)
            self._accumulate_int(self.linear_state_output_hist_source_elements, module_name, hist_source_numel)

            if input_tensor is not None:
                metrics["input_abs_mean"] = input_tensor.abs().mean()
                input_rms = self._rms(input_tensor)
                metrics["input_rms"] = input_rms
                metrics["output_input_rms_ratio"] = metrics["output_rms"] / input_rms.clamp_min(1e-12)

                if input_tensor.shape == output_tensor.shape:
                    delta = output_tensor - input_tensor
                    metrics["input_output_delta_abs_mean"] = delta.abs().mean()
                    input_output_delta_rms = self._rms(delta)
                    metrics["input_output_delta_rms"] = input_output_delta_rms
                    metrics["input_output_delta_rms_ratio"] = input_output_delta_rms / input_rms.clamp_min(1e-12)
                    metrics["output_input_cosine"] = self._cosine_similarity(input_tensor, output_tensor)

                    delta_hist_samples, delta_hist_source_numel = self._histogram_samples(delta.abs())
                    self._update_histogram_samples(self.linear_state_input_output_delta_hist_samples, module_name, delta_hist_samples)
                    self._accumulate_int(self.linear_state_input_output_delta_hist_source_elements, module_name, delta_hist_source_numel)

            for metric_name, value in metrics.items():
                self._accumulate_tensor(self.linear_state_stats[metric_name], module_name, value.detach().reshape(()).cpu())
            self._accumulate_int(self.linear_state_counts, module_name, 1)
            self._track_token_group_behavior("linear_state", module_name, inputs, output_tensor, channel_dim=-1)

    def _track_linear_state_component(self, parent_name: str, component_name: str, module, inputs, output) -> None:
        with torch.no_grad():
            if not self._should_update_module(component_name, "linear_state_component"):
                return

            tensor = self._extract_primary_tensor(output)
            if tensor is None or tensor.ndim == 0:
                return

            tensor = tensor.detach().float()
            abs_tensor = tensor.abs()
            channel_dim, reduce_dims = self._channel_reduce_dims(tensor, module)
            stats = abs_tensor.mean(dim=reduce_dims) if reduce_dims else abs_tensor
            self._accumulate(self.linear_state_component_stats, self.linear_state_component_counts, component_name, stats.cpu())
            self.linear_state_component_parents.setdefault(component_name, parent_name)

            hist_samples, hist_source_numel = self._histogram_samples(abs_tensor)
            self._update_histogram_samples(self.linear_state_component_hist_samples, component_name, hist_samples)
            self._accumulate_int(self.linear_state_component_hist_source_elements, component_name, hist_source_numel)

            active_mask = abs_tensor > self.ACTIVE_THRESHOLD
            channel_size = int(active_mask.shape[channel_dim]) if active_mask.ndim > 0 else 1
            positions = int(active_mask.numel() / max(channel_size, 1))
            self._accumulate_int(self.linear_state_component_active_elements, component_name, int(active_mask.sum().item()))
            self._accumulate_int(self.linear_state_component_total_elements, component_name, int(active_mask.numel()))
            self._accumulate_int(self.linear_state_component_position_counts, component_name, positions)

            channel_fire_counts = active_mask.float().sum(dim=reduce_dims) if reduce_dims else active_mask.float()
            if self._accumulate_tensor(self.linear_state_component_channel_fire_counts, component_name, channel_fire_counts.cpu()):
                self._accumulate_int(self.linear_state_component_channel_fire_denoms, component_name, positions)
            self._track_token_group_behavior(
                "linear_state_component",
                component_name,
                inputs,
                tensor,
                channel_dim=channel_dim,
            )

    def remove_hooks(self) -> None:
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def aggregate_statistics(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        return (
            {name: tensor.clone() for name, tensor in self.mlp_stats.items()},
            {name: tensor.clone() for name, tensor in self.attn_stats.items()},
        )

    def export_state(self) -> dict[str, Any]:
        mlp_stats, attn_stats = self.aggregate_statistics()
        mlp_hist_bin_edges = _build_histogram_bin_edges(
            self.mlp_hist_samples,
            num_bins=self.MLP_HIST_NUM_BINS,
            scale="log",
        )
        attn_hist_bin_edges = _build_histogram_bin_edges(
            self.attn_hist_samples,
            num_bins=self.ATTN_HIST_NUM_BINS,
            scale="linear",
        )
        linear_state_output_hist_bin_edges = _build_histogram_bin_edges(
            self.linear_state_output_hist_samples,
            num_bins=self.LINEAR_STATE_HIST_NUM_BINS,
            scale="log",
        )
        linear_state_input_output_delta_hist_bin_edges = _build_histogram_bin_edges(
            self.linear_state_input_output_delta_hist_samples,
            num_bins=self.LINEAR_STATE_HIST_NUM_BINS,
            scale="log",
        )
        linear_state_component_hist_bin_edges = _build_histogram_bin_edges(
            self.linear_state_component_hist_samples,
            num_bins=self.LINEAR_STATE_HIST_NUM_BINS,
            scale="log",
        )
        return {
            "mlp_stats": mlp_stats,
            "attn_stats": attn_stats,
            "mlp_counts": dict(self.mlp_counts),
            "attn_counts": dict(self.attn_counts),
            "linear_state_stats": {
                metric_name: {module_name: tensor.clone() for module_name, tensor in stats.items()}
                for metric_name, stats in self.linear_state_stats.items()
            },
            "linear_state_counts": dict(self.linear_state_counts),
            "linear_state_component_stats": {name: tensor.clone() for name, tensor in self.linear_state_component_stats.items()},
            "linear_state_component_counts": dict(self.linear_state_component_counts),
            "mlp_hist_samples": {name: tensor.clone() for name, tensor in self.mlp_hist_samples.items()},
            "attn_hist_samples": {name: tensor.clone() for name, tensor in self.attn_hist_samples.items()},
            "linear_state_output_hist_samples": {name: tensor.clone() for name, tensor in self.linear_state_output_hist_samples.items()},
            "linear_state_input_output_delta_hist_samples": {name: tensor.clone() for name, tensor in self.linear_state_input_output_delta_hist_samples.items()},
            "linear_state_component_hist_samples": {name: tensor.clone() for name, tensor in self.linear_state_component_hist_samples.items()},
            "mlp_hist_counts": _histogram_counts_from_sample_dict(self.mlp_hist_samples, mlp_hist_bin_edges),
            "attn_hist_counts": _histogram_counts_from_sample_dict(self.attn_hist_samples, attn_hist_bin_edges),
            "linear_state_output_hist_counts": _histogram_counts_from_sample_dict(self.linear_state_output_hist_samples, linear_state_output_hist_bin_edges),
            "linear_state_input_output_delta_hist_counts": _histogram_counts_from_sample_dict(self.linear_state_input_output_delta_hist_samples, linear_state_input_output_delta_hist_bin_edges),
            "linear_state_component_hist_counts": _histogram_counts_from_sample_dict(self.linear_state_component_hist_samples, linear_state_component_hist_bin_edges),
            "mlp_hist_source_elements": dict(self.mlp_hist_source_elements),
            "attn_hist_source_elements": dict(self.attn_hist_source_elements),
            "linear_state_output_hist_source_elements": dict(self.linear_state_output_hist_source_elements),
            "linear_state_input_output_delta_hist_source_elements": dict(self.linear_state_input_output_delta_hist_source_elements),
            "linear_state_component_hist_source_elements": dict(self.linear_state_component_hist_source_elements),
            "mlp_hist_bin_edges": mlp_hist_bin_edges,
            "attn_hist_bin_edges": attn_hist_bin_edges,
            "linear_state_output_hist_bin_edges": linear_state_output_hist_bin_edges,
            "linear_state_input_output_delta_hist_bin_edges": linear_state_input_output_delta_hist_bin_edges,
            "linear_state_component_hist_bin_edges": linear_state_component_hist_bin_edges,
            "mlp_active_elements": dict(self.mlp_active_elements),
            "mlp_total_elements": dict(self.mlp_total_elements),
            "mlp_position_counts": dict(self.mlp_position_counts),
            "mlp_channel_fire_counts": {name: tensor.clone() for name, tensor in self.mlp_channel_fire_counts.items()},
            "mlp_channel_fire_denoms": dict(self.mlp_channel_fire_denoms),
            "linear_state_component_active_elements": dict(self.linear_state_component_active_elements),
            "linear_state_component_total_elements": dict(self.linear_state_component_total_elements),
            "linear_state_component_position_counts": dict(self.linear_state_component_position_counts),
            "linear_state_component_channel_fire_counts": {name: tensor.clone() for name, tensor in self.linear_state_component_channel_fire_counts.items()},
            "linear_state_component_channel_fire_denoms": dict(self.linear_state_component_channel_fire_denoms),
            "attn_valid_elements": dict(self.attn_valid_elements),
            "attn_total_elements": dict(self.attn_total_elements),
            "attn_entropy": {name: tensor.clone() for name, tensor in self.attn_entropy.items()},
            "attn_peak_mass": {name: tensor.clone() for name, tensor in self.attn_peak_mass.items()},
            "token_group_output_abs_sums": {name: tensor.clone() for name, tensor in self.token_group_output_abs_sums.items()},
            "token_group_output_sq_sums": {name: tensor.clone() for name, tensor in self.token_group_output_sq_sums.items()},
            "token_group_input_sq_sums": {name: tensor.clone() for name, tensor in self.token_group_input_sq_sums.items()},
            "token_group_delta_abs_sums": {name: tensor.clone() for name, tensor in self.token_group_delta_abs_sums.items()},
            "token_group_delta_sq_sums": {name: tensor.clone() for name, tensor in self.token_group_delta_sq_sums.items()},
            "token_group_element_counts": dict(self.token_group_element_counts),
            "token_group_token_counts": dict(self.token_group_token_counts),
            "token_group_update_counts": dict(self.token_group_update_counts),
            "token_group_cosine_sums": {name: tensor.clone() for name, tensor in self.token_group_cosine_sums.items()},
            "token_group_cosine_counts": dict(self.token_group_cosine_counts),
            "attn_route_stats": {name: tensor.clone() for name, tensor in self.attn_route_stats.items()},
            "attn_route_query_counts": dict(self.attn_route_query_counts),
            "attn_route_update_counts": dict(self.attn_route_update_counts),
            "linear_state_component_parents": dict(self.linear_state_component_parents),
            "linear_state_component_roles": dict(self.linear_state_component_roles),
            "module_registration_order": list(self.module_registration_order),
            "tracking_config": {
                "mlp_active_threshold_abs": self.ACTIVE_THRESHOLD,
                "update_stride": self.update_stride,
                "max_histogram_samples": self.max_histogram_samples,
                "mlp_hist_num_bins": self.MLP_HIST_NUM_BINS,
                "attn_hist_num_bins": self.ATTN_HIST_NUM_BINS,
                "linear_state_hist_num_bins": self.LINEAR_STATE_HIST_NUM_BINS,
                "mlp_histogram_binning": "dynamic_log",
                "attention_histogram_binning": "dynamic_linear",
                "linear_state_histogram_binning": "dynamic_log",
                "attention_histogram_ignores_zero_masked_entries": True,
                "attention_stats_ignore_zero_masked_entries": True,
                "token_group_tracking": True,
                "attention_routing_tracking": True,
            },
            "num_hooks": len(self.hook_handles),
        }


def resolve_pruning_target(model_or_wrapper: Any) -> nn.Module | None:
    if isinstance(model_or_wrapper, nn.Module):
        return model_or_wrapper

    for attr_name in ("model", "_model"):
        try:
            candidate = getattr(model_or_wrapper, attr_name)
        except Exception:
            continue
        if isinstance(candidate, nn.Module):
            return candidate

    accelerator = getattr(model_or_wrapper, "accelerator", None)
    wrapped_model = getattr(model_or_wrapper, "_model", None)
    unwrap_model = getattr(accelerator, "unwrap_model", None)
    if callable(unwrap_model) and isinstance(wrapped_model, nn.Module):
        try:
            candidate = unwrap_model(wrapped_model)
        except Exception:
            candidate = None
        if isinstance(candidate, nn.Module):
            return candidate

    for value in vars(model_or_wrapper).values():
        if isinstance(value, nn.Module):
            return value

    return None


def _iter_model_configs(model: Any) -> Iterable[Any]:
    seen: set[int] = set()
    for candidate in (model, getattr(model, "module", None)):
        if candidate is None:
            continue
        for attr_name in ("config", "generation_config"):
            config = getattr(candidate, attr_name, None)
            if config is not None and id(config) not in seen:
                seen.add(id(config))
                yield config
                for nested_attr in ("text_config", "vision_config", "vision_config_dict"):
                    nested_config = getattr(config, nested_attr, None)
                    if nested_config is not None and id(nested_config) not in seen:
                        seen.add(id(nested_config))
                        yield nested_config


def _collect_token_ids_from_attrs(objects: Iterable[Any], attr_names: Iterable[str]) -> set[int]:
    token_ids: set[int] = set()
    for obj in objects:
        for attr_name in attr_names:
            value = getattr(obj, attr_name, None)
            if value is None:
                continue
            if isinstance(value, int):
                token_ids.add(int(value))
            elif isinstance(value, (list, tuple, set)):
                for nested in value:
                    if isinstance(nested, int):
                        token_ids.add(int(nested))
    return token_ids


def _collect_token_ids_from_strings(tokenizer: Any, token_strings: Iterable[str]) -> set[int]:
    token_ids: set[int] = set()
    convert = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(convert):
        return token_ids
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for token in token_strings:
        try:
            token_id = convert(token)
        except Exception:
            continue
        if isinstance(token_id, int) and token_id >= 0 and token_id != unk_id:
            token_ids.add(int(token_id))
    return token_ids


def _ids_mask(input_ids: torch.Tensor, token_ids: set[int]) -> torch.Tensor:
    mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for token_id in token_ids:
        mask |= input_ids == int(token_id)
    return mask


def infer_token_group_context(
    input_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None = None,
    tokenizer: Any | None = None,
    model: Any | None = None,
    task_name: str | None = None,
    category: str | None = None,
) -> dict[str, Any] | None:
    """Build best-effort token-group masks for behavior tracking.

    The returned masks are intentionally coarse and model-agnostic. Model wrappers
    can pass richer masks later without changing the hook aggregation format.
    """

    if input_ids is None or not torch.is_tensor(input_ids):
        return None
    ids = input_ids.detach()
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2 or ids.numel() == 0:
        return None

    if attention_mask is not None and torch.is_tensor(attention_mask):
        valid = attention_mask.detach().to(device=ids.device, dtype=torch.bool)
        if valid.ndim == 1:
            valid = valid.unsqueeze(0)
        if valid.shape != ids.shape:
            valid = torch.ones_like(ids, dtype=torch.bool)
    else:
        pad_token_id = getattr(tokenizer, "pad_token_id", None)
        valid = ids != int(pad_token_id) if isinstance(pad_token_id, int) else torch.ones_like(ids, dtype=torch.bool)

    config_objects = list(_iter_model_configs(model))
    vision_attr_names = (
        "image_token_id",
        "image_token_index",
        "video_token_id",
        "video_token_index",
        "vision_token_id",
        "vision_token_index",
        "vision_start_token_id",
        "vision_end_token_id",
        "image_start_token_id",
        "image_end_token_id",
        "video_start_token_id",
        "video_end_token_id",
    )
    vision_token_ids = _collect_token_ids_from_attrs(config_objects, vision_attr_names)
    if tokenizer is not None:
        vision_token_ids.update(
            _collect_token_ids_from_strings(
                tokenizer,
                (
                    "<image>",
                    "<video>",
                    "<|image_pad|>",
                    "<|video_pad|>",
                    "<|vision_start|>",
                    "<|vision_end|>",
                    "<|image|>",
                    "<image_soft_token>",
                    "<img>",
                    "<start_of_image>",
                    "<end_of_image>",
                ),
            )
        )

    special_token_ids: set[int] = set()
    if tokenizer is not None:
        for token_id in getattr(tokenizer, "all_special_ids", []) or []:
            if isinstance(token_id, int):
                special_token_ids.add(int(token_id))
        for attr_name in ("pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id"):
            value = getattr(tokenizer, attr_name, None)
            if isinstance(value, int):
                special_token_ids.add(int(value))

    vision_mask = _ids_mask(ids, vision_token_ids) & valid if vision_token_ids else torch.zeros_like(valid)
    special_mask = _ids_mask(ids, special_token_ids) & valid if special_token_ids else torch.zeros_like(valid)
    text_mask = valid & ~vision_mask & ~special_mask
    nonvision_mask = valid & ~vision_mask

    first_token_mask = torch.zeros_like(valid)
    first_four_mask = torch.zeros_like(valid)
    for batch_index in range(valid.shape[0]):
        valid_positions = torch.nonzero(valid[batch_index], as_tuple=False).reshape(-1)
        if valid_positions.numel() == 0:
            continue
        first_token_mask[batch_index, valid_positions[0]] = True
        first_four_mask[batch_index, valid_positions[:4]] = True

    candidate_groups = {
        "all_input": valid,
        "vision": vision_mask,
        "text": text_mask,
        "nonvision": nonvision_mask,
        "special": special_mask,
        "sink_first_token": first_token_mask,
        "sink_first_4": first_four_mask,
    }
    token_groups = {name: mask for name, mask in candidate_groups.items() if bool(mask.any().item())}
    if not token_groups:
        return None

    return {
        "sequence_length": int(ids.shape[1]),
        "batch_size": int(ids.shape[0]),
        "token_groups": token_groups,
        "task_name": task_name,
        "category": category,
        "vision_token_ids": sorted(vision_token_ids),
        "special_token_ids": sorted(special_token_ids),
    }


@contextmanager
def pruning_tracker_context(model: Any, context: dict[str, Any] | None):
    tracker = getattr(model, "_lmms_eval_pruning_tracker", None)
    if tracker is None or not hasattr(tracker, "track_context"):
        yield
        return
    with tracker.track_context(context):
        yield


def configure_model_for_explicit_attention(model: nn.Module, force_output_attentions: bool = False) -> dict[str, int | bool]:
    """Best-effort request for explicit HF attention tensors.

    By default this only switches HF attention implementations to eager mode.
    Setting ``config.output_attentions=True`` globally is intentionally opt-in
    because generation code can retain every layer's full ``B x H x N x N``
    attention maps and quickly exhaust GPU memory.
    """

    if not isinstance(model, nn.Module):
        raise TypeError(f"configure_model_for_explicit_attention expects an nn.Module, got {type(model).__name__}")

    seen_configs: set[int] = set()
    output_attentions_updates = 0
    output_attentions_disabled = 0
    attn_impl_updates = 0
    setter_calls = 0

    for module in model.modules():
        set_attn_implementation = getattr(module, "set_attn_implementation", None)
        if callable(set_attn_implementation):
            try:
                set_attn_implementation("eager")
                setter_calls += 1
            except Exception:
                pass

        for attr_name in ("config", "generation_config"):
            config = getattr(module, attr_name, None)
            if config is None:
                continue
            config_id = id(config)
            if config_id in seen_configs:
                continue
            seen_configs.add(config_id)

            if hasattr(config, "output_attentions"):
                target_output_attentions = bool(force_output_attentions)
                if getattr(config, "output_attentions") is not target_output_attentions:
                    try:
                        setattr(config, "output_attentions", target_output_attentions)
                        output_attentions_updates += 1
                        if not target_output_attentions:
                            output_attentions_disabled += 1
                    except Exception:
                        pass

            for impl_attr in ("_attn_implementation", "attn_implementation"):
                if hasattr(config, impl_attr) and getattr(config, impl_attr) != "eager":
                    try:
                        setattr(config, impl_attr, "eager")
                        attn_impl_updates += 1
                    except Exception:
                        pass

    return {
        "configs_seen": len(seen_configs),
        "output_attentions_updates": output_attentions_updates,
        "output_attentions_disabled": output_attentions_disabled,
        "attn_implementation_updates": attn_impl_updates,
        "set_attn_implementation_calls": setter_calls,
        "force_output_attentions": bool(force_output_attentions),
    }


def merge_pruning_payloads(payloads: Iterable[dict[str, Any]]) -> dict[str, Any]:
    merged = {
        "mlp_stats": {},
        "attn_stats": {},
        "mlp_counts": {},
        "attn_counts": {},
        "linear_state_stats": {metric: {} for metric in PruningTracker.LINEAR_STATE_METRIC_NAMES},
        "linear_state_counts": {},
        "linear_state_component_stats": {},
        "linear_state_component_counts": {},
        "mlp_hist_samples": {},
        "attn_hist_samples": {},
        "linear_state_output_hist_samples": {},
        "linear_state_input_output_delta_hist_samples": {},
        "linear_state_component_hist_samples": {},
        "mlp_hist_counts": {},
        "attn_hist_counts": {},
        "linear_state_output_hist_counts": {},
        "linear_state_input_output_delta_hist_counts": {},
        "linear_state_component_hist_counts": {},
        "mlp_hist_source_elements": {},
        "attn_hist_source_elements": {},
        "linear_state_output_hist_source_elements": {},
        "linear_state_input_output_delta_hist_source_elements": {},
        "linear_state_component_hist_source_elements": {},
        "mlp_hist_bin_edges": [],
        "attn_hist_bin_edges": [],
        "linear_state_output_hist_bin_edges": [],
        "linear_state_input_output_delta_hist_bin_edges": [],
        "linear_state_component_hist_bin_edges": [],
        "mlp_active_elements": {},
        "mlp_total_elements": {},
        "mlp_position_counts": {},
        "mlp_channel_fire_counts": {},
        "mlp_channel_fire_denoms": {},
        "linear_state_component_active_elements": {},
        "linear_state_component_total_elements": {},
        "linear_state_component_position_counts": {},
        "linear_state_component_channel_fire_counts": {},
        "linear_state_component_channel_fire_denoms": {},
        "attn_valid_elements": {},
        "attn_total_elements": {},
        "attn_entropy": {},
        "attn_peak_mass": {},
        "token_group_output_abs_sums": {},
        "token_group_output_sq_sums": {},
        "token_group_input_sq_sums": {},
        "token_group_delta_abs_sums": {},
        "token_group_delta_sq_sums": {},
        "token_group_element_counts": {},
        "token_group_token_counts": {},
        "token_group_update_counts": {},
        "token_group_cosine_sums": {},
        "token_group_cosine_counts": {},
        "attn_route_stats": {},
        "attn_route_query_counts": {},
        "attn_route_update_counts": {},
        "linear_state_component_parents": {},
        "linear_state_component_roles": {},
        "module_registration_order": [],
        "tracking_config": {},
        "num_hooks": 0,
    }

    for payload in payloads:
        if not payload:
            continue

        merged["num_hooks"] = max(int(payload.get("num_hooks", 0)), merged["num_hooks"])

        for key in (
            "mlp_stats",
            "attn_stats",
            "linear_state_component_stats",
            "mlp_channel_fire_counts",
            "linear_state_component_channel_fire_counts",
            "attn_entropy",
            "attn_peak_mass",
            "token_group_output_abs_sums",
            "token_group_output_sq_sums",
            "token_group_input_sq_sums",
            "token_group_delta_abs_sums",
            "token_group_delta_sq_sums",
            "token_group_cosine_sums",
            "attn_route_stats",
        ):
            for module_name, tensor in payload.get(key, {}).items():
                current = merged[key].get(module_name)
                if current is None:
                    merged[key][module_name] = tensor.clone()
                elif current.shape == tensor.shape:
                    merged[key][module_name] = current + tensor

        for metric_name, metric_payload in payload.get("linear_state_stats", {}).items():
            target = merged["linear_state_stats"].setdefault(metric_name, {})
            for module_name, tensor in metric_payload.items():
                current = target.get(module_name)
                if current is None:
                    target[module_name] = tensor.clone()
                elif current.shape == tensor.shape:
                    target[module_name] = current + tensor

        for key in (
            "mlp_counts",
            "attn_counts",
            "linear_state_counts",
            "linear_state_component_counts",
            "mlp_hist_source_elements",
            "attn_hist_source_elements",
            "linear_state_output_hist_source_elements",
            "linear_state_input_output_delta_hist_source_elements",
            "linear_state_component_hist_source_elements",
            "mlp_active_elements",
            "mlp_total_elements",
            "mlp_position_counts",
            "mlp_channel_fire_denoms",
            "linear_state_component_active_elements",
            "linear_state_component_total_elements",
            "linear_state_component_position_counts",
            "linear_state_component_channel_fire_denoms",
            "attn_valid_elements",
            "attn_total_elements",
            "token_group_element_counts",
            "token_group_token_counts",
            "token_group_update_counts",
            "token_group_cosine_counts",
            "attn_route_query_counts",
            "attn_route_update_counts",
        ):
            for module_name, count in payload.get(key, {}).items():
                merged[key][module_name] = merged[key].get(module_name, 0) + int(count)

        for key in (
            "mlp_hist_bin_edges",
            "attn_hist_bin_edges",
            "linear_state_output_hist_bin_edges",
            "linear_state_input_output_delta_hist_bin_edges",
            "linear_state_component_hist_bin_edges",
        ):
            if not merged[key]:
                merged[key] = list(payload.get(key, []))

        if not merged["tracking_config"]:
            merged["tracking_config"] = dict(payload.get("tracking_config", {}))
        for module_name, parent_name in payload.get("linear_state_component_parents", {}).items():
            merged["linear_state_component_parents"].setdefault(module_name, str(parent_name))
        for module_name, role in payload.get("linear_state_component_roles", {}).items():
            merged["linear_state_component_roles"].setdefault(module_name, str(role))
        for module_name in payload.get("module_registration_order", []):
            if module_name and module_name not in merged["module_registration_order"]:
                merged["module_registration_order"].append(module_name)

        max_histogram_samples = int(payload.get("tracking_config", {}).get("max_histogram_samples", 65536) or 65536)
        for key in (
            "mlp_hist_samples",
            "attn_hist_samples",
            "linear_state_output_hist_samples",
            "linear_state_input_output_delta_hist_samples",
            "linear_state_component_hist_samples",
        ):
            _merge_histogram_sample_dict(
                merged[key],
                payload.get(key, {}),
                max_histogram_samples=max_histogram_samples,
            )

        if not payload.get("mlp_hist_samples"):
            for key in (
                "mlp_hist_counts",
                "attn_hist_counts",
                "linear_state_output_hist_counts",
                "linear_state_input_output_delta_hist_counts",
                "linear_state_component_hist_counts",
            ):
                for module_name, tensor in payload.get(key, {}).items():
                    current = merged[key].get(module_name)
                    if current is None:
                        merged[key][module_name] = tensor.clone()
                    elif current.shape == tensor.shape:
                        merged[key][module_name] = current + tensor

    if merged["mlp_hist_samples"]:
        merged["mlp_hist_bin_edges"] = _build_histogram_bin_edges(
            merged["mlp_hist_samples"],
            num_bins=int(merged["tracking_config"].get("mlp_hist_num_bins", PruningTracker.MLP_HIST_NUM_BINS) or PruningTracker.MLP_HIST_NUM_BINS),
            scale="log",
        )
        merged["mlp_hist_counts"] = _histogram_counts_from_sample_dict(merged["mlp_hist_samples"], merged["mlp_hist_bin_edges"])
    if merged["attn_hist_samples"]:
        merged["attn_hist_bin_edges"] = _build_histogram_bin_edges(
            merged["attn_hist_samples"],
            num_bins=int(merged["tracking_config"].get("attn_hist_num_bins", PruningTracker.ATTN_HIST_NUM_BINS) or PruningTracker.ATTN_HIST_NUM_BINS),
            scale="linear",
        )
        merged["attn_hist_counts"] = _histogram_counts_from_sample_dict(merged["attn_hist_samples"], merged["attn_hist_bin_edges"])
    if merged["linear_state_output_hist_samples"]:
        merged["linear_state_output_hist_bin_edges"] = _build_histogram_bin_edges(
            merged["linear_state_output_hist_samples"],
            num_bins=int(merged["tracking_config"].get("linear_state_hist_num_bins", PruningTracker.LINEAR_STATE_HIST_NUM_BINS) or PruningTracker.LINEAR_STATE_HIST_NUM_BINS),
            scale="log",
        )
        merged["linear_state_output_hist_counts"] = _histogram_counts_from_sample_dict(
            merged["linear_state_output_hist_samples"],
            merged["linear_state_output_hist_bin_edges"],
        )
    if merged["linear_state_input_output_delta_hist_samples"]:
        merged["linear_state_input_output_delta_hist_bin_edges"] = _build_histogram_bin_edges(
            merged["linear_state_input_output_delta_hist_samples"],
            num_bins=int(merged["tracking_config"].get("linear_state_hist_num_bins", PruningTracker.LINEAR_STATE_HIST_NUM_BINS) or PruningTracker.LINEAR_STATE_HIST_NUM_BINS),
            scale="log",
        )
        merged["linear_state_input_output_delta_hist_counts"] = _histogram_counts_from_sample_dict(
            merged["linear_state_input_output_delta_hist_samples"],
            merged["linear_state_input_output_delta_hist_bin_edges"],
        )
    if merged["linear_state_component_hist_samples"]:
        merged["linear_state_component_hist_bin_edges"] = _build_histogram_bin_edges(
            merged["linear_state_component_hist_samples"],
            num_bins=int(merged["tracking_config"].get("linear_state_hist_num_bins", PruningTracker.LINEAR_STATE_HIST_NUM_BINS) or PruningTracker.LINEAR_STATE_HIST_NUM_BINS),
            scale="log",
        )
        merged["linear_state_component_hist_counts"] = _histogram_counts_from_sample_dict(
            merged["linear_state_component_hist_samples"],
            merged["linear_state_component_hist_bin_edges"],
        )

    return merged


def gather_pruning_payload(local_payload: dict[str, Any]) -> dict[str, Any] | None:
    if not (dist.is_available() and dist.is_initialized()):
        return local_payload

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    gathered = [None] * world_size if rank == 0 else None
    dist.gather_object(local_payload, gathered, dst=0)

    if rank == 0:
        return merge_pruning_payloads(gathered)
    return None


def _normalize_module_tensor(tensor: torch.Tensor, count: int) -> torch.Tensor:
    values = tensor.detach().float().reshape(-1).cpu()
    if count > 0:
        return values / count
    return values


def _take_evenly_spaced_samples(values: torch.Tensor, max_samples: int) -> torch.Tensor:
    values = values.detach().float().reshape(-1).cpu()
    if values.numel() <= max_samples:
        return values
    if max_samples <= 1:
        return values[:1]
    positions = torch.arange(max_samples, dtype=torch.int64)
    indices = torch.div(positions * (values.numel() - 1), max_samples - 1, rounding_mode="floor")
    return values[indices]


def _merge_histogram_sample_dict(
    target: dict[str, torch.Tensor],
    source: dict[str, torch.Tensor],
    max_histogram_samples: int,
) -> None:
    for module_name, tensor in source.items():
        sampled = _take_evenly_spaced_samples(tensor, max_histogram_samples)
        if module_name in target:
            sampled = _take_evenly_spaced_samples(torch.cat([target[module_name], sampled]), max_histogram_samples)
        target[module_name] = sampled


def _combine_histogram_samples(samples_by_module: dict[str, torch.Tensor]) -> torch.Tensor:
    if not samples_by_module:
        return torch.zeros(0, dtype=torch.float32)

    samples = [tensor.detach().float().reshape(-1).cpu() for tensor in samples_by_module.values() if tensor.numel() > 0]
    if not samples:
        return torch.zeros(0, dtype=torch.float32)
    return torch.cat(samples)


def _make_edges_strictly_increasing(edges: torch.Tensor) -> torch.Tensor:
    edges = edges.detach().float().clone().reshape(-1)
    if edges.numel() < 2:
        return edges

    for index in range(1, edges.numel()):
        if edges[index] <= edges[index - 1]:
            increment = max(1e-6, abs(float(edges[index - 1].item())) * 1e-6)
            edges[index] = edges[index - 1] + increment
    return edges


def _build_histogram_bin_edges(
    samples_by_module: dict[str, torch.Tensor],
    num_bins: int,
    scale: str,
) -> list[float]:
    samples = _combine_histogram_samples(samples_by_module)
    if samples.numel() == 0 or num_bins < 1:
        return []

    if scale == "log":
        positive = samples[samples > 0]
        if positive.numel() == 0:
            return [0.0, 1.0]
        positive_min = float(positive.min().item())
        positive_max = float(positive.max().item())
        if positive_max <= positive_min:
            edges = torch.linspace(0.0, max(positive_max, 1.0), steps=num_bins + 1)
        else:
            log_edges = torch.logspace(
                math.log10(positive_min),
                math.log10(positive_max),
                steps=num_bins,
            )
            edges = torch.cat([torch.tensor([0.0], dtype=torch.float32), log_edges.to(torch.float32)])
        return [float(value) for value in _make_edges_strictly_increasing(edges).tolist()]

    start = min(0.0, float(samples.min().item()))
    end = float(samples.max().item())
    if end <= start:
        end = start + 1e-6
    edges = torch.linspace(start, end, steps=num_bins + 1)
    return [float(value) for value in _make_edges_strictly_increasing(edges).tolist()]


def _histogram_counts_for_samples(samples: torch.Tensor, bin_edges: list[float] | tuple[float, ...]) -> torch.Tensor:
    edges = torch.tensor(bin_edges, dtype=torch.float32)
    if samples.numel() == 0 or edges.numel() < 2:
        return torch.zeros(max(edges.numel() - 1, 0), dtype=torch.float32)

    clipped = samples.detach().float().reshape(-1).cpu().clamp(min=float(edges[0].item()), max=float(edges[-1].item()))
    bucket_ids = torch.bucketize(clipped, edges[1:-1], right=False)
    return torch.bincount(bucket_ids, minlength=edges.numel() - 1).to(torch.float32)


def _histogram_counts_from_sample_dict(
    samples_by_module: dict[str, torch.Tensor],
    bin_edges: list[float] | tuple[float, ...],
) -> dict[str, torch.Tensor]:
    if not bin_edges:
        return {}
    return {
        module_name: _histogram_counts_for_samples(samples, bin_edges)
        for module_name, samples in samples_by_module.items()
    }


def _summarize_histogram(
    samples_by_module: dict[str, torch.Tensor] | None,
    counts_by_module: dict[str, torch.Tensor],
    source_elements_by_module: dict[str, int],
    bin_edges: list[float] | tuple[float, ...],
    num_bins: int,
    scale: str,
) -> dict[str, Any]:
    sample_values = _combine_histogram_samples(samples_by_module or {})
    num_source_values = int(sum(int(value) for value in source_elements_by_module.values()))

    if sample_values.numel() > 0:
        edges = _build_histogram_bin_edges(samples_by_module or {}, num_bins=num_bins, scale=scale)
        total_counts = _histogram_counts_for_samples(sample_values, edges)
        num_samples = int(total_counts.sum().item())
        return {
            "available": bool(num_samples),
            "bin_edges": [float(edge) for edge in edges],
            "counts": [int(value) for value in total_counts.tolist()],
            "num_samples": num_samples,
            "num_source_values": num_source_values,
            "is_sampled": num_source_values > num_samples,
            "sampling_ratio": (float(num_samples / num_source_values) if num_source_values > 0 else None),
        }

    edges = [float(edge) for edge in bin_edges]
    if not edges:
        return {
            "available": False,
            "bin_edges": [],
            "counts": [],
            "num_samples": 0,
            "num_source_values": 0,
            "is_sampled": False,
            "sampling_ratio": None,
        }

    total_counts = torch.zeros(len(edges) - 1, dtype=torch.float32)
    for counts in counts_by_module.values():
        if counts.numel() != total_counts.numel():
            continue
        total_counts = total_counts + counts.detach().float().cpu()
    num_samples = int(total_counts.sum().item())

    return {
        "available": bool(num_samples),
        "bin_edges": edges,
        "counts": [int(value) for value in total_counts.tolist()],
        "num_samples": num_samples,
        "num_source_values": num_source_values,
        "is_sampled": num_source_values > num_samples,
        "sampling_ratio": (float(num_samples / num_source_values) if num_source_values > 0 else None),
    }


def _module_name_segments(module_name: str) -> list[str]:
    return [segment for segment in str(module_name).split(".") if segment]


def _module_layer_indices(module_name: str) -> list[int]:
    return [int(segment) for segment in _module_name_segments(module_name) if segment.isdigit()]


def _normalize_stage_pattern(module_name: str) -> str:
    segments = ["*" if segment.isdigit() else segment for segment in _module_name_segments(module_name)]
    return ".".join(segments)


def _natural_module_sort_key(module_name: str) -> tuple[Any, ...]:
    return tuple((0, int(segment)) if segment.isdigit() else (1, segment) for segment in _module_name_segments(module_name))


def _ordered_module_names(payload: dict[str, Any], module_names: Iterable[str]) -> list[str]:
    registration_order = [str(name) for name in payload.get("module_registration_order", []) if name]
    registration_lookup = {name: idx for idx, name in enumerate(registration_order)}
    unique_names = {str(name) for name in module_names if name}
    return sorted(
        unique_names,
        key=lambda name: (
            0,
            registration_lookup[name],
        )
        if name in registration_lookup
        else (
            1,
            _natural_module_sort_key(name),
            name,
        ),
    )


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def _summarize_stat_dict(
    stats: dict[str, torch.Tensor],
    counts: dict[str, int],
    hist_samples: dict[str, torch.Tensor] | None = None,
    hist_counts: dict[str, torch.Tensor] | None = None,
    hist_source_elements: dict[str, int] | None = None,
    sources: dict[str, str] | None = None,
    hist_bin_edges: list[float] | tuple[float, ...] | None = None,
    hist_num_bins: int = 20,
    hist_scale: str = "linear",
    top_k: int = 10,
    include_module_rows: bool = False,
) -> dict[str, Any]:
    module_rows: list[dict[str, Any]] = []
    flat_values: list[torch.Tensor] = []

    for module_name, tensor in stats.items():
        values = _normalize_module_tensor(tensor, int(counts.get(module_name, 0)))
        if values.numel() == 0:
            continue

        module_rows.append(
            {
                "module": module_name,
                "mean": float(values.mean().item()),
                "std": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
                "min": float(values.min().item()),
                "max": float(values.max().item()),
                "num_values": int(values.numel()),
                "updates": int(counts.get(module_name, 0)),
                "source": (sources or {}).get(module_name),
            }
        )
        flat_values.append(values)

    module_rows.sort(key=lambda row: row["mean"], reverse=True)

    summary: dict[str, Any] = {
        "available": bool(module_rows),
        "num_modules": len(module_rows),
        "num_updates": int(sum(counts.values())),
        "top_modules": module_rows[:top_k],
    }

    if flat_values:
        all_values = torch.cat(flat_values)
        summary.update(
            {
                "num_values": int(all_values.numel()),
                "mean": float(all_values.mean().item()),
                "std": float(all_values.std(unbiased=False).item()) if all_values.numel() > 1 else 0.0,
                "min": float(all_values.min().item()),
                "max": float(all_values.max().item()),
            }
        )
    else:
        summary.update(
            {
                "num_values": 0,
                "mean": None,
                "std": None,
                "min": None,
                "max": None,
            }
        )

    summary["histogram"] = _summarize_histogram(
        hist_samples,
        hist_counts or {},
        hist_source_elements or {},
        hist_bin_edges or [],
        num_bins=hist_num_bins,
        scale=hist_scale,
    )

    if include_module_rows:
        summary["module_rows"] = module_rows

    return summary


def _summarize_mlp_payload(
    payload: dict[str, Any],
    top_k: int = 10,
    include_module_rows: bool = False,
) -> dict[str, Any]:
    summary = _summarize_stat_dict(
        payload.get("mlp_stats", {}),
        payload.get("mlp_counts", {}),
        hist_samples=payload.get("mlp_hist_samples", {}),
        hist_counts=payload.get("mlp_hist_counts", {}),
        hist_source_elements=payload.get("mlp_hist_source_elements", {}),
        sources=None,
        hist_bin_edges=payload.get("mlp_hist_bin_edges", []),
        hist_num_bins=int(payload.get("tracking_config", {}).get("mlp_hist_num_bins", PruningTracker.MLP_HIST_NUM_BINS) or PruningTracker.MLP_HIST_NUM_BINS),
        hist_scale="log",
        top_k=top_k,
        include_module_rows=include_module_rows,
    )

    global_active_elements = 0
    global_total_elements = 0
    global_positions = 0
    all_channel_rates: list[torch.Tensor] = []
    active_threshold = payload.get("tracking_config", {}).get("mlp_active_threshold_abs")

    module_rows = summary.get("module_rows", [])
    row_lookup = {row["module"]: row for row in module_rows}

    for module_name in payload.get("mlp_stats", {}):
        active_elements = int(payload.get("mlp_active_elements", {}).get(module_name, 0))
        total_elements = int(payload.get("mlp_total_elements", {}).get(module_name, 0))
        positions = int(payload.get("mlp_position_counts", {}).get(module_name, 0))
        global_active_elements += active_elements
        global_total_elements += total_elements
        global_positions += positions

        row = row_lookup.get(module_name)
        if row is not None:
            row["mean_active_neurons"] = float(active_elements / positions) if positions > 0 else None
            row["mean_active_fraction"] = float(active_elements / total_elements) if total_elements > 0 else None

        fire_counts = payload.get("mlp_channel_fire_counts", {}).get(module_name)
        fire_denom = int(payload.get("mlp_channel_fire_denoms", {}).get(module_name, 0))
        if fire_counts is None or fire_denom <= 0:
            continue

        channel_rates = fire_counts.detach().float().reshape(-1).cpu() / fire_denom
        all_channel_rates.append(channel_rates)
        if row is not None:
            row["mean_channel_firing_rate"] = float(channel_rates.mean().item())
            row["low_activity_channel_fraction"] = float((channel_rates <= 0.01).float().mean().item())

    summary["mean_active_neurons"] = float(global_active_elements / global_positions) if global_positions > 0 else None
    summary["mean_active_fraction"] = float(global_active_elements / global_total_elements) if global_total_elements > 0 else None
    summary["active_threshold_abs"] = float(active_threshold) if active_threshold is not None else None

    if all_channel_rates:
        flat_rates = torch.cat(all_channel_rates)
        summary["mean_channel_firing_rate"] = float(flat_rates.mean().item())
        summary["low_activity_channel_fraction"] = float((flat_rates <= 0.01).float().mean().item())
    else:
        summary["mean_channel_firing_rate"] = None
        summary["low_activity_channel_fraction"] = None

    return summary


def _summarize_attention_payload(
    payload: dict[str, Any],
    top_k: int = 10,
    include_module_rows: bool = False,
) -> dict[str, Any]:
    summary = _summarize_stat_dict(
        payload.get("attn_stats", {}),
        payload.get("attn_counts", {}),
        hist_samples=payload.get("attn_hist_samples", {}),
        hist_counts=payload.get("attn_hist_counts", {}),
        hist_source_elements=payload.get("attn_hist_source_elements", {}),
        sources=None,
        hist_bin_edges=payload.get("attn_hist_bin_edges", []),
        hist_num_bins=int(payload.get("tracking_config", {}).get("attn_hist_num_bins", PruningTracker.ATTN_HIST_NUM_BINS) or PruningTracker.ATTN_HIST_NUM_BINS),
        hist_scale="linear",
        top_k=top_k,
        include_module_rows=include_module_rows,
    )

    entropy_values: list[torch.Tensor] = []
    peak_values: list[torch.Tensor] = []
    global_valid_elements = 0
    global_total_elements = 0

    module_rows = summary.get("module_rows", [])
    row_lookup = {row["module"]: row for row in module_rows}

    for module_name in payload.get("attn_stats", {}):
        updates = int(payload.get("attn_counts", {}).get(module_name, 0))
        entropy = payload.get("attn_entropy", {}).get(module_name)
        peak_mass = payload.get("attn_peak_mass", {}).get(module_name)
        valid_elements = int(payload.get("attn_valid_elements", {}).get(module_name, 0))
        total_elements = int(payload.get("attn_total_elements", {}).get(module_name, 0))
        global_valid_elements += valid_elements
        global_total_elements += total_elements
        row = row_lookup.get(module_name)
        if row is not None:
            row["attention_valid_fraction"] = float(valid_elements / total_elements) if total_elements > 0 else None

        if entropy is not None:
            normalized_entropy = _normalize_module_tensor(entropy, updates)
            entropy_values.append(normalized_entropy)
            if row is not None:
                row["mean_entropy"] = float(normalized_entropy.mean().item())

        if peak_mass is not None:
            normalized_peak = _normalize_module_tensor(peak_mass, updates)
            peak_values.append(normalized_peak)
            if row is not None:
                row["mean_peak_mass"] = float(normalized_peak.mean().item())

    if entropy_values:
        flat_entropy = torch.cat(entropy_values)
        summary["mean_entropy"] = float(flat_entropy.mean().item())
        summary["std_entropy"] = float(flat_entropy.std(unbiased=False).item()) if flat_entropy.numel() > 1 else 0.0
    else:
        summary["mean_entropy"] = None
        summary["std_entropy"] = None

    if peak_values:
        flat_peak = torch.cat(peak_values)
        summary["mean_peak_mass"] = float(flat_peak.mean().item())
        summary["std_peak_mass"] = float(flat_peak.std(unbiased=False).item()) if flat_peak.numel() > 1 else 0.0
    else:
        summary["mean_peak_mass"] = None
        summary["std_peak_mass"] = None
    summary["attention_valid_fraction"] = float(global_valid_elements / global_total_elements) if global_total_elements > 0 else None

    return summary


def _summarize_linear_state_payload(
    payload: dict[str, Any],
    top_k: int = 10,
    include_module_rows: bool = False,
) -> dict[str, Any]:
    stats_by_metric = payload.get("linear_state_stats", {}) or {}
    counts = payload.get("linear_state_counts", {}) or {}
    module_names = set(counts.keys())
    for metric_stats in stats_by_metric.values():
        module_names.update(metric_stats.keys())

    module_rows: list[dict[str, Any]] = []
    metric_values: dict[str, list[float]] = {metric: [] for metric in PruningTracker.LINEAR_STATE_METRIC_NAMES}

    for module_name in sorted(module_names, key=_natural_module_sort_key):
        updates = int(counts.get(module_name, 0))
        row: dict[str, Any] = {
            "module": module_name,
            "updates": updates,
        }
        has_metric = False
        for metric_name in PruningTracker.LINEAR_STATE_METRIC_NAMES:
            tensor = (stats_by_metric.get(metric_name, {}) or {}).get(module_name)
            if tensor is None:
                row[metric_name] = None
                continue
            value_tensor = _normalize_module_tensor(tensor, updates)
            if value_tensor.numel() == 0:
                row[metric_name] = None
                continue
            value = float(value_tensor.mean().item())
            row[metric_name] = value
            metric_values[metric_name].append(value)
            has_metric = True
        if has_metric:
            module_rows.append(row)

    delta_rows = [row for row in module_rows if row.get("input_output_delta_rms_ratio") is not None]
    top_delta_rows = sorted(delta_rows, key=lambda row: float(row["input_output_delta_rms_ratio"]), reverse=True)[:top_k]
    low_delta_rows = sorted(delta_rows, key=lambda row: float(row["input_output_delta_rms_ratio"]))[:top_k]

    summary: dict[str, Any] = {
        "available": bool(module_rows),
        "num_modules": len(module_rows),
        "num_updates": int(sum(int(value) for value in counts.values())),
        "metric_names": list(PruningTracker.LINEAR_STATE_METRIC_NAMES),
        "top_delta_modules": top_delta_rows,
        "low_delta_modules": low_delta_rows,
        "histograms": {
            "output_abs": _summarize_histogram(
                payload.get("linear_state_output_hist_samples", {}),
                payload.get("linear_state_output_hist_counts", {}),
                payload.get("linear_state_output_hist_source_elements", {}),
                payload.get("linear_state_output_hist_bin_edges", []),
                num_bins=int(payload.get("tracking_config", {}).get("linear_state_hist_num_bins", PruningTracker.LINEAR_STATE_HIST_NUM_BINS) or PruningTracker.LINEAR_STATE_HIST_NUM_BINS),
                scale="log",
            ),
            "input_output_delta_abs": _summarize_histogram(
                payload.get("linear_state_input_output_delta_hist_samples", {}),
                payload.get("linear_state_input_output_delta_hist_counts", {}),
                payload.get("linear_state_input_output_delta_hist_source_elements", {}),
                payload.get("linear_state_input_output_delta_hist_bin_edges", []),
                num_bins=int(payload.get("tracking_config", {}).get("linear_state_hist_num_bins", PruningTracker.LINEAR_STATE_HIST_NUM_BINS) or PruningTracker.LINEAR_STATE_HIST_NUM_BINS),
                scale="log",
            ),
        },
    }

    for metric_name, values in metric_values.items():
        if values:
            tensor = torch.tensor(values, dtype=torch.float32)
            summary[f"mean_{metric_name}"] = float(tensor.mean().item())
            summary[f"std_{metric_name}"] = float(tensor.std(unbiased=False).item()) if tensor.numel() > 1 else 0.0
        else:
            summary[f"mean_{metric_name}"] = None
            summary[f"std_{metric_name}"] = None

    if include_module_rows:
        summary["module_rows"] = module_rows

    return summary


def _summarize_linear_state_component_payload(
    payload: dict[str, Any],
    top_k: int = 10,
    include_module_rows: bool = False,
) -> dict[str, Any]:
    summary = _summarize_stat_dict(
        payload.get("linear_state_component_stats", {}),
        payload.get("linear_state_component_counts", {}),
        hist_samples=payload.get("linear_state_component_hist_samples", {}),
        hist_counts=payload.get("linear_state_component_hist_counts", {}),
        hist_source_elements=payload.get("linear_state_component_hist_source_elements", {}),
        sources=payload.get("linear_state_component_roles", {}),
        hist_bin_edges=payload.get("linear_state_component_hist_bin_edges", []),
        hist_num_bins=int(payload.get("tracking_config", {}).get("linear_state_hist_num_bins", PruningTracker.LINEAR_STATE_HIST_NUM_BINS) or PruningTracker.LINEAR_STATE_HIST_NUM_BINS),
        hist_scale="log",
        top_k=top_k,
        include_module_rows=include_module_rows,
    )

    global_active_elements = 0
    global_total_elements = 0
    global_positions = 0
    all_channel_rates: list[torch.Tensor] = []
    parents = payload.get("linear_state_component_parents", {}) or {}
    roles = payload.get("linear_state_component_roles", {}) or {}

    module_rows = summary.get("module_rows", [])
    row_lookup = {row["module"]: row for row in module_rows}

    for module_name in payload.get("linear_state_component_stats", {}):
        active_elements = int(payload.get("linear_state_component_active_elements", {}).get(module_name, 0))
        total_elements = int(payload.get("linear_state_component_total_elements", {}).get(module_name, 0))
        positions = int(payload.get("linear_state_component_position_counts", {}).get(module_name, 0))
        global_active_elements += active_elements
        global_total_elements += total_elements
        global_positions += positions

        row = row_lookup.get(module_name)
        if row is not None:
            row["parent_module"] = parents.get(module_name)
            row["component_name"] = module_name.rsplit(".", 1)[-1]
            row["component_role"] = roles.get(module_name)
            row["mean_active_components"] = float(active_elements / positions) if positions > 0 else None
            row["mean_active_fraction"] = float(active_elements / total_elements) if total_elements > 0 else None

        fire_counts = payload.get("linear_state_component_channel_fire_counts", {}).get(module_name)
        fire_denom = int(payload.get("linear_state_component_channel_fire_denoms", {}).get(module_name, 0))
        if fire_counts is None or fire_denom <= 0:
            continue

        channel_rates = fire_counts.detach().float().reshape(-1).cpu() / fire_denom
        all_channel_rates.append(channel_rates)
        if row is not None:
            row["mean_channel_firing_rate"] = float(channel_rates.mean().item())
            row["low_activity_channel_fraction"] = float((channel_rates <= 0.01).float().mean().item())

    summary["mean_active_components"] = float(global_active_elements / global_positions) if global_positions > 0 else None
    summary["mean_active_fraction"] = float(global_active_elements / global_total_elements) if global_total_elements > 0 else None

    if all_channel_rates:
        flat_rates = torch.cat(all_channel_rates)
        summary["mean_channel_firing_rate"] = float(flat_rates.mean().item())
        summary["low_activity_channel_fraction"] = float((flat_rates <= 0.01).float().mean().item())
    else:
        summary["mean_channel_firing_rate"] = None
        summary["low_activity_channel_fraction"] = None

    return summary


def _tensor_scalar(payload: dict[str, torch.Tensor], key: str) -> float:
    value = payload.get(key)
    if value is None:
        return 0.0
    if torch.is_tensor(value):
        return float(value.detach().float().reshape(-1).sum().item())
    return float(value)


def _summarize_token_group_payload(
    payload: dict[str, Any],
    top_k: int = 10,
    include_module_rows: bool = False,
) -> dict[str, Any]:
    element_counts = payload.get("token_group_element_counts", {}) or {}
    token_counts = payload.get("token_group_token_counts", {}) or {}
    update_counts = payload.get("token_group_update_counts", {}) or {}
    output_abs_sums = payload.get("token_group_output_abs_sums", {}) or {}
    output_sq_sums = payload.get("token_group_output_sq_sums", {}) or {}
    input_sq_sums = payload.get("token_group_input_sq_sums", {}) or {}
    delta_abs_sums = payload.get("token_group_delta_abs_sums", {}) or {}
    delta_sq_sums = payload.get("token_group_delta_sq_sums", {}) or {}
    cosine_sums = payload.get("token_group_cosine_sums", {}) or {}
    cosine_counts = payload.get("token_group_cosine_counts", {}) or {}

    rows: list[dict[str, Any]] = []
    aggregate: dict[tuple[str, str, str], dict[str, Any]] = {}

    for key, element_count in element_counts.items():
        element_count = int(element_count)
        if element_count <= 0:
            continue
        stream, module_name, phase, group_name = PruningTracker._split_behavior_key(key)
        output_abs_sum = _tensor_scalar(output_abs_sums, key)
        output_sq_sum = _tensor_scalar(output_sq_sums, key)
        input_sq_sum = _tensor_scalar(input_sq_sums, key)
        delta_abs_sum = _tensor_scalar(delta_abs_sums, key)
        delta_sq_sum = _tensor_scalar(delta_sq_sums, key)
        cosine_count = int(cosine_counts.get(key, 0) or 0)

        output_abs_mean = output_abs_sum / element_count
        output_rms = math.sqrt(max(output_sq_sum / element_count, 0.0))
        input_rms = math.sqrt(max(input_sq_sum / element_count, 0.0)) if input_sq_sum > 0 else None
        delta_abs_mean = delta_abs_sum / element_count if delta_abs_sum > 0 else None
        delta_rms = math.sqrt(max(delta_sq_sum / element_count, 0.0)) if delta_sq_sum > 0 else None
        output_input_rms_ratio = output_rms / max(input_rms, 1e-12) if input_rms is not None else None
        delta_input_rms_ratio = delta_rms / max(input_rms, 1e-12) if delta_rms is not None and input_rms is not None else None
        output_input_cosine = _tensor_scalar(cosine_sums, key) / cosine_count if cosine_count > 0 else None

        row = {
            "stream": stream,
            "module": module_name,
            "phase": phase,
            "token_group": group_name,
            "updates": int(update_counts.get(key, 0) or 0),
            "token_count": int(token_counts.get(key, 0) or 0),
            "element_count": element_count,
            "output_abs_mean": float(output_abs_mean),
            "output_rms": float(output_rms),
            "input_rms": float(input_rms) if input_rms is not None else None,
            "input_output_delta_abs_mean": float(delta_abs_mean) if delta_abs_mean is not None else None,
            "input_output_delta_rms": float(delta_rms) if delta_rms is not None else None,
            "output_input_rms_ratio": float(output_input_rms_ratio) if output_input_rms_ratio is not None else None,
            "input_output_delta_rms_ratio": float(delta_input_rms_ratio) if delta_input_rms_ratio is not None else None,
            "output_input_cosine": float(output_input_cosine) if output_input_cosine is not None else None,
        }
        rows.append(row)

        aggregate_key = (stream, phase, group_name)
        agg = aggregate.setdefault(
            aggregate_key,
            {
                "stream": stream,
                "phase": phase,
                "token_group": group_name,
                "num_modules": 0,
                "updates": 0,
                "token_count": 0,
                "element_count": 0,
                "output_abs_sum": 0.0,
                "output_sq_sum": 0.0,
                "input_sq_sum": 0.0,
                "delta_abs_sum": 0.0,
                "delta_sq_sum": 0.0,
                "cosine_sum": 0.0,
                "cosine_count": 0,
            },
        )
        agg["num_modules"] += 1
        agg["updates"] += int(update_counts.get(key, 0) or 0)
        agg["token_count"] += int(token_counts.get(key, 0) or 0)
        agg["element_count"] += element_count
        agg["output_abs_sum"] += output_abs_sum
        agg["output_sq_sum"] += output_sq_sum
        agg["input_sq_sum"] += input_sq_sum
        agg["delta_abs_sum"] += delta_abs_sum
        agg["delta_sq_sum"] += delta_sq_sum
        agg["cosine_sum"] += _tensor_scalar(cosine_sums, key)
        agg["cosine_count"] += cosine_count

    rows.sort(key=lambda row: (row["stream"], row["phase"], row["token_group"], row["module"]))
    aggregate_rows = []
    for agg in aggregate.values():
        element_count = int(agg["element_count"])
        if element_count <= 0:
            continue
        output_rms = math.sqrt(max(float(agg["output_sq_sum"]) / element_count, 0.0))
        input_rms = math.sqrt(max(float(agg["input_sq_sum"]) / element_count, 0.0)) if agg["input_sq_sum"] > 0 else None
        delta_rms = math.sqrt(max(float(agg["delta_sq_sum"]) / element_count, 0.0)) if agg["delta_sq_sum"] > 0 else None
        aggregate_rows.append(
            {
                "stream": agg["stream"],
                "phase": agg["phase"],
                "token_group": agg["token_group"],
                "num_modules": int(agg["num_modules"]),
                "updates": int(agg["updates"]),
                "token_count": int(agg["token_count"]),
                "element_count": element_count,
                "output_abs_mean": float(agg["output_abs_sum"] / element_count),
                "output_rms": float(output_rms),
                "input_rms": float(input_rms) if input_rms is not None else None,
                "input_output_delta_abs_mean": float(agg["delta_abs_sum"] / element_count) if agg["delta_abs_sum"] > 0 else None,
                "input_output_delta_rms": float(delta_rms) if delta_rms is not None else None,
                "output_input_rms_ratio": float(output_rms / max(input_rms, 1e-12)) if input_rms is not None else None,
                "input_output_delta_rms_ratio": float(delta_rms / max(input_rms, 1e-12)) if delta_rms is not None and input_rms is not None else None,
                "output_input_cosine": float(agg["cosine_sum"] / agg["cosine_count"]) if agg["cosine_count"] > 0 else None,
            }
        )
    aggregate_rows.sort(key=lambda row: (row["stream"], row["phase"], row["token_group"]))

    summary: dict[str, Any] = {
        "available": bool(rows),
        "num_rows": len(rows),
        "num_groups": len(aggregate_rows),
        "aggregate_rows": aggregate_rows,
        "top_output_rms": sorted(rows, key=lambda row: row["output_rms"], reverse=True)[:top_k],
    }
    if include_module_rows:
        summary["module_rows"] = rows
    return summary


def _summarize_attention_routing_payload(
    payload: dict[str, Any],
    top_k: int = 10,
    include_module_rows: bool = False,
) -> dict[str, Any]:
    route_stats = payload.get("attn_route_stats", {}) or {}
    query_counts = payload.get("attn_route_query_counts", {}) or {}
    update_counts = payload.get("attn_route_update_counts", {}) or {}
    rows: list[dict[str, Any]] = []
    aggregate: dict[tuple[str, str, str], dict[str, Any]] = {}

    for key, query_count in query_counts.items():
        query_count = int(query_count)
        if query_count <= 0:
            continue
        tensor = route_stats.get(key)
        if tensor is None:
            continue
        values = tensor.detach().float().reshape(-1).cpu() / query_count
        if values.numel() == 0:
            continue
        module_name, phase, query_group, key_group = PruningTracker._split_route_key(key)
        mean_mass = float(values.mean().item())
        row = {
            "module": module_name,
            "phase": phase,
            "query_group": query_group,
            "key_group": key_group,
            "mean_mass": mean_mass,
            "std_mass": float(values.std(unbiased=False).item()) if values.numel() > 1 else 0.0,
            "min_mass": float(values.min().item()),
            "max_mass": float(values.max().item()),
            "num_heads": int(values.numel()),
            "query_count": query_count,
            "updates": int(update_counts.get(key, 0) or 0),
        }
        rows.append(row)

        aggregate_key = (phase, query_group, key_group)
        agg = aggregate.setdefault(
            aggregate_key,
            {
                "phase": phase,
                "query_group": query_group,
                "key_group": key_group,
                "module_count": 0,
                "weighted_mass_sum": 0.0,
                "head_query_count": 0,
                "updates": 0,
            },
        )
        agg["module_count"] += 1
        agg["weighted_mass_sum"] += float(tensor.detach().float().sum().item())
        agg["head_query_count"] += query_count * int(values.numel())
        agg["updates"] += int(update_counts.get(key, 0) or 0)

    rows.sort(key=lambda row: (row["phase"], row["query_group"], row["key_group"], row["module"]))
    aggregate_rows = []
    for agg in aggregate.values():
        denom = int(agg["head_query_count"])
        aggregate_rows.append(
            {
                "phase": agg["phase"],
                "query_group": agg["query_group"],
                "key_group": agg["key_group"],
                "mean_mass": float(agg["weighted_mass_sum"] / denom) if denom > 0 else None,
                "module_count": int(agg["module_count"]),
                "updates": int(agg["updates"]),
                "head_query_count": denom,
            }
        )
    aggregate_rows.sort(key=lambda row: (row["phase"], row["query_group"], row["key_group"]))

    summary: dict[str, Any] = {
        "available": bool(rows),
        "num_routes": len(rows),
        "aggregate_rows": aggregate_rows,
        "top_routes": sorted(rows, key=lambda row: row["mean_mass"], reverse=True)[:top_k],
    }
    if include_module_rows:
        summary["module_rows"] = rows
    return summary


def _summarize_network_payload(
    payload: dict[str, Any],
    mlp_summary: dict[str, Any],
    attention_summary: dict[str, Any],
    linear_state_summary: dict[str, Any],
) -> dict[str, Any]:
    mlp_rows = {row["module"]: row for row in mlp_summary.get("module_rows", [])}
    attention_rows = {row["module"]: row for row in attention_summary.get("module_rows", [])}
    linear_state_rows = {row["module"]: row for row in linear_state_summary.get("module_rows", [])}
    ordered_modules = _ordered_module_names(payload, [*mlp_rows.keys(), *attention_rows.keys(), *linear_state_rows.keys()])

    module_sequence: list[dict[str, Any]] = []
    for sequence_index, module_name in enumerate(ordered_modules):
        layer_indices = _module_layer_indices(module_name)
        mlp_row = mlp_rows.get(module_name, {})
        attention_row = attention_rows.get(module_name, {})
        linear_state_row = linear_state_rows.get(module_name, {})

        is_mlp = bool(mlp_row)
        has_explicit_attention = bool(attention_row)
        is_linear_state = bool(linear_state_row)
        kinds = []
        if has_explicit_attention:
            kinds.append("attention")
        if is_linear_state:
            kinds.append("linear_state")
        if is_mlp:
            kinds.append("mlp")

        module_sequence.append(
            {
                "sequence_index": sequence_index,
                "module": module_name,
                "kind": kinds[0] if len(kinds) == 1 else "+".join(kinds) if kinds else "unknown",
                "stage_pattern": _normalize_stage_pattern(module_name),
                "path_depth": len(_module_name_segments(module_name)),
                "layer_indices": layer_indices,
                "primary_layer_index": layer_indices[0] if layer_indices else None,
                "is_mlp": is_mlp,
                "has_explicit_attention": has_explicit_attention,
                "is_linear_state": is_linear_state,
                "mlp_mean": mlp_row.get("mean"),
                "mlp_active_fraction": mlp_row.get("mean_active_fraction"),
                "mlp_mean_active_neurons": mlp_row.get("mean_active_neurons"),
                "attention_mean": attention_row.get("mean"),
                "attention_entropy": attention_row.get("mean_entropy"),
                "attention_peak_mass": attention_row.get("mean_peak_mass"),
                "attention_valid_fraction": attention_row.get("attention_valid_fraction"),
                "linear_state_output_abs_mean": linear_state_row.get("output_abs_mean"),
                "linear_state_input_output_delta_abs_mean": linear_state_row.get("input_output_delta_abs_mean"),
                "linear_state_output_input_rms_ratio": linear_state_row.get("output_input_rms_ratio"),
                "linear_state_input_output_delta_rms_ratio": linear_state_row.get("input_output_delta_rms_ratio"),
                "linear_state_output_input_cosine": linear_state_row.get("output_input_cosine"),
            }
        )

    pattern_aggregates: dict[str, dict[str, Any]] = {}
    for row in module_sequence:
        stage_pattern = row["stage_pattern"]
        aggregate = pattern_aggregates.setdefault(
            stage_pattern,
            {
                "stage_pattern": stage_pattern,
                "kind": row["kind"],
                "count": 0,
                "first_sequence_index": row["sequence_index"],
                "mlp_mean_values": [],
                "mlp_active_fraction_values": [],
                "mlp_mean_active_neurons_values": [],
                "attention_mean_values": [],
                "attention_entropy_values": [],
                "attention_peak_mass_values": [],
                "attention_valid_fraction_values": [],
                "linear_state_output_abs_mean_values": [],
                "linear_state_input_output_delta_abs_mean_values": [],
                "linear_state_output_input_rms_ratio_values": [],
                "linear_state_input_output_delta_rms_ratio_values": [],
                "linear_state_output_input_cosine_values": [],
            },
        )
        aggregate["count"] += 1
        aggregate["first_sequence_index"] = min(aggregate["first_sequence_index"], row["sequence_index"])
        if row.get("mlp_mean") is not None:
            aggregate["mlp_mean_values"].append(float(row["mlp_mean"]))
        if row.get("mlp_active_fraction") is not None:
            aggregate["mlp_active_fraction_values"].append(float(row["mlp_active_fraction"]))
        if row.get("mlp_mean_active_neurons") is not None:
            aggregate["mlp_mean_active_neurons_values"].append(float(row["mlp_mean_active_neurons"]))
        if row.get("attention_mean") is not None:
            aggregate["attention_mean_values"].append(float(row["attention_mean"]))
        if row.get("attention_entropy") is not None:
            aggregate["attention_entropy_values"].append(float(row["attention_entropy"]))
        if row.get("attention_peak_mass") is not None:
            aggregate["attention_peak_mass_values"].append(float(row["attention_peak_mass"]))
        if row.get("attention_valid_fraction") is not None:
            aggregate["attention_valid_fraction_values"].append(float(row["attention_valid_fraction"]))
        if row.get("linear_state_output_abs_mean") is not None:
            aggregate["linear_state_output_abs_mean_values"].append(float(row["linear_state_output_abs_mean"]))
        if row.get("linear_state_input_output_delta_abs_mean") is not None:
            aggregate["linear_state_input_output_delta_abs_mean_values"].append(float(row["linear_state_input_output_delta_abs_mean"]))
        if row.get("linear_state_output_input_rms_ratio") is not None:
            aggregate["linear_state_output_input_rms_ratio_values"].append(float(row["linear_state_output_input_rms_ratio"]))
        if row.get("linear_state_input_output_delta_rms_ratio") is not None:
            aggregate["linear_state_input_output_delta_rms_ratio_values"].append(float(row["linear_state_input_output_delta_rms_ratio"]))
        if row.get("linear_state_output_input_cosine") is not None:
            aggregate["linear_state_output_input_cosine_values"].append(float(row["linear_state_output_input_cosine"]))

    stage_patterns = sorted(pattern_aggregates.values(), key=lambda row: (row["first_sequence_index"], row["stage_pattern"]))
    for row in stage_patterns:
        row["mlp_mean"] = _mean_or_none(row.pop("mlp_mean_values"))
        row["mlp_active_fraction"] = _mean_or_none(row.pop("mlp_active_fraction_values"))
        row["mlp_mean_active_neurons"] = _mean_or_none(row.pop("mlp_mean_active_neurons_values"))
        row["attention_mean"] = _mean_or_none(row.pop("attention_mean_values"))
        row["attention_entropy"] = _mean_or_none(row.pop("attention_entropy_values"))
        row["attention_peak_mass"] = _mean_or_none(row.pop("attention_peak_mass_values"))
        row["attention_valid_fraction"] = _mean_or_none(row.pop("attention_valid_fraction_values"))
        row["linear_state_output_abs_mean"] = _mean_or_none(row.pop("linear_state_output_abs_mean_values"))
        row["linear_state_input_output_delta_abs_mean"] = _mean_or_none(row.pop("linear_state_input_output_delta_abs_mean_values"))
        row["linear_state_output_input_rms_ratio"] = _mean_or_none(row.pop("linear_state_output_input_rms_ratio_values"))
        row["linear_state_input_output_delta_rms_ratio"] = _mean_or_none(row.pop("linear_state_input_output_delta_rms_ratio_values"))
        row["linear_state_output_input_cosine"] = _mean_or_none(row.pop("linear_state_output_input_cosine_values"))

    return {
        "available": bool(module_sequence),
        "num_modules": len(module_sequence),
        "module_sequence": module_sequence,
        "stage_patterns": stage_patterns,
    }


def summarize_pruning_payload(
    payload: dict[str, Any],
    top_k: int = 10,
    include_module_rows: bool = False,
) -> dict[str, Any]:
    mlp_summary = _summarize_mlp_payload(
        payload,
        top_k=top_k,
        include_module_rows=True,
    )
    attention_summary = _summarize_attention_payload(
        payload,
        top_k=top_k,
        include_module_rows=True,
    )
    linear_state_summary = _summarize_linear_state_payload(
        payload,
        top_k=top_k,
        include_module_rows=True,
    )
    linear_state_component_summary = _summarize_linear_state_component_payload(
        payload,
        top_k=top_k,
        include_module_rows=True,
    )
    token_group_summary = _summarize_token_group_payload(
        payload,
        top_k=top_k,
        include_module_rows=True,
    )
    attention_routing_summary = _summarize_attention_routing_payload(
        payload,
        top_k=top_k,
        include_module_rows=True,
    )
    network_summary = _summarize_network_payload(
        payload,
        mlp_summary=mlp_summary,
        attention_summary=attention_summary,
        linear_state_summary=linear_state_summary,
    )

    if not include_module_rows:
        mlp_summary.pop("module_rows", None)
        attention_summary.pop("module_rows", None)
        linear_state_summary.pop("module_rows", None)
        linear_state_component_summary.pop("module_rows", None)
        token_group_summary.pop("module_rows", None)
        attention_routing_summary.pop("module_rows", None)

    return {
        "mlp": mlp_summary,
        "attention": attention_summary,
        "linear_state": linear_state_summary,
        "linear_state_components": linear_state_component_summary,
        "token_groups": token_group_summary,
        "attention_routing": attention_routing_summary,
        "network": network_summary,
        "cross_modal_attention": {
            "available": bool(attention_routing_summary.get("available")),
            "reason": (
                "Use attention_routing rows with query_group/key_group when token span metadata is available."
                if attention_routing_summary.get("available")
                else "No token span metadata and explicit attention maps were jointly available."
            ),
        },
    }
