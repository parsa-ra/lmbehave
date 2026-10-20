from __future__ import annotations

import os
import re
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterable

import torch
import torch.distributed as dist
import torch.nn as nn


_LAYER_PATH_RE = re.compile(r"^(?P<prefix>.*(?:layers|blocks|\.h))\.(?P<index>\d+)$")
_MIXER_CHILD_NAMES = ("self_attn", "linear_attn", "attn", "attention", "mixer")
_MLP_CHILD_NAMES = ("mlp", "ffn", "feed_forward")
_VALID_PHASES = frozenset({"prefill", "decode"})
_VALID_POOLING = frozenset({"last_valid", "mean_valid"})
_VALID_DECODE_AGGREGATION = frozenset({"mean_updates", "first_update", "all_updates"})
_VALID_STORAGE_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


@dataclass(frozen=True)
class _LayerSpec:
    layer_index: int
    layer_name: str
    layer_module: nn.Module
    mixer_name: str
    mixer_module: nn.Module
    mixer_stream: str
    mlp_name: str
    mlp_module: nn.Module
    attention_delta_exact: bool
    mlp_delta_exact: bool
    additional_residual_branch: str | None
    post_residual_transform: str | None

    def metadata(self) -> dict[str, Any]:
        return {
            "layer_index": self.layer_index,
            "layer_name": self.layer_name,
            "layer_class": type(self.layer_module).__name__,
            "mixer_name": self.mixer_name,
            "mixer_class": type(self.mixer_module).__name__,
            "mixer_stream": self.mixer_stream,
            "mlp_name": self.mlp_name,
            "mlp_class": type(self.mlp_module).__name__,
            "attention_delta_exact": self.attention_delta_exact,
            "mlp_delta_exact": self.mlp_delta_exact,
            "additional_residual_branch": self.additional_residual_branch,
            "post_residual_transform": self.post_residual_transform,
            "residual_change_equals_reported_branch_sum": bool(
                self.attention_delta_exact
                and self.mlp_delta_exact
                and self.additional_residual_branch is None
                and self.post_residual_transform is None
            ),
        }


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    if isinstance(value, dict):
        for item in value.values():
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
        return None
    if hasattr(value, "to_tuple"):
        try:
            return _first_tensor(value.to_tuple())
        except Exception:
            return None
    return None


def _direct_child(module: nn.Module, names: Iterable[str]) -> tuple[str, nn.Module] | None:
    children = dict(module.named_children())
    for name in names:
        child = children.get(name)
        if child is not None:
            return name, child
    return None


def _stack_priority(prefix: str) -> int:
    lowered = prefix.lower()
    preference = 0
    if any(token in lowered for token in ("language_model", "text_model", "decoder")):
        preference += 1000
    if any(token in lowered for token in ("vision", "visual", "encoder")):
        preference -= 1000
    return preference


def _discover_decoder_stack(model: nn.Module) -> tuple[str, list[_LayerSpec]]:
    grouped: dict[str, list[_LayerSpec]] = defaultdict(list)

    for module_name, module in model.named_modules():
        match = _LAYER_PATH_RE.match(module_name)
        if match is None:
            continue
        mixer = _direct_child(module, _MIXER_CHILD_NAMES)
        mlp = _direct_child(module, _MLP_CHILD_NAMES)
        if mixer is None or mlp is None:
            continue

        mixer_child_name, mixer_module = mixer
        mlp_child_name, mlp_module = mlp
        class_name = type(module).__name__.lower()
        # Linear-attention layers are the architecture's attention/token-mixing
        # branch and therefore implement the same requested Attn-delta stream.
        mixer_stream = "attention_delta"
        attention_delta_exact = False
        mlp_delta_exact = False
        additional_residual_branch = None
        post_residual_transform = None

        # Gemma3 normalizes each branch output after the branch module and before
        # residual addition. Hooking these post-branch norms gives the actual delta.
        if (
            "gemma3decoderlayer" in class_name
            or "gemma4textdecoderlayer" in class_name
            or "olmo3decoderlayer" in class_name
        ):
            post_attention_norm = getattr(module, "post_attention_layernorm", None)
            post_feedforward_norm = getattr(module, "post_feedforward_layernorm", None)
            if isinstance(post_attention_norm, nn.Module) and isinstance(post_feedforward_norm, nn.Module):
                mixer_child_name = "post_attention_layernorm"
                mixer_module = post_attention_norm
                mlp_child_name = "post_feedforward_layernorm"
                mlp_module = post_feedforward_norm
                attention_delta_exact = True
                mlp_delta_exact = True
            if "gemma4textdecoderlayer" in class_name and bool(
                getattr(module, "hidden_size_per_layer_input", 0)
            ):
                # Small Gemma 4 PLE models add another branch after the
                # feed-forward residual update. The three requested streams
                # remain exact, but they are not a complete branch breakdown.
                additional_residual_branch = "per_layer_input"
            if "gemma4textdecoderlayer" in class_name and hasattr(module, "layer_scalar"):
                # Gemma 4 scales the complete layer output after all residual
                # additions. The two hooked branch tensors remain the exact
                # pre-addition updates, but their sum is not the full change
                # between adjacent residual states.
                post_residual_transform = "layer_scalar"
        elif class_name in {
            "apertusdecoderlayer",
            "apertus1p5textdecoderlayer",
            "gptossdecoderlayer",
        }:
            # These pre-norm implementations add the direct attention and MLP
            # outputs to their residual streams without a post-branch transform.
            # GPT-OSS returns router scores alongside the MLP update; the hook
            # tensor extractor selects the update tensor without altering it.
            attention_delta_exact = True
            mlp_delta_exact = True
        elif "phi3decoderlayer" in class_name:
            # Phi-3/Phi-4 apply explicit residual dropout after each branch.
            # Hook the dropout outputs so exactness also holds outside eval mode.
            attention_dropout = getattr(module, "resid_attn_dropout", None)
            mlp_dropout = getattr(module, "resid_mlp_dropout", None)
            if isinstance(attention_dropout, nn.Module) and isinstance(mlp_dropout, nn.Module):
                mixer_child_name = "resid_attn_dropout"
                mixer_module = attention_dropout
                mlp_child_name = "resid_mlp_dropout"
                mlp_module = mlp_dropout
                attention_delta_exact = True
                mlp_delta_exact = True
        elif "qwen3_5decoderlayer" in class_name:
            # Qwen3.5 adds the mixer and MLP outputs directly to their residuals.
            attention_delta_exact = True
            mlp_delta_exact = True

        layer_index = int(match.group("index"))
        prefix = match.group("prefix")
        grouped[prefix].append(
            _LayerSpec(
                layer_index=layer_index,
                layer_name=module_name,
                layer_module=module,
                mixer_name=f"{module_name}.{mixer_child_name}",
                mixer_module=mixer_module,
                mixer_stream=mixer_stream,
                mlp_name=f"{module_name}.{mlp_child_name}",
                mlp_module=mlp_module,
                attention_delta_exact=attention_delta_exact,
                mlp_delta_exact=mlp_delta_exact,
                additional_residual_branch=additional_residual_branch,
                post_residual_transform=post_residual_transform,
            )
        )

    if not grouped:
        raise ValueError(
            "RepresentationDynamicsTracker could not find a decoder stack with indexed layers "
            "and direct attention/mixer plus MLP children."
        )

    priorities = {prefix: _stack_priority(prefix) for prefix in grouped}
    best_priority = max(priorities.values())
    candidates = sorted(prefix for prefix, priority in priorities.items() if priority == best_priority)
    if len(candidates) != 1:
        raise ValueError(
            "RepresentationDynamicsTracker found ambiguous decoder stacks with equal semantic priority: "
            f"{candidates}. Bind the tracker to the intended text-generating module explicitly."
        )
    selected_prefix = candidates[0]
    selected_specs = grouped[selected_prefix]
    selected_specs = sorted(selected_specs, key=lambda spec: spec.layer_index)
    indices = [spec.layer_index for spec in selected_specs]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Decoder stack {selected_prefix!r} contains duplicate layer indices: {indices}")
    return selected_prefix, selected_specs


def _normalize_string_list(value: Any, batch_size: int, default: str) -> list[str]:
    if value is None:
        return [default] * batch_size
    if isinstance(value, str):
        return [value] * batch_size
    normalized = [default if item is None else str(item) for item in value]
    if len(normalized) != batch_size:
        raise ValueError(f"Expected {batch_size} metadata values, received {len(normalized)}")
    return normalized


def build_representation_context(
    input_ids: torch.Tensor | None,
    attention_mask: torch.Tensor | None,
    *,
    task_names: Iterable[str] | str,
    sample_ids: Iterable[Any],
    split_names: Iterable[str] | str | None = None,
    pair_ids: Iterable[str | None] | None = None,
) -> dict[str, Any] | None:
    """Build per-example metadata used by representation hooks during generation."""

    if input_ids is None or not torch.is_tensor(input_ids):
        return None
    ids = input_ids.detach()
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2 or ids.numel() == 0:
        return None

    batch_size, sequence_length = int(ids.shape[0]), int(ids.shape[1])
    if attention_mask is not None and torch.is_tensor(attention_mask):
        valid_mask = attention_mask.detach().to(device=ids.device, dtype=torch.bool)
        if valid_mask.ndim == 1:
            valid_mask = valid_mask.unsqueeze(0)
        if valid_mask.shape != ids.shape:
            raise ValueError(
                f"Representation attention mask shape {tuple(valid_mask.shape)} does not match input IDs {tuple(ids.shape)}"
            )
    else:
        valid_mask = torch.ones_like(ids, dtype=torch.bool)

    tasks = _normalize_string_list(task_names, batch_size, "unknown")
    splits = _normalize_string_list(split_names, batch_size, "unknown")
    raw_sample_ids = list(sample_ids)
    if len(raw_sample_ids) != batch_size:
        raise ValueError(f"Expected {batch_size} sample IDs, received {len(raw_sample_ids)}")
    pairs = list(pair_ids) if pair_ids is not None else [None] * batch_size
    if len(pairs) != batch_size:
        raise ValueError(f"Expected {batch_size} pair IDs, received {len(pairs)}")

    normalized_sample_ids = [f"{tasks[index]}:{splits[index]}:{raw_sample_ids[index]}" for index in range(batch_size)]
    prompt_lengths = valid_mask.sum(dim=1).detach().cpu().to(dtype=torch.int64).tolist()
    return {
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "valid_mask": valid_mask,
        "task_names": tasks,
        "split_names": splits,
        "sample_ids": normalized_sample_ids,
        "pair_ids": [None if pair is None else str(pair) for pair in pairs],
        "prompt_lengths": [int(length) for length in prompt_lengths],
    }


class RepresentationDynamicsTracker:
    """Extract pooled per-example decoder representations without changing outputs."""

    SCHEMA_VERSION = 1

    def __init__(
        self,
        model: nn.Module,
        *,
        pooling: str = "last_valid",
        phases: Iterable[str] = ("prefill",),
        decode_aggregation: str = "mean_updates",
        decode_max_updates: int | None = None,
        storage_dtype: str = "float32",
    ) -> None:
        if pooling not in _VALID_POOLING:
            raise ValueError(f"pooling must be one of {sorted(_VALID_POOLING)}, got {pooling!r}")
        normalized_phases = tuple(dict.fromkeys(str(phase) for phase in phases))
        if not normalized_phases or any(phase not in _VALID_PHASES for phase in normalized_phases):
            raise ValueError(f"phases must be drawn from {sorted(_VALID_PHASES)}, got {normalized_phases!r}")
        if storage_dtype not in _VALID_STORAGE_DTYPES:
            raise ValueError(f"storage_dtype must be one of {sorted(_VALID_STORAGE_DTYPES)}, got {storage_dtype!r}")
        if decode_aggregation not in _VALID_DECODE_AGGREGATION:
            raise ValueError(
                "decode_aggregation must be one of "
                f"{sorted(_VALID_DECODE_AGGREGATION)}, got {decode_aggregation!r}"
            )
        if decode_max_updates is not None and int(decode_max_updates) < 1:
            raise ValueError("decode_max_updates must be positive or None")

        self.model = model
        self.pooling = pooling
        self.phases = normalized_phases
        self.decode_aggregation = decode_aggregation
        self.decode_max_updates = None if decode_max_updates is None else int(decode_max_updates)
        self.storage_dtype_name = storage_dtype
        self.storage_dtype = _VALID_STORAGE_DTYPES[storage_dtype]
        self.selected_stack, self.layer_specs = _discover_decoder_stack(model)
        self.hook_handles: list[Any] = []
        self.records: list[dict[str, Any]] = []
        self.skipped: Counter[str] = Counter()
        self.truncated: Counter[str] = Counter()
        self._context_stack: list[dict[str, Any]] = []
        self._context_counter = 0
        self._register_hooks()

    def _register_hooks(self) -> None:
        for spec in self.layer_specs:
            self.hook_handles.append(
                spec.mixer_module.register_forward_hook(
                    self._make_hook(
                        layer_index=spec.layer_index,
                        stream=spec.mixer_stream,
                        module_name=spec.mixer_name,
                        residual_delta_exact=spec.attention_delta_exact,
                    )
                )
            )
            self.hook_handles.append(
                spec.mlp_module.register_forward_hook(
                    self._make_hook(
                        layer_index=spec.layer_index,
                        stream="mlp_delta",
                        module_name=spec.mlp_name,
                        residual_delta_exact=spec.mlp_delta_exact,
                    )
                )
            )
            self.hook_handles.append(
                spec.layer_module.register_forward_hook(
                    self._make_hook(
                        layer_index=spec.layer_index,
                        stream="residual_state",
                        module_name=spec.layer_name,
                        residual_delta_exact=True,
                    )
                )
            )

    def _make_hook(self, *, layer_index: int, stream: str, module_name: str, residual_delta_exact: bool):
        def hook(_module, _inputs, output):
            tensor = _first_tensor(output)
            if tensor is not None:
                self._capture(
                    tensor,
                    layer_index=layer_index,
                    stream=stream,
                    module_name=module_name,
                    residual_delta_exact=residual_delta_exact,
                )

        return hook

    @contextmanager
    def track_context(self, context: dict[str, Any] | None):
        if not context:
            yield
            return
        normalized = dict(context)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else int(os.environ.get("RANK", 0))
        normalized["rank"] = rank
        normalized["context_id"] = f"rank{rank}:context{self._context_counter}"
        normalized["accumulators"] = {}
        normalized["decode_mean_updates"] = defaultdict(list)
        normalized["decode_update_counts"] = {}
        normalized["valid_decode_updates"] = None
        self._context_counter += 1
        self._context_stack.append(normalized)
        succeeded = False
        try:
            yield
            succeeded = True
        finally:
            active = self._context_stack.pop()
            if succeeded:
                self._finalize_context(active)
            else:
                self.skipped["failed_context"] += 1

    def _current_context(self) -> dict[str, Any] | None:
        return self._context_stack[-1] if self._context_stack else None

    def record_generated_tokens(
        self,
        generated_token_ids: torch.Tensor,
        *,
        eos_token_id: int | Iterable[int] | None,
        pad_token_id: int | None,
    ) -> None:
        """Bind response-valid cached updates to the active generation context.

        Prefill predicts the first generated token. A cached decode forward for
        generated token ``j`` exists only when generation continues to predict
        token ``j+1``. Updates at or after EOS/padding are therefore excluded.
        """

        context = self._current_context()
        if context is None:
            raise RuntimeError("Generated tokens must be recorded inside track_context")
        if not torch.is_tensor(generated_token_ids):
            raise TypeError("generated_token_ids must be a tensor")
        tokens = generated_token_ids.detach().cpu()
        if tokens.ndim == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.ndim != 2 or int(tokens.shape[0]) != int(context["batch_size"]):
            raise ValueError("Generated-token batch does not match the representation context")
        eos_ids = (
            set()
            if eos_token_id is None
            else {int(eos_token_id)}
            if isinstance(eos_token_id, int)
            else {int(value) for value in eos_token_id}
        )
        pad_id = None if pad_token_id is None else int(pad_token_id)
        valid_updates: list[int] = []
        for row in tokens.tolist():
            special_index = None
            for index, token in enumerate(row):
                if int(token) in eos_ids or (pad_id is not None and int(token) == pad_id):
                    special_index = index
                    break
            if special_index is None:
                valid_updates.append(max(len(row) - 1, 0))
            else:
                valid_updates.append(int(special_index))
        context["valid_decode_updates"] = valid_updates

    def _phase_for_tensor(self, tensor: torch.Tensor, context: dict[str, Any]) -> tuple[str, int, int] | None:
        batch_size = int(context["batch_size"])
        if tensor.ndim == 3:
            tensor_batch, sequence_length = int(tensor.shape[0]), int(tensor.shape[1])
        elif tensor.ndim == 2:
            tensor_batch, sequence_length = int(tensor.shape[0]), 1
        else:
            self.skipped["unsupported_tensor_rank"] += 1
            return None
        if tensor_batch != batch_size:
            self.skipped["batch_mismatch"] += 1
            return None
        prompt_length = int(context["sequence_length"])
        if sequence_length == prompt_length:
            return "prefill", tensor_batch, sequence_length
        if sequence_length == 1 and prompt_length != 1:
            return "decode", tensor_batch, sequence_length
        self.skipped["unknown_phase"] += 1
        return None

    def _pool(self, tensor: torch.Tensor, phase: str, context: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
        values = tensor.detach().float()
        if values.ndim == 2:
            values = values.unsqueeze(1)
        batch_size, sequence_length = int(values.shape[0]), int(values.shape[1])

        if phase == "decode":
            pooled = values[:, -1, :]
            valid_examples = torch.ones(batch_size, dtype=torch.bool, device=values.device)
            return pooled, valid_examples

        mask = context["valid_mask"].detach().to(device=values.device, dtype=torch.bool)
        if tuple(mask.shape) != (batch_size, sequence_length):
            raise ValueError(
                f"Representation valid mask shape {tuple(mask.shape)} does not match hooked tensor {(batch_size, sequence_length)}"
            )
        valid_examples = mask.any(dim=1)
        if self.pooling == "mean_valid":
            weights = mask.to(dtype=values.dtype).unsqueeze(-1)
            denominator = weights.sum(dim=1).clamp_min(1.0)
            pooled = (values * weights).sum(dim=1) / denominator
            return pooled, valid_examples

        positions = torch.arange(sequence_length, device=values.device).unsqueeze(0).expand(batch_size, sequence_length)
        last_indices = positions.masked_fill(~mask, -1).max(dim=1).values.clamp_min(0)
        pooled = values[torch.arange(batch_size, device=values.device), last_indices]
        return pooled, valid_examples

    def _capture(
        self,
        tensor: torch.Tensor,
        *,
        layer_index: int,
        stream: str,
        module_name: str,
        residual_delta_exact: bool,
    ) -> None:
        context = self._current_context()
        if context is None:
            return
        phase_info = self._phase_for_tensor(tensor, context)
        if phase_info is None:
            return
        phase, batch_size, _ = phase_info
        if phase not in self.phases:
            return

        with torch.no_grad():
            pooled, valid_examples = self._pool(tensor, phase, context)
            pooled_cpu = pooled.detach().cpu()
            valid_cpu = valid_examples.detach().cpu()
        accumulators = context["accumulators"]
        for sample_index in range(batch_size):
            if not bool(valid_cpu[sample_index].item()):
                self.skipped["empty_valid_mask"] += 1
                continue
            base_key = (sample_index, layer_index, stream, module_name, phase, residual_delta_exact)
            decode_step = None
            if phase == "decode":
                decode_step = int(context["decode_update_counts"].get(base_key, 0))
                context["decode_update_counts"][base_key] = decode_step + 1
            if phase == "decode" and self.decode_aggregation == "all_updates":
                if self.decode_max_updates is not None and decode_step is not None and decode_step >= self.decode_max_updates:
                    self.truncated["decode_updates_after_cap"] += 1
                    continue
            elif phase == "decode" and self.decode_aggregation == "first_update":
                decode_step = 0
            elif phase == "decode" and self.decode_aggregation == "mean_updates":
                context["decode_mean_updates"][base_key].append((int(decode_step), pooled_cpu[sample_index].clone()))
                continue
            key = (*base_key, decode_step)
            vector = pooled_cpu[sample_index]
            existing = accumulators.get(key)
            if existing is None:
                accumulators[key] = [vector.clone(), 1]
            else:
                if phase == "decode" and self.decode_aggregation in {"first_update", "all_updates"}:
                    continue
                if existing[0].shape != vector.shape:
                    self.skipped["hidden_size_mismatch"] += 1
                    continue
                existing[0].add_(vector)
                existing[1] += 1

    def _finalize_context(self, context: dict[str, Any]) -> None:
        has_decode = bool(context["decode_update_counts"])
        valid_decode_updates = context.get("valid_decode_updates")
        if has_decode and valid_decode_updates is None:
            raise RuntimeError(
                "Decode activations were captured without generated-token validity metadata; "
                "refusing to include possible EOS or padding updates."
            )
        for base_key, updates in context["decode_mean_updates"].items():
            sample_index = int(base_key[0])
            valid_count = int(valid_decode_updates[sample_index])
            retained = [vector for step, vector in updates if step < valid_count]
            if not retained:
                self.skipped["decode_after_eos_or_padding"] += len(updates)
                continue
            removed = len(updates) - len(retained)
            if removed:
                self.skipped["decode_after_eos_or_padding"] += removed
            context["accumulators"][(*base_key, None)] = [torch.stack(retained).sum(dim=0), len(retained)]

        for key, (vector_sum, update_count) in context["accumulators"].items():
            sample_index, layer_index, stream, module_name, phase, residual_delta_exact, decode_step = key
            if phase == "decode" and decode_step is not None:
                valid_count = int(valid_decode_updates[sample_index])
                if int(decode_step) >= valid_count:
                    self.skipped["decode_after_eos_or_padding"] += 1
                    continue
            averaged = vector_sum / max(int(update_count), 1)
            if not bool(torch.isfinite(averaged).all().item()):
                raise RuntimeError(
                    "Representation vector contains non-finite values before storage "
                    f"(sample={context['sample_ids'][sample_index]!r}, layer={layer_index}, "
                    f"stream={stream}, module={module_name!r}, phase={phase})."
                )
            max_abs = float(averaged.abs().max().item()) if averaged.numel() else 0.0
            vector = averaged.to(dtype=self.storage_dtype)
            if not bool(torch.isfinite(vector).all().item()):
                raise RuntimeError(
                    "Representation storage conversion produced non-finite values "
                    f"(dtype={self.storage_dtype_name}, max_abs_before_cast={max_abs:.9g}, "
                    f"sample={context['sample_ids'][sample_index]!r}, layer={layer_index}, "
                    f"stream={stream}, module={module_name!r}, phase={phase}). "
                    "Use float32 storage or investigate non-finite model activations."
                )
            self.records.append(
                {
                    "context_id": context["context_id"],
                    "rank": int(context["rank"]),
                    "sample_index": int(sample_index),
                    "sample_id": context["sample_ids"][sample_index],
                    "pair_id": context["pair_ids"][sample_index],
                    "task_name": context["task_names"][sample_index],
                    "split_name": context["split_names"][sample_index],
                    "prompt_length": int(context["prompt_lengths"][sample_index]),
                    "layer_index": int(layer_index),
                    "stream": stream,
                    "module_name": module_name,
                    "phase": phase,
                    "pooling": self.pooling,
                    "decode_aggregation": self.decode_aggregation,
                    "decode_step": decode_step,
                    "decode_max_updates": self.decode_max_updates,
                    "num_updates": int(update_count),
                    "hidden_size": int(vector.numel()),
                    "residual_delta_exact": bool(residual_delta_exact),
                    "vector": vector,
                }
            )

    def remove_hooks(self) -> None:
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()

    def export_state(self) -> dict[str, Any]:
        if self._context_stack:
            raise RuntimeError("Cannot export representation state while a tracking context is active")
        return {
            "schema_version": self.SCHEMA_VERSION,
            "selected_stack": self.selected_stack,
            "layer_specs": [spec.metadata() for spec in self.layer_specs],
            "pooling": self.pooling,
            "decode_aggregation": self.decode_aggregation,
            "decode_max_updates": self.decode_max_updates,
            "phases": list(self.phases),
            "storage_dtype": self.storage_dtype_name,
            "num_hooks": len(self.hook_handles),
            "num_contexts": self._context_counter,
            "num_records": len(self.records),
            "skipped": dict(self.skipped),
            "truncated": dict(self.truncated),
            "records": self.records,
        }


@contextmanager
def representation_tracker_context(model: Any, context: dict[str, Any] | None):
    tracker = getattr(model, "_lmms_eval_representation_tracker", None)
    if tracker is None or not hasattr(tracker, "track_context"):
        yield None
        return
    with tracker.track_context(context):
        yield tracker


def gather_representation_payload(local_payload: dict[str, Any]) -> dict[str, Any] | None:
    """Gather data-parallel representation records on rank zero.

    Representation payloads are much larger than pruning summaries. This helper is
    intended for small calibration runs; large distributed sweeps should write one
    shard per rank instead of gathering Python objects.
    """

    if not dist.is_available() or not dist.is_initialized():
        return local_payload
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    gathered: list[dict[str, Any] | None] | None = [None] * world_size if rank == 0 else None
    dist.gather_object(local_payload, gathered, dst=0)
    if rank != 0:
        return None
    assert gathered is not None
    payloads = [payload for payload in gathered if payload is not None]
    if not payloads:
        return None

    reference = payloads[0]
    compatibility_keys = (
        "schema_version",
        "selected_stack",
        "pooling",
        "phases",
        "decode_aggregation",
        "decode_max_updates",
        "storage_dtype",
        "layer_specs",
    )
    for payload in payloads[1:]:
        for key in compatibility_keys:
            if payload.get(key) != reference.get(key):
                raise ValueError(f"Cannot merge representation payloads with different {key}")

    merged = {
        key: value
        for key, value in reference.items()
        if key not in {"records", "skipped", "truncated", "num_contexts", "num_records"}
    }
    merged_records: list[dict[str, Any]] = []
    merged_skipped: Counter[str] = Counter()
    merged_truncated: Counter[str] = Counter()
    total_contexts = 0
    for payload in payloads:
        merged_records.extend(payload.get("records", []))
        merged_skipped.update(payload.get("skipped", {}))
        merged_truncated.update(payload.get("truncated", {}))
        total_contexts += int(payload.get("num_contexts", 0))
    merged.update(
        {
            "records": merged_records,
            "skipped": dict(merged_skipped),
            "truncated": dict(merged_truncated),
            "num_contexts": total_contexts,
            "num_records": len(merged_records),
            "world_size": world_size,
        }
    )
    return merged
