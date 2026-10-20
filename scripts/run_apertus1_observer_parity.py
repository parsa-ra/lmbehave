#!/usr/bin/env python3
"""Run same-process observer parity for frozen Apertus 1.0 checkpoints.

This is an engineering gate, not a benchmark result. It loads one immutable revision,
checks the selected decoder stack, compares logits and greedy token IDs with monitoring
off/on/removed, and writes a sample-free JSON report only after every assertion passes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmms_eval.representations import RepresentationDynamicsTracker, build_representation_context


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "experiments" / "apertus1_longitudinal_v1.json"
DEFAULT_OUTPUT_ROOT = ROOT / "logs" / "apertus1-observer-parity-v1"
PROMPTS = (
    "Question: Which organ pumps blood through the human body?\nAnswer:",
    "Question: If x + 7 = 12, what is x?\nAnswer:",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str | None:
    result = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=path,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def git_dirty(path: Path) -> bool | None:
    result = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=path,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return bool(result.stdout.strip()) if result.returncode == 0 else None


def count_forward_hooks(model: torch.nn.Module) -> int:
    return sum(len(module._forward_hooks) for module in model.modules())


def xielu_runtime(model: torch.nn.Module) -> dict[str, Any]:
    activations = [
        module
        for module in model.modules()
        if type(module).__name__ == "XIELUActivation"
    ]
    fused = [getattr(module, "_xielu_cuda_obj", None) is not None for module in activations]
    return {
        "activation_modules": len(activations),
        "fused_cuda_modules": sum(fused),
        "implementation": "fused_cuda" if fused and all(fused) else "python_fallback",
    }


def tensor_digest(tensor: torch.Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    return hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest()


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def checkpoint_from_manifest(manifest: dict[str, Any], checkpoint_id: str) -> dict[str, Any]:
    matches = [item for item in manifest["checkpoints"] if item["id"] == checkpoint_id]
    if len(matches) != 1:
        known = ", ".join(item["id"] for item in manifest["checkpoints"])
        raise ValueError(f"Unknown checkpoint {checkpoint_id!r}; choose one of: {known}")
    return matches[0]


def exact_stack_report(payload: dict[str, Any], expected_layers: int) -> dict[str, Any]:
    specs = payload["layer_specs"]
    indices = [int(spec["layer_index"]) for spec in specs]
    layer_classes = sorted({str(spec["layer_class"]) for spec in specs})
    attention_exact = all(bool(spec["attention_delta_exact"]) for spec in specs)
    mlp_exact = all(bool(spec["mlp_delta_exact"]) for spec in specs)
    contiguous = indices == list(range(expected_layers))
    direct_sites = all(
        str(spec["mixer_name"]).endswith(".self_attn")
        and str(spec["mlp_name"]).endswith(".mlp")
        for spec in specs
    )
    passed = (
        len(specs) == expected_layers
        and contiguous
        and attention_exact
        and mlp_exact
        and direct_sites
        and layer_classes == ["ApertusDecoderLayer"]
    )
    return {
        "passed": passed,
        "selected_stack": payload["selected_stack"],
        "layers": len(specs),
        "indices_contiguous": contiguous,
        "layer_classes": layer_classes,
        "all_attention_delta_exact": attention_exact,
        "all_mlp_delta_exact": mlp_exact,
        "all_sites_are_direct_self_attention_and_mlp": direct_sites,
        "hooks": int(payload["num_hooks"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This smoke requires exactly one visible CUDA device")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")

    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = checkpoint_from_manifest(manifest, args.checkpoint)
    output_dir = args.output_root.resolve() / checkpoint["id"]
    report_path = output_dir / "observer_parity.json"
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing smoke output: {output_dir}")
    output_dir.mkdir(parents=True)

    torch.manual_seed(int(manifest["extraction"]["seed"]))
    repository = checkpoint["repository"]
    revision = checkpoint["revision"]
    tokenizer = AutoTokenizer.from_pretrained(repository, revision=revision)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Apertus tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    encoded = tokenizer(list(PROMPTS), return_tensors="pt", padding=True)

    model = AutoModelForCausalLM.from_pretrained(
        repository,
        revision=revision,
        dtype=torch.bfloat16,
        device_map={"": 0},
    ).eval()
    activation_runtime = xielu_runtime(model)
    if activation_runtime["activation_modules"] != int(manifest["model"]["layers"]):
        raise RuntimeError(f"Unexpected xIELU activation layout: {activation_runtime}")
    model_device = next(model.parameters()).device
    encoded = {key: value.to(model_device) for key, value in encoded.items()}
    generation_kwargs = {
        **encoded,
        "do_sample": False,
        "max_new_tokens": args.max_new_tokens,
        "use_cache": True,
        "pad_token_id": tokenizer.pad_token_id,
    }

    with torch.inference_mode():
        baseline_logits = model(**encoded, use_cache=False).logits
        baseline_tokens = model.generate(**generation_kwargs)
    synchronize()
    hooks_before = count_forward_hooks(model)
    if hooks_before != 0:
        raise RuntimeError(f"Expected no pre-existing forward hooks, found {hooks_before}")

    tracker = RepresentationDynamicsTracker(
        model,
        pooling="last_valid",
        phases=("prefill",),
        storage_dtype="float32",
    )
    stack = exact_stack_report(tracker.export_state(), int(manifest["model"]["layers"]))
    expected_hooks = int(manifest["model"]["layers"]) * 3
    if not stack["passed"] or stack["hooks"] != expected_hooks:
        tracker.remove_hooks()
        raise RuntimeError(f"Apertus decoder-stack audit failed: {json.dumps(stack, sort_keys=True)}")

    forward_context = build_representation_context(
        encoded["input_ids"],
        encoded.get("attention_mask"),
        task_names=("knowledge-smoke", "reasoning-smoke"),
        sample_ids=("forward-0", "forward-1"),
        split_names="smoke",
    )
    generation_context = build_representation_context(
        encoded["input_ids"],
        encoded.get("attention_mask"),
        task_names=("knowledge-smoke", "reasoning-smoke"),
        sample_ids=("generate-0", "generate-1"),
        split_names="smoke",
    )
    try:
        with torch.inference_mode(), tracker.track_context(forward_context):
            observed_logits = model(**encoded, use_cache=False).logits
        with torch.inference_mode(), tracker.track_context(generation_context):
            observed_tokens = model.generate(**generation_kwargs)
        synchronize()
        payload = tracker.export_state()
    finally:
        tracker.remove_hooks()

    hooks_after_removal = count_forward_hooks(model)
    with torch.inference_mode():
        removed_logits = model(**encoded, use_cache=False).logits
        removed_tokens = model.generate(**generation_kwargs)
    synchronize()

    records = payload["records"]
    expected_records = len(PROMPTS) * int(manifest["model"]["layers"]) * 3 * 2
    finite_records = all(bool(torch.isfinite(record["vector"]).all().item()) for record in records)
    exact_records = all(bool(record["residual_delta_exact"]) for record in records)
    streams = sorted({str(record["stream"]) for record in records})
    hidden_sizes = sorted({int(record["hidden_size"]) for record in records})
    phases = sorted({str(record["phase"]) for record in records})
    num_updates = sorted({int(record["num_updates"]) for record in records})
    logits_on_equal = torch.equal(baseline_logits, observed_logits)
    logits_removed_equal = torch.equal(baseline_logits, removed_logits)
    tokens_on_equal = torch.equal(baseline_tokens, observed_tokens)
    tokens_removed_equal = torch.equal(baseline_tokens, removed_tokens)
    passed = all(
        (
            stack["passed"],
            logits_on_equal,
            logits_removed_equal,
            tokens_on_equal,
            tokens_removed_equal,
            hooks_after_removal == 0,
            len(records) == expected_records,
            finite_records,
            exact_records,
            not payload["skipped"],
            streams == ["attention_delta", "mlp_delta", "residual_state"],
            hidden_sizes == [int(manifest["model"]["hidden_width"])],
            phases == ["prefill"],
            num_updates == [1],
        )
    )
    report = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "engineering observer-parity gate; not benchmark evidence",
        "manifest": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "experiment_id": manifest["experiment_id"],
        "checkpoint": checkpoint,
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0),
            "transformers_git_head": git_head(ROOT / "transformers"),
            "transformers_git_dirty": git_dirty(ROOT / "transformers"),
            "lmms_eval_git_head": git_head(ROOT / "lmms-eval"),
            "lmms_eval_git_dirty": git_dirty(ROOT / "lmms-eval"),
            "observer_script_sha256": sha256_file(Path(__file__).resolve()),
            "xielu": activation_runtime,
        },
        "stack": stack,
        "parity": {
            "logits_equal_observer_on": logits_on_equal,
            "logits_equal_after_removal": logits_removed_equal,
            "greedy_tokens_equal_observer_on": tokens_on_equal,
            "greedy_tokens_equal_after_removal": tokens_removed_equal,
            "baseline_logits_sha256": tensor_digest(baseline_logits),
            "observed_logits_sha256": tensor_digest(observed_logits),
            "removed_logits_sha256": tensor_digest(removed_logits),
            "baseline_tokens_sha256": tensor_digest(baseline_tokens),
            "observed_tokens_sha256": tensor_digest(observed_tokens),
            "removed_tokens_sha256": tensor_digest(removed_tokens),
            "generated_continuations": [
                tokenizer.decode(row[encoded["input_ids"].shape[1] :], skip_special_tokens=False)
                for row in baseline_tokens
            ],
        },
        "capture": {
            "expected_records": expected_records,
            "observed_records": len(records),
            "finite": finite_records,
            "exact": exact_records,
            "streams": streams,
            "hidden_sizes": hidden_sizes,
            "phases": phases,
            "num_updates": num_updates,
            "skipped": payload["skipped"],
            "truncated": payload["truncated"],
        },
        "cleanup": {
            "hooks_before": hooks_before,
            "hooks_registered": stack["hooks"],
            "hooks_after_removal": hooks_after_removal,
        },
    }
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("Apertus 1.0 observer parity failed")


if __name__ == "__main__":
    main()
