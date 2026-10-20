from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from lmms_eval.pruning import PruningTracker, configure_model_for_explicit_attention, summarize_pruning_payload


DEFAULT_TEXTS = [
    "Structured pruning can reduce inference cost when the saliency signal is reliable.",
    "Forward hooks let you collect activation statistics without modifying every model implementation.",
]

MODEL_LOADERS = {
    "causal_lm": AutoModelForCausalLM,
    "auto_model": AutoModel,
}

DTYPE_MAP = {
    "auto": None,
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone Hugging Face example for lmms_eval.pruning.PruningTracker.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="Hugging Face model identifier or local path.",
    )
    parser.add_argument(
        "--model-class",
        choices=sorted(MODEL_LOADERS),
        default="causal_lm",
        help="Which AutoModel loader to use.",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Calibration text. Repeat --text to run multiple forwards.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=128,
        help="Tokenizer truncation length.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./hf_pruning_tracker_dump",
        help="Directory where the tracker payload and summaries will be saved.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPE_MAP),
        default="auto",
        help="torch_dtype passed to from_pretrained when supported.",
    )
    parser.add_argument(
        "--output-attentions",
        action="store_true",
        help="Ask the HF model to expose explicit attention maps when supported.",
    )
    parser.add_argument(
        "--attn-implementation",
        choices=["none", "eager", "sdpa", "flash_attention_2"],
        default="eager",
        help="Attention implementation passed to from_pretrained. Use eager for explicit attention maps when the model supports it.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Forwarded to AutoTokenizer/AutoModel.from_pretrained.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional HF revision or git ref.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> Path | None:
    if not rows:
        return None

    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})
    return path


def load_tokenizer(model_id: str, trust_remote_code: bool, revision: str | None):
    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
    }
    if revision:
        kwargs["revision"] = revision
    return AutoTokenizer.from_pretrained(model_id, **kwargs)


def load_model(
    model_id: str,
    model_class: str,
    dtype_name: str,
    attn_implementation: str,
    trust_remote_code: bool,
    revision: str | None,
) -> torch.nn.Module:
    loader = MODEL_LOADERS[model_class]
    kwargs: dict[str, Any] = {
        "trust_remote_code": trust_remote_code,
        "low_cpu_mem_usage": True,
    }
    if revision:
        kwargs["revision"] = revision
    if DTYPE_MAP[dtype_name] is not None:
        kwargs["torch_dtype"] = DTYPE_MAP[dtype_name]
    if attn_implementation != "none":
        kwargs["attn_implementation"] = attn_implementation
    return loader.from_pretrained(model_id, **kwargs)


def run_forward_passes(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    device: torch.device,
    max_length: int,
    output_attentions: bool,
) -> None:
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        inputs = {name: tensor.to(device) for name, tensor in inputs.items()}

        forward_kwargs: dict[str, Any] = {"return_dict": True}
        if output_attentions:
            forward_kwargs["output_attentions"] = True
        if hasattr(getattr(model, "config", None), "use_cache"):
            forward_kwargs["use_cache"] = False

        try:
            with torch.inference_mode():
                model(**inputs, **forward_kwargs)
        except TypeError:
            if output_attentions and "output_attentions" in forward_kwargs:
                forward_kwargs.pop("output_attentions", None)
                with torch.inference_mode():
                    model(**inputs, **forward_kwargs)
            else:
                raise


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    texts = args.text or list(DEFAULT_TEXTS)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(
        args.model_id,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    model = load_model(
        args.model_id,
        model_class=args.model_class,
        dtype_name=args.dtype,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        revision=args.revision,
    )
    model.eval()
    model.to(device)
    explicit_attention_config = None
    if args.output_attentions:
        explicit_attention_config = configure_model_for_explicit_attention(model, force_output_attentions=True)

    tracker = PruningTracker(model)
    try:
        run_forward_passes(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            device=device,
            max_length=args.max_length,
            output_attentions=args.output_attentions,
        )
        payload = tracker.export_state()
    finally:
        tracker.remove_hooks()

    summary = summarize_pruning_payload(payload, include_module_rows=True)
    metadata = {
        "model_id": args.model_id,
        "model_class": args.model_class,
        "device": str(device),
        "dtype": args.dtype,
        "num_texts": len(texts),
        "max_length": args.max_length,
        "output_attentions_requested": bool(args.output_attentions),
        "attn_implementation": args.attn_implementation,
        "explicit_attention_config": explicit_attention_config,
    }

    torch.save(
        {
            "metadata": metadata,
            **payload,
            "summary": summarize_pruning_payload(payload),
        },
        output_dir / "tracker_payload.pt",
    )
    write_json(
        output_dir / "summary.json",
        {
            "metadata": metadata,
            "summary": summary,
        },
    )

    write_csv(output_dir / "mlp_modules.csv", summary.get("mlp", {}).get("module_rows", []))
    write_csv(output_dir / "attention_modules.csv", summary.get("attention", {}).get("module_rows", []))
    write_csv(output_dir / "linear_state_modules.csv", summary.get("linear_state", {}).get("module_rows", []))
    write_csv(output_dir / "linear_state_components.csv", summary.get("linear_state_components", {}).get("module_rows", []))
    write_csv(output_dir / "network_sequence.csv", summary.get("network", {}).get("module_sequence", []))

    print(f"Saved tracker payload to {output_dir / 'tracker_payload.pt'}")
    print(f"Saved summary JSON to {output_dir / 'summary.json'}")
    print(f"Tracked MLP modules: {summary.get('mlp', {}).get('num_modules', 0)}")
    print(f"Tracked explicit attention modules: {summary.get('attention', {}).get('num_modules', 0)}")
    print(f"Tracked linear-state modules: {summary.get('linear_state', {}).get('num_modules', 0)}")
    print(f"Tracked linear-state components: {summary.get('linear_state_components', {}).get('num_modules', 0)}")


if __name__ == "__main__":
    main()
