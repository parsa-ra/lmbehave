import types
import unittest
from unittest.mock import patch

import torch
from transformers import (
    ApertusConfig,
    ApertusForCausalLM,
    GptOssConfig,
    GptOssForCausalLM,
    Olmo3Config,
    Olmo3ForCausalLM,
    Phi3Config,
    Phi3ForCausalLM,
)

from lmms_eval.models.simple.hf_causal import HFCausal
from lmms_eval.representations import RepresentationDynamicsTracker


class _FakeBatch(dict):
    def to(self, device):
        return _FakeBatch({key: value.to(device) for key, value in self.items()})


class _FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<eos>"
    padding_side = "right"
    chat_template = "fake"

    def encode(self, text, **kwargs):
        return [ord(char) % 17 + 1 for char in text]

    def decode(self, token_id):
        return "<eos>"

    def __call__(self, texts, **kwargs):
        self.call_kwargs = kwargs
        rows = [[index + 1, index + 2] for index, _ in enumerate(texts)]
        return _FakeBatch(
            {
                "input_ids": torch.tensor(rows, dtype=torch.long),
                "attention_mask": torch.ones((len(rows), 2), dtype=torch.long),
            }
        )

    def batch_decode(self, rows, **kwargs):
        return ["answer STOP ignored" for _ in rows]

    def apply_chat_template(self, messages, **kwargs):
        self.chat_messages = messages
        self.chat_kwargs = kwargs
        return f"<user>{messages[0]['content']}</user><assistant>"


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.config = types.SimpleNamespace(max_position_embeddings=128)

    def generate(self, input_ids, attention_mask, **kwargs):
        suffix = torch.full((input_ids.shape[0], 1), 9, dtype=input_ids.dtype, device=input_ids.device)
        return torch.cat((input_ids, suffix), dim=1)


class _FakeAccelerator:
    num_processes = 1

    def unwrap_model(self, model):
        return model


class TestHFCausal(unittest.TestCase):
    def _make_model(self):
        model = HFCausal.__new__(HFCausal)
        model.accelerator = _FakeAccelerator()
        model._model = _FakeModel()
        model._tokenizer = _FakeTokenizer()
        model._device = torch.device("cpu")
        model.device_map = "cpu"
        model.batch_size_per_gpu = 2
        model.use_cache = True
        model.apply_chat_template = False
        model._rank = 0
        model._world_size = 1
        model.task_dict = {"demo": {"test": [{"id": 0}, {"id": 1}]}}
        model.cache_hook = types.SimpleNamespace(add_partial=lambda *args, **kwargs: None)
        return model

    def test_generate_until_uses_plain_text_and_representation_context(self):
        model = self._make_model()
        model._lmms_eval_representation_tracker = object()
        requests = [
            types.SimpleNamespace(args=(f"question {index}", {"until": ["STOP"]}, lambda doc: [], index, "demo", "test"))
            for index in range(2)
        ]

        with (
            patch("lmms_eval.models.simple.hf_causal.build_representation_context", return_value={"context": True}) as build_context,
            patch("lmms_eval.models.simple.hf_causal.representation_tracker_context") as tracker_context,
        ):
            outputs = model.generate_until(requests)

        self.assertEqual(outputs, ["answer ", "answer "])
        self.assertFalse(model.tokenizer.call_kwargs["return_token_type_ids"])
        build_context.assert_called_once()
        self.assertEqual(build_context.call_args.kwargs["task_names"], ["demo", "demo"])
        self.assertEqual(build_context.call_args.kwargs["sample_ids"], [0, 1])
        self.assertIs(tracker_context.call_args.args[0], model)

    def test_generate_until_applies_chat_template_only_when_requested(self):
        model = self._make_model()
        model.apply_chat_template = True
        requests = [
            types.SimpleNamespace(args=("question", {"until": ["STOP"]}, lambda doc: [], 0, "demo", "test"))
        ]

        outputs = model.generate_until(requests)

        self.assertEqual(outputs, ["answer "])
        self.assertEqual(model.tokenizer.chat_messages, [{"role": "user", "content": "question"}])
        self.assertTrue(model.tokenizer.chat_kwargs["add_generation_prompt"])

    def test_loglikelihood_returns_summed_continuation_loss(self):
        model = self._make_model()
        config = Olmo3Config(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            vocab_size=128,
            pad_token_id=0,
            eos_token_id=127,
        )
        torch.manual_seed(0)
        model._model = Olmo3ForCausalLM(config).eval()
        model._config = model._model.config
        continuations = [" A", " longer"]
        requests = [
            types.SimpleNamespace(
                args=("question", continuation, lambda doc: [], index, "demo", "test")
            )
            for index, continuation in enumerate(continuations)
        ]

        outputs = model.loglikelihood(requests)

        self.assertEqual(len(outputs), 2)
        self.assertTrue(all(loss > 0 for loss, _is_greedy in outputs))
        for continuation, (observed_loss, observed_greedy) in zip(continuations, outputs):
            context_ids = model.tokenizer.encode("question", add_special_tokens=False)
            combined_ids = model.tokenizer.encode("question" + continuation, add_special_tokens=False)
            input_ids = torch.tensor([combined_ids])
            with torch.inference_mode():
                logits = model.model(input_ids=input_ids, use_cache=False).logits.float()
            start = len(context_ids)
            prediction = logits[0, start - 1 : len(combined_ids) - 1]
            targets = input_ids[0, start:]
            expected_loss = torch.nn.functional.cross_entropy(prediction, targets, reduction="sum").item()
            self.assertAlmostEqual(observed_loss, expected_loss, places=5)
            self.assertEqual(observed_greedy, bool(torch.equal(prediction.argmax(dim=-1), targets)))

    def test_loglikelihood_rejects_ambiguous_representation_collection(self):
        model = self._make_model()
        model._lmms_eval_representation_tracker = object()
        with self.assertRaisesRegex(ValueError, "one request per answer choice"):
            model.loglikelihood([])

    def test_olmo3_uses_exact_normalized_residual_branches(self):
        config = Olmo3Config(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            vocab_size=128,
        )
        model = Olmo3ForCausalLM(config)
        tracker = RepresentationDynamicsTracker(model, storage_dtype="float32")
        try:
            self.assertEqual(tracker.selected_stack, "model.layers")
            self.assertEqual(len(tracker.layer_specs), 2)
            for spec in tracker.layer_specs:
                self.assertTrue(spec.attention_delta_exact)
                self.assertTrue(spec.mlp_delta_exact)
                self.assertTrue(spec.mixer_name.endswith("post_attention_layernorm"))
                self.assertTrue(spec.mlp_name.endswith("post_feedforward_layernorm"))
        finally:
            tracker.remove_hooks()

    def test_additional_text_families_have_exact_branch_hooks(self):
        common = dict(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=128,
            pad_token_id=0,
        )
        models = [
            ApertusForCausalLM(ApertusConfig(**common, head_dim=16)),
            GptOssForCausalLM(
                GptOssConfig(
                    **common,
                    head_dim=16,
                    num_local_experts=4,
                    num_experts_per_tok=2,
                )
            ),
            Phi3ForCausalLM(Phi3Config(**common)),
        ]
        for model in models:
            tracker = RepresentationDynamicsTracker(model, storage_dtype="float32")
            try:
                self.assertEqual(len(tracker.layer_specs), 2)
                self.assertTrue(all(spec.attention_delta_exact for spec in tracker.layer_specs))
                self.assertTrue(all(spec.mlp_delta_exact for spec in tracker.layer_specs))
            finally:
                tracker.remove_hooks()

    def test_tiny_olmo3_generation_collects_complete_prefill_records(self):
        wrapper = self._make_model()
        config = Olmo3Config(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=16,
            vocab_size=128,
            pad_token_id=0,
            eos_token_id=127,
        )
        torch.manual_seed(0)
        wrapper._model = Olmo3ForCausalLM(config).eval()
        tracker = RepresentationDynamicsTracker(wrapper.model, storage_dtype="float32")
        wrapper._lmms_eval_representation_tracker = tracker
        requests = [
            types.SimpleNamespace(args=(f"question {index}", {"max_new_tokens": 1}, lambda doc: [], index, "demo", "test"))
            for index in range(2)
        ]
        try:
            outputs = wrapper.generate_until(requests)
            payload = tracker.export_state()
        finally:
            tracker.remove_hooks()
            del wrapper._lmms_eval_representation_tracker

        self.assertEqual(outputs, ["answer STOP ignored", "answer STOP ignored"])
        self.assertEqual(payload["num_contexts"], 1)
        self.assertEqual(payload["num_records"], 2 * 2 * 3)
        self.assertEqual(payload["skipped"], {})
        self.assertEqual(
            {(record["layer_index"], record["stream"]) for record in payload["records"]},
            {
                (layer, stream)
                for layer in range(2)
                for stream in ("attention_delta", "mlp_delta", "residual_state")
            },
        )
        self.assertTrue(all(record["residual_delta_exact"] for record in payload["records"]))


if __name__ == "__main__":
    unittest.main()
