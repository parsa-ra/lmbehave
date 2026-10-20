import types
import unittest
from unittest.mock import patch

import torch
from lmms_eval.models.simple.apertus1p5 import Apertus1p5


class _FakeBatch(dict):
    def to(self, device):
        return _FakeBatch({key: value.to(device) for key, value in self.items()})


class _FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text):
        return [1, 2]

    def decode(self, token_id):
        return "<eos>"


class _FakeProcessor:
    tokenizer = _FakeTokenizer()

    def apply_chat_template(self, conversations, **kwargs):
        self.template_input = conversations
        self.template_kwargs = kwargs
        if not kwargs.get("tokenize", False):
            return "rendered"
        batch = len(conversations)
        return _FakeBatch(
            {
                "input_ids": torch.tensor([[1, 2]] * batch),
                "attention_mask": torch.ones((batch, 2), dtype=torch.long),
            }
        )

    def batch_decode(self, rows, **kwargs):
        return ["answer STOP ignored" for _ in rows]


class _FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))

    @property
    def device(self):
        return self.anchor.device

    def generate(self, input_ids, attention_mask, **kwargs):
        suffix = torch.full((input_ids.shape[0], 1), 9, dtype=input_ids.dtype)
        return torch.cat((input_ids, suffix), dim=1)


class _FakeAccelerator:
    def unwrap_model(self, model):
        return model


class TestApertus1p5(unittest.TestCase):
    def _make_model(self):
        model = Apertus1p5.__new__(Apertus1p5)
        model.accelerator = _FakeAccelerator()
        model._model = _FakeModel()
        model.processor = _FakeProcessor()
        model._tokenizer = model.processor.tokenizer
        model._device = torch.device("cpu")
        model.device_map = "cpu"
        model.batch_size_per_gpu = 2
        model.use_cache = True
        model.enable_thinking = False
        model.system_prompt = None
        model._rank = 0
        model._world_size = 1
        model.task_dict = {"demo": {"test": [{"id": 0}, {"id": 1}]}}
        model.cache_hook = types.SimpleNamespace(
            add_partial=lambda *args, **kwargs: None
        )
        return model

    def test_text_generation_uses_fixed_nonthinking_template_and_context(self):
        model = self._make_model()
        model._lmms_eval_representation_tracker = object()
        requests = [
            types.SimpleNamespace(
                args=(
                    f"question {index}",
                    {"until": ["STOP"]},
                    lambda doc: [],
                    index,
                    "demo",
                    "test",
                )
            )
            for index in range(2)
        ]

        with (
            patch(
                "lmms_eval.models.simple.apertus1p5.build_representation_context",
                return_value={"context": True},
            ) as build_context,
            patch(
                "lmms_eval.models.simple.apertus1p5.representation_tracker_context"
            ) as tracker_context,
        ):
            outputs = model.generate_until(requests)

        self.assertEqual(outputs, ["answer ", "answer "])
        self.assertFalse(model.processor.template_kwargs["enable_thinking"])
        self.assertEqual(
            model.processor.template_kwargs["processor_kwargs"], {"padding": True}
        )
        build_context.assert_called_once()
        self.assertEqual(build_context.call_args.kwargs["task_names"], ["demo", "demo"])
        self.assertEqual(build_context.call_args.kwargs["sample_ids"], [0, 1])
        self.assertIs(tracker_context.call_args.args[0], model)

    def test_media_is_rejected_until_a_multimodal_protocol_is_frozen(self):
        model = self._make_model()
        request = types.SimpleNamespace(
            args=("question", {}, lambda doc: [object()], 0, "demo", "test")
        )
        with self.assertRaisesRegex(ValueError, "text-only"):
            model.generate_until([request])

    def test_thinking_mode_is_not_silently_enabled(self):
        model = Apertus1p5.__new__(Apertus1p5)
        with (
            self.assertRaisesRegex(ValueError, "enable_thinking=false"),
            patch(
                "lmms_eval.models.simple.apertus1p5.lmms.__init__", return_value=None
            ),
        ):
            Apertus1p5.__init__(model, enable_thinking=True)
