import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

from lmms_eval.models.simple.qwen3_vl import Qwen3_VL
from lmms_eval.models.chat.qwen3_vl import Qwen3_VL as Qwen3_VL_Chat


class _FakeTokenizer:
    eos_token_id = 0
    pad_token_id = 0

    def encode(self, text):
        return [1, 2, 3]

    def decode(self, token_id):
        return "<eos>"


class _FakeInputs(dict):
    def __init__(self):
        super().__init__(input_ids=torch.tensor([[10, 11]]))

    @property
    def input_ids(self):
        return self["input_ids"]

    def to(self, device):
        return self


class _FakeProcessor:
    def __init__(self):
        self.calls = []

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return ["prompt"]

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return _FakeInputs()

    def batch_decode(self, generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        return ["final answer"]


class _FakeModel:
    def generate(self, **kwargs):
        return torch.tensor([[10, 11, 12]])


class _FakeFrame:
    def asnumpy(self):
        return np.zeros((2, 2, 3), dtype=np.uint8)


class _FakeVideoReader:
    def __init__(self, path):
        self.path = path

    def __getitem__(self, index):
        return _FakeFrame()


class _VideoMetadata:
    def __init__(self, frames_indices):
        self.frames_indices = np.asarray(frames_indices)


class TestQwen3VLSimple(unittest.TestCase):
    def _make_model(self, max_num_frames=3):
        model = Qwen3_VL.__new__(Qwen3_VL)
        model._tokenizer = _FakeTokenizer()
        model.processor = _FakeProcessor()
        model._model = _FakeModel()
        model.max_pixels = 1024
        model.min_pixels = 256
        model.max_num_frames = max_num_frames
        model.fps = None
        model.total_pixels = None
        model.system_prompt = "You are a helpful assistant."
        model.interleave_visuals = False
        model.enable_thinking = None
        model.reasoning_prompt = None
        model.batch_size_per_gpu = 1
        model.use_cache = False
        model.device_map = "cpu"
        model._device = torch.device("cpu")
        model._rank = 0
        model._world_size = 1
        model.task_dict = {"demo_task": {"test": [{"id": 0}]}}
        model.cache_hook = types.SimpleNamespace(add_partial=lambda *args, **kwargs: None)
        return model

    def test_generate_until_passes_video_metadata_and_kwargs_to_processor(self):
        model = self._make_model(max_num_frames=3)
        metadata = _VideoMetadata([0, 10, 20, 30, 40])
        video_tensor = torch.arange(20, dtype=torch.float32).reshape(5, 4)
        request = types.SimpleNamespace(
            args=("Describe the video", {}, lambda doc: ["demo.mp4"], 0, "demo_task", "test"),
        )

        with (
            patch("lmms_eval.models.simple.qwen3_vl.process_vision_info", return_value=(None, [(video_tensor.clone(), metadata)], {"fps": 30.0, "max_frames": 5})),
            patch(
                "lmms_eval.models.simple.qwen3_vl.decord.VideoReader",
                _FakeVideoReader,
            ),
        ):
            result = model.generate_until([request])

        self.assertEqual(result, ["final answer"])
        self.assertEqual(len(model.processor.calls), 1)

        processor_call = model.processor.calls[0]
        expected_indices = np.array([0, 2, 4])
        self.assertTrue(torch.equal(processor_call["videos"][0], video_tensor[expected_indices]))
        self.assertIs(processor_call["video_metadata"][0], metadata)
        self.assertEqual(processor_call["fps"], 30.0)
        self.assertEqual(processor_call["max_frames"], 5)
        self.assertTrue(np.array_equal(metadata.frames_indices, np.array([0, 20, 40])))

    def test_generate_until_passes_task_name_when_pruning_tracker_is_attached(self):
        model = self._make_model()
        model._model._lmms_eval_pruning_tracker = object()
        request = types.SimpleNamespace(
            args=("Describe the image", {}, lambda doc: [], 0, "demo_task", "test"),
        )

        with (
            patch("lmms_eval.models.simple.qwen3_vl.process_vision_info", return_value=(None, None, {})),
            patch("lmms_eval.models.simple.qwen3_vl.infer_token_group_context", return_value=None) as infer_context,
        ):
            result = model.generate_until([request])

        self.assertEqual(result, ["final answer"])
        infer_context.assert_called_once()
        self.assertEqual(infer_context.call_args.kwargs["task_name"], "demo_task")

    def test_generate_until_passes_per_example_metadata_to_representation_tracker(self):
        model = self._make_model()
        model._lmms_eval_representation_tracker = object()
        request = types.SimpleNamespace(
            args=("Describe the image", {}, lambda doc: [], 0, "demo_task", "test"),
        )

        with (
            patch("lmms_eval.models.simple.qwen3_vl.process_vision_info", return_value=(None, None, {})),
            patch("lmms_eval.models.simple.qwen3_vl.build_representation_context", return_value=None) as build_context,
            patch("lmms_eval.models.simple.qwen3_vl.representation_tracker_context") as tracker_context,
        ):
            result = model.generate_until([request])

        self.assertEqual(result, ["final answer"])
        build_context.assert_called_once()
        self.assertEqual(build_context.call_args.kwargs["task_names"], ["demo_task"])
        self.assertEqual(build_context.call_args.kwargs["split_names"], ["test"])
        self.assertEqual(build_context.call_args.kwargs["sample_ids"], [0])
        self.assertIs(tracker_context.call_args.args[0], model)

    def test_chat_generate_until_passes_per_example_metadata_to_representation_tracker(self):
        model = self._make_model()
        model.__class__ = Qwen3_VL_Chat
        model._lmms_eval_representation_tracker = object()
        request = types.SimpleNamespace(
            args=(
                "unused-context",
                lambda doc: [{"role": "user", "content": [{"type": "text", "text": "Describe the image"}]}],
                {},
                0,
                "demo_task",
                "test",
            ),
        )

        with (
            patch("lmms_eval.models.chat.qwen3_vl.process_vision_info", return_value=(None, None, {})),
            patch("lmms_eval.models.chat.qwen3_vl.build_representation_context", return_value=None) as build_context,
            patch("lmms_eval.models.chat.qwen3_vl.representation_tracker_context") as tracker_context,
        ):
            result = model.generate_until([request])

        self.assertEqual([item.text for item in result], ["final answer"])
        build_context.assert_called_once()
        self.assertEqual(build_context.call_args.kwargs["task_names"], ["demo_task"])
        self.assertEqual(build_context.call_args.kwargs["split_names"], ["test"])
        self.assertEqual(build_context.call_args.kwargs["sample_ids"], [0])
        self.assertIs(tracker_context.call_args.args[0], model)

    def test_chat_generate_until_applies_task_stop_strings(self):
        model = self._make_model()
        model.__class__ = Qwen3_VL_Chat
        request = types.SimpleNamespace(
            args=(
                "unused-context",
                lambda doc: [{"role": "user", "content": [{"type": "text", "text": "Question"}]}],
                {"until": ["STOP"]},
                0,
                "demo_task",
                "test",
            ),
        )
        model.processor.batch_decode = lambda *args, **kwargs: ["answer STOP ignored"]

        with patch("lmms_eval.models.chat.qwen3_vl.process_vision_info", return_value=(None, None, {})):
            result = model.generate_until([request])

        self.assertEqual([item.text for item in result], ["answer "])

    def test_subsample_video_inputs_updates_dict_metadata_frames(self):
        model = self._make_model(max_num_frames=3)
        video_inputs = [
            torch.arange(10, dtype=torch.float32).reshape(5, 2),
            torch.arange(12, dtype=torch.float32).reshape(6, 2),
        ]
        video_metadatas = [
            {"frames_indices": [0, 1, 2, 3, 4]},
            {"frames_indices": [10, 11, 12, 13, 14, 15]},
        ]

        model._subsample_video_inputs(video_inputs, video_metadatas)

        self.assertEqual(video_inputs[0].shape[0], 3)
        self.assertEqual(video_inputs[1].shape[0], 3)
        self.assertEqual(video_metadatas[0]["frames_indices"], [0, 2, 4])
        self.assertEqual(video_metadatas[1]["frames_indices"], [10, 12, 15])


if __name__ == "__main__":
    unittest.main()
