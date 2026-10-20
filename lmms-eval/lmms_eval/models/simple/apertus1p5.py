from __future__ import annotations

import torch
from accelerate import Accelerator, DistributedType
from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.representations import (
    build_representation_context,
    representation_tracker_context,
)
from tqdm import tqdm
from transformers import AutoProcessor


@register_model("apertus1p5")
class Apertus1p5(lmms):
    """Text-only evaluation path for the multimodal Apertus 1.5 checkpoint.

    Apertus 1.5 is not yet in a released Transformers version. The official
    Swiss-AI Transformers revision must therefore be installed explicitly.
    This adapter intentionally rejects media until the study defines a matched
    multimodal protocol.
    """

    def __init__(
        self,
        pretrained: str = "swiss-ai/Apertus-v1.5-8B",
        revision: str | None = None,
        device: str | None = "cuda",
        device_map: str | None = "auto",
        batch_size: int | str | None = 1,
        use_cache: bool = True,
        enable_thinking: bool = False,
        system_prompt: str | None = None,
        attn_implementation: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(f"Unexpected apertus1p5 arguments: {sorted(kwargs)}")
        if enable_thinking:
            raise ValueError(
                "The current direct-answer study requires enable_thinking=false; "
                "thinking mode needs a separately frozen trace and stopping protocol"
            )

        # This symbol exists only in the pinned Swiss-AI branch at present.
        try:
            from transformers import AutoModelForMultimodalLM
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Apertus 1.5 requires the official Swiss-AI Transformers revision "
                "3797303dda74844e3d1f8977ff5518bb91f818b4"
            ) from exc

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "dtype": "auto",
            "device_map": self.device_map,
            "revision": revision,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        self._model = AutoModelForMultimodalLM.from_pretrained(
            pretrained, **model_kwargs
        ).eval()
        self.processor = AutoProcessor.from_pretrained(pretrained, revision=revision)
        self._tokenizer = self.processor.tokenizer
        if self._tokenizer.pad_token_id is None:
            if self._tokenizer.eos_token_id is None:
                raise ValueError(
                    "apertus1p5 requires a tokenizer pad token or EOS token"
                )
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        self.pretrained = pretrained
        self.revision = revision
        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = bool(use_cache)
        self.enable_thinking = False
        self.system_prompt = system_prompt

        if accelerator.num_processes > 1:
            if accelerator.distributed_type not in {
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            }:
                raise ValueError(
                    "apertus1p5 supports only FSDP or multi-GPU Accelerate execution"
                )
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(
                    self.model, evaluation_mode=True
                )
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self.accelerator.unwrap_model(self._model)

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        text_config = getattr(self.config, "text_config", self.config)
        return int(getattr(text_config, "max_position_embeddings", 262_144))

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: list[Instance]) -> list[tuple[float, bool]]:
        raise NotImplementedError(
            "Apertus 1.5 is currently integrated only for direct-answer generate_until tasks"
        )

    def _messages(self, context: str) -> list[dict]:
        messages: list[dict] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": context})
        return messages

    def generate_until(self, requests: list[Instance]) -> list[str]:
        results: list[str] = []

        def _collate(request_args):
            rendered = self.processor.apply_chat_template(
                self._messages(request_args[0]),
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=False,
            )
            return -len(self.tokenizer.encode(rendered)), request_args[0]

        reorder = utils.Collator(
            [request.args for request in requests], _collate, grouping=True
        )
        chunks = reorder.get_batched(n=self.batch_size, batch_fn=None)
        progress = tqdm(
            total=len(requests), disable=(self.rank != 0), desc="Model Responding"
        )

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_ids, tasks, splits = zip(
                *chunk
            )
            for visual_fn, doc_id, task, split in zip(
                doc_to_visual, doc_ids, tasks, splits
            ):
                visuals = visual_fn(self.task_dict[task][split][doc_id])
                if visuals:
                    raise ValueError(
                        "The apertus1p5 study adapter currently accepts text-only tasks; "
                        "received a non-empty media input"
                    )

            generation_kwargs = dict(all_gen_kwargs[0])
            until = generation_kwargs.pop(
                "until", [self.tokenizer.decode(self.eot_token_id)]
            )
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise TypeError(
                    f"Expected until to be a string or list, got {type(until).__name__}"
                )
            until = [str(item) for item in until if str(item)]

            conversations = [self._messages(context) for context in contexts]
            inputs = self.processor.apply_chat_template(
                conversations,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
                processor_kwargs={"padding": True},
            ).to(self.model.device)

            current = {
                "max_new_tokens": 128,
                "temperature": 0.0,
                "top_p": None,
                "num_beams": 1,
                **generation_kwargs,
            }
            temperature = float(current.get("temperature") or 0.0)
            do_sample = bool(current.get("do_sample", temperature > 0))
            if not do_sample:
                temperature = None
                current["top_p"] = None

            representation_owner = (
                self
                if getattr(self, "_lmms_eval_representation_tracker", None) is not None
                else self.model
            )
            representation_context = None
            if (
                getattr(representation_owner, "_lmms_eval_representation_tracker", None)
                is not None
            ):
                representation_context = build_representation_context(
                    inputs.get("input_ids"),
                    inputs.get("attention_mask"),
                    task_names=list(tasks),
                    split_names=list(splits),
                    sample_ids=list(doc_ids),
                )

            with (
                torch.inference_mode(),
                representation_tracker_context(
                    representation_owner, representation_context
                ) as active_representation_tracker,
            ):
                generated = self.model.generate(
                    **inputs,
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.pad_token_id,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=current.get("top_p"),
                    num_beams=int(current.get("num_beams", 1)),
                    max_new_tokens=int(current["max_new_tokens"]),
                    use_cache=self.use_cache,
                )
                if active_representation_tracker is not None:
                    active_representation_tracker.record_generated_tokens(
                        generated[:, int(inputs["input_ids"].shape[1]) :],
                        eos_token_id=self.tokenizer.eos_token_id,
                        pad_token_id=self.tokenizer.pad_token_id,
                    )

            prompt_width = int(inputs["input_ids"].shape[1])
            answers = self.processor.batch_decode(
                generated[:, prompt_width:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for answer, context in zip(answers, contexts):
                for stop in until:
                    answer = answer.split(stop, 1)[0]
                results.append(answer)
                self.cache_hook.add_partial(
                    "generate_until", (context, all_gen_kwargs[0]), answer
                )
                progress.update(1)

        progress.close()
        return reorder.get_original(results)

    def generate_until_multi_round(self, requests):
        raise NotImplementedError(
            "Multi-round generation is not implemented for apertus1p5"
        )
