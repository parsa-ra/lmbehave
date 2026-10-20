from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedType
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lmms_eval import utils
from lmms_eval.api.instance import Instance
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.representations import build_representation_context, representation_tracker_context


@register_model("hf_causal")
class HFCausal(lmms):
    """Plain-text Hugging Face causal model adapter.

    This path deliberately avoids chat templates and multimodal processing so the
    same prompt can be used for base and post-trained checkpoints.
    """

    def __init__(
        self,
        pretrained: str,
        revision: Optional[str] = None,
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        use_cache: bool = True,
        apply_chat_template: bool = False,
        attn_implementation: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if kwargs:
            raise ValueError(f"Unexpected hf_causal arguments: {sorted(kwargs)}")

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        model_kwargs = {
            "dtype": torch.bfloat16,
            "device_map": self.device_map,
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        self._model = AutoModelForCausalLM.from_pretrained(pretrained, **model_kwargs).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(
            pretrained,
            revision=revision,
            trust_remote_code=trust_remote_code,
        )
        if self._tokenizer.pad_token_id is None:
            if self._tokenizer.eos_token_id is None:
                raise ValueError("hf_causal requires a tokenizer pad token or EOS token")
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"

        self.pretrained = pretrained
        self.revision = revision
        self._config = self.model.config
        self.batch_size_per_gpu = int(batch_size)
        self.use_cache = bool(use_cache)
        self.apply_chat_template = bool(apply_chat_template)

        if accelerator.num_processes > 1:
            if accelerator.distributed_type not in {
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
            }:
                raise ValueError("hf_causal supports only FSDP or multi-GPU Accelerate execution")
            if accelerator.distributed_type == DistributedType.FSDP:
                self._model = accelerator.prepare(self.model)
            else:
                self._model = accelerator.prepare_model(self.model, evaluation_mode=True)
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
        return int(getattr(self.config, "max_position_embeddings", 2048))

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

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        if getattr(self, "_lmms_eval_representation_tracker", None) is not None:
            raise ValueError(
                "hf_causal loglikelihood creates one request per answer choice; "
                "representation collection is disabled until choice-level records have a predeclared aggregation"
            )

        results: list[Tuple[float, bool]] = []
        progress = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        for start in range(0, len(requests), self.batch_size):
            chunk = requests[start : start + self.batch_size]
            encoded: list[tuple[list[int], int]] = []
            for request in chunk:
                context, doc_to_target, doc_to_visual, doc_id, task, split = request.args
                visuals = doc_to_visual(self.task_dict[task][split][doc_id])
                if visuals:
                    raise ValueError("hf_causal accepts text-only tasks; received a non-empty visual input")
                continuation = (
                    doc_to_target
                    if isinstance(doc_to_target, str)
                    else doc_to_target(self.task_dict[task][split][doc_id])
                )
                context_ids = self.tokenizer.encode(context, add_special_tokens=False)
                combined_ids = self.tokenizer.encode(context + continuation, add_special_tokens=False)
                continuation_start = len(context_ids)
                if continuation_start < 1 or continuation_start >= len(combined_ids):
                    raise ValueError("Each loglikelihood request needs non-empty context and continuation tokens")
                if len(combined_ids) > self.max_length:
                    removed = len(combined_ids) - self.max_length
                    combined_ids = combined_ids[removed:]
                    continuation_start -= removed
                    if continuation_start < 1:
                        raise ValueError("Context truncation removed the token preceding the continuation")
                encoded.append((combined_ids, continuation_start))

            width = max(len(token_ids) for token_ids, _continuation_start in encoded)
            input_ids = torch.full(
                (len(encoded), width),
                self.tokenizer.pad_token_id,
                dtype=torch.long,
                device=self.device,
            )
            attention_mask = torch.zeros_like(input_ids)
            for row, (token_ids, _continuation_start) in enumerate(encoded):
                length = len(token_ids)
                input_ids[row, :length] = torch.tensor(token_ids, dtype=torch.long, device=self.device)
                attention_mask[row, :length] = 1

            with torch.inference_mode():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
            log_probabilities = F.log_softmax(logits.float(), dim=-1)
            for row, (token_ids, continuation_start) in enumerate(encoded):
                length = len(token_ids)
                targets = input_ids[row, continuation_start:length]
                predicting_logits = log_probabilities[row, continuation_start - 1 : length - 1]
                token_log_probabilities = predicting_logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
                greedy_tokens = predicting_logits.argmax(dim=-1)
                negative_log_likelihood = -float(token_log_probabilities.sum().item())
                results.append((negative_log_likelihood, bool(torch.equal(greedy_tokens, targets))))
                progress.update(1)
        progress.close()
        return results

    def generate_until(self, requests: List[Instance]) -> List[str]:
        results: list[str] = []

        def _format_context(context: str) -> str:
            if not self.apply_chat_template:
                return context
            if not getattr(self.tokenizer, "chat_template", None):
                raise ValueError("apply_chat_template=true requires a tokenizer chat template")
            return self.tokenizer.apply_chat_template(
                [{"role": "user", "content": context}],
                tokenize=False,
                add_generation_prompt=True,
            )

        def _collate(request_args):
            tokens = self.tokenizer.encode(_format_context(request_args[0]))
            return -len(tokens), request_args[0]

        reorder = utils.Collator([request.args for request in requests], _collate, grouping=True)
        chunks = reorder.get_batched(n=self.batch_size, batch_fn=None)
        progress = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")

        for chunk in chunks:
            contexts, all_gen_kwargs, doc_to_visual, doc_ids, tasks, splits = zip(*chunk)
            for visual_fn, doc_id, task, split in zip(doc_to_visual, doc_ids, tasks, splits):
                visuals = visual_fn(self.task_dict[task][split][doc_id])
                if visuals:
                    raise ValueError("hf_causal accepts text-only tasks; received a non-empty visual input")

            generation_kwargs = dict(all_gen_kwargs[0])
            until = generation_kwargs.pop("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected until to be a string or list, got {type(until).__name__}")
            until = [str(item) for item in until if str(item)]

            formatted_contexts = [_format_context(context) for context in contexts]
            inputs = self.tokenizer(
                formatted_contexts,
                padding=True,
                return_tensors="pt",
                return_token_type_ids=False,
            )
            target_device = self.model.device if self.device_map == "auto" else self.device
            inputs = inputs.to(target_device)

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

            representation_owner = self if getattr(self, "_lmms_eval_representation_tracker", None) is not None else self.model
            representation_context = None
            if getattr(representation_owner, "_lmms_eval_representation_tracker", None) is not None:
                representation_context = build_representation_context(
                    inputs.get("input_ids"),
                    inputs.get("attention_mask"),
                    task_names=list(tasks),
                    split_names=list(splits),
                    sample_ids=list(doc_ids),
                )

            with torch.inference_mode():
                with representation_tracker_context(representation_owner, representation_context) as active_representation_tracker:
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
            answers = self.tokenizer.batch_decode(
                generated[:, prompt_width:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for answer, context in zip(answers, contexts):
                for stop in until:
                    answer = answer.split(stop, 1)[0]
                results.append(answer)
                self.cache_hook.add_partial("generate_until", (context, all_gen_kwargs[0]), answer)
                progress.update(1)

        progress.close()
        return reorder.get_original(results)

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("Multi-round generation is not implemented for hf_causal")
