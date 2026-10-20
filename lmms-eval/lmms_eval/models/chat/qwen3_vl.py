import time
from typing import List

from loguru import logger as eval_logger
from tqdm import tqdm

from lmms_eval import utils
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL as Qwen3_VLSimple
from lmms_eval.protocol import ChatMessages
from lmms_eval.representations import build_representation_context, representation_tracker_context

process_vision_info, _has_qwen_vl = optional_import("qwen_vl_utils", "process_vision_info")
if not _has_qwen_vl:
    eval_logger.warning("Failed to import qwen_vl_utils; Please install it via `pip install qwen-vl-utils`")


@register_model("qwen3_vl_chat")
class Qwen3_VL(Qwen3_VLSimple):
    is_simple = False

    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        res = []

        def _collate(x):
            return x[0], x[0]

        re_ords = utils.Collator(
            [reg.args for reg in requests],
            _collate,
            group_fn=lambda x: x[2],
            grouping=True,
        )
        chunks = re_ords.get_batched(n=self.batch_size, batch_fn=None)
        num_iters = len(requests) // self.batch_size if len(requests) % self.batch_size == 0 else len(requests) // self.batch_size + 1
        pbar = tqdm(total=num_iters, disable=(self.rank != 0), desc="Model Responding")
        total_elapsed_time = 0
        total_tokens = 0

        for chunk in chunks:
            ctx, doc_to_messages, all_gen_kwargs, doc_id, task, split = zip(*chunk)

            chat_messages: List[ChatMessages] = []
            visuals = []
            videos = []
            for idx, (ids, task_name, split_name) in enumerate(zip(doc_id, task, split)):
                messages = doc_to_messages[idx](self.task_dict[task_name][split_name][ids])
                messages.insert(0, {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]})
                chat_message = ChatMessages(**{"messages": messages})
                visual, video, _ = chat_message.extract_media()
                visuals.append(visual)
                videos.append(video)
                chat_messages.append(chat_message)

            visuals = self.flatten(visuals)
            videos = self.flatten(videos)
            gen_kwargs = all_gen_kwargs[0]
            until = gen_kwargs.get("until", [self.tokenizer.decode(self.eot_token_id)])
            if isinstance(until, str):
                until = [until]
            elif not isinstance(until, list):
                raise ValueError(f"Expected `gen_kwargs['until']` to be str or list, got {type(until)}")
            until = [term for term in until if term]

            video_kwargs = self._build_video_kwargs()
            batched_messages = [chat_message.to_hf_messages(video_kwargs=video_kwargs) for chat_message in chat_messages]

            texts = self._apply_chat_template(batched_messages)

            image_inputs, video_inputs, video_kwargs_qwen = process_vision_info(
                batched_messages,
                return_video_kwargs=True,
                image_patch_size=16,
                return_video_metadata=True,
            )
            video_kwargs = {**video_kwargs, **video_kwargs_qwen}

            video_metadatas = None
            if video_inputs is not None:
                video_inputs, video_metadatas = zip(*video_inputs)
                video_inputs, video_metadatas = list(video_inputs), list(video_metadatas)

            if self.batch_size > 1:
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    video_metadata=video_metadatas,
                    **video_kwargs,
                    do_resize=False,
                    padding=True,
                    padding_side="left",
                    return_tensors="pt",
                )
            else:
                inputs = self.processor(
                    text=texts,
                    images=image_inputs,
                    videos=video_inputs,
                    video_metadata=video_metadatas,
                    **video_kwargs,
                    do_resize=False,
                    return_tensors="pt",
                )

            if self.device_map == "auto":
                inputs = inputs.to("cuda")
            else:
                inputs = inputs.to(self.device)

            generate_kwargs = self._build_generate_kwargs(gen_kwargs)

            representation_owner = self if getattr(self, "_lmms_eval_representation_tracker", None) is not None else self.model
            representation_context = None
            if getattr(representation_owner, "_lmms_eval_representation_tracker", None) is not None:
                representation_context = build_representation_context(
                    getattr(inputs, "input_ids", inputs.get("input_ids", None)),
                    getattr(inputs, "attention_mask", inputs.get("attention_mask", None)),
                    task_names=list(task),
                    split_names=list(split),
                    sample_ids=list(doc_id),
                )

            start_time = time.time()
            with representation_tracker_context(representation_owner, representation_context) as active_representation_tracker:
                cont = self.model.generate(**inputs, **generate_kwargs)
                if active_representation_tracker is not None:
                    active_representation_tracker.record_generated_tokens(
                        cont[:, int(inputs.input_ids.shape[1]) :],
                        eos_token_id=generate_kwargs.get(
                            "eos_token_id", self.tokenizer.eos_token_id
                        ),
                        pad_token_id=generate_kwargs.get(
                            "pad_token_id", self.tokenizer.pad_token_id
                        ),
                    )
            end_time = time.time()

            generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, cont)]
            answers = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )

            total_elapsed_time += end_time - start_time
            total_tokens += sum(len(ids) for ids in generated_ids_trimmed)

            for i, (ans, context) in enumerate(zip(answers, texts)):
                ans = self._strip_thinking(ans)
                for term in until:
                    ans = ans.split(term, 1)[0]
                res.append(GenerationResult(text=ans, token_counts=TokenCounts(output_tokens=len(generated_ids_trimmed[i]))))
                self.cache_hook.add_partial("generate_until", (context, gen_kwargs), ans)

                eval_logger.debug(f"Question: {context}")
                eval_logger.debug(f"Model Response: {ans}")
            pbar.update(1)

        res = re_ords.get_original(res)

        avg_speed = total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
        log_metrics(
            total_gen_tokens=total_tokens,
            total_elapsed_time=total_elapsed_time,
            avg_speed=avg_speed,
            additional_metrics={"rank": self.rank},
        )

        pbar.close()
        return res
