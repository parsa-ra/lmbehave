import base64
import collections
import copy
import inspect
import itertools
import json
import mimetypes
import os
import random
import re
from datetime import timedelta
from typing import Any, Callable, List, Optional, Union

import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from loguru import logger as eval_logger
from tqdm import tqdm

# Increase default torch.distributed process group timeout from 30 min to 24 hours.
# With dp=16 across 2 nodes, workload imbalance causes fast ranks to finish hours
# before slow ranks (e.g. 1h24m vs 4h24m).  3h was not enough — increase to 24h.
torch.distributed.distributed_c10d.default_pg_timeout = timedelta(hours=24)

import lmms_eval.api
import lmms_eval.api.metrics
import lmms_eval.api.registry
from lmms_eval import models
from lmms_eval.api.instance import Instance, unwrap_generation_output
from lmms_eval.api.model import lmms
from lmms_eval.api.reasoning import parse_reasoning_tags_config, strip_reasoning_tags
from lmms_eval.api.task import Task
from lmms_eval.baselines import (
    BASELINE_REGISTRY,
    get_baseline_display_name,
    load_baseline,
)
from lmms_eval.caching.response_cache import ResponseCache
from lmms_eval.evaluator_utils import (
    compute_baseline_comparison,
    consolidate_group_results,
    consolidate_results,
    get_sample_size,
    get_subtask_list,
    get_task_list,
    prepare_print_tasks,
    print_writeout,
    run_task_tests,
)
from lmms_eval.llm_judge.launcher import get_launcher
from lmms_eval.loggers.evaluation_tracker import EvaluationTracker
from lmms_eval.models.model_utils.efficiency_metrics import build_efficiency_summary
from lmms_eval.models.model_utils.usage_metrics import (
    is_budget_exceeded,
    reset_usage_metrics,
    set_budget,
    set_task_context,
    summarize_usage_metrics,
)
from lmms_eval.pruning import (
    PruningTracker,
    configure_model_for_explicit_attention,
    gather_pruning_payload,
    resolve_pruning_target,
)
from lmms_eval.representations import RepresentationDynamicsTracker, gather_representation_payload
from lmms_eval.tasks import TaskManager, get_task_dict
from lmms_eval.utils import (
    create_iterator,
    get_datetime_str,
    get_git_branch_name,
    get_git_commit_hash,
    get_lmms_eval_version_string,
    handle_non_serializable,
    hash_string,
    is_multimodal_content,
    positional_deprecated,
    run_task_tests,
    simple_parse_args_string,
)


def _append_model_response(reqtype: str, response: Any, request: Instance) -> None:
    """Store likelihood tuples unchanged and unwrap only generation outputs."""

    if reqtype == "loglikelihood":
        request.resps.append(response)
        request.token_counts.append(None)
    else:
        text, token_counts = unwrap_generation_output(response)
        request.resps.append(text)
        request.token_counts.append(token_counts)


def _raw_responses_for_sample_log(requests: list[Instance]) -> list[list[Any]]:
    """Return model outputs before task filters for an auditable sample log."""

    return [request.resps for request in requests]


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff")

_REDACTED_VALUE = "<redacted>"
_SENSITIVE_KEY_RE = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?token|auth[_-]?token|credential|password|secret|token)(?:$|[_-])",
    flags=re.IGNORECASE,
)
_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|credential|password|secret|token)\s*[=:]\s*)([^,\s]+)",
    flags=re.IGNORECASE,
)
_CREDENTIAL_VALUE_RE = re.compile(r"\b(?:hf_|gh[pousr]_|github_pat_|sk-)[A-Za-z0-9_-]{20,}\b")


def redact_sensitive_data(value, *, key: object = None):
    """Return a structure-safe copy with credential-like values redacted."""

    if key is not None and _SENSITIVE_KEY_RE.search(str(key)):
        return _REDACTED_VALUE
    if isinstance(value, dict):
        return {item_key: redact_sensitive_data(item_value, key=item_key) for item_key, item_value in value.items()}
    if isinstance(value, list):
        return [redact_sensitive_data(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_sensitive_data(item) for item in value)
    if isinstance(value, str):
        value = _CREDENTIAL_ASSIGNMENT_RE.sub(r"\1<redacted>", value)
        return _CREDENTIAL_VALUE_RE.sub(_REDACTED_VALUE, value)
    return value


def resolved_cli_args_for_logging(cli_args) -> dict:
    """Build JSON-safe reproducibility arguments without serializing secrets."""

    resolved = {}
    for key, value in vars(cli_args).items():
        safe_value = redact_sensitive_data(value, key=key)
        try:
            json.dumps(safe_value)
            resolved[key] = safe_value
        except (TypeError, ValueError):
            resolved[key] = redact_sensitive_data(str(safe_value), key=key)
    return resolved


def _model_class_accepts_init_arg(model_cls, arg_name: str) -> bool:
    try:
        signature = inspect.signature(model_cls.__init__)
    except (TypeError, ValueError):
        return False
    return arg_name in signature.parameters


def _append_or_override_model_arg(model_args: Optional[Union[str, dict]], key: str, value: str) -> tuple[Optional[Union[str, dict]], bool, object]:
    if isinstance(model_args, dict):
        previous = model_args.get(key)
        if previous == value:
            return model_args, False, previous
        updated = dict(model_args)
        updated[key] = value
        return updated, True, previous

    args_string = "" if model_args is None else str(model_args)
    parsed_args = simple_parse_args_string(args_string)
    previous = parsed_args.get(key)
    if previous == value:
        return model_args, False, previous

    suffix = f"{key}={value}"
    if args_string.strip():
        return f"{args_string},{suffix}", True, previous
    return suffix, True, previous


def _enable_reentrant_filelocks() -> None:
    """Force singleton file locks so nested load_dataset calls cannot deadlock.

    filelock >= 3.16 detects in-thread deadlocks via a global ``_registry``
    that tracks *all* held locks by canonical path and instance ``id()``.
    Meanwhile, ``is_singleton=True`` caches instances per-class in
    ``cls._instances``.  ``filelock.FileLock`` (``UnixFileLock``) and
    ``datasets.utils._filelock.FileLock`` are **different classes**, so each
    maintains a separate singleton cache — two distinct instances can end up
    targeting the same OS lock file, triggering the global deadlock check.

    Fix: maintain a **cross-class** singleton cache keyed on the canonical
    lock path.  Any target-class ``FileLock(path)`` call returns the single
    cached instance regardless of which class originated it.
    """
    if getattr(_enable_reentrant_filelocks, "_patched", False):
        return

    import filelock as _fl
    import filelock._api as _fl_api
    from datasets.utils import _filelock as _datasets_filelock

    target_lock_classes = tuple(
        cls
        for cls in (
            getattr(_fl, "FileLock", None),
            getattr(_datasets_filelock, "FileLock", None),
        )
        if cls is not None
    )

    original_call = _fl_api.FileLockMeta.__call__
    _cross_class_cache: dict[str, _fl_api.BaseFileLock] = {}

    def _patched_call(cls, lock_file, *args, **kwargs):
        if cls in target_lock_classes:
            canonical = str(lock_file)
            cached = _cross_class_cache.get(canonical)
            if cached is not None:
                return cached
            kwargs.setdefault("is_singleton", True)
            if cls is getattr(_datasets_filelock, "FileLock", None):
                unset_mode = getattr(_fl_api, "_UNSET_FILE_MODE", None)
                if kwargs.get("mode", unset_mode) == unset_mode:
                    umask = os.umask(0o666)
                    os.umask(umask)
                    kwargs["mode"] = 0o666 & ~umask
            instance = original_call(cls, lock_file, *args, **kwargs)
            _cross_class_cache[canonical] = instance
            return instance
        return original_call(cls, lock_file, *args, **kwargs)

    _fl_api.FileLockMeta.__call__ = _patched_call
    _enable_reentrant_filelocks._patched = True


def _looks_like_image_ref(value: str) -> bool:
    lowered = value.lower().strip()
    if lowered.startswith("data:image/"):
        return True
    if lowered.startswith(("http://", "https://", "file://")):
        return any(ext in lowered for ext in IMAGE_EXTENSIONS)
    return lowered.endswith(IMAGE_EXTENSIONS)


def _guess_image_mime(path_hint: Optional[str]) -> str:
    if path_hint:
        guessed, _ = mimetypes.guess_type(path_hint)
        if guessed and guessed.startswith("image/"):
            return guessed
    return "image/png"


def _append_image_source(target: list[str], source: str, seen: set[str], max_items: int) -> None:
    if not source or source in seen or len(target) >= max_items:
        return
    seen.add(source)
    target.append(source)


def _extract_image_sources(value, out: list[str], seen: set[str], max_items: int = 4, max_inline_bytes: int = 300_000) -> None:
    if len(out) >= max_items:
        return

    if isinstance(value, str):
        if _looks_like_image_ref(value):
            _append_image_source(out, value, seen, max_items)
        return

    if isinstance(value, dict):
        path_hint: Optional[str] = None
        for key in ("url", "uri", "path", "image", "image_url", "image_path"):
            candidate = value.get(key)
            if isinstance(candidate, str):
                if key == "path":
                    path_hint = candidate
                if _looks_like_image_ref(candidate):
                    _append_image_source(out, candidate, seen, max_items)

        raw_bytes = value.get("bytes")
        if isinstance(raw_bytes, (bytes, bytearray)) and 0 < len(raw_bytes) <= max_inline_bytes and len(out) < max_items:
            mime = _guess_image_mime(path_hint)
            encoded = base64.b64encode(raw_bytes).decode("ascii")
            _append_image_source(out, f"data:{mime};base64,{encoded}", seen, max_items)

        for nested in value.values():
            _extract_image_sources(nested, out, seen, max_items=max_items, max_inline_bytes=max_inline_bytes)
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            _extract_image_sources(item, out, seen, max_items=max_items, max_inline_bytes=max_inline_bytes)
            if len(out) >= max_items:
                break


def _collect_input_media(doc: dict, request_args: list) -> list[str]:
    sources: list[str] = []
    seen: set[str] = set()
    _extract_image_sources(doc, sources, seen)
    for arg in request_args:
        if len(sources) >= 4:
            break
        _extract_image_sources(arg, sources, seen)
    return sources


@positional_deprecated
def simple_evaluate(
    model,
    model_args: Optional[Union[str, dict]] = None,
    launcher_args: Optional[Union[str, dict]] = None,
    tasks: Optional[List[Union[str, dict, object]]] = None,
    num_fewshot: Optional[int] = None,
    batch_size: Optional[Union[int, str]] = None,
    max_batch_size: Optional[int] = None,
    device: Optional[str] = None,
    use_cache: Optional[str] = None,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    delete_requests_cache: bool = False,
    limit: Optional[Union[int, float]] = None,
    offset: int = 0,
    bootstrap_iters: int = 100000,
    check_integrity: bool = False,
    write_out: bool = False,
    log_samples: bool = True,
    evaluation_tracker: Optional[EvaluationTracker] = None,
    system_instruction: Optional[str] = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    gen_kwargs: Optional[str] = None,
    task_manager: Optional[TaskManager] = None,
    verbosity: str = "INFO",
    predict_only: bool = False,
    random_seed: int = 0,
    numpy_random_seed: int = 1234,
    torch_random_seed: int = 1234,
    fewshot_random_seed: int = 1234,
    datetime_str: str = get_datetime_str(),
    distributed_executor_backend: str = "accelerate",
    cli_args=None,
    force_simple: bool = False,
    repeats: int = 1,
    baseline: Optional[str] = None,
    max_tokens: Optional[int] = None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param model: Union[str, LM]
        Name of model or LM object, see lmms_eval.models.get_model
    :param model_args: Optional[str, dict]
        String or dict arguments for each model class, see LM.create_from_arg_string and LM.create_from_arg_object.
        Ignored if `model` argument is a LM object.
    :param tasks: list[Union[str, dict, Task]]
        List of task names or Task objects. Task objects will be taken to have name task.EVAL_HARNESS_NAME if defined and type(task).__name__ otherwise.
    :param num_fewshot: int
        Number of examples in few-shot context
    :param batch_size: int or str, optional
        Batch size for model
    :param max_batch_size: int, optional
        Maximal batch size to try with automatic batch size detection
    :param device: str, optional
        PyTorch device (e.g. "cpu" or "cuda:0") for running models
    :param use_cache: str, optional
        Path to a response-cache root directory or SQLite .db file for response-level caching. `None` to disable.
    :param cache_requests: bool, optional
        Speed up evaluation by caching the building of dataset requests. `None` if not caching.
    :param rewrite_requests_cache: bool, optional
        Rewrites all of the request cache if set to `True`. `None` if not desired.
    :param delete_requests_cache: bool, optional
        Deletes all of the request cache if set to `True`. `None` if not desired.
    :param limit: int or float, optional
        Limit the number of examples per task (only use this for testing), If <1, limit is a percentage of the total number of examples.
    :param offset: int, optional
        Start evaluation from this dataset index for each task.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics, used when calculating stderrs. set to 0 for no stderr calculations to be performed.
    :param check_integrity: bool
        Whether to run the relevant part of the test suite for the tasks
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param repeats: int
        Number of repeated generations per question for k-samples stability metrics.
    :param system_instruction: str
        System instruction to be applied to the prompt
    :param apply_chat_template: bool
        If True, apply chat template to the prompt
    :param fewshot_as_multiturn: bool
        Whether to provide the fewshot examples as a multiturn conversation or a single user turn.
    :param gen_kwargs: str
        String arguments for model generation
        Ignored for all tasks with loglikelihood output_type
    :param predict_only: bool
        If true only model outputs will be generated and returned. Metrics will not be evaluated
    :param random_seed: int
        Random seed for python's random module. If set to None, the seed will not be set.
    :param numpy_random_seed: int
        Random seed for numpy. If set to None, the seed will not be set.
    :param torch_random_seed: int
        Random seed for torch. If set to None, the seed will not be set.
    :param fewshot_random_seed: int
        Random seed for fewshot sampler random generator. If set to None, the seed of generator will be set to None.
    :param distributed_executor_backend: str
        The backend to use for distributed execution, `accelerate` or `torchrun`. Defaults to "accelerate" for the `accelerate` library.
    :return
        Dictionary of results
    """
    seed_message = []
    if random_seed is not None:
        # See https://github.com/EleutherAI/lm-evaluation-harness/pull/1412
        seed_message.append(f"Setting random seed to {random_seed}")
        random.seed(random_seed)

    if numpy_random_seed is not None:
        seed_message.append(f"Setting numpy seed to {numpy_random_seed}")
        np.random.seed(numpy_random_seed)

    if torch_random_seed is not None:
        seed_message.append(f"Setting torch manual seed to {torch_random_seed}")
        torch.manual_seed(torch_random_seed)

    if seed_message:
        eval_logger.info(" | ".join(seed_message))

    assert tasks != [], "No tasks specified, or no tasks found. Please verify the task names."

    assert distributed_executor_backend in {"accelerate", "torchrun"}, f"Invalid distributed executor backend: {distributed_executor_backend}. Choose either 'accelerate' or 'torchrun'."

    if gen_kwargs:
        gen_kwargs = simple_parse_args_string(gen_kwargs)
        eval_logger.warning("generation_kwargs specified through cli, these settings will be used over set parameters in yaml tasks.")
        if gen_kwargs == "":
            gen_kwargs = None

    if model_args is None:
        model_args = ""
    pruning_enabled = bool(getattr(cli_args, "pruning_tracker", False))
    force_eager_attention = bool(getattr(cli_args, "pruning_tracker_force_eager_attention", True))
    force_output_attentions = bool(getattr(cli_args, "pruning_tracker_force_output_attentions", False))
    representation_enabled = bool(getattr(cli_args, "representation_dynamics", False))

    if launcher_args is not None:
        launcher_args = simple_parse_args_string(launcher_args)
        launcher_name = launcher_args.pop("name")
        eval_launcher = get_launcher(launcher_name)(**launcher_args)
    else:
        eval_launcher = None

    if task_manager is None:
        task_manager = TaskManager(verbosity, model_name=model)

    if isinstance(model, str):
        if model_args is None:
            model_args = ""
        model_cls = models.get_model(model, force_simple)
        if pruning_enabled and force_eager_attention:
            if _model_class_accepts_init_arg(model_cls, "attn_implementation"):
                model_args, changed_model_args, previous_attn_impl = _append_or_override_model_arg(model_args, "attn_implementation", "eager")
                if changed_model_args:
                    if previous_attn_impl is None:
                        eval_logger.info("Pruning tracker requested explicit attention; adding attn_implementation=eager to model_args before model load.")
                    else:
                        eval_logger.info(
                            "Pruning tracker requested explicit attention; overriding model_args attn_implementation={} with eager before model load.",
                            previous_attn_impl,
                        )
            else:
                eval_logger.warning(
                    "Pruning tracker requested explicit attention, but {}.__init__ does not expose attn_implementation; "
                    "will only try post-load attention configuration.",
                    model_cls.__name__,
                )

        lm = model_cls.create_from_arg_string(
            model_args,
            {
                "batch_size": batch_size,
                "max_batch_size": max_batch_size,
                "device": device,
            },
        )
    elif isinstance(model, lmms_eval.api.model.lmms):
        lm = model
    task_type = "simple" if lm.is_simple else "chat"
    pruning_tracker = None
    pruning_payload = None
    pruning_target_name = None
    explicit_attention_config = None
    representation_tracker = None
    representation_payload = None
    representation_target = None
    representation_target_name = None

    if pruning_enabled:
        pruning_target = resolve_pruning_target(lm)
        if pruning_target is None:
            eval_logger.warning(
                "Pruning tracker requested, but lmms_eval could not resolve a torch.nn.Module from {}.",
                type(lm).__name__,
            )
        else:
            pruning_target_name = type(pruning_target).__name__
            if force_eager_attention:
                explicit_attention_config = configure_model_for_explicit_attention(
                    pruning_target,
                    force_output_attentions=force_output_attentions,
                )
                eval_logger.info(
                    "Requested explicit attention outputs for pruning tracker on {} "
                    "(configs_seen={}, output_attentions_updates={}, attn_implementation_updates={}, "
                    "set_attn_implementation_calls={}, force_output_attentions={}, output_attentions_disabled={}).",
                    pruning_target_name,
                    explicit_attention_config["configs_seen"],
                    explicit_attention_config["output_attentions_updates"],
                    explicit_attention_config["attn_implementation_updates"],
                    explicit_attention_config["set_attn_implementation_calls"],
                    explicit_attention_config["force_output_attentions"],
                    explicit_attention_config["output_attentions_disabled"],
                )
            pruning_tracker = PruningTracker(
                pruning_target,
                update_stride=int(getattr(cli_args, "pruning_tracker_stride", 1) or 1),
                max_histogram_samples=int(getattr(cli_args, "pruning_tracker_max_histogram_samples", 65536) or 65536),
            )
            try:
                setattr(pruning_target, "_lmms_eval_pruning_tracker", pruning_tracker)
            except Exception:
                pass
            eval_logger.info(
                "Attached pruning tracker to {} with {} forward hooks (stride={}, max_histogram_samples={}).",
                pruning_target_name,
                len(pruning_tracker.hook_handles),
                pruning_tracker.update_stride,
                pruning_tracker.max_histogram_samples,
            )

    if representation_enabled:
        representation_target = resolve_pruning_target(lm)
        if representation_target is None:
            if pruning_tracker is not None:
                try:
                    if getattr(pruning_target, "_lmms_eval_pruning_tracker", None) is pruning_tracker:
                        delattr(pruning_target, "_lmms_eval_pruning_tracker")
                except Exception:
                    pass
                pruning_tracker.remove_hooks()
            raise ValueError(
                f"Representation dynamics requested, but lmms_eval could not resolve a torch.nn.Module from {type(lm).__name__}."
            )
        representation_target_name = type(representation_target).__name__
        phase_option = str(getattr(cli_args, "representation_dynamics_phases", "prefill"))
        representation_phases = ("prefill", "decode") if phase_option == "both" else (phase_option,)
        try:
            representation_tracker = RepresentationDynamicsTracker(
                representation_target,
                pooling=str(getattr(cli_args, "representation_dynamics_pooling", "last_valid")),
                phases=representation_phases,
                decode_aggregation=str(
                    getattr(cli_args, "representation_dynamics_decode_aggregation", "mean_updates")
                ),
                decode_max_updates=getattr(cli_args, "representation_dynamics_decode_max_updates", None),
                storage_dtype=str(getattr(cli_args, "representation_dynamics_storage_dtype", "float32")),
            )
        except Exception:
            if pruning_tracker is not None:
                try:
                    if getattr(pruning_target, "_lmms_eval_pruning_tracker", None) is pruning_tracker:
                        delattr(pruning_target, "_lmms_eval_pruning_tracker")
                except Exception:
                    pass
                pruning_tracker.remove_hooks()
            raise
        setattr(representation_target, "_lmms_eval_representation_tracker", representation_tracker)
        # Keep the tracker on the lmms wrapper as well as the resolved torch module.
        # Some wrappers resolve ``self.model`` dynamically (for example through
        # Accelerator.unwrap_model), so the wrapper is the stable owner used to
        # establish per-batch tracking contexts.
        setattr(lm, "_lmms_eval_representation_tracker", representation_tracker)
        eval_logger.info(
            "Attached representation dynamics tracker to {} with {} hooks over {} decoder layers "
            "(stack={}, pooling={}, phases={}, decode_aggregation={}).",
            representation_target_name,
            len(representation_tracker.hook_handles),
            len(representation_tracker.layer_specs),
            representation_tracker.selected_stack,
            representation_tracker.pooling,
            representation_tracker.phases,
            representation_tracker.decode_aggregation,
        )

    # filelock >= 3.16 decides singleton behavior in the metaclass __call__ path,
    # before __init__ runs. datasets.load_dataset() uses datasets.utils._filelock.FileLock,
    # which wraps filelock separately, so patch the shared metaclass instead.
    _enable_reentrant_filelocks()

    # Make SSLContext picklable so dill/datasets can fingerprint closures that
    # capture the HF Hub HTTP session.  Without this, datasets 4.x raises
    # "TypeError: cannot pickle 'SSLContext' object" during load_dataset().
    import ssl

    if not hasattr(ssl.SSLContext, "_orig_reduce"):
        ssl.SSLContext._orig_reduce = True  # sentinel to avoid double-patching
        ssl.SSLContext.__reduce__ = lambda self: (ssl.SSLContext, (self.protocol,))

    # Stagger dataset loading across ranks to avoid cross-process FileLock contention
    # on shared FS.  Rank 0 loads first (populates HF cache), then all others load
    # from warm cache after a barrier.
    # NOTE: use torch.distributed directly (not `dist`) because later code in this function
    # has local `import torch.distributed as dist` which shadows the module-level import.
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
        if rank == 0:
            eval_logger.info("Rank 0: loading datasets first to warm HF cache")
            task_dict = get_task_dict(tasks, task_manager, task_type)
        torch.distributed.barrier()
        if rank != 0:
            eval_logger.info(f"Rank {rank}: loading datasets from warm cache")
            task_dict = get_task_dict(tasks, task_manager, task_type)
    else:
        task_dict = get_task_dict(tasks, task_manager, task_type)

    # helper function to recursively apply config overrides to leaf subtasks, skipping their constituent groups.
    # (setting of num_fewshot ; bypassing metric calculation ; setting fewshot seed)
    def _adjust_config(task_dict):
        adjusted_task_dict = {}
        for task_name, task_obj in task_dict.items():
            if isinstance(task_obj, dict):
                adjusted_task_dict = {
                    **adjusted_task_dict,
                    **{task_name: _adjust_config(task_obj)},
                }

            else:
                task_obj = task_dict[task_name]
                if type(task_obj) == tuple:
                    group, task_obj = task_obj
                    if task_obj is None:
                        continue
                lm.task_dict[task_name] = task_obj.dataset
                if "generate_until" in task_obj.get_config("output_type"):
                    if gen_kwargs is not None:
                        task_obj.set_config(key="generation_kwargs", value=gen_kwargs, update=True)

                if predict_only:
                    eval_logger.info(f"Processing {task_name} in output-only mode. Metrics will not be calculated!")
                    # we have to change the class properties post-hoc. This is pretty hacky.
                    task_obj.override_metric(metric_name="bypass")

                # override tasks' fewshot values to the provided num_fewshot arg value
                # except if tasks have it set to 0 manually in their configs--then we should never overwrite that
                if num_fewshot is not None:
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) == 0:
                        eval_logger.info(f"num_fewshot has been set to 0 for {task_name} in its config. Manual configuration will be ignored.")
                    else:
                        eval_logger.warning(f"Overwriting default num_fewshot of {task_name} from {default_num_fewshot} to {num_fewshot}")
                        task_obj.set_config(key="num_fewshot", value=num_fewshot)
                else:
                    # if num_fewshot not provided, and the task does not define a default one, default to 0
                    if (default_num_fewshot := task_obj.get_config("num_fewshot")) is None:
                        task_obj.set_config(key="num_fewshot", value=0)
                # fewshot_random_seed set for tasks, even with a default num_fewshot (e.g. in the YAML file)
                task_obj.set_fewshot_seed(seed=fewshot_random_seed)
                # eval_logger.info(f"Setting fewshot random generator seed to {fewshot_random_seed}")

                # Handle repeated generations for model stability measurement (k-samples mode)
                if repeats > 1:
                    default_repeats = task_obj.get_config("repeats") or 1
                    eval_logger.info(f"[Model Stability] Setting repeats={repeats} for {task_name} (was: {default_repeats})")
                    task_obj.set_config(key="repeats", value=repeats)

                adjusted_task_dict[task_name] = task_obj

        return adjusted_task_dict

    task_dict = _adjust_config(task_dict)

    if check_integrity:
        run_task_tests(task_list=tasks)

    if evaluation_tracker is not None:
        evaluation_tracker.general_config_tracker.log_experiment_args(
            model_source=model,
            model_args=model_args,
            system_instruction=system_instruction,
            chat_template=lm.chat_template if apply_chat_template else None,
            fewshot_as_multiturn=fewshot_as_multiturn,
        )

    from lmms_eval.models.model_utils.gen_metrics import reset_logged_metrics

    reset_logged_metrics()
    reset_usage_metrics()
    if max_tokens is not None:
        set_budget(max_tokens=max_tokens)

    # Getting the rank settings
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    response_cache = None
    if use_cache is not None:
        response_cache = ResponseCache.create(
            cache_root=use_cache,
            model=model,
            model_args=model_args,
            task_dict=task_dict,
            world_size=world_size,
            global_rank=global_rank,
        )

    eval_succeeded = False
    try:
        results = evaluate(
            lm=lm,
            task_dict=task_dict,
            limit=limit,
            offset=offset,
            cache_requests=cache_requests,
            rewrite_requests_cache=rewrite_requests_cache,
            bootstrap_iters=bootstrap_iters,
            write_out=write_out,
            log_samples=True if predict_only else log_samples,
            system_instruction=system_instruction,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            verbosity=verbosity,
            distributed_executor_backend=distributed_executor_backend,
            cli_args=cli_args,
            eval_server_launcher=eval_launcher,
            response_cache=response_cache,
        )
        eval_succeeded = True
    finally:
        if response_cache is not None:
            response_cache.finalize(
                success=eval_succeeded,
                dist_backend=distributed_executor_backend,
                accelerator=getattr(lm, "accelerator", None),
            )
        try:
            if pruning_tracker is not None and eval_succeeded:
                pruning_payload = gather_pruning_payload(pruning_tracker.export_state())
            if representation_tracker is not None and eval_succeeded:
                representation_payload = gather_representation_payload(representation_tracker.export_state())
                if representation_payload is not None:
                    if int(representation_payload.get("num_contexts", 0)) == 0:
                        raise RuntimeError(
                            "Representation dynamics was enabled, but no generation context reached the tracker. "
                            "Refusing to emit an empty successful artifact."
                        )
                    if int(representation_payload.get("num_records", 0)) == 0:
                        raise RuntimeError(
                            "Representation dynamics was enabled, but no activation records were captured. "
                            "Inspect wrapper context propagation, selected hooks, and phase settings."
                        )
        finally:
            if pruning_tracker is not None:
                try:
                    if getattr(pruning_target, "_lmms_eval_pruning_tracker", None) is pruning_tracker:
                        delattr(pruning_target, "_lmms_eval_pruning_tracker")
                except Exception:
                    pass
                pruning_tracker.remove_hooks()
            if representation_tracker is not None:
                try:
                    if getattr(representation_target, "_lmms_eval_representation_tracker", None) is representation_tracker:
                        delattr(representation_target, "_lmms_eval_representation_tracker")
                except Exception:
                    pass
                try:
                    if getattr(lm, "_lmms_eval_representation_tracker", None) is representation_tracker:
                        delattr(lm, "_lmms_eval_representation_tracker")
                except Exception:
                    pass
                representation_tracker.remove_hooks()
    if global_rank == 0:
        from lmms_eval.models.model_utils.gen_metrics import summarize_logged_metrics

        if isinstance(model, str):
            model_name = model
        elif hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
            model_name = model.config._name_or_path
        else:
            model_name = type(model).__name__

        # add info about the model and few shot config
        results["config"] = {
            "model": model_name,
            "model_args": redact_sensitive_data(model_args),
        }
        # add more detailed model info if available TODO: add model info
        # if isinstance(lm, lmms_eval.models.huggingface.HFLM):
        #     results["config"].update(lm.get_model_info())
        # add info about execution
        results["config"].update(
            {
                "batch_size": batch_size,
                "batch_sizes": (list(lm.batch_sizes.values()) if hasattr(lm, "batch_sizes") else []),
                "device": device,
                "use_cache": use_cache,
                "limit": limit,
                "offset": offset,
                "bootstrap_iters": bootstrap_iters,
                "gen_kwargs": redact_sensitive_data(gen_kwargs),
                "random_seed": random_seed,
                "numpy_seed": numpy_random_seed,
                "torch_seed": torch_random_seed,
                "fewshot_seed": fewshot_random_seed,
            }
        )
        if pruning_enabled:
            pruning_config = {
                "enabled": True,
                "attached": pruning_tracker is not None,
                "force_eager_attention": force_eager_attention,
                "force_output_attentions": force_output_attentions,
            }
            if pruning_target_name is not None:
                pruning_config["target_module"] = pruning_target_name
            if explicit_attention_config is not None:
                pruning_config["explicit_attention_config"] = explicit_attention_config
            pruning_config["update_stride"] = int(getattr(cli_args, "pruning_tracker_stride", 1) or 1)
            pruning_config["max_histogram_samples"] = int(getattr(cli_args, "pruning_tracker_max_histogram_samples", 65536) or 65536)
            if pruning_payload is not None:
                pruning_config.update(
                    {
                        "num_hooks": pruning_payload.get("num_hooks", 0),
                        "num_mlp_modules": len(pruning_payload.get("mlp_stats", {})),
                        "num_attn_modules": len(pruning_payload.get("attn_stats", {})),
                        "num_linear_state_modules": len(pruning_payload.get("linear_state_counts", {})),
                        "num_linear_state_components": len(pruning_payload.get("linear_state_component_stats", {})),
                    }
                )
                results["_pruning_tracker_data"] = pruning_payload
                if not pruning_payload.get("attn_stats"):
                    eval_logger.warning(
                        "Pruning tracker did not capture explicit attention maps. "
                        "HF eager attention was requested where possible, but this model/backend may not expose "
                        "square (B, H, N, N) attention tensors on the hooked modules. "
                        "Linear-attention modules often do not materialize such maps.",
                    )
            results["config"]["pruning_tracker"] = pruning_config
        if representation_enabled:
            representation_config = {
                "enabled": True,
                "attached": representation_tracker is not None,
                "pooling": str(getattr(cli_args, "representation_dynamics_pooling", "last_valid")),
                "phases": str(getattr(cli_args, "representation_dynamics_phases", "prefill")),
                "decode_aggregation": str(
                    getattr(cli_args, "representation_dynamics_decode_aggregation", "mean_updates")
                ),
                "decode_max_updates": getattr(cli_args, "representation_dynamics_decode_max_updates", None),
                "storage_dtype": str(getattr(cli_args, "representation_dynamics_storage_dtype", "float32")),
            }
            if representation_target_name is not None:
                representation_config["target_module"] = representation_target_name
            if representation_payload is not None:
                representation_config.update(
                    {
                        "selected_stack": representation_payload.get("selected_stack"),
                        "num_layers": len(representation_payload.get("layer_specs", [])),
                        "num_hooks": representation_payload.get("num_hooks", 0),
                        "num_contexts": representation_payload.get("num_contexts", 0),
                        "num_records": representation_payload.get("num_records", 0),
                        "skipped": representation_payload.get("skipped", {}),
                    }
                )
                results["_representation_dynamics_data"] = representation_payload
            results["config"]["representation_dynamics"] = representation_config
        # Store full resolved CLI args for reproducibility
        if cli_args is not None:
            results["config"]["resolved_cli_args"] = resolved_cli_args_for_logging(cli_args)

        results["git_hash"] = get_git_commit_hash()
        results["git_branch"] = get_git_branch_name()
        results["lmms_eval_version"] = get_lmms_eval_version_string()
        results["date"] = datetime_str
        throughput_summary = summarize_logged_metrics()
        if throughput_summary:
            results["throughput"] = throughput_summary
        usage_summary = summarize_usage_metrics()
        results["usage"] = usage_summary
        efficiency_summary = build_efficiency_summary(results)
        if efficiency_summary:
            results["efficiency"] = efficiency_summary
        # add_env_info(results)  # additional environment info to results
        # add_tokenizer_info(results, lm)  # additional info about tokenizer

        # Baseline comparison (paired t-test)
        if baseline:
            baseline_display_name = get_baseline_display_name(baseline)

            for task_name in results.get("results", {}).keys():
                try:
                    baseline_scores_dict, baseline_agg = load_baseline(baseline, task_name)
                    # Extract current scores from samples
                    if "samples" in results and task_name in results["samples"]:
                        current_samples = results["samples"][task_name]
                        # Get score_key from task config, default to "score"
                        task_config = results.get("configs", {}).get(task_name, {})
                        score_key = task_config.get("score_key", "score")

                        current_scores = []
                        baseline_scores = []
                        for sample in current_samples:
                            doc_id = sample.get("doc_id")
                            if doc_id in baseline_scores_dict:
                                # Extract score: first try exact score_key, then search for *_score fields
                                score = None
                                if score_key in sample:
                                    val = sample[score_key]
                                    if isinstance(val, (int, float)):
                                        score = float(val)
                                    elif isinstance(val, dict) and "score" in val:
                                        score = float(val["score"])
                                # Fallback: search for fields ending with "_score" (e.g., videomme_perception_score)
                                if score is None:
                                    for key in sample:
                                        if key.endswith("_score") and key != score_key:
                                            val = sample[key]
                                            if isinstance(val, (int, float)):
                                                score = float(val)
                                                break
                                            elif isinstance(val, dict) and "score" in val:
                                                score = float(val["score"])
                                                break
                                if score is not None:
                                    current_scores.append(score)
                                    baseline_scores.append(baseline_scores_dict[doc_id])

                        if current_scores and baseline_scores:
                            comparison = compute_baseline_comparison(current_scores, baseline_scores, baseline_display_name)
                            task_results = results["results"][task_name]
                            task_results["paired_baseline"] = comparison["baseline_name"]
                            task_results["paired_baseline_score"] = comparison["baseline_mean"] * 100
                            task_results["paired_ci_lower"] = comparison["ci_lower"] * 100
                            task_results["paired_ci_upper"] = comparison["ci_upper"] * 100
                            task_results["paired_pvalue"] = comparison["p_value"]
                            eval_logger.info(f"[Baseline] {task_name}: diff={comparison['mean_diff']*100:.2f}%, p={comparison['p_value']:.4f}")
                        else:
                            eval_logger.debug(f"[Baseline] Skipping {task_name}: no valid scores found with score_key='{score_key}'")
                except Exception as e:
                    eval_logger.warning(f"[Baseline] Failed for {task_name}: {e}")

        return results
    else:
        return None


decontaminate_suffix = "_decontaminate"


def _run_generate_until_agentic(
    lm,
    requests: list[Instance],
    agentic_trace_mode: str = "basic",
    response_cache: Optional[ResponseCache] = None,
) -> list[str]:
    responses: list[str] = []

    for req in requests:
        (
            current_context,
            generation_kwargs,
            current_doc_to_visual,
            doc_to_text,
            doc_id,
            task_name,
            split,
        ) = req.args

        if not callable(doc_to_text):
            raise ValueError("generate_until_agentic requires callable doc_to_text")

        max_agentic_steps = int(generation_kwargs.get("max_agentic_steps", 12))
        base_generation_kwargs = copy.deepcopy(generation_kwargs)
        base_generation_kwargs.pop("max_agentic_steps", None)

        model_outputs: list[str] = []
        previous_round_info = None
        final_response = ""
        full_round_trace: list[dict] = []

        for round_idx in range(max_agentic_steps):
            round_input_context = current_context
            if getattr(lm, "is_simple", False):
                single_req = Instance(
                    request_type="generate_until",
                    arguments=(current_context, copy.deepcopy(base_generation_kwargs), current_doc_to_visual, doc_id, task_name, split),
                    idx=0,
                    metadata=req.metadata,
                )
            else:
                current_doc = lm.task_dict[task_name][split][doc_id]

                def _agentic_doc_to_messages(_doc):
                    visuals = current_doc_to_visual(_doc)
                    if visuals is None:
                        visuals = []
                    content = []
                    for visual in visuals:
                        if isinstance(visual, dict):
                            content.append({"type": "audio", "url": visual})
                        elif isinstance(visual, str):
                            content.append({"type": "video", "url": visual})
                        else:
                            content.append({"type": "image", "url": visual})
                    content.append({"type": "text", "text": current_context})
                    return [{"role": "user", "content": content}]

                single_req = Instance(
                    request_type="generate_until",
                    arguments=(current_context, _agentic_doc_to_messages, copy.deepcopy(base_generation_kwargs), doc_id, task_name, split),
                    idx=0,
                    metadata=req.metadata,
                )
            if response_cache is not None:
                current_raw_output = response_cache.execute(lm, "generate_until", [single_req])[0]
            else:
                current_raw_output = lm.generate_until([single_req])[0]
            current_output, _ = unwrap_generation_output(current_raw_output)
            model_outputs.append(current_output)
            final_response = current_output

            step_payload = doc_to_text(
                lm.task_dict[task_name][split][doc_id],
                previous_output=model_outputs,
                round_idx=round_idx + 1,
                previous_round_info=previous_round_info,
            )

            if isinstance(step_payload, tuple) and len(step_payload) == 5:
                visuals, next_context, terminal_signal, updated_outputs, next_round_info = step_payload
                if updated_outputs is not None:
                    model_outputs = list(updated_outputs)
                    if model_outputs:
                        final_response = model_outputs[-1]
                previous_round_info = next_round_info

                if agentic_trace_mode == "full":
                    round_record = {
                        "round_idx": round_idx + 1,
                        "round_input": round_input_context,
                        "model_output": current_output,
                        "terminal": bool(terminal_signal),
                    }
                    if isinstance(next_round_info, dict):
                        round_record["state"] = next_round_info.get("state")
                        round_record["tool_result"] = next_round_info.get("last_tool_result")
                        round_record["tool_calls"] = next_round_info.get("tool_calls")
                        round_record["valid_tool_calls"] = next_round_info.get("valid_tool_calls")
                        round_record["invalid_steps"] = next_round_info.get("invalid_steps")
                    if next_context is not None:
                        round_record["next_input"] = next_context
                    full_round_trace.append(round_record)

                if terminal_signal:
                    break

                if next_context is not None:
                    current_context = next_context
                if visuals is not None:
                    current_doc_to_visual = lambda _doc, _visuals=visuals: _visuals
            elif isinstance(step_payload, str):
                current_context = step_payload
            else:
                break

        if previous_round_info is not None and not (isinstance(final_response, str) and final_response.strip().startswith("{")):
            state = previous_round_info.get("state", {}) if isinstance(previous_round_info, dict) else {}
            valid_tool_calls = float(previous_round_info.get("valid_tool_calls", previous_round_info.get("tool_calls", 0))) if isinstance(previous_round_info, dict) else 0.0
            invalid_steps = float(previous_round_info.get("invalid_steps", 0.0)) if isinstance(previous_round_info, dict) else 0.0
            fallback_payload = {
                "success": False,
                "error": "max_agentic_steps_reached",
                "tool_calls": float(previous_round_info.get("tool_calls", 0)) if isinstance(previous_round_info, dict) else 0.0,
                "valid_tool_calls": valid_tool_calls,
                "invalid_steps": invalid_steps,
                "state": state,
                "last_model_output": final_response,
                "trace": model_outputs,
            }
            if isinstance(state, dict):
                for key in ["cash", "days_elapsed", "inventory", "mobile_data_working"]:
                    if key in state:
                        fallback_payload[key] = state[key]
            final_response = json.dumps(fallback_payload, ensure_ascii=False)

        if agentic_trace_mode == "full":
            try:
                parsed_response = json.loads(final_response) if isinstance(final_response, str) else None
                if isinstance(parsed_response, dict):
                    parsed_response["agentic_trace_mode"] = "full"
                    parsed_response["agentic_rounds"] = full_round_trace
                    final_response = json.dumps(parsed_response, ensure_ascii=False)
            except (TypeError, json.JSONDecodeError):
                pass

        responses.append(final_response)

    return responses


@positional_deprecated
def evaluate(
    lm,
    task_dict,
    limit: Optional[int] = None,
    offset: int = 0,
    cache_requests: bool = False,
    rewrite_requests_cache: bool = False,
    bootstrap_iters: Optional[int] = 100000,
    write_out: bool = False,
    log_samples: bool = True,
    system_instruction: Optional[str] = None,
    apply_chat_template: bool = False,
    fewshot_as_multiturn: bool = False,
    verbosity: str = "INFO",
    distributed_executor_backend: str = "accelerate",
    eval_server_launcher: Optional[Union[str, Callable]] = None,
    cli_args=None,
    response_cache: Optional[ResponseCache] = None,
):
    """Instantiate and evaluate a model on a list of tasks.

    :param lm: obj
        Language Model
    :param task_dict: dict[str, Task]
        Dictionary of tasks. Tasks will be taken to have name type(task).config.task .
    :param limit: int, optional
        Limit the number of examples per task (only use this for testing)
    :param offset: int, optional
        Start evaluation from this dataset index for each task.
    :param bootstrap_iters:
        Number of iterations for bootstrap statistics, used when calculating stderr. Set to 0 for skipping all stderr calculations.
    :param write_out: bool
        If True, write out an example document and model input for checking task integrity
    :param log_samples: bool
        If True, write out all model outputs and documents for per-sample measurement and post-hoc analysis
    :param system_instruction: str
        System instruction to be applied to the prompt
    :param apply_chat_template: bool
        If True, apply chat template to the prompt
    :param fewshot_as_multiturn: bool
        Whether to provide the fewshot examples as a multiturn conversation or a single user turn.
    :param distributed_executor_backend: str
        The backend to use for distributed execution, `accelerate` or `torchrun`. Defaults to "accelerate" for the `accelerate` library.
    :return
        Dictionary of results
    """

    # stores the final result for each task, for each metric/filter pair.
    results = collections.defaultdict(dict)
    # Tracks each task's version.
    versions = collections.defaultdict(dict)
    # Tracks the YAML configs of all chosen tasks.
    configs = collections.defaultdict(dict)
    # logs info about each document evaluated.
    samples = collections.defaultdict(list)
    # tracks all Instances/requests a model must generate output on.
    requests = collections.defaultdict(list)
    # Aggregated task scores presented with groups
    results_agg = collections.defaultdict(dict)
    # Aggregated groups scores only
    groups_agg = collections.defaultdict(dict)
    # stores the amount to pad out reqs per req. type so that
    # number of fwd passes per distributed rank is equal
    padding_requests = collections.defaultdict(int)
    # store the hierarchy to do proper ordering
    task_hierarchy = collections.defaultdict(list)
    # store the ordering of tasks and groups
    task_order = collections.defaultdict(int)
    task_group_alias = collections.defaultdict(dict)
    # store num-fewshot value per task
    num_fewshot = collections.defaultdict(int)
    # Getting the rank settings
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    global_rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    eval_logger.info(f"Running on rank {global_rank} (local rank {local_rank})")

    def _infer_task_request_type(task_obj: Task) -> Optional[str]:
        if task_obj.instances:
            return task_obj.instances[0].request_type

        output_type = getattr(task_obj, "OUTPUT_TYPE", None)
        if output_type == "multiple_choice":
            return "loglikelihood"
        if isinstance(output_type, str):
            return output_type
        return None

    # get lists of group hierarchy and each type of request
    eval_tasks = get_task_list(task_dict)
    name_to_task = {}
    if not log_samples:
        if not all("bypass" not in getattr(task_output.task, "_metric_fn_list", {}).keys() for task_output in eval_tasks):
            raise ValueError("log_samples must be True for 'bypass' metric-only tasks")

    if distributed_executor_backend == "accelerate" and not hasattr(lm, "accelerator"):
        lm.accelerator = Accelerator()

    for task_output in eval_tasks:
        task = task_output.task
        task_name = task_output.task_name
        task.args = cli_args

        name_to_task[task_name] = task

        if type(task) == tuple:
            group_name, task = task
            task_hierarchy[group_name].append(task_name)
            versions[group_name] = "N/A"
        else:
            group_name = None
            task_hierarchy[task_name] = []

        if task is None:
            continue

        versions[task_name] = task.VERSION
        configs[task_name] = dict(task.dump_config())

        if "num_fewshot" in configs[task_name]:
            n_shot = configs[task_name]["num_fewshot"]
        else:
            n_shot = 0
        num_fewshot[task_name] = n_shot

        if "task_alias" in configs[task_name]:
            task_group_alias[task_name] = configs[task_name]["task_alias"]

        if ("group_alias" in configs[task_name]) and (group_name not in task_group_alias) and (group_name is not None):
            task_group_alias[group_name] = configs[task_name]["group_alias"]

        limit = get_sample_size(task, limit)
        task.build_all_requests(
            limit=limit,
            offset=offset,
            rank=global_rank,
            world_size=world_size,
            cache_requests=cache_requests,  # later we will add them
            rewrite_requests_cache=rewrite_requests_cache,
            system_instruction=system_instruction,
            apply_chat_template=apply_chat_template,
            fewshot_as_multiturn=fewshot_as_multiturn,
            chat_template=getattr(lm, "apply_chat_template") if apply_chat_template else None,
            tokenizer_name=getattr(lm, "tokenizer_name", "") if apply_chat_template else "",
        )
        eval_logger.debug(f"Task: {task_output.task_name}; number of requests on this rank: {len(task._instances)}")
        if write_out:
            eval_logger.warning(
                "DEPRECATION WARNING: --write_out is deprecated and will be removed in v0.5.0. "
                "Use --log_samples instead for saving model outputs and debugging. "
                "The write_out flag only prints the first few documents and impacts performance."
            )
            print_writeout(task)
        # aggregate Instances by LM method requested to get output.
        for instance in task.instances:
            reqtype = instance.request_type
            requests[reqtype].append(instance)

        if world_size > 1:
            if distributed_executor_backend == "accelerate":
                instances_rnk = torch.tensor(len(task._instances), device=lm.device)
                gathered_item = lm.accelerator.gather(instances_rnk).cpu().detach().numpy().tolist()
            elif distributed_executor_backend == "torchrun":
                instances_rnk = torch.tensor(len(task._instances), device=lm.device)
                gathered_item = torch.zeros(world_size * 1, dtype=instances_rnk.dtype, device=lm.device)
                dist.all_gather_into_tensor(gathered_item, instances_rnk)
                gathered_item = gathered_item.cpu().detach().numpy().tolist()
            else:
                raise ValueError(f"Invalid distributed_executor_backend: {distributed_executor_backend}. Choose either 'accelerate' or 'torchrun'.")

            # "multiple_choice" task types dispatch "loglikelihood" requests.
            local_reqtype = _infer_task_request_type(task)
            reqtype = local_reqtype
            if dist.is_available() and dist.is_initialized():
                gathered_reqtypes = [None] * world_size
                dist.all_gather_object(gathered_reqtypes, local_reqtype)
                reqtype = next((rt for rt in gathered_reqtypes if rt is not None), None)

            # compute number of pseudo-batches to pad with (FSDP/DDP require even batches among ranks)
            numpad = max(gathered_item) - gathered_item[lm.rank]
            if reqtype is None:
                eval_logger.warning(f"Task: {task_output.task_name}; unable to infer request type on rank {global_rank}, skipping padding computation.")
            else:
                # todo: may not account for padding in cases like SquadV2 which has multiple req types
                padding_requests[reqtype] += numpad

    ### Run LMM on inputs, get all outputs ###
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        local_reqtypes = list(requests.keys())
        gathered_reqtypes = [None] * world_size
        dist.all_gather_object(gathered_reqtypes, local_reqtypes)
        canonical_reqtypes = []
        for rank_reqtypes in gathered_reqtypes:
            if not rank_reqtypes:
                continue
            for reqtype in rank_reqtypes:
                if reqtype not in canonical_reqtypes:
                    canonical_reqtypes.append(reqtype)
    else:
        canonical_reqtypes = list(requests.keys())

    # execute each type of request
    for reqtype in canonical_reqtypes:
        reqs = requests.get(reqtype, [])
        eval_logger.info("Running {} requests".format(reqtype))
        # create `K` copies of each request `req` based off `K = req.repeats`
        cloned_reqs = []
        pad_source = reqs[-1] if reqs else None
        for req in reqs:
            cloned_reqs.extend([req] * req.repeats)

        if (world_size > 1) and (padding_requests[reqtype] > 0):
            if pad_source is None:
                eval_logger.warning(f"Running {reqtype} requests but could not find a pad source request on rank {global_rank}; skipping rank padding.")
            else:
                for _ in range(padding_requests[reqtype]):
                    cloned_reqs.extend([pad_source] * pad_source.repeats)

        # run requests through model (with optional response cache)
        if reqtype == "generate_until_agentic":
            trace_mode = "basic"
            if cli_args is not None:
                trace_mode = getattr(cli_args, "agentic_trace_mode", "basic")
            resps = _run_generate_until_agentic(
                lm,
                cloned_reqs,
                agentic_trace_mode=trace_mode,
                response_cache=response_cache,
            )
        elif response_cache is not None:
            resps = response_cache.execute(lm, reqtype, cloned_reqs)
        else:
            resps = getattr(lm, reqtype)(cloned_reqs)

        for x, req in zip(resps, cloned_reqs):
            _append_model_response(reqtype, x, req)

        if is_budget_exceeded():
            eval_logger.warning("Token budget reached after '{}' requests. Skipping remaining request types.", reqtype)
            break

        if world_size > 1:
            if distributed_executor_backend == "accelerate":
                lm.accelerator.wait_for_everyone()
            elif distributed_executor_backend == "torchrun":
                dist.barrier()
            else:
                raise ValueError(f"Invalid distributed_executor_backend: {distributed_executor_backend}. Choose either 'accelerate' or 'torchrun'.")

    # Cleaning lm's cuda memory if you are launching llm as judge in local
    lm.clean()
    RANK = global_rank
    WORLD_SIZE = world_size
    if eval_server_launcher is not None and RANK == 0:
        eval_server_launcher.launch()

    if world_size > 1:
        if distributed_executor_backend == "accelerate":
            lm.accelerator.wait_for_everyone()
        elif distributed_executor_backend == "torchrun":
            dist.barrier()

    ### Postprocess outputs ###
    # TODO: del model here, maybe (idea: allow user to specify device of e.g. reward model separately)
    for task_output in eval_tasks:
        task = task_output.task
        task.apply_filters()
        set_task_context(task_output.task_name)

        ### Collect values of metrics on all datapoints ###
        # # unpack results and sort back in order and return control to Task
        # TODO: make it possible to use a different metric per filter
        # Pre-process task.instances to group by doc_id
        instances_by_doc_id = collections.defaultdict(list)
        for instance in task.instances:
            instances_by_doc_id[instance.doc_id].append(instance)
        # Sort instances within each group
        for instances in instances_by_doc_id.values():
            instances.sort(key=lambda x: x.idx)
        # iterate over different filters used
        local_filter_keys = list(task.instances[0].filtered_resps.keys()) if task.instances else []
        if WORLD_SIZE > 1 and dist.is_available() and dist.is_initialized():
            gathered_filter_keys = [None] * WORLD_SIZE
            dist.all_gather_object(gathered_filter_keys, local_filter_keys)
            filter_keys = []
            for rank_keys in gathered_filter_keys:
                if not rank_keys:
                    continue
                for filter_key in rank_keys:
                    if filter_key not in filter_keys:
                        filter_keys.append(filter_key)
        else:
            filter_keys = local_filter_keys

        if len(filter_keys) == 0:
            eval_logger.warning(f"Task: {task_output.task_name}; no filter keys available on rank {RANK}.")
            continue

        for filter_key in filter_keys:
            # Resolve reasoning tags for this task
            cli_reasoning_tags = getattr(cli_args, "reasoning_tags", None) if cli_args else None
            task_reasoning_tags = getattr(task.config, "reasoning_tags", None)
            reasoning_tags = parse_reasoning_tags_config(cli_value=cli_reasoning_tags, task_value=task_reasoning_tags)

            if cli_args is not None and not cli_args.process_with_media:
                doc_iterator = create_iterator(
                    enumerate(task.eval_docs_no_media),
                    rank=RANK,
                    limit=int(limit) if limit else None,
                    world_size=WORLD_SIZE,
                    offset=offset,
                )
            else:
                doc_iterator = task.doc_iterator(rank=RANK, limit=limit, world_size=WORLD_SIZE, offset=offset)
            doc_iterator_for_counting = (
                create_iterator(
                    range(len(task.test_docs())),
                    rank=RANK,
                    limit=limit,
                    world_size=WORLD_SIZE,
                    offset=offset,
                )
                if task.has_test_docs()
                else create_iterator(
                    range(len(task.validation_docs())),
                    rank=RANK,
                    limit=limit,
                    world_size=WORLD_SIZE,
                    offset=offset,
                )
            )
            total_docs = sum(1 for _ in doc_iterator_for_counting)
            pbar = tqdm(total=total_docs, desc="Postprocessing", disable=(RANK != 0))
            for doc_id, doc in doc_iterator:
                requests = instances_by_doc_id[doc_id]

                # Strip reasoning tags before scoring
                if reasoning_tags is not None:
                    for req in requests:
                        raw_resp = req.filtered_resps[filter_key]
                        req.raw_filtered_resps[filter_key] = raw_resp
                        if isinstance(raw_resp, str):
                            req.filtered_resps[filter_key] = strip_reasoning_tags(raw_resp, reasoning_tags)
                        elif isinstance(raw_resp, list):
                            req.filtered_resps[filter_key] = [strip_reasoning_tags(r, reasoning_tags) if isinstance(r, str) else r for r in raw_resp]

                metrics = task.process_results(doc, [req.filtered_resps[filter_key] for req in requests])

                # For stability metrics: compute per-sample scores when repeats > 1
                repeats = task.config.repeats if hasattr(task, "config") and hasattr(task.config, "repeats") else 1
                if repeats > 1 and len(requests) == repeats:
                    # Compute per-sample scores by calling process_results for each sample individually
                    per_sample_scores = {}
                    for req in requests:
                        sample_metrics = task.process_results(doc, [req.filtered_resps[filter_key]])
                        for metric_name, value in sample_metrics.items():
                            if metric_name not in per_sample_scores:
                                per_sample_scores[metric_name] = []
                            per_sample_scores[metric_name].append(value)
                    # Store per-sample scores grouped by doc_id
                    for metric_name, scores in per_sample_scores.items():
                        task_output.per_sample_metrics[(metric_name, filter_key)].append(scores)

                if log_samples:
                    target = task.doc_to_target(doc)
                    saved_doc = {}
                    for key, value in doc.items():
                        if not is_multimodal_content(value):
                            saved_doc[key] = value
                    filtered_arguments = []
                    for req in requests:
                        # check if req.args is a list of tuples, and each item in the list is a serializable object
                        for value in req.args:
                            if isinstance(value, (str, int, float, bool, list, dict, type(None))):
                                filtered_arguments.append(value)
                            # else:
                            #     filtered_arguments.append(_handle_non_serializable(value))

                    input_media = _collect_input_media(doc, filtered_arguments)

                    per_sample_tc = []
                    for req in requests:
                        if req.token_counts:
                            tc = req.token_counts[0]
                            per_sample_tc.append(tc.to_dict() if tc is not None else None)
                        else:
                            per_sample_tc.append(None)

                    example = {
                        "doc_id": doc_id,
                        "doc": saved_doc,
                        "target": target,
                        "arguments": filtered_arguments,
                        # Keep the established resps field unchanged for
                        # downstream consumers, and persist the actual model
                        # output under a separate, unambiguous key.
                        "resps": [req.raw_filtered_resps.get(filter_key, req.resps) for req in requests],
                        "raw_resps": _raw_responses_for_sample_log(requests),
                        "filtered_resps": [req.filtered_resps[filter_key] for req in requests],
                        "token_counts": per_sample_tc,
                        "doc_hash": hash_string(
                            json.dumps(
                                requests[0].doc,
                                indent=2,
                                default=handle_non_serializable,
                                ensure_ascii=False,
                            )
                        ),
                    }
                    if input_media:
                        example["input_media"] = input_media
                    example.update(metrics)
                    task_output.logged_samples.append(example)
                for metric, value in metrics.items():
                    task_output.sample_metrics[(metric, filter_key)].append(value)
                pbar.update(1)

            pbar.close()
        set_task_context(None)

    if WORLD_SIZE > 1:
        # if multigpu, then gather data across all ranks to rank 0
        # first gather logged samples across all ranks
        for task_output in eval_tasks:
            if log_samples:
                # for task_name, task_samples in list(samples.items()):
                full_samples = [None] * WORLD_SIZE if RANK == 0 else None
                per_rank_samples = []
                for sample in task_output.logged_samples:
                    per_rank_samples.append(sample)

                torch.distributed.gather_object(
                    obj=per_rank_samples,
                    object_gather_list=full_samples,
                    dst=0,
                )

                if RANK == 0:
                    task_output.logged_samples = list(itertools.chain.from_iterable(full_samples))

            # then collect metrics across all ranks
            # All ranks must iterate over metric keys in the SAME order,
            # otherwise gather_object calls will misalign values between keys.
            # Gather all keys, merge, sort, and broadcast a canonical order.
            # this important when returning many keys from a benchmark, to avoid misalignments between ranks.
            all_metric_keys = list(task_output.sample_metrics.keys())
            gathered_keys = [None] * WORLD_SIZE if RANK == 0 else None
            torch.distributed.gather_object(
                obj=all_metric_keys,
                object_gather_list=gathered_keys,
                dst=0,
            )

            if RANK == 0:
                all_keys_set = set()
                for rank_keys in gathered_keys:
                    if rank_keys:
                        all_keys_set.update(rank_keys)
                canonical_keys = sorted(all_keys_set, key=lambda x: str(x))
            else:
                canonical_keys = None

            broadcast_list = [canonical_keys] if RANK == 0 else [None]
            torch.distributed.broadcast_object_list(broadcast_list, src=0)
            canonical_keys = broadcast_list[0]

            for metrics in canonical_keys:
                if metrics in task_output.sample_metrics:
                    pre_gather = task_output.sample_metrics[metrics]
                else:
                    pre_gather = []

                metric_list = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=pre_gather,
                    object_gather_list=metric_list,
                    dst=0,
                )
                if RANK == 0:
                    task_output.sample_metrics[metrics] = list(itertools.chain.from_iterable(metric_list))

            # gather per_sample_metrics for stability metrics (same canonical ordering)
            all_ps_keys = list(task_output.per_sample_metrics.keys())
            gathered_ps_keys = [None] * WORLD_SIZE if RANK == 0 else None
            torch.distributed.gather_object(
                obj=all_ps_keys,
                object_gather_list=gathered_ps_keys,
                dst=0,
            )

            if RANK == 0:
                all_ps_set = set()
                for rank_keys in gathered_ps_keys:
                    if rank_keys:
                        all_ps_set.update(rank_keys)
                canonical_ps_keys = sorted(all_ps_set, key=lambda x: str(x))
            else:
                canonical_ps_keys = None

            broadcast_ps = [canonical_ps_keys] if RANK == 0 else [None]
            torch.distributed.broadcast_object_list(broadcast_ps, src=0)
            canonical_ps_keys = broadcast_ps[0]

            for metrics in canonical_ps_keys:
                if metrics in task_output.per_sample_metrics:
                    pre_gather = task_output.per_sample_metrics[metrics]
                else:
                    pre_gather = []

                metric_list = [None] * WORLD_SIZE if RANK == 0 else None
                torch.distributed.gather_object(
                    obj=pre_gather,
                    object_gather_list=metric_list,
                    dst=0,
                )
                if RANK == 0:
                    task_output.per_sample_metrics[metrics] = list(itertools.chain.from_iterable(metric_list))

        dist.barrier()  # Ensure all processes are synced before proceeding

    if RANK == 0:
        if eval_server_launcher is not None:
            eval_server_launcher.clean()
        ### Aggregate results over all datapoints ###
        # aggregate results ; run bootstrap CIs
        for task_output in eval_tasks:
            task_output.calculate_aggregate_metric(bootstrap_iters=bootstrap_iters)
            task_output.calculate_clt_aggregate_metric()
            task_output.calculate_stability_metrics()
        (
            results,
            samples,
            configs,
            versions,
            num_fewshot,
            higher_is_better,
        ) = consolidate_results(eval_tasks)

        ### Calculate group metrics ###
        if bool(results):
            results, versions, show_group_table, *_ = consolidate_group_results(results, versions, task_dict)

        results_agg, group_agg = prepare_print_tasks(task_dict, results)
        subtask_list = get_subtask_list(task_dict)

        # collect all higher_is_better values for metrics
        # in the group's subtasks.
        # TODO: clean this up ; unify with the below metric_list loop?
        _higher_is_better = {}
        for group, task_list in subtask_list.items():
            if len(task_list) != 0:  # subtask list will list "task_name": [] for solo tasks
                for task in task_list:
                    for m, h in higher_is_better[task].items():
                        if m not in _higher_is_better.keys():
                            _higher_is_better[m] = h

                        if m in _higher_is_better and _higher_is_better[m] is not None and _higher_is_better[m] != h:
                            eval_logger.warning(f"Higher_is_better values for metric {m} in group {group} are not consistent. Defaulting to None.")
                            _higher_is_better[m] = None
                higher_is_better[group] = _higher_is_better

        results_dict = {
            "results": dict(results_agg.items()),
            **({"groups": dict(group_agg.items())} if (bool(group_agg) & show_group_table) else {}),
            "group_subtasks": dict(reversed(subtask_list.items())),
            "configs": dict(sorted(configs.items())),
            "versions": dict(sorted(versions.items())),
            "n-shot": dict(sorted(num_fewshot.items())),
            "higher_is_better": dict(sorted(higher_is_better.items())),
            "n-samples": {
                task_output.task_name: {
                    "original": len(task_output.task.eval_docs),
                    "effective": min(
                        limit if limit else len(task_output.task.eval_docs),
                        len(task_output.task.eval_docs),
                    ),
                }
                for task_output in eval_tasks
            },
        }
        if log_samples:
            results_dict["samples"] = dict(samples)
    else:
        results_dict = None

    if WORLD_SIZE > 1:
        # if muti-gpu, wait for all processes to finish
        if distributed_executor_backend == "accelerate":
            # this should work for torchrun as well since it internally calls torch.distributed.barrier()
            Accelerator().wait_for_everyone()
        elif distributed_executor_backend == "torchrun":
            dist.barrier()
        else:
            raise ValueError(f"Invalid distributed_executor_backend: {distributed_executor_backend}. Choose either 'accelerate' or 'torchrun'.")

    return results_dict


def request_caching_arg_to_dict(cache_requests: str) -> dict:
    request_caching_args = {
        "cache_requests": cache_requests in {"true", "refresh"},
        "rewrite_requests_cache": cache_requests == "refresh",
        "delete_requests_cache": cache_requests == "delete",
    }

    return request_caching_args
