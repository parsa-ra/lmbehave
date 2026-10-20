> **Research fork.** Branch `research/representation-dynamics` adds an opt-in, read-only
> activation tracker and a task-level inference toolkit to `lmms-eval`. They were used for
> *Do Knowledge- and Reasoning-Oriented Benchmarks Induce Stable Representation Signatures in
> LLMs?* (NeurIPS 2026 Workshop on Interpretability as a Science; project page:
> <https://parsa-ra.github.io/lmbehave/>). The study-specific documentation is in the section
> [Representation-dynamics study](#representation-dynamics-study) at the end of this file; the
> upstream `lmms-eval` README follows unchanged.

<p align="center" width="70%">
<img src="https://i.postimg.cc/KvkLzbF9/WX20241212-014400-2x.png">
</p>

# LMMs-Eval: Probing Intelligence in the Real World

[![PyPI](https://img.shields.io/pypi/v/lmms-eval)](https://pypi.org/project/lmms-eval)
![PyPI - Downloads](https://img.shields.io/pypi/dm/lmms-eval)
[![GitHub contributors](https://img.shields.io/github/contributors/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/graphs/contributors)
[![issue resolution](https://img.shields.io/github/issues-closed-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)
[![open issues](https://img.shields.io/github/issues-raw/EvolvingLMMs-Lab/lmms-eval)](https://github.com/EvolvingLMMs-Lab/lmms-eval/issues)

> We are building the unified evaluation toolkit for frontier models and probing the abilities in real world, shape what we build next.

<details>
<summary>🌐 Available in 17 languages</summary>

[简体中文](docs/i18n/README_zh-CN.md) | [繁體中文](docs/i18n/README_zh-TW.md) | [日本語](docs/i18n/README_ja.md) | [한국어](docs/i18n/README_ko.md) | [Español](docs/i18n/README_es.md) | [Français](docs/i18n/README_fr.md) | [Deutsch](docs/i18n/README_de.md) | [Português](docs/i18n/README_pt-BR.md) | [Русский](docs/i18n/README_ru.md) | [Italiano](docs/i18n/README_it.md) | [Nederlands](docs/i18n/README_nl.md) | [Polski](docs/i18n/README_pl.md) | [Türkçe](docs/i18n/README_tr.md) | [العربية](docs/i18n/README_ar.md) | [हिन्दी](docs/i18n/README_hi.md) | [Tiếng Việt](docs/i18n/README_vi.md) | [Indonesia](docs/i18n/README_id.md)

</details>

📚 [Documentation](docs/README.md) | 📖 [100+ Tasks](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/advanced/current_tasks.md) | 🌟 [30+ Models](https://github.com/EvolvingLMMs-Lab/lmms-eval/tree/main/lmms_eval/models) | ⚡ [Quickstart](docs/getting-started/quickstart.md)

🏠 [Homepage](https://www.lmms-lab.com/) | 💬 [Discord](https://discord.gg/8xTM6jWnXa) | 🤝 [Contributing](CONTRIBUTING.md)

---

## Why `lmms-eval`?

Benchmarks decide what gets built next. A model team that trusts its eval numbers can focus on real improvements instead of chasing noise. But the multimodal evaluation ecosystem is fragmented - scattered datasets, inconsistent post-processing, and single-number accuracy scores that hide whether a gain is real or random. Two teams evaluating the same model on the same benchmark routinely report different results.

We believe [better evals lead to better models](https://arxiv.org/pdf/2211.09110). Good evaluation maps the border of what models can do and shapes what we build next.

We are building `lmms-eval` and focusing on three core principles:

- **Reproducible** - One pipeline, deterministic results. Same model, same benchmark, same numbers, every time.
- **Efficient** - Evaluation should not be the bottleneck, even at large scale. Async serving, adaptive batching, and video I/O optimizations keep your GPUs saturated end to end.
- **Trustworthy** - Not just accuracy. Confidence intervals, clustered standard errors, paired comparisons, and ongoing research into evaluation methodology. Results you can trust enough to act on.

For how the pipeline works and the concrete mechanisms behind these principles, see [How the Evaluation Pipeline Works](docs/README.md#how-the-evaluation-pipeline-works) and [Why it's Efficient and Trustworthy](docs/README.md#why-its-efficient-and-trustworthy).

## What's New

**v0.7** (Feb 2026) - Operational simplicity and pipeline maturity. 25+ new tasks across 8 domains, 2 new model backends, agentic task evaluation (`generate_until_agentic`), video I/O overhaul with TorchCodec (up to 3.58x faster), Lance-backed video distribution on Hugging Face, safety/red-teaming baselines, efficiency metrics (per-sample token counts, run-level throughput), and streamlined flattened JSONL log output for cleaner post-analysis. [Release notes](docs/releases/lmms-eval-0.7.md) | [Changelog](docs/releases/CHANGELOG.md).

**v0.6** (Feb 2026) - Evaluation as a service. Standalone HTTP eval server, ~7.5x throughput over v0.5, statistically grounded results (CI, paired t-test), 50+ new tasks. [Release notes](docs/releases/lmms-eval-0.6.md) | [Changelog](docs/releases/CHANGELOG.md).

**v0.5** (Oct 2025) - Audio expansion. Comprehensive audio evaluation, response caching, 50+ benchmark variants across audio, vision, and reasoning. [Release notes](docs/releases/lmms-eval-0.5.md).

<details>
<summary>Older updates</summary>

- [2025-01] [Video-MMMU](https://arxiv.org/abs/2501.13826) - Knowledge acquisition from multi-discipline professional videos.
- [2024-12] [MME-Survey](https://arxiv.org/pdf/2411.15296) - Comprehensive survey on evaluation of multimodal LLMs.
- [2024-11] **v0.3** - Audio evaluation support (Qwen2-Audio, Gemini-Audio). [Release notes](docs/releases/lmms-eval-0.3.md).
- [2024-06] **v0.2** - Video evaluation (LLaVA-NeXT Video, Gemini 1.5 Pro, VideoMME, EgoSchema). [Blog](https://lmms-lab.github.io/posts/lmms-eval-0.2/).
- [2024-03] **v0.1** - First release. [Blog](https://lmms-lab.github.io/posts/lmms-eval-0.1/).

</details>

## Quickstart

Install and run your first evaluation in under 5 minutes:

```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval.git
cd lmms-eval && uv pip install -e ".[all]"

# Run a quick evaluation (Qwen2.5-VL on MME, 8 samples)
python -m lmms_eval \
  --model qwen2_5_vl \
  --model_args pretrained=Qwen/Qwen2.5-VL-3B-Instruct \
  --tasks mme \
  --batch_size 1 \
  --limit 8
```

If it prints metrics, your environment is ready. For the full guide, see [`docs/getting-started/quickstart.md`](docs/getting-started/quickstart.md).

## Installation

### Using `uv` (Recommended for consistent environments)

We use `uv` for package management to ensure all developers use exactly the same package versions. First, install uv:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For development with consistent environment:
```bash
git clone https://github.com/EvolvingLMMs-Lab/lmms-eval
cd lmms-eval
# Recommend
uv pip install -e ".[all]"
# If you want to use uv sync
# uv sync  # This creates/updates your environment from uv.lock
```

To run commands:
```bash
uv run python -m lmms_eval --help  # Run any command with uv run
```

To add new dependencies:
```bash
uv add <package>  # Updates both pyproject.toml and uv.lock
```

### Alternative Installation

For direct usage from Git:
```bash
uv venv eval
uv venv --python 3.12
source eval/bin/activate
# You might need to add and include your own task yaml if using this installation
uv pip install git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git
```

<details>
<summary>Reproduction of LLaVA-1.5's paper results</summary>

You can check the [torch environment info](miscs/repr_torch_envs.txt) and [results check](miscs/llava_result_check.md) to **reproduce LLaVA-1.5's paper results**. We found torch/cuda versions difference would cause small variations in the results.

</details>

If you want to test on caption dataset such as `coco`, `refcoco`, and `nocaps`, you will need to have `java==1.8.0` to let pycocoeval api to work. If you don't have it, you can install by using conda
```
conda install openjdk=8
```
you can then check your java version by `java -version`


<details>
<summary>Comprehensive Evaluation Results of LLaVA Family Models</summary>
<br>

As demonstrated by the extensive table below, we aim to provide detailed information for readers to understand the datasets included in lmms-eval and some specific details about these datasets (we remain grateful for any corrections readers may have during our evaluation process).

We provide a Google Sheet for the detailed results of the LLaVA series models on different datasets. You can access the sheet [here](https://docs.google.com/spreadsheets/d/1a5ImfdKATDI8T7Cwh6eH-bEsnQFzanFraFUgcS9KHWc/edit?usp=sharing). It's a live sheet, and we are updating it with new results.

<p align="center" width="100%">
<img src="https://i.postimg.cc/jdw497NS/WX20240307-162526-2x.png"  width="100%" height="80%">
</p>

We also provide the raw data exported from Weights & Biases for the detailed results of the LLaVA series models on different datasets. You can access the raw data [here](https://docs.google.com/spreadsheets/d/1AvaEmuG4csSmXaHjgu4ei1KBMmNNW8wflOD_kkTDdv8/edit?usp=sharing).

</details>
<br>

If you want to test [VILA](https://github.com/NVlabs/VILA), you should install the following dependencies:

```bash
pip install s2wrapper@git+https://github.com/bfshi/scaling_on_scales
```

Our Development will be continuing on the main branch, and we encourage you to give us feedback on what features are desired and how to improve the library further, or ask questions, either in issues or PRs on GitHub.

## Usage Examples

> More examples can be found in [examples/models](examples/models)

### Evaluation with vLLM

**Qwen2.5-VL:**
```bash
bash examples/models/vllm_qwen2vl.sh
```

**Qwen3-VL:**
```bash
bash examples/models/vllm_qwen3vl.sh
```

**Qwen3.5:**
```bash
bash examples/models/vllm_qwen35.sh
```

### Evaluation with SGLang

```bash
bash examples/models/sglang.sh
```

**Qwen3.5:**
```bash
bash examples/models/sglang_qwen35.sh
```

### Evaluation of OpenAI-Compatible Model

```bash
bash examples/models/openai_compatible.sh
```

### Evaluation of Qwen2.5-VL

```bash
bash examples/models/qwen25vl.sh
```

### Evaluation of Qwen3-VL

```bash
bash examples/models/qwen3vl.sh
```

**More Parameters**

```bash
python3 -m lmms_eval --help
```

### Collect and Analyze Representation Dynamics

`lmms-eval` provides an opt-in, per-example tracker for objective comparison of
knowledge-heavy and capability-first/reasoning task sets. It targets the semantic
language-decoder stack and records three pooled vectors for every decoder layer:

- `attention_delta`: attention or linear-attention branch output before residual addition.
- `mlp_delta`: MLP branch output before residual addition.
- `residual_state`: decoder-layer output after its residual updates.

Gemma 3 and Qwen 3/3.5 use architecture-aware targets so the branch outputs are exact;
each record carries `residual_delta_exact` so structurally discovered fallbacks cannot be
silently interpreted as exact residual contributions. This tracker is separate from
`PruningTracker`: it does not request eager attention, retain attention maps, replace
outputs, modify weights, or enable pruning.

```bash
accelerate launch --num_processes=1 -m lmms_eval \
  --model qwen3_5 \
  --model_args pretrained="$MODEL",enable_thinking=false \
  --tasks "$TASK" \
  --batch_size 1 \
  --output_path ./logs/results \
  --representation_dynamics \
  --representation_dynamics_output "./logs/representations/$TASK" \
  --representation_dynamics_pooling last_valid \
  --representation_dynamics_phases prefill \
  --representation_dynamics_storage_dtype float16
```

The default `last_valid` estimator selects the final non-padding prompt token.
`mean_valid` averages only valid prompt positions. Decode capture may average updates,
retain only the first update, or emit every update with an explicit zero-based
`decode_step`. Records include task, split,
document/sample ID, distributed rank, layer, stream, module, phase, prompt length, pooling
rule, update count, hidden size, and the vector. A run writes:

- `representations.pt`: tensor-bearing artifact for analysis.
- `records.csv`: vector-free record manifest with vector norms.
- `metadata.json`: model, evaluator revision, arguments, selected stack, counts, and skips.

Analyze artifacts with task labels supplied independently of folder names:

```bash
python -m lmms_eval.representations.analysis \
  /path/to/run-1/representations.pt /path/to/run-2/representations.pt \
  --knowledge-tasks k1,k2,k3,k4,k5 \
  --reasoning-tasks r1,r2,r3,r4,r5 \
  --output-dir /path/to/analysis \
  --cv-folds 5 --permutations 251 --seed 0
```

`layer_separability.csv` reports, per layer and stream:

- PCA-space Fisher discriminant trace ratio plus mean/maximum per-PC ratios;
- task-held-out logistic-regression accuracy and macro-F1;
- task-label permutation p-values and Benjamini--Hochberg q-values;
- RBF-SVM accuracy and macro-F1 as the direct nonlinear probe;
- ordinary linear CKA only for explicitly row-paired `pair_id` observations;
- normalized feature-covariance alignment for unpaired task sets; and
- cosine silhouette in the original high-dimensional feature space.

`report.json` identifies extrema globally and separately per model, calibrates the
maximum and BH rejection count over balanced task partitions, and predeclares each model's
maximum FDR as its primary divergence location. `permutation_nulls.json.gz` preserves the
aligned sample-free null score vectors. CKA and covariance alignment are
similarity scores, so their minimum is the largest divergence. No t-SNE or UMAP quantity
is computed or used as evidence. A separate branch-only field identifies whether the
largest attention/MLP divergence occurs in `attention_delta` or `mlp_delta`.
Use `--fdr-permutations` independently of `--permutations` when an exhaustive Fisher
maximum is required but an exhaustive probe refit at every layer would be wasteful. A
separate count does not itself provide the secondary probe at the Fisher-selected row.

Scientific constraints:

- Cross-validation holds out complete tasks, not random examples. Five folds require at
  least five tasks per category. With two tasks per category, only three complementary
  class-symmetric outcomes exist. Five per category gives 252 oriented assignments but
  only 126 complementary outcomes, so the exact p-value floor is `1/126`. Exhaustive
  balanced linear probes fit one orientation per complementary pair and store the same
  score for both orientations; this preserves the full randomization support while
  halving duplicate fits. RBF probes are not deduplicated because their numerical
  complement symmetry is itself a required diagnostic.
- `p < 0.01` is evidence against the task-label null, not proof. Use q-values when screening
  layers/modules and confirm the selected layer on held-out tasks.
- Standard CKA requires row alignment. It is `NaN` for unpaired benchmarks by design;
  covariance alignment is the unpaired second-order comparison.
- Task-category prediction can reflect modality, domain, length, prompt template,
  correctness, or decoding. Freeze these factors or measure them as controls before
  interpreting the axis as knowledge versus reasoning.
- Hook outputs are passive, but observer parity still requires identical tracker-off and
  tracker-on calibration runs. Measure production latency with tracking disabled.

The tracker implementation is in `lmms_eval/representations/tracker.py`; quantitative
analysis is in `lmms_eval/representations/analysis.py`; evaluator lifecycle and artifact
writing are in `lmms_eval/evaluator.py` and `lmms_eval/__main__.py`; Qwen/Gemma wrappers
supply sample identity and masks around their unchanged `generate()` calls.



### Collect Pruning Statistics

`lmms-eval` can optionally attach a forward-hook tracker during evaluation to collect MLP/channel and attention-head statistics for downstream structured pruning.

```bash
accelerate launch --num_processes=8 --main_process_port=12346 -m lmms_eval \
  --model qwen3_5 \
  --model_args pretrained=$MODEL,max_pixels=$MAX_PIXELS,min_pixels=$MIN_PIXELS,max_frames=$MAX_NUM_FRAMES,enable_thinking=false \
  --gen_kwargs max_new_tokens=$MAX_NEW_TOKENS,temperature=0.7,top_p=0.8,top_k=20,repetition_penalty=1.0 \
  --tasks $TASKS \
  --batch_size $BATCH_SIZE \
  --output_path ./logs/ \
  --pruning_tracker \
  --pruning_tracker_output ./logs/pruning/ \
  --pruning_tracker_stride 4
```

This feature is implemented as a generic `PruningTracker` that is attached once, right after the model is constructed inside `lmms_eval.evaluator.simple_evaluate()`. The integration is intentionally model-agnostic, so it works from the `lmms_eval` entry point without modifying individual model wrappers.

When `--pruning_tracker` is enabled, the entry point prioritizes real attention scores without globally retaining all attentions: it tries to pass `attn_implementation=eager` to wrappers that expose that constructor argument, then the hooks consume attention tensors directly from attention-module outputs. HF-style global `output_attentions=True` is kept disabled by default because generation can keep every layer's full `B x H x N x N` attention maps and cause OOM. Projection-based attention proxies have been removed so the reported attention metrics are actual attention-map statistics only.

**What The Implementation Does**

- Resolves the actual `torch.nn.Module` to track from the `lmms-eval` model wrapper. It handles direct modules, wrappers exposing `.model` or `._model`, and distributed wrappers that need `accelerator.unwrap_model(...)`.
- Iterates over `model.named_modules()` and registers forward hooks dynamically.
- Registers an MLP hook for module names containing `mlp`, `ffn`, or `dense`.
- Excludes `dense` projections that live under attention paths, so attention output projections are not accidentally mixed into `mlp_stats`.
- Registers an attention hook for modules whose names or classes indicate `attention` or `attn`.
- Registers a separate linear-state hook for modules whose own names or classes indicate `linear_attn`, `linear_attention`, `GatedDeltaNet`, DeltaNet, RetNet/retention, Mamba, SSM, or RWKV-style recurrent/state blocks.
- Registers lightweight component hooks on direct children of those state blocks, such as `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `conv1d`, `norm`, and `out_proj`.
- Requests explicit Hugging Face attention outputs by default for supported wrappers by forcing `attn_implementation=eager` before model load, while keeping global `output_attentions=True` off unless explicitly requested.
- Prefers full attention blocks over internal attention submodules at registration time, which reduces false positives from layers such as internal query/key normalization modules.
- Reduces tensors on-device first and only moves compact summaries to CPU, which avoids the previous bottleneck of copying full activations or attention maps out of GPU memory on every hook call.
- Supports `--pruning_tracker_stride N` to track every `N`-th hooked forward when a faster, lower-overhead calibration pass is preferred.
- Uses adaptive histograms whose bin edges are derived from retained model samples; `--pruning_tracker_max_histogram_samples` caps the number of sampled values kept per tracked module so this stays lightweight even for very large activations or attention maps.
- Uses integer-safe evenly spaced sample retention for those adaptive histograms, so very large tensors do not fail due to floating-point index rounding.
- The implementation stays model-agnostic at the module-statistics level: it does not hardcode Qwen-specific logic and instead relies on generic module structure, naming, and output-shape checks.
- Builds a structural network-order summary after the run so the saved reports can show how metrics evolve across the module sequence seen by incoming tokens.

**Registration And Extraction Rules**

The tracker uses PyTorch forward hooks only. It does not patch model internals, modify outputs, or require Qwen/Gemma-specific imports. Registration is based on `model.named_modules()` traversal, module names, class names, and output tensor shapes.

| Stream | Hook target | Registration condition | Accepted tensor |
| --- | --- | --- | --- |
| `mlp` | Feed-forward or dense channel-producing modules | Name or class contains `mlp` or `ffn`; name or class containing `dense` is accepted unless it is a dense-only match inside an attention path. | First rank >= 1 tensor found in tensor, tuple/list, dictionary, or Hugging Face `ModelOutput`. |
| `attention` | Explicit attention blocks | Name or class contains `attention` or `attn`; full blocks are preferred, and internal helper tensors are rejected later by shape checks. | A square rank-4 tensor with shape compatible with `(batch, heads, query_tokens, key_tokens)`. HF-style `output[1]` is checked first, then nested structures are searched recursively. |
| `linear_state` | Linear-attention or recurrent/state-style sequence mixers | Own basename or class contains `linear_attn`, `linear_attention`, `GatedDeltaNet`, `DeltaNet`, `retention`, `RetNet`, `Mamba`, `SSM`, or `RWKV`, and the module has children. | Primary input tensor and primary output tensor from the module call. |
| `linear_state_component` | Direct children of a registered linear-state block | Child name matches known observable projections, gates, convolutions, norms, or readouts such as `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`, `q_proj`, `k_proj`, `v_proj`, `qkv`, `conv1d`, `x_proj`, `dt_proj`, `norm`, `out_proj`, or `o_proj`. | Primary output tensor of the child module. `Conv1d` uses channel dimension 1; other modules use the last dimension as channel. |

Component roles are assigned from names only: `gate` for gate-like names such as `in_proj_z`, `in_proj_a`, `in_proj_b`, or `dt_proj`; `write` for q/k/v/qkv/input projections and convolutions; `readout` for output projections and norms; `component` otherwise.

**What "Model-Agnostic" Means Here**

- The tracker is not limited to Qwen or Gemma classes.
- The explicit-attention path is fully generic: if an attention block returns a square 4D tensor like `(B, H, N, N)`, it will be consumed regardless of model family.
- Linear-attention modules are supported through the separate `linear_state` path even when they do not expose real square attention maps.
- Many linear-attention implementations, including recurrent/state-space style blocks, do not form an explicit softmax `N x N` matrix, so the tracker skips them for `attn_stats` while still reporting block contribution and component activity under `linear_state` and `linear_state_components`.
- If `attn_stats` is empty, the model/backend did not expose explicit square attention maps to the hooked modules. This can happen for custom kernels, linear-attention layers, or wrappers whose attention modules only return maps when global `output_attentions=True` is set.

**How The Statistics Are Computed**

- Let `Y` be an MLP-like output tensor. The channel score is `s_c = mean_{all dims except last} |Y[..., c]|`. For standard hidden states `(B, N, C)`, this is the mean absolute activation per channel over batch and token positions.
- For MLP-like modules, the hook extracts the first tensor it can find from the module output. This handles plain tensors, tuples/lists, dictionaries, and Hugging Face `ModelOutput` objects.
- In the same hook, `lmms-eval` accumulates a histogram of `|Y|`, active elements using `|Y| > 1e-6`, the mean number of active neurons per token/position, the mean active fraction, and per-channel firing rates.
- Let `A` be an explicit attention tensor with accepted shape `(B, H, N, N)`. The head score is `s_h = sum |A[:, h, :, :]| / max(count(A[:, h, :, :] != 0), 1)`.
- For attention-like modules, the parent hook first searches the output structure for a 4D tensor, ideally of shape `(B, H, N, N)`.
- The tracker intentionally ignores non-square 4D tensors from internal submodules, such as reshaped query/key states, to avoid mixing them with real attention maps.
- When a real attention map is exposed, attention mass is averaged over nonzero entries only. This avoids diluting causal decoder statistics with upper-triangular zeros or padding-mask zeros.
- The attention histogram also samples nonzero attention values only, and the reports include `attention_valid_fraction` to show what fraction of each `N x N` map was nonzero.
- For explicit attention outputs, the tracker accumulates a nonzero weight histogram, per-head entropy, and per-head peak attention mass. Entropy and peak mass are computed after row-normalizing `|A|` over the key dimension.
- For linear-state modules, output magnitude and output RMS are always recorded. If a primary input tensor exists, the tracker records `output_input_rms_ratio = rms(output) / max(rms(input), 1e-12)`.
- If a linear-state input and output have identical shapes, the tracker additionally records `input_output_delta_abs_mean`, `input_output_delta_rms`, `input_output_delta_rms_ratio = rms(output - input) / max(rms(input), 1e-12)`, and flattened `output_input_cosine`.
- The same linear-state hook also records output activation and input-output delta histograms without storing full tensors.
- For direct children of linear-state modules, component hooks accumulate `s_c = mean_{all non-channel dims} |Z_c|`, activity fraction, per-channel firing rates, and histograms. Component rows include `parent_module` and a coarse role: `write`, `gate`, `readout`, or `component`.
- The generic tracker does not read private recurrent cache objects unless a model exposes them through module inputs or outputs. The current linear-state path therefore measures observable block outputs and component activations, not hidden kernel-internal state tensors.
- The reduction does not assume a fixed sequence length, so it is compatible with variable-length text and cross-modal token streams used in VLMs.
- Tensor reductions run under `torch.no_grad()` and call `detach()` before summaries are moved to CPU. Full activation tensors and full attention matrices are not retained after the hook returns.
- Histogram edges are generated from retained samples at export time. MLP and linear-state histograms use dynamic log-spaced bins, while explicit attention histograms use dynamic linear bins. Retained samples are capped by `--pruning_tracker_max_histogram_samples`.
- If a later observation for a module has a different channel/head vector shape than earlier observations, that update is skipped for that accumulator rather than combining incompatible tensors. This matters for heterogeneous or dynamically shaped modules.
- After aggregation, `lmms-eval` also organizes modules that produced statistics into network order using the recorded `named_modules()` registration sequence. This is what allows the saved reports to line up repeated Gemma-style blocks and mixed Qwen-style stacks under one ordered view of the network.

**Interpretation For Explicit And Linear Attention**

`attn_stats` contains real explicit attention-map statistics only. Projection-based attention proxies were removed because they are not attention scores.

Linear-attention and state-space modules usually do not instantiate a softmax `N x N` attention matrix. They are therefore reported under `linear_state` and `linear_state_components`, not `attn_stats`. For Qwen-style `GatedDeltaNet` blocks, the tracked parent output is the observable branch output returned by the module before the decoder residual addition. In that case, `output_input_rms_ratio` is the most direct branch-strength measure. Delta and cosine metrics compare the module's primary input and primary output; they should only be interpreted as full residual-stream change if that specific module returns a full transformed hidden state.

**What The Layer Visualizations Mean**

- `visualizations.json` now contains ordered line traces for MLP mean activation, MLP active fraction, explicit attention mean mass, attention entropy, attention peak mass, attention valid fraction, linear-state input-output delta ratio, linear-state output ratio, and linear-state output/input cosine.
- It also contains histograms and bar charts for linear-state output magnitudes, linear-state input-output deltas, and linear-state component activity.
- It also contains layer-head heatmaps for explicit attention statistics. In these heatmaps, rows follow the structural module order through the network and columns correspond to head indices.
- The file additionally stores `stage_pattern` summaries, where numeric layer indices are replaced with `*`. This makes repeated patterns such as `model.layers.*.self_attn` or `model.layers.*.mlp` easy to compare across architectures.
- These are aggregated layer/head summaries, not raw token-token attention matrices. The tracker intentionally avoids storing full `N x N` attention maps for every request because that would be too expensive in memory and I/O.

**Verification Status**

The current implementation has local regression coverage for:

- explicit square 4D attention maps,
- attention blocks with internal `q_norm`-style submodules that should not be mistaken for attention maps,
- linear-attention modules without explicit square maps being skipped for `attn_stats` while still being captured under `linear_state`,
- linear-state direct child component activity,
- causal/padding zero entries being excluded from attention means and histograms,
- and distributed payload merging.

This does not prove universal compatibility across all Transformer variants, but it does verify that the current logic is pattern-based and not hardcoded to Qwen or Gemma class names.

**Distributed Behavior**

- Each process accumulates its own local statistics during evaluation.
- At the end of the run, local payloads are gathered with `torch.distributed.gather_object(...)`.
- Rank 0 merges the per-rank tensors by summation and writes a single artifact to disk.
- Hook cleanup is automatic after evaluation to avoid handle leaks.

**Per-Run Local Report Directory**

Each run now creates a local pruning report directory rather than only a single `.pt` file.

Default layout:

- `<output_path>/pruning_stats/pruning_stats_<model>_<datetime>/`
- or `<pruning_tracker_output>/pruning_stats_<model>_<datetime>/` if `--pruning_tracker_output` is a directory
- or `<parent>/<custom_name>/` if `--pruning_tracker_output` is passed as a `.pt` filename

Typical contents:

- `pruning_stats.pt`: full raw payload with all aggregated tensors and counts.
- `summary.json`: detailed run summary with global metrics and per-module rows.
- `histograms.json`: local copy of the global MLP activation histogram and explicit attention-weight histogram.
- `visualizations.json`: plot-ready histogram, ordered-trace, bar-chart, and layer-head heatmap payloads in JSON, including whether histogram bins were built from sampled values.
- `plots/`: rendered PNG figures for the exported histogram, line, bar, and heatmap views when `matplotlib` is available.
- `plots_manifest.json`: JSON index mapping plot names such as `mlp_mean_by_network_order` to the generated local PNG files.
- `network_sequence.csv`: one row per tracked module in structural network order, with fields such as `stage_pattern`, `primary_layer_index`, and the ordered MLP/attention metrics.
- `mlp_modules.csv`: per-module MLP summary rows.
- `attention_modules.csv`: per-module attention summary rows when attention maps are available.
- `linear_state_modules.csv`: per-module linear-state block contribution rows.
- `linear_state_components.csv`: per-component activity rows for direct children of linear-state blocks.

The saved `pruning_stats.pt` payload contains:

- `metadata`: run metadata including model name, model args, selected tasks, timestamp, tracked module counts, and the local run directory.
- `mlp_stats`: dictionary mapping module name to the aggregated channel-wise score tensor.
- `attn_stats`: dictionary mapping module name to the aggregated head-wise score tensor computed over nonzero attention entries.
- `linear_state_stats`: nested dictionaries keyed by metric name, then module name, for linear-state block contribution metrics.
- `linear_state_component_stats`: dictionary mapping direct linear-state component name to channel-wise activity tensors.
- `mlp_counts`: number of hook updates accumulated for each MLP-like module.
- `attn_counts`: number of hook updates accumulated for each attention-like module.
- `linear_state_counts` and `linear_state_component_counts`: hook update counts for linear-state block and component summaries.
- `mlp_hist_counts`, `attn_hist_counts`, `linear_state_output_hist_counts`, `linear_state_input_output_delta_hist_counts`, and `linear_state_component_hist_counts`: histogram counts accumulated during the run; attention histograms exclude zero-masked entries.
- `mlp_active_elements`, `mlp_total_elements`, `mlp_position_counts`, `mlp_channel_fire_counts`: MLP activity and firing-rate aggregates.
- `attn_entropy` and `attn_peak_mass`: attention concentration summaries accumulated per head.
- `attn_valid_elements` and `attn_total_elements`: counts used to report the nonzero attention fraction per module.
- `linear_state_component_parents` and `linear_state_component_roles`: mappings used to group linear-state child components back to their parent block and coarse state role.
- `module_registration_order`: the structural module order used to build the layer-wise summaries and heatmaps.
- `tracking_config`: tracker settings such as `update_stride`, `max_histogram_samples`, and the adaptive histogram binning mode.

Example:

```python
import torch

payload = torch.load("./logs/pruning/pruning_stats_qwen3_5_<timestamp>/pruning_stats.pt")
print(payload["metadata"])
print(payload["mlp_stats"].keys())
print(payload["attn_stats"].keys())
print(payload["linear_state_stats"].keys())
```

**Standalone Hugging Face Example**

If you want to reuse the pruning code outside the `lmms_eval` CLI, there is a minimal standalone example in [`examples/pruning/hf_pruning_tracker_example.py`](examples/pruning/hf_pruning_tracker_example.py).

Example:

```bash
cd lmms-eval
../.venv/bin/python examples/pruning/hf_pruning_tracker_example.py \
  --model-id sshleifer/tiny-gpt2 \
  --output-attentions \
  --text "Pruning can reduce inference cost." \
  --text "Forward hooks can collect pruning signals." \
  --output-dir ./hf_pruning_tracker_dump
```

What this standalone example does:

- loads a tokenizer and HF model with `AutoTokenizer` and `AutoModelForCausalLM` by default,
- passes `attn_implementation=eager` by default and requests `output_attentions` when `--output-attentions` is used,
- attaches `PruningTracker` directly to the loaded `torch.nn.Module`,
- runs a few calibration-style forward passes,
- saves `tracker_payload.pt` and `summary.json`,
- and exports `mlp_modules.csv`, `attention_modules.csv`, `linear_state_modules.csv`, `linear_state_components.csv`, and `network_sequence.csv`.

**Weights & Biases Logging**

If `--wandb_args ...` is enabled in the same run, `lmms-eval` now also reports a pruning summary to W&B:

- scalar summaries for MLP and attention statistics such as mean, std, min, max, number of tracked modules, and update counts,
- scalar MLP activity summaries such as average active neurons per token, average active fraction, and low-activity channel fraction,
- scalar attention concentration summaries such as mean entropy and mean peak mass,
- scalar linear-state summaries such as mean input-output delta RMS ratio, output/input RMS ratio, output/input cosine, and component activity,
- W&B tables with one row per tracked module,
- W&B histograms for global MLP activation magnitudes, explicit attention weights, linear-state output magnitudes, linear-state input-output deltas, and linear-state component magnitudes,
- W&B-accessible paths back to the local `visualizations.json`, `plots/`, `plots_manifest.json`, and `network_sequence.csv` artifacts,
- and the full local pruning report directory as a W&B artifact.

This gives a run-level view of which layers or submodules carry the largest activation or attention mass, how sparse the MLP pathways are during the run, how concentrated the attention maps are, and how strongly linear-state blocks transform the input/output stream.

**Performance Note**

The pruning tracker is still extra work on top of generation, especially when explicit attention maps are exposed, but it avoids global attention-output retention by default. If you want a lighter calibration run, increase `--pruning_tracker_stride` to values such as `4` or `8`, reduce the retained-sample budget for adaptive histograms with `--pruning_tracker_max_histogram_samples`, or disable eager-attention requests with `--no-pruning_tracker_force_eager_attention`.

**Credential And Task Isolation**

Evaluation-tracker credentials are kept in runtime-only arguments. Credential-like values in tracker arguments, model arguments, console summaries, W&B-generated names, pruning metadata, and `resolved_cli_args` are redacted before logging or persistence. Existing result files and remote artifacts are not sanitized retroactively, so rotate any credential captured by an earlier run. Pruning statistics currently aggregate across the process; run exactly one task per process when comparing benchmark categories.

Git revision and branch metadata are resolved from the installed `lmms-eval` source directory rather than the shell working directory. This prevents a surrounding meta-repository from being recorded as the evaluator revision.

**Important Limitation**

The tracker requests eager HF attention by default when `--pruning_tracker` is used, but this is still best-effort. Some wrappers do not expose an `attn_implementation` load argument, some custom kernels do not return attention maps, and many linear-attention modules do not have a standard softmax `(B, H, N, N)` attention matrix. In those cases `attn_stats` can still be empty, but `linear_state` can still be populated. This is intentional; the tracker no longer reports projection magnitudes as attention scores.

If a model only exposes attention maps through global HF `output_attentions=True`, use `--pruning_tracker_force_output_attentions` as a fallback. This is high-memory during generation and should be avoided for large VLM sequences or large batches unless you have confirmed enough GPU memory.

The current W&B summary is generic and module-level. A true vision-vs-text attention breakdown is not reported yet, because the entry-point tracker does not currently have token-span metadata telling it which attention positions correspond to visual tokens and which correspond to text tokens. That would require model-specific boundary information from the processor or wrapper.

**Environmental Variables**

Before running experiments and evaluations, we recommend you to export following environment variables to your environment. Some are necessary for certain tasks to run.

```bash
export OPENAI_API_KEY="<YOUR_API_KEY>"
export HF_HOME="<Path to HF cache>"
export HF_TOKEN="<YOUR_API_KEY>"
export HF_HUB_ENABLE_HF_TRANSFER="1"
export REKA_API_KEY="<YOUR_API_KEY>"
# Other possible environment variables include
# ANTHROPIC_API_KEY,DASHSCOPE_API_KEY etc.
```

**Common Environment Issues**

Sometimes you might encounter some common issues for example error related to httpx or protobuf. To solve these issues, you can first try

```bash
python3 -m pip install httpx==0.23.3;
python3 -m pip install protobuf==3.20;
# If you are using numpy==2.x, sometimes may causing errors
python3 -m pip install numpy==1.26;
# Someties sentencepiece are required for tokenizer to work
python3 -m pip install sentencepiece;
```

## Custom Model Integration

`lmms-eval` supports two types of models: **Chat (recommended)** and **Simple (legacy)**.

### Chat Models (Recommended) 🌟

- Location: `lmms_eval/models/chat/`
- Use: `doc_to_messages` function from task
- Input: Structured `ChatMessages` with roles (`user`, `system`, `assistant`) and content types (`text`, `image`, `video`, `audio`)
- Supports: Interleaved multimodal content
- Uses: Model's `apply_chat_template()` method
- Reference: `lmms_eval/models/chat/qwen2_5_vl.py` or `lmms_eval/models/chat/qwen3_vl.py`

**Example input format:**
```python
[
    {"role": "user", "content": [
        {"type": "image", "url": <image>},
        {"type": "text", "text": "What's in this image?"}
    ]}
]
```

### Simple Models (Legacy)

- Location: `lmms_eval/models/simple/`
- Use: `doc_to_visual` + `doc_to_text` functions from task
- Input: Plain text with `<image>` placeholders + separate visual list
- Supports: Limited (mainly images)
- Manual processing: No chat template support
- Reference: `lmms_eval/models/simple/instructblip.py`

**Example input format:**
```python
# Separate visual and text
doc_to_visual -> [PIL.Image]
doc_to_text -> "What's in this image?"
```

### Key Differences

| Aspect | Chat Models | Simple Models |
|--------|-------------|---------------|
| File location | `models/chat/` | `models/simple/` |
| Input method | `doc_to_messages` | `doc_to_visual` + `doc_to_text` |
| Message format | Structured (roles + content types) | Plain text with placeholders |
| Interleaved support | ✅ Yes | ❌ Limited |
| Chat template | ✅ Built-in | ❌ Manual/None |
| Recommendation | **Use this** | Legacy only |

### Why Use Chat Models?

- ✅ Built-in chat template support
- ✅ Interleaved multimodal content
- ✅ Structured message protocol
- ✅ Better video/audio support
- ✅ Consistent with modern LLM APIs

### Chat Model Implementation Example

```python
from lmms_eval.api.registry import register_model
from lmms_eval.api.model import lmms
from lmms_eval.protocol import ChatMessages

@register_model("my_chat_model")
class MyChatModel(lmms):
    is_simple = False  # Use chat interface

    def generate_until(self, requests):
        for request in requests:
            # 5 elements for chat models
            doc_to_messages, gen_kwargs, doc_id, task, split = request.args

            # Get structured messages
            raw_messages = doc_to_messages(self.task_dict[task][split][doc_id])
            messages = ChatMessages(messages=raw_messages)

            # Extract media and apply chat template
            images, videos, audios = messages.extract_media()
            hf_messages = messages.to_hf_messages()
            text = self.processor.apply_chat_template(hf_messages)

            # Generate...
```

For more details, see the [Model Guide](docs/guides/model_guide.md).

## Custom Dataset Integration

### Task Configuration with `doc_to_messages`

Implement `doc_to_messages` to transform dataset documents into structured chat messages:

```python
def my_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    # Extract visuals and text from doc
    visuals = my_doc_to_visual(doc)
    text = my_doc_to_text(doc, lmms_eval_specific_kwargs)

    # Build structured messages
    messages = [{"role": "user", "content": []}]

    # Add visuals first
    for visual in visuals:
        messages[0]["content"].append({"type": "image", "url": visual})

    # Add text
    messages[0]["content"].append({"type": "text", "text": text})

    return messages
```

### YAML Configuration

```yaml
task: "my_benchmark"
dataset_path: "my-org/my-dataset"
test_split: test
output_type: generate_until

# For chat models (recommended)
doc_to_messages: !function utils.my_doc_to_messages

# OR legacy approach:
doc_to_visual: !function utils.my_doc_to_visual
doc_to_text: !function utils.my_doc_to_text

process_results: !function utils.my_process_results
metric_list:
  - metric: acc
```

### Key Features

#### `doc_to_messages`

- Transforms dataset document into structured chat messages
- Returns: List of message dicts with `role` and `content`
- Content supports: `text`, `image`, `video`, `audio` types
- Protocol: Defined in `lmms_eval/protocol.py` (`ChatMessages` class)
- Auto-fallback: If not provided, uses `doc_to_visual` + `doc_to_text`


For more details, see the [Task Guide](docs/guides/task_guide.md).

## Web UI

LMMS-Eval includes an optional Web UI for interactive evaluation configuration.

### Requirements

- Node.js 18+ (for building the frontend, auto-built on first run)

### Usage

```bash
# Start the Web UI (opens browser automatically)
uv run lmms-eval-ui

# Custom port
LMMS_SERVER_PORT=3000 uv run lmms-eval-ui
```

The web UI provides:
- Model selection from all available models
- Task selection with search/filter
- Real-time command preview
- Live evaluation output streaming
- Start/Stop evaluation controls
- Log Viewer for browsing saved evaluation results and samples

For more details, see [Web UI README](lmms_eval/tui/README.md).

## HTTP Evaluation Server

LMMS-Eval includes a production-ready HTTP server for remote evaluation workflows.

### Why Use Eval Server?

- **Decoupled evaluation**: Run evaluations on dedicated GPU nodes while training continues
- **Async workflow**: Submit jobs without blocking training loops
- **Queue management**: Sequential job processing with automatic resource management
- **Remote access**: Evaluate models from any machine

### Start Server

```python
from lmms_eval.entrypoints import ServerArgs, launch_server

# Configure server
args = ServerArgs(
    host="0.0.0.0",
    port=8000,
    max_completed_jobs=200,
    temp_dir_prefix="lmms_eval_"
)

# Launch server
launch_server(args)
```

Server runs at `http://host:port` with auto-generated API docs at `/docs`

### Client Usage

**Sync Client:**
```python
from lmms_eval.entrypoints import EvalClient

client = EvalClient("http://eval-server:8000")

# Submit evaluation (non-blocking)
job = client.evaluate(
    model="qwen2_5_vl",
    tasks=["mmmu_val", "mme"],
    model_args={"pretrained": "Qwen/Qwen2.5-VL-7B-Instruct"},
    num_fewshot=0,
    batch_size=1,
    device="cuda:0",
)

# Continue training...
# Later, retrieve results
result = client.wait_for_job(job["job_id"])
print(result["result"])
```

**Async Client:**
```python
from lmms_eval.entrypoints import AsyncEvalClient

async with AsyncEvalClient("http://eval-server:8000") as client:
    job = await client.evaluate(
        model="qwen3_vl",
        tasks=["mmmu_val"],
        model_args={"pretrained": "Qwen/Qwen3-VL-4B-Instruct"},
    )
    result = await client.wait_for_job(job["job_id"])
```

### Server API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Server health check |
| `/evaluate` | POST | Submit evaluation job |
| `/jobs/{job_id}` | GET | Get job status and results |
| `/queue` | GET | View queue status |
| `/tasks` | GET | List available tasks |
| `/models` | GET | List available models |
| `/jobs/{job_id}` | DELETE | Cancel queued job |
| `/merge` | POST | Merge FSDP2 sharded checkpoints |

### Example Workflow

```python
# Training loop pseudocode
for epoch in range(num_epochs):
    train_one_epoch()

    # After every N epochs, evaluate checkpoint
    if epoch % 5 == 0:
        checkpoint_path = f"checkpoints/epoch_{epoch}"

        # Submit async evaluation (non-blocking)
        eval_job = client.evaluate(
            model="vllm",
            model_args={"model": checkpoint_path},
            tasks=["mmmu_val", "mathvista"],
        )

        # Training continues immediately
        print(f"Evaluation job submitted: {eval_job['job_id']}")

# After training completes, retrieve all results
results = []
for job_id in eval_jobs:
    result = client.wait_for_job(job_id)
    results.append(result)
```

### Security Note

⚠️ **This server is intended for trusted environments only**. Do NOT expose to untrusted networks without additional security layers (authentication, rate limiting, network isolation).

For more details, see the [v0.6 release notes](docs/releases/lmms-eval-0.6.md).

## Frequently Asked Questions

<details>
<summary><strong>What models does lmms-eval support?</strong></summary>

We support 30+ model families out of the box, including Qwen2.5-VL, Qwen3-VL, LLaVA-OneVision, InternVL-2, VILA, and more. Any OpenAI-compatible API endpoint is also supported. See the full list in [`lmms_eval/models/`](lmms_eval/models/).

Qwen3.5 is supported through existing runtime backends (`--model vllm` and `--model sglang`) by setting `model=Qwen/Qwen3.5-397B-A17B` in `--model_args`.

The Qwen3.5 example scripts align with official runtime references (for example, `max_model_len/context_length=262144` and `reasoning_parser=qwen3`).

If a new model family is already fully supported by vLLM or SGLang at runtime, we generally only need documentation and examples instead of adding a dedicated model wrapper.

</details>

<details>
<summary><strong>What benchmarks and tasks are available?</strong></summary>

Over 100 evaluation tasks across image, video, and audio modalities, including MMMU, MME, MMBench, MathVista, VideoMME, EgoSchema, and many more. Check [`docs/advanced/current_tasks.md`](docs/advanced/current_tasks.md) for the full list.

</details>

<details>
<summary><strong>How do I add my own benchmark?</strong></summary>

Create a YAML config under `lmms_eval/tasks/` with dataset path, splits, and a `doc_to_messages` function. See [`docs/guides/task_guide.md`](docs/guides/task_guide.md) for a step-by-step guide.

</details>

<details>
<summary><strong>Can I evaluate a model behind an API (e.g., GPT-4o, Claude)?</strong></summary>

Yes. Use `--model openai` with `--model_args model=gpt-4o` and set `OPENAI_API_KEY`. Any OpenAI-compatible endpoint works, including local vLLM/SGLang servers.

</details>

<details>
<summary><strong>How do I run evaluations on multiple GPUs?</strong></summary>

Use `accelerate launch` or pass `--device cuda` with tensor parallelism via vLLM/SGLang backends. See [`docs/getting-started/commands.md`](docs/getting-started/commands.md) for multi-GPU flags.

</details>

<details>
<summary><strong>How do I cite lmms-eval?</strong></summary>

Use the BibTeX entries below, or click the "Cite this repository" button in the GitHub sidebar (powered by our [`CITATION.cff`](CITATION.cff)).

</details>

## Acknowledgement

lmms_eval is a fork of [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness). We recommend you to read through the [docs of lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness/tree/main/docs) for relevant information.

---

Below are the changes we made to the original API:
- Build context now only pass in idx and process image and doc during the model responding phase. This is due to the fact that dataset now contains lots of images and we can't store them in the doc like the original lm-eval-harness otherwise the cpu memory would explode.
- Instance.args (lmms_eval/api/instance.py) now contains a list of images to be inputted to lmms.
- lm-eval-harness supports all HF language models as single model class. Currently this is not possible of lmms because the input/output format of lmms in HF are not yet unified. Therefore, we have to create a new class for each lmms model. This is not ideal and we will try to unify them in the future.

---

## Citations

```bibtex
@misc{zhang2024lmmsevalrealitycheckevaluation,
      title={LMMs-Eval: Reality Check on the Evaluation of Large Multimodal Models},
      author={Kaichen Zhang and Bo Li and Peiyuan Zhang and Fanyi Pu and Joshua Adrian Cahyono and Kairui Hu and Shuai Liu and Yuanhan Zhang and Jingkang Yang and Chunyuan Li and Ziwei Liu},
      year={2024},
      eprint={2407.12772},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.12772},
}

@misc{lmms_eval2024,
    title={LMMs-Eval: Accelerating the Development of Large Multimoal Models},
    url={https://github.com/EvolvingLMMs-Lab/lmms-eval},
    author={Bo Li*, Peiyuan Zhang*, Kaichen Zhang*, Fanyi Pu*, Xinrun Du, Yuhao Dong, Haotian Liu, Yuanhan Zhang, Ge Zhang, Chunyuan Li and Ziwei Liu},
    publisher    = {Zenodo},
    version      = {v0.1.0},
    month={March},
    year={2024}
}
```


---

## Representation-dynamics study

This section documents the additions of the `research/representation-dynamics` branch and
how the paper's results were produced from them. Evaluation behaviour without the
`--representation_dynamics` flag is unchanged from upstream.

### What the fork adds

| Path | Purpose |
| --- | --- |
| `lmms_eval/representations/tracker.py` | `RepresentationDynamicsTracker`: discovers the text decoder stack, attaches three read-only forward hooks per layer (token-mixing update, feed-forward update, residual state), pools the final valid prompt token (or decode updates), and records one vector per example/layer/stream with exactness metadata. Hooks return `None`, never modify weights, and are removed on exit. |
| `lmms_eval/representations/analysis.py` | PCA-space mean-shift ratio, rank diagnostics, task-held-out linear and RBF probes, exhaustive balanced task-label permutation tests with selection-aware maxima, Benjamini–Hochberg q-values, covariance alignment, silhouette, paired CKA; scikit-learn and cuML backends. |
| `lmms_eval/representations/{selected_site_probe,prompt_baseline,matched_excess,lexical_diagnostics,probe_calibration,matched_excess_calibration,representation_width_sensitivity,split_unit_sensitivity,whitened_selector}.py` | The controls reported in the paper: selection-aware probe at the mean-shift site, prompt-text baselines, matched activation-minus-prompt test, lexical residualization diagnostics, injected-signal calibration, width and inferential-unit sensitivities, regularized-whitened selector. |
| `lmms_eval/tasks/representation_dynamics_mmlu/` | The 16 single-prompt MMLU subject tasks of the primary panel (`hails/mmlu_no_train`, pinned revision, A/B/C/D answers, four-token ceiling). |
| `lmms_eval/tasks/representation_dynamics_text/`, `.../representation_dynamics_controls/`, `.../representation_dynamics_response_trajectory/` | Direct-answer adapters for the MMLU-Pro and mixed-benchmark panels, the multimodal appendix panel, and the answer-then-explain response-trajectory pilot. |
| `lmms_eval/models/simple/{hf_causal,gemma3,gemma4,qwen3_vl,qwen3_5,apertus1p5}.py` | Model adapters that build the per-example context around `generate()` so the tracker can attribute activations to samples. |
| `test/eval/test_representation_dynamics.py` and related tests | Unit tests for extraction, decode modes, exactness flags, analysis metrics, permutation design, and the control modules. |

CLI flags (all optional; absent flags leave evaluation untouched): `--representation_dynamics`,
`--representation_dynamics_output`, `--representation_dynamics_pooling {last_valid,mean_valid}`,
`--representation_dynamics_phases {prefill,decode,both}`,
`--representation_dynamics_decode_aggregation {mean_updates,first_update,all_updates}`,
`--representation_dynamics_decode_max_updates`,
`--representation_dynamics_storage_dtype {float16,bfloat16,float32}`.

### Running

```bash
# Collect activations for one task (writes representations.pt, records.csv, metadata.json)
python -m lmms_eval --model hf_causal \
  --model_args pretrained=allenai/Olmo-3-7B-Instruct,revision=6e5971d9eba42665f5bd5a0fcf047f299ce1dccc,apply_chat_template=True \
  --tasks repdyn_mmlu_anatomy --batch_size 32 \
  --representation_dynamics --representation_dynamics_output /path/to/run/anatomy

# Analyse a set of runs with task-level inference
python -m lmms_eval.representations.analysis /path/to/run/*/representations.pt \
  --knowledge-tasks repdyn_mmlu_anatomy,... --reasoning-tasks repdyn_mmlu_abstract_algebra,... \
  --output-dir /path/to/analysis --cv-folds 5 --permutations 12869 --seed 0

# Tests
python -m pytest -q test/eval/test_representation_dynamics.py test/eval/test_cli_parse_args.py
```

The frozen experiment manifests (models, immutable revisions, batch sizes, sample freezes,
analysis settings) and the orchestration scripts that call these commands live in the study
repository [`parsa-ra/vlmbehave`](https://github.com/parsa-ra/vlmbehave) under `experiments/`
and `scripts/`.

### Results and artifacts

All reported results were produced with this repository at commit `1bdf31d` and the study
repository at commit `34c3105`. The sample-free tables behind every number in the paper are
committed in the study repository under `results/`, and each `report.json` there records the
SHA-256 of the raw files it was computed from.

The raw outputs (activation tensors, per-record tables, sample logs, analysis reports, and
permutation nulls; 133.7 GB uncompressed) are released as the archive
`persistent-data.tar.zst` (81,699,077,557 bytes, SHA-256
`160ab6cb5cfcf83ac35f43c4b946957cdb2511d33005a617daa71520546d1b22`) together with a
per-file SHA-256 manifest and the revisions above, in the Google Drive folder
`lmbehave-release` linked from the project page. The study repository's
`docs/ARTIFACTS.md` maps every results directory to the archive directories it derives from
and gives the restore and recomputation commands.

### Citation

```bibtex
@inproceedings{rahiminoshanagh2026knowledge,
  title     = {Do Knowledge- and Reasoning-Oriented Benchmarks Induce Stable
               Representation Signatures in {LLMs}?},
  author    = {Rahimi Noshanagh, Parsa and Dehghani Amirabad, Azim},
  booktitle = {NeurIPS 2026 Workshop on Interpretability as a Science},
  year      = {2026},
  url       = {https://parsa-ra.github.io/lmbehave/}
}
```
