# Repository Audit — baseline 2026-07-21

This records the audited local baseline; no remote was fetched while the evidence below was collected. Since that audit, the root and active `lmms-eval` branches have been placed on their private SSH remotes without changing the recorded research code. Exact child revisions remain in [`repos.lock.json`](../repos.lock.json).

## Topology and active work

The workspace contains six independent Git repositories. The root initially had an empty `.git` directory, so it did not preserve the nested repositories or research scripts.

| Repository | Local state | Research role |
| --- | --- | --- |
| `lmms-eval` | Clean `research/representation-dynamics` at `c4fa3931`; recorded as a root submodule; private `origin` prepared and public `upstream` retained | Active behavior tracker and representation-separability integration |
| `transformers_qwenvl` | Local `AdaptToken` commit plus 2 modified Qwen3-VL source files | Active visual-token pruning prototype |
| `transformers` | Clean detached `v5.5.4` shallow checkout | Upstream baseline currently selected by `install.sh` |
| `vllm` | Clean | Optional inference reference |
| `vLLM-Hook` | Clean | Optional hook reference |
| `IVC-Prune` | Clean | Optional pruning baseline/reference |

The root `logs/` tree is approximately 7.6 GB. It contains 46 files above GitHub's 100 MB file limit. Raw results, tracker tensors, sample JSONL, and W&B state are excluded from the prepared meta-repository; no artifact was deleted or rewritten.

## Critical findings

1. A Hugging Face access token is present in 84 log/W&B text files and all 10 local W&B run binaries. It entered output through mutation and logging of `hf_hub_log_args`. The credential must be revoked/rotated; a private repository is not a safe secret store. The code now keeps the environment token runtime-only and redacts credential-like tracker/model arguments at console and artifact boundaries, but old artifacts remain unchanged.
2. Qwen3-VL/Qwen3.5 tracking reached an undefined `task` variable before `model.generate`. The wrapper now returns the batch task name from preprocessing; a mocked regression test covers the tracker-enabled path.
3. The aggregate `PruningTracker` still pools task observations. The primary `RepresentationDynamicsTracker` now exports per-example task, split, document, rank, layer, stream, and phase identity. Publication runs should still use one task per process for isolation and simpler failure recovery.
4. Existing results are not code-reproducible: all 68 result JSONs have a null Git hash, 66 report `unknown@unknown`, and 65 tracker summaries lack a source revision. The lookup previously used the launch directory; it now resolves from the `lmms-eval` source checkout. Existing artifacts remain unresolved, and limits/tracker schemas also vary across runs.
5. Enabling the tracker normally requests eager attention. This changes the kernel path, memory, speed, and potentially numerical/sampled outputs. It is not a neutral timing mode.

## Gemma and benchmark scripts

`lmms-eval/lmms_eval/models/simple/gemma3.py` has two local changes:

- a tracker context around `model.generate`, which does not replace model outputs but adds hook work when a tracker is attached;
- normalization of `None`, a single image, or a single path into an iterable visual list. Standard list inputs are unchanged; the `None` case repairs text-only tasks such as GPQA that previously failed before generation.

Both [`bench_gemma.sh`](../bench_gemma.sh) and [`bench_qwen.sh`](../bench_qwen.sh) now enable `--representation_dynamics` and write one per-example representation directory per task. Important mismatches remain:

- The original Gemma script set `LIM=128` but passed `--limit $((LIM/BATCH_SIZE))`, so it evaluated 16 documents, not 128. Preparation now names the effective default `NUM_SAMPLES=16`; Qwen's unchanged effective default is named `NUM_SAMPLES=512`.
- Qwen evaluates up to 512 documents and samples at temperature 0.7; Gemma evaluates 16 and uses the wrapper's greedy default. Their behavior statistics are not directly comparable.
- The Gemma script selects 4B, while the main saved analysis is dominated by 12B and also contains 27B runs.
- Gemma pads batches to 2,048 tokens. Global hook summaries include padded positions, while token-group metrics use a validity mask; global Gemma statistics are therefore not directly comparable with dynamically padded Qwen statistics.
- Current wrapper defaults also allow substantially different image resolutions between Gemma and Qwen3.5, adding a visual-token-count confound.
- The labels are hypothesis labels, not clean causal categories. MedQA/GPQA are text-only while the capability set is visual; MME, MMMU, AI2D, OCRBench, TextVQA, and DocVQA mix perception, reasoning, task format, and knowledge.
- Both scripts contain site-specific Slurm account/email settings. They parse as Bash but are not portable job specifications.

The scripts still default to the same effective sample counts as before preparation. No benchmark list, decoding setting, or model selection was changed; only the opt-in monitoring path changed from aggregate pruning summaries to per-example representation dynamics.

Saved evidence confirms that newer Gemma-3-12B runs captured token-group and routing rows, whereas saved Qwen3.5-9B summaries did not. The current Gemma-4B output tree is empty, the Gemma-27B series is incomplete, and the latest Qwen3.5-9B series does not cover the full ten-task matrix. Cross-model token-aware claims are not yet supported.

## Qwen pruning prototype

The `transformers_qwenvl` overlay adds vision-depth control, token pruning, attention-guided importance, dropped-token summaries, and a visual/text QK monitor. `install.sh` defaults to the clean `transformers/` checkout and requires an explicit `TRANSFORMERS_REPO=transformers_qwenvl` selection for this prototype.

The prototype has no feature tests. Notable risks include: silent L2 fallback for attention pruning without warmup; a summary token beyond the requested keep budget; unvalidated methods/layer specs; video-frame segmentation mismatch; stale monitor state; and monitoring of unmasked, unsoftmaxed QK logits rather than attention probability. The documented monitor exporter is absent. Default-disabled equivalence has not been demonstrated.

## Research interpretation

Current statistics are useful for hypothesis generation, not an ICLR claim. The main confounds are modality, input/output length, generation phase, decoding policy, task format, model size, and unequal sample counts. In particular:

- MLP/linear-state updates mix one prefill with many decode steps, so longer answers receive more weight.
- Mean nonzero softmax attention is strongly determined by mask/sequence length and is not head importance.
- The `1e-6` activity threshold is likely saturated for smooth dense activations.
- MLP matching includes parent blocks and internal projections, mixing channel spaces and double-counting conceptual blocks.
- Passive scores are candidate pruning signals; causal pruning/ablation is required to validate them.

The next defensible protocol is described in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md).
