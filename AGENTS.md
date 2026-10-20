# LMBehave development contract

This public repository contains the study protocols, sample-free results, orchestration, and bundled evaluation source for the NeurIPS 2026 Interpretability as a Science Workshop paper.

- Article: https://arxiv.org/abs/2610.24029
- Project page: https://parsa-ra.github.io/lmbehave/
- Website source: https://github.com/parsa-ra/lmbehave-page
- Start with README.md, docs/SCIENTIFIC_STATUS.md, docs/ARTIFACTS.md, and docs/OPERATIONS.md.
- lmms-eval/ is ordinary tracked source, bundled from revision 2f121d64e804b83fe8724981fd180d1d5a4c3047. Preserve its upstream MIT license.
- Representation instrumentation belongs only in lmms-eval/lmms_eval/representations/ and its adapters. Do not instrument Transformers internals for this study.
- Keep raw prompts, activations, sample logs, credentials, and model weights out of Git. Existing reports retain original provenance hashes and compute paths.
- Make focused changes; preserve experiment settings and the distinction between primary, corrected, historical, and appendix evidence.
- Use the root Git repository for all source changes. There are no private submodules.
- Commit or push only when requested. Preserve user work and report verification limits.

## 2. Scientific contract

These constraints are non-negotiable unless the researcher explicitly revises the study
design and the paper documents that revision.

1. Primary evidence must be quantitative. Raw t-SNE/UMAP inspection is prohibited as a
   primary metric.
2. Collect three streams at every selected decoder layer: `attention_delta`, `mlp_delta`,
   and `residual_state`.
3. Collection must be opt-in. With `--representation_dynamics` absent, evaluation behavior
   must remain unchanged.
4. Hooks must return `None`, never replace outputs, never modify weights, and be removed in
   cleanup paths. Do not force a different attention implementation merely to collect the
   three representation streams.
5. Save one vector per example/layer/stream/phase, with task, split, document/sample ID,
   model, module, layer, rank, prompt length, pooling, update count, and exactness metadata.
6. Compare within model. Do not concatenate or directly average raw features from models
   with different architectures or hidden widths.
7. Probe validation must hold out entire tasks, not random examples. Label permutations
   must also occur at task level.
8. Preserve failed or undefined quantities as missing values. Never convert absent CKA,
   unsupported hooks, or failed analysis rows to zero.
9. Predeclare primary decisions. The current primary localization statistic is maximum
   per-model PCA-space Fisher ratio. Report the overall maximum and the attention-versus-MLP
   branch maximum separately.
10. Statistical significance is evidence against the specified null, not proof. Report raw
    permutation p-values, minimum attainable p-values, and Benjamini–Hochberg q-values.
11. Passive separability is not causal importance and is not pruning safety. Any pruning
    claim requires an intervention with matched budgets, random and established baselines,
    accuracy/quality, latency, throughput, and peak-memory measurements.
12. Preserve behavior during maintenance. Do not alter task lists, prompts, sample counts,
    batch sizes, generation arguments, model revisions, pooling, or storage precision as a
    cleanup side effect.

## Verification

Run focused tests for changed code. Packaging checks:

```bash
python3 scripts/check_private_ready.py
git diff --check
bash -n install.sh scripts/setup_machine.sh
```

Do not claim real-model, GPU, or output-parity validation from packaging checks. No protocol or analysis defaults should change during release maintenance.
