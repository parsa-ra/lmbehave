# Normalized response-position pilot

This directory contains sample-free aggregates for the completed OLMo 3 7B descriptive
pilot. Raw activations, prompts, and responses remain on persistent storage and are bound by
the hashes in `report.json`.

The observer-off replay, observer parity, exactness, completeness, eight-bin coverage, and
selection-aware inference gates passed. The maximum decode-minus-prefill Fisher increase is
0.286113 at layer 31's token-mixing update in relative bin 5/8 (0.193350 to 0.479463), with
exact p=2/70=0.028571 after reselecting layer, stream, and bin for every assignment.

This is not confirmatory evidence: 4+4 tasks cannot attain the paper's p<.01 threshold,
the answer-first protocol measures post-answer dynamics, and relative bins are not semantic
reasoning steps.
