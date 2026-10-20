# Primary-panel lexical activation control

These sample-free tables summarize the exact five-model sensitivity analysis that removes
activation variance linearly predictable from fold-local character TF--IDF representations.
Full per-assignment outputs remain on persistent storage at the path recorded in
`report.json`; SHA-256 bindings cover all ten source reports and the compact outputs.

For each of the 12,870 balanced task-label assignments, the analysis reselects that
assignment's maximum from the unadjusted Fisher profile. Within every grouped probe fold,
it fits the character 3--5-gram vocabulary, inverse-document frequencies, truncated SVD,
activation Ridge map, PCA, scaling, and classifier without the held-out tasks. The 6,435
class-label-complement pairs share one fit, and all ten reports pass exact support,
complement-symmetry, clean-revision, and distribution-hash gates.

At 64 components, macro-F1 falls from `.781--.805` before adjustment to `.613--.698`.
Exact p-values are `.01197`, `.06605`, `.01476`, `.04165`, and `.02020` for OLMo 3,
Apertus 1.5, Qwen 9B, Qwen 27B, and Gemma 4. None meets the paper's declared `.01`
criterion, although four remain below a nominal per-model `.05` threshold. The 32-component
sensitivity gives p-values `.01880`, `.06402`, `.02704`, `.03124`, and `.02362`; it also
has no p-value below `.01`.

This is evidence that the unadjusted selected-site result is sensitive to a measured
low-rank lexical control. It does not show that lexical form causes the representation
result, remove nonlinear or model-rendered lexical information, or establish absence of
task-partition information.
