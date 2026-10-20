# Result provenance notes

Every directory here is a sample-free summary computed from raw artifacts that are released
as one verified bundle; `../docs/ARTIFACTS.md` lists, for each directory, the bundle paths
its `report.json` hashes and how to restore and recompute them.

Most older committed result directories are immutable, sample-free summaries of analyses
completed before the current complement-aware inference revision. Their observed scores and empirical
permutation p-values remain historical measurements, but any metadata field that advertises
`1/252` (`0.003968...`) as the minimum attainable p-value is wrong for class-symmetric
accuracy, macro-F1, and Fisher statistics. With five tasks per class, the 252 oriented
assignments form 126 complementary pairs, so the attainable floor is `1/126` (`0.007936...`).

Do not rewrite old CSV or JSON fields in place: doing so would break their recorded hashes
and obscure provenance. New analyses must use the revised `lmms-eval` implementation, retain
`permutation_nulls.json.gz`, pass complement-symmetry checks, and report selection-aware
model-level Fisher maxima. The old row-wise rejection counts and Fisher maxima are
descriptive until regenerated from the hash-bound activation artifacts.

`representation_dynamics_main_track_fisher_v1/` is the regenerated exception: it contains
the complement-aware, selection-aware five-model Fisher summary and SHA-256 bindings to the
full persistent null artifacts. It does not alter any legacy output in place.

`representation_dynamics_main_track_selected_probe_v1/` contains the completed exact
five-model probe at the Fisher-selected site, with site selection repeated for every task
assignment. `apertus1_longitudinal_v1/` contains compact aggregates and hash bindings for
the completed eleven-checkpoint Apertus 1.0 trajectory.

`representation_dynamics_text_expansion_v1/` has an additional local note because it also
contains the superseded Apertus 1.0 condition and malformed-MuSR cross-panel summaries. The
corrected current-model aggregates are under
`representation_dynamics_text_expansion_corrected_v1/`.
