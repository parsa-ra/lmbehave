# Selection-aware linear probe

This directory contains sample-free aggregates from the exact 8+8-task linear probe at
the Fisher-selected site. Fisher selection is repeated for every balanced task-label
assignment before the grouped probe is evaluated. Each report preserves all 12,870
oriented scores while sharing one probe refit across each label-complement pair.

All five reports pass and have exact accuracy and macro-F1 complement symmetry. The
macro-F1 randomization p-values range from 0.001554 to 0.004196. This supports linear
recoverability of the declared subject partition under this composite statistic. It is not
a prompt-adjusted result, causal localization, or evidence for a common site across models.
Full score distributions and tensor artifacts remain on persistent storage.
