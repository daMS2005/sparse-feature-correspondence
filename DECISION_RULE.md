# Decision rule: shared-predictor control

Written 2026-10-15, before any control is run.

## Question

Is the brain–LM feature-importance agreement from the thesis (0.937–0.986 across six runs)
explained by the shared predictor matrix alone?

## Rule

For each run, the LM targets are circularly shifted by 100 offsets drawn uniformly from
[10, n_TR − 10] with a fixed seed, and the agreement is recomputed. If the observed agreement
does not exceed the 95th percentile of these 100 values, it is attributed to the shared
predictor matrix.

Each run is judged separately. No aggregate verdict is defined. If verdicts differ across
runs, they are reported by story and model.

If any run exceeds the threshold, Figures 1 and 3 are finished, Figure 2 is skipped, and the
results are reported before further analysis.

The LM-shifted, brain-shifted, white-noise and phase-randomized distributions are reported
for every run.
