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

## Amendment, 2026-10-16

Added before the control runs finished: the fallback is triggered only if the observed
agreement exceeds the 95th percentile of the LM-shifted distribution in at least two of the
six runs. Per-run verdicts are still reported for every run.

Disclosure: while testing the control script on 2026-10-15, partial control output for one
run, Gemma 2 2B Physical, was printed (16 of 100 LM-shifted, 8 of 100 white-noise and 2 of 100
phase-randomized replicates). No other control output had been examined when this amendment
was committed.

## Note, 2026-10-19

Figure 2 (importance expected from the predictor matrix alone) was made for all six runs after
the control results had been seen, at the supervisor's request. The rule above had said to skip it.
