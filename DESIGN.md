# Scientific design

This document defines the estimand, analysis choices, and interpretation
boundary. Execution and verification guidance is in
[`REPRODUCE.md`](REPRODUCE.md).

## Scientific question and estimand

For a selected SAE feature bank, the workflow constructs matched feature
coordinates for two prediction problems:

- **BOLD target:** participant-specific Schaefer-200 cortical responses, with
  five temporal lags per SAE feature;
- **language-model target:** the TR-averaged final hidden state, using the same
  feature coordinates without hemodynamic lags.

Ridge models are trained separately for the two targets. For feature group
`j`, reliance is the held-out change in out-of-sample variance explained when
that group is replaced by its training mean:

```text
reliance(j) = R²(full model) - R²(model with feature group j ablated)
```

For BOLD, a group contains all five lagged columns for one SAE feature. For the
language-model target, it contains the corresponding unlagged column. The
fitted weights stay fixed during ablation. In the primary transfer analysis,
reliance is averaged over target dimensions and, for BOLD, participants, then
the two profiles are compared by Spearman correlation. Within-story analyses
compare profiles separately in each outer fold before averaging the
correlations, because feature banks can differ across folds.

## Analysis specification

The release configuration records the choices used for the reported results.

| Component | Choice |
| --- | --- |
| Stories | `shapesphysical`, `shapessocial` |
| TR duration | 1.5 seconds |
| Expected participants | 59 per story |
| Expected TRs | 303 physical, 305 social |
| Cortical target | Schaefer-200 parcels |
| BOLD lags | 0, 1, 2, 3, 4 TRs |
| Outer/inner folds | 5 contiguous folds each |
| Purge | 4 adjacent TRs |
| Feature-bank sizes | 64, 128, 256, 512, 1024 |
| Primary bank size | 256 |
| Primary bank ranking | binary temporal variance, `p(1-p)` |
| Primary prediction metric | Pearson correlation |
| Variance metric | out-of-sample R² |
| Primary correspondence | Spearman correlation |
| Temporal null | 199 circular target shifts |
| Identity null | 10,000 within-layer permutations |
| Bootstrap/sign flips | 10,000 draws |
| Multiplicity | Benjamini-Hochberg at 0.05 |

The confirmatory model set contains Gemma 2 2B, Gemma 2 9B, and Llama 3.1 8B.
Gemma 3 1B, 4B, and 12B are replication extensions. Six residual-stream layers
are specified for each model; the exact model identifiers, SAE releases,
widths, and layer indices are recorded in the analysis configuration.

## Analysis flow

```text
timed transcript
      |
      v
model tokens and SAE activations
      |
      v
TR-aligned predictor arrays -----------+
      |                                |
      v                                v
source-only feature banks       matched LM/BOLD targets
      |                                |
      +---------------+----------------+
                      v
          purged nested ridge regression
                      |
            +---------+---------+
            v                   v
   held-out prediction   held-out feature ablation
                                |
                                v
                    matched reliance profiles
                                |
                  +-------------+-------------+
                  v             v             v
          correspondence   structured nulls   sensitivities
                                |
                                v
                    aggregate and verification
```

## Token and TR alignment

Transcript words are assigned to TRs using the midpoint of each word's start
and end times. SAE activations are aggregated within each TR. The primary rule
is mean positive activation over all assigned tokens, including zero
activations in the denominator; empty TR bins are zero-filled.

The primary BOLD design uses five lags. The language-model design is unlagged.
Pooling sensitivities evaluate maximum positive activation, sum of positive
activation, and positive-event count when token-level records are available.

## Feature banks

Feature selection does not use BOLD targets, language-model targets, fitted
coefficients, or evaluation metrics. A candidate feature must activate in at
least five source-story TRs, activate in no more than 95% of source-story TRs,
and belong to a specified layer.

The primary bank ranks eligible features by binary temporal variance,
`p(1-p)`, with positive-activation RMS and feature ID as deterministic
tie-breakers. Allocation across layers is equal, and bank sizes are nested
prefixes when enough features are available. The `p(1-p)` ranking favors
features active in about half the source TRs, so selection itself makes
prevalence relevant to interpreting the reliance profiles.

Within-story banks are rebuilt from each purged outer-training fold.
Cross-story banks are selected from the complete source story and applied to
the other story. Paired BOLD and language-model fits use the same
`(layer, feature_id)` coordinates. Sensitivities include active-TR prevalence
and within-layer complete-linkage correlation pruning at thresholds 0.70,
0.80, and 0.90.

## Prediction, reliance, and nulls

Ridge penalties are selected within training data from 0.1 through `10^10`.
Predictor standardization is fitted on training rows only. Adjacent TRs are
purged around each held-out block. For cross-story transfer, nested source-story
fits select the penalty, a final predictor is refit on the full source story,
and its scaling and weights are applied unchanged to the other story.

Two structured null families test different failure modes:

1. The primary transfer temporal null independently circularly shifts the
   source and held-out language-model targets by at least 10 TRs and repeats
   penalty selection and fitting for each shift. Offset pairs are shared
   across the six models within each direction and bank condition. The
   source-only bank is not reselected because its construction is independent
   of the shifted targets.
2. The feature-identity null permutes feature identities within layer while
   preserving layer composition.

The alternative-pooling temporal nulls also share offsets across models, but
shift only the source language-model target. Their null distributions are
therefore not directly comparable to the primary transfer null.

The grouped sensitivity ablates correlation-defined feature clusters to test
whether substitutable features make single-feature reliance unstable and uses
corresponding retuned temporal nulls. Null p-values and multiplicity correction
are calculated from stored null draws.

## Controls and sensitivities

The analysis includes leave-one-participant-out and split-half BOLD
reliability; reliance stability across folds, participants, and transfer
directions; low-level, position, and surprisal controls; dense residual-stream
and matched-random-SAE controls; alternative source-story banks and pooling
definitions; and participant-average BOLD sensitivity.

These analyses diagnose measurement quality and specification dependence.
They do not change the primary estimand.

## Semi-synthetic detectability

Power designs preserve the empirical predictor geometry, folds, feature
groups, censoring pattern, target dimensionality, participant structure, and
residual covariance. A cell is detected when held-out reliance correspondence
exceeds the 95th percentile of the same-design `rho=0` distribution. The
specified identifiability criterion is 80% detection at oracle correlation
0.40. Null cells contain 500 replicates and positive cells contain 250.

### Planted correspondence

The statistic the pipeline scores is a held-out ablation loss. In a linear
mapping the noiseless ablation loss of feature group `k` is
`beta_k^2 * ||sum_{j in G_k} x~_j||^2` on the standardized design, independent
of the coefficient's sign. The corrected calibration
(`fmri_sae_correspondence.power_oracle`) therefore prescribes the
correspondence in that metric: two positive rank-score loss profiles with the
prescribed Spearman correlation are generated by reversing a rank block of a
random ordering, and each system's coefficients are set to
`s_k * sqrt(p_k / c_k)` with a sign vector drawn independently per system, so
the noiseless ablation-loss ordering follows the prescribed profile. The
realized correlation, including finite-rank quantization and ties, is recorded
for every replicate. Feature groups silent on the
held-out story receive coefficient zero and tie at the bottom in both systems.
Planted signals are scaled to one or two times each system's mean per-target
residual standard deviation and added to resampled, recentred residual rows.
Every other step (design artifacts, folds, alpha grid, seed contract, ridge
solver, grouped ablation, statistic, detection rule) is the configured power
stage, imported unchanged.

The eight committed aggregates under `evidence/power_oracle/` cover Gemma 2
2B and Llama 3.1 8B, the variance and correlation-pruned banks, and both
transfer directions: 80 grid cells from 192 validated shards. The realized
planted correlations are 0.21--0.23, 0.40--0.42, 0.61--0.62, and 0.80--0.81 at
the four nominal levels in every design. Detection over the 64 positive cells
ranges from 0.016 to 0.104 (mean 0.060); no cell approaches the 80% criterion,
at either signal scale. At nominal `rho=0.8`, detection ranges from 0.040 to
0.104. The recovered correlation's mean lies between -0.001
and 0.029 at `rho=0` and between 0.004 and 0.039 at `rho=0.8`, against
replicate standard deviations of 0.06--0.10 and null thresholds of 0.11--0.16.
The `rho=0` exceedance fraction is 0.050 by construction.

The calibration fixes the banks and fits within the held-out story rather
than repeating source-to-target transfer. It omits both structured null tests,
resamples residual rows independently (removing their autocorrelation), and
uses one planted profile across target dimensions and BOLD participants. These
choices are intended to favor recovery, but they do not establish a formal
upper bound on the full test's sensitivity. Only Gemma 2 2B and Llama 3.1 8B
were calibrated.

### Superseded first calibration

The configured power stage (`fmri_sae_correspondence.power`) plants two
signed, centred rank-score profiles with a prescribed Spearman correlation and
uses them directly as ridge coefficients. Because ablation loss follows
coefficient magnitude, that construction does not prescribe the
ablation-ordering correlation: on the same generator the noiseless
ablation-loss correlation is about 0.76 at nominal `rho=0` and about 0.55 at
nominal `rho=0.8` for 256 features. Its monotone detection curve arose because
the fitted ablation losses carry a signed cross-term and the block-reversal
generator makes the two systems' coefficient signs agree more often as the
nominal correlation rises; with the corrected loss profiles but a sign vector
shared by both systems, the recovered correlation on a real design was 0.41
and 0.12 at nominal `rho=0`, and with independent signs 0.00 and -0.11 at
`rho=0` and 0.06 and -0.06 at `rho=0.8`. The eight aggregates under
`evidence/power/` and the four within-story aggregates under
`evidence/power_within_story/` were produced with that construction; they are
retained and cataloged, with a `SUPERSEDED.md` notice in each directory, and
none of their detection fractions is a usable power estimate. The
within-story design was not recalibrated.

### Identifiability of the reliance ordering (exploratory)

Separate exploratory simulations vary the number of feature groups, K, using
the corrected planting construction and a signal scale of one residual
standard deviation. They distinguish self-recovery (a fitted profile's
correlation with its own planted profile) from cross-system correspondence
(the correlation between the two fitted profiles).

In the self-recovery sweep, eight transfer-derived designs and five seeds give
40 replicates per bank size. Median BOLD self-recovery is 0.990 at K=24,
0.813 at K=48, 0.601 at K=64, and 0.278 at K=128. A separate paired sweep uses
eight transfer-derived designs and three seeds (24 replicates per bank size).
At a planted cross-system correlation of 0.80, median recovered correspondence
is 0.556 at K=24, 0.394 at K=48, and 0.023 at K=256. The analogous sweep on
four within-description designs and five seeds (20 replicates per bank size)
gives 0.683 at K=24 and 0.037 at K=256. These simulations fit within fixed
designs; they do not rerun the full transfer test.

A separate predictor/noise control at K=256 uses eight designs and four seeds
per condition. With empirical predictors and residual noise, median BOLD
self-recovery is 0.060 and LM self-recovery is 0.140. Replacing the predictors
with i.i.d. Gaussian columns of the same shape while retaining empirical noise
gives 0.998 and 0.302, respectively. These are self-recovery values, not
cross-system correspondence. Changing the predictor matrix alters temporal
structure, repeated rows, and correlations together, so this comparison does
not isolate feature correlation or rule out effects of sample size and noise.
The exploratory sweeps suggest that predictor structure and bank size matter
for recovery, but do not establish a universal feature-count limit or adequate
sensitivity for the full transfer test.

Outputs are under `evidence/identifiability/`, scripts under
`scripts/identifiability/`, and bank-size summaries in
`results/identifiability_summary.json`. These exploratory runs followed the
configured grid and use fewer replicates than the corrected calibration.

## Interpretation boundary

The analysis did not establish reproducible sparse-feature correspondence.
Its measured sensitivity is insufficient to conclude that correspondence of
any tested size is absent. Prediction alone does not establish shared feature
reliance, and correspondence does not establish causal or biological
equivalence.

A positive correspondence result requires both structured nulls, both transfer
directions, three adjacent bank sizes including 256, two model families, and
the primary and secondary bank constructions. An identifiable-null conclusion
additionally requires stable conditional and grouped results across bank size
and pooling and at least 80% detection at oracle correlation 0.40. Those
conditions were not met.

Detection remains at or below 0.104 across all positive calibration conditions,
including nominal correlation 0.80 at both signal scales. This limited
sensitivity prevents interpreting the empirical non-detection as evidence
that brain-model correspondence is absent. The Gemma 3 systems were not
calibrated.
