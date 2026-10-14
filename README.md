# Can We Detect Shared Features Between Language Models and the Brain?

Code, calibration evidence and reproducibility checks for the paper by
**Daniel Mora**.

**Paper:** [paper.pdf](paper.pdf)

Sparse language-model features can predict both human brain responses (fMRI)
and a language model's own hidden states. That does not mean the two rely on
the *same* features in the same way. This repository tests that question
directly on two spoken stories, `shapesphysical` and `shapessocial`, from the
Narratives fMRI dataset (59 participants).

## Main results

- **Prediction transfers, correspondence does not show up.** Across six
  model-SAE pairs and both transfer directions, the mean rank correlation
  between brain and model feature-reliance profiles is -0.056, and no model
  passes both the timing and feature-identity null tests in both directions.
- **The null is inconclusive, and the calibration shows why.** With 256
  features, a planted positive correspondence is detected in only 1.6% to
  10.4% of replicates across 64 conditions (false-positive rate fixed at 5%),
  including a planted correlation of 0.80.
- **Bank size matters.** In an exploratory sweep, median recovered
  correspondence is 0.556 with 24 features and 0.023 with 256 features at the
  same planted correlation of 0.80.

The takeaway: before reading a null like this as "no shared organization", an
analysis needs to show that it can recover known feature agreement.

This is a predictive correspondence analysis. It does not imply that the brain
implements a language model, that SAE coordinates are biological units, or
that the measured associations are causal.

## Method

Words and model activations are aligned to the stories' 1.5-second TR bins.
Source-story SAE activations define feature banks without using evaluation
outcomes. The cortical models use five feature lags (0-4 TRs); the language-model
target is the unlagged, TR-averaged final hidden state. Five-fold nested ridge
regression on the source story uses contiguous folds and a four-TR purge to
select penalties. Each predictor is then refit on the full source story and
evaluated on the other story with its scaling and weights fixed. The reverse
direction is fitted separately.

Feature reliance is measured as the held-out change in out-of-sample variance
explained after ablation. The primary comparison is the Spearman correlation
between matched reliance profiles. Circular target shifts and within-layer
feature permutations provide the two main null distributions. Full details,
including controls and sensitivity analyses, are in [`DESIGN.md`](DESIGN.md).

## What is included

The repository ships the analysis package, the schema-v2 configuration, the
eight aggregates of the corrected semi-synthetic calibration, deterministic
reporting outputs, and checksum catalogs. The calibration aggregates cover
Gemma 2 2B and Llama 3.1 8B, two feature-bank constructions, both transfer
directions, and two injected-signal scales. The earlier power aggregates and
the within-story aggregates remain cataloged but are superseded; their planted
profiles prescribed the correlation of signed coefficients rather than of the
ablation-loss orderings the test measures (see [`DESIGN.md`](DESIGN.md)).

The corrected calibration plants a known correlation between ablation-loss
profiles in banks of 256 features. Detection ranges from 0.016 to 0.104 across
64 positive conditions: eight designs, four nominal correlations (0.20--0.80),
and signal scales of one and two residual standard deviations. At nominal
correlation 0.80, detection ranges from 0.040 to 0.104. The false-positive
fraction is 0.050 by construction, using the same-design zero-correlation
replicates to set the threshold. No condition reaches the 80% detection
criterion.

These simulations fix the feature banks and fit within the held-out story;
they do not repeat source-to-target transfer or the two structured null tests.
They demonstrate limited sensitivity under the tested conditions, without
establishing a bound on the full transfer test's sensitivity. The empirical
results do not establish sparse-feature correspondence, and the calibration
prevents interpreting non-detection as evidence that correspondence is absent.

An exploratory sweep examines whether smaller banks improve recovery. At a
planted cross-system correlation of 0.80, median recovered correspondence is
0.556 with 24 features and 0.023 with 256 features on the transfer-derived
designs. Separate controls measure recovery of each system's own planted
profile; those values are not cross-system correspondence. The sweep does not
establish a universal feature-count limit or adequate sensitivity for the full
transfer test. Details and the distinct simulation arms are in
[`DESIGN.md`](DESIGN.md#identifiability-of-the-reliance-ordering-exploratory).

Raw fMRI data, participant identifiers, stimuli, transcripts, model weights,
SAE weights, cached activations, and intermediate model outputs are not part of
this repository. The 192 calibration shards are also not distributed, so
readers can verify the eight aggregates and the reports derived from them,
but not each individual shard. See [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) for the
complete data boundary.

## Run the checks

The default development installation is data-free and does not download model
weights:

```bash
python -m pip install -e ".[dev]"
fmri-sae-correspondence --version
fmri-sae-correspondence config validate
fmri-sae-correspondence results verify
python -m pytest -q
./verify.sh evidence/power evidence/power_within_story evidence/power_oracle
```

The command-line package can be used outside the repository root. Full
recomputation requires authorized inputs and the optional scientific
dependencies; [`REPRODUCE.md`](REPRODUCE.md) documents that setup and the
available analysis stages.

## Layout

```text
src/fmri_sae_correspondence/  analysis code and command-line interface
configs/analysis.yaml         scientific configuration
evidence/power_oracle/        eight corrected-calibration aggregates and shard hashes
evidence/identifiability/     outputs of the identifiability sweep
scripts/identifiability/      the sweep scripts and their summary
evidence/power/               eight superseded power aggregates
evidence/power_within_story/  four superseded within-story aggregates and shard hashes
scripts/within_story_power/   design, shard, and aggregation scripts for that sensitivity
results/                      JSON and LaTeX summaries
checksums/                    source and evidence integrity catalogs
provenance/                   scientific-equivalence record
tests/                        deterministic tests using synthetic data
```

## Citation

If you use this code, please cite the paper (citation metadata is also in
[`CITATION.cff`](CITATION.cff)):

> Daniel Mora (2026). *Can We Detect Shared Features Between Language Models
> and the Brain?*

## License

MIT. See [`LICENSE`](LICENSE).
