# Changelog

## Unreleased

- Corrects the calibration detection ranges and their scope, distinguishes
  self-recovery from cross-system correspondence in the exploratory sweeps,
  and removes the unsupported upper-bound interpretation.
- Clarifies fixed-predictor ablation, source-to-target fitting, and the primary
  and alternative-pooling temporal nulls; corrects the documented penalty grid.
- Shares temporal-null offset schedules across models for each condition,
  matching the coordinated null results used in the paper. Adds regression
  checks for shared schedules and the reported primary offset vectors.
- Numerical outputs and the analysis configuration are unchanged.

## 0.1.4 - 2026-09-09

- Adds the identifiability sweep: scripts under `scripts/identifiability/`
  that plant an ablation-loss ordering with the `power_oracle` construction
  and measure its recovery by the configured estimator as the number of real
  feature groups varies, its sanitized outputs under `evidence/identifiability/`
  with a checksum catalog, and `results/identifiability_summary.json`.

## 0.1.3 - 2026-09-09

- Adds `fmri_sae_correspondence.power_oracle`, a calibration stage that plants
  the correspondence in the ablation-loss metric with signs drawn
  independently per system, and the `run power-oracle` command.
- Publishes the eight sanitized aggregates of that calibration, the SHA-256
  digests of their 192 undistributed shards, their checksum catalog, and the
  corresponding summary, table, and macro products.
- Marks the previous power aggregates and the within-story aggregates as
  superseded: their planted profiles prescribed the correlation of signed
  coefficients rather than of ablation-loss orderings.

## 0.1.2 - 2026-09-07

- Adds four sanitized within-story power aggregates as an exploratory
  sensitivity, with their checksum catalog, shard hash inventory, and the
  scripts that produce them from the configured power stage.
- Documents the within-story calibration and its upper-bound interpretation.

## 0.1.1 - 2026-09-04

- Provides the `fmri_sae_correspondence` package and
  `fmri-sae-correspondence` CLI.
- Publishes eight sanitized power aggregates and deterministic reporting
  products.
- Verifies source identity, aggregate integrity, grid completeness, and report
  equivalence without restricted data.
- Documents the bounded non-detection and the limits of the distributed
  evidence.
