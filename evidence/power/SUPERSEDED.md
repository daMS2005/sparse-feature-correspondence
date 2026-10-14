# Superseded aggregates

The eight aggregates in this directory were produced by the configured power
stage, whose planted profiles are signed, centred rank scores with a prescribed
Spearman correlation, used directly as ridge coefficients. The statistic the
pipeline scores is a held-out ablation loss, which follows coefficient
magnitude and is blind to sign, so the prescribed correlation of the signed
profiles is not the correlation of the ablation-loss orderings the test
measures. On that generator the noiseless ablation-loss correlation is about
.76 at nominal `rho = 0` and about .55 at nominal `rho = .8` for 256 features.
The monotone detection curve these aggregates report reflects the fraction of
features whose coefficient signs agree across the two systems, which the
block-reversal generator raises with the nominal correlation, rather than
recovery of a planted ordering.

The detection fractions in these files, and the reporting products derived
from them (`results/power_summary.json`, `results/power_summary_table.tex`,
`results/power_summary_macros.tex`), must not be cited as achieved power.
They are retained so that the record is complete and so that the catalog in
`checksums/power_aggregates.sha256` remains verifiable.

The corrected calibration, which plants the correspondence in the
ablation-loss metric with signs drawn independently per system, is provided by
`fmri_sae_correspondence.power_oracle` and released under
`evidence/power_oracle/` with its own catalog and summary. See `DESIGN.md`.
