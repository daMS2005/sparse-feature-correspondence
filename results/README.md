# Power reporting products

This directory contains the compact evidence used to report semi-synthetic
detectability.

## Corrected calibration

- [`../evidence/power_oracle/`](../evidence/power_oracle/) contains eight
  sanitized aggregate JSON files covering two model families, two feature-bank
  constructions, both transfer directions, and two injected-signal scales, with
  the planted correspondence defined in the ablation-loss metric;
- `power_oracle_summary.json` is the authoritative machine-readable summary;
- `power_oracle_table.tex` is the manuscript-facing table; and
- `power_oracle_macros.tex` provides the corresponding manuscript values.

The aggregate hashes are cataloged in
[`../checksums/power_oracle_aggregates.sha256`](../checksums/power_oracle_aggregates.sha256).
The 192 intermediate shards are not included; their digests are listed under
`../evidence/power_oracle/shard_hashes/`.

The realized planted correlations are 0.21--0.23, 0.40--0.42, 0.61--0.62, and
0.80--0.81 at the four nominal levels. Detection over the 64 positive cells
ranges from 0.016 to 0.104; at nominal correlation 0.80 alone, it ranges from
0.040 to 0.104. No cell reaches the 80% criterion at either signal scale. The
zero-correlation exceedance fraction is 0.050 by construction because those
replicates also set the detection threshold.

The calibration fits within fixed designs and does not repeat bank selection,
source-to-target transfer, or the two structured null tests. Its limited
sensitivity prevents interpreting the empirical non-detection as evidence
that correspondence is absent; it does not establish an upper bound on the
full transfer test's sensitivity.

## Superseded products

`power_summary.json`, `power_summary_table.tex`, and `power_summary_macros.tex`
are derived from the eight aggregates under
[`../evidence/power/`](../evidence/power/), and the four within-story
aggregates under
[`../evidence/power_within_story/`](../evidence/power_within_story/) are
cataloged in
[`../checksums/power_within_story_aggregates.sha256`](../checksums/power_within_story_aggregates.sha256).
Both sets were produced with a planted-profile construction whose prescribed
correlation applies to signed coefficients rather than to ablation-loss
orderings (see the `SUPERSEDED.md` notice in each directory and
[`../DESIGN.md`](../DESIGN.md)). They are retained for the record and their
detection fractions are not usable power estimates.
