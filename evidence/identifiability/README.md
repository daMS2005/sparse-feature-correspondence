# Identifiability sweep outputs

Outputs of exploratory simulations examining how predictor structure and bank
size affect recovery. The runs use the `power_oracle` planting construction
(positive rank-score loss profiles, signs drawn independently per system) and
the configured estimator. Separate arms measure recovery of each system's own
planted profile and correspondence between the two fitted profiles. They do not
repeat the full transfer test or establish a universal feature-count limit.
The scripts are under `scripts/identifiability/`; bank-size summaries are in
`results/identifiability_summary.json`.

| directory | what varies | rows |
|---|---|---|
| `positive_control/` | i.i.d. Gaussian predictors and noise; feature groups, rows, targets, signal scale | per-cell JSON shards |
| `predictor_noise_isolation/` | real vs i.i.d. predictors crossed with empirical vs white noise, eight designs and four seeds per condition | one JSON per design |
| `k_sweep/` | random subsets of K real feature groups, self-recovery only | one JSON per design |
| `surface/` | K by rows-per-column by real vs i.i.d. predictors | JSONL shards |
| `threshold/` | K from 24 to 128 on real predictors at full sample size, eight designs, five seeds | JSONL shards |
| `cross_system_transfer/` | recovered cross-system correlation at a planted .80 by K, eight transfer-derived designs and three seeds | JSONL shards |
| `cross_system_within/` | the same on four within-description designs and five seeds | JSONL shards |

Fields: `self_brain` and `self_lm` are the Spearman correlation between a fitted
reliance profile and its own planted ablation-loss profile; `recovered` is the
Spearman correlation between the two fitted profiles, the manuscript's statistic;
`planted_oracle` is the realized correlation of the two planted loss profiles.
The `threshold/` and `cross_system_transfer/` arms use different runs and seed
counts; their self-recovery values should not be combined as one paired result.
These runs are exploratory and were produced after the configured grid was frozen;
they require the restricted design artifacts to reproduce. Hashes are cataloged in
`checksums/identifiability_outputs.sha256`.
