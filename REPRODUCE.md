# Reproducing and verifying the analysis

The release supports three distinct levels of inspection. They should not be
treated as equivalent:

1. Source validation requires only this repository.
2. Aggregate and report validation uses the eight sanitized aggregates of the
   corrected calibration under `evidence/power_oracle/`, and the superseded
   aggregates under `evidence/power/` and `evidence/power_within_story/`.
3. Full scientific recomputation additionally requires restricted fMRI,
   stimulus, model, and SAE inputs described in
   [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md).

The intermediate shards (192 for the corrected calibration, 256 and 128 for
the superseded grids) are not distributed. Their recorded coverage can be
checked from the committed aggregates, but the individual shards cannot be
independently revalidated from this repository.

## Installation

Python 3.11 is the reference runtime. The data-free checks also run under
Python 3.12.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

The package lives at `src/fmri_sae_correspondence`. The installed entry point
is:

```bash
fmri-sae-correspondence --help
```

Configuration and external input locations are supplied explicitly through
the CLI and documented environment variables. The public configuration does
not assume a username, host, or machine-specific directory layout.

## 1. Source validation

```bash
python -m pip install -e ".[dev]"
fmri-sae-correspondence --help
python -m pytest -q
./verify.sh
```

These checks validate installation, lightweight interfaces, deterministic unit
tests, and the public source catalog. They do not validate raw scientific
inputs or rerun the analysis.

## 2. Aggregate integrity

The aggregate catalog contains bare filenames. From the repository root:

```bash
cd evidence/power
shasum -a 256 -c ../../checksums/power_aggregates.sha256
```

On systems that provide `sha256sum` instead of `shasum`:

```bash
cd evidence/power
sha256sum -c ../../checksums/power_aggregates.sha256
```

Expected: eight successful checks and no failures.

The committed aggregates cover:

- Gemma 2 2B and Llama 3.1 8B;
- source-story variance and correlation-pruned feature banks;
- both `shapesphysical -> shapessocial` and
  `shapessocial -> shapesphysical` transfer directions;
- 160 power-grid cells;
- oracle correlations `0.0, 0.2, 0.4, 0.6, 0.8`; and
- residual-scale multipliers `0.25, 0.5, 1.0, 2.0`.

Aggregate validation checks the expected model, bank, direction, cell,
replicate, and metadata coordinates. It does not recreate or inspect the 256
undistributed intermediate shards.

The four within-story aggregates have their own catalog:

```bash
cd evidence/power_within_story
shasum -a 256 -c ../../checksums/power_within_story_aggregates.sha256
```

Expected: four successful checks. Each aggregate records 32 shards at residual
scale 1.0 only; `evidence/power_within_story/shard_hashes/` lists the SHA-256
digest of every one of the 128 shards, which are not distributed.

The within-story designs, shards, and aggregates are produced by the scripts
in `scripts/within_story_power/`, which call the configured power stage's
functions unchanged with the two specification changes described in
[`DESIGN.md`](DESIGN.md):

```bash
python scripts/within_story_power/materialize_within_story_designs.py --out-dir designs
python scripts/within_story_power/run_within_story_shard.py --design designs/DESIGN.npz \
  --shard-index 0 --shard-count 32 --shard-output shards/DESIGN.s000of032.json
python scripts/within_story_power/aggregate_within_story_shards.py --design designs/DESIGN.npz \
  --shards-dir shards --scales 1.0 --output aggregates/DESIGN.json
```

They require the same restricted inputs and completed within-story fit
artifacts as the configured stage.

### Corrected calibration

The corrected calibration is the `power-oracle` stage. It consumes the same
design artifacts as the configured power stage and writes shards and
aggregates of the same shape, with the planted correspondence defined in the
ablation-loss metric:

```bash
cd evidence/power_oracle
shasum -a 256 -c ../../checksums/power_oracle_aggregates.sha256
cd ../..
fmri-sae-correspondence run power-oracle --design designs/DESIGN.npz --scales 1.0,2.0 \
  --shard-index 0 --shard-count 24 --shard-output shards/DESIGN.s000of024.json
fmri-sae-correspondence run power-oracle --design designs/DESIGN.npz --scales 1.0,2.0 --aggregate \
  --shard-paths shards/DESIGN.s*of024.json --aggregate-output aggregates/DESIGN.json
fmri-sae-correspondence run power-oracle --summarize evidence/power_oracle/power_oracle_*.json \
  --summary-output results/power_oracle_summary.json
```

Expected for the catalog: eight successful checks. Each aggregate records 24
shards at residual scales 1.0 and 2.0; `evidence/power_oracle/shard_hashes/`
lists the digest of every one of the 192 shards. Shard and aggregate modes
require the restricted inputs and completed transfer artifacts.

### Identifiability sweep

The scripts under `scripts/identifiability/` plant an ablation-loss ordering
with the `power_oracle` construction over a subset of a design's real feature
groups and measure its recovery by the configured estimator. They take the
design artifacts and an alpha grid on the command line and write JSON or JSONL
outputs; the committed outputs under `evidence/identifiability/` are cataloged
in `checksums/identifiability_outputs.sha256`, and
`scripts/identifiability/summarize.py` regenerates
`results/identifiability_summary.json` from them:

```bash
cd evidence/identifiability
shasum -a 256 -c ../../checksums/identifiability_outputs.sha256
cd ../..
python scripts/identifiability/summarize.py
```

## 3. Report verification

The corrected-calibration aggregates deterministically produce
`results/power_oracle_summary.json`, and the manuscript products
`results/power_oracle_table.tex` and `results/power_oracle_macros.tex` are
derived from the same aggregates. The superseded aggregates produce:

- `results/power_summary.json`;
- `results/power_summary_table.tex`; and
- `results/power_summary_macros.tex`.

The reporting validator checks that all eight aggregate hashes match, that the
expected grid is complete, and that recomputed summary values agree with the
committed products. Report regeneration is a deterministic transformation of
the committed aggregates; it is not a rerun of model fitting, fMRI processing,
or power simulation.

The machine-readable summary is authoritative when a displayed value is
rounded differently in a manuscript table.

## Full scientific recomputation

Full recomputation requires authorized copies of the inputs listed in
[`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md), compatible model and SAE
assets, and sufficient compute. The analysis configuration records the models,
stories, layers, feature-bank construction, alignment, folds, purge, ridge
grid, nulls, controls, seeds, and power design.

Install the complete optional dependency set and provide external roots before
running a data-consuming stage:

```bash
python -m pip install -e ".[full]"
export FMRI_SAE_DATA_ROOT=/path/to/authorized-inputs
export FMRI_SAE_OUTPUT_ROOT=/path/to/generated-artifacts
export FMRI_SAE_CACHE_ROOT=/path/to/model-cache
fmri-sae-correspondence run feature-banks --help
```

Each stage exposes its preserved scientific options through `run STAGE
--help`. Explicit CLI root options take precedence over the environment
variables. Existing artifacts are reused only when their fingerprints match;
otherwise a stage fails closed unless its `--force` option is supplied.

The broad stage order is:

1. align transcript tokens and SAE activations to TRs;
2. construct source-only feature banks;
3. fit within-story and cross-story ridge models;
4. derive held-out feature-reliance profiles;
5. evaluate temporal and feature-identity nulls;
6. run reliability, grouped-reliance, control, and pooling sensitivities;
7. calculate semi-synthetic detectability; and
8. validate aggregates and generate reporting products.

The public repository does not claim that full recomputation is possible
without the restricted inputs.

## Interpretation boundary

Successful integrity checks establish that the committed files match their
catalogs and that the reports follow from the committed aggregates. They do not
establish that the raw fMRI data, stimuli, model weights, or SAE weights are
publicly available, and they do not turn a predictive correspondence into a
causal result.
