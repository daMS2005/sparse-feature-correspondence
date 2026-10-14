# Integrity catalogs

The release contains separate catalogs for source and scientific aggregates.
Checksums establish file identity; they do not rerun the scientific analysis.

| File | Scope |
| --- | --- |
| `SCIENTIFIC_CODE.sha256` | Released scientific source files |
| `SCIENTIFIC_CODE_MANIFEST.json` | Source roles, release-relative paths, sizes, and hashes |
| `power_aggregates.sha256` | Eight committed sanitized power aggregates |
| `power_within_story_aggregates.sha256` | Four committed within-story aggregates (superseded) |
| `power_oracle_aggregates.sha256` | Eight committed aggregates of the corrected calibration |
| `identifiability_outputs.sha256` | Outputs of the identifiability sweep, by release-relative path |

## Source verification

From the repository root:

```bash
./verify.sh
```

The source catalog can also be checked directly:

```bash
shasum -a 256 -c checksums/SCIENTIFIC_CODE.sha256
```

The JSON manifest describes the same public source set using only
release-relative identifiers. A successful check establishes that the source
files match the release catalog.

## Aggregate verification

The aggregate catalog lists bare filenames. From the repository root:

```bash
cd evidence/power
shasum -a 256 -c ../../checksums/power_aggregates.sha256
```

On systems that provide `sha256sum`:

```bash
cd evidence/power
sha256sum -c ../../checksums/power_aggregates.sha256
```

Expected: eight successful checks and no failures. These files cover the
complete model, bank, direction, and 160-cell power-grid union represented in
the released summary.

The aggregates record coverage from 256 validated intermediate shards. The
shards are not committed, so the catalog verifies the eight aggregates rather
than the individual shards.

## Corrected calibration aggregates

```bash
cd evidence/power_oracle
shasum -a 256 -c ../../checksums/power_oracle_aggregates.sha256
```

Expected: eight successful checks. Each aggregate records 24 shards at
residual scales 1.0 and 2.0; `evidence/power_oracle/shard_hashes/` lists the
SHA-256 digest of every one of the 192 shards, which are not distributed. The
aggregates under `evidence/power/` and `evidence/power_within_story/` remain
cataloged but are superseded; see the `SUPERSEDED.md` file in each directory.

## Report verification

The installed `fmri-sae-correspondence results verify` command validates
aggregate coordinates and checks the reporting products against the committed
aggregates. Explicit `--write --force` flags are required to regenerate them.

Verification should establish that:

- all eight aggregate hashes match;
- the expected models, banks, directions, oracle correlations, residual scales,
  and 160 grid cells are present;
- replicate counts agree with the analysis configuration; and
- regenerated JSON and LaTeX values agree with the committed reports.

These checks establish aggregate and reporting integrity. They do not attest to
the restricted fMRI data, stimuli, model weights, SAE weights, or the
undistributed intermediate shards required for full scientific recomputation.
