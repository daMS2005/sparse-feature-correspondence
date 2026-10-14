# Scientific provenance

The public release binds four components:

1. the analysis configuration defining the scientific choices;
2. the source implementing those choices;
3. eight sanitized semi-synthetic power aggregates; and
4. the JSON and LaTeX reports derived from those aggregates.

[`SCIENTIFIC_EQUIVALENCE.json`](SCIENTIFIC_EQUIVALENCE.json) records equal
schema-v1 and schema-v2 scientific-projection hashes, equal before/after
scientific and numerical projections for all eight aggregates, and the exact
raw hash required for the deterministically reconstructed eighth aggregate.

The checksum catalogs are documented in
[`../checksums/README.md`](../checksums/README.md). Public provenance records
use release-relative paths and content hashes; they do not depend on
unavailable workspace locations.

Repository curation changed names, documentation, packaging, and path
resolution. It did not rerun model extraction, fMRI processing, regression,
null tests, or power simulation. Scientific choices and reported numerical
values were preserved. The committed aggregates are sanitized copies of the
evidence used for reporting, and their hashes bind them to the released
summary.

The eight corrected-calibration aggregates under `evidence/power_oracle/` were
produced after the main analysis by the `power_oracle` stage, which imports
the configured power stage unchanged for every step except the planted
profiles; they are cataloged separately, and the sanitized copies omit the
analysis-tree module hashes that the originals record. The four within-story
aggregates are an exploratory sensitivity produced with two declared
specification changes; they, like the eight aggregates in the equivalence
record above, are superseded by the corrected calibration (see
[`../DESIGN.md`](../DESIGN.md)).

The 256 intermediate power shards are not distributed. Their coverage and
validation status are represented in the eight aggregates, but only aggregate,
grid, and report verification is claimed publicly.

This record establishes release integrity and the boundary of the shipped
evidence. It does not substitute for access to the restricted scientific inputs
needed for full recomputation.
