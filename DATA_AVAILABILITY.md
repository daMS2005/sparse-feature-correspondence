# Data availability

This repository includes:

- scientific source code and analysis configuration;
- eight sanitized aggregates of the corrected semi-synthetic calibration and
  the SHA-256 digests of their 192 undistributed shards;
- eight superseded power aggregates and four superseded within-story
  aggregates, with the digests of the latter's 128 undistributed shards;
- the outputs of the identifiability sweep and their checksum catalog;
- the machine-readable summaries and LaTeX reporting products; and
- source and aggregate checksum catalogs.

The committed aggregates support integrity, grid-completeness, and report
verification. They record the validated intermediate shards each calculation
used, but those shards are not distributed and cannot be independently
checked shard by shard.

The repository does not distribute:

- raw or derived participant-level fMRI data;
- participant identifiers;
- story audio, transcripts, or word timings;
- language-model checkpoints or tokenizer assets;
- sparse-autoencoder weights;
- cached hidden states or activations;
- fitted participant-level outputs; or
- other intermediate scientific artifacts.

Full recomputation requires authorized copies of the cleaned
participant-by-time BOLD targets, language-model predictors and targets,
transcript timing metadata, model assets, and SAE assets. External locations
must be supplied explicitly through configuration or documented environment
variables; no private machine layout is assumed.

The distributed files are sanitized to remove machine identifiers and contain
no participant-level data.
