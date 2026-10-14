"""Tests for the ablation-loss oracle calibration."""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np

from fmri_sae_correspondence import power, power_oracle

ROOT = Path(__file__).resolve().parents[1]


def _design(rng: np.random.Generator, n_feat: int = 24, n_rows: int = 120, lags: int = 3) -> dict:
    brain_x = rng.normal(size=(n_rows, n_feat * lags)) + 0.4 * rng.normal(size=(n_rows, 1))
    lm_x = rng.normal(size=(n_rows // 2, n_feat)) + 0.4 * rng.normal(size=(n_rows // 2, 1))
    return {
        "design_schema": "paired_target_design_v1",
        "participant_structure": "paired_system_rows",
        "design_label": "gemma_2_2b/source_story_variance/shapesphysical_to_shapessocial",
        "sha256": "synthetic-design-sha",
        "model_key": "gemma_2_2b",
        "bank_type": "source_story_variance",
        "direction": "shapesphysical_to_shapessocial",
        "censoring_manifest": "synthetic-test",
        "brain_x": brain_x,
        "lm_x": lm_x,
        "brain_residual": rng.normal(scale=2.0, size=(n_rows, 5)),
        "lm_residual": rng.normal(scale=3.0, size=(n_rows // 2, 4)),
        "brain_participant_ids": np.repeat(["p0", "p1"], n_rows // 2),
        "lm_participant_ids": np.asarray(["shared"] * (n_rows // 2)),
        "brain_censor_mask": np.zeros(n_rows, dtype=bool),
        "lm_censor_mask": np.zeros(n_rows // 2, dtype=bool),
        "brain_folds": power.repeated_participant_purged_folds(2, n_rows // 2, n_folds=2, purge_adjacent=1),
        "lm_folds": power.contiguous_purged_folds(n_rows // 2, n_folds=2, purge_adjacent=1),
        "brain_feature_groups": [np.arange(i * lags, (i + 1) * lags) for i in range(n_feat)],
        "lm_feature_groups": [np.asarray([i]) for i in range(n_feat)],
    }


def test_noiseless_ablation_loss_equals_the_planted_profile() -> None:
    rng = np.random.default_rng(3)
    d = _design(rng)
    norms = power_oracle.group_signal_norms(d["brain_x"], d["brain_feature_groups"])
    p, _ = power_oracle.make_ablation_oracle_profiles(24, 0.6, rng)
    beta = power_oracle.oracle_coefficients(p, norms, rng.choice(np.asarray([-1.0, 1.0]), size=24))
    x = d["brain_x"]
    xs = (x - x.mean(0)) / x.std(0)
    cols = np.zeros(x.shape[1])
    for b, g in zip(beta, d["brain_feature_groups"], strict=False):
        cols[g] = b
    y = xs @ cols
    loss = np.empty(24)
    for k, g in enumerate(d["brain_feature_groups"]):
        keep = cols.copy()
        keep[g] = 0.0
        loss[k] = float(np.sum((y - xs @ keep) ** 2))
    assert np.allclose(loss / loss.sum(), p / p.sum())


def test_realized_correlation_matches_every_grid_level() -> None:
    rng = np.random.default_rng(11)
    d = _design(rng, n_feat=256, n_rows=64, lags=2)
    nb = power_oracle.group_signal_norms(d["brain_x"], d["brain_feature_groups"])
    nl = power_oracle.group_signal_norms(d["lm_x"], d["lm_feature_groups"])
    for target in (0.0, 0.2, 0.4, 0.6, 0.8):
        r = np.random.default_rng(int(target * 100))
        p, q = power_oracle.make_ablation_oracle_profiles(256, target, r)
        lb = power_oracle.noiseless_ablation_loss(power_oracle.oracle_coefficients(p, nb, np.ones(256)), nb)
        ll = power_oracle.noiseless_ablation_loss(power_oracle.oracle_coefficients(q, nl, np.ones(256)), nl)
        assert abs(power.profile_correlation(lb, ll, method="spearman") - target) < 0.01


def test_signed_profiles_do_not_prescribe_the_ablation_ordering() -> None:
    rng = np.random.default_rng(5)
    d = _design(rng, n_feat=256, n_rows=64, lags=2)
    nb = power_oracle.group_signal_norms(d["brain_x"], d["brain_feature_groups"])
    nl = power_oracle.group_signal_norms(d["lm_x"], d["lm_feature_groups"])
    values = []
    for seed in range(20):
        a, b = power.make_reliance_profiles(256, 0.0, np.random.default_rng(seed), correlation_method="spearman")
        values.append(power.profile_correlation(a**2 * nb, b**2 * nl, method="spearman"))
    assert float(np.mean(values)) > 0.5


def test_silent_groups_get_zero_loss_and_the_realized_value_is_recorded() -> None:
    rng = np.random.default_rng(23)
    d = _design(rng)
    for g in d["brain_feature_groups"][:2]:
        d["brain_x"][:, g] = 0.0
    d["lm_x"][:, :2] = 0.0
    case = power_oracle.simulate_case(
        d, oracle_correlation=0.6, residual_scale_multiplier=1.0, alpha_grid=[0.1, 10.0], seed=9, solver="solve"
    )
    assert case["silent_feature_groups"] == 2
    assert 0.6 < case["realized_oracle_spearman"] < 0.75
    assert np.isfinite(case["observed_reliance_spearman"])


def test_shard_aggregate_and_summary_roundtrip() -> None:
    rng = np.random.default_rng(17)
    d = _design(rng)
    config = {
        "designs": ["gemma_2_2b"],
        "bank_types": ["source_story_variance"],
        "directions": ["shapesphysical_to_shapessocial"],
        "oracle_correlations": [0.0, 0.4],
        "residual_scale_multipliers": [1.0],
        "null_replicates": 4,
        "positive_replicates_per_grid_cell": 3,
        "identifiability_power_threshold": 0.8,
        "seed": 5,
        "alpha_grid": [0.1, 10.0],
        "purge_adjacent_trs": 1,
    }
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        paths = []
        for i in range(2):
            out = root / f"s{i}.json"
            power_oracle.run_shard(d, config, output_path=out, shard_index=i, shard_count=2)
            paths.append(out)
        agg = power_oracle.aggregate(d, config, paths, output_path=root / "agg.json")
        summary = power_oracle.summarize([root / "agg.json"])
    assert agg["oracle_definition"] == "ablation_loss"
    rows = {r["oracle_correlation"]: r for r in agg["rows"]}
    assert abs(rows[0.4]["realized_oracle_mean"] - 0.4) < 0.05
    assert rows[0.0]["replicates"] == 4 and rows[0.4]["replicates"] == 3
    assert summary["grid"] == {"designs": 1, "cells": 2, "positive_cells": 1, "shards": 2, "residual_scale_multipliers": [1.0]}


def test_released_oracle_summary_matches_aggregates_and_catalog() -> None:
    summary = json.loads((ROOT / "results/power_oracle_summary.json").read_text())
    assert summary["grid"] == {
        "designs": 8,
        "cells": 80,
        "positive_cells": 64,
        "shards": 192,
        "residual_scale_multipliers": [1.0, 2.0],
    }
    assert summary["detection"]["cells_at_or_above_0.80"] == 0
    assert summary["detection"]["maximum"] < 0.11
    catalog = {}
    for line in (ROOT / "checksums/power_oracle_aggregates.sha256").read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        catalog[filename.strip()] = digest
    assert catalog == {item["aggregate_file"]: item["aggregate_sha256"] for item in summary["source_aggregates"]}
    for filename, digest in catalog.items():
        assert hashlib.sha256((ROOT / "evidence/power_oracle" / filename).read_bytes()).hexdigest() == digest
    assert len(list((ROOT / "evidence/power_oracle/shard_hashes").glob("*.sha256"))) == 8
