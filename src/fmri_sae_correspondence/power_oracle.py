"""Semi-synthetic calibration with the planted correspondence defined in the ablation-loss metric.

The configured power stage (:mod:`.power`) plants two signed, centred rank-score
profiles with a prescribed Spearman correlation and uses them directly as ridge
coefficients.  The statistic the pipeline scores is a held-out ablation loss,
and in a linear mapping the noiseless ablation loss of feature group ``k`` is
``beta_k**2 * ||sum_{j in G_k} x~_j||**2``: it depends on coefficient magnitude,
not sign.  Prescribing the correlation of signed coefficient profiles therefore
does not prescribe the correlation of the ablation-loss orderings the test
measures; on that generator the noiseless ablation-loss correlation is about
.76 at nominal ``rho = 0`` for 256 features.  The detection fractions produced
by that stage are superseded by this module.

This module reuses the configured stage unchanged for every other step: design
artifacts, folds, alpha grid, seed contract, residual resampling, signal
scaling in residual-SD units, ridge solver, held-out grouped ablation, the
Spearman statistic, and the 95th-percentile detection rule.  One thing
changes.  For each system with standardized design columns ``x~`` (the same
centring and scaling ``inject_targets`` applies), let
``c_k = ||sum_{j in G_k} x~_j||**2``.  Two positive rank-score profiles ``p``
and ``q`` with the prescribed Spearman correlation are drawn, and the
coefficients are set to ``beta_k = s_k * sqrt(p_k / c_k)`` (brain) and
``s'_k * sqrt(q_k / c_k)`` (LM), with sign vectors ``s`` and ``s'`` drawn
independently per system.  The noiseless ablation-loss profile of each planted
mapping then equals ``p`` or ``q`` exactly, so the planted ablation-ordering
correlation equals the prescribed value up to the generator's finite-rank
quantization.  The realized correlation of the two planted loss profiles is
computed from the design geometry and recorded for every replicate.  Signs do
not enter the noiseless loss; they are drawn independently because a sign
vector shared by both systems induces between-system agreement in the fitted
losses through a signed cross-term.

Feature groups that are silent on the held-out description have zero signal
norm in both systems; they receive coefficient zero and tie at the bottom of
both planted profiles, and the recorded realized correlation accounts for the
ties.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

try:
    from scipy.stats import rankdata
except ImportError:  # pragma: no cover
    rankdata = None  # type: ignore[assignment]

from . import power
from .config import default_config_path, load_config
from .feature_banks import sha256_file, write_json

ORACLE_DEFINITION = "ablation_loss"
SHARD_ARTIFACT_KIND = "power_shard_ablation_oracle"
SHARD_SCHEMA_VERSION = 3
ORACLE_CONSTRUCTION = (
    "positive rank-score ablation-loss targets; coefficients sqrt(target/group_signal_norm) "
    "with independent random sign vectors per system"
)


def group_signal_norms(x: np.ndarray, groups: Sequence[Sequence[int]]) -> np.ndarray:
    """Return ``c_k = ||sum_{j in G_k} x~_j||^2`` with the standardization ``inject_targets`` uses."""
    x = np.asarray(x, dtype=float)
    centered = x - np.mean(x, axis=0, keepdims=True)
    scale = np.std(centered, axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    standardized = centered / scale
    norms = np.empty(len(groups), dtype=float)
    for index, group in enumerate(groups):
        summed = standardized[:, np.asarray(group, dtype=int)].sum(axis=1)
        norms[index] = float(summed @ summed)
    if not np.all(np.isfinite(norms)) or np.any(norms < 0):
        raise ValueError("feature-group signal norms must be finite and non-negative")
    if np.sum(norms > 0) < max(3, len(norms) // 2):
        raise ValueError("fewer than half of the feature groups carry signal on this design")
    return norms


def make_ablation_oracle_profiles(
    n_features: int, oracle_correlation: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Positive ablation-loss target profiles with the prescribed Spearman correlation."""
    if rankdata is None:
        raise RuntimeError("scipy is required")
    a, b = power.make_reliance_profiles(
        int(n_features), float(oracle_correlation), rng, correlation_method="spearman"
    )
    return rankdata(a, method="ordinal") / float(n_features), rankdata(b, method="ordinal") / float(n_features)


def oracle_coefficients(loss_profile: np.ndarray, norms: np.ndarray, signs: np.ndarray) -> np.ndarray:
    """Coefficients whose noiseless ablation loss equals ``loss_profile`` up to a constant."""
    loss_profile = np.asarray(loss_profile, dtype=float)
    norms = np.asarray(norms, dtype=float)
    if np.any(loss_profile < 0):
        raise ValueError("ablation-loss targets must be non-negative")
    active = norms > 0
    if np.any(loss_profile[~active] != 0):
        raise ValueError("a silent feature group cannot carry a non-zero ablation-loss target")
    out = np.zeros(len(norms), dtype=float)
    out[active] = np.asarray(signs, dtype=float)[active] * np.sqrt(loss_profile[active] / norms[active])
    return out


def noiseless_ablation_loss(coefficients: np.ndarray, norms: np.ndarray) -> np.ndarray:
    """Exact noiseless grouped ablation loss, up to the target normalization, of a planted mapping."""
    return np.asarray(coefficients, dtype=float) ** 2 * np.asarray(norms, dtype=float)


def simulate_case(
    design: Mapping[str, Any],
    *,
    oracle_correlation: float,
    residual_scale_multiplier: float,
    alpha_grid: Sequence[float],
    seed: int,
    x_caches: Mapping[str, Mapping[str, Any]] | None = None,
    solver: str = "cholesky",
    norms: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """One paired case with the planted correspondence defined in the ablation-loss metric."""
    alpha_values = [float(value) for value in alpha_grid]
    if not alpha_values or any(value <= 0 for value in alpha_values):
        raise ValueError("power simulation requires a positive alpha grid")
    rng = np.random.default_rng(int(seed))
    brain_groups = design["brain_feature_groups"]
    lm_groups = design["lm_feature_groups"]
    n_features = len(brain_groups)
    if len(lm_groups) != n_features:
        raise ValueError("brain and LM designs must share the feature grouping")
    if norms is None:
        norms = {
            "brain": group_signal_norms(design["brain_x"], brain_groups),
            "lm": group_signal_norms(design["lm_x"], lm_groups),
        }
    active = (norms["brain"] > 0) & (norms["lm"] > 0)
    n_active = int(np.sum(active))
    p_active, q_active = make_ablation_oracle_profiles(n_active, float(oracle_correlation), rng)
    loss_brain = np.zeros(n_features, dtype=float)
    loss_lm = np.zeros(n_features, dtype=float)
    loss_brain[active] = p_active
    loss_lm[active] = q_active
    signs_brain = rng.choice(np.asarray([-1.0, 1.0]), size=n_features)
    signs_lm = rng.choice(np.asarray([-1.0, 1.0]), size=n_features)
    beta_brain = oracle_coefficients(loss_brain, norms["brain"], signs_brain)
    beta_lm = oracle_coefficients(loss_lm, norms["lm"], signs_lm)
    realized = power.profile_correlation(
        noiseless_ablation_loss(beta_brain, norms["brain"]),
        noiseless_ablation_loss(beta_lm, norms["lm"]),
        method="spearman",
    )
    brain_columns = np.zeros(design["brain_x"].shape[1], dtype=float)
    lm_columns = np.zeros(design["lm_x"].shape[1], dtype=float)
    for value_brain, value_lm, brain_group, lm_group in zip(beta_brain, beta_lm, brain_groups, lm_groups, strict=False):
        brain_columns[np.asarray(brain_group, dtype=int)] = float(value_brain)
        lm_columns[np.asarray(lm_group, dtype=int)] = float(value_lm)

    residual_brain = np.asarray(design["brain_residual"], dtype=float)
    residual_lm = np.asarray(design["lm_residual"], dtype=float)
    brain_residual_sd = power._mean_empirical_residual_sd(residual_brain)
    lm_residual_sd = power._mean_empirical_residual_sd(residual_lm)
    brain_target = power.inject_targets(
        design["brain_x"],
        power._noise_from_residual(residual_brain, rng, len(design["brain_x"]), residual_brain.shape[1]),
        brain_columns,
        signal_scale=float(residual_scale_multiplier) * max(brain_residual_sd, 1e-12),
        rng=rng,
    )
    lm_target = power.inject_targets(
        design["lm_x"],
        power._noise_from_residual(residual_lm, rng, len(design["lm_x"]), residual_lm.shape[1]),
        lm_columns,
        signal_scale=float(residual_scale_multiplier) * max(lm_residual_sd, 1e-12),
        rng=rng,
    )
    brain_censor = np.asarray(design["brain_censor_mask"], dtype=bool)
    lm_censor = np.asarray(design["lm_censor_mask"], dtype=bool)
    if np.any(brain_censor):
        brain_target[brain_censor] = np.nan
    if np.any(lm_censor):
        lm_target[lm_censor] = np.nan
    brain_finite = np.all(np.isfinite(brain_target), axis=1)
    lm_finite = np.all(np.isfinite(lm_target), axis=1)
    if not np.array_equal(brain_finite, ~brain_censor) or not np.array_equal(lm_finite, ~lm_censor):
        raise ValueError("synthetic target finiteness changed; refusing cached fold masks")
    brain_cache = x_caches.get("brain") if x_caches else None
    lm_cache = x_caches.get("lm") if x_caches else None
    purge = int(design.get("purge_adjacent_trs", 4))
    brain_alphas = power.select_alpha_by_fold(
        design["brain_x"], brain_target, design["brain_folds"], alpha_values, purge_adjacent=purge,
        finite_mask=brain_finite if brain_cache is not None else None,
        x_cache=brain_cache["selection"] if brain_cache is not None else None, solver=solver,
    )
    lm_alphas = power.select_alpha_by_fold(
        design["lm_x"], lm_target, design["lm_folds"], alpha_values, purge_adjacent=purge,
        finite_mask=lm_finite if lm_cache is not None else None,
        x_cache=lm_cache["selection"] if lm_cache is not None else None, solver=solver,
    )
    brain_fit = power.heldout_grouped_reliance(
        design["brain_x"], brain_target, design["brain_folds"], alpha=brain_alphas,
        feature_groups=brain_groups, finite_mask=brain_finite if brain_cache is not None else None,
        x_cache=brain_cache["heldout"] if brain_cache is not None else None, solver=solver,
    )
    lm_fit = power.heldout_grouped_reliance(
        design["lm_x"], lm_target, design["lm_folds"], alpha=lm_alphas,
        feature_groups=lm_groups, finite_mask=lm_finite if lm_cache is not None else None,
        x_cache=lm_cache["heldout"] if lm_cache is not None else None, solver=solver,
    )
    observed = power.profile_correlation(
        brain_fit["reliance_profile"], lm_fit["reliance_profile"], method="spearman"
    )
    return {
        "oracle_correlation": float(oracle_correlation),
        "oracle_definition": ORACLE_DEFINITION,
        "oracle_correlation_method": "spearman",
        "realized_oracle_spearman": float(realized),
        "silent_feature_groups": int(n_features - n_active),
        "residual_scale_multiplier": float(residual_scale_multiplier),
        "brain_empirical_residual_sd": float(brain_residual_sd),
        "lm_empirical_residual_sd": float(lm_residual_sd),
        "observed_reliance_spearman": float(observed),
        "brain_reliance_profile": np.asarray(brain_fit["reliance_profile"], dtype=np.float32),
        "lm_reliance_profile": np.asarray(lm_fit["reliance_profile"], dtype=np.float32),
        "brain_mean_r2_oos": float(np.nanmean([row["mean_r2_oos"] for row in brain_fit["fold_scores"]])),
        "lm_mean_r2_oos": float(np.nanmean([row["mean_r2_oos"] for row in lm_fit["fold_scores"]])),
        "brain_alphas": [float(value) for value in brain_alphas],
        "lm_alphas": [float(value) for value in lm_alphas],
    }


def run_shard(
    design: Mapping[str, Any],
    power_spec: Mapping[str, Any],
    *,
    output_path: Path,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    """One deterministic, resumable shard with the configured stage's coordinates and seeds."""
    design = power._validate_paired_design(design)
    power._validate_power_coordinates(design, power_spec)
    if int(shard_count) < 1 or not 0 <= int(shard_index) < int(shard_count):
        raise ValueError("shard_index must be in [0, shard_count)")
    alpha_grid = [float(value) for value in power_spec.get("alpha_grid", ())]
    if not alpha_grid:
        raise ValueError("full power requires a non-empty alpha_grid")
    base_seed = int(power_spec.get("seed", 2026082704))
    purge = int(design.get("purge_adjacent_trs", 4))
    x_caches = {
        "brain": power._prepare_x_cache(
            design["brain_x"], design["brain_folds"], ~np.asarray(design["brain_censor_mask"], dtype=bool),
            purge_adjacent=purge, alpha_grid=alpha_grid,
        ),
        "lm": power._prepare_x_cache(
            design["lm_x"], design["lm_folds"], ~np.asarray(design["lm_censor_mask"], dtype=bool),
            purge_adjacent=purge, alpha_grid=alpha_grid,
        ),
    }
    norms = {
        "brain": group_signal_norms(design["brain_x"], design["brain_feature_groups"]),
        "lm": group_signal_norms(design["lm_x"], design["lm_feature_groups"]),
    }
    silent = int(np.sum(~((norms["brain"] > 0) & (norms["lm"] > 0))))
    rows: list[dict[str, Any]] = []
    for rho, scale, replicate_count in power._power_grid_coordinates(power_spec):
        replicate_ids = list(range(int(shard_index), int(replicate_count), int(shard_count)))
        values: list[float] = []
        realized: list[float] = []
        for replicate in replicate_ids:
            case = simulate_case(
                design,
                oracle_correlation=rho,
                residual_scale_multiplier=scale,
                alpha_grid=alpha_grid,
                seed=power._case_seed(base_seed, design, rho, scale, replicate),
                x_caches=x_caches,
                solver="cholesky",
                norms=norms,
            )
            values.append(float(case["observed_reliance_spearman"]))
            realized.append(float(case["realized_oracle_spearman"]))
        rows.append(
            {
                "oracle_correlation": float(rho),
                "residual_scale_multiplier": float(scale),
                "replicates_total": int(replicate_count),
                "replicate_ids": replicate_ids,
                "replicates_in_shard": int(len(replicate_ids)),
                "values": values,
                "realized_oracle_spearman": realized,
                "silent_feature_groups": silent,
            }
        )
    result = {
        "schema_version": SHARD_SCHEMA_VERSION,
        "artifact_kind": SHARD_ARTIFACT_KIND,
        "scientific_evidence": False,
        "design_schema": "paired_target_design_v1",
        "design_label": str(design["design_label"]),
        "design_sha256": str(design.get("sha256", design.get("design_sha256"))),
        "model": str(design["model_key"]),
        "bank_type": str(design["bank_type"]),
        "bank_variant": str(design.get("bank_variant", design["bank_type"])),
        "direction": str(design["direction"]),
        "source_story": str(design.get("source_story", "")),
        "target_story": str(design.get("target_story", "")),
        "shard_index": int(shard_index),
        "shard_count": int(shard_count),
        "seed_contract": "sha256(base_seed, design_sha256, cell coordinates, replicate id)",
        "oracle_correlation_method": "spearman",
        "oracle_definition": ORACLE_DEFINITION,
        "oracle_construction": ORACLE_CONSTRUCTION,
        "residual_scale_definition": "mean per-target population SD of the corresponding empirical residual pool",
        "brain_empirical_residual_sd": power._mean_empirical_residual_sd(design["brain_residual"]),
        "lm_empirical_residual_sd": power._mean_empirical_residual_sd(design["lm_residual"]),
        "alpha_grid": alpha_grid,
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "frozen_module_sha256": sha256_file(Path(power.__file__).resolve()),
        "rows": rows,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    return result


def aggregate(
    design: Mapping[str, Any],
    power_spec: Mapping[str, Any],
    shard_paths: Sequence[Path],
    *,
    output_path: Path,
) -> dict[str, Any]:
    """Aggregate shards with the configured aggregator, then attach the realized planted correlation."""
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in shard_paths]
    for record in records:
        if record.get("artifact_kind") != SHARD_ARTIFACT_KIND or record.get("oracle_definition") != ORACLE_DEFINITION:
            raise ValueError(f"not an ablation-oracle shard: {record.get('artifact_kind')}")
    output = power.aggregate_power_shards(design, power_spec, [Path(p) for p in shard_paths], output_path=output_path)
    realized: dict[tuple[float, float], dict[int, float]] = {}
    for record in records:
        for row in record["rows"]:
            key = (float(row["oracle_correlation"]), float(row["residual_scale_multiplier"]))
            target = realized.setdefault(key, {})
            for replicate, value in zip(row["replicate_ids"], row["realized_oracle_spearman"], strict=False):
                target[int(replicate)] = float(value)
    for row in output["rows"]:
        key = (float(row["oracle_correlation"]), float(row["residual_scale_multiplier"]))
        values = np.asarray([realized[key][i] for i in sorted(realized[key])], dtype=float)
        row["realized_oracle_mean"] = float(np.mean(values))
        row["realized_oracle_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else None
    output["schema_version"] = SHARD_SCHEMA_VERSION
    output["analysis_role"] = "semi_synthetic_power_ablation_oracle"
    output["oracle_definition"] = ORACLE_DEFINITION
    output["oracle_construction"] = ORACLE_CONSTRUCTION
    output["code_sha256"] = records[0].get("code_sha256")
    output["frozen_module_sha256"] = records[0].get("frozen_module_sha256")
    output["decision_rule"] = (
        "detect if paired held-out reliance Spearman exceeds the 95th percentile of the same-design "
        "rho=0 distribution, where rho is the noiseless ablation-loss correlation of the planted mappings; "
        "identifiable if detection fraction >= configured threshold"
    )
    write_json(Path(output_path), output)
    return output


def summarize(aggregate_paths: Sequence[Path]) -> dict[str, Any]:
    """Compact machine-readable summary of a set of ablation-oracle aggregates."""
    records = [json.loads(Path(p).read_text(encoding="utf-8")) for p in aggregate_paths]
    cells = []
    for record, path in zip(records, aggregate_paths, strict=False):
        if record.get("oracle_definition") != ORACLE_DEFINITION:
            raise ValueError(f"{path}: not an ablation-oracle aggregate")
        for row in record["rows"]:
            cells.append(
                {
                    "model": record["model"], "bank_type": record["bank_type"], "direction": record["direction"],
                    "residual_scale_multiplier": row["residual_scale_multiplier"],
                    "oracle_correlation": row["oracle_correlation"],
                    "realized_oracle_mean": row["realized_oracle_mean"],
                    "observed_mean": row["observed_mean"], "observed_sd": row["observed_sd"],
                    "null_95_threshold": row["null_95_threshold"], "detection_fraction": row["detection_fraction"],
                }
            )
    positive = [c for c in cells if c["oracle_correlation"] > 0]
    by_level: dict[str, dict[str, float]] = {}
    for rho in sorted({c["oracle_correlation"] for c in positive}):
        level = [c for c in positive if c["oracle_correlation"] == rho]
        by_level[f"{rho:.1f}"] = {
            "realized_oracle_min": min(c["realized_oracle_mean"] for c in level),
            "realized_oracle_max": max(c["realized_oracle_mean"] for c in level),
            "detection_min": min(c["detection_fraction"] for c in level),
            "detection_max": max(c["detection_fraction"] for c in level),
        }
    return {
        "analysis_role": "semi_synthetic_power_ablation_oracle_summary",
        "oracle_definition": ORACLE_DEFINITION,
        "oracle_construction": ORACLE_CONSTRUCTION,
        "source_aggregates": [
            {"aggregate_file": Path(p).name, "aggregate_sha256": sha256_file(Path(p))} for p in aggregate_paths
        ],
        "grid": {
            "designs": len(records),
            "cells": len(cells),
            "positive_cells": len(positive),
            "shards": sum(int(r["shard_count"]) for r in records),
            "residual_scale_multipliers": sorted({c["residual_scale_multiplier"] for c in cells}),
        },
        "detection": {
            "minimum": min(c["detection_fraction"] for c in positive),
            "maximum": max(c["detection_fraction"] for c in positive),
            "mean": float(np.mean([c["detection_fraction"] for c in positive])),
            "cells_at_or_above_0.80": sum(c["detection_fraction"] >= 0.8 for c in positive),
            "null_threshold_min": min(c["null_95_threshold"] for c in cells),
            "null_threshold_max": max(c["null_95_threshold"] for c in cells),
        },
        "by_oracle_correlation": by_level,
        "claim_boundary": {
            "allowed": "The test as configured does not recover a planted ablation-loss correspondence at any tested level, so its null is uninformative.",
            "forbidden": "The data establish that sparse-feature correspondence is present or absent.",
        },
        "cells": cells,
    }


def _power_config(spec: Mapping[str, Any], scales: Sequence[float] | None) -> dict[str, Any]:
    power_config = dict(spec["power"])
    power_config["alpha_grid"] = list(spec["cross_validation"]["alpha_grid"])
    power_config["purge_adjacent_trs"] = int(spec["cross_validation"]["purge_adjacent_trs"])
    power_config.setdefault("seed", int(spec["nulls"]["seeds"]["power"]))
    if scales:
        configured = [float(v) for v in power_config["residual_scale_multipliers"]]
        wanted = [float(v) for v in scales]
        unknown = [v for v in wanted if not any(abs(v - c) < 1e-9 for c in configured)]
        if unknown:
            raise ValueError(f"residual scales {unknown} are not in the configured grid {configured}")
        if wanted != configured:
            power_config["residual_scale_multipliers"] = wanted
            power_config["scales_restricted_from"] = configured
    return power_config


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ablation-loss oracle calibration: shard, aggregate, or summarize")
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--design", type=Path, default=None)
    parser.add_argument("--scales", type=str, default=None, help="comma-separated subset of the configured residual scales")
    parser.add_argument("--shard-index", type=int, default=None)
    parser.add_argument("--shard-count", type=int, default=None)
    parser.add_argument("--shard-output", type=Path, default=None)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--shard-paths", nargs="+", type=Path, default=None)
    parser.add_argument("--aggregate-output", type=Path, default=None)
    parser.add_argument("--summarize", nargs="+", type=Path, default=None, help="aggregate files to summarize")
    parser.add_argument("--summary-output", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.summarize:
        summary = summarize(args.summarize)
        if args.summary_output is not None:
            write_json(args.summary_output, summary)
        print(json.dumps({k: v for k, v in summary.items() if k != "cells"}, indent=1, sort_keys=True))
        return 0
    if args.design is None:
        parser.error("--design is required for shard and aggregate modes")
    spec = load_config(args.config)
    scales = [float(v) for v in args.scales.split(",")] if args.scales else None
    power_config = _power_config(spec, scales)
    design = power._load_design(args.design.resolve())
    if args.aggregate:
        if not args.shard_paths or args.aggregate_output is None:
            parser.error("--aggregate requires --shard-paths and --aggregate-output")
        result = aggregate(design, power_config, args.shard_paths, output_path=args.aggregate_output)
        print(json.dumps({"status": "aggregated", "output": str(args.aggregate_output), "rows": len(result["rows"])}))
        return 0
    if args.shard_index is None or args.shard_count is None or args.shard_output is None:
        parser.error("shard mode requires --shard-index, --shard-count, --shard-output")
    result = run_shard(
        design, power_config, output_path=args.shard_output, shard_index=args.shard_index, shard_count=args.shard_count
    )
    print(json.dumps({"status": "shard_complete", "output": str(args.shard_output), "rows": len(result["rows"])}))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
