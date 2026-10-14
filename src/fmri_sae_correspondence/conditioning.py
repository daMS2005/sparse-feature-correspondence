#!/usr/bin/env python3
"""Feature-bank coverage and conditioning diagnostics for analysis.

The diagnostics in this module are predictor-side diagnostics.  Source-story
statistics are computed only on the rows used to construct a bank (all source
rows for a transfer bank, or the purged outer-training rows for a within-story
bank).  Rows from the target story or held-out block are never used to choose a
feature.  They are reported separately as distribution-shift diagnostics.

The module intentionally keeps the full per-feature summaries in a compact
JSON artifact.  This makes a result auditable without requiring a reader to
reconstruct the selected columns from a model-specific NPZ file.  Pairwise
correlations and SVD summaries are stored as aggregate statistics because the
full matrices are redundant and can be very large.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata

from .config import default_config_path, load_config
from .feature_banks import sha256_file, write_json
from .integrity import provenance_fingerprint, resolve_data_root, resolve_output_root

HERE = Path(__file__).resolve().parent


def _resolve_data_root(config_path: Path, spec: Mapping[str, Any]) -> Path:
    del config_path
    return resolve_data_root(spec=spec)


def _source_bundle_path(
    data_root: Path,
    spec: Mapping[str, Any],
    model_key: str,
    story: str,
) -> Path:
    relative = str(spec["paths"]["predictor_pattern"]).format(
        model=model_key,
        story=story,
    )
    return data_root / relative


def _bank_path(output_root: Path, model_key: str, source: str, target: str, *, fold: int | None = None) -> Path:
    if fold is None:
        return output_root / "feature_banks" / "cross_story" / model_key / f"{source}_to_{target}" / "banks.json"
    return output_root / "feature_banks" / "within_story" / model_key / source / f"fold_{int(fold):02d}" / "banks.json"


def _resolve_bank_root(config_path: Path, spec: Mapping[str, Any]) -> Path:
    del config_path
    return (
        resolve_output_root(spec=spec)
        / str(spec["paths"]["artifact_root"])
        / str(spec["paths"]["feature_bank_root"])
    ).resolve()


def _load_source_bundle(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"missing all-feature predictor bundle: {path}")
    with np.load(path, allow_pickle=True) as z:
        required = {"all_sae_tr_means", "selected_layers", "tr_indices"}
        missing = sorted(required.difference(z.files))
        if missing:
            raise ValueError(f"{path}: missing predictor fields {missing}")
        values = np.asarray(z["all_sae_tr_means"], dtype=np.float64)
        layers = np.asarray(z["selected_layers"], dtype=np.int64)
        trs = np.asarray(z["tr_indices"], dtype=np.int64)
        model_id = str(np.asarray(z["model_id"]).reshape(-1)[0]) if "model_id" in z else ""
        story = str(np.asarray(z["story"]).reshape(-1)[0]) if "story" in z else ""
    if values.ndim != 3 or values.shape[0] != len(layers) or values.shape[1] != len(trs):
        raise ValueError(f"{path}: all_sae_tr_means must be [layer,time,feature] with matching metadata")
    return {
        "all_sae_tr_means": values,
        "selected_layers": layers,
        "tr_indices": trs,
        "model_id": model_id,
        "story": story,
        "path": path,
        "sha256": sha256_file(path),
    }


def _entries_for_bank(bank_record: Mapping[str, Any], bank_name: str, size: int) -> tuple[list[int], list[int], list[dict[str, Any]]]:
    """Return selected layer IDs, feature IDs, and entry metadata."""
    if bank_name.startswith("source_story_correlation_pruned@"):
        threshold = bank_name.split("@", 1)[1]
        pruned_records = bank_record["correlation_pruned"]
        # JSON encoders preserve 0.7 as ``"0.7"`` whereas the configured
        # configuration spells it ``0.70``.  Resolve by numeric equality so
        # the artifact remains insensitive to that harmless representation.
        if threshold not in pruned_records:
            matches = [key for key in pruned_records if np.isclose(float(key), float(threshold))]
            if len(matches) != 1:
                raise KeyError(f"correlation-pruned threshold {threshold} unavailable; keys={sorted(pruned_records)}")
            threshold = matches[0]
        source_bank = pruned_records[threshold]
    else:
        source_bank = bank_record[bank_name]
    selected = source_bank["banks"][str(int(size))]
    return (
        [int(v) for v in selected["layers"]],
        [int(v) for v in selected["feature_ids"]],
        list(selected.get("entries", [])),
    )


def _extract_selected(all_sae: np.ndarray, available_layers: Sequence[int], layers: Sequence[int], feature_ids: Sequence[int]) -> np.ndarray:
    offsets = {int(layer): idx for idx, layer in enumerate(available_layers)}
    if len(layers) != len(feature_ids):
        raise ValueError("bank layer and feature arrays have different lengths")
    cols: list[np.ndarray] = []
    for layer, feature in zip(layers, feature_ids, strict=False):
        if int(layer) not in offsets:
            raise KeyError(f"selected layer {layer} absent from extraction")
        if feature < 0 or feature >= all_sae.shape[2]:
            raise IndexError(f"feature ID {feature} outside dictionary width {all_sae.shape[2]}")
        cols.append(np.asarray(all_sae[offsets[int(layer)], :, int(feature)], dtype=np.float64))
    return np.column_stack(cols) if cols else np.empty((all_sae.shape[1], 0), dtype=np.float64)


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must have shape [time, feature]")
    if not np.all(np.isfinite(values)):
        # A predictor-side NaN would make the diagnostic ambiguous.  Do not
        # silently impute it; the caller receives the exact failure.
        raise ValueError("predictor values contain non-finite values")
    return values


def _column_stats(values: np.ndarray) -> dict[str, Any]:
    values = _finite(values)
    n_rows, n_features = values.shape
    active = values > 0
    counts = active.sum(axis=0, dtype=np.int64)
    prevalence = counts.astype(np.float64) / max(n_rows, 1)
    binary_variance = prevalence * (1.0 - prevalence)
    positive_sq = np.sum(np.square(np.where(active, values, 0.0)), axis=0)
    positive_rms = np.sqrt(positive_sq / np.maximum(counts, 1))
    continuous_mean = np.mean(values, axis=0) if n_rows else np.full(n_features, np.nan)
    continuous_variance = np.var(values, axis=0) if n_rows else np.full(n_features, np.nan)
    return {
        "n_rows": int(n_rows),
        "n_features": int(n_features),
        "active_count": counts,
        "active_fraction": prevalence,
        "zero_fraction": 1.0 - prevalence,
        "binary_temporal_variance": binary_variance,
        "activation_mean": continuous_mean,
        "activation_variance": continuous_variance,
        "positive_activation_rms": positive_rms,
        "features_active_at_most_2_trs_count": int(np.sum(counts <= 2)),
        "features_active_at_most_5_trs_count": int(np.sum(counts <= 5)),
        "features_active_at_most_10_trs_count": int(np.sum(counts <= 10)),
        "features_active_at_most_2_trs_fraction": float(np.mean(counts <= 2)) if n_features else np.nan,
        "features_active_at_most_5_trs_fraction": float(np.mean(counts <= 5)) if n_features else np.nan,
        "features_active_at_most_10_trs_fraction": float(np.mean(counts <= 10)) if n_features else np.nan,
    }


def _spearman_abs_summary(values: np.ndarray) -> dict[str, Any]:
    values = _finite(values)
    n_rows, n_features = values.shape
    if n_features < 2:
        return {
            "n_pairs": 0,
            "quantiles": {"p50": np.nan, "p95": np.nan, "p99": np.nan, "max": np.nan},
            "fractions_at_least": {"0.50": np.nan, "0.80": np.nan, "0.90": np.nan, "0.99": np.nan},
            "constant_columns": int(np.sum(np.std(values, axis=0) == 0)) if n_features else 0,
        }
    ranks = np.column_stack(
        [rankdata(values[:, idx], method="average") for idx in range(n_features)]
    ).astype(np.float64)
    ranks -= ranks.mean(axis=0, keepdims=True)
    denom = np.sqrt(np.sum(ranks * ranks, axis=0))
    valid = denom > 0
    normalized = np.zeros_like(ranks)
    normalized[:, valid] = ranks[:, valid] / denom[valid]
    corr = np.abs(np.clip(normalized.T @ normalized, -1.0, 1.0))
    upper = corr[np.triu_indices(n_features, k=1)]
    thresholds = (0.50, 0.80, 0.90, 0.99)
    return {
        "n_pairs": int(len(upper)),
        "quantiles": {
            "p50": float(np.quantile(upper, 0.50)),
            "p95": float(np.quantile(upper, 0.95)),
            "p99": float(np.quantile(upper, 0.99)),
            "max": float(np.max(upper)),
        },
        "fractions_at_least": {f"{threshold:.2f}": float(np.mean(upper >= threshold)) for threshold in thresholds},
        "constant_columns": int(np.sum(~valid)),
    }


def _rank_summary(values: np.ndarray) -> dict[str, Any]:
    values = _finite(values)
    n_rows, n_features = values.shape
    if n_features == 0 or n_rows == 0:
        return {
            "numerical_rank": 0,
            "effective_rank": 0.0,
            "stable_rank": 0.0,
            "condition_number": np.inf,
            "singular_value_max": 0.0,
            "singular_value_min_nonzero": np.nan,
            "singular_values_count": 0,
        }
    centered = values - values.mean(axis=0, keepdims=True)
    scale = centered.std(axis=0, ddof=0)
    valid = scale > 0
    if not np.any(valid):
        return {
            "numerical_rank": 0,
            "effective_rank": 0.0,
            "stable_rank": 0.0,
            "condition_number": np.inf,
            "singular_value_max": 0.0,
            "singular_value_min_nonzero": np.nan,
            "singular_values_count": int(min(n_rows, n_features)),
        }
    standardized = centered[:, valid] / scale[valid]
    singular = np.linalg.svd(standardized, full_matrices=False, compute_uv=False)
    positive = singular[singular > (singular[0] * max(standardized.shape) * np.finfo(float).eps)] if len(singular) else singular
    if not len(positive):
        return {
            "numerical_rank": 0,
            "effective_rank": 0.0,
            "stable_rank": 0.0,
            "condition_number": np.inf,
            "singular_value_max": 0.0,
            "singular_value_min_nonzero": np.nan,
            "singular_values_count": int(len(singular)),
        }
    probabilities = np.square(positive) / np.sum(np.square(positive))
    entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, np.finfo(float).tiny)))
    return {
        "numerical_rank": int(len(positive)),
        "effective_rank": float(np.exp(entropy)),
        "stable_rank": float(np.sum(np.square(positive)) / np.square(positive[0])),
        "condition_number": float(positive[0] / positive[-1]),
        "singular_value_max": float(positive[0]),
        "singular_value_min_nonzero": float(positive[-1]),
        "singular_values_count": int(len(singular)),
    }


def summarize_values(values: np.ndarray, *, include_profiles: bool = True) -> dict[str, Any]:
    """Return all requested coverage and conditioning summaries for a bank."""
    values = _finite(values)
    columns = _column_stats(values)
    pairwise = _spearman_abs_summary(values)
    ranks = _rank_summary(values)
    result: dict[str, Any] = {
        "n_rows": columns["n_rows"],
        "n_features": columns["n_features"],
        "coverage": {
            key: columns[key]
            for key in (
                "features_active_at_most_2_trs_count",
                "features_active_at_most_5_trs_count",
                "features_active_at_most_10_trs_count",
                "features_active_at_most_2_trs_fraction",
                "features_active_at_most_5_trs_fraction",
                "features_active_at_most_10_trs_fraction",
            )
        },
        "pairwise_abs_spearman": pairwise,
        "rank_and_conditioning": ranks,
    }
    if include_profiles:
        result["per_feature"] = {
            key: _jsonable(columns[key])
            for key in (
                "active_count",
                "active_fraction",
                "zero_fraction",
                "binary_temporal_variance",
                "activation_mean",
                "activation_variance",
                "positive_activation_rms",
            )
        }
    return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(x) for x in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and (not np.isfinite(value)):
        return None
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(x) for x in value]
    return value


def _per_layer_summary(values: np.ndarray, layers: Sequence[int]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for layer in sorted(set(int(layer) for layer in layers)):
        idx = [i for i, item in enumerate(layers) if int(item) == int(layer)]
        output[str(layer)] = {
            "selected_feature_count": int(len(idx)),
            "feature_positions": [int(i) for i in idx],
            # Global source/target summaries retain the per-feature vectors.
            # Per-layer entries carry the requested aggregates only; repeating
            # every vector six times made the all-condition JSON needlessly
            # large without adding information.
            "summary": summarize_values(values[:, idx], include_profiles=False) if idx else summarize_values(np.empty((len(values), 0)), include_profiles=False),
        }
    return output


def _cluster_summary(bank_record: Mapping[str, Any], bank_name: str, size: int) -> dict[str, Any] | None:
    if not bank_name.startswith("source_story_correlation_pruned@"):
        return None
    threshold = bank_name.split("@", 1)[1]
    pruned_records = bank_record["correlation_pruned"]
    if threshold not in pruned_records:
        matches = [key for key in pruned_records if np.isclose(float(key), float(threshold))]
        if len(matches) != 1:
            return None
        threshold = matches[0]
    pruned = pruned_records[threshold]
    exact = pruned.get("clusters_over_exact_source_variance_banks", {}).get(str(int(size)))
    if not exact:
        return None
    clusters = [
        cluster
        for layer_clusters in exact.get("clusters_by_layer", {}).values()
        for cluster in layer_clusters
    ]
    sizes = np.asarray([int(cluster["size"]) for cluster in clusters], dtype=np.int64)
    if len(sizes):
        quantiles = {"p50": float(np.quantile(sizes, 0.50)), "p95": float(np.quantile(sizes, 0.95)), "max": int(np.max(sizes))}
    else:
        quantiles = {"p50": None, "p95": None, "max": 0}
    return {
        "threshold": float(threshold),
        "cluster_count": int(len(clusters)),
        "cluster_size_quantiles": quantiles,
        "cluster_sizes": [int(value) for value in sizes.tolist()],
        "complete_exact_k_predictor_membership": bool(exact.get("complete_exact_k_predictor_membership", False)),
    }


def distribution_shift(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    """Summarize target distribution relative to source training rows."""
    source = _finite(source)
    target = _finite(target)
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target banks must have the same feature count")
    src = _column_stats(source)
    tgt = _column_stats(target)
    source_std = np.sqrt(np.asarray(src["activation_variance"], dtype=np.float64))
    shift = (np.asarray(tgt["activation_mean"]) - np.asarray(src["activation_mean"])) / np.maximum(source_std, 1e-12)
    source_min = np.min(source, axis=0) if len(source) else np.full(source.shape[1], np.nan)
    source_max = np.max(source, axis=0) if len(source) else np.full(source.shape[1], np.nan)
    out_of_range = np.mean((target < source_min[None, :]) | (target > source_max[None, :]), axis=0) if len(target) else np.full(source.shape[1], np.nan)
    return {
        "source_rows": int(len(source)),
        "target_rows": int(len(target)),
        "active_fraction_difference": _jsonable(np.asarray(tgt["active_fraction"]) - np.asarray(src["active_fraction"])),
        "activation_mean_standardized_shift": _jsonable(shift),
        "target_out_of_source_range_rate": _jsonable(out_of_range),
        "summary": {
            "active_fraction_difference_mean_abs": float(np.mean(np.abs(np.asarray(tgt["active_fraction"]) - np.asarray(src["active_fraction"])))) if source.shape[1] else None,
            "standardized_mean_shift_abs_median": float(np.median(np.abs(shift))) if len(shift) else None,
            "standardized_mean_shift_abs_p95": float(np.quantile(np.abs(shift), 0.95)) if len(shift) else None,
            "out_of_source_range_rate_mean": float(np.mean(out_of_range)) if len(out_of_range) else None,
            "out_of_source_range_rate_p95": float(np.quantile(out_of_range, 0.95)) if len(out_of_range) else None,
        },
    }


def _bank_specs(bank_record: Mapping[str, Any], sizes: Sequence[int], thresholds: Sequence[float]) -> list[str]:
    names = ["source_variance", "source_prevalence"]
    names.extend(f"source_story_correlation_pruned@{float(t):.2f}" for t in thresholds)
    return names


def _summarize_bank(
    source_bundle: Mapping[str, Any],
    bank_record: Mapping[str, Any],
    bank_name: str,
    size: int,
    source_rows: np.ndarray,
    target_rows: np.ndarray | None,
    *,
    model_key: str,
    source_story: str,
    target_story: str | None,
    fold: int | None,
) -> dict[str, Any]:
    layers, feature_ids, entries = _entries_for_bank(bank_record, bank_name, size)
    values = _extract_selected(source_bundle["all_sae_tr_means"], source_bundle["selected_layers"], layers, feature_ids)
    source_values = values[source_rows]
    record: dict[str, Any] = {
        "model_key": model_key,
        "source_story": source_story,
        "target_story": target_story,
        "fold": None if fold is None else int(fold),
        "bank_name": bank_name,
        "bank_size": int(size),
        "layer_ids": layers,
        "feature_ids": feature_ids,
        "feature_names": [str(row.get("feature_name", f"layer_{layer}_feature_{feature}")) for layer, feature, row in zip(layers, feature_ids, entries, strict=False)],
        "source_rows": [int(row) for row in source_rows.tolist()],
        "source_summary": summarize_values(source_values, include_profiles=True),
        "per_layer_source_summary": _per_layer_summary(source_values, layers),
        "cluster_summary": _cluster_summary(bank_record, bank_name, size),
        "outcome_censoring_used_for_selection": False,
    }
    if target_rows is not None:
        target_values = values[target_rows]
        record["target_rows"] = [int(row) for row in target_rows.tolist()]
        record["target_summary"] = summarize_values(target_values, include_profiles=True)
        record["per_layer_target_summary"] = _per_layer_summary(target_values, layers)
        record["source_to_target_distribution_shift"] = distribution_shift(source_values, target_values)
    else:
        record["target_rows"] = []
        record["target_summary"] = None
        record["source_to_target_distribution_shift"] = None
    return record


def _identity_set(bank_record: Mapping[str, Any], bank_name: str, size: int) -> set[tuple[int, int]]:
    layers, feature_ids, _ = _entries_for_bank(bank_record, bank_name, size)
    return set(zip(layers, feature_ids, strict=False))


def _overlap_records(
    bank_dir: Mapping[tuple[str, str], Mapping[str, Any]],
    model_key: str,
    sizes: Sequence[int],
    bank_names: Sequence[str],
) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    left = bank_dir[("shapesphysical", "shapessocial")]
    right = bank_dir[("shapessocial", "shapesphysical")]
    for name in bank_names:
        for size in sizes:
            a = _identity_set(left, name, size)
            b = _identity_set(right, name, size)
            result.append({
                "model_key": model_key,
                "bank_name": name,
                "bank_size": int(size),
                "physical_source_count": int(len(a)),
                "social_source_count": int(len(b)),
                "intersection_count": int(len(a & b)),
                "union_count": int(len(a | b)),
                "jaccard": float(len(a & b) / len(a | b)) if a | b else None,
            })
    return result


def run(*, config_path: Path, spec: Mapping[str, Any], args: argparse.Namespace | None = None) -> dict[str, Any]:
    """Run all available cross-story and within-story conditioning diagnostics."""
    config_path = Path(config_path).resolve()
    data_root = _resolve_data_root(config_path, spec)
    output_root = resolve_output_root(spec=spec)
    artifact_root = output_root / str(spec["paths"]["artifact_root"]) / "conditioning"
    if artifact_root.exists() and any(artifact_root.iterdir()) and not bool(getattr(args, "force", False)):
        raise FileExistsError(f"conditioning output exists; pass --force to replace: {artifact_root}")
    artifact_root.mkdir(parents=True, exist_ok=True)
    bank_root = _resolve_bank_root(config_path, spec)
    sizes = tuple(int(value) for value in spec["feature_bank"]["sizes"])
    thresholds = tuple(float(value) for value in next(item for item in spec["feature_bank"]["banks"] if item["name"] == "source_story_correlation_pruned")["thresholds"])
    model_keys = list(spec["models"].keys())
    stories = list(spec["stories"].keys())
    scope = str(getattr(args, "scope", "all") if args is not None else "all")
    include_cross = scope in {"all", "cross"}
    include_within = scope in {"all", "within"}
    selected_models = list(getattr(args, "models", None) or model_keys) if args is not None else model_keys
    selected_stories = list(getattr(args, "stories", None) or stories) if args is not None else stories
    bank_names = _bank_specs({}, sizes, thresholds)

    bundles: dict[tuple[str, str], dict[str, Any]] = {}
    for model_key in selected_models:
        for story in selected_stories:
            bundles[(model_key, story)] = _load_source_bundle(
                _source_bundle_path(data_root, spec, model_key, story)
            )

    records: list[dict[str, Any]] = []
    overlap: list[dict[str, Any]] = []
    bank_manifest_entries: list[dict[str, Any]] = []
    for model_key in selected_models:
        # Keep each cross-story bank record in memory only for its identity
        # overlap calculation; it is small compared with all-feature arrays.
        cross_records: dict[tuple[str, str], dict[str, Any]] = {}
        for source_story in selected_stories:
            other_stories = [story for story in selected_stories if story != source_story]
            for target_story in other_stories:
                path = bank_root / "cross_story" / model_key / f"{source_story}_to_{target_story}" / "banks.json"
                if not path.is_file():
                    raise FileNotFoundError(f"feature-bank artifact missing: {path}")
                bank_record = json.loads(path.read_text(encoding="utf-8"))
                cross_records[(source_story, target_story)] = bank_record
                source_bundle = bundles[(model_key, source_story)]
                source_rows = np.arange(source_bundle["all_sae_tr_means"].shape[1], dtype=np.int64)
                target_rows = np.arange(bundles[(model_key, target_story)]["all_sae_tr_means"].shape[1], dtype=np.int64)
                # Target values are extracted from the target story's all-SAE
                # bundle using the source-story identities, not from the
                # source bundle.  This is the distribution-shift comparison
                # promised by the analysis specification.
                target_bundle = bundles[(model_key, target_story)]
                for bank_name in bank_names:
                    for size in sizes:
                        layers, feature_ids, _ = _entries_for_bank(bank_record, bank_name, size)
                        source_values = _extract_selected(source_bundle["all_sae_tr_means"], source_bundle["selected_layers"], layers, feature_ids)
                        target_values = _extract_selected(target_bundle["all_sae_tr_means"], target_bundle["selected_layers"], layers, feature_ids)
                        record = {
                            "model_key": model_key,
                            "source_story": source_story,
                            "target_story": target_story,
                            "fold": None,
                            "bank_name": bank_name,
                            "bank_size": int(size),
                            "actual_bank_size": int(len(feature_ids)),
                            "layer_ids": layers,
                            "feature_ids": feature_ids,
                            "feature_names": [f"layer_{layer}_feature_{feature}" for layer, feature in zip(layers, feature_ids, strict=False)],
                            "source_rows": [int(row) for row in source_rows.tolist()],
                            "target_rows": [int(row) for row in target_rows.tolist()],
                            "source_summary": summarize_values(source_values, include_profiles=True),
                            "per_layer_source_summary": _per_layer_summary(source_values, layers),
                            "target_summary": summarize_values(target_values, include_profiles=True),
                            "per_layer_target_summary": _per_layer_summary(target_values, layers),
                            "source_to_target_distribution_shift": distribution_shift(source_values, target_values),
                            "cluster_summary": _cluster_summary(bank_record, bank_name, size),
                            "selection_source_sha256": str(bank_record.get("source_predictor_sha256", "")),
                            "bank_bundle_sha256": str(bank_record.get("bank_bundle_sha256", "")),
                            "outcome_censoring_used_for_selection": False,
                            "selection_is_training_only": False,
                        }
                        records.append(record)
                bank_manifest_entries.append({"path": path.relative_to(bank_root).as_posix(), "sha256": sha256_file(path), "mode": "cross_story", "model_key": model_key, "source_story": source_story, "target_story": target_story})
        if include_cross and set(cross_records) == {(stories[0], stories[1]), (stories[1], stories[0])}:
            overlap.extend(_overlap_records(cross_records, model_key, sizes, bank_names))

        if include_within:
            for story in selected_stories:
                bundle = bundles[(model_key, story)]
                for fold in range(int(spec["cross_validation"]["outer_folds"])):
                    path = bank_root / "within_story" / model_key / story / f"fold_{fold:02d}" / "banks.json"
                    if not path.is_file():
                        raise FileNotFoundError(f"feature-bank artifact missing: {path}")
                    bank_record = json.loads(path.read_text(encoding="utf-8"))
                    source_rows = np.asarray(bank_record.get("selection_rows", []), dtype=np.int64)
                    held_rows = np.asarray(bank_record.get("held_out_rows", []), dtype=np.int64)
                    if len(np.intersect1d(source_rows, held_rows)):
                        raise ValueError(f"within-story fold {path}: source/held rows overlap")
                    for bank_name in bank_names:
                        for size in sizes:
                            layers, feature_ids, _ = _entries_for_bank(bank_record, bank_name, size)
                            values = _extract_selected(bundle["all_sae_tr_means"], bundle["selected_layers"], layers, feature_ids)
                            record = {
                                "model_key": model_key,
                                "source_story": story,
                                "target_story": story,
                                "fold": int(fold),
                                "bank_name": bank_name,
                                "bank_size": int(size),
                                "actual_bank_size": int(len(feature_ids)),
                                "layer_ids": layers,
                                "feature_ids": feature_ids,
                                "feature_names": [f"layer_{layer}_feature_{feature}" for layer, feature in zip(layers, feature_ids, strict=False)],
                                "source_rows": [int(row) for row in source_rows.tolist()],
                                "target_rows": [int(row) for row in held_rows.tolist()],
                                "source_summary": summarize_values(values[source_rows], include_profiles=True),
                                "per_layer_source_summary": _per_layer_summary(values[source_rows], layers),
                                "target_summary": summarize_values(values[held_rows], include_profiles=True),
                                "per_layer_target_summary": _per_layer_summary(values[held_rows], layers),
                                "source_to_target_distribution_shift": distribution_shift(values[source_rows], values[held_rows]),
                                "cluster_summary": _cluster_summary(bank_record, bank_name, size),
                                "selection_source_sha256": str(bank_record.get("source_predictor_sha256", "")),
                                "bank_bundle_sha256": str(bank_record.get("bank_bundle_sha256", "")),
                                "outcome_censoring_used_for_selection": False,
                                "purge_adjacent_trs": int(bank_record.get("purge_adjacent_trs", 4)),
                                "selection_is_training_only": True,
                            }
                            records.append(record)
                    bank_manifest_entries.append({"path": path.relative_to(bank_root).as_posix(), "sha256": sha256_file(path), "mode": "within_story", "model_key": model_key, "story": story, "fold": int(fold)})

    input_paths = [_source_bundle_path(data_root, spec, model_key, story) for model_key in selected_models for story in selected_stories]
    input_paths.extend(bank_root / entry["path"] for entry in bank_manifest_entries)
    code_hash = sha256_file(Path(__file__).resolve())
    config_hash = sha256_file(config_path)
    fingerprint = provenance_fingerprint(
        config_path,
        input_paths=sorted(set(input_paths), key=str),
        code_paths=[Path(__file__).resolve()],
        extra={
            "scope": scope,
            "models": selected_models,
            "stories": selected_stories,
            "sizes": sizes,
            "thresholds": thresholds,
        },
    )
    payload = {
        "schema_version": 1,
        "analysis_role": "correspondence_feature_coverage_conditioning",
        "analysis_fingerprint": fingerprint,
        "analysis_config_sha256": config_hash,
        "code_sha256": code_hash,
        "scope": scope,
        "models": selected_models,
        "stories": selected_stories,
        "sizes": [int(size) for size in sizes],
        "thresholds": [float(value) for value in thresholds],
        "selection_contract": {
            "activation_aggregation": "mean_nonnegative_sae_activation_over_all_tokens_assigned_to_tr_including_zero_events",
            "minimum_active_trs": int(spec["feature_bank"]["primary"]["minimum_active_trs"]),
            "maximum_active_fraction": float(spec["feature_bank"]["primary"]["maximum_active_fraction"]),
            "source_rows_only": True,
            "within_story_rows_purged": True,
            "outcome_censoring_used_for_selection": False,
        },
        "records": records,
        "source_bank_overlap": overlap,
        "input_artifacts": bank_manifest_entries,
    }
    summary_path = artifact_root / "conditioning_summary.json"
    write_json(summary_path, payload)
    manifest = {
        "schema_version": 1,
        "analysis_fingerprint": fingerprint,
        "summary": "conditioning_summary.json",
        "summary_sha256": sha256_file(summary_path),
        "record_count": len(records),
        "overlap_record_count": len(overlap),
        "input_count": len(bank_manifest_entries),
        "code_sha256": code_hash,
        "analysis_config_sha256": config_hash,
    }
    write_json(artifact_root / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--scope", choices=("all", "cross", "within"), default="all")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--stories", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.data_root is not None:
        import os

        os.environ["FMRI_SAE_DATA_ROOT"] = str(args.data_root.resolve())
    if args.output_root is not None:
        import os

        os.environ["FMRI_SAE_OUTPUT_ROOT"] = str(args.output_root.resolve())
    config_path = (args.config or default_config_path()).resolve()
    result = run(config_path=config_path, spec=load_config(config_path), args=args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
