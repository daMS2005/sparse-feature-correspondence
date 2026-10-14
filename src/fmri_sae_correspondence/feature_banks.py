#!/usr/bin/env python3
"""Build outcome-blind source-story SAE feature banks.

This module deliberately consumes the uncapped ``all_sae_tr_means`` arrays
saved in the canonical evaluation-blind extraction bundles.  It never reads
an fMRI target or a fitted result.  Cross-story banks are selected from the
source story only; within-story banks are selected from the outer-fold
training TRs only.

The generated banks are nested and approximately layer-balanced.  The
correlation-pruned banks are exploratory diagnostics: clustering is performed
on the top source-variance candidates within each layer and the complete
cluster membership is retained for grouped-ablation analyses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import rankdata

from .config import default_config_path, load_config
from .integrity import scientific_fingerprint
from .paths import RuntimePaths, resolve_below

HERE = Path(__file__).resolve().parent
DEFAULT_SIZES = (64, 128, 256, 512, 1024)
DEFAULT_THRESHOLDS = (0.70, 0.80, 0.90)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    def convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(key): convert(val) for key, val in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(item) for item in obj]
        if isinstance(obj, np.ndarray):
            return convert(obj.tolist())
        if isinstance(obj, (np.integer, np.floating, np.bool_)):
            return obj.item()
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(value), indent=2, sort_keys=True) + "\n")


def _model_story_name(model_key: str, story: str) -> str:
    return f"{story}_lm_controls.npz"


def validate_analysis_selection_contract(
    analysis_config: dict[str, Any],
    *,
    sizes: Sequence[int],
    thresholds: Sequence[float],
    min_active_trs: int,
    max_active_fraction: float,
    max_candidate_size: int,
) -> None:
    """Fail closed when CLI selection settings diverge from the analysis config."""

    configured_sizes = tuple(sorted(int(value) for value in analysis_config["feature_bank"]["sizes"]))
    observed_sizes = tuple(sorted(int(value) for value in sizes))
    if observed_sizes != configured_sizes:
        raise ValueError(
            f"K grid differs from analysis configuration: observed {observed_sizes}, expected {configured_sizes}"
        )
    pruned = next(
        item
        for item in analysis_config["feature_bank"]["banks"]
        if item.get("name") == "source_story_correlation_pruned"
    )
    configured_thresholds = tuple(sorted(float(value) for value in pruned["thresholds"]))
    observed_thresholds = tuple(sorted(float(value) for value in thresholds))
    if observed_thresholds != configured_thresholds:
        raise ValueError(
            "correlation thresholds differ from analysis configuration: "
            f"observed {observed_thresholds}, expected {configured_thresholds}"
        )
    primary = analysis_config["feature_bank"]["primary"]
    if int(min_active_trs) != int(primary["minimum_active_trs"]):
        raise ValueError("minimum active-TR filter differs from analysis configuration")
    if not np.isclose(float(max_active_fraction), float(primary["maximum_active_fraction"])):
        raise ValueError("maximum active-fraction filter differs from analysis configuration")
    if int(max_candidate_size) != max(configured_sizes):
        raise ValueError("max candidate size must equal the largest configured K")


def load_all_sae_bundle(path: Path) -> dict[str, Any]:
    """Load and validate an uncapped all-feature predictor bundle.

    Only predictor-side fields are returned.  In particular, the fMRI target
    and any target-side metadata are intentionally ignored by this loader.
    """

    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as bundle:
        required = {"all_sae_tr_means", "selected_layers", "tr_indices"}
        missing = sorted(required - set(bundle.files))
        if missing:
            raise ValueError(f"{path}: missing uncapped predictor fields {missing}")
        all_sae = np.asarray(bundle["all_sae_tr_means"])
        layers = np.asarray(bundle["selected_layers"], dtype=np.int64)
        tr_indices = np.asarray(bundle["tr_indices"], dtype=np.int64)
        model_id = str(np.asarray(bundle["model_id"]).reshape(-1)[0]) if "model_id" in bundle else ""
        story = str(np.asarray(bundle["story"]).reshape(-1)[0]) if "story" in bundle else ""

    if all_sae.ndim != 3:
        raise ValueError(f"{path}: all_sae_tr_means must be [layer,time,feature]")
    if all_sae.shape[0] != len(layers):
        raise ValueError(f"{path}: layer axis does not match selected_layers")
    if all_sae.shape[1] != len(tr_indices):
        raise ValueError(f"{path}: time axis does not match tr_indices")
    if len(np.unique(layers)) != len(layers):
        raise ValueError(f"{path}: selected_layers contains duplicates")
    if len(np.unique(tr_indices)) != len(tr_indices):
        raise ValueError(f"{path}: tr_indices contains duplicates")
    return {
        "all_sae_tr_means": all_sae,
        "selected_layers": layers,
        "tr_indices": tr_indices,
        "model_id": model_id,
        "story": story,
        "source_path": path,
        "source_sha256": sha256_file(path),
    }


def _positive_rms(values: np.ndarray, active_count: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    sums = np.sum(np.square(values), axis=0)
    return np.sqrt(sums / np.maximum(active_count, 1))


def compute_feature_stats(
    all_sae: np.ndarray,
    selection_rows: Sequence[int],
    *,
    min_active_trs: int = 5,
    max_active_fraction: float = 0.95,
) -> dict[str, Any]:
    """Compute predictor-only prevalence and variance statistics.

    Activity is defined by the canonical non-negative SAE value being strictly
    greater than zero.  A feature is eligible when it is active in at least
    ``min_active_trs`` selection TRs and in no more than
    ``max_active_fraction`` of them.  The function returns per-layer arrays so
    callers can build either full-story or fold-only banks.
    """

    all_sae = np.asarray(all_sae)
    rows = np.asarray(selection_rows, dtype=np.int64)
    if all_sae.ndim != 3:
        raise ValueError("all_sae must have shape [layer,time,feature]")
    if rows.ndim != 1 or len(rows) == 0:
        raise ValueError("selection_rows must be a non-empty one-dimensional index array")
    if np.any(rows < 0) or np.any(rows >= all_sae.shape[1]):
        raise IndexError("selection_rows contains an out-of-range row")
    if min_active_trs < 1:
        raise ValueError("min_active_trs must be positive")
    if not 0.0 < max_active_fraction <= 1.0:
        raise ValueError("max_active_fraction must be in (0, 1]")

    n_rows = len(rows)
    prevalence: list[np.ndarray] = []
    active_count: list[np.ndarray] = []
    positive_rms: list[np.ndarray] = []
    binary_variance: list[np.ndarray] = []
    eligible: list[np.ndarray] = []
    for layer_values in all_sae[:, rows, :]:
        values = np.asarray(layer_values, dtype=np.float64)
        active = values > 0
        counts = np.sum(active, axis=0, dtype=np.int64)
        p = counts.astype(np.float64) / float(n_rows)
        rms = _positive_rms(np.where(active, values, 0.0), counts)
        variance = p * (1.0 - p)
        keep = (counts >= int(min_active_trs)) & (p <= float(max_active_fraction))
        prevalence.append(p)
        active_count.append(counts)
        positive_rms.append(rms)
        binary_variance.append(variance)
        eligible.append(np.flatnonzero(keep).astype(np.int64))

    return {
        "n_selection_trs": int(n_rows),
        "min_active_trs": int(min_active_trs),
        "max_active_fraction": float(max_active_fraction),
        "prevalence": prevalence,
        "active_count": active_count,
        "positive_rms": positive_rms,
        "binary_variance": binary_variance,
        "eligible": eligible,
    }


def _rank_layer_features(
    feature_ids: np.ndarray,
    *,
    score: np.ndarray,
    positive_rms: np.ndarray,
    prevalence: np.ndarray,
    method: str,
) -> list[int]:
    if method == "source_variance":
        # Score, then scale-aware tie break, then feature ID.  The primary
        # score is binary variance and therefore does not favor activation
        # magnitude.
        ordered = sorted(
            (int(feature_id) for feature_id in feature_ids.tolist()),
            key=lambda feature_id: (
                -float(score[feature_id]),
                -float(positive_rms[feature_id]),
                feature_id,
            ),
        )
    elif method == "source_prevalence":
        ordered = sorted(
            (int(feature_id) for feature_id in feature_ids.tolist()),
            key=lambda feature_id: (
                -float(prevalence[feature_id]),
                -float(positive_rms[feature_id]),
                feature_id,
            ),
        )
    else:
        raise ValueError(f"unknown ranking method {method}")
    return ordered


def _interleave_layer_orders(
    layer_orders: Sequence[Sequence[int]],
    layers: Sequence[int],
    *,
    max_size: int,
) -> list[tuple[int, int, int]]:
    """Return ``(layer, feature_id, within_layer_rank)`` in balanced order."""

    output: list[tuple[int, int, int]] = []
    for within_rank in range(max((len(order) for order in layer_orders), default=0)):
        for layer_index, order in enumerate(layer_orders):
            if within_rank >= len(order):
                continue
            output.append((int(layers[layer_index]), int(order[within_rank]), within_rank + 1))
            if len(output) >= max_size:
                return output
    return output


def _entry_from_stats(
    layer_index: int,
    layer: int,
    feature_id: int,
    within_layer_rank: int,
    stats: dict[str, Any],
) -> dict[str, Any]:
    return {
        "layer_index": int(layer_index),
        "layer": int(layer),
        "feature_id": int(feature_id),
        "feature_name": f"layer_{int(layer)}_feature_{int(feature_id)}",
        "within_layer_rank": int(within_layer_rank),
        "active_count": int(stats["active_count"][layer_index][feature_id]),
        "active_fraction": float(stats["prevalence"][layer_index][feature_id]),
        "binary_temporal_variance": float(stats["binary_variance"][layer_index][feature_id]),
        "positive_activation_rms": float(stats["positive_rms"][layer_index][feature_id]),
    }


def _banks_from_order(
    order: Sequence[tuple[int, int, int]],
    *,
    layers: Sequence[int],
    stats: dict[str, Any],
    layer_to_index: dict[int, int],
    sizes: Sequence[int],
    method: str,
    parent_order_hash: str,
) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    for layer, feature_id, within_rank in order:
        entries.append(
            _entry_from_stats(
                layer_to_index[int(layer)], int(layer), int(feature_id), int(within_rank), stats
            )
        )
    output: dict[str, Any] = {
        "method": method,
        "nested": True,
        "requested_sizes": [int(size) for size in sizes],
        "available_ranked_features": len(entries),
        "parent_order_sha256": parent_order_hash,
        "banks": {},
    }
    for requested in sizes:
        selected = entries[: int(requested)]
        requested_quota = _requested_layer_quota(int(requested), layers)
        realized_quota = {
            str(int(layer)): int(sum(1 for row in selected if int(row["layer"]) == int(layer)))
            for layer in layers
        }
        output["banks"][str(int(requested))] = {
            "requested_size": int(requested),
            "actual_size": len(selected),
            "complete": len(selected) == int(requested),
            "feature_names": [row["feature_name"] for row in selected],
            "layers": [row["layer"] for row in selected],
            "feature_ids": [row["feature_id"] for row in selected],
            "entries": selected,
            "requested_layer_quota": requested_quota,
            "realized_layer_quota": realized_quota,
            "quota_deficit_redistributed": requested_quota != realized_quota,
        }
    return output


def _requested_layer_quota(size: int, layers: Sequence[int]) -> dict[str, int]:
    """Return the deterministic balanced quota before short-layer redistribution."""

    n_layers = len(layers)
    base, remainder = divmod(int(size), n_layers)
    return {
        str(int(layer)): int(base + (1 if index < remainder else 0))
        for index, layer in enumerate(layers)
    }


def build_ranked_banks(
    all_sae: np.ndarray,
    layers: Sequence[int],
    selection_rows: Sequence[int],
    *,
    method: str,
    sizes: Sequence[int] = DEFAULT_SIZES,
    min_active_trs: int = 5,
    max_active_fraction: float = 0.95,
    max_candidate_size: int | None = None,
) -> dict[str, Any]:
    """Build nested source-variance or source-prevalence banks."""

    layers = [int(layer) for layer in layers]
    sizes = tuple(sorted({int(size) for size in sizes}))
    if not sizes or min(sizes) < 1:
        raise ValueError("sizes must contain positive integers")
    max_size = int(max(sizes) if max_candidate_size is None else max_candidate_size)
    if max_size < max(sizes):
        raise ValueError("max_candidate_size cannot be smaller than the largest requested size")
    stats = compute_feature_stats(
        all_sae,
        selection_rows,
        min_active_trs=min_active_trs,
        max_active_fraction=max_active_fraction,
    )
    layer_orders: list[list[int]] = []
    for layer_index, feature_ids in enumerate(stats["eligible"]):
        order = _rank_layer_features(
            feature_ids,
            score=stats["binary_variance"][layer_index],
            positive_rms=stats["positive_rms"][layer_index],
            prevalence=stats["prevalence"][layer_index],
            method=method,
        )
        layer_orders.append(order)
    order = _interleave_layer_orders(layer_orders, layers, max_size=max_size)
    layer_to_index = {int(layer): index for index, layer in enumerate(layers)}
    order_hash = stable_hash(order)
    banks = _banks_from_order(
        order,
        layers=layers,
        stats=stats,
        layer_to_index=layer_to_index,
        sizes=sizes,
        method=method,
        parent_order_hash=order_hash,
    )
    banks.update(
        {
            "n_layers": len(layers),
            "layers": layers,
            "selection_rows": [int(row) for row in selection_rows],
            "stats": {
                "n_selection_trs": stats["n_selection_trs"],
                "min_active_trs": stats["min_active_trs"],
                "max_active_fraction": stats["max_active_fraction"],
                "eligible_count_by_layer": [int(len(values)) for values in stats["eligible"]],
                "dictionary_size_by_layer": [int(all_sae.shape[2]) for _ in layers],
            },
            "candidate_order": [
                _entry_from_stats(
                    layer_to_index[layer], layer, feature_id, within_rank, stats
                )
                for layer, feature_id, within_rank in order
            ],
        }
    )
    return banks


def bank_identity(bank: dict[str, Any], size: int) -> tuple[tuple[int, int], ...]:
    entry = bank["banks"][str(int(size))]
    return tuple(zip((int(layer) for layer in entry["layers"]), (int(fid) for fid in entry["feature_ids"]), strict=False))


def assert_same_bank_identity(
    bank: dict[str, Any],
    *,
    size: int,
    observed_layers: Sequence[int],
    observed_feature_ids: Sequence[int],
) -> None:
    expected = bank_identity(bank, size)
    observed = tuple(zip((int(layer) for layer in observed_layers), (int(fid) for fid in observed_feature_ids), strict=False))
    if expected != observed:
        raise AssertionError(
            f"feature identity mismatch for size {size}: expected {expected[:3]}..., observed {observed[:3]}..."
        )


def extract_bank_predictors(
    all_sae: np.ndarray,
    layers: Sequence[int],
    bank: dict[str, Any],
    size: int,
) -> np.ndarray:
    """Extract columns using a bank, preserving its exact layer/feature IDs."""

    all_sae = np.asarray(all_sae)
    layer_to_index = {int(layer): index for index, layer in enumerate(layers)}
    entry = bank["banks"][str(int(size))]
    columns = [
        all_sae[layer_to_index[int(layer)], :, int(feature_id)]
        for layer, feature_id in zip(entry["layers"], entry["feature_ids"], strict=False)
    ]
    if not columns:
        return np.empty((all_sae.shape[1], 0), dtype=all_sae.dtype)
    return np.column_stack(columns)


def _spearman_abs_correlation(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("values must be [time, feature]")
    if values.shape[1] == 0:
        return np.empty((0, 0), dtype=np.float64)
    ranks = np.column_stack([rankdata(values[:, j], method="average") for j in range(values.shape[1])])
    centered = ranks - ranks.mean(axis=0, keepdims=True)
    denominator = np.sqrt(np.sum(centered * centered, axis=0))
    valid = denominator > 0
    corr = np.zeros((values.shape[1], values.shape[1]), dtype=np.float64)
    if np.any(valid):
        normalized = np.zeros_like(centered)
        normalized[:, valid] = centered[:, valid] / denominator[valid]
        corr = normalized.T @ normalized
    corr[~np.isfinite(corr)] = 0.0
    np.fill_diagonal(corr, 1.0)
    return np.abs(np.clip(corr, -1.0, 1.0))


def cluster_feature_candidates(
    values: np.ndarray,
    feature_ids: Sequence[int],
    *,
    threshold: float,
    feature_ranks: dict[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Cluster candidates by complete-linkage distance ``1-|rho|``.

    The returned clusters are sorted by the rank of their representative, so
    representative prefixes are deterministic at every requested bank size.
    """

    if not 0.0 < float(threshold) < 1.0:
        raise ValueError("correlation threshold must be strictly between zero and one")
    ids = [int(feature_id) for feature_id in feature_ids]
    if len(set(ids)) != len(ids):
        raise ValueError("feature_ids must be unique")
    if np.asarray(values).ndim != 2 or np.asarray(values).shape[1] != len(ids):
        raise ValueError("values must have one column per feature_id")
    if not ids:
        return []
    ranks = feature_ranks or {feature_id: index + 1 for index, feature_id in enumerate(ids)}
    abs_corr = _spearman_abs_correlation(values)
    distance = np.clip(1.0 - abs_corr, 0.0, 1.0)
    if len(ids) == 1:
        labels = np.ones(1, dtype=np.int64)
    else:
        condensed = squareform(distance, checks=False)
        tree = linkage(condensed, method="complete", optimal_ordering=False)
        # Floating-point tolerances should not make an exact threshold
        # boundary nondeterministic.
        labels = fcluster(tree, t=max(0.0, 1.0 - float(threshold) + 1e-12), criterion="distance")
    raw_clusters: dict[int, list[int]] = {}
    for feature_id, label in zip(ids, labels.tolist(), strict=False):
        raw_clusters.setdefault(int(label), []).append(feature_id)
    clusters: list[dict[str, Any]] = []
    for members in raw_clusters.values():
        members = sorted(members, key=lambda feature_id: (int(ranks[feature_id]), feature_id))
        representative = members[0]
        clusters.append(
            {
                "representative_feature_id": int(representative),
                "representative_rank": int(ranks[representative]),
                "members": [int(member) for member in members],
                "size": len(members),
            }
        )
    clusters.sort(key=lambda cluster: (cluster["representative_rank"], cluster["representative_feature_id"]))
    for cluster_id, cluster in enumerate(clusters, start=1):
        cluster["cluster_id"] = int(cluster_id)
    return clusters


def build_correlation_pruned_banks(
    all_sae: np.ndarray,
    layers: Sequence[int],
    selection_rows: Sequence[int],
    *,
    sizes: Sequence[int] = DEFAULT_SIZES,
    threshold: float,
    min_active_trs: int = 5,
    max_active_fraction: float = 0.95,
    max_candidate_size: int | None = None,
) -> dict[str, Any]:
    """Build representative banks and retain complete correlation clusters."""

    primary = build_ranked_banks(
        all_sae,
        layers,
        selection_rows,
        method="source_variance",
        sizes=sizes,
        min_active_trs=min_active_trs,
        max_active_fraction=max_active_fraction,
        max_candidate_size=max_candidate_size,
    )
    layers = [int(layer) for layer in layers]
    layer_to_index = {layer: index for index, layer in enumerate(layers)}
    source_rows = np.asarray(selection_rows, dtype=np.int64)
    candidates_by_layer: dict[int, list[dict[str, Any]]] = {layer: [] for layer in layers}
    for row in primary["candidate_order"]:
        candidates_by_layer[int(row["layer"])].append(row)
    clusters_by_layer: dict[str, list[dict[str, Any]]] = {}
    representatives_by_layer: list[list[int]] = []
    for layer_index, layer in enumerate(layers):
        candidates = candidates_by_layer[int(layer)]
        ids = [int(row["feature_id"]) for row in candidates]
        ranks = {int(row["feature_id"]): int(row["within_layer_rank"]) for row in candidates}
        values = np.asarray(all_sae[layer_index, source_rows, :][:, ids], dtype=np.float64)
        clusters = cluster_feature_candidates(
            values,
            ids,
            threshold=float(threshold),
            feature_ranks=ranks,
        )
        clusters_by_layer[str(int(layer))] = clusters
        representatives_by_layer.append(
            [int(cluster["representative_feature_id"]) for cluster in clusters]
        )
    # Keep a second, exact-K cluster product.  These memberships are defined
    # on the exact source-variance K-column predictor matrix and are therefore
    # suitable for grouped ablation at that K.  The extended-candidate
    # clusters above are used only to construct representative-only pruned
    # banks and must not be substituted for this product.
    exact_clusters_by_size: dict[str, Any] = {}
    for requested in sorted(set(int(size) for size in sizes)):
        exact_entry = primary["banks"][str(requested)]
        exact_by_layer: dict[str, list[dict[str, Any]]] = {}
        for layer_index, layer in enumerate(layers):
            positions = [
                index
                for index, row in enumerate(exact_entry["entries"])
                if int(row["layer"]) == int(layer)
            ]
            ids = [int(exact_entry["entries"][index]["feature_id"]) for index in positions]
            ranks = {
                int(exact_entry["entries"][index]["feature_id"]): int(
                    exact_entry["entries"][index]["within_layer_rank"]
                )
                for index in positions
            }
            values = np.asarray(all_sae[layer_index, source_rows, :][:, ids], dtype=np.float64)
            exact_by_layer[str(int(layer))] = cluster_feature_candidates(
                values,
                ids,
                threshold=float(threshold),
                feature_ranks=ranks,
            )
        exact_clusters_by_size[str(requested)] = {
            "requested_size": int(requested),
            "actual_size": int(exact_entry["actual_size"]),
            "clusters_by_layer": exact_by_layer,
            "total_clusters": int(sum(len(values) for values in exact_by_layer.values())),
            "complete_exact_k_predictor_membership": bool(exact_entry["complete"]),
        }
    representative_order = _interleave_layer_orders(
        representatives_by_layer,
        layers,
        max_size=max(sizes),
    )
    # The representative's within-layer rank refers to the source-variance
    # candidate rank, not to the representative-prefix rank.
    rank_lookup = {
        (int(row["layer"]), int(row["feature_id"])): int(row["within_layer_rank"])
        for row in primary["candidate_order"]
    }
    stats = compute_feature_stats(
        all_sae,
        source_rows,
        min_active_trs=min_active_trs,
        max_active_fraction=max_active_fraction,
    )
    entries = [
        _entry_from_stats(
            layer_to_index[int(layer)], int(layer), int(feature_id), rank_lookup[(int(layer), int(feature_id))], stats
        )
        for layer, feature_id, _ in representative_order
    ]
    output: dict[str, Any] = {
        "method": "correlation_pruned_source_variance",
        "nested": True,
        "threshold": float(threshold),
        "correlation": "absolute_spearman_complete_linkage",
        "requested_sizes": [int(size) for size in sorted(set(int(size) for size in sizes))],
        "available_representatives": len(entries),
        "representative_order_sha256": stable_hash(representative_order),
        "source_variance_parent_order_sha256": primary["parent_order_sha256"],
        "clusters_by_layer": clusters_by_layer,
        "clusters_over_exact_source_variance_banks": exact_clusters_by_size,
        "cluster_membership_definition": "exact_source_variance_K_predictors",
        "representative_bank_definition": "representatives_of_extended_source_variance_candidates",
        "banks": {},
        "layers": layers,
        "selection_rows": [int(row) for row in source_rows.tolist()],
        "stats": primary["stats"],
    }
    for requested in output["requested_sizes"]:
        selected = entries[: int(requested)]
        requested_quota = _requested_layer_quota(int(requested), layers)
        realized_quota = {
            str(int(layer)): int(sum(1 for row in selected if int(row["layer"]) == int(layer)))
            for layer in layers
        }
        output["banks"][str(int(requested))] = {
            "requested_size": int(requested),
            "actual_size": len(selected),
            "complete": len(selected) == int(requested),
            "feature_names": [row["feature_name"] for row in selected],
            "layers": [row["layer"] for row in selected],
            "feature_ids": [row["feature_id"] for row in selected],
            "entries": selected,
            "requested_layer_quota": requested_quota,
            "realized_layer_quota": realized_quota,
            "quota_deficit_redistributed": requested_quota != realized_quota,
        }
    return output


def build_source_bank_bundle(
    *,
    model_key: str,
    model_id: str,
    source_story: str,
    target_story: str | None,
    mode: str,
    source_bundle: dict[str, Any],
    selection_rows: Sequence[int],
    held_out_rows: Sequence[int] | None,
    purge_adjacent_trs: int,
    sizes: Sequence[int],
    thresholds: Sequence[float],
    min_active_trs: int,
    max_active_fraction: float,
    max_candidate_size: int,
    analysis_spec_sha256: str,
    analysis_config_sha256: str,
    generator_sha256: str,
) -> dict[str, Any]:
    """Build all bank variants and a self-contained provenance record."""

    all_sae = source_bundle["all_sae_tr_means"]
    layers = source_bundle["selected_layers"]
    selection_rows = np.asarray(selection_rows, dtype=np.int64)
    held_out_rows = np.asarray([] if held_out_rows is None else held_out_rows, dtype=np.int64)
    if np.intersect1d(selection_rows, held_out_rows).size:
        raise ValueError("selection rows overlap held-out rows")
    if purge_adjacent_trs < 0:
        raise ValueError("purge_adjacent_trs must be non-negative")
    if len(held_out_rows):
        nearby = np.concatenate(
            [
                np.arange(
                    max(0, int(row) - int(purge_adjacent_trs)),
                    min(all_sae.shape[1], int(row) + int(purge_adjacent_trs) + 1),
                    dtype=np.int64,
                )
                for row in held_out_rows
            ]
        )
        if np.intersect1d(selection_rows, nearby).size:
            raise ValueError("selection rows overlap held-out rows after temporal purge")
    variance = build_ranked_banks(
        all_sae,
        layers,
        selection_rows,
        method="source_variance",
        sizes=sizes,
        min_active_trs=min_active_trs,
        max_active_fraction=max_active_fraction,
        max_candidate_size=max_candidate_size,
    )
    prevalence = build_ranked_banks(
        all_sae,
        layers,
        selection_rows,
        method="source_prevalence",
        sizes=sizes,
        min_active_trs=min_active_trs,
        max_active_fraction=max_active_fraction,
        max_candidate_size=max_candidate_size,
    )
    pruned = {
        str(threshold): build_correlation_pruned_banks(
            all_sae,
            layers,
            selection_rows,
            sizes=sizes,
            threshold=float(threshold),
            min_active_trs=min_active_trs,
            max_active_fraction=max_active_fraction,
            max_candidate_size=max_candidate_size,
        )
        for threshold in thresholds
    }
    config = {
        "sizes": [int(size) for size in sizes],
        "thresholds": [float(value) for value in thresholds],
        "min_active_trs": int(min_active_trs),
        "max_active_fraction": float(max_active_fraction),
        "max_candidate_size": int(max_candidate_size),
        "selection_rule": "active_count >= min_active_trs and active_fraction <= max_active_fraction",
        "activity_rule": "canonical non-negative activation > 0",
        "primary_score": "p*(1-p)",
        "tie_break": "positive_activation_rms descending, feature_id ascending",
        "layer_allocation": "round_robin_interleave_within_layer_rank",
    }
    record = {
        "schema_version": 1,
        "analysis_role": "correspondence_source_story_feature_bank",
        "mode": mode,
        "model_key": model_key,
        "model_id": model_id,
        "source_story": source_story,
        "target_story": target_story,
        "source_predictor_path": str(source_bundle["source_path"]),
        "source_predictor_sha256": source_bundle["source_sha256"],
        "analysis_spec_sha256": analysis_spec_sha256,
        "analysis_config_sha256": analysis_config_sha256,
        "generator_sha256": generator_sha256,
        "selected_layers": [int(layer) for layer in layers.tolist()],
        "dictionary_size_by_layer": [int(all_sae.shape[2]) for _ in layers],
        "n_source_trs": int(len(selection_rows)),
        "source_predictor_representation": {
            "field": "all_sae_tr_means",
            "aggregation": "mean_over_all_transcript_tokens_in_each_tr",
            "positive_activation_clipped": True,
            "zero_activation_tokens_in_denominator": True,
            "denominator_field": "token_count",
        },
        "selection_rows": [int(row) for row in selection_rows.tolist()],
        "held_out_rows": [int(row) for row in held_out_rows.tolist()],
        "purge_adjacent_trs": int(purge_adjacent_trs),
        "target_excluded_from_selection": target_story is not None and source_story != target_story,
        "within_story_held_out_excluded": bool(len(held_out_rows)),
        "outcome_inputs_used": False,
        "selection_config": config,
        "source_variance": variance,
        "source_prevalence": prevalence,
        "correlation_pruned": pruned,
    }
    record["bank_bundle_sha256"] = stable_hash(
        {
            "source_sha256": record["source_predictor_sha256"],
            "source_story": source_story,
            "target_story": target_story,
            "mode": mode,
            "selection_rows": record["selection_rows"],
            "config": config,
            "generator_sha256": generator_sha256,
            "variance_order": variance["parent_order_sha256"],
            "prevalence_order": prevalence["parent_order_sha256"],
            "pruned_orders": {key: value["representative_order_sha256"] for key, value in pruned.items()},
        }
    )
    return record


def contiguous_fold_rows(
    n_rows: int,
    n_folds: int = 5,
    *,
    purge_adjacent_trs: int = 4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if n_rows < n_folds or n_folds < 2:
        raise ValueError("n_rows must be at least n_folds and n_folds must be >= 2")
    folds = [np.asarray(values, dtype=np.int64) for values in np.array_split(np.arange(n_rows), n_folds)]
    if purge_adjacent_trs < 0:
        raise ValueError("purge_adjacent_trs must be non-negative")
    output = []
    all_rows = np.arange(n_rows, dtype=np.int64)
    for held in folds:
        train = np.setdiff1d(all_rows, held, assume_unique=True)
        if purge_adjacent_trs:
            forbidden = np.concatenate(
                [
                    np.arange(
                        max(0, int(row) - int(purge_adjacent_trs)),
                        min(n_rows, int(row) + int(purge_adjacent_trs) + 1),
                        dtype=np.int64,
                    )
                    for row in held
                ]
            )
            train = np.setdiff1d(train, np.unique(forbidden), assume_unique=True)
        output.append((train, held))
    return output


def _format_threshold(value: float) -> str:
    return f"{float(value):.2f}".replace(".", "p")


def _resolve_source_paths(spec: dict[str, Any], data_root: Path) -> dict[tuple[str, str], Path]:
    output: dict[tuple[str, str], Path] = {}
    for model_key in spec["models"]:
        for story in spec["stories"]:
            logical_path = str(spec["paths"]["predictor_pattern"]).format(
                model=model_key,
                story=story,
            )
            path = resolve_below(data_root, logical_path)
            output[(model_key, story)] = path
    return output


def generate_all(
    *,
    config_path: Path,
    data_root: Path,
    output_root: Path,
    sizes: Sequence[int] = DEFAULT_SIZES,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    min_active_trs: int = 5,
    max_active_fraction: float = 0.95,
    max_candidate_size: int | None = None,
    n_folds: int = 5,
    include_within_story_folds: bool = True,
    model_keys: Sequence[str] | None = None,
    stories: Sequence[str] | None = None,
) -> dict[str, Any]:
    spec = load_config(config_path)
    source_paths = _resolve_source_paths(spec, data_root)
    model_keys = list(model_keys or spec["models"].keys())
    stories = list(stories or spec["stories"].keys())
    sizes = tuple(sorted({int(size) for size in sizes}))
    thresholds = tuple(sorted({float(value) for value in thresholds}))
    max_candidate_size = int(max_candidate_size or max(sizes))
    if max_candidate_size < max(sizes):
        raise ValueError("max_candidate_size must be >= largest requested size")
    validate_analysis_selection_contract(
        spec,
        sizes=sizes,
        thresholds=thresholds,
        min_active_trs=min_active_trs,
        max_active_fraction=max_active_fraction,
        max_candidate_size=max_candidate_size,
    )
    analysis_spec_sha256 = scientific_fingerprint(spec)
    analysis_config_sha256 = analysis_spec_sha256
    generator_sha256 = sha256_file(Path(__file__).resolve())
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "analysis_role": "correspondence_source_story_feature_bank_manifest",
        "analysis_spec_sha256": analysis_spec_sha256,
        "analysis_config_sha256": analysis_config_sha256,
        "analysis_config_id": spec.get("spec_id"),
        "generator_sha256": generator_sha256,
        "output_role": "feature_banks",
        "models": model_keys,
        "stories": stories,
        "sizes": [int(size) for size in sizes],
        "thresholds": [float(value) for value in thresholds],
        "include_within_story_folds": bool(include_within_story_folds),
        "n_folds": int(n_folds),
        "within_story_purge_adjacent_trs": 4,
        "selection_config": {
            "min_active_trs": int(min_active_trs),
            "max_active_fraction": float(max_active_fraction),
            "max_candidate_size": int(max_candidate_size),
        },
        "outcome_inputs_used": False,
        "cross_story": [],
        "within_story": [],
    }
    loaded: dict[tuple[str, str], dict[str, Any]] = {}
    for model_key in model_keys:
        if model_key not in spec["models"]:
            raise KeyError(f"unknown model {model_key}")
        for story in stories:
            if story not in spec["stories"]:
                raise KeyError(f"unknown story {story}")
            bundle_path = source_paths[(model_key, story)]
            bundle = load_all_sae_bundle(bundle_path)
            loaded[(model_key, story)] = bundle
            expected_model = str(spec["models"][model_key].get("model_id", ""))
            if expected_model and bundle["model_id"] and bundle["model_id"] != expected_model:
                raise ValueError(f"{bundle_path}: model_id {bundle['model_id']} != spec {expected_model}")
            if bundle["story"] and bundle["story"] != story:
                raise ValueError(f"{bundle_path}: story {bundle['story']} != requested {story}")

    for model_key in model_keys:
        model_id = str(spec["models"][model_key].get("model_id", ""))
        for source_story in stories:
            targets = [story for story in stories if story != source_story]
            for target_story in targets:
                source_bundle = loaded[(model_key, source_story)]
                n_rows = int(source_bundle["all_sae_tr_means"].shape[1])
                record = build_source_bank_bundle(
                    model_key=model_key,
                    model_id=model_id,
                    source_story=source_story,
                    target_story=target_story,
                    mode="cross_story_source",
                    source_bundle=source_bundle,
                    selection_rows=np.arange(n_rows, dtype=np.int64),
                    held_out_rows=np.asarray([], dtype=np.int64),
                    purge_adjacent_trs=0,
                    sizes=sizes,
                    thresholds=thresholds,
                    min_active_trs=min_active_trs,
                    max_active_fraction=max_active_fraction,
                    max_candidate_size=max_candidate_size,
                    analysis_spec_sha256=analysis_spec_sha256,
                    analysis_config_sha256=analysis_config_sha256,
                    generator_sha256=generator_sha256,
                )
                out_path = output_root / "cross_story" / model_key / f"{source_story}_to_{target_story}" / "banks.json"
                write_json(out_path, record)
                manifest["cross_story"].append({
                    "model_key": model_key,
                    "source_story": source_story,
                    "target_story": target_story,
                    "path": out_path.relative_to(output_root).as_posix(),
                    "sha256": sha256_file(out_path),
                    "bank_bundle_sha256": record["bank_bundle_sha256"],
                })
            if include_within_story_folds:
                source_bundle = loaded[(model_key, source_story)]
                n_rows = int(source_bundle["all_sae_tr_means"].shape[1])
                for fold_index, (train_rows, held_rows) in enumerate(
                    contiguous_fold_rows(n_rows, n_folds, purge_adjacent_trs=4)
                ):
                    record = build_source_bank_bundle(
                        model_key=model_key,
                        model_id=model_id,
                        source_story=source_story,
                        target_story=source_story,
                        mode="within_story_outer_fold",
                        source_bundle=source_bundle,
                        selection_rows=train_rows,
                        held_out_rows=held_rows,
                        purge_adjacent_trs=4,
                        sizes=sizes,
                        thresholds=thresholds,
                        min_active_trs=min_active_trs,
                        max_active_fraction=max_active_fraction,
                        max_candidate_size=max_candidate_size,
                        analysis_spec_sha256=analysis_spec_sha256,
                        analysis_config_sha256=analysis_config_sha256,
                        generator_sha256=generator_sha256,
                    )
                    out_path = output_root / "within_story" / model_key / source_story / f"fold_{fold_index:02d}" / "banks.json"
                    write_json(out_path, record)
                    manifest["within_story"].append({
                        "model_key": model_key,
                        "story": source_story,
                        "fold": fold_index,
                        "path": out_path.relative_to(output_root).as_posix(),
                        "sha256": sha256_file(out_path),
                        "bank_bundle_sha256": record["bank_bundle_sha256"],
                    })
    manifest_path = output_root / "feature_bank_manifest.json"
    write_json(manifest_path, manifest)
    manifest["manifest_sha256"] = sha256_file(manifest_path)
    write_json(manifest_path, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--sizes", nargs="+", type=int, default=list(DEFAULT_SIZES))
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--min-active-trs", type=int, default=5)
    parser.add_argument("--max-active-fraction", type=float, default=0.95)
    parser.add_argument("--max-candidate-size", type=int, default=None)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--no-within-story-folds", action="store_true")
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--stories", nargs="+", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    config_path = (args.config or default_config_path()).resolve()
    spec = load_config(config_path)
    runtime_paths = RuntimePaths.resolve(
        data_root=args.data_root,
        output_root=args.output_root,
    )
    data_root = runtime_paths.require_data_root()
    artifact_root = resolve_below(
        runtime_paths.require_output_root(),
        str(spec["paths"]["artifact_root"]),
    )
    output_root = resolve_below(artifact_root, str(spec["paths"]["feature_bank_root"]))
    if output_root.exists() and any(output_root.iterdir()) and not args.force:
        raise FileExistsError(f"feature-bank output exists; pass --force to replace: {output_root}")
    generate_all(
        config_path=config_path,
        data_root=data_root,
        output_root=output_root,
        sizes=args.sizes,
        thresholds=args.thresholds,
        min_active_trs=args.min_active_trs,
        max_active_fraction=args.max_active_fraction,
        max_candidate_size=args.max_candidate_size,
        n_folds=args.folds,
        include_within_story_folds=not args.no_within_story_folds,
        model_keys=args.models,
        stories=args.stories,
    )
    print(f"wrote source-story feature banks under {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
