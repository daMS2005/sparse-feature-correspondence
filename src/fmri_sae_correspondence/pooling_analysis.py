#!/usr/bin/env python3
"""Fit and null-test pooled SAE predictors for the analysis.

This module is the generalized downstream stage for ``pooling.py``.
The latter only materializes outcome-free predictor matrices; the present
stage fits the same participant-specific, purged/nested ridge analyses used
by the canonical analysis, but reads ``pooling/<model>/<story>/<pooling>``
bundles instead of the uncapped ``all_sae_tr_means`` arrays.

The implementation is deliberately fail-closed.  A pooled bundle must be
present, its coordinate order must contain the requested source-story bank,
and the target bundle must be complete before fitting starts.  No fallback to
the canonical mean matrix is allowed: doing so would make a max, sum, or
event-count result silently become a mean result.

Observed fits use both Pearson-r-selected and out-of-sample-R2-selected
penalties; held-out conditional reliance is taken from the latter.  The
feature-identity null is within layer.  The temporal null shifts only the LM
target, refits penalty selection for every shift, and leaves the source-side
feature bank fixed.  For transfer, the source LM target is shifted and the
unshifted target story is evaluated with the refitted source mapping.

The command expects token-pooling bundles to have been materialized first.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .config import default_config_path
from .feature_banks import contiguous_fold_rows  # noqa: E402
from .fitting import (  # noqa: E402
    _artifact_root,
    _brain_path,
    _load_bank,
    bank_path,
    evaluate_fit,
    global_finite_training_rows,
    select_alpha_on_outer_train,
)
from .grouped_reliance import _cluster_groups  # noqa: E402
from .integrity import (  # noqa: E402
    AnalysisError,
    assert_cache_fingerprint,
    atomic_write_json,
    load_spec,
    provenance_fingerprint,
    resolve_data_root,
    sha256_file,
)
from .modeling import (  # noqa: E402
    block_labels,
    correspondence,
    feature_groups,
    fit_ridge,
    fit_transfer,
    lag_expand,
    modal_alpha,
    select_alpha,
    within_layer_identity_null,
)
from .nulls import fixed_shift_offsets  # noqa: E402
from .pooling import _pooling_root  # noqa: E402

HERE = Path(__file__).resolve().parent

PRIMARY_SIZE = 256
DEFAULT_POOLINGS = (
    "mean_positive_activation",
    "max_positive_activation",
    "sum_positive_activation",
    "event_count",
)
DEFAULT_VARIANTS = (
    "source_story_variance",
    "source_story_correlation_pruned_0p70",
    "source_story_correlation_pruned_0p80",
    "source_story_correlation_pruned_0p90",
)
GROUP_THRESHOLDS = (0.70, 0.80, 0.90)


def _scalar_string(values: Any, default: str = "") -> str:
    if values is None:
        return default
    array = np.asarray(values)
    return default if array.size == 0 else str(array.reshape(-1)[0])


def _pool_bundle_path(
    config_path: Path, spec: Mapping[str, Any], model: str, story: str, pooling: str
) -> Path:
    return _pooling_root(config_path, spec) / model / story / pooling / "predictors.npz"


def load_pooled_predictor(
    config_path: Path,
    spec: Mapping[str, Any],
    model: str,
    story: str,
    pooling: str,
) -> dict[str, Any]:
    """Load one validated pooled predictor bundle without touching all SAE data."""

    path = _pool_bundle_path(config_path, spec, model, story, pooling)
    if not path.is_file():
        raise FileNotFoundError(
            f"pooled predictor is missing: {path}; run analysis_pooling after token extraction"
        )
    # A predictor file by itself is not enough to establish that the four
    # pooling representations came from token records.  Require the
    # materializer's sibling manifest and its content-addressed token input;
    # this prevents a hand-created or stale ``predictors.npz`` from entering
    # a scientific fit.
    materialization_manifest = path.parent.parent / "manifest.json"
    if not materialization_manifest.is_file():
        raise AnalysisError(
            f"pooling materialization manifest is missing: {materialization_manifest}"
        )
    try:
        materialization = json.loads(materialization_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(
            f"pooling materialization manifest is unreadable: {materialization_manifest}"
        ) from exc
    token_path_raw = materialization.get("token_records")
    token_hash = materialization.get("token_records_sha256")
    if not token_path_raw or not token_hash:
        raise AnalysisError(
            f"pooling manifest lacks token-record provenance: {materialization_manifest}"
        )
    token_path = Path(str(token_path_raw))
    if not token_path.is_absolute():
        token_path = (materialization_manifest.parent / token_path).resolve()
    if not token_path.is_file():
        raise AnalysisError(
            f"token-level record referenced by pooling manifest is missing: {token_path}"
        )
    if sha256_file(token_path) != str(token_hash):
        raise AnalysisError(
            f"token-level record hash mismatch in pooling manifest: {token_path}"
        )
    output_entry = materialization.get("outputs", {}).get(pooling)
    if not isinstance(output_entry, Mapping):
        raise AnalysisError(
            f"pooling manifest has no validated output entry for {pooling}: {materialization_manifest}"
        )
    if output_entry.get("predictor_sha256") and sha256_file(path) != str(output_entry["predictor_sha256"]):
        raise AnalysisError(f"pooled predictor hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as bundle:
        required = {"predictors", "tr_indices", "feature_layers", "feature_ids"}
        missing = sorted(required - set(bundle.files))
        if missing:
            raise ValueError(f"{path}: missing pooled predictor fields {missing}")
        predictors = np.asarray(bundle["predictors"], dtype=np.float64)
        tr_indices = np.asarray(bundle["tr_indices"], dtype=np.int64)
        feature_layers = np.asarray(bundle["feature_layers"], dtype=np.int64)
        feature_ids = np.asarray(bundle["feature_ids"], dtype=np.int64)
        feature_names = (
            np.asarray(bundle["feature_names"], dtype=str)
            if "feature_names" in bundle.files
            else np.asarray(
                [
                    f"layer_{layer}_feature_{feature}"
                    for layer, feature in zip(feature_layers, feature_ids, strict=False)
                ],
                dtype=str,
            )
        )
        model_key = _scalar_string(bundle["model_key"] if "model_key" in bundle.files else None)
        bundle_story = _scalar_string(bundle["story"] if "story" in bundle.files else None)
        bundle_pooling = _scalar_string(bundle["pooling"] if "pooling" in bundle.files else None)
        semantics = _scalar_string(
            bundle["pooling_semantics"] if "pooling_semantics" in bundle.files else None
        )
        causal = bool(np.asarray(bundle["causal"]).reshape(-1)[0]) if "causal" in bundle.files else None
        context = int(np.asarray(bundle["context_tokens"]).reshape(-1)[0]) if "context_tokens" in bundle.files else None
    if predictors.ndim != 2:
        raise ValueError(f"{path}: predictors must have shape [TR,coordinate]")
    n_time, n_coordinates = predictors.shape
    if tr_indices.shape != (n_time,) or len(np.unique(tr_indices)) != n_time:
        raise ValueError(f"{path}: invalid TR index vector")
    expected_trs = int(spec["stories"][story]["expected_trs"])
    if n_time != expected_trs:
        raise ValueError(f"{path}: {n_time} TRs found, expected {expected_trs}")
    if feature_layers.shape != (n_coordinates,) or feature_ids.shape != (n_coordinates,):
        raise ValueError(f"{path}: coordinate arrays do not match predictors")
    if feature_names.shape != (n_coordinates,):
        raise ValueError(f"{path}: feature_names do not match predictors")
    if len(set(zip(feature_layers.tolist(), feature_ids.tolist(), strict=False))) != n_coordinates:
        raise ValueError(f"{path}: duplicate feature-layer coordinates")
    if not np.all(np.isfinite(predictors)) or np.any(predictors < -1e-6):
        raise ValueError(f"{path}: pooled predictors must be finite and non-negative")
    if model_key and model_key != model:
        raise ValueError(f"{path}: model_key={model_key!r}, expected {model!r}")
    if bundle_story and bundle_story != story:
        raise ValueError(f"{path}: story={bundle_story!r}, expected {story!r}")
    if bundle_pooling and bundle_pooling != pooling:
        raise ValueError(f"{path}: pooling={bundle_pooling!r}, expected {pooling!r}")
    if causal is not None and not causal:
        raise ValueError(f"{path}: pooled target requires causal LM extraction")
    expected_context = int(spec["targets"]["lm"]["maximum_context_tokens"])
    if context is not None and context != expected_context:
        raise ValueError(f"{path}: context_tokens={context}, expected {expected_context}")
    return {
        "path": path,
        "predictors": predictors,
        "tr_indices": tr_indices,
        "feature_layers": feature_layers,
        "feature_ids": feature_ids,
        "feature_names": feature_names,
        "model_key": model_key,
        "story": bundle_story,
        "pooling": bundle_pooling,
        "pooling_semantics": semantics,
        "causal": causal,
        "context_tokens": context,
        "sha256": sha256_file(path),
    }


def _require_tr_alignment(
    bundle: Mapping[str, Any], target_tr_indices: np.ndarray, *, context: str
) -> None:
    """Reject equal-length but differently indexed pooled/target arrays."""

    pooled = np.asarray(bundle["tr_indices"], dtype=np.int64)
    target = np.asarray(target_tr_indices, dtype=np.int64)
    if pooled.shape != target.shape or not np.array_equal(pooled, target):
        raise AnalysisError(
            f"{context}: pooled predictor TR indices do not exactly match its target bundle"
        )


def _load_targets_only(
    config_path: Path, spec: Mapping[str, Any], model: str, story: str
) -> dict[str, Any]:
    """Load LM/BOLD targets while deliberately avoiding all SAE columns."""

    root = resolve_data_root(config_path, spec)
    predictor_path = root / str(spec["paths"]["predictor_pattern"]).format(
        model=model,
        story=story,
    )
    brain_path = _brain_path(root, spec, story)
    if not predictor_path.is_file():
        raise FileNotFoundError(f"canonical LM target bundle is missing: {predictor_path}")
    if not brain_path.is_file():
        raise FileNotFoundError(f"canonical BOLD target bundle is missing: {brain_path}")
    with np.load(predictor_path, allow_pickle=True) as bundle:
        required = {"lm_target", "tr_indices"}
        missing = sorted(required - set(bundle.files))
        if missing:
            raise ValueError(f"{predictor_path}: missing target fields {missing}")
        lm_target = np.asarray(bundle["lm_target"], dtype=np.float64)
        tr_indices = np.asarray(bundle["tr_indices"], dtype=np.int64)
        model_id = _scalar_string(bundle["model_id"] if "model_id" in bundle.files else None)
        bundle_story = _scalar_string(bundle["story"] if "story" in bundle.files else None)
    if lm_target.ndim != 2 or len(lm_target) != len(tr_indices):
        raise ValueError(f"{predictor_path}: LM target/time dimensions are inconsistent")
    if bundle_story and bundle_story != story:
        raise ValueError(f"{predictor_path}: story={bundle_story!r}, expected {story!r}")
    with np.load(brain_path, allow_pickle=True) as bundle:
        required = {"values", "tr_indices", "subject_ids", "censor_mask", "target_names"}
        missing = sorted(required - set(bundle.files))
        if missing:
            raise ValueError(f"{brain_path}: missing BOLD fields {missing}")
        values = np.asarray(bundle["values"], dtype=np.float64)
        brain_tr = np.asarray(bundle["tr_indices"], dtype=np.int64)
        subjects = np.asarray(bundle["subject_ids"], dtype=str)
        censor = np.asarray(bundle["censor_mask"], dtype=bool)
        target_names = np.asarray(bundle["target_names"], dtype=str)
    if values.ndim != 2 or len(values) != len(brain_tr) or len(values) != len(subjects):
        raise ValueError(f"{brain_path}: BOLD row dimensions are inconsistent")
    if censor.shape != (len(values),):
        raise ValueError(f"{brain_path}: censor mask length mismatch")
    lookup = {int(tr): index for index, tr in enumerate(tr_indices.tolist())}
    subject_targets: dict[str, np.ndarray] = {}
    for subject in sorted(np.unique(subjects).tolist()):
        target = np.full((len(tr_indices), values.shape[1]), np.nan, dtype=np.float64)
        for row in np.flatnonzero(subjects == subject):
            tr = int(brain_tr[row])
            if tr in lookup and not bool(censor[row]):
                target[lookup[tr]] = values[row]
        subject_targets[str(subject)] = target
    expected_subjects = int(spec["stories"][story]["expected_subjects"])
    expected_trs = int(spec["stories"][story]["expected_trs"])
    if len(subject_targets) != expected_subjects:
        raise ValueError(
            f"{story}: expected {expected_subjects} participants, found {len(subject_targets)}"
        )
    if len(tr_indices) != expected_trs:
        raise ValueError(f"{story}: expected {expected_trs} TRs, found {len(tr_indices)}")
    return {
        "model": model,
        "story": story,
        "model_id": model_id,
        "predictor_path": predictor_path,
        "brain_path": brain_path,
        "lm_target": lm_target,
        "tr_indices": tr_indices,
        "subject_targets": subject_targets,
        "target_names": target_names,
    }


def _slice_bank(bundle: Mapping[str, Any], bank: Mapping[str, Any], size: int) -> tuple[np.ndarray, np.ndarray]:
    """Slice pooled coordinates using an exact source-bank identity."""

    entry = bank["banks"][str(int(size))]
    expected = list(zip((int(x) for x in entry["layers"]), (int(x) for x in entry["feature_ids"]), strict=False))
    coordinates = {
        (int(layer), int(feature)): index
        for index, (layer, feature) in enumerate(
            zip(bundle["feature_layers"].tolist(), bundle["feature_ids"].tolist(), strict=False)
        )
    }
    missing = [pair for pair in expected if pair not in coordinates]
    if missing:
        raise AnalysisError(
            f"pooled predictor {bundle['path']} is missing {len(missing)} requested bank coordinates; "
            "token extraction must include the full source-bank union"
        )
    columns = np.asarray([coordinates[pair] for pair in expected], dtype=np.int64)
    values = np.asarray(bundle["predictors"][:, columns], dtype=np.float64)
    names = np.asarray(entry["feature_names"], dtype=str)
    if values.shape != (len(bundle["tr_indices"]), int(size)) or names.shape != (int(size),):
        raise ValueError("sliced pooled predictor shape does not match bank size")
    return values, names


def _feature_bank_file(
    config_path: Path,
    spec: Mapping[str, Any],
    model: str,
    story: str,
    *,
    fold: int | None = None,
    target_story: str | None = None,
) -> Path:
    if fold is not None:
        return bank_path(config_path, spec, model, story, fold=int(fold))
    if target_story is None:
        raise ValueError("cross-story bank requires target_story")
    return bank_path(
        config_path,
        spec,
        model,
        story,
        source_story=story,
        target_story=target_story,
    )


def _single_groups(size: int) -> list[np.ndarray]:
    return feature_groups(int(size), 1)


def _group_label(threshold: float) -> str:
    return f"threshold_{float(threshold):.2f}".replace(".", "p")


def _cluster_groups_for_variant(
    bank_record: Mapping[str, Any], variant: str, size: int
) -> tuple[list[np.ndarray], list[str], list[dict[str, Any]]]:
    """Return discovery-cluster groups in the fitted variant's columns.

    For the full source-variance bank, the existing grouped helper returns
    multi-member groups over the exact K=256 columns.  A correlation-pruned
    bank contains only one representative per discovery cluster; its grouped
    analysis therefore maps each cluster to the representative column that
    survived pruning.  Those groups are intentionally labelled
    ``representative_cluster`` because they test the fitted representative
    basis and must not be described as a redundant-member ablation.
    """

    if variant == "source_story_variance":
        # Caller supplies a threshold through the variant suffix for a
        # representative bank, while the source-variance condition computes
        # all three thresholds in its outer loop.  This branch is reached by
        # _cluster_groups_for_variant_with_threshold below.
        raise ValueError("source-variance grouped definitions require an explicit threshold")
    prefix = "source_story_correlation_pruned_"
    if not variant.startswith(prefix):
        raise ValueError(f"grouped definitions are unavailable for {variant}")
    raw = variant[len(prefix) :].replace("p", ".")
    threshold = float(raw)
    pruned = bank_record.get("correlation_pruned", {})
    key = next((candidate for candidate in pruned if np.isclose(float(candidate), threshold)), None)
    if key is None:
        raise KeyError(f"correlation-pruned threshold {threshold} is absent")
    representative_entry = pruned[key]["banks"][str(int(size))]
    representative_positions = {
        (int(layer), int(feature)): index
        for index, (layer, feature) in enumerate(
            zip(representative_entry["layers"], representative_entry["feature_ids"], strict=False)
        )
    }
    exact = pruned[key]["clusters_over_exact_source_variance_banks"][str(int(size))]
    groups: list[np.ndarray] = []
    labels: list[str] = []
    metadata: list[dict[str, Any]] = []
    for layer, clusters in exact.get("clusters_by_layer", {}).items():
        for cluster in clusters:
            members = [
                (int(layer), int(feature)) for feature in cluster.get("members", [])
            ]
            columns = sorted(
                representative_positions[identity]
                for identity in members
                if identity in representative_positions
            )
            if not columns:
                continue
            cluster_id = int(cluster.get("cluster_id", len(groups) + 1))
            group_id = f"layer_{int(layer)}_cluster_{cluster_id}"
            groups.append(np.asarray(columns, dtype=np.int64))
            labels.append(group_id)
            metadata.append(
                {
                    "group_id": group_id,
                    "layer": int(layer),
                    "cluster_id": cluster_id,
                    "representative_columns": [int(value) for value in columns],
                    "source_cluster_members": [int(feature) for _, feature in members],
                    "size": len(columns),
                    "threshold": threshold,
                    "group_type": "representative_cluster",
                }
            )
    if not groups:
        raise AnalysisError(
            f"{variant}: no discovery clusters map to the representative bank"
        )
    return groups, labels, metadata


def _cluster_groups_for_variant_with_threshold(
    bank_record: Mapping[str, Any], variant: str, size: int, threshold: float
) -> tuple[list[np.ndarray], list[str], list[dict[str, Any]]]:
    """Dispatch full-bank or representative-bank grouped definitions."""

    if variant == "source_story_variance":
        groups, metadata, labels = _cluster_groups(
            dict(bank_record), size=int(size), threshold=float(threshold)
        )
        return groups, labels, metadata
    return _cluster_groups_for_variant(bank_record, variant, size)


def _corr(a: np.ndarray, b: np.ndarray, size: int | None = None) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.size < 3:
        return {"spearman": float("nan"), "pearson": float("nan"), "topk_jaccard": float("nan")}
    top_k = max(1, min(32, round((size or len(a)) * 0.125)))
    return correspondence(a, b, top_k=top_k)


def _pad_profiles(rows: Sequence[np.ndarray], *, dtype: Any = np.float32) -> tuple[np.ndarray, np.ndarray]:
    max_len = max((len(np.asarray(row)) for row in rows), default=0)
    padded = np.full((len(rows), max_len), np.nan, dtype=dtype)
    counts = np.zeros(len(rows), dtype=np.int64)
    for index, row in enumerate(rows):
        values = np.asarray(row, dtype=dtype)
        counts[index] = len(values)
        padded[index, : len(values)] = values
    return padded, counts


def _write_arrays_and_summary(
    arrays_path: Path, summary_path: Path, arrays: Mapping[str, Any], summary: dict[str, Any]
) -> dict[str, Any]:
    arrays_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = arrays_path.with_name(f".{arrays_path.name}.tmp")
    np.savez_compressed(temporary, **arrays)
    generated = temporary if temporary.suffix == ".npz" else temporary.with_suffix(temporary.suffix + ".npz")
    os.replace(generated, arrays_path)
    summary["arrays_path"] = str(arrays_path)
    summary["arrays_sha256"] = sha256_file(arrays_path)
    atomic_write_json(summary_path, summary)
    return summary


def _base_fingerprint(
    config_path: Path,
    *,
    input_paths: Sequence[Path],
    extra: Mapping[str, Any],
) -> tuple[str, list[Path]]:
    code_paths = [
        HERE / "pooling_analysis.py",
        HERE / "pooling.py",
        HERE / "fitting.py",
        HERE / "grouped_reliance.py",
        HERE / "feature_banks.py",
        HERE / "modeling.py",
    ]
    return provenance_fingerprint(
        config_path, input_paths=input_paths, code_paths=code_paths, extra=dict(extra)
    ), code_paths


def _metric_fit(
    x: np.ndarray,
    y: np.ndarray,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    *,
    alphas: np.ndarray,
    metric: str,
    n_time: int,
    inner_folds: int,
    purge: int,
    groups: Sequence[np.ndarray],
) -> tuple[dict[str, Any], float, np.ndarray, int]:
    alpha, grid, used = select_alpha_on_outer_train(
        x,
        y,
        outer_train,
        alphas=alphas,
        metric=metric,
        n_time=n_time,
        inner_folds=inner_folds,
        purge_adjacent=purge,
    )
    train = global_finite_training_rows(y, outer_train)
    fit = fit_ridge(x, y, train, alpha)
    evaluation = evaluate_fit(fit, x[outer_test], y[outer_test], groups)
    return evaluation, float(alpha), np.asarray(grid, dtype=np.float64), int(used)


def _fold_metadata(
    fold: int,
    train: np.ndarray,
    test: np.ndarray,
    *,
    r_alpha: float,
    r2_alpha: float,
    r_inner: int,
    r2_inner: int,
) -> dict[str, Any]:
    return {
        "fold": int(fold),
        "train_rows": [int(x) for x in np.asarray(train).tolist()],
        "test_rows": [int(x) for x in np.asarray(test).tolist()],
        "r_selected_alpha": float(r_alpha),
        "r2_selected_alpha": float(r2_alpha),
        "r_inner_folds_used": int(r_inner),
        "r2_inner_folds_used": int(r2_inner),
    }


def _identity_null(
    brain: np.ndarray,
    lm: np.ndarray,
    names: np.ndarray,
    spec: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    return within_layer_identity_null(
        np.asarray(brain, dtype=np.float64),
        np.asarray(lm, dtype=np.float64),
        np.asarray(names, dtype=str),
        n_permutations=int(spec["nulls"]["feature_identity"]["permutations"]),
        seed=int(seed),
    )


def fit_within_pooling(
    config_path: Path,
    spec: Mapping[str, Any],
    model: str,
    story: str,
    pooling: str,
    variant: str,
    *,
    size: int = PRIMARY_SIZE,
    force: bool = False,
) -> dict[str, Any]:
    """Fit one pooled model/story/variant condition with participant-specific BOLD."""

    if size != PRIMARY_SIZE:
        raise ValueError("pooled analysis runner is fixed at K=256")
    bundle = load_pooled_predictor(config_path, spec, model, story, pooling)
    target = _load_targets_only(config_path, spec, model, story)
    _require_tr_alignment(bundle, target["tr_indices"], context=f"{model}/{story}/{pooling}")
    cv = spec["cross_validation"]
    n_folds = int(cv["outer_folds"])
    inner_folds = int(cv["inner_folds"])
    purge = int(cv["purge_adjacent_trs"])
    alphas = np.asarray(cv["alpha_grid"], dtype=np.float64)
    fold_rows = contiguous_fold_rows(len(bundle["tr_indices"]), n_folds, purge_adjacent_trs=purge)
    bank_files = [
        _feature_bank_file(config_path, spec, model, story, fold=index)
        for index in range(n_folds)
    ]
    input_paths = [bundle["path"], target["predictor_path"], target["brain_path"], *bank_files]
    fingerprint, code_paths = _base_fingerprint(
        config_path,
        input_paths=input_paths,
        extra={
            "analysis": "correspondence_pooling_within_observed",
            "model": model,
            "story": story,
            "pooling": pooling,
            "variant": variant,
            "size": int(size),
        },
    )
    output_dir = (
        _artifact_root(config_path, spec)
        / "pooling_analysis"
        / "observed"
        / "within_story"
        / story
        / model
        / pooling
        / variant
        / f"n{size}"
    )
    summary_path = output_dir / "summary.json"
    arrays_path = output_dir / "arrays.npz"
    if not force:
        cached = assert_cache_fingerprint(summary_path, fingerprint)
        if cached is not None:
            if not arrays_path.is_file():
                raise AnalysisError(f"observed cache exists without arrays: {summary_path}")
            return cached

    subject_ids = np.asarray(sorted(target["subject_targets"]), dtype=str)
    n_subjects = len(subject_ids)
    n_time = len(bundle["tr_indices"])
    n_parcels = next(iter(target["subject_targets"].values())).shape[1]
    lm_dims = target["lm_target"].shape[1]
    brain_r_reliance = np.full((n_folds, n_subjects, size), np.nan, dtype=np.float32)
    brain_r2_reliance = np.full_like(brain_r_reliance, np.nan)
    lm_r_reliance = np.full((n_folds, size), np.nan, dtype=np.float32)
    lm_r2_reliance = np.full_like(lm_r_reliance, np.nan)
    brain_r = np.full((n_folds, n_subjects), np.nan, dtype=np.float32)
    brain_r2 = np.full_like(brain_r, np.nan)
    brain_r2_oos = np.full_like(brain_r, np.nan)
    brain_r2_alpha = np.full((n_folds, n_subjects), np.nan, dtype=np.float64)
    brain_r_alpha = np.full_like(brain_r2_alpha, np.nan)
    lm_r = np.full(n_folds, np.nan, dtype=np.float32)
    lm_r2 = np.full_like(lm_r, np.nan)
    lm_r_alpha = np.full(n_folds, np.nan, dtype=np.float64)
    lm_r2_alpha = np.full_like(lm_r_alpha, np.nan)
    fold_names: list[list[str]] = []
    fold_summary: list[dict[str, Any]] = []
    fold_corr_r: list[dict[str, float]] = []
    fold_corr_r2: list[dict[str, float]] = []
    identity_nulls: list[np.ndarray] = []
    brain_groups_by_fold: dict[str, list[list[np.ndarray]]] = { _group_label(t): [] for t in GROUP_THRESHOLDS }
    lm_groups_by_fold: dict[str, list[np.ndarray]] = { _group_label(t): [] for t in GROUP_THRESHOLDS }
    grouped_labels_by_fold: dict[str, list[list[str]]] = { _group_label(t): [] for t in GROUP_THRESHOLDS }
    started = time.time()

    for fold, (outer_train, outer_test) in enumerate(fold_rows):
        bank_record, bank = _load_bank(bank_files[fold], variant, size)
        selection_rows = np.asarray(bank_record.get("selection_rows", []), dtype=np.int64)
        if not np.array_equal(selection_rows, np.asarray(outer_train, dtype=np.int64)):
            raise AnalysisError(
                f"{bank_files[fold]}: bank selection rows do not equal purged outer training rows"
            )
        x_base, names = _slice_bank(bundle, bank, size)
        fold_names.append(names.tolist())
        x_lm = x_base
        x_brain = lag_expand(x_base, spec["targets"]["brain"]["lags"])
        lm_groups = _single_groups(size)
        brain_groups = feature_groups(size, len(spec["targets"]["brain"]["lags"]))
        lm_eval_r, lm_alpha_r, _, lm_inner_r = _metric_fit(
            x_lm, target["lm_target"], outer_train, outer_test,
            alphas=alphas, metric="pearson_r", n_time=n_time,
            inner_folds=inner_folds, purge=purge, groups=lm_groups,
        )
        lm_eval_r2, lm_alpha_r2, _, lm_inner_r2 = _metric_fit(
            x_lm, target["lm_target"], outer_train, outer_test,
            alphas=alphas, metric="r2_oos", n_time=n_time,
            inner_folds=inner_folds, purge=purge, groups=lm_groups,
        )
        lm_r_reliance[fold] = np.asarray(lm_eval_r["reliance"], dtype=np.float32)
        lm_r2_reliance[fold] = np.asarray(lm_eval_r2["reliance"], dtype=np.float32)
        lm_r[fold] = float(lm_eval_r["mean_r"])
        lm_r2[fold] = float(lm_eval_r2["mean_r"])
        lm_r_alpha[fold] = lm_alpha_r
        lm_r2_alpha[fold] = lm_alpha_r2

        fold_brain_r2_profiles: list[np.ndarray] = []
        # Grouped profiles are defined over discovery clusters.  For the full
        # source-variance bank these groups contain all members; for a pruned
        # representative bank they contain the surviving representative(s)
        # and are labelled as such in the summary.
        group_defs: dict[str, tuple[list[np.ndarray], list[str], list[dict[str, Any]]]] = {}
        if variant == "source_story_variance" or variant.startswith("source_story_correlation_pruned_"):
            grouped_thresholds = GROUP_THRESHOLDS if variant == "source_story_variance" else (float(variant.rsplit("_", 1)[-1].replace("p", ".")),)
            for threshold in grouped_thresholds:
                groups, labels, metadata = _cluster_groups_for_variant_with_threshold(bank_record, variant, size, threshold)
                group_defs[_group_label(threshold)] = (groups, labels, metadata)
                brain_groups_by_fold[_group_label(threshold)].append([])
                lm_groups_by_fold[_group_label(threshold)].append(np.asarray([], dtype=np.float64))
                grouped_labels_by_fold[_group_label(threshold)].append(labels)

        subject_r_records: list[dict[str, Any]] = []
        subject_r2_records: list[dict[str, Any]] = []
        for subject_index, subject in enumerate(subject_ids.tolist()):
            y = target["subject_targets"][str(subject)]
            brain_eval_r, brain_alpha_r, _, brain_inner_r = _metric_fit(
                x_brain, y, outer_train, outer_test,
                alphas=alphas, metric="pearson_r", n_time=n_time,
                inner_folds=inner_folds, purge=purge, groups=brain_groups,
            )
            brain_eval_r2, brain_alpha_r2, _, brain_inner_r2 = _metric_fit(
                x_brain, y, outer_train, outer_test,
                alphas=alphas, metric="r2_oos", n_time=n_time,
                inner_folds=inner_folds, purge=purge, groups=brain_groups,
            )
            brain_r_reliance[fold, subject_index] = np.asarray(brain_eval_r["reliance"], dtype=np.float32)
            brain_r2_reliance[fold, subject_index] = np.asarray(brain_eval_r2["reliance"], dtype=np.float32)
            brain_r[fold, subject_index] = float(brain_eval_r["mean_r"])
            brain_r2[fold, subject_index] = float(brain_eval_r2["mean_r"])
            brain_r2_oos[fold, subject_index] = float(brain_eval_r2["mean_r2_oos"])
            brain_r_alpha[fold, subject_index] = brain_alpha_r
            brain_r2_alpha[fold, subject_index] = brain_alpha_r2
            fold_brain_r2_profiles.append(np.asarray(brain_eval_r2["reliance"], dtype=np.float64))
            subject_r_records.append({"subject": str(subject), "alpha": brain_alpha_r, "mean_r": brain_eval_r["mean_r"], "mean_r2_oos": brain_eval_r["mean_r2_oos"], "inner_folds_used": brain_inner_r})
            subject_r2_records.append({"subject": str(subject), "alpha": brain_alpha_r2, "mean_r": brain_eval_r2["mean_r"], "mean_r2_oos": brain_eval_r2["mean_r2_oos"], "inner_folds_used": brain_inner_r2})
            if group_defs:
                for label, (groups, _labels, _metadata) in group_defs.items():
                    if len(brain_groups_by_fold[label][fold]) == 0:
                        brain_groups_by_fold[label][fold] = []
                    group_columns = [
                        np.asarray(
                            [int(feature) * len(spec["targets"]["brain"]["lags"]) + lag for feature in group.tolist() for lag in range(len(spec["targets"]["brain"]["lags"]))],
                            dtype=np.int64,
                        )
                        for group in groups
                    ]
                    grouped_eval = evaluate_fit(
                        fit_ridge(x_brain, y, global_finite_training_rows(y, outer_train), brain_alpha_r2),
                        x_brain[outer_test], y[outer_test], group_columns,
                    )
                    brain_groups_by_fold[label][fold].append(np.asarray(grouped_eval["reliance"], dtype=np.float64))

        brain_profile_r = np.nanmean(brain_r_reliance[fold], axis=0)
        brain_profile_r2 = np.nanmean(brain_r2_reliance[fold], axis=0)
        corr_r = _corr(brain_profile_r, lm_r_reliance[fold], size)
        corr_r2 = _corr(brain_profile_r2, lm_r2_reliance[fold], size)
        fold_corr_r.append(corr_r)
        fold_corr_r2.append(corr_r2)
        identity = _identity_null(
            brain_profile_r2, lm_r2_reliance[fold], np.asarray(names, dtype=str), spec,
            int(spec["nulls"]["seeds"]["feature_identity"]) + fold,
        )
        identity_nulls.append(np.asarray(identity["null"], dtype=np.float32))
        # LM grouped reliance is evaluated with the r2-selected fit.  Refit
        # once here rather than changing the primary conditional profile.
        if group_defs:
            lm_fit_r2 = fit_ridge(
                x_lm,
                target["lm_target"],
                global_finite_training_rows(target["lm_target"], outer_train),
                lm_alpha_r2,
            )
            for label, (groups, _labels, _metadata) in group_defs.items():
                grouped_eval = evaluate_fit(lm_fit_r2, x_lm[outer_test], target["lm_target"][outer_test], groups)
                lm_groups_by_fold[label][fold] = np.asarray(grouped_eval["reliance"], dtype=np.float64)
        fold_summary.append(
            {
                **_fold_metadata(fold, outer_train, outer_test, r_alpha=lm_alpha_r, r2_alpha=lm_alpha_r2, r_inner=lm_inner_r, r2_inner=lm_inner_r2),
                "lm_mean_r_selected": float(lm_eval_r["mean_r"]),
                "lm_mean_r2_oos_selected": float(lm_eval_r2["mean_r2_oos"]),
                "brain_mean_r_selected": float(np.nanmean(brain_r[fold])),
                "brain_mean_r2_oos_selected": float(np.nanmean(brain_r2_oos[fold])),
                "correspondence_r_selected": corr_r,
                "correspondence_r2_selected": corr_r2,
                "feature_names": names.tolist(),
                "identity_null_r2_selected": {k: v for k, v in identity.items() if k != "null"},
                "brain_r_selected": subject_r_records,
                "brain_r2_selected": subject_r2_records,
            }
        )

    aggregate_r = {key: float(np.nanmean([row[key] for row in fold_corr_r])) for key in ("spearman", "pearson", "topk_jaccard")}
    aggregate_r2 = {key: float(np.nanmean([row[key] for row in fold_corr_r2])) for key in ("spearman", "pearson", "topk_jaccard")}
    arrays: dict[str, Any] = {
        "feature_names_by_fold": np.asarray(fold_names, dtype=str),
        "brain_r_reliance_by_fold": brain_r_reliance,
        "brain_r2_reliance_by_fold": brain_r2_reliance,
        "lm_r_reliance_by_fold": lm_r_reliance,
        "lm_r2_reliance_by_fold": lm_r2_reliance,
        "brain_mean_r_by_fold": brain_r,
        "brain_mean_r2_by_fold": brain_r2,
        "brain_mean_r2_oos_by_fold": brain_r2_oos,
        "lm_mean_r_by_fold": lm_r,
        "lm_mean_r2_by_fold": lm_r2,
        "brain_r_alpha_by_fold": brain_r_alpha,
        "brain_r2_alpha_by_fold": brain_r2_alpha,
        "lm_r_alpha_by_fold": lm_r_alpha,
        "lm_r2_alpha_by_fold": lm_r2_alpha,
        "identity_null_r2_by_fold": np.asarray(identity_nulls, dtype=np.float32),
        "tr_indices": np.asarray(bundle["tr_indices"], dtype=np.int64),
    }
    grouped_summary: dict[str, Any] = {}
    if group_defs:
        for label in sorted(group_defs):
            brain_rows = brain_groups_by_fold[label]
            # ``brain_rows[fold]`` is a list over subjects; pad both axes.
            max_groups = max((len(profile) for fold_rows_local in brain_rows for profile in fold_rows_local), default=0)
            brain_group_array = np.full((n_folds, n_subjects, max_groups), np.nan, dtype=np.float32)
            for fold, rows_local in enumerate(brain_rows):
                for subject_index, profile in enumerate(rows_local):
                    brain_group_array[fold, subject_index, : len(profile)] = np.asarray(profile, dtype=np.float32)
            lm_group_array, group_counts = _pad_profiles(lm_groups_by_fold[label])
            arrays[f"grouped_{label}_brain_reliance_by_fold"] = brain_group_array
            arrays[f"grouped_{label}_lm_reliance_by_fold"] = lm_group_array
            arrays[f"grouped_{label}_count_by_fold"] = group_counts
            grouped_corr = []
            for fold in range(n_folds):
                count = int(group_counts[fold])
                brain_profile = np.nanmean(brain_group_array[fold, :, :count], axis=0) if count else np.asarray([])
                grouped_corr.append(_corr(brain_profile, lm_group_array[fold, :count], count))
            grouped_summary[label] = {
                "threshold": float(label.replace("threshold_", "").replace("p", ".")),
                "n_groups_by_fold": group_counts.tolist(),
                "correspondence_by_fold": grouped_corr,
                "mean_correspondence": {key: float(np.nanmean([row[key] for row in grouped_corr])) for key in ("spearman", "pearson", "topk_jaccard")},
                "group_membership_fixed_within_fold": True,
                "model_variant": variant,
                "group_metadata_by_fold": [
                    [dict(item) for item in _cluster_groups_for_variant_with_threshold(
                        _load_bank(bank_files[fold], variant, size)[0],
                        variant,
                        size,
                        float(label.replace("threshold_", "").replace("p", ".")),
                    )[2]]
                    for fold in range(n_folds)
                ],
            }
    summary = {
        "schema_version": 1,
        "analysis_role": "correspondence_pooling_within_observed_fit",
        "analysis_fingerprint": fingerprint,
        "model_key": model,
        "model_id": target["model_id"],
        "story": story,
        "pooling": pooling,
        "pooling_semantics": bundle["pooling_semantics"],
        "feature_bank_variant": variant,
        "bank_size": int(size),
        "n_subjects": int(n_subjects),
        "n_time": int(n_time),
        "n_parcels": int(n_parcels),
        "lm_dimensions": int(lm_dims),
        "n_outer_folds": int(n_folds),
        "n_inner_folds": int(inner_folds),
        "purge_adjacent_trs": int(purge),
        "brain_lags": [int(x) for x in spec["targets"]["brain"]["lags"]],
        "selection_metrics": ["pearson_r", "r2_oos"],
        "alpha_grid": [float(x) for x in alphas.tolist()],
        "training_only_standardization": True,
        "primary_reliance_metric": "r2_selected_held_out_delta_r2_oos",
        "paired_feature_coordinates_within_fold": True,
        "feature_selection_outcome_free": True,
        "correspondence_aggregation": "mean_of_fold_level_correspondence",
        "folds": fold_summary,
        "correspondence_r_selected": {"folds": fold_corr_r, "mean": aggregate_r},
        "correspondence_r2_selected": {"folds": fold_corr_r2, "mean": aggregate_r2},
        "primary_correspondence": aggregate_r2,
        "identity_null_r2_selected": {
            "n_permutations_per_fold": int(spec["nulls"]["feature_identity"]["permutations"]),
            "complete": True,
        },
        "grouped_reliance": grouped_summary,
        "grouped_reliance_status": "complete" if group_defs else "not_applicable",
        "input_sha256": {str(path): sha256_file(path) for path in input_paths},
        "code_sha256": {str(path): sha256_file(path) for path in code_paths},
        "elapsed_seconds": float(time.time() - started),
        "confirmatory_eligible": False,
        "complete": True,
    }
    return _write_arrays_and_summary(arrays_path, summary_path, arrays, summary)


def _select_source_alpha(
    x: np.ndarray,
    y: np.ndarray,
    blocks: np.ndarray,
    *,
    alphas: np.ndarray,
    metric: str,
    purge: int,
) -> tuple[float, list[dict[str, Any]]]:
    selected: list[dict[str, Any]] = []
    for fold in sorted(np.unique(blocks).tolist()):
        alpha, scores = select_alpha(
            x, y, blocks, int(fold), alphas, metric=metric, purge_adjacent=purge
        )
        selected.append({"fold": int(fold), "alpha": float(alpha), "scores": np.asarray(scores, dtype=np.float64).tolist()})
    return modal_alpha(selected), selected


def fit_transfer_pooling(
    config_path: Path,
    spec: Mapping[str, Any],
    model: str,
    source_story: str,
    target_story: str,
    pooling: str,
    variant: str,
    *,
    size: int = PRIMARY_SIZE,
    force: bool = False,
) -> dict[str, Any]:
    """Fit source-story weights and evaluate unchanged on the target story."""

    if size != PRIMARY_SIZE:
        raise ValueError("pooled analysis runner is fixed at K=256")
    source_bundle = load_pooled_predictor(config_path, spec, model, source_story, pooling)
    target_bundle = load_pooled_predictor(config_path, spec, model, target_story, pooling)
    source = _load_targets_only(config_path, spec, model, source_story)
    target = _load_targets_only(config_path, spec, model, target_story)
    _require_tr_alignment(
        source_bundle,
        source["tr_indices"],
        context=f"{model}/{source_story}/{pooling}",
    )
    _require_tr_alignment(
        target_bundle,
        target["tr_indices"],
        context=f"{model}/{target_story}/{pooling}",
    )
    bank_file = _feature_bank_file(config_path, spec, model, source_story, target_story=target_story)
    reverse_bank_file = _feature_bank_file(config_path, spec, model, target_story, target_story=source_story)
    bank_record, bank = _load_bank(bank_file, variant, size)
    _load_bank(reverse_bank_file, variant, size)
    x_source, names = _slice_bank(source_bundle, bank, size)
    x_target, target_names = _slice_bank(target_bundle, bank, size)
    if not np.array_equal(names, target_names):
        raise AnalysisError("source and target pooled slices changed the source bank identity")
    input_paths = [source_bundle["path"], target_bundle["path"], source["predictor_path"], source["brain_path"], target["predictor_path"], target["brain_path"], bank_file, reverse_bank_file]
    fingerprint, code_paths = _base_fingerprint(
        config_path,
        input_paths=input_paths,
        extra={
            "analysis": "correspondence_pooling_cross_story_observed",
            "model": model,
            "source_story": source_story,
            "target_story": target_story,
            "pooling": pooling,
            "variant": variant,
            "size": int(size),
        },
    )
    label = f"{source_story}_to_{target_story}"
    output_dir = _artifact_root(config_path, spec) / "pooling_analysis" / "observed" / "cross_story" / label / model / pooling / variant / f"n{size}"
    summary_path = output_dir / "summary.json"
    arrays_path = output_dir / "arrays.npz"
    if not force:
        cached = assert_cache_fingerprint(summary_path, fingerprint)
        if cached is not None:
            if not arrays_path.is_file():
                raise AnalysisError(f"transfer cache exists without arrays: {summary_path}")
            return cached
    common_subjects = sorted(set(source["subject_targets"]) & set(target["subject_targets"]))
    expected_matched = int(spec["cross_validation"].get("expected_matched_subjects", len(common_subjects)))
    if len(common_subjects) != expected_matched:
        raise AnalysisError(f"expected {expected_matched} matched subjects, found {len(common_subjects)}")
    cv = spec["cross_validation"]
    alphas = np.asarray(cv["alpha_grid"], dtype=np.float64)
    purge = int(cv["purge_adjacent_trs"])
    n_folds = int(cv["outer_folds"])
    blocks = block_labels(len(x_source), n_folds)
    x_source_brain = lag_expand(x_source, spec["targets"]["brain"]["lags"])
    x_target_brain = lag_expand(x_target, spec["targets"]["brain"]["lags"])
    lm_groups = _single_groups(size)
    brain_groups = feature_groups(size, len(spec["targets"]["brain"]["lags"]))
    n_subjects = len(common_subjects)
    n_parcels = next(iter(target["subject_targets"].values())).shape[1]
    brain_r_reliance: list[np.ndarray] = []
    brain_r2_reliance: list[np.ndarray] = []
    brain_r: list[float] = []
    brain_r2: list[float] = []
    brain_r2_oos: list[float] = []
    brain_r_alpha: list[float] = []
    brain_r2_alpha: list[float] = []
    grouped: dict[str, dict[str, Any]] = {}
    source_group_defs: dict[str, tuple[list[np.ndarray], list[str], list[dict[str, Any]]]] = {}
    if variant == "source_story_variance" or variant.startswith("source_story_correlation_pruned_"):
        grouped_thresholds = GROUP_THRESHOLDS if variant == "source_story_variance" else (float(variant.rsplit("_", 1)[-1].replace("p", ".")),)
        for threshold in grouped_thresholds:
            groups, labels, metadata = _cluster_groups_for_variant_with_threshold(bank_record, variant, size, threshold)
            source_group_defs[_group_label(threshold)] = (groups, labels, metadata)
    started = time.time()
    lm_alpha_r, lm_cv_r = _select_source_alpha(x_source, source["lm_target"], blocks, alphas=alphas, metric="pearson_r", purge=purge)
    lm_alpha_r2, lm_cv_r2 = _select_source_alpha(x_source, source["lm_target"], blocks, alphas=alphas, metric="r2_oos", purge=purge)
    lm_fit_r = fit_ridge(x_source, source["lm_target"], np.flatnonzero(np.all(np.isfinite(source["lm_target"]), axis=1)), lm_alpha_r)
    lm_fit_r2 = fit_ridge(x_source, source["lm_target"], np.flatnonzero(np.all(np.isfinite(source["lm_target"]), axis=1)), lm_alpha_r2)
    lm_eval_r = evaluate_fit(lm_fit_r, x_target, target["lm_target"], lm_groups)
    lm_eval_r2 = evaluate_fit(lm_fit_r2, x_target, target["lm_target"], lm_groups)
    for subject in common_subjects:
        y_source = source["subject_targets"][subject]
        y_target = target["subject_targets"][subject]
        alpha_r, _cv_r = _select_source_alpha(x_source_brain, y_source, blocks, alphas=alphas, metric="pearson_r", purge=purge)
        alpha_r2, _cv_r2 = _select_source_alpha(x_source_brain, y_source, blocks, alphas=alphas, metric="r2_oos", purge=purge)
        result_r = fit_transfer(x_source_brain, y_source, x_target_brain, y_target, alpha=alpha_r, groups=brain_groups)
        result_r2 = fit_transfer(x_source_brain, y_source, x_target_brain, y_target, alpha=alpha_r2, groups=brain_groups)
        brain_r_reliance.append(np.asarray(result_r["reliance"], dtype=np.float64))
        brain_r2_reliance.append(np.asarray(result_r2["reliance"], dtype=np.float64))
        brain_r.append(float(result_r["mean_r"]))
        brain_r2.append(float(result_r2["mean_r"]))
        brain_r2_oos.append(float(result_r2["mean_r2_oos"]))
        brain_r_alpha.append(float(alpha_r))
        brain_r2_alpha.append(float(alpha_r2))
        if source_group_defs:
            for label_name, (groups, labels, metadata) in source_group_defs.items():
                group_columns = [
                    np.asarray([int(feature) * len(spec["targets"]["brain"]["lags"]) + lag for feature in group.tolist() for lag in range(len(spec["targets"]["brain"]["lags"]))], dtype=np.int64)
                    for group in groups
                ]
                result_group = fit_transfer(x_source_brain, y_source, x_target_brain, y_target, alpha=alpha_r2, groups=group_columns)
                grouped.setdefault(label_name, {"brain": [], "labels": labels, "metadata": metadata})["brain"].append(np.asarray(result_group["reliance"], dtype=np.float64))
    brain_profile_r = np.nanmean(np.asarray(brain_r_reliance), axis=0)
    brain_profile_r2 = np.nanmean(np.asarray(brain_r2_reliance), axis=0)
    corr_r = _corr(brain_profile_r, lm_eval_r["reliance"], size)
    corr_r2 = _corr(brain_profile_r2, lm_eval_r2["reliance"], size)
    identity = _identity_null(brain_profile_r2, lm_eval_r2["reliance"], names, spec, int(spec["nulls"]["seeds"]["feature_identity"]))
    arrays: dict[str, Any] = {
        "feature_names": np.asarray(names, dtype=str),
        "subject_ids": np.asarray(common_subjects, dtype=str),
        "brain_r_reliance": np.asarray(brain_r_reliance, dtype=np.float32),
        "brain_r2_reliance": np.asarray(brain_r2_reliance, dtype=np.float32),
        "lm_r_reliance": np.asarray(lm_eval_r["reliance"], dtype=np.float32),
        "lm_r2_reliance": np.asarray(lm_eval_r2["reliance"], dtype=np.float32),
        "brain_r_mean_r": np.asarray(brain_r, dtype=np.float32),
        "brain_r2_mean_r": np.asarray(brain_r2, dtype=np.float32),
        "brain_r2_mean_r2_oos": np.asarray(brain_r2_oos, dtype=np.float32),
        "brain_r_alpha": np.asarray(brain_r_alpha, dtype=np.float64),
        "brain_r2_alpha": np.asarray(brain_r2_alpha, dtype=np.float64),
        "lm_r_alpha": np.asarray([lm_alpha_r], dtype=np.float64),
        "lm_r2_alpha": np.asarray([lm_alpha_r2], dtype=np.float64),
        "identity_null_r2": np.asarray(identity["null"], dtype=np.float32),
    }
    grouped_summary: dict[str, Any] = {}
    for label_name, entry in grouped.items():
        padded, counts = _pad_profiles(entry["brain"])
        lm_groups_def = source_group_defs[label_name][0]
        lm_group_eval = evaluate_fit(lm_fit_r2, x_target, target["lm_target"], lm_groups_def)
        arrays[f"grouped_{label_name}_brain_reliance"] = padded
        arrays[f"grouped_{label_name}_count"] = counts
        arrays[f"grouped_{label_name}_lm_reliance"] = np.asarray(lm_group_eval["reliance"], dtype=np.float32)
        brain_group_profile = np.nanmean(padded[:, : int(np.max(counts))], axis=0) if len(padded) else np.asarray([])
        grouped_corr = _corr(brain_group_profile, lm_group_eval["reliance"], len(lm_group_eval["reliance"]))
        grouped_summary[label_name] = {
            "threshold": float(label_name.replace("threshold_", "").replace("p", ".")),
            "n_groups": len(lm_group_eval["reliance"]),
            "correspondence": grouped_corr,
            "group_membership_fixed": True,
            "group_metadata": entry["metadata"],
            "labels": entry["labels"],
        }
    summary = {
        "schema_version": 1,
        "analysis_role": "correspondence_pooling_cross_story_observed_fit",
        "analysis_fingerprint": fingerprint,
        "model_key": model,
        "model_id": source["model_id"],
        "source_story": source_story,
        "target_story": target_story,
        "pooling": pooling,
        "pooling_semantics": source_bundle["pooling_semantics"],
        "feature_bank_variant": variant,
        "bank_size": int(size),
        "n_matched_subjects": int(n_subjects),
        "n_source_time": len(x_source),
        "n_target_time": len(x_target),
        "n_parcels": int(n_parcels),
        "lm_dimensions": int(target["lm_target"].shape[1]),
        "outer_folds_for_source_penalty": int(n_folds),
        "purge_adjacent_trs": int(purge),
        "brain_lags": [int(x) for x in spec["targets"]["brain"]["lags"]],
        "selection_metrics": ["pearson_r", "r2_oos"],
        "alpha_grid": [float(x) for x in alphas.tolist()],
        "training_only_standardization": True,
        "source_bank_selection_only": True,
        "target_story_excluded_from_bank_selection": True,
        "paired_targets_use_exact_same_feature_layer_coordinates": True,
        "source_scaling_and_weights_applied_unchanged": True,
        "primary_reliance_metric": "r2_selected_held_out_delta_r2_oos",
        "lm_r_selected": {"alpha": lm_alpha_r, "mean_r": lm_eval_r["mean_r"], "mean_r2_oos": lm_eval_r["mean_r2_oos"], "source_cv": lm_cv_r},
        "lm_r2_selected": {"alpha": lm_alpha_r2, "mean_r": lm_eval_r2["mean_r"], "mean_r2_oos": lm_eval_r2["mean_r2_oos"], "source_cv": lm_cv_r2},
        "brain_r_selected": {"mean_r": float(np.nanmean(brain_r)), "mean_r2_oos": float(np.nanmean([float(x) for x in brain_r2_oos])), "alphas": brain_r_alpha},
        "brain_r2_selected": {"mean_r": float(np.nanmean(brain_r2)), "mean_r2_oos": float(np.nanmean(brain_r2_oos)), "alphas": brain_r2_alpha},
        "held_out_reliance_correspondence": {"r_selected": corr_r, "r2_selected": corr_r2},
        "primary_correspondence": corr_r2,
        "identity_null_r2_selected": {"n_permutations": int(spec["nulls"]["feature_identity"]["permutations"]), "complete": True},
        "grouped_reliance": grouped_summary,
        "grouped_reliance_status": "complete" if source_group_defs else "not_applicable",
        "input_sha256": {str(path): sha256_file(path) for path in input_paths},
        "code_sha256": {str(path): sha256_file(path) for path in code_paths},
        "confirmatory_eligible": False,
        "complete": True,
        "elapsed_seconds": float(time.time() - started),
    }
    return _write_arrays_and_summary(arrays_path, summary_path, arrays, summary)


def _observed_path(
    config_path: Path, spec: Mapping[str, Any], role: str, model: str, story_or_label: str, pooling: str, variant: str, size: int
) -> Path:
    if role == "within_story":
        return _artifact_root(config_path, spec) / "pooling_analysis" / "observed" / "within_story" / story_or_label / model / pooling / variant / f"n{size}" / "summary.json"
    return _artifact_root(config_path, spec) / "pooling_analysis" / "observed" / "cross_story" / story_or_label / model / pooling / variant / f"n{size}" / "summary.json"


def _load_summary_arrays(summary_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    arrays_path = Path(str(summary.get("arrays_path", summary_path.with_name("arrays.npz"))))
    if not arrays_path.is_absolute():
        arrays_path = (summary_path.parent / arrays_path).resolve()
    if not arrays_path.is_file():
        raise FileNotFoundError(arrays_path)
    with np.load(arrays_path, allow_pickle=False) as bundle:
        arrays = {str(key): np.asarray(bundle[key]) for key in bundle.files}
    return summary, arrays


def run_within_temporal_null(
    config_path: Path,
    spec: Mapping[str, Any],
    model: str,
    story: str,
    pooling: str,
    variant: str,
    *,
    size: int = PRIMARY_SIZE,
    force: bool = False,
) -> dict[str, Any]:
    """Run 199 fully retuned LM target shifts for one within-story condition."""

    observed_path = _observed_path(config_path, spec, "within_story", model, story, pooling, variant, size)
    observed, observed_arrays = _load_summary_arrays(observed_path)
    bundle = load_pooled_predictor(config_path, spec, model, story, pooling)
    target = _load_targets_only(config_path, spec, model, story)
    _require_tr_alignment(bundle, target["tr_indices"], context=f"{model}/{story}/{pooling}")
    n_folds = int(spec["cross_validation"]["outer_folds"])
    purge = int(spec["cross_validation"]["purge_adjacent_trs"])
    inner_folds = int(spec["cross_validation"]["inner_folds"])
    alphas = np.asarray(spec["cross_validation"]["alpha_grid"], dtype=np.float64)
    offsets = fixed_shift_offsets(len(bundle["tr_indices"]), int(spec["nulls"]["temporal_target"]["shifts"]), int(spec["nulls"]["temporal_target"]["minimum_shift_trs"]), int(spec["nulls"]["seeds"]["temporal_target"]))
    fold_rows = contiguous_fold_rows(len(bundle["tr_indices"]), n_folds, purge_adjacent_trs=purge)
    bank_files = [_feature_bank_file(config_path, spec, model, story, fold=fold) for fold in range(n_folds)]
    input_paths = [bundle["path"], target["predictor_path"], target["brain_path"], observed_path, *bank_files]
    fingerprint, code_paths = _base_fingerprint(config_path, input_paths=input_paths, extra={"analysis": "correspondence_pooling_within_temporal_null", "model": model, "story": story, "pooling": pooling, "variant": variant, "size": int(size), "offsets": offsets.tolist()})
    label = story
    output_dir = _artifact_root(config_path, spec) / "pooling_analysis" / "nulls" / "within_story" / label / model / pooling / variant / f"n{size}"
    summary_path = output_dir / "summary.json"
    arrays_path = output_dir / "arrays.npz"
    if not force:
        cached = assert_cache_fingerprint(summary_path, fingerprint)
        if cached is not None:
            if not arrays_path.is_file():
                raise AnalysisError(f"temporal-null cache exists without arrays: {summary_path}")
            return cached
    observed_brain = np.asarray(observed_arrays["brain_r2_reliance_by_fold"], dtype=np.float64)
    if observed_brain.shape[:2] != (n_folds, len(target["subject_targets"])):
        raise ValueError("observed pooled within artifact has an unexpected BOLD reliance shape")
    null_corr = np.full((len(offsets), n_folds, 3), np.nan, dtype=np.float64)
    null_reliance = np.full((len(offsets), n_folds, size), np.nan, dtype=np.float32)
    selected = np.full((len(offsets), n_folds), np.nan, dtype=np.float64)
    grouped_null: dict[str, np.ndarray] = {}
    grouped_brain: dict[str, np.ndarray] = {}
    grouped_counts: dict[str, np.ndarray] = {}
    for label_name in observed.get("grouped_reliance", {}):
        grouped_null[label_name] = np.full((len(offsets), n_folds, 3), np.nan, dtype=np.float64)
        key = f"grouped_{label_name}_brain_reliance_by_fold"
        count_key = f"grouped_{label_name}_count_by_fold"
        if key in observed_arrays and count_key in observed_arrays:
            grouped_brain[label_name] = np.asarray(observed_arrays[key], dtype=np.float64)
            grouped_counts[label_name] = np.asarray(observed_arrays[count_key], dtype=np.int64)
    started = time.time()
    for shift_index, shift in enumerate(offsets.tolist()):
        shifted_lm = np.roll(target["lm_target"], int(shift), axis=0)
        for fold, (outer_train, outer_test) in enumerate(fold_rows):
            bank_record, bank = _load_bank(bank_files[fold], variant, size)
            x_base, _ = _slice_bank(bundle, bank, size)
            alpha, _, _ = select_alpha_on_outer_train(x_base, shifted_lm, outer_train, alphas=alphas, metric="r2_oos", n_time=len(x_base), inner_folds=inner_folds, purge_adjacent=purge)
            selected[shift_index, fold] = alpha
            fit = fit_ridge(x_base, shifted_lm, global_finite_training_rows(shifted_lm, outer_train), alpha)
            evaluation = evaluate_fit(fit, x_base[outer_test], shifted_lm[outer_test], _single_groups(size))
            null_reliance[shift_index, fold] = np.asarray(evaluation["reliance"], dtype=np.float32)
            brain_profile = np.nanmean(observed_brain[fold], axis=0)
            corr = _corr(brain_profile, evaluation["reliance"], size)
            null_corr[shift_index, fold] = [corr["spearman"], corr["pearson"], corr["topk_jaccard"]]
            if grouped_null:
                for label_name in grouped_null:
                    groups, _, _ = _cluster_groups_for_variant_with_threshold(
                        bank_record,
                        variant,
                        size,
                        float(label_name.replace("threshold_", "").replace("p", ".")),
                    )
                    group_eval = evaluate_fit(fit, x_base[outer_test], shifted_lm[outer_test], groups)
                    count = int(grouped_counts[label_name][fold])
                    if count >= 3:
                        fixed = np.nanmean(grouped_brain[label_name][fold, :, :count], axis=0)
                        group_corr = _corr(fixed, group_eval["reliance"], count)
                        grouped_null[label_name][shift_index, fold] = [group_corr["spearman"], group_corr["pearson"], group_corr["topk_jaccard"]]
    arrays: dict[str, Any] = {"offsets": offsets, "null_correspondence_by_fold": null_corr.astype(np.float32), "null_lm_reliance_by_fold": null_reliance, "selected_alphas": selected.astype(np.float32)}
    for label_name, values in grouped_null.items():
        arrays[f"grouped_{label_name}_null_correspondence_by_fold"] = values.astype(np.float32)
    summary = {
        "schema_version": 1,
        "analysis_role": "correspondence_pooling_within_temporal_target_null",
        "analysis_fingerprint": fingerprint,
        "model_key": model,
        "story": story,
        "pooling": pooling,
        "feature_bank_variant": variant,
        "bank_size": int(size),
        "n_shifts": len(offsets),
        "offsets": offsets.tolist(),
        "minimum_shift_trs": int(spec["nulls"]["temporal_target"]["minimum_shift_trs"]),
        "target_shifted": "lm_target_only",
        "feature_bank_reselected": False,
        "penalty_selection_refit_per_shift": True,
        "brain_profile_fixed_from_observed_fit": True,
        "observed_correspondence": observed.get("primary_correspondence", {}),
        "null_mean": {key: float(np.nanmean(null_corr[:, :, index])) for index, key in enumerate(("spearman", "pearson", "topk_jaccard"))},
        "null_sd": {key: float(np.nanstd(null_corr[:, :, index], ddof=1)) for index, key in enumerate(("spearman", "pearson", "topk_jaccard"))},
        "grouped_nulls": {label_name: {"n_shifts": len(offsets), "retuned": True} for label_name in grouped_null},
        "source_observed_summary": str(observed_path),
        "input_sha256": {str(path): sha256_file(path) for path in input_paths},
        "code_sha256": {str(path): sha256_file(path) for path in code_paths},
        "elapsed_seconds": float(time.time() - started),
        "complete": True,
    }
    return _write_arrays_and_summary(arrays_path, summary_path, arrays, summary)


def run_transfer_temporal_null(
    config_path: Path,
    spec: Mapping[str, Any],
    model: str,
    source_story: str,
    target_story: str,
    pooling: str,
    variant: str,
    *,
    size: int = PRIMARY_SIZE,
    force: bool = False,
) -> dict[str, Any]:
    """Run retuned source-target LM shifts for one transfer direction."""

    label = f"{source_story}_to_{target_story}"
    observed_path = _observed_path(config_path, spec, "cross_story", model, label, pooling, variant, size)
    observed, observed_arrays = _load_summary_arrays(observed_path)
    source_bundle = load_pooled_predictor(config_path, spec, model, source_story, pooling)
    target_bundle = load_pooled_predictor(config_path, spec, model, target_story, pooling)
    source = _load_targets_only(config_path, spec, model, source_story)
    target = _load_targets_only(config_path, spec, model, target_story)
    _require_tr_alignment(
        source_bundle,
        source["tr_indices"],
        context=f"{model}/{source_story}/{pooling}",
    )
    _require_tr_alignment(
        target_bundle,
        target["tr_indices"],
        context=f"{model}/{target_story}/{pooling}",
    )
    bank_file = _feature_bank_file(config_path, spec, model, source_story, target_story=target_story)
    bank_record, bank = _load_bank(bank_file, variant, size)
    x_source, _ = _slice_bank(source_bundle, bank, size)
    x_target, _ = _slice_bank(target_bundle, bank, size)
    n_folds = int(spec["cross_validation"]["outer_folds"])
    purge = int(spec["cross_validation"]["purge_adjacent_trs"])
    alphas = np.asarray(spec["cross_validation"]["alpha_grid"], dtype=np.float64)
    blocks = block_labels(len(x_source), n_folds)
    offsets = fixed_shift_offsets(len(x_source), int(spec["nulls"]["temporal_target"]["shifts"]), int(spec["nulls"]["temporal_target"]["minimum_shift_trs"]), int(spec["nulls"]["seeds"]["temporal_target"]))
    input_paths = [source_bundle["path"], target_bundle["path"], source["predictor_path"], source["brain_path"], target["predictor_path"], target["brain_path"], bank_file, observed_path]
    fingerprint, code_paths = _base_fingerprint(config_path, input_paths=input_paths, extra={"analysis": "correspondence_pooling_cross_story_temporal_null", "model": model, "source_story": source_story, "target_story": target_story, "pooling": pooling, "variant": variant, "size": int(size), "offsets": offsets.tolist()})
    output_dir = _artifact_root(config_path, spec) / "pooling_analysis" / "nulls" / "cross_story" / label / model / pooling / variant / f"n{size}"
    summary_path = output_dir / "summary.json"
    arrays_path = output_dir / "arrays.npz"
    if not force:
        cached = assert_cache_fingerprint(summary_path, fingerprint)
        if cached is not None:
            if not arrays_path.is_file():
                raise AnalysisError(f"transfer null cache exists without arrays: {summary_path}")
            return cached
    observed_brain = np.asarray(observed_arrays["brain_r2_reliance"], dtype=np.float64)
    observed_lm = np.asarray(observed_arrays["lm_r2_reliance"], dtype=np.float64)
    brain_profile = np.nanmean(observed_brain, axis=0)
    observed_corr = _corr(brain_profile, observed_lm, size)
    null_corr = np.full((len(offsets), 3), np.nan, dtype=np.float64)
    null_reliance = np.full((len(offsets), size), np.nan, dtype=np.float32)
    selected = np.full((len(offsets), n_folds), np.nan, dtype=np.float64)
    grouped_null: dict[str, np.ndarray] = {}
    grouped_brain: dict[str, np.ndarray] = {}
    for label_name in observed.get("grouped_reliance", {}):
        key = f"grouped_{label_name}_brain_reliance"
        if key in observed_arrays:
            grouped_brain[label_name] = np.asarray(observed_arrays[key], dtype=np.float64)
            grouped_null[label_name] = np.full((len(offsets), 3), np.nan, dtype=np.float64)
    started = time.time()
    for shift_index, shift in enumerate(offsets.tolist()):
        shifted_source = np.roll(source["lm_target"], int(shift), axis=0)
        fold_selected: list[dict[str, Any]] = []
        for fold in range(n_folds):
            alpha, _ = select_alpha(x_source, shifted_source, blocks, fold, alphas, metric="r2_oos", purge_adjacent=purge)
            selected[shift_index, fold] = alpha
            fold_selected.append({"alpha": float(alpha)})
        alpha = modal_alpha(fold_selected)
        fit = fit_ridge(x_source, shifted_source, np.flatnonzero(np.all(np.isfinite(shifted_source), axis=1)), alpha)
        evaluation = evaluate_fit(fit, x_target, target["lm_target"], _single_groups(size))
        null_reliance[shift_index] = np.asarray(evaluation["reliance"], dtype=np.float32)
        corr = _corr(brain_profile, evaluation["reliance"], size)
        null_corr[shift_index] = [corr["spearman"], corr["pearson"], corr["topk_jaccard"]]
        for label_name in grouped_null:
            threshold = float(label_name.replace("threshold_", "").replace("p", "."))
            groups, _, _ = _cluster_groups_for_variant_with_threshold(
                bank_record, variant, size, threshold
            )
            group_eval = evaluate_fit(fit, x_target, target["lm_target"], groups)
            fixed = np.nanmean(grouped_brain[label_name], axis=0)
            group_corr = _corr(fixed, group_eval["reliance"], len(group_eval["reliance"]))
            grouped_null[label_name][shift_index] = [group_corr["spearman"], group_corr["pearson"], group_corr["topk_jaccard"]]
    arrays: dict[str, Any] = {"offsets": offsets, "null_correspondence": null_corr.astype(np.float32), "null_lm_reliance": null_reliance, "selected_alphas": selected.astype(np.float32)}
    for label_name, values in grouped_null.items():
        arrays[f"grouped_{label_name}_null_correspondence"] = values.astype(np.float32)
    summary = {
        "schema_version": 1,
        "analysis_role": "correspondence_pooling_cross_story_temporal_target_null",
        "analysis_fingerprint": fingerprint,
        "model_key": model,
        "source_story": source_story,
        "target_story": target_story,
        "pooling": pooling,
        "feature_bank_variant": variant,
        "bank_size": int(size),
        "n_shifts": len(offsets),
        "offsets": offsets.tolist(),
        "minimum_shift_trs": int(spec["nulls"]["temporal_target"]["minimum_shift_trs"]),
        "targets_shifted": "source_lm_target_only",
        "feature_bank_reselected": False,
        "penalty_selection_refit_per_shift": True,
        "brain_profile_fixed_from_observed_transfer": True,
        "observed_correspondence": observed_corr,
        "null_mean": {key: float(np.nanmean(null_corr[:, index])) for index, key in enumerate(("spearman", "pearson", "topk_jaccard"))},
        "null_sd": {key: float(np.nanstd(null_corr[:, index], ddof=1)) for index, key in enumerate(("spearman", "pearson", "topk_jaccard"))},
        "grouped_nulls": {label_name: {"n_shifts": len(offsets), "retuned": True} for label_name in grouped_null},
        "source_observed_summary": str(observed_path),
        "input_sha256": {str(path): sha256_file(path) for path in input_paths},
        "code_sha256": {str(path): sha256_file(path) for path in code_paths},
        "elapsed_seconds": float(time.time() - started),
        "complete": True,
    }
    return _write_arrays_and_summary(arrays_path, summary_path, arrays, summary)


def build_analysis_task_plan(spec: Mapping[str, Any], *, models: Sequence[str] | None = None, poolings: Sequence[str] = DEFAULT_POOLINGS, variants: Sequence[str] = DEFAULT_VARIANTS) -> dict[str, Any]:
    """Return a deterministic compute decomposition without executing it."""

    selected_models = tuple(str(x) for x in (models or spec["models"].keys()))
    selected_poolings = tuple(str(x) for x in poolings)
    selected_variants = tuple(str(x) for x in variants)
    n_models = len(selected_models)
    n_poolings = len(selected_poolings)
    n_variants = len(selected_variants)
    within = n_models * 2 * n_poolings * n_variants
    transfer = n_models * 2 * n_poolings * n_variants
    shifts = int(spec["nulls"]["temporal_target"]["shifts"])
    # Full source-variance banks expose three thresholded, multi-member
    # clusters.  Each pruned bank exposes one representative-cluster family
    # at its own threshold; these are retained as a distinct, conservative
    # grouped diagnostic rather than silently treating omitted members as
    # ablated.
    grouped_variants = sum(
        3 if variant == "source_story_variance" else 1
        for variant in selected_variants
        if variant == "source_story_variance" or variant.startswith("source_story_correlation_pruned_")
    )
    grouped_within = n_models * 2 * n_poolings * grouped_variants
    grouped_transfer = n_models * 2 * n_poolings * grouped_variants
    return {
        "status": "planned_only",
        "models": list(selected_models),
        "poolings": list(selected_poolings),
        "variants": list(selected_variants),
        "bank_size": PRIMARY_SIZE,
        "observed_within_conditions": within,
        "observed_transfer_conditions": transfer,
        "observed_conditions_total": within + transfer,
        "feature_level_temporal_null_conditions": within + transfer,
        "feature_level_temporal_refits": (within + transfer) * shifts,
        "grouped_within_conditions": grouped_within,
        "grouped_transfer_conditions": grouped_transfer,
        "grouped_temporal_null_conditions": grouped_within + grouped_transfer,
        "grouped_temporal_refits": (grouped_within + grouped_transfer) * shifts,
        "total_temporal_null_shift_fits": (within + transfer + grouped_within + grouped_transfer) * shifts,
        "no_jobs_submitted": True,
        "fail_closed_if_token_records_or_pooled_bundles_missing": True,
    }


def _selected_models(spec: Mapping[str, Any], args: Any) -> list[str]:
    values = getattr(args, "models", None)
    models = [str(x) for x in (values or spec["models"].keys())]
    unknown = sorted(set(models) - set(spec["models"]))
    if unknown:
        raise KeyError(f"unknown models: {unknown}")
    return models


def _selected_poolings(args: Any) -> tuple[str, ...]:
    values = getattr(args, "poolings", None)
    poolings = tuple(str(x) for x in (values or DEFAULT_POOLINGS))
    unknown = sorted(set(poolings) - set(DEFAULT_POOLINGS))
    if unknown:
        raise KeyError(f"unknown poolings: {unknown}")
    return poolings


def _selected_variants(args: Any) -> tuple[str, ...]:
    values = getattr(args, "variants", None)
    variants = tuple(str(x) for x in (values or DEFAULT_VARIANTS))
    unknown = sorted(set(variants) - set(DEFAULT_VARIANTS))
    if unknown:
        raise KeyError(f"unknown pooled-analysis variants: {unknown}")
    return variants


def _manifest(config_path: Path, spec: Mapping[str, Any], role: str, summaries: Sequence[Mapping[str, Any]]) -> Path:
    root = _artifact_root(config_path, spec) / "pooling_analysis"
    path = root / f"{role}_manifest.json"
    atomic_write_json(path, {
        "schema_version": 1,
        "analysis_role": f"correspondence_pooling_{role}_manifest",
        "conditions": [
            {key: row.get(key) for key in ("analysis_role", "model_key", "story", "source_story", "target_story", "pooling", "feature_bank_variant", "bank_size", "analysis_fingerprint", "arrays_path")}
            for row in summaries
        ],
        "n_conditions": len(summaries),
        "complete": True,
    })
    return path


def run(*, config_path: Path, spec: Mapping[str, Any], args: Any) -> dict[str, Any]:
    mode = str(getattr(args, "mode", "plan"))
    models = _selected_models(spec, args)
    poolings = _selected_poolings(args)
    variants = _selected_variants(args)
    force = bool(getattr(args, "force", False))
    plan = build_analysis_task_plan(spec, models=models, poolings=poolings, variants=variants)
    if mode == "plan":
        return {"stage": "pooling_analysis", "plan": plan}
    observed: list[dict[str, Any]] = []
    nulls: list[dict[str, Any]] = []
    if mode in {"observed", "all"}:
        for model in models:
            for story in spec["stories"]:
                for pooling in poolings:
                    for variant in variants:
                        observed.append(fit_within_pooling(config_path, spec, model, story, pooling, variant, force=force))
                for source_story, target_story in (("shapesphysical", "shapessocial"), ("shapessocial", "shapesphysical")):
                    for pooling in poolings:
                        for variant in variants:
                            observed.append(fit_transfer_pooling(config_path, spec, model, source_story, target_story, pooling, variant, force=force))
        _manifest(config_path, spec, "observed", observed)
    if mode in {"nulls", "all"}:
        for model in models:
            for story in spec["stories"]:
                for pooling in poolings:
                    for variant in variants:
                        nulls.append(run_within_temporal_null(config_path, spec, model, story, pooling, variant, force=force))
                for source_story, target_story in (("shapesphysical", "shapessocial"), ("shapessocial", "shapesphysical")):
                    for pooling in poolings:
                        for variant in variants:
                            nulls.append(run_transfer_temporal_null(config_path, spec, model, source_story, target_story, pooling, variant, force=force))
        _manifest(config_path, spec, "nulls", nulls)
    return {
        "stage": "pooling_analysis",
        "mode": mode,
        "plan": plan,
        "observed_conditions": len(observed),
        "null_conditions": len(nulls),
        "observed_manifest": str(_artifact_root(config_path, spec) / "pooling_analysis" / "observed_manifest.json") if observed else None,
        "null_manifest": str(_artifact_root(config_path, spec) / "pooling_analysis" / "nulls_manifest.json") if nulls else None,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--mode", choices=("plan", "observed", "nulls", "all"), default="plan")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--poolings", nargs="+")
    parser.add_argument("--variants", nargs="+")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.data_root is not None:
        os.environ["FMRI_SAE_DATA_ROOT"] = str(args.data_root.resolve())
    if args.output_root is not None:
        os.environ["FMRI_SAE_OUTPUT_ROOT"] = str(args.output_root.resolve())
    config_path = (args.config or default_config_path()).resolve()
    try:
        spec = load_spec(config_path)
        result = run(config_path=config_path, spec=spec, args=args)
    except (FileNotFoundError, KeyError, AnalysisError, RuntimeError, ValueError) as exc:
        print(f"pooling analysis failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
