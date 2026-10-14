#!/usr/bin/env python3
"""Fully retuned target-shift and identity-null analyses.

The temporal null in this module is deliberately a refit null.  A circular
shift is applied to the LM target, the nested ridge penalty is selected again
using only the shifted training target, and a fresh mapping is evaluated on
the held-out rows.  Feature banks are never rebuilt: their definition uses
the predictor matrix only and is therefore fixed for a target permutation.

This module consumes the observed-fit artifacts produced by ``fitting`` and
``transfer``. It does not alter the specified modeling code or the analysis
specification.  Missing observed artifacts are represented as explicit
``skipped`` records so an incomplete null cannot be mistaken for a completed
negative result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import default_config_path
from .feature_banks import contiguous_fold_rows  # noqa: E402
from .fitting import (  # noqa: E402
    DEFAULT_SIZES,
    _artifact_root,
    _extract_bank,
    _feature_bank_root,
    _load_bank,
    bank_path,
    evaluate_fit,
    load_condition,
    select_alpha_on_outer_train,
)
from .grouped_reliance import _corr as _grouped_correspondence  # noqa: E402
from .integrity import (  # noqa: E402
    assert_cache_fingerprint,
    atomic_write_json,
    provenance_fingerprint,
    sha256_file,
)
from .modeling import (  # noqa: E402
    block_labels,
    correspondence,
    feature_groups,
    fit_ridge,
    fit_transfer,
    modal_alpha,
    select_alpha,
)

HERE = Path(__file__).resolve().parent

DEFAULT_VARIANTS = (
    "source_story_variance",
    "source_story_prevalence",
    "source_story_correlation_pruned_0p70",
    "source_story_correlation_pruned_0p80",
    "source_story_correlation_pruned_0p90",
)

NULL_PARTS = ("all", "feature", "within-grouped", "cross-grouped")


def _selected_part(args: Any) -> str:
    """Return the requested operational resume part, defaulting to all."""

    part = str(getattr(args, "parts", None) or "all")
    if part not in NULL_PARTS:
        raise ValueError(f"unknown null run part {part!r}; choose one of {NULL_PARTS}")
    return part


def _selected_models(spec: dict[str, Any], args: Any) -> list[str]:
    values = getattr(args, "models", None)
    models = [str(value) for value in (values or spec["models"].keys())]
    for model in models:
        if model not in spec["models"]:
            raise KeyError(model)
    return models


def _selected_variants(spec: dict[str, Any], args) -> list[str]:
    values = getattr(args, "variants", None)
    variants = [str(value) for value in (values or DEFAULT_VARIANTS)]
    known = {"source_story_variance", "source_story_prevalence", "source_story_correlation_pruned_0p70", "source_story_correlation_pruned_0p80", "source_story_correlation_pruned_0p90", "historical_five_story_reciprocal_rank"}
    unknown = sorted(set(variants) - known)
    if unknown:
        raise KeyError(f"unknown null variants: {unknown}")
    return variants


def _variant_sizes(spec: dict[str, Any], variant: str, args: Any) -> tuple[int, ...]:
    supplied = getattr(args, "sizes", None)
    if supplied:
        return tuple(sorted({int(value) for value in supplied}))
    all_sizes = tuple(int(value) for value in spec["feature_bank"].get("sizes", DEFAULT_SIZES))
    if variant == "source_story_variance":
        return all_sizes
    if variant == "source_story_prevalence":
        return tuple(size for size in (256, 1024) if size in all_sizes)
    # The grouped/pruned product is specified at the primary size.  The
    # grouped module separately uses exact source-variance memberships.
    if variant.startswith("source_story_correlation_pruned"):
        return (int(spec["feature_bank"]["primary_size"]),)
    return (int(spec["feature_bank"]["primary_size"]),)


def fixed_shift_offsets(
    n_time: int,
    n_shifts: int,
    minimum_shift: int,
    seed: int,
) -> np.ndarray:
    """Return deterministic, unique, nontrivial circular shifts.

    Offsets use their canonical positive representation modulo ``n_time``.
    This prevents sampling both ``-d`` and ``n_time-d`` as if they were two
    different permutations.  Every selected offset has circular distance at
    least ``minimum_shift`` from the unshifted target.
    """

    n_time = int(n_time)
    n_shifts = int(n_shifts)
    minimum_shift = int(minimum_shift)
    if n_time < 2 or n_shifts < 1:
        raise ValueError("n_time and n_shifts must be positive")
    if minimum_shift < 1 or minimum_shift >= n_time:
        raise ValueError("minimum_shift must lie between 1 and n_time-1")
    candidates = np.arange(
        minimum_shift,
        n_time - minimum_shift + 1,
        dtype=np.int64,
    )
    rng = np.random.default_rng(int(seed))
    if n_shifts > len(candidates):
        raise ValueError(
            f"requested {n_shifts} unique shifts, but only {len(candidates)} "
            "satisfy the configured circular-distance threshold"
        )
    chosen = rng.choice(candidates, size=n_shifts, replace=False)
    return np.asarray(chosen, dtype=np.int64)


def fixed_independent_transfer_offsets(
    n_source: int,
    n_target: int,
    n_shifts: int,
    minimum_shift: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return deterministic, independently retimed source/target offsets.

    Cross-story transfer contains two LM targets.  Applying the same signed
    shift to both can preserve event alignment between the two Shapes
    descriptions.  This helper preserves each target's autocorrelation while
    breaking both predictor-target alignment and source-target co-retiming.
    The two offset vectors are unique within story and never use the same
    signed offset in a replicate.
    """

    source = fixed_shift_offsets(
        n_source, n_shifts, minimum_shift, _condition_seed(seed, "source")
    )
    for attempt in range(10_000):
        target = fixed_shift_offsets(
            n_target,
            n_shifts,
            minimum_shift,
            _condition_seed(seed, "target", attempt),
        )
        if np.all(source != target):
            return source, target
    raise RuntimeError("could not construct independent cross-story shift pairs")


def _output_dir(
    config_path: Path,
    spec: dict[str, Any],
    role: str,
    model: str,
    label: str,
    variant: str,
    size: int,
) -> Path:
    return (
        _artifact_root(config_path, spec)
        / "nulls"
        / role
        / label
        / model
        / variant
        / f"n{int(size)}"
    )


def _load_observed_summary_arrays(summary_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    arrays_path = Path(str(summary.get("arrays_path", summary_path.with_name("arrays.npz"))))
    if not arrays_path.is_absolute():
        arrays_path = (summary_path.parent / arrays_path).resolve()
    if not arrays_path.is_file():
        raise FileNotFoundError(arrays_path)
    with np.load(arrays_path, allow_pickle=True) as z:
        arrays = {str(key): np.asarray(z[key]) for key in z.files}
    return summary, arrays


def _fit_summary_path(
    config_path: Path,
    spec: dict[str, Any],
    model: str,
    story: str,
    variant: str,
    size: int,
) -> Path:
    return _artifact_root(config_path, spec) / "fit" / f"{model}_{story}" / variant / f"n{int(size)}" / "summary.json"


def _transfer_summary_path(
    config_path: Path,
    spec: dict[str, Any],
    model: str,
    source_story: str,
    target_story: str,
    variant: str,
    size: int,
) -> Path:
    return _artifact_root(config_path, spec) / "transfer" / f"{model}_{source_story}_to_{target_story}" / variant / f"n{int(size)}" / "summary.json"


def _identity_status(summary: dict[str, Any], arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Verify that the observed artifact actually contains the 10k null."""

    candidates = [
        arrays.get("identity_null_r2_by_fold"),
        arrays.get("identity_null_r2"),
    ]
    found = next((value for value in candidates if value is not None), None)
    expected = int(summary.get("identity_null_r2_selected", {}).get("n_permutations_per_fold", 0))
    if found is None:
        return {
            "status": "missing",
            "expected_per_fold": expected or 10000,
            "observed_per_fold": 0,
        }
    if found.ndim == 1:
        observed = int(found.shape[0])
    else:
        observed = int(found.shape[-1])
    return {
        "status": "verified" if observed == 10000 else "incomplete",
        "expected_per_fold": expected or 10000,
        "observed_per_fold": observed,
        "array_key": "identity_null_r2_by_fold" if arrays.get("identity_null_r2_by_fold") is not None else "identity_null_r2",
    }


def _aggregate_correspondence(brain: np.ndarray, lm: np.ndarray, size: int) -> dict[str, float]:
    return correspondence(
        np.asarray(brain, dtype=np.float64),
        np.asarray(lm, dtype=np.float64),
        top_k=max(1, min(32, int(round(float(size) * 0.125)))),
    )


def _condition_seed(seed: int, *parts: Any) -> int:
    """Derive a process-independent 32-bit seed for one condition."""

    payload = "|".join([str(int(seed)), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little", signed=False)


def _p_value(null: np.ndarray, observed: float) -> float:
    null = np.asarray(null, dtype=np.float64)
    finite = null[np.isfinite(null)]
    if not np.isfinite(observed) or len(finite) == 0:
        return float("nan")
    return float((np.sum(finite >= float(observed)) + 1) / (len(finite) + 1))


def _within_temporal_null(
    config_path: Path,
    spec: dict[str, Any],
    model: str,
    story: str,
    variant: str,
    size: int,
    *,
    n_shifts: int,
    seed: int,
    force: bool,
) -> dict[str, Any]:
    observed_summary_path = _fit_summary_path(config_path, spec, model, story, variant, size)
    observed, observed_arrays = _load_observed_summary_arrays(observed_summary_path)
    condition = load_condition(config_path, spec, model, story)
    n_time = len(condition["tr_indices"])
    offsets = fixed_shift_offsets(
        n_time,
        n_shifts,
        int(spec["nulls"]["temporal_target"]["minimum_shift_trs"]),
        int(seed),
    )
    output_dir = _output_dir(config_path, spec, "within_story", model, story, variant, size)
    summary_path = output_dir / "summary.json"
    arrays_path = output_dir / "arrays.npz"
    bank_paths = [bank_path(config_path, spec, model, story, fold=index) for index in range(int(spec["cross_validation"]["outer_folds"]))]
    source_paths = [condition["predictor_path"], condition["brain_path"], observed_summary_path, *bank_paths]
    code_paths = [HERE / "nulls.py", HERE / "fitting.py", HERE / "feature_banks.py", HERE / "modeling.py"]
    fingerprint = provenance_fingerprint(
        config_path,
        input_paths=source_paths,
        code_paths=code_paths,
        extra={
            "analysis": "correspondence_within_temporal_target_null",
            "model": model,
            "story": story,
            "variant": variant,
            "size": int(size),
            "n_shifts": int(n_shifts),
            "offsets": offsets.tolist(),
            "selection_metric": "r2_oos",
        },
    )
    if not force:
        cached = assert_cache_fingerprint(summary_path, fingerprint)
        if cached is not None:
            if not arrays_path.is_file():
                raise RuntimeError(f"null cache exists without arrays: {summary_path}")
            return cached

    n_folds = int(spec["cross_validation"]["outer_folds"])
    inner_folds = int(spec["cross_validation"]["inner_folds"])
    purge = int(spec["cross_validation"]["purge_adjacent_trs"])
    alphas = np.asarray(spec["cross_validation"]["alpha_grid"], dtype=np.float64)
    lm_groups = feature_groups(int(size), 1)
    fold_brain_observed = np.asarray(observed_arrays.get("brain_r2_reliance_by_fold"), dtype=np.float64)
    fold_lm_observed = np.asarray(observed_arrays.get("lm_r2_reliance_by_fold"), dtype=np.float64)
    if fold_brain_observed.shape[:2] != (n_folds, len(condition["subject_targets"])):
        raise ValueError(f"{observed_summary_path}: unexpected brain reliance shape {fold_brain_observed.shape}")
    if fold_lm_observed.shape != (n_folds, int(size)):
        raise ValueError(f"{observed_summary_path}: unexpected LM reliance shape {fold_lm_observed.shape}")
    observed_fold_corr: list[dict[str, float]] = []
    for fold in range(n_folds):
        brain_profile = np.nanmean(fold_brain_observed[fold], axis=0)
        observed_fold_corr.append(_aggregate_correspondence(brain_profile, fold_lm_observed[fold], size))
    observed_corr = {key: float(np.nanmean([row[key] for row in observed_fold_corr])) for key in ("spearman", "pearson", "topk_jaccard")}

    null_corr = np.full((len(offsets), 3), np.nan, dtype=np.float64)
    null_fold_corr = np.full((len(offsets), n_folds, 3), np.nan, dtype=np.float64)
    null_reliance = np.full((len(offsets), n_folds, int(size)), np.nan, dtype=np.float32)
    selected_alphas = np.full((len(offsets), n_folds), np.nan, dtype=np.float64)
    started = time.time()
    fold_rows = [
        (np.asarray(fold["train_rows"], dtype=np.int64), np.asarray(fold["test_rows"], dtype=np.int64))
        for fold in observed.get("folds", [])
    ]
    if len(fold_rows) != n_folds:
        fold_rows = contiguous_fold_rows(n_time, n_folds, purge_adjacent_trs=purge)
        fold_rows = [(train, test) for train, test in fold_rows]
    for shift_index, shift in enumerate(offsets.tolist()):
        shifted_target = np.roll(condition["lm_target"], int(shift), axis=0)
        for fold_index, (outer_train, outer_test) in enumerate(fold_rows):
            bank_record, bank = _load_bank(bank_paths[fold_index], variant, size)
            selection_rows = np.asarray(bank_record.get("selection_rows", []), dtype=np.int64)
            if not np.array_equal(selection_rows, outer_train):
                raise RuntimeError(f"{bank_paths[fold_index]}: fold bank selection does not match observed rows")
            x_fold, names = _extract_bank(condition, bank, size)
            alpha, _, _ = select_alpha_on_outer_train(
                x_fold,
                shifted_target,
                outer_train,
                alphas=alphas,
                metric="r2_oos",
                n_time=n_time,
                inner_folds=inner_folds,
                purge_adjacent=purge,
            )
            selected_alphas[shift_index, fold_index] = alpha
            lm_fit = fit_ridge(x_fold, shifted_target, outer_train, alpha)
            lm_eval = evaluate_fit(lm_fit, x_fold[outer_test], shifted_target[outer_test], lm_groups)
            brain_profile = np.nanmean(fold_brain_observed[fold_index], axis=0)
            corr = _aggregate_correspondence(brain_profile, lm_eval["reliance"], size)
            null_fold_corr[shift_index, fold_index] = [corr["spearman"], corr["pearson"], corr["topk_jaccard"]]
            null_reliance[shift_index, fold_index] = np.asarray(lm_eval["reliance"], dtype=np.float32)
        null_corr[shift_index] = np.nanmean(null_fold_corr[shift_index], axis=0)

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_path,
        offsets=offsets,
        observed_fold_spearman=np.asarray([row["spearman"] for row in observed_fold_corr], dtype=np.float32),
        observed_fold_pearson=np.asarray([row["pearson"] for row in observed_fold_corr], dtype=np.float32),
        observed_fold_topk_jaccard=np.asarray([row["topk_jaccard"] for row in observed_fold_corr], dtype=np.float32),
        null_correspondence=null_corr.astype(np.float32),
        null_fold_correspondence=null_fold_corr.astype(np.float32),
        null_lm_reliance=null_reliance,
        selected_alphas=selected_alphas.astype(np.float32),
    )
    identity = _identity_status(observed, observed_arrays)
    summary = {
        "schema_version": 1,
        "analysis_role": "correspondence_within_temporal_target_null",
        "analysis_fingerprint": fingerprint,
        "model_key": model,
        "story": story,
        "feature_bank_variant": variant,
        "bank_size": int(size),
        "n_shifts": int(len(offsets)),
        "offsets": offsets.tolist(),
        "minimum_shift_trs": int(spec["nulls"]["temporal_target"]["minimum_shift_trs"]),
        "target_shifted": "lm_target_only",
        "feature_bank_reselected": False,
        "penalty_selection_refit_per_shift": True,
        "selection_metric": "r2_oos",
        "brain_profile_fixed_from_observed_fit": True,
        "observed_correspondence": observed_corr,
        "null_mean": {key: float(np.nanmean(null_corr[:, index])) for index, key in enumerate(("spearman", "pearson", "topk_jaccard"))},
        "null_sd": {key: float(np.nanstd(null_corr[:, index], ddof=1)) for index, key in enumerate(("spearman", "pearson", "topk_jaccard"))},
        "p_value_one_sided": {key: _p_value(null_corr[:, index], observed_corr[key]) for index, key in enumerate(("spearman", "pearson", "topk_jaccard"))},
        "identity_null_verification": identity,
        "source_fit_summary": str(observed_summary_path),
        "arrays_path": str(arrays_path),
        "input_sha256": {str(path): sha256_file(path) for path in source_paths},
        "code_sha256": {str(path): sha256_file(path) for path in code_paths},
        "elapsed_seconds": float(time.time() - started),
        "complete": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary_path, summary)
    return summary


def _cross_temporal_null(
    config_path: Path,
    spec: dict[str, Any],
    model: str,
    source_story: str,
    target_story: str,
    variant: str,
    size: int,
    *,
    n_shifts: int,
    seed: int,
    force: bool,
) -> dict[str, Any]:
    observed_summary_path = _transfer_summary_path(config_path, spec, model, source_story, target_story, variant, size)
    observed, observed_arrays = _load_observed_summary_arrays(observed_summary_path)
    source = load_condition(config_path, spec, model, source_story)
    target = load_condition(config_path, spec, model, target_story)
    bank_file = _feature_bank_root(config_path, spec) / "cross_story" / model / f"{source_story}_to_{target_story}" / "banks.json"
    bank_record, bank = _load_bank(bank_file, variant, size)
    x_source, names = _extract_bank(source, bank, size)
    x_target, target_names = _extract_bank(target, bank, size)
    if not np.array_equal(names, target_names):
        raise RuntimeError("cross-story temporal null changed feature identity during target extraction")
    n_folds = int(spec["cross_validation"]["outer_folds"])
    blocks = block_labels(len(x_source), n_folds)
    purge = int(spec["cross_validation"]["purge_adjacent_trs"])
    alphas = np.asarray(spec["cross_validation"]["alpha_grid"], dtype=np.float64)
    source_offsets, target_offsets = fixed_independent_transfer_offsets(
        len(x_source),
        len(x_target),
        n_shifts,
        int(spec["nulls"]["temporal_target"]["minimum_shift_trs"]),
        int(seed),
    )
    output_dir = _output_dir(config_path, spec, "cross_story", model, f"{source_story}_to_{target_story}", variant, size)
    summary_path = output_dir / "summary.json"
    arrays_path = output_dir / "arrays.npz"
    source_paths = [source["predictor_path"], source["brain_path"], target["predictor_path"], target["brain_path"], bank_file, observed_summary_path]
    code_paths = [HERE / "nulls.py", HERE / "transfer.py", HERE / "fitting.py", HERE / "feature_banks.py", HERE / "modeling.py"]
    fingerprint = provenance_fingerprint(
        config_path,
        input_paths=source_paths,
        code_paths=code_paths,
        extra={
            "analysis": "correspondence_cross_story_temporal_target_null",
            "model": model,
            "source_story": source_story,
            "target_story": target_story,
            "variant": variant,
            "size": int(size),
            "n_shifts": int(n_shifts),
            "source_offsets": source_offsets.tolist(),
            "target_offsets": target_offsets.tolist(),
            "selection_metric": "r2_oos",
        },
    )
    if not force:
        cached = assert_cache_fingerprint(summary_path, fingerprint)
        if cached is not None:
            if not arrays_path.is_file():
                raise RuntimeError(f"null cache exists without arrays: {summary_path}")
            return cached

    brain_reliance = np.asarray(observed_arrays.get("brain_r2_reliance"), dtype=np.float64)
    lm_observed = np.asarray(observed_arrays.get("lm_r2_reliance"), dtype=np.float64)
    if brain_reliance.ndim != 2 or brain_reliance.shape[1] != int(size):
        raise ValueError(f"{observed_summary_path}: unexpected cross-story brain reliance shape {brain_reliance.shape}")
    if lm_observed.shape != (int(size),):
        raise ValueError(f"{observed_summary_path}: unexpected cross-story LM reliance shape {lm_observed.shape}")
    brain_profile = np.nanmean(brain_reliance, axis=0)
    observed_corr = _aggregate_correspondence(brain_profile, lm_observed, size)
    lm_groups = feature_groups(int(size), 1)
    null_corr = np.full((len(source_offsets), 3), np.nan, dtype=np.float64)
    null_reliance = np.full((len(source_offsets), int(size)), np.nan, dtype=np.float32)
    selected_alphas = np.full((len(source_offsets), n_folds), np.nan, dtype=np.float64)
    started = time.time()
    for shift_index, (source_shift, target_shift) in enumerate(
        zip(source_offsets.tolist(), target_offsets.tolist(), strict=True)
    ):
        shifted_source = np.roll(source["lm_target"], int(source_shift), axis=0)
        shifted_target = np.roll(target["lm_target"], int(target_shift), axis=0)
        fold_alphas: list[float] = []
        for fold in range(n_folds):
            alpha, _ = select_alpha(
                x_source,
                shifted_source,
                blocks,
                fold,
                alphas,
                metric="r2_oos",
                purge_adjacent=purge,
            )
            selected_alphas[shift_index, fold] = alpha
            fold_alphas.append(float(alpha))
        alpha = modal_alpha([{"alpha": value} for value in fold_alphas])
        lm_result = fit_transfer(
            x_source,
            shifted_source,
            x_target,
            shifted_target,
            alpha=alpha,
            groups=lm_groups,
        )
        null_reliance[shift_index] = np.asarray(lm_result["reliance"], dtype=np.float32)
        corr = _aggregate_correspondence(brain_profile, lm_result["reliance"], size)
        null_corr[shift_index] = [corr["spearman"], corr["pearson"], corr["topk_jaccard"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_path,
        offsets=source_offsets,
        source_offsets=source_offsets,
        target_offsets=target_offsets,
        observed_brain_reliance=brain_profile.astype(np.float32),
        observed_lm_reliance=lm_observed.astype(np.float32),
        null_correspondence=null_corr.astype(np.float32),
        null_lm_reliance=null_reliance,
        selected_alphas=selected_alphas.astype(np.float32),
    )
    identity = _identity_status(observed, observed_arrays)
    summary = {
        "schema_version": 1,
        "analysis_role": "correspondence_cross_story_temporal_target_null",
        "analysis_fingerprint": fingerprint,
        "model_key": model,
        "source_story": source_story,
        "target_story": target_story,
        "feature_bank_variant": variant,
        "bank_size": int(size),
        "n_shifts": int(len(source_offsets)),
        "offsets": source_offsets.tolist(),
        "source_offsets": source_offsets.tolist(),
        "target_offsets": target_offsets.tolist(),
        "effective_source_offsets": [int(value % len(x_source)) for value in source_offsets.tolist()],
        "effective_target_offsets": [int(value % len(x_target)) for value in target_offsets.tolist()],
        "minimum_shift_trs": int(spec["nulls"]["temporal_target"]["minimum_shift_trs"]),
        "targets_shifted": "source_and_target_lm_targets_by_independent_fixed_offsets",
        "feature_bank_reselected": False,
        "penalty_selection_refit_per_shift": True,
        "selection_metric": "r2_oos",
        "brain_profile_fixed_from_observed_transfer": True,
        "observed_correspondence": observed_corr,
        "null_mean": {key: float(np.nanmean(null_corr[:, index])) for index, key in enumerate(("spearman", "pearson", "topk_jaccard"))},
        "null_sd": {key: float(np.nanstd(null_corr[:, index], ddof=1)) for index, key in enumerate(("spearman", "pearson", "topk_jaccard"))},
        "p_value_one_sided": {key: _p_value(null_corr[:, index], observed_corr[key]) for index, key in enumerate(("spearman", "pearson", "topk_jaccard"))},
        "identity_null_verification": identity,
        "source_bank_path": str(bank_file),
        "source_transfer_summary": str(observed_summary_path),
        "arrays_path": str(arrays_path),
        "input_sha256": {str(path): sha256_file(path) for path in source_paths},
        "code_sha256": {str(path): sha256_file(path) for path in code_paths},
        "elapsed_seconds": float(time.time() - started),
        "complete": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary_path, summary)
    return summary


def _groups_from_metadata(metadata: Sequence[dict[str, Any]]) -> list[np.ndarray]:
    """Derive source-variance column groups from a grouped-stage record."""

    groups = []
    for item in metadata:
        columns = np.asarray(item.get("source_variance_columns", []), dtype=np.int64)
        if len(columns) == 0 or len(np.unique(columns)) != len(columns):
            raise ValueError("grouped metadata contains an empty or duplicate column group")
        groups.append(columns)
    if len(groups) != len(metadata):
        raise ValueError("grouped metadata and group arrays differ in length")
    return groups


def _grouped_temporal_null_within(
    config_path: Path,
    spec: dict[str, Any],
    model: str,
    story: str,
    threshold: float,
    *,
    n_shifts: int,
    seed: int,
    force: bool,
) -> dict[str, Any]:
    """Retune LM grouped reliance against fixed BOLD grouped profiles."""

    observed_summary_path = _artifact_root(config_path, spec) / "grouped" / "within_story" / story / model / f"source_story_correlation_pruned_{float(threshold):.2f}".replace(".", "p") / "n256" / "summary.json"
    observed, observed_arrays = _load_observed_summary_arrays(observed_summary_path)
    condition = load_condition(config_path, spec, model, story)
    n_time = len(condition["tr_indices"])
    offsets = fixed_shift_offsets(n_time, n_shifts, int(spec["nulls"]["temporal_target"]["minimum_shift_trs"]), seed)
    n_folds = int(spec["cross_validation"]["outer_folds"])
    purge = int(spec["cross_validation"]["purge_adjacent_trs"])
    inner_folds = int(spec["cross_validation"]["inner_folds"])
    alphas = np.asarray(spec["cross_validation"]["alpha_grid"], dtype=np.float64)
    bank_paths = [bank_path(config_path, spec, model, story, fold=fold) for fold in range(n_folds)]
    # Preserve the threshold in the directory rather than relying on the
    # representative-bank naming convention; grouped inputs are exact-K
    # source-variance columns.
    output_dir = _artifact_root(config_path, spec) / "nulls" / "grouped" / "within_story" / story / model / f"threshold_{float(threshold):.2f}".replace(".", "p") / "n256"
    summary_path = output_dir / "summary.json"
    arrays_path = output_dir / "arrays.npz"
    source_paths = [condition["predictor_path"], condition["brain_path"], observed_summary_path, *bank_paths]
    code_paths = [HERE / "nulls.py", HERE / "grouped_reliance.py", HERE / "fitting.py", HERE / "feature_banks.py", HERE / "modeling.py"]
    fingerprint = provenance_fingerprint(config_path, input_paths=source_paths, code_paths=code_paths, extra={"analysis": "correspondence_within_grouped_temporal_target_null", "model": model, "story": story, "threshold": float(threshold), "size": 256, "n_shifts": int(n_shifts), "offsets": offsets.tolist()})
    if not force:
        cached = assert_cache_fingerprint(summary_path, fingerprint)
        if cached is not None:
            if not arrays_path.is_file():
                raise RuntimeError(f"grouped null cache exists without arrays: {summary_path}")
            return cached
    fold_rows = [
        (np.asarray(row["train_rows"], dtype=np.int64), np.asarray(row["test_rows"], dtype=np.int64))
        for row in observed.get("folds", [])
    ]
    if len(fold_rows) != n_folds:
        fold_rows = contiguous_fold_rows(n_time, n_folds, purge_adjacent_trs=purge)
    fixed_brain = np.asarray(observed_arrays["grouped_brain_reliance_by_fold"], dtype=np.float64)
    counts = np.asarray(observed_arrays["grouped_count_by_fold"], dtype=np.int64)
    if len(fixed_brain) != n_folds or len(counts) != n_folds:
        raise ValueError("grouped observed artifact has an unexpected fold axis")
    null_corr = np.full((len(offsets), 3), np.nan, dtype=np.float64)
    null_fold = np.full((len(offsets), n_folds, 3), np.nan, dtype=np.float64)
    null_reliance = np.full((len(offsets), n_folds, max(1, fixed_brain.shape[1])), np.nan, dtype=np.float32)
    selected_alphas = np.full((len(offsets), n_folds), np.nan, dtype=np.float64)
    started = time.time()
    for shift_index, shift in enumerate(offsets.tolist()):
        shifted_target = np.roll(condition["lm_target"], int(shift), axis=0)
        for fold_index, (outer_train, outer_test) in enumerate(fold_rows):
            bank_record, bank = _load_bank(bank_paths[fold_index], "source_story_variance", 256)
            metadata = observed["groups_by_fold"][fold_index]
            lm_groups = _groups_from_metadata(metadata)
            x_fold, _ = _extract_bank(condition, bank, 256)
            alpha, _, _ = select_alpha_on_outer_train(x_fold, shifted_target, outer_train, alphas=alphas, metric="r2_oos", n_time=n_time, inner_folds=inner_folds, purge_adjacent=purge)
            selected_alphas[shift_index, fold_index] = alpha
            fit = fit_ridge(x_fold, shifted_target, outer_train, alpha)
            evaluation = evaluate_fit(fit, x_fold[outer_test], shifted_target[outer_test], lm_groups)
            group_count = int(counts[fold_index])
            fixed = fixed_brain[fold_index, :group_count]
            corr = correspondence(fixed, evaluation["reliance"], top_k=max(1, min(32, int(round(group_count * 0.125))))) if group_count >= 3 else {"spearman": np.nan, "pearson": np.nan, "topk_jaccard": np.nan}
            null_fold[shift_index, fold_index] = [corr["spearman"], corr["pearson"], corr["topk_jaccard"]]
            null_reliance[shift_index, fold_index, : len(evaluation["reliance"])] = np.asarray(evaluation["reliance"], dtype=np.float32)
        null_corr[shift_index] = np.nanmean(null_fold[shift_index], axis=0)
    observed_corr = {key: float(np.nanmean([row[key] for row in observed.get("correspondence_by_fold", [])])) for key in ("spearman", "pearson", "topk_jaccard")}
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, offsets=offsets, null_correspondence=null_corr.astype(np.float32), null_fold_correspondence=null_fold.astype(np.float32), null_grouped_lm_reliance=null_reliance, selected_alphas=selected_alphas.astype(np.float32))
    summary = {
        "schema_version": 1,
        "analysis_role": "correspondence_within_grouped_temporal_target_null",
        "analysis_fingerprint": fingerprint,
        "model_key": model,
        "story": story,
        "threshold": float(threshold),
        "bank_size": 256,
        "n_shifts": int(len(offsets)),
        "offsets": offsets.tolist(),
        "target_shifted": "lm_target_only",
        "grouped_model_refit": True,
        "group_membership_fixed": True,
        "observed_correspondence": observed_corr,
        "null_mean": {key: float(np.nanmean(null_corr[:, index])) for index, key in enumerate(("spearman", "pearson", "topk_jaccard"))},
        "null_sd": {key: float(np.nanstd(null_corr[:, index], ddof=1)) for index, key in enumerate(("spearman", "pearson", "topk_jaccard"))},
        "p_value_one_sided": {key: _p_value(null_corr[:, index], observed_corr[key]) for index, key in enumerate(("spearman", "pearson", "topk_jaccard"))},
        "identity_null_status": "not_applicable_grouped_profiles; retain feature-level identity null from observed fit",
        "source_grouped_summary": str(observed_summary_path),
        "arrays_path": str(arrays_path),
        "input_sha256": {str(path): sha256_file(path) for path in source_paths},
        "code_sha256": {str(path): sha256_file(path) for path in code_paths},
        "elapsed_seconds": float(time.time() - started),
        "complete": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary_path, summary)
    return summary


def _grouped_temporal_null_cross(
    config_path: Path,
    spec: dict[str, Any],
    model: str,
    source_story: str,
    target_story: str,
    threshold: float,
    *,
    n_shifts: int,
    seed: int,
    force: bool,
) -> dict[str, Any]:
    """Retune a transferred LM mapping against fixed grouped BOLD reliance."""

    label = f"{source_story}_to_{target_story}"
    observed_summary_path = _artifact_root(config_path, spec) / "grouped" / "cross_story" / label / model / f"source_story_correlation_pruned_{float(threshold):.2f}".replace(".", "p") / "n256" / "summary.json"
    observed, observed_arrays = _load_observed_summary_arrays(observed_summary_path)
    source = load_condition(config_path, spec, model, source_story)
    target = load_condition(config_path, spec, model, target_story)
    bank_file = _feature_bank_root(config_path, spec) / "cross_story" / model / label / "banks.json"
    bank_record, bank = _load_bank(bank_file, "source_story_variance", 256)
    x_source, _ = _extract_bank(source, bank, 256)
    x_target, _ = _extract_bank(target, bank, 256)
    groups = _groups_from_metadata(observed["groups"])
    n_folds = int(spec["cross_validation"]["outer_folds"])
    blocks = block_labels(len(x_source), n_folds)
    purge = int(spec["cross_validation"]["purge_adjacent_trs"])
    alphas = np.asarray(spec["cross_validation"]["alpha_grid"], dtype=np.float64)
    source_offsets, target_offsets = fixed_independent_transfer_offsets(
        len(x_source),
        len(x_target),
        n_shifts,
        int(spec["nulls"]["temporal_target"]["minimum_shift_trs"]),
        seed,
    )
    output_dir = _artifact_root(config_path, spec) / "nulls" / "grouped" / "cross_story" / label / model / f"threshold_{float(threshold):.2f}".replace(".", "p") / "n256"
    summary_path = output_dir / "summary.json"
    arrays_path = output_dir / "arrays.npz"
    source_paths = [source["predictor_path"], source["brain_path"], target["predictor_path"], target["brain_path"], bank_file, observed_summary_path]
    code_paths = [HERE / "nulls.py", HERE / "grouped_reliance.py", HERE / "fitting.py", HERE / "feature_banks.py", HERE / "modeling.py"]
    fingerprint = provenance_fingerprint(config_path, input_paths=source_paths, code_paths=code_paths, extra={"analysis": "correspondence_cross_grouped_temporal_target_null", "model": model, "source_story": source_story, "target_story": target_story, "threshold": float(threshold), "size": 256, "n_shifts": int(n_shifts), "source_offsets": source_offsets.tolist(), "target_offsets": target_offsets.tolist()})
    if not force:
        cached = assert_cache_fingerprint(summary_path, fingerprint)
        if cached is not None:
            if not arrays_path.is_file():
                raise RuntimeError(f"grouped null cache exists without arrays: {summary_path}")
            return cached
    brain_profile = np.asarray(observed_arrays["grouped_brain_reliance_mean"], dtype=np.float64)
    lm_observed = np.asarray(observed_arrays["grouped_lm_reliance"], dtype=np.float64)
    observed_corr = _grouped_correspondence(brain_profile, lm_observed)
    lm_groups = groups
    null_corr = np.full((len(source_offsets), 3), np.nan, dtype=np.float64)
    null_reliance = np.full((len(source_offsets), len(groups)), np.nan, dtype=np.float32)
    selected_alphas = np.full((len(source_offsets), n_folds), np.nan, dtype=np.float64)
    started = time.time()
    for shift_index, (source_shift, target_shift) in enumerate(
        zip(source_offsets.tolist(), target_offsets.tolist(), strict=True)
    ):
        shifted_source = np.roll(source["lm_target"], int(source_shift), axis=0)
        shifted_target = np.roll(target["lm_target"], int(target_shift), axis=0)
        selected = []
        for fold in range(n_folds):
            alpha, _ = select_alpha(x_source, shifted_source, blocks, fold, alphas, metric="r2_oos", purge_adjacent=purge)
            selected_alphas[shift_index, fold] = alpha
            selected.append({"alpha": float(alpha)})
        alpha = modal_alpha(selected)
        fit = fit_ridge(x_source, shifted_source, np.flatnonzero(np.all(np.isfinite(shifted_source), axis=1)), alpha)
        evaluation = evaluate_fit(fit, x_target, shifted_target, lm_groups)
        null_reliance[shift_index] = np.asarray(evaluation["reliance"], dtype=np.float32)
        corr = _grouped_correspondence(brain_profile, evaluation["reliance"])
        null_corr[shift_index] = [corr["spearman"], corr["pearson"], corr["topk_jaccard"]]
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(arrays_path, offsets=source_offsets, source_offsets=source_offsets, target_offsets=target_offsets, null_correspondence=null_corr.astype(np.float32), null_grouped_lm_reliance=null_reliance, selected_alphas=selected_alphas.astype(np.float32))
    summary = {
        "schema_version": 1,
        "analysis_role": "correspondence_cross_grouped_temporal_target_null",
        "analysis_fingerprint": fingerprint,
        "model_key": model,
        "source_story": source_story,
        "target_story": target_story,
        "threshold": float(threshold),
        "bank_size": 256,
        "n_shifts": int(len(source_offsets)),
        "offsets": source_offsets.tolist(),
        "source_offsets": source_offsets.tolist(),
        "target_offsets": target_offsets.tolist(),
        "targets_shifted": "source_and_target_lm_targets_by_independent_fixed_offsets",
        "grouped_model_refit": True,
        "group_membership_fixed": True,
        "observed_correspondence": observed_corr,
        "null_mean": {key: float(np.nanmean(null_corr[:, index])) for index, key in enumerate(("spearman", "pearson", "topk_jaccard"))},
        "null_sd": {key: float(np.nanstd(null_corr[:, index], ddof=1)) for index, key in enumerate(("spearman", "pearson", "topk_jaccard"))},
        "p_value_one_sided": {key: _p_value(null_corr[:, index], observed_corr[key]) for index, key in enumerate(("spearman", "pearson", "topk_jaccard"))},
        "identity_null_status": "not_applicable_grouped_profiles; retain feature-level identity null from observed transfer",
        "source_grouped_summary": str(observed_summary_path),
        "arrays_path": str(arrays_path),
        "input_sha256": {str(path): sha256_file(path) for path in source_paths},
        "code_sha256": {str(path): sha256_file(path) for path in code_paths},
        "elapsed_seconds": float(time.time() - started),
        "complete": True,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary_path, summary)
    return summary


def _missing_record(
    *,
    role: str,
    label: str,
    model: str,
    variant: str,
    size: int,
    reason: str,
    path: Path,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "analysis_role": role,
        "status": "skipped",
        "model_key": model,
        "label": label,
        "feature_bank_variant": variant,
        "bank_size": int(size),
        "reason": reason,
        "required_input": str(path),
        "complete": False,
    }


def _group_average_status(config_path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    """Report the separately computed participant-average sensitivity.

    The target is derived deterministically from the explicit participant
    bundles, so a second target-bundle path is unnecessary.  This scanner
    fails closed unless every model/story artifact records a complete,
    fully-retuned 199-shift temporal null.
    """

    statuses = []
    artifact_root = _artifact_root(config_path, spec) / "group_average"
    for model in spec["models"]:
        for story in ("shapesphysical", "shapessocial"):
            path = artifact_root / model / story / "source_story_variance" / "n256" / "summary.json"
            if not path.is_file():
                statuses.append({"model_key": model, "story": story, "status": "missing", "path": str(path)})
                continue
            try:
                record = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                statuses.append({"model_key": model, "story": story, "status": "invalid", "path": str(path), "reason": str(exc)})
                continue
            temporal = record.get("conditional_temporal_null", {})
            grouped = record.get("grouped_temporal_null", {})
            complete = bool(record.get("complete")) and int(temporal.get("n_shifts", 0)) == 199 and int(grouped.get("n_shifts", 0)) == 199 and bool(temporal.get("fully_retuned")) and bool(grouped.get("fully_retuned"))
            statuses.append({"model_key": model, "story": story, "status": "complete" if complete else "incomplete", "path": str(path), "analysis_fingerprint": record.get("analysis_fingerprint")})
    complete = len(statuses) == 12 and all(row["status"] == "complete" for row in statuses)
    return {
        "analysis_role": "correspondence_group_average_temporal_null",
        "status": "complete" if complete else "incomplete",
        "conditions": statuses,
        "complete": complete,
    }


def run(*, config_path: Path, spec: dict[str, Any], args: Any) -> dict[str, Any]:
    models = _selected_models(spec, args)
    variants = _selected_variants(spec, args)
    part = _selected_part(args)
    force = bool(getattr(args, "force", False))
    n_shifts = int(getattr(args, "shifts", None) or spec["nulls"]["temporal_target"]["shifts"])
    seed_base = int(spec["nulls"]["seeds"]["temporal_target"])
    # A pooled replicate must retime every model identically. Keep the
    # condition in the seed, but exclude model identity from all schedules.
    records: list[dict[str, Any]] = []
    if part in ("all", "feature"):
        for model in models:
            for story in ("shapesphysical", "shapessocial"):
                for variant in variants:
                    for size in _variant_sizes(spec, variant, args):
                        observed_path = _fit_summary_path(config_path, spec, model, story, variant, size)
                        try:
                            print(f"analysis within temporal null {model}/{story}/{variant}/n{size}", flush=True)
                            records.append(_within_temporal_null(config_path, spec, model, story, variant, size, n_shifts=n_shifts, seed=_condition_seed(seed_base, story, variant, size), force=force))
                        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
                            records.append(_missing_record(role="correspondence_within_temporal_target_null", label=story, model=model, variant=variant, size=size, reason=f"observed_fit_unavailable:{type(exc).__name__}:{exc}", path=observed_path))
        for model in models:
            for source_story, target_story in (("shapesphysical", "shapessocial"), ("shapessocial", "shapesphysical")):
                label = f"{source_story}_to_{target_story}"
                for variant in variants:
                    for size in _variant_sizes(spec, variant, args):
                        observed_path = _transfer_summary_path(config_path, spec, model, source_story, target_story, variant, size)
                        try:
                            print(f"analysis cross temporal null {model}/{label}/{variant}/n{size}", flush=True)
                            records.append(_cross_temporal_null(config_path, spec, model, source_story, target_story, variant, size, n_shifts=n_shifts, seed=_condition_seed(seed_base, source_story, target_story, variant, size), force=force))
                        except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
                            records.append(_missing_record(role="correspondence_cross_story_temporal_target_null", label=label, model=model, variant=variant, size=size, reason=f"observed_transfer_unavailable:{type(exc).__name__}:{exc}", path=observed_path))
    # Grouped reliance is a separate exact-K analysis.  Its LM null must
    # reuse the discovery-defined cluster memberships from the grouped stage;
    # representative-pruned feature identities are not substituted.
    grouped_thresholds = tuple(
        float(value)
        for value in (
            getattr(args, "grouped_thresholds", None)
            or spec["feature_bank"]["banks"][2].get("thresholds", (0.70, 0.80, 0.90))
        )
    )
    if part in ("all", "within-grouped"):
        for model in models:
            for story in ("shapesphysical", "shapessocial"):
                for threshold in grouped_thresholds:
                    grouped_path = _artifact_root(config_path, spec) / "grouped" / "within_story" / story / model / f"source_story_correlation_pruned_{float(threshold):.2f}".replace(".", "p") / "n256" / "summary.json"
                    try:
                        print(f"analysis within grouped temporal null {model}/{story}/threshold={threshold}", flush=True)
                        records.append(_grouped_temporal_null_within(config_path, spec, model, story, threshold, n_shifts=n_shifts, seed=_condition_seed(seed_base, "grouped", story, threshold), force=force))
                    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
                        records.append(_missing_record(role="correspondence_within_grouped_temporal_target_null", label=story, model=model, variant=f"grouped_threshold_{threshold}", size=256, reason=f"grouped_observed_unavailable:{type(exc).__name__}:{exc}", path=grouped_path))
    if part in ("all", "cross-grouped"):
        for model in models:
            for source_story, target_story in (("shapesphysical", "shapessocial"), ("shapessocial", "shapesphysical")):
                label = f"{source_story}_to_{target_story}"
                for threshold in grouped_thresholds:
                    grouped_path = _artifact_root(config_path, spec) / "grouped" / "cross_story" / label / model / f"source_story_correlation_pruned_{float(threshold):.2f}".replace(".", "p") / "n256" / "summary.json"
                    try:
                        print(f"analysis cross grouped temporal null {model}/{label}/threshold={threshold}", flush=True)
                        records.append(_grouped_temporal_null_cross(config_path, spec, model, source_story, target_story, threshold, n_shifts=n_shifts, seed=_condition_seed(seed_base, "grouped", source_story, target_story, threshold), force=force))
                    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
                        records.append(_missing_record(role="correspondence_cross_grouped_temporal_target_null", label=label, model=model, variant=f"grouped_threshold_{threshold}", size=256, reason=f"grouped_observed_unavailable:{type(exc).__name__}:{exc}", path=grouped_path))
    if part == "all":
        group_average = _group_average_status(config_path, spec)
    else:
        group_average = {
            "analysis_role": "correspondence_group_average_temporal_null",
            "status": "not_run",
            "complete": False,
            "reason": "omitted_from_partial_null_run",
            "run_part": part,
            "conditions": [],
        }
    null_root = _artifact_root(config_path, spec) / "nulls"
    global_manifest_path = null_root / "manifest.json"
    manifest_path = global_manifest_path if part == "all" else null_root / f"manifest_{part.replace('-', '_')}.json"
    manifest = {
        "schema_version": 1,
        "analysis_role": "correspondence_null_manifest",
        "n_requested_shifts": n_shifts,
        "records": records,
        "group_average": group_average,
        "complete_records": int(sum(bool(row.get("complete")) for row in records)),
        "skipped_records": int(sum(row.get("status") == "skipped" for row in records)),
    }
    if part != "all":
        manifest.update(
            {
                "manifest_scope": part,
                "partial_run": True,
                "global_manifest_path": str(global_manifest_path),
                "global_manifest_untouched": True,
            }
        )
    atomic_write_json(manifest_path, manifest)
    return {"stage": "nulls", "manifest": str(manifest_path), **manifest}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--sizes", nargs="+", type=int, default=None)
    parser.add_argument("--shifts", type=int, default=None)
    parser.add_argument("--parts", choices=NULL_PARTS, default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    from .integrity import load_spec

    if args.data_root is not None:
        os.environ["FMRI_SAE_DATA_ROOT"] = str(args.data_root.resolve())
    if args.output_root is not None:
        os.environ["FMRI_SAE_OUTPUT_ROOT"] = str(args.output_root.resolve())
    config_path = (args.config or default_config_path()).resolve()
    spec = load_spec(config_path)
    result = run(config_path=config_path, spec=spec, args=args)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
