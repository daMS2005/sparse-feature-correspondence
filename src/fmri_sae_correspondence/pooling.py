#!/usr/bin/env python3
"""Materialize token-pooling predictors for the correspondence analysis.

The canonical evaluation bundles contain one mean-positive SAE value per TR,
but they do not contain token-level activations.  Consequently, a maximum,
sum, or event-count sensitivity cannot be reconstructed from those arrays.
This module consumes the outcome-free ``token_records.npz`` files written by
``extraction.py`` and materializes the four explicitly
defined pooling representations without running a second model forward pass.

This stage intentionally does *not* pretend that materialization is a fitted
result. It writes validated predictor bundles and an exact compute plan for
the generalized fit/transfer/reliance/null runner. The ``fitting`` and
``transfer`` modules are coupled to ``all_sae_tr_means`` and are
therefore not called on these bundles: silently adapting them by replacing a
mean matrix would produce the wrong scientific analysis.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import default_config_path, load_config
from .extraction import (
    POOLING_NAMES,
    atomic_json,
    pool_token_values,
    sha256_file,
    stable_hash,
)
from .extraction import (
    _predictor_path as _canonical_predictor_path,
)
from .integrity import AnalysisError, atomic_write_json, resolve_output_root

HERE = Path(__file__).resolve().parent

DEFAULT_VARIANTS = (
    "source_story_variance",
    "source_story_correlation_pruned_0p70",
    "source_story_correlation_pruned_0p80",
    "source_story_correlation_pruned_0p90",
)
DEFAULT_POOLINGS = POOLING_NAMES

POOLING_SEMANTICS = {
    "mean_positive_activation": (
        "mean of non-negative SAE activation over all transcript tokens assigned "
        "to the TR; zero-activation tokens remain in the denominator"
    ),
    "max_positive_activation": (
        "maximum non-negative SAE activation over assigned transcript tokens; "
        "zero for an empty TR"
    ),
    "sum_positive_activation": (
        "sum of non-negative SAE activation over assigned transcript tokens"
    ),
    "event_count": (
        "number of assigned transcript tokens whose SAE activation is strictly "
        "greater than zero"
    ),
}


def _artifact_root(config_path: Path, spec: Mapping[str, Any]) -> Path:
    del config_path
    return (resolve_output_root(spec=spec) / str(spec["paths"]["artifact_root"])).resolve()


def _record_root(
    config_path: Path, spec: Mapping[str, Any], output_root: Path | None = None
) -> Path:
    if output_root is not None:
        return Path(output_root).expanduser().resolve()
    return _artifact_root(config_path, spec) / "pooling_token_records"


def _pooling_root(config_path: Path, spec: Mapping[str, Any]) -> Path:
    return _artifact_root(config_path, spec) / "pooling"


def _scalar_string(values: Any, default: str = "") -> str:
    if values is None:
        return default
    array = np.asarray(values)
    if array.size == 0:
        return default
    return str(array.reshape(-1)[0])


def _scalar_bool(values: Any, default: bool = False) -> bool:
    if values is None:
        return default
    array = np.asarray(values)
    if array.size == 0:
        return default
    return bool(array.reshape(-1)[0])


def _scalar_int(values: Any, default: int = 0) -> int:
    if values is None:
        return default
    array = np.asarray(values)
    if array.size == 0:
        return default
    return int(array.reshape(-1)[0])


def load_token_records(path: Path) -> dict[str, Any]:
    """Load and validate one token-record artifact.

    Validation is deliberately stricter than ``np.load``: the pooling
    calculation must not accept duplicate coordinates, negative activations,
    invalid TR assignments, or a token-count field that disagrees with the
    actual alignment.  This makes partial or stale extraction caches fail
    closed before they can enter a fit.
    """

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"token-level pooling record is missing: {path}")
    with np.load(path, allow_pickle=False) as bundle:
        required = {
            "token_values",
            "token_to_tr",
            "tr_indices",
            "token_count",
            "feature_layers",
            "feature_ids",
        }
        missing = sorted(required - set(bundle.files))
        if missing:
            raise ValueError(f"{path}: missing token-record fields {missing}")
        values = np.asarray(bundle["token_values"], dtype=np.float32)
        assignment = np.asarray(bundle["token_to_tr"], dtype=np.int64)
        tr_indices = np.asarray(bundle["tr_indices"], dtype=np.int64)
        token_count = np.asarray(bundle["token_count"], dtype=np.int32)
        feature_layers = np.asarray(bundle["feature_layers"], dtype=np.int32)
        feature_ids = np.asarray(bundle["feature_ids"], dtype=np.int32)
        if "feature_names" in bundle.files:
            feature_names = np.asarray(bundle["feature_names"], dtype=str)
        else:
            feature_names = np.asarray(
                [
                    f"layer_{layer}_feature_{feature}"
                    for layer, feature in zip(feature_layers, feature_ids, strict=False)
                ],
                dtype=str,
            )
        token_positions = (
            np.asarray(bundle["token_positions"], dtype=np.int64)
            if "token_positions" in bundle.files
            else np.arange(len(values), dtype=np.int64)
        )
        token_ids = (
            np.asarray(bundle["token_ids"], dtype=np.int64)
            if "token_ids" in bundle.files
            else None
        )
        tokens = (
            np.asarray(bundle["tokens"], dtype=str)
            if "tokens" in bundle.files
            else None
        )
        model_id = _scalar_string(bundle["model_id"] if "model_id" in bundle.files else None)
        model_key = _scalar_string(bundle["model_key"] if "model_key" in bundle.files else None)
        story = _scalar_string(bundle["story"] if "story" in bundle.files else None)
        context_tokens = _scalar_int(
            bundle["context_tokens"] if "context_tokens" in bundle.files else None,
            1024,
        )
        causal = _scalar_bool(
            bundle["causal"] if "causal" in bundle.files else None,
            True,
        )
        tr_start_s = (
            np.asarray(bundle["tr_start_s"], dtype=np.float64)
            if "tr_start_s" in bundle.files
            else None
        )
        tr_end_s = (
            np.asarray(bundle["tr_end_s"], dtype=np.float64)
            if "tr_end_s" in bundle.files
            else None
        )

    if values.ndim != 2:
        raise ValueError(f"{path}: token_values must have shape [token,coordinate]")
    n_tokens, n_coordinates = values.shape
    if assignment.shape != (n_tokens,):
        raise ValueError(f"{path}: token_to_tr length does not match token_values")
    if tr_indices.ndim != 1 or len(tr_indices) < 1:
        raise ValueError(f"{path}: tr_indices must be a non-empty vector")
    n_tr = len(tr_indices)
    if token_count.shape != (n_tr,):
        raise ValueError(f"{path}: token_count length does not match tr_indices")
    if feature_layers.shape != (n_coordinates,) or feature_ids.shape != (n_coordinates,):
        raise ValueError(f"{path}: feature coordinate lengths do not match token_values")
    if feature_names.shape != (n_coordinates,):
        raise ValueError(f"{path}: feature_names length does not match token_values")
    if token_positions.shape != (n_tokens,) or not np.array_equal(
        token_positions, np.arange(n_tokens, dtype=np.int64)
    ):
        raise ValueError(f"{path}: token_positions must enumerate transcript order exactly")
    if token_ids is not None and token_ids.shape != (n_tokens,):
        raise ValueError(f"{path}: token_ids length does not match token_values")
    if tokens is not None and tokens.shape != (n_tokens,):
        raise ValueError(f"{path}: tokens length does not match token_values")
    if tr_start_s is not None and tr_start_s.shape != (n_tr,):
        raise ValueError(f"{path}: tr_start_s length does not match tr_indices")
    if tr_end_s is not None and tr_end_s.shape != (n_tr,):
        raise ValueError(f"{path}: tr_end_s length does not match tr_indices")
    if not np.all(np.isfinite(values)) or np.any(values < -1e-7):
        raise ValueError(f"{path}: token SAE values must be finite and non-negative")
    valid_assignment = (assignment >= -1) & (assignment < n_tr)
    if not np.all(valid_assignment):
        raise ValueError(f"{path}: token_to_tr contains an invalid TR index")
    observed_counts = np.bincount(
        assignment[assignment >= 0], minlength=n_tr
    ).astype(np.int32)
    if not np.array_equal(observed_counts, token_count):
        raise ValueError(
            f"{path}: token_count disagrees with token_to_tr; "
            f"observed={observed_counts.tolist()} stored={token_count.tolist()}"
        )
    coordinate_pairs = list(zip(feature_layers.tolist(), feature_ids.tolist(), strict=False))
    if len(set(coordinate_pairs)) != n_coordinates:
        raise ValueError(f"{path}: duplicate (layer, feature_id) coordinates")
    return {
        "path": path,
        "token_values": values,
        "token_to_tr": assignment,
        "tr_indices": tr_indices,
        "token_count": token_count,
        "feature_layers": feature_layers,
        "feature_ids": feature_ids,
        "feature_names": feature_names,
        "token_ids": token_ids,
        "tokens": tokens,
        "tr_start_s": tr_start_s,
        "tr_end_s": tr_end_s,
        "model_id": model_id,
        "model_key": model_key,
        "story": story,
        "context_tokens": context_tokens,
        "causal": causal,
        "sha256": sha256_file(path),
    }


def load_coordinate_manifest(path: Path, records: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the coordinate/membership manifest next to token records."""

    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"coordinate manifest is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    coordinates = data.get("coordinate_records")
    if not isinstance(coordinates, list) or len(coordinates) != records["token_values"].shape[1]:
        raise ValueError(f"{path}: coordinate_records length does not match token records")
    observed = []
    for index, row in enumerate(coordinates):
        if not isinstance(row, Mapping):
            raise ValueError(f"{path}: coordinate record {index} is not a mapping")
        if int(row.get("index", -1)) != index:
            raise ValueError(f"{path}: coordinate index is not contiguous at {index}")
        observed.append((int(row["layer"]), int(row["feature_id"])))
    record_coords = list(
        zip(records["feature_layers"].tolist(), records["feature_ids"].tolist(), strict=False)
    )
    if observed != record_coords:
        raise ValueError(f"{path}: coordinate order differs from token-record order")
    memberships = data.get("bank_memberships")
    if not isinstance(memberships, Mapping) or not memberships:
        raise ValueError(f"{path}: bank_memberships is missing")
    for bank_name, indices in memberships.items():
        if not isinstance(indices, list) or not indices:
            raise ValueError(f"{path}: bank {bank_name} has no coordinate membership")
        if len(set(int(index) for index in indices)) != len(indices):
            raise ValueError(f"{path}: bank {bank_name} contains duplicate coordinate indices")
        if any(int(index) < 0 or int(index) >= len(observed) for index in indices):
            raise ValueError(f"{path}: bank {bank_name} contains an out-of-range index")
    data["path"] = str(path)
    data["sha256"] = sha256_file(path)
    return data


def _canonical_mean_check(
    config_path: Path,
    spec: Mapping[str, Any],
    records: Mapping[str, Any],
    pooled_mean: np.ndarray,
    *,
    model_key: str,
    story: str,
) -> dict[str, Any]:
    """Check that token mean pooling reproduces the canonical predictor."""

    predictor_path = _canonical_predictor_path(config_path, spec, model_key, story)
    with np.load(predictor_path, allow_pickle=True) as bundle:
        required = {"all_sae_tr_means", "tr_indices", "selected_layers"}
        missing = sorted(required - set(bundle.files))
        if missing:
            raise ValueError(f"{predictor_path}: missing canonical mean fields {missing}")
        canonical = np.asarray(bundle["all_sae_tr_means"], dtype=np.float32)
        canonical_tr = np.asarray(bundle["tr_indices"], dtype=np.int64)
        canonical_layers = np.asarray(bundle["selected_layers"], dtype=np.int64)
    if not np.array_equal(canonical_tr, records["tr_indices"]):
        raise ValueError("token records and canonical means have different TR indices")
    offsets = {int(layer): idx for idx, layer in enumerate(canonical_layers.tolist())}
    expected = np.zeros_like(pooled_mean, dtype=np.float32)
    for column, (layer, feature) in enumerate(
        zip(records["feature_layers"].tolist(), records["feature_ids"].tolist(), strict=False)
    ):
        if int(layer) not in offsets:
            raise ValueError(f"coordinate layer {layer} is absent from canonical means")
        if int(feature) < 0 or int(feature) >= canonical.shape[2]:
            raise ValueError(f"coordinate feature {feature} is outside canonical dictionary width")
        expected[:, column] = canonical[offsets[int(layer)], :, int(feature)]
    valid = records["token_count"] > 0
    difference = (
        np.abs(pooled_mean[valid] - expected[valid])
        if np.any(valid)
        else np.asarray([], dtype=np.float32)
    )
    max_abs = float(np.max(difference)) if difference.size else 0.0
    max_rel = (
        float(np.max(difference / np.maximum(np.abs(expected[valid]), 1e-6)))
        if difference.size
        else 0.0
    )
    passed = (
        bool(np.allclose(pooled_mean[valid], expected[valid], atol=3e-3, rtol=3e-2))
        if np.any(valid)
        else True
    )
    if not passed:
        raise ValueError(
            f"token mean pooling does not reproduce canonical means for {model_key}/{story}; "
            f"max_abs={max_abs:.6g}, max_relative={max_rel:.6g}"
        )
    return {
        "passed": passed,
        "predictor_path": str(predictor_path),
        "predictor_sha256": sha256_file(predictor_path),
        "atol": 3e-3,
        "rtol": 3e-2,
        "max_abs_error": max_abs,
        "max_relative_error": max_rel,
        "n_checked": int(np.count_nonzero(valid) * pooled_mean.shape[1]),
    }


def materialize_pooled_predictors(
    *,
    config_path: Path,
    spec: Mapping[str, Any],
    model_key: str,
    story: str,
    poolings: Sequence[str] = DEFAULT_POOLINGS,
    record_root: Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Pool one model/story token record into reusable predictor bundles."""

    poolings = tuple(str(value) for value in poolings)
    unknown = sorted(set(poolings) - set(POOLING_NAMES))
    if unknown:
        raise KeyError(f"unknown pooling methods: {unknown}")
    record_dir = _record_root(config_path, spec, record_root) / model_key / story
    token_path = record_dir / "token_records.npz"
    manifest_path = record_dir / "coordinate_manifest.json"
    records = load_token_records(token_path)
    coordinate_manifest = load_coordinate_manifest(manifest_path, records)
    if records["model_key"] and records["model_key"] != model_key:
        raise ValueError(f"token record model mismatch: {records['model_key']} != {model_key}")
    if records["story"] and records["story"] != story:
        raise ValueError(f"token record story mismatch: {records['story']} != {story}")
    fingerprint = stable_hash(
        {
            "analysis": "correspondence_pooling_materialization",
            "config_sha256": sha256_file(config_path),
            "model_key": model_key,
            "story": story,
            "poolings": list(poolings),
            "token_records_sha256": records["sha256"],
            "coordinate_manifest_sha256": coordinate_manifest["sha256"],
            "canonical_predictor_sha256": sha256_file(
                _canonical_predictor_path(config_path, spec, model_key, story)
            ),
            "pooling_semantics": {name: POOLING_SEMANTICS[name] for name in poolings},
        }
    )
    output_root = _pooling_root(config_path, spec) / model_key / story
    manifest_out = output_root / "manifest.json"
    if not force and manifest_out.is_file():
        observed = json.loads(manifest_out.read_text(encoding="utf-8"))
        if observed.get("analysis_fingerprint") != fingerprint:
            raise AnalysisError(
                f"stale pooling materialization cache rejected: {manifest_out}"
            )
        outputs = observed.get("outputs", {})
        if set(outputs) == set(poolings) and all(
            Path(str(item["predictor_path"])).is_file()
            for item in outputs.values()
            if isinstance(item, Mapping)
        ):
            return observed
        raise AnalysisError(
            f"pooling manifest exists with missing output: {manifest_out}"
        )

    pooled_values: dict[str, np.ndarray] = {}
    for pooling in poolings:
        pooled, counts = pool_token_values(
            records["token_values"],
            records["token_to_tr"],
            len(records["tr_indices"]),
            pooling,
        )
        if not np.array_equal(counts, records["token_count"]):
            raise AnalysisError(
                f"pooling count audit failed for {model_key}/{story}/{pooling}"
            )
        pooled_values[pooling] = pooled

    mean_audit = None
    if "mean_positive_activation" in pooled_values:
        mean_audit = _canonical_mean_check(
            config_path,
            spec,
            records,
            pooled_values["mean_positive_activation"],
            model_key=model_key,
            story=story,
        )

    outputs: dict[str, Any] = {}
    for pooling in poolings:
        directory = output_root / pooling
        predictor_path = directory / "predictors.npz"
        summary_path = directory / "summary.json"
        directory.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            prefix=f".{predictor_path.name}.", suffix=".npz", dir=directory, delete=False
        ) as handle:
            temporary = Path(handle.name)
        try:
            np.savez_compressed(
                temporary,
                predictors=pooled_values[pooling].astype(np.float32, copy=False),
                tr_indices=records["tr_indices"],
                token_count=records["token_count"],
                feature_layers=records["feature_layers"],
                feature_ids=records["feature_ids"],
                feature_names=records["feature_names"],
                model_id=np.asarray(
                    [records["model_id"] or spec["models"][model_key]["model_id"]]
                ),
                model_key=np.asarray([model_key]),
                story=np.asarray([story]),
                pooling=np.asarray([pooling]),
                pooling_semantics=np.asarray([POOLING_SEMANTICS[pooling]]),
                context_tokens=np.asarray([records["context_tokens"]], dtype=np.int32),
                causal=np.asarray([records["causal"]], dtype=bool),
            )
            os.replace(temporary, predictor_path)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        summary = {
            "schema_version": 1,
            "analysis_role": "correspondence_pooling_predictors",
            "analysis_fingerprint": fingerprint,
            "model_key": model_key,
            "model_id": records["model_id"] or spec["models"][model_key]["model_id"],
            "story": story,
            "pooling": pooling,
            "pooling_semantics": POOLING_SEMANTICS[pooling],
            "token_records": str(token_path),
            "token_records_sha256": records["sha256"],
            "coordinate_manifest": str(manifest_path),
            "coordinate_manifest_sha256": coordinate_manifest["sha256"],
            "predictor_path": str(predictor_path),
            "predictor_sha256": sha256_file(predictor_path),
            "n_tr": int(len(records["tr_indices"])),
            "n_coordinates": int(pooled_values[pooling].shape[1]),
            "token_count_semantics": "all transcript tokens assigned to each TR",
            "empty_tr_value": 0.0,
            "causal": bool(records["causal"]),
            "context_tokens": int(records["context_tokens"]),
            "outcome_free": True,
        }
        atomic_json(summary_path, summary)
        outputs[pooling] = {
            "predictor_path": str(predictor_path),
            "summary_path": str(summary_path),
            "predictor_sha256": summary["predictor_sha256"],
            "pooling_semantics": summary["pooling_semantics"],
        }

    materialization = {
        "schema_version": 1,
        "analysis_role": "correspondence_pooling_manifest",
        "analysis_fingerprint": fingerprint,
        "model_key": model_key,
        "model_id": records["model_id"] or spec["models"][model_key]["model_id"],
        "story": story,
        "pooling_names": list(poolings),
        "token_records": str(token_path),
        "token_records_sha256": records["sha256"],
        "coordinate_manifest": str(manifest_path),
        "coordinate_manifest_sha256": coordinate_manifest["sha256"],
        "outputs": outputs,
        "coordinate_count": int(records["token_values"].shape[1]),
        "tr_count": int(len(records["tr_indices"])),
        "canonical_mean_reproduction": mean_audit,
        "pooling_semantics": {name: POOLING_SEMANTICS[name] for name in poolings},
        "outcome_free": True,
        "feature_bank_memberships": sorted(coordinate_manifest["bank_memberships"]),
    }
    atomic_json(manifest_out, materialization)
    return materialization


def _selected_models(spec: Mapping[str, Any], args: Any) -> list[str]:
    values = getattr(args, "models", None)
    models = [
        str(value)
        for value in (
            values
            or spec["model_sets"].get("all_models", spec["models"].keys())
        )
    ]
    unknown = sorted(set(models) - set(spec["models"]))
    if unknown:
        raise KeyError(f"unknown models: {unknown}")
    return models


def _selected_stories(spec: Mapping[str, Any], args: Any) -> list[str]:
    values = getattr(args, "stories", None)
    stories = [str(value) for value in (values or spec["stories"].keys())]
    unknown = sorted(set(stories) - set(spec["stories"]))
    if unknown:
        raise KeyError(f"unknown stories: {unknown}")
    return stories


def _selected_poolings(args: Any) -> tuple[str, ...]:
    values = getattr(args, "poolings", None)
    poolings = tuple(str(value) for value in (values or DEFAULT_POOLINGS))
    unknown = sorted(set(poolings) - set(POOLING_NAMES))
    if unknown:
        raise KeyError(f"unknown poolings: {unknown}")
    return poolings


def build_remaining_compute_plan(
    spec: Mapping[str, Any],
    *,
    models: Sequence[str],
    stories: Sequence[str],
    poolings: Sequence[str],
    variants: Sequence[str] = DEFAULT_VARIANTS,
) -> dict[str, Any]:
    """Describe exact fitting work still required after materialization."""

    models = tuple(models)
    stories = tuple(stories)
    poolings = tuple(poolings)
    variants = tuple(variants)
    n_models = len(models)
    n_stories = len(stories)
    n_poolings = len(poolings)
    n_variants = len(variants)
    within_conditions = n_models * n_stories * n_poolings * n_variants
    directions = n_stories * (n_stories - 1)
    transfer_conditions = n_models * directions * n_poolings * n_variants
    temporal_shift_count = int(spec["nulls"]["temporal_target"]["shifts"])
    within_null_refits = within_conditions * temporal_shift_count
    transfer_null_refits = transfer_conditions * temporal_shift_count
    return {
        "status": "pending_generalized_pooling_fit_runner",
        "configured_cv": {
            "outer_folds": int(spec["cross_validation"]["outer_folds"]),
            "inner_folds": int(spec["cross_validation"]["inner_folds"]),
            "purge_adjacent_trs": int(spec["cross_validation"]["purge_adjacent_trs"]),
            "selection_metrics": [
                str(value) for value in spec["cross_validation"]["selection_metrics"]
            ],
            "alpha_grid": [
                float(value) for value in spec["cross_validation"]["alpha_grid"]
            ],
            "training_only_standardization": True,
        },
        "models": list(models),
        "stories": list(stories),
        "poolings": list(poolings),
        "bank_variants": list(variants),
        "pooled_predictor_bundles": n_models * n_stories * n_poolings,
        "within_story_paired_conditions": within_conditions,
        "cross_story_directions": int(directions),
        "cross_story_paired_conditions": transfer_conditions,
        "observed_paired_conditions_total": within_conditions + transfer_conditions,
        "per_condition_targets": {
            "lm": "TR-averaged final hidden state from canonical causal <=1024-token windows",
            "brain": "participant-specific cortical BOLD with SAE lags 0-4",
        },
        "reliance": {
            "primary": "held-out conditional single-feature delta R2_oos",
            "brain_ablation": "all five lag columns for one feature",
            "lm_ablation": "one unlagged feature column",
            "correspondence": "Spearman across matched feature identities",
        },
        "temporal_null": {
            "shifts_per_condition": temporal_shift_count,
            "within_story_lm_refits": within_null_refits,
            "cross_story_lm_refits": transfer_null_refits,
            "total_lm_refits": within_null_refits + transfer_null_refits,
            "refit_penalty_selection": bool(
                spec["nulls"]["temporal_target"]["refit_penalty_selection"]
            ),
            "feature_bank_reselected": bool(
                spec["nulls"]["temporal_target"]["feature_bank_reselected"]
            ),
        },
        "not_run_by_this_stage": [
            "nested ridge fitting",
            "participant-specific BOLD prediction",
            "LM final-hidden-state prediction",
            "cross-story transfer",
            "held-out conditional reliance",
            "199 fully refitted temporal-target nulls",
        ],
        "reason": (
            "fitting and transfer consume all_sae_tr_means and cannot "
            "be pointed at pooled predictor matrices without a generalized loader; "
            "this stage therefore refuses to claim those results"
        ),
    }


def run(*, config_path: Path, spec: dict[str, Any], args: Any) -> dict[str, Any]:
    """Materialize available token records and report exact remaining work."""

    models = _selected_models(spec, args)
    stories = _selected_stories(spec, args)
    poolings = _selected_poolings(args)
    force = bool(getattr(args, "force", False))
    record_root = getattr(args, "record_root", None)
    summaries: list[dict[str, Any]] = []
    missing: list[str] = []
    for model_key in models:
        for story in stories:
            record_path = (
                _record_root(config_path, spec, record_root)
                / model_key
                / story
                / "token_records.npz"
            )
            if not record_path.is_file():
                missing.append(str(record_path))
                continue
            summaries.append(
                materialize_pooled_predictors(
                    config_path=config_path,
                    spec=spec,
                    model_key=model_key,
                    story=story,
                    poolings=poolings,
                    record_root=record_root,
                    force=force,
                )
            )
    if missing:
        raise AnalysisError(
            "pooling stage failed closed because token records are missing; "
            "run the explicit model extraction first:\n- " + "\n- ".join(missing)
        )
    plan = build_remaining_compute_plan(
        spec, models=models, stories=stories, poolings=poolings
    )
    plan_path = _pooling_root(config_path, spec) / "REMAINING_COMPUTE_PLAN.json"
    plan_payload = {
        **plan,
        "analysis_fingerprint": stable_hash(
            {"config_sha256": sha256_file(config_path), "plan": plan}
        ),
        "materialized_manifest_paths": [
            str(_pooling_root(config_path, spec) / row["model_key"] / row["story"] / "manifest.json")
            for row in summaries
        ],
    }
    atomic_write_json(plan_path, plan_payload)
    return {
        "stage": "pooling",
        "status": "materialized_predictors_pending_fits",
        "n_materialized_model_story_bundles": len(summaries),
        "summaries": summaries,
        "remaining_compute_plan": str(plan_path),
        "remaining_compute": plan,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--stories", nargs="+")
    parser.add_argument("--poolings", nargs="+", choices=POOLING_NAMES)
    parser.add_argument("--record-root", type=Path, default=None)
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
    spec = load_config(config_path)
    try:
        result = run(config_path=config_path, spec=spec, args=args)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as exc:
        print(f"pooling stage failed closed: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
