#!/usr/bin/env python3
"""Reproducible control analyses for the analysis release.

This module is deliberately separate from the source-story reliance fit.  Its
historical/dense/random controls write to ``artifacts/controls``.  Its
primary source-story baseline/full comparisons write to
``artifacts/primary_controls`` and never read the historical bank.  It
never writes the primary ``artifacts/fit`` or ``artifacts/transfer`` trees.

The saved evaluation-blind bundles contain the following predictor-side
inputs:

* word counts and phoneme counts can be reconstructed from the checked-in
  stimulus timing tables;
* audio-envelope RMS and first differences can be reconstructed from the
  checked-in WAV files;
* model-specific mean and summed surprisal are stored in every bundle;
* dense residual-stream layer means and all-feature SAE TR means are stored
  in every bundle; and
* the historical 99 matched random banks are stored for all six models.

The loader fails closed when any required source is missing.  In particular,
it does not substitute zero-valued audio, phoneme, dense, or random-bank
controls.  ``inventory`` records this decision before a fit is attempted.

The default control fit is participant-specific, five-fold contiguous,
four-TR-purged, and uses the two independently selected metrics required by
the analysis specification.  The baseline contains low-level,
position, and surprisal blocks.  The multiband model contains the same two
control blocks plus the historical SAE bank, with independently searched
block weights and ridge penalties.  The primary stages use fold-specific
purged-training banks within story and source-selected banks for transfer.

Example:

    fmri-sae-correspondence run controls --config configs/analysis.yaml \
        --stage baseline --models gemma_2_2b --stories shapesphysical \
        --sizes 256
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .config import default_config_path
from .feature_banks import contiguous_fold_rows
from .fitting import (
    _extract_bank as extract_source_bank,
)
from .fitting import (
    _load_bank as load_source_bank,
)
from .fitting import (
    bank_path as source_bank_path,
)
from .fitting import (
    global_finite_training_rows,
    load_condition,
)
from .integrity import (
    assert_cache_fingerprint,
    atomic_write_json,
    provenance_fingerprint,
    resolve_data_root,
    resolve_output_root,
    sha256_file,
)
from .modeling import (
    benjamini_hochberg,
    block_labels,
    bootstrap_mean,
    build_enhanced_baseline,
    feature_groups,
    finite_rows,
    fit_multiband,
    fit_multiband_transfer,
    fixed_band_weights,
    lag_expand,
    mean_metric,
    modal_multiband_parameters,
    nested_multiband_cv,
    nested_subject_cv,
    pearson_columns,
    purged_training_positions,
    sign_flip_test,
)

HERE = Path(__file__).resolve().parent
PARENT_ENGINE = HERE / "modeling.py"

SCHEMA_VERSION = 1
DEFAULT_RANDOM_COUNT = 99
DEFAULT_RANDOM_SIZE = 256
CONTROL_NAMES = (
    "word_count",
    "phoneme_count",
    "envelope_rms",
    "envelope_delta",
    "position_plateau",
    "surprisal_mean",
    "surprisal_sum",
)


def _data_root(config_path: Path, spec: dict[str, Any]) -> Path:
    return resolve_data_root(config_path, spec)


def _artifact_root(config_path: Path, spec: dict[str, Any]) -> Path:
    del config_path
    return (resolve_output_root(spec=spec) / str(spec["paths"]["artifact_root"])).resolve()


def _data_root(
    config_path: Path,
    spec: dict[str, Any],
    override: str | Path | None = None,
) -> Path:
    if override is not None:
        return Path(override).expanduser().resolve()
    return _data_root(config_path, spec)


def _transcript_root(
    config_path: Path,
    spec: dict[str, Any],
    data_root: str | Path | None = None,
) -> Path:
    return _data_root(config_path, spec, data_root) / str(spec["paths"]["transcript_root"])


def _predictor_path(
    config_path: Path,
    spec: dict[str, Any],
    model_key: str,
    story: str,
) -> Path:
    relative = str(spec["paths"]["predictor_pattern"]).format(model=model_key, story=story)
    return _data_root(config_path, spec) / relative


def _random_bank_path(
    config_path: Path,
    spec: dict[str, Any],
    model_key: str,
) -> Path:
    relative = str(spec["paths"]["random_bank_pattern"]).format(model=model_key)
    return _data_root(config_path, spec) / relative


def _historical_ranked_path(
    config_path: Path,
    spec: dict[str, Any],
    model_key: str,
    size: int,
) -> Path:
    relative = str(spec["paths"]["historical_ranked_pattern"]).format(
        model=model_key,
        size=int(size),
    )
    return _data_root(config_path, spec) / relative


def _brain_path(
    config_path: Path,
    spec: dict[str, Any],
    story: str,
) -> Path:
    root = _data_root(config_path, spec)
    value = Path(str(spec["stories"][story]["brain_bundle"]))
    return value.resolve() if value.is_absolute() else (root / value).resolve()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number}: malformed JSONL") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path}:{line_number}: row is not an object")
        rows.append(row)
    return rows


def _required_control_sources(
    transcript_root: Path,
    story: str,
) -> dict[str, Path]:
    story_root = transcript_root / story
    return {
        "metadata": story_root / "metadata.json",
        "tr_aligned": story_root / f"{story}_tr_aligned.tsv",
        "phonemes_by_tr": story_root / f"{story}_phonemes_by_tr.tsv",
        "audio": transcript_root / "audio" / f"{story}_audio.wav",
    }


def _source_status(paths: dict[str, Path]) -> dict[str, Any]:
    missing = {key: str(path) for key, path in paths.items() if not path.is_file()}
    present = {key: str(path) for key, path in paths.items() if path.is_file()}
    return {
        "complete": not missing,
        "present": present,
        "missing": missing,
        "sha256": {key: sha256_file(path) for key, path in paths.items() if path.is_file()},
    }


def _load_predictor_metadata(
    config_path: Path,
    spec: dict[str, Any],
    model_key: str,
    story: str,
) -> dict[str, Any]:
    path = _predictor_path(config_path, spec, model_key, story)
    if not path.is_file():
        raise FileNotFoundError(f"missing saved evaluation-blind predictor bundle: {path}")
    required = {
        "all_sae_tr_means",
        "dense_layer_means",
        "lm_target",
        "surprisal_mean",
        "surprisal_sum",
        "tr_indices",
        "selected_layers",
    }
    with np.load(path, allow_pickle=True) as bundle:
        missing = sorted(required - set(bundle.files))
        if missing:
            raise ValueError(f"{path}: missing required predictor fields: {missing}")
        result = {
            "path": path,
            "model_key": model_key,
            "story": story,
            "all_sae_tr_means": np.asarray(bundle["all_sae_tr_means"]),
            "dense_layer_means": np.asarray(bundle["dense_layer_means"], dtype=np.float64),
            "lm_target": np.asarray(bundle["lm_target"], dtype=np.float64),
            "surprisal_mean": np.asarray(bundle["surprisal_mean"], dtype=np.float64),
            "surprisal_sum": np.asarray(bundle["surprisal_sum"], dtype=np.float64),
            "tr_indices": np.asarray(bundle["tr_indices"], dtype=np.int64),
            "selected_layers": np.asarray(bundle["selected_layers"], dtype=np.int64),
            "model_id": str(np.asarray(bundle["model_id"]).reshape(-1)[0]) if "model_id" in bundle else "",
            "context_tokens": int(np.asarray(bundle["context_tokens"]).reshape(-1)[0]) if "context_tokens" in bundle else None,
            "causal": bool(np.asarray(bundle["causal"]).reshape(-1)[0]) if "causal" in bundle else None,
        }
    all_sae = result["all_sae_tr_means"]
    dense = result["dense_layer_means"]
    n_time = len(result["tr_indices"])
    if all_sae.ndim != 3 or all_sae.shape[1] != n_time:
        raise ValueError(f"{path}: all_sae_tr_means must be [layers,time,features]")
    if dense.ndim != 3 or dense.shape[1] != n_time:
        raise ValueError(f"{path}: dense_layer_means must be [layers,time,hidden]")
    if dense.shape[0] != len(result["selected_layers"]):
        raise ValueError(f"{path}: dense layer axis does not match selected_layers")
    if len(result["lm_target"]) != n_time:
        raise ValueError(f"{path}: LM target length differs from TR index length")
    return result


def _load_predictor_only(
    config_path: Path,
    spec: dict[str, Any],
    model_key: str,
    story: str,
) -> dict[str, Any]:
    """Alias documenting that this loader never reads BOLD outcomes."""

    return _load_predictor_metadata(config_path, spec, model_key, story)


def _build_controls(
    config_path: Path,
    spec: dict[str, Any],
    model_key: str,
    story: str,
    predictor: dict[str, Any],
    data_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build checked-in low-level/position controls plus stored surprisal.

    The shared ``build_enhanced_baseline`` implementation retains the exact
    transcript/audio alignment used by the analysis.
    """

    transcript_root = _transcript_root(config_path, spec, data_root)
    sources = _required_control_sources(transcript_root, story)
    status = _source_status(sources)
    if not status["complete"]:
        raise FileNotFoundError(
            "enhanced control sources are incomplete; refusing to synthesize "
            f"missing controls: {json.dumps(status['missing'], sort_keys=True)}"
        )
    metadata = json.loads(sources["metadata"].read_text(encoding="utf-8"))
    tr_seconds = float(metadata.get("tr_s", float("nan")))
    if not np.isclose(tr_seconds, 1.5):
        raise ValueError(f"{sources['metadata']}: tr_s={tr_seconds} is not the configured 1.5 seconds")
    low = build_enhanced_baseline(
        spec,
        story,
        len(predictor["tr_indices"]),
        predictor["surprisal_mean"],
        predictor["surprisal_sum"],
        transcript_root=transcript_root,
    )
    low_level_position = np.asarray(low["low_level_position"], dtype=np.float64)
    surprisal = np.asarray(low["surprisal"], dtype=np.float64)
    if low_level_position.shape[0] != len(predictor["tr_indices"]):
        raise ValueError(f"{story}: low-level controls do not match predictor TR count")
    if surprisal.shape != (len(predictor["tr_indices"]), 2):
        raise ValueError(f"{story}: surprisal controls have unexpected shape {surprisal.shape}")
    return {
        "low_level_position": low_level_position,
        "surprisal": surprisal,
        "low_level_position_names": np.asarray(low["low_level_position_names"], dtype=str),
        "surprisal_names": np.asarray(low["surprisal_names"], dtype=str),
        "source_paths": [Path(path) for path in sources.values()],
        "source_status": status,
        "transcript_root": transcript_root,
    }


def _read_historical_bank(
    config_path: Path,
    spec: dict[str, Any],
    model_key: str,
    predictor: dict[str, Any],
    size: int,
) -> dict[str, Any]:
    """Reconstruct the old five-story bank from stored all-feature arrays."""

    size = int(size)
    if size < 1 or size > 512:
        raise ValueError("historical bank is available only for 1 <= size <= 512")
    ranked_path = _historical_ranked_path(config_path, spec, model_key, size)
    rows = _read_jsonl(ranked_path)
    if len(rows) < size:
        raise ValueError(f"{ranked_path}: requested {size}, only {len(rows)} rows available")
    layers = np.asarray(predictor["selected_layers"], dtype=np.int64)
    lookup = {int(layer): index for index, layer in enumerate(layers.tolist())}
    selected = rows[:size]
    identities: list[tuple[int, int]] = []
    columns: list[np.ndarray] = []
    for row in selected:
        layer = int(row["layer"])
        feature_id = int(row["feature_id"])
        if layer not in lookup:
            raise ValueError(f"{ranked_path}: layer {layer} is not in saved model layers")
        identities.append((layer, feature_id))
        columns.append(
            np.asarray(
                predictor["all_sae_tr_means"][lookup[layer], :, feature_id],
                dtype=np.float64,
            )
        )
    if len(set(identities)) != size:
        raise ValueError(f"{ranked_path}: duplicate historical feature-layer identity")
    x = np.column_stack(columns)
    if size == 256 and "predictors" in _open_npz_keys(predictor["path"]):
        # The check is informational but catches accidental mismatch between
        # the compact old bank and its full-array reconstruction.
        with np.load(predictor["path"], allow_pickle=True) as bundle:
            compact = np.asarray(bundle["predictors"], dtype=np.float64)
        if compact.shape == x.shape and not np.allclose(compact, x, rtol=1e-4, atol=1e-4):
            raise ValueError(
                f"{predictor['path']}: historical 256 bank does not match all-feature reconstruction"
            )
    return {
        "x": x,
        "layers": np.asarray([layer for layer, _ in identities], dtype=np.int64),
        "feature_ids": np.asarray([feature for _, feature in identities], dtype=np.int64),
        "feature_names": np.asarray(
            [f"layer_{layer}_feature_{feature}" for layer, feature in identities], dtype=str
        ),
        "ranked_path": ranked_path,
        "selection_corpus": ["alice", "black", "bronx", "forgot", "piemanpni"],
        "evaluation_stories_excluded": ["shapesphysical", "shapessocial"],
        "outcome_free": True,
    }


def _open_npz_keys(path: Path) -> set[str]:
    with np.load(path, allow_pickle=True) as bundle:
        return set(bundle.files)


def _load_random_banks(
    config_path: Path,
    spec: dict[str, Any],
    model_key: str,
    predictor: dict[str, Any],
) -> dict[str, Any]:
    path = _random_bank_path(config_path, spec, model_key)
    if not path.is_file():
        raise FileNotFoundError(f"missing saved matched random-bank bundle: {path}")
    with np.load(path, allow_pickle=True) as bundle:
        required = {"feature_ids", "layers", "selected_feature_ids", "selected_layers"}
        missing = sorted(required - set(bundle.files))
        if missing:
            raise ValueError(f"{path}: missing matched-bank fields {missing}")
        feature_ids = np.asarray(bundle["feature_ids"], dtype=np.int64)
        bank_layers = np.asarray(bundle["layers"], dtype=np.int64)
        selected_feature_ids = np.asarray(bundle["selected_feature_ids"], dtype=np.int64)
        selected_layers = np.asarray(bundle["selected_layers"], dtype=np.int64)
        matching_source = str(np.asarray(bundle["matching_source"]).reshape(-1)[0]) if "matching_source" in bundle else ""
        matching_variables = np.asarray(bundle["matching_variables"], dtype=str).tolist() if "matching_variables" in bundle else []
    expected_count = int(spec.get("random_banks", {}).get("count", DEFAULT_RANDOM_COUNT))
    if feature_ids.ndim != 2 or feature_ids.shape[0] != expected_count:
        raise ValueError(f"{path}: expected {expected_count} random banks, got {feature_ids.shape}")
    if feature_ids.shape[1] != DEFAULT_RANDOM_SIZE:
        raise ValueError(f"{path}: expected {DEFAULT_RANDOM_SIZE} features per bank, got {feature_ids.shape[1]}")
    if bank_layers.shape != (DEFAULT_RANDOM_SIZE,) or selected_layers.shape != (DEFAULT_RANDOM_SIZE,):
        raise ValueError(f"{path}: malformed random-bank layer shape")
    if selected_feature_ids.shape != (DEFAULT_RANDOM_SIZE,):
        raise ValueError(f"{path}: malformed selected feature identity shape")
    if not np.array_equal(selected_layers, bank_layers):
        raise ValueError(f"{path}: random-bank layers differ from selected bank layers")
    predictor_layers = set(np.asarray(predictor["selected_layers"], dtype=np.int64).tolist())
    if not set(selected_layers.tolist()).issubset(predictor_layers):
        raise ValueError(f"{path}: random-bank layers not available in predictor bundle")
    if not set(bank_layers.tolist()).issubset(predictor_layers):
        raise ValueError(f"{path}: random-bank layer set is not available in predictor bundle")
    return {
        "path": path,
        "feature_ids": feature_ids,
        "layers": bank_layers,
        "selected_feature_ids": selected_feature_ids,
        "selected_layers": selected_layers,
        "matching_source": matching_source,
        "matching_variables": matching_variables,
        "count": int(feature_ids.shape[0]),
        "size": int(feature_ids.shape[1]),
    }


def _random_predictors(
    predictor: dict[str, Any],
    random_banks: dict[str, Any],
    bank_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    layers = np.asarray(predictor["selected_layers"], dtype=np.int64)
    lookup = {int(layer): index for index, layer in enumerate(layers.tolist())}
    identities = []
    columns = []
    for layer, feature_id in zip(
        random_banks["layers"].tolist(), random_banks["feature_ids"][int(bank_index)].tolist(), strict=False
    ):
        if int(layer) not in lookup:
            raise ValueError(f"random bank layer {layer} unavailable in predictor bundle")
        identities.append((int(layer), int(feature_id)))
        columns.append(
            np.asarray(
                predictor["all_sae_tr_means"][lookup[int(layer)], :, int(feature_id)],
                dtype=np.float64,
            )
        )
    return np.column_stack(columns), np.asarray([x[0] for x in identities]), np.asarray([x[1] for x in identities])


def _flatten_dense(predictor: dict[str, Any]) -> np.ndarray:
    dense = np.asarray(predictor["dense_layer_means"], dtype=np.float64)
    return np.transpose(dense, (1, 0, 2)).reshape(dense.shape[1], -1)


def _alpha_grid(spec: dict[str, Any]) -> np.ndarray:
    values = spec.get("cross_validation", {}).get("alpha_grid")
    if not values:
        raise ValueError("analysis configuration has no cross_validation.alpha_grid")
    return np.asarray(values, dtype=np.float64)


def _gamma_candidates(
    config_path: Path,
    analysis_config: dict[str, Any],
) -> np.ndarray:
    del config_path
    controls = analysis_config["controls"]
    n_candidates = int(controls["band_random_search_candidates"])
    concentrations = controls["band_dirichlet_concentrations"]
    seed = int(analysis_config["nulls"]["seeds"]["primary"])
    return fixed_band_weights(3, n_candidates, concentrations, seed)


def _gamma_candidates_for_blocks(
    config_path: Path,
    analysis_config: dict[str, Any],
    n_blocks: int,
) -> np.ndarray:
    """Return the configured search grid for an explicit number of bands.

    The source-story comparison uses two independently weighted blocks in the
    enhanced baseline (low-level/position and surprisal) and three in the
    full model (those two plus SAE).  A common ridge alpha and per-block
    kernel weights imply independently varying effective penalties
    ``alpha / gamma_b`` for every block.
    """

    del config_path
    controls = analysis_config["controls"]
    n_candidates = int(controls["band_random_search_candidates"])
    concentrations = controls["band_dirichlet_concentrations"]
    seed = int(analysis_config["nulls"]["seeds"]["primary"]) + int(n_blocks)
    return fixed_band_weights(int(n_blocks), n_candidates, concentrations, seed)


def select_multiband_on_outer_train(
    x_blocks: Sequence[np.ndarray],
    y: np.ndarray,
    outer_train: np.ndarray,
    *,
    alphas: np.ndarray,
    gammas: np.ndarray,
    metric: str,
    n_time: int,
    inner_folds: int,
    purge_adjacent: int,
) -> tuple[np.ndarray, float, np.ndarray, int]:
    """Nested multiband selection restricted to one purged outer train set.

    This is the multiband counterpart of
    :func:`fitting.select_alpha_on_outer_train`. It is required for
    fold-specific source banks: the predictor identity is already fixed for
    the current outer fold, so an engine routine that reconstructs a global
    outer split cannot be used safely.
    """

    outer_train = np.asarray(outer_train, dtype=np.int64)
    y = np.asarray(y, dtype=np.float64)
    x_blocks = [np.asarray(block, dtype=np.float64) for block in x_blocks]
    alphas = np.asarray(alphas, dtype=np.float64)
    gammas = np.asarray(gammas, dtype=np.float64)
    if len(outer_train) < int(inner_folds):
        raise ValueError("outer training set is smaller than the inner-fold count")
    if gammas.ndim != 2 or gammas.shape[1] != len(x_blocks):
        raise ValueError("gamma grid width must equal the number of predictor blocks")
    if any(len(block) != int(n_time) for block in x_blocks) or len(y) != int(n_time):
        raise ValueError("multiband predictors and target must share the full-story time axis")

    valid_y = finite_rows(y)
    scores = np.zeros((len(gammas), len(alphas)), dtype=np.float64)
    counts = np.zeros_like(scores, dtype=np.int64)
    used = 0
    for validation_all in np.array_split(outer_train, int(inner_folds)):
        validation_all = np.asarray(validation_all, dtype=np.int64)
        if len(validation_all) == 0:
            continue
        validation = np.intersect1d(validation_all, valid_y, assume_unique=True)
        inner_train = purged_training_positions(
            int(n_time),
            validation_all,
            allowed_positions=outer_train,
            purge_adjacent=int(purge_adjacent),
        )
        inner_train = np.intersect1d(inner_train, valid_y, assume_unique=True)
        if len(validation) < 3 or len(inner_train) < 3:
            continue
        y_train = y[inner_train]
        y_mean = y_train.mean(axis=0)
        y_scale = y_train.std(axis=0)
        y_scale[y_scale == 0] = 1.0
        ys = (y_train - y_mean) / y_scale
        train_standardized: list[np.ndarray] = []
        validation_standardized: list[np.ndarray] = []
        for block in x_blocks:
            train_block = block[inner_train]
            block_mean = train_block.mean(axis=0)
            block_scale = train_block.std(axis=0)
            block_scale[block_scale == 0] = 1.0
            train_standardized.append((train_block - block_mean) / block_scale)
            validation_standardized.append(
                (block[validation] - block_mean) / block_scale
            )
        for gamma_index, gamma in enumerate(gammas):
            kernel = np.zeros((len(inner_train), len(inner_train)), dtype=np.float64)
            cross_kernel = np.zeros((len(validation), len(inner_train)), dtype=np.float64)
            for weight, train_block, validation_block in zip(
                gamma, train_standardized, validation_standardized, strict=False
            ):
                kernel += float(weight) * (train_block @ train_block.T)
                cross_kernel += float(weight) * (validation_block @ train_block.T)
            eigenvalues, eigenvectors = np.linalg.eigh(kernel)
            eigenvalues = np.maximum(eigenvalues, 0.0)
            projected_y = eigenvectors.T @ ys
            projected_test = cross_kernel @ eigenvectors
            for alpha_index, alpha in enumerate(alphas):
                prediction_std = (
                    projected_test / (eigenvalues[None, :] + float(alpha))
                ) @ projected_y
                prediction = prediction_std * y_scale + y_mean
                score = mean_metric(y[validation], prediction, y_mean, metric)
                if np.isfinite(score):
                    scores[gamma_index, alpha_index] += float(score)
                    counts[gamma_index, alpha_index] += 1
        used += 1
    if used == 0:
        raise RuntimeError("no usable inner folds for multiband selection")
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.divide(
            scores,
            counts,
            out=np.full_like(scores, np.nan),
            where=counts > 0,
        )
    finite = np.isfinite(scores)
    if not np.any(finite):
        raise RuntimeError("all inner-fold multiband scores were non-finite")
    best = np.argwhere(finite & (scores == np.nanmax(scores)))
    gamma_index, alpha_index = (int(value) for value in best[0])
    return (
        np.asarray(gammas[gamma_index], dtype=np.float64),
        float(alphas[alpha_index]),
        scores,
        int(used),
    )


def evaluate_multiband_fit(
    fit: Any,
    x_test_blocks: Sequence[np.ndarray],
    y_test: np.ndarray,
    *,
    reliance_block: int | None = None,
    reliance_groups: Sequence[np.ndarray] = (),
) -> dict[str, Any]:
    """Evaluate a fixed multiband fit without refitting target-story data."""

    y_test = np.asarray(y_test, dtype=np.float64)
    valid = finite_rows(y_test)
    if len(valid) == 0:
        raise ValueError("test target has no finite rows")
    blocks = [np.asarray(block, dtype=np.float64)[valid] for block in x_test_blocks]
    y_valid = y_test[valid]
    prediction = fit.predict(blocks)
    sse = np.sum((y_valid - prediction) ** 2, axis=0)
    denominator = np.sum((y_valid - fit.y_mean[None, :]) ** 2, axis=0)
    reliance = np.empty(0, dtype=np.float64)
    if reliance_block is not None and reliance_groups:
        deltas = np.zeros((len(reliance_groups), y_valid.shape[1]), dtype=np.float64)
        for group_index, columns in enumerate(reliance_groups):
            contribution = fit.block_column_contribution(
                blocks,
                int(reliance_block),
                np.asarray(columns, dtype=np.int64),
            )
            ablated = prediction - contribution
            deltas[group_index] = np.sum((y_valid - ablated) ** 2, axis=0) - sse
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled = np.divide(
                deltas,
                denominator[None, :],
                out=np.full_like(deltas, np.nan),
                where=denominator[None, :] > 0,
            )
        reliance = np.nanmean(scaled, axis=1)
    return {
        "valid_positions": valid,
        "prediction": prediction,
        "mean_r": float(mean_metric(y_valid, prediction, fit.y_mean, "pearson_r")),
        "mean_r2_oos": float(mean_metric(y_valid, prediction, fit.y_mean, "r2_oos")),
        "sse": sse,
        "denominator": denominator,
        "reliance": reliance,
    }


def _effective_penalties(alpha: float, gamma: np.ndarray) -> list[float]:
    gamma = np.asarray(gamma, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.divide(float(alpha), gamma)
    return [float(value) for value in values.tolist()]


def _pooled_oof_r(y: np.ndarray, prediction: np.ndarray) -> float:
    valid = finite_rows(prediction)
    valid = np.intersect1d(valid, finite_rows(y), assume_unique=True)
    if len(valid) < 3:
        return float("nan")
    return float(np.nanmean(pearson_columns(y[valid], prediction[valid])))


def _pooled_r2(total_sse: np.ndarray, total_denominator: np.ndarray) -> float:
    with np.errstate(divide="ignore", invalid="ignore"):
        values = 1.0 - np.divide(
            np.asarray(total_sse, dtype=np.float64),
            np.asarray(total_denominator, dtype=np.float64),
            out=np.full_like(np.asarray(total_sse, dtype=np.float64), np.nan),
            where=np.asarray(total_denominator, dtype=np.float64) > 0,
        )
    return float(np.nanmean(values))


def _fit_outer_multiband_model(
    x_blocks: Sequence[np.ndarray],
    y: np.ndarray,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    *,
    alphas: np.ndarray,
    gammas: np.ndarray,
    metric: str,
    inner_folds: int,
    purge_adjacent: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select, fit, and evaluate one multiband model in one outer fold."""

    gamma, alpha, scores, used = select_multiband_on_outer_train(
        x_blocks,
        y,
        outer_train,
        alphas=alphas,
        gammas=gammas,
        metric=metric,
        n_time=len(y),
        inner_folds=inner_folds,
        purge_adjacent=purge_adjacent,
    )
    train = global_finite_training_rows(y, np.asarray(outer_train, dtype=np.int64))
    fit = fit_multiband(x_blocks, y, train, gamma=gamma, alpha=alpha)
    evaluated = evaluate_multiband_fit(
        fit,
        [block[np.asarray(outer_test, dtype=np.int64)] for block in x_blocks],
        y[np.asarray(outer_test, dtype=np.int64)],
    )
    record = {
        "alpha": float(alpha),
        "gamma": np.asarray(gamma, dtype=np.float64).tolist(),
        "effective_block_penalties": _effective_penalties(alpha, gamma),
        "selection_metric": str(metric),
        "n_inner_folds_used": int(used),
        "n_train": int(len(train)),
        "n_test": int(len(evaluated["valid_positions"])),
        "mean_r": float(evaluated["mean_r"]),
        "mean_r2_oos": float(evaluated["mean_r2_oos"]),
        "inner_score_shape": list(np.asarray(scores).shape),
        "inner_score_max": float(np.nanmax(scores)),
    }
    return record, evaluated


def _selected_subjects(condition: dict[str, Any], subject_limit: int | None) -> list[str]:
    subjects = sorted(str(value) for value in condition["subject_targets"])
    if subject_limit is not None:
        if int(subject_limit) < 1:
            raise ValueError("subject_limit must be positive")
        subjects = subjects[: int(subject_limit)]
    return subjects


def _strip_result(value: Any) -> Any:
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            if key in {"oof", "fold_scaling", "reliance", "fold_reliance"}:
                continue
            result[str(key)] = _strip_result(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_strip_result(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _aggregate_result_rows(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = np.asarray([row[key] for row in rows], dtype=np.float64)
    finite = values[np.isfinite(values)]
    return {
        "mean": float(np.mean(finite)) if len(finite) else float("nan"),
        "median": float(np.median(finite)) if len(finite) else float("nan"),
        "sd": float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan"),
        "n": int(len(finite)),
    }


def _fit_one_subject_models(
    condition: dict[str, Any],
    controls: dict[str, Any],
    sae_x: np.ndarray,
    spec: dict[str, Any],
    parent_spec: dict[str, Any],
    subject: str,
    size: int,
    *,
    save_oof: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Fit baseline, SAE-only, and multiband models for one participant."""

    cv = spec["cross_validation"]
    alphas = _alpha_grid(spec)
    purge = int(cv["purge_adjacent_trs"])
    blocks = block_labels(len(sae_x), int(cv["outer_folds"]))
    lags = [int(value) for value in spec["targets"]["brain"]["lags"]]
    target = np.asarray(condition["subject_targets"][subject], dtype=np.float64)
    baseline_raw = np.column_stack([controls["low_level_position"], controls["surprisal"]])
    baseline_brain = lag_expand(baseline_raw, lags)
    sae_brain = lag_expand(sae_x, lags)
    baseline_blocks = [
        lag_expand(controls["low_level_position"], lags),
        lag_expand(controls["surprisal"], lags),
        sae_brain,
    ]
    # The caller passes the already fixed gamma candidates through the
    # private metadata field below.  This keeps all control fitting on the
    # analysis grid without mutating the parent specification.
    gamma_candidates = np.asarray(parent_spec["_gamma_candidates"], dtype=np.float64)
    baseline_groups = feature_groups(baseline_brain.shape[1], 1)
    sae_groups = feature_groups(int(size), len(lags))

    baseline_r = nested_subject_cv(
        baseline_brain,
        target,
        blocks,
        alphas,
        selection_metric="pearson_r",
        purge_adjacent=purge,
        groups=baseline_groups,
    )
    baseline_r2 = nested_subject_cv(
        baseline_brain,
        target,
        blocks,
        alphas,
        selection_metric="r2_oos",
        purge_adjacent=purge,
        groups=baseline_groups,
    )
    sae_r = nested_subject_cv(
        sae_brain,
        target,
        blocks,
        alphas,
        selection_metric="pearson_r",
        purge_adjacent=purge,
        groups=sae_groups,
    )
    sae_r2 = nested_subject_cv(
        sae_brain,
        target,
        blocks,
        alphas,
        selection_metric="r2_oos",
        purge_adjacent=purge,
        groups=sae_groups,
    )
    multiband_r = nested_multiband_cv(
        baseline_blocks,
        target,
        blocks,
        alphas,
        gamma_candidates,
        selection_metric="pearson_r",
        purge_adjacent=purge,
        reliance_block=2,
        reliance_groups=sae_groups,
    )
    multiband_r2 = nested_multiband_cv(
        baseline_blocks,
        target,
        blocks,
        alphas,
        gamma_candidates,
        selection_metric="r2_oos",
        purge_adjacent=purge,
        reliance_block=2,
        reliance_groups=sae_groups,
    )
    row = {
        "subject": str(subject),
        "baseline_r_selected": float(baseline_r["pooled_oof_r"]),
        "baseline_r2_selected_r": float(baseline_r2["pooled_oof_r"]),
        "baseline_r2_selected_r2_oos": float(baseline_r2["pooled_r2_oos"]),
        "sae_r_selected": float(sae_r["pooled_oof_r"]),
        "sae_r2_selected_r": float(sae_r2["pooled_oof_r"]),
        "sae_r2_selected_r2_oos": float(sae_r2["pooled_r2_oos"]),
        "multiband_r_selected": float(multiband_r["pooled_oof_r"]),
        "multiband_r2_selected_r": float(multiband_r2["pooled_oof_r"]),
        "multiband_r2_selected_r2_oos": float(multiband_r2["pooled_r2_oos"]),
        "delta_r_multiband_minus_baseline": float(multiband_r["pooled_oof_r"] - baseline_r["pooled_oof_r"]),
        "delta_r2_multiband_minus_baseline": float(multiband_r2["pooled_r2_oos"] - baseline_r2["pooled_r2_oos"]),
        "baseline_r_alphas": [float(fold["alpha"]) for fold in baseline_r["folds"]],
        "baseline_r2_alphas": [float(fold["alpha"]) for fold in baseline_r2["folds"]],
        "sae_r_alphas": [float(fold["alpha"]) for fold in sae_r["folds"]],
        "sae_r2_alphas": [float(fold["alpha"]) for fold in sae_r2["folds"]],
        "multiband_r_alphas": [float(fold["alpha"]) for fold in multiband_r["folds"]],
        "multiband_r2_alphas": [float(fold["alpha"]) for fold in multiband_r2["folds"]],
        "multiband_r_gammas": [np.asarray(fold["gamma"], dtype=float).tolist() for fold in multiband_r["folds"]],
        "multiband_r2_gammas": [np.asarray(fold["gamma"], dtype=float).tolist() for fold in multiband_r2["folds"]],
    }
    arrays = {}
    if save_oof:
        arrays = {
            "baseline_r_oof": np.asarray(baseline_r["oof"], dtype=np.float32),
            "baseline_r2_oof": np.asarray(baseline_r2["oof"], dtype=np.float32),
            "sae_r_oof": np.asarray(sae_r["oof"], dtype=np.float32),
            "sae_r2_oof": np.asarray(sae_r2["oof"], dtype=np.float32),
            "multiband_r_oof": np.asarray(multiband_r["oof"], dtype=np.float32),
            "multiband_r2_oof": np.asarray(multiband_r2["oof"], dtype=np.float32),
            "multiband_r2_reliance": np.asarray(multiband_r2["reliance"], dtype=np.float32),
        }
    return row, arrays


def _fit_lm_controls(
    condition: dict[str, Any],
    controls: dict[str, Any],
    sae_x: np.ndarray,
    spec: dict[str, Any],
    parent_spec: dict[str, Any],
    *,
    save_oof: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Fit the same control/SAE models to the unlagged LM target."""

    cv = spec["cross_validation"]
    alphas = _alpha_grid(spec)
    purge = int(cv["purge_adjacent_trs"])
    blocks = block_labels(len(sae_x), int(cv["outer_folds"]))
    target = np.asarray(condition["lm_target"], dtype=np.float64)
    baseline = np.column_stack([controls["low_level_position"], controls["surprisal"]])
    baseline_blocks = [controls["low_level_position"], controls["surprisal"], sae_x]
    gamma_candidates = np.asarray(parent_spec["_gamma_candidates"], dtype=np.float64)
    baseline_groups = feature_groups(baseline.shape[1], 1)
    sae_groups = feature_groups(sae_x.shape[1], 1)
    baseline_r = nested_subject_cv(
        baseline, target, blocks, alphas,
        selection_metric="pearson_r", purge_adjacent=purge, groups=baseline_groups,
    )
    baseline_r2 = nested_subject_cv(
        baseline, target, blocks, alphas,
        selection_metric="r2_oos", purge_adjacent=purge, groups=baseline_groups,
    )
    sae_r = nested_subject_cv(
        sae_x, target, blocks, alphas,
        selection_metric="pearson_r", purge_adjacent=purge, groups=sae_groups,
    )
    sae_r2 = nested_subject_cv(
        sae_x, target, blocks, alphas,
        selection_metric="r2_oos", purge_adjacent=purge, groups=sae_groups,
    )
    multiband_r = nested_multiband_cv(
        baseline_blocks, target, blocks, alphas, gamma_candidates,
        selection_metric="pearson_r", purge_adjacent=purge,
        reliance_block=2, reliance_groups=sae_groups,
    )
    multiband_r2 = nested_multiband_cv(
        baseline_blocks, target, blocks, alphas, gamma_candidates,
        selection_metric="r2_oos", purge_adjacent=purge,
        reliance_block=2, reliance_groups=sae_groups,
    )
    row = {
        "baseline_r_selected": float(baseline_r["pooled_oof_r"]),
        "baseline_r2_selected_r": float(baseline_r2["pooled_oof_r"]),
        "baseline_r2_selected_r2_oos": float(baseline_r2["pooled_r2_oos"]),
        "sae_r_selected": float(sae_r["pooled_oof_r"]),
        "sae_r2_selected_r": float(sae_r2["pooled_oof_r"]),
        "sae_r2_selected_r2_oos": float(sae_r2["pooled_r2_oos"]),
        "multiband_r_selected": float(multiband_r["pooled_oof_r"]),
        "multiband_r2_selected_r": float(multiband_r2["pooled_oof_r"]),
        "multiband_r2_selected_r2_oos": float(multiband_r2["pooled_r2_oos"]),
        "delta_r_multiband_minus_baseline": float(multiband_r["pooled_oof_r"] - baseline_r["pooled_oof_r"]),
        "delta_r2_multiband_minus_baseline": float(multiband_r2["pooled_r2_oos"] - baseline_r2["pooled_r2_oos"]),
        "baseline_r_alphas": [float(fold["alpha"]) for fold in baseline_r["folds"]],
        "baseline_r2_alphas": [float(fold["alpha"]) for fold in baseline_r2["folds"]],
        "sae_r_alphas": [float(fold["alpha"]) for fold in sae_r["folds"]],
        "sae_r2_alphas": [float(fold["alpha"]) for fold in sae_r2["folds"]],
        "multiband_r_alphas": [float(fold["alpha"]) for fold in multiband_r["folds"]],
        "multiband_r2_alphas": [float(fold["alpha"]) for fold in multiband_r2["folds"]],
        "multiband_r_gammas": [np.asarray(fold["gamma"], dtype=float).tolist() for fold in multiband_r["folds"]],
        "multiband_r2_gammas": [np.asarray(fold["gamma"], dtype=float).tolist() for fold in multiband_r2["folds"]],
    }
    arrays = {}
    if save_oof:
        arrays = {
            "baseline_r_oof": np.asarray(baseline_r["oof"], dtype=np.float32),
            "baseline_r2_oof": np.asarray(baseline_r2["oof"], dtype=np.float32),
            "sae_r_oof": np.asarray(sae_r["oof"], dtype=np.float32),
            "sae_r2_oof": np.asarray(sae_r2["oof"], dtype=np.float32),
            "multiband_r_oof": np.asarray(multiband_r["oof"], dtype=np.float32),
            "multiband_r2_oof": np.asarray(multiband_r2["oof"], dtype=np.float32),
            "multiband_r2_reliance": np.asarray(multiband_r2["reliance"], dtype=np.float32),
        }
    return row, arrays


def _base_fingerprint_inputs(
    config_path: Path,
    spec: dict[str, Any],
    model_key: str,
    story: str,
    controls: dict[str, Any],
    *,
    extra_paths: Iterable[Path] = (),
) -> list[Path]:
    paths = [
        Path(config_path),
        _predictor_path(config_path, spec, model_key, story),
        _brain_path(config_path, spec, story),
        *controls["source_paths"],
        *list(extra_paths),
    ]
    return [Path(path) for path in paths]


def _control_output_dir(
    config_path: Path,
    spec: dict[str, Any],
    model_key: str,
    story: str,
    family: str,
    size: int | None = None,
    bank_index: int | None = None,
) -> Path:
    root = _artifact_root(config_path, spec) / "controls" / f"{model_key}_{story}"
    if family == "random":
        if bank_index is None:
            raise ValueError("random output requires bank_index")
        return root / f"random_bank_{int(bank_index):03d}"
    if size is not None:
        return root / f"{family}_n{int(size)}"
    return root / family


def _write_control_artifact(
    output_dir: Path,
    summary: dict[str, Any],
    arrays: dict[str, np.ndarray],
    *,
    force: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    arrays_path = output_dir / "arrays.npz"
    if summary_path.is_file() and not force:
        cached = assert_cache_fingerprint(summary_path, str(summary["analysis_fingerprint"]))
        if cached is not None:
            if not arrays_path.is_file():
                raise RuntimeError(f"summary cache exists without arrays: {summary_path}")
            return cached
    if arrays:
        np.savez_compressed(arrays_path, **arrays)
    else:
        # An empty array file is still a useful explicit marker for a metric-
        # only random-bank artifact.
        np.savez_compressed(arrays_path, schema_marker=np.asarray([SCHEMA_VERSION], dtype=np.int16))
    atomic_write_json(summary_path, summary)
    return summary


def inventory(
    config_path: Path,
    spec: dict[str, Any],
    *,
    models: Sequence[str],
    stories: Sequence[str],
    data_root: str | Path | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Inventory exact controls and saved artifacts without fitting."""

    rows: list[dict[str, Any]] = []
    for model_key in models:
        if model_key not in spec["models"]:
            raise KeyError(model_key)
        for story in stories:
            if story not in spec["stories"]:
                raise KeyError(story)
            predictor_path = _predictor_path(config_path, spec, model_key, story)
            random_path = _random_bank_path(config_path, spec, model_key)
            ranked_256 = _historical_ranked_path(config_path, spec, model_key, 256)
            ranked_512 = _historical_ranked_path(config_path, spec, model_key, 512)
            opposite_story = "shapessocial" if story == "shapesphysical" else "shapesphysical"
            primary_fold_banks = [
                source_bank_path(config_path, spec, model_key, story, fold=fold)
                for fold in range(int(spec["cross_validation"]["outer_folds"]))
            ]
            primary_transfer_bank = source_bank_path(
                config_path,
                spec,
                model_key,
                story,
                source_story=story,
                target_story=opposite_story,
            )
            transcript_sources = _required_control_sources(_transcript_root(config_path, spec, data_root), story)
            row: dict[str, Any] = {
                "model_key": model_key,
                "story": story,
                "predictor_bundle": {
                    "path": str(predictor_path),
                    "exists": predictor_path.is_file(),
                    "sha256": sha256_file(predictor_path) if predictor_path.is_file() else None,
                },
                "saved_fields": [],
                "required_saved_fields": [
                    "all_sae_tr_means",
                    "dense_layer_means",
                    "lm_target",
                    "surprisal_mean",
                    "surprisal_sum",
                ],
                "transcript_control_sources": _source_status(transcript_sources),
                "historical_bank": {
                    "selected_256": str(ranked_256),
                    "selected_512": str(ranked_512),
                    "selected_256_exists": ranked_256.is_file(),
                    "selected_512_exists": ranked_512.is_file(),
                    "selection_corpus": ["alice", "black", "bronx", "forgot", "piemanpni"],
                    "evaluation_stories_excluded": ["shapesphysical", "shapessocial"],
                    "requires_refit_under_analysis_spec": True,
                },
                "matched_random_banks": {
                    "path": str(random_path),
                    "exists": random_path.is_file(),
                    "requires_refit_under_analysis_spec": True,
                },
                "primary_source_story_banks": {
                    "within_fold_paths": [str(path) for path in primary_fold_banks],
                    "within_fold_complete": all(path.is_file() for path in primary_fold_banks),
                    "transfer_path": str(primary_transfer_bank),
                    "transfer_exists": primary_transfer_bank.is_file(),
                    "historical_bank_used": False,
                },
                "reuse_status": {
                    "reused_predictor_arrays": predictor_path.is_file(),
                    "reused_dense_layer_means": predictor_path.is_file(),
                    "reused_surprisal": predictor_path.is_file(),
                    "reconstructed_word_phoneme_audio_controls": _source_status(transcript_sources)["complete"],
                    "old_control_results_reused_as_current": False,
                },
            }
            if predictor_path.is_file():
                with np.load(predictor_path, allow_pickle=True) as bundle:
                    row["saved_fields"] = sorted(bundle.files)
                    row["shapes"] = {
                        key: list(bundle[key].shape)
                        for key in (
                            "all_sae_tr_means",
                            "dense_layer_means",
                            "lm_target",
                            "surprisal_mean",
                            "surprisal_sum",
                        )
                        if key in bundle
                    }
            rows.append(row)
    result = {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": "correspondence_control_inventory",
        "analysis_config": "configs/analysis.yaml",
        "analysis_config_sha256": sha256_file(config_path),
        "rows": rows,
        "available_control_families": {
            "low_level_position": "reconstructable only when transcript timing, phoneme, metadata, and WAV files are present",
            "surprisal": "saved in every evaluation-blind bundle",
            "dense_residual_stream": "saved as dense_layer_means in every evaluation-blind bundle",
            "historical_five_story_sae": "reconstructable from saved all_sae_tr_means and ranked JSONL",
            "matched_random_sae": "saved as matched_random_banks.npz for every configured model",
            "primary_source_story_variance": "saved as five purged-training fold banks plus two directional source banks per model",
        },
        "unavailable_control_families": {
            "token_level_max_sum_event_count": "not reconstructed by this script; all-feature token records are not saved",
        },
        "result_policy": "control outputs are computed under the documented analysis contract",
    }
    if output_path is None:
        output_path = _artifact_root(config_path, spec) / "controls" / "inventory.json"
    atomic_write_json(output_path, result)
    return result


def fit_historical_condition(
    config_path: Path,
    spec: dict[str, Any],
    model_key: str,
    story: str,
    size: int,
    *,
    data_root: str | Path | None = None,
    subject_limit: int | None = None,
    save_oof: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Refit baseline, multiband, and historical SAE controls."""

    predictor = _load_predictor_only(config_path, spec, model_key, story)
    condition = load_condition(config_path, spec, model_key, story)
    controls = _build_controls(config_path, spec, model_key, story, predictor, data_root)
    bank = _read_historical_bank(config_path, spec, model_key, predictor, int(size))
    parent_spec = {"_gamma_candidates": _gamma_candidates(config_path, spec)}
    source_paths = _base_fingerprint_inputs(
        config_path,
        spec,
        model_key,
        story,
        controls,
        extra_paths=[bank["ranked_path"]],
    )
    code_paths = [HERE / "controls.py", PARENT_ENGINE, HERE / "fitting.py"]
    fingerprint = provenance_fingerprint(
        config_path,
        input_paths=source_paths,
        code_paths=code_paths,
        extra={
            "analysis": "correspondence_historical_controls",
            "model_key": model_key,
            "story": story,
            "bank_kind": "historical_five_story_reciprocal_rank",
            "bank_size": int(size),
            "subject_limit": subject_limit,
            "save_oof": bool(save_oof),
        },
    )
    output_dir = _control_output_dir(config_path, spec, model_key, story, "historical", int(size))
    summary_path = output_dir / "summary.json"
    if summary_path.is_file() and not force:
        cached = assert_cache_fingerprint(summary_path, fingerprint)
        if cached is not None and (output_dir / "arrays.npz").is_file():
            return cached
    subjects = _selected_subjects(condition, subject_limit)
    brain_rows: list[dict[str, Any]] = []
    brain_arrays: dict[str, list[np.ndarray]] = {}
    started = time.time()
    for index, subject in enumerate(subjects, start=1):
        print(f"historical controls {model_key}/{story} subject {index}/{len(subjects)} {subject}", flush=True)
        row, arrays = _fit_one_subject_models(
            condition,
            controls,
            bank["x"],
            spec,
            parent_spec,
            subject,
            int(size),
            save_oof=save_oof,
        )
        brain_rows.append(row)
        for key, value in arrays.items():
            brain_arrays.setdefault(key, []).append(value)
    lm_row, lm_arrays = _fit_lm_controls(
        condition,
        controls,
        bank["x"],
        spec,
        parent_spec,
        save_oof=save_oof,
    )
    arrays_to_save: dict[str, np.ndarray] = {
        "subject_ids": np.asarray(subjects, dtype=str),
        "historical_layers": bank["layers"],
        "historical_feature_ids": bank["feature_ids"],
        "historical_predictors": np.asarray(bank["x"], dtype=np.float32),
    }
    for key, values in brain_arrays.items():
        arrays_to_save[f"brain_{key}"] = np.asarray(values)
    for key, value in lm_arrays.items():
        arrays_to_save[f"lm_{key}"] = np.asarray(value)
    results = {
        "brain": {
            key: _aggregate_result_rows(brain_rows, key)
            for key in (
                "baseline_r_selected",
                "baseline_r2_selected_r2_oos",
                "sae_r_selected",
                "sae_r2_selected_r2_oos",
                "multiband_r_selected",
                "multiband_r2_selected_r2_oos",
                "delta_r_multiband_minus_baseline",
                "delta_r2_multiband_minus_baseline",
            )
        },
        "lm": lm_row,
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": "correspondence_historical_control_refit",
        "analysis_fingerprint": fingerprint,
        "model_key": model_key,
        "model_id": predictor["model_id"],
        "story": story,
        "bank_kind": "historical_five_story_reciprocal_rank",
        "bank_size": int(size),
        "bank_selection_corpus": bank["selection_corpus"],
        "evaluation_stories_excluded_from_selection": bank["evaluation_stories_excluded"],
        "bank_outcome_free": True,
        "n_subjects": len(subjects),
        "subject_limit": subject_limit,
        "n_time": int(len(predictor["tr_indices"])),
        "n_parcels": int(next(iter(condition["subject_targets"].values())).shape[1]),
        "lm_dimensions": int(condition["lm_target"].shape[1]),
        "context_tokens": predictor["context_tokens"],
        "causal_lm_target": predictor["causal"],
        "controls": {
            "low_level_position_names": controls["low_level_position_names"].tolist(),
            "surprisal_names": controls["surprisal_names"].tolist(),
            "low_level_position_columns": int(controls["low_level_position"].shape[1]),
            "surprisal_columns": int(controls["surprisal"].shape[1]),
            "brain_lags": [int(value) for value in spec["targets"]["brain"]["lags"]],
            "brain_baseline_columns": int((controls["low_level_position"].shape[1] + controls["surprisal"].shape[1]) * len(spec["targets"]["brain"]["lags"])),
            "brain_sae_columns": int(size * len(spec["targets"]["brain"]["lags"])),
        },
        "cv": {
            "outer_folds": int(spec["cross_validation"]["outer_folds"]),
            "inner_folds": int(spec["cross_validation"]["inner_folds"]),
            "split": "contiguous",
            "purge_adjacent_trs": int(spec["cross_validation"]["purge_adjacent_trs"]),
            "selection_metrics": ["pearson_r", "r2_oos"],
            "training_fold_only_standardization": True,
            "alpha_grid": _alpha_grid(spec).tolist(),
            "multiband_gamma_candidates": parent_spec["_gamma_candidates"].tolist(),
        },
        "results": results,
        "participant_rows": brain_rows,
        "lm_row": lm_row,
        "reused_saved_inputs": [
            str(predictor["path"]),
            "all_sae_tr_means",
            "dense_layer_means",
            "surprisal_mean",
            "surprisal_sum",
        ],
        "refit_required": True,
        "old_control_results_reused_as_current": False,
        "source_sha256": {str(path): sha256_file(path) for path in source_paths},
        "code_sha256": {str(path): sha256_file(path) for path in code_paths},
        "arrays_path": str(output_dir / "arrays.npz"),
        "elapsed_seconds": float(time.time() - started),
        "confirmatory_eligible": False,
    }
    return _write_control_artifact(output_dir, summary, arrays_to_save, force=force)


def fit_dense_condition(
    config_path: Path,
    spec: dict[str, Any],
    model_key: str,
    story: str,
    *,
    data_root: str | Path | None = None,
    subject_limit: int | None = None,
    save_oof: bool = True,
    force: bool = False,
) -> dict[str, Any]:
    """Fit dense residual-stream controls at the six saved SAE layers."""

    predictor = _load_predictor_only(config_path, spec, model_key, story)
    condition = load_condition(config_path, spec, model_key, story)
    controls = _build_controls(config_path, spec, model_key, story, predictor, data_root)
    dense_x = _flatten_dense(predictor)
    source_paths = _base_fingerprint_inputs(config_path, spec, model_key, story, controls)
    code_paths = [HERE / "controls.py", PARENT_ENGINE, HERE / "fitting.py"]
    fingerprint = provenance_fingerprint(
        config_path,
        input_paths=source_paths,
        code_paths=code_paths,
        extra={
            "analysis": "correspondence_dense_residual_stream_control",
            "model_key": model_key,
            "story": story,
            "subject_limit": subject_limit,
        },
    )
    output_dir = _control_output_dir(config_path, spec, model_key, story, "dense")
    summary_path = output_dir / "summary.json"
    if summary_path.is_file() and not force:
        cached = assert_cache_fingerprint(summary_path, fingerprint)
        if cached is not None and (output_dir / "arrays.npz").is_file():
            return cached
    cv = spec["cross_validation"]
    alphas = _alpha_grid(spec)
    purge = int(cv["purge_adjacent_trs"])
    blocks = block_labels(len(dense_x), int(cv["outer_folds"]))
    subjects = _selected_subjects(condition, subject_limit)
    brain_rows: list[dict[str, Any]] = []
    brain_arrays: list[np.ndarray] = []
    brain_r2_arrays: list[np.ndarray] = []
    started = time.time()
    for index, subject in enumerate(subjects, start=1):
        print(f"dense control {model_key}/{story} subject {index}/{len(subjects)} {subject}", flush=True)
        target = condition["subject_targets"][subject]
        x_brain = lag_expand(dense_x, spec["targets"]["brain"]["lags"])
        r = nested_subject_cv(
            x_brain, target, blocks, alphas,
            selection_metric="pearson_r", purge_adjacent=purge, groups=[],
        )
        r2 = nested_subject_cv(
            x_brain, target, blocks, alphas,
            selection_metric="r2_oos", purge_adjacent=purge, groups=[],
        )
        brain_rows.append({
            "subject": str(subject),
            "selected_r": float(r["pooled_oof_r"]),
            "selected_r2_r": float(r2["pooled_oof_r"]),
            "selected_r2_oos": float(r2["pooled_r2_oos"]),
            "r_alphas": [float(fold["alpha"]) for fold in r["folds"]],
            "r2_alphas": [float(fold["alpha"]) for fold in r2["folds"]],
        })
        if save_oof:
            brain_arrays.append(np.asarray(r["oof"], dtype=np.float32))
            brain_r2_arrays.append(np.asarray(r2["oof"], dtype=np.float32))
    lm_r = nested_subject_cv(
        dense_x,
        condition["lm_target"],
        blocks,
        alphas,
        selection_metric="pearson_r",
        purge_adjacent=purge,
        groups=[],
    )
    lm_r2 = nested_subject_cv(
        dense_x,
        condition["lm_target"],
        blocks,
        alphas,
        selection_metric="r2_oos",
        purge_adjacent=purge,
        groups=[],
    )
    arrays = {
        "subject_ids": np.asarray(subjects, dtype=str),
        "dense_predictor_dimensions": np.asarray([dense_x.shape[1]], dtype=np.int32),
    }
    if save_oof:
        arrays.update({
            "brain_r_oof": np.asarray(brain_arrays, dtype=np.float32),
            "brain_r2_oof": np.asarray(brain_r2_arrays, dtype=np.float32),
            "lm_r_oof": np.asarray(lm_r["oof"], dtype=np.float32),
            "lm_r2_oof": np.asarray(lm_r2["oof"], dtype=np.float32),
        })
    summary = {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": "correspondence_dense_residual_stream_control",
        "analysis_fingerprint": fingerprint,
        "model_key": model_key,
        "model_id": predictor["model_id"],
        "story": story,
        "n_subjects": len(subjects),
        "subject_limit": subject_limit,
        "dense_predictor_dimensions": int(dense_x.shape[1]),
        "dense_layers": np.asarray(predictor["selected_layers"], dtype=int).tolist(),
        "brain_lags": [int(value) for value in spec["targets"]["brain"]["lags"]],
        "brain": {
            "selected_r": _aggregate_result_rows(brain_rows, "selected_r"),
            "selected_r2_r": _aggregate_result_rows(brain_rows, "selected_r2_r"),
            "selected_r2_oos": _aggregate_result_rows(brain_rows, "selected_r2_oos"),
            "subjects": brain_rows,
        },
        "lm": {
            "selected_r": float(lm_r["pooled_oof_r"]),
            "selected_r2_r": float(lm_r2["pooled_oof_r"]),
            "selected_r2_oos": float(lm_r2["pooled_r2_oos"]),
            "r_alphas": [float(fold["alpha"]) for fold in lm_r["folds"]],
            "r2_alphas": [float(fold["alpha"]) for fold in lm_r2["folds"]],
        },
        "reused_saved_inputs": [str(predictor["path"]), "dense_layer_means"],
        "refit_required": True,
        "old_control_results_reused_as_current": False,
        "source_sha256": {str(path): sha256_file(path) for path in source_paths},
        "code_sha256": {str(path): sha256_file(path) for path in code_paths},
        "arrays_path": str(output_dir / "arrays.npz"),
        "elapsed_seconds": float(time.time() - started),
        "confirmatory_eligible": False,
    }
    return _write_control_artifact(output_dir, summary, arrays, force=force)


def fit_random_condition(
    config_path: Path,
    spec: dict[str, Any],
    model_key: str,
    story: str,
    bank_index: int,
    *,
    data_root: str | Path | None = None,
    subject_limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Fit one matched random SAE bank."""

    predictor = _load_predictor_only(config_path, spec, model_key, story)
    condition = load_condition(config_path, spec, model_key, story)
    controls = _build_controls(config_path, spec, model_key, story, predictor, data_root)
    random_banks = _load_random_banks(config_path, spec, model_key, predictor)
    if not 0 <= int(bank_index) < random_banks["count"]:
        raise IndexError(f"random bank index {bank_index} outside [0,{random_banks['count']})")
    sae_x, bank_layers, bank_feature_ids = _random_predictors(predictor, random_banks, int(bank_index))
    source_paths = _base_fingerprint_inputs(
        config_path, spec, model_key, story, controls, extra_paths=[random_banks["path"]]
    )
    code_paths = [HERE / "controls.py", PARENT_ENGINE, HERE / "fitting.py"]
    fingerprint = provenance_fingerprint(
        config_path,
        input_paths=source_paths,
        code_paths=code_paths,
        extra={
            "analysis": "correspondence_matched_random_sae_control",
            "model_key": model_key,
            "story": story,
            "bank_index": int(bank_index),
            "bank_size": int(random_banks["size"]),
            "subject_limit": subject_limit,
        },
    )
    output_dir = _control_output_dir(config_path, spec, model_key, story, "random", bank_index=int(bank_index))
    summary_path = output_dir / "summary.json"
    if summary_path.is_file() and not force:
        cached = assert_cache_fingerprint(summary_path, fingerprint)
        if cached is not None and (output_dir / "arrays.npz").is_file():
            return cached
    cv = spec["cross_validation"]
    alphas = _alpha_grid(spec)
    purge = int(cv["purge_adjacent_trs"])
    blocks = block_labels(len(sae_x), int(cv["outer_folds"]))
    subjects = _selected_subjects(condition, subject_limit)
    x_brain = lag_expand(sae_x, spec["targets"]["brain"]["lags"])
    brain_r_rows = []
    started = time.time()
    for index, subject in enumerate(subjects, start=1):
        print(f"random bank {model_key}/{story} bank={bank_index} subject {index}/{len(subjects)} {subject}", flush=True)
        target = condition["subject_targets"][subject]
        r = nested_subject_cv(
            x_brain, target, blocks, alphas,
            selection_metric="pearson_r", purge_adjacent=purge, groups=[],
        )
        r2 = nested_subject_cv(
            x_brain, target, blocks, alphas,
            selection_metric="r2_oos", purge_adjacent=purge, groups=[],
        )
        brain_r_rows.append({
            "subject": str(subject),
            "selected_r": float(r["pooled_oof_r"]),
            "selected_r2_oos": float(r2["pooled_r2_oos"]),
            "r2_selected_r": float(r2["pooled_oof_r"]),
            "r_alphas": [float(fold["alpha"]) for fold in r["folds"]],
            "r2_alphas": [float(fold["alpha"]) for fold in r2["folds"]],
        })
    lm_r = nested_subject_cv(
        sae_x, condition["lm_target"], blocks, alphas,
        selection_metric="pearson_r", purge_adjacent=purge, groups=[],
    )
    lm_r2 = nested_subject_cv(
        sae_x, condition["lm_target"], blocks, alphas,
        selection_metric="r2_oos", purge_adjacent=purge, groups=[],
    )
    arrays = {
        "subject_ids": np.asarray(subjects, dtype=str),
        "bank_layers": np.asarray(bank_layers, dtype=np.int64),
        "bank_feature_ids": np.asarray(bank_feature_ids, dtype=np.int64),
        "brain_selected_r": np.asarray([row["selected_r"] for row in brain_r_rows], dtype=np.float32),
        "brain_selected_r2_oos": np.asarray([row["selected_r2_oos"] for row in brain_r_rows], dtype=np.float32),
        "lm_selected_r": np.asarray([lm_r["pooled_oof_r"]], dtype=np.float32),
        "lm_selected_r2_oos": np.asarray([lm_r2["pooled_r2_oos"]], dtype=np.float32),
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": "correspondence_matched_random_sae_control",
        "analysis_fingerprint": fingerprint,
        "model_key": model_key,
        "model_id": predictor["model_id"],
        "story": story,
        "bank_index": int(bank_index),
        "bank_size": int(random_banks["size"]),
        "bank_layers": np.asarray(bank_layers, dtype=int).tolist(),
        "bank_feature_ids": np.asarray(bank_feature_ids, dtype=int).tolist(),
        "matching_source": random_banks["matching_source"],
        "matching_variables": random_banks["matching_variables"],
        "matched_without_outcomes": True,
        "n_subjects": len(subjects),
        "subject_limit": subject_limit,
        "brain": {
            "selected_r": _aggregate_result_rows(brain_r_rows, "selected_r"),
            "selected_r2_oos": _aggregate_result_rows(brain_r_rows, "selected_r2_oos"),
            "subjects": brain_r_rows,
        },
        "lm": {
            "selected_r": float(lm_r["pooled_oof_r"]),
            "selected_r2_oos": float(lm_r2["pooled_r2_oos"]),
            "r_alphas": [float(fold["alpha"]) for fold in lm_r["folds"]],
            "r2_alphas": [float(fold["alpha"]) for fold in lm_r2["folds"]],
        },
        "reused_saved_inputs": [str(predictor["path"]), "all_sae_tr_means", str(random_banks["path"])],
        "refit_required": True,
        "old_control_results_reused_as_current": False,
        "source_sha256": {str(path): sha256_file(path) for path in source_paths},
        "code_sha256": {str(path): sha256_file(path) for path in code_paths},
        "arrays_path": str(output_dir / "arrays.npz"),
        "elapsed_seconds": float(time.time() - started),
        "confirmatory_eligible": False,
    }
    return _write_control_artifact(output_dir, summary, arrays, force=force)


def _primary_control_dir(
    config_path: Path,
    spec: dict[str, Any],
    regime: str,
    condition_name: str,
    *,
    size: int,
    subject_limit: int | None,
) -> Path:
    root = _artifact_root(config_path, spec) / "primary_controls" / regime
    path = root / condition_name / "source_story_variance" / f"n{int(size)}"
    if subject_limit is not None:
        path = path / f"smoke_subjects{int(subject_limit)}"
    return path


def _validate_primary_size_variant(size: int, variant: str) -> None:
    if int(size) != 256:
        raise ValueError("primary control analysis is fixed at K=256")
    if str(variant) != "source_story_variance":
        raise ValueError("primary control analysis requires source_story_variance")


def _pair_metrics_from_oof(
    y: np.ndarray,
    baseline_r_oof: np.ndarray,
    full_r_oof: np.ndarray,
    baseline_r2_sse: np.ndarray,
    full_r2_sse: np.ndarray,
    r2_denominator: np.ndarray,
) -> dict[str, float]:
    baseline_r = _pooled_oof_r(y, baseline_r_oof)
    full_r = _pooled_oof_r(y, full_r_oof)
    baseline_r2 = _pooled_r2(baseline_r2_sse, r2_denominator)
    full_r2 = _pooled_r2(full_r2_sse, r2_denominator)
    return {
        "baseline_r": baseline_r,
        "full_r": full_r,
        "delta_r": float(full_r - baseline_r),
        "baseline_r2_oos": baseline_r2,
        "full_r2_oos": full_r2,
        "delta_r2_oos": float(full_r2 - baseline_r2),
    }


def fit_primary_within_condition(
    config_path: Path,
    spec: dict[str, Any],
    model_key: str,
    story: str,
    *,
    size: int = 256,
    variant: str = "source_story_variance",
    data_root: str | Path | None = None,
    subject_limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Fit primary enhanced-baseline and three-band models within story.

    Every outer fold reads the bank selected from that fold's purged training
    TRs.  The bank is shared by the BOLD and LM targets in the fold.  Baseline
    and full models independently tune their block weights and ridge alpha.
    """

    _validate_primary_size_variant(size, variant)
    condition = load_condition(config_path, spec, model_key, story)
    predictor = _load_predictor_only(config_path, spec, model_key, story)
    if not np.array_equal(condition["tr_indices"], predictor["tr_indices"]):
        raise RuntimeError("condition and control bundles use different TR indices")
    controls = _build_controls(config_path, spec, model_key, story, predictor, data_root)
    cv = spec["cross_validation"]
    n_folds = int(cv["outer_folds"])
    inner_folds = int(cv["inner_folds"])
    purge = int(cv["purge_adjacent_trs"])
    alphas = _alpha_grid(spec)
    gamma_baseline = _gamma_candidates_for_blocks(config_path, spec, 2)
    gamma_full = _gamma_candidates_for_blocks(config_path, spec, 3)
    fold_rows = contiguous_fold_rows(len(condition["tr_indices"]), n_folds, purge_adjacent_trs=purge)
    bank_files = [
        source_bank_path(config_path, spec, model_key, story, fold=fold)
        for fold in range(n_folds)
    ]
    source_paths = _base_fingerprint_inputs(config_path, spec, model_key, story, controls, extra_paths=bank_files)
    code_paths = [HERE / "controls.py", HERE / "fitting.py", HERE / "feature_banks.py", PARENT_ENGINE]
    fingerprint = provenance_fingerprint(
        config_path,
        input_paths=source_paths,
        code_paths=code_paths,
        extra={
            "analysis": "primary_source_story_controls_within",
            "model_key": model_key,
            "story": story,
            "variant": variant,
            "size": int(size),
            "subject_limit": subject_limit,
            "fold_specific_bank": True,
        },
    )
    output_dir = _primary_control_dir(
        config_path, spec, "within", f"{model_key}_{story}", size=size, subject_limit=subject_limit
    )
    summary_path = output_dir / "summary.json"
    if summary_path.is_file() and not force:
        cached = assert_cache_fingerprint(summary_path, fingerprint)
        if cached is not None and (output_dir / "arrays.npz").is_file():
            return cached

    subjects = _selected_subjects(condition, subject_limit)
    n_subjects = len(subjects)
    n_time = len(condition["tr_indices"])
    n_parcels = next(iter(condition["subject_targets"].values())).shape[1]
    lm_dims = condition["lm_target"].shape[1]
    brain_lags = [int(value) for value in spec["targets"]["brain"]["lags"]]

    brain_oof = {
        key: np.full((n_subjects, n_time, n_parcels), np.nan, dtype=np.float32)
        for key in ("baseline_r", "full_r", "baseline_r2", "full_r2")
    }
    lm_oof = {
        key: np.full((n_time, lm_dims), np.nan, dtype=np.float32)
        for key in ("baseline_r", "full_r", "baseline_r2", "full_r2")
    }
    brain_sse = {
        key: np.zeros((n_subjects, n_parcels), dtype=np.float64)
        for key in ("baseline_r2", "full_r2")
    }
    brain_denominator = np.zeros((n_subjects, n_parcels), dtype=np.float64)
    lm_sse = {key: np.zeros(lm_dims, dtype=np.float64) for key in ("baseline_r2", "full_r2")}
    lm_denominator = np.zeros(lm_dims, dtype=np.float64)
    fold_feature_names: list[list[str]] = []
    fold_bank_hashes: list[str | None] = []
    fold_records: list[dict[str, Any]] = []
    started = time.time()

    for fold_index, (outer_train, outer_test) in enumerate(fold_rows):
        bank_record, bank = load_source_bank(bank_files[fold_index], variant, int(size))
        selection_rows = np.asarray(bank_record.get("selection_rows", []), dtype=np.int64)
        if not np.array_equal(selection_rows, outer_train):
            raise RuntimeError(
                f"{bank_files[fold_index]}: bank rows differ from purged outer training rows"
            )
        sae_x, feature_names = extract_source_bank(condition, bank, int(size))
        fold_feature_names.append(feature_names.tolist())
        fold_bank_hashes.append(bank_record.get("bank_bundle_sha256"))
        lm_baseline_blocks = [controls["low_level_position"], controls["surprisal"]]
        lm_full_blocks = [*lm_baseline_blocks, sae_x]
        brain_baseline_blocks = [lag_expand(block, brain_lags) for block in lm_baseline_blocks]
        brain_full_blocks = [*brain_baseline_blocks, lag_expand(sae_x, brain_lags)]
        current: dict[str, Any] = {
            "fold": int(fold_index),
            "train_rows": [int(value) for value in outer_train.tolist()],
            "test_rows": [int(value) for value in outer_test.tolist()],
            "bank_path": str(bank_files[fold_index]),
            "bank_bundle_sha256": bank_record.get("bank_bundle_sha256"),
            "feature_names": feature_names.tolist(),
            "lm": {},
            "brain": [],
        }

        for metric, suffix in (("pearson_r", "r"), ("r2_oos", "r2")):
            baseline_record, baseline_eval = _fit_outer_multiband_model(
                lm_baseline_blocks,
                condition["lm_target"],
                outer_train,
                outer_test,
                alphas=alphas,
                gammas=gamma_baseline,
                metric=metric,
                inner_folds=inner_folds,
                purge_adjacent=purge,
            )
            full_record, full_eval = _fit_outer_multiband_model(
                lm_full_blocks,
                condition["lm_target"],
                outer_train,
                outer_test,
                alphas=alphas,
                gammas=gamma_full,
                metric=metric,
                inner_folds=inner_folds,
                purge_adjacent=purge,
            )
            baseline_global = outer_test[np.asarray(baseline_eval["valid_positions"], dtype=np.int64)]
            full_global = outer_test[np.asarray(full_eval["valid_positions"], dtype=np.int64)]
            lm_oof[f"baseline_{suffix}"][baseline_global] = np.asarray(baseline_eval["prediction"], dtype=np.float32)
            lm_oof[f"full_{suffix}"][full_global] = np.asarray(full_eval["prediction"], dtype=np.float32)
            if suffix == "r2":
                lm_sse["baseline_r2"] += baseline_eval["sse"]
                lm_sse["full_r2"] += full_eval["sse"]
                lm_denominator += baseline_eval["denominator"]
            current["lm"][suffix] = {"baseline": baseline_record, "full": full_record}

        for subject_index, subject in enumerate(subjects):
            target = condition["subject_targets"][subject]
            subject_record: dict[str, Any] = {"subject": subject}
            for metric, suffix in (("pearson_r", "r"), ("r2_oos", "r2")):
                baseline_record, baseline_eval = _fit_outer_multiband_model(
                    brain_baseline_blocks,
                    target,
                    outer_train,
                    outer_test,
                    alphas=alphas,
                    gammas=gamma_baseline,
                    metric=metric,
                    inner_folds=inner_folds,
                    purge_adjacent=purge,
                )
                full_record, full_eval = _fit_outer_multiband_model(
                    brain_full_blocks,
                    target,
                    outer_train,
                    outer_test,
                    alphas=alphas,
                    gammas=gamma_full,
                    metric=metric,
                    inner_folds=inner_folds,
                    purge_adjacent=purge,
                )
                baseline_global = outer_test[np.asarray(baseline_eval["valid_positions"], dtype=np.int64)]
                full_global = outer_test[np.asarray(full_eval["valid_positions"], dtype=np.int64)]
                brain_oof[f"baseline_{suffix}"][subject_index, baseline_global] = np.asarray(
                    baseline_eval["prediction"], dtype=np.float32
                )
                brain_oof[f"full_{suffix}"][subject_index, full_global] = np.asarray(
                    full_eval["prediction"], dtype=np.float32
                )
                if suffix == "r2":
                    brain_sse["baseline_r2"][subject_index] += baseline_eval["sse"]
                    brain_sse["full_r2"][subject_index] += full_eval["sse"]
                    brain_denominator[subject_index] += baseline_eval["denominator"]
                subject_record[suffix] = {"baseline": baseline_record, "full": full_record}
            current["brain"].append(subject_record)
        fold_records.append(current)

    participant_rows = []
    for subject_index, subject in enumerate(subjects):
        participant_rows.append(
            {
                "subject": subject,
                **_pair_metrics_from_oof(
                    condition["subject_targets"][subject],
                    brain_oof["baseline_r"][subject_index],
                    brain_oof["full_r"][subject_index],
                    brain_sse["baseline_r2"][subject_index],
                    brain_sse["full_r2"][subject_index],
                    brain_denominator[subject_index],
                ),
            }
        )
    lm_metrics = _pair_metrics_from_oof(
        condition["lm_target"],
        lm_oof["baseline_r"],
        lm_oof["full_r"],
        lm_sse["baseline_r2"],
        lm_sse["full_r2"],
        lm_denominator,
    )
    arrays = {
        "subject_ids": np.asarray(subjects, dtype=str),
        "fold_feature_names": np.asarray(fold_feature_names, dtype=str),
        "brain_delta_r": np.asarray([row["delta_r"] for row in participant_rows], dtype=np.float32),
        "brain_delta_r2_oos": np.asarray([row["delta_r2_oos"] for row in participant_rows], dtype=np.float32),
        "lm_delta_r": np.asarray([lm_metrics["delta_r"]], dtype=np.float32),
        "lm_delta_r2_oos": np.asarray([lm_metrics["delta_r2_oos"]], dtype=np.float32),
        **{f"brain_{key}_oof": value for key, value in brain_oof.items()},
        **{f"lm_{key}_oof": value for key, value in lm_oof.items()},
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": "correspondence_primary_source_story_controls_within",
        "analysis_fingerprint": fingerprint,
        "model_key": model_key,
        "model_id": condition["model_id"],
        "story": story,
        "feature_bank_variant": variant,
        "bank_size": int(size),
        "bank_policy": "fold_specific_purged_outer_training_rows_only",
        "historical_bank_used": False,
        "n_subjects": int(n_subjects),
        "subject_limit": subject_limit,
        "controls": {
            "baseline_blocks": ["low_level_position", "surprisal"],
            "full_blocks": ["low_level_position", "surprisal", "sae"],
            "independent_effective_penalties": True,
            "baseline_gamma_grid_shape": list(gamma_baseline.shape),
            "full_gamma_grid_shape": list(gamma_full.shape),
        },
        "cv": {
            "outer_folds": n_folds,
            "inner_folds": inner_folds,
            "purge_adjacent_trs": purge,
            "selection_metrics": ["pearson_r", "r2_oos"],
            "training_only_scaling": True,
        },
        "brain": {
            "participants": participant_rows,
            "delta_r": _aggregate_result_rows(participant_rows, "delta_r"),
            "delta_r2_oos": _aggregate_result_rows(participant_rows, "delta_r2_oos"),
        },
        "lm": lm_metrics,
        "folds": fold_records,
        "fold_bank_bundle_sha256": fold_bank_hashes,
        "input_sha256": {str(path): sha256_file(path) for path in source_paths},
        "code_sha256": {str(path): sha256_file(path) for path in code_paths},
        "arrays_path": str(output_dir / "arrays.npz"),
        "elapsed_seconds": float(time.time() - started),
        "smoke_only": subject_limit is not None,
        "control_inference_eligible": subject_limit is None,
    }
    return _write_control_artifact(output_dir, summary, arrays, force=force)


def _multiband_transfer_pair(
    source_blocks: Sequence[np.ndarray],
    source_y: np.ndarray,
    target_blocks: Sequence[np.ndarray],
    target_y: np.ndarray,
    *,
    blocks: np.ndarray,
    alphas: np.ndarray,
    gammas: np.ndarray,
    metric: str,
    purge: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    training = nested_multiband_cv(
        source_blocks,
        source_y,
        blocks,
        alphas,
        gammas,
        selection_metric=metric,
        purge_adjacent=purge,
        reliance_block=0,
        reliance_groups=[],
    )
    gamma, alpha = modal_multiband_parameters(training["folds"])
    transferred = fit_multiband_transfer(
        source_blocks,
        source_y,
        target_blocks,
        target_y,
        gamma=gamma,
        alpha=alpha,
        reliance_block=0,
        reliance_groups=[],
    )
    record = {
        "selection_metric": metric,
        "alpha": float(alpha),
        "gamma": np.asarray(gamma, dtype=np.float64).tolist(),
        "effective_block_penalties": _effective_penalties(alpha, gamma),
        "mean_r": float(transferred["mean_r"]),
        "mean_r2_oos": float(transferred["mean_r2_oos"]),
        "n_train": int(transferred["n_train"]),
        "n_test": int(transferred["n_test"]),
        "source_cv": _strip_result(training),
    }
    return record, transferred


def fit_primary_transfer_condition(
    config_path: Path,
    spec: dict[str, Any],
    model_key: str,
    source_story: str,
    target_story: str,
    *,
    size: int = 256,
    variant: str = "source_story_variance",
    data_root: str | Path | None = None,
    subject_limit: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Fit source-selected control/full models and transfer unchanged."""

    _validate_primary_size_variant(size, variant)
    if source_story == target_story:
        raise ValueError("cross-story transfer requires different source and target stories")
    source_condition = load_condition(config_path, spec, model_key, source_story)
    target_condition = load_condition(config_path, spec, model_key, target_story)
    source_predictor = _load_predictor_only(config_path, spec, model_key, source_story)
    target_predictor = _load_predictor_only(config_path, spec, model_key, target_story)
    source_controls = _build_controls(
        config_path, spec, model_key, source_story, source_predictor, data_root
    )
    target_controls = _build_controls(
        config_path, spec, model_key, target_story, target_predictor, data_root
    )
    bank_file = source_bank_path(
        config_path,
        spec,
        model_key,
        source_story,
        source_story=source_story,
        target_story=target_story,
    )
    bank_record, bank = load_source_bank(bank_file, variant, int(size))
    if str(bank_record.get("source_story", source_story)) != source_story:
        raise RuntimeError(f"{bank_file}: source story provenance mismatch")
    source_x, source_names = extract_source_bank(source_condition, bank, int(size))
    target_x, target_names = extract_source_bank(target_condition, bank, int(size))
    if not np.array_equal(source_names, target_names):
        raise RuntimeError("target extraction changed source-selected feature identities")

    source_paths = [
        source_condition["predictor_path"],
        source_condition["brain_path"],
        target_condition["predictor_path"],
        target_condition["brain_path"],
        bank_file,
        *source_controls["source_paths"],
        *target_controls["source_paths"],
    ]
    code_paths = [HERE / "controls.py", HERE / "fitting.py", HERE / "feature_banks.py", PARENT_ENGINE]
    fingerprint = provenance_fingerprint(
        config_path,
        input_paths=source_paths,
        code_paths=code_paths,
        extra={
            "analysis": "primary_source_story_controls_transfer",
            "model_key": model_key,
            "source_story": source_story,
            "target_story": target_story,
            "variant": variant,
            "size": int(size),
            "subject_limit": subject_limit,
        },
    )
    output_dir = _primary_control_dir(
        config_path,
        spec,
        "transfer",
        f"{model_key}_{source_story}_to_{target_story}",
        size=size,
        subject_limit=subject_limit,
    )
    summary_path = output_dir / "summary.json"
    if summary_path.is_file() and not force:
        cached = assert_cache_fingerprint(summary_path, fingerprint)
        if cached is not None and (output_dir / "arrays.npz").is_file():
            return cached

    common_subjects = sorted(
        set(source_condition["subject_targets"]) & set(target_condition["subject_targets"])
    )
    expected = int(spec.get("cross_validation", {}).get("expected_matched_subjects", 59))
    if len(common_subjects) != expected:
        raise RuntimeError(
            f"expected {expected} matched participants, found {len(common_subjects)}"
        )
    if subject_limit is not None:
        common_subjects = common_subjects[: int(subject_limit)]
    cv = spec["cross_validation"]
    alphas = _alpha_grid(spec)
    purge = int(cv["purge_adjacent_trs"])
    time_blocks = block_labels(len(source_x), int(cv["outer_folds"]))
    gamma_baseline = _gamma_candidates_for_blocks(config_path, spec, 2)
    gamma_full = _gamma_candidates_for_blocks(config_path, spec, 3)
    brain_lags = [int(value) for value in spec["targets"]["brain"]["lags"]]

    source_lm_baseline = [source_controls["low_level_position"], source_controls["surprisal"]]
    target_lm_baseline = [target_controls["low_level_position"], target_controls["surprisal"]]
    source_lm_full = [*source_lm_baseline, source_x]
    target_lm_full = [*target_lm_baseline, target_x]
    source_brain_baseline = [lag_expand(block, brain_lags) for block in source_lm_baseline]
    target_brain_baseline = [lag_expand(block, brain_lags) for block in target_lm_baseline]
    source_brain_full = [*source_brain_baseline, lag_expand(source_x, brain_lags)]
    target_brain_full = [*target_brain_baseline, lag_expand(target_x, brain_lags)]

    started = time.time()
    participant_rows: list[dict[str, Any]] = []
    participant_records: list[dict[str, Any]] = []
    brain_predictions: dict[str, list[np.ndarray]] = {
        key: [] for key in ("baseline_r", "full_r", "baseline_r2", "full_r2")
    }
    for subject in common_subjects:
        source_y = source_condition["subject_targets"][subject]
        target_y = target_condition["subject_targets"][subject]
        record: dict[str, Any] = {"subject": subject}
        metrics: dict[str, float] = {}
        for metric, suffix in (("pearson_r", "r"), ("r2_oos", "r2")):
            baseline_record, baseline_transfer = _multiband_transfer_pair(
                source_brain_baseline,
                source_y,
                target_brain_baseline,
                target_y,
                blocks=time_blocks,
                alphas=alphas,
                gammas=gamma_baseline,
                metric=metric,
                purge=purge,
            )
            full_record, full_transfer = _multiband_transfer_pair(
                source_brain_full,
                source_y,
                target_brain_full,
                target_y,
                blocks=time_blocks,
                alphas=alphas,
                gammas=gamma_full,
                metric=metric,
                purge=purge,
            )
            metrics[f"baseline_{suffix if suffix == 'r' else 'r2_oos'}"] = float(
                baseline_transfer["mean_r"] if suffix == "r" else baseline_transfer["mean_r2_oos"]
            )
            metrics[f"full_{suffix if suffix == 'r' else 'r2_oos'}"] = float(
                full_transfer["mean_r"] if suffix == "r" else full_transfer["mean_r2_oos"]
            )
            baseline_prediction = np.full(
                (len(target_condition["tr_indices"]), target_y.shape[1]),
                np.nan,
                dtype=np.float32,
            )
            full_prediction = np.full_like(baseline_prediction, np.nan)
            baseline_prediction[np.asarray(baseline_transfer["test_positions"], dtype=np.int64)] = np.asarray(
                baseline_transfer["prediction"], dtype=np.float32
            )
            full_prediction[np.asarray(full_transfer["test_positions"], dtype=np.int64)] = np.asarray(
                full_transfer["prediction"], dtype=np.float32
            )
            brain_predictions[f"baseline_{suffix}"].append(baseline_prediction)
            brain_predictions[f"full_{suffix}"].append(full_prediction)
            record[suffix] = {"baseline": baseline_record, "full": full_record}
        metrics["delta_r"] = float(metrics["full_r"] - metrics["baseline_r"])
        metrics["delta_r2_oos"] = float(metrics["full_r2_oos"] - metrics["baseline_r2_oos"])
        participant_rows.append({"subject": subject, **metrics})
        participant_records.append(record)

    lm_record: dict[str, Any] = {}
    lm_metrics: dict[str, float] = {}
    lm_predictions: dict[str, np.ndarray] = {}
    for metric, suffix in (("pearson_r", "r"), ("r2_oos", "r2")):
        baseline_record, baseline_transfer = _multiband_transfer_pair(
            source_lm_baseline,
            source_condition["lm_target"],
            target_lm_baseline,
            target_condition["lm_target"],
            blocks=time_blocks,
            alphas=alphas,
            gammas=gamma_baseline,
            metric=metric,
            purge=purge,
        )
        full_record, full_transfer = _multiband_transfer_pair(
            source_lm_full,
            source_condition["lm_target"],
            target_lm_full,
            target_condition["lm_target"],
            blocks=time_blocks,
            alphas=alphas,
            gammas=gamma_full,
            metric=metric,
            purge=purge,
        )
        lm_metrics[f"baseline_{suffix if suffix == 'r' else 'r2_oos'}"] = float(
            baseline_transfer["mean_r"] if suffix == "r" else baseline_transfer["mean_r2_oos"]
        )
        lm_metrics[f"full_{suffix if suffix == 'r' else 'r2_oos'}"] = float(
            full_transfer["mean_r"] if suffix == "r" else full_transfer["mean_r2_oos"]
        )
        lm_predictions[f"baseline_{suffix}"] = np.asarray(baseline_transfer["prediction"], dtype=np.float32)
        lm_predictions[f"full_{suffix}"] = np.asarray(full_transfer["prediction"], dtype=np.float32)
        lm_record[suffix] = {"baseline": baseline_record, "full": full_record}
    lm_metrics["delta_r"] = float(lm_metrics["full_r"] - lm_metrics["baseline_r"])
    lm_metrics["delta_r2_oos"] = float(lm_metrics["full_r2_oos"] - lm_metrics["baseline_r2_oos"])

    arrays = {
        "subject_ids": np.asarray(common_subjects, dtype=str),
        "feature_names": np.asarray(source_names, dtype=str),
        "brain_delta_r": np.asarray([row["delta_r"] for row in participant_rows], dtype=np.float32),
        "brain_delta_r2_oos": np.asarray([row["delta_r2_oos"] for row in participant_rows], dtype=np.float32),
        **{f"brain_{key}_prediction": np.asarray(values, dtype=np.float32) for key, values in brain_predictions.items()},
        **{f"lm_{key}_prediction": value for key, value in lm_predictions.items()},
    }
    summary = {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": "correspondence_primary_source_story_controls_transfer",
        "analysis_fingerprint": fingerprint,
        "model_key": model_key,
        "model_id": source_condition["model_id"],
        "source_story": source_story,
        "target_story": target_story,
        "feature_bank_variant": variant,
        "bank_size": int(size),
        "source_bank_path": str(bank_file),
        "source_bank_bundle_sha256": bank_record.get("bank_bundle_sha256"),
        "source_bank_selection_only": True,
        "target_story_excluded_from_bank_selection": True,
        "source_scaling_and_weights_applied_unchanged": True,
        "historical_bank_used": False,
        "n_matched_subjects": len(common_subjects),
        "subject_limit": subject_limit,
        "controls": {
            "baseline_blocks": ["low_level_position", "surprisal"],
            "full_blocks": ["low_level_position", "surprisal", "sae"],
            "independent_effective_penalties": True,
        },
        "brain": {
            "participants": participant_rows,
            "delta_r": _aggregate_result_rows(participant_rows, "delta_r"),
            "delta_r2_oos": _aggregate_result_rows(participant_rows, "delta_r2_oos"),
            "fit_records": participant_records,
        },
        "lm": {**lm_metrics, "fit_records": lm_record},
        "input_sha256": {str(path): sha256_file(path) for path in source_paths},
        "code_sha256": {str(path): sha256_file(path) for path in code_paths},
        "arrays_path": str(output_dir / "arrays.npz"),
        "elapsed_seconds": float(time.time() - started),
        "smoke_only": subject_limit is not None,
        "control_inference_eligible": subject_limit is None,
    }
    return _write_control_artifact(output_dir, summary, arrays, force=force)


def _inference_stat(values: np.ndarray, *, bootstrap_draws: int, sign_draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        raise ValueError("paired inference requires at least two finite participants")
    boot = bootstrap_mean(values, draws=int(bootstrap_draws), seed=int(seed))
    sign = sign_flip_test(values, draws=int(sign_draws), seed=int(seed) + 1)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "sd": float(np.std(values, ddof=1)),
        "n": int(len(values)),
        "bootstrap_ci95": [float(value) for value in np.asarray(boot["ci95"]).tolist()],
        "sign_flip_p_one_sided": float(sign["p_one_sided"]),
        "sign_flip_p_two_sided": float(sign["p_two_sided"]),
    }


def run_primary_control_inference(
    config_path: Path,
    spec: dict[str, Any],
    *,
    regime: str,
    size: int = 256,
    variant: str = "source_story_variance",
    force: bool = False,
) -> dict[str, Any]:
    """Paired participant inference with BH and global mean-effect tests."""

    _validate_primary_size_variant(size, variant)
    if regime not in {"within", "transfer"}:
        raise ValueError("inference regime must be 'within' or 'transfer'")
    models = list(spec["models"].keys())
    if len(models) != 6:
        raise RuntimeError(f"BH contract expects six models, found {len(models)}")
    if regime == "within":
        groups = {
            story: [(model, f"{model}_{story}") for model in models]
            for story in spec["stories"]
        }
    else:
        groups = {
            f"{source}_to_{target}": [
                (model, f"{model}_{source}_to_{target}") for model in models
            ]
            for source, target in (
                ("shapesphysical", "shapessocial"),
                ("shapessocial", "shapesphysical"),
            )
        }
    artifact_root = _artifact_root(config_path, spec) / "primary_controls"
    input_paths: list[Path] = []
    loaded: dict[str, list[dict[str, Any]]] = {}
    for group_name, conditions in groups.items():
        loaded[group_name] = []
        for model_key, condition_name in conditions:
            condition_dir = artifact_root / regime / condition_name / variant / f"n{int(size)}"
            summary_path = condition_dir / "summary.json"
            arrays_path = condition_dir / "arrays.npz"
            if not summary_path.is_file() or not arrays_path.is_file():
                raise FileNotFoundError(
                    f"missing full primary-control condition for inference: {condition_dir}"
                )
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("smoke_only") or not summary.get("control_inference_eligible"):
                raise RuntimeError(f"{summary_path}: smoke/ineligible artifact cannot enter inference")
            if summary.get("historical_bank_used") is not False:
                raise RuntimeError(f"{summary_path}: primary inference cannot use historical bank")
            with np.load(arrays_path, allow_pickle=True) as bundle:
                required = {"subject_ids", "brain_delta_r", "brain_delta_r2_oos"}
                missing = sorted(required - set(bundle.files))
                if missing:
                    raise ValueError(f"{arrays_path}: missing inference arrays {missing}")
                row = {
                    "model_key": model_key,
                    "subject_ids": np.asarray(bundle["subject_ids"], dtype=str),
                    "delta_r": np.asarray(bundle["brain_delta_r"], dtype=np.float64),
                    "delta_r2_oos": np.asarray(bundle["brain_delta_r2_oos"], dtype=np.float64),
                    "summary_path": summary_path,
                    "arrays_path": arrays_path,
                }
            if not (
                len(row["subject_ids"]) == len(row["delta_r"]) == len(row["delta_r2_oos"])
            ):
                raise ValueError(f"{arrays_path}: participant/delta length mismatch")
            loaded[group_name].append(row)
            input_paths.extend([summary_path, arrays_path])

    code_paths = [HERE / "controls.py", PARENT_ENGINE]
    fingerprint = provenance_fingerprint(
        config_path,
        input_paths=input_paths,
        code_paths=code_paths,
        extra={
            "analysis": "primary_control_paired_inference",
            "regime": regime,
            "variant": variant,
            "size": int(size),
        },
    )
    output_dir = artifact_root / "inference" / regime / variant / f"n{int(size)}"
    summary_path = output_dir / "summary.json"
    if summary_path.is_file() and not force:
        cached = assert_cache_fingerprint(summary_path, fingerprint)
        if cached is not None and (output_dir / "arrays.npz").is_file():
            return cached

    bootstrap_draws = int(spec["nulls"]["bootstrap_draws"])
    sign_draws = int(spec["nulls"]["sign_flip_draws"])
    seed_base = int(spec["nulls"]["seeds"]["bootstrap"])
    result_groups: dict[str, Any] = {}
    arrays: dict[str, np.ndarray] = {}
    for group_index, (group_name, rows) in enumerate(loaded.items()):
        reference_subjects = rows[0]["subject_ids"]
        for row in rows[1:]:
            if not np.array_equal(row["subject_ids"], reference_subjects):
                raise RuntimeError(f"{group_name}: participant order differs across models")
        group_result: dict[str, Any] = {"models": {}, "global_mean_effect": {}}
        for metric_index, metric in enumerate(("delta_r", "delta_r2_oos")):
            model_stats = []
            p_one = []
            p_two = []
            stacked = []
            for model_index, row in enumerate(rows):
                values = np.asarray(row[metric], dtype=np.float64)
                stat = _inference_stat(
                    values,
                    bootstrap_draws=bootstrap_draws,
                    sign_draws=sign_draws,
                    seed=seed_base + group_index * 1000 + metric_index * 100 + model_index * 3,
                )
                model_stats.append((row["model_key"], stat))
                p_one.append(stat["sign_flip_p_one_sided"])
                p_two.append(stat["sign_flip_p_two_sided"])
                stacked.append(values)
                arrays[f"{group_name}_{row['model_key']}_{metric}"] = values.astype(np.float32)
            q_one = benjamini_hochberg(p_one)
            q_two = benjamini_hochberg(p_two)
            for index, (model_key, stat) in enumerate(model_stats):
                stat["bh_q_one_sided_across_six_models"] = float(q_one[index])
                stat["bh_q_two_sided_across_six_models"] = float(q_two[index])
                group_result["models"].setdefault(model_key, {})[metric] = stat
            model_average_per_participant = np.nanmean(np.vstack(stacked), axis=0)
            group_result["global_mean_effect"][metric] = _inference_stat(
                model_average_per_participant,
                bootstrap_draws=bootstrap_draws,
                sign_draws=sign_draws,
                seed=seed_base + group_index * 1000 + metric_index * 100 + 97,
            )
            arrays[f"{group_name}_global_model_average_{metric}"] = model_average_per_participant.astype(np.float32)
        arrays[f"{group_name}_subject_ids"] = reference_subjects.astype(str)
        result_groups[group_name] = group_result

    summary = {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": "correspondence_primary_control_paired_inference",
        "analysis_fingerprint": fingerprint,
        "regime": regime,
        "feature_bank_variant": variant,
        "bank_size": int(size),
        "historical_bank_used": False,
        "participant_pairing": True,
        "bootstrap_draws": bootstrap_draws,
        "sign_flip_draws": sign_draws,
        "bh_family": "six_models_separately_within_each_story_or_transfer_direction_and_metric",
        "global_test_unit": "participant_level_mean_across_six_models",
        "groups": result_groups,
        "input_sha256": {str(path): sha256_file(path) for path in input_paths},
        "code_sha256": {str(path): sha256_file(path) for path in code_paths},
        "arrays_path": str(output_dir / "arrays.npz"),
        "control_inference_complete": True,
    }
    return _write_control_artifact(output_dir, summary, arrays, force=force)


def _selected_models_stories(spec: dict[str, Any], args: argparse.Namespace) -> tuple[list[str], list[str]]:
    models = list(args.models) if args.models else list(spec["models"].keys())
    stories = list(args.stories) if args.stories else list(spec["stories"].keys())
    for model in models:
        if model not in spec["models"]:
            raise KeyError(model)
    for story in stories:
        if story not in spec["stories"]:
            raise KeyError(story)
    return models, stories


def verify_outputs(config_path: Path, spec: dict[str, Any]) -> dict[str, Any]:
    roots = [
        _artifact_root(config_path, spec) / "controls",
        _artifact_root(config_path, spec) / "primary_controls",
    ]
    summaries = sorted(
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("summary.json")
    )
    checked = []
    errors = []
    for summary_path in summaries:
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("schema_version mismatch")
            if not summary.get("analysis_fingerprint"):
                raise ValueError("missing analysis_fingerprint")
            arrays_path = summary_path.with_name("arrays.npz")
            if not arrays_path.is_file():
                raise ValueError("missing arrays.npz")
            # Re-read NPZ metadata to catch truncated/invalid writes.
            with np.load(arrays_path, allow_pickle=True) as bundle:
                keys = list(bundle.files)
            checked.append({"summary": str(summary_path), "arrays": str(arrays_path), "array_keys": keys})
        except Exception as exc:  # pragma: no cover - defensive reporting
            errors.append({"summary": str(summary_path), "error": str(exc)})
    result = {
        "schema_version": SCHEMA_VERSION,
        "analysis_role": "correspondence_control_verification",
        "root": str(roots[0]),
        "roots": [str(root) for root in roots],
        "n_summaries": len(summaries),
        "checked": checked,
        "errors": errors,
        "passed": not errors,
    }
    verification_root = roots[0]
    verification_root.mkdir(parents=True, exist_ok=True)
    atomic_write_json(verification_root / "verification.json", result)
    if errors:
        raise RuntimeError(f"control verification failed for {len(errors)} artifacts")
    return result


def run(config_path: Path, spec: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    models, stories = _selected_models_stories(spec, args)
    stage = str(args.stage)
    data_root = args.data_root
    if stage == "inventory":
        result = inventory(
            config_path,
            spec,
            models=models,
            stories=stories,
            data_root=data_root,
        )
        return {"stage": stage, "n_rows": len(result["rows"]), "inventory": result}
    if stage == "verify":
        return verify_outputs(config_path, spec)
    if stage == "inference":
        return run_primary_control_inference(
            config_path,
            spec,
            regime=str(args.regime),
            size=256,
            variant="source_story_variance",
            force=args.force,
        )
    outputs: list[dict[str, Any]] = []
    sizes = [int(value) for value in (args.sizes or [256])]
    subject_limit = int(args.subject_limit) if args.subject_limit is not None else None
    for model_key in models:
        for story in stories:
            if stage == "primary-within":
                outputs.append(
                    fit_primary_within_condition(
                        config_path,
                        spec,
                        model_key,
                        story,
                        size=256,
                        variant="source_story_variance",
                        data_root=data_root,
                        subject_limit=subject_limit,
                        force=args.force,
                    )
                )
            if stage in {"historical", "baseline", "all"}:
                for size in sizes:
                    outputs.append(
                        fit_historical_condition(
                            config_path,
                            spec,
                            model_key,
                            story,
                            size,
                            data_root=data_root,
                            subject_limit=subject_limit,
                            save_oof=not args.no_oof,
                            force=args.force,
                        )
                    )
            if stage in {"dense", "all"}:
                outputs.append(
                    fit_dense_condition(
                        config_path,
                        spec,
                        model_key,
                        story,
                        data_root=data_root,
                        subject_limit=subject_limit,
                        save_oof=not args.no_oof,
                        force=args.force,
                    )
                )
            if stage in {"random", "all"}:
                random_path = _random_bank_path(config_path, spec, model_key)
                if not random_path.is_file():
                    raise FileNotFoundError(random_path)
                with np.load(random_path, allow_pickle=True) as bundle:
                    count = int(np.asarray(bundle["feature_ids"]).shape[0])
                indices = [int(args.random_index)] if args.random_index is not None else list(range(count))
                if args.random_limit is not None:
                    indices = indices[: int(args.random_limit)]
                for bank_index in indices:
                    outputs.append(
                        fit_random_condition(
                            config_path,
                            spec,
                            model_key,
                            story,
                            bank_index,
                            data_root=data_root,
                            subject_limit=subject_limit,
                            force=args.force,
                        )
                    )
        if stage == "primary-transfer":
            if args.source_story or args.target_story:
                if not args.source_story or not args.target_story:
                    raise ValueError("primary-transfer requires both --source-story and --target-story")
                directions = [(str(args.source_story), str(args.target_story))]
            else:
                directions = [
                    ("shapesphysical", "shapessocial"),
                    ("shapessocial", "shapesphysical"),
                ]
            for source_story, target_story in directions:
                outputs.append(
                    fit_primary_transfer_condition(
                        config_path,
                        spec,
                        model_key,
                        source_story,
                        target_story,
                        size=256,
                        variant="source_story_variance",
                        data_root=data_root,
                        subject_limit=subject_limit,
                        force=args.force,
                    )
                )
    manifest_family = "primary_controls" if stage.startswith("primary-") else "controls"
    manifest_path = _artifact_root(config_path, spec) / manifest_family / f"{stage}_manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": SCHEMA_VERSION,
            "analysis_role": f"correspondence_control_{stage}_manifest",
            "n_outputs": len(outputs),
            "outputs": [
                {
                    "model_key": row.get("model_key"),
                    "story": row.get("story"),
                    "source_story": row.get("source_story"),
                    "target_story": row.get("target_story"),
                    "analysis_role": row.get("analysis_role"),
                    "analysis_fingerprint": row.get("analysis_fingerprint"),
                    "arrays_path": row.get("arrays_path"),
                }
                for row in outputs
            ],
        },
    )
    return {"stage": stage, "n_outputs": len(outputs), "manifest": str(manifest_path)}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "inventory",
            "historical",
            "baseline",
            "dense",
            "random",
            "primary-within",
            "primary-transfer",
            "inference",
            "all",
            "verify",
        ],
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--stories", nargs="+", default=None)
    parser.add_argument("--sizes", nargs="+", type=int, default=None)
    parser.add_argument("--random-index", type=int, default=None)
    parser.add_argument("--random-limit", type=int, default=None)
    parser.add_argument("--subject-limit", type=int, default=None)
    parser.add_argument("--source-story", default=None)
    parser.add_argument("--target-story", default=None)
    parser.add_argument("--regime", choices=["within", "transfer"], default="within")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--no-oof", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    from .integrity import load_spec

    if args.data_root is not None:
        os.environ["FMRI_SAE_DATA_ROOT"] = str(args.data_root.resolve())
    if args.output_root is not None:
        os.environ["FMRI_SAE_OUTPUT_ROOT"] = str(args.output_root.resolve())
    config_path = (args.config or default_config_path()).resolve()
    result = run(config_path, load_spec(config_path), args)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
