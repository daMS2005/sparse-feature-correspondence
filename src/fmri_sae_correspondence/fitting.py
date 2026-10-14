#!/usr/bin/env python3
"""Observed within-story fits for the correspondence analysis.

This module is intentionally separate from :mod:`modeling`.  The shared
modeling helpers implement the ridge objective, scaling, metrics,
and held-out ablation.  The new code only supplies the fold-specific feature
bank and records the paired, bank-aware results.  In particular, no target
array is read while choosing a bank: each outer fold reads a bank generated
from the corresponding purged training rows by ``feature_banks``.

The LM and BOLD fits in one fold use exactly the same ``(layer, feature_id)``
coordinates.  Because feature identities can differ between outer folds, the
primary within-story correspondence is the mean of fold-level correlations;
profiles are never averaged across unmatched coordinates.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .config import default_config_path
from .feature_banks import (
    contiguous_fold_rows,
    extract_bank_predictors,
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
    build_ridge_path,
    correspondence,
    feature_groups,
    finite_rows,
    fit_ridge,
    lag_expand,
    mean_metric,
    purged_training_positions,
    within_layer_identity_null,
)

HERE = Path(__file__).resolve().parent

DEFAULT_SIZES = (64, 128, 256, 512, 1024)
DEFAULT_VARIANTS = ("source_story_variance", "source_story_prevalence")


def _data_root(config_path: Path, spec: dict[str, Any]) -> Path:
    return resolve_data_root(config_path, spec)


def _artifact_root(config_path: Path, spec: dict[str, Any]) -> Path:
    del config_path
    return (resolve_output_root(spec=spec) / str(spec["paths"]["artifact_root"])).resolve()


def _feature_bank_root(config_path: Path, spec: dict[str, Any]) -> Path:
    """Resolve either accepted feature-bank layout without changing inputs."""

    return _artifact_root(config_path, spec) / str(spec["paths"]["feature_bank_root"])


def _path_from_repo(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _predictor_path(root: Path, spec: dict[str, Any], model_key: str, story: str) -> Path:
    relative = str(spec["paths"]["predictor_pattern"]).format(
        model=model_key,
        story=story,
    )
    return _path_from_repo(root, relative)


def _brain_path(root: Path, spec: dict[str, Any], story: str) -> Path:
    return _path_from_repo(root, spec["stories"][story]["brain_bundle"])


def load_condition(
    config_path: Path, spec: dict[str, Any], model_key: str, story: str
) -> dict[str, Any]:
    """Load predictor-side arrays, LM targets, and participant BOLD targets.

    This loader intentionally accepts only the canonical evaluation-blind
    predictor bundle and the canonical full participant target bundle.  It
    does not inspect old fitted results, which prevents an outcome from
    entering source-story bank selection accidentally.
    """

    root = _data_root(config_path, spec)
    predictor_path = _predictor_path(root, spec, model_key, story)
    brain_path = _brain_path(root, spec, story)
    if not predictor_path.is_file():
        raise FileNotFoundError(f"missing canonical predictor bundle: {predictor_path}")
    if not brain_path.is_file():
        raise FileNotFoundError(f"missing canonical BOLD target bundle: {brain_path}")

    with np.load(predictor_path, allow_pickle=True) as z:
        required = {"all_sae_tr_means", "lm_target", "tr_indices", "selected_layers"}
        missing = sorted(required - set(z.files))
        if missing:
            raise ValueError(f"{predictor_path}: missing fields {missing}")
        all_sae = np.asarray(z["all_sae_tr_means"])
        lm_target = np.asarray(z["lm_target"], dtype=np.float64)
        tr_indices = np.asarray(z["tr_indices"], dtype=np.int64)
        layers = np.asarray(z["selected_layers"], dtype=np.int64)
        model_id = str(np.asarray(z["model_id"]).reshape(-1)[0]) if "model_id" in z else ""
        bundle_story = str(np.asarray(z["story"]).reshape(-1)[0]) if "story" in z else ""
    if all_sae.ndim != 3:
        raise ValueError(f"{predictor_path}: all_sae_tr_means must be [layer,time,feature]")
    if all_sae.shape[0] != len(layers) or all_sae.shape[1] != len(tr_indices):
        raise ValueError(f"{predictor_path}: inconsistent all_sae/layer/time dimensions")
    if len(lm_target) != len(tr_indices):
        raise ValueError(f"{predictor_path}: LM target and TR lengths differ")
    if bundle_story and bundle_story != story:
        raise ValueError(f"{predictor_path}: story={bundle_story!r}, expected {story!r}")

    with np.load(brain_path, allow_pickle=True) as z:
        required = {"values", "tr_indices", "subject_ids", "censor_mask", "target_names"}
        missing = sorted(required - set(z.files))
        if missing:
            raise ValueError(f"{brain_path}: missing fields {missing}")
        values = np.asarray(z["values"], dtype=np.float64)
        brain_tr = np.asarray(z["tr_indices"], dtype=np.int64)
        subjects = np.asarray(z["subject_ids"], dtype=str)
        censor = np.asarray(z["censor_mask"], dtype=bool)
        target_names = np.asarray(z["target_names"], dtype=str)
    if values.ndim != 2 or len(values) != len(brain_tr) or len(values) != len(subjects):
        raise ValueError(f"{brain_path}: inconsistent BOLD row dimensions")
    if len(censor) != len(values):
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
    if len(subject_targets) != expected_subjects:
        raise ValueError(
            f"{story}: expected {expected_subjects} participants, found {len(subject_targets)}"
        )
    expected_trs = int(spec["stories"][story]["expected_trs"])
    if len(tr_indices) != expected_trs:
        raise ValueError(f"{story}: expected {expected_trs} TRs, found {len(tr_indices)}")
    return {
        "model_key": model_key,
        "story": story,
        "predictor_path": predictor_path,
        "brain_path": brain_path,
        "all_sae_tr_means": all_sae,
        "lm_target": lm_target,
        "tr_indices": tr_indices,
        "selected_layers": layers,
        "model_id": model_id,
        "subject_targets": subject_targets,
        "target_names": target_names,
    }


def bank_path(
    config_path: Path,
    spec: dict[str, Any],
    model_key: str,
    story: str,
    *,
    fold: int | None = None,
    source_story: str | None = None,
    target_story: str | None = None,
) -> Path:
    """Resolve a source-feature-bank JSON path from the configured artifact layout."""

    root = _feature_bank_root(config_path, spec)
    if fold is None:
        if source_story is None or target_story is None:
            raise ValueError("cross-story bank paths require source_story and target_story")
        return root / "cross_story" / model_key / f"{source_story}_to_{target_story}" / "banks.json"
    return root / "within_story" / model_key / story / f"fold_{int(fold):02d}" / "banks.json"


def _variant_entries(record: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return all supported bank variants in deterministic order."""

    output: list[tuple[str, dict[str, Any]]] = []
    output.append(("source_story_variance", record["source_variance"]))
    output.append(("source_story_prevalence", record["source_prevalence"]))
    pruned = record.get("correlation_pruned", {})
    for threshold in sorted(pruned, key=lambda value: float(value)):
        label = f"source_story_correlation_pruned_{float(threshold):.2f}".replace(".", "p")
        output.append((label, pruned[threshold]))
    return output


def get_variant(record: dict[str, Any], variant: str) -> dict[str, Any]:
    for name, value in _variant_entries(record):
        if name == variant:
            return value
    available = [name for name, _ in _variant_entries(record)]
    raise KeyError(f"unknown bank variant {variant!r}; available={available}")


def _load_bank(path: Path, variant: str, size: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(
            f"source feature bank missing: {path}; run the banks stage before fitting"
        )
    record = json.loads(path.read_text(encoding="utf-8"))
    bank = get_variant(record, variant)
    key = str(int(size))
    if key not in bank.get("banks", {}):
        raise KeyError(f"{path}: variant {variant} has no size {size}")
    entry = bank["banks"][key]
    if not bool(entry.get("complete")) or int(entry.get("actual_size", -1)) != int(size):
        raise ValueError(f"{path}: incomplete bank {variant}/{size}: {entry.get('actual_size')}")
    if len(entry.get("layers", [])) != int(size) or len(entry.get("feature_ids", [])) != int(size):
        raise ValueError(f"{path}: malformed bank identity for {variant}/{size}")
    if len(set(zip(entry["layers"], entry["feature_ids"], strict=False))) != int(size):
        raise ValueError(f"{path}: duplicate feature-layer identities for {variant}/{size}")
    return record, bank


def _extract_bank(condition: dict[str, Any], bank: dict[str, Any], size: int) -> tuple[np.ndarray, np.ndarray]:
    entry = bank["banks"][str(int(size))]
    expected_layers = np.asarray(condition["selected_layers"], dtype=np.int64)
    if not np.array_equal(
        np.asarray(bank.get("layers", expected_layers), dtype=np.int64), expected_layers
    ):
        raise ValueError("bank sampled layers differ from the predictor bundle layers")
    x = extract_bank_predictors(
        condition["all_sae_tr_means"],
        expected_layers,
        bank,
        int(size),
    ).astype(np.float64, copy=False)
    names = np.asarray(entry["feature_names"], dtype=str)
    if x.shape != (len(condition["tr_indices"]), int(size)):
        raise ValueError(f"bank predictor shape mismatch: {x.shape}")
    if len(names) != int(size):
        raise ValueError("bank feature name length mismatch")
    return x, names


def select_alpha_on_outer_train(
    x: np.ndarray,
    y: np.ndarray,
    outer_train: np.ndarray,
    *,
    alphas: np.ndarray,
    metric: str,
    n_time: int,
    inner_folds: int,
    purge_adjacent: int,
) -> tuple[float, np.ndarray, int]:
    """Select alpha using only one outer-training set.

    ``modeling.select_alpha`` assumes a fixed predictor matrix and derives
    the outer training rows itself.  The current within-story analysis has a
    fold-specific bank, so this equivalent helper takes the already-fixed
    outer-training rows explicitly.  Inner validation blocks are contiguous
    chunks of those rows and are purged in the original TR coordinate system.
    """

    outer_train = np.asarray(outer_train, dtype=np.int64)
    if len(outer_train) < inner_folds:
        raise ValueError("outer training set is smaller than the inner-fold count")
    valid_y = finite_rows(y)
    scores = np.zeros(len(alphas), dtype=np.float64)
    score_counts = np.zeros(len(alphas), dtype=np.int64)
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
        path = build_ridge_path(x, y, inner_train)
        for index, alpha in enumerate(np.asarray(alphas, dtype=np.float64)):
            fit = path.at(float(alpha))
            prediction = fit.predict(x[validation])
            value = mean_metric(y[validation], prediction, fit.y_mean, metric)
            if np.isfinite(value):
                scores[index] += float(value)
                score_counts[index] += 1
        used += 1
    if used == 0:
        raise RuntimeError("no usable inner folds for alpha selection")
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = np.divide(
            scores,
            score_counts,
            out=np.full_like(scores, np.nan),
            where=score_counts > 0,
        )
    finite = np.isfinite(scores)
    if not np.any(finite):
        raise RuntimeError("all inner-fold alpha scores were non-finite")
    best_indices = np.flatnonzero(finite & (scores == np.nanmax(scores)))
    return float(np.asarray(alphas, dtype=np.float64)[int(best_indices[0])]), scores, used


def evaluate_fit(
    fit: Any,
    x_test: np.ndarray,
    y_test: np.ndarray,
    groups: Sequence[np.ndarray],
) -> dict[str, Any]:
    """Evaluate prediction and held-out conditional reliance for one fit."""

    valid = finite_rows(y_test)
    if len(valid) == 0:
        raise ValueError("test target has no finite rows")
    x_valid = np.asarray(x_test[valid], dtype=np.float64)
    y_valid = np.asarray(y_test[valid], dtype=np.float64)
    prediction = fit.predict(x_valid)
    full_sse = np.sum((y_valid - prediction) ** 2, axis=0)
    denominator = np.sum((y_valid - fit.y_mean[None, :]) ** 2, axis=0)
    deltas = np.zeros((len(groups), y_valid.shape[1]), dtype=np.float64)
    for group_index, columns in enumerate(groups):
        ablated_prediction = prediction - fit.contribution(x_valid, np.asarray(columns, dtype=int))
        ablated_sse = np.sum((y_valid - ablated_prediction) ** 2, axis=0)
        deltas[group_index] = ablated_sse - full_sse
    with np.errstate(divide="ignore", invalid="ignore"):
        scaled = np.full_like(deltas, np.nan)
        np.divide(deltas, denominator[None, :], out=scaled, where=denominator[None, :] > 0)
        reliance = np.nanmean(scaled, axis=1)
    return {
        "valid_positions": valid,
        "prediction": prediction,
        "mean_r": float(mean_metric(y_valid, prediction, fit.y_mean, "pearson_r")),
        "mean_r2_oos": float(mean_metric(y_valid, prediction, fit.y_mean, "r2_oos")),
        "reliance": reliance,
        "sse": full_sse,
        "denominator": denominator,
    }


def _fold_record(
    *,
    fold: int,
    alpha: float,
    score_grid: np.ndarray,
    n_inner_used: int,
    train: np.ndarray,
    test: np.ndarray,
    fit_result: dict[str, Any],
) -> dict[str, Any]:
    return {
        "fold": int(fold),
        "alpha": float(alpha),
        "inner_score_grid": np.asarray(score_grid, dtype=np.float64).tolist(),
        "n_inner_folds_used": int(n_inner_used),
        "n_train": int(len(train)),
        "n_test": int(len(fit_result["valid_positions"])),
        "mean_r": float(fit_result["mean_r"]),
        "mean_r2_oos": float(fit_result["mean_r2_oos"]),
    }


def _fold_correspondence(
    brain_profile: np.ndarray, lm_profile: np.ndarray, size: int
) -> dict[str, float]:
    return correspondence(
        np.asarray(brain_profile, dtype=np.float64),
        np.asarray(lm_profile, dtype=np.float64),
        top_k=max(1, min(32, int(round(float(size) * 0.125)))),
    )


def _finite_mean(values: np.ndarray, axis: int | None = None) -> np.ndarray | float:
    with np.errstate(invalid="ignore"):
        return np.nanmean(np.asarray(values, dtype=np.float64), axis=axis)


def global_finite_training_rows(y: np.ndarray, allowed_rows: np.ndarray) -> np.ndarray:
    """Return finite training positions in the original full-story index.

    Calling ``finite_rows(y[allowed_rows])`` alone returns positions local to
    the sliced array.  Passing those local positions to a fit over full-story
    ``X``/``Y`` silently trains on the wrong temporal rows whenever the outer
    training set does not begin at zero.
    """

    return np.intersect1d(
        np.asarray(allowed_rows, dtype=np.int64),
        finite_rows(y),
        assume_unique=True,
    )


def fit_within_condition(
    config_path: Path,
    spec: dict[str, Any],
    model_key: str,
    story: str,
    variant: str,
    size: int,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Fit one model/story/bank-size within-story condition."""

    condition = load_condition(config_path, spec, model_key, story)
    cv = spec["cross_validation"]
    n_folds = int(cv["outer_folds"])
    inner_folds = int(cv["inner_folds"])
    purge = int(cv["purge_adjacent_trs"])
    alphas = np.asarray(cv["alpha_grid"], dtype=np.float64)
    fold_rows = contiguous_fold_rows(
        len(condition["tr_indices"]), n_folds, purge_adjacent_trs=purge
    )
    bank_paths = [
        bank_path(config_path, spec, model_key, story, fold=index)
        for index in range(n_folds)
    ]
    source_paths: list[Path] = [condition["predictor_path"], condition["brain_path"], *bank_paths]
    code_paths = [
        HERE / "fitting.py",
        HERE / "feature_banks.py",
        HERE / "modeling.py",
        HERE / "cli.py",
    ]
    fingerprint = provenance_fingerprint(
        config_path,
        input_paths=source_paths,
        code_paths=code_paths,
        extra={
            "analysis": "correspondence_within_story",
            "model_key": model_key,
            "story": story,
            "variant": variant,
            "size": int(size),
            "inner_folds": inner_folds,
            "outer_folds": n_folds,
        },
    )
    output_dir = _artifact_root(config_path, spec) / "fit" / f"{model_key}_{story}" / variant / f"n{int(size)}"
    summary_path = output_dir / "summary.json"
    arrays_path = output_dir / "arrays.npz"
    if not force:
        cached = assert_cache_fingerprint(summary_path, fingerprint)
        if cached is not None:
            if not arrays_path.is_file():
                raise RuntimeError(f"summary cache exists without arrays: {summary_path}")
            return cached

    n_subjects = len(condition["subject_targets"])
    n_time = len(condition["tr_indices"])
    n_parcels = next(iter(condition["subject_targets"].values())).shape[1]
    lm_dims = condition["lm_target"].shape[1]
    brain_r_reliance = np.full((n_folds, n_subjects, int(size)), np.nan, dtype=np.float32)
    brain_r2_reliance = np.full_like(brain_r_reliance, np.nan)
    lm_r_reliance = np.full((n_folds, int(size)), np.nan, dtype=np.float32)
    lm_r2_reliance = np.full_like(lm_r_reliance, np.nan)
    brain_r_mean = np.full((n_folds, n_subjects), np.nan, dtype=np.float32)
    brain_r2_mean = np.full_like(brain_r_mean, np.nan)
    brain_r2_oos = np.full_like(brain_r_mean, np.nan)
    brain_r_oos = np.full_like(brain_r_mean, np.nan)
    lm_r_oos = np.full(n_folds, np.nan, dtype=np.float32)
    lm_r2_oos = np.full_like(lm_r_oos, np.nan)
    brain_r_oof = np.full((n_subjects, n_time, n_parcels), np.nan, dtype=np.float32)
    brain_r2_oof = np.full_like(brain_r_oof, np.nan)
    lm_r_oof = np.full((n_time, lm_dims), np.nan, dtype=np.float32)
    lm_r2_oof = np.full_like(lm_r_oof, np.nan)
    subject_ids = np.asarray(sorted(condition["subject_targets"]), dtype=str)
    folds_summary: list[dict[str, Any]] = []
    fold_names: list[list[str]] = []
    fold_correspondence_r: list[dict[str, float]] = []
    fold_correspondence_r2: list[dict[str, float]] = []
    fold_identity_nulls: list[np.ndarray] = []
    fold_bank_bundle_hashes: list[str | None] = []
    lm_r_fold_records: list[dict[str, Any]] = []
    lm_r2_fold_records: list[dict[str, Any]] = []
    brain_r_records: list[list[dict[str, Any]]] = []
    brain_r2_records: list[list[dict[str, Any]]] = []
    scaling: dict[str, list[np.ndarray]] = {
        "lm_r_x_mean": [],
        "lm_r_x_scale": [],
        "lm_r_y_mean": [],
        "lm_r_y_scale": [],
        "lm_r2_x_mean": [],
        "lm_r2_x_scale": [],
        "lm_r2_y_mean": [],
        "lm_r2_y_scale": [],
        "brain_r2_x_mean": [],
        "brain_r2_x_scale": [],
        "brain_r2_y_mean": [],
        "brain_r2_y_scale": [],
    }
    started = time.time()

    for fold_index, (outer_train, outer_test) in enumerate(fold_rows):
        bank_file = bank_paths[fold_index]
        bank_record, _ = _load_bank(bank_file, variant, int(size))
        fold_bank_bundle_hashes.append(bank_record.get("bank_bundle_sha256"))
        selection_rows = np.asarray(bank_record.get("selection_rows", []), dtype=np.int64)
        if not np.array_equal(selection_rows, outer_train):
            raise RuntimeError(
                f"{bank_file}: fold selection rows do not equal purged outer training rows"
            )
        # Every target uses this same bank object and hence the same identities.
        x_base, names = _extract_bank(condition, get_variant(bank_record, variant), int(size))
        fold_names.append(names.tolist())
        x_lm = x_base
        x_brain = lag_expand(x_base, spec["targets"]["brain"]["lags"])
        brain_groups = feature_groups(int(size), len(spec["targets"]["brain"]["lags"]))
        lm_groups = feature_groups(int(size), 1)

        # LM fits: one target matrix, two independently selected penalties.
        lm_r_alpha, lm_r_grid, lm_r_inner = select_alpha_on_outer_train(
            x_lm,
            condition["lm_target"],
            outer_train,
            alphas=alphas,
            metric="pearson_r",
            n_time=n_time,
            inner_folds=inner_folds,
            purge_adjacent=purge,
        )
        lm_valid_train = global_finite_training_rows(
            condition["lm_target"], outer_train
        )
        lm_r_fit = fit_ridge(
            x_lm,
            condition["lm_target"],
            lm_valid_train,
            lm_r_alpha,
        )
        lm_r_eval = evaluate_fit(lm_r_fit, x_lm[outer_test], condition["lm_target"][outer_test], lm_groups)
        # ``valid_positions`` is local to ``outer_test``; use explicit index
        # mapping back to the global TR coordinate system.
        lm_valid_global = outer_test[np.asarray(lm_r_eval["valid_positions"], dtype=int)]
        lm_r_oof[lm_valid_global] = lm_r_eval["prediction"].astype(np.float32)
        lm_r_reliance[fold_index] = np.asarray(lm_r_eval["reliance"], dtype=np.float32)
        lm_r_oos[fold_index] = float(lm_r_eval["mean_r2_oos"])
        lm_r_fold_records.append(
            _fold_record(
                fold=fold_index,
                alpha=lm_r_alpha,
                score_grid=lm_r_grid,
                n_inner_used=lm_r_inner,
                train=outer_train,
                test=outer_test,
                fit_result=lm_r_eval,
            )
        )
        scaling["lm_r_x_mean"].append(np.asarray(lm_r_fit.x_mean))
        scaling["lm_r_x_scale"].append(np.asarray(lm_r_fit.x_scale))
        scaling["lm_r_y_mean"].append(np.asarray(lm_r_fit.y_mean))
        scaling["lm_r_y_scale"].append(np.asarray(lm_r_fit.y_scale))

        lm_r2_alpha, lm_r2_grid, lm_r2_inner = select_alpha_on_outer_train(
            x_lm,
            condition["lm_target"],
            outer_train,
            alphas=alphas,
            metric="r2_oos",
            n_time=n_time,
            inner_folds=inner_folds,
            purge_adjacent=purge,
        )
        lm_r2_fit = fit_ridge(
            x_lm,
            condition["lm_target"],
            lm_valid_train,
            lm_r2_alpha,
        )
        lm_r2_eval = evaluate_fit(lm_r2_fit, x_lm[outer_test], condition["lm_target"][outer_test], lm_groups)
        lm_r2_valid_global = outer_test[np.asarray(lm_r2_eval["valid_positions"], dtype=int)]
        lm_r2_oof[lm_r2_valid_global] = lm_r2_eval["prediction"].astype(np.float32)
        lm_r2_reliance[fold_index] = np.asarray(lm_r2_eval["reliance"], dtype=np.float32)
        lm_r2_oos[fold_index] = float(lm_r2_eval["mean_r2_oos"])
        lm_r2_fold_records.append(
            _fold_record(
                fold=fold_index,
                alpha=lm_r2_alpha,
                score_grid=lm_r2_grid,
                n_inner_used=lm_r2_inner,
                train=outer_train,
                test=outer_test,
                fit_result=lm_r2_eval,
            )
        )
        scaling["lm_r2_x_mean"].append(np.asarray(lm_r2_fit.x_mean))
        scaling["lm_r2_x_scale"].append(np.asarray(lm_r2_fit.x_scale))
        scaling["lm_r2_y_mean"].append(np.asarray(lm_r2_fit.y_mean))
        scaling["lm_r2_y_scale"].append(np.asarray(lm_r2_fit.y_scale))

        # Participant-specific BOLD fits.
        fold_brain_r_records: list[dict[str, Any]] = []
        fold_brain_r2_records: list[dict[str, Any]] = []
        fold_brain_r2_profiles: list[np.ndarray] = []
        for subject_index, subject in enumerate(subject_ids.tolist()):
            target = condition["subject_targets"][str(subject)]
            r_alpha, r_grid, r_inner = select_alpha_on_outer_train(
                x_brain,
                target,
                outer_train,
                alphas=alphas,
                metric="pearson_r",
                n_time=n_time,
                inner_folds=inner_folds,
                purge_adjacent=purge,
            )
            valid_train = global_finite_training_rows(target, outer_train)
            r_fit = fit_ridge(x_brain, target, valid_train, r_alpha)
            r_eval = evaluate_fit(r_fit, x_brain[outer_test], target[outer_test], brain_groups)
            r_valid_global = outer_test[np.asarray(r_eval["valid_positions"], dtype=int)]
            brain_r_oof[subject_index, r_valid_global] = r_eval["prediction"].astype(np.float32)
            brain_r_reliance[fold_index, subject_index] = np.asarray(r_eval["reliance"], dtype=np.float32)
            brain_r_mean[fold_index, subject_index] = float(r_eval["mean_r"])
            brain_r_oos[fold_index, subject_index] = float(r_eval["mean_r2_oos"])
            fold_brain_r_records.append(
                _fold_record(
                    fold=fold_index,
                    alpha=r_alpha,
                    score_grid=r_grid,
                    n_inner_used=r_inner,
                    train=outer_train,
                    test=outer_test,
                    fit_result=r_eval,
                )
            )

            r2_alpha, r2_grid, r2_inner = select_alpha_on_outer_train(
                x_brain,
                target,
                outer_train,
                alphas=alphas,
                metric="r2_oos",
                n_time=n_time,
                inner_folds=inner_folds,
                purge_adjacent=purge,
            )
            r2_fit = fit_ridge(x_brain, target, valid_train, r2_alpha)
            r2_eval = evaluate_fit(r2_fit, x_brain[outer_test], target[outer_test], brain_groups)
            r2_valid_global = outer_test[np.asarray(r2_eval["valid_positions"], dtype=int)]
            brain_r2_oof[subject_index, r2_valid_global] = r2_eval["prediction"].astype(np.float32)
            brain_r2_reliance[fold_index, subject_index] = np.asarray(r2_eval["reliance"], dtype=np.float32)
            brain_r2_mean[fold_index, subject_index] = float(r2_eval["mean_r"])
            brain_r2_oos[fold_index, subject_index] = float(r2_eval["mean_r2_oos"])
            fold_brain_r2_records.append(
                _fold_record(
                    fold=fold_index,
                    alpha=r2_alpha,
                    score_grid=r2_grid,
                    n_inner_used=r2_inner,
                    train=outer_train,
                    test=outer_test,
                    fit_result=r2_eval,
                )
            )
            # The x scaling is target-independent; retain one copy per fold.
            if subject_index == 0:
                scaling["brain_r2_x_mean"].append(np.asarray(r2_fit.x_mean))
                scaling["brain_r2_x_scale"].append(np.asarray(r2_fit.x_scale))
            scaling["brain_r2_y_mean"].append(np.asarray(r2_fit.y_mean))
            scaling["brain_r2_y_scale"].append(np.asarray(r2_fit.y_scale))
            fold_brain_r2_profiles.append(np.asarray(r2_eval["reliance"], dtype=np.float64))
        brain_r_records.append(fold_brain_r_records)
        brain_r2_records.append(fold_brain_r2_records)

        mean_brain_r = np.nanmean(brain_r_reliance[fold_index].astype(np.float64), axis=0)
        mean_brain_r2 = np.nanmean(brain_r2_reliance[fold_index].astype(np.float64), axis=0)
        fold_correspondence_r.append(_fold_correspondence(mean_brain_r, lm_r_reliance[fold_index], int(size)))
        fold_correspondence_r2.append(_fold_correspondence(mean_brain_r2, lm_r2_reliance[fold_index], int(size)))
        identity = within_layer_identity_null(
            mean_brain_r2,
            lm_r2_reliance[fold_index],
            np.asarray(fold_names[-1], dtype=str),
            n_permutations=int(spec["nulls"]["feature_identity"]["permutations"]),
            seed=int(spec["nulls"]["seeds"]["feature_identity"]) + fold_index,
        )
        fold_identity_nulls.append(np.asarray(identity["null"], dtype=np.float32))
        folds_summary.append(
            {
                "fold": int(fold_index),
                "train_rows": [int(value) for value in outer_train.tolist()],
                "test_rows": [int(value) for value in outer_test.tolist()],
                "feature_names": fold_names[-1],
                "brain_r_selected": fold_brain_r_records,
                "brain_r2_selected": fold_brain_r2_records,
                "lm_r_selected": lm_r_fold_records[-1],
                "lm_r2_selected": lm_r2_fold_records[-1],
                "correspondence_r_selected": fold_correspondence_r[-1],
                "correspondence_r2_selected": fold_correspondence_r2[-1],
                "identity_null_r2_selected": {
                    key: value for key, value in identity.items() if key != "null"
                },
            }
        )

    # The primary within-story statistic is a fold-level mean.  This avoids
    # treating fold-specific banks as if they were one common coordinate set.
    aggregate_r = {
        key: float(np.nanmean([row[key] for row in fold_correspondence_r]))
        for key in ("spearman", "pearson", "topk_jaccard")
    }
    aggregate_r2 = {
        key: float(np.nanmean([row[key] for row in fold_correspondence_r2]))
        for key in ("spearman", "pearson", "topk_jaccard")
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    save_arrays: dict[str, Any] = {
        "subject_ids": subject_ids,
        "fold_feature_names": np.asarray(fold_names, dtype=str),
        "brain_r_reliance_by_fold": brain_r_reliance,
        "brain_r2_reliance_by_fold": brain_r2_reliance,
        "lm_r_reliance_by_fold": lm_r_reliance,
        "lm_r2_reliance_by_fold": lm_r2_reliance,
        "brain_r_mean_r_by_fold": brain_r_mean,
        "brain_r2_mean_r_by_fold": brain_r2_mean,
        "brain_r_mean_r2_oos_by_fold": brain_r_oos,
        "brain_r2_mean_r2_oos_by_fold": brain_r2_oos,
        "lm_r_mean_r2_oos_by_fold": lm_r_oos,
        "lm_r2_mean_r2_oos_by_fold": lm_r2_oos,
        "brain_r_oof": brain_r_oof,
        "brain_r2_oof": brain_r2_oof,
        "lm_r_oof": lm_r_oof,
        "lm_r2_oof": lm_r2_oof,
        "identity_null_r2_by_fold": np.asarray(fold_identity_nulls, dtype=np.float32),
    }
    for key, values in scaling.items():
        if values:
            save_arrays[key] = np.asarray(values, dtype=np.float32)
    np.savez_compressed(arrays_path, **save_arrays)
    summary = {
        "schema_version": 1,
        "analysis_role": "correspondence_within_story_observed_fit",
        "analysis_fingerprint": fingerprint,
        "model_key": model_key,
        "model_id": condition["model_id"],
        "story": story,
        "feature_bank_variant": variant,
        "bank_size": int(size),
        "n_subjects": int(n_subjects),
        "n_time": int(n_time),
        "n_parcels": int(n_parcels),
        "lm_dimensions": int(lm_dims),
        "n_outer_folds": int(n_folds),
        "n_inner_folds": int(inner_folds),
        "purge_adjacent_trs": int(purge),
        "brain_lags": [int(value) for value in spec["targets"]["brain"]["lags"]],
        "selection_metric_models": ["pearson_r", "r2_oos"],
        "primary_reliance_metric": "r2_selected_held_out_delta_r2_oos",
        "paired_feature_coordinates_within_fold": True,
        "feature_selection_outcome_free": True,
        "feature_selection_rows_are_purged_outer_training_rows": True,
        "correspondence_aggregation": "mean_of_fold_level_correspondence",
        "brain_r_selected": {
            "mean_fold_r": float(np.nanmean(brain_r_mean)),
            "mean_fold_r2_oos": float(np.nanmean(brain_r_oos)),
        },
        "brain_r2_selected": {
            "mean_fold_r": float(np.nanmean(brain_r2_mean)),
            "mean_fold_r2_oos": float(np.nanmean(brain_r2_oos)),
        },
        "lm_r_selected": {
            "mean_fold_r2_oos": float(np.nanmean(lm_r_oos)),
            "folds": lm_r_fold_records,
        },
        "lm_r2_selected": {
            "mean_fold_r2_oos": float(np.nanmean(lm_r2_oos)),
            "folds": lm_r2_fold_records,
        },
        "correspondence_r_selected": {
            "folds": fold_correspondence_r,
            "mean_of_fold_correspondence": aggregate_r,
        },
        "correspondence_r2_selected": {
            "folds": fold_correspondence_r2,
            "mean_of_fold_correspondence": aggregate_r2,
        },
        "identity_null_r2_selected": {
            "n_permutations_per_fold": int(spec["nulls"]["feature_identity"]["permutations"]),
            "folds": [
                {
                    "observed": row["identity_null_r2_selected"]["observed"],
                    "p_value": row["identity_null_r2_selected"]["p_value"],
                    "null_mean": row["identity_null_r2_selected"]["null_mean"],
                    "null_sd": row["identity_null_r2_selected"]["null_sd"],
                }
                for row in folds_summary
            ],
        },
        "folds": folds_summary,
        "fold_bank_bundle_sha256": fold_bank_bundle_hashes,
        "input_sha256": {str(path): sha256_file(path) for path in source_paths},
        "code_sha256": {str(path): sha256_file(path) for path in code_paths},
        "arrays_path": str(arrays_path),
        "elapsed_seconds": float(time.time() - started),
        "confirmatory_eligible": False,
    }
    atomic_write_json(summary_path, summary)
    return summary


def _selected_models_stories(spec: dict[str, Any], args: Any) -> tuple[list[str], list[str]]:
    model_filter = getattr(args, "models", None)
    story_filter = getattr(args, "stories", None)
    models = [str(value) for value in (model_filter or spec["models"].keys())]
    stories = [str(value) for value in (story_filter or spec["stories"].keys())]
    for model in models:
        if model not in spec["models"]:
            raise KeyError(model)
    for story in stories:
        if story not in spec["stories"]:
            raise KeyError(story)
    return models, stories


def run(*, config_path: Path, spec: dict[str, Any], args: Any) -> dict[str, Any]:
    """Run selected within-story conditions from the package CLI."""

    sizes = tuple(
        int(value)
        for value in (
            getattr(args, "sizes", None)
            or spec["feature_bank"].get("sizes", DEFAULT_SIZES)
        )
    )
    variants = tuple(
        str(value)
        for value in (
            getattr(args, "variants", None)
            or ["source_story_variance", "source_story_prevalence", "source_story_correlation_pruned_0p80"]
        )
    )
    models, stories = _selected_models_stories(spec, args)
    force = bool(getattr(args, "force", False))
    summaries: list[dict[str, Any]] = []
    for model_key in models:
        for story in stories:
            for variant in variants:
                for size in sizes:
                    print(
                        f"analysis fit {model_key}/{story}/{variant}/n{size}",
                        flush=True,
                    )
                    summaries.append(
                        fit_within_condition(
                            config_path,
                            spec,
                            model_key,
                            story,
                            variant,
                            int(size),
                            force=force,
                        )
                    )
    manifest = {
        "schema_version": 1,
        "analysis_role": "correspondence_within_story_fit_manifest",
        "conditions": [
            {
                "model_key": row["model_key"],
                "story": row["story"],
                "feature_bank_variant": row["feature_bank_variant"],
                "bank_size": row["bank_size"],
                "analysis_fingerprint": row["analysis_fingerprint"],
                "summary_path": row["arrays_path"].replace("arrays.npz", "summary.json"),
            }
            for row in summaries
        ],
    }
    manifest_path = _artifact_root(config_path, spec) / "fit" / "manifest.json"
    atomic_write_json(manifest_path, manifest)
    return {
        "stage": "fit",
        "n_conditions": len(summaries),
        "manifest": str(manifest_path),
        "summaries": summaries,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--stories", nargs="+", default=None)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--sizes", nargs="+", type=int, default=None)
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
    # Full condition summaries are persisted to artifact JSON/NPZ files; keep
    # command output compact for batch and interactive runs.
    print(json.dumps({key: value for key, value in result.items() if key != "summaries"}, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
