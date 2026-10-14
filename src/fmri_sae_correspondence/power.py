#!/usr/bin/env python3
"""Fail-closed semi-synthetic power analysis.

The analysis specification calls for a power analysis based on the
actual predictor geometry, folds, censoring pattern, residual covariance, and
target dimensionality. This module supplies that implementation and does not
manufacture a result when the required design is unavailable.

The full runner can consume a design ``.npz`` containing:

``x``
    Predictor matrix [time, feature] after the prescribed alignment.
``brain_residual`` and ``lm_residual``
    Residual matrices [time, target] with empirical covariance and the desired
    target dimensionalities.  They must be derived without using simulated
    target information.
``folds``
    Optional object array of ``(train_indices, test_indices)`` pairs.  If
    omitted, five contiguous purged folds are generated from ``x``.
``feature_groups``
    Optional object array of feature-index arrays; default is one group per
    predictor column.  For lag-expanded BOLD designs, all lag columns for one
    SAE feature should be one group.

The simulation injects two target systems with prescribed latent reliance
profiles and evaluates their held-out grouped-ablation profiles.  The paired
simulation's oracle grid is defined in the primary Spearman metric (the legacy
single-matrix helper retains its Pearson contract).  Signal multipliers are
calibrated separately to each system's mean per-target empirical residual SD.
It uses training-only scaling and training-mean out-of-sample R² throughout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

try:
    from scipy.stats import rankdata
except ImportError:  # pragma: no cover - the repository runtime includes scipy
    rankdata = None  # type: ignore[assignment]

try:
    from scipy.linalg import cho_factor, cho_solve
except ImportError:  # pragma: no cover - full power already requires scipy
    cho_factor = None  # type: ignore[assignment]
    cho_solve = None  # type: ignore[assignment]

from .config import default_config_path, load_config
from .feature_banks import sha256_file, stable_hash, write_json
from .integrity import resolve_output_root

HERE = Path(__file__).resolve().parent


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float).reshape(-1)
    b = np.asarray(b, dtype=float).reshape(-1)
    mask = np.isfinite(a) & np.isfinite(b)
    if int(mask.sum()) < 3:
        return float("nan")
    a = a[mask] - np.mean(a[mask])
    b = b[mask] - np.mean(b[mask])
    denominator = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    return float(np.sum(a * b) / denominator) if denominator > 0 else float("nan")


def profile_correlation(a: np.ndarray, b: np.ndarray, *, method: str = "spearman") -> float:
    """Correlate two feature-reliance profiles without silently dropping all ties."""
    if method == "spearman":
        if rankdata is None:
            raise RuntimeError("scipy is required for Spearman power calculations")
        return _pearson(rankdata(np.asarray(a, dtype=float)), rankdata(np.asarray(b, dtype=float)))
    if method == "pearson":
        return _pearson(a, b)
    raise ValueError(f"unknown profile correlation method: {method}")


def contiguous_purged_folds(n_rows: int, *, n_folds: int = 5, purge_adjacent: int = 4) -> list[tuple[np.ndarray, np.ndarray]]:
    """Construct contiguous test blocks and remove their adjacent rows from train."""
    if n_rows < n_folds or n_folds < 2:
        raise ValueError("n_rows must be at least n_folds and n_folds must be >= 2")
    all_rows = np.arange(int(n_rows), dtype=np.int64)
    blocks = [np.asarray(block, dtype=np.int64) for block in np.array_split(all_rows, n_folds)]
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for test in blocks:
        train = np.setdiff1d(all_rows, test, assume_unique=True)
        if purge_adjacent:
            forbidden = np.concatenate(
                [
                    np.arange(max(0, int(row) - purge_adjacent), min(n_rows, int(row) + purge_adjacent + 1), dtype=np.int64)
                    for row in test
                ]
            )
            train = np.setdiff1d(train, np.unique(forbidden), assume_unique=True)
        result.append((train, test))
    return result


def repeated_participant_purged_folds(
    n_participants: int,
    n_time: int,
    *,
    n_folds: int = 5,
    purge_adjacent: int = 4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Make contiguous temporal folds for a participant-by-time design.

    Rows are ordered as ``participant_0``'s complete time series followed by
    ``participant_1``'s complete time series, and so on.  The same temporal
    test block is evaluated for every participant; only rows within each
    participant's local time series are purged.  This is the fold geometry
    used by the paired BOLD design materializer and avoids accidentally
    purging rows of unrelated participants merely because their flattened row
    indices are adjacent.
    """
    if n_participants < 1 or n_time < n_folds or n_folds < 2:
        raise ValueError("need at least one participant, n_time >= n_folds, and n_folds >= 2")
    blocks = [np.asarray(block, dtype=np.int64) for block in np.array_split(np.arange(int(n_time)), int(n_folds))]
    all_rows = np.arange(int(n_participants) * int(n_time), dtype=np.int64)
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for block in blocks:
        test_parts: list[np.ndarray] = []
        train_mask = np.ones(len(all_rows), dtype=bool)
        for participant in range(int(n_participants)):
            offset = participant * int(n_time)
            test_rows = offset + block
            test_parts.append(test_rows)
            train_mask[test_rows] = False
            if purge_adjacent:
                for row in block.tolist():
                    lo = max(0, int(row) - int(purge_adjacent))
                    hi = min(int(n_time), int(row) + int(purge_adjacent) + 1)
                    train_mask[offset + np.arange(lo, hi, dtype=np.int64)] = False
        test = np.concatenate(test_parts).astype(np.int64, copy=False)
        output.append((all_rows[train_mask], test))
    return output


def _validate_folds(folds: Sequence[tuple[np.ndarray, np.ndarray]], n_rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for index, (train_raw, test_raw) in enumerate(folds):
        train = np.asarray(train_raw, dtype=np.int64).reshape(-1)
        test = np.asarray(test_raw, dtype=np.int64).reshape(-1)
        if len(np.intersect1d(train, test)):
            raise ValueError(f"fold {index}: train and test overlap")
        if np.any(train < 0) or np.any(train >= n_rows) or np.any(test < 0) or np.any(test >= n_rows):
            raise ValueError(f"fold {index}: row outside design")
        output.append((train, test))
    if not output:
        raise ValueError("at least one fold is required")
    return output


def _standardize_train_test(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(x_train, axis=0)
    scale = np.std(x_train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    return (x_train - mean) / scale, (x_test - mean) / scale, mean, scale


def _standardize_target_train_test(y_train: np.ndarray, y_test: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(y_train, axis=0)
    scale = np.std(y_train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 0), scale, 1.0)
    return (y_train - mean) / scale, (y_test - mean) / scale, mean, scale


def ridge_fit(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    *,
    gram: np.ndarray | None = None,
) -> np.ndarray:
    """Fit multi-output ridge on already standardized training data."""
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if x_train.ndim != 2 or y_train.ndim != 2 or len(x_train) != len(y_train):
        raise ValueError("ridge inputs must be [rows, predictors] and [rows, targets]")
    # ``gram`` is an optional X-only cache.  It is copied before adding the
    # penalty so callers can safely reuse the exact unregularized Gram matrix
    # for every target/alpha fit without changing the solve path.
    gram = np.array(x_train.T @ x_train if gram is None else gram, copy=True)
    gram.flat[:: gram.shape[0] + 1] += float(alpha)
    try:
        return np.linalg.solve(gram, x_train.T @ y_train)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(gram) @ (x_train.T @ y_train)


def ridge_fit_cholesky(
    x_train: np.ndarray,
    y_train: np.ndarray,
    alpha: float,
    *,
    factor: tuple[np.ndarray, bool] | None = None,
) -> np.ndarray:
    """Fit ridge using a cached Cholesky factor of ``X.T @ X + alpha I``.

    The objective and RHS are identical to :func:`ridge_fit`; only the
    factorization is moved out of the replicate loop.  ``cho_solve`` is used
    instead of an inverse, avoiding a less stable and less equivalent solver.
    """
    if cho_factor is None or cho_solve is None:
        raise RuntimeError("scipy is required for the cached Cholesky solver")
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train, dtype=float)
    if x_train.ndim != 2 or y_train.ndim != 2 or len(x_train) != len(y_train):
        raise ValueError("ridge inputs must be [rows, predictors] and [rows, targets]")
    if factor is None:
        gram = x_train.T @ x_train
        gram.flat[:: gram.shape[0] + 1] += float(alpha)
        factor = cho_factor(gram, check_finite=False)
    return cho_solve(factor, x_train.T @ y_train, check_finite=False)


def _r2_oos(y_true: np.ndarray, y_pred: np.ndarray, train_mean: np.ndarray) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    baseline = np.asarray(train_mean, dtype=float).reshape(1, -1)
    denominator = np.sum(np.square(y_true - baseline), axis=0)
    numerator = np.sum(np.square(y_true - y_pred), axis=0)
    return np.divide(denominator - numerator, denominator, out=np.full_like(denominator, np.nan), where=denominator > 0)


def _groups_from_input(groups: Sequence[Sequence[int]] | None, n_features: int) -> list[np.ndarray]:
    if groups is None:
        return [np.asarray([index], dtype=np.int64) for index in range(n_features)]
    result: list[np.ndarray] = []
    seen: set[int] = set()
    for index, group in enumerate(groups):
        array = np.asarray(group, dtype=np.int64).reshape(-1)
        if len(array) == 0 or np.any(array < 0) or np.any(array >= n_features):
            raise ValueError(f"feature group {index} is empty or out of range")
        if len(np.unique(array)) != len(array):
            raise ValueError(f"feature group {index} contains duplicate columns")
        overlap = seen.intersection(int(item) for item in array.tolist())
        if overlap:
            raise ValueError(f"feature groups overlap columns {sorted(overlap)}")
        seen.update(int(item) for item in array.tolist())
        result.append(array)
    if seen != set(range(n_features)):
        raise ValueError("feature groups must cover every predictor column")
    return result


def heldout_grouped_reliance(
    x: np.ndarray,
    y: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    alpha: float | Sequence[float],
    feature_groups: Sequence[Sequence[int]] | None = None,
    finite_mask: np.ndarray | None = None,
    x_cache: Sequence[Mapping[str, np.ndarray]] | None = None,
    solver: str = "solve",
) -> dict[str, Any]:
    """Estimate held-out conditional grouped reliance under train-only scaling.

    The returned profile is the held-out decrease in R², averaged over target
    dimensions and folds.  Ablation replaces standardized test columns with
    zero, which is the training mean in standardized coordinates.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 2 or y.ndim != 2 or len(x) != len(y):
        raise ValueError("x and y must have matching row counts")
    groups = _groups_from_input(feature_groups, x.shape[1])
    if finite_mask is not None:
        finite_mask = np.asarray(finite_mask, dtype=bool).reshape(-1)
        if len(finite_mask) != len(x):
            raise ValueError("finite_mask must match x rows")
    if x_cache is not None and len(x_cache) != len(folds):
        raise ValueError("x_cache must match the number of folds")
    if solver not in {"solve", "cholesky"}:
        raise ValueError(f"unknown ridge solver: {solver}")
    fold_profiles: list[np.ndarray] = []
    fold_scores: list[dict[str, Any]] = []
    for fold_index, (train, test) in enumerate(_validate_folds(folds, len(x))):
        if finite_mask is None:
            train_mask = np.all(np.isfinite(y[train]), axis=1)
            test_mask = np.all(np.isfinite(y[test]), axis=1)
            train = train[train_mask]
            test = test[test_mask]
        else:
            train = train[finite_mask[train]]
            test = test[finite_mask[test]]
        if len(train) < 2 or len(test) < 2:
            raise ValueError(f"fold {fold_index}: insufficient finite rows")
        if x_cache is None:
            x_train, x_test, x_mean, x_scale = _standardize_train_test(x[train], x[test])
            cached_gram = None
            cached_factor = None
        else:
            cached = x_cache[fold_index]
            x_train = cached["x_train"]
            x_test = cached["x_test"]
            x_mean = cached["x_mean"]
            x_scale = cached["x_scale"]
            cached_gram = cached["gram"]
            cached_factor = None
        y_train, y_test, y_mean, y_scale = _standardize_target_train_test(y[train], y[test])
        if np.isscalar(alpha):
            fold_alpha = float(alpha)
        else:
            alpha_values = list(alpha)
            if len(alpha_values) != len(folds):
                raise ValueError("per-fold alpha sequence must match the number of folds")
            fold_alpha = float(alpha_values[fold_index])
        if x_cache is not None and solver == "cholesky":
            cached_factor = x_cache[fold_index].get("cholesky", {}).get(float(fold_alpha))
        if solver == "cholesky":
            coefficients = ridge_fit_cholesky(x_train, y_train, fold_alpha, factor=cached_factor)
        else:
            coefficients = ridge_fit(x_train, y_train, fold_alpha, gram=cached_gram)
        prediction = x_test @ coefficients
        full_r2 = _r2_oos(y_test, prediction, np.zeros(y_test.shape[1]))
        profile = np.empty(len(groups), dtype=float)
        for group_index, group in enumerate(groups):
            ablated = x_test.copy()
            ablated[:, group] = 0.0
            ablated_prediction = ablated @ coefficients
            ablated_r2 = _r2_oos(y_test, ablated_prediction, np.zeros(y_test.shape[1]))
            profile[group_index] = float(np.nanmean(full_r2 - ablated_r2))
        fold_profiles.append(profile)
        fold_scores.append(
            {
                "fold": int(fold_index),
                "alpha": fold_alpha,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "mean_r2_oos": float(np.nanmean(full_r2)),
                "full_r2_oos_by_target": _jsonable(full_r2),
                "x_mean": _jsonable(x_mean),
                "x_scale": _jsonable(x_scale),
                "y_mean": _jsonable(y_mean),
                "y_scale": _jsonable(y_scale),
            }
        )
    profile_matrix = np.asarray(fold_profiles, dtype=float)
    return {
        "alpha": float(alpha) if np.isscalar(alpha) else [float(value) for value in alpha],
        "reliance_profile": np.nanmean(profile_matrix, axis=0),
        "fold_reliance_profiles": profile_matrix,
        "fold_scores": fold_scores,
        "n_groups": int(len(groups)),
        "group_sizes": [int(len(group)) for group in groups],
    }


def select_alpha_by_fold(
    x: np.ndarray,
    y: np.ndarray,
    outer_folds: Sequence[tuple[np.ndarray, np.ndarray]],
    alpha_grid: Sequence[float],
    *,
    purge_adjacent: int = 4,
    finite_mask: np.ndarray | None = None,
    x_cache: Sequence[Sequence[Mapping[str, np.ndarray]]] | None = None,
    solver: str = "solve",
) -> list[float]:
    """Select R²-optimizing ridge penalties inside each outer training fold.

    This is the same nested structure used by the empirical analysis: inner
    contiguous folds are made from the outer-training rows, and no validation
    row contributes to the fitted scaling or coefficient.  It is deliberately
    separate from :func:`heldout_grouped_reliance` so a power replicate can
    retune penalties after its target has been generated.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    grid = sorted({float(value) for value in alpha_grid})
    if not grid or any(value <= 0 for value in grid):
        raise ValueError("alpha_grid must contain positive values")
    if finite_mask is not None:
        finite_mask = np.asarray(finite_mask, dtype=bool).reshape(-1)
        if len(finite_mask) != len(x):
            raise ValueError("finite_mask must match x rows")
    if x_cache is not None and len(x_cache) != len(outer_folds):
        raise ValueError("x_cache must match the number of outer folds")
    if solver not in {"solve", "cholesky"}:
        raise ValueError(f"unknown ridge solver: {solver}")
    selected: list[float] = []
    for fold_index, (outer_train, _outer_test) in enumerate(_validate_folds(outer_folds, len(x))):
        outer_train = np.asarray(outer_train, dtype=np.int64)
        inner_local = contiguous_purged_folds(len(outer_train), n_folds=min(5, len(outer_train)), purge_adjacent=purge_adjacent)
        inner_folds = [(outer_train[train], outer_train[test]) for train, test in inner_local]
        scores: list[tuple[float, float]] = []
        for alpha in grid:
            fold_scores: list[float] = []
            for inner_index, (train, test) in enumerate(inner_folds):
                if finite_mask is None:
                    finite_train = np.all(np.isfinite(y[train]), axis=1)
                    finite_test = np.all(np.isfinite(y[test]), axis=1)
                    train = train[finite_train]
                    test = test[finite_test]
                else:
                    train = train[finite_mask[train]]
                    test = test[finite_mask[test]]
                if len(train) < 2 or len(test) < 2:
                    continue
                if x_cache is None:
                    x_train, x_test, _, _ = _standardize_train_test(x[train], x[test])
                    cached_gram = None
                    cached_factor = None
                else:
                    cached = x_cache[fold_index][inner_index]
                    x_train = cached["x_train"]
                    x_test = cached["x_test"]
                    cached_gram = cached["gram"]
                    cached_factor = cached.get("cholesky", {}).get(float(alpha)) if solver == "cholesky" else None
                y_train, y_test, y_mean, _ = _standardize_target_train_test(y[train], y[test])
                if solver == "cholesky":
                    beta = ridge_fit_cholesky(x_train, y_train, alpha, factor=cached_factor)
                else:
                    beta = ridge_fit(x_train, y_train, alpha, gram=cached_gram)
                fold_scores.append(float(np.nanmean(_r2_oos(y_test, x_test @ beta, np.zeros(y_test.shape[1])))))
            scores.append((float(np.nanmean(fold_scores)) if fold_scores else -np.inf, alpha))
        # Python's stable max plus the secondary alpha key makes ties choose
        # the smaller penalty, as in the configured empirical selection contract.
        best_score = max(score for score, _ in scores)
        selected.append(float(min(alpha for score, alpha in scores if np.isclose(score, best_score, equal_nan=False))))
    return selected


def _noise_from_residual(residual: np.ndarray, rng: np.random.Generator, n_rows: int, n_targets: int) -> np.ndarray:
    """Resample residual rows, preserving target covariance without fake iid noise."""
    residual = np.asarray(residual, dtype=float)
    if residual.ndim != 2 or residual.shape[1] != n_targets or len(residual) < 2:
        raise ValueError("residual must be [rows, requested target dimensions] with >=2 rows")
    indices = rng.integers(0, len(residual), size=int(n_rows))
    sampled = residual[indices].copy()
    sampled -= np.mean(residual, axis=0, keepdims=True)
    return sampled


def _mean_empirical_residual_sd(residual: np.ndarray) -> float:
    """Return the mean per-target SD used to scale one synthetic target system."""
    residual = np.asarray(residual, dtype=float)
    if residual.ndim != 2 or residual.shape[1] < 1 or len(residual) < 2:
        raise ValueError("residual must be a two-dimensional array with at least two rows")
    scales = np.std(residual, axis=0, ddof=0)
    finite = scales[np.isfinite(scales)]
    if len(finite) == 0:
        raise ValueError("residual has no finite target standard deviations")
    return float(np.mean(finite))


def _make_spearman_profiles(
    n_features: int,
    oracle_correlation: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate continuous profiles whose *rank* correlation targets ``rho``.

    Spearman correspondence is the primary feature-reliance estimand.  A
    Pearson-linear mixture does not preserve that estimand, especially for
    highly correlated or sparse designs.  We therefore search deterministic
    rank perturbations of one random profile and retain the candidate whose
    measured Spearman correlation is closest to the requested grid value.
    The search is deliberately internal to profile construction; it does not
    inspect predictors, targets, folds, or fitted results.
    """
    if rankdata is None:
        raise RuntimeError("scipy is required for Spearman power calculations")
    n_features = int(n_features)
    if n_features < 3:
        raise ValueError("at least three features are required for Spearman profiles")
    target = float(oracle_correlation)
    # The configured grid is non-negative, but accepting the full correlation
    # interval keeps this helper useful for calibration tests and future work.
    if not -1.0 <= target <= 1.0:
        raise ValueError("oracle_correlation must be in [-1,1]")
    base_raw = rng.normal(size=n_features)
    base = (base_raw - base_raw.mean()) / max(float(base_raw.std()), 1e-12)
    base_rank = rankdata(base, method="average")
    # For a permutation without ties, Spearman rho is
    # ``1 - 6*sum(d_i**2)/(n*(n**2-1))``.  Reversing a contiguous rank block
    # of length ``m`` gives ``sum(d_i**2) = (m**3-m)/3``.  Selecting the
    # closest block length therefore calibrates the finite-rank grid directly,
    # without a costly Monte-Carlo search per power replicate.  A random block
    # location and random base ordering keep the profiles non-canonical while
    # preserving the known rank correlation.
    block_lengths = np.arange(n_features + 1, dtype=float)
    candidate_rhos = 1.0 - 2.0 * (block_lengths**3 - block_lengths) / (
        n_features * (n_features**2 - 1.0)
    )
    block_length = int(np.argmin(np.abs(candidate_rhos - target)))
    candidate_rank = np.arange(1, n_features + 1, dtype=float)
    if block_length > 1:
        start = int(rng.integers(0, n_features - block_length + 1))
        candidate_rank[start : start + block_length] = candidate_rank[
            start : start + block_length
        ][::-1]
    # Convert the rank order to feature-index scores.  Rank scores provide
    # stable, non-tied magnitudes and make Spearman the profile's defining
    # property rather than its accidental Pearson scale.
    best_profile = np.empty(n_features, dtype=float)
    best_profile[np.argsort(base_rank, kind="mergesort")] = candidate_rank
    best_profile = (best_profile - best_profile.mean()) / max(float(best_profile.std()), 1e-12)
    return base, best_profile


def make_reliance_profiles(
    n_features: int,
    oracle_correlation: float,
    rng: np.random.Generator,
    *,
    correlation_method: str = "pearson",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate two centered feature profiles with a prescribed correlation.

    ``pearson`` retains the legacy helper contract.  The paired-target power
    analysis passes ``spearman`` because its primary correspondence metric is
    rank correlation; the requested oracle grid is therefore interpreted in
    that same metric rather than as Pearson correlation.
    """
    if correlation_method == "spearman":
        return _make_spearman_profiles(n_features, oracle_correlation, rng)
    if correlation_method != "pearson":
        raise ValueError(f"unknown profile correlation method: {correlation_method}")
    if not 0.0 <= float(oracle_correlation) <= 1.0:
        raise ValueError("oracle_correlation must be in [0,1]")
    base = rng.normal(size=int(n_features))
    orthogonal = rng.normal(size=int(n_features))
    base = (base - base.mean()) / max(float(base.std()), 1e-12)
    orthogonal = (orthogonal - orthogonal.mean()) / max(float(orthogonal.std()), 1e-12)
    # Remove the component parallel to the first profile so the requested
    # correlation is a property of the generated profiles, not an
    # approximation that varies with the random draw.
    orthogonal = orthogonal - (float(np.dot(orthogonal, base)) / max(float(np.dot(base, base)), 1e-12)) * base
    orthogonal = (orthogonal - orthogonal.mean()) / max(float(orthogonal.std()), 1e-12)
    second = float(oracle_correlation) * base + np.sqrt(max(0.0, 1.0 - float(oracle_correlation) ** 2)) * orthogonal
    second = (second - second.mean()) / max(float(second.std()), 1e-12)
    return base, second


def inject_targets(
    x: np.ndarray,
    residual: np.ndarray,
    profile: np.ndarray,
    *,
    signal_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Inject a profile-coded signal into empirical residual draws.

    A random target loading vector gives each target dimension a different
    mixture of the same feature-derived latent.  Thus target dimensionality
    is retained while the feature reliance profile is shared.
    """
    x = np.asarray(x, dtype=float)
    profile = np.asarray(profile, dtype=float).reshape(-1)
    if x.ndim != 2 or x.shape[1] != len(profile):
        raise ValueError("profile length must match x columns")
    residual = np.asarray(residual, dtype=float)
    if residual.ndim != 2 or len(residual) != len(x):
        raise ValueError("residual rows must match x")
    centered = x - np.mean(x, axis=0, keepdims=True)
    scale = np.std(centered, axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    z = (centered / scale) @ profile
    z /= max(float(np.std(z)), 1e-12)
    loading = rng.normal(size=residual.shape[1])
    # Normalize by mean absolute loading rather than L2 norm so
    # ``signal_scale`` is in the requested mean per-target SD units: across
    # target dimensions, the injected signal has mean SD approximately equal
    # to ``signal_scale`` regardless of whether the system has three BOLD
    # parcels or thousands of LM dimensions.
    loading /= max(float(np.mean(np.abs(loading))), 1e-12)
    signal = float(signal_scale) * z[:, None] * loading[None, :]
    return residual + signal


def _prepare_x_cache(
    x: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    finite_mask: np.ndarray,
    *,
    purge_adjacent: int,
    alpha_grid: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Prepare invariant standardized-X data for one target system.

    The cache contains no target-derived values.  It is therefore safe to
    reuse across power replicates while retaining the exact standardization,
    Gram construction, and ``np.linalg.solve`` operations of the uncached
    path.  A copy of each unregularized Gram matrix is supplied to
    :func:`ridge_fit`, which adds the replicate-specific alpha to its copy.
    """
    x = np.asarray(x, dtype=float)
    finite_mask = np.asarray(finite_mask, dtype=bool).reshape(-1)
    if len(finite_mask) != len(x):
        raise ValueError("finite_mask must match x rows")
    validated_folds = _validate_folds(folds, len(x))
    grid = sorted({float(value) for value in (alpha_grid or ())})
    if any(value <= 0 for value in grid):
        raise ValueError("alpha_grid must contain positive values")

    def factors(gram: np.ndarray) -> dict[float, tuple[np.ndarray, bool]]:
        if not grid:
            return {}
        if cho_factor is None:
            raise RuntimeError("scipy is required for the cached Cholesky solver")
        output: dict[float, tuple[np.ndarray, bool]] = {}
        for alpha in grid:
            regularized = np.array(gram, copy=True)
            regularized.flat[:: regularized.shape[0] + 1] += alpha
            try:
                output[alpha] = cho_factor(regularized, check_finite=False)
            except Exception as exc:  # noqa: BLE001 - fail closed on cache construction
                raise RuntimeError(f"Cholesky factorization failed for alpha={alpha}") from exc
        return output

    heldout: list[dict[str, np.ndarray]] = []
    selection: list[list[dict[str, np.ndarray]]] = []
    for outer_train_raw, outer_test_raw in validated_folds:
        # Inner fold boundaries are defined on the original outer-training
        # rows, before target-finite filtering, exactly as in
        # ``select_alpha_by_fold``.  Censored rows are removed only after the
        # local indices have been mapped back to global rows.
        outer_train_full = np.asarray(outer_train_raw, dtype=np.int64)
        outer_train = outer_train_raw[finite_mask[outer_train_raw]]
        outer_test = outer_test_raw[finite_mask[outer_test_raw]]
        if len(outer_train) < 2 or len(outer_test) < 2:
            raise ValueError("fold has insufficient finite rows")
        x_train, x_test, x_mean, x_scale = _standardize_train_test(x[outer_train], x[outer_test])
        gram = x_train.T @ x_train
        heldout.append(
            {
                "x_train": x_train,
                "x_test": x_test,
                "x_mean": x_mean,
                "x_scale": x_scale,
                "gram": gram,
                "cholesky": factors(gram),
            }
        )
        inner_local = contiguous_purged_folds(
            len(outer_train_full),
            n_folds=min(5, len(outer_train_full)),
            purge_adjacent=purge_adjacent,
        )
        inner_cache: list[dict[str, np.ndarray]] = []
        for inner_train_local, inner_test_local in inner_local:
            train = outer_train_full[inner_train_local]
            test = outer_train_full[inner_test_local]
            train = train[finite_mask[train]]
            test = test[finite_mask[test]]
            if len(train) < 2 or len(test) < 2:
                # Keep a marker with empty arrays so the selection loop has
                # exactly the same skip behavior as the uncached path.
                inner_cache.append(
                    {
                        "x_train": np.empty((0, x.shape[1]), dtype=float),
                        "x_test": np.empty((0, x.shape[1]), dtype=float),
                        "gram": np.empty((x.shape[1], x.shape[1]), dtype=float),
                        "cholesky": {},
                    }
                )
                continue
            x_train, x_test, _x_mean, _x_scale = _standardize_train_test(x[train], x[test])
            gram = x_train.T @ x_train
            inner_cache.append(
                {
                    "x_train": x_train,
                    "x_test": x_test,
                    "gram": gram,
                    "cholesky": factors(gram),
                }
            )
        selection.append(inner_cache)
    return {"heldout": heldout, "selection": selection}


def _simulate_paired_power_case_validated(
    design: Mapping[str, Any],
    *,
    oracle_correlation: float,
    residual_scale_multiplier: float,
    alpha_grid: Sequence[float],
    seed: int,
    x_caches: Mapping[str, Mapping[str, Any]] | None = None,
    solver: str = "solve",
) -> dict[str, Any]:
    """Internal paired simulation assuming design validation is complete."""
    alpha_values = [float(value) for value in alpha_grid]
    if not alpha_values or any(value <= 0 for value in alpha_values):
        raise ValueError("paired power simulation requires a positive alpha grid")
    rng = np.random.default_rng(int(seed))
    n_features = len(design["brain_feature_groups"])
    profile_brain, profile_lm = make_reliance_profiles(
        n_features,
        float(oracle_correlation),
        rng,
        correlation_method="spearman",
    )
    brain_columns = np.zeros(design["brain_x"].shape[1], dtype=float)
    lm_columns = np.zeros(design["lm_x"].shape[1], dtype=float)
    for score_brain, score_lm, brain_group, lm_group in zip(
        profile_brain,
        profile_lm,
        design["brain_feature_groups"],
        design["lm_feature_groups"], strict=False,
    ):
        brain_columns[brain_group] = float(score_brain)
        lm_columns[lm_group] = float(score_lm)
    residual_brain = np.asarray(design["brain_residual"], dtype=float)
    residual_lm = np.asarray(design["lm_residual"], dtype=float)
    brain_residual_sd = _mean_empirical_residual_sd(residual_brain)
    lm_residual_sd = _mean_empirical_residual_sd(residual_lm)
    brain_signal_scale = float(residual_scale_multiplier) * max(brain_residual_sd, 1e-12)
    lm_signal_scale = float(residual_scale_multiplier) * max(lm_residual_sd, 1e-12)
    brain_target = inject_targets(
        design["brain_x"],
        _noise_from_residual(residual_brain, rng, len(design["brain_x"]), residual_brain.shape[1]),
        brain_columns,
        signal_scale=brain_signal_scale,
        rng=rng,
    )
    lm_target = inject_targets(
        design["lm_x"],
        _noise_from_residual(residual_lm, rng, len(design["lm_x"]), residual_lm.shape[1]),
        lm_columns,
        signal_scale=lm_signal_scale,
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
    expected_brain_finite = ~brain_censor
    expected_lm_finite = ~lm_censor
    if not np.array_equal(brain_finite, expected_brain_finite) or not np.array_equal(lm_finite, expected_lm_finite):
        raise ValueError("synthetic target finiteness changed; refusing cached fold masks")
    brain_cache = x_caches.get("brain") if x_caches else None
    lm_cache = x_caches.get("lm") if x_caches else None
    brain_alphas = select_alpha_by_fold(
        design["brain_x"],
        brain_target,
        design["brain_folds"],
        alpha_values,
        purge_adjacent=int(design.get("purge_adjacent_trs", 4)),
        finite_mask=brain_finite if brain_cache is not None else None,
        x_cache=brain_cache["selection"] if brain_cache is not None else None,
        solver=solver,
    )
    lm_alphas = select_alpha_by_fold(
        design["lm_x"],
        lm_target,
        design["lm_folds"],
        alpha_values,
        purge_adjacent=int(design.get("purge_adjacent_trs", 4)),
        finite_mask=lm_finite if lm_cache is not None else None,
        x_cache=lm_cache["selection"] if lm_cache is not None else None,
        solver=solver,
    )
    brain_fit = heldout_grouped_reliance(
        design["brain_x"],
        brain_target,
        design["brain_folds"],
        alpha=brain_alphas,
        feature_groups=design["brain_feature_groups"],
        finite_mask=brain_finite if brain_cache is not None else None,
        x_cache=brain_cache["heldout"] if brain_cache is not None else None,
        solver=solver,
    )
    lm_fit = heldout_grouped_reliance(
        design["lm_x"],
        lm_target,
        design["lm_folds"],
        alpha=lm_alphas,
        feature_groups=design["lm_feature_groups"],
        finite_mask=lm_finite if lm_cache is not None else None,
        x_cache=lm_cache["heldout"] if lm_cache is not None else None,
        solver=solver,
    )
    observed = profile_correlation(
        brain_fit["reliance_profile"], lm_fit["reliance_profile"], method="spearman"
    )
    achieved_pearson = profile_correlation(profile_brain, profile_lm, method="pearson")
    achieved_spearman = profile_correlation(profile_brain, profile_lm, method="spearman")
    return {
        "oracle_correlation": float(oracle_correlation),
        "oracle_correlation_method": "spearman",
        "profile_achieved_spearman": float(achieved_spearman),
        "profile_achieved_pearson": float(achieved_pearson),
        "residual_scale_multiplier": float(residual_scale_multiplier),
        "residual_scale_definition": "mean per-target population SD of the corresponding empirical residual pool",
        "brain_empirical_residual_sd": float(brain_residual_sd),
        "lm_empirical_residual_sd": float(lm_residual_sd),
        "brain_signal_scale_absolute": float(brain_signal_scale),
        "lm_signal_scale_absolute": float(lm_signal_scale),
        "observed_reliance_spearman": float(observed),
        "brain_reliance_profile": np.asarray(brain_fit["reliance_profile"], dtype=np.float32),
        "lm_reliance_profile": np.asarray(lm_fit["reliance_profile"], dtype=np.float32),
        "brain_mean_r2_oos": float(np.nanmean([row["mean_r2_oos"] for row in brain_fit["fold_scores"]])),
        "lm_mean_r2_oos": float(np.nanmean([row["mean_r2_oos"] for row in lm_fit["fold_scores"]])),
        "brain_alphas": [float(value) for value in brain_alphas],
        "lm_alphas": [float(value) for value in lm_alphas],
    }


def simulate_power_case(
    x: np.ndarray,
    brain_residual: np.ndarray,
    lm_residual: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    *,
    oracle_correlation: float,
    residual_scale_multiplier: float,
    alpha: float | None,
    alpha_grid: Sequence[float] | None = None,
    feature_groups: Sequence[Sequence[int]] | None,
    seed: int,
    censor_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Run one complete semi-synthetic paired-target case."""
    rng = np.random.default_rng(int(seed))
    groups = _groups_from_input(feature_groups, np.asarray(x).shape[1])
    # Profiles are generated at the grouping level.  If a group contains lag
    # columns, repeat its score across those columns; the ablation is still one
    # feature unit.
    p_a, p_b = make_reliance_profiles(len(groups), oracle_correlation, rng)
    column_a = np.zeros(np.asarray(x).shape[1], dtype=float)
    column_b = np.zeros(np.asarray(x).shape[1], dtype=float)
    for value_a, value_b, group in zip(p_a, p_b, groups, strict=False):
        column_a[group] = value_a
        column_b[group] = value_b
    # The reference residual scale is the mean target residual SD.  It is used
    # only to define a reproducible signal-to-noise multiplier.
    residual_a = np.asarray(brain_residual, dtype=float)
    residual_b = np.asarray(lm_residual, dtype=float)
    reference = float(np.nanmean(np.concatenate([np.std(residual_a, axis=0), np.std(residual_b, axis=0)])))
    signal_scale = float(residual_scale_multiplier) * max(reference, 1e-12)
    target_a = inject_targets(x, _noise_from_residual(residual_a, rng, len(x), residual_a.shape[1]), column_a, signal_scale=signal_scale, rng=rng)
    target_b = inject_targets(x, _noise_from_residual(residual_b, rng, len(x), residual_b.shape[1]), column_b, signal_scale=signal_scale, rng=rng)
    if censor_mask is not None:
        censor_mask = np.asarray(censor_mask, dtype=bool).reshape(-1)
        if len(censor_mask) != len(x):
            raise ValueError("censor_mask must match design rows")
        # Censoring is a BOLD target decision.  The language-model target is
        # still available at every TR, so only the brain target is masked.
        target_a[censor_mask] = np.nan
    if alpha_grid is not None:
        alpha_a = select_alpha_by_fold(x, target_a, folds, alpha_grid)
        alpha_b = select_alpha_by_fold(x, target_b, folds, alpha_grid)
    elif alpha is not None:
        alpha_a = alpha_b = alpha
    else:
        raise ValueError("provide alpha or alpha_grid")
    fit_a = heldout_grouped_reliance(x, target_a, folds, alpha=alpha_a, feature_groups=groups)
    fit_b = heldout_grouped_reliance(x, target_b, folds, alpha=alpha_b, feature_groups=groups)
    observed = profile_correlation(fit_a["reliance_profile"], fit_b["reliance_profile"], method="spearman")
    return {
        "oracle_correlation": float(oracle_correlation),
        "residual_scale_multiplier": float(residual_scale_multiplier),
        "signal_scale_absolute": signal_scale,
        "observed_reliance_spearman": float(observed),
        "brain_reliance_profile": fit_a["reliance_profile"],
        "lm_reliance_profile": fit_b["reliance_profile"],
        "brain_mean_r2_oos": float(np.nanmean([row["mean_r2_oos"] for row in fit_a["fold_scores"]])),
        "lm_mean_r2_oos": float(np.nanmean([row["mean_r2_oos"] for row in fit_b["fold_scores"]])),
        "brain_alphas": alpha_a,
        "lm_alphas": alpha_b,
        "fold_brain_reliance": fit_a["fold_reliance_profiles"],
        "fold_lm_reliance": fit_b["fold_reliance_profiles"],
    }


def _load_design(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as z:
        design_schema = str(np.asarray(z["design_schema"]).reshape(-1)[0]) if "design_schema" in z else ""
        if design_schema == "paired_target_design_v1":
            required = {
                "brain_x",
                "lm_x",
                "brain_residual",
                "lm_residual",
                "brain_participant_ids",
                "lm_participant_ids",
                "brain_censor_mask",
                "lm_censor_mask",
                "brain_folds",
                "lm_folds",
                "brain_feature_groups",
                "lm_feature_groups",
            }
            missing = sorted(required.difference(z.files))
            if missing:
                raise ValueError(f"{path}: missing paired design fields {missing}")
            loaded: dict[str, Any] = {
                "design_schema": design_schema,
                "brain_x": np.asarray(z["brain_x"], dtype=float),
                "lm_x": np.asarray(z["lm_x"], dtype=float),
                "brain_residual": np.asarray(z["brain_residual"], dtype=float),
                "lm_residual": np.asarray(z["lm_residual"], dtype=float),
                "brain_participant_ids": np.asarray(z["brain_participant_ids"], dtype=str).reshape(-1),
                "lm_participant_ids": np.asarray(z["lm_participant_ids"], dtype=str).reshape(-1),
                "brain_censor_mask": np.asarray(z["brain_censor_mask"], dtype=bool).reshape(-1),
                "lm_censor_mask": np.asarray(z["lm_censor_mask"], dtype=bool).reshape(-1),
                "brain_folds": [
                    (np.asarray(item[0], dtype=np.int64), np.asarray(item[1], dtype=np.int64))
                    for item in z["brain_folds"].tolist()
                ],
                "lm_folds": [
                    (np.asarray(item[0], dtype=np.int64), np.asarray(item[1], dtype=np.int64))
                    for item in z["lm_folds"].tolist()
                ],
                "brain_feature_groups": [
                    np.asarray(item, dtype=np.int64) for item in z["brain_feature_groups"].tolist()
                ],
                "lm_feature_groups": [
                    np.asarray(item, dtype=np.int64) for item in z["lm_feature_groups"].tolist()
                ],
                "path": path,
                "sha256": sha256_file(path),
            }
            for key in (
                "design_label",
                "model_key",
                "bank_type",
                "bank_variant",
                "direction",
                "source_story",
                "target_story",
                "participant_structure",
                "censoring_manifest",
                "purge_adjacent_trs",
                "evaluation_regime",
                "row_order",
                "feature_identity_source",
                "analysis_fingerprint",
            ):
                if key in z:
                    value = np.asarray(z[key]).reshape(-1)
                    loaded[key] = str(value[0]) if len(value) else ""
            for key in ("provenance_source_paths", "provenance_source_hashes", "provenance_code_hashes"):
                stored = f"provenance_{key}"
                if stored in z:
                    value = np.asarray(z[stored]).reshape(-1)
                    loaded[key] = str(value[0]) if len(value) else ""
            loaded = _validate_paired_design(loaded)
            return loaded
        required = {"x", "brain_residual", "lm_residual", "participant_ids", "censor_mask"}
        missing = sorted(required.difference(z.files))
        if missing:
            raise ValueError(f"{path}: missing design fields {missing}")
        x = np.asarray(z["x"], dtype=float)
        brain = np.asarray(z["brain_residual"], dtype=float)
        lm = np.asarray(z["lm_residual"], dtype=float)
        participant_ids = np.asarray(z["participant_ids"], dtype=str).reshape(-1)
        censor_mask = np.asarray(z["censor_mask"], dtype=bool).reshape(-1)
        if "folds" in z:
            folds = [(np.asarray(item[0], dtype=np.int64), np.asarray(item[1], dtype=np.int64)) for item in z["folds"].tolist()]
        else:
            folds = contiguous_purged_folds(len(x), n_folds=5, purge_adjacent=4)
        groups = None
        if "feature_groups" in z:
            groups = [np.asarray(item, dtype=np.int64) for item in z["feature_groups"].tolist()]
    if x.ndim != 2 or brain.ndim != 2 or lm.ndim != 2 or len(x) != len(brain) or len(x) != len(lm):
        raise ValueError(f"{path}: x and residual rows must match and be two-dimensional")
    if len(participant_ids) != len(x) or len(censor_mask) != len(x):
        raise ValueError(f"{path}: participant_ids and censor_mask must match design rows")
    metadata: dict[str, Any] = {}
    with np.load(path, allow_pickle=True) as z:
        for key in ("design_label", "model_key", "bank_type", "direction", "participant_structure", "source_story", "target_story", "censoring_manifest"):
            if key in z:
                value = np.asarray(z[key]).reshape(-1)
                metadata[key] = str(value[0]) if len(value) else ""
    if not metadata.get("design_label") or not metadata.get("model_key") or not metadata.get("bank_type") or not metadata.get("direction"):
        raise ValueError(f"{path}: design metadata must identify model_key, bank_type, direction, and design_label")
    if metadata.get("participant_structure") != "participant_by_time_rows":
        raise ValueError(f"{path}: participant_structure must be participant_by_time_rows")
    if not metadata.get("censoring_manifest"):
        raise ValueError(f"{path}: censoring_manifest is required for a power design")
    return {
        "x": x,
        "brain_residual": brain,
        "lm_residual": lm,
        "folds": _validate_folds(folds, len(x)),
        "feature_groups": groups,
        "participant_ids": participant_ids,
        "censor_mask": censor_mask,
        **metadata,
        "path": path,
        "sha256": sha256_file(path),
    }


def write_design_artifact(
    path: Path,
    *,
    x: np.ndarray,
    brain_residual: np.ndarray,
    lm_residual: np.ndarray,
    participant_ids: Sequence[str],
    censor_mask: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    feature_groups: Sequence[Sequence[int]] | None,
    metadata: Mapping[str, str],
) -> dict[str, Any]:
    """Write a provenance-complete power design from corrected fit residuals.

    The caller must provide residuals produced by the corrected observed-fit
    pipeline.  This helper does not derive residuals from raw targets and does
    not silently substitute old artifacts.
    """
    path = Path(path)
    x = np.asarray(x, dtype=np.float64)
    brain_residual = np.asarray(brain_residual, dtype=np.float64)
    lm_residual = np.asarray(lm_residual, dtype=np.float64)
    participant_ids = np.asarray(participant_ids, dtype=str).reshape(-1)
    censor_mask = np.asarray(censor_mask, dtype=bool).reshape(-1)
    if x.ndim != 2 or brain_residual.ndim != 2 or lm_residual.ndim != 2:
        raise ValueError("design arrays must be two-dimensional")
    if len(x) != len(brain_residual) or len(x) != len(lm_residual) or len(x) != len(participant_ids) or len(x) != len(censor_mask):
        raise ValueError("design arrays and participant/censor metadata must share rows")
    required_metadata = ("design_label", "model_key", "bank_type", "direction", "participant_structure", "censoring_manifest")
    missing = [key for key in required_metadata if not str(metadata.get(key, ""))]
    if missing:
        raise ValueError(f"design metadata missing {missing}")
    groups = _groups_from_input(feature_groups, x.shape[1])
    folds = _validate_folds(folds, len(x))
    path.parent.mkdir(parents=True, exist_ok=True)
    fold_object = np.asarray([(train, test) for train, test in folds], dtype=object)
    group_object = np.asarray(groups, dtype=object)
    np.savez_compressed(
        path,
        x=x,
        brain_residual=brain_residual,
        lm_residual=lm_residual,
        participant_ids=participant_ids,
        censor_mask=censor_mask,
        folds=fold_object,
        feature_groups=group_object,
        **{key: np.asarray([str(value)]) for key, value in metadata.items()},
    )
    return {
        "schema_version": 1,
        "path": str(path),
        "sha256": sha256_file(path),
        "rows": int(len(x)),
        "features": int(x.shape[1]),
        "brain_target_dimensions": int(brain_residual.shape[1]),
        "lm_target_dimensions": int(lm_residual.shape[1]),
        "participant_count": int(len(np.unique(participant_ids))),
        "censored_rows": int(np.sum(censor_mask)),
        "fold_count": int(len(folds)),
        **{key: str(value) for key, value in metadata.items()},
    }


def write_paired_design_artifact(
    path: Path,
    *,
    brain_x: np.ndarray,
    lm_x: np.ndarray,
    brain_residual: np.ndarray,
    lm_residual: np.ndarray,
    brain_participant_ids: Sequence[str],
    lm_participant_ids: Sequence[str],
    brain_censor_mask: np.ndarray,
    lm_censor_mask: np.ndarray,
    brain_folds: Sequence[tuple[np.ndarray, np.ndarray]],
    lm_folds: Sequence[tuple[np.ndarray, np.ndarray]],
    brain_feature_groups: Sequence[Sequence[int]],
    lm_feature_groups: Sequence[Sequence[int]],
    metadata: Mapping[str, str],
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a paired BOLD/LM power design without collapsing their geometry.

    A BOLD design has participant-by-time rows and lag-expanded columns,
    whereas an LM design has one row per stimulus TR and unlagged columns.
    Treating these as one matrix would make the power calculation answer a
    different question from the observed analysis.  This schema therefore
    stores the two predictor systems, fold sets, feature groups, and censoring
    masks separately.  ``brain_residual`` and ``lm_residual`` are finite
    empirical residual pools; they need not have the same number of rows as
    the predictors because they are resampled to generate synthetic targets.
    """
    path = Path(path)
    brain_x = np.asarray(brain_x, dtype=np.float32)
    lm_x = np.asarray(lm_x, dtype=np.float32)
    brain_residual = np.asarray(brain_residual, dtype=np.float32)
    lm_residual = np.asarray(lm_residual, dtype=np.float32)
    brain_participant_ids = np.asarray(brain_participant_ids, dtype=str).reshape(-1)
    lm_participant_ids = np.asarray(lm_participant_ids, dtype=str).reshape(-1)
    brain_censor_mask = np.asarray(brain_censor_mask, dtype=bool).reshape(-1)
    lm_censor_mask = np.asarray(lm_censor_mask, dtype=bool).reshape(-1)
    for label, array in (
        ("brain_x", brain_x),
        ("lm_x", lm_x),
        ("brain_residual", brain_residual),
        ("lm_residual", lm_residual),
    ):
        if array.ndim != 2:
            raise ValueError(f"{label} must be two-dimensional")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{label} contains non-finite values")
    if len(brain_participant_ids) != len(brain_x) or len(brain_censor_mask) != len(brain_x):
        raise ValueError("brain participant IDs/censor mask must match brain_x rows")
    if len(lm_participant_ids) != len(lm_x) or len(lm_censor_mask) != len(lm_x):
        raise ValueError("LM participant IDs/censor mask must match lm_x rows")
    required_metadata = (
        "design_label",
        "model_key",
        "bank_type",
        "direction",
        "participant_structure",
        "censoring_manifest",
    )
    missing = [key for key in required_metadata if not str(metadata.get(key, ""))]
    if missing:
        raise ValueError(f"paired design metadata missing {missing}")
    if str(metadata.get("participant_structure")) != "paired_system_rows":
        raise ValueError("paired design participant_structure must be paired_system_rows")
    brain_groups = _groups_from_input(brain_feature_groups, brain_x.shape[1])
    lm_groups = _groups_from_input(lm_feature_groups, lm_x.shape[1])
    brain_folds = _validate_folds(brain_folds, len(brain_x))
    lm_folds = _validate_folds(lm_folds, len(lm_x))
    if len(brain_groups) != len(lm_groups):
        raise ValueError("brain and LM feature groups must represent the same number of SAE features")
    if np.any(brain_censor_mask & ~np.all(np.isfinite(brain_x), axis=1)):
        raise ValueError("brain censor mask cannot analysis non-finite predictors")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": np.asarray([2], dtype=np.int32),
        "design_schema": np.asarray(["paired_target_design_v1"]),
        "brain_x": brain_x,
        "lm_x": lm_x,
        "brain_residual": brain_residual,
        "lm_residual": lm_residual,
        "brain_participant_ids": brain_participant_ids,
        "lm_participant_ids": lm_participant_ids,
        "brain_censor_mask": brain_censor_mask,
        "lm_censor_mask": lm_censor_mask,
        "brain_folds": np.asarray([(train, test) for train, test in brain_folds], dtype=object),
        "lm_folds": np.asarray([(train, test) for train, test in lm_folds], dtype=object),
        "brain_feature_groups": np.asarray([np.asarray(group, dtype=np.int64) for group in brain_groups], dtype=object),
        "lm_feature_groups": np.asarray([np.asarray(group, dtype=np.int64) for group in lm_groups], dtype=object),
    }
    for key, value in metadata.items():
        payload[str(key)] = np.asarray([str(value)])
    for key, value in (provenance or {}).items():
        if isinstance(value, (dict, list, tuple)):
            payload[f"provenance_{key}"] = np.asarray([json.dumps(_jsonable(value), sort_keys=True)])
        else:
            payload[f"provenance_{key}"] = np.asarray([str(value)])
    np.savez_compressed(path, **payload)
    result = {
        "schema_version": 2,
        "design_schema": "paired_target_design_v1",
        "path": str(path),
        "sha256": sha256_file(path),
        "design_rows": {"brain": int(len(brain_x)), "lm": int(len(lm_x))},
        "predictor_columns": {"brain": int(brain_x.shape[1]), "lm": int(lm_x.shape[1])},
        "residual_pool_rows": {"brain": int(len(brain_residual)), "lm": int(len(lm_residual))},
        "target_dimensions": {"brain": int(brain_residual.shape[1]), "lm": int(lm_residual.shape[1])},
        "participant_count": int(len(np.unique(brain_participant_ids))),
        "censored_rows": {"brain": int(np.sum(brain_censor_mask)), "lm": int(np.sum(lm_censor_mask))},
        "fold_count": {"brain": int(len(brain_folds)), "lm": int(len(lm_folds))},
        "feature_group_count": int(len(brain_groups)),
        "residual_scale_definition": "mean per-target population SD of the corresponding empirical residual pool",
        "brain_empirical_residual_sd": _mean_empirical_residual_sd(brain_residual),
        "lm_empirical_residual_sd": _mean_empirical_residual_sd(lm_residual),
        "oracle_correlation_method": "spearman",
        **{key: str(value) for key, value in metadata.items()},
        "provenance": _jsonable(dict(provenance or {})),
    }
    sidecar = path.with_suffix(".json")
    write_json(sidecar, result)
    return result


def _validate_paired_design(design: Mapping[str, Any]) -> dict[str, Any]:
    """Validate an in-memory paired design before any simulation is run."""
    required = {
        "brain_x",
        "lm_x",
        "brain_residual",
        "lm_residual",
        "brain_participant_ids",
        "lm_participant_ids",
        "brain_censor_mask",
        "lm_censor_mask",
        "brain_folds",
        "lm_folds",
        "brain_feature_groups",
        "lm_feature_groups",
    }
    missing = sorted(required.difference(design))
    if missing:
        raise ValueError(f"paired power design is missing fields {missing}")
    brain_x = np.asarray(design["brain_x"], dtype=float)
    lm_x = np.asarray(design["lm_x"], dtype=float)
    brain_residual = np.asarray(design["brain_residual"], dtype=float)
    lm_residual = np.asarray(design["lm_residual"], dtype=float)
    if any(array.ndim != 2 for array in (brain_x, lm_x, brain_residual, lm_residual)):
        raise ValueError("paired design matrices must be two-dimensional")
    if not all(np.all(np.isfinite(array)) for array in (brain_x, lm_x, brain_residual, lm_residual)):
        raise ValueError("paired design matrices and residual pools must be finite")
    brain_ids = np.asarray(design["brain_participant_ids"], dtype=str).reshape(-1)
    lm_ids = np.asarray(design["lm_participant_ids"], dtype=str).reshape(-1)
    brain_censor = np.asarray(design["brain_censor_mask"], dtype=bool).reshape(-1)
    lm_censor = np.asarray(design["lm_censor_mask"], dtype=bool).reshape(-1)
    if len(brain_ids) != len(brain_x) or len(brain_censor) != len(brain_x):
        raise ValueError("brain row metadata does not match brain_x")
    if len(lm_ids) != len(lm_x) or len(lm_censor) != len(lm_x):
        raise ValueError("LM row metadata does not match lm_x")
    brain_groups = _groups_from_input(design["brain_feature_groups"], brain_x.shape[1])
    lm_groups = _groups_from_input(design["lm_feature_groups"], lm_x.shape[1])
    if len(brain_groups) != len(lm_groups):
        raise ValueError("paired design systems must have equal SAE feature-group counts")
    brain_folds = _validate_folds(design["brain_folds"], len(brain_x))
    lm_folds = _validate_folds(design["lm_folds"], len(lm_x))
    if str(design.get("participant_structure", "")) != "paired_system_rows":
        raise ValueError("paired power design participant_structure must be paired_system_rows")
    for key in ("design_label", "model_key", "bank_type", "direction", "censoring_manifest"):
        if not str(design.get(key, "")):
            raise ValueError(f"paired power design missing metadata {key}")
    return {
        **dict(design),
        "brain_x": brain_x,
        "lm_x": lm_x,
        "brain_residual": brain_residual,
        "lm_residual": lm_residual,
        "brain_participant_ids": brain_ids,
        "lm_participant_ids": lm_ids,
        "brain_censor_mask": brain_censor,
        "lm_censor_mask": lm_censor,
        "brain_feature_groups": brain_groups,
        "lm_feature_groups": lm_groups,
        "brain_folds": brain_folds,
        "lm_folds": lm_folds,
    }


def run_full_grid(design: Mapping[str, Any], power_spec: Mapping[str, Any], *, output_root: Path) -> dict[str, Any]:
    """Run the configured grid on an explicitly supplied design artifact.

    This function is intentionally explicit about its input: it cannot infer
    residuals from an arbitrary result directory, which prevents silently
    using an old or target-contaminated fit as a power design.
    """
    # The analysis design keeps the lag-expanded BOLD and unlagged LM
    # geometries separate.  Dispatch before the legacy single-matrix path so
    # an accidental flattening cannot silently change the estimand.
    if "brain_x" in design or str(design.get("design_schema", "")).startswith("paired_target_design"):
        return run_full_paired_grid(design, power_spec, output_root=output_root)

    x = np.asarray(design["x"], dtype=float)
    brain = np.asarray(design["brain_residual"], dtype=float)
    lm = np.asarray(design["lm_residual"], dtype=float)
    folds = design["folds"]
    groups = design.get("feature_groups")
    participant_ids = np.asarray(design.get("participant_ids", []), dtype=str).reshape(-1)
    censor_mask = np.asarray(design.get("censor_mask", []), dtype=bool).reshape(-1)
    if design.get("participant_structure") != "participant_by_time_rows":
        raise ValueError("power design must explicitly represent participant_by_time_rows")
    if len(participant_ids) != len(x) or len(censor_mask) != len(x):
        raise ValueError("power design participant_ids/censor_mask must match x rows")
    # Each design file represents one exact model/bank/direction.
    # Evaluate each design artifact in a separate call.
    model = str(design.get("model_key", ""))
    bank = str(design.get("bank_type", ""))
    direction = str(design.get("direction", ""))
    if model not in set(power_spec["designs"]):
        raise ValueError(f"design model_key {model!r} is not in the configured power designs")
    allowed_banks = set(str(value) for value in power_spec["bank_types"])
    bank_matches = bank in allowed_banks or any(
        bank.startswith(f"{value}_") or bank.startswith(f"{value}@") for value in allowed_banks
    )
    if not bank_matches:
        raise ValueError(f"design bank_type {bank!r} is not in the configured power bank types")
    if direction not in set(power_spec["directions"]):
        raise ValueError(f"design direction {direction!r} is not in the configured power directions")
    oracle_values = [float(v) for v in power_spec["oracle_correlations"]]
    scales = [float(v) for v in power_spec["residual_scale_multipliers"]]
    null_reps = int(power_spec["null_replicates"])
    positive_reps = int(power_spec["positive_replicates_per_grid_cell"])
    seed = int(power_spec.get("seed", 2026082704))
    # A full run uses design-specific residuals and groups supplied by the
    # caller.  Refuse an attempt to imply that one matrix covers multiple
    # models, banks, or transfer directions.
    if not design.get("design_label"):
        raise ValueError("full power design must include an explicit design_label (model/bank/direction provenance)")
    rows: list[dict[str, Any]] = []
    alpha_grid = [float(value) for value in power_spec.get("alpha_grid", ())]
    if not alpha_grid:
        raise ValueError("full power requires cross_validation alpha_grid for per-replicate nested selection")
    for rho in oracle_values:
        for scale in scales:
            reps = null_reps if rho == 0.0 else positive_reps
            values = np.empty(reps, dtype=float)
            for replicate in range(reps):
                case = simulate_power_case(
                    x,
                    brain,
                    lm,
                    folds,
                    oracle_correlation=rho,
                    residual_scale_multiplier=scale,
                    alpha=None,
                    alpha_grid=alpha_grid,
                    feature_groups=groups,
                    seed=seed + len(rows) * 100000 + replicate,
                    censor_mask=censor_mask,
                )
                values[replicate] = case["observed_reliance_spearman"]
            rows.append(
                {
                    "design_label": design["design_label"],
                    "model": model,
                    "bank_type": bank,
                    "direction": direction,
                    "oracle_correlation": rho,
                    "residual_scale_multiplier": scale,
                    "replicates": reps,
                    "observed_mean": float(np.nanmean(values)),
                    "observed_sd": float(np.nanstd(values, ddof=1)) if reps > 1 else None,
                    "values": values,
                }
            )
    null_by_key = {
        (row["model"], row["bank_type"], row["direction"], row["residual_scale_multiplier"]): np.asarray(row["values"], dtype=float)
        for row in rows
        if row["oracle_correlation"] == 0.0
    }
    for row in rows:
        key = (row["model"], row["bank_type"], row["direction"], row["residual_scale_multiplier"])
        null = null_by_key[key]
        threshold = float(np.nanquantile(null, 0.95))
        row["null_95_threshold"] = threshold
        row["detection_fraction"] = float(np.mean(np.asarray(row["values"]) > threshold))
        row["identifiable_at_80_percent"] = bool(row["detection_fraction"] >= float(power_spec["identifiability_power_threshold"]))
        del row["values"]
    output_root.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 1,
        "scientific_evidence": True,
        "analysis_role": "correspondence_semi_synthetic_power",
        "design_label": design["design_label"],
        "design_sha256": design.get("sha256"),
        "design_rows": int(len(x)),
        "design_features": int(x.shape[1]),
        "brain_target_dimensions": int(brain.shape[1]),
        "lm_target_dimensions": int(lm.shape[1]),
        "participant_count": int(len(np.unique(participant_ids))),
        "censored_rows": int(np.sum(censor_mask)),
        "censoring_manifest": design.get("censoring_manifest"),
        "fold_count": int(len(folds)),
        "row_index_semantics": "fold indices are global indices into the supplied participant-by-time design; no local sliced indices are reused",
        "rows": rows,
        "decision_rule": "detect if observed reliance Spearman exceeds the 95th percentile of the same-design rho=0 distribution; identifiable if detection fraction >= configured threshold",
    }
    write_json(output_root / f"power_{str(design['design_label']).replace('/', '_')}.json", result)
    return result


def _case_seed(base_seed: int, design: Mapping[str, Any], rho: float, scale: float, replicate: int) -> int:
    """Derive a stable seed from scientific coordinates, not loop order."""
    digest = stable_hash(
        {
            "base_seed": int(base_seed),
            "design_sha256": str(design.get("sha256", design.get("design_sha256", ""))),
            "design_label": str(design.get("design_label", "")),
            "oracle_correlation": float(rho),
            "residual_scale_multiplier": float(scale),
            "replicate": int(replicate),
        }
    )
    return int(digest[:16], 16) % (2**63 - 1)


def simulate_paired_power_case(
    design: Mapping[str, Any],
    *,
    oracle_correlation: float,
    residual_scale_multiplier: float,
    alpha_grid: Sequence[float],
    seed: int,
) -> dict[str, Any]:
    """Simulate one paired-target case using the actual two-system geometry.

    The latent feature profiles are shared at the SAE-feature-group level,
    but each target is generated from its own observed design matrix.  Thus
    lag-expanded BOLD predictors and unlagged LM predictors are not conflated.
    Empirical residual pools preserve each system's target covariance and the
    supplied censoring/fold structure is applied exactly during fitting.
    """
    design = _validate_paired_design(design)
    return _simulate_paired_power_case_validated(
        design,
        oracle_correlation=oracle_correlation,
        residual_scale_multiplier=residual_scale_multiplier,
        alpha_grid=alpha_grid,
        seed=seed,
    )

    # Kept below as the original uncached implementation while the optimized
    # shard path is validated.  The return above is the sole public path.
    alpha_values = [float(value) for value in alpha_grid]
    if not alpha_values or any(value <= 0 for value in alpha_values):
        raise ValueError("paired power simulation requires a positive alpha grid")
    rng = np.random.default_rng(int(seed))
    n_features = len(design["brain_feature_groups"])
    # The primary correspondence estimand is Spearman rho.  Generate the
    # latent profiles in that metric; a Pearson mixture would otherwise make
    # the nominal power grid answer a different question from the analysis.
    profile_brain, profile_lm = make_reliance_profiles(
        n_features,
        float(oracle_correlation),
        rng,
        correlation_method="spearman",
    )
    brain_columns = np.zeros(design["brain_x"].shape[1], dtype=float)
    lm_columns = np.zeros(design["lm_x"].shape[1], dtype=float)
    for score_brain, score_lm, brain_group, lm_group in zip(
        profile_brain,
        profile_lm,
        design["brain_feature_groups"],
        design["lm_feature_groups"], strict=False,
    ):
        brain_columns[brain_group] = float(score_brain)
        lm_columns[lm_group] = float(score_lm)
    residual_brain = np.asarray(design["brain_residual"], dtype=float)
    residual_lm = np.asarray(design["lm_residual"], dtype=float)
    # Keep the configured multiplier grid, but calibrate it separately in each
    # target system's empirical residual units.  A shared average would make
    # the BOLD and LM injections have different, undocumented SNRs whenever
    # their residual scales differ.
    brain_residual_sd = _mean_empirical_residual_sd(residual_brain)
    lm_residual_sd = _mean_empirical_residual_sd(residual_lm)
    brain_signal_scale = float(residual_scale_multiplier) * max(brain_residual_sd, 1e-12)
    lm_signal_scale = float(residual_scale_multiplier) * max(lm_residual_sd, 1e-12)
    brain_target = inject_targets(
        design["brain_x"],
        _noise_from_residual(residual_brain, rng, len(design["brain_x"]), residual_brain.shape[1]),
        brain_columns,
        signal_scale=brain_signal_scale,
        rng=rng,
    )
    lm_target = inject_targets(
        design["lm_x"],
        _noise_from_residual(residual_lm, rng, len(design["lm_x"]), residual_lm.shape[1]),
        lm_columns,
        signal_scale=lm_signal_scale,
        rng=rng,
    )
    brain_censor = np.asarray(design["brain_censor_mask"], dtype=bool)
    lm_censor = np.asarray(design["lm_censor_mask"], dtype=bool)
    if np.any(brain_censor):
        brain_target[brain_censor] = np.nan
    if np.any(lm_censor):
        lm_target[lm_censor] = np.nan
    brain_alphas = select_alpha_by_fold(
        design["brain_x"],
        brain_target,
        design["brain_folds"],
        alpha_values,
        purge_adjacent=int(design.get("purge_adjacent_trs", 4)),
    )
    lm_alphas = select_alpha_by_fold(
        design["lm_x"],
        lm_target,
        design["lm_folds"],
        alpha_values,
        purge_adjacent=int(design.get("purge_adjacent_trs", 4)),
    )
    brain_fit = heldout_grouped_reliance(
        design["brain_x"],
        brain_target,
        design["brain_folds"],
        alpha=brain_alphas,
        feature_groups=design["brain_feature_groups"],
    )
    lm_fit = heldout_grouped_reliance(
        design["lm_x"],
        lm_target,
        design["lm_folds"],
        alpha=lm_alphas,
        feature_groups=design["lm_feature_groups"],
    )
    observed = profile_correlation(
        brain_fit["reliance_profile"], lm_fit["reliance_profile"], method="spearman"
    )
    achieved_pearson = profile_correlation(profile_brain, profile_lm, method="pearson")
    achieved_spearman = profile_correlation(profile_brain, profile_lm, method="spearman")
    return {
        "oracle_correlation": float(oracle_correlation),
        "oracle_correlation_method": "spearman",
        "profile_achieved_spearman": float(achieved_spearman),
        "profile_achieved_pearson": float(achieved_pearson),
        "residual_scale_multiplier": float(residual_scale_multiplier),
        "residual_scale_definition": "mean per-target population SD of the corresponding empirical residual pool",
        "brain_empirical_residual_sd": float(brain_residual_sd),
        "lm_empirical_residual_sd": float(lm_residual_sd),
        "brain_signal_scale_absolute": float(brain_signal_scale),
        "lm_signal_scale_absolute": float(lm_signal_scale),
        "observed_reliance_spearman": float(observed),
        "brain_reliance_profile": np.asarray(brain_fit["reliance_profile"], dtype=np.float32),
        "lm_reliance_profile": np.asarray(lm_fit["reliance_profile"], dtype=np.float32),
        "brain_mean_r2_oos": float(np.nanmean([row["mean_r2_oos"] for row in brain_fit["fold_scores"]])),
        "lm_mean_r2_oos": float(np.nanmean([row["mean_r2_oos"] for row in lm_fit["fold_scores"]])),
        "brain_alphas": [float(value) for value in brain_alphas],
        "lm_alphas": [float(value) for value in lm_alphas],
    }


def _validate_power_coordinates(design: Mapping[str, Any], power_spec: Mapping[str, Any]) -> None:
    model = str(design.get("model_key", ""))
    bank = str(design.get("bank_type", ""))
    direction = str(design.get("direction", ""))
    if model not in {str(value) for value in power_spec["designs"]}:
        raise ValueError(f"design model_key {model!r} is not in the configured power designs")
    allowed_banks = {str(value) for value in power_spec["bank_types"]}
    bank_matches = bank in allowed_banks or any(
        bank.startswith(f"{value}_") or bank.startswith(f"{value}@") for value in allowed_banks
    )
    if not bank_matches:
        raise ValueError(f"design bank_type {bank!r} is not in the configured power bank types")
    if direction not in {str(value) for value in power_spec["directions"]}:
        raise ValueError(f"design direction {direction!r} is not in the configured power directions")
    if not str(design.get("design_label", "")):
        raise ValueError("power design must include an explicit design_label")
    if not str(design.get("sha256", design.get("design_sha256", ""))):
        raise ValueError("power design must include its SHA-256 fingerprint")


def _power_grid_coordinates(power_spec: Mapping[str, Any]) -> Iterable[tuple[float, float, int]]:
    null_reps = int(power_spec["null_replicates"])
    positive_reps = int(power_spec["positive_replicates_per_grid_cell"])
    for rho in (float(value) for value in power_spec["oracle_correlations"]):
        for scale in (float(value) for value in power_spec["residual_scale_multipliers"]):
            yield rho, scale, null_reps if np.isclose(rho, 0.0) else positive_reps


def run_power_grid_shard(
    design: Mapping[str, Any],
    power_spec: Mapping[str, Any],
    *,
    output_path: Path,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    """Run one deterministic, resumable shard of one exact power design."""
    design = _validate_paired_design(design)
    _validate_power_coordinates(design, power_spec)
    if int(shard_count) < 1 or not 0 <= int(shard_index) < int(shard_count):
        raise ValueError("shard_index must be in [0, shard_count)")
    alpha_grid = [float(value) for value in power_spec.get("alpha_grid", ())]
    if not alpha_grid:
        raise ValueError("full power requires a non-empty alpha_grid")
    base_seed = int(power_spec.get("seed", 2026082704))
    # Validation and all X-only fold work are invariant across replicates.
    # Prepare them once per shard; target scaling, coefficients, and scores
    # remain replicate-specific below.
    x_caches = {
        "brain": _prepare_x_cache(
            design["brain_x"],
            design["brain_folds"],
            ~np.asarray(design["brain_censor_mask"], dtype=bool),
            purge_adjacent=int(design.get("purge_adjacent_trs", 4)),
            alpha_grid=alpha_grid,
        ),
        "lm": _prepare_x_cache(
            design["lm_x"],
            design["lm_folds"],
            ~np.asarray(design["lm_censor_mask"], dtype=bool),
            purge_adjacent=int(design.get("purge_adjacent_trs", 4)),
            alpha_grid=alpha_grid,
        ),
    }
    rows: list[dict[str, Any]] = []
    for rho, scale, replicate_count in _power_grid_coordinates(power_spec):
        replicate_ids = list(range(int(shard_index), int(replicate_count), int(shard_count)))
        values: list[float] = []
        for replicate in replicate_ids:
            case = _simulate_paired_power_case_validated(
                design,
                oracle_correlation=rho,
                residual_scale_multiplier=scale,
                alpha_grid=alpha_grid,
                seed=_case_seed(base_seed, design, rho, scale, replicate),
                x_caches=x_caches,
                solver="cholesky",
            )
            values.append(float(case["observed_reliance_spearman"]))
        rows.append(
            {
                "oracle_correlation": float(rho),
                "residual_scale_multiplier": float(scale),
                "replicates_total": int(replicate_count),
                "replicate_ids": replicate_ids,
                "replicates_in_shard": int(len(replicate_ids)),
                "values": values,
            }
        )
    result = {
        "schema_version": 2,
        "artifact_kind": "power_shard",
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
        "residual_scale_definition": "mean per-target population SD of the corresponding empirical residual pool",
        "brain_empirical_residual_sd": _mean_empirical_residual_sd(design["brain_residual"]),
        "lm_empirical_residual_sd": _mean_empirical_residual_sd(design["lm_residual"]),
        "alpha_grid": alpha_grid,
        "rows": rows,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, result)
    return result


def aggregate_power_shards(
    design: Mapping[str, Any],
    power_spec: Mapping[str, Any],
    shard_paths: Sequence[Path],
    *,
    output_path: Path,
) -> dict[str, Any]:
    """Require complete shard coverage, then write one canonical power result."""
    design = _validate_paired_design(design)
    _validate_power_coordinates(design, power_spec)
    if not shard_paths:
        raise ValueError("at least one shard is required")
    shard_records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in shard_paths]
    expected_count = int(shard_records[0].get("shard_count", 0))
    if expected_count < 1:
        raise ValueError("shard records must declare a positive shard_count")
    if len(shard_records) != expected_count:
        raise ValueError(f"expected {expected_count} shard records, found {len(shard_records)}")
    expected_indices = set(range(expected_count))
    observed_indices = {int(record.get("shard_index", -1)) for record in shard_records}
    if observed_indices != expected_indices:
        raise ValueError(f"shard index coverage is incomplete: {sorted(observed_indices)}")
    design_identity = str(design.get("sha256", design.get("design_sha256")))
    cell_values: dict[tuple[float, float], dict[int, float]] = {}
    for record in shard_records:
        for key in ("design_label", "design_sha256", "model", "bank_type", "direction"):
            if str(record.get(key, "")) != str(
                design_identity if key == "design_sha256" else design.get({"model": "model_key", "bank_type": "bank_type", "direction": "direction"}.get(key, key), design.get(key, ""))
            ):
                raise ValueError(f"shard provenance mismatch for {key}")
        for row in record.get("rows", []):
            key = (float(row["oracle_correlation"]), float(row["residual_scale_multiplier"]))
            ids = [int(value) for value in row.get("replicate_ids", [])]
            values = [float(value) for value in row.get("values", [])]
            if len(ids) != len(values):
                raise ValueError(f"shard row {key} has mismatched replicate IDs and values")
            target = cell_values.setdefault(key, {})
            for replicate, value in zip(ids, values, strict=False):
                if replicate in target:
                    raise ValueError(f"duplicate replicate {replicate} for grid cell {key}")
                target[replicate] = value
    canonical_rows: list[dict[str, Any]] = []
    null_values: dict[float, np.ndarray] = {}
    expected_cells = {(rho, scale): count for rho, scale, count in _power_grid_coordinates(power_spec)}
    for (rho, scale), replicate_count in expected_cells.items():
        values_by_id = cell_values.get((rho, scale), {})
        expected_ids = set(range(int(replicate_count)))
        if set(values_by_id) != expected_ids:
            missing = sorted(expected_ids - set(values_by_id))[:5]
            extra = sorted(set(values_by_id) - expected_ids)[:5]
            raise ValueError(f"incomplete grid cell {(rho, scale)}; missing={missing}, extra={extra}")
        values = np.asarray([values_by_id[index] for index in range(int(replicate_count))], dtype=float)
        if np.isclose(rho, 0.0):
            null_values[float(scale)] = values
        canonical_rows.append(
            {
                "design_label": str(design["design_label"]),
                "model": str(design["model_key"]),
                "bank_type": str(design["bank_type"]),
                "bank_variant": str(design.get("bank_variant", design["bank_type"])),
                "direction": str(design["direction"]),
                "oracle_correlation": float(rho),
                "residual_scale_multiplier": float(scale),
                "replicates": int(replicate_count),
                "observed_mean": float(np.nanmean(values)),
                "observed_sd": float(np.nanstd(values, ddof=1)) if len(values) > 1 else None,
                "values_internal": values,
            }
        )
    for row in canonical_rows:
        null = null_values[float(row["residual_scale_multiplier"])]
        threshold = float(np.nanquantile(null, 0.95))
        values = np.asarray(row.pop("values_internal"), dtype=float)
        row["null_95_threshold"] = threshold
        row["detection_fraction"] = float(np.mean(values > threshold))
        row["identifiable_at_80_percent"] = bool(
            row["detection_fraction"] >= float(power_spec["identifiability_power_threshold"])
        )
    output = {
        "schema_version": 2,
        "scientific_evidence": True,
        "analysis_role": "correspondence_semi_synthetic_power",
        "design_schema": "paired_target_design_v1",
        "design_label": str(design["design_label"]),
        "design_sha256": design_identity,
        "model": str(design["model_key"]),
        "bank_type": str(design["bank_type"]),
        "bank_variant": str(design.get("bank_variant", design["bank_type"])),
        "direction": str(design["direction"]),
        "source_story": str(design.get("source_story", "")),
        "target_story": str(design.get("target_story", "")),
        "censoring_manifest": str(design["censoring_manifest"]),
        "design_rows": {
            "brain": int(len(design["brain_x"])),
            "lm": int(len(design["lm_x"])),
        },
        "predictor_columns": {
            "brain": int(design["brain_x"].shape[1]),
            "lm": int(design["lm_x"].shape[1]),
        },
        "target_dimensions": {
            "brain": int(design["brain_residual"].shape[1]),
            "lm": int(design["lm_residual"].shape[1]),
        },
        "participant_count": int(len(np.unique(design["brain_participant_ids"]))),
        "censored_rows": {
            "brain": int(np.sum(design["brain_censor_mask"])),
            "lm": int(np.sum(design["lm_censor_mask"])),
        },
        "fold_count": {
            "brain": int(len(design["brain_folds"])),
            "lm": int(len(design["lm_folds"])),
        },
        "feature_group_count": int(len(design["brain_feature_groups"])),
        "alpha_grid": [float(value) for value in power_spec.get("alpha_grid", ())],
        "null_replicates": int(power_spec["null_replicates"]),
        "positive_replicates_per_grid_cell": int(power_spec["positive_replicates_per_grid_cell"]),
        "shard_count": int(expected_count),
        "shard_paths": [str(Path(path)) for path in shard_paths],
        "seed_contract": "sha256(base_seed, design_sha256, cell coordinates, replicate id)",
        "oracle_correlation_method": "spearman",
        "residual_scale_definition": "mean per-target population SD of the corresponding empirical residual pool",
        "brain_empirical_residual_sd": _mean_empirical_residual_sd(design["brain_residual"]),
        "lm_empirical_residual_sd": _mean_empirical_residual_sd(design["lm_residual"]),
        "row_index_semantics": "BOLD rows are participant-by-time; LM rows are stimulus TRs; each system retains its own folds and feature groups",
        "decision_rule": "detect if paired held-out reliance Spearman exceeds the 95th percentile of the same-design rho=0 distribution; identifiable if detection fraction >= configured threshold",
        "rows": canonical_rows,
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, output)
    return output


def run_full_paired_grid(
    design: Mapping[str, Any], power_spec: Mapping[str, Any], *, output_root: Path
) -> dict[str, Any]:
    """Run and aggregate one paired design as a single local shard.

    This keeps ``--full --design`` backwards-compatible for a small test or a
    one-shot run. Larger runs may call :func:`run_power_grid_shard` multiple
    times and then :func:`aggregate_power_shards` after all shards are present.
    """
    design = _validate_paired_design(design)
    output_root = Path(output_root)
    label = str(design["design_label"]).replace("/", "_")
    shard_path = output_root / "shards" / f"power_{label}.s000of001.json"
    run_power_grid_shard(
        design,
        power_spec,
        output_path=shard_path,
        shard_index=0,
        shard_count=1,
    )
    output_path = output_root / f"power_{label}.json"
    return aggregate_power_shards(
        design,
        power_spec,
        [shard_path],
        output_path=output_path,
    )


def run(*, config_path: Path, spec: Mapping[str, Any], args: argparse.Namespace | None = None) -> dict[str, Any]:
    """Run a configured paired design or report that the design is unavailable."""
    output_root = (
        resolve_output_root(override=getattr(args, "output_root", None))
        / str(spec["paths"]["artifact_root"])
        / "power"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    design_arg = getattr(args, "design", None) if args is not None else None
    full = bool(getattr(args, "full", False)) if args is not None else False
    aggregate = bool(getattr(args, "aggregate", False)) if args is not None else False
    if full and design_arg:
        design = _load_design(Path(design_arg).resolve())
        power_config = dict(spec["power"])
        power_config["alpha_grid"] = list(spec["cross_validation"]["alpha_grid"])
        power_config["purge_adjacent_trs"] = int(spec["cross_validation"]["purge_adjacent_trs"])
        shard_count = int(getattr(args, "shard_count", 1) or 1)
        shard_index = int(getattr(args, "shard_index", 0) or 0)
        if shard_count > 1 or shard_index:
            shard_output = getattr(args, "shard_output", None)
            if shard_output is None:
                label = str(design["design_label"]).replace("/", "_")
                shard_output = output_root / "shards" / f"power_{label}.s{shard_index:03d}of{shard_count:03d}.json"
            shard = run_power_grid_shard(
                design,
                power_config,
                output_path=Path(shard_output),
                shard_index=shard_index,
                shard_count=shard_count,
            )
            return {"status": "shard_complete", "scientific_evidence": False, "shard": shard}
        power = run_full_grid(design, power_config, output_root=output_root)
        return {"status": "complete", "scientific_evidence": True, "power": power}
    if aggregate and design_arg:
        design = _load_design(Path(design_arg).resolve())
        power_config = dict(spec["power"])
        power_config["alpha_grid"] = list(spec["cross_validation"]["alpha_grid"])
        power_config["purge_adjacent_trs"] = int(spec["cross_validation"]["purge_adjacent_trs"])
        shard_paths = [Path(value).resolve() for value in (getattr(args, "shard_paths", None) or ())]
        if not shard_paths:
            raise ValueError("--aggregate requires --shard-paths")
        label = str(design["design_label"]).replace("/", "_")
        aggregate_output = getattr(args, "aggregate_output", None)
        if aggregate_output is None:
            aggregate_output = output_root / f"power_{label}.json"
        power = aggregate_power_shards(
            design,
            power_config,
            shard_paths,
            output_path=Path(aggregate_output),
        )
        return {"status": "complete", "scientific_evidence": True, "power": power}
    status = {
        "schema_version": 1,
        "analysis_role": "correspondence_semi_synthetic_power",
        "status": "blocked_pending_design",
        "scientific_evidence": False,
        "reason": "Full power requires a provenance-labeled design artifact containing real predictor geometry, empirical residual covariance, target dimensionality, participant-by-time rows, feature groups, folds, and censoring decisions. No design was supplied to this stage; reporting power without these fields would be underidentified.",
        "required_design_fields": ["brain_x", "lm_x", "brain_residual", "lm_residual", "brain_folds", "lm_folds", "brain_feature_groups", "lm_feature_groups", "brain_participant_ids", "lm_participant_ids", "brain_censor_mask", "lm_censor_mask", "design_label", "model_key", "bank_type", "direction", "participant_structure", "censoring_manifest"],
        "configured_grid": _jsonable(spec["power"]),
    }
    write_json(output_root / "power_status.json", status)
    manifest = {
        "schema_version": 1,
        "status": status["status"],
        "scientific_evidence": False,
        "status_artifact": str(output_root / "power_status.json"),
        "code_sha256": sha256_file(Path(__file__).resolve()),
        "config_sha256": sha256_file(config_path.resolve()),
    }
    write_json(output_root / "manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--design", type=Path, default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-output", type=Path, default=None)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--shard-paths", nargs="+", type=Path, default=None)
    parser.add_argument("--aggregate-output", type=Path, default=None)
    args = parser.parse_args(argv)
    config_path = (args.config or default_config_path()).resolve()
    result = run(config_path=config_path, spec=load_config(config_path), args=args)
    print(json.dumps(_jsonable(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
