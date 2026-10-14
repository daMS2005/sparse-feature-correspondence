from __future__ import annotations

import hashlib
import json
import math
import re
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import yaml
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent


def load_spec(path: Path) -> dict[str, Any]:
    spec = yaml.safe_load(path.read_text())
    if int(spec.get("schema_version", -1)) != 2:
        raise ValueError("analysis specification must have schema_version=2")
    return spec


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def analysis_fingerprint(
    spec_path: Path,
    source_paths: Iterable[Path],
    code_paths: Iterable[Path],
    *,
    extra: Any | None = None,
) -> str:
    payload = {
        "spec": yaml.safe_load(spec_path.read_text()),
        "sources": {str(p): sha256_file(p) for p in sorted(set(source_paths))},
        "code": {str(p): sha256_file(p) for p in sorted(set(code_paths))},
        "extra": extra,
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def write_json(path: Path, value: Any) -> None:
    def convert(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {str(k): convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [convert(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return convert(obj.tolist())
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        return obj

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(convert(value), indent=2, sort_keys=True) + "\n")


def assert_cache_fingerprint(path: Path, expected: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    observed = data.get("analysis_fingerprint")
    if observed != expected:
        raise RuntimeError(
            f"stale cache rejected: {path}\nexpected {expected}\nobserved {observed}"
        )
    return data


def block_labels(n_time: int, n_blocks: int) -> np.ndarray:
    labels = np.empty(n_time, dtype=int)
    for block, indices in enumerate(np.array_split(np.arange(n_time), n_blocks)):
        labels[indices] = block
    return labels


def purged_training_positions(
    n_time: int,
    held_positions: np.ndarray,
    *,
    allowed_positions: np.ndarray | None = None,
    purge_adjacent: int,
) -> np.ndarray:
    allowed = np.ones(n_time, dtype=bool)
    if allowed_positions is not None:
        allowed[:] = False
        allowed[np.asarray(allowed_positions, dtype=int)] = True
    held = np.asarray(held_positions, dtype=int)
    if held.size:
        lo = max(0, int(held.min()) - purge_adjacent)
        hi = min(n_time, int(held.max()) + purge_adjacent + 1)
        allowed[lo:hi] = False
    return np.flatnonzero(allowed)


def lag_expand(values: np.ndarray, lags: Sequence[int]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    n_time, n_features = values.shape
    out = np.zeros((n_time, n_features * len(lags)), dtype=np.float64)
    for feature in range(n_features):
        for lag_index, lag in enumerate(lags):
            column = feature * len(lags) + lag_index
            if lag == 0:
                out[:, column] = values[:, feature]
            else:
                out[lag:, column] = values[:-lag, feature]
    return out


def feature_groups(n_features: int, n_lags: int) -> list[np.ndarray]:
    return [np.arange(i * n_lags, (i + 1) * n_lags, dtype=int) for i in range(n_features)]


def finite_rows(y: np.ndarray) -> np.ndarray:
    return np.flatnonzero(np.all(np.isfinite(y), axis=1))


def pearson_columns(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    yc = y - y.mean(axis=0)
    pc = prediction - prediction.mean(axis=0)
    denom = np.sqrt(np.sum(yc * yc, axis=0) * np.sum(pc * pc, axis=0))
    result = np.full(y.shape[1], np.nan, dtype=float)
    valid = denom > 0
    result[valid] = np.sum(yc[:, valid] * pc[:, valid], axis=0) / denom[valid]
    return result


def r2_oos_columns(y: np.ndarray, prediction: np.ndarray, training_mean: np.ndarray) -> np.ndarray:
    residual = np.sum((y - prediction) ** 2, axis=0)
    denominator = np.sum((y - np.asarray(training_mean)[None, :]) ** 2, axis=0)
    result = np.full(y.shape[1], np.nan, dtype=float)
    valid = denominator > 0
    result[valid] = 1.0 - residual[valid] / denominator[valid]
    return result


def mean_metric(
    y: np.ndarray,
    prediction: np.ndarray,
    training_mean: np.ndarray,
    metric: str,
) -> float:
    if metric == "pearson_r":
        values = pearson_columns(y, prediction)
    elif metric == "r2_oos":
        values = r2_oos_columns(y, prediction, training_mean)
    else:
        raise ValueError(f"unknown metric {metric}")
    return float(np.nanmean(values))


@dataclass
class RidgeFit:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    weights_standardized: np.ndarray
    alpha: float

    def standardized_x(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.x_mean) / self.x_scale

    def predict(self, x: np.ndarray) -> np.ndarray:
        pred_std = self.standardized_x(x) @ self.weights_standardized
        return pred_std * self.y_scale + self.y_mean

    def contribution(self, x: np.ndarray, columns: np.ndarray) -> np.ndarray:
        xt = self.standardized_x(x)
        pred_std = xt[:, columns] @ self.weights_standardized[columns]
        return pred_std * self.y_scale


@dataclass
class RidgePath:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_scale: np.ndarray
    xs: np.ndarray
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    projected_y: np.ndarray

    def at(self, alpha: float) -> RidgeFit:
        dual = self.eigenvectors @ (
            self.projected_y / (self.eigenvalues[:, None] + float(alpha))
        )
        weights = self.xs.T @ dual
        return RidgeFit(
            self.x_mean,
            self.x_scale,
            self.y_mean,
            self.y_scale,
            weights,
            float(alpha),
        )


def build_ridge_path(x: np.ndarray, y: np.ndarray, train_positions: np.ndarray) -> RidgePath:
    train_positions = np.asarray(train_positions, dtype=int)
    x_train = np.asarray(x[train_positions], dtype=np.float64)
    y_train = np.asarray(y[train_positions], dtype=np.float64)
    if not np.all(np.isfinite(y_train)):
        raise ValueError("ridge training target contains non-finite values")
    x_mean = x_train.mean(axis=0)
    x_scale = x_train.std(axis=0)
    x_scale[x_scale == 0] = 1.0
    y_mean = y_train.mean(axis=0)
    y_scale = y_train.std(axis=0)
    y_scale[y_scale == 0] = 1.0
    xs = (x_train - x_mean) / x_scale
    ys = (y_train - y_mean) / y_scale
    gram = xs @ xs.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    return RidgePath(
        x_mean,
        x_scale,
        y_mean,
        y_scale,
        xs,
        eigenvalues,
        eigenvectors,
        eigenvectors.T @ ys,
    )


def fit_ridge(x: np.ndarray, y: np.ndarray, train_positions: np.ndarray, alpha: float) -> RidgeFit:
    return build_ridge_path(x, y, train_positions).at(alpha)


def fixed_band_weights(
    n_blocks: int,
    n_candidates: int,
    concentrations: Sequence[float],
    seed: int,
) -> np.ndarray:
    if n_candidates < n_blocks + 1:
        raise ValueError("band random search must include equal and block-dominant candidates")
    rng = np.random.default_rng(seed)
    rows = [np.ones(n_blocks, dtype=np.float64)]
    epsilon = 1e-6
    for block in range(n_blocks):
        row = np.full(n_blocks, epsilon, dtype=np.float64)
        row[block] = float(n_blocks)
        rows.append(row)
    concentration_values = list(concentrations)
    while len(rows) < n_candidates:
        concentration = float(concentration_values[(len(rows) - n_blocks - 1) % len(concentration_values)])
        row = rng.dirichlet(np.full(n_blocks, concentration)) * n_blocks
        rows.append(row)
    return np.asarray(rows[:n_candidates])


@dataclass
class MultiBandFit:
    x_means: list[np.ndarray]
    x_scales: list[np.ndarray]
    y_mean: np.ndarray
    y_scale: np.ndarray
    train_blocks_standardized: list[np.ndarray]
    dual: np.ndarray
    gamma: np.ndarray
    alpha: float

    def standardized_blocks(self, blocks: Sequence[np.ndarray]) -> list[np.ndarray]:
        return [
            (np.asarray(block, dtype=np.float64) - mean) / scale
            for block, mean, scale in zip(blocks, self.x_means, self.x_scales, strict=False)
        ]

    def predict(self, blocks: Sequence[np.ndarray]) -> np.ndarray:
        standardized = self.standardized_blocks(blocks)
        prediction_std = np.zeros((len(standardized[0]), self.dual.shape[1]), dtype=np.float64)
        for gamma, test, train in zip(self.gamma, standardized, self.train_blocks_standardized, strict=False):
            prediction_std += float(gamma) * (test @ train.T) @ self.dual
        return prediction_std * self.y_scale + self.y_mean

    def block_column_contribution(
        self,
        blocks: Sequence[np.ndarray],
        block_index: int,
        columns: np.ndarray,
    ) -> np.ndarray:
        standardized = self.standardized_blocks(blocks)[block_index]
        train = self.train_blocks_standardized[block_index]
        contribution_std = (
            float(self.gamma[block_index])
            * standardized[:, columns]
            @ train[:, columns].T
            @ self.dual
        )
        return contribution_std * self.y_scale


def fit_multiband(
    blocks: Sequence[np.ndarray],
    y: np.ndarray,
    train_positions: np.ndarray,
    *,
    gamma: np.ndarray,
    alpha: float,
) -> MultiBandFit:
    train_positions = np.asarray(train_positions, dtype=int)
    y_train = np.asarray(y[train_positions], dtype=np.float64)
    y_mean = y_train.mean(axis=0)
    y_scale = y_train.std(axis=0)
    y_scale[y_scale == 0] = 1.0
    ys = (y_train - y_mean) / y_scale
    x_means = []
    x_scales = []
    standardized = []
    for block in blocks:
        train = np.asarray(block[train_positions], dtype=np.float64)
        mean = train.mean(axis=0)
        scale = train.std(axis=0)
        scale[scale == 0] = 1.0
        x_means.append(mean)
        x_scales.append(scale)
        standardized.append((train - mean) / scale)
    kernel = np.zeros((len(train_positions), len(train_positions)), dtype=np.float64)
    for weight, block in zip(gamma, standardized, strict=False):
        kernel += float(weight) * (block @ block.T)
    eigenvalues, eigenvectors = np.linalg.eigh(kernel)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    dual = eigenvectors @ ((eigenvectors.T @ ys) / (eigenvalues[:, None] + float(alpha)))
    return MultiBandFit(
        x_means=x_means,
        x_scales=x_scales,
        y_mean=y_mean,
        y_scale=y_scale,
        train_blocks_standardized=standardized,
        dual=dual,
        gamma=np.asarray(gamma, dtype=np.float64),
        alpha=float(alpha),
    )


def select_multiband_parameters(
    x_blocks: Sequence[np.ndarray],
    y: np.ndarray,
    blocks: np.ndarray,
    outer_block: int,
    alphas: np.ndarray,
    gammas: np.ndarray,
    *,
    metric: str,
    purge_adjacent: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    n_time = len(blocks)
    valid_y = finite_rows(y)
    outer_test = np.flatnonzero(blocks == outer_block)
    outer_train = purged_training_positions(n_time, outer_test, purge_adjacent=purge_adjacent)
    outer_train = np.intersect1d(outer_train, valid_y, assume_unique=True)
    scores = np.zeros((len(gammas), len(alphas)), dtype=np.float64)
    used = 0
    for inner_block in sorted(np.unique(blocks).tolist()):
        if inner_block == outer_block:
            continue
        validation = np.intersect1d(
            np.flatnonzero(blocks == inner_block), outer_train, assume_unique=True
        )
        inner_train = purged_training_positions(
            n_time,
            validation,
            allowed_positions=outer_train,
            purge_adjacent=purge_adjacent,
        )
        inner_train = np.intersect1d(inner_train, valid_y, assume_unique=True)
        if len(validation) < 3 or len(inner_train) < 3:
            continue
        y_train = np.asarray(y[inner_train], dtype=np.float64)
        y_mean = y_train.mean(axis=0)
        y_scale = y_train.std(axis=0)
        y_scale[y_scale == 0] = 1.0
        ys = (y_train - y_mean) / y_scale
        train_standardized = []
        validation_standardized = []
        for block in x_blocks:
            train_block = np.asarray(block[inner_train], dtype=np.float64)
            mean = train_block.mean(axis=0)
            scale = train_block.std(axis=0)
            scale[scale == 0] = 1.0
            train_standardized.append((train_block - mean) / scale)
            validation_standardized.append(
                (np.asarray(block[validation], dtype=np.float64) - mean) / scale
            )
        for gamma_index, gamma in enumerate(gammas):
            kernel = np.zeros((len(inner_train), len(inner_train)), dtype=np.float64)
            for weight, block in zip(gamma, train_standardized, strict=False):
                kernel += float(weight) * (block @ block.T)
            eigenvalues, eigenvectors = np.linalg.eigh(kernel)
            eigenvalues = np.maximum(eigenvalues, 0.0)
            projected_y = eigenvectors.T @ ys
            cross_kernel = np.zeros((len(validation), len(inner_train)), dtype=np.float64)
            for weight, test, train in zip(
                gamma, validation_standardized, train_standardized, strict=False
            ):
                cross_kernel += float(weight) * (test @ train.T)
            projected_test = cross_kernel @ eigenvectors
            for alpha_index, alpha in enumerate(alphas):
                prediction_std = (projected_test / (eigenvalues + float(alpha))) @ projected_y
                prediction = prediction_std * y_scale + y_mean
                scores[gamma_index, alpha_index] += mean_metric(
                    y[validation], prediction, y_mean, metric
                )
        used += 1
    if used == 0:
        raise RuntimeError(f"outer block {outer_block} has no multiband inner folds")
    scores /= used
    best = np.argwhere(scores == np.nanmax(scores))
    gamma_index, alpha_index = best[0]
    return gammas[int(gamma_index)], float(alphas[int(alpha_index)]), scores


def nested_multiband_cv(
    x_blocks: Sequence[np.ndarray],
    y: np.ndarray,
    blocks: np.ndarray,
    alphas: np.ndarray,
    gammas: np.ndarray,
    *,
    selection_metric: str,
    purge_adjacent: int,
    reliance_block: int,
    reliance_groups: list[np.ndarray],
) -> dict[str, Any]:
    n_time, n_targets = y.shape
    valid_y = finite_rows(y)
    oof = np.full((n_time, n_targets), np.nan, dtype=np.float64)
    total_sse = np.zeros(n_targets, dtype=np.float64)
    total_denom = np.zeros(n_targets, dtype=np.float64)
    total_delta = np.zeros((len(reliance_groups), n_targets), dtype=np.float64)
    folds = []
    fold_reliance = []
    fold_scaling = []
    for outer_block in sorted(np.unique(blocks).tolist()):
        test = np.intersect1d(np.flatnonzero(blocks == outer_block), valid_y, assume_unique=True)
        outer_test_full = np.flatnonzero(blocks == outer_block)
        train = purged_training_positions(n_time, outer_test_full, purge_adjacent=purge_adjacent)
        train = np.intersect1d(train, valid_y, assume_unique=True)
        gamma, alpha, inner_scores = select_multiband_parameters(
            x_blocks, y, blocks, outer_block, alphas, gammas,
            metric=selection_metric, purge_adjacent=purge_adjacent,
        )
        fit = fit_multiband(x_blocks, y, train, gamma=gamma, alpha=alpha)
        test_blocks = [block[test] for block in x_blocks]
        prediction = fit.predict(test_blocks)
        oof[test] = prediction
        full_sse = np.sum((y[test] - prediction) ** 2, axis=0)
        denominator = np.sum((y[test] - fit.y_mean[None, :]) ** 2, axis=0)
        total_sse += full_sse
        total_denom += denominator
        fold_delta = np.zeros((len(reliance_groups), n_targets), dtype=np.float64)
        for group_index, columns in enumerate(reliance_groups):
            ablated = prediction - fit.block_column_contribution(
                test_blocks, reliance_block, columns
            )
            fold_delta[group_index] = np.sum((y[test] - ablated) ** 2, axis=0) - full_sse
        total_delta += fold_delta
        with np.errstate(divide="ignore", invalid="ignore"):
            fold_scaled = np.full_like(fold_delta, np.nan)
            np.divide(fold_delta, denominator[None, :], out=fold_scaled, where=denominator[None, :] > 0)
        fold_reliance.append(np.nanmean(fold_scaled, axis=1))
        fold_scaling.append(
            {
                "x_means": fit.x_means,
                "x_scales": fit.x_scales,
                "y_mean": fit.y_mean,
                "y_scale": fit.y_scale,
                "train_positions": train,
                "test_positions": test,
            }
        )
        folds.append(
            {
                "fold": int(outer_block),
                "alpha": alpha,
                "gamma": gamma,
                "effective_block_penalties": alpha / gamma,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "mean_r": mean_metric(y[test], prediction, fit.y_mean, "pearson_r"),
                "mean_r2_oos": mean_metric(y[test], prediction, fit.y_mean, "r2_oos"),
                "inner_scores": inner_scores,
            }
        )
    valid_oof = finite_rows(oof)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.full_like(total_sse, np.nan)
        np.divide(total_sse, total_denom, out=ratio, where=total_denom > 0)
        scaled_delta = np.full_like(total_delta, np.nan)
        np.divide(total_delta, total_denom[None, :], out=scaled_delta, where=total_denom[None, :] > 0)
    return {
        "selection_metric": selection_metric,
        "mean_fold_r": float(np.mean([fold["mean_r"] for fold in folds])),
        "pooled_oof_r": float(np.nanmean(pearson_columns(y[valid_oof], oof[valid_oof]))),
        "pooled_r2_oos": float(np.nanmean(1.0 - ratio)),
        "folds": folds,
        "oof": oof,
        "reliance": np.nanmean(scaled_delta, axis=1),
        "fold_reliance": np.asarray(fold_reliance),
        "fold_scaling": fold_scaling,
    }

def modal_multiband_parameters(folds: list[dict[str, Any]]) -> tuple[np.ndarray, float]:
    rows = [
        (float(fold["alpha"]), *np.asarray(fold["gamma"], dtype=float).tolist())
        for fold in folds
    ]
    counts: dict[tuple[float, ...], int] = {}
    for row in rows:
        counts[row] = counts.get(row, 0) + 1
    maximum = max(counts.values())
    selected = min(row for row, count in counts.items() if count == maximum)
    return np.asarray(selected[1:], dtype=np.float64), float(selected[0])


def fit_multiband_transfer(
    train_blocks: Sequence[np.ndarray],
    y_train: np.ndarray,
    test_blocks: Sequence[np.ndarray],
    y_test: np.ndarray,
    *,
    gamma: np.ndarray,
    alpha: float,
    reliance_block: int,
    reliance_groups: list[np.ndarray],
) -> dict[str, Any]:
    train = finite_rows(y_train)
    test = finite_rows(y_test)
    fit = fit_multiband(train_blocks, y_train, train, gamma=gamma, alpha=alpha)
    selected_test_blocks = [block[test] for block in test_blocks]
    prediction = fit.predict(selected_test_blocks)
    full_sse = np.sum((y_test[test] - prediction) ** 2, axis=0)
    denominator = np.sum((y_test[test] - fit.y_mean[None, :]) ** 2, axis=0)
    delta = np.zeros((len(reliance_groups), y_test.shape[1]), dtype=np.float64)
    for group_index, columns in enumerate(reliance_groups):
        ablated = prediction - fit.block_column_contribution(
            selected_test_blocks, reliance_block, columns
        )
        delta[group_index] = np.sum((y_test[test] - ablated) ** 2, axis=0) - full_sse
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.full_like(full_sse, np.nan)
        np.divide(full_sse, denominator, out=ratio, where=denominator > 0)
        scaled_delta = np.full_like(delta, np.nan)
        np.divide(
            delta,
            denominator[None, :],
            out=scaled_delta,
            where=denominator[None, :] > 0,
        )
    return {
        "alpha": float(alpha),
        "gamma": np.asarray(gamma, dtype=np.float64),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "mean_r": float(np.nanmean(pearson_columns(y_test[test], prediction))),
        "mean_r2_oos": float(np.nanmean(1.0 - ratio)),
        "reliance": np.nanmean(scaled_delta, axis=1),
        "prediction": prediction,
        "test_positions": test,
        "x_means": fit.x_means,
        "x_scales": fit.x_scales,
        "y_mean": fit.y_mean,
        "y_scale": fit.y_scale,
    }


def select_alpha(
    x: np.ndarray,
    y: np.ndarray,
    blocks: np.ndarray,
    outer_block: int,
    alphas: np.ndarray,
    *,
    metric: str,
    purge_adjacent: int,
) -> tuple[float, np.ndarray]:
    n_time = len(blocks)
    valid_y = finite_rows(y)
    outer_test = np.flatnonzero(blocks == outer_block)
    outer_train = purged_training_positions(
        n_time, outer_test, purge_adjacent=purge_adjacent
    )
    outer_train = np.intersect1d(outer_train, valid_y, assume_unique=True)
    scores = np.zeros(len(alphas), dtype=float)
    used = 0
    for inner_block in sorted(np.unique(blocks).tolist()):
        if inner_block == outer_block:
            continue
        validation = np.intersect1d(
            np.flatnonzero(blocks == inner_block), outer_train, assume_unique=True
        )
        inner_train = purged_training_positions(
            n_time,
            validation,
            allowed_positions=outer_train,
            purge_adjacent=purge_adjacent,
        )
        inner_train = np.intersect1d(inner_train, valid_y, assume_unique=True)
        if validation.size < 3 or inner_train.size < 3:
            continue
        path = build_ridge_path(x, y, inner_train)
        for alpha_index, alpha in enumerate(alphas):
            fit = path.at(float(alpha))
            prediction = fit.predict(x[validation])
            scores[alpha_index] += mean_metric(
                y[validation], prediction, fit.y_mean, metric
            )
        used += 1
    if used == 0:
        raise RuntimeError(f"outer block {outer_block} has no usable inner folds")
    scores /= used
    best = np.flatnonzero(scores == np.nanmax(scores))
    return float(alphas[int(best[0])]), scores


def _fold_reliance(
    fit: RidgeFit,
    x_test: np.ndarray,
    y_test: np.ndarray,
    groups: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    prediction = fit.predict(x_test)
    full_sse = np.sum((y_test - prediction) ** 2, axis=0)
    denominator = np.sum((y_test - fit.y_mean[None, :]) ** 2, axis=0)
    delta_sse = np.zeros((len(groups), y_test.shape[1]), dtype=np.float64)
    for feature_index, columns in enumerate(groups):
        ablated = prediction - fit.contribution(x_test, columns)
        ablated_sse = np.sum((y_test - ablated) ** 2, axis=0)
        delta_sse[feature_index] = ablated_sse - full_sse
    return prediction, full_sse, denominator, delta_sse


def nested_subject_cv(
    x: np.ndarray,
    y: np.ndarray,
    blocks: np.ndarray,
    alphas: np.ndarray,
    *,
    selection_metric: str,
    purge_adjacent: int,
    groups: list[np.ndarray],
) -> dict[str, Any]:
    n_time, n_targets = y.shape
    oof = np.full((n_time, n_targets), np.nan, dtype=np.float64)
    total_sse = np.zeros(n_targets, dtype=float)
    total_denom = np.zeros(n_targets, dtype=float)
    total_delta = np.zeros((len(groups), n_targets), dtype=float)
    fold_reliance = []
    folds = []
    fold_scaling = []
    valid_y = finite_rows(y)
    for outer_block in sorted(np.unique(blocks).tolist()):
        test = np.intersect1d(
            np.flatnonzero(blocks == outer_block), valid_y, assume_unique=True
        )
        outer_test_full = np.flatnonzero(blocks == outer_block)
        train = purged_training_positions(
            n_time, outer_test_full, purge_adjacent=purge_adjacent
        )
        train = np.intersect1d(train, valid_y, assume_unique=True)
        alpha, inner_scores = select_alpha(
            x,
            y,
            blocks,
            outer_block,
            alphas,
            metric=selection_metric,
            purge_adjacent=purge_adjacent,
        )
        fit = fit_ridge(x, y, train, alpha)
        prediction, sse, denominator, delta = _fold_reliance(
            fit, x[test], y[test], groups
        )
        oof[test] = prediction
        total_sse += sse
        total_denom += denominator
        total_delta += delta
        with np.errstate(divide="ignore", invalid="ignore"):
            scaled_delta = np.full_like(delta, np.nan)
            np.divide(
                delta,
                denominator[None, :],
                out=scaled_delta,
                where=denominator[None, :] > 0,
            )
            per_feature = np.nanmean(scaled_delta, axis=1)
        fold_reliance.append(per_feature)
        fold_scaling.append(
            {
                "x_mean": fit.x_mean,
                "x_scale": fit.x_scale,
                "y_mean": fit.y_mean,
                "y_scale": fit.y_scale,
                "train_positions": train,
                "test_positions": test,
            }
        )
        folds.append(
            {
                "fold": int(outer_block),
                "alpha": alpha,
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                "mean_r": mean_metric(y[test], prediction, fit.y_mean, "pearson_r"),
                "mean_r2_oos": mean_metric(y[test], prediction, fit.y_mean, "r2_oos"),
                "inner_scores": inner_scores,
            }
        )
    valid_oof = finite_rows(oof)
    pooled_r = float(np.nanmean(pearson_columns(y[valid_oof], oof[valid_oof])))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.full_like(total_sse, np.nan)
        np.divide(total_sse, total_denom, out=ratio, where=total_denom > 0)
        pooled_r2_cols = 1.0 - ratio
        scaled_delta = np.full_like(total_delta, np.nan)
        np.divide(
            total_delta,
            total_denom[None, :],
            out=scaled_delta,
            where=total_denom[None, :] > 0,
        )
        reliance = np.nanmean(scaled_delta, axis=1)
    return {
        "selection_metric": selection_metric,
        "mean_fold_r": float(np.mean([fold["mean_r"] for fold in folds])),
        "pooled_oof_r": pooled_r,
        "pooled_r2_oos": float(np.nanmean(pooled_r2_cols)),
        "folds": folds,
        "oof": oof,
        "reliance": reliance,
        "fold_reliance": np.asarray(fold_reliance),
        "fold_scaling": fold_scaling,
    }


def modal_alpha(folds: list[dict[str, Any]]) -> float:
    values = np.asarray([float(fold["alpha"]) for fold in folds])
    unique, counts = np.unique(values, return_counts=True)
    return float(np.min(unique[counts == counts.max()]))


def fit_temporal_shift_null(
    x: np.ndarray,
    y: np.ndarray,
    shift: int,
    blocks: np.ndarray,
    alphas: np.ndarray,
    *,
    selection_metric: str,
    purge_adjacent: int,
    groups: list[np.ndarray],
) -> dict[str, Any]:
    """Shift the target first, then rerun the complete nested fit."""
    shifted_target = np.roll(np.asarray(y), int(shift), axis=0)
    return nested_subject_cv(
        x,
        shifted_target,
        blocks,
        alphas,
        selection_metric=selection_metric,
        purge_adjacent=purge_adjacent,
        groups=groups,
    )


def fit_transfer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    alpha: float,
    groups: list[np.ndarray],
) -> dict[str, Any]:
    train = finite_rows(y_train)
    test = finite_rows(y_test)
    fit = fit_ridge(x_train, y_train, train, alpha)
    prediction, sse, denominator, delta = _fold_reliance(
        fit, x_test[test], y_test[test], groups
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.full_like(sse, np.nan)
        np.divide(sse, denominator, out=ratio, where=denominator > 0)
        r2 = 1.0 - ratio
        scaled_delta = np.full_like(delta, np.nan)
        np.divide(
            delta,
            denominator[None, :],
            out=scaled_delta,
            where=denominator[None, :] > 0,
        )
        reliance = np.nanmean(scaled_delta, axis=1)
    return {
        "alpha": float(alpha),
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "mean_r": float(np.nanmean(pearson_columns(y_test[test], prediction))),
        "mean_r2_oos": float(np.nanmean(r2)),
        "reliance": reliance,
        "prediction": prediction,
        "test_positions": test,
        "x_mean": fit.x_mean,
        "x_scale": fit.x_scale,
        "y_mean": fit.y_mean,
        "y_scale": fit.y_scale,
    }


def fit_cross_story_shift_null(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    *,
    train_shift: int,
    test_shift: int,
    blocks: np.ndarray,
    alphas: np.ndarray,
    selection_metric: str,
    purge_adjacent: int,
    groups: list[np.ndarray],
) -> dict[str, Any]:
    """Retune on a shifted training target, then transfer to a shifted test target."""
    shifted_train = np.roll(np.asarray(y_train), int(train_shift), axis=0)
    shifted_test = np.roll(np.asarray(y_test), int(test_shift), axis=0)
    training_cv = nested_subject_cv(
        x_train,
        shifted_train,
        blocks,
        alphas,
        selection_metric=selection_metric,
        purge_adjacent=purge_adjacent,
        groups=groups,
    )
    result = fit_transfer(
        x_train,
        shifted_train,
        x_test,
        shifted_test,
        alpha=modal_alpha(training_cv["folds"]),
        groups=groups,
    )
    result["training_cv"] = training_cv
    return result


def scalar_pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return float("nan")
    ac = a[valid] - a[valid].mean()
    bc = b[valid] - b[valid].mean()
    denominator = math.sqrt(float(np.sum(ac * ac) * np.sum(bc * bc)))
    return float(np.sum(ac * bc) / denominator) if denominator > 0 else float("nan")


def scalar_spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return float("nan")
    return scalar_pearson(rankdata(a[valid]), rankdata(b[valid]))


def topk_jaccard(a: np.ndarray, b: np.ndarray, k: int) -> float:
    left = set(np.argsort(np.asarray(a))[-k:].tolist())
    right = set(np.argsort(np.asarray(b))[-k:].tolist())
    return float(len(left & right) / len(left | right))


def correspondence(a: np.ndarray, b: np.ndarray, top_k: int = 32) -> dict[str, float]:
    return {
        "spearman": scalar_spearman(a, b),
        "pearson": scalar_pearson(a, b),
        "topk_jaccard": topk_jaccard(a, b, top_k),
    }


def within_layer_identity_null(
    brain_profile: np.ndarray,
    lm_profile: np.ndarray,
    feature_names: np.ndarray,
    *,
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    def layer_key(name: Any) -> str:
        text = str(name)
        match = re.search(r"layer[_:]?(\d+)", text)
        if match:
            return f"layer_{int(match.group(1))}"
        match = re.match(r"l(\d+)[a-z_]", text)
        if match:
            return f"layer_{int(match.group(1))}"
        raise ValueError(f"cannot parse layer identity from feature name: {text}")

    layers = np.asarray([layer_key(name) for name in feature_names])
    indices = [np.flatnonzero(layers == layer) for layer in np.unique(layers)]
    rng = np.random.default_rng(seed)
    observed = scalar_spearman(brain_profile, lm_profile)
    null = np.empty(n_permutations, dtype=float)
    for permutation in range(n_permutations):
        shuffled = np.asarray(lm_profile).copy()
        for group in indices:
            shuffled[group] = shuffled[rng.permutation(group)]
        null[permutation] = scalar_spearman(brain_profile, shuffled)
    p_value = float((np.sum(null >= observed) + 1) / (len(null) + 1))
    return {
        "observed": observed,
        "p_value": p_value,
        "null_mean": float(np.mean(null)),
        "null_sd": float(np.std(null)),
        "null": null,
    }


def bootstrap_mean(values: np.ndarray, *, draws: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    samples = values[rng.integers(0, len(values), size=(draws, len(values)))].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "ci95": np.quantile(samples, [0.025, 0.975]),
        "draws": samples,
    }


def sign_flip_test(values: np.ndarray, *, draws: int, seed: int) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    observed = float(values.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice(np.asarray([-1.0, 1.0]), size=(draws, len(values)))
    null = np.mean(signs * values[None, :], axis=1)
    return {
        "mean": observed,
        "p_one_sided": float((np.sum(null >= observed) + 1) / (draws + 1)),
        "p_two_sided": float((np.sum(np.abs(null) >= abs(observed)) + 1) / (draws + 1)),
    }


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * len(p) / np.arange(1, len(p) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.clip(adjusted, 0.0, 1.0)
    return result


def position_plateau_basis(n_time: int, minimum: int, maximum: int) -> np.ndarray:
    time = np.arange(n_time, dtype=float)
    columns = []
    for plateau in range(int(minimum), int(maximum) + 1):
        columns.append(np.minimum(time, float(plateau - 1)))
    return np.column_stack(columns)


def _tsv_token_counts(path: Path, column: str) -> dict[int, int]:
    rows = [line.rstrip("\n").split("\t") for line in path.open(encoding="utf-8")]
    header = rows[0]
    column_index = header.index(column)
    tr_index = header.index("tr_index")
    counts: dict[int, int] = {}
    for row in rows[1:]:
        if len(row) <= max(column_index, tr_index):
            continue
        text = row[column_index].strip()
        counts[int(row[tr_index])] = 0 if not text else len(text.split())
    return counts


def _audio_envelope(
    path: Path,
    n_time: int,
    tr_seconds: float,
    stimulus_onset_seconds: float,
) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        sample_rate = handle.getframerate()
        channels = handle.getnchannels()
        width = handle.getsampwidth()
        raw = handle.readframes(handle.getnframes())
    dtype = {1: np.int8, 2: np.int16, 4: np.int32}[width]
    audio = np.frombuffer(raw, dtype=dtype).astype(np.float64)
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    audio /= max(float(np.max(np.abs(audio))), 1.0)
    rms = np.zeros(n_time, dtype=np.float64)
    for tr in range(n_time):
        start_seconds = tr * tr_seconds - stimulus_onset_seconds
        start = int(max(start_seconds, 0.0) * sample_rate)
        end = int(max(start_seconds + tr_seconds, 0.0) * sample_rate)
        if end > start and start < len(audio):
            segment = audio[start : min(end, len(audio))]
            if len(segment):
                rms[tr] = math.sqrt(float(np.mean(segment * segment)))
    return np.column_stack([rms, np.concatenate([[0.0], np.diff(rms)])])


def build_enhanced_baseline(
    spec: dict[str, Any],
    story: str,
    n_time: int,
    surprisal_mean: np.ndarray,
    surprisal_sum: np.ndarray,
    *,
    transcript_root: Path,
) -> dict[str, Any]:
    root = Path(transcript_root)
    story_root = root / story
    metadata = json.loads((story_root / "metadata.json").read_text(encoding="utf-8"))
    word_counts = _tsv_token_counts(story_root / f"{story}_tr_aligned.tsv", "text")
    phoneme_counts = _tsv_token_counts(
        story_root / f"{story}_phonemes_by_tr.tsv", "phonemes"
    )
    envelope = _audio_envelope(
        root / "audio" / f"{story}_audio.wav",
        n_time,
        float(metadata["tr_s"]),
        float(metadata.get("stimulus_onset_s", 0.0)),
    )
    low = np.column_stack(
        [
            np.asarray([word_counts.get(tr, 0) for tr in range(n_time)], dtype=float),
            np.asarray([phoneme_counts.get(tr, 0) for tr in range(n_time)], dtype=float),
            envelope,
            position_plateau_basis(
                n_time,
                int(spec["baselines"]["position_plateau_min_tr"]),
                int(spec["baselines"]["position_plateau_max_tr"]),
            ),
        ]
    )
    surprisal = np.column_stack([surprisal_mean, surprisal_sum]).astype(np.float64)
    return {
        "low_level_position": low,
        "surprisal": surprisal,
        "low_level_position_names": np.asarray(
            ["word_count", "phoneme_count", "envelope_rms", "envelope_delta"]
            + [
                f"position_plateau_{value}"
                for value in range(
                    int(spec["baselines"]["position_plateau_min_tr"]),
                    int(spec["baselines"]["position_plateau_max_tr"]) + 1,
                )
            ]
        ),
        "surprisal_names": np.asarray(["surprisal_mean", "surprisal_sum"]),
        "source_paths": [
            story_root / f"{story}_tr_aligned.tsv",
            story_root / f"{story}_phonemes_by_tr.tsv",
            story_root / "metadata.json",
            root / "audio" / f"{story}_audio.wav",
        ],
    }
