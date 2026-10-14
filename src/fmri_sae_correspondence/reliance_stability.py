#!/usr/bin/env python3
"""Estimate stability of reliance profiles from the current primary fits.

This is a diagnostic stage, not a model-fitting stage.  It reads only the
versioned ``artifacts/fit`` and ``artifacts/transfer`` outputs produced by the
source-story feature-bank analyses.  In particular, it never falls back to
the historical evaluation-story bank.  Profiles are compared only when the
feature coordinates are identical for the profiles being compared:

* BOLD participant profiles are comparable within a fit fold, and in a
  transfer direction (the transfer bank is fixed).
* LM fold profiles are comparable only when all outer folds use the exact
  same feature-name vector.  Fold-specific source banks are therefore
  reported as not estimable rather than being compared by array position.

The output includes participant-pair and split-half Spearman stability, LM
fold Spearman stability when estimable, deterministic bootstrap intervals,
source hashes, and the feature-coordinate fingerprints used for every
comparison.  Planned incomplete rows are recorded and skipped; a row marked
complete that fails validation aborts before writing an aggregate result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
MODEL_KEYS = (
    "gemma_2_2b", "gemma_2_9b", "llama_3_1_8b",
    "gemma_3_1b", "gemma_3_4b", "gemma_3_12b",
)
STORIES = ("shapesphysical", "shapessocial")
FIT_VARIANTS = ("source_story_variance", "source_story_prevalence")
CACHE_SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(_jsonable(value), indent=2, sort_keys=True) + "\n"
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _write_text_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **arrays)
        with open(tmp_name, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _finite_pairwise_spearman(profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pairwise Spearman values and indices for a profile matrix."""
    profiles = np.asarray(profiles, dtype=np.float64)
    if profiles.ndim != 2:
        raise ValueError("profiles must be a two-dimensional matrix")
    values: list[float] = []
    pairs: list[tuple[int, int]] = []
    for i in range(profiles.shape[0]):
        for j in range(i + 1, profiles.shape[0]):
            valid = np.isfinite(profiles[i]) & np.isfinite(profiles[j])
            if int(valid.sum()) < 3:
                continue
            left = rankdata(profiles[i, valid], method="average")
            right = rankdata(profiles[j, valid], method="average")
            left -= left.mean()
            right -= right.mean()
            denom = float(np.linalg.norm(left) * np.linalg.norm(right))
            if denom > 0:
                values.append(float(np.dot(left, right) / denom))
                pairs.append((i, j))
    return np.asarray(values, dtype=np.float64), np.asarray(pairs, dtype=np.int32).reshape(-1, 2)


def _row_rank_unit(profiles: np.ndarray) -> np.ndarray:
    """Center/rank rows and normalize, preserving invalid rows as NaN."""
    profiles = np.asarray(profiles, dtype=np.float64)
    ranked = np.full_like(profiles, np.nan, dtype=np.float64)
    for i, row in enumerate(profiles):
        valid = np.isfinite(row)
        if valid.sum() >= 3:
            vals = rankdata(row[valid], method="average")
            vals -= vals.mean()
            norm = float(np.linalg.norm(vals))
            if norm > 0:
                ranked[i, valid] = vals / norm
    return ranked


def bootstrap_pairwise_mean(
    profiles: np.ndarray, *, draws: int, seed: int
) -> np.ndarray:
    """Profile-resampling bootstrap of mean pairwise Spearman stability."""
    profiles = np.asarray(profiles, dtype=np.float64)
    result = np.full(int(draws), np.nan, dtype=np.float64)
    if profiles.ndim != 2 or profiles.shape[0] < 2:
        return result
    if not np.isfinite(profiles).all():
        rng = np.random.default_rng(seed)
        for d in range(int(draws)):
            idx = rng.integers(0, profiles.shape[0], profiles.shape[0])
            vals, _ = _finite_pairwise_spearman(profiles[idx])
            if vals.size:
                result[d] = float(vals.mean())
        return result
    unit = _row_rank_unit(profiles)
    rng = np.random.default_rng(seed)
    n = profiles.shape[0]
    # For unit rank rows, the average over all unordered pairs is obtained
    # from the squared norm of the row sum.  This avoids a correlation matrix
    # per draw while retaining the exact bootstrap statistic.
    for start in range(0, int(draws), 256):
        stop = min(start + 256, int(draws))
        idx = rng.integers(0, n, size=(stop - start, n))
        sampled = unit[idx]
        sums = np.sum(sampled, axis=1)
        result[start:stop] = (np.sum(sums * sums, axis=1) - n) / (n * (n - 1))
    return result


def bootstrap_split_half_mean(
    profiles: np.ndarray, *, draws: int, seed: int
) -> np.ndarray:
    """Bootstrap correlations between two participant-half mean profiles."""
    profiles = np.asarray(profiles, dtype=np.float64)
    result = np.full(int(draws), np.nan, dtype=np.float64)
    if profiles.ndim != 2 or profiles.shape[0] < 4:
        return result
    if not np.isfinite(profiles).all():
        return result
    rng = np.random.default_rng(seed)
    n = profiles.shape[0]
    left_n = n // 2
    # Draw balanced random splits in chunks.  The row-wise rank transform is
    # equivalent to Pearson correlation of the two half means.
    for start in range(0, int(draws), 128):
        stop = min(start + 128, int(draws))
        perms = np.asarray([rng.permutation(n) for _ in range(stop - start)])
        left = profiles[perms[:, :left_n]].mean(axis=1)
        right = profiles[perms[:, left_n:]].mean(axis=1)
        left_rank = rankdata(left, axis=1, method="average")
        right_rank = rankdata(right, axis=1, method="average")
        left_rank -= left_rank.mean(axis=1, keepdims=True)
        right_rank -= right_rank.mean(axis=1, keepdims=True)
        denom = np.linalg.norm(left_rank, axis=1) * np.linalg.norm(right_rank, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            result[start:stop] = np.divide(
                np.sum(left_rank * right_rank, axis=1), denom,
                out=np.full(stop - start, np.nan), where=denom > 0,
            )
    return result


def _ci(values: np.ndarray, *, estimate: float | None = None) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "estimate": float(np.mean(finite)) if estimate is None and finite.size else (
            float(estimate) if estimate is not None else float("nan")
        ),
        "ci95_low": float(np.quantile(finite, 0.025)) if finite.size else float("nan"),
        "ci95_high": float(np.quantile(finite, 0.975)) if finite.size else float("nan"),
        "valid_draws": int(finite.size),
    }


def summarize_profiles(
    profiles: np.ndarray, *, draws: int, seed: int, unit: str,
    feature_names: Iterable[str],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Summarize one comparable profile collection."""
    profiles = np.asarray(profiles, dtype=np.float64)
    if profiles.ndim != 2:
        raise ValueError("profile collection must have shape n_profiles x n_features")
    names = np.asarray(list(feature_names), dtype=str)
    if names.shape != (profiles.shape[1],):
        raise ValueError("feature-name count does not match profile width")
    pairwise, pair_indices = _finite_pairwise_spearman(profiles)
    pair_boot = bootstrap_pairwise_mean(profiles, draws=draws, seed=seed)
    split_boot = bootstrap_split_half_mean(profiles, draws=draws, seed=seed + 1)
    split_valid = split_boot[np.isfinite(split_boot)]
    split_estimate = float(np.mean(split_valid)) if split_valid.size else float("nan")
    summary = {
        "status": "complete" if profiles.shape[0] >= 2 and pairwise.size else "not_estimable",
        "profile_unit": unit,
        "profile_count": int(profiles.shape[0]),
        "feature_count": int(profiles.shape[1]),
        "feature_name_sha256": hashlib.sha256("\n".join(names.tolist()).encode()).hexdigest(),
        "pair_count": int(pairwise.size),
        "pairwise_spearman": _ci(pair_boot, estimate=float(np.mean(pairwise)) if pairwise.size else None),
        "split_half_spearman": {
            **_ci(split_boot, estimate=split_estimate),
            "minimum_profiles": 4,
        },
        "bootstrap": {"draws": int(draws), "seed_pairwise": int(seed), "seed_split_half": int(seed + 1)},
    }
    arrays = {
        "profiles": profiles.astype(np.float32),
        "feature_names": names,
        "pairwise_spearman": pairwise.astype(np.float32),
        "pair_indices": pair_indices.astype(np.int32),
        "pairwise_bootstrap": pair_boot.astype(np.float32),
        "split_half_bootstrap": split_boot.astype(np.float32),
    }
    return summary, arrays


def _validate_summary(summary_path: Path, *, expected_role: str) -> dict[str, Any]:
    if not summary_path.is_file():
        raise FileNotFoundError(f"missing summary for arrays: {summary_path}")
    try:
        summary = json.loads(summary_path.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid summary JSON: {summary_path}") from exc
    if summary.get("analysis_role") != expected_role:
        raise ValueError(
            f"unexpected analysis_role in {summary_path}: {summary.get('analysis_role')!r}"
        )
    if summary.get("analysis_status") not in (None, "complete"):
        raise ValueError(f"source summary is not complete: {summary_path}")
    return summary


def _validate_arrays(path: Path, *, transfer: bool) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    required = ({"subject_ids", "brain_r_reliance", "brain_r2_reliance", "feature_names"}
                if transfer else {
                    "subject_ids", "fold_feature_names", "brain_r_reliance_by_fold",
                    "brain_r2_reliance_by_fold", "lm_r_reliance_by_fold", "lm_r2_reliance_by_fold",
                })
    # NPZ members are lazy.  Load only the reliance diagnostics needed here,
    # not the much larger OOF prediction tensors stored in the same artifact.
    with np.load(path, allow_pickle=False) as loaded:
        missing = sorted(required - set(loaded.files))
        if missing:
            raise ValueError(f"{path} missing required stability keys: {missing}")
        arrays = {key: np.asarray(loaded[key]) for key in sorted(required)}
    if transfer:
        names = arrays["feature_names"].astype(str)
        for key in ("brain_r_reliance", "brain_r2_reliance"):
            if arrays[key].ndim != 2 or arrays[key].shape[1] != names.size:
                raise ValueError(f"{path}: {key} shape is incompatible with feature_names")
        if arrays["subject_ids"].ndim != 1 or arrays["brain_r_reliance"].shape[0] != arrays["subject_ids"].size:
            raise ValueError(f"{path}: participant axis is inconsistent")
    else:
        names = arrays["fold_feature_names"].astype(str)
        if names.ndim != 2:
            raise ValueError(f"{path}: fold_feature_names must be fold x feature")
        folds = names.shape[0]
        for key in ("brain_r_reliance_by_fold", "brain_r2_reliance_by_fold"):
            if arrays[key].ndim != 3 or arrays[key].shape[:2] != (folds, arrays["subject_ids"].size) or arrays[key].shape[2] != names.shape[1]:
                raise ValueError(f"{path}: {key} shape is incompatible with fold_feature_names")
        for key in ("lm_r_reliance_by_fold", "lm_r2_reliance_by_fold"):
            if arrays[key].shape != (folds, names.shape[1]):
                raise ValueError(f"{path}: {key} shape is incompatible with fold_feature_names")
    return arrays


def _condition_key_from_fit(path: Path) -> tuple[str, str, str, int] | None:
    # .../fit/{model}_{story}/{variant}/n{K}/arrays.npz
    if len(path.parts) < 4:
        return None
    try:
        model_story, variant, n_name = path.parent.parent.parent.name, path.parent.parent.name, path.parent.name
        model, story = next((m, s) for m in MODEL_KEYS for s in STORIES if model_story == f"{m}_{s}")
        if not n_name.startswith("n") or not n_name[1:].isdigit():
            return None
        return model, story, variant, int(n_name[1:])
    except StopIteration:
        return None


def _condition_key_from_transfer(path: Path) -> tuple[str, str, str, str, int] | None:
    # .../transfer/{model}_{train}_to_{test}/{variant}/n{K}/arrays.npz
    model_story, variant, n_name = path.parent.parent.parent.name, path.parent.parent.name, path.parent.name
    for model in MODEL_KEYS:
        prefix = model + "_"
        if model_story.startswith(prefix) and "_to_" in model_story[len(prefix):]:
            train, test = model_story[len(prefix):].split("_to_", 1)
            if train in STORIES and test in STORIES and n_name.startswith("n") and n_name[1:].isdigit():
                return model, train, test, variant, int(n_name[1:])
    return None


def _discover(root: Path, *, transfer: bool) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Discover complete fit/transfer rows from the condition aggregate.

    The per-stage manifests are progress reports and can be stale while
    analyses are active. The aggregate is the authoritative condition
    inventory. Only the requested stage is inspected; missing
    null, control, or grouped rows do not block this diagnostic.
    """
    manifest = root / "aggregate_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"required aggregate condition manifest missing: {manifest}")
    try:
        manifest_data = json.loads(manifest.read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid aggregate condition manifest: {manifest}") from exc
    if manifest_data.get("manifest_role") != "correspondence_condition_aggregate":
        raise ValueError(f"unexpected manifest role in {manifest}: {manifest_data.get('manifest_role')!r}")
    manifest_sha256 = sha256_file(manifest)
    requested_stage = "transfer" if transfer else "fit"
    requested_role = "observed_cross_story" if transfer else "observed_within_story"
    rows = [row for row in manifest_data.get("conditions", []) if row.get("stage") == requested_stage]
    if not rows:
        raise ValueError(f"aggregate manifest contains no {requested_stage} condition rows")
    found: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for condition in rows:
        key_text = condition.get("key")
        if condition.get("status") != "complete" or condition.get("problems"):
            # Preserve planned but incomplete rows without fabricating a
            # profile or aborting the usable subset.
            skipped.append({
                "key": key_text,
                "stage": requested_stage,
                "status": condition.get("status"),
                "problems": condition.get("problems", []),
                "path": condition.get("path"),
                "reason": "incomplete_or_manifest_flagged_problem",
            })
            continue
        if condition.get("role") != requested_role:
            raise ValueError(f"unexpected {requested_stage} condition role for {key_text}: {condition.get('role')!r}")
        summary_path = Path(str(condition.get("path", "")))
        if not summary_path.is_file():
            raise FileNotFoundError(f"manifest summary path is missing for {key_text}: {summary_path}")
        recorded_summary_hash = condition.get("summary_sha256")
        actual_summary_hash = sha256_file(summary_path)
        if not recorded_summary_hash or recorded_summary_hash != actual_summary_hash:
            raise ValueError(f"summary hash mismatch for {key_text}: {summary_path}")
        arrays_path = summary_path.with_name("arrays.npz")
        key = _condition_key_from_transfer(arrays_path) if transfer else _condition_key_from_fit(arrays_path)
        if key is None:
            raise ValueError(f"unrecognized artifact condition path: {arrays_path}")
        expected_key = (
            (condition.get("model_key"), condition.get("train_story"), condition.get("test_story"), condition.get("variant"), condition.get("size"))
            if transfer else
            (condition.get("model_key"), condition.get("story"), condition.get("variant"), condition.get("size"))
        )
        if key != expected_key:
            raise ValueError(f"condition key/path mismatch for {key_text}: {key!r} != {expected_key!r}")
        summary = _validate_summary(summary_path, expected_role=(
            "correspondence_cross_story_source_bank_transfer" if transfer else "correspondence_within_story_observed_fit"
        ))
        if summary.get("model_key") != condition.get("model_key") or summary.get("bank_size") != condition.get("size") or summary.get("feature_bank_variant") != condition.get("variant"):
            raise ValueError(f"summary metadata mismatch for {key_text}")
        _validate_arrays(arrays_path, transfer=transfer)
        found.append({
            "arrays_path": arrays_path, "summary_path": summary_path, "summary": summary,
            "key": key, "manifest_path": manifest, "manifest_record": condition,
            # Hash each immutable source exactly once.
            "arrays_sha256": sha256_file(arrays_path),
            "summary_sha256": actual_summary_hash,
            "manifest_sha256": manifest_sha256,
        })
    return found, skipped


def _write_condition(
    out_root: Path, record: dict[str, Any], *, draws: int, seed: int, transfer: bool,
) -> list[dict[str, Any]]:
    model = record["key"][0]
    variant = record["key"][-2] if transfer else record["key"][2]
    size = record["key"][-1]
    arrays = _validate_arrays(record["arrays_path"], transfer=transfer)
    results: list[dict[str, Any]] = []
    if transfer:
        train, test = record["key"][1:3]
        direction = f"{train}_to_{test}"
        roles = [("brain", "r", arrays["brain_r_reliance"]), ("brain", "r2", arrays["brain_r2_reliance"])]
        names = arrays["feature_names"].astype(str)
        lm_roles = [("lm", "r", None), ("lm", "r2", None)]
    else:
        story = record["key"][1]
        direction = "within_story"
        names_by_fold = arrays["fold_feature_names"].astype(str)
        roles = []
        for role, metric, key in (("brain", "r", "brain_r_reliance_by_fold"), ("brain", "r2", "brain_r2_reliance_by_fold")):
            profiles = arrays[key]
            for fold in range(profiles.shape[0]):
                fold_summary, fold_arrays = summarize_profiles(
                    profiles[fold], draws=draws, seed=seed + fold * 100,
                    unit="participant", feature_names=names_by_fold[fold],
                )
                cond = _save_result(out_root, model, story, direction, variant, size, role, metric, fold, fold_summary, fold_arrays, record, transfer)
                results.append(cond)
            roles.append((role, metric, profiles))
        # LM fold stability is not valid for fold-varying feature identities.
        lm_names_equal = all(np.array_equal(names_by_fold[0], x) for x in names_by_fold[1:])
        for metric, key in (("r", "lm_r_reliance_by_fold"), ("r2", "lm_r2_reliance_by_fold")):
            if lm_names_equal:
                lm_summary, lm_arrays = summarize_profiles(
                    arrays[key], draws=draws, seed=seed + 900,
                    unit="temporal_fold", feature_names=names_by_fold[0],
                )
                results.append(_save_result(out_root, model, story, direction, variant, size, "lm", metric, "all_folds", lm_summary, lm_arrays, record, transfer))
            else:
                results.append(_save_not_estimable(out_root, model, story, direction, variant, size, "lm", metric, "all_folds", "fold-specific feature identities differ; exact-coordinate fold stability is not estimable", record, transfer))
        return results
    for role, metric, profiles in roles:
        # Transfer has one fixed profile per participant; exact coordinates are
        # shared by both targets and therefore participant stability is valid.
        stable_summary, stable_arrays = summarize_profiles(
            profiles, draws=draws, seed=seed, unit="participant", feature_names=names,
        )
        results.append(_save_result(out_root, model, direction, direction, variant, size, role, metric, "all_participants", stable_summary, stable_arrays, record, transfer, train=train, test=test))
    for _, metric, _ in lm_roles:
        results.append(_save_not_estimable(out_root, model, direction, direction, variant, size, "lm", metric, "all_folds", "transfer artifact has a single LM reliance profile and no fold axis", record, transfer, train=train, test=test))
    return results


def _condition_cache_path(out_root: Path, record: dict[str, Any], *, transfer: bool) -> Path:
    regime = "transfer" if transfer else "fit"
    key = "__".join(str(part).replace("/", "_") for part in record["key"])
    return out_root / ".condition_cache" / f"{regime}__{key}.json"


def _result_key(
    model: str, story: str, direction: str, variant: str, size: int,
    role: str, metric: str, axis: str, *, transfer: bool,
) -> str:
    """Return a collision-free storage key without changing analysis identity."""
    evaluation = direction if transfer else f"{story}_{direction}"
    return f"{model}_{evaluation}_{variant}_n{size}_{role}_{metric}_{axis}"


def _condition_fingerprint(
    record: dict[str, Any], *, draws: int, seed: int, transfer: bool,
    calculator_sha256: str,
) -> str:
    return _stable_hash({
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "calculator_sha256": calculator_sha256,
        "source_arrays_sha256": record["arrays_sha256"],
        "source_summary_sha256": record["summary_sha256"],
        "source_aggregate_manifest_sha256": record["manifest_sha256"],
        "source_analysis_fingerprint": record["summary"].get("analysis_fingerprint"),
        "source_condition_key": record["manifest_record"].get("key"),
        "parsed_condition_key": record["key"],
        "bootstrap_draws": int(draws),
        "seed": int(seed),
        "transfer": bool(transfer),
    })


def _validate_cached_row(
    row: dict[str, Any], *, condition_fingerprint: str,
) -> dict[str, Any]:
    summary_path = Path(str(row.get("summary_path", "")))
    expected_summary_hash = row.get("summary_sha256")
    if not summary_path.is_file() or not expected_summary_hash:
        raise ValueError(f"cached stability summary is missing: {summary_path}")
    actual_summary_hash = sha256_file(summary_path)
    if actual_summary_hash != expected_summary_hash:
        raise ValueError(f"cached stability summary hash mismatch: {summary_path}")
    summary = json.loads(summary_path.read_text())
    if summary.get("condition_fingerprint") != condition_fingerprint:
        raise ValueError(f"stale stability cache fingerprint: {summary_path}")
    derived = summary.get("derived_arrays")
    if derived:
        derived_path = Path(str(derived))
        expected_derived_hash = summary.get("derived_arrays_sha256")
        if not derived_path.is_file() or sha256_file(derived_path) != expected_derived_hash:
            raise ValueError(f"cached stability arrays hash mismatch: {derived_path}")
    summary["summary_sha256"] = actual_summary_hash
    return summary


def _write_condition_cached(
    out_root: Path,
    record: dict[str, Any],
    *,
    draws: int,
    seed: int,
    transfer: bool,
    calculator_sha256: str,
) -> list[dict[str, Any]]:
    """Run one independently seeded condition, or reuse its exact cache."""
    if sha256_file(Path(__file__).resolve()) != calculator_sha256:
        raise ValueError("stability calculator changed after task fingerprinting")
    for path_key, hash_key in (
        ("arrays_path", "arrays_sha256"),
        ("summary_path", "summary_sha256"),
    ):
        source_path = Path(record[path_key])
        if not source_path.is_file() or sha256_file(source_path) != record[hash_key]:
            raise ValueError(f"source changed after validation: {source_path}")
    fingerprint = _condition_fingerprint(
        record, draws=draws, seed=seed, transfer=transfer,
        calculator_sha256=calculator_sha256,
    )
    record = {**record, "condition_fingerprint": fingerprint}
    cache_path = _condition_cache_path(out_root, record, transfer=transfer)
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        if cache.get("cache_schema_version") != CACHE_SCHEMA_VERSION:
            raise ValueError(f"unsupported stability cache schema: {cache_path}")
        if cache.get("condition_fingerprint") != fingerprint:
            raise ValueError(f"stale stability condition cache: {cache_path}")
        return [
            _validate_cached_row(row, condition_fingerprint=fingerprint)
            for row in cache.get("results", [])
        ]

    rows = _write_condition(out_root, record, draws=draws, seed=seed, transfer=transfer)
    cached_rows = []
    for row in rows:
        key = _result_key(
            row["model_key"], row.get("story", row["direction"]), row["direction"],
            row["feature_bank_variant"], row["bank_size"], row["profile_role"],
            row["metric"], row["axis"], transfer=row["evaluation_regime"] == "cross_story_transfer",
        )
        cached_rows.append({
            "summary_path": str(out_root / key / "summary.json"),
            "summary_sha256": row["summary_sha256"],
        })
    _write_json(cache_path, {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "condition_fingerprint": fingerprint,
        "source_condition_key": record["manifest_record"].get("key"),
        "results": cached_rows,
    })
    return rows


def _reuse_existing_result(path: Path, *, condition_fingerprint: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    if not path.is_dir():
        raise ValueError(f"stability output path is not a directory: {path}")
    summary_path = path / "summary.json"
    if not summary_path.is_file():
        raise ValueError(f"partial stability output exists without summary: {path}")
    row = json.loads(summary_path.read_text())
    if row.get("condition_fingerprint") != condition_fingerprint:
        raise ValueError(f"stale stability output exists: {path}")
    derived = row.get("derived_arrays")
    if derived:
        derived_path = Path(str(derived))
        if not derived_path.is_file() or sha256_file(derived_path) != row.get("derived_arrays_sha256"):
            raise ValueError(f"stability output arrays are incomplete or modified: {path}")
    row["summary_sha256"] = sha256_file(summary_path)
    return row


def _save_not_estimable(out_root: Path, model: str, story: str, direction: str, variant: str, size: int, role: str, metric: str, axis: str, reason: str, record: dict[str, Any], transfer: bool, **extra: Any) -> dict[str, Any]:
    key = _result_key(model, story, direction, variant, size, role, metric, axis, transfer=transfer)
    path = out_root / key
    fingerprint = record["condition_fingerprint"]
    reused = _reuse_existing_result(path, condition_fingerprint=fingerprint)
    if reused is not None:
        return reused
    row = {
        "schema_version": 1, "artifact_role": "primary_reliance_profile_stability",
        "analysis_status": "not_estimable", "model_key": model,
        "evaluation_regime": "cross_story_transfer" if transfer else "within_story",
        "direction": direction, "feature_bank_variant": variant, "bank_size": int(size),
        "profile_role": role, "metric": metric, "axis": axis, "reason": reason,
        **extra, "condition_fingerprint": fingerprint,
        "source_arrays": str(record["arrays_path"]), "source_arrays_sha256": record["arrays_sha256"],
        "source_summary": str(record["summary_path"]), "source_summary_sha256": record["summary_sha256"],
        "source_analysis_fingerprint": record["summary"].get("analysis_fingerprint"),
        "source_aggregate_manifest": str(record["manifest_path"]),
        "source_aggregate_manifest_sha256": record["manifest_sha256"],
        "source_condition_key": record["manifest_record"].get("key"),
    }
    if not transfer:
        row["story"] = story
    out_root.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=out_root))
    try:
        _write_json(tmp_path / "summary.json", row)
        os.replace(tmp_path, path)
    except BaseException:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise
    row["summary_sha256"] = sha256_file(path / "summary.json")
    return row


def _save_result(out_root: Path, model: str, story: str, direction: str, variant: str, size: int, role: str, metric: str, axis: str, result: dict[str, Any], arrays: dict[str, np.ndarray], record: dict[str, Any], transfer: bool, **extra: Any) -> dict[str, Any]:
    key = _result_key(model, story, direction, variant, size, role, metric, axis, transfer=transfer)
    path = out_root / key
    fingerprint = record["condition_fingerprint"]
    reused = _reuse_existing_result(path, condition_fingerprint=fingerprint)
    if reused is not None:
        return reused
    out_root.mkdir(parents=True, exist_ok=True)
    tmp_path = Path(tempfile.mkdtemp(prefix=f".{key}.", dir=out_root))
    array_path = path / "arrays.npz"
    tmp_array_path = tmp_path / "arrays.npz"
    _write_npz_atomic(tmp_array_path, arrays)
    row = {
        "schema_version": 1, "artifact_role": "primary_reliance_profile_stability",
        "analysis_status": result.get("status", "complete"), "model_key": model,
        "evaluation_regime": "cross_story_transfer" if transfer else "within_story",
        "direction": direction, "feature_bank_variant": variant, "bank_size": int(size),
        "profile_role": role, "metric": metric, "axis": axis,
        "profile_summary": result, **extra,
        "condition_fingerprint": fingerprint,
        "source_arrays": str(record["arrays_path"]), "source_arrays_sha256": record["arrays_sha256"],
        "source_summary": str(record["summary_path"]), "source_summary_sha256": record["summary_sha256"],
        "source_analysis_fingerprint": record["summary"].get("analysis_fingerprint"),
        "source_aggregate_manifest": str(record["manifest_path"]),
        "source_aggregate_manifest_sha256": record["manifest_sha256"],
        "source_condition_key": record["manifest_record"].get("key"),
        "derived_arrays": str(array_path), "derived_arrays_sha256": sha256_file(tmp_array_path),
        "coordinate_policy": "Spearman comparisons use exact feature names; no array-position matching across fold-specific banks",
    }
    if not transfer:
        row["story"] = story
    try:
        _write_json(tmp_path / "summary.json", row)
        os.replace(tmp_path, path)
    except BaseException:
        shutil.rmtree(tmp_path, ignore_errors=True)
        raise
    row["summary_sha256"] = sha256_file(path / "summary.json")
    return row


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args(argv)
    if args.bootstrap_draws < 100:
        raise ValueError("at least 100 bootstrap draws are required")
    artifact_root = args.artifact_root.resolve()
    out_root = args.output_root.resolve()
    # Validate both trees before writing any derived files: a malformed
    # transfer cannot leave behind a seemingly complete mixed aggregate.
    fit, skipped_fit = _discover(artifact_root, transfer=False)
    transfer, skipped_transfer = _discover(artifact_root, transfer=True)
    if not fit and not transfer:
        raise ValueError("aggregate manifest has no complete fit or transfer conditions to analyze")
    calculator_path = Path(__file__).resolve()
    calculator_sha256 = sha256_file(calculator_path)
    source_manifest_path = artifact_root / "aggregate_manifest.json"
    source_manifest_sha256 = sha256_file(source_manifest_path)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate([*fit, *transfer]):
        is_transfer = index >= len(fit)
        rows.extend(
            _write_condition_cached(
                out_root,
                record,
                draws=args.bootstrap_draws,
                seed=args.seed + index * 1000,
                transfer=is_transfer,
                calculator_sha256=calculator_sha256,
            )
        )
    if sha256_file(source_manifest_path) != source_manifest_sha256:
        raise ValueError("source aggregate manifest changed during stability calculation")
    checksums: list[str] = []
    for path in sorted(out_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksums.append(f"{sha256_file(path)}  {path.relative_to(out_root)}")
    out_root.mkdir(parents=True, exist_ok=True)
    _write_text_atomic(out_root / "SHA256SUMS.txt", "\n".join(checksums) + "\n")
    aggregate = {
        "schema_version": 1,
        "artifact_role": "primary_reliance_profile_stability_aggregate",
        "analysis_status": "complete" if not (skipped_fit or skipped_transfer) else "partial_inputs",
        "calculator": "reliance_stability.py",
        "calculator_sha256": calculator_sha256,
        "parameters": {"bootstrap_draws": int(args.bootstrap_draws), "seed": int(args.seed), "metric": "Spearman correlation across exact SAE feature coordinates"},
        "source_manifest": "aggregate_manifest.json",
        "source_manifest_sha256": source_manifest_sha256,
        "conditions": rows,
        "skipped_incomplete_conditions": skipped_fit + skipped_transfer,
        "coverage": {
            "source_fit_conditions": len(fit), "transfer_conditions": len(transfer),
            "participant_profile_results": sum(r["profile_role"] == "brain" and r["analysis_status"] == "complete" for r in rows),
            "lm_fold_results": sum(r["profile_role"] == "lm" and r["analysis_status"] == "complete" for r in rows),
            "not_estimable": sum(r["analysis_status"] == "not_estimable" for r in rows),
            "skipped_incomplete_conditions": len(skipped_fit) + len(skipped_transfer),
        },
        "interpretation_boundary": "Stability of fitted reliance profiles is a reliability diagnostic. It is not a BOLD acquisition noise ceiling, a cross-system correspondence test, or evidence of a shared biological mechanism.",
    }
    _write_json(out_root / "aggregate.json", aggregate)
    # Recompute after aggregate is written so it is covered by the manifest.
    checksums = [f"{sha256_file(p)}  {p.relative_to(out_root)}" for p in sorted(out_root.rglob("*")) if p.is_file() and p.name != "SHA256SUMS.txt"]
    _write_text_atomic(out_root / "SHA256SUMS.txt", "\n".join(checksums) + "\n")
    print(f"Wrote {len(rows)} primary reliance stability records to {out_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
