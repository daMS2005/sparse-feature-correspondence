#!/usr/bin/env python3
"""Compute BOLD shared-response reliability diagnostics.

This is a post-confirmatory, outcome-independent diagnostic.  It reads the
already-cleaned parcel target bundles, uses their saved censor masks, and
never reads language-model targets, feature weights, or prediction results.

The two reported families are deliberately separated:

* leave-one-participant-out ISC asks whether an individual's uncensored time
  series follows the mean time series of the other participants; and
* split-half reliability asks whether two randomly formed participant groups
  produce similar parcel time series.

The script retains the full parcelwise arrays so downstream manuscript tables
and figures can be regenerated without recomputing the analysis.  It also
records the input hashes and calculation parameters in a sidecar JSON.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n")


def percentile_ci(values: np.ndarray, *, draws: int, rng: np.random.Generator) -> dict[str, float]:
    """Bootstrap a scalar mean, median, or another already-computed statistic.

    This helper is intentionally used on participant-level or split-draw-level
    summaries, not on the individual parcel values as if parcels were
    independent observations.
    """
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"estimate": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    indices = rng.integers(0, values.size, size=(int(draws), values.size))
    sampled = values[indices]
    means = np.mean(sampled, axis=1)
    return {
        "estimate": float(np.mean(values)),
        "ci95_low": float(np.quantile(means, 0.025)),
        "ci95_high": float(np.quantile(means, 0.975)),
    }


def bootstrap_stat(
    values: np.ndarray,
    statistic: str,
    *,
    draws: int,
    seed: int,
) -> dict[str, float]:
    """Bootstrap a participant/draw-level mean or median with a fixed seed."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "estimate": float("nan"),
            "ci95_low": float("nan"),
            "ci95_high": float("nan"),
            "draws": int(draws),
        }
    rng = np.random.default_rng(seed)
    # Chunking bounds temporary memory when the source vector is later made
    # larger (for example, if more participants are added).
    result = np.empty(int(draws), dtype=np.float64)
    chunk = 512
    for start in range(0, int(draws), chunk):
        stop = min(start + chunk, int(draws))
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        sampled = values[indices]
        if statistic == "mean":
            result[start:stop] = np.mean(sampled, axis=1)
        elif statistic == "median":
            result[start:stop] = np.median(sampled, axis=1)
        else:
            raise ValueError(f"unsupported bootstrap statistic: {statistic}")
    return {
        "estimate": float(np.mean(values) if statistic == "mean" else np.median(values)),
        "ci95_low": float(np.quantile(result, 0.025)),
        "ci95_high": float(np.quantile(result, 0.975)),
        "draws": int(draws),
        "seed": int(seed),
        "statistic": statistic,
    }


def _correlation_vectors(
    left: np.ndarray,
    right: np.ndarray,
    *,
    min_valid: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pearson correlation for columns with independent finite masks.

    ``left`` and ``right`` are T x P.  Censored values are represented as
    non-finite entries before this function is called.  The function returns
    a P-vector of correlations and the corresponding number of valid time
    points, making the censor intersection explicit in the artifact.
    """
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("left and right must have equal T x P shapes")
    valid = np.isfinite(left) & np.isfinite(right)
    count = valid.sum(axis=0).astype(np.int32)
    safe_left = np.where(valid, left, 0.0)
    safe_right = np.where(valid, right, 0.0)
    denominator_count = np.maximum(count, 1)
    mean_left = safe_left.sum(axis=0) / denominator_count
    mean_right = safe_right.sum(axis=0) / denominator_count
    centered_left = np.where(valid, left - mean_left, 0.0)
    centered_right = np.where(valid, right - mean_right, 0.0)
    covariance = np.sum(centered_left * centered_right, axis=0)
    variance_left = np.sum(centered_left * centered_left, axis=0)
    variance_right = np.sum(centered_right * centered_right, axis=0)
    denominator = np.sqrt(variance_left * variance_right)
    result = np.full(left.shape[1], np.nan, dtype=np.float64)
    keep = (count >= int(min_valid)) & (denominator > 0)
    result[keep] = covariance[keep] / denominator[keep]
    return result, count


def _load_grid(path: Path, story: str) -> dict[str, Any]:
    """Load and validate one target bundle into subject x TR x parcel form."""
    required = {
        "values",
        "subject_ids",
        "run_ids",
        "tr_indices",
        "target_names",
        "censor_mask",
    }
    with np.load(path, allow_pickle=False) as bundle:
        missing = required - set(bundle.files)
        if missing:
            raise ValueError(f"{path} is missing required keys: {sorted(missing)}")
        values = np.asarray(bundle["values"], dtype=np.float64)
        subject_ids_rows = np.asarray(bundle["subject_ids"], dtype=str)
        run_ids_rows = np.asarray(bundle["run_ids"], dtype=str)
        tr_indices_rows = np.asarray(bundle["tr_indices"], dtype=np.int64)
        target_names = np.asarray(bundle["target_names"], dtype=str)
        censor_rows = np.asarray(bundle["censor_mask"], dtype=bool)
        fd_rows = (
            np.asarray(bundle["framewise_displacement"], dtype=np.float64)
            if "framewise_displacement" in bundle.files
            else None
        )
        dvars_rows = (
            np.asarray(bundle["std_dvars"], dtype=np.float64)
            if "std_dvars" in bundle.files
            else None
        )

    if values.ndim != 2:
        raise ValueError("values must be a row x parcel matrix")
    n_rows, n_parcels = values.shape
    arrays = (subject_ids_rows, run_ids_rows, tr_indices_rows, censor_rows)
    if any(len(a) != n_rows for a in arrays):
        raise ValueError("row metadata lengths do not match values")
    if len(target_names) != n_parcels:
        raise ValueError("target_names length does not match parcel count")

    subjects = np.asarray(sorted(np.unique(subject_ids_rows).tolist()), dtype=str)
    runs_per_subject = {
        subject: sorted(np.unique(run_ids_rows[subject_ids_rows == subject]).tolist())
        for subject in subjects
    }
    if any(len(runs) != 1 for runs in runs_per_subject.values()):
        raise ValueError(
            "ISC implementation requires one run per participant; observed "
            f"multiple runs for {[(s, r) for s, r in runs_per_subject.items() if len(r) != 1]}"
        )
    trs = np.asarray(sorted(np.unique(tr_indices_rows).tolist()), dtype=np.int64)
    expected_trs = np.arange(int(trs.min()), int(trs.max()) + 1, dtype=np.int64)
    if not np.array_equal(trs, expected_trs):
        raise ValueError("TR indices must form a complete integer grid")
    if not np.array_equal(trs, np.arange(len(trs), dtype=np.int64)):
        raise ValueError("TR indices must start at zero for this target bundle")

    subject_lookup = {subject: i for i, subject in enumerate(subjects.tolist())}
    tr_lookup = {int(tr): i for i, tr in enumerate(trs.tolist())}
    grid = np.full((len(subjects), len(trs), n_parcels), np.nan, dtype=np.float64)
    valid = np.zeros((len(subjects), len(trs)), dtype=bool)
    seen: set[tuple[str, str, int]] = set()
    for row in range(n_rows):
        key = (str(subject_ids_rows[row]), str(run_ids_rows[row]), int(tr_indices_rows[row]))
        if key in seen:
            raise ValueError(f"duplicate participant/run/TR key: {key}")
        seen.add(key)
        subject_index = subject_lookup[str(subject_ids_rows[row])]
        tr_index = tr_lookup[int(tr_indices_rows[row])]
        row_finite = bool(np.all(np.isfinite(values[row])))
        row_valid = (not bool(censor_rows[row])) and row_finite
        if row_valid:
            grid[subject_index, tr_index] = values[row]
            valid[subject_index, tr_index] = True

    expected_rows = len(subjects) * len(trs)
    if len(seen) != expected_rows:
        raise ValueError(
            f"target does not contain a rectangular grid: {len(seen)} keys, "
            f"expected {expected_rows}"
        )

    return {
        "story": story,
        "path": path,
        "values": grid,
        "valid": valid,
        "subjects": subjects,
        "trs": trs,
        "target_names": target_names,
        "censor_rows": censor_rows,
        "fd_rows": fd_rows,
        "dvars_rows": dvars_rows,
        "n_rows": n_rows,
        "n_parcels": n_parcels,
        "runs_per_subject": runs_per_subject,
    }


def _loo_isc(data: dict[str, Any], *, min_valid_tr: int, min_other_participants: int) -> dict[str, Any]:
    values = np.asarray(data["values"], dtype=np.float64)
    n_subjects, n_time, n_parcels = values.shape
    finite_values = np.isfinite(values)
    total_sum = np.nansum(values, axis=0)
    total_count = np.sum(finite_values, axis=0).astype(np.int32)
    isc = np.full((n_subjects, n_parcels), np.nan, dtype=np.float64)
    valid_counts = np.zeros((n_subjects, n_parcels), dtype=np.int32)
    for subject_index in range(n_subjects):
        own = values[subject_index]
        own_finite = np.isfinite(own)
        own_clean = np.where(own_finite, own, 0.0)
        other_sum = total_sum - own_clean
        other_count = total_count - own_finite.astype(np.int32)
        other_mean = np.divide(
            other_sum,
            other_count,
            out=np.full_like(other_sum, np.nan),
            where=other_count >= int(min_other_participants),
        )
        isc[subject_index], valid_counts[subject_index] = _correlation_vectors(
            own,
            other_mean,
            min_valid=min_valid_tr,
        )

    participant_mean = np.nanmean(isc, axis=1)
    participant_median = np.nanmedian(isc, axis=1)
    parcel_mean = np.nanmean(isc, axis=0)
    parcel_median = np.nanmedian(isc, axis=0)
    all_values = isc[np.isfinite(isc)]
    return {
        "isc": isc,
        "valid_counts": valid_counts,
        "participant_mean": participant_mean,
        "participant_median": participant_median,
        "parcel_mean": parcel_mean,
        "parcel_median": parcel_median,
        "all_mean": float(np.mean(all_values)) if all_values.size else float("nan"),
        "all_median": float(np.median(all_values)) if all_values.size else float("nan"),
        "n_valid_entries": int(all_values.size),
    }


def _make_split_assignments(n_subjects: int, draws: int, seed: int) -> np.ndarray:
    if n_subjects < 4:
        raise ValueError("split-half reliability requires at least four participants")
    rng = np.random.default_rng(seed)
    assignments = np.zeros((int(draws), int(n_subjects)), dtype=np.uint8)
    half = n_subjects // 2
    for draw in range(int(draws)):
        chosen = rng.permutation(n_subjects)[:half]
        assignments[draw, chosen] = 1
    return assignments


def _batched_correlations(
    left: np.ndarray,
    right: np.ndarray,
    *,
    min_valid_tr: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Correlate B group pairs, each T x P, in one vectorized operation."""
    if left.shape != right.shape or left.ndim != 3:
        raise ValueError("batched inputs must have equal B x T x P shapes")
    valid = np.isfinite(left) & np.isfinite(right)
    count = valid.sum(axis=1).astype(np.int16)
    safe_left = np.where(valid, left, 0.0)
    safe_right = np.where(valid, right, 0.0)
    denominator_count = np.maximum(count, 1)
    mean_left = safe_left.sum(axis=1) / denominator_count
    mean_right = safe_right.sum(axis=1) / denominator_count
    centered_left = np.where(valid, left - mean_left[:, None, :], 0.0)
    centered_right = np.where(valid, right - mean_right[:, None, :], 0.0)
    covariance = np.sum(centered_left * centered_right, axis=1)
    variance_left = np.sum(centered_left * centered_left, axis=1)
    variance_right = np.sum(centered_right * centered_right, axis=1)
    denominator = np.sqrt(variance_left * variance_right)
    result = np.full((left.shape[0], left.shape[2]), np.nan, dtype=np.float64)
    keep = (count >= int(min_valid_tr)) & (denominator > 0)
    result[keep] = (covariance / denominator)[keep]
    return result, count


def _split_half(data: dict[str, Any], *, draws: int, seed: int, min_valid_tr: int) -> dict[str, Any]:
    values = np.asarray(data["values"], dtype=np.float64)
    n_subjects, n_time, n_parcels = values.shape
    assignments = _make_split_assignments(n_subjects, draws, seed)
    finite = np.isfinite(values)
    clean = np.where(finite, values, 0.0)
    total_sum = clean.sum(axis=0)
    total_valid_tr_count = finite.any(axis=2).sum(axis=0).astype(np.int16)
    split_r = np.full((int(draws), n_parcels), np.nan, dtype=np.float32)
    split_valid_counts = np.zeros((int(draws), n_parcels), dtype=np.int16)
    # 64 draws use about 32 MB for each group tensor at 200 parcels x 303 TRs.
    # Chunking also keeps this safe if the atlas or number of participants grows.
    chunk_size = 64
    for start in range(0, int(draws), chunk_size):
        stop = min(start + chunk_size, int(draws))
        a = assignments[start:stop].astype(np.float64)
        group_a_sum = np.einsum("bs,stp->btp", a, clean, optimize=True)
        group_a_count = np.einsum("bs,st->bt", a, finite.any(axis=2), optimize=True).astype(np.int16)
        group_b_sum = total_sum[None, :, :] - group_a_sum
        group_b_count = total_valid_tr_count[None, :] - group_a_count
        group_a = np.divide(
            group_a_sum,
            group_a_count[:, :, None],
            out=np.full_like(group_a_sum, np.nan),
            where=group_a_count[:, :, None] > 0,
        )
        group_b = np.divide(
            group_b_sum,
            group_b_count[:, :, None],
            out=np.full_like(group_b_sum, np.nan),
            where=group_b_count[:, :, None] > 0,
        )
        corr, counts = _batched_correlations(group_a, group_b, min_valid_tr=min_valid_tr)
        split_r[start:stop] = corr.astype(np.float32)
        split_valid_counts[start:stop] = counts.astype(np.int16)
    draw_mean = np.nanmean(split_r, axis=1).astype(np.float32)
    draw_median = np.nanmedian(split_r, axis=1).astype(np.float32)
    all_values = split_r[np.isfinite(split_r)]
    raw_median = float(np.median(all_values)) if all_values.size else float("nan")
    raw_mean = float(np.mean(all_values)) if all_values.size else float("nan")
    return {
        "assignments": assignments,
        "r": split_r,
        "valid_counts": split_valid_counts,
        "draw_mean": draw_mean,
        "draw_median": draw_median,
        "all_mean": raw_mean,
        "all_median": raw_median,
        "draw_mean_median": float(np.median(draw_mean)),
        "draw_median_median": float(np.median(draw_median)),
        "draw_mean_ci": bootstrap_stat(draw_mean, "median", draws=10000, seed=seed + 101),
        "draw_median_ci": bootstrap_stat(draw_median, "median", draws=10000, seed=seed + 102),
        "draw_mean_mean_ci": bootstrap_stat(draw_mean, "mean", draws=10000, seed=seed + 103),
        "spearman_brown_from_all_median": (
            float((2.0 * raw_median) / (1.0 + raw_median))
            if np.isfinite(raw_median) and raw_median > -1.0
            else float("nan")
        ),
        "spearman_brown_from_draw_mean_median": (
            float((2.0 * float(np.median(draw_mean))) / (1.0 + float(np.median(draw_mean))))
            if np.isfinite(np.median(draw_mean)) and np.median(draw_mean) > -1.0
            else float("nan")
        ),
        "spearman_brown_from_draw_median_median": (
            float((2.0 * float(np.median(draw_median))) / (1.0 + float(np.median(draw_median))))
            if np.isfinite(np.median(draw_median)) and np.median(draw_median) > -1.0
            else float("nan")
        ),
    }


def _network_name(label: str) -> str | None:
    parts = str(label).split("_")
    # Target names are e.g. 7Networks_LH_Vis_1.  Only use the inherited
    # Schaefer network label; do not imply a separately measured Yeo atlas.
    if len(parts) >= 4 and parts[0] == "7Networks" and parts[1] in {"LH", "RH"}:
        return parts[2]
    return None


def _network_summary(
    target_names: np.ndarray,
    loo: dict[str, Any],
    split: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = {}
    for index, label in enumerate(target_names.tolist()):
        name = _network_name(str(label))
        if name is not None:
            grouped.setdefault(name, []).append(index)
    rows: list[dict[str, Any]] = []
    for network, indices in sorted(grouped.items()):
        index_array = np.asarray(indices, dtype=int)
        loo_parcel = np.asarray(loo["parcel_mean"])[index_array]
        split_parcel = np.asarray(split["r"], dtype=np.float64)[:, index_array]
        split_values = split_parcel[np.isfinite(split_parcel)]
        rows.append(
            {
                "network": network,
                "parcel_count": int(len(indices)),
                "loo_parcel_mean": float(np.nanmean(loo_parcel)),
                "loo_parcel_median": float(np.nanmedian(loo_parcel)),
                "split_half_mean": float(np.mean(split_values)) if split_values.size else float("nan"),
                "split_half_median": float(np.median(split_values)) if split_values.size else float("nan"),
                "label_source": "Schaefer2018 7Networks label encoded in target_names",
            }
        )
    return rows


def analyze_story(
    story: str,
    source_path: Path,
    output_root: Path,
    *,
    split_draws: int,
    bootstrap_draws: int,
    seed: int,
    min_valid_tr: int,
    min_other_participants: int,
) -> dict[str, Any]:
    data = _load_grid(source_path, story)
    loo = _loo_isc(
        data,
        min_valid_tr=min_valid_tr,
        min_other_participants=min_other_participants,
    )
    split = _split_half(
        data,
        draws=split_draws,
        seed=seed,
        min_valid_tr=min_valid_tr,
    )
    participant_bootstrap_mean = bootstrap_stat(
        loo["participant_mean"], "mean", draws=bootstrap_draws, seed=seed + 201
    )
    participant_bootstrap_median = bootstrap_stat(
        loo["participant_median"], "median", draws=bootstrap_draws, seed=seed + 202
    )
    parcel_bootstrap_mean = bootstrap_stat(
        loo["parcel_mean"], "mean", draws=bootstrap_draws, seed=seed + 203
    )
    parcel_bootstrap_median = bootstrap_stat(
        loo["parcel_median"], "median", draws=bootstrap_draws, seed=seed + 204
    )

    source_summary_path = source_path.with_suffix(".summary.json")
    source_summary = None
    if source_summary_path.exists():
        try:
            source_summary = json.loads(source_summary_path.read_text())
        except json.JSONDecodeError:
            source_summary = {"read_error": "invalid JSON"}
    source_rel = source_path.name
    artifact_dir = output_root / story
    artifact_dir.mkdir(parents=True, exist_ok=True)
    arrays_path = artifact_dir / "arrays.npz"
    np.savez_compressed(
        arrays_path,
        subject_ids=data["subjects"].astype(str),
        tr_indices=data["trs"].astype(np.int64),
        target_names=data["target_names"].astype(str),
        subject_valid_mask=data["valid"].astype(bool),
        subject_censor_mask=(~data["valid"]).astype(bool),
        loo_isc=loo["isc"].astype(np.float32),
        loo_valid_counts=loo["valid_counts"].astype(np.int16),
        loo_participant_mean=loo["participant_mean"].astype(np.float32),
        loo_participant_median=loo["participant_median"].astype(np.float32),
        loo_parcel_mean=loo["parcel_mean"].astype(np.float32),
        loo_parcel_median=loo["parcel_median"].astype(np.float32),
        split_half_assignment=split["assignments"].astype(np.uint8),
        split_half_r=split["r"].astype(np.float32),
        split_half_valid_counts=split["valid_counts"].astype(np.int16),
        split_half_draw_mean=split["draw_mean"].astype(np.float32),
        split_half_draw_median=split["draw_median"].astype(np.float32),
    )
    source_censor_count = int(np.sum(~data["valid"]))
    subject_rows = []
    for index, subject in enumerate(data["subjects"].tolist()):
        subject_rows.append(
            {
                "subject_id": str(subject),
                "run_ids": data["runs_per_subject"][str(subject)],
                "total_trs": int(data["valid"].shape[1]),
                "censored_trs": int(np.sum(~data["valid"][index])),
                "valid_trs": int(np.sum(data["valid"][index])),
                "loo_isc_mean": float(loo["participant_mean"][index]),
                "loo_isc_median": float(loo["participant_median"][index]),
            }
        )
    finite_loo_counts = loo["valid_counts"][np.isfinite(loo["isc"])]
    finite_split_counts = split["valid_counts"][np.isfinite(split["r"])]
    split_ci = {
        "raw_all_parcel_r": {
            "mean": split["all_mean"],
            "median": split["all_median"],
            "q025": float(np.quantile(all_values, 0.025)) if (all_values := split["r"][np.isfinite(split["r"])]) .size else float("nan"),
            "q975": float(np.quantile(all_values, 0.975)) if all_values.size else float("nan"),
        },
        "draw_mean": {
            "median": float(np.median(split["draw_mean"])),
            "mean": float(np.mean(split["draw_mean"])),
            "q025": float(np.quantile(split["draw_mean"], 0.025)),
            "q975": float(np.quantile(split["draw_mean"], 0.975)),
        },
        "draw_median": {
            "median": float(np.median(split["draw_median"])),
            "mean": float(np.mean(split["draw_median"])),
            "q025": float(np.quantile(split["draw_median"], 0.025)),
            "q975": float(np.quantile(split["draw_median"], 0.975)),
        },
    }
    summary = {
        "schema_version": 1,
        "artifact_role": "correspondence_bold_shared_response_reliability",
        "analysis_status": "complete",
        "analysis_scope": "post-confirmatory descriptive reliability/estimability diagnostic",
        "story": story,
        "source_path": source_rel,
        "source_sha256": sha256_file(source_path),
        "source_summary_path": source_summary_path.name if source_summary_path.exists() else None,
        "source_summary": source_summary,
        "parameters": {
            "split_half_draws": int(split_draws),
            "bootstrap_draws": int(bootstrap_draws),
            "seed": int(seed),
            "min_valid_tr": int(min_valid_tr),
            "min_other_participants_for_loo_mean": int(min_other_participants),
            "censor_semantics": "censor_mask=true is excluded; values remain in source bundle but are never used",
            "loo_mean_semantics": "each participant correlated with the mean of the other participants at each TR using available uncensored participants",
            "split_half_semantics": "random half-group parcel means correlated across TRs using the intersection of group-available TRs",
            "spearman_brown_semantics": "2r/(1+r) applied to the chosen raw split-half correlation summary; descriptive, not a single-subject ceiling",
        },
        "software": {
            "calculator": "isc_reliability.py",
            "calculator_sha256": sha256_file(Path(__file__).resolve()),
        },
        "dimensions": {
            "subjects": int(len(data["subjects"])),
            "time_points": int(len(data["trs"])),
            "parcels": int(data["n_parcels"]),
            "source_rows": int(data["n_rows"]),
        },
        "validity_counts": {
            "source_censored_rows": source_censor_count,
            "source_uncensored_rows": int(data["valid"].sum()),
            "source_censor_fraction": float(source_censor_count / data["valid"].size),
            "per_subject": subject_rows,
            "loo_valid_tr_summary": {
                "minimum": int(np.min(finite_loo_counts)) if finite_loo_counts.size else 0,
                "median": float(np.median(finite_loo_counts)) if finite_loo_counts.size else float("nan"),
                "maximum": int(np.max(finite_loo_counts)) if finite_loo_counts.size else 0,
                "entries_meeting_minimum": int(np.sum(finite_loo_counts >= min_valid_tr)),
            },
            "split_half_valid_tr_summary": {
                "minimum": int(np.min(finite_split_counts)) if finite_split_counts.size else 0,
                "median": float(np.median(finite_split_counts)) if finite_split_counts.size else float("nan"),
                "maximum": int(np.max(finite_split_counts)) if finite_split_counts.size else 0,
                "entries_meeting_minimum": int(np.sum(finite_split_counts >= min_valid_tr)),
            },
        },
        "loo_isc": {
            "all_parcel_entries": {
                "mean": loo["all_mean"],
                "median": loo["all_median"],
                "n": int(loo["n_valid_entries"]),
            },
            "participant_mean_summary": {
                "mean": float(np.nanmean(loo["participant_mean"])),
                "median": float(np.nanmedian(loo["participant_mean"])),
                "q025": float(np.nanquantile(loo["participant_mean"], 0.025)),
                "q975": float(np.nanquantile(loo["participant_mean"], 0.975)),
                "bootstrap_mean_ci95": participant_bootstrap_mean,
            },
            "participant_median_summary": {
                "mean": float(np.nanmean(loo["participant_median"])),
                "median": float(np.nanmedian(loo["participant_median"])),
                "q025": float(np.nanquantile(loo["participant_median"], 0.025)),
                "q975": float(np.nanquantile(loo["participant_median"], 0.975)),
                "bootstrap_median_ci95": participant_bootstrap_median,
            },
            "parcel_mean_summary": {
                "mean": float(np.nanmean(loo["parcel_mean"])),
                "median": float(np.nanmedian(loo["parcel_mean"])),
                "q025": float(np.nanquantile(loo["parcel_mean"], 0.025)),
                "q975": float(np.nanquantile(loo["parcel_mean"], 0.975)),
                "bootstrap_mean_ci95": parcel_bootstrap_mean,
            },
            "parcel_median_summary": {
                "mean": float(np.nanmean(loo["parcel_median"])),
                "median": float(np.nanmedian(loo["parcel_median"])),
                "q025": float(np.nanquantile(loo["parcel_median"], 0.025)),
                "q975": float(np.nanquantile(loo["parcel_median"], 0.975)),
                "bootstrap_median_ci95": parcel_bootstrap_median,
            },
        },
        "split_half": {
            "draws": int(split_draws),
            "group_sizes": [int(len(data["subjects"]) // 2), int(len(data["subjects"]) - len(data["subjects"]) // 2)],
            "summary": split_ci,
            "spearman_brown_from_all_parcel_median": split["spearman_brown_from_all_median"],
            "spearman_brown_from_median_draw_mean": split["spearman_brown_from_draw_mean_median"],
            "spearman_brown_from_median_draw_median": split["spearman_brown_from_draw_median_median"],
            "bootstrap_ci95": {
                "median_of_draw_means": split["draw_mean_ci"],
                "median_of_draw_medians": split["draw_median_ci"],
                "mean_of_draw_means": split["draw_mean_mean_ci"],
            },
        },
        "network_summary": _network_summary(data["target_names"], loo, split),
        "network_summary_note": "Network rows are descriptive summaries of the Schaefer 7Networks labels; no independent Yeo or ROI analysis was performed.",
        "arrays_path": "arrays.npz",
        "arrays_sha256": sha256_file(arrays_path),
    }
    write_json(artifact_dir / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--physical", type=Path, required=True)
    parser.add_argument("--social", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split-draws", type=int, default=10000)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--min-valid-tr", type=int, default=100)
    parser.add_argument("--min-other-participants", type=int, default=2)
    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    all_summaries = []
    for story, path in (("shapesphysical", args.physical), ("shapessocial", args.social)):
        path = path.resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        print(f"Analyzing {story}: {path}", flush=True)
        all_summaries.append(
            analyze_story(
                story,
                path,
                output_root,
                split_draws=args.split_draws,
                bootstrap_draws=args.bootstrap_draws,
                seed=args.seed + (0 if story == "shapesphysical" else 1),
                min_valid_tr=args.min_valid_tr,
                min_other_participants=args.min_other_participants,
            )
        )
    aggregate = {
        "schema_version": 1,
        "artifact_role": "correspondence_bold_reliability_aggregate",
        "analysis_status": "complete",
        "calculator": "isc_reliability.py",
        "calculator_sha256": sha256_file(Path(__file__).resolve()),
        "parameters": {
            "split_half_draws": int(args.split_draws),
            "bootstrap_draws": int(args.bootstrap_draws),
            "seed": int(args.seed),
            "min_valid_tr": int(args.min_valid_tr),
            "min_other_participants": int(args.min_other_participants),
        },
        "stories": [
            {
                "story": item["story"],
                "summary_path": f"{item['story']}/summary.json",
                "summary_sha256": sha256_file(output_root / item["story"] / "summary.json"),
                "arrays_path": item["arrays_path"],
                "arrays_sha256": item["arrays_sha256"],
                "subjects": item["dimensions"]["subjects"],
                "time_points": item["dimensions"]["time_points"],
                "parcels": item["dimensions"]["parcels"],
                "loo_all_median": item["loo_isc"]["all_parcel_entries"]["median"],
                "loo_participant_mean_median": item["loo_isc"]["participant_mean_summary"]["median"],
                "split_half_all_median": item["split_half"]["summary"]["raw_all_parcel_r"]["median"],
                "split_half_spearman_brown": item["split_half"]["spearman_brown_from_median_draw_mean"],
            }
            for item in all_summaries
        ],
        "interpretation": (
            "These quantities diagnose shared stimulus-locked BOLD structure and the "
            "stability of the acquisition-level response. They are not exact single-subject "
            "noise ceilings, are not reliability-corrected model scores, and do not establish "
            "feature-level correspondence."
        ),
    }
    write_json(output_root / "aggregate.json", aggregate)
    checksums = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksums.append(f"{sha256_file(path)}  {path.relative_to(output_root)}")
    (output_root / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")
    print(f"Wrote reliability artifacts to {output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
