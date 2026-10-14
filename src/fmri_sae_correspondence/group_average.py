#!/usr/bin/env python3
"""Participant-averaged BOLD sensitivity with a fully refitted temporal null.

This module is intentionally separate from the participant-specific primary
analysis.  It averages the already-cleaned, run-wise standardized parcel
targets across uncensored participants at each TR, then repeats the current
purged temporal analysis with the fold-specific source-story variance bank.
The result estimates temporal prediction of a participant-average response;
it is not held-out-participant generalization and is never confirmatory.
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
    _artifact_root,
    _extract_bank,
    _load_bank,
    bank_path,
    evaluate_fit,
    global_finite_training_rows,
    load_condition,
    select_alpha_on_outer_train,
)
from .group_average_manifest import materialize_group_average_manifest  # noqa: E402
from .grouped_reliance import _cluster_groups, _expand_brain_groups  # noqa: E402
from .integrity import (  # noqa: E402
    assert_cache_fingerprint,
    atomic_write_json,
    provenance_fingerprint,
    sha256_file,
)
from .modeling import (  # noqa: E402
    correspondence,
    feature_groups,
    fit_ridge,
    lag_expand,
    within_layer_identity_null,
)
from .nulls import _p_value, fixed_shift_offsets  # noqa: E402

HERE = Path(__file__).resolve().parent

PRIMARY_SIZE = 256
PRIMARY_THRESHOLD = 0.80
MINIMUM_PARTICIPANT_FRACTION = 0.80


def _seed(base: int, *parts: Any) -> int:
    payload = "|".join([str(int(base)), *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little", signed=False)


def build_group_mean_target(
    subject_targets: dict[str, np.ndarray],
    *,
    minimum_participant_fraction: float = MINIMUM_PARTICIPANT_FRACTION,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Average finite participant targets per TR and retain well-sampled TRs."""

    subject_ids = np.asarray(sorted(subject_targets), dtype=str)
    if len(subject_ids) < 2:
        raise ValueError("group-average sensitivity requires at least two participants")
    stack = np.asarray([subject_targets[str(subject)] for subject in subject_ids], dtype=np.float64)
    if stack.ndim != 3:
        raise ValueError("participant targets must form [participant,time,parcel]")
    row_valid = np.all(np.isfinite(stack), axis=2)
    counts = np.sum(row_valid, axis=0).astype(np.int64)
    minimum = int(np.ceil(float(minimum_participant_fraction) * len(subject_ids)))
    with np.errstate(invalid="ignore"):
        target = np.nanmean(stack, axis=0)
    target[counts < minimum] = np.nan
    valid = counts >= minimum
    if not np.any(valid):
        raise RuntimeError("no TR meets the minimum participant fraction")
    if not np.all(np.isfinite(target[valid])):
        raise RuntimeError("retained group-average target contains nonfinite values")
    return target, counts, subject_ids


def _corr(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    n = min(len(a), len(b))
    if n < 3:
        return {"spearman": float("nan"), "pearson": float("nan"), "topk_jaccard": float("nan")}
    return correspondence(
        np.asarray(a, dtype=np.float64),
        np.asarray(b, dtype=np.float64),
        top_k=max(1, min(32, int(round(n * 0.125)))),
    )


def _fit_fold(
    x: np.ndarray,
    y: np.ndarray,
    outer_train: np.ndarray,
    outer_test: np.ndarray,
    groups: Sequence[np.ndarray],
    *,
    alphas: np.ndarray,
    metric: str,
    n_time: int,
    inner_folds: int,
    purge: int,
) -> tuple[Any, dict[str, Any], float, np.ndarray]:
    alpha, grid, _ = select_alpha_on_outer_train(
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
    return fit, evaluation, float(alpha), np.asarray(grid, dtype=np.float64)


def run_condition(
    config_path: Path,
    spec: dict[str, Any],
    model: str,
    story: str,
    *,
    n_shifts: int,
    force: bool = False,
) -> dict[str, Any]:
    condition = load_condition(config_path, spec, model, story)
    n_time = len(condition["tr_indices"])
    n_folds = int(spec["cross_validation"]["outer_folds"])
    inner_folds = int(spec["cross_validation"]["inner_folds"])
    purge = int(spec["cross_validation"]["purge_adjacent_trs"])
    alphas = np.asarray(spec["cross_validation"]["alpha_grid"], dtype=np.float64)
    lags = tuple(int(value) for value in spec["targets"]["brain"]["lags"])
    fold_rows = contiguous_fold_rows(n_time, n_folds, purge_adjacent_trs=purge)
    bank_paths = [bank_path(config_path, spec, model, story, fold=fold) for fold in range(n_folds)]
    output_dir = _artifact_root(config_path, spec) / "group_average" / model / story / "source_story_variance" / f"n{PRIMARY_SIZE}"
    summary_path = output_dir / "summary.json"
    arrays_path = output_dir / "arrays.npz"
    offsets = fixed_shift_offsets(
        n_time,
        int(n_shifts),
        int(spec["nulls"]["temporal_target"]["minimum_shift_trs"]),
        _seed(int(spec["nulls"]["seeds"]["temporal_target"]), "group_average", model, story),
    )
    source_paths = [condition["predictor_path"], condition["brain_path"], *bank_paths]
    code_paths = [
        HERE / "group_average.py",
        HERE / "fitting.py",
        HERE / "grouped_reliance.py",
        HERE / "nulls.py",
        HERE / "feature_banks.py",
        HERE / "modeling.py",
    ]
    fingerprint = provenance_fingerprint(
        config_path,
        input_paths=source_paths,
        code_paths=code_paths,
        extra={
            "analysis": "correspondence_group_average_sensitivity",
            "model": model,
            "story": story,
            "bank": "source_story_variance",
            "size": PRIMARY_SIZE,
            "group_threshold": PRIMARY_THRESHOLD,
            "minimum_participant_fraction": MINIMUM_PARTICIPANT_FRACTION,
            "n_shifts": int(n_shifts),
            "offsets": offsets.tolist(),
        },
    )
    if not force:
        cached = assert_cache_fingerprint(summary_path, fingerprint)
        if cached is not None:
            if not arrays_path.is_file():
                raise RuntimeError(f"group-average cache exists without arrays: {summary_path}")
            return cached

    target, participant_counts, subject_ids = build_group_mean_target(condition["subject_targets"])
    conditional_brain = np.full((n_folds, PRIMARY_SIZE), np.nan, dtype=np.float32)
    conditional_lm = np.full_like(conditional_brain, np.nan)
    observed_corr = np.full((n_folds, 3), np.nan, dtype=np.float64)
    identity_null = np.full((n_folds, int(spec["nulls"]["feature_identity"]["permutations"])), np.nan, dtype=np.float32)
    grouped_brain: list[np.ndarray] = []
    grouped_lm: list[np.ndarray] = []
    grouped_corr = np.full((n_folds, 3), np.nan, dtype=np.float64)
    group_counts = np.zeros(n_folds, dtype=np.int64)
    group_metadata: list[list[dict[str, Any]]] = []
    brain_r_metrics = np.full((n_folds, 2), np.nan, dtype=np.float64)
    brain_r2_metrics = np.full((n_folds, 2), np.nan, dtype=np.float64)
    lm_r2_metrics = np.full((n_folds, 2), np.nan, dtype=np.float64)
    selected = {"brain_r": [], "brain_r2": [], "lm_r2": []}
    fold_x: list[np.ndarray] = []
    fold_names: list[np.ndarray] = []
    fold_groups: list[list[np.ndarray]] = []
    started = time.time()

    for fold, (outer_train, outer_test) in enumerate(fold_rows):
        bank_record, bank = _load_bank(bank_paths[fold], "source_story_variance", PRIMARY_SIZE)
        if not np.array_equal(np.asarray(bank_record.get("selection_rows", []), dtype=np.int64), outer_train):
            raise RuntimeError(f"{bank_paths[fold]}: selection rows differ from purged training rows")
        x_base, names = _extract_bank(condition, bank, PRIMARY_SIZE)
        x_brain = lag_expand(x_base, lags)
        conditional_brain_groups = feature_groups(PRIMARY_SIZE, len(lags))
        conditional_lm_groups = feature_groups(PRIMARY_SIZE, 1)
        clusters, metadata, _ = _cluster_groups(
            bank_record,
            size=PRIMARY_SIZE,
            threshold=PRIMARY_THRESHOLD,
        )
        brain_cluster_groups = _expand_brain_groups(clusters, len(lags))

        _, brain_r_eval, alpha, _ = _fit_fold(
            x_brain, target, outer_train, outer_test, conditional_brain_groups,
            alphas=alphas, metric="pearson_r", n_time=n_time, inner_folds=inner_folds, purge=purge,
        )
        selected["brain_r"].append(alpha)
        brain_r_metrics[fold] = [brain_r_eval["mean_r"], brain_r_eval["mean_r2_oos"]]

        brain_fit, brain_r2_eval, alpha, _ = _fit_fold(
            x_brain, target, outer_train, outer_test, conditional_brain_groups,
            alphas=alphas, metric="r2_oos", n_time=n_time, inner_folds=inner_folds, purge=purge,
        )
        selected["brain_r2"].append(alpha)
        brain_r2_metrics[fold] = [brain_r2_eval["mean_r"], brain_r2_eval["mean_r2_oos"]]
        conditional_brain[fold] = np.asarray(brain_r2_eval["reliance"], dtype=np.float32)
        brain_group_eval = evaluate_fit(brain_fit, x_brain[outer_test], target[outer_test], brain_cluster_groups)
        grouped_brain.append(np.asarray(brain_group_eval["reliance"], dtype=np.float64))

        lm_fit, lm_r2_eval, alpha, _ = _fit_fold(
            x_base, condition["lm_target"], outer_train, outer_test, conditional_lm_groups,
            alphas=alphas, metric="r2_oos", n_time=n_time, inner_folds=inner_folds, purge=purge,
        )
        selected["lm_r2"].append(alpha)
        lm_r2_metrics[fold] = [lm_r2_eval["mean_r"], lm_r2_eval["mean_r2_oos"]]
        conditional_lm[fold] = np.asarray(lm_r2_eval["reliance"], dtype=np.float32)
        lm_group_eval = evaluate_fit(lm_fit, x_base[outer_test], condition["lm_target"][outer_test], clusters)
        grouped_lm.append(np.asarray(lm_group_eval["reliance"], dtype=np.float64))

        corr = _corr(conditional_brain[fold], conditional_lm[fold])
        observed_corr[fold] = [corr["spearman"], corr["pearson"], corr["topk_jaccard"]]
        group_corr = _corr(grouped_brain[-1], grouped_lm[-1])
        grouped_corr[fold] = [group_corr["spearman"], group_corr["pearson"], group_corr["topk_jaccard"]]
        group_counts[fold] = len(clusters)
        group_metadata.append(metadata)
        identity = within_layer_identity_null(
            conditional_brain[fold],
            conditional_lm[fold],
            names,
            n_permutations=int(spec["nulls"]["feature_identity"]["permutations"]),
            seed=_seed(int(spec["nulls"]["seeds"]["feature_identity"]), "group_average", model, story, fold),
        )
        identity_null[fold] = np.asarray(identity["null"], dtype=np.float32)
        fold_x.append(x_base)
        fold_names.append(names)
        fold_groups.append(clusters)

    max_groups = int(max(group_counts, default=0))
    grouped_brain_padded = np.full((n_folds, max_groups), np.nan, dtype=np.float32)
    grouped_lm_padded = np.full_like(grouped_brain_padded, np.nan)
    for fold in range(n_folds):
        grouped_brain_padded[fold, : group_counts[fold]] = grouped_brain[fold]
        grouped_lm_padded[fold, : group_counts[fold]] = grouped_lm[fold]

    null_conditional = np.full((len(offsets), n_folds, 3), np.nan, dtype=np.float64)
    null_grouped = np.full_like(null_conditional, np.nan)
    null_alphas = np.full((len(offsets), n_folds), np.nan, dtype=np.float64)
    for shift_index, offset in enumerate(offsets.tolist()):
        shifted = np.roll(condition["lm_target"], int(offset), axis=0)
        for fold, (outer_train, outer_test) in enumerate(fold_rows):
            x_base = fold_x[fold]
            alpha, _, _ = select_alpha_on_outer_train(
                x_base,
                shifted,
                outer_train,
                alphas=alphas,
                metric="r2_oos",
                n_time=n_time,
                inner_folds=inner_folds,
                purge_adjacent=purge,
            )
            null_alphas[shift_index, fold] = alpha
            lm_fit = fit_ridge(x_base, shifted, global_finite_training_rows(shifted, outer_train), alpha)
            conditional_eval = evaluate_fit(
                lm_fit,
                x_base[outer_test],
                shifted[outer_test],
                feature_groups(PRIMARY_SIZE, 1),
            )
            grouped_eval = evaluate_fit(
                lm_fit,
                x_base[outer_test],
                shifted[outer_test],
                fold_groups[fold],
            )
            corr = _corr(conditional_brain[fold], conditional_eval["reliance"])
            null_conditional[shift_index, fold] = [corr["spearman"], corr["pearson"], corr["topk_jaccard"]]
            corr = _corr(grouped_brain[fold], grouped_eval["reliance"])
            null_grouped[shift_index, fold] = [corr["spearman"], corr["pearson"], corr["topk_jaccard"]]

    observed_mean = np.nanmean(observed_corr, axis=0)
    grouped_observed_mean = np.nanmean(grouped_corr, axis=0)
    null_conditional_mean = np.nanmean(null_conditional, axis=1)
    null_grouped_mean = np.nanmean(null_grouped, axis=1)
    identity_mean = np.nanmean(identity_null, axis=0)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_path,
        group_mean_target=target.astype(np.float32),
        participant_counts=participant_counts.astype(np.int16),
        subject_ids=subject_ids,
        conditional_brain_reliance_by_fold=conditional_brain,
        conditional_lm_reliance_by_fold=conditional_lm,
        grouped_brain_reliance_by_fold=grouped_brain_padded,
        grouped_lm_reliance_by_fold=grouped_lm_padded,
        grouped_count_by_fold=group_counts,
        observed_conditional_correspondence_by_fold=observed_corr.astype(np.float32),
        observed_grouped_correspondence_by_fold=grouped_corr.astype(np.float32),
        identity_null_by_fold=identity_null,
        temporal_offsets=offsets,
        temporal_null_conditional_by_shift_fold=null_conditional.astype(np.float32),
        temporal_null_grouped_by_shift_fold=null_grouped.astype(np.float32),
        temporal_null_selected_alphas=null_alphas.astype(np.float32),
        brain_r_metrics_by_fold=brain_r_metrics.astype(np.float32),
        brain_r2_metrics_by_fold=brain_r2_metrics.astype(np.float32),
        lm_r2_metrics_by_fold=lm_r2_metrics.astype(np.float32),
    )
    keys = ("spearman", "pearson", "topk_jaccard")
    summary = {
        "schema_version": 1,
        "analysis_role": "correspondence_group_average_sensitivity",
        "analysis_fingerprint": fingerprint,
        "confirmatory_eligible": False,
        "model_key": model,
        "story": story,
        "bank_variant": "source_story_variance",
        "bank_size": PRIMARY_SIZE,
        "group_threshold": PRIMARY_THRESHOLD,
        "target_definition": "parcelwise mean across uncensored participant targets at each TR after run-wise target standardization",
        "generalization_scope": "held-out_story_time_for_participant_average_target_not_held_out_participants",
        "n_participants": int(len(subject_ids)),
        "participant_count_per_tr": {
            "minimum": int(np.min(participant_counts)),
            "maximum": int(np.max(participant_counts)),
            "mean": float(np.mean(participant_counts)),
            "minimum_required": int(np.ceil(MINIMUM_PARTICIPANT_FRACTION * len(subject_ids))),
            "excluded_trs": int(np.sum(~np.all(np.isfinite(target), axis=1))),
        },
        "brain_r_selected": {"mean_fold_r": float(np.nanmean(brain_r_metrics[:, 0])), "mean_fold_r2_oos": float(np.nanmean(brain_r_metrics[:, 1])), "alphas": selected["brain_r"]},
        "brain_r2_selected": {"mean_fold_r": float(np.nanmean(brain_r2_metrics[:, 0])), "mean_fold_r2_oos": float(np.nanmean(brain_r2_metrics[:, 1])), "alphas": selected["brain_r2"]},
        "lm_r2_selected": {"mean_fold_r": float(np.nanmean(lm_r2_metrics[:, 0])), "mean_fold_r2_oos": float(np.nanmean(lm_r2_metrics[:, 1])), "alphas": selected["lm_r2"]},
        "conditional_observed": {key: float(observed_mean[index]) for index, key in enumerate(keys)},
        "conditional_identity_null": {
            "n_permutations": int(identity_null.shape[1]),
            "p_value_one_sided": _p_value(identity_mean, float(observed_mean[0])),
            "aggregation": "mean_null_statistic_across_folds_per_permutation",
        },
        "conditional_temporal_null": {
            "n_shifts": int(len(offsets)),
            "fully_retuned": True,
            "p_value_one_sided": {key: _p_value(null_conditional_mean[:, index], float(observed_mean[index])) for index, key in enumerate(keys)},
            "null_mean": {key: float(np.nanmean(null_conditional_mean[:, index])) for index, key in enumerate(keys)},
        },
        "grouped_observed": {key: float(grouped_observed_mean[index]) for index, key in enumerate(keys)},
        "grouped_temporal_null": {
            "n_shifts": int(len(offsets)),
            "fully_retuned": True,
            "p_value_one_sided": {key: _p_value(null_grouped_mean[:, index], float(grouped_observed_mean[index])) for index, key in enumerate(keys)},
            "null_mean": {key: float(np.nanmean(null_grouped_mean[:, index])) for index, key in enumerate(keys)},
        },
        "group_counts_by_fold": group_counts.tolist(),
        "group_metadata_by_fold": group_metadata,
        "arrays_path": str(arrays_path),
        "input_sha256": {str(path): sha256_file(path) for path in source_paths},
        "code_sha256": {str(path): sha256_file(path) for path in code_paths},
        "elapsed_seconds": float(time.time() - started),
        "complete": True,
    }
    atomic_write_json(summary_path, summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--stories", nargs="+")
    parser.add_argument("--shifts", type=int, default=199)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    from .integrity import load_spec

    if args.data_root is not None:
        os.environ["FMRI_SAE_DATA_ROOT"] = str(args.data_root.resolve())
    if args.output_root is not None:
        os.environ["FMRI_SAE_OUTPUT_ROOT"] = str(args.output_root.resolve())
    config_path = (args.config or default_config_path()).resolve()
    spec = load_spec(config_path)
    models = list(args.models or spec["models"].keys())
    stories = list(args.stories or spec["stories"].keys())
    records = []
    for model in models:
        if model not in spec["models"]:
            raise KeyError(model)
        for story in stories:
            if story not in spec["stories"]:
                raise KeyError(story)
            print(f"analysis group average {model}/{story}", flush=True)
            records.append(
                run_condition(
                    config_path,
                    spec,
                    model,
                    story,
                    n_shifts=int(args.shifts),
                    force=bool(args.force),
                )
            )
    group_root = _artifact_root(config_path, spec) / "group_average"
    manifest = materialize_group_average_manifest(group_root, spec)
    manifest_path = group_root / "manifest.json"
    print(json.dumps({
        "conditions_computed_by_process": len(records),
        "conditions_in_manifest": manifest["observed_condition_count"],
        "manifest_complete": manifest["complete"],
        "manifest": str(manifest_path),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
