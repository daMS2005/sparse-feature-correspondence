#!/usr/bin/env python3
"""Grouped-reliance analyses over exact source-variance feature banks.

Correlation-pruned banks are useful for displaying representative features,
but they are not the input to this analysis.  For every requested threshold,
groups are reconstructed from the complete cluster memberships defined on
the exact 256-feature source-variance predictor matrix.  The fitted model
therefore still contains all 256 columns; grouped ablation jointly zeros all
members of a discovery-defined cluster (and all five lag columns for BOLD).

The stage reports within-story fold-level and cross-story transfer
correspondence.  It does not average profiles whose group identities differ
between folds.  If fewer than three independent groups are available, the
condition is retained as ``underidentified`` rather than being interpreted
as evidence for or against correspondence.
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
from .feature_banks import contiguous_fold_rows
from .fitting import (  # noqa: E402
    _artifact_root,
    _extract_bank,
    _feature_bank_root,
    _load_bank,
    evaluate_fit,
    load_condition,
    select_alpha_on_outer_train,
)
from .integrity import (  # noqa: E402
    assert_cache_fingerprint,
    atomic_write_json,
    provenance_fingerprint,
    sha256_file,
)
from .modeling import (  # noqa: E402
    block_labels,
    fit_ridge,
    lag_expand,
    modal_alpha,
    select_alpha,
)

HERE = Path(__file__).resolve().parent

PRIMARY_SIZE = 256
DEFAULT_THRESHOLDS = (0.70, 0.80, 0.90)


def _threshold_key(threshold: float) -> str:
    return str(float(threshold))


def _variant_name(threshold: float) -> str:
    return f"source_story_correlation_pruned_{float(threshold):.2f}".replace(".", "p")


def _bank_file(
    config_path: Path,
    spec: dict[str, Any],
    model: str,
    story: str,
    *,
    fold: int | None = None,
    target_story: str | None = None,
) -> Path:
    root = _feature_bank_root(config_path, spec)
    if fold is not None:
        return root / "within_story" / model / story / f"fold_{int(fold):02d}" / "banks.json"
    if target_story is None:
        raise ValueError("cross-story bank path requires target_story")
    return root / "cross_story" / model / f"{story}_to_{target_story}" / "banks.json"


def _cluster_groups(
    bank_record: dict[str, Any],
    *,
    size: int,
    threshold: float,
) -> tuple[list[np.ndarray], list[dict[str, Any]], list[str]]:
    """Map exact-K cluster memberships to source-variance column positions."""

    source = bank_record.get("source_variance", {})
    entry = source.get("banks", {}).get(str(int(size)))
    if not isinstance(entry, dict) or not bool(entry.get("complete")):
        raise ValueError("exact source-variance bank is missing or incomplete")
    pruned = bank_record.get("correlation_pruned", {})
    key = _threshold_key(threshold)
    if key not in pruned:
        # JSON writers sometimes normalize 0.70 to 0.7; accept only an exact
        # numeric match, never a neighboring threshold.
        matches = [candidate for candidate in pruned if np.isclose(float(candidate), float(threshold))]
        if len(matches) != 1:
            raise KeyError(f"correlation-pruned threshold {threshold} is absent")
        key = matches[0]
    exact = pruned[key].get("clusters_over_exact_source_variance_banks", {}).get(str(int(size)))
    if not isinstance(exact, dict) or not bool(exact.get("complete_exact_k_predictor_membership")):
        raise ValueError("exact-K cluster membership is unavailable or incomplete")
    layers = [int(value) for value in bank_record.get("selected_layers", bank_record.get("source_variance", {}).get("layers", []))]
    if not layers:
        raise ValueError("cluster record contains no sampled layers")
    identities = [(int(layer), int(feature)) for layer, feature in zip(entry["layers"], entry["feature_ids"], strict=False)]
    positions = {identity: index for index, identity in enumerate(identities)}
    groups: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    labels: list[str] = []
    clusters_by_layer = exact.get("clusters_by_layer", {})
    for layer in layers:
        clusters = clusters_by_layer.get(str(layer), [])
        for cluster in clusters:
            members = [(int(layer), int(feature)) for feature in cluster.get("members", [])]
            columns = [positions[identity] for identity in members if identity in positions]
            if not columns:
                continue
            group_id = f"layer_{layer}_cluster_{int(cluster.get('cluster_id', len(groups) + 1))}"
            groups.append(np.asarray(sorted(set(columns)), dtype=np.int64))
            labels.append(group_id)
            metadata.append(
                {
                    "group_id": group_id,
                    "layer": int(layer),
                    "cluster_id": int(cluster.get("cluster_id", len(groups))),
                    "members": [int(feature) for _, feature in members if (int(layer), int(feature)) in positions],
                    "source_variance_columns": [int(value) for value in sorted(set(columns))],
                    "size": int(len(set(columns))),
                    "threshold": float(threshold),
                }
            )
    if len(groups) != len(metadata) or len(groups) != len(labels):
        raise AssertionError("group metadata length mismatch")
    return groups, metadata, labels


def _expand_brain_groups(groups: Sequence[np.ndarray], n_lags: int) -> list[np.ndarray]:
    return [
        np.asarray(
            [int(feature) * int(n_lags) + lag for feature in group.tolist() for lag in range(int(n_lags))],
            dtype=np.int64,
        )
        for group in groups
    ]


def _aggregate_profile(values: Sequence[np.ndarray]) -> np.ndarray:
    if not values:
        return np.asarray([], dtype=np.float64)
    return np.nanmean(np.asarray(values, dtype=np.float64), axis=0)


def _corr(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    from .modeling import correspondence

    if len(a) < 3 or len(b) < 3:
        return {"spearman": float("nan"), "pearson": float("nan"), "topk_jaccard": float("nan")}
    top_k = max(1, min(32, int(round(len(a) * 0.125))))
    return correspondence(a, b, top_k=top_k)


def _pad_profiles(profiles: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    max_groups = max((len(profile) for profile in profiles), default=0)
    padded = np.full((len(profiles), max_groups), np.nan, dtype=np.float32)
    counts = np.zeros(len(profiles), dtype=np.int64)
    for index, profile in enumerate(profiles):
        counts[index] = len(profile)
        if len(profile):
            padded[index, : len(profile)] = np.asarray(profile, dtype=np.float32)
    return padded, counts


def _missing_record(role: str, label: str, model: str, threshold: float, reason: str, path: Path) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "analysis_role": role,
        "status": "skipped",
        "model_key": model,
        "label": label,
        "threshold": float(threshold),
        "bank_size": PRIMARY_SIZE,
        "reason": reason,
        "required_input": str(path),
        "complete": False,
    }


def _within_condition(
    config_path: Path,
    spec: dict[str, Any],
    model: str,
    story: str,
    threshold: float,
    *,
    force: bool,
) -> dict[str, Any]:
    condition = load_condition(config_path, spec, model, story)
    n_folds = int(spec["cross_validation"]["outer_folds"])
    inner_folds = int(spec["cross_validation"]["inner_folds"])
    purge = int(spec["cross_validation"]["purge_adjacent_trs"])
    alphas = np.asarray(spec["cross_validation"]["alpha_grid"], dtype=np.float64)
    fold_rows = contiguous_fold_rows(
        len(condition["tr_indices"]), n_folds, purge_adjacent_trs=purge
    )
    bank_paths = [_bank_file(config_path, spec, model, story, fold=fold) for fold in range(n_folds)]
    output_dir = _artifact_root(config_path, spec) / "grouped" / "within_story" / story / model / _variant_name(threshold) / f"n{PRIMARY_SIZE}"
    summary_path = output_dir / "summary.json"
    arrays_path = output_dir / "arrays.npz"
    source_paths = [condition["predictor_path"], condition["brain_path"], *bank_paths]
    code_paths = [HERE / "grouped_reliance.py", HERE / "fitting.py", HERE / "feature_banks.py", HERE / "modeling.py"]
    fingerprint = provenance_fingerprint(
        config_path,
        input_paths=source_paths,
        code_paths=code_paths,
        extra={"analysis": "correspondence_within_grouped_reliance", "model": model, "story": story, "threshold": float(threshold), "size": PRIMARY_SIZE},
    )
    if not force:
        cached = assert_cache_fingerprint(summary_path, fingerprint)
        if cached is not None:
            if not arrays_path.is_file():
                raise RuntimeError(f"grouped cache exists without arrays: {summary_path}")
            return cached

    n_time = len(condition["tr_indices"])
    subject_ids = np.asarray(sorted(condition["subject_targets"]), dtype=str)
    n_subjects = len(subject_ids)
    fold_brain: list[np.ndarray] = []
    fold_lm: list[np.ndarray] = []
    fold_corr: list[dict[str, float]] = []
    fold_metadata: list[list[dict[str, Any]]] = []
    fold_labels: list[list[str]] = []
    fold_alphas_lm: list[float] = []
    fold_alpha_brain: list[list[float]] = []
    started = time.time()
    for fold_index, (outer_train, outer_test) in enumerate(fold_rows):
        bank_record, bank = _load_bank(bank_paths[fold_index], "source_story_variance", PRIMARY_SIZE)
        if not np.array_equal(
            np.asarray(bank_record.get("selection_rows", []), dtype=np.int64),
            np.asarray(outer_train, dtype=np.int64),
        ):
            raise RuntimeError(
                f"{bank_paths[fold_index]}: grouped analysis bank is not selected from the purged outer-training rows"
            )
        groups, metadata, labels = _cluster_groups(bank_record, size=PRIMARY_SIZE, threshold=threshold)
        x_base, _ = _extract_bank(condition, bank, PRIMARY_SIZE)
        x_lm = x_base
        x_brain = lag_expand(x_base, spec["targets"]["brain"]["lags"])
        lm_groups = groups
        brain_groups = _expand_brain_groups(groups, len(spec["targets"]["brain"]["lags"]))
        lm_alpha, _, _ = select_alpha_on_outer_train(
            x_lm, condition["lm_target"], outer_train, alphas=alphas, metric="r2_oos", n_time=n_time, inner_folds=inner_folds, purge_adjacent=purge
        )
        lm_fit = fit_ridge(x_lm, condition["lm_target"], outer_train, lm_alpha)
        lm_eval = evaluate_fit(lm_fit, x_lm[outer_test], condition["lm_target"][outer_test], lm_groups)
        fold_lm.append(np.asarray(lm_eval["reliance"], dtype=np.float64))
        fold_alphas_lm.append(float(lm_alpha))
        subject_profiles: list[np.ndarray] = []
        subject_alphas: list[float] = []
        for subject in subject_ids.tolist():
            target = condition["subject_targets"][str(subject)]
            valid_train = np.intersect1d(outer_train, np.flatnonzero(np.all(np.isfinite(target), axis=1)), assume_unique=True)
            alpha, _, _ = select_alpha_on_outer_train(
                x_brain, target, outer_train, alphas=alphas, metric="r2_oos", n_time=n_time, inner_folds=inner_folds, purge_adjacent=purge
            )
            fit = fit_ridge(x_brain, target, valid_train, alpha)
            evaluation = evaluate_fit(fit, x_brain[outer_test], target[outer_test], brain_groups)
            subject_profiles.append(np.asarray(evaluation["reliance"], dtype=np.float64))
            subject_alphas.append(float(alpha))
        brain_profile = _aggregate_profile(subject_profiles)
        fold_brain.append(brain_profile)
        fold_alpha_brain.append(subject_alphas)
        fold_corr.append(_corr(brain_profile, fold_lm[-1]))
        fold_metadata.append(metadata)
        fold_labels.append(labels)
    counts = [len(profile) for profile in fold_brain]
    status = "underidentified" if any(count < 3 for count in counts) else "complete"
    brain_padded, _ = _pad_profiles(fold_brain)
    lm_padded, _ = _pad_profiles(fold_lm)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_path,
        grouped_brain_reliance_by_fold=brain_padded,
        grouped_lm_reliance_by_fold=lm_padded,
        grouped_count_by_fold=np.asarray(counts, dtype=np.int64),
        grouped_spearman_by_fold=np.asarray([row["spearman"] for row in fold_corr], dtype=np.float32),
        grouped_pearson_by_fold=np.asarray([row["pearson"] for row in fold_corr], dtype=np.float32),
        grouped_topk_jaccard_by_fold=np.asarray([row["topk_jaccard"] for row in fold_corr], dtype=np.float32),
        lm_r2_selected_alpha_by_fold=np.asarray(fold_alphas_lm, dtype=np.float32),
        brain_r2_selected_alpha_by_fold=np.asarray(fold_alpha_brain, dtype=np.float32),
    )
    summary = {
        "schema_version": 1,
        "analysis_role": "correspondence_within_grouped_reliance",
        "analysis_fingerprint": fingerprint,
        "status": status,
        "complete": True,
        "model_key": model,
        "story": story,
        "threshold": float(threshold),
        "bank_size": PRIMARY_SIZE,
        "group_definition": "complete_linkage_absolute_spearman_clusters_on_exact_source_variance_256_bank",
        "model_input_bank": "exact_source_story_variance_256_not_representative_pruned_bank",
        "brain_group_unit": "all_members_and_all_five_lag_columns",
        "lm_group_unit": "all_members_one_unlagged_column",
        "selection_metric": "r2_oos",
        "n_outer_folds": n_folds,
        "n_subjects": n_subjects,
        "group_count_by_fold": counts,
        "minimum_independent_group_requirement": 3,
        "underidentified_reason": "fewer_than_three_discovery_defined_groups_in_at_least_one_fold" if status == "underidentified" else None,
        "correspondence_by_fold": fold_corr,
        "mean_of_fold_correspondence": {key: float(np.nanmean([row[key] for row in fold_corr])) for key in ("spearman", "pearson", "topk_jaccard")},
        "groups_by_fold": fold_metadata,
        "group_labels_by_fold": fold_labels,
        "arrays_path": str(arrays_path),
        "input_sha256": {str(path): sha256_file(path) for path in source_paths},
        "code_sha256": {str(path): sha256_file(path) for path in code_paths},
        "elapsed_seconds": float(time.time() - started),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary_path, summary)
    return summary


def _modal_source_alpha(x: np.ndarray, y: np.ndarray, blocks: np.ndarray, alphas: np.ndarray, purge: int) -> float:
    selected = []
    for outer_block in sorted(np.unique(blocks).tolist()):
        alpha, _ = select_alpha(x, y, blocks, int(outer_block), alphas, metric="r2_oos", purge_adjacent=purge)
        selected.append({"alpha": float(alpha)})
    return modal_alpha(selected)


def _cross_condition(
    config_path: Path,
    spec: dict[str, Any],
    model: str,
    source_story: str,
    target_story: str,
    threshold: float,
    *,
    force: bool,
) -> dict[str, Any]:
    source = load_condition(config_path, spec, model, source_story)
    target = load_condition(config_path, spec, model, target_story)
    bank_file = _bank_file(config_path, spec, model, source_story, target_story=target_story)
    bank_record, bank = _load_bank(bank_file, "source_story_variance", PRIMARY_SIZE)
    groups, metadata, labels = _cluster_groups(bank_record, size=PRIMARY_SIZE, threshold=threshold)
    x_source, names = _extract_bank(source, bank, PRIMARY_SIZE)
    x_target, target_names = _extract_bank(target, bank, PRIMARY_SIZE)
    if not np.array_equal(names, target_names):
        raise RuntimeError("cross-story grouped extraction changed source feature identities")
    output_dir = _artifact_root(config_path, spec) / "grouped" / "cross_story" / f"{source_story}_to_{target_story}" / model / _variant_name(threshold) / f"n{PRIMARY_SIZE}"
    summary_path = output_dir / "summary.json"
    arrays_path = output_dir / "arrays.npz"
    source_paths = [source["predictor_path"], source["brain_path"], target["predictor_path"], target["brain_path"], bank_file]
    code_paths = [HERE / "grouped_reliance.py", HERE / "fitting.py", HERE / "feature_banks.py", HERE / "modeling.py"]
    fingerprint = provenance_fingerprint(
        config_path,
        input_paths=source_paths,
        code_paths=code_paths,
        extra={"analysis": "correspondence_cross_story_grouped_reliance", "model": model, "source_story": source_story, "target_story": target_story, "threshold": float(threshold), "size": PRIMARY_SIZE},
    )
    if not force:
        cached = assert_cache_fingerprint(summary_path, fingerprint)
        if cached is not None:
            if not arrays_path.is_file():
                raise RuntimeError(f"grouped cache exists without arrays: {summary_path}")
            return cached
    purge = int(spec["cross_validation"]["purge_adjacent_trs"])
    n_folds = int(spec["cross_validation"]["outer_folds"])
    alphas = np.asarray(spec["cross_validation"]["alpha_grid"], dtype=np.float64)
    blocks = block_labels(len(x_source), n_folds)
    brain_groups = _expand_brain_groups(groups, len(spec["targets"]["brain"]["lags"]))
    lm_groups = groups
    subjects = sorted(set(source["subject_targets"]) & set(target["subject_targets"]))
    expected = int(spec["cross_validation"].get("expected_matched_subjects", len(subjects)))
    if len(subjects) != expected:
        raise RuntimeError(f"expected {expected} matched participants, found {len(subjects)}")
    brain_profiles: list[np.ndarray] = []
    brain_alphas: list[float] = []
    started = time.time()
    for subject in subjects:
        source_y = source["subject_targets"][subject]
        target_y = target["subject_targets"][subject]
        brain_alpha = _modal_source_alpha(lag_expand(x_source, spec["targets"]["brain"]["lags"]), source_y, blocks, alphas, purge)
        brain_fit = fit_ridge(lag_expand(x_source, spec["targets"]["brain"]["lags"]), source_y, np.flatnonzero(np.all(np.isfinite(source_y), axis=1)), brain_alpha)
        brain_eval = evaluate_fit(brain_fit, lag_expand(x_target, spec["targets"]["brain"]["lags"]), target_y, brain_groups)
        brain_profiles.append(np.asarray(brain_eval["reliance"], dtype=np.float64))
        brain_alphas.append(float(brain_alpha))
    lm_alpha = _modal_source_alpha(x_source, source["lm_target"], blocks, alphas, purge)
    lm_fit = fit_ridge(x_source, source["lm_target"], np.flatnonzero(np.all(np.isfinite(source["lm_target"]), axis=1)), lm_alpha)
    lm_eval = evaluate_fit(lm_fit, x_target, target["lm_target"], lm_groups)
    brain_profile = _aggregate_profile(brain_profiles)
    lm_profile = np.asarray(lm_eval["reliance"], dtype=np.float64)
    corr = _corr(brain_profile, lm_profile)
    status = "underidentified" if len(groups) < 3 else "complete"
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_path,
        grouped_brain_reliance_by_subject=np.asarray(brain_profiles, dtype=np.float32),
        grouped_brain_reliance_mean=brain_profile.astype(np.float32),
        grouped_lm_reliance=lm_profile.astype(np.float32),
        grouped_correspondence=np.asarray([corr["spearman"], corr["pearson"], corr["topk_jaccard"]], dtype=np.float32),
        brain_r2_selected_alpha_by_subject=np.asarray(brain_alphas, dtype=np.float32),
        lm_r2_selected_alpha=np.asarray([lm_alpha], dtype=np.float32),
    )
    summary = {
        "schema_version": 1,
        "analysis_role": "correspondence_cross_story_grouped_reliance",
        "analysis_fingerprint": fingerprint,
        "status": status,
        "complete": True,
        "model_key": model,
        "source_story": source_story,
        "target_story": target_story,
        "threshold": float(threshold),
        "bank_size": PRIMARY_SIZE,
        "group_definition": "complete_linkage_absolute_spearman_clusters_on_exact_source_variance_256_bank",
        "model_input_bank": "exact_source_story_variance_256_not_representative_pruned_bank",
        "brain_group_unit": "all_members_and_all_five_lag_columns",
        "lm_group_unit": "all_members_one_unlagged_column",
        "selection_metric": "r2_oos",
        "n_matched_subjects": len(subjects),
        "n_groups": len(groups),
        "minimum_independent_group_requirement": 3,
        "underidentified_reason": "fewer_than_three_discovery_defined_groups" if status == "underidentified" else None,
        "correspondence": corr,
        "groups": metadata,
        "group_labels": labels,
        "arrays_path": str(arrays_path),
        "input_sha256": {str(path): sha256_file(path) for path in source_paths},
        "code_sha256": {str(path): sha256_file(path) for path in code_paths},
        "elapsed_seconds": float(time.time() - started),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(summary_path, summary)
    return summary


def _selected_models(spec: dict[str, Any], args: Any) -> list[str]:
    values = getattr(args, "models", None)
    models = [str(value) for value in (values or spec["models"].keys())]
    for model in models:
        if model not in spec["models"]:
            raise KeyError(model)
    return models


def run(*, config_path: Path, spec: dict[str, Any], args: Any) -> dict[str, Any]:
    models = _selected_models(spec, args)
    thresholds = tuple(float(value) for value in (getattr(args, "thresholds", None) or spec["feature_bank"]["banks"][2].get("thresholds", DEFAULT_THRESHOLDS)))
    force = bool(getattr(args, "force", False))
    records: list[dict[str, Any]] = []
    for model in models:
        for story in ("shapesphysical", "shapessocial"):
            for threshold in thresholds:
                try:
                    print(f"analysis within grouped {model}/{story}/threshold={threshold}", flush=True)
                    records.append(_within_condition(config_path, spec, model, story, threshold, force=force))
                except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
                    records.append(_missing_record("correspondence_within_grouped_reliance", story, model, threshold, f"input_unavailable:{type(exc).__name__}:{exc}", _bank_file(config_path, spec, model, story, fold=0)))
        for source_story, target_story in (("shapesphysical", "shapessocial"), ("shapessocial", "shapesphysical")):
            label = f"{source_story}_to_{target_story}"
            for threshold in thresholds:
                try:
                    print(f"analysis cross grouped {model}/{label}/threshold={threshold}", flush=True)
                    records.append(_cross_condition(config_path, spec, model, source_story, target_story, threshold, force=force))
                except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
                    records.append(_missing_record("correspondence_cross_story_grouped_reliance", label, model, threshold, f"input_unavailable:{type(exc).__name__}:{exc}", _bank_file(config_path, spec, model, source_story, target_story=target_story)))
    manifest_path = _artifact_root(config_path, spec) / "grouped" / "manifest.json"
    manifest = {
        "schema_version": 1,
        "analysis_role": "correspondence_grouped_reliance_manifest",
        "bank_size": PRIMARY_SIZE,
        "thresholds": list(thresholds),
        "records": records,
        "complete_records": int(sum(bool(row.get("complete")) for row in records)),
        "skipped_records": int(sum(row.get("status") == "skipped" for row in records)),
        "underidentified_records": int(sum(row.get("status") == "underidentified" for row in records)),
    }
    atomic_write_json(manifest_path, manifest)
    return {"stage": "grouped", "manifest": str(manifest_path), **manifest}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--thresholds", nargs="+", type=float, default=None)
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
