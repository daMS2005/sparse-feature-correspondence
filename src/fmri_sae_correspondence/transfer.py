#!/usr/bin/env python3
"""Cross-story source-bank transfer fits for analysis.

Each direction selects its feature bank from the source story only.  A model
is fitted and scaled on the source story, and the resulting coefficients and
training statistics are applied unchanged to the target story.  The BOLD and
LM transfer fits use the same source-bank ``(layer, feature_id)`` sequence.

This stage deliberately excludes temporal-target nulls.  Those are a later
stage because they require a complete LM refit for every shift.
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
from .fitting import (  # noqa: E402
    DEFAULT_SIZES,
    _artifact_root,
    _extract_bank,
    _feature_bank_root,
    _load_bank,
    get_variant,
    load_condition,
)
from .integrity import (  # noqa: E402
    assert_cache_fingerprint,
    atomic_write_json,
    provenance_fingerprint,
)
from .modeling import (  # noqa: E402
    block_labels,
    correspondence,
    feature_groups,
    fit_transfer,
    lag_expand,
    modal_alpha,
    nested_subject_cv,
    within_layer_identity_null,
)
from .modeling import (
    sha256_file as engine_sha256_file,
)

HERE = Path(__file__).resolve().parent


def _strip_arrays(value: Any, *, drop_large: bool = True) -> Any:
    """Make engine result metadata JSON-safe without embedding predictions."""

    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            if drop_large and key in {
                "oof",
                "reliance",
                "fold_reliance",
                "fold_scaling",
                "prediction",
            }:
                continue
            output[str(key)] = _strip_arrays(item, drop_large=drop_large)
        return output
    if isinstance(value, (list, tuple)):
        return [_strip_arrays(item, drop_large=drop_large) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    return value


def _jaccard_pairs(
    left: dict[str, Any], right: dict[str, Any], size: int
) -> dict[str, Any]:
    """Compare directional bank identities without treating order as semantic."""

    l_entry = left["banks"][str(int(size))]
    r_entry = right["banks"][str(int(size))]
    left_pairs = {(int(layer), int(feature)) for layer, feature in zip(l_entry["layers"], l_entry["feature_ids"], strict=False)}
    right_pairs = {(int(layer), int(feature)) for layer, feature in zip(r_entry["layers"], r_entry["feature_ids"], strict=False)}
    union = left_pairs | right_pairs
    result: dict[str, Any] = {
        "overall": float(len(left_pairs & right_pairs) / len(union)) if union else float("nan"),
        "intersection": int(len(left_pairs & right_pairs)),
        "union": int(len(union)),
    }
    per_layer: dict[str, float] = {}
    for layer in sorted({pair[0] for pair in left_pairs | right_pairs}):
        lset = {feature for current_layer, feature in left_pairs if current_layer == layer}
        rset = {feature for current_layer, feature in right_pairs if current_layer == layer}
        layer_union = lset | rset
        per_layer[str(layer)] = float(len(lset & rset) / len(layer_union)) if layer_union else float("nan")
    result["per_layer"] = per_layer
    return result


def _transfer_summary(
    result: dict[str, Any], subject: str
) -> dict[str, Any]:
    return {
        "subject": str(subject),
        "alpha": float(result["alpha"]),
        "n_train": int(result["n_train"]),
        "n_test": int(result["n_test"]),
        "mean_r": float(result["mean_r"]),
        "mean_r2_oos": float(result["mean_r2_oos"]),
    }


def fit_transfer_condition(
    config_path: Path,
    spec: dict[str, Any],
    model_key: str,
    train_story: str,
    test_story: str,
    variant: str,
    size: int,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Fit one source-story transfer condition."""

    train_condition = load_condition(config_path, spec, model_key, train_story)
    test_condition = load_condition(config_path, spec, model_key, test_story)
    train_bank_file = _bank_file(
        config_path, spec, model_key, train_story, test_story
    )
    reverse_bank_file = _bank_file(
        config_path, spec, model_key, test_story, train_story
    )
    train_bank_record, train_bank = _load_bank(train_bank_file, variant, int(size))
    reverse_bank_record, reverse_bank = _load_bank(reverse_bank_file, variant, int(size))
    x_train, feature_names = _extract_bank(train_condition, train_bank, int(size))
    x_test, target_names = _extract_bank(test_condition, train_bank, int(size))
    if not np.array_equal(feature_names, target_names):
        raise RuntimeError("target extraction changed the source bank feature identities")

    # Source-bank identity comparison is an invariant for the paired targets;
    # the reverse-direction bank is used only for the descriptive overlap.
    reverse_variant = get_variant(reverse_bank_record, variant)
    _ = reverse_variant
    source_paths = [
        train_condition["predictor_path"],
        train_condition["brain_path"],
        test_condition["predictor_path"],
        test_condition["brain_path"],
        train_bank_file,
        reverse_bank_file,
    ]
    code_paths = [
        HERE / "transfer.py",
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
            "analysis": "correspondence_cross_story_transfer",
            "model_key": model_key,
            "train_story": train_story,
            "test_story": test_story,
            "variant": variant,
            "size": int(size),
        },
    )
    output_dir = (
        _artifact_root(config_path, spec)
        / "transfer"
        / f"{model_key}_{train_story}_to_{test_story}"
        / variant
        / f"n{int(size)}"
    )
    summary_path = output_dir / "summary.json"
    arrays_path = output_dir / "arrays.npz"
    if not force:
        cached = assert_cache_fingerprint(summary_path, fingerprint)
        if cached is not None:
            if not arrays_path.is_file():
                raise RuntimeError(f"summary cache exists without arrays: {summary_path}")
            return cached

    expected_matched = int(spec.get("cross_validation", {}).get("expected_matched_subjects", 59))
    common_subjects = sorted(
        set(train_condition["subject_targets"]) & set(test_condition["subject_targets"])
    )
    if len(common_subjects) != expected_matched:
        raise RuntimeError(
            f"{model_key}/{train_story}->{test_story}: expected {expected_matched} matched participants, found {len(common_subjects)}"
        )
    cv = spec["cross_validation"]
    alphas = np.asarray(cv["alpha_grid"], dtype=np.float64)
    purge = int(cv["purge_adjacent_trs"])
    blocks = block_labels(len(x_train), int(cv["outer_folds"]))
    brain_lags = spec["targets"]["brain"]["lags"]
    x_train_brain = lag_expand(x_train, brain_lags)
    x_test_brain = lag_expand(x_test, brain_lags)
    brain_groups = feature_groups(int(size), len(brain_lags))
    lm_groups = feature_groups(int(size), 1)
    started = time.time()

    brain_r: list[dict[str, Any]] = []
    brain_r2: list[dict[str, Any]] = []
    brain_r_reliance: list[np.ndarray] = []
    brain_r2_reliance: list[np.ndarray] = []
    brain_r_predictions: list[np.ndarray] = []
    brain_r2_predictions: list[np.ndarray] = []
    brain_r2_source_cv: list[dict[str, Any]] = []
    brain_r_source_cv: list[dict[str, Any]] = []
    brain_r2_x_mean: list[np.ndarray] = []
    brain_r2_x_scale: list[np.ndarray] = []
    brain_r2_y_mean: list[np.ndarray] = []
    brain_r2_y_scale: list[np.ndarray] = []

    n_test = len(test_condition["tr_indices"])
    n_parcels = next(iter(test_condition["subject_targets"].values())).shape[1]
    for subject in common_subjects:
        y_train = train_condition["subject_targets"][subject]
        y_test = test_condition["subject_targets"][subject]
        training_r = nested_subject_cv(
            x_train_brain,
            y_train,
            blocks,
            alphas,
            selection_metric="pearson_r",
            purge_adjacent=purge,
            groups=brain_groups,
        )
        training_r2 = nested_subject_cv(
            x_train_brain,
            y_train,
            blocks,
            alphas,
            selection_metric="r2_oos",
            purge_adjacent=purge,
            groups=brain_groups,
        )
        transfer_r = fit_transfer(
            x_train_brain,
            y_train,
            x_test_brain,
            y_test,
            alpha=modal_alpha(training_r["folds"]),
            groups=brain_groups,
        )
        transfer_r2 = fit_transfer(
            x_train_brain,
            y_train,
            x_test_brain,
            y_test,
            alpha=modal_alpha(training_r2["folds"]),
            groups=brain_groups,
        )
        brain_r.append(_transfer_summary(transfer_r, subject))
        brain_r2.append(_transfer_summary(transfer_r2, subject))
        brain_r_reliance.append(np.asarray(transfer_r["reliance"], dtype=np.float32))
        brain_r2_reliance.append(np.asarray(transfer_r2["reliance"], dtype=np.float32))
        pred_r = np.full((n_test, n_parcels), np.nan, dtype=np.float32)
        pred_r2 = np.full_like(pred_r, np.nan)
        pred_r[np.asarray(transfer_r["test_positions"], dtype=np.int64)] = np.asarray(transfer_r["prediction"], dtype=np.float32)
        pred_r2[np.asarray(transfer_r2["test_positions"], dtype=np.int64)] = np.asarray(transfer_r2["prediction"], dtype=np.float32)
        brain_r_predictions.append(pred_r)
        brain_r2_predictions.append(pred_r2)
        brain_r_source_cv.append(_strip_arrays(training_r))
        brain_r2_source_cv.append(_strip_arrays(training_r2))
        brain_r2_x_mean.append(np.asarray(transfer_r2["x_mean"], dtype=np.float32))
        brain_r2_x_scale.append(np.asarray(transfer_r2["x_scale"], dtype=np.float32))
        brain_r2_y_mean.append(np.asarray(transfer_r2["y_mean"], dtype=np.float32))
        brain_r2_y_scale.append(np.asarray(transfer_r2["y_scale"], dtype=np.float32))

    # The LM target is fit on the source story and transferred using the
    # exact source-trained scaling and coefficients.
    lm_training_r = nested_subject_cv(
        x_train,
        train_condition["lm_target"],
        blocks,
        alphas,
        selection_metric="pearson_r",
        purge_adjacent=purge,
        groups=lm_groups,
    )
    lm_training_r2 = nested_subject_cv(
        x_train,
        train_condition["lm_target"],
        blocks,
        alphas,
        selection_metric="r2_oos",
        purge_adjacent=purge,
        groups=lm_groups,
    )
    lm_transfer_r = fit_transfer(
        x_train,
        train_condition["lm_target"],
        x_test,
        test_condition["lm_target"],
        alpha=modal_alpha(lm_training_r["folds"]),
        groups=lm_groups,
    )
    lm_transfer_r2 = fit_transfer(
        x_train,
        train_condition["lm_target"],
        x_test,
        test_condition["lm_target"],
        alpha=modal_alpha(lm_training_r2["folds"]),
        groups=lm_groups,
    )

    brain_r_arr = np.asarray(brain_r_reliance, dtype=np.float64)
    brain_r2_arr = np.asarray(brain_r2_reliance, dtype=np.float64)
    brain_r_mean = np.nanmean(brain_r_arr, axis=0)
    brain_r2_mean = np.nanmean(brain_r2_arr, axis=0)
    correspondences = {
        "r_selected": correspondence(brain_r_mean, lm_transfer_r["reliance"], top_k=max(1, min(32, int(round(size * 0.125))))),
        "r2_selected": correspondence(brain_r2_mean, lm_transfer_r2["reliance"], top_k=max(1, min(32, int(round(size * 0.125))))),
    }
    identity = within_layer_identity_null(
        brain_r2_mean,
        lm_transfer_r2["reliance"],
        feature_names,
        n_permutations=int(spec["nulls"]["feature_identity"]["permutations"]),
        seed=int(spec["nulls"]["seeds"]["feature_identity"]),
    )
    overlap = _jaccard_pairs(
        get_variant(train_bank_record, variant),
        get_variant(reverse_bank_record, variant),
        int(size),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_path,
        feature_names=np.asarray(feature_names, dtype=str),
        subject_ids=np.asarray(common_subjects, dtype=str),
        brain_r_reliance=brain_r_arr.astype(np.float32),
        brain_r2_reliance=brain_r2_arr.astype(np.float32),
        brain_r_prediction=np.asarray(brain_r_predictions, dtype=np.float32),
        brain_r2_prediction=np.asarray(brain_r2_predictions, dtype=np.float32),
        lm_r_reliance=np.asarray(lm_transfer_r["reliance"], dtype=np.float32),
        lm_r2_reliance=np.asarray(lm_transfer_r2["reliance"], dtype=np.float32),
        lm_r_prediction=np.asarray(lm_transfer_r["prediction"], dtype=np.float32),
        lm_r2_prediction=np.asarray(lm_transfer_r2["prediction"], dtype=np.float32),
        identity_null_r2=np.asarray(identity["null"], dtype=np.float32),
        brain_r2_x_mean=np.asarray(brain_r2_x_mean, dtype=np.float32),
        brain_r2_x_scale=np.asarray(brain_r2_x_scale, dtype=np.float32),
        brain_r2_y_mean=np.asarray(brain_r2_y_mean, dtype=np.float32),
        brain_r2_y_scale=np.asarray(brain_r2_y_scale, dtype=np.float32),
        lm_r2_x_mean=np.asarray(lm_transfer_r2["x_mean"], dtype=np.float32),
        lm_r2_x_scale=np.asarray(lm_transfer_r2["x_scale"], dtype=np.float32),
        lm_r2_y_mean=np.asarray(lm_transfer_r2["y_mean"], dtype=np.float32),
        lm_r2_y_scale=np.asarray(lm_transfer_r2["y_scale"], dtype=np.float32),
    )
    summary = {
        "schema_version": 1,
        "analysis_role": "correspondence_cross_story_source_bank_transfer",
        "analysis_fingerprint": fingerprint,
        "model_key": model_key,
        "model_id": train_condition["model_id"],
        "train_story": train_story,
        "test_story": test_story,
        "feature_bank_variant": variant,
        "bank_size": int(size),
        "feature_names": feature_names.tolist(),
        "n_matched_subjects": int(len(common_subjects)),
        "n_train_time": int(len(x_train)),
        "n_test_time": int(len(x_test)),
        "n_parcels": int(n_parcels),
        "lm_dimensions": int(test_condition["lm_target"].shape[1]),
        "purge_adjacent_trs": int(purge),
        "brain_lags": [int(value) for value in brain_lags],
        "selection_metric_models": ["pearson_r", "r2_oos"],
        "primary_reliance_metric": "r2_selected_held_out_delta_r2_oos",
        "source_bank_selection_only": True,
        "target_story_excluded_from_bank_selection": True,
        "paired_targets_use_exact_same_feature_layer_coordinates": True,
        "source_scaling_and_weights_applied_unchanged": True,
        "brain_r_selected": {
            "subjects": brain_r,
            "mean_r": float(np.nanmean([row["mean_r"] for row in brain_r])),
            "mean_r2_oos": float(np.nanmean([row["mean_r2_oos"] for row in brain_r])),
            "source_training_cv": brain_r_source_cv,
        },
        "brain_r2_selected": {
            "subjects": brain_r2,
            "mean_r": float(np.nanmean([row["mean_r"] for row in brain_r2])),
            "mean_r2_oos": float(np.nanmean([row["mean_r2_oos"] for row in brain_r2])),
            "source_training_cv": brain_r2_source_cv,
        },
        "lm_r_selected": {
            "alpha": float(lm_transfer_r["alpha"]),
            "mean_r": float(lm_transfer_r["mean_r"]),
            "mean_r2_oos": float(lm_transfer_r["mean_r2_oos"]),
            "source_training_cv": _strip_arrays(lm_training_r),
        },
        "lm_r2_selected": {
            "alpha": float(lm_transfer_r2["alpha"]),
            "mean_r": float(lm_transfer_r2["mean_r"]),
            "mean_r2_oos": float(lm_transfer_r2["mean_r2_oos"]),
            "source_training_cv": _strip_arrays(lm_training_r2),
        },
        "held_out_reliance_correspondence": correspondences,
        "primary_correspondence": correspondences["r2_selected"],
        "within_layer_identity_null_r2_selected": {
            key: value for key, value in identity.items() if key != "null"
        },
        "directional_bank_overlap": overlap,
        "source_bank_path": str(train_bank_file),
        "reverse_bank_path": str(reverse_bank_file),
        "source_bank_bundle_sha256": train_bank_record.get("bank_bundle_sha256"),
        "reverse_bank_bundle_sha256": reverse_bank_record.get("bank_bundle_sha256"),
        "input_sha256": {str(path): engine_sha256_file(path) for path in source_paths},
        "code_sha256": {str(path): engine_sha256_file(path) for path in code_paths},
        "arrays_path": str(arrays_path),
        "elapsed_seconds": float(time.time() - started),
        "confirmatory_eligible": False,
        "temporal_target_null_status": "not_run_in_observed_transfer_stage",
    }
    atomic_write_json(summary_path, summary)
    return summary


def _bank_file(
    config_path: Path,
    spec: dict[str, Any],
    model_key: str,
    source_story: str,
    target_story: str,
) -> Path:
    return (
        _feature_bank_root(config_path, spec)
        / "cross_story"
        / model_key
        / f"{source_story}_to_{target_story}"
        / "banks.json"
    )


def _selected_models(spec: dict[str, Any], args: Any) -> list[str]:
    values = getattr(args, "models", None)
    models = [str(value) for value in (values or spec["models"].keys())]
    for model in models:
        if model not in spec["models"]:
            raise KeyError(model)
    return models


def run(*, config_path: Path, spec: dict[str, Any], args: Any) -> dict[str, Any]:
    models = _selected_models(spec, args)
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
    force = bool(getattr(args, "force", False))
    summaries: list[dict[str, Any]] = []
    for model_key in models:
        for train_story, test_story in (
            ("shapesphysical", "shapessocial"),
            ("shapessocial", "shapesphysical"),
        ):
            for variant in variants:
                for size in sizes:
                    print(
                        f"analysis transfer {model_key}/{train_story}->{test_story}/{variant}/n{size}",
                        flush=True,
                    )
                    summaries.append(
                        fit_transfer_condition(
                            config_path,
                            spec,
                            model_key,
                            train_story,
                            test_story,
                            variant,
                            int(size),
                            force=force,
                        )
                    )
    manifest_path = _artifact_root(config_path, spec) / "transfer" / "manifest.json"
    atomic_write_json(
        manifest_path,
        {
            "schema_version": 1,
            "analysis_role": "correspondence_cross_story_transfer_manifest",
            "conditions": [
                {
                    "model_key": row["model_key"],
                    "train_story": row["train_story"],
                    "test_story": row["test_story"],
                    "feature_bank_variant": row["feature_bank_variant"],
                    "bank_size": row["bank_size"],
                    "analysis_fingerprint": row["analysis_fingerprint"],
                    "summary_path": row["arrays_path"].replace("arrays.npz", "summary.json"),
                }
                for row in summaries
            ],
        },
    )
    return {"stage": "transfer", "n_conditions": len(summaries), "manifest": str(manifest_path), "summaries": summaries}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=None)
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
    # Full condition summaries are persisted to artifact JSON/NPZ files.  Do
    # not duplicate them into command output.
    print(json.dumps({key: value for key, value in result.items() if key != "summaries"}, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
