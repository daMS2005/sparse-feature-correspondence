#!/usr/bin/env python3
"""Materialize provenance-complete paired designs for the power stage.

The power analysis is deliberately downstream of the corrected observed
transfer fits.  This module never substitutes an old ``n64`` result, a
single-story fit, a target-informed bank, or synthetic arrays.  It builds one
design for each of the eight configured combinations:

``Gemma 2 2B, Llama 3.1 8B`` x
``source-story variance, source-story correlation-pruned (.80)`` x
``Physical -> Social, Social -> Physical``.

Each design stores the actual target-story predictor geometry, including the
five lagged columns used by BOLD and the unlagged columns used by the LM,
empirical residual pools taken from the corrected transfer predictions,
participant-by-time BOLD folds, TR-level LM folds, censoring masks, feature
groups, and all input hashes.  Missing or malformed corrected artifacts cause
the entire command to fail closed before any design is written.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import default_config_path
from .feature_banks import write_json  # noqa: E402
from .fitting import (  # noqa: E402
    _artifact_root,
    _extract_bank,
    _load_bank,
    bank_path,
    load_condition,
)
from .integrity import (  # noqa: E402
    load_spec,
    provenance_fingerprint,
    sha256_file,
)
from .modeling import lag_expand  # noqa: E402
from .power import (  # noqa: E402
    repeated_participant_purged_folds,
    write_paired_design_artifact,
)

HERE = Path(__file__).resolve().parent

DEFAULT_MODELS = ("gemma_2_2b", "llama_3_1_8b")
DEFAULT_DIRECTIONS = (
    ("shapesphysical", "shapessocial"),
    ("shapessocial", "shapesphysical"),
)
POWER_BANKS = {
    "source_story_variance": "source_story_variance",
    # The configured power design uses the .80 representative threshold.
    "source_story_correlation_pruned": "source_story_correlation_pruned_0p80",
}
POWER_MODELS = set(DEFAULT_MODELS)


def _scalar(value: Any) -> str:
    array = np.asarray(value).reshape(-1)
    return str(array[0]) if len(array) else ""


def _direction_label(source_story: str, target_story: str) -> str:
    return f"{source_story}_to_{target_story}"


def _target_subjects_and_censor(condition: Mapping[str, Any]) -> tuple[list[str], np.ndarray]:
    """Return sorted subjects and a subject-major censor mask from the target."""
    subject_ids = sorted(str(value) for value in condition["subject_targets"])
    masks: list[np.ndarray] = []
    for subject in subject_ids:
        target = np.asarray(condition["subject_targets"][subject], dtype=float)
        if target.ndim != 2:
            raise ValueError(f"{condition['story']}/{subject}: BOLD target must be 2D")
        masks.append(~np.all(np.isfinite(target), axis=1))
    return subject_ids, np.concatenate(masks).astype(bool, copy=False)


def _load_transfer_inputs(
    config_path: Path,
    spec: Mapping[str, Any],
    model_key: str,
    source_story: str,
    target_story: str,
    bank_family: str,
) -> dict[str, Any]:
    """Preflight one exact corrected transfer condition and its dependencies."""
    if bank_family not in POWER_BANKS:
        raise ValueError(f"unsupported power bank family {bank_family!r}")
    variant = POWER_BANKS[bank_family]
    source_condition = load_condition(config_path, dict(spec), model_key, source_story)
    target_condition = load_condition(config_path, dict(spec), model_key, target_story)
    transfer_summary = (
        _artifact_root(config_path, dict(spec))
        / "transfer"
        / f"{model_key}_{source_story}_to_{target_story}"
        / variant
        / "n256"
        / "summary.json"
    )
    transfer_arrays = transfer_summary.with_name("arrays.npz")
    if not transfer_summary.is_file() or not transfer_arrays.is_file():
        raise FileNotFoundError(
            "corrected n256 transfer artifacts are required before power design "
            f"materialization: {transfer_summary}"
        )
    summary = json.loads(transfer_summary.read_text(encoding="utf-8"))
    required_summary = {
        "analysis_role": "correspondence_cross_story_source_bank_transfer",
        "model_key": model_key,
        "train_story": source_story,
        "test_story": target_story,
        "feature_bank_variant": variant,
        "bank_size": 256,
        "source_scaling_and_weights_applied_unchanged": True,
    }
    for key, expected in required_summary.items():
        if summary.get(key) != expected:
            raise ValueError(
                f"{transfer_summary}: corrected transfer invariant {key!r}="
                f"{summary.get(key)!r} does not equal {expected!r}"
            )
    bank_file = bank_path(
        config_path,
        dict(spec),
        model_key,
        source_story,
        source_story=source_story,
        target_story=target_story,
    )
    bank_record, bank_variant = _load_bank(bank_file, variant, 256)
    if bank_record.get("mode") != "cross_story_source":
        raise ValueError(f"{bank_file}: power design requires a cross-story source bank")
    if bank_record.get("source_story") != source_story or bank_record.get("target_story") != target_story:
        raise ValueError(f"{bank_file}: source/target metadata does not match direction")
    if bank_record.get("target_excluded_from_selection") is not True:
        raise ValueError(f"{bank_file}: target story was not excluded from bank selection")
    if bank_record.get("outcome_inputs_used") is not False:
        raise ValueError(f"{bank_file}: outcome_inputs_used is not false")
    entry = bank_variant["banks"]["256"]
    bank_layers = np.asarray(entry["layers"], dtype=np.int64)
    bank_features = np.asarray(entry["feature_ids"], dtype=np.int64)
    if len(bank_layers) != 256 or len(bank_features) != 256:
        raise ValueError(f"{bank_file}: malformed K=256 bank")
    if len(set(zip(bank_layers.tolist(), bank_features.tolist(), strict=False))) != 256:
        raise ValueError(f"{bank_file}: duplicate K=256 feature-layer identities")
    with np.load(transfer_arrays, allow_pickle=True) as arrays:
        required_arrays = {
            "feature_names",
            "subject_ids",
            "brain_r2_prediction",
            "lm_r2_prediction",
        }
        missing = sorted(required_arrays.difference(arrays.files))
        if missing:
            raise ValueError(f"{transfer_arrays}: missing corrected transfer fields {missing}")
        feature_names = np.asarray(arrays["feature_names"], dtype=str).reshape(-1)
        subject_ids = np.asarray(arrays["subject_ids"], dtype=str).reshape(-1)
        brain_prediction = np.asarray(arrays["brain_r2_prediction"], dtype=float)
        lm_prediction = np.asarray(arrays["lm_r2_prediction"], dtype=float)
    if len(feature_names) != 256:
        raise ValueError(f"{transfer_arrays}: transfer feature_names are not K=256")
    if not np.array_equal(subject_ids, np.asarray(sorted(target_condition["subject_targets"]), dtype=str)):
        raise ValueError(f"{transfer_arrays}: subject ordering does not match target condition")
    target_trs = len(target_condition["tr_indices"])
    n_parcels = next(iter(target_condition["subject_targets"].values())).shape[1]
    lm_dims = int(target_condition["lm_target"].shape[1])
    if brain_prediction.shape != (len(subject_ids), target_trs, n_parcels):
        raise ValueError(
            f"{transfer_arrays}: brain_r2_prediction shape {brain_prediction.shape} "
            f"does not match {(len(subject_ids), target_trs, n_parcels)}"
        )
    if lm_prediction.shape != (target_trs, lm_dims):
        raise ValueError(
            f"{transfer_arrays}: lm_r2_prediction shape {lm_prediction.shape} "
            f"does not match {(target_trs, lm_dims)}"
        )
    if not np.all(np.isfinite(lm_prediction)):
        raise ValueError(f"{transfer_arrays}: corrected LM transfer predictions contain non-finite values")
    # The observed transfer writer stores NaN predictions at censored BOLD
    # rows.  Those rows are retained in the full participant-by-time design
    # but excluded by the fit; non-finite predictions at an uncensored row are
    # still a hard provenance/data error.
    for subject_index, subject in enumerate(subject_ids.tolist()):
        actual = np.asarray(target_condition["subject_targets"][str(subject)], dtype=float)
        uncensored = np.all(np.isfinite(actual), axis=1)
        if not np.all(np.isfinite(brain_prediction[subject_index][uncensored])):
            raise ValueError(
                f"{transfer_arrays}: non-finite BOLD transfer prediction at an uncensored row for {subject}"
            )
    return {
        "model_key": model_key,
        "source_story": source_story,
        "target_story": target_story,
        "bank_family": bank_family,
        "bank_variant": variant,
        "source_condition": source_condition,
        "target_condition": target_condition,
        "transfer_summary_path": transfer_summary,
        "transfer_arrays_path": transfer_arrays,
        "transfer_summary": summary,
        "bank_path": bank_file,
        "bank_record": bank_record,
        "bank_variant_record": bank_variant,
        "feature_names": feature_names,
        "subject_ids": subject_ids,
        "brain_prediction": brain_prediction,
        "lm_prediction": lm_prediction,
    }


def _build_design_from_inputs(
    config_path: Path,
    spec: Mapping[str, Any],
    inputs: Mapping[str, Any],
    *,
    output_path: Path,
) -> dict[str, Any]:
    source_story = str(inputs["source_story"])
    target_story = str(inputs["target_story"])
    model_key = str(inputs["model_key"])
    bank_family = str(inputs["bank_family"])
    target = inputs["target_condition"]
    bank_variant_record = inputs["bank_variant_record"]
    # Apply the source-selected identities to the target-story predictors.
    base_x, names = _extract_bank(target, bank_variant_record, 256)
    if not np.array_equal(names, np.asarray(inputs["feature_names"], dtype=str)):
        raise ValueError("target extraction changed corrected transfer feature identities")
    if not np.array_equal(
        np.asarray(bank_variant_record["banks"]["256"]["layers"], dtype=np.int64),
        np.asarray(target["selected_layers"], dtype=np.int64)[
            [list(target["selected_layers"]).index(int(layer)) for layer in bank_variant_record["banks"]["256"]["layers"]]
        ],
    ):
        # The explicit _extract_bank layer check is the authoritative check;
        # this branch is retained only to make a mismatched layer map fail
        # closed rather than relying on an implicit column order.
        raise ValueError("source bank contains a layer absent from target predictor bundle")
    lm_x = np.asarray(base_x, dtype=np.float64)
    brain_x_base = np.asarray(base_x, dtype=np.float64)
    lags = [int(value) for value in spec["targets"]["brain"]["lags"]]
    brain_x_one_subject = lag_expand(brain_x_base, lags)
    subject_ids = [str(value) for value in inputs["subject_ids"].tolist()]
    n_subjects = len(subject_ids)
    n_time = len(target["tr_indices"])
    brain_x = np.tile(brain_x_one_subject, (n_subjects, 1))
    brain_targets: list[np.ndarray] = []
    brain_residuals: list[np.ndarray] = []
    brain_censor_parts: list[np.ndarray] = []
    for subject_index, subject in enumerate(subject_ids):
        actual = np.asarray(target["subject_targets"][subject], dtype=float)
        prediction = np.asarray(inputs["brain_prediction"][subject_index], dtype=float)
        if actual.shape != prediction.shape:
            raise ValueError(f"{target_story}/{subject}: target/prediction shape mismatch")
        censor = ~np.all(np.isfinite(actual), axis=1)
        residual = actual - prediction
        finite = np.all(np.isfinite(residual), axis=1) & ~censor
        if int(np.sum(finite)) < 2:
            raise ValueError(f"{target_story}/{subject}: too few finite empirical residual rows")
        brain_targets.append(actual)
        brain_residuals.append(residual[finite])
        brain_censor_parts.append(censor)
    brain_censor = np.concatenate(brain_censor_parts).astype(bool, copy=False)
    brain_residual = np.concatenate(brain_residuals, axis=0).astype(np.float32, copy=False)
    lm_actual = np.asarray(target["lm_target"], dtype=float)
    lm_prediction = np.asarray(inputs["lm_prediction"], dtype=float)
    lm_residual = lm_actual - lm_prediction
    lm_finite = np.all(np.isfinite(lm_residual), axis=1)
    if int(np.sum(lm_finite)) < 2:
        raise ValueError(f"{target_story}: too few finite empirical LM residual rows")
    lm_residual = lm_residual[lm_finite].astype(np.float32, copy=False)
    lm_x = lm_x.astype(np.float32, copy=False)
    brain_x = brain_x.astype(np.float32, copy=False)
    brain_folds = repeated_participant_purged_folds(
        n_subjects,
        n_time,
        n_folds=int(spec["cross_validation"]["outer_folds"]),
        purge_adjacent=int(spec["cross_validation"]["purge_adjacent_trs"]),
    )
    from .power import contiguous_purged_folds

    lm_folds = contiguous_purged_folds(
        n_time,
        n_folds=int(spec["cross_validation"]["outer_folds"]),
        purge_adjacent=int(spec["cross_validation"]["purge_adjacent_trs"]),
    )
    brain_groups = [
        np.arange(index * len(lags), (index + 1) * len(lags), dtype=np.int64)
        for index in range(256)
    ]
    lm_groups = [np.asarray([index], dtype=np.int64) for index in range(256)]
    direction = _direction_label(source_story, target_story)
    design_label = f"{model_key}/{bank_family}/{direction}"
    source_artifacts = {
        "source_predictor": Path(inputs["source_condition"]["predictor_path"]),
        "target_predictor": Path(inputs["target_condition"]["predictor_path"]),
        "source_bold": Path(inputs["source_condition"]["brain_path"]),
        "target_bold": Path(inputs["target_condition"]["brain_path"]),
        "feature_bank": Path(inputs["bank_path"]),
        "transfer_summary": Path(inputs["transfer_summary_path"]),
        "transfer_arrays": Path(inputs["transfer_arrays_path"]),
        "analysis_config": config_path,
    }
    source_paths = list(source_artifacts.values())
    code_paths = [
        HERE / "power_designs.py",
        HERE / "power.py",
        HERE / "fitting.py",
        HERE / "feature_banks.py",
        HERE / "modeling.py",
    ]
    source_hashes = {
        role: sha256_file(path) for role, path in source_artifacts.items()
    }
    code_hashes = {path.name: sha256_file(path) for path in code_paths}
    fingerprint = provenance_fingerprint(
        config_path,
        input_paths=source_paths,
        code_paths=code_paths,
        extra={
            "analysis": "correspondence_paired_power_design",
            "design_label": design_label,
            "bank_variant": str(inputs["bank_variant"]),
            "target_story": target_story,
            "rows": {"brain": int(len(brain_x)), "lm": int(len(lm_x))},
        },
    )
    metadata = {
        "design_label": design_label,
        "model_key": model_key,
        "bank_type": bank_family,
        "bank_variant": str(inputs["bank_variant"]),
        "direction": direction,
        "source_story": source_story,
        "target_story": target_story,
        "participant_structure": "paired_system_rows",
        "censoring_manifest": "target_bold",
        "evaluation_regime": "target_story_transfer_geometry",
        "purge_adjacent_trs": str(spec["cross_validation"]["purge_adjacent_trs"]),
        "row_order": "subject_major_then_tr_within_subject_for_brain; tr_order_for_lm",
        "feature_identity_source": "feature_bank",
        "analysis_fingerprint": fingerprint,
    }
    provenance = {
        "source_hashes": source_hashes,
        "code_hashes": code_hashes,
        "analysis_config_sha256": sha256_file(config_path),
        "transfer_analysis_fingerprint": inputs["transfer_summary"].get("analysis_fingerprint", ""),
        "bank_bundle_sha256": inputs["bank_record"].get("bank_bundle_sha256", ""),
        "brain_censor_definition": "target subject BOLD rows with any non-finite parcel are censored; stored in full participant-major grid",
        "residual_definition": "target observation minus corrected transfer r2-selected prediction; finite uncensored rows form empirical residual pool",
        "brain_lag_definition": lags,
        "feature_group_definition": "one SAE feature per LM column; all five BOLD lag columns per SAE feature",
    }
    result = write_paired_design_artifact(
        output_path,
        brain_x=brain_x,
        lm_x=lm_x,
        brain_residual=brain_residual,
        lm_residual=lm_residual,
        brain_participant_ids=np.repeat(np.asarray(subject_ids, dtype=str), n_time),
        lm_participant_ids=np.asarray(["shared_stimulus"] * n_time, dtype=str),
        brain_censor_mask=brain_censor,
        lm_censor_mask=np.zeros(n_time, dtype=bool),
        brain_folds=brain_folds,
        lm_folds=lm_folds,
        brain_feature_groups=brain_groups,
        lm_feature_groups=lm_groups,
        metadata=metadata,
        provenance=provenance,
    )
    result["source_hashes"] = source_hashes
    result["code_hashes"] = code_hashes
    result["analysis_fingerprint"] = fingerprint
    result["bank_identity"] = {
        "layers": [int(value) for value in bank_variant_record["banks"]["256"]["layers"]],
        "feature_ids": [int(value) for value in bank_variant_record["banks"]["256"]["feature_ids"]],
        "feature_names": [str(value) for value in names.tolist()],
    }
    write_json(output_path.with_suffix(".json"), result)
    return result


def materialize(
    config_path: Path,
    spec: Mapping[str, Any],
    *,
    models: Sequence[str] = DEFAULT_MODELS,
    directions: Sequence[tuple[str, str]] = DEFAULT_DIRECTIONS,
    bank_families: Sequence[str] = tuple(POWER_BANKS),
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    models = [str(value) for value in models]
    bank_families = [str(value) for value in bank_families]
    for model in models:
        if model not in POWER_MODELS:
            raise ValueError(f"power design model {model!r} is not allowed")
        if model not in spec.get("models", {}):
            raise ValueError(f"model {model!r} is absent from analysis configuration")
    for family in bank_families:
        if family not in POWER_BANKS:
            raise ValueError(f"power bank family {family!r} is not allowed")
    normalized_directions = [(str(source), str(target)) for source, target in directions]
    for source, target in normalized_directions:
        if (source, target) not in DEFAULT_DIRECTIONS:
            raise ValueError(f"power direction {(source, target)!r} is not allowed")
    # Preflight every requested condition before writing any design.  This
    # makes a partial release impossible when one transfer array is unavailable.
    preflight: list[dict[str, Any]] = []
    failures: list[str] = []
    for model in models:
        for bank_family in bank_families:
            for source_story, target_story in normalized_directions:
                try:
                    preflight.append(
                        _load_transfer_inputs(
                            config_path,
                            spec,
                            model,
                            source_story,
                            target_story,
                            bank_family,
                        )
                    )
                except Exception as exc:  # noqa: BLE001 - report every missing gate
                    failures.append(f"{model}/{bank_family}/{source_story}->{target_story}: {exc}")
    output_root = _artifact_root(config_path, dict(spec)) / "power" / "designs"
    if failures:
        status = {
            "schema_version": 2,
            "status": "blocked_pending_corrected_transfer_artifacts",
            "scientific_evidence": False,
            "requested_design_count": len(models) * len(bank_families) * len(normalized_directions),
            "failures": failures,
            "required_bank_variant_map": POWER_BANKS,
        }
        output_root.mkdir(parents=True, exist_ok=True)
        write_json(output_root / "materialization_status.json", status)
        raise RuntimeError("power design materialization is fail-closed:\n" + "\n".join(failures))
    if dry_run:
        return {
            "schema_version": 2,
            "status": "preflight_complete",
            "scientific_evidence": False,
            "design_count": len(preflight),
            "design_labels": [
                f"{row['model_key']}/{row['bank_family']}/{_direction_label(row['source_story'], row['target_story'])}"
                for row in preflight
            ],
        }
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for inputs in preflight:
        label = (
            f"{inputs['model_key']}__{inputs['bank_family']}__"
            f"{_direction_label(inputs['source_story'], inputs['target_story'])}"
        )
        output_path = output_root / f"{label}.npz"
        if output_path.exists() and not force:
            # Do not silently reuse a design merely because its label matches:
            # the upstream transfer arrays, bank, code, or configured config may
            # have changed.  Rebuilding with --force recomputes the complete
            # input/code fingerprint and replaces the paired file.
            raise RuntimeError(
                f"existing power design cache requires explicit --force for "
                f"fingerprint revalidation: {output_path}"
            )
        records.append(
            _build_design_from_inputs(
                config_path,
                spec,
                inputs,
                output_path=output_path,
            )
        )
    index_path = output_root / "design_index.tsv"
    lines = ["design_label\tmodel_key\tbank_type\tbank_variant\tdirection\tpath\tsha256"]
    for record in records:
        lines.append(
            "\t".join(
                [
                    str(record["design_label"]),
                    str(record["model_key"]),
                    str(record["bank_type"]),
                    str(record.get("bank_variant", "")),
                    str(record["direction"]),
                    str(record["path"]),
                    str(record["sha256"]),
                ]
            )
        )
    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 2,
        "status": "complete",
        "scientific_evidence": False,
        "analysis_role": "correspondence_paired_power_design_manifest",
        "design_count": len(records),
        "expected_design_count": len(models) * len(bank_families) * len(normalized_directions),
        "models": models,
        "bank_families": bank_families,
        "bank_variant_map": POWER_BANKS,
        "directions": [list(value) for value in normalized_directions],
        "records": records,
        "index_path": str(index_path),
        "config_sha256": sha256_file(config_path),
    }
    write_json(output_root / "design_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=["shapesphysical_to_shapessocial", "shapessocial_to_shapesphysical"],
        default=[_direction_label(*direction) for direction in DEFAULT_DIRECTIONS],
    )
    parser.add_argument("--bank-families", nargs="+", default=list(POWER_BANKS))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.data_root is not None:
        os.environ["FMRI_SAE_DATA_ROOT"] = str(args.data_root.resolve())
    if args.output_root is not None:
        os.environ["FMRI_SAE_OUTPUT_ROOT"] = str(args.output_root.resolve())
    config_path = (args.config or default_config_path()).resolve()
    spec = load_spec(config_path)
    directions = []
    for label in args.directions:
        source, target = label.split("_to_", 1)
        directions.append((source, target))
    result = materialize(
        config_path,
        spec,
        models=args.models,
        directions=directions,
        bank_families=args.bank_families,
        force=args.force,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
