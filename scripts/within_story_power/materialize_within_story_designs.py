#!/usr/bin/env python3
"""Materialize within-story semi-synthetic power designs (exploratory sensitivity).

The configured power stage calibrates cross-story transfer.  This script builds
the analogous design for the within-story analysis so that the within-story null
can be bounded on the same terms.  It reads the same observed artifacts the
package writes and never modifies the configured power stage.

The within-story analysis reselects its feature bank inside every outer fold,
so it has no single feature axis.  A power design requires one, so this design
fixes the bank to the whole-story variance bank at K=256 (the bank the
cross-story stage selects when the story is the source).  Fixing the bank
removes the variance that per-fold reselection contributes; the resulting
detection fractions are therefore upper bounds on within-story detectability.
Residual pools come from the within-story out-of-fold predictions and live in
target space, so they do not depend on which bank produced them.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from fmri_sae_correspondence.config import default_config_path, load_config
from fmri_sae_correspondence.fitting import (
    _artifact_root,
    _extract_bank,
    _load_bank,
    load_condition,
)
from fmri_sae_correspondence.integrity import provenance_fingerprint, sha256_file
from fmri_sae_correspondence.modeling import lag_expand
from fmri_sae_correspondence.power import (
    contiguous_purged_folds,
    repeated_participant_purged_folds,
)

OTHER = {"shapesphysical": "shapessocial", "shapessocial": "shapesphysical"}
BANK_FAMILY = "source_story_variance"
BANK_SIZE = 256


def build(config_path: Path, spec: Mapping[str, Any], model_key: str, story: str, out_dir: Path) -> dict[str, Any]:
    artifacts = _artifact_root(config_path, dict(spec))
    condition = load_condition(config_path, dict(spec), model_key, story)

    bank_path = (
        artifacts / "feature_banks" / "cross_story" / model_key / f"{story}_to_{OTHER[story]}" / "banks.json"
    )
    if not bank_path.is_file():
        raise FileNotFoundError(f"whole-story bank missing: {bank_path}")
    bank_record, bank_variant = _load_bank(bank_path, BANK_FAMILY, BANK_SIZE)
    base_x, names = _extract_bank(condition, bank_variant, BANK_SIZE)
    if base_x.shape[1] != BANK_SIZE:
        raise ValueError(f"expected K={BANK_SIZE}, got {base_x.shape[1]}")

    fit_arrays = artifacts / "fit" / f"{model_key}_{story}" / BANK_FAMILY / f"n{BANK_SIZE}" / "arrays.npz"
    if not fit_arrays.is_file():
        raise FileNotFoundError(f"within-story fit arrays missing: {fit_arrays}")
    with np.load(fit_arrays, allow_pickle=True) as fitted:
        brain_oof = np.asarray(fitted["brain_r2_oof"], dtype=float)
        lm_oof = np.asarray(fitted["lm_r2_oof"], dtype=float)
        subject_ids = [str(value) for value in fitted["subject_ids"].tolist()]

    lags = [int(value) for value in spec["targets"]["brain"]["lags"]]
    n_time = len(condition["tr_indices"])
    n_subjects = len(subject_ids)
    if brain_oof.shape[:2] != (n_subjects, n_time):
        raise ValueError(f"out-of-fold shape {brain_oof.shape} does not match {n_subjects} x {n_time}")

    brain_x = np.tile(lag_expand(np.asarray(base_x, dtype=np.float64), lags), (n_subjects, 1))
    brain_x = brain_x.astype(np.float32)
    lm_x = np.asarray(base_x, dtype=np.float32)

    residuals: list[np.ndarray] = []
    censor_parts: list[np.ndarray] = []
    for index, subject in enumerate(subject_ids):
        actual = np.asarray(condition["subject_targets"][subject], dtype=float)
        prediction = brain_oof[index]
        if actual.shape != prediction.shape:
            raise ValueError(f"{story}/{subject}: target {actual.shape} vs prediction {prediction.shape}")
        censor = ~np.all(np.isfinite(actual), axis=1)
        residual = actual - prediction
        finite = np.all(np.isfinite(residual), axis=1) & ~censor
        if int(finite.sum()) < 2:
            raise ValueError(f"{story}/{subject}: too few finite residual rows")
        residuals.append(residual[finite])
        censor_parts.append(censor)
    brain_residual = np.concatenate(residuals, axis=0).astype(np.float32)
    brain_censor = np.concatenate(censor_parts).astype(bool)

    lm_residual = np.asarray(condition["lm_target"], dtype=float) - lm_oof
    lm_finite = np.all(np.isfinite(lm_residual), axis=1)
    if int(lm_finite.sum()) < 2:
        raise ValueError(f"{story}: too few finite LM residual rows")
    lm_residual = lm_residual[lm_finite].astype(np.float32)

    n_folds = int(spec["cross_validation"]["outer_folds"])
    purge = int(spec["cross_validation"]["purge_adjacent_trs"])
    brain_folds = repeated_participant_purged_folds(n_subjects, n_time, n_folds=n_folds, purge_adjacent=purge)
    lm_folds = contiguous_purged_folds(n_time, n_folds=n_folds, purge_adjacent=purge)
    brain_groups = [
        np.arange(index * len(lags), (index + 1) * len(lags), dtype=np.int64) for index in range(BANK_SIZE)
    ]
    lm_groups = [np.asarray([index], dtype=np.int64) for index in range(BANK_SIZE)]

    direction = f"within_{story}"
    design_label = f"{model_key}/{BANK_FAMILY}/{direction}"
    fingerprint = provenance_fingerprint(
        config_path,
        input_paths=[Path(condition["predictor_path"]), Path(condition["brain_path"]), bank_path, fit_arrays],
        code_paths=[Path(__file__)],
        extra={"analysis": "within_story_paired_power_design", "design_label": design_label},
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{model_key}__{BANK_FAMILY}__within_{story}.npz"
    np.savez(
        out_path,
        schema_version=np.asarray([2], dtype=np.int32),
        design_schema=np.asarray(["paired_target_design_v1"]),
        brain_x=brain_x,
        lm_x=lm_x,
        brain_residual=brain_residual,
        lm_residual=lm_residual,
        brain_participant_ids=np.asarray(np.repeat(subject_ids, n_time)),
        lm_participant_ids=np.asarray([f"lm_tr_{index:04d}" for index in range(n_time)]),
        brain_censor_mask=brain_censor,
        lm_censor_mask=np.zeros(n_time, dtype=bool),
        brain_folds=np.asarray(brain_folds, dtype=object),
        lm_folds=np.asarray(lm_folds, dtype=object),
        brain_feature_groups=np.asarray(brain_groups, dtype=object),
        lm_feature_groups=np.asarray(lm_groups, dtype=object),
        design_label=np.asarray([design_label]),
        model_key=np.asarray([model_key]),
        bank_type=np.asarray([BANK_FAMILY]),
        bank_variant=np.asarray([BANK_FAMILY]),
        direction=np.asarray([direction]),
        source_story=np.asarray([story]),
        target_story=np.asarray([story]),
        participant_structure=np.asarray(["paired_system_rows"]),
        censoring_manifest=np.asarray([Path(condition["brain_path"]).name]),
        evaluation_regime=np.asarray(["within_story_held_out_time"]),
        purge_adjacent_trs=np.asarray([str(purge)]),
        row_order=np.asarray(["subject_major_then_tr_within_subject_for_brain; tr_order_for_lm"]),
        feature_identity_source=np.asarray(
            [f"whole-story variance bank K={BANK_SIZE} from {bank_path.name}; fold reselection not simulated"]
        ),
        analysis_fingerprint=np.asarray([str(fingerprint)]),
        residual_definition=np.asarray(
            ["within-story observation minus R2-selected out-of-fold prediction; finite uncensored rows"]
        ),
        design_caveat=np.asarray(
            ["bank fixed to the whole story while the observed analysis reselects per fold; "
             "detection fractions are upper bounds on within-story detectability"]
        ),
        bank_bundle_sha256=np.asarray([str(bank_record.get("bank_bundle_sha256", ""))]),
        config_sha256=np.asarray([sha256_file(config_path)]),
    )
    return {
        "design_label": design_label,
        "path": str(out_path),
        "design_sha256": sha256_file(out_path),
        "brain_x": list(brain_x.shape),
        "lm_x": list(lm_x.shape),
        "brain_residual": list(brain_residual.shape),
        "lm_residual": list(lm_residual.shape),
        "n_subjects": n_subjects,
        "n_time": n_time,
        "feature_names": int(len(names)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["gemma_2_2b", "llama_3_1_8b"])
    parser.add_argument("--stories", nargs="+", default=["shapesphysical", "shapessocial"])
    args = parser.parse_args()
    config_path = (args.config or default_config_path()).resolve()
    spec = load_config(config_path)
    rows = [build(config_path, spec, model, story, args.out_dir) for model in args.models for story in args.stories]
    print(json.dumps({"designs": len(rows), "rows": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
