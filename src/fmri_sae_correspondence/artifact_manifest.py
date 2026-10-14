#!/usr/bin/env python3
"""Build an aggregate, condition-level manifest for analysis outputs.

The individual analysis stages write stage-level manifests, while this module
treats per-condition ``summary.json`` files as authoritative and derives an
aggregate inventory from the documented analysis configuration.

The inventory is deliberately useful while the analysis is incomplete.  It
records missing and invalid conditions without pretending that they are
scientific evidence.  Consumers that need publication inputs must call
``require_complete_manifest``; it raises with an actionable list instead of
silently plotting a partial result.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .config import default_config_path
from .integrity import (
    AnalysisError,
    atomic_write_json,
    config_digest,
    load_spec,
    provenance_fingerprint,
    resolve_data_root,
    resolve_output_root,
    resolve_path,
    sha256_file,
)

HERE = Path(__file__).resolve().parent
# ``ExpectedArtifact.role`` is this module's internal grouping label; stored
# analysis artifacts use the ``correspondence_*`` role vocabulary.
# The two are not interchangeable, so map one onto the other rather than
# comparing an internal label against a recorded field.
PRIMARY_CONTROL_ANALYSIS_ROLES = {
    "primary_source_story_control_within": "correspondence_primary_source_story_controls_within",
    "primary_source_story_control_transfer": "correspondence_primary_source_story_controls_transfer",
}


class ManifestIncomplete(AnalysisError):
    """Raised when a consumer requests manuscript-ready artifacts too early."""


@dataclass(frozen=True)
class ExpectedArtifact:
    stage: str
    role: str
    key: str
    path: Path
    model_key: str | None = None
    story: str | None = None
    train_story: str | None = None
    test_story: str | None = None
    variant: str | None = None
    size: int | None = None
    threshold: float | None = None


def artifact_root(config_path: Path, spec: Mapping[str, Any]) -> Path:
    """Resolve the new artifact root independently of the caller's cwd."""

    return resolve_path(config_path, spec, "artifact_root", root="output")


def variants(spec: Mapping[str, Any]) -> list[str]:
    """Return the exact variant labels emitted by the analysis stages."""

    # The analysis config stores correlation-pruned thresholds under the bank
    # list.  Keep this lookup defensive so a temporary test spec can provide
    # the same contract in either form.
    bank_entries = spec.get("feature_bank", {}).get("banks", [])
    values: Sequence[Any] = ()
    if isinstance(bank_entries, Sequence):
        for entry in bank_entries:
            if isinstance(entry, Mapping) and entry.get("name") == "source_story_correlation_pruned":
                raw = entry.get("thresholds", ())
                if isinstance(raw, Sequence):
                    values = raw
                break
    if not values:
        values = (0.70, 0.80, 0.90)
    result = ["source_story_variance", "source_story_prevalence"]
    result.extend(
        f"source_story_correlation_pruned_{float(value):.2f}".replace(".", "p")
        for value in values
    )
    return result


def sizes(spec: Mapping[str, Any]) -> list[int]:
    return [int(value) for value in spec["feature_bank"]["sizes"]]


def variant_sizes(spec: Mapping[str, Any]) -> dict[str, list[int]]:
    """Return the condition sizes that are publication-required per bank.

    Source-variance and source-prevalence banks are nested primary/sensitivity
    curves, so all configured ``K`` values are required for their observed
    fits.  A correlation-pruned bank is a representative bank and can contain
    fewer than the requested number of representatives after clustering.  The
    analysis config uses that bank for the grouped-reliance sensitivity at
    ``K=256``; requiring an impossible ``K=1024`` representative bank would
    turn a valid underidentified sensitivity into a permanently incomplete
    release.  The bank generator still records every requested K and its
    actual size, but only K=256 is a publication condition for this variant.
    """

    all_sizes = sizes(spec)
    if 256 not in all_sizes:
        raise ManifestIncomplete("configured feature-bank size grid must include K=256")
    return {
        "source_story_variance": list(all_sizes),
        "source_story_prevalence": list(all_sizes),
        **{
            variant: [256]
            for variant in variants(spec)
            if variant.startswith("source_story_correlation_pruned_")
        },
    }


def directions(spec: Mapping[str, Any]) -> list[tuple[str, str]]:
    stories = list(spec["stories"].keys())
    if set(stories) != {"shapesphysical", "shapessocial"}:
        raise ManifestIncomplete(f"unexpected story set in analysis configuration: {stories}")
    return [("shapesphysical", "shapessocial"), ("shapessocial", "shapesphysical")]


def _condition(
    stage: str,
    role: str,
    key: str,
    path: Path,
    *,
    model_key: str | None = None,
    story: str | None = None,
    train_story: str | None = None,
    test_story: str | None = None,
    variant: str | None = None,
    size: int | None = None,
    threshold: float | None = None,
) -> ExpectedArtifact:
    return ExpectedArtifact(
        stage=stage,
        role=role,
        key=key,
        path=path,
        model_key=model_key,
        story=story,
        train_story=train_story,
        test_story=test_story,
        variant=variant,
        size=size,
        threshold=threshold,
    )


def expected_artifacts(config_path: Path, spec: Mapping[str, Any]) -> list[ExpectedArtifact]:
    """Enumerate publication-required condition summaries from the spec.

    The inventory is intentionally explicit.  It includes every observed
    fit/transfer condition and the nulls required by the revised plan:
    primary variance banks at all sizes, prevalence at 256/1024, and each
    correlation-pruned threshold at 256.  Grouped reliance and its retuned
    temporal nulls are separately required at 256.  The participant-average
    BOLD sensitivity is included as an explicit appendix condition because it
    must either be completed with its fully retuned temporal null or be
    excluded from the release.  Historical-bank outputs are not required for
    the primary figure gate; they remain appendix sensitivities.  The primary
    source-story control comparison is also enumerated as its own stage: 12
    within-story and 12 directional-transfer conditions.  This keeps the
    baseline-versus-multiband evidence from being confused with historical,
    dense, or random controls.
    """

    root = artifact_root(config_path, spec)
    model_keys = [str(value) for value in spec["models"].keys()]
    stories = [str(value) for value in spec["stories"].keys()]
    all_sizes = sizes(spec)
    requested_sizes = variant_sizes(spec)
    result: list[ExpectedArtifact] = []

    for model in model_keys:
        for story in stories:
            for variant, variant_k in requested_sizes.items():
                for size in variant_k:
                    path = root / "fit" / f"{model}_{story}" / variant / f"n{size}" / "summary.json"
                    result.append(_condition("fit", "observed_within_story", f"{model}/{story}/{variant}/n{size}", path, model_key=model, story=story, variant=variant, size=size))

    for model in model_keys:
        for source, target in directions(spec):
            for variant, variant_k in requested_sizes.items():
                for size in variant_k:
                    path = root / "transfer" / f"{model}_{source}_to_{target}" / variant / f"n{size}" / "summary.json"
                    result.append(_condition("transfer", "observed_cross_story", f"{model}/{source}->{target}/{variant}/n{size}", path, model_key=model, train_story=source, test_story=target, variant=variant, size=size))

    # Primary source-story controls are written by controls.py below
    # primary_controls/.  The control implementation calls the model
    # containing low-level/position, surprisal, and SAE blocks ``full`` in its
    # per-fold records; the validator below maps that name to the declared
    # multiband model and checks both independent selection metrics.
    for model in model_keys:
        for story in stories:
            path = root / "primary_controls" / "within" / f"{model}_{story}" / "source_story_variance" / "n256" / "summary.json"
            result.append(
                _condition(
                    "primary_controls",
                    "primary_source_story_control_within",
                    f"within/{model}/{story}/source_story_variance/n256",
                    path,
                    model_key=model,
                    story=story,
                    variant="source_story_variance",
                    size=256,
                )
            )
        for source, target in directions(spec):
            path = root / "primary_controls" / "transfer" / f"{model}_{source}_to_{target}" / "source_story_variance" / "n256" / "summary.json"
            result.append(
                _condition(
                    "primary_controls",
                    "primary_source_story_control_transfer",
                    f"transfer/{model}/{source}->{target}/source_story_variance/n256",
                    path,
                    model_key=model,
                    train_story=source,
                    test_story=target,
                    variant="source_story_variance",
                    size=256,
                )
            )

    primary = "source_story_variance"
    null_pairs = [(primary, size) for size in all_sizes]
    null_pairs.extend([("source_story_prevalence", 256), ("source_story_prevalence", 1024)])
    pruned_thresholds = [float(value) for value in (0.70, 0.80, 0.90)]
    null_pairs.extend(
        (f"source_story_correlation_pruned_{value:.2f}".replace(".", "p"), 256)
        for value in pruned_thresholds
    )
    # Grouped temporal nulls are emitted under grouped_* variant labels and
    # are required for the grouped-reliance sensitivity.
    grouped_pairs = [
        (f"threshold_{value:.2f}".replace(".", "p"), 256)
        for value in pruned_thresholds
    ]

    for model in model_keys:
        for story in stories:
            for variant, size in null_pairs:
                path = root / "nulls" / "within_story" / story / model / variant / f"n{size}" / "summary.json"
                result.append(_condition("nulls", "temporal_null_within_story", f"within/{model}/{story}/{variant}/n{size}", path, model_key=model, story=story, variant=variant, size=size))
        for source, target in directions(spec):
            for variant, size in null_pairs:
                path = root / "nulls" / "cross_story" / f"{source}_to_{target}" / model / variant / f"n{size}" / "summary.json"
                result.append(_condition("nulls", "temporal_null_cross_story", f"cross/{model}/{source}->{target}/{variant}/n{size}", path, model_key=model, train_story=source, test_story=target, variant=variant, size=size))
            for variant, size in grouped_pairs:
                label = f"{source}_to_{target}"
                path = root / "nulls" / "grouped" / "cross_story" / label / model / variant / f"n{size}" / "summary.json"
                result.append(_condition("nulls", "temporal_null_grouped_cross_story", f"grouped_cross/{model}/{label}/{variant}/n{size}", path, model_key=model, train_story=source, test_story=target, variant=variant, size=size))
        for variant, size in grouped_pairs:
            for story in stories:
                concrete = root / "nulls" / "grouped" / "within_story" / story / model / variant / f"n{size}" / "summary.json"
                result.append(_condition("nulls", "temporal_null_grouped_within_story", f"grouped_within/{model}/{story}/{variant}/n{size}", concrete, model_key=model, story=story, variant=variant, size=size))

    for model in model_keys:
        for story in stories:
            for threshold in pruned_thresholds:
                variant = f"source_story_correlation_pruned_{threshold:.2f}".replace(".", "p")
                path = root / "grouped" / "within_story" / story / model / variant / "n256" / "summary.json"
                result.append(_condition("grouped", "grouped_reliance_within_story", f"within/{model}/{story}/{variant}/n256", path, model_key=model, story=story, variant=variant, size=256, threshold=threshold))
        for source, target in directions(spec):
            label = f"{source}_to_{target}"
            for threshold in pruned_thresholds:
                variant = f"source_story_correlation_pruned_{threshold:.2f}".replace(".", "p")
                path = root / "grouped" / "cross_story" / label / model / variant / "n256" / "summary.json"
                result.append(_condition("grouped", "grouped_reliance_cross_story", f"cross/{model}/{label}/{variant}/n256", path, model_key=model, train_story=source, test_story=target, variant=variant, size=256, threshold=threshold))

    # The participant-average BOLD analysis is not confirmatory, but it is an
    # explicitly planned appendix sensitivity.  Treating each model/story
    # result as a condition prevents a stale group-average manifest from
    # making the release look complete when one condition is absent.
    for model in model_keys:
        for story in stories:
            path = root / "group_average" / model / story / "source_story_variance" / "n256" / "summary.json"
            result.append(_condition("group_average", "participant_average_bold_sensitivity", f"{model}/{story}/source_story_variance/n256", path, model_key=model, story=story, variant="source_story_variance", size=256))
    return result


def canonicalize_evidence_path(
    raw: str | os.PathLike[str],
    config_path: Path,
    spec: Mapping[str, Any],
    *,
    expected_summary_path: Path | None = None,
    arrays: bool = False,
) -> tuple[Path, str]:
    """Resolve evidence only within the configured data and output roots."""

    original = Path(str(raw))
    data_root = resolve_data_root(config_path, spec)
    output_root = resolve_output_root(config_path, spec)
    if not original.is_absolute():
        candidates = []
        if expected_summary_path is not None:
            candidates.append(Path(expected_summary_path).resolve().parent / original)
        candidates.extend((output_root / original, data_root / original))
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve(), "configured_relative"
        return candidates[0].resolve(), "configured_relative_missing"

    resolved = original.resolve()
    for root, reason in ((output_root, "configured_output"), (data_root, "configured_data")):
        if resolved.is_relative_to(root.resolve()):
            return resolved, reason
    return resolved, "unmapped_absolute"


def _resolve_summary_ref(
    raw: str | os.PathLike[str],
    config_path: Path,
    spec: Mapping[str, Any],
    *,
    expected_summary_path: Path | None = None,
    arrays: bool = False,
) -> Path:
    """Resolve a summary reference using the strict portability layer."""

    return canonicalize_evidence_path(
        raw,
        config_path,
        spec,
        expected_summary_path=expected_summary_path,
        arrays=arrays,
    )[0]


def _summary_valid(
    expected: ExpectedArtifact,
    *,
    config_path: Path,
    spec: Mapping[str, Any],
    full_hashes: bool,
) -> tuple[str, list[str], dict[str, Any] | None]:
    """Validate one summary and its direct artifact references."""

    path = expected.path.resolve()
    if not path.is_file():
        return "missing", ["summary.json not found"], None
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "invalid", [f"unreadable summary: {exc}"], None
    if not isinstance(summary, Mapping):
        return "invalid", ["summary is not a JSON object"], None
    problems: list[str] = []
    if int(summary.get("schema_version", 0)) < 1:
        problems.append("missing schema_version")
    if not summary.get("analysis_fingerprint"):
        problems.append("missing analysis_fingerprint")
    if expected.stage in {"fit", "transfer"}:
        if summary.get("feature_bank_variant") != expected.variant:
            problems.append(f"feature_bank_variant={summary.get('feature_bank_variant')!r}, expected {expected.variant!r}")
        if int(summary.get("bank_size", -1)) != int(expected.size or -1):
            problems.append(f"bank_size={summary.get('bank_size')!r}, expected {expected.size}")
        if summary.get("feature_selection_outcome_free") is False or summary.get("source_bank_selection_only") is False:
            problems.append("feature selection is not marked outcome-free")
        if summary.get("paired_feature_coordinates_within_fold") is False or summary.get("paired_targets_use_exact_same_feature_layer_coordinates") is False:
            problems.append("paired feature-coordinate invariant is absent or false")
    if expected.stage == "primary_controls":
        role = PRIMARY_CONTROL_ANALYSIS_ROLES.get(str(expected.role), str(expected.role))
        if summary.get("analysis_role") != role:
            problems.append(
                f"analysis_role={summary.get('analysis_role')!r}, expected {role!r}"
            )
        if summary.get("feature_bank_variant") != "source_story_variance":
            problems.append("primary control does not use source_story_variance")
        if int(summary.get("bank_size", -1)) != 256:
            problems.append("primary control is not fixed at K=256")
        if summary.get("historical_bank_used") is not False:
            problems.append("primary control uses or fails to exclude historical bank")
        if summary.get("smoke_only") is not False or summary.get("control_inference_eligible") is not True:
            problems.append("primary control is smoke-only or not inference-eligible")
        if expected.story is not None and summary.get("story") != expected.story:
            problems.append("within-story control story identity mismatch")
        if expected.train_story is not None and summary.get("source_story") != expected.train_story:
            problems.append("transfer control source-story identity mismatch")
        if expected.test_story is not None and summary.get("target_story") != expected.test_story:
            problems.append("transfer control target-story identity mismatch")
        controls = summary.get("controls")
        if not isinstance(controls, Mapping):
            problems.append("primary control block specification is missing")
        else:
            if list(controls.get("baseline_blocks", ())) != ["low_level_position", "surprisal"]:
                problems.append("primary baseline blocks are not low-level/position plus surprisal")
            if list(controls.get("full_blocks", ())) != ["low_level_position", "surprisal", "sae"]:
                problems.append("primary full model is not the declared multiband model")
            if controls.get("independent_effective_penalties") is not True:
                problems.append("primary baseline and multiband penalties are not independent")
        # The within-story control selects penalties by inner-fold CV on the
        # training rows, so it must declare both selection metrics and
        # training-only scaling.  The transfer control performs no target-side
        # selection at all: it applies the source scaling and weights unchanged,
        # which is a stronger guarantee than training-only scaling and is
        # recorded under different fields.  Each is checked on its own terms.
        if expected.role == "primary_source_story_control_transfer":
            if summary.get("source_scaling_and_weights_applied_unchanged") is not True:
                problems.append("transfer control does not record unchanged source scaling and weights")
            if summary.get("source_bank_selection_only") is not True:
                problems.append("transfer control does not record source-only bank selection")
            if summary.get("target_story_excluded_from_bank_selection") is not True:
                problems.append("transfer control does not exclude the target story from bank selection")
        else:
            cv = summary.get("cv", {})
            if not isinstance(cv, Mapping):
                problems.append("primary control CV contract is missing")
            else:
                selection_metrics = set(str(value) for value in cv.get("selection_metrics", ()))
                if not {"pearson_r", "r2_oos"}.issubset(selection_metrics):
                    problems.append("primary control does not record separate r and R2 selection")
                if cv.get("training_only_scaling") is not True:
                    problems.append("primary control does not record training-only scaling")

        def valid_selection_record(record: Any, metric: str) -> bool:
            if not isinstance(record, Mapping):
                return False
            baseline = record.get("baseline")
            full = record.get("full")
            return (
                isinstance(baseline, Mapping)
                and isinstance(full, Mapping)
                and baseline.get("selection_metric") == metric
                and full.get("selection_metric") == metric
            )

        # Both primary functions use the same record contract but place the
        # records in different containers: within-story stores them per outer
        # fold (and per subject for BOLD), while transfer stores them per
        # participant plus one LM record.  Check every selected r/R2 record so
        # a summary with only one tuned metric cannot pass accidentally.
        if role.endswith("within"):
            folds = summary.get("folds")
            if not isinstance(folds, Sequence) or isinstance(folds, (str, bytes)) or not folds:
                problems.append("within control has no outer-fold records")
            else:
                for fold_index, fold in enumerate(folds):
                    if not isinstance(fold, Mapping):
                        problems.append(f"within fold {fold_index} is not an object")
                        continue
                    lm_records = fold.get("lm", {})
                    for metric, label in (("pearson_r", "r"), ("r2_oos", "r2")):
                        if not valid_selection_record(lm_records.get(label) if isinstance(lm_records, Mapping) else None, metric):
                            problems.append(f"within fold {fold_index} lacks separately selected LM {label} baseline/multiband records")
                    brain_records = fold.get("brain", ())
                    if not isinstance(brain_records, Sequence) or isinstance(brain_records, (str, bytes)) or not brain_records:
                        problems.append(f"within fold {fold_index} has no participant brain records")
                    else:
                        for subject_record in brain_records:
                            for metric, label in (("pearson_r", "r"), ("r2_oos", "r2")):
                                if not valid_selection_record(subject_record.get(label) if isinstance(subject_record, Mapping) else None, metric):
                                    problems.append(f"within fold {fold_index} lacks separately selected brain {label} baseline/multiband records")
        else:
            fit_records = summary.get("brain", {}).get("fit_records", ()) if isinstance(summary.get("brain"), Mapping) else ()
            if not isinstance(fit_records, Sequence) or isinstance(fit_records, (str, bytes)) or not fit_records:
                problems.append("transfer control has no participant brain fit records")
            else:
                for subject_record in fit_records:
                    for metric, label in (("pearson_r", "r"), ("r2_oos", "r2")):
                        if not valid_selection_record(subject_record.get(label) if isinstance(subject_record, Mapping) else None, metric):
                            problems.append("transfer participant lacks separately selected brain r/R2 baseline/multiband records")
            lm_fit_records = summary.get("lm", {}).get("fit_records", {}) if isinstance(summary.get("lm"), Mapping) else {}
            for metric, label in (("pearson_r", "r"), ("r2_oos", "r2")):
                if not valid_selection_record(lm_fit_records.get(label) if isinstance(lm_fit_records, Mapping) else None, metric):
                    problems.append(f"transfer lacks separately selected LM {label} baseline/multiband records")

        brain = summary.get("brain")
        participants = brain.get("participants") if isinstance(brain, Mapping) else None
        if not isinstance(participants, Sequence) or isinstance(participants, (str, bytes)) or not participants:
            problems.append("primary control has no participant-level brain rows")
        else:
            for participant in participants:
                if not isinstance(participant, Mapping) or not all(
                    key in participant for key in ("delta_r", "delta_r2_oos")
                ):
                    problems.append("primary control participant rows lack baseline-versus-multiband deltas")
                    break
    if expected.stage == "nulls":
        if summary.get("complete") is not True:
            problems.append("null summary is not complete")
        required_shifts = int(spec["nulls"]["temporal_target"]["shifts"])
        if int(summary.get("n_shifts", -1)) != required_shifts:
            problems.append(f"n_shifts={summary.get('n_shifts')!r}, expected {required_shifts}")
        if summary.get("penalty_selection_refit_per_shift") is not True:
            # Grouped-null summaries use an explicit grouped_model_refit
            # flag instead of the feature-level name.
            if not expected.role.startswith("temporal_null_grouped") or summary.get("grouped_model_refit") is not True:
                problems.append("null did not refit penalty selection per shift")
        if expected.role.startswith("temporal_null_grouped") and summary.get("group_membership_fixed") is not True:
            problems.append("grouped null did not keep group membership fixed")
    if expected.stage == "grouped":
        if summary.get("complete") is not True or summary.get("status") not in {"complete", None}:
            problems.append("grouped summary is incomplete or underidentified")
        if int(summary.get("bank_size", -1)) != 256:
            problems.append("grouped analysis is not at n=256")
    if expected.stage == "group_average":
        if summary.get("complete") is not True:
            problems.append("group-average summary is not complete")
        if summary.get("confirmatory_eligible") is not False:
            problems.append("group-average sensitivity is not marked non-confirmatory")
        if summary.get("bank_variant") != "source_story_variance" or int(summary.get("bank_size", -1)) != 256:
            problems.append("group-average summary does not use the fixed source-variance K=256 bank")
        temporal = summary.get("conditional_temporal_null", {})
        if temporal.get("fully_retuned") is not True:
            problems.append("group-average temporal null is not fully retuned")
        if int(temporal.get("n_shifts", -1)) != int(spec["nulls"]["temporal_target"]["shifts"]):
            problems.append("group-average temporal null has the wrong shift count")

    # A condition can only enter the release if it carries the provenance
    # needed to audit it later.  Older/stale summaries that happen to contain
    # numeric scores therefore remain visible in the inventory but are not
    # eligible evidence.  The maps are intentionally required for every
    # scientific condition rather than inferred from a stage-level manifest.
    if expected.stage in {"fit", "transfer", "primary_controls", "nulls", "grouped"}:
        for field in ("input_sha256", "code_sha256"):
            value = summary.get(field)
            if not isinstance(value, Mapping) or not value:
                problems.append(f"{field} is missing or empty")

    arrays_raw = summary.get("arrays_path")
    canonical_paths: dict[str, Any] = {
        "summary_path": {
            "original": str(path),
            "resolved": str(path.resolve()),
        },
        "arrays_path": None,
        "input_sha256": {},
        "code_sha256": {},
    }
    if not arrays_raw:
        problems.append("missing arrays_path")
    else:
        arrays_path, arrays_reason = canonicalize_evidence_path(
            str(arrays_raw),
            config_path,
            spec,
            expected_summary_path=path,
            arrays=True,
        )
        canonical_paths["arrays_path"] = {
            "original": str(arrays_raw),
            "resolved": str(arrays_path),
            "mapping": arrays_reason,
        }
        if arrays_reason == "unmapped_absolute":
            problems.append(f"arrays_path is an unmappable absolute path: {arrays_raw}")
        if not arrays_path.is_file():
            problems.append(f"arrays file not found: {arrays_path}")
        elif full_hashes and summary.get("arrays_sha256"):
            if sha256_file(arrays_path) != str(summary["arrays_sha256"]):
                problems.append("arrays_sha256 mismatch")

    if expected.stage == "primary_controls" and arrays_raw:
        arrays_path, _ = canonicalize_evidence_path(
            str(arrays_raw),
            config_path,
            spec,
            expected_summary_path=path,
            arrays=True,
        )
        if arrays_path.is_file():
            try:
                with np.load(arrays_path, allow_pickle=True) as bundle:
                    array_keys = set(bundle.files)
                    required = {"subject_ids", "brain_delta_r", "brain_delta_r2_oos"}
                    missing = sorted(required - array_keys)
                    if missing:
                        problems.append(f"primary control arrays missing participant deltas: {missing}")
                    else:
                        subject_ids = np.asarray(bundle["subject_ids"])
                        delta_r = np.asarray(bundle["brain_delta_r"])
                        delta_r2 = np.asarray(bundle["brain_delta_r2_oos"])
                        if subject_ids.ndim != 1 or len(subject_ids) < 2:
                            problems.append("primary control participant array has fewer than two subjects")
                        if delta_r.ndim != 1 or delta_r2.ndim != 1:
                            problems.append("primary control participant delta arrays are not one-dimensional")
                        if not (len(subject_ids) == len(delta_r) == len(delta_r2)):
                            problems.append("primary control participant/delta array lengths differ")
                        if not np.isfinite(delta_r).any() or not np.isfinite(delta_r2).any():
                            problems.append("primary control participant deltas contain no finite values")
            except (OSError, ValueError) as exc:
                problems.append(f"primary control arrays unreadable: {exc}")

    # Validate source/code hashes when the stage recorded them.  This catches
    # stale copied results without imposing an O(number-of-arrays) rehash on a
    # routine manifest scan.  Full array hashing is an explicit opt-in.
    for field in ("input_sha256", "code_sha256"):
        mapping = summary.get(field, {})
        if mapping is None:
            continue
        if not isinstance(mapping, Mapping):
            problems.append(f"{field} is not a mapping")
            continue
        for raw_path, recorded in mapping.items():
            item, mapping_reason = canonicalize_evidence_path(
                str(raw_path), config_path, spec
            )
            canonical_paths[field][str(raw_path)] = {
                "original": str(raw_path),
                "resolved": str(item),
                "mapping": mapping_reason,
                "recorded_sha256": str(recorded),
            }
            if mapping_reason == "unmapped_absolute":
                problems.append(f"{field} is an unmappable absolute path: {raw_path}")
            if not item.is_file():
                problems.append(f"{field} path missing: {item}")
                continue
            try:
                observed = sha256_file(item)
            except OSError as exc:
                problems.append(f"{field} unreadable {item}: {exc}")
                continue
            if observed != str(recorded):
                problems.append(f"{field} hash mismatch: {item}")
    if problems:
        validated = dict(summary)
        validated["_canonical_provenance_paths"] = canonical_paths
        return "invalid", problems, validated
    validated = dict(summary)
    validated["_canonical_provenance_paths"] = canonical_paths
    return "complete", [], validated


def _stage_marker_status(root: Path, stage: str) -> tuple[str, list[str]]:
    """Read a stage-level status marker without trusting its condition list."""

    problems: list[str] = []
    if stage == "reliability":
        return "not_applicable", problems
    marker = root / stage / "stage_summary.json"
    if not marker.is_file():
        return "missing", [f"stage marker missing: {marker}"]
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "invalid", [f"stage marker unreadable: {exc}"]
    if not data.get("analysis_fingerprint"):
        problems.append("stage marker missing analysis_fingerprint")
    return ("complete" if not problems else "invalid"), problems


def _reliability_inventory(config_path: Path, spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    root = artifact_root(config_path, spec)
    records: list[dict[str, Any]] = []
    for story in spec["stories"]:
        path = root / "reliability" / "bold" / str(story) / "summary.json"
        status = "missing"
        problems: list[str] = []
        summary: dict[str, Any] | None = None
        if path.is_file():
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
                if summary.get("analysis_status") != "complete":
                    problems.append("analysis_status is not complete")
                if not summary.get("arrays_path"):
                    problems.append("reliability arrays missing")
                else:
                    arrays_path, mapping_reason = canonicalize_evidence_path(
                        summary["arrays_path"],
                        config_path,
                        spec,
                        expected_summary_path=path,
                        arrays=True,
                    )
                    if mapping_reason == "unmapped_absolute":
                        problems.append("reliability arrays path is unmappable")
                    if not arrays_path.is_file():
                        problems.append("reliability arrays missing")
                status = "complete" if not problems else "invalid"
            except (OSError, json.JSONDecodeError) as exc:
                status = "invalid"
                problems.append(str(exc))
        else:
            problems.append("reliability summary missing")
        records.append({"stage": "reliability", "key": str(story), "path": str(path), "status": status, "problems": problems, "summary_sha256": sha256_file(path) if path.is_file() else None})
    return records


def _conditioning_inventory(config_path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    root = artifact_root(config_path, spec)
    path = root / "conditioning" / "conditioning_summary.json"
    problems: list[str] = []
    if not path.is_file():
        return {"status": "missing", "path": str(path), "problems": ["conditioning_summary.json missing"]}
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "invalid", "path": str(path), "problems": [str(exc)]}
    expected_inputs = len(spec["models"]) * 2  # one source bank per direction
    expected_inputs += len(spec["models"]) * 2 * int(spec["cross_validation"]["outer_folds"])
    if len(summary.get("input_artifacts", [])) != expected_inputs:
        problems.append(f"input_artifacts={len(summary.get('input_artifacts', []))}, expected {expected_inputs}")
    if not summary.get("records"):
        problems.append("no conditioning records")
    if summary.get("analysis_config_sha256") != config_digest(config_path):
        problems.append("conditioning analysis configuration hash mismatch")
    return {"status": "complete" if not problems else "invalid", "path": str(path), "problems": problems, "record_count": len(summary.get("records", [])), "summary_sha256": sha256_file(path)}


def _pooling_inventory(config_path: Path, spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate token-level inputs required by pooling sensitivities.

    The pooling extractor writes one summary per model/story and does not call
    the output a fitted ``complete`` result.  A canonical mean array alone is
    insufficient for max/sum/event-count pooling, so only a complete
    token-record summary with all four definitions can satisfy this check.
    """

    root = artifact_root(config_path, spec)
    required_pooling = {
        "mean_positive_activation",
        "max_positive_activation",
        "sum_positive_activation",
        "event_count",
    }
    records: list[dict[str, Any]] = []
    for model in spec["models"]:
        for story in spec["stories"]:
            path = root / "pooling_token_records" / str(model) / str(story) / "summary.json"
            status = "missing"
            problems: list[str] = []
            if not path.is_file():
                problems.append("token-level pooling summary missing")
            else:
                try:
                    loaded = json.loads(path.read_text(encoding="utf-8"))
                    if not isinstance(loaded, Mapping):
                        problems.append("pooling summary is not a JSON object")
                    else:
                        summary = loaded
                        if summary.get("analysis_role") != "correspondence_token_level_pooling_inputs":
                            problems.append("unexpected pooling summary role")
                        if summary.get("outcome_free") is not True or summary.get("causal") is not True:
                            problems.append("pooling inputs are not marked outcome-free and causal")
                        if int(summary.get("window_context_tokens", -1)) != int(spec["targets"]["lm"]["maximum_context_tokens"]):
                            problems.append("pooling context-window size differs from configured LM target")
                        if set(summary.get("pooling_names", ())) != required_pooling:
                            problems.append("pooling summary does not contain all four configured pooling definitions")
                        coordinate = None
                        coordinate_reason = None
                        if summary.get("coordinate_manifest"):
                            coordinate, coordinate_reason = canonicalize_evidence_path(
                                str(summary["coordinate_manifest"]), config_path, spec
                            )
                            if coordinate_reason == "unmapped_absolute":
                                problems.append("coordinate manifest path is unmappable")
                        token_records = None
                        token_reason = None
                        if summary.get("token_records"):
                            token_records, token_reason = canonicalize_evidence_path(
                                str(summary["token_records"]), config_path, spec
                            )
                            if token_reason == "unmapped_absolute":
                                problems.append("token records path is unmappable")
                        if coordinate is None or not coordinate.is_file():
                            problems.append("coordinate manifest missing")
                        if token_records is None or not token_records.is_file():
                            problems.append("token records missing")
                        if token_records is not None and token_records.is_file():
                            recorded_hash = summary.get("token_records_sha256")
                            if not recorded_hash or sha256_file(token_records) != str(recorded_hash):
                                problems.append("token_records_sha256 missing or mismatched")
                        if summary.get("predictor_sha256") and summary.get("predictor_path"):
                            predictor, predictor_reason = canonicalize_evidence_path(
                                str(summary["predictor_path"]), config_path, spec
                            )
                            if predictor_reason == "unmapped_absolute":
                                problems.append("canonical predictor path is unmappable")
                            if not predictor.is_file():
                                problems.append("canonical predictor referenced by pooling summary is missing")
                            elif sha256_file(predictor) != str(summary["predictor_sha256"]):
                                problems.append("predictor_sha256 mismatch")
                        if not summary.get("analysis_fingerprint"):
                            problems.append("pooling summary missing analysis_fingerprint")
                        status = "complete" if not problems else "invalid"
                except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                    problems.append(f"unreadable pooling summary: {exc}")
            records.append({"stage": "pooling", "key": f"{model}/{story}", "path": str(path), "status": status, "problems": problems, "summary_sha256": sha256_file(path) if path.is_file() else None})
    return records


def _pooling_analysis_inventory(config_path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Inventory the generalized pooled-predictor fits and retuned nulls.

    ``pooling.py`` is only a materialization stage.  Its predictor
    bundles become scientific evidence only after the downstream runner has
    produced every model/story/pooling/bank condition and the corresponding
    199-shift null.  Keep this inventory separate from the historical
    ``fit``/``transfer`` rows so an old canonical mean analysis cannot make a
    partially run pooling sensitivity appear complete.
    """

    root = artifact_root(config_path, spec) / "pooling_analysis"
    models = list(spec["models"])
    # The generalized runner is intentionally fixed at K=256 and covers the
    # primary source-variance bank plus the three pruned sensitivities.
    variants = [
        "source_story_variance",
        "source_story_correlation_pruned_0p70",
        "source_story_correlation_pruned_0p80",
        "source_story_correlation_pruned_0p90",
    ]
    poolings = [
        "mean_positive_activation",
        "max_positive_activation",
        "sum_positive_activation",
        "event_count",
    ]
    observed_expected = len(models) * 2 * len(poolings) * len(variants) * 2
    null_expected = observed_expected
    records: list[dict[str, Any]] = []
    for path in sorted(root.glob("observed/**/summary.json")) if root.is_dir() else []:
        records.append(_pooling_analysis_summary_record(path, config_path, spec, "observed"))
    for path in sorted(root.glob("nulls/**/summary.json")) if root.is_dir() else []:
        records.append(_pooling_analysis_summary_record(path, config_path, spec, "nulls"))
    observed = [row for row in records if row["regime"] == "observed"]
    nulls = [row for row in records if row["regime"] == "nulls"]
    problems: list[str] = []
    if len(observed) != observed_expected:
        problems.append(f"observed pooling-analysis summaries={len(observed)}, expected {observed_expected}")
    if len(nulls) != null_expected:
        problems.append(f"null pooling-analysis summaries={len(nulls)}, expected {null_expected}")

    def condition_key(row: Mapping[str, Any]) -> tuple[Any, ...] | None:
        role = str(row.get("analysis_role", ""))
        common = (
            row.get("model_key"),
            row.get("pooling"),
            row.get("feature_bank_variant"),
            row.get("bank_size"),
        )
        if row.get("regime") == "observed":
            if role == "correspondence_pooling_within_observed_fit":
                return ("within_story", row.get("story"), *common)
            if role == "correspondence_pooling_cross_story_observed_fit":
                return ("cross_story", row.get("source_story"), row.get("target_story"), *common)
        elif row.get("regime") == "nulls":
            if role == "correspondence_pooling_within_temporal_target_null":
                return ("within_story", row.get("story"), *common)
            if role == "correspondence_pooling_cross_story_temporal_target_null":
                return ("cross_story", row.get("source_story"), row.get("target_story"), *common)
        return None

    def expected_keys() -> set[tuple[Any, ...]]:
        result: set[tuple[Any, ...]] = set()
        stories = [str(value) for value in spec["stories"]]
        directions = [(stories[0], stories[1]), (stories[1], stories[0])]
        for model in models:
            for pooling in poolings:
                for variant in variants:
                    common = (model, pooling, variant, 256)
                    result.update(("within_story", story, *common) for story in stories)
                    result.update(("cross_story", source, target, *common) for source, target in directions)
        return result

    for regime, rows in (("observed", observed), ("nulls", nulls)):
        keys = [condition_key(row) for row in rows]
        if any(key is None for key in keys):
            problems.append(f"{regime} pooling-analysis summary has an unexpected analysis role")
        concrete = [key for key in keys if key is not None]
        if len(concrete) != len(set(concrete)):
            problems.append(f"{regime} pooling-analysis summaries contain duplicate condition identities")
        expected = expected_keys()
        missing = expected - set(concrete)
        unexpected = set(concrete) - expected
        if missing:
            problems.append(f"{regime} pooling-analysis conditions missing: {len(missing)}")
        if unexpected:
            problems.append(f"{regime} pooling-analysis conditions unexpected: {len(unexpected)}")
    problems.extend(
        f"{row['path']}: {problem}"
        for row in records
        if row["status"] != "complete"
        for problem in row["problems"]
    )
    return {
        "status": "complete" if not problems else "invalid",
        "path": str(root),
        "problems": problems,
        "observed_expected_count": observed_expected,
        "observed_complete_count": sum(row["status"] == "complete" for row in observed),
        "null_expected_count": null_expected,
        "null_complete_count": sum(row["status"] == "complete" for row in nulls),
        "records": records,
    }


def _pooling_analysis_summary_record(
    path: Path,
    config_path: Path,
    spec: Mapping[str, Any],
    regime: str,
) -> dict[str, Any]:
    """Validate one pooled-analysis summary without trusting stage manifests."""

    problems: list[str] = []
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"path": str(path), "regime": regime, "status": "invalid", "problems": [str(exc)]}
    if not isinstance(summary, Mapping):
        return {"path": str(path), "regime": regime, "status": "invalid", "problems": ["summary is not a JSON object"]}
    if summary.get("complete") is not True:
        problems.append("complete is not true")
    if not summary.get("analysis_fingerprint"):
        problems.append("analysis_fingerprint missing")
    if regime == "nulls":
        expected_shifts = int(spec["nulls"]["temporal_target"]["shifts"])
        if int(summary.get("n_shifts", -1)) != expected_shifts:
            problems.append(f"n_shifts={summary.get('n_shifts')!r}, expected {expected_shifts}")
        if summary.get("penalty_selection_refit_per_shift") is not True:
            problems.append("null is not marked fully retuned")
    arrays_raw = summary.get("arrays_path")
    if not arrays_raw:
        problems.append("arrays_path missing")
    else:
        arrays, mapping_reason = canonicalize_evidence_path(
            str(arrays_raw),
            config_path,
            spec,
            expected_summary_path=path,
            arrays=True,
        )
        if mapping_reason == "unmapped_absolute":
            problems.append("arrays_path is an unmappable absolute path")
        if not arrays.is_file():
            problems.append(f"arrays file missing: {arrays}")
        elif summary.get("arrays_sha256") and sha256_file(arrays) != str(summary["arrays_sha256"]):
            problems.append("arrays_sha256 mismatch")
    for field in ("input_sha256", "code_sha256"):
        value = summary.get(field)
        if not isinstance(value, Mapping) or not value:
            problems.append(f"{field} missing or empty")
    return {
        "path": str(path),
        "regime": regime,
        "status": "complete" if not problems else "invalid",
        "problems": problems,
        "analysis_role": summary.get("analysis_role"),
        "model_key": summary.get("model_key"),
        "story": summary.get("story"),
        "source_story": summary.get("source_story"),
        "target_story": summary.get("target_story"),
        "pooling": summary.get("pooling"),
        "feature_bank_variant": summary.get("feature_bank_variant"),
        "bank_size": summary.get("bank_size"),
        "summary_sha256": sha256_file(path),
    }


def _power_bank_family(raw: Any) -> str:
    value = str(raw or "")
    if value == "source_story_variance" or value.startswith("source_story_variance"):
        return "source_story_variance"
    if value == "source_story_correlation_pruned" or value.startswith("source_story_correlation_pruned"):
        return "source_story_correlation_pruned"
    return value


def _power_artifact_status(path: Path, spec: Mapping[str, Any]) -> tuple[str, list[str], tuple[str, str, str] | None]:
    """Validate one complete semi-synthetic power-grid JSON artifact.

    ``power.run_full_grid`` intentionally omits a generic ``status``
    field from the result file and uses ``detection_fraction`` rather than the
    plotting-friendly ``detection_probability`` name.  Validation therefore
    checks the actual scientific contract: one model/bank/direction design,
    the complete rho-by-scale grid, and the configured null/positive replicate
    counts.  Smoke/status/manifest JSON files are never eligible.
    """

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "invalid", [f"unreadable power artifact: {exc}"], None
    if not isinstance(data, Mapping):
        return "invalid", ["power artifact is not a JSON object"], None
    problems: list[str] = []
    if data.get("scientific_evidence") is not True:
        problems.append("scientific_evidence is not true")
    if data.get("analysis_role") != "correspondence_semi_synthetic_power":
        problems.append("analysis_role is not the completed semi-synthetic power role")
    model = str(data.get("model", ""))
    bank = _power_bank_family(data.get("bank_type", ""))
    direction = str(data.get("direction", ""))
    allowed_models = {str(value) for value in spec.get("power", {}).get("designs", ())}
    allowed_banks = {str(value) for value in spec.get("power", {}).get("bank_types", ())}
    allowed_directions = {str(value) for value in spec.get("power", {}).get("directions", ())}
    if model not in allowed_models:
        problems.append(f"model {model!r} is not in the configured power design list")
    if bank not in allowed_banks:
        problems.append(f"bank family {bank!r} is not in the configured power design list")
    if direction not in allowed_directions:
        problems.append(f"direction {direction!r} is not in the configured power design list")
    if not data.get("design_label"):
        problems.append("missing design_label")
    if not data.get("design_sha256"):
        problems.append("missing design_sha256")
    if not data.get("censoring_manifest"):
        problems.append("missing censoring_manifest")
    if data.get("oracle_correlation_method") != "spearman":
        problems.append("oracle_correlation_method is not the primary Spearman metric")
    expected_scale_definition = "mean per-target population SD of the corresponding empirical residual pool"
    if data.get("residual_scale_definition") != expected_scale_definition:
        problems.append("residual_scale_definition does not identify system-specific empirical scaling")
    for scale_key in ("brain_empirical_residual_sd", "lm_empirical_residual_sd"):
        try:
            scale_value = float(data[scale_key])
        except (KeyError, TypeError, ValueError):
            problems.append(f"missing or malformed {scale_key}")
            continue
        if not np.isfinite(scale_value) or scale_value <= 0:
            problems.append(f"{scale_key} is not finite and positive")
    rows = data.get("rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        problems.append("rows is not a list")
        rows = []
    oracle_values = [float(value) for value in spec["power"]["oracle_correlations"]]
    scale_values = [float(value) for value in spec["power"]["residual_scale_multipliers"]]
    null_reps = int(spec["power"]["null_replicates"])
    positive_reps = int(spec["power"]["positive_replicates_per_grid_cell"])
    expected = {(rho, scale) for rho in oracle_values for scale in scale_values}
    seen: set[tuple[float, float]] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            problems.append(f"row {index} is not an object")
            continue
        try:
            rho = float(row["oracle_correlation"])
            scale = float(row["residual_scale_multiplier"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"row {index} lacks numeric oracle/scale coordinates")
            continue
        key = (rho, scale)
        if key not in expected:
            problems.append(f"row {index} has an unexpected oracle/scale coordinate {key}")
        if key in seen:
            problems.append(f"duplicate oracle/scale row {key}")
        seen.add(key)
        expected_reps = null_reps if np.isclose(rho, 0.0) else positive_reps
        try:
            replicates = int(row["replicates"])
            detection = float(row.get("detection_fraction", row.get("detection_probability")))
        except (KeyError, TypeError, ValueError):
            problems.append(f"row {index} lacks replicate count or detection fraction")
            continue
        if replicates != expected_reps:
            problems.append(f"row {index} replicates={replicates}, expected {expected_reps}")
        if not np.isfinite(detection) or not 0.0 <= detection <= 1.0:
            problems.append(f"row {index} detection fraction is outside [0,1]")
    missing = sorted(expected - seen)
    if missing:
        problems.append(f"missing power-grid coordinates: {missing[:8]}")
    if len(rows) != len(expected):
        problems.append(f"row count={len(rows)}, expected {len(expected)}")
    # Keep a design key even when the grid is malformed so the inventory can
    # explain which design failed; malformed/duplicate designs never count as
    # complete below.
    design_key = (model, bank, direction) if model and bank and direction else None
    return ("complete" if not problems else "invalid"), problems, design_key


def _power_inventory(config_path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    root = artifact_root(config_path, spec) / "power"
    candidates = sorted(root.rglob("*.json")) if root.is_dir() else []
    good: list[Path] = []
    records: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    problems: list[str] = []
    for path in candidates:
        if path.name in {"manifest.json", "power_status.json", "smoke.json"}:
            continue
        # Paired design sidecars and resumable shard files are provenance
        # support artifacts, not completed scientific power grids.  Treating
        # them as result JSON would incorrectly mark a valid partial
        # run as malformed (and would make the shard files count as duplicate
        # designs).  They are listed separately for auditability.
        if "designs" in path.parts or "shards" in path.parts:
            support.append({"path": str(path), "kind": "power_design_or_shard_support"})
            continue
        status, local, design_key = _power_artifact_status(path, spec)
        if status == "complete":
            good.append(path)
        records.append({"path": str(path), "status": status, "design": list(design_key) if design_key else None, "problems": local})
    expected_designs = {
        (str(model), str(bank), str(direction))
        for model in spec["power"]["designs"]
        for bank in spec["power"]["bank_types"]
        for direction in spec["power"]["directions"]
    }
    complete_designs = set()
    for row in records:
        if row["status"] == "complete" and row.get("design"):
            complete_designs.add(tuple(row["design"]))
    missing_designs = sorted(expected_designs - complete_designs)
    if missing_designs:
        problems.append(f"missing complete power designs: {missing_designs[:12]}")
    if not records:
        status = root / "power_status.json"
        if status.is_file():
            try:
                marker = json.loads(status.read_text(encoding="utf-8"))
                problems.append(f"power status is {marker.get('status', 'unknown')!r}; a real design/grid artifact is required")
            except Exception:
                problems.append("power status marker is unreadable")
        else:
            problems.append("no complete scientific power artifact")
    return {
        "status": "complete" if not problems else "invalid",
        "path": str(root),
        "problems": problems,
        "scientific_artifacts": [str(path) for path in good],
        "support_artifacts": support,
        "expected_design_count": len(expected_designs),
        "complete_design_count": len(complete_designs),
        "records": records,
    }


def _bank_inventory(config_path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    root = artifact_root(config_path, spec) / "feature_banks"
    expected = len(spec["models"]) * 2 + len(spec["models"]) * 2 * int(spec["cross_validation"]["outer_folds"])
    records: list[dict[str, Any]] = []
    problems: list[str] = []
    for path in sorted(root.glob("cross_story/*/*/banks.json")) + sorted(root.glob("within_story/*/*/fold_*/banks.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            records.append({"path": str(path), "status": "invalid", "problems": [str(exc)]})
            continue
        local: list[str] = []
        if data.get("analysis_config_sha256") != config_digest(config_path):
            local.append("analysis_config_sha256 mismatch")
        if data.get("outcome_inputs_used") is not False:
            local.append("outcome_inputs_used is not false")
        if data.get("target_excluded_from_selection") is not True and data.get("mode") == "cross_story_source":
            local.append("target story was not marked excluded")
        for variant in ("source_variance", "source_prevalence"):
            section = data.get(variant, {})
            for size in sizes(spec):
                entry = section.get("banks", {}).get(str(size)) if isinstance(section, Mapping) else None
                if not isinstance(entry, Mapping) or entry.get("complete") is not True or int(entry.get("actual_size", -1)) != int(size):
                    local.append(f"{variant}/n{size} incomplete")
        for threshold in (0.70, 0.80, 0.90):
            key = str(threshold)
            section = data.get("correlation_pruned", {}).get(key, {})
            if not section:
                local.append(f"correlation_pruned@{threshold:.2f} missing")
            # Only the K=256 representative bank is required for the
            # grouped-reliance sensitivity.  Higher requested K values are
            # retained in the bundle as diagnostics and may be incomplete
            # when clustering leaves fewer than K representatives.
            for size in (256,):
                entry = section.get("banks", {}).get(str(size)) if isinstance(section, Mapping) else None
                cluster = section.get("clusters_over_exact_source_variance_banks", {}).get(str(size)) if isinstance(section, Mapping) else None
                if not isinstance(entry, Mapping) or entry.get("complete") is not True:
                    local.append(f"correlation_pruned@{threshold:.2f}/n{size} incomplete")
                if not isinstance(cluster, Mapping) or cluster.get("complete_exact_k_predictor_membership") is not True:
                    local.append(f"exact clusters@{threshold:.2f}/n{size} incomplete")
        records.append({"path": str(path), "status": "complete" if not local else "invalid", "problems": local})
    if len(records) != expected:
        problems.append(f"bank bundle count={len(records)}, expected {expected}")
    problems.extend(f"{row['path']}: {problem}" for row in records if row["status"] != "complete" for problem in row["problems"])
    return {"status": "complete" if not problems else "invalid", "path": str(root), "problems": problems, "bundle_count": len(records), "expected_bundle_count": expected, "records": records}


def _inference_stat_valid(
    value: Any,
    *,
    require_bh: bool,
) -> list[str]:
    """Validate the compact statistic emitted by control inference.

    This intentionally checks the fields used by the paper's inference table,
    rather than treating a finite mean as sufficient evidence.  In particular,
    a primary release needs participant-bootstrap intervals and sign-flip p
    values, and each model-level statistic needs both BH-adjusted q values.
    """

    problems: list[str] = []
    if not isinstance(value, Mapping):
        return ["inference statistic is not an object"]
    try:
        mean = float(value["mean"])
        sd = float(value["sd"])
        n = int(value["n"])
    except (KeyError, TypeError, ValueError):
        problems.append("inference statistic lacks mean/sd/n")
        mean = sd = float("nan")
        n = 0
    if not np.isfinite(mean) or not np.isfinite(sd) or n < 2:
        problems.append("inference statistic has invalid mean/sd/n")
    interval = value.get("bootstrap_ci95")
    if not isinstance(interval, Sequence) or isinstance(interval, (str, bytes)) or len(interval) != 2:
        problems.append("participant-bootstrap ci95 is missing or malformed")
    else:
        try:
            limits = [float(v) for v in interval]
            if not all(np.isfinite(v) for v in limits) or limits[0] > limits[1]:
                problems.append("participant-bootstrap ci95 is not finite/ordered")
        except (TypeError, ValueError):
            problems.append("participant-bootstrap ci95 is nonnumeric")
    for field in ("sign_flip_p_one_sided", "sign_flip_p_two_sided"):
        try:
            p = float(value[field])
            if not np.isfinite(p) or not 0.0 <= p <= 1.0:
                problems.append(f"{field} is outside [0,1]")
        except (KeyError, TypeError, ValueError):
            problems.append(f"{field} is missing")
    if require_bh:
        for field in (
            "bh_q_one_sided_across_six_models",
            "bh_q_two_sided_across_six_models",
        ):
            try:
                q = float(value[field])
                if not np.isfinite(q) or not 0.0 <= q <= 1.0:
                    problems.append(f"{field} is outside [0,1]")
            except (KeyError, TypeError, ValueError):
                problems.append(f"{field} is missing")
    return problems


def _primary_inference_status(
    config_path: Path,
    spec: Mapping[str, Any],
    regime: str,
) -> tuple[str, list[str], Path]:
    """Validate one of the two participant-paired primary inference summaries."""

    root = artifact_root(config_path, spec)
    path = root / "primary_controls" / "inference" / regime / "source_story_variance" / "n256" / "summary.json"
    if not path.is_file():
        return "missing", [f"primary {regime} inference summary missing"], path
    try:
        summary = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return "invalid", [f"primary {regime} inference summary unreadable: {exc}"], path
    problems: list[str] = []
    if not isinstance(summary, Mapping):
        return "invalid", ["primary inference summary is not a JSON object"], path
    if summary.get("analysis_role") != "correspondence_primary_control_paired_inference":
        problems.append("unexpected primary inference analysis_role")
    if summary.get("regime") != regime:
        problems.append("primary inference regime mismatch")
    if summary.get("feature_bank_variant") != "source_story_variance" or int(summary.get("bank_size", -1)) != 256:
        problems.append("primary inference does not use source-story variance K=256")
    if summary.get("historical_bank_used") is not False:
        problems.append("primary inference is not marked historical-bank-free")
    if summary.get("participant_pairing") is not True:
        problems.append("primary inference is not participant-paired")
    if summary.get("control_inference_complete") is not True:
        problems.append("primary inference is not marked complete")
    def _int_field(name: str) -> int:
        try:
            return int(summary.get(name, -1))
        except (TypeError, ValueError):
            problems.append(f"primary inference {name} is not an integer")
            return -1

    if _int_field("bootstrap_draws") != int(spec["nulls"]["bootstrap_draws"]):
        problems.append("primary inference has unexpected bootstrap draw count")
    if _int_field("sign_flip_draws") != int(spec["nulls"]["sign_flip_draws"]):
        problems.append("primary inference has unexpected sign-flip draw count")
    models = [str(value) for value in spec["models"].keys()]
    if len(models) != 6:
        problems.append(f"configured model list has {len(models)} entries; six-model BH is required")
    if regime == "within":
        expected_groups = [str(value) for value in spec["stories"].keys()]
    else:
        expected_groups = [
            "shapesphysical_to_shapessocial",
            "shapessocial_to_shapesphysical",
        ]
    groups = summary.get("groups")
    if not isinstance(groups, Mapping):
        problems.append("primary inference groups are missing")
        groups = {}
    if set(str(key) for key in groups) != set(expected_groups):
        problems.append("primary inference groups do not cover the configured stories/directions")
    for group_name in expected_groups:
        group = groups.get(group_name)
        if not isinstance(group, Mapping):
            problems.append(f"primary inference group {group_name} is missing")
            continue
        model_stats = group.get("models")
        if not isinstance(model_stats, Mapping) or set(str(key) for key in model_stats) != set(models):
            problems.append(f"primary inference group {group_name} does not contain all six models")
            model_stats = {}
        for model in models:
            result = model_stats.get(model)
            if not isinstance(result, Mapping):
                problems.append(f"primary inference group {group_name}/{model} is missing")
                continue
            for metric in ("delta_r", "delta_r2_oos"):
                local = _inference_stat_valid(result.get(metric), require_bh=True)
                problems.extend(f"{group_name}/{model}/{metric}: {problem}" for problem in local)
        global_stats = group.get("global_mean_effect")
        if not isinstance(global_stats, Mapping):
            problems.append(f"primary inference group {group_name} lacks global mean-effect tests")
        else:
            for metric in ("delta_r", "delta_r2_oos"):
                local = _inference_stat_valid(global_stats.get(metric), require_bh=False)
                problems.extend(f"{group_name}/global/{metric}: {problem}" for problem in local)

    arrays_raw = summary.get("arrays_path")
    if not arrays_raw:
        problems.append("primary inference arrays_path is missing")
    else:
        arrays_path, arrays_reason = canonicalize_evidence_path(
            str(arrays_raw),
            config_path,
            spec,
            expected_summary_path=path,
            arrays=True,
        )
        if arrays_reason == "unmapped_absolute":
            problems.append("primary inference arrays_path is an unmappable absolute path")
        if not arrays_path.is_file():
            problems.append(f"primary inference arrays file missing: {arrays_path}")
        else:
            try:
                with np.load(arrays_path, allow_pickle=True) as bundle:
                    keys = set(bundle.files)
                for group_name in expected_groups:
                    if f"{group_name}_subject_ids" not in keys:
                        problems.append(f"primary inference arrays lack {group_name} subject IDs")
                    for model in models:
                        for metric in ("delta_r", "delta_r2_oos"):
                            if f"{group_name}_{model}_{metric}" not in keys:
                                problems.append(f"primary inference arrays lack {group_name}/{model}/{metric}")
                        for metric in ("delta_r", "delta_r2_oos"):
                            if f"{group_name}_global_model_average_{metric}" not in keys:
                                problems.append(f"primary inference arrays lack global {group_name}/{metric}")
            except (OSError, ValueError) as exc:
                problems.append(f"primary inference arrays unreadable: {exc}")
    for field in ("input_sha256", "code_sha256"):
        mapping = summary.get(field)
        if not isinstance(mapping, Mapping) or not mapping:
            problems.append(f"primary inference {field} map is missing or empty")
        else:
            for raw_path, recorded in mapping.items():
                item, mapping_reason = canonicalize_evidence_path(
                    str(raw_path), config_path, spec
                )
                if mapping_reason == "unmapped_absolute":
                    problems.append(f"primary inference {field} path is unmappable: {raw_path}")
                if not item.is_file():
                    problems.append(f"primary inference {field} path missing: {item}")
                else:
                    try:
                        if sha256_file(item) != str(recorded):
                            problems.append(f"primary inference {field} hash mismatch: {item}")
                    except OSError as exc:
                        problems.append(f"primary inference {field} unreadable: {item}: {exc}")
    return ("complete" if not problems else "invalid"), problems, path


def _primary_control_inventory(
    config_path: Path,
    spec: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Inventory the 24 primary control conditions and two inference summaries."""

    primary_rows = [row for row in rows if row.get("stage") == "primary_controls"]
    within_rows = [row for row in primary_rows if row.get("role") == "primary_source_story_control_within"]
    transfer_rows = [row for row in primary_rows if row.get("role") == "primary_source_story_control_transfer"]
    condition_problems: list[str] = []
    if len(within_rows) != len(spec["models"]) * len(spec["stories"]):
        condition_problems.append(
            f"within primary condition count={len(within_rows)}, expected {len(spec['models']) * len(spec['stories'])}"
        )
    if len(transfer_rows) != len(spec["models"]) * 2:
        condition_problems.append(
            f"transfer primary condition count={len(transfer_rows)}, expected {len(spec['models']) * 2}"
        )
    condition_problems.extend(
        f"{row.get('key')}: {row.get('status')}" for row in primary_rows if row.get("status") != "complete"
    )
    inference_records: list[dict[str, Any]] = []
    for regime in ("within", "transfer"):
        status, problems, path = _primary_inference_status(config_path, spec, regime)
        inference_records.append(
            {
                "regime": regime,
                "path": str(path),
                "status": status,
                "problems": problems,
                "summary_sha256": sha256_file(path) if path.is_file() else None,
            }
        )
    problems = condition_problems + [
        f"inference/{row['regime']}: {problem}"
        for row in inference_records
        if row["status"] != "complete"
        for problem in row["problems"]
    ]
    return {
        "status": "complete" if not problems else "invalid",
        "scope": "primary_source_story_controls_only",
        "condition_count": len(primary_rows),
        "expected_condition_count": len(spec["models"]) * len(spec["stories"]) + len(spec["models"]) * 2,
        "within_condition_count": len(within_rows),
        "expected_within_condition_count": len(spec["models"]) * len(spec["stories"]),
        "transfer_condition_count": len(transfer_rows),
        "expected_transfer_condition_count": len(spec["models"]) * 2,
        "conditions": [dict(row) for row in primary_rows],
        "inference": inference_records,
        "problems": problems,
    }


def _optional_control_inventory(
    config_path: Path,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Inventory historical/dense/random controls without making them a gate."""

    root = artifact_root(config_path, spec) / "controls"
    records: list[dict[str, Any]] = []
    if root.is_dir():
        for path in sorted(root.rglob("summary.json")):
            try:
                summary = json.loads(path.read_text(encoding="utf-8"))
                role = str(summary.get("analysis_role", "unknown"))
            except (OSError, json.JSONDecodeError) as exc:
                role = "unreadable"
                summary = {}
                records.append({"path": str(path), "role": role, "status": "invalid", "problems": [str(exc)]})
                continue
            if "historical" in role:
                family = "historical"
            elif "dense" in role:
                family = "dense"
            elif "random" in role:
                family = "random"
            else:
                family = "other"
            records.append(
                {
                    "path": str(path),
                    "role": role,
                    "family": family,
                    "status": "optional",
                    "appendix_only": True,
                    "summary_sha256": sha256_file(path),
                }
            )
    counts: dict[str, int] = {}
    for record in records:
        family = str(record.get("family", "other"))
        counts[family] = counts.get(family, 0) + 1
    return {
        "status": "optional",
        "scope": "appendix_only",
        "required_for_release": False,
        "families": counts,
        "records": records,
    }


def build_manifest(
    config_path: Path,
    spec: Mapping[str, Any],
    *,
    full_hashes: bool = False,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Scan per-condition outputs and write an aggregate manifest."""

    config_path = Path(config_path).resolve()
    root = artifact_root(config_path, spec)
    expected = expected_artifacts(config_path, spec)
    rows: list[dict[str, Any]] = []
    for item in expected:
        status, problems, summary = _summary_valid(item, config_path=config_path, spec=spec, full_hashes=full_hashes)
        row = {
            "stage": item.stage,
            "role": item.role,
            "key": item.key,
            "path": item.path.relative_to(root).as_posix(),
            "status": status,
            "problems": problems,
            "summary_sha256": sha256_file(item.path) if item.path.is_file() else None,
        }
        if summary is not None:
            row.update({"model_key": item.model_key, "story": item.story, "train_story": item.train_story, "test_story": item.test_story, "variant": item.variant, "size": item.size})
            row["provenance_paths"] = summary.get("_canonical_provenance_paths", {})
        rows.append(row)

    reliability = _reliability_inventory(config_path, spec)
    pooling = _pooling_inventory(config_path, spec)
    pooling_analysis = _pooling_analysis_inventory(config_path, spec)
    conditioning = _conditioning_inventory(config_path, spec)
    banks = _bank_inventory(config_path, spec)
    power = _power_inventory(config_path, spec)
    primary_controls = _primary_control_inventory(config_path, spec, rows)
    optional_controls = _optional_control_inventory(config_path, spec)
    counts: dict[str, dict[str, int]] = {}
    for row in rows:
        stage = str(row["stage"])
        counts.setdefault(stage, {}).setdefault(str(row["status"]), 0)
        counts[stage][str(row["status"])] += 1
    required_complete = (
        all(row["status"] == "complete" for row in rows)
        and all(row["status"] == "complete" for row in reliability)
        and all(row["status"] == "complete" for row in pooling)
        and pooling_analysis["status"] == "complete"
        and primary_controls["status"] == "complete"
        and conditioning["status"] == "complete"
        and banks["status"] == "complete"
        and power["status"] == "complete"
    )
    fingerprint = provenance_fingerprint(
        config_path,
        input_paths=[item.path for item in expected if item.path.is_file()],
        code_paths=[HERE / "artifact_manifest.py", config_path],
        extra={
            "full_hashes": bool(full_hashes),
            "expected_count": len(rows),
            "primary_control_condition_count": primary_controls["condition_count"],
            "primary_control_inference_count": len(primary_controls["inference"]),
        },
    )
    manifest = {
        "schema_version": 1,
        "manifest_role": "correspondence_condition_aggregate",
        "analysis_fingerprint": fingerprint,
        "config_sha256": config_digest(config_path),
        "generated_by": "artifact_manifest.py",
        "path_resolution": {
            "allowed_roots": ["FMRI_SAE_DATA_ROOT", "FMRI_SAE_OUTPUT_ROOT"],
            "unknown_absolute_paths_rejected": True,
            "condition_path_records": "conditions[].provenance_paths",
        },
        "expected_condition_count": len(rows),
        "condition_counts": counts,
        "conditions": rows,
        "reliability": reliability,
        "pooling": pooling,
        "pooling_analysis": pooling_analysis,
        "primary_controls": primary_controls,
        "optional_controls": optional_controls,
        "conditioning": conditioning,
        "banks": banks,
        "power": power,
        "complete": bool(required_complete),
        "scientific_evidence": bool(required_complete),
        "full_hashes": bool(full_hashes),
    }
    output_path = output_path or (root / "aggregate_manifest.json")
    atomic_write_json(output_path, manifest)
    manifest["manifest_path"] = str(output_path)
    return manifest


def require_complete_manifest(manifest: Mapping[str, Any]) -> None:
    """Raise with concise diagnostics if a consumer tries to use partial data."""

    if manifest.get("complete") is True and manifest.get("scientific_evidence") is True:
        return
    missing: list[str] = []
    for row in manifest.get("conditions", []):
        if row.get("status") != "complete":
            missing.append(f"{row.get('stage')}:{row.get('key')} ({row.get('status')})")
            if len(missing) >= 12:
                break
    for section in ("reliability", "pooling", "pooling_analysis", "primary_controls", "conditioning", "banks", "power"):
        value = manifest.get(section)
        if isinstance(value, Mapping) and value.get("status") != "complete":
            missing.append(f"{section}: {value.get('status')}")
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if any(isinstance(row, Mapping) and row.get("status") != "complete" for row in value):
                missing.append(f"{section}: incomplete conditions")
    detail = "; ".join(missing) if missing else "manifest did not satisfy complete/scientific_evidence flags"
    raise ManifestIncomplete("publication figures are fail-closed until the artifact manifest is complete: " + detail)


def run(*, config_path: Path, spec: Mapping[str, Any], args: Any = None) -> dict[str, Any]:
    full_hashes = bool(getattr(args, "full_hashes", False)) if args is not None else False
    return build_manifest(config_path, spec, full_hashes=full_hashes)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--full-hashes", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.data_root is not None:
        os.environ["FMRI_SAE_DATA_ROOT"] = str(args.data_root.resolve())
    if args.output_root is not None:
        os.environ["FMRI_SAE_OUTPUT_ROOT"] = str(args.output_root.resolve())
    config_path = (args.config or default_config_path()).resolve()
    try:
        spec = load_spec(config_path)
        result = run(config_path=config_path, spec=spec, args=args)
    except (AnalysisError, FileNotFoundError, ValueError) as exc:
        print(f"aggregate manifest failed: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result if args.json else {k: result[k] for k in ("manifest_path", "complete", "scientific_evidence", "condition_counts", "reliability", "pooling", "primary_controls", "optional_controls", "conditioning", "banks", "power")}, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
