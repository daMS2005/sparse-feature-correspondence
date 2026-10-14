"""Loading and validation for the portable schema-2 analysis contract."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sysconfig
from collections.abc import Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any

import yaml

PACKAGE_NAME = "fmri-sae-correspondence"
CONFIG_ENV = "FMRI_SAE_CONFIG"
SCIENTIFIC_SECTIONS = (
    "models",
    "model_sets",
    "stories",
    "feature_bank",
    "alignment",
    "targets",
    "controls",
    "baselines",
    "cross_validation",
    "reliance",
    "nulls",
    "power",
    "manuscript_gates",
)
EXPECTED_ALPHA_GRID = [
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
    10000.0,
    100000.0,
    1000000.0,
    10000000.0,
    100000000.0,
    1000000000.0,
    10000000000.0,
]


class ConfigError(ValueError):
    """Raised when the documented analysis contract is incomplete or changed."""


def default_config_path(environ: Mapping[str, str] | None = None) -> Path:
    """Locate the config without depending on the caller's working directory."""

    environment = os.environ if environ is None else environ
    override = environment.get(CONFIG_ENV)
    if override:
        return Path(override).expanduser().resolve()

    source_tree = Path(__file__).resolve().parents[2] / "configs" / "analysis.yaml"
    if source_tree.is_file():
        return source_tree

    installed = (
        Path(sysconfig.get_path("data"))
        / "share"
        / PACKAGE_NAME
        / "analysis.yaml"
    )
    return installed.resolve()


def _mapping(value: Any, name: str, errors: list[str]) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        errors.append(f"{name} must be a mapping")
        return {}
    return value


def _is_relative_path(value: Any) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    path = Path(value)
    return not path.is_absolute() and not PureWindowsPath(value).is_absolute() and ".." not in path.parts


def validate_config(spec: Mapping[str, Any]) -> None:
    """Fail closed if a scientific choice or portable-path invariant changed."""

    errors: list[str] = []
    if spec.get("schema_version") != 2:
        errors.append("schema_version must be 2")
    if not spec.get("spec_id"):
        errors.append("spec_id is required")

    paths = _mapping(spec.get("paths"), "paths", errors)
    required_paths = {
        "artifact_root",
        "predictor_root",
        "predictor_pattern",
        "random_bank_pattern",
        "historical_ranked_pattern",
        "transcript_root",
        "feature_bank_root",
        "sae_root",
        "sae_weight_pattern",
        "evidence_manifest",
    }
    for key in sorted(required_paths):
        if not _is_relative_path(paths.get(key)):
            errors.append(f"paths.{key} must be a non-escaping relative path")
    models = _mapping(spec.get("models"), "models", errors)
    if len(models) != 6:
        errors.append("models must contain the six configured models")
    for name, model_value in models.items():
        model = _mapping(model_value, f"models.{name}", errors)
        layers = model.get("layers")
        if not isinstance(layers, list) or len(layers) != 6 or len(set(layers)) != 6:
            errors.append(f"models.{name}.layers must contain six unique layers")
        for key in ("model_id", "sae_release", "sae_width"):
            if not model.get(key):
                errors.append(f"models.{name}.{key} is required")

    model_sets = _mapping(spec.get("model_sets"), "model_sets", errors)
    if set(model_sets.get("all_models", [])) != set(models):
        errors.append("model_sets.all_models must list exactly the configured models")

    stories = _mapping(spec.get("stories"), "stories", errors)
    if set(stories) != {"shapesphysical", "shapessocial"}:
        errors.append("stories must define shapesphysical and shapessocial")
    for story, story_value in stories.items():
        values = _mapping(story_value, f"stories.{story}", errors)
        if values.get("tr_seconds") != 1.5:
            errors.append(f"stories.{story}.tr_seconds must remain 1.5")
        if values.get("expected_subjects") != 59:
            errors.append(f"stories.{story}.expected_subjects must remain 59")
        if not _is_relative_path(values.get("brain_bundle")):
            errors.append(f"stories.{story}.brain_bundle must be data-root-relative")

    bank = _mapping(spec.get("feature_bank"), "feature_bank", errors)
    sizes = bank.get("sizes")
    if sizes != [64, 128, 256, 512, 1024]:
        errors.append("feature_bank.sizes changed from the configured grid")
    if bank.get("primary_size") != 256:
        errors.append("feature_bank.primary_size must remain 256")
    primary = _mapping(bank.get("primary"), "feature_bank.primary", errors)
    if primary.get("source") != "source_story":
        errors.append("feature_bank.primary.source must remain source_story")
    if bank.get("evaluation_outcomes_allowed") is not False:
        errors.append("feature-bank selection must disallow evaluation outcomes")
    if bank.get("same_feature_layer_coordinates_for_paired_targets") is not True:
        errors.append("paired targets must use identical feature-layer coordinates")

    alignment = _mapping(spec.get("alignment"), "alignment", errors)
    if alignment.get("primary_pooling") != "mean_positive_activation":
        errors.append("alignment.primary_pooling must remain mean_positive_activation")
    if alignment.get("all_pooling_sensitivity_fail_closed_if_missing") is not True:
        errors.append("pooling sensitivity must fail closed when token records are missing")

    targets = _mapping(spec.get("targets"), "targets", errors)
    brain = _mapping(targets.get("brain"), "targets.brain", errors)
    lm = _mapping(targets.get("lm"), "targets.lm", errors)
    if brain.get("lags") != [0, 1, 2, 3, 4]:
        errors.append("targets.brain.lags must remain [0, 1, 2, 3, 4]")
    if lm.get("maximum_context_tokens") != 1024 or lm.get("causal") is not True:
        errors.append("LM extraction must remain causal with a 1,024-token context")
    if lm.get("add_special_tokens") is not False or lm.get("lagged") is not False:
        errors.append("LM extraction must remain unlagged and must not add special tokens")

    controls = _mapping(spec.get("controls"), "controls", errors)
    inherited_controls = {
        "matched_random_bank_count": 99,
        "matched_random_bank_primary_size": 256,
        "matched_random_bank_strata": [
            "layer",
            "active_fraction_decile",
            "log_variance_decile",
        ],
        "matched_random_bank_seed": 20260825,
        "band_ratios": [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
        "band_random_search_candidates": 50,
        "band_dirichlet_concentrations": [0.1, 1.0, 10.0],
    }
    for key, expected in inherited_controls.items():
        if controls.get(key) != expected:
            errors.append(f"controls.{key} must remain {expected!r}")

    baselines = _mapping(spec.get("baselines"), "baselines", errors)
    inherited_baselines = {
        "low_level": ["word_count", "phoneme_count", "envelope_rms", "envelope_delta"],
        "position_plateau_min_tr": 3,
        "position_plateau_max_tr": 51,
        "model_specific": ["surprisal_mean", "surprisal_sum"],
        "multiband_blocks": ["low_level_position", "surprisal", "sae"],
    }
    for key, expected in inherited_baselines.items():
        if baselines.get(key) != expected:
            errors.append(f"baselines.{key} must remain {expected!r}")

    cv = _mapping(spec.get("cross_validation"), "cross_validation", errors)
    if cv.get("outer_folds") != 5 or cv.get("inner_folds") != 5:
        errors.append("cross-validation must use five outer and five inner folds")
    if cv.get("purge_adjacent_trs") != 4:
        errors.append("cross_validation.purge_adjacent_trs must remain 4")
    if cv.get("selection_metrics") != ["pearson_r", "r2_oos"]:
        errors.append("cross_validation.selection_metrics changed")
    if cv.get("standardization") != "training_fold_only":
        errors.append("cross_validation.standardization must remain training_fold_only")
    if cv.get("expected_matched_subjects") != 59:
        errors.append("cross_validation.expected_matched_subjects must remain 59")
    if cv.get("alpha_grid") != EXPECTED_ALPHA_GRID:
        errors.append("cross_validation.alpha_grid changed")

    nulls = _mapping(spec.get("nulls"), "nulls", errors)
    temporal = _mapping(nulls.get("temporal_target"), "nulls.temporal_target", errors)
    identity = _mapping(nulls.get("feature_identity"), "nulls.feature_identity", errors)
    if temporal.get("shifts") != 199 or temporal.get("refit_penalty_selection") is not True:
        errors.append("temporal null must retain 199 fully retuned shifts")
    if identity.get("permutations") != 10000 or identity.get("constraint") != "within_layer":
        errors.append("identity null must retain 10,000 within-layer permutations")

    power = _mapping(spec.get("power"), "power", errors)
    if power.get("null_replicates") != 500:
        errors.append("power.null_replicates must remain 500")
    if power.get("positive_replicates_per_grid_cell") != 250:
        errors.append("power.positive_replicates_per_grid_cell must remain 250")
    if power.get("seed") != 2026082704:
        errors.append("power.seed must remain 2026082704")

    for section in SCIENTIFIC_SECTIONS:
        if section not in spec:
            errors.append(f"{section} is required")

    if errors:
        raise ConfigError("invalid schema-2 analysis config:\n- " + "\n- ".join(errors))


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load and validate the documented YAML configuration."""

    config_path = default_config_path() if path is None else Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"analysis config not found: {config_path}")
    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ConfigError("analysis config must be a YAML mapping")
    validate_config(loaded)
    return loaded


def scientific_projection(spec: Mapping[str, Any]) -> dict[str, Any]:
    """Return only choices that can alter scientific interpretation or values."""

    validate_config(spec)
    projection = {section: copy.deepcopy(spec[section]) for section in SCIENTIFIC_SECTIONS}
    for story in projection["stories"].values():
        story.pop("brain_bundle", None)
        story.pop("brain_artifact_role", None)
    return projection


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the documented config without importing scientific frameworks."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=default_config_path())
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    spec = load_config(args.config)
    from .integrity import scientific_fingerprint

    result = {
        "schema_version": spec["schema_version"],
        "spec_id": spec["spec_id"],
        "scientific_fingerprint": scientific_fingerprint(spec),
        "status": "valid",
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(f"valid: {result['spec_id']} ({result['scientific_fingerprint']})")
    return 0


__all__ = [
    "CONFIG_ENV",
    "ConfigError",
    "SCIENTIFIC_SECTIONS",
    "default_config_path",
    "load_config",
    "main",
    "scientific_projection",
    "validate_config",
]
