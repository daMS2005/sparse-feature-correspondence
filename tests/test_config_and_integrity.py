from __future__ import annotations

import copy
import tomllib
from pathlib import Path

import pytest

from fmri_sae_correspondence.config import ConfigError, load_config, validate_config
from fmri_sae_correspondence.config import main as config_main
from fmri_sae_correspondence.integrity import (
    AnalysisError,
    FingerprintError,
    artifact_fingerprint,
    assert_cache_fingerprint,
    atomic_write_json,
    config_digest,
    load_spec,
    provenance_fingerprint,
    resolve_cache_root,
    resolve_data_root,
    resolve_output_root,
    resolve_path,
    scientific_fingerprint,
    tree_digest,
)
from fmri_sae_correspondence.paths import (
    CACHE_ROOT_ENV,
    DATA_ROOT_ENV,
    OUTPUT_ROOT_ENV,
    PathConfigurationError,
    RuntimePaths,
    resolve_below,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "analysis.yaml"
BASELINE_SCIENTIFIC_FINGERPRINT = "c6fed9e46c609654da607c2792a96598be2a1c20f16ddac4cb1ef366bc4bb943"


def test_package_metadata_names_neutral_package() -> None:
    metadata = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert metadata["project"]["name"] == "fmri-sae-correspondence"
    assert metadata["project"]["version"] == "0.1.4"
    assert metadata["project"]["requires-python"] == ">=3.11"
    assert "PyYAML==6.0.2" in metadata["project"]["dependencies"]


def test_schema_two_config_loads_without_parent_spec() -> None:
    spec = load_config(CONFIG_PATH)
    assert spec["schema_version"] == 2
    assert all(not str(key).startswith("prior_") for key in spec["paths"])
    assert set(spec["stories"]) == {"shapesphysical", "shapessocial"}


def test_inherited_scientific_values_are_explicit_and_exact() -> None:
    spec = load_config(CONFIG_PATH)
    controls = spec["controls"]
    baselines = spec["baselines"]
    assert baselines["position_plateau_min_tr"] == 3
    assert baselines["position_plateau_max_tr"] == 51
    assert controls["band_random_search_candidates"] == 50
    assert controls["band_dirichlet_concentrations"] == [0.1, 1.0, 10.0]
    assert controls["band_ratios"] == [
        0.0001,
        0.001,
        0.01,
        0.1,
        1.0,
        10.0,
        100.0,
        1000.0,
        10000.0,
    ]
    assert controls["matched_random_bank_count"] == 99
    assert spec["power"]["seed"] == 2026082704


def test_load_spec_reexports_neutral_schema_loader() -> None:
    assert load_spec(CONFIG_PATH) == load_config(CONFIG_PATH)
    assert config_digest(CONFIG_PATH) == scientific_fingerprint(load_config(CONFIG_PATH))


def test_scientific_projection_matches_baseline() -> None:
    assert scientific_fingerprint(load_config(CONFIG_PATH)) == BASELINE_SCIENTIFIC_FINGERPRINT


def test_config_validation_entrypoint_is_lightweight(capsys: pytest.CaptureFixture[str]) -> None:
    assert config_main(["--config", str(CONFIG_PATH), "--json"]) == 0
    assert BASELINE_SCIENTIFIC_FINGERPRINT in capsys.readouterr().out


def test_missing_inherited_value_is_rejected() -> None:
    spec = load_config(CONFIG_PATH)
    del spec["baselines"]["position_plateau_min_tr"]
    with pytest.raises(ConfigError, match="position_plateau_min_tr"):
        validate_config(spec)


def test_config_contains_only_relative_runtime_paths() -> None:
    spec = load_config(CONFIG_PATH)
    values = list(spec["paths"].values()) + [
        story["brain_bundle"] for story in spec["stories"].values()
    ]
    assert values
    assert all(not Path(value).is_absolute() for value in values)
    assert all(".." not in Path(value).parts for value in values)


def test_explicit_roots_override_environment(tmp_path: Path) -> None:
    environment = {
        DATA_ROOT_ENV: str(tmp_path / "env-data"),
        OUTPUT_ROOT_ENV: str(tmp_path / "env-output"),
        CACHE_ROOT_ENV: str(tmp_path / "env-cache"),
    }
    resolved = RuntimePaths.resolve(
        data_root=tmp_path / "cli-data",
        output_root=tmp_path / "cli-output",
        cache_root=tmp_path / "cli-cache",
        environ=environment,
    )
    assert resolved.data_root == (tmp_path / "cli-data").resolve()
    assert resolved.output_root == (tmp_path / "cli-output").resolve()
    assert resolved.cache_root == (tmp_path / "cli-cache").resolve()


def test_environment_roots_are_used_when_cli_values_are_absent(tmp_path: Path) -> None:
    environment = {
        DATA_ROOT_ENV: str(tmp_path / "data"),
        OUTPUT_ROOT_ENV: str(tmp_path / "output"),
        CACHE_ROOT_ENV: str(tmp_path / "cache"),
    }
    resolved = RuntimePaths.resolve(environ=environment)
    assert resolved.data_root == (tmp_path / "data").resolve()
    assert resolved.output_root == (tmp_path / "output").resolve()
    assert resolved.cache_root == (tmp_path / "cache").resolve()


def test_integrity_root_helpers_share_the_same_precedence(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    environment = {
        DATA_ROOT_ENV: str(data),
        OUTPUT_ROOT_ENV: str(tmp_path / "output"),
        CACHE_ROOT_ENV: str(tmp_path / "cache"),
    }
    assert resolve_data_root(environ=environment) == data.resolve()
    assert resolve_output_root(environ=environment) == (tmp_path / "output").resolve()
    assert resolve_cache_root(environ=environment) == (tmp_path / "cache").resolve()


def test_missing_root_fails_only_when_required() -> None:
    resolved = RuntimePaths.resolve(environ={})
    assert resolved.data_root is None
    with pytest.raises(PathConfigurationError, match="FMRI_SAE_DATA_ROOT"):
        resolved.require_data_root()


def test_resolve_below_rejects_absolute_and_escaping_paths(tmp_path: Path) -> None:
    assert resolve_below(tmp_path, "nested/file.npz") == (tmp_path / "nested/file.npz").resolve()
    with pytest.raises(PathConfigurationError, match="relative"):
        resolve_below(tmp_path, tmp_path / "absolute.npz")
    with pytest.raises(PathConfigurationError, match="escapes"):
        resolve_below(tmp_path, "../outside.npz")


def test_resolve_path_uses_selected_runtime_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.setenv(DATA_ROOT_ENV, str(data))
    monkeypatch.setenv(OUTPUT_ROOT_ENV, str(tmp_path / "output"))
    monkeypatch.setenv(CACHE_ROOT_ENV, str(tmp_path / "cache"))
    spec = load_config(CONFIG_PATH)
    assert resolve_path(CONFIG_PATH, spec, "transcript_root", root="data") == data / "transcripts"
    assert resolve_path(CONFIG_PATH, spec, "artifact_root", root="output") == tmp_path / "output/artifacts"


def test_scientific_fingerprint_ignores_runtime_paths_and_release_metadata() -> None:
    spec = load_config(CONFIG_PATH)
    relocated = copy.deepcopy(spec)
    relocated["paths"]["artifact_root"] = "different-output-layout"
    relocated["stories"]["shapessocial"]["brain_bundle"] = "elsewhere/social.npz"
    relocated["status"] = "replicated"
    assert scientific_fingerprint(relocated) == scientific_fingerprint(spec)


def test_scientific_fingerprint_changes_with_scientific_contract() -> None:
    spec = load_config(CONFIG_PATH)
    changed = copy.deepcopy(spec)
    changed["models"]["gemma_2_2b"]["display_name"] = "Gemma Two 2B"
    assert scientific_fingerprint(changed) != scientific_fingerprint(spec)


def test_artifact_fingerprint_is_independent_of_root(tmp_path: Path) -> None:
    first = tmp_path / "first" / "payload.npz"
    second = tmp_path / "second" / "renamed.npz"
    first.parent.mkdir()
    second.parent.mkdir()
    first.write_bytes(b"identical numerical payload")
    second.write_bytes(first.read_bytes())
    left = artifact_fingerprint({"brain_targets": first}, namespace="fit-inputs")
    right = artifact_fingerprint({"brain_targets": second}, namespace="fit-inputs")
    assert left == right


def test_artifact_fingerprint_changes_with_bytes_role_or_science(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"version-one")
    spec = load_config(CONFIG_PATH)
    original = artifact_fingerprint(
        {"predictor": artifact}, namespace="fit-inputs", scientific_config=spec
    )
    role_changed = artifact_fingerprint(
        {"target": artifact}, namespace="fit-inputs", scientific_config=spec
    )
    artifact.write_bytes(b"version-two")
    bytes_changed = artifact_fingerprint(
        {"predictor": artifact}, namespace="fit-inputs", scientific_config=spec
    )
    assert original != role_changed
    assert original != bytes_changed


def test_artifact_fingerprint_requires_logical_namespace_and_inputs() -> None:
    with pytest.raises(FingerprintError, match="namespace"):
        artifact_fingerprint({}, namespace="")
    with pytest.raises(FingerprintError, match="at least one artifact"):
        artifact_fingerprint({}, namespace="fit-inputs")


def test_tree_and_provenance_fingerprints_ignore_external_roots(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    for root in (first, second):
        (root / "nested").mkdir(parents=True)
        (root / "nested/input.bin").write_bytes(b"same input")
        (root / "stage.py").write_text("VALUE = 1\n", encoding="utf-8")
    assert tree_digest(first) == tree_digest(second)
    left = provenance_fingerprint(
        CONFIG_PATH,
        input_paths=[first / "nested/input.bin"],
        code_paths=[first / "stage.py"],
        extra={"stage": "fit", "fold": 2},
    )
    right = provenance_fingerprint(
        CONFIG_PATH,
        input_paths=[second / "nested/input.bin"],
        code_paths=[second / "stage.py"],
        extra={"stage": "fit", "fold": 2},
    )
    assert left == right


def test_atomic_cache_marker_round_trip_and_stale_rejection(tmp_path: Path) -> None:
    marker = tmp_path / "cache/summary.json"
    atomic_write_json(marker, {"analysis_fingerprint": "expected", "complete": True})
    assert assert_cache_fingerprint(marker, "expected") == {
        "analysis_fingerprint": "expected",
        "complete": True,
    }
    with pytest.raises(AnalysisError, match="stale cache"):
        assert_cache_fingerprint(marker, "different")
    assert assert_cache_fingerprint(tmp_path / "missing.json", "anything") is None
