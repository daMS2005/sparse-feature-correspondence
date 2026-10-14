from __future__ import annotations

import copy
import os
import subprocess
import sys
from pathlib import Path

import pytest

from fmri_sae_correspondence.config import ConfigError, load_config, validate_config

ROOT = Path(__file__).resolve().parents[1]


def test_analysis_config_loads_and_validates() -> None:
    spec = load_config()
    assert spec["schema_version"] == 2
    assert spec["cross_validation"]["outer_folds"] == 5
    assert spec["cross_validation"]["purge_adjacent_trs"] == 4
    assert set(spec["stories"]) == {"shapesphysical", "shapessocial"}


def test_validation_rejects_changed_purge_window() -> None:
    spec = copy.deepcopy(load_config())
    spec["cross_validation"]["purge_adjacent_trs"] = 0
    with pytest.raises(ConfigError, match="purge_adjacent_trs"):
        validate_config(spec)


def test_public_cli_help_is_lightweight() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-m", "fmri_sae_correspondence", "--help"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "fmri-sae-correspondence" in result.stdout


def test_public_cli_version_is_lightweight() -> None:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-m", "fmri_sae_correspondence", "--version"],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.1.4"
