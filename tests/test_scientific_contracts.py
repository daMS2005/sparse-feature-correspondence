from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fmri_sae_correspondence.feature_banks import build_ranked_banks
from fmri_sae_correspondence.modeling import block_labels, select_alpha
from fmri_sae_correspondence.power_reporting import PowerSummaryError, read_design

ROOT = Path(__file__).resolve().parents[1]


def test_ranked_feature_banks_are_nested_prefixes() -> None:
    values = np.zeros((2, 24, 8), dtype=float)
    for layer in range(values.shape[0]):
        for feature in range(values.shape[2]):
            period = feature + 2
            values[layer, ::period, feature] = 1.0 + layer + feature / 10
    banks = build_ranked_banks(
        values,
        [4, 8],
        np.arange(values.shape[1]),
        method="source_variance",
        sizes=[4, 8],
        min_active_trs=2,
        max_active_fraction=0.95,
    )
    small = banks["banks"]["4"]["feature_names"]
    large = banks["banks"]["8"]["feature_names"]
    assert large[:4] == small


def test_ridge_selection_returns_a_configured_candidate() -> None:
    x = np.linspace(-1.0, 1.0, 60)[:, None]
    y = np.column_stack((2.0 * x[:, 0], -x[:, 0]))
    alphas = np.array([0.1, 1.0, 10.0])
    selected, scores = select_alpha(
        x,
        y,
        block_labels(len(x), 5),
        outer_block=0,
        alphas=alphas,
        metric="r2_oos",
        purge_adjacent=1,
    )
    assert selected in alphas
    assert scores.shape == alphas.shape
    assert np.all(np.isfinite(scores))


def test_malformed_power_aggregate_is_rejected(tmp_path: Path) -> None:
    source = next((ROOT / "evidence" / "power").glob("power_*.json"))
    payload = json.loads(source.read_text(encoding="utf-8"))
    payload["analysis_role"] = "unexpected"
    malformed = tmp_path / source.name
    malformed.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(PowerSummaryError, match="not a scientific power aggregate"):
        read_design(malformed)


def test_equivalence_record_verifies_all_eight_aggregates() -> None:
    record = json.loads(
        (ROOT / "provenance" / "SCIENTIFIC_EQUIVALENCE.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["verification"]["artifact_count"] == 8
    assert record["verification"]["all_numerical_projections_equal"] is True
    assert record["verification"]["all_scientific_projections_equal"] is True
    assert record["configuration"]["scientific_projection_equal"] is True
    assert all(item["numerical_projection_equal"] for item in record["artifacts"])


def test_cli_runs_outside_checkout_without_runtime_roots(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("FMRI_SAE_DATA_ROOT", None)
    environment.pop("FMRI_SAE_OUTPUT_ROOT", None)
    environment.pop("FMRI_SAE_CACHE_ROOT", None)
    result = subprocess.run(
        [sys.executable, "-m", "fmri_sae_correspondence", "config", "validate"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "fmri_sae_correspondence_v2" in result.stdout


def test_reporting_refuses_implicit_overwrite() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "fmri_sae_correspondence",
            "results",
            "verify",
            "--write",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "refusing to overwrite" in result.stderr
