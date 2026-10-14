from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_source_catalog_matches_all_released_scientific_files() -> None:
    manifest = json.loads((ROOT / "checksums/SCIENTIFIC_CODE_MANIFEST.json").read_text())
    assert manifest["file_count"] == 36
    assert len(manifest["files"]) == 36
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        assert path.is_file()
        assert path.stat().st_size == entry["size_bytes"]
        assert sha256(path) == entry["sha256"]


def test_source_only_verification_states_its_boundary() -> None:
    result = subprocess.run(
        ["bash", "verify.sh"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "source catalog verified: 36 files" in result.stdout
    assert "derived scientific evidence was not checked" in result.stdout


def test_released_power_summary_is_complete_and_bounded() -> None:
    summary = json.loads((ROOT / "results/power_summary.json").read_text())
    assert summary["grid"]["designs"] == 8
    assert summary["grid"]["cells"] == 160
    assert summary["grid"]["shards"] == 256
    assert len(summary["source_aggregates"]) == 8
    assert summary["decision_rule"]["coverage_complete"] is True
    assert summary["decision_rule"]["configured_detectability_criterion_met"] is False
    assert summary["decision_rule"]["interpretation"] == "bounded_non_detection_underidentified"
    assert summary["families"]["gemma_2_2b"]["by_oracle_correlation"]["0.4"]["minimum"] == 0.22
    assert summary["families"]["llama_3_1_8b"]["by_oracle_correlation"]["0.4"]["maximum"] == 0.248


def test_power_catalog_agrees_with_summary() -> None:
    summary = json.loads((ROOT / "results/power_summary.json").read_text())
    expected = {
        item["aggregate_file"]: item["aggregate_sha256"]
        for item in summary["source_aggregates"]
    }
    observed: dict[str, str] = {}
    for line in (ROOT / "checksums/power_aggregates.sha256").read_text().splitlines():
        digest, filename = line.split(maxsplit=1)
        observed[filename.lstrip("* ")] = digest
    assert observed == expected
