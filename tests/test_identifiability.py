"""The committed identifiability summary matches its outputs and catalog."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_identifiability_summary_and_catalog_agree() -> None:
    summary = json.loads((ROOT / "results/identifiability_summary.json").read_text())
    brain = summary["brain_self_recovery_by_k"]
    assert brain["24"]["median"] > 0.95 and brain["128"]["median"] < 0.35
    transfer = summary["cross_system_transfer_by_k"]
    assert transfer["24"]["median"] > 0.5 and transfer["256"]["median"] < 0.1
    for line in (ROOT / "checksums/identifiability_outputs.sha256").read_text().splitlines():
        digest, path = line.split(maxsplit=1)
        assert hashlib.sha256((ROOT / "evidence/identifiability" / path.strip()).read_bytes()).hexdigest() == digest
    result = subprocess.run([sys.executable, "scripts/identifiability/summarize.py", "/dev/null"], cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
