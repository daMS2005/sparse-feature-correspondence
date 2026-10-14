#!/usr/bin/env python3
"""Summarize the identifiability sweep outputs into results/identifiability_summary.json."""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "evidence/identifiability"


def _rows(directory: Path) -> list[dict]:
    rows: list[dict] = []
    for path in sorted(directory.glob("*.jsonl")):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


def _by_k(rows: list[dict], key: str, **filters: str) -> dict[str, dict[str, float]]:
    groups: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        if all(row.get(name) == value for name, value in filters.items()) and row.get(key) is not None:
            groups[int(row["K"])].append(float(row[key]))
    return {
        str(k): {"n": len(v), "median": float(np.median(v)), "q25": float(np.quantile(v, 0.25)), "q75": float(np.quantile(v, 0.75))}
        for k, v in sorted(groups.items())
    }


def summarize() -> dict:
    threshold = _rows(EVIDENCE / "threshold")
    return {
        "analysis_role": "identifiability_sweep_summary",
        "planted_correspondence": 0.8,
        "brain_self_recovery_by_k": _by_k(threshold, "self_recovery", system="brain"),
        "lm_self_recovery_by_k": _by_k(threshold, "self_recovery", system="lm"),
        "cross_system_transfer_by_k": _by_k(_rows(EVIDENCE / "cross_system_transfer"), "recovered"),
        "cross_system_within_by_k": _by_k(_rows(EVIDENCE / "cross_system_within"), "recovered"),
    }


def main(argv: list[str] | None = None) -> int:
    out = Path(argv[0]) if argv else ROOT / "results/identifiability_summary.json"
    summary = summarize()
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: {kk: round(vv["median"], 3) for kk, vv in v.items()} for k, v in summary.items() if k.endswith("_by_k")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
