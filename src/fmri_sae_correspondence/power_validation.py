#!/usr/bin/env python3
"""Validate one power shard against its fixed design and configuration."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import yaml


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_replicate_ids(
    config_path: Path, shard_index: int, shard_count: int
) -> dict[tuple[float, float], list[int]]:
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    power = spec["power"]
    null_reps = int(power["null_replicates"])
    positive_reps = int(power["positive_replicates_per_grid_cell"])
    expected: dict[tuple[float, float], list[int]] = {}
    for raw_rho in power["oracle_correlations"]:
        rho = float(raw_rho)
        for raw_scale in power["residual_scale_multipliers"]:
            scale = float(raw_scale)
            count = null_reps if math.isclose(rho, 0.0) else positive_reps
            expected[(rho, scale)] = list(range(shard_index, count, shard_count))
    return expected


def validate(
    path: Path,
    config_path: Path,
    *,
    design_sha256: str,
    design_label: str,
    model: str,
    bank_type: str,
    direction: str,
    shard_index: int,
    shard_count: int,
) -> tuple[bool, str]:
    sidecar = path.with_name(path.name + ".sha256")
    if not path.is_file() or not sidecar.is_file():
        return False, "shard or SHA-256 sidecar is missing"
    recorded = sidecar.read_text(encoding="utf-8").split()
    if not recorded or recorded[0] != file_sha256(path):
        return False, "SHA-256 sidecar does not match shard bytes"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, f"shard JSON is unreadable: {exc}"
    expected_meta = {
        "schema_version": 2,
        "artifact_kind": "power_shard",
        "scientific_evidence": False,
        "design_schema": "paired_target_design_v1",
        "oracle_correlation_method": "spearman",
        "residual_scale_definition": "mean per-target population SD of the corresponding empirical residual pool",
        "design_sha256": design_sha256,
        "design_label": design_label,
        "model": model,
        "bank_type": bank_type,
        "direction": direction,
        "shard_index": shard_index,
        "shard_count": shard_count,
    }
    for key, expected in expected_meta.items():
        if record.get(key) != expected:
            return False, f"metadata mismatch for {key}"
    for key in ("brain_empirical_residual_sd", "lm_empirical_residual_sd"):
        try:
            value = float(record[key])
        except (KeyError, TypeError, ValueError):
            return False, f"{key} is missing or malformed"
        if not math.isfinite(value) or value <= 0:
            return False, f"{key} must be finite and positive"
    expected_rows = expected_replicate_ids(config_path, shard_index, shard_count)
    spec = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    expected_alpha_grid = [float(value) for value in spec["cross_validation"]["alpha_grid"]]
    try:
        actual_alpha_grid = [float(value) for value in record["alpha_grid"]]
    except (KeyError, TypeError, ValueError):
        return False, "alpha_grid is missing or malformed"
    if actual_alpha_grid != expected_alpha_grid:
        return False, "alpha_grid does not match the configured specification"
    rows: dict[tuple[float, float], bool] = {}
    for row in record.get("rows", []):
        key = (float(row["oracle_correlation"]), float(row["residual_scale_multiplier"]))
        if key in rows:
            return False, f"duplicate grid cell {key}"
        ids = [int(value) for value in row.get("replicate_ids", [])]
        values = row.get("values", [])
        if ids != expected_rows.get(key) or len(ids) != len(values):
            return False, f"replicate coverage mismatch for grid cell {key}"
        try:
            if not all(math.isfinite(float(value)) for value in values):
                return False, f"non-finite replicate value for grid cell {key}"
        except (TypeError, ValueError):
            return False, f"malformed replicate value for grid cell {key}"
        rows[key] = True
    if set(rows) != set(expected_rows):
        return False, "grid-cell coverage is incomplete"
    return True, "valid"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--design-sha256", required=True)
    parser.add_argument("--design-label", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--bank-type", required=True)
    parser.add_argument("--direction", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args(argv)
    valid, reason = validate(
        args.path,
        args.config,
        design_sha256=args.design_sha256,
        design_label=args.design_label,
        model=args.model,
        bank_type=args.bank_type,
        direction=args.direction,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    if not valid:
        print(reason)
        return 1
    print(reason)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
