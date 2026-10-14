#!/usr/bin/env python3
"""Validate and summarize the completed semi-synthetic power grid."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from .paths import release_path

EXPECTED_MODELS = ("gemma_2_2b", "llama_3_1_8b")
EXPECTED_BANKS = ("source_story_correlation_pruned", "source_story_variance")
EXPECTED_DIRECTIONS = (
    "shapesphysical_to_shapessocial",
    "shapessocial_to_shapesphysical",
)
EXPECTED_RHOS = (0.0, 0.2, 0.4, 0.6, 0.8)
EXPECTED_RESIDUAL_SCALES = (0.25, 0.5, 1.0, 2.0)
EXPECTED_SHARDS_PER_DESIGN = 32
EXPECTED_NULL_REPLICATES = 500
EXPECTED_POSITIVE_REPLICATES = 250
IDENTIFIABILITY_THRESHOLD = 0.8
REFERENCE_RHO = 0.4


class PowerSummaryError(RuntimeError):
    """Raised when the power evidence is incomplete or malformed."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PowerSummaryError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PowerSummaryError(f"{path}: expected a JSON object")
    return value


def _as_float(value: Any, *, field: str, path: Path) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise PowerSummaryError(f"{path}: invalid {field}: {value!r}") from exc
    if not 0.0 <= result <= 1.0 and field == "detection_fraction":
        raise PowerSummaryError(f"{path}: {field} outside [0, 1]: {result}")
    return result


def _expected_designs() -> set[tuple[str, str, str]]:
    return {
        (model, bank, direction)
        for model in EXPECTED_MODELS
        for bank in EXPECTED_BANKS
        for direction in EXPECTED_DIRECTIONS
    }


def read_design(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    if payload.get("analysis_role") != "semi_synthetic_power":
        raise PowerSummaryError(f"{path}: not a scientific power aggregate")
    if payload.get("scientific_evidence") is not True:
        raise PowerSummaryError(f"{path}: scientific_evidence is not true")

    model = str(payload.get("model", ""))
    bank = str(payload.get("bank_type", ""))
    direction = str(payload.get("direction", ""))
    design = (model, bank, direction)
    if design not in _expected_designs():
        raise PowerSummaryError(f"{path}: unexpected design {design}")
    if int(payload.get("shard_count", -1)) != EXPECTED_SHARDS_PER_DESIGN:
        raise PowerSummaryError(f"{path}: expected 32 shards")
    shard_paths = payload.get("shard_paths")
    if not isinstance(shard_paths, list) or len(shard_paths) != EXPECTED_SHARDS_PER_DESIGN:
        raise PowerSummaryError(f"{path}: expected 32 recorded shard paths")
    valid_shard_paths = all(isinstance(item, str) for item in shard_paths)
    if not valid_shard_paths or len(set(shard_paths)) != len(shard_paths):
        raise PowerSummaryError(f"{path}: shard paths must be unique strings")
    expected_label = "/".join(design)
    if payload.get("design_label") != expected_label:
        raise PowerSummaryError(f"{path}: design_label does not match its coordinates")
    design_sha256 = payload.get("design_sha256")
    if (
        not isinstance(design_sha256, str)
        or len(design_sha256) != 64
        or any(character not in "0123456789abcdef" for character in design_sha256)
    ):
        raise PowerSummaryError(f"{path}: missing or malformed design_sha256")

    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise PowerSummaryError(f"{path}: rows must be a list")

    expected_coordinates = {
        (rho, scale) for rho in EXPECTED_RHOS for scale in EXPECTED_RESIDUAL_SCALES
    }
    parsed_rows: list[dict[str, float | int]] = []
    seen_coordinates: set[tuple[float, float]] = set()
    for row in rows:
        if not isinstance(row, dict):
            raise PowerSummaryError(f"{path}: every row must be an object")
        rho = _as_float(row.get("oracle_correlation"), field="oracle_correlation", path=path)
        scale = _as_float(
            row.get("residual_scale_multiplier"),
            field="residual_scale_multiplier",
            path=path,
        )
        detection = _as_float(
            row.get("detection_fraction"), field="detection_fraction", path=path
        )
        replicates = int(row.get("replicates", -1))
        expected_replicates = (
            EXPECTED_NULL_REPLICATES if rho == 0.0 else EXPECTED_POSITIVE_REPLICATES
        )
        if replicates != expected_replicates:
            raise PowerSummaryError(
                f"{path}: ({rho}, {scale}) has {replicates} replicates; "
                f"expected {expected_replicates}"
            )
        coordinate = (rho, scale)
        if coordinate in seen_coordinates:
            raise PowerSummaryError(f"{path}: duplicate coordinate {coordinate}")
        seen_coordinates.add(coordinate)
        parsed_rows.append(
            {
                "oracle_correlation": rho,
                "residual_scale_multiplier": scale,
                "detection_fraction": detection,
                "replicates": replicates,
            }
        )
    if seen_coordinates != expected_coordinates:
        missing = sorted(expected_coordinates - seen_coordinates)
        extra = sorted(seen_coordinates - expected_coordinates)
        raise PowerSummaryError(f"{path}: grid mismatch; missing={missing}, extra={extra}")

    return {
        "model": model,
        "bank_type": bank,
        "direction": direction,
        "design_label": payload.get("design_label"),
        "design_sha256": payload.get("design_sha256"),
        "aggregate_sha256": sha256_file(path),
        "aggregate_file": path.name,
        "shard_count": EXPECTED_SHARDS_PER_DESIGN,
        "rows": sorted(
            parsed_rows,
            key=lambda row: (
                float(row["oracle_correlation"]),
                float(row["residual_scale_multiplier"]),
            ),
        ),
    }


def build_summary(power_dir: Path) -> dict[str, Any]:
    paths = sorted(
        path
        for path in power_dir.glob("power_*.json")
        if path.name not in {"power_status.json"}
    )
    designs = [read_design(path) for path in paths]
    observed = {
        (str(item["model"]), str(item["bank_type"]), str(item["direction"]))
        for item in designs
    }
    expected = _expected_designs()
    if observed != expected or len(designs) != len(expected):
        raise PowerSummaryError(
            f"expected exactly eight designs; missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )

    values: dict[tuple[str, float], list[float]] = defaultdict(list)
    for design in designs:
        for row in design["rows"]:
            values[(str(design["model"]), float(row["oracle_correlation"]))].append(
                float(row["detection_fraction"])
            )

    families: dict[str, Any] = {}
    for model in EXPECTED_MODELS:
        by_rho: dict[str, Any] = {}
        model_values: list[float] = []
        for rho in EXPECTED_RHOS:
            cell_values = values[(model, rho)]
            if len(cell_values) != 16:
                raise PowerSummaryError(
                    f"{model} rho={rho}: expected 16 cells, found {len(cell_values)}"
                )
            model_values.extend(cell_values)
            by_rho[f"{rho:.1f}"] = {
                "minimum": min(cell_values),
                "maximum": max(cell_values),
                "mean": round(sum(cell_values) / len(cell_values), 12),
                "cells_at_or_above_0.80": sum(
                    value >= IDENTIFIABILITY_THRESHOLD for value in cell_values
                ),
            }
        reference = by_rho[f"{REFERENCE_RHO:.1f}"]
        families[model] = {
            "cells": len(model_values),
            "cells_at_or_above_0.80": sum(
                value >= IDENTIFIABILITY_THRESHOLD for value in model_values
            ),
            "maximum_detection_fraction": max(model_values),
            "configured_rho_0.40_threshold_met": (
                reference["maximum"] >= IDENTIFIABILITY_THRESHOLD
            ),
            "by_oracle_correlation": by_rho,
        }

    rho_zero = [
        float(row["detection_fraction"])
        for design in designs
        for row in design["rows"]
        if float(row["oracle_correlation"]) == 0.0
    ]
    return {
        "schema_version": 1,
        "analysis_role": "completed_two_family_power_summary",
        "generator_sha256": sha256_file(Path(__file__).resolve()),
        "grid": {
            "models": list(EXPECTED_MODELS),
            "designs": len(designs),
            "shards": sum(int(item["shard_count"]) for item in designs),
            "cells": sum(len(item["rows"]) for item in designs),
            "oracle_correlations": list(EXPECTED_RHOS),
            "residual_scale_multipliers": list(EXPECTED_RESIDUAL_SCALES),
            "null_replicates_per_cell": EXPECTED_NULL_REPLICATES,
            "positive_replicates_per_cell": EXPECTED_POSITIVE_REPLICATES,
        },
        "decision_rule": {
            "identifiability_threshold": IDENTIFIABILITY_THRESHOLD,
            "configured_reference_oracle_correlation": REFERENCE_RHO,
            "coverage_complete": True,
            "configured_detectability_criterion_met": all(
                family["configured_rho_0.40_threshold_met"]
                for family in families.values()
            ),
            "interpretation": "bounded_non_detection_underidentified",
        },
        "rho_zero_realized_detection_fraction": {
            "cells": len(rho_zero),
            "minimum": min(rho_zero),
            "maximum": max(rho_zero),
            "mean": round(sum(rho_zero) / len(rho_zero), 12),
        },
        "families": families,
        "source_aggregates": [
            {key: design[key] for key in (
                "aggregate_file",
                "aggregate_sha256",
                "design_label",
                "design_sha256",
                "model",
                "bank_type",
                "direction",
                "shard_count",
            )}
            for design in designs
        ],
        "claim_boundary": {
            "allowed": (
                "No reproducible correspondence was established; the completed grid "
                "shows that moderate correspondence remains unresolved."
            ),
            "forbidden": "The data establish that sparse-feature correspondence is absent.",
        },
    }


def render_tex(summary: dict[str, Any]) -> str:
    rows = []
    for rho in EXPECTED_RHOS:
        cells = []
        for model in EXPECTED_MODELS:
            values = summary["families"][model]["by_oracle_correlation"][f"{rho:.1f}"]
            cells.append(
                f"${_tex_number(values['minimum'])}$--${_tex_number(values['maximum'])}$"
            )
        rows.append(f"${_tex_number(rho)}$ & {cells[0]} & {cells[1]} \\\\")
    return "\n".join(
        [
            "% Generated by summarize_power.py; do not edit by hand.",
            "\\begin{tabular}{lrr}",
            "\\toprule",
            "Oracle $\\rho$ & Gemma 2 2B & Llama 3.1 8B \\\\",
            "\\midrule",
            *rows,
            "\\bottomrule",
            "\\end{tabular}",
            "",
        ]
    )


def _tex_number(value: float, digits: int = 2) -> str:
    rendered = f"{value:.{digits}f}"
    return rendered[1:] if rendered.startswith("0.") else rendered


def render_macros(summary: dict[str, Any]) -> str:
    gemma = summary["families"]["gemma_2_2b"]
    llama = summary["families"]["llama_3_1_8b"]

    def interval(family: dict[str, Any], rho: float) -> str:
        values = family["by_oracle_correlation"][f"{rho:.1f}"]
        return f"{_tex_number(values['minimum'])}--{_tex_number(values['maximum'])}"

    values = {
        "PowerTotalDesigns": str(summary["grid"]["designs"]),
        "PowerTotalShards": str(summary["grid"]["shards"]),
        "PowerNullRate": _tex_number(
            summary["rho_zero_realized_detection_fraction"]["mean"], 3
        ),
        "PowerGemmaIdentifiableCells": str(gemma["cells_at_or_above_0.80"]),
        "PowerGemmaMaximum": _tex_number(gemma["maximum_detection_fraction"]),
        "PowerLlamaMaximum": _tex_number(llama["maximum_detection_fraction"]),
        "PowerGemmaAtFourRange": interval(gemma, 0.4),
        "PowerLlamaAtFourRange": interval(llama, 0.4),
        "PowerGemmaAtSixRange": interval(gemma, 0.6),
        "PowerLlamaAtSixRange": interval(llama, 0.6),
    }
    return "\n".join(
        [
            "% Generated by summarize_power.py; do not edit by hand.",
            *(f"\\newcommand{{\\{name}}}{{{value}}}" for name, value in values.items()),
            "",
        ]
    )


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def verify_outputs(power_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Verify that committed reports are exact transforms of the aggregates."""

    summary = build_summary(power_dir)
    expected = {
        "power_summary.json": json.dumps(
            summary, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        "power_summary_table.tex": render_tex(summary),
        "power_summary_macros.tex": render_macros(summary),
    }
    mismatches = [
        name
        for name, content in expected.items()
        if not (output_dir / name).is_file()
        or (output_dir / name).read_text(encoding="utf-8") != content
    ]
    if mismatches:
        raise PowerSummaryError(
            "report products do not match the shipped aggregates: "
            + ", ".join(mismatches)
        )
    return {"aggregates": 8, "reports": len(expected), "verified": True}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power-dir", type=Path, default=release_path("evidence/power"))
    parser.add_argument("--output-dir", type=Path, default=release_path("results"))
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    try:
        power_dir = args.power_dir.resolve()
        output_dir = args.output_dir.resolve()
        if not args.write:
            print(json.dumps(verify_outputs(power_dir, output_dir), sort_keys=True))
            return 0
        summary = build_summary(power_dir)
        output_paths = [
            output_dir / "power_summary.json",
            output_dir / "power_summary_table.tex",
            output_dir / "power_summary_macros.tex",
        ]
        existing = [str(path) for path in output_paths if path.exists()]
        if existing and not args.force:
            raise PowerSummaryError(
                "refusing to overwrite report products without --force: "
                + ", ".join(existing)
            )
        atomic_write(
            output_dir / "power_summary.json",
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        atomic_write(output_dir / "power_summary_table.tex", render_tex(summary))
        atomic_write(output_dir / "power_summary_macros.tex", render_macros(summary))
    except (OSError, PowerSummaryError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"designs": 8, "output_dir": str(output_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
