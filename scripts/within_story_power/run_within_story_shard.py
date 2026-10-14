#!/usr/bin/env python3
"""Run one within-story power shard (exploratory sensitivity).

This calls the configured power stage's shard function unchanged.  The only
differences from the configured cross-story run are in the power specification
handed to it, both declared here and recorded in the shard output:

1. the permitted ``directions`` list is extended with the design's
   within-story label, which the coordinate validator would otherwise reject;
2. the residual-scale axis may be restricted to a subset of the configured
   multipliers (``--scales``), defaulting to the empirical level 1.0.

Alpha grid, seed contract, oracle correlations, replicate counts, and the
numerical core are those of the configured stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from fmri_sae_correspondence.config import default_config_path, load_config
from fmri_sae_correspondence.power import _load_design, run_power_grid_shard


def within_story_power_spec(spec: dict, design: dict, scales: str) -> dict:
    power_spec = dict(spec["power"])
    power_spec["alpha_grid"] = list(spec["cross_validation"]["alpha_grid"])
    power_spec["purge_adjacent_trs"] = int(spec["cross_validation"]["purge_adjacent_trs"])
    power_spec.setdefault("seed", spec["nulls"]["seeds"]["power"])
    configured = [float(value) for value in power_spec["residual_scale_multipliers"]]
    wanted = [float(value) for value in scales.split(",") if value.strip()]
    unknown = [value for value in wanted if not any(abs(value - c) < 1e-9 for c in configured)]
    if unknown:
        raise SystemExit(f"scales {unknown} are not in the configured grid {configured}")
    if wanted != configured:
        power_spec["residual_scale_multipliers"] = wanted
        power_spec["scales_restricted_from"] = configured
    direction = str(design.get("direction", ""))
    if direction not in {str(value) for value in power_spec["directions"]}:
        power_spec["directions"] = [*[str(value) for value in power_spec["directions"]], direction]
        power_spec["direction_added"] = direction
    return power_spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--shard-output", type=Path, required=True)
    parser.add_argument("--scales", default="1.0", help="comma-separated residual-scale multipliers")
    args = parser.parse_args()
    config_path = (args.config or default_config_path()).resolve()
    spec = load_config(config_path)
    design = _load_design(args.design.resolve())
    power_spec = within_story_power_spec(spec, design, args.scales)
    args.shard_output.parent.mkdir(parents=True, exist_ok=True)
    result = run_power_grid_shard(
        design,
        power_spec,
        output_path=args.shard_output,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    print(
        json.dumps(
            {
                "shard": args.shard_index,
                "design": str(design.get("direction", "")),
                "residual_scales": power_spec["residual_scale_multipliers"],
                "rows": result.get("rows_written", result.get("row_count")),
                "output": str(args.shard_output),
            },
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
