#!/usr/bin/env python3
"""Aggregate within-story power shards (exploratory sensitivity).

This calls the configured power stage's aggregation function unchanged, with
the same two declared specification changes as ``run_within_story_shard.py``:
the ``directions`` list is extended with the design's within-story label, and
``residual_scale_multipliers`` is set to exactly the scales whose shards are
being pooled so that the aggregator's expected-cell check matches the shards.

Several shard directories may be supplied.  Records are merged by shard index
before aggregation because the aggregator requires exactly one record per
shard; shard sets that cover disjoint residual scales therefore combine into
one record per index whose rows are the union.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from run_within_story_shard import within_story_power_spec

from fmri_sae_correspondence.config import default_config_path, load_config
from fmri_sae_correspondence.power import _load_design, aggregate_power_shards


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--shards-dir", type=Path, action="append", required=True)
    parser.add_argument("--scales", required=True, help="comma-separated, e.g. 1.0 or 0.25,0.5,1.0,2.0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config_path = (args.config or default_config_path()).resolve()
    spec = load_config(config_path)
    design = _load_design(args.design.resolve())
    power_spec = within_story_power_spec(spec, design, args.scales)

    by_index: dict[int, dict] = {}
    for directory in args.shards_dir:
        paths = sorted(Path(directory).glob("*.json"))
        if not paths:
            raise SystemExit(f"no shards in {directory}")
        for path in paths:
            record = json.loads(path.read_text(encoding="utf-8"))
            index = int(record["shard_index"])
            if index in by_index:
                by_index[index]["rows"] = [*by_index[index].get("rows", []), *record.get("rows", [])]
            else:
                by_index[index] = record
    merged = [by_index[index] for index in sorted(by_index)]

    with tempfile.TemporaryDirectory() as temporary:
        paths = []
        for record in merged:
            path = Path(temporary) / f"shard_{int(record['shard_index']):04d}.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            paths.append(path)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = aggregate_power_shards(design, power_spec, paths, output_path=args.output)

    print(
        json.dumps(
            {
                "design": str(design.get("direction", "")),
                "scales": power_spec["residual_scale_multipliers"],
                "output": str(args.output),
                "rows": len(result.get("rows", [])),
            },
            default=str,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
