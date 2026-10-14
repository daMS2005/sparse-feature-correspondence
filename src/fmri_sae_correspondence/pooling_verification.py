#!/usr/bin/env python3
"""Fail-closed verification for the complete token-pooling materialization gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .extraction import _load_spec
from .pooling import _pooling_root

EXPECTED_POOLINGS = frozenset(
    {
        "mean_positive_activation",
        "max_positive_activation",
        "sum_positive_activation",
        "event_count",
    }
)


def verify(config_path: Path) -> dict[str, Any]:
    config_path = Path(config_path).resolve()
    spec = _load_spec(config_path)
    root = _pooling_root(config_path, spec)
    missing: list[str] = []
    manifests: list[dict[str, Any]] = []
    for model in spec["model_sets"]["all_models"]:
        for story in spec["stories"]:
            path = root / model / story / "manifest.json"
            if not path.is_file():
                missing.append(f"manifest:{model}/{story}")
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if set(record.get("outputs", {})) != EXPECTED_POOLINGS:
                missing.append(f"pooling_outputs:{model}/{story}")
            if not record.get("analysis_fingerprint"):
                missing.append(f"analysis_fingerprint:{model}/{story}")
            if not record.get("token_records_sha256"):
                missing.append(f"token_records_sha256:{model}/{story}")
            manifests.append({"model_key": model, "story": story, "path": str(path)})
    if missing:
        raise RuntimeError("pooling gate failed closed: " + ", ".join(missing))
    return {
        "schema_version": 1,
        "analysis_role": "correspondence_pooling_materialization_gate",
        "config_path": str(config_path),
        "model_story_count": len(manifests),
        "pooling_names": sorted(EXPECTED_POOLINGS),
        "manifests": manifests,
        "complete": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    result = verify(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
