"""Runner adapter for the analysis reliability diagnostics.

The underlying analysis keeps full parcelwise and split-half arrays.  This
adapter binds it to the documented specification, verifies the resulting artifact
hashes, and writes a cache marker whose fingerprint includes inputs, code, and
configuration.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .config import default_config_path
from .integrity import (
    assert_cache_fingerprint,
    atomic_write_json,
    provenance_fingerprint,
    resolve_data_root,
    resolve_output_root,
    sha256_file,
)
from .isc_reliability import main as reliability_main
from .reliance_stability import main as primary_stability_main

HERE = Path(__file__).resolve().parent


def _data_path(config_path: Path, spec: dict[str, Any], value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (resolve_data_root(config_path, spec) / candidate).resolve()


def run(*, config_path: Path, spec: dict[str, Any], args: Any) -> dict[str, Any]:
    output_root = (
        resolve_output_root(spec=spec)
        / str(spec["paths"]["artifact_root"])
        / "reliability"
    ).resolve()
    physical = _data_path(config_path, spec, spec["stories"]["shapesphysical"]["brain_bundle"])
    social = _data_path(config_path, spec, spec["stories"]["shapessocial"]["brain_bundle"])
    code = [
        HERE / "reliability.py",
        HERE / "isc_reliability.py",
        HERE / "reliance_stability.py",
    ]
    fingerprint = provenance_fingerprint(
        config_path,
        input_paths=[physical, social],
        code_paths=code,
        extra={"stage": "reliability", "split_draws": 10000, "bootstrap_draws": 10000, "seed": 20260827},
    )
    marker = output_root / "stage_summary.json"
    cached = None if bool(getattr(args, "force", False)) else assert_cache_fingerprint(marker, fingerprint)
    if cached is not None:
        return cached

    status = reliability_main(
        [
            "--physical", str(physical),
            "--social", str(social),
            "--output-root", str(output_root / "bold"),
            "--split-draws", "10000",
            "--bootstrap-draws", "10000",
            "--seed", "20260827",
        ]
    )
    if status != 0:
        raise RuntimeError(f"BOLD reliability analysis exited with status {status}")
    primary_stability_status = primary_stability_main(
        [
            "--artifact-root", str((resolve_output_root(spec=spec) / str(spec["paths"]["artifact_root"])).resolve()),
            "--output-root", str(output_root / "primary_reliance_profile_stability"),
            "--bootstrap-draws", "10000",
            "--seed", "20260827",
        ]
    )
    if primary_stability_status != 0:
        raise RuntimeError(f"primary reliance stability analysis exited with status {primary_stability_status}")

    aggregate_path = output_root / "bold" / "aggregate.json"
    primary_stability_path = output_root / "primary_reliance_profile_stability" / "aggregate.json"
    if not aggregate_path.is_file() or not primary_stability_path.is_file():
        raise RuntimeError("reliability stage did not produce all required aggregate artifacts")
    result = {
        "schema_version": 1,
        "stage": "reliability",
        "analysis_fingerprint": fingerprint,
        "bold_aggregate": str(aggregate_path),
        "bold_aggregate_sha256": sha256_file(aggregate_path),
        "primary_reliance_stability_aggregate": str(primary_stability_path),
        "primary_reliance_stability_aggregate_sha256": sha256_file(primary_stability_path),
        "interpretation_boundary": "Shared-response reliability diagnostics, not exact noise ceilings.",
    }
    atomic_write_json(marker, result)
    return json.loads(marker.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    if args.data_root is not None:
        os.environ["FMRI_SAE_DATA_ROOT"] = str(args.data_root.resolve())
    if args.output_root is not None:
        os.environ["FMRI_SAE_OUTPUT_ROOT"] = str(args.output_root.resolve())
    from .integrity import load_spec

    config_path = (args.config or default_config_path()).resolve()
    result = run(config_path=config_path, spec=load_spec(config_path), args=args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
