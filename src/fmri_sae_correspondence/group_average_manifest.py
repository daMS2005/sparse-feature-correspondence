#!/usr/bin/env python3
"""Parallel-safe manifest materialization for group-average artifacts."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Sequence

PRIMARY_SIZE = 256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except BaseException:
        try:
            os.unlink(temp_name)
        except FileNotFoundError:
            pass
        raise


def materialize_group_average_manifest(
    group_root: Path,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Scan condition artifacts and atomically publish their union under a lock."""

    group_root = Path(group_root).resolve()
    group_root.mkdir(parents=True, exist_ok=True)
    lock_path = group_root / ".manifest.lock"
    manifest_path = group_root / "manifest.json"
    expected = {
        (str(model), str(story))
        for model in spec["models"]
        for story in spec["stories"]
    }
    with lock_path.open("a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            records: dict[tuple[str, str], dict[str, Any]] = {}
            pattern = "*/*/source_story_variance/n256/summary.json"
            for summary_path in sorted(group_root.glob(pattern)):
                relative = summary_path.relative_to(group_root)
                model, story, variant, size_name, _ = relative.parts
                try:
                    summary = json.loads(summary_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError as exc:
                    raise RuntimeError(f"invalid group-average summary JSON: {summary_path}") from exc
                arrays_path = summary_path.with_name("arrays.npz")
                if not arrays_path.is_file():
                    raise RuntimeError(f"group-average summary exists without arrays: {summary_path}")
                if summary.get("analysis_role") != "correspondence_group_average_sensitivity":
                    raise RuntimeError(f"unexpected group-average analysis role: {summary_path}")
                if not bool(summary.get("complete")):
                    raise RuntimeError(f"incomplete group-average summary: {summary_path}")
                if summary.get("model_key") != model or summary.get("story") != story:
                    raise RuntimeError(f"group-average path/summary identity mismatch: {summary_path}")
                if summary.get("bank_variant") != variant or int(summary.get("bank_size", -1)) != PRIMARY_SIZE or size_name != f"n{PRIMARY_SIZE}":
                    raise RuntimeError(f"unexpected group-average bank identity: {summary_path}")
                key = (model, story)
                if key not in expected:
                    raise RuntimeError(f"unexpected group-average condition: {model}/{story}")
                if key in records:
                    raise RuntimeError(f"duplicate group-average condition: {model}/{story}")
                records[key] = {
                    "model_key": model,
                    "story": story,
                    "analysis_fingerprint": summary.get("analysis_fingerprint"),
                    "complete": True,
                    "summary_path": str(summary_path),
                    "summary_sha256": sha256_file(summary_path),
                    "arrays_path": str(arrays_path),
                    "arrays_sha256": sha256_file(arrays_path),
                }
            missing = sorted(expected - set(records))
            manifest = {
                "schema_version": 2,
                "analysis_role": "correspondence_group_average_manifest",
                "conditions": [records[key] for key in sorted(records)],
                "expected_condition_count": len(expected),
                "observed_condition_count": len(records),
                "missing_conditions": [
                    {"model_key": model, "story": story} for model, story in missing
                ],
                "complete": not missing and len(records) == len(expected),
                "materializer": "group_average_manifest.py",
                "materializer_sha256": sha256_file(Path(__file__).resolve()),
            }
            atomic_write_json(manifest_path, manifest)
            return manifest
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group-root", type=Path, required=True)
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--stories", nargs="+", required=True)
    args = parser.parse_args(argv)
    manifest = materialize_group_average_manifest(
        args.group_root,
        {"models": {key: {} for key in args.models}, "stories": {key: {} for key in args.stories}},
    )
    print(json.dumps({
        "manifest": str(args.group_root.resolve() / "manifest.json"),
        "observed": manifest["observed_condition_count"],
        "expected": manifest["expected_condition_count"],
        "complete": manifest["complete"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
