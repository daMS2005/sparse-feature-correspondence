#!/usr/bin/env python3
"""Build or verify the repository-relative scientific source catalog."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "checksums" / "SCIENTIFIC_CODE_MANIFEST.json"
CHECKSUM_PATH = ROOT / "checksums" / "SCIENTIFIC_CODE.sha256"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_paths() -> list[Path]:
    package = ROOT / "src" / "fmri_sae_correspondence"
    return [ROOT / "configs" / "analysis.yaml", *sorted(package.glob("*.py"))]


def role_for(path: Path) -> str:
    if path == ROOT / "configs" / "analysis.yaml":
        return "scientific_configuration"
    if path.name in {
        "__init__.py",
        "__main__.py",
        "alignment.py",
        "cli.py",
        "config.py",
        "integrity.py",
        "model_adapter.py",
        "paths.py",
    }:
        return "shared_interface"
    return "scientific_stage"


def build_catalog() -> tuple[dict[str, object], str]:
    records = []
    for path in source_paths():
        relative = path.relative_to(ROOT).as_posix()
        records.append(
            {
                "path": relative,
                "role": role_for(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    manifest: dict[str, object] = {
        "schema_version": 2,
        "manifest_role": "scientific_source_integrity",
        "file_count": len(records),
        "files": records,
    }
    checksums = "".join(f"{row['sha256']}  {row['path']}\n" for row in records)
    return manifest, checksums


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--write", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest, checksums = build_catalog()
    manifest_text = json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not MANIFEST_PATH.is_file() or not CHECKSUM_PATH.is_file():
            parser.error("source catalogs are missing")
        if MANIFEST_PATH.read_text(encoding="utf-8") != manifest_text:
            parser.error("scientific source manifest is stale")
        if CHECKSUM_PATH.read_text(encoding="utf-8") != checksums:
            parser.error("scientific source checksum catalog is stale")
        print(f"source catalog verified: {manifest['file_count']} files")
        return 0

    existing = [path for path in (MANIFEST_PATH, CHECKSUM_PATH) if path.exists()]
    if existing and not args.force:
        parser.error("refusing to overwrite source catalogs without --force")
    MANIFEST_PATH.write_text(manifest_text, encoding="utf-8")
    CHECKSUM_PATH.write_text(checksums, encoding="utf-8")
    print(f"source catalog written: {manifest['file_count']} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
