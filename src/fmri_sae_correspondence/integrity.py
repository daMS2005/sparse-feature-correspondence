"""Deterministic, path-independent fingerprints for science and artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .config import load_config, scientific_projection
from .paths import RuntimePaths, resolve_below


class AnalysisError(RuntimeError):
    """Base exception for configuration, provenance, and cache failures."""


class FingerprintError(ValueError):
    """Raised when fingerprint input is ambiguous or non-deterministic."""


def _canonical_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FingerprintError("fingerprints do not accept NaN or infinite values")
        return value
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise FingerprintError("fingerprint mapping keys must be strings")
            output[key] = _canonical_value(item)
        return output
    if isinstance(value, (list, tuple)):
        return [_canonical_value(item) for item in value]
    raise FingerprintError(f"unsupported fingerprint value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible values in a stable, platform-neutral form."""

    return json.dumps(
        _canonical_value(value),
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: str | Path) -> str:
    """Hash file bytes without incorporating the file's location."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    if not path.is_dir():
        raise FileNotFoundError(f"hash source does not exist: {path}")
    yield from sorted(
        (item for item in path.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(path).as_posix(),
    )


def tree_digest(path: str | Path) -> tuple[str, int, int]:
    """Hash a file or tree without incorporating the tree's absolute root."""

    root = Path(path)
    if root.is_symlink():
        raise AnalysisError(f"symlinks are not accepted as hash roots: {root}")
    if root.is_file():
        return sha256_file(root), root.stat().st_size, 1
    if not root.is_dir():
        raise FileNotFoundError(f"hash source does not exist: {root}")

    digest = hashlib.sha256()
    total_bytes = 0
    file_count = 0
    for item in _iter_files(root):
        if item.is_symlink():
            raise AnalysisError(f"symlinks are not accepted in hashed trees: {item}")
        relative = item.relative_to(root).as_posix()
        size = item.stat().st_size
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(size).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\n")
        total_bytes += size
        file_count += 1
    return digest.hexdigest(), total_bytes, file_count


def stable_fingerprint(value: Any) -> str:
    """Hash a canonical JSON value."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def scientific_fingerprint(spec: Mapping[str, Any]) -> str:
    """Hash scientific choices while excluding runtime paths and metadata."""

    return stable_fingerprint(
        {
            "fingerprint_schema": 1,
            "scientific_projection": scientific_projection(spec),
        }
    )


def load_spec(path: str | Path | None = None) -> dict[str, Any]:
    """Load the neutral schema-2 configuration for stage modules."""

    return load_config(path)


def config_digest(config_path: str | Path) -> str:
    """Hash the config's scientific projection, excluding runtime locations."""

    return scientific_fingerprint(load_spec(config_path))


def _root(
    kind: str,
    *,
    override: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    keyword = {f"{kind}_root": override, "environ": environ}
    paths = RuntimePaths.resolve(**keyword)
    if kind == "data":
        return paths.require_data_root()
    if kind == "output":
        return paths.require_output_root()
    if kind == "cache":
        return paths.require_cache_root()
    raise ValueError(f"unknown runtime root: {kind}")


def resolve_data_root(
    config_path: str | Path | None = None,
    spec: Mapping[str, Any] | None = None,
    *,
    override: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the external input root; config arguments are intentionally ignored."""

    del config_path, spec
    return _root("data", override=override, environ=environ)


def resolve_output_root(
    config_path: str | Path | None = None,
    spec: Mapping[str, Any] | None = None,
    *,
    override: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the generated-output root independently of the source checkout."""

    del config_path, spec
    return _root("output", override=override, environ=environ)


def resolve_cache_root(
    config_path: str | Path | None = None,
    spec: Mapping[str, Any] | None = None,
    *,
    override: str | Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the reusable model/cache root independently of the source checkout."""

    del config_path, spec
    return _root("cache", override=override, environ=environ)


def resolve_path(
    config_path: str | Path,
    spec: Mapping[str, Any],
    key: str,
    *,
    root: str,
) -> Path:
    """Resolve one configured relative path below a selected runtime root."""

    del config_path
    if root == "data":
        base = resolve_data_root(spec=spec)
    elif root == "output":
        base = resolve_output_root(spec=spec)
    elif root == "cache":
        base = resolve_cache_root(spec=spec)
    else:
        raise AnalysisError(f"unknown runtime root {root!r}")
    paths = spec.get("paths")
    if not isinstance(paths, Mapping) or key not in paths:
        raise AnalysisError(f"configuration has no paths.{key}")
    value = paths[key]
    if not isinstance(value, (str, Path)):
        raise AnalysisError(f"paths.{key} is not a single relative path")
    return resolve_below(base, value)


def _content_records(paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        digest, size, count = tree_digest(path)
        records.append({"sha256": digest, "size_bytes": size, "file_count": count})
    return sorted(records, key=lambda row: (row["sha256"], row["size_bytes"], row["file_count"]))


def provenance_fingerprint(
    config_path: str | Path,
    *,
    input_paths: Iterable[str | Path] = (),
    code_paths: Iterable[str | Path] = (),
    extra: Any | None = None,
) -> str:
    """Hash science, input bytes, implementation bytes, and semantic options."""

    return stable_fingerprint(
        {
            "fingerprint_schema": 2,
            "scientific_fingerprint": config_digest(config_path),
            "inputs": _content_records(input_paths),
            "code": _content_records(code_paths),
            "extra": extra,
        }
    )


def assert_cache_fingerprint(path: str | Path, expected: str) -> dict[str, Any] | None:
    """Return a verified cache marker, reject stale or unreadable markers."""

    marker = Path(path)
    if not marker.exists():
        return None
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"cache marker is unreadable; refusing reuse: {marker}") from exc
    if not isinstance(data, dict):
        raise AnalysisError(f"cache marker must be a JSON object: {marker}")
    observed = data.get("analysis_fingerprint")
    if observed != expected:
        raise AnalysisError(
            f"stale cache rejected: {marker}\nexpected {expected}\nobserved {observed}"
        )
    return data


def atomic_write_json(path: str | Path, value: Any) -> None:
    """Atomically replace a JSON document so partial cache markers are impossible."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def artifact_manifest(artifacts: Mapping[str, str | Path]) -> dict[str, dict[str, Any]]:
    """Describe artifact bytes by logical role, never by filesystem location."""

    manifest: dict[str, dict[str, Any]] = {}
    for role, raw_path in sorted(artifacts.items()):
        if not isinstance(role, str) or not role.strip():
            raise FingerprintError("artifact roles must be non-empty strings")
        path = Path(raw_path)
        if not path.is_file():
            raise FileNotFoundError(f"artifact for role {role!r} not found: {path}")
        manifest[role] = {
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
    if not manifest:
        raise FingerprintError("at least one artifact is required")
    return manifest


def artifact_fingerprint(
    artifacts: Mapping[str, str | Path],
    *,
    namespace: str,
    scientific_config: Mapping[str, Any] | None = None,
) -> str:
    """Hash labeled artifact content and, optionally, the scientific contract."""

    if not isinstance(namespace, str) or not namespace.strip():
        raise FingerprintError("artifact fingerprint namespace is required")
    payload: dict[str, Any] = {
        "fingerprint_schema": 1,
        "namespace": namespace,
        "artifacts": artifact_manifest(artifacts),
    }
    if scientific_config is not None:
        payload["scientific_fingerprint"] = scientific_fingerprint(scientific_config)
    return stable_fingerprint(payload)


__all__ = [
    "AnalysisError",
    "FingerprintError",
    "assert_cache_fingerprint",
    "artifact_fingerprint",
    "artifact_manifest",
    "atomic_write_json",
    "canonical_json",
    "config_digest",
    "load_spec",
    "provenance_fingerprint",
    "resolve_cache_root",
    "resolve_data_root",
    "resolve_output_root",
    "resolve_path",
    "scientific_fingerprint",
    "sha256_file",
    "stable_fingerprint",
    "tree_digest",
]
