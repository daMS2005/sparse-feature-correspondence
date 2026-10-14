"""Portable resolution of external input, output, and cache roots."""

from __future__ import annotations

import os
import sysconfig
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

DATA_ROOT_ENV = "FMRI_SAE_DATA_ROOT"
OUTPUT_ROOT_ENV = "FMRI_SAE_OUTPUT_ROOT"
CACHE_ROOT_ENV = "FMRI_SAE_CACHE_ROOT"
PACKAGE_NAME = "fmri-sae-correspondence"


class PathConfigurationError(ValueError):
    """Raised when a required runtime root or relative path is invalid."""


def _resolve_root(
    explicit: str | Path | None,
    env_name: str,
    environ: Mapping[str, str],
) -> Path | None:
    if explicit is not None:
        raw = str(explicit).strip()
        if not raw:
            raise PathConfigurationError(f"explicit {env_name} override is empty")
        return Path(raw).expanduser().resolve()
    environment_value = environ.get(env_name)
    if environment_value is None or not environment_value.strip():
        return None
    return Path(environment_value).expanduser().resolve()


@dataclass(frozen=True, slots=True)
class RuntimePaths:
    """Runtime roots resolved with explicit CLI values taking precedence."""

    data_root: Path | None
    output_root: Path | None
    cache_root: Path | None

    @classmethod
    def resolve(
        cls,
        *,
        data_root: str | Path | None = None,
        output_root: str | Path | None = None,
        cache_root: str | Path | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> RuntimePaths:
        environment = os.environ if environ is None else environ
        return cls(
            data_root=_resolve_root(data_root, DATA_ROOT_ENV, environment),
            output_root=_resolve_root(output_root, OUTPUT_ROOT_ENV, environment),
            cache_root=_resolve_root(cache_root, CACHE_ROOT_ENV, environment),
        )

    def require_data_root(self) -> Path:
        root = self._require(self.data_root, "data", DATA_ROOT_ENV)
        if not root.is_dir():
            raise PathConfigurationError(f"configured data root does not exist: {root}")
        return root

    def require_output_root(self) -> Path:
        return self._require(self.output_root, "output", OUTPUT_ROOT_ENV)

    def require_cache_root(self) -> Path:
        return self._require(self.cache_root, "cache", CACHE_ROOT_ENV)

    @staticmethod
    def _require(value: Path | None, label: str, env_name: str) -> Path:
        if value is None:
            raise PathConfigurationError(
                f"{label} root is required; provide --{label}-root or {env_name}"
            )
        return value

    def data_path(self, value: str | Path) -> Path:
        return resolve_below(self.require_data_root(), value)

    def output_path(self, value: str | Path) -> Path:
        return resolve_below(self.require_output_root(), value)

    def cache_path(self, value: str | Path) -> Path:
        return resolve_below(self.require_cache_root(), value)


def resolve_below(root: Path, value: str | Path) -> Path:
    """Resolve a relative value below ``root`` and reject path traversal."""

    base = Path(root).expanduser().resolve()
    raw_text = str(value)
    raw = Path(raw_text)
    if raw.is_absolute() or PureWindowsPath(raw_text).is_absolute():
        raise PathConfigurationError(f"expected a path relative to its runtime root: {value}")
    resolved = (base / raw).resolve()
    try:
        resolved.relative_to(base)
    except ValueError as exc:
        raise PathConfigurationError(f"relative path escapes its runtime root: {value}") from exc
    return resolved


def release_path(relative: str | Path) -> Path:
    """Locate a small file shipped in either a source checkout or a wheel."""

    source_path = Path(__file__).resolve().parents[2] / relative
    if source_path.exists():
        return source_path
    return (
        Path(sysconfig.get_path("data"))
        / "share"
        / PACKAGE_NAME
        / relative
    ).resolve()


__all__ = [
    "CACHE_ROOT_ENV",
    "DATA_ROOT_ENV",
    "OUTPUT_ROOT_ENV",
    "PathConfigurationError",
    "RuntimePaths",
    "release_path",
    "resolve_below",
]
