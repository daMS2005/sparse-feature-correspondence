"""Sparse-feature correspondence analysis for language models and fMRI."""

import tomllib
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

try:
    __version__ = version("fmri-sae-correspondence")
except PackageNotFoundError:  # pragma: no cover - source checkout without installation
    metadata = tomllib.loads(
        (Path(__file__).resolve().parents[2] / "pyproject.toml").read_text(encoding="utf-8")
    )
    __version__ = str(metadata["project"]["version"])

__all__ = ["__version__"]
