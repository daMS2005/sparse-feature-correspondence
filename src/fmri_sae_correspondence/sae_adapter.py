"""Small selected-coordinate adapter for official JumpReLU SAE weights.

This module is kept separate from the historical extraction pipeline so the
pooling stage cannot silently fall back to the old top-k token records.  It
implements the same encoder equation as ``extract_v2.LocalNPZSAEAdapter`` but
returns only requested dictionary coordinates.

The canonical analysis extraction encoded the dictionary in contiguous
1,024-coordinate chunks.  That chunk shape is part of the numerical execution
contract: on reduced-precision CUDA kernels, multiplying by a thin matrix of
only selected columns need not round identically to multiplying by the
canonical 1,024-column chunk.  Near a JumpReLU threshold, the small rounding
difference can change whether an activation is present.  This adapter therefore
computes every canonical chunk touched by the requested coordinates and slices
the requested values only after the JumpReLU operation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

CANONICAL_ENCODER_CHUNK_SIZE = 1024


def canonical_chunk_bounds(
    width: int,
    feature_ids: Sequence[int],
    *,
    chunk_size: int = CANONICAL_ENCODER_CHUNK_SIZE,
) -> tuple[tuple[int, int], ...]:
    """Return canonical encoder chunks touched by ``feature_ids``."""

    width = int(width)
    chunk_size = int(chunk_size)
    if width < 1 or chunk_size < 1:
        raise ValueError("width and chunk_size must be positive")
    ids = [int(value) for value in feature_ids]
    if not ids or min(ids) < 0 or max(ids) >= width:
        raise IndexError(f"feature id outside SAE width {width}: {ids}")
    starts = sorted({(feature_id // chunk_size) * chunk_size for feature_id in ids})
    return tuple((start, min(width, start + chunk_size)) for start in starts)


class SelectedSAEAdapter:
    """Inference-compatible selected-coordinate official SAE adapter."""

    def __init__(
        self,
        root: Path,
        *,
        device: str | torch.device,
        dtype: torch.dtype,
        chunk_size: int = CANONICAL_ENCODER_CHUNK_SIZE,
    ) -> None:
        self.root = Path(root).resolve()
        self.device = device
        self.dtype = dtype
        self.chunk_size = int(chunk_size)
        if self.chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self._cache: dict[int, dict[str, torch.Tensor]] = {}

    def _load(self, layer: int) -> dict[str, torch.Tensor]:
        layer = int(layer)
        if layer not in self._cache:
            path = self.root / f"layer_{layer}" / "params.npz"
            if not path.is_file():
                raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as data:
                required = {"W_enc", "b_enc", "threshold", "apply_b_dec_to_input"}
                missing = sorted(required - set(data.files))
                if missing:
                    raise ValueError(f"{path}: missing explicit SAE fields {missing}")
                apply_b_dec = bool(np.asarray(data["apply_b_dec_to_input"]).reshape(-1)[0])
                values: dict[str, torch.Tensor] = {
                    key: torch.as_tensor(data[key], device=self.device, dtype=self.dtype)
                    for key in ("W_enc", "b_enc", "threshold")
                }
                values["apply_b_dec_to_input"] = torch.as_tensor(apply_b_dec, device=self.device)
                if apply_b_dec:
                    if "b_dec" not in data.files:
                        raise ValueError(f"{path}: apply_b_dec_to_input=true but b_dec is missing")
                    values["b_dec"] = torch.as_tensor(data["b_dec"], device=self.device, dtype=self.dtype)
                self._cache[layer] = values
        return self._cache[layer]

    def encode_selected(
        self,
        layer: int,
        residual: torch.Tensor,
        feature_ids: Sequence[int],
    ) -> np.ndarray:
        """Return ``[sequence, selected_feature]`` positive JumpReLU values.

        The method evaluates the same fixed-width encoder chunks as the
        canonical full-dictionary extractor, then returns the requested
        columns in their original order.  A NumPy result keeps the extractor
        output independent of autograd and accelerator memory.
        """

        ids = np.asarray([int(value) for value in feature_ids], dtype=np.int64)
        if ids.ndim != 1 or len(ids) == 0:
            raise ValueError("feature_ids must be a non-empty one-dimensional sequence")
        if len(np.unique(ids)) != len(ids) or np.any(ids < 0):
            raise ValueError("feature_ids must be unique and non-negative")
        weights = self._load(int(layer))
        width = int(weights["W_enc"].shape[1])
        if np.any(ids >= width):
            raise IndexError(f"feature id outside SAE width {width}: {ids.tolist()}")
        value = residual.to(self.device, dtype=self.dtype)
        if bool(weights["apply_b_dec_to_input"].item()):
            value = value - weights["b_dec"]

        # Preserve the canonical 1,024-output GEMM shape.  Selecting columns
        # before the matrix multiplication is algebraically equivalent in real
        # arithmetic, but not numerically equivalent in bfloat16/float16.
        selected_columns: dict[int, torch.Tensor] = {}
        for start, end in canonical_chunk_bounds(
            width,
            ids.tolist(),
            chunk_size=self.chunk_size,
        ):
            pre = (
                value @ weights["W_enc"][:, start:end]
                + weights["b_enc"][start:end]
            )
            latent = torch.relu(pre) * (
                pre > weights["threshold"][start:end]
            ).to(pre.dtype)
            for feature_id in ids.tolist():
                if start <= feature_id < end:
                    selected_columns[feature_id] = latent[..., feature_id - start]

        ordered = torch.stack(
            [selected_columns[int(feature_id)] for feature_id in ids.tolist()],
            dim=-1,
        )
        return ordered.squeeze(0).detach().to("cpu", dtype=torch.float32).numpy()


__all__ = [
    "CANONICAL_ENCODER_CHUNK_SIZE",
    "SelectedSAEAdapter",
    "canonical_chunk_bounds",
]
