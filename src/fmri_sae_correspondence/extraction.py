#!/usr/bin/env python3
"""Extract token-level SAE values needed by the pooling sensitivities.

The canonical analysis bundles contain the mean-positive SAE value per
TR for every dictionary coordinate, but they do not contain token-level
values. Consequently, a TR maximum and an event count cannot be reconstructed
from those bundles.  This extractor performs a fresh, outcome-blind forward
pass and writes only the union of coordinates used by the source-story
variance and correlation-pruned K=256 banks.

The output is deliberately a token record rather than a fitted result.  It
can therefore be pooled repeatedly without another model forward pass.  The
pooling definitions are:

* ``mean_positive_activation``: sum of positive token activations divided by
  the number of all transcript tokens assigned to the TR (including tokens
  with zero activation); this is the canonical analysis mean.
* ``sum_positive_activation``: sum of positive token activations.
* ``max_positive_activation``: maximum positive activation, with zero for an
  empty TR.
* ``event_count``: number of assigned transcript tokens with strictly
  positive activation.  It is a positive-token count, not a count of all
  tokens and not an activation-weighted count.

The script never reads fMRI targets or fitted coefficients.  Existing model
and SAE caches are reused when explicitly supplied; missing caches fail
closed.  It is safe to run one model/story at a time on a GPU node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from .alignment import (
    assign_words_to_trs,
    build_token_to_word_index,
    group_tokens_with_model_tokenizer,
    load_tr_bins,
    load_word_rows,
    resolve_transcript_paths,
    validate_tr_alignment,
    validate_word_alignment,
)
from .config import default_config_path, load_config
from .paths import RuntimePaths, resolve_below

HERE = Path(__file__).resolve().parent
POOLING_NAMES = (
    "mean_positive_activation",
    "max_positive_activation",
    "sum_positive_activation",
    "event_count",
)
REQUIRED_BANK_KINDS = ("source_story_variance", "source_story_correlation_pruned")
PRUNED_THRESHOLDS = (0.70, 0.80, 0.90)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, default=str)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _feature_bank_root(
    config_path: Path,
    spec: Mapping[str, Any],
    output_root: Path | None = None,
) -> Path:
    del config_path
    runtime_root = output_root or RuntimePaths.resolve().require_output_root()
    root = resolve_below(
        resolve_below(runtime_root, str(spec["paths"]["artifact_root"])),
        str(spec["paths"]["feature_bank_root"]),
    )
    if not root.is_dir():
        raise FileNotFoundError(
            "source feature-bank root is missing; run the feature-banks stage first"
        )
    return root


def _predictor_path(
    config_path: Path,
    spec: Mapping[str, Any],
    model_key: str,
    story: str,
    data_root: Path | None = None,
) -> Path:
    del config_path
    logical_path = str(spec["paths"]["predictor_pattern"]).format(
        model=model_key,
        story=story,
    )
    return resolve_below(
        data_root or RuntimePaths.resolve().require_data_root(),
        logical_path,
    )


def _bank_file(
    config_path: Path,
    spec: Mapping[str, Any],
    model_key: str,
    *,
    source_story: str,
    target_story: str | None = None,
    fold: int | None = None,
    output_root: Path | None = None,
) -> Path:
    root = _feature_bank_root(config_path, spec, output_root)
    if fold is None:
        if target_story is None:
            raise ValueError("target_story is required for a cross-story bank")
        return root / "cross_story" / model_key / f"{source_story}_to_{target_story}" / "banks.json"
    return root / "within_story" / model_key / source_story / f"fold_{int(fold):02d}" / "banks.json"


def _bank_entry(record: Mapping[str, Any], variant: str, size: int) -> Mapping[str, Any]:
    if variant == "source_story_variance":
        bank = record.get("source_variance")
    elif variant.startswith("source_story_correlation_pruned_"):
        threshold = float(variant.rsplit("_", 1)[-1].replace("p", "."))
        pruned = record.get("correlation_pruned", {})
        bank = pruned.get(str(threshold))
        if bank is None:
            bank = pruned.get(f"{threshold:.1f}")
        if bank is None:
            bank = pruned.get(f"{threshold:.2f}")
    else:
        raise KeyError(f"unsupported pooling bank variant: {variant}")
    if not isinstance(bank, Mapping):
        raise KeyError(f"{variant}: missing bank in record")
    entry = bank.get("banks", {}).get(str(int(size)))
    if not isinstance(entry, Mapping):
        raise KeyError(f"{variant}/{size}: missing bank entry")
    if not bool(entry.get("complete")) or int(entry.get("actual_size", -1)) != int(size):
        raise ValueError(f"{variant}/{size}: incomplete bank")
    layers = [int(value) for value in entry.get("layers", [])]
    feature_ids = [int(value) for value in entry.get("feature_ids", [])]
    if len(layers) != int(size) or len(feature_ids) != int(size):
        raise ValueError(f"{variant}/{size}: malformed bank identity")
    if len(set(zip(layers, feature_ids, strict=False))) != int(size):
        raise ValueError(f"{variant}/{size}: duplicate bank identity")
    return entry


def _variant_labels() -> tuple[str, ...]:
    return (
        "source_story_variance",
        *(f"source_story_correlation_pruned_{threshold:.2f}".replace(".", "p") for threshold in PRUNED_THRESHOLDS),
    )


def collect_required_coordinates(
    config_path: Path,
    spec: Mapping[str, Any],
    model_key: str,
    *,
    size: int = 256,
    output_root: Path | None = None,
) -> dict[str, Any]:
    """Collect the coordinate union needed by all relevant K=256 banks.

    The returned ``bank_memberships`` maps each exact bank to indices in the
    sorted coordinate array.  This permits downstream pooled matrices to be
    sliced without guessing whether two bank JSON files use the same order.
    """

    if int(size) != 256:
        raise ValueError("pooling extraction is fixed to K=256 bank identities")
    if model_key not in spec["models"]:
        raise KeyError(model_key)
    model_layers = [int(value) for value in spec["models"][model_key]["layers"]]
    coordinate_set: set[tuple[int, int]] = set()
    memberships: dict[str, list[tuple[int, int]]] = {}
    bank_sources: dict[str, str] = {}
    labels = _variant_labels()
    for source_story, target_story in (
        ("shapesphysical", "shapessocial"),
        ("shapessocial", "shapesphysical"),
    ):
        path = _bank_file(
            config_path,
            spec,
            model_key,
            source_story=source_story,
            target_story=target_story,
            output_root=output_root,
        )
        if not path.is_file():
            raise FileNotFoundError(path)
        record = json.loads(path.read_text(encoding="utf-8"))
        for variant in labels:
            entry = _bank_entry(record, variant, size)
            coords = tuple(zip((int(v) for v in entry["layers"]), (int(v) for v in entry["feature_ids"]), strict=False))
            key = f"cross_story:{source_story}_to_{target_story}:{variant}:n{int(size)}"
            memberships[key] = list(coords)
            bank_sources[key] = (
                f"feature_banks/cross_story/{model_key}/"
                f"{source_story}_to_{target_story}/banks.json"
            )
            coordinate_set.update(coords)
    for story in ("shapesphysical", "shapessocial"):
        for fold in range(int(spec["cross_validation"]["outer_folds"])):
            path = _bank_file(
                config_path,
                spec,
                model_key,
                source_story=story,
                fold=fold,
                output_root=output_root,
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            record = json.loads(path.read_text(encoding="utf-8"))
            for variant in labels:
                entry = _bank_entry(record, variant, size)
                coords = tuple(zip((int(v) for v in entry["layers"]), (int(v) for v in entry["feature_ids"]), strict=False))
                key = f"within_story:{story}:fold_{fold:02d}:{variant}:n{int(size)}"
                memberships[key] = list(coords)
                bank_sources[key] = (
                    f"feature_banks/within_story/{model_key}/{story}/"
                    f"fold_{fold:02d}/banks.json"
                )
                coordinate_set.update(coords)
    layer_order = {layer: index for index, layer in enumerate(model_layers)}
    unknown = sorted({layer for layer, _ in coordinate_set if layer not in layer_order})
    if unknown:
        raise ValueError(f"bank coordinates contain layers not listed in spec: {unknown}")
    coordinates = sorted(coordinate_set, key=lambda pair: (layer_order[pair[0]], pair[1]))
    index = {pair: idx for idx, pair in enumerate(coordinates)}
    coordinate_records = [
        {
            "index": int(idx),
            "layer": int(layer),
            "feature_id": int(feature_id),
            "feature_name": f"layer_{int(layer)}_feature_{int(feature_id)}",
        }
        for idx, (layer, feature_id) in enumerate(coordinates)
    ]
    membership_indices = {
        key: [int(index[coord]) for coord in coords]
        for key, coords in sorted(memberships.items())
    }
    return {
        "schema_version": 1,
        "model_key": model_key,
        "model_id": str(spec["models"][model_key]["model_id"]),
        "selected_layers": model_layers,
        "size": int(size),
        "variant_labels": list(labels),
        "coordinate_records": coordinate_records,
        "bank_memberships": membership_indices,
        "bank_coordinate_memberships": {
            key: [[int(layer), int(feature_id)] for layer, feature_id in memberships[key]]
            for key in sorted(memberships)
        },
        "bank_sources": bank_sources,
        "bank_file_sha256": {
            logical_id: sha256_file(
                _bank_file(
                    config_path,
                    spec,
                    model_key,
                    source_story=key.split(":")[1].split("_to_")[0]
                    if key.startswith("cross_story:")
                    else key.split(":")[1],
                    target_story=key.split(":")[1].split("_to_")[1]
                    if key.startswith("cross_story:")
                    else None,
                    fold=int(key.split(":")[2].split("_")[1])
                    if key.startswith("within_story:")
                    else None,
                    output_root=output_root,
                )
            )
            for key, logical_id in sorted(bank_sources.items())
        },
    }


def _resolve_sae_root(
    cache_root: Path,
    spec: Mapping[str, Any],
    model_key: str,
    override: str | Path | None,
    required_layers: Sequence[int] | None = None,
) -> Path:
    if override:
        raw = str(override).replace("{model_key}", model_key)
        candidate = Path(raw).expanduser().resolve()
    else:
        candidate = resolve_below(cache_root, str(spec["paths"]["sae_root"])) / model_key
    layers = tuple(int(value) for value in (required_layers or ()))
    if not candidate.is_dir():
        raise FileNotFoundError(f"SAE weights are missing for {model_key}: {candidate}")
    missing = [
        candidate / f"layer_{layer}" / "params.npz"
        for layer in layers
        if not (candidate / f"layer_{layer}" / "params.npz").is_file()
    ]
    if missing:
        raise FileNotFoundError(f"SAE weights are incomplete; missing {missing[0]}")
    return candidate


def _sae_parameter_hashes(sae_root: Path, layers: Sequence[int]) -> dict[str, str]:
    """Hash the exact official SAE parameter files used by an extraction.

    The weights are often large, but hashing them once is important here: a
    path-only provenance record cannot distinguish two releases mounted at
    the same directory.  This helper is called before the cache check so a
    changed weight file is rejected rather than silently reused.
    """

    paths = {
        f"layer_{int(layer)}/params.npz": sha256_file(
            Path(sae_root) / f"layer_{int(layer)}" / "params.npz"
        )
        for layer in layers
    }
    return dict(sorted(paths.items()))


def _render_runtime_config(
    output_dir: Path,
    model_info: Mapping[str, Any],
    *,
    seq_len: int,
) -> Path:
    raw = {
        "model": {
            "base_model_id": model_info["model_id"],
            "scope_release": model_info["sae_release"],
            "scope_width": model_info["sae_width"],
            "layer_selection": [int(value) for value in model_info["layers"]],
        },
        "tokenization": {"seq_len": int(seq_len), "add_special_tokens": False},
    }
    path = output_dir / "runtime_config.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def _load_model_and_sae(
    *,
    model_info: Mapping[str, Any],
    sae_root: Path,
    output_dir: Path,
    seq_len: int,
) -> tuple[Any, Any, Path]:
    if not sae_root.is_dir():
        raise FileNotFoundError(
            f"SAE cache is missing: {sae_root}; pass --sae-root or prepare the official release first"
        )
    for layer in model_info["layers"]:
        params = sae_root / f"layer_{int(layer)}" / "params.npz"
        if not params.is_file():
            raise FileNotFoundError(f"SAE cache is incomplete for layer {layer}: {params}")
    from .model_adapter import TextModelAdapter, load_model_config
    from .sae_adapter import SelectedSAEAdapter

    runtime_config = _render_runtime_config(output_dir, model_info, seq_len=seq_len)
    model = TextModelAdapter(load_model_config(runtime_config))
    sae = SelectedSAEAdapter(sae_root, device=model.device, dtype=model.dtype)
    return model, sae, runtime_config


def build_token_to_tr(
    *,
    model: Any,
    story: str,
    transcript_root: Path,
) -> tuple[list[int], list[str], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Reconstruct the canonical transcript-token/TR mapping.

    The timing functions are the same functions used by ``extract_v2.py``.
    Returned token rows remain in transcript order, while ``token_to_tr`` is
    ``-1`` for tokenizer pieces that cannot be assigned to a timed word.
    """

    paths = resolve_transcript_paths(transcript_root, story)
    transcript_text = paths.transcript_txt.read_text(encoding="utf-8").strip()
    word_rows = load_word_rows(paths.words_tsv)
    tr_bins = load_tr_bins(paths.tr_aligned_tsv)
    metadata = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    token_ids = model.tokenize_document(transcript_text)
    tokens = model.tokenizer.convert_ids_to_tokens(token_ids)
    groups = group_tokens_with_model_tokenizer(
        model_id=model.config.model.base_model_id,
        tokens=tokens,
        transcript_text=transcript_text,
        tokenizer=model.tokenizer,
    )
    validate_word_alignment(groups, word_rows, transcript_text)
    token_to_word = build_token_to_word_index(groups)
    word_to_tr = assign_words_to_trs(
        word_rows=word_rows,
        tr_bins=tr_bins,
        stimulus_onset_s=float(metadata.get("stimulus_onset_s", 0.0)),
    )
    validate_tr_alignment(
        word_rows=word_rows,
        word_to_tr_index=word_to_tr,
        tr_bins=tr_bins,
        stimulus_onset_s=float(metadata.get("stimulus_onset_s", 0.0)),
    )
    token_to_tr = map_token_words_to_trs(
        len(token_ids),
        token_to_word,
        word_to_tr,
    )
    tr_indices = np.asarray([int(item.tr_index) for item in tr_bins], dtype=np.int64)
    start_s = np.asarray([float(item.start_s) for item in tr_bins], dtype=np.float64)
    end_s = np.asarray([float(item.end_s) for item in tr_bins], dtype=np.float64)
    return token_ids, tokens, token_to_tr, tr_indices, np.column_stack([start_s, end_s]), metadata


def map_token_words_to_trs(
    n_tokens: int,
    token_to_word: Mapping[int, int],
    word_to_tr: Sequence[int],
) -> np.ndarray:
    """Map tokenizer pieces to TRs using canonical word-index semantics.

    ``assign_words_to_trs`` returns a list whose *position* is the word index
    and whose value is the assigned TR index.  It is not a mapping keyed by TR
    or word identifiers.  Tokenizer pieces absent from ``token_to_word`` remain
    unassigned (``-1``), matching the canonical extraction.
    """

    n_tokens = int(n_tokens)
    if n_tokens < 0:
        raise ValueError("n_tokens must be non-negative")
    assignments = np.full(n_tokens, -1, dtype=np.int32)
    n_words = len(word_to_tr)
    for raw_token_index, raw_word_index in token_to_word.items():
        token_index = int(raw_token_index)
        word_index = int(raw_word_index)
        if token_index < 0 or token_index >= n_tokens:
            raise IndexError(f"token index outside transcript: {token_index}")
        if word_index < 0 or word_index >= n_words:
            raise IndexError(f"word index outside timed transcript: {word_index}")
        tr_index = int(word_to_tr[word_index])
        if tr_index < 0:
            raise ValueError(f"word {word_index} has a negative TR assignment: {tr_index}")
        assignments[token_index] = tr_index
    return assignments


def _encode_hidden_selected(sae: Any, layer: int, residual: Any, feature_ids: Sequence[int]) -> np.ndarray:
    """Encode selected JumpReLU coordinates without materializing width 16k/32k."""

    return sae.encode_selected(int(layer), residual, feature_ids)


def pool_token_values(
    token_values: np.ndarray,
    token_to_tr: Sequence[int],
    n_tr: int,
    pooling: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Pool token records with explicit empty-bin and denominator semantics.

    Returns ``(values, token_counts)``.  ``token_counts`` counts all tokens
    assigned to each TR, not only active tokens; it is retained so the
    canonical mean can be independently audited.  For ``event_count`` the
    values are counts of tokens with a strictly positive activation.
    """

    values = np.asarray(token_values, dtype=np.float32)
    assignment = np.asarray(token_to_tr, dtype=np.int64)
    if values.ndim != 2 or len(values) != len(assignment):
        raise ValueError("token_values must be [token,feature] and match token_to_tr")
    if int(n_tr) < 1:
        raise ValueError("n_tr must be positive")
    if pooling not in POOLING_NAMES:
        raise ValueError(f"unknown pooling: {pooling}")
    if np.any(values < -1e-7):
        raise ValueError("SAE values must be non-negative after JumpReLU")
    output = np.zeros((int(n_tr), values.shape[1]), dtype=np.float32)
    token_counts = np.zeros(int(n_tr), dtype=np.int32)
    for tr in range(int(n_tr)):
        rows = np.flatnonzero(assignment == tr)
        if len(rows) == 0:
            continue
        selected = values[rows]
        token_counts[tr] = int(len(rows))
        if pooling == "mean_positive_activation":
            output[tr] = np.mean(selected, axis=0, dtype=np.float64).astype(np.float32)
        elif pooling == "max_positive_activation":
            output[tr] = np.max(selected, axis=0).astype(np.float32)
        elif pooling == "sum_positive_activation":
            output[tr] = np.sum(selected, axis=0, dtype=np.float64).astype(np.float32)
        else:  # event_count
            output[tr] = np.sum(selected > 0.0, axis=0, dtype=np.float64).astype(np.float32)
    return output, token_counts


def _write_npz_atomic(path: Path, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".npz", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _hidden_states_for_window(model: Any, input_ids: Sequence[int]) -> Any:
    # Use the same adapter entry point as the canonical extractor.  Calling
    # ``model.model.model`` directly avoids the LM head, but it also bypasses
    # any wrapper-level behavior that is part of the configured hidden-state
    # execution path.  The canonical-mean gate is intentionally strict, so a
    # speed shortcut is not safe unless it is proven equivalent for the exact
    # runtime and checkpoint.
    outputs, _ = model.forward_outputs(list(input_ids), require_grad=False)
    return outputs


def extract_story(
    *,
    config_path: Path,
    spec: Mapping[str, Any],
    model_key: str,
    story: str,
    data_root: Path,
    analysis_output_root: Path,
    output_root: Path,
    transcript_root: Path,
    sae_root: Path,
    force: bool = False,
    size: int = 256,
) -> dict[str, Any]:
    if story not in spec["stories"]:
        raise KeyError(story)
    coordinate_manifest = collect_required_coordinates(
        config_path,
        spec,
        model_key,
        size=size,
        output_root=analysis_output_root,
    )
    model_info = spec["models"][model_key]
    predictor_path = _predictor_path(
        config_path,
        spec,
        model_key,
        story,
        data_root=data_root,
    )
    if not predictor_path.is_file():
        raise FileNotFoundError(predictor_path)
    with np.load(predictor_path, allow_pickle=True) as bundle:
        required = {
            "all_sae_tr_means",
            "tr_indices",
            "selected_layers",
            "token_count",
            "context_tokens",
            "causal",
            "add_special_tokens",
        }
        missing = sorted(required - set(bundle.files))
        if missing:
            raise ValueError(f"{predictor_path}: missing canonical audit fields {missing}")
        canonical_means = np.asarray(bundle["all_sae_tr_means"], dtype=np.float32)
        canonical_tr_indices = np.asarray(bundle["tr_indices"], dtype=np.int64)
        canonical_layers = np.asarray(bundle["selected_layers"], dtype=np.int64)
        canonical_token_count = np.asarray(bundle["token_count"], dtype=np.int32)
        canonical_context_tokens = int(np.asarray(bundle["context_tokens"]).reshape(-1)[0])
        canonical_causal = bool(np.asarray(bundle["causal"]).reshape(-1)[0])
        canonical_add_special_tokens = bool(
            np.asarray(bundle["add_special_tokens"]).reshape(-1)[0]
        )
        bundle_model = str(np.asarray(bundle["model_id"]).reshape(-1)[0]) if "model_id" in bundle else ""
        bundle_story = str(np.asarray(bundle["story"]).reshape(-1)[0]) if "story" in bundle else ""
    if bundle_model and bundle_model != str(model_info["model_id"]):
        raise ValueError(f"canonical predictor model mismatch: {bundle_model} != {model_info['model_id']}")
    if bundle_story and bundle_story != story:
        raise ValueError(f"canonical predictor story mismatch: {bundle_story} != {story}")
    if not np.array_equal(canonical_layers, np.asarray(model_info["layers"], dtype=np.int64)):
        raise ValueError("canonical predictor layers differ from analysis configuration")
    if len(canonical_tr_indices) != int(spec["stories"][story]["expected_trs"]):
        raise ValueError("canonical predictor TR count differs from analysis configuration")
    if canonical_context_tokens != 1024 or not canonical_causal or canonical_add_special_tokens:
        raise ValueError(
            "canonical predictor does not use the configured causal 1,024-token, "
            "no-special-token extraction contract"
        )
    # Validate and fingerprint all required parameter files before looking at
    # the cache.  A complete output produced with different weights is not a
    # valid pooling artifact, even when its directory and filenames match.
    sae_parameter_sha256 = _sae_parameter_hashes(sae_root, model_info["layers"])
    coords = coordinate_manifest["coordinate_records"]
    layer_offsets = {int(layer): offset for offset, layer in enumerate(canonical_layers.tolist())}
    for row in coords:
        if int(row["layer"]) not in layer_offsets:
            raise ValueError(f"coordinate layer is absent from canonical predictor: {row}")
        if int(row["feature_id"]) >= canonical_means.shape[2]:
            raise ValueError(f"coordinate feature is outside dictionary width: {row}")
    output_dir = Path(output_root).resolve() / model_key / story
    token_path = output_dir / "token_records.npz"
    manifest_path = output_dir / "coordinate_manifest.json"
    summary_path = output_dir / "summary.json"
    fingerprint = stable_hash(
        {
            "analysis": "correspondence_token_pooling_extraction",
            "config_sha256": sha256_file(config_path),
            "model_key": model_key,
            "story": story,
            "bank_size": int(size),
            "predictor_sha256": sha256_file(predictor_path),
            "coordinate_manifest": coordinate_manifest,
            "model_id": model_info["model_id"],
            "sae_release": model_info["sae_release"],
            "sae_parameter_sha256": sae_parameter_sha256,
            "code_sha256": {
                "extraction.py": sha256_file(Path(__file__).resolve()),
                "sae_adapter.py": sha256_file(
                    HERE / "sae_adapter.py"
                ),
                "model_adapter.py": sha256_file(HERE / "model_adapter.py"),
                "alignment.py": sha256_file(HERE / "alignment.py"),
            },
        }
    )
    if not force and summary_path.exists():
        observed = json.loads(summary_path.read_text(encoding="utf-8"))
        if observed.get("analysis_fingerprint") != fingerprint:
            raise RuntimeError(f"stale pooling extraction cache rejected: {summary_path}")
        if not token_path.is_file() or not manifest_path.is_file():
            raise RuntimeError(f"pooling extraction cache is incomplete: {output_dir}")
        return observed

    output_dir.mkdir(parents=True, exist_ok=True)
    runtime_config = _render_runtime_config(output_dir, model_info, seq_len=1024)
    # We construct the adapter after the cache check so an unchanged complete
    # extraction never loads a multi-billion-parameter model.
    model, sae, _ = _load_model_and_sae(
        model_info=model_info,
        sae_root=sae_root,
        output_dir=output_dir,
        seq_len=1024,
    )
    token_ids, tokens, token_to_tr, tr_indices, tr_timing, metadata = build_token_to_tr(
        model=model,
        story=story,
        transcript_root=transcript_root,
    )
    if not np.array_equal(canonical_tr_indices, tr_indices):
        raise ValueError("canonical predictor and transcript timing TR indices differ")
    reconstructed_token_count = np.bincount(
        token_to_tr[token_to_tr >= 0],
        minlength=len(tr_indices),
    ).astype(np.int32)
    if not np.array_equal(canonical_token_count, reconstructed_token_count):
        mismatch = np.flatnonzero(canonical_token_count != reconstructed_token_count)
        first = int(mismatch[0])
        raise ValueError(
            "canonical predictor and reconstructed transcript assignment use different "
            f"token counts; first mismatch at TR {first}: "
            f"canonical={int(canonical_token_count[first])}, "
            f"reconstructed={int(reconstructed_token_count[first])}"
        )
    n_tokens = len(token_ids)
    token_values = np.zeros((n_tokens, len(coords)), dtype=np.float32)
    coord_by_layer: dict[int, list[tuple[int, int]]] = {}
    for output_index, row in enumerate(coords):
        coord_by_layer.setdefault(int(row["layer"]), []).append((output_index, int(row["feature_id"])))
    windows_seen: list[tuple[int, int]] = []
    started = time.time()
    windows = model.make_windows(
        token_ids,
        metadata_mode="heuristic",
        window_len=min(model.max_context_window_tokens(), 1024),
    )
    for window_index, window in enumerate(windows, start=1):
        start = int(window.window_start)
        end = start + len(window.input_ids)
        if start < 0 or end > n_tokens or end <= start:
            raise RuntimeError(f"invalid model window {start}:{end}")
        # The canonical adapter currently emits adjacent, non-overlapping
        # windows.  Require that contract explicitly: allowing a gap would
        # silently drop transcript tokens, while allowing an overlap would
        # duplicate their contribution to every pooling statistic.
        if windows_seen:
            previous_end = windows_seen[-1][1]
            if start != previous_end:
                relation = "overlap" if start < previous_end else "gap"
                raise RuntimeError(
                    f"model windows contain a {relation} at {previous_end}:{start}; "
                    "refusing incomplete/duplicated token records"
                )
        outputs = _hidden_states_for_window(model, window.input_ids)
        hidden_states = outputs.hidden_states
        for layer, output_pairs in sorted(coord_by_layer.items()):
            feature_ids = [feature_id for _, feature_id in output_pairs]
            residual = hidden_states[int(layer) + 1]
            encoded = _encode_hidden_selected(sae, int(layer), residual, feature_ids)
            if encoded.shape != (len(window.input_ids), len(feature_ids)):
                raise RuntimeError(f"selected SAE shape mismatch at layer {layer}: {encoded.shape}")
            for local_column, (output_index, _) in enumerate(output_pairs):
                token_values[start:end, output_index] = encoded[:, local_column]
        windows_seen.append((start, end))
        del outputs, hidden_states
        if getattr(model, "device", None) == "mps":
            __import__("torch").mps.empty_cache()
        print(
            f"pooling extraction {model_key}/{story}: window {window_index}/{len(windows)}",
            flush=True,
        )
    if not windows_seen or windows_seen[0][0] != 0:
        raise RuntimeError("model windows do not start at transcript token zero")
    if windows_seen[-1][1] != n_tokens or sum(end - start for start, end in windows_seen) != n_tokens:
        raise RuntimeError("model windows do not cover every transcript token exactly once")
    for row in coords:
        row_index = int(row["index"])
        layer_offset = layer_offsets[int(row["layer"])]
        feature_id = int(row["feature_id"])
        pooled_mean, counts = pool_token_values(
            token_values[:, row_index : row_index + 1],
            token_to_tr,
            len(tr_indices),
            "mean_positive_activation",
        )
        expected = canonical_means[layer_offset, :, feature_id]
        valid = counts > 0
        if np.any(valid) and not np.allclose(
            pooled_mean[valid, 0], expected[valid], atol=3e-3, rtol=3e-2
        ):
            observed_valid = pooled_mean[valid, 0]
            expected_valid = expected[valid]
            absolute_error = np.abs(observed_valid - expected_valid)
            valid_trs = np.flatnonzero(valid)
            worst_local = int(np.argmax(absolute_error))
            worst_tr = int(valid_trs[worst_local])
            raise RuntimeError(
                f"token pooling does not reproduce canonical mean for {row['feature_name']}; "
                f"worst TR={worst_tr}, observed={float(pooled_mean[worst_tr, 0]):.8g}, "
                f"expected={float(expected[worst_tr]):.8g}, "
                f"max_abs_error={float(absolute_error[worst_local]):.8g}"
            )
    coordinate_manifest["analysis_fingerprint"] = fingerprint
    coordinate_manifest["token_records_sha256"] = None
    coordinate_manifest["runtime_config_sha256"] = sha256_file(runtime_config)
    coordinate_manifest["predictor_sha256"] = sha256_file(predictor_path)
    coordinate_manifest["canonical_mean_reproduction_tolerance"] = {"atol": 3e-3, "rtol": 3e-2}
    atomic_json(manifest_path, coordinate_manifest)
    _write_npz_atomic(
        token_path,
        token_values=token_values,
        token_positions=np.arange(n_tokens, dtype=np.int64),
        token_ids=np.asarray(token_ids, dtype=np.int64),
        tokens=np.asarray(tokens, dtype=str),
        token_to_tr=token_to_tr.astype(np.int32),
        tr_indices=tr_indices,
        tr_start_s=tr_timing[:, 0],
        tr_end_s=tr_timing[:, 1],
        token_count=canonical_token_count,
        feature_layers=np.asarray([int(row["layer"]) for row in coords], dtype=np.int32),
        feature_ids=np.asarray([int(row["feature_id"]) for row in coords], dtype=np.int32),
        feature_names=np.asarray([str(row["feature_name"]) for row in coords], dtype=str),
        model_id=np.asarray([str(model_info["model_id"])]),
        model_key=np.asarray([model_key]),
        story=np.asarray([story]),
        context_tokens=np.asarray([1024], dtype=np.int32),
        causal=np.asarray([True]),
        stimulus_onset_s=np.asarray([float(metadata.get("stimulus_onset_s", 0.0))], dtype=np.float64),
    )
    coordinate_manifest["token_records_sha256"] = sha256_file(token_path)
    atomic_json(manifest_path, coordinate_manifest)
    summary = {
        "schema_version": 1,
        "analysis_role": "correspondence_token_level_pooling_inputs",
        "analysis_fingerprint": fingerprint,
        "model_key": model_key,
        "model_id": str(model_info["model_id"]),
        "story": story,
        "sae_release": str(model_info["sae_release"]),
        "sae_width": str(model_info["sae_width"]),
        "layers": [int(value) for value in model_info["layers"]],
        "n_tokens": int(n_tokens),
        "n_tr": int(len(tr_indices)),
        "n_coordinates": int(len(coords)),
        "coordinate_manifest": "coordinate_manifest.json",
        "token_records": "token_records.npz",
        "predictor_id": f"predictors/{model_key}/{story}_lm_controls.npz",
        "predictor_sha256": sha256_file(predictor_path),
        "sae_parameter_sha256": sae_parameter_sha256,
        "token_records_sha256": coordinate_manifest["token_records_sha256"],
        "pooling_names": list(POOLING_NAMES),
        "sum_semantics": "sum of positive SAE activation over assigned transcript tokens",
        "max_semantics": "maximum positive SAE activation over assigned transcript tokens; zero for empty TR",
        "event_count_semantics": "count of assigned transcript tokens with activation strictly greater than zero",
        "mean_denominator_semantics": "all assigned transcript tokens, including zero-activation tokens",
        "windows": [[int(start), int(end)] for start, end in windows_seen],
        "window_context_tokens": 1024,
        "causal": True,
        "outcome_free": True,
        "elapsed_seconds": float(time.time() - started),
    }
    atomic_json(summary_path, summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", required=True)
    parser.add_argument("--story", nargs="+", choices=("shapesphysical", "shapessocial"), required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--sae-root", type=str, default=None, help="SAE cache directory; may contain {model_key}")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--json", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = (args.config or default_config_path()).resolve()
    spec = load_config(config_path)
    if args.model not in spec["models"]:
        raise SystemExit(f"unknown model key: {args.model}")
    runtime_paths = RuntimePaths.resolve(
        data_root=args.data_root,
        output_root=args.output_root,
        cache_root=args.cache_root,
    )
    data_root = runtime_paths.require_data_root()
    analysis_output_root = runtime_paths.require_output_root()
    cache_root = runtime_paths.require_cache_root()
    transcript_root = resolve_below(data_root, str(spec["paths"]["transcript_root"]))
    output_root = (
        resolve_below(analysis_output_root, str(spec["paths"]["artifact_root"]))
        / "pooling_token_records"
    )
    sae_root = _resolve_sae_root(
        cache_root,
        spec,
        args.model,
        args.sae_root,
        required_layers=spec["models"][args.model]["layers"],
    )
    summaries = [
        extract_story(
            config_path=config_path,
            spec=spec,
            model_key=args.model,
            story=story,
            data_root=data_root,
            analysis_output_root=analysis_output_root,
            output_root=output_root,
            transcript_root=transcript_root,
            sae_root=sae_root,
            force=args.force,
            size=args.size,
        )
        for story in args.story
    ]
    result = {
        "stage": "pooling_token_extraction",
        "model": args.model,
        "stories": list(args.story),
        "summaries": summaries,
        "output_root": str(Path(output_root).resolve()),
    }
    print(json.dumps(result, indent=2, sort_keys=True, default=str) if args.json else result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
