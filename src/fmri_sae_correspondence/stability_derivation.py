#!/usr/bin/env python3
"""Derive reliability summaries from already-saved reliance profiles.

This stage deliberately does not refit any model and does not infer missing
participant or fold profiles.  It reads only schemas that expose an explicit
profile axis (participants for BOLD, temporal folds for the LM), computes
pairwise Spearman correlations, and writes a versioned diagnostic artifact.
The result is a stability diagnostic for the *derived* reliance profile, not a
BOLD acquisition noise ceiling and not a new feature-correspondence test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
MODEL_KEYS = (
    "gemma_2_2b",
    "gemma_2_9b",
    "llama_3_1_8b",
    "gemma_3_1b",
    "gemma_3_4b",
    "gemma_3_12b",
)
STORIES = ("shapesphysical", "shapessocial")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(value), indent=2, sort_keys=True) + "\n")


def pairwise_spearman(profiles: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the upper-triangle pairwise Spearman values and indices.

    Rows are independent profiles.  A pair is omitted if either profile is
    constant or has fewer than three jointly finite feature entries.
    """
    profiles = np.asarray(profiles, dtype=np.float64)
    if profiles.ndim != 2:
        raise ValueError("profiles must have shape n_profiles x n_features")
    # The saved reliance arrays are normally complete.  Rank all rows once in
    # that common case; this is both exact and substantially faster than
    # ranking each profile pair independently.
    if np.isfinite(profiles).all():
        ranks = rankdata(profiles, axis=1, method="average").astype(np.float64)
        ranks -= ranks.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(ranks, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = (ranks @ ranks.T) / (norms[:, None] * norms[None, :])
        upper = np.triu_indices(profiles.shape[0], k=1)
        return corr[upper], np.column_stack(upper).astype(np.int32)

    values: list[float] = []
    pairs: list[tuple[int, int]] = []
    for i in range(profiles.shape[0]):
        for j in range(i + 1, profiles.shape[0]):
            valid = np.isfinite(profiles[i]) & np.isfinite(profiles[j])
            if int(valid.sum()) < 3:
                continue
            left = rankdata(profiles[i, valid], method="average")
            right = rankdata(profiles[j, valid], method="average")
            left -= left.mean()
            right -= right.mean()
            denominator = float(np.sqrt(np.sum(left**2) * np.sum(right**2)))
            if denominator <= 0:
                continue
            values.append(float(np.sum(left * right) / denominator))
            pairs.append((i, j))
    return np.asarray(values, dtype=np.float64), np.asarray(pairs, dtype=np.int32).reshape(-1, 2)


def bootstrap_pairwise_mean(
    profiles: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap the mean pairwise profile correlation by profile resampling."""
    profiles = np.asarray(profiles, dtype=np.float64)
    result = np.full(int(draws), np.nan, dtype=np.float64)
    if profiles.shape[0] < 2:
        return result
    rng = np.random.default_rng(seed)
    n_profiles = profiles.shape[0]
    if np.isfinite(profiles).all():
        # Spearman is Pearson correlation after rank transformation.  For
        # unit-norm centered rows, the mean of all pairwise correlations in a
        # bootstrap sample can be obtained from the squared norm of the row
        # sum, avoiding an O(n_profiles^2) correlation matrix per draw.
        ranks = rankdata(profiles, axis=1, method="average").astype(np.float64)
        ranks -= ranks.mean(axis=1, keepdims=True)
        norms = np.linalg.norm(ranks, axis=1, keepdims=True)
        normalized = ranks / norms
        pair_count = n_profiles * (n_profiles - 1) / 2
        chunk = 256
        for start in range(0, int(draws), chunk):
            stop = min(start + chunk, int(draws))
            indices = rng.integers(0, n_profiles, size=(stop - start, n_profiles))
            sampled = normalized[indices]
            sum_vectors = sampled.sum(axis=1)
            # Each sampled row has unit squared norm.
            pair_sums = np.sum(sum_vectors * sum_vectors, axis=1) - n_profiles
            result[start:stop] = pair_sums / (2.0 * pair_count)
        return result
    for draw in range(int(draws)):
        indices = rng.integers(0, profiles.shape[0], size=profiles.shape[0])
        pairwise, _ = pairwise_spearman(profiles[indices])
        if pairwise.size:
            result[draw] = float(np.mean(pairwise))
    return result


def summarize_profile(
    profiles: np.ndarray,
    *,
    bootstrap_draws: int,
    seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    profiles = np.asarray(profiles, dtype=np.float64)
    pairwise, pairs = pairwise_spearman(profiles)
    boot = bootstrap_pairwise_mean(profiles, draws=bootstrap_draws, seed=seed)
    finite_boot = boot[np.isfinite(boot)]
    summary = {
        "profile_count": int(profiles.shape[0]),
        "feature_count": int(profiles.shape[1]),
        "pair_count": int(pairwise.size),
        "pairwise_mean_spearman": float(np.mean(pairwise)) if pairwise.size else float("nan"),
        "pairwise_median_spearman": float(np.median(pairwise)) if pairwise.size else float("nan"),
        "pairwise_q025": float(np.quantile(pairwise, 0.025)) if pairwise.size else float("nan"),
        "pairwise_q975": float(np.quantile(pairwise, 0.975)) if pairwise.size else float("nan"),
        "bootstrap_mean_pairwise_spearman": {
            "valid_draws": int(finite_boot.size),
            "requested_draws": int(bootstrap_draws),
            "estimate": float(np.mean(pairwise)) if pairwise.size else float("nan"),
            "ci95_low": float(np.quantile(finite_boot, 0.025)) if finite_boot.size else float("nan"),
            "ci95_high": float(np.quantile(finite_boot, 0.975)) if finite_boot.size else float("nan"),
            "seed": int(seed),
        },
    }
    return summary, {
        "profiles": profiles.astype(np.float32),
        "pairwise_spearman": pairwise.astype(np.float32),
        "pair_indices": pairs.astype(np.int32),
        "bootstrap_mean_pairwise_spearman": boot.astype(np.float32),
    }


def _condition_from_directory(directory: Path) -> tuple[str, str] | None:
    name = directory.name
    for story in STORIES:
        suffix = "_" + story
        if name.endswith(suffix):
            model = name[: -len(suffix)]
            if model in MODEL_KEYS:
                return model, story
    return None


def discover_sources(
    source_dirs: list[Path],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Discover explicit participant/fold schemas without fabricating pairs."""
    candidates: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    for root in source_dirs:
        root = Path(root).resolve()
        if not root.exists():
            inventory.append({"source": root.name, "status": "missing_source_directory"})
            continue
        for arrays_path in sorted(root.glob("*/arrays.npz")):
            condition = _condition_from_directory(arrays_path.parent)
            if condition is None:
                inventory.append({
                    "path": f"{root.name}/{arrays_path.relative_to(root).as_posix()}",
                    "status": "ignored_unrecognized_condition_name",
                })
                continue
            model, story = condition
            summary_path = arrays_path.with_name("summary.json")
            candidates.append(
                {
                    "model": model,
                    "story": story,
                    "arrays_path": arrays_path,
                    "summary_path": summary_path if summary_path.exists() else None,
                    "source_root": root.name,
                }
            )
    return candidates, inventory


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args(argv)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    candidates, inventory = discover_sources(args.source_root)
    conditions: list[dict[str, Any]] = []
    seen_roles: set[tuple[str, str, str, str]] = set()
    for candidate in candidates:
        arrays_path = candidate["arrays_path"]
        with np.load(arrays_path, allow_pickle=False) as arrays:
            keys = set(arrays.files)
            # Prefer the participant-level bank when a source exposes
            # both the participant BOLD and LM fold schemas.  The LM-only
            # precompute source fills missing LM rows for the other models.
            profile_specs = []
            if "brain_r2_reliance" in keys and np.asarray(arrays["brain_r2_reliance"]).ndim == 2:
                profile_specs.append(("brain", "r2", np.asarray(arrays["brain_r2_reliance"])))
            if "brain_r_reliance" in keys and np.asarray(arrays["brain_r_reliance"]).ndim == 2:
                profile_specs.append(("brain", "r", np.asarray(arrays["brain_r_reliance"])))
            if "lm_r2_fold_reliance" in keys and np.asarray(arrays["lm_r2_fold_reliance"]).ndim == 2:
                profile_specs.append(("lm", "r2", np.asarray(arrays["lm_r2_fold_reliance"])))
            if "lm_r_fold_reliance" in keys and np.asarray(arrays["lm_r_fold_reliance"]).ndim == 2:
                profile_specs.append(("lm", "r", np.asarray(arrays["lm_r_fold_reliance"])))
            if not profile_specs:
                inventory.append({
                    "path": f"{candidate['source_root']}/{arrays_path.parent.name}/arrays.npz",
                    "status": "no_explicit_profile_axis",
                    "keys": sorted(keys),
                })
                continue
            summary = {}
            if candidate["summary_path"] is not None:
                try:
                    summary = json.loads(candidate["summary_path"].read_text())
                except json.JSONDecodeError:
                    summary = {"summary_read_error": "invalid JSON"}
            # Older summaries do not always expose n_features; the explicit
            # profile width is the authoritative bank size in that case.
            bank_size = int(summary.get("n_features", 0)) if summary else 0
            for role, metric, profiles in profile_specs:
                role_key = (candidate["model"], candidate["story"], role, metric)
                if role_key in seen_roles:
                    inventory.append({
                        "path": f"{candidate['source_root']}/{arrays_path.parent.name}/arrays.npz",
                        "status": "duplicate_role_preferred_existing_source",
                        "role": role,
                        "metric": metric,
                    })
                    continue
                seen_roles.add(role_key)
                seed_offset = (
                    MODEL_KEYS.index(candidate["model"]) * 100
                    + STORIES.index(candidate["story"]) * 10
                    + (0 if role == "brain" else 1)
                    + (0 if metric == "r2" else 2)
                )
                profile_summary, derived = summarize_profile(
                    profiles,
                    bootstrap_draws=args.bootstrap_draws,
                    seed=args.seed + seed_offset,
                )
                if bank_size == 0:
                    bank_size = int(profiles.shape[1])
                condition_name = f"{candidate['model']}_{candidate['story']}_{role}_{metric}"
                condition_dir = output_root / condition_name
                condition_dir.mkdir(parents=True, exist_ok=True)
                derived_path = condition_dir / "arrays.npz"
                np.savez_compressed(derived_path, **derived)
                source_summary = summary.get("analysis_role") if summary else None
                row = {
                    "schema_version": 1,
                    "artifact_role": "derived_reliance_profile_stability",
                    "analysis_status": "complete",
                    "analysis_scope": "post-confirmatory descriptive stability diagnostic",
                    "model_key": candidate["model"],
                    "story": candidate["story"],
                    "profile_role": role,
                    "profile_unit": "participant" if role == "brain" else "temporal_fold",
                    "metric": metric,
                    "bank_size": bank_size,
                    "source_path": f"{candidate['source_root']}/{arrays_path.parent.name}/arrays.npz",
                    "source_sha256": sha256_file(arrays_path),
                    "source_analysis_role": source_summary,
                    "profile_summary": profile_summary,
                    "derived_arrays_path": derived_path.relative_to(output_root).as_posix(),
                    "derived_arrays_sha256": sha256_file(derived_path),
                    "interpretation_note": (
                        "Participant profile correlations quantify stability of the saved "
                        "reliance summaries across participants; LM fold correlations quantify "
                        "stability across temporal folds. Neither is an acquisition-level BOLD "
                        "noise ceiling or evidence of cross-system correspondence."
                    ),
                }
                write_json(condition_dir / "summary.json", row)
                row["summary_sha256"] = sha256_file(condition_dir / "summary.json")
                conditions.append(row)

    aggregate = {
        "schema_version": 1,
        "artifact_role": "correspondence_derived_reliance_profile_stability_aggregate",
        "analysis_status": "complete",
        "calculator": "stability_derivation.py",
        "calculator_sha256": sha256_file(Path(__file__).resolve()),
        "parameters": {
            "bootstrap_draws": int(args.bootstrap_draws),
            "seed": int(args.seed),
            "pairwise_metric": "Spearman correlation across feature coordinates",
            "bootstrap_unit": "participant profiles for BOLD; temporal fold profiles for LM",
        },
        "conditions": sorted(conditions, key=lambda r: (MODEL_KEYS.index(r["model_key"]), STORIES.index(r["story"]), r["profile_role"], r["metric"])),
        "source_inventory": inventory,
        "coverage": {
            "participant_brain_conditions": int(sum(c["profile_role"] == "brain" for c in conditions)),
            "lm_fold_conditions": int(sum(c["profile_role"] == "lm" for c in conditions)),
            "paired_model_story_metric_conditions": int(len({(c["model_key"], c["story"], c["metric"]) for c in conditions if c["profile_role"] == "brain"} & {(c["model_key"], c["story"], c["metric"]) for c in conditions if c["profile_role"] == "lm"})),
        },
        "interpretation": (
            "This stage only summarizes explicit profile arrays already present in the "
            "repository. It does not reconstruct missing participant profiles, does not refit "
            "models, and should not be treated as a reliability-corrected feature correspondence."
        ),
    }
    write_json(output_root / "aggregate.json", aggregate)
    checksums: list[str] = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name != "SHA256SUMS.txt":
            checksums.append(f"{sha256_file(path)}  {path.relative_to(output_root)}")
    (output_root / "SHA256SUMS.txt").write_text("\n".join(checksums) + "\n")
    print(f"Wrote {len(conditions)} reliance-profile stability summaries to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
