#!/usr/bin/env python3
"""Generate analysis figures only from a complete evidence release.

This module is intentionally fail-closed.  It first builds the condition-level
aggregate manifest from per-condition summaries and requires that every
publication condition, reliability diagnostic, conditioning bundle, and real
power artifact is complete.  A partial run therefore creates an
inventory but cannot create a scientific chart.

The output images contain panel labels and axis labels, but no figure captions.
Captions are written separately as Markdown and LaTeX so the manuscript owns
caption placement and can keep the caption outside the rendered image.
"""

from __future__ import annotations

import argparse
import json
import os
import textwrap
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .artifact_manifest import (
    _power_artifact_status,
    artifact_root,
    build_manifest,
    directions,
    require_complete_manifest,
    sizes,
)
from .config import default_config_path
from .integrity import (
    AnalysisError,
    atomic_write_json,
    load_spec,
    provenance_fingerprint,
    sha256_file,
)

HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT_NAME = "figures"


class FigureDataError(AnalysisError):
    """Raised when a complete manifest points to a malformed result."""


COLORS = {
    "ink": "#14213D",
    "blue": "#277DA1",
    "teal": "#2A9D8F",
    "gold": "#E9C46A",
    "coral": "#E76F51",
    "purple": "#6D597A",
    "gray": "#6C757D",
    "grid": "#D9E2EC",
    "paper": "#FAFCFE",
}
MODEL_SHORT = {
    "gemma_2_2b": "G2-2B",
    "gemma_2_9b": "G2-9B",
    "llama_3_1_8b": "L3.1-8B",
    "gemma_3_1b": "G3-1B",
    "gemma_3_4b": "G3-4B",
    "gemma_3_12b": "G3-12B",
}
MODEL_COLORS = {
    "gemma_2_2b": COLORS["blue"],
    "gemma_2_9b": COLORS["teal"],
    "llama_3_1_8b": COLORS["coral"],
    "gemma_3_1b": COLORS["purple"],
    "gemma_3_4b": "#577590",
    "gemma_3_12b": "#F4A261",
}


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    except ImportError as exc:  # pragma: no cover - depends on runtime image
        raise FigureDataError("matplotlib is required to generate vector figures") from exc
    return plt, FancyArrowPatch, FancyBboxPatch


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise FigureDataError(f"cannot read JSON artifact {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise FigureDataError(f"JSON artifact is not an object: {path}")
    return value


def _npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FigureDataError(f"missing array artifact: {path}")
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: np.asarray(data[key]) for key in data.files}
    except (OSError, ValueError) as exc:
        raise FigureDataError(f"cannot load array artifact {path}: {exc}") from exc


def _summary_arrays(summary_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    summary = _json(summary_path)
    raw = summary.get("arrays_path")
    if not raw:
        raise FigureDataError(f"{summary_path} has no arrays_path")
    arrays_path = Path(str(raw))
    if not arrays_path.is_absolute():
        arrays_path = (summary_path.parent / arrays_path).resolve()
    return summary, _npz(arrays_path)


def _fit_path(root: Path, model: str, story: str, variant: str, size: int) -> Path:
    return root / "fit" / f"{model}_{story}" / variant / f"n{int(size)}" / "summary.json"


def _transfer_path(root: Path, model: str, source: str, target: str, variant: str, size: int) -> Path:
    return root / "transfer" / f"{model}_{source}_to_{target}" / variant / f"n{int(size)}" / "summary.json"


def _null_path(root: Path, model: str, source: str, target: str, variant: str, size: int) -> Path:
    return root / "nulls" / "cross_story" / f"{source}_to_{target}" / model / variant / f"n{int(size)}" / "summary.json"


def _finite_scalar(value: Any, label: str) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise FigureDataError(f"{label} is not numeric: {value!r}") from exc
    if not np.isfinite(numeric):
        raise FigureDataError(f"{label} is non-finite")
    return numeric


def _array_mean(values: np.ndarray, label: str) -> float:
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise FigureDataError(f"{label} contains no finite values")
    return float(np.mean(finite))


def _style_axes(ax: Any) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def _save_figure(fig: Any, output_root: Path, stem: str) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    pdf = output_root / f"{stem}.pdf"
    png = output_root / f"{stem}.png"
    # PDF is the manuscript/vector artifact; PNG is for visual QA and review.
    fig.savefig(pdf, bbox_inches="tight", facecolor="white", metadata={"Creator": "analysis figure generator"})
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    import matplotlib.pyplot as plt
    plt.close(fig)
    return [pdf, png]


def _wrapped(text: str, width: int = 30) -> str:
    return "\n".join(textwrap.wrap(text, width=width))


def make_figure1(output_root: Path) -> list[Path]:
    """Create the method schematic; captions are intentionally external."""

    plt, FancyArrowPatch, FancyBboxPatch = _require_matplotlib()
    fig, ax = plt.subplots(figsize=(13.2, 5.0), constrained_layout=True)
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 5.0)
    ax.axis("off")
    boxes = [
        (0.2, 3.08, 2.2, 1.32, "Source-story selection", "variance bank\nprevalence/pruned sensitivities", COLORS["blue"]),
        (2.8, 3.08, 2.2, 1.32, "Aligned predictors", "SAE features → tokens → TRs\ntraining-side identities fixed", COLORS["teal"]),
        (5.4, 3.43, 2.25, 1.1, "Cortical BOLD", "participant-specific\n0–4 TR lagged", COLORS["coral"]),
        (5.4, 1.28, 2.25, 1.1, "LM final state", "causal ≤1,024 tokens\nTR-average target", COLORS["gold"]),
        (8.2, 3.43, 2.1, 1.1, "Within story", "purged nested CV\nheld-out ablation", COLORS["coral"]),
        (8.2, 1.28, 2.1, 1.1, "Cross story", "fit source → transfer\nweights/scaling fixed", COLORS["teal"]),
        (10.75, 2.18, 2.25, 1.52, "Inference", "conditional reliance\ntemporal target null\nwithin-layer identity null", COLORS["ink"]),
    ]
    for x, y, w, h, title, subtitle, color in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.08", linewidth=1.7, edgecolor=color, facecolor="white")
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h * 0.68, title, ha="center", va="center", fontsize=10.3, weight="bold", color=color)
        ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center", fontsize=8.7, color=COLORS["ink"])
    arrows = [
        ((2.4, 3.74), (2.8, 3.74)),
        ((5.0, 3.74), (5.4, 3.98)),
        ((5.0, 3.74), (5.4, 1.72)),
        ((7.65, 3.98), (8.2, 3.98)),
        ((7.65, 1.72), (8.2, 1.72)),
        ((10.3, 3.98), (10.75, 3.10)),
        ((10.3, 1.72), (10.75, 2.78)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.4, color=COLORS["gray"]))
    ax.text(6.6, 0.42, "same feature–layer coordinates within each paired target comparison; target-specific temporal models retained", ha="center", fontsize=10.0, weight="bold", color=COLORS["ink"])
    return _save_figure(fig, output_root, "figure_1_validity_pipeline")


def _cross_story_rows(root: Path, spec: Mapping[str, Any], size: int = 256) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    variant = "source_story_variance"
    for model in spec["models"]:
        for source, target in directions(spec):
            path = _transfer_path(root, str(model), source, target, variant, size)
            summary, arrays = _summary_arrays(path)
            if summary.get("confirmatory_eligible") is True:
                # The stage deliberately leaves this false until the release
                # gate; the figure gate itself is the stronger condition.
                pass
            # Pearson r and R2_oos are tuned separately in the configured
            # specification.  Keep their selected summaries separate here;
            # reading mean_r from the R2-selected object silently reports the
            # correlation at the wrong penalty.
            brain_r = summary.get("brain_r_selected", {})
            brain_r2 = summary.get("brain_r2_selected", {})
            lm_r = summary.get("lm_r_selected", {})
            lm_r2 = summary.get("lm_r2_selected", {})
            rows.append({
                "model": str(model),
                "source": source,
                "target": target,
                "label": f"{MODEL_SHORT.get(str(model), str(model))} {source[6].upper()}→{target[6].upper()}",
                "brain_r": _finite_scalar(brain_r.get("mean_r"), f"{path}: brain mean_r"),
                "brain_r2": _finite_scalar(brain_r2.get("mean_r2_oos"), f"{path}: brain mean_r2_oos"),
                "lm_r": _finite_scalar(lm_r.get("mean_r"), f"{path}: LM mean_r"),
                "lm_r2": _finite_scalar(lm_r2.get("mean_r2_oos"), f"{path}: LM mean_r2_oos"),
                "correspondence": _finite_scalar(summary.get("primary_correspondence", {}).get("spearman"), f"{path}: correspondence"),
                "arrays": arrays,
                "summary_path": path,
            })
    if not rows:
        raise FigureDataError("no complete cross-story rows")
    return rows


def _reliability_rows(root: Path, spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for story in spec["stories"]:
        path = root / "reliability" / "bold" / str(story) / "summary.json"
        summary = _json(path)
        loo = summary.get("loo_isc", {}).get("all_parcel_entries", {})
        split = summary.get("split_half", {})
        rows.append({
            "story": str(story),
            "loo_median": _finite_scalar(loo.get("median"), f"{path}: LOO ISC median"),
            "loo_mean": _finite_scalar(loo.get("mean"), f"{path}: LOO ISC mean"),
            "split_half": _finite_scalar(split.get("spearman_brown_from_median_draw_mean"), f"{path}: split-half reliability"),
        })
    return rows


def make_figure2(root: Path, spec: Mapping[str, Any], output_root: Path) -> tuple[list[Path], list[Path], list[dict[str, Any]], list[dict[str, Any]]]:
    """Reliability plus held-out prediction across the two transfer directions."""

    plt, _, _ = _require_matplotlib()
    rows = _cross_story_rows(root, spec, size=256)
    reliability = _reliability_rows(root, spec)
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.55), constrained_layout=True)

    ax = axes[0]
    x = np.arange(len(reliability))
    width = 0.24
    ax.bar(x - width, [r["loo_median"] for r in reliability], width, color=COLORS["blue"], label="LOO ISC median")
    ax.bar(x, [r["loo_mean"] for r in reliability], width, color=COLORS["teal"], label="LOO ISC mean")
    ax.bar(x + width, [r["split_half"] for r in reliability], width, color=COLORS["gold"], label="split-half SB")
    ax.set_xticks(x)
    ax.set_xticklabels(["Physical", "Social"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("shared-response reliability")
    ax.set_title("A  Reliability diagnostic", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=7.5)
    _style_axes(ax)

    # Keep the model order stable across runs and distinguish source direction
    # using marker shape.  The points are deliberately not connected: these
    # are independent transfer conditions, not a time series.
    models = [str(model) for model in spec["models"]]
    ax = axes[1]
    for direction, marker, dx in [("shapesphysical", "o", -0.09), ("shapessocial", "s", 0.09)]:
        sub = [row for row in rows if row["source"] == direction]
        xpos = np.arange(len(models)) + dx
        values = [next(row["brain_r"] for row in sub if row["model"] == model) for model in models]
        ax.scatter(xpos, values, s=42, marker=marker, color=[MODEL_COLORS.get(model, COLORS["ink"]) for model in models], edgecolor="white", linewidth=0.7, label=f"{direction.replace('shapes', '').title()}→other")
    ax.axhline(0, color=COLORS["gray"], linewidth=0.9)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([MODEL_SHORT.get(model, model) for model in models], rotation=35, ha="right")
    ax.set_ylabel("held-out BOLD Pearson r")
    ax.set_title("B  Cross-story BOLD prediction", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=7.5)
    _style_axes(ax)

    ax = axes[2]
    for direction, marker, dx in [("shapesphysical", "o", -0.09), ("shapessocial", "s", 0.09)]:
        sub = [row for row in rows if row["source"] == direction]
        xpos = np.arange(len(models)) + dx
        values = [next(row["lm_r"] for row in sub if row["model"] == model) for model in models]
        ax.scatter(xpos, values, s=42, marker=marker, color=[MODEL_COLORS.get(model, COLORS["ink"]) for model in models], edgecolor="white", linewidth=0.7, label=f"{direction.replace('shapes', '').title()}→other")
    ax.axhline(0, color=COLORS["gray"], linewidth=0.9)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels([MODEL_SHORT.get(model, model) for model in models], rotation=35, ha="right")
    ax.set_ylabel("held-out LM Pearson r")
    ax.set_title("C  Cross-story LM prediction", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=7.5)
    _style_axes(ax)
    return _save_figure(fig, output_root, "figure_2_reliability_and_prediction"), [row["summary_path"] for row in rows] + [root / "reliability" / "bold" / str(row["story"]) / "summary.json" for row in reliability], rows, reliability


def _conditioning_curve(root: Path, spec: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    path = root / "conditioning" / "conditioning_summary.json"
    summary = _json(path)
    records = summary.get("records", [])
    xs: list[int] = []
    ys: list[float] = []
    for record in records:
        if record.get("bank_name") != "source_variance" or record.get("mode") != "cross_story":
            continue
        bank_size = int(record.get("bank_size", -1))
        rank_summary = record.get("source_summary", {}).get("rank_summary", {})
        condition = rank_summary.get("condition_number")
        if bank_size in sizes(spec) and condition is not None and np.isfinite(float(condition)):
            xs.append(bank_size)
            ys.append(float(condition))
    if not ys:
        raise FigureDataError("conditioning artifact contains no source-variance condition numbers")
    grouped: dict[int, list[float]] = {}
    for x, y in zip(xs, ys, strict=False):
        grouped.setdefault(x, []).append(y)
    return np.asarray(sorted(grouped), dtype=int), np.asarray([np.median(grouped[x]) for x in sorted(grouped)], dtype=float), [path]


def _rho_curve(root: Path, spec: Mapping[str, Any]) -> tuple[dict[str, list[float]], list[Path]]:
    curve: dict[str, list[float]] = {}
    sources: list[Path] = []
    for model in spec["models"]:
        for source, target in directions(spec):
            label = f"{MODEL_SHORT.get(str(model), str(model))} {source[6].upper()}→{target[6].upper()}"
            values: list[float] = []
            for size in sizes(spec):
                path = _transfer_path(root, str(model), source, target, "source_story_variance", size)
                summary = _json(path)
                value = _finite_scalar(summary.get("primary_correspondence", {}).get("spearman"), f"{path}: correspondence")
                values.append(value)
                sources.append(path)
            curve[label] = values
    return curve, sources


def _null_data(root: Path, spec: Mapping[str, Any], size: int = 256) -> tuple[list[dict[str, Any]], list[Path]]:
    rows: list[dict[str, Any]] = []
    sources: list[Path] = []
    for model in spec["models"]:
        for source, target in directions(spec):
            observed_path = _transfer_path(root, str(model), source, target, "source_story_variance", size)
            null_path = _null_path(root, str(model), source, target, "source_story_variance", size)
            observed = _json(observed_path)
            null_summary, arrays = _summary_arrays(null_path)
            if "null_correspondence" not in arrays:
                raise FigureDataError(f"{null_path} does not contain null_correspondence")
            null_values = np.asarray(arrays["null_correspondence"], dtype=float)
            if null_values.ndim != 2 or null_values.shape[1] < 1:
                raise FigureDataError(f"{null_path}: malformed null_correspondence shape {null_values.shape}")
            rho_null = null_values[:, 0]
            if not np.isfinite(rho_null).any():
                raise FigureDataError(f"{null_path}: null Spearman values are all non-finite")
            observed_value = _finite_scalar(observed.get("primary_correspondence", {}).get("spearman"), f"{observed_path}: observed correspondence")
            rows.append({"label": f"{MODEL_SHORT.get(str(model), str(model))} {source[6].upper()}→{target[6].upper()}", "observed": observed_value, "null_low": float(np.nanquantile(rho_null, 0.025)), "null_high": float(np.nanquantile(rho_null, 0.975)), "summary_path": observed_path, "null_path": null_path})
            sources.extend([observed_path, null_path])
    return rows, sources


def _walk_mappings(value: Any) -> Iterable[Mapping[str, Any]]:
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for child in value:
            yield from _walk_mappings(child)


def _power_points(root: Path, spec: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[Path]]:
    """Extract the configured power grid without assuming one output schema."""

    points: list[dict[str, Any]] = []
    sources: list[Path] = []
    for path in sorted(root.rglob("*.json")):
        if path.name in {"manifest.json", "power_status.json", "smoke.json"}:
            continue
        status, problems, _ = _power_artifact_status(path, spec)
        if status != "complete":
            # A malformed or partial file is not a plot source.  The
            # aggregate manifest normally catches it before this function is
            # called; keeping this local guard prevents direct callers from
            # accidentally mixing smoke/partial rows into a figure.
            continue
        data = _json(path)
        for mapping in _walk_mappings(data):
            oracle = mapping.get("oracle_correlation", mapping.get("rho_star"))
            detection = mapping.get(
                "detection_fraction",
                mapping.get("detection_probability", mapping.get("power")),
            )
            if oracle is None or detection is None:
                continue
            try:
                oracle_value = float(oracle)
                detection_value = float(detection)
            except (TypeError, ValueError):
                continue
            if np.isfinite(oracle_value) and np.isfinite(detection_value) and 0 <= detection_value <= 1:
                points.append({
                    "oracle": oracle_value,
                    "detection": detection_value,
                    "model": str(mapping.get("model", data.get("model", "unknown"))),
                    "design": str(mapping.get("design_label", mapping.get("design", path.stem))),
                    "path": path,
                })
    if not points:
        raise FigureDataError("complete power stage contains no recognizable oracle/detection grid")
    sources = sorted({row["path"] for row in points})
    return points, sources


def make_figure3(root: Path, spec: Mapping[str, Any], output_root: Path) -> tuple[list[Path], list[Path], dict[str, Any]]:
    plt, _, _ = _require_matplotlib()
    conditioning_x, conditioning_y, conditioning_sources = _conditioning_curve(root, spec)
    rho, rho_sources = _rho_curve(root, spec)
    null_rows, null_sources = _null_data(root, spec, size=256)
    power_points, power_sources = _power_points(root / "power", spec)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.0), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(conditioning_x, np.log10(conditioning_y), marker="o", color=COLORS["blue"], linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_xticks(conditioning_x)
    ax.set_xticklabels([str(v) for v in conditioning_x])
    ax.set_xlabel("source-variance bank size K")
    ax.set_ylabel("median log₁₀ condition number")
    ax.set_title("A  Predictor conditioning", loc="left", weight="bold")
    _style_axes(ax)

    ax = axes[0, 1]
    x = np.asarray(sizes(spec), dtype=float)
    for label, values in rho.items():
        model_short = label.split(" ", 1)[0]
        model_key = next((key for key, value in MODEL_SHORT.items() if value == model_short), None)
        direction = "P→S" if "P→S" in label else "S→P"
        ax.plot(x, values, marker="o" if direction == "P→S" else "s", linewidth=1.4, markersize=4, color=MODEL_COLORS.get(model_key, COLORS["ink"]), alpha=0.85, label=label)
    ax.axhline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(x)
    ax.set_xticklabels([str(int(v)) for v in x])
    ax.set_xlabel("source-variance bank size K")
    ax.set_ylabel("cross-story Spearman ρ")
    ax.set_title("B  Reliance correspondence across K", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=6.2, ncol=2)
    _style_axes(ax)

    ax = axes[1, 0]
    x = np.arange(len(null_rows))
    ax.vlines(x, [row["null_low"] for row in null_rows], [row["null_high"] for row in null_rows], color=COLORS["gray"], linewidth=6, alpha=0.50, label="temporal-target null 95% interval")
    ax.scatter(x, [row["observed"] for row in null_rows], color=COLORS["ink"], marker="D", s=28, zorder=3, label="observed")
    ax.axhline(0, color=COLORS["gray"], linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([row["label"] for row in null_rows], rotation=65, ha="right", fontsize=6.5)
    ax.set_ylabel("Spearman ρ")
    ax.set_title("C  Observed reliance versus retuned temporal null", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=7)
    _style_axes(ax)

    ax = axes[1, 1]
    models = sorted({str(point["model"]) for point in power_points})
    for model in models:
        model_points = [point for point in power_points if point["model"] == model]
        oracle_values = sorted({float(point["oracle"]) for point in model_points})
        means = []
        lows = []
        highs = []
        for oracle in oracle_values:
            detections = np.asarray(
                [point["detection"] for point in model_points if point["oracle"] == oracle],
                dtype=float,
            )
            means.append(float(np.mean(detections)))
            lows.append(float(np.min(detections)))
            highs.append(float(np.max(detections)))
        color = MODEL_COLORS.get(model, COLORS["ink"])
        label = MODEL_SHORT.get(model, model)
        ax.fill_between(oracle_values, lows, highs, color=color, alpha=0.16)
        ax.plot(oracle_values, means, marker="o", linewidth=1.8, color=color, label=label)
    ax.axhline(0.8, color=COLORS["coral"], linewidth=1.2, linestyle="--", label="80% threshold")
    ax.set_xlim(0, 0.82)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("oracle correspondence ρ⋆")
    ax.set_ylabel("detection probability")
    ax.set_title("D  Detectability under empirical design", loc="left", weight="bold")
    ax.legend(frameon=False, fontsize=7)
    _style_axes(ax)
    sources = conditioning_sources + rho_sources + null_sources + power_sources
    payload = {
        "conditioning_bank_sizes": conditioning_x.tolist(),
        "conditioning_median_condition_number": conditioning_y.tolist(),
        "rho_curve": rho,
        "null_rows": [{key: value for key, value in row.items() if key not in {"summary_path", "null_path"}} for row in null_rows],
        "power_points": [{"oracle": float(row["oracle"]), "detection": float(row["detection"]), "model": row["model"], "design": row["design"]} for row in power_points],
    }
    return _save_figure(fig, output_root, "figure_3_conditioning_correspondence_nulls_power"), sources, payload


CAPTIONS = {
    "figure_1_validity_pipeline": "Evaluation-first analysis pipeline. A source-story SAE bank is selected without target outcomes, aligned to transcript time and fMRI repetition bins, and shared by paired BOLD and language-model targets. Within-story and cross-story fits are evaluated with held-out conditional ablations; correspondence is tested against retuned temporal-target and within-layer identity nulls.",
    "figure_2_reliability_and_prediction": "Reliability and cross-story prediction. (A) Leave-one-participant-out intersubject correlation and split-half Spearman–Brown diagnostics quantify shared response reliability; they are not exact acquisition noise ceilings. (B–C) Pearson prediction scores for participant-specific cortical BOLD and TR-averaged final hidden-state targets at the configured K=256 source-variance bank, with marker shape denoting transfer direction.",
    "figure_3_conditioning_correspondence_nulls_power": "Conditioning, correspondence, nulls, and detectability. (A) Predictor condition number across source-variance bank sizes. (B) Cross-story Spearman correspondence across the configured bank-size curve. (C) Observed K=256 correspondence with 95% intervals from fully refitted circular temporal-target nulls. (D) Mean detection probability by model family across bank, direction, and residual-scale conditions; shaded bands show the full range and the dashed line marks the configured 80% identifiability threshold.",
}


def write_external_captions(output_root: Path) -> list[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    md = output_root / "figure_captions.md"
    tex = output_root / "figure_captions.tex"
    md_lines = ["# Figure captions", "", "Captions are intentionally stored outside the rendered image files.", ""]
    tex_lines = ["% Captions are intentionally external to the rendered image files."]
    for stem, caption in CAPTIONS.items():
        md_lines.extend([f"## {stem}", "", caption, ""])
        label = stem.replace("_", ":")
        tex_lines.extend([f"% {stem}", f"\\caption{{{caption}}}", f"\\label{{fig:{label}}}", ""])
    md.write_text("\n".join(md_lines), encoding="utf-8")
    tex.write_text("\n".join(tex_lines), encoding="utf-8")
    return [md, tex]


def run(*, config_path: Path, spec: Mapping[str, Any], args: Any = None) -> dict[str, Any]:
    """Build the evidence manifest, require completeness, then render."""

    full_hashes = bool(getattr(args, "full_hashes", False)) if args is not None else False
    manifest = build_manifest(config_path, spec, full_hashes=full_hashes)
    require_complete_manifest(manifest)
    root = artifact_root(config_path, spec)
    output_root = root / DEFAULT_OUTPUT_ROOT_NAME
    generated: list[Path] = []
    sources: list[Path] = []
    generated.extend(make_figure1(output_root))
    fig2, fig2_sources, prediction_rows, reliability_rows = make_figure2(root, spec, output_root)
    generated.extend(fig2)
    sources.extend(fig2_sources)
    fig3, fig3_sources, correspondence_payload = make_figure3(root, spec, output_root)
    generated.extend(fig3)
    sources.extend(fig3_sources)
    captions = write_external_captions(output_root)
    generated.extend(captions)
    data_path = output_root / "figure_data.json"
    atomic_write_json(
        data_path,
        {
            "schema_version": 1,
            "source_role": "canonical_figure_data_from_complete_correspondence_artifacts",
            "prediction_rows": [
                {key: value for key, value in row.items() if key not in {"arrays"}}
                for row in prediction_rows
            ],
            "reliability_rows": reliability_rows,
            "correspondence": correspondence_payload,
        },
    )
    generated.append(data_path)
    code_paths = [HERE / "figures.py", HERE / "artifact_manifest.py", config_path]
    fingerprint = provenance_fingerprint(config_path, input_paths=[Path(path) for path in sources], code_paths=code_paths, extra={"stage": "figures", "manifest_fingerprint": manifest.get("analysis_fingerprint")})
    figure_manifest = {
        "schema_version": 1,
        "analysis_role": "correspondence_publication_figures",
        "analysis_fingerprint": fingerprint,
        "aggregate_manifest": str(root / "aggregate_manifest.json"),
        "aggregate_manifest_sha256": sha256_file(root / "aggregate_manifest.json"),
        "caption_embedded_in_images": False,
        "figure_data_path": str(data_path),
        "source_artifacts": {str(Path(path)): sha256_file(Path(path)) for path in sorted(set(sources))},
        "generated_artifacts": {str(path): sha256_file(path) for path in sorted(set(generated))},
        "scientific_evidence": True,
        "complete": True,
    }
    atomic_write_json(output_root / "figure_manifest.json", figure_manifest)
    figure_manifest["manifest_path"] = str(output_root / "figure_manifest.json")
    return figure_manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--full-hashes", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    if args.data_root is not None:
        os.environ["FMRI_SAE_DATA_ROOT"] = str(args.data_root.resolve())
    if args.output_root is not None:
        os.environ["FMRI_SAE_OUTPUT_ROOT"] = str(args.output_root.resolve())
    try:
        config_path = (args.config or default_config_path()).resolve()
        spec = load_spec(config_path)
        result = run(config_path=config_path, spec=spec, args=args)
    except (AnalysisError, FileNotFoundError, ValueError) as exc:
        print(f"analysis figures failed closed: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
