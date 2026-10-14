#!/usr/bin/env python3
"""Verify analysis provenance, condition coverage, and figure identity.

The verifier is separate from the scientific stages so it can be run on a
completed tree without re-fitting anything.  It never treats a stage-level
manifest as authoritative: coverage is recomputed from the per-condition
summaries by :mod:`artifact_manifest`. A non-strict run writes a
diagnostic report even for an incomplete pipeline; ``--strict`` exits nonzero
until the full release is internally consistent.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .artifact_manifest import (
    artifact_root,
    build_manifest,
    canonicalize_evidence_path,
    expected_artifacts,
)
from .config import default_config_path
from .integrity import (
    AnalysisError,
    atomic_write_json,
    config_digest,
    load_spec,
    provenance_fingerprint,
    sha256_file,
)

HERE = Path(__file__).resolve().parent


class VerificationError(AnalysisError):
    """Raised by strict verification when one or more release checks fail."""


def _json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VerificationError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VerificationError(f"expected JSON object: {path}")
    return value


def _check_condition_counts(manifest: Mapping[str, Any], config_path: Path, spec: Mapping[str, Any]) -> dict[str, Any]:
    expected = expected_artifacts(config_path, spec)
    expected_counts: dict[str, int] = {}
    for item in expected:
        expected_counts[item.stage] = expected_counts.get(item.stage, 0) + 1
    observed_counts = {
        stage: int(sum(1 for row in manifest.get("conditions", []) if row.get("stage") == stage))
        for stage in expected_counts
    }
    mismatches = {
        stage: {"expected": expected_counts[stage], "observed": observed_counts.get(stage, 0)}
        for stage in expected_counts
        if observed_counts.get(stage, 0) != expected_counts[stage]
    }
    incomplete = [row for row in manifest.get("conditions", []) if row.get("status") != "complete"]
    return {
        "passed": not mismatches and not incomplete,
        "expected_counts": expected_counts,
        "observed_counts": observed_counts,
        "count_mismatches": mismatches,
        "incomplete_condition_count": len(incomplete),
        "incomplete_examples": [
            {"stage": row.get("stage"), "key": row.get("key"), "status": row.get("status"), "problems": row.get("problems", [])}
            for row in incomplete[:20]
        ],
    }


def _check_supporting_inventories(manifest: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    """Check reliability/pooling inventories independently of top-level flags."""

    expected = {
        "reliability": len(spec["stories"]),
        "pooling": len(spec["models"]) * len(spec["stories"]),
    }
    results: dict[str, Any] = {}
    passed = True
    for section, expected_count in expected.items():
        rows = manifest.get(section, [])
        rows = list(rows) if isinstance(rows, Sequence) and not isinstance(rows, (str, bytes)) else []
        incomplete = [
            {"key": row.get("key"), "status": row.get("status"), "problems": row.get("problems", [])}
            for row in rows
            if isinstance(row, Mapping) and row.get("status") != "complete"
        ]
        section_passed = len(rows) == expected_count and not incomplete
        passed = passed and section_passed
        results[section] = {
            "passed": section_passed,
            "expected_count": expected_count,
            "observed_count": len(rows),
            "incomplete_count": len(incomplete),
            "incomplete_examples": incomplete[:20],
        }
    # The generalized token-pooling analysis is a separate downstream
    # evidence family.  Older manifests may not contain it; in that case the
    # supporting check reports the missing family explicitly rather than
    # allowing canonical mean-only fits to stand in for max/sum/event-count
    # results.
    pooled_analysis = manifest.get("pooling_analysis")
    if isinstance(pooled_analysis, Mapping):
        pooled_status = str(pooled_analysis.get("status", "missing"))
        pooled_problems = list(pooled_analysis.get("problems", []))
    else:
        pooled_status = "missing"
        pooled_problems = ["pooling_analysis inventory is missing"]
    results["pooling_analysis"] = {
        "passed": pooled_status == "complete",
        "expected_count": int(len(spec["models"]) * 2 * 4 * 4 * 4),
        "observed_count": int(
            (pooled_analysis or {}).get("observed_complete_count", 0)
            + (pooled_analysis or {}).get("null_complete_count", 0)
        ) if isinstance(pooled_analysis, Mapping) else 0,
        "status": pooled_status,
        "problems": pooled_problems[:20],
    }
    passed = passed and pooled_status == "complete"
    return {"passed": passed, "sections": results}


def _check_primary_control_support(manifest: Mapping[str, Any], spec: Mapping[str, Any]) -> dict[str, Any]:
    """Require primary source-story controls and both paired inference summaries.

    The ordinary condition-count check catches absent primary summaries after
    they are enumerated by the aggregate manifest.  This separate check is
    intentional: it makes the release contract legible and prevents a future
    manifest refactor from allowing the generic fit/transfer curves—or the
    appendix historical/dense/random controls—to stand in for the required
    baseline-versus-multiband evidence.
    """

    expected_within = len(spec["models"]) * len(spec["stories"])
    expected_transfer = len(spec["models"]) * 2
    expected_total = expected_within + expected_transfer
    section = manifest.get("primary_controls")
    problems: list[str] = []
    if not isinstance(section, Mapping):
        return {
            "passed": False,
            "expected_within_conditions": expected_within,
            "expected_transfer_conditions": expected_transfer,
            "observed_within_conditions": 0,
            "observed_transfer_conditions": 0,
            "expected_inference_summaries": 2,
            "observed_inference_summaries": 0,
            "problems": ["primary_controls inventory is missing"],
        }
    if section.get("scope") != "primary_source_story_controls_only":
        problems.append("primary_controls inventory has an unexpected scope")
    def _count(name: str) -> int:
        try:
            return int(section.get(name, -1))
        except (TypeError, ValueError):
            problems.append(f"primary control {name} is not an integer")
            return -1

    within = _count("within_condition_count")
    transfer = _count("transfer_condition_count")
    total = _count("condition_count")
    if within != expected_within:
        problems.append(f"within primary controls={within}, expected {expected_within}")
    if transfer != expected_transfer:
        problems.append(f"transfer primary controls={transfer}, expected {expected_transfer}")
    if total != expected_total:
        problems.append(f"primary control total={total}, expected {expected_total}")
    if section.get("status") != "complete":
        problems.append(f"primary controls inventory status is {section.get('status')!r}")
    condition_rows = section.get("conditions")
    if not isinstance(condition_rows, Sequence) or isinstance(condition_rows, (str, bytes)):
        problems.append("primary controls inventory has no condition rows")
        condition_rows = []
    else:
        for row in condition_rows:
            if not isinstance(row, Mapping) or row.get("status") != "complete":
                problems.append("primary controls contain an incomplete condition")
                break
    inference = section.get("inference")
    if not isinstance(inference, Sequence) or isinstance(inference, (str, bytes)):
        problems.append("primary controls inventory has no inference summaries")
        inference = []
    inference_rows = list(inference)
    if len(inference_rows) != 2:
        problems.append(f"primary inference summaries={len(inference_rows)}, expected 2")
    if {str(row.get("regime")) for row in inference_rows if isinstance(row, Mapping)} != {"within", "transfer"}:
        problems.append("primary inference summaries do not cover within and transfer regimes")
    for row in inference_rows:
        if not isinstance(row, Mapping) or row.get("status") != "complete":
            problems.append(f"primary inference summary is {row.get('status') if isinstance(row, Mapping) else 'malformed'}")
    # Optional controls are deliberately not part of this gate.  If present,
    # they must remain explicitly labelled appendix-only so a copied old
    # result cannot silently become the primary evidence.
    optional = manifest.get("optional_controls")
    if isinstance(optional, Mapping) and optional.get("required_for_release") is True:
        problems.append("historical/dense/random controls are incorrectly marked release-required")
    return {
        "passed": not problems,
        "expected_within_conditions": expected_within,
        "expected_transfer_conditions": expected_transfer,
        "observed_within_conditions": within,
        "observed_transfer_conditions": transfer,
        "expected_total_conditions": expected_total,
        "observed_total_conditions": total,
        "expected_inference_summaries": 2,
        "observed_inference_summaries": len(inference_rows),
        "problems": problems[:100],
    }


def _check_code_config_fingerprints(
    manifest: Mapping[str, Any],
    config_path: Path,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    problems: list[str] = []
    expected_config = config_digest(config_path)
    if manifest.get("config_sha256") != expected_config:
        problems.append("aggregate manifest config_sha256 does not match the current configuration")
    summary_count = 0
    code_count = 0
    input_count = 0
    for row in manifest.get("conditions", []):
        if row.get("status") != "complete":
            continue
        path = Path(str(row["path"]))
        try:
            summary = _json(path)
        except VerificationError as exc:
            problems.append(str(exc))
            continue
        if not summary.get("analysis_fingerprint"):
            problems.append(f"{path}: missing analysis_fingerprint")
        input_hashes = summary.get("input_sha256")
        code_hashes = summary.get("code_sha256")
        if not isinstance(input_hashes, Mapping) or not input_hashes:
            problems.append(f"{path}: missing input_sha256 map")
        else:
            input_count += len(input_hashes)
            for raw_path, recorded in input_hashes.items():
                item, mapping_reason = canonicalize_evidence_path(
                    str(raw_path), config_path, spec
                )
                if mapping_reason == "unmapped_absolute":
                    problems.append(f"{path}: input_sha256 path is unmappable: {raw_path}")
                    continue
                if not item.is_file():
                    problems.append(f"{path}: input_sha256 path missing after canonicalization: {item}")
                elif sha256_file(item) != str(recorded):
                    problems.append(f"{path}: input_sha256 hash mismatch: {item}")
        if not isinstance(code_hashes, Mapping) or not code_hashes:
            problems.append(f"{path}: missing code_sha256 map")
        else:
            code_count += len(code_hashes)
            for raw_path, recorded in code_hashes.items():
                item, mapping_reason = canonicalize_evidence_path(
                    str(raw_path), config_path, spec
                )
                if mapping_reason == "unmapped_absolute":
                    problems.append(f"{path}: code_sha256 path is unmappable: {raw_path}")
                    continue
                if not item.is_file():
                    problems.append(f"{path}: code_sha256 path missing after canonicalization: {item}")
                elif sha256_file(item) != str(recorded):
                    problems.append(f"{path}: code_sha256 hash mismatch: {item}")
        summary_count += 1
    return {
        "passed": not problems,
        "current_config_sha256": expected_config,
        "complete_summaries_checked": summary_count,
        "input_hash_entries_checked": input_count,
        "code_hash_entries_checked": code_count,
        "problems": problems[:100],
    }


def _check_figures(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    figure_root = root / "figures"
    figure_manifest_path = figure_root / "figure_manifest.json"
    if not figure_manifest_path.is_file():
        return {"passed": False, "status": "missing", "problems": [f"figure manifest missing: {figure_manifest_path}"]}
    problems: list[str] = []
    try:
        figure_manifest = _json(figure_manifest_path)
    except VerificationError as exc:
        return {"passed": False, "status": "invalid", "problems": [str(exc)]}
    if figure_manifest.get("complete") is not True or figure_manifest.get("scientific_evidence") is not True:
        problems.append("figure manifest is not marked complete scientific evidence")
    aggregate_path = root / "aggregate_manifest.json"
    current_aggregate_hash = sha256_file(aggregate_path) if aggregate_path.is_file() else None
    if current_aggregate_hash is None or figure_manifest.get("aggregate_manifest_sha256") != current_aggregate_hash:
        problems.append("figure manifest does not reference the current aggregate manifest hash")
    if figure_manifest.get("caption_embedded_in_images") is not False:
        problems.append("figure manifest does not declare external captions")
    generated_map = figure_manifest.get("generated_artifacts", {})
    if not isinstance(generated_map, Mapping) or not generated_map:
        problems.append("figure manifest has no generated artifact hash map")
        generated_map = {}
    required_names = {
        "figure_1_validity_pipeline.pdf",
        "figure_1_validity_pipeline.png",
        "figure_2_reliability_and_prediction.pdf",
        "figure_2_reliability_and_prediction.png",
        "figure_3_conditioning_correspondence_nulls_power.pdf",
        "figure_3_conditioning_correspondence_nulls_power.png",
        "figure_captions.md",
        "figure_captions.tex",
        "figure_data.json",
    }
    observed_names = {Path(str(raw_path)).name for raw_path in generated_map}
    for missing_name in sorted(required_names - observed_names):
        problems.append(f"required figure artifact is absent from hash map: {missing_name}")
    for raw_path, recorded in generated_map.items():
        path = Path(str(raw_path))
        if not path.is_absolute():
            path = (figure_root / path).resolve()
        if not path.is_file():
            problems.append(f"generated figure artifact missing: {path}")
        elif sha256_file(path) != str(recorded):
            problems.append(f"generated figure artifact hash mismatch: {path}")
    data_path = Path(str(figure_manifest.get("figure_data_path", figure_root / "figure_data.json")))
    if not data_path.is_absolute():
        data_path = (figure_root / data_path).resolve()
    if not data_path.is_file():
        problems.append(f"canonical figure data missing: {data_path}")
    return {"passed": not problems, "status": "complete" if not problems else "invalid", "problems": problems, "figure_manifest_path": str(figure_manifest_path), "figure_data_path": str(data_path)}


def _check_figure_data_identity(root: Path, spec: Mapping[str, Any], figure_check: Mapping[str, Any]) -> dict[str, Any]:
    """Compare canonical figure data to the source summaries.

    The check is intentionally independent of the plotting code.  It catches
    the common failure mode in which a table/figure is copied from an earlier
    run while the source artifact tree has changed.
    """

    if not figure_check.get("figure_data_path") or not Path(str(figure_check["figure_data_path"])).is_file():
        return {"passed": False, "problems": ["figure data unavailable for source-value identity check"]}
    data_path = Path(str(figure_check["figure_data_path"]))
    data = _json(data_path)
    problems: list[str] = []
    rows = data.get("prediction_rows", [])
    expected: dict[tuple[str, str], dict[str, float]] = {}
    for model in spec["models"]:
        for source, target in (("shapesphysical", "shapessocial"), ("shapessocial", "shapesphysical")):
            path = root / "transfer" / f"{model}_{source}_to_{target}" / "source_story_variance" / "n256" / "summary.json"
            if not path.is_file():
                problems.append(f"source summary missing: {path}")
                continue
            summary = _json(path)
            expected[(str(model), source)] = {
                "brain_r": float(summary["brain_r2_selected"]["mean_r"]),
                "brain_r2": float(summary["brain_r2_selected"]["mean_r2_oos"]),
                "lm_r": float(summary["lm_r2_selected"]["mean_r"]),
                "lm_r2": float(summary["lm_r2_selected"]["mean_r2_oos"]),
                "correspondence": float(summary["primary_correspondence"]["spearman"]),
            }
    seen: set[tuple[str, str]] = set()
    for row in rows:
        key = (str(row.get("model")), str(row.get("source")))
        if key not in expected:
            problems.append(f"unexpected figure row key: {key}")
            continue
        seen.add(key)
        for field, expected_value in expected[key].items():
            try:
                observed = float(row[field])
            except (KeyError, TypeError, ValueError):
                problems.append(f"figure row {key} missing numeric field {field}")
                continue
            if not np.isclose(observed, expected_value, rtol=0.0, atol=1e-12):
                problems.append(f"figure row {key} field {field} differs from source summary")
    for key in sorted(set(expected) - seen):
        problems.append(f"figure data is missing source row {key}")
    return {"passed": not problems, "rows_checked": len(seen), "source_rows_expected": len(expected), "problems": problems[:100]}


def verify_release(config_path: Path, spec: Mapping[str, Any], *, full_hashes: bool = False) -> dict[str, Any]:
    manifest = build_manifest(config_path, spec, full_hashes=full_hashes)
    root = artifact_root(config_path, spec)
    counts = _check_condition_counts(manifest, config_path, spec)
    provenance = _check_code_config_fingerprints(manifest, config_path, spec)
    figures = _check_figures(root, manifest)
    figure_values = _check_figure_data_identity(root, spec, figures)
    supporting = _check_supporting_inventories(manifest, spec)
    primary_controls = _check_primary_control_support(manifest, spec)
    checks = {"condition_counts": counts, "primary_control_support": primary_controls, "supporting_inventories": supporting, "code_config_fingerprints": provenance, "figures": figures, "figure_source_value_identity": figure_values}
    passed = bool(manifest.get("complete")) and all(bool(value.get("passed")) for value in checks.values())
    report_path = root / "verification" / "verification_report.json"
    fingerprint = provenance_fingerprint(config_path, input_paths=[root / "aggregate_manifest.json"], code_paths=[HERE / "verification.py", HERE / "artifact_manifest.py"], extra={"full_hashes": full_hashes})
    report = {
        "schema_version": 1,
        "analysis_role": "correspondence_release_verification",
        "analysis_fingerprint": fingerprint,
        "aggregate_manifest_path": str(root / "aggregate_manifest.json"),
        "aggregate_manifest_sha256": sha256_file(root / "aggregate_manifest.json"),
        "checks": checks,
        "complete": passed,
        "scientific_evidence": passed,
        "full_hashes": bool(full_hashes),
        "report_path": str(report_path),
    }
    atomic_write_json(report_path, report)
    report["report_path"] = str(report_path)
    return report


def run(*, config_path: Path, spec: Mapping[str, Any], args: Any = None) -> dict[str, Any]:
    full_hashes = bool(getattr(args, "full_hashes", False)) if args is not None else False
    strict = bool(getattr(args, "strict", False)) if args is not None else False
    report = verify_release(config_path, spec, full_hashes=full_hashes)
    if strict and not report["complete"]:
        raise VerificationError(f"release verification failed; see {report['report_path']}")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--full-hashes", action="store_true")
    parser.add_argument("--strict", action="store_true")
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
        print(f"analysis verification failed: {exc}", file=os.sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, default=str))
    return 0 if result.get("complete") else 1


if __name__ == "__main__":
    raise SystemExit(main())
