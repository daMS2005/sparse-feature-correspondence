"""Merge control replicates into one CSV per condition and apply the decision rule.

usage: collect_controls.py RUNS_DIR OUT_DIR REPLICATE_DIR [REPLICATE_DIR ...]
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

RUNS_DIR, out, *REPLICATE_DIRS = map(Path, sys.argv[1:])
RUNS = ["gemma_2_2b_shapesphysical_r4", "gemma_2_9b_shapesphysical_r5", "llama_3_1_8b_shapesphysical_r4",
        "gemma_2_2b_shapessocial_r4", "gemma_2_9b_shapessocial_r5", "llama_3_1_8b_shapessocial_r4"]
CONDITIONS = ["lm_shift", "brain_shift", "white_noise", "phase"]
out.mkdir(parents=True, exist_ok=True)

rows = {c: [] for c in CONDITIONS}
for run in RUNS:
    for c in CONDITIONS:
        files = [f for d in REPLICATE_DIRS for f in sorted((d / run).glob(f"{c}_*.json"))]
        reps = [json.loads(f.read_text()) | {"run": run} for f in files]
        assert sorted(r["rep"] for r in reps) == list(range(100)), (run, c, len(reps))
        rows[c] += sorted(reps, key=lambda r: r["rep"])

for c, reps in rows.items():
    key = "offset" if "shift" in c else "seed"
    with open(out / f"{c}.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "rep", key, "selected_alpha", "importance_correlation"])
        for r in reps:
            w.writerow([r["run"], r["rep"], r[key], r["selected_alpha"], r["importance_correlation"]])

summary = []
for run in RUNS:
    obs = json.loads((RUNS_DIR / f"{run}_average_all_layers_final_hidden_state/all_layers/feature_importance_summary.json")
                     .read_text())["importance_correlation"]
    row = {"run": run, "observed": obs}
    for c in CONDITIONS:
        v = np.array([r["importance_correlation"] for r in rows[c] if r["run"] == run])
        row[c] = {"p95": float(np.percentile(v, 95)), "median": float(np.median(v)),
                  "min": float(v.min()), "max": float(v.max()),
                  "fraction_at_or_above_observed": float((v >= obs).mean())}
    row["exceeds_lm_shift_p95"] = bool(obs > row["lm_shift"]["p95"])
    summary.append(row)

n_exceed = sum(r["exceeds_lm_shift_p95"] for r in summary)
result = {"runs": summary, "runs_exceeding": n_exceed, "fallback_triggered": n_exceed >= 2,
          "rule": "fallback if observed > 95th percentile of LM-shifted distribution in >= 2 of 6 runs"}
(out / "summary.json").write_text(json.dumps(result, indent=1))

print(f"{'run':32s} {'obs':>6s} | {'LMshift p95':>11s} {'frac>=obs':>9s} | {'brain p95':>9s} {'white p95':>9s} {'phase p95':>9s} | verdict")
for r in summary:
    print(f"{r['run']:32s} {r['observed']:6.3f} | {r['lm_shift']['p95']:11.3f} {r['lm_shift']['fraction_at_or_above_observed']:9.2f} | "
          f"{r['brain_shift']['p95']:9.3f} {r['white_noise']['p95']:9.3f} {r['phase']['p95']:9.3f} | "
          f"{'EXCEEDS' if r['exceeds_lm_shift_p95'] else 'does not exceed'}")
print(f"runs exceeding: {n_exceed}/6 -> fallback {'TRIGGERED' if n_exceed >= 2 else 'not triggered'}")
