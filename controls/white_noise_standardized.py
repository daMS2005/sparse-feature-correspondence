"""White-noise control with the agreement computed from standardized weights (weights_scaled) on both sides.

The noise targets and LM fits are those of the white-noise control (same seeds, same penalty search);
fit_ridge_with_standardization's raw-unit factor y_scale / x_scale is undone before the norms are taken.

usage: white_noise_standardized.py RUN_DIR BRAIN_TARGETS_NPZ OUT_CSV
"""
import csv
import sys
from pathlib import Path

import numpy as np

from structure_comparison._implementation import (
    build_brain_design_matrix, compare_feature_importance, consensus_alpha, contiguous_block_groups,
    fit_final_model, load_brain_targets, run_grouped_ridge_cv)

ALPHA_GRID, LAGS, SEED = [0.1, 1.0, 10.0, 100.0, 1000.0], [0, 1, 2, 3, 4], 20261015


def sd(v):
    s = v.std(0)
    s[s == 0] = 1.0
    return s


run_dir, brain_npz, out_csv = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
feats = np.load(run_dir / "intermediates/tr_feature_average.npz", allow_pickle=True)
hidden = np.load(run_dir / "intermediates/tr_lm_target_final_hidden_average.npz", allow_pickle=True)
bm = np.load(run_dir / "all_layers/brain_final_model.npz", allow_pickle=True)
lm = np.load(run_dir / "all_layers/lm_final_model.npz", allow_pickle=True)
x, tr, ids = np.asarray(feats["values"], float), np.asarray(feats["tr_indices"], int), feats["sample_ids"].astype(str)
y_lm, names = np.asarray(hidden["values"], float), bm["base_feature_names"].astype(str)

d = build_brain_design_matrix(tr_predictors=x, tr_indices=tr, feature_names=names,
                              brain_targets=load_brain_targets(brain_npz), lags=LAGS)
assert (d["feature_names"] == bm["feature_names"].astype(str)).all()
brain_std = (np.asarray(bm["weights"], float) * sd(d["x"])[:, None] / sd(d["y"])[None, :]).reshape(len(names), len(LAGS), -1).sum(1)
lm_std = np.asarray(lm["weights"], float) * sd(x)[:, None] / sd(y_lm)[None, :]
observed = compare_feature_importance(brain_std, lm_std, names)["importance_correlation"]

with open(out_csv, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["run", "rep", "seed", "selected_alpha", "observed_standardized", "importance_correlation_standardized"])
    for rep in range(100):
        noise = np.random.default_rng([SEED, 1, rep]).standard_normal(y_lm.shape)
        cv = run_grouped_ridge_cv(x=x, y=noise, groups=contiguous_block_groups(len(tr), 5), sample_ids=ids,
                                  target_names=hidden["target_names"].astype(str), alpha_grid=ALPHA_GRID,
                                  family_name="all_layers:lm")
        alpha = consensus_alpha(cv["outer_folds"])
        noise_std = fit_final_model(x, noise, alpha, "all_layers:lm")["weights"] * sd(x)[:, None] / sd(noise)[None, :]
        value = compare_feature_importance(brain_std, noise_std, names)["importance_correlation"]
        w.writerow([run_dir.name.replace("_average_all_layers_final_hidden_state", ""), rep, [SEED, 1, rep], alpha, observed, value])
