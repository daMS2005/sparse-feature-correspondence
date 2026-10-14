#!/usr/bin/env python3
"""Self-recovery of a planted ablation-loss ordering as the number of real feature groups grows.

Draws a random subset of K real feature groups of one transfer design, keeps only
their columns, plants an ablation-loss ordering with the ``power_oracle``
construction, fits with the configured estimator, and records the Spearman
correlation between the fitted and the planted profiles. Groups keep their
lag structure, so K counts real SAE features.

argv: design_npz alpha_grid_json out_json [seeds]
"""
import json
import sys
from pathlib import Path

import numpy as np

DESIGN = Path(sys.argv[1])
from fmri_sae_correspondence import power as rp
from fmri_sae_correspondence import power_oracle as v2

d = rp._validate_paired_design(rp._load_design(DESIGN))
alpha_grid = [float(v) for v in json.loads(sys.argv[2])]
n_seeds = int(sys.argv[4]) if len(sys.argv) > 4 else 3
purge = int(d.get("purge_adjacent_trs", 4))
rows = []
for tag in ("brain", "lm"):
    X0 = np.asarray(d[f"{tag}_x"], float)
    G0 = [np.asarray(g, int) for g in d[f"{tag}_feature_groups"]]
    res = np.asarray(d[f"{tag}_residual"], float)
    cm = np.asarray(d[f"{tag}_censor_mask"], bool)
    folds = d[f"{tag}_folds"]
    scale = rp._mean_empirical_residual_sd(res)
    for K in (8, 16, 32, 64, 128, len(G0)):
        if K > len(G0):
            continue
        for seed in range(n_seeds):
            rng = np.random.default_rng(90000 + 7 * K + seed)
            pick = rng.choice(len(G0), size=K, replace=False)
            cols = np.concatenate([G0[i] for i in pick])
            X = X0[:, cols]
            # regroup into contiguous blocks of the same widths
            groups, at = [], 0
            for i in pick:
                w = len(G0[i]); groups.append(list(range(at, at + w))); at += w
            norms = v2.group_signal_norms(X, groups)
            act = norms > 0
            if int(act.sum()) < 3:
                continue
            p, _ = v2.make_ablation_oracle_profiles(int(act.sum()), 0.8, rng)
            loss = np.zeros(K); loss[act] = p
            signs = rng.choice(np.asarray([-1.0, 1.0]), size=K)
            beta = v2.oracle_coefficients(loss, norms, signs)
            cvec = np.zeros(X.shape[1])
            for b, g in zip(beta, groups, strict=False):
                cvec[np.asarray(g, int)] = b
            noise = rp._noise_from_residual(res, rng, len(X), res.shape[1])
            y = rp.inject_targets(X, noise, cvec, signal_scale=scale, rng=rng)
            y[cm] = np.nan
            fin = ~cm
            cache = rp._prepare_x_cache(X, folds, fin, purge_adjacent=purge, alpha_grid=alpha_grid)
            a = rp.select_alpha_by_fold(X, y, folds, alpha_grid, purge_adjacent=purge,
                                        finite_mask=fin, x_cache=cache["selection"], solver="cholesky")
            fit = rp.heldout_grouped_reliance(X, y, folds, alpha=a, feature_groups=groups,
                                              finite_mask=fin, x_cache=cache["heldout"], solver="cholesky")
            Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
            C = np.corrcoef(Xs.T); off = C[~np.eye(len(C), dtype=bool)]
            off = off[np.isfinite(off)]
            rows.append(dict(system=tag, K=int(K), seed=seed,
                             self_recovery=rp.profile_correlation(fit["reliance_profile"],
                                                                  v2.noiseless_ablation_loss(beta, norms),
                                                                  method="spearman"),
                             r2=float(np.nanmean([r["mean_r2_oos"] for r in fit["fold_scores"]])),
                             rows_per_col=float(len(X) / X.shape[1]),
                             median_abs_corr=float(np.median(np.abs(off))) if len(off) else float("nan")))
            print("  %-5s K=%3d seed=%d  self=%+.3f  r2=%.3f  rows/col=%.1f  |r|=%.3f"
                  % (tag, K, seed, rows[-1]["self_recovery"], rows[-1]["r2"],
                     rows[-1]["rows_per_col"], rows[-1]["median_abs_corr"]), flush=True)
Path(sys.argv[3]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[3]).write_text(json.dumps({"design": DESIGN.name, "rows": rows}, indent=1))
