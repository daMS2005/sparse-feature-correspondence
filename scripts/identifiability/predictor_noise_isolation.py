#!/usr/bin/env python3
"""Two-by-two isolation of predictors and noise on one transfer design.

Runs the planted-ordering recovery with real predictors and empirical residual
noise (the actual design), real predictors and white noise, i.i.d. Gaussian
predictors and empirical noise, and i.i.d. predictors and white noise, with the
configured estimator unchanged throughout, so that the source of any loss of
recovery can be attributed to the predictors or to the noise.

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
purge = int(d.get("purge_adjacent_trs", 4))
rows = []
for use_real_x in (True, False):
    for use_real_noise in (True, False):
        for seed in range(int(sys.argv[4]) if len(sys.argv) > 4 else 2):
            rng = np.random.default_rng(7000 + seed)
            out = {}
            for tag in ("brain", "lm"):
                X0 = np.asarray(d[f"{tag}_x"], float)
                X = X0 if use_real_x else rng.standard_normal(X0.shape)
                groups = d[f"{tag}_feature_groups"]
                res = np.asarray(d[f"{tag}_residual"], float)
                norms = v2.group_signal_norms(X, groups)
                act = norms > 0
                p, _q = v2.make_ablation_oracle_profiles(int(act.sum()), 0.8, rng)
                loss = np.zeros(len(groups)); loss[act] = p
                signs = rng.choice(np.asarray([-1.0, 1.0]), size=len(groups))
                beta = v2.oracle_coefficients(loss, norms, signs)
                cols = np.zeros(X.shape[1]) 
                for b, g in zip(beta, groups, strict=False): cols[np.asarray(g, int)] = b
                n_t = res.shape[1]
                noise = (rp._noise_from_residual(res, rng, len(X), n_t) if use_real_noise
                         else rng.standard_normal((len(X), n_t)) * float(np.nanstd(res)))
                y = rp.inject_targets(X, noise, cols, signal_scale=rp._mean_empirical_residual_sd(res), rng=rng)
                cm = np.asarray(d[f"{tag}_censor_mask"], bool)
                y[cm] = np.nan
                fin = ~cm
                cache = rp._prepare_x_cache(X, d[f"{tag}_folds"], fin, purge_adjacent=purge, alpha_grid=alpha_grid)
                a = rp.select_alpha_by_fold(X, y, d[f"{tag}_folds"], alpha_grid, purge_adjacent=purge,
                                            finite_mask=fin, x_cache=cache["selection"], solver="cholesky")
                fit = rp.heldout_grouped_reliance(X, y, d[f"{tag}_folds"], alpha=a, feature_groups=groups,
                                                  finite_mask=fin, x_cache=cache["heldout"], solver="cholesky")
                out[tag] = (rp.profile_correlation(fit["reliance_profile"],
                                                   v2.noiseless_ablation_loss(beta, norms), method="spearman"),
                            float(np.nanmean([r["mean_r2_oos"] for r in fit["fold_scores"]])))
            rows.append(dict(real_X=use_real_x, real_noise=use_real_noise, seed=seed,
                             self_brain=out["brain"][0], self_lm=out["lm"][0],
                             r2_brain=out["brain"][1], r2_lm=out["lm"][1]))
            print("  X=%-9s noise=%-9s seed=%d  self_brain=%+.3f self_lm=%+.3f  r2=%.3f/%.3f"
                  % ("real" if use_real_x else "iid", "real" if use_real_noise else "white", seed,
                     rows[-1]["self_brain"], rows[-1]["self_lm"], rows[-1]["r2_brain"], rows[-1]["r2_lm"]), flush=True)
Path(sys.argv[3]).parent.mkdir(parents=True, exist_ok=True)
Path(sys.argv[3]).write_text(json.dumps({"design": DESIGN.name, "rows": rows}, indent=1))
