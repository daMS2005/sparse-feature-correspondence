#!/usr/bin/env python3
"""Positive control for the fitting-and-ablation estimator on well-conditioned synthetic data.

Walks the number of feature groups, rows per system, target dimensions, and
injected signal relative to noise with i.i.d. Gaussian predictors and noise,
plants an ablation-loss correspondence of .8 with signs drawn independently
per system, and reports per cell the Spearman correlation between each fitted
reliance profile and its own planted profile, the correlation between the two
fitted profiles, and the held-out R2. Everything numerical is the configured
estimator; only the data are synthetic.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def contiguous_folds(n_rows: int, k: int):
    edges = np.linspace(0, n_rows, k + 1).astype(int)
    out = []
    for i in range(k):
        test = np.arange(edges[i], edges[i + 1])
        train = np.setdiff1d(np.arange(n_rows), test, assume_unique=True)
        out.append((train, test))
    return out


def run_cell(rp, v2, *, n_groups, n_rows, n_targets, signal_scale, rho, lags, alpha_grid, seed, folds_k=5):
    rng = np.random.default_rng(seed)
    cols = n_groups * lags
    X = rng.standard_normal((n_rows, cols))
    groups = [list(range(g * lags, (g + 1) * lags)) for g in range(n_groups)]
    norms = v2.group_signal_norms(X, groups)
    active = norms > 0
    p, q = v2.make_ablation_oracle_profiles(int(active.sum()), float(rho), rng)
    loss_b = np.zeros(n_groups); loss_l = np.zeros(n_groups)
    loss_b[active] = p; loss_l[active] = q
    sb = rng.choice(np.asarray([-1.0, 1.0]), size=n_groups)
    sl = rng.choice(np.asarray([-1.0, 1.0]), size=n_groups)
    beta_b = v2.oracle_coefficients(loss_b, norms, sb)
    beta_l = v2.oracle_coefficients(loss_l, norms, sl)
    cb = np.zeros(cols); cl = np.zeros(cols)
    for v_b, v_l, g in zip(beta_b, beta_l, groups, strict=False):
        cb[g] = v_b; cl[g] = v_l
    noise_b = rng.standard_normal((n_rows, n_targets))
    noise_l = rng.standard_normal((n_rows, n_targets))
    yb = rp.inject_targets(X, noise_b, cb, signal_scale=float(signal_scale), rng=rng)
    yl = rp.inject_targets(X, noise_l, cl, signal_scale=float(signal_scale), rng=rng)
    folds = contiguous_folds(n_rows, folds_k)
    finite = np.ones(n_rows, dtype=bool)
    cache = rp._prepare_x_cache(X, folds, finite, purge_adjacent=0, alpha_grid=alpha_grid)
    out = {}
    for tag, y in (("brain", yb), ("lm", yl)):
        a = rp.select_alpha_by_fold(X, y, folds, alpha_grid, purge_adjacent=0,
                                    finite_mask=finite, x_cache=cache["selection"], solver="cholesky")
        fit = rp.heldout_grouped_reliance(X, y, folds, alpha=a, feature_groups=groups,
                                          finite_mask=finite, x_cache=cache["heldout"], solver="cholesky")
        out[tag] = (np.asarray(fit["reliance_profile"], dtype=float),
                    float(np.nanmean([r["mean_r2_oos"] for r in fit["fold_scores"]])),
                    [float(v) for v in a])
    nl_b = v2.noiseless_ablation_loss(beta_b, norms)
    nl_l = v2.noiseless_ablation_loss(beta_l, norms)
    return {
        "n_groups": n_groups, "n_rows": n_rows, "n_targets": n_targets, "lags": lags,
        "signal_scale": signal_scale, "rho": rho, "seed": seed,
        "rows_per_column": round(n_rows / cols, 3),
        "self_brain": rp.profile_correlation(out["brain"][0], nl_b, method="spearman"),
        "self_lm": rp.profile_correlation(out["lm"][0], nl_l, method="spearman"),
        "recovered": rp.profile_correlation(out["brain"][0], out["lm"][0], method="spearman"),
        "planted_oracle": rp.profile_correlation(nl_b, nl_l, method="spearman"),
        "brain_r2": out["brain"][1], "lm_r2": out["lm"][1],
        "brain_alphas": out["brain"][2], "lm_alphas": out["lm"][2],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--shard-count", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args(argv)
    from fmri_sae_correspondence import power as rp
    from fmri_sae_correspondence import power_oracle as v2

    alpha_grid = [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]
    cells = []
    for n_groups in (16, 32, 64, 128, 256):
        for n_rows in (1000, 4000, 16000):
            for n_targets in (1, 20, 200):
                for signal_scale in (0.5, 1.0, 2.0, 4.0, 8.0):
                    for seed in range(a.seeds):
                        cells.append(dict(n_groups=n_groups, n_rows=n_rows, n_targets=n_targets,
                                          signal_scale=signal_scale, rho=0.8, lags=5,
                                          alpha_grid=alpha_grid, seed=1000 + seed))
    mine = [c for i, c in enumerate(cells) if i % a.shard_count == a.shard_index]
    print(f"shard {a.shard_index}/{a.shard_count}: {len(mine)} of {len(cells)} cells", flush=True)
    rows = []
    for i, c in enumerate(mine):
        try:
            rows.append(run_cell(rp, v2, **c))
        except Exception as exc:  # a cell that cannot be fit is itself a result
            rows.append({k: v for k, v in c.items() if k != "alpha_grid"} | {"error": repr(exc)[:200]})
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(mine)}", flush=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({"cells": rows, "n_cells": len(rows)}, indent=1), encoding="utf-8")
    print(f"wrote {a.out} ({len(rows)} cells)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
