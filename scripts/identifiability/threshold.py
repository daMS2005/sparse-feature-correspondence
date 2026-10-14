#!/usr/bin/env python3
"""Self-recovery of a planted ablation-loss ordering on real predictors as K varies.

For each transfer design and each K from 24 to 128, a random subset of K real
feature groups is kept at full sample size, an ablation-loss ordering is
planted with the ``power_oracle`` construction, the configured estimator fits
and ablates, and the Spearman correlation between the fitted reliance profile
and the planted profile is recorded for each system. Writes JSONL
incrementally, one line per cell.

argv: designs_dir alpha_grid_json out_jsonl shard_index shard_count [seeds]
"""
import itertools
import json
import sys
from pathlib import Path

import numpy as np

DES = Path(sys.argv[1])
from fmri_sae_correspondence import power as rp
from fmri_sae_correspondence import power_oracle as v2

alpha_grid = [float(v) for v in json.loads(sys.argv[2])]
out = Path(sys.argv[3]); shard_i = int(sys.argv[4]); shard_n = int(sys.argv[5])
n_seeds = int(sys.argv[6]) if len(sys.argv) > 6 else 3

DESIGNS = sorted(q.stem for q in DES.glob("*.npz"))
KS = (24, 32, 40, 48, 56, 64, 80, 96, 128)
FRACS = (1.0,)
ARMS = ("real",)
cells = [c for c in itertools.product(DESIGNS, KS, FRACS, ARMS, range(n_seeds))]
mine = [c for i, c in enumerate(cells) if i % shard_n == shard_i]
print(f"shard {shard_i}/{shard_n}: {len(mine)} of {len(cells)} cells", flush=True)

out.parent.mkdir(parents=True, exist_ok=True)
done = set()
if out.exists():
    for ln in out.read_text().splitlines():
        try: r = json.loads(ln); done.add((r["design"], r["K"], r["frac"], r["arm"], r["seed"], r["system"]))
        except Exception: pass

cache_design = {}
fh = out.open("a")
for design, K, frac, arm, seed in mine:
    if design not in cache_design:
        cache_design.clear()
        cache_design[design] = rp._validate_paired_design(rp._load_design(DES / f"{design}.npz"))
    d = cache_design[design]
    purge = int(d.get("purge_adjacent_trs", 4))
    for tag in ("brain", "lm"):
        key = (design, K, frac, arm, seed, tag)
        if key in done: continue
        X0 = np.asarray(d[f"{tag}_x"], float)
        G0 = [np.asarray(g, int) for g in d[f"{tag}_feature_groups"]]
        if K > len(G0): continue
        res = np.asarray(d[f"{tag}_residual"], float); cm = np.asarray(d[f"{tag}_censor_mask"], bool)
        rng = np.random.default_rng(500000 + 13 * K + 7 * seed + int(frac * 100) + (1 if arm == "iid" else 0))
        pick = rng.choice(len(G0), size=K, replace=False)
        cols = np.concatenate([G0[i] for i in pick])
        Xr = X0[:, cols]
        if arm == "iid": Xr = rng.standard_normal(Xr.shape)
        groups, at = [], 0
        for i in pick:
            w = len(G0[i]); groups.append(list(range(at, at + w))); at += w
        n_keep = max(int(round(len(Xr) * frac)), 50)
        Xr = Xr[:n_keep]; cmk = cm[:n_keep]
        folds = [(np.setdiff1d(np.arange(n_keep), np.arange(a, b)), np.arange(a, b))
                 for a, b in zip(np.linspace(0, n_keep, 6).astype(int)[:-1], np.linspace(0, n_keep, 6).astype(int)[1:], strict=False)]
        norms = v2.group_signal_norms(Xr, groups); act = norms > 0
        if int(act.sum()) < 3: continue
        p, _ = v2.make_ablation_oracle_profiles(int(act.sum()), 0.8, rng)
        loss = np.zeros(K); loss[act] = p
        beta = v2.oracle_coefficients(loss, norms, rng.choice(np.asarray([-1.0, 1.0]), size=K))
        cvec = np.zeros(Xr.shape[1])
        for b, g in zip(beta, groups, strict=False): cvec[np.asarray(g, int)] = b
        y = rp.inject_targets(Xr, rp._noise_from_residual(res, rng, len(Xr), res.shape[1]), cvec,
                              signal_scale=rp._mean_empirical_residual_sd(res), rng=rng)
        y[cmk] = np.nan; fin = ~cmk
        try:
            cache = rp._prepare_x_cache(Xr, folds, fin, purge_adjacent=purge, alpha_grid=alpha_grid)
            a = rp.select_alpha_by_fold(Xr, y, folds, alpha_grid, purge_adjacent=purge, finite_mask=fin,
                                        x_cache=cache["selection"], solver="cholesky")
            fit = rp.heldout_grouped_reliance(Xr, y, folds, alpha=a, feature_groups=groups, finite_mask=fin,
                                              x_cache=cache["heldout"], solver="cholesky")
            self_rec = rp.profile_correlation(fit["reliance_profile"], v2.noiseless_ablation_loss(beta, norms), method="spearman")
            r2 = float(np.nanmean([r["mean_r2_oos"] for r in fit["fold_scores"]]))
        except Exception:
            self_rec, r2 = float("nan"), float("nan")
        Xs = (Xr - Xr.mean(0)) / (Xr.std(0) + 1e-12)
        C = np.corrcoef(Xs.T); off = C[~np.eye(len(C), dtype=bool)]; off = off[np.isfinite(off)]
        rec = dict(design=design, K=K, frac=frac, arm=arm, seed=seed, system=tag,
                   self_recovery=self_rec, r2=r2, n_rows=int(len(Xr)), n_cols=int(Xr.shape[1]),
                   rows_per_col=float(len(Xr) / Xr.shape[1]),
                   median_abs_corr=float(np.median(np.abs(off))) if len(off) else float("nan"))
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        print("  %-4s %-5s K=%3d f=%.2f s=%d self=%+.3f rows/col=%.1f |r|=%.3f"
              % (arm, tag, K, frac, seed, self_rec, rec["rows_per_col"], rec["median_abs_corr"]), flush=True)
fh.close()
print("SHARD_DONE", flush=True)
