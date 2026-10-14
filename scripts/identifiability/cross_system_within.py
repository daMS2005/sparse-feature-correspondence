#!/usr/bin/env python3
"""Recovered cross-system correlation at a planted rho = .8 by bank size, within description.

The same procedure as ``cross_system_transfer.py`` on the four within-description
designs (Gemma 2 2B and Llama 3.1 8B on each description), so that the
identifiability limit is measured in both evaluation regimes rather than
inferred from one.

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
DESIGNS = sorted(q.stem for q in DES.glob("*within*.npz"))
KS = (24, 32, 48, 64, 96, 128, 192, 256)
cells = [c for c in itertools.product(DESIGNS, KS, range(n_seeds))]
mine = [c for i, c in enumerate(cells) if i % shard_n == shard_i]
print(f"shard {shard_i}/{shard_n}: {len(mine)} of {len(cells)} cells", flush=True)
out.parent.mkdir(parents=True, exist_ok=True)
done = set()
if out.exists():
    for ln in out.read_text().splitlines():
        try: r = json.loads(ln); done.add((r["design"], r["K"], r["seed"]))
        except Exception: pass
fh = out.open("a"); cur = {}
for design, K, seed in mine:
    if (design, K, seed) in done: continue
    if design not in cur:
        cur.clear(); cur[design] = rp._validate_paired_design(rp._load_design(DES / f"{design}.npz"))
    d = cur[design]; purge = int(d.get("purge_adjacent_trs", 4))
    rng = np.random.default_rng(770000 + 31 * K + seed)
    nG = len(d["brain_feature_groups"])
    if K > nG: continue
    pick = rng.choice(nG, size=K, replace=False)
    fits, oracles = {}, {}
    for tag in ("brain", "lm"):
        G0 = [np.asarray(g, int) for g in d[f"{tag}_feature_groups"]]
        X = np.asarray(d[f"{tag}_x"], float)[:, np.concatenate([G0[i] for i in pick])]
        groups, at = [], 0
        for i in pick:
            w = len(G0[i]); groups.append(list(range(at, at + w))); at += w
        fits[tag] = (X, groups, np.asarray(d[f"{tag}_residual"], float),
                     np.asarray(d[f"{tag}_censor_mask"], bool), d[f"{tag}_folds"],
                     v2.group_signal_norms(X, groups))
    act = (fits["brain"][5] > 0) & (fits["lm"][5] > 0)
    if int(act.sum()) < 3: continue
    p, q = v2.make_ablation_oracle_profiles(int(act.sum()), 0.8, rng)
    prof = {}
    for tag, target in (("brain", p), ("lm", q)):
        X, groups, res, cm, folds, norms = fits[tag]
        loss = np.zeros(K); loss[act] = target
        beta = v2.oracle_coefficients(loss, norms, rng.choice(np.asarray([-1.0, 1.0]), size=K))
        oracles[tag] = v2.noiseless_ablation_loss(beta, norms)
        cvec = np.zeros(X.shape[1])
        for b, g in zip(beta, groups, strict=False): cvec[np.asarray(g, int)] = b
        y = rp.inject_targets(X, rp._noise_from_residual(res, rng, len(X), res.shape[1]), cvec,
                              signal_scale=rp._mean_empirical_residual_sd(res), rng=rng)
        y[cm] = np.nan; fin = ~cm
        cache = rp._prepare_x_cache(X, folds, fin, purge_adjacent=purge, alpha_grid=alpha_grid)
        a = rp.select_alpha_by_fold(X, y, folds, alpha_grid, purge_adjacent=purge, finite_mask=fin,
                                    x_cache=cache["selection"], solver="cholesky")
        f = rp.heldout_grouped_reliance(X, y, folds, alpha=a, feature_groups=groups, finite_mask=fin,
                                        x_cache=cache["heldout"], solver="cholesky")
        prof[tag] = np.asarray(f["reliance_profile"], float)
    rec = dict(design=design, K=int(K), seed=seed,
               planted_oracle=rp.profile_correlation(oracles["brain"], oracles["lm"], method="spearman"),
               recovered=rp.profile_correlation(prof["brain"], prof["lm"], method="spearman"),
               self_brain=rp.profile_correlation(prof["brain"], oracles["brain"], method="spearman"),
               self_lm=rp.profile_correlation(prof["lm"], oracles["lm"], method="spearman"))
    fh.write(json.dumps(rec) + "\n"); fh.flush()
    print("  K=%3d s=%d planted=%+.3f recovered=%+.3f (self %.2f/%.2f)"
          % (K, seed, rec["planted_oracle"], rec["recovered"], rec["self_brain"], rec["self_lm"]), flush=True)
fh.close(); print("SHARD_DONE", flush=True)
