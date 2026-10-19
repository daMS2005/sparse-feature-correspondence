"""Shifted-target and noise controls for the thesis feature-importance correlation.

Uses the thesis pipeline functions unchanged; only the LM or brain target is replaced
before fitting. Penalties are reselected by the same nested CV for every replicate.

usage: run_controls.py RUN_DIR BRAIN_TARGETS_NPZ CONDITION REP_START REP_STOP OUT_DIR
CONDITION: original | lm_shift | brain_shift | white_noise | phase
"""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from structure_comparison._implementation import (
    build_brain_design_matrix,
    collapse_lagged_weights,
    compare_feature_importance,
    consensus_alpha,
    contiguous_block_groups,
    fit_final_model,
    load_brain_targets,
    regression_metrics,
    run_grouped_ridge_cv,
)

ALPHA_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]
LAGS = [0, 1, 2, 3, 4]
LM_FOLDS = 5
SEED = 20261015
N_REP = 100


def fit(x, y, groups, sample_ids, target_names, name):
    cv = run_grouped_ridge_cv(x=x, y=y, groups=groups, sample_ids=sample_ids,
                              target_names=target_names, alpha_grid=ALPHA_GRID, family_name=name)
    alpha = consensus_alpha(cv["outer_folds"])
    return alpha, fit_final_model(x, y, alpha, name)["weights"]


def fast_brain_alpha(x, y, groups):
    """run_grouped_ridge_cv + consensus_alpha for the brain, from per-run sufficient statistics.

    Same nested leave-one-run-out search, standardization and score as the thesis code;
    the training-set z-scored Gram matrix is assembled from per-run sums instead of raw rows.
    """
    labels = np.unique(groups)
    rows = {g: np.where(groups == g)[0] for g in labels}
    stats = {}
    for g, i in rows.items():
        stats[g] = np.empty(7, dtype=object)
        stats[g][:] = [len(i), x[i].sum(0), (x[i] ** 2).sum(0), x[i].T @ x[i],
                       y[i].sum(0), (y[i] ** 2).sum(0), x[i].T @ y[i]]
    total = sum(stats.values())

    def best_alpha(train_stats, held_out):
        scores = np.zeros(len(ALPHA_GRID))
        for v in held_out:
            n, sx, sxx, xx, sy, syy, xy = train_stats - stats[v]
            mx, my = sx / n, sy / n
            sdx = np.sqrt(np.maximum(sxx / n - mx ** 2, 0.0))
            sdy = np.sqrt(np.maximum(syy / n - my ** 2, 0.0))
            sdx[sdx == 0] = 1.0
            sdy[sdy == 0] = 1.0
            gram = (xx - n * np.outer(mx, mx)) / np.outer(sdx, sdx)
            rhs = (xy - n * np.outer(mx, my)) / np.outer(sdx, sdy)
            evals, evecs = np.linalg.eigh(gram)
            proj = evecs.T @ rhs
            xv = (x[rows[v]] - mx) / sdx
            for a, alpha in enumerate(ALPHA_GRID):
                w = evecs @ (proj / (evals + alpha)[:, None])
                pred = xv @ w * sdy + my
                scores[a] += regression_metrics(y[rows[v]], pred)["mean_correlation"]
        return ALPHA_GRID[int(np.argmax(scores))]

    outer = [{"selected_alpha": best_alpha(total - stats[g], [h for h in labels if h != g])} for g in labels]
    return consensus_alpha(outer), [f["selected_alpha"] for f in outer]


def shift_brain(targets, offset):
    """np.roll each subject-run time series by the same offset."""
    values = targets.values.copy()
    runs = np.char.add(np.char.add(targets.subject_ids, ":"), targets.run_ids)
    for run in np.unique(runs):
        rows = np.where(runs == run)[0]
        rows = rows[np.argsort(targets.tr_indices[rows], kind="stable")]
        values[rows] = np.roll(values[rows], offset, axis=0)
    return replace(targets, values=values)


def phase_randomize(y, rng):
    n = y.shape[0]
    spectrum = np.fft.rfft(y, axis=0)
    phases = rng.uniform(0, 2 * np.pi, spectrum.shape[0])
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0
    return np.fft.irfft(spectrum * np.exp(1j * phases)[:, None], n=n, axis=0)


def main():
    run_dir, brain_npz, condition, start, stop, out_dir = sys.argv[1:7]
    run_dir, out_dir = Path(run_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    feats = np.load(run_dir / "intermediates/tr_feature_average.npz", allow_pickle=True)
    hidden = np.load(run_dir / "intermediates/tr_lm_target_final_hidden_average.npz", allow_pickle=True)
    bm = np.load(run_dir / "all_layers/brain_final_model.npz", allow_pickle=True)
    lm = np.load(run_dir / "all_layers/lm_final_model.npz", allow_pickle=True)
    reported = json.loads((run_dir / "all_layers/feature_importance_summary.json").read_text())["importance_correlation"]

    x = np.asarray(feats["values"], float)
    tr = np.asarray(feats["tr_indices"], int)
    sample_ids = feats["sample_ids"].astype(str)
    names = bm["base_feature_names"].astype(str)
    y_lm = np.asarray(hidden["values"], float)
    assert (hidden["sample_ids"].astype(str) == sample_ids).all()
    lm_names = hidden["target_names"].astype(str)
    lm_groups = contiguous_block_groups(len(tr), LM_FOLDS)
    n_tr = len(tr)
    offsets = np.random.default_rng(SEED).integers(10, n_tr - 10, size=N_REP, endpoint=True)

    brain_w = np.asarray(bm["collapsed_weights"], float)
    lm_w = np.asarray(lm["weights"], float)
    brain = load_brain_targets(Path(brain_npz)) if condition in ("original", "brain_shift") else None

    def fit_lm(y):
        return fit(x, y, lm_groups, sample_ids, lm_names, "all_layers:lm")

    def fit_brain(targets):
        design = build_brain_design_matrix(tr_predictors=x, tr_indices=tr, feature_names=names,
                                           brain_targets=targets, lags=LAGS)
        assert (design["feature_names"] == bm["feature_names"].astype(str)).all()
        alpha, fold_alphas = fast_brain_alpha(design["x"], design["y"], design["groups"])
        w = fit_final_model(design["x"], design["y"], alpha, "all_layers:brain")["weights"]
        return alpha, collapse_lagged_weights(w, base_feature_count=x.shape[1], lags=LAGS), fold_alphas

    def corr(bw, lw):
        return compare_feature_importance(bw, lw, names)["importance_correlation"]

    if condition == "original":
        a_l, w_l = fit_lm(y_lm)
        a_b, w_b, folds_b = fit_brain(brain)
        saved_folds = [f["selected_alpha"] for f in json.loads((run_dir / "all_layers/brain_cv_summary.json").read_text())["outer_folds"]]
        row = {"condition": condition, "brain_fold_alphas_match_saved": folds_b == saved_folds, "reported": reported, "reproduced": corr(w_b, w_l),
               "lm_alpha": a_l, "brain_alpha": a_b,
               "lm_alpha_saved": float(lm["alpha"][0]), "brain_alpha_saved": float(bm["alpha"][0]),
               "lm_weights_max_abs_diff": float(np.abs(w_l - lm_w).max()),
               "brain_weights_max_abs_diff": float(np.abs(w_b - brain_w).max())}
        (out_dir / "original.json").write_text(json.dumps(row, indent=1))
        return

    for rep in range(int(start), int(stop)):
        row = {"condition": condition, "rep": rep}
        if condition == "lm_shift":
            row["offset"] = int(offsets[rep])
            alpha, w = fit_lm(np.roll(y_lm, row["offset"], axis=0))
            row.update(selected_alpha=alpha, importance_correlation=corr(brain_w, w))
        elif condition == "brain_shift":
            row["offset"] = int(offsets[rep])
            alpha, w, _ = fit_brain(shift_brain(brain, row["offset"]))
            row.update(selected_alpha=alpha, importance_correlation=corr(w, lm_w))
        elif condition == "white_noise":
            row["seed"] = [SEED, 1, rep]
            alpha, w = fit_lm(np.random.default_rng(row["seed"]).standard_normal(y_lm.shape))
            row.update(selected_alpha=alpha, importance_correlation=corr(brain_w, w))
        elif condition == "phase":
            row["seed"] = [SEED, 2, rep]
            alpha, w = fit_lm(phase_randomize(y_lm, np.random.default_rng(row["seed"])))
            row.update(selected_alpha=alpha, importance_correlation=corr(brain_w, w))
        else:
            raise ValueError(condition)
        (out_dir / f"{condition}_{rep:03d}.json").write_text(json.dumps(row))


if __name__ == "__main__":
    main()
