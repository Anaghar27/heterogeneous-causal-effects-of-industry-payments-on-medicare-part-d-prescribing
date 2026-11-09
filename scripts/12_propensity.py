"""Stage 3a: cross-fitted, calibrated propensity scores, trimming and weights.

Three specifications of who is compared (pre-treatment covariates only, all from 2022):
  all       every prescriber; covariates as in Report 1
  all_hist  every prescriber; + 2022 payment history (paid_2022, log 2022 dollars)
  incident  only prescribers with no 2022 payment: newly paid in 2023 vs never paid

For each, logistic regression and gradient boosting are fit with 5-fold cross-fitting, so every
prescriber's score comes from a model that never saw them (needed later for doubly robust estimators).
Each model is also isotonic-calibrated on a slice of its training fold.

Outputs (data/processed is git-ignored; logs and figures are aggregates):
  data/processed/propensity_2023_2024.parquet   NPI, spec membership, out-of-fold scores, weights
  data/logs/propensity.json                     discrimination, calibration, overlap, trimming
  reports/figures/model/p01_calibration.png, p02_overlap_by_spec.png
"""
import importlib
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import LOGS, PROCESSED, PROJECT

eda = importlib.import_module("08_eda")  # covariate lists, design(), plot style
FIG = PROJECT / "reports" / "figures" / "model"
K_FOLDS = 5
CAL_FRACTION = 0.15  # share of each training fold held back to fit the isotonic calibrator
EPS = 1e-6
SEED = 0

HIST_NUM = ["hist_log1p_usd_2022"]
HIST_FLAGS = ["paid_2022_f"]
SPECS = {
    "all": {"rows": lambda m: np.ones(len(m), bool), "num": eda.NUM, "flags": eda.FLAGS},
    "all_hist": {"rows": lambda m: np.ones(len(m), bool), "num": eda.NUM + HIST_NUM, "flags": eda.FLAGS + HIST_FLAGS},
    "incident": {"rows": lambda m: ~m["paid_2022"].to_numpy(), "num": eda.NUM, "flags": eda.FLAGS},
}


def design(m):
    X = eda.design(m)
    X["hist_log1p_usd_2022"] = np.log1p(m["total_payment_usd_2022"].to_numpy())
    X["paid_2022_f"] = m["paid_2022"].astype(float).to_numpy()
    return X


def logistic(num):
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), eda.CAT)])
    return make_pipeline(pre, LogisticRegression(max_iter=2000, C=1.0))


def boosting():
    return HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08, max_leaf_nodes=63, l2_regularization=1.0,
                                          min_samples_leaf=200, categorical_features="from_dtype",
                                          early_stopping=True, validation_fraction=0.1, n_iter_no_change=20,
                                          random_state=SEED)


def ece(y, p, bins=20):
    """expected calibration error: mean |observed rate - predicted| over equal-count bins."""
    q = pd.qcut(p, bins, labels=False, duplicates="drop")
    g = pd.DataFrame({"y": y, "p": p, "q": q}).groupby("q").agg(y=("y", "mean"), p=("p", "mean"), n=("y", "size"))
    return float((g["n"] * (g["y"] - g["p"]).abs()).sum() / g["n"].sum())


def crossfit(X, t, num, flags):
    """out-of-fold raw and isotonic-calibrated scores for logistic and boosting."""
    cols = num + flags + eda.CAT
    Xl = X[cols]
    Xg = Xl.copy()
    for c in eda.CAT:
        Xg[c] = Xg[c].astype("category")
    out = {k: np.full(len(t), np.nan) for k in ("lr", "lr_cal", "gb", "gb_cal")}
    fold = np.full(len(t), -1)
    rng = np.random.default_rng(SEED)
    skf = StratifiedKFold(K_FOLDS, shuffle=True, random_state=SEED)
    for k, (tr, te) in enumerate(skf.split(Xl, t)):
        fold[te] = k
        cal_mask = rng.random(len(tr)) < CAL_FRACTION
        fit_i, cal_i = tr[~cal_mask], tr[cal_mask]
        for name, model, Xm in (("lr", logistic(num + flags), Xl), ("gb", boosting(), Xg)):
            t0 = time.time()
            model.fit(Xm.iloc[fit_i], t[fit_i])
            iso = IsotonicRegression(out_of_bounds="clip", y_min=EPS, y_max=1 - EPS)
            iso.fit(model.predict_proba(Xm.iloc[cal_i])[:, 1], t[cal_i])
            raw = model.predict_proba(Xm.iloc[te])[:, 1]
            out[name][te] = raw
            out[f"{name}_cal"][te] = iso.predict(raw)
            print(f"    fold {k + 1}/{K_FOLDS} {name}: {time.time() - t0:,.0f}s", flush=True)
    return {k: np.clip(v, EPS, 1 - EPS) for k, v in out.items()}, fold


def crump_alpha(p):
    """Crump et al. (2009) optimal trimming threshold for the ATE: keep alpha <= e(x) <= 1 - alpha."""
    g = 1 / (p * (1 - p))
    if g.max() <= 2 * g.mean():
        return 0.0
    for a in np.linspace(0.001, 0.2, 400):
        thr = 1 / (a * (1 - a))
        if thr <= 2 * g[g <= thr].mean():
            return float(a)
    return 0.1


def weights(t, p):
    pbar = t.mean()
    return {
        "w_ate": np.where(t, pbar / p, (1 - pbar) / (1 - p)),  # stabilized inverse probability weights
        "w_att": np.where(t, 1.0, p / (1 - p)),  # treated keep weight 1; controls reweighted to look treated
        "w_ato": np.where(t, 1 - p, p),  # overlap weights: emphasize the comparable middle, bounded in [0, 1]
    }


def ess(w):
    return float(w.sum() ** 2 / (w ** 2).sum())


def main():
    t_start = time.time()
    m = pd.read_parquet(PROCESSED / "model_2023_2024.parquet")
    X = design(m)
    T = m["treated"].to_numpy()
    store = pd.DataFrame({"NPI": m["NPI"], "treated": T})
    report, curves = {}, {}

    for spec, cfg in SPECS.items():
        rows = cfg["rows"](m)
        idx = np.flatnonzero(rows)
        t = T[idx]
        print(f"\n== {spec}: {len(idx):,} prescribers, {t.mean():.1%} paid", flush=True)
        scores, fold = crossfit(X.iloc[idx].reset_index(drop=True), t, cfg["num"], cfg["flags"])

        metrics = {}
        for name, p in scores.items():
            metrics[name] = {"auc": round(roc_auc_score(t, p), 4), "brier": round(brier_score_loss(t, p), 5),
                             "log_loss": round(log_loss(t, p), 5), "ece": round(ece(t, p), 5)}
        # primary score: best (lowest) Brier among the four; Brier rewards both discrimination and calibration
        primary = min(metrics, key=lambda k: metrics[k]["brier"])
        p = scores[primary]
        alpha = crump_alpha(p)
        keep = (p >= alpha) & (p <= 1 - alpha)
        w = weights(t, p)
        report[spec] = {
            "n": int(len(idx)), "pct_paid": round(100 * float(t.mean()), 2), "metrics": metrics, "primary": primary,
            "crump_alpha": round(alpha, 4), "pct_trimmed_crump": round(100 * float((~keep).mean()), 2),
            "pct_trimmed_crump_paid": round(100 * float((~keep & t).sum() / t.sum()), 2),
            "pct_ps_0.1_0.9": round(100 * float(((p > 0.1) & (p < 0.9)).mean()), 2),
            "pct_ps_lt_0.05": round(100 * float((p < 0.05).mean()), 2),
            "pct_ps_gt_0.95": round(100 * float((p > 0.95).mean()), 2),
            "max_w_ate": round(float(w["w_ate"].max()), 2),
            "ess_ate_pct": round(100 * ess(w["w_ate"]) / len(t), 1),
            "ess_ate_trimmed_pct": round(100 * ess(w["w_ate"][keep]) / keep.sum(), 1),
            "ess_ato_pct": round(100 * ess(w["w_ato"]) / len(t), 1),
        }
        print(json.dumps(report[spec], indent=1), flush=True)

        col = pd.Series(np.nan, index=store.index)
        store[f"in_{spec}"] = rows
        for name, s in scores.items():
            store[f"ps_{spec}_{name}"] = col.copy()
            store.loc[idx, f"ps_{spec}_{name}"] = s
        store[f"ps_{spec}"] = store[f"ps_{spec}_{primary}"]
        store[f"fold_{spec}"] = -1
        store.loc[idx, f"fold_{spec}"] = fold
        store[f"keep_{spec}"] = False
        store.loc[idx, f"keep_{spec}"] = keep
        for k, v in w.items():
            store[f"{k}_{spec}"] = col.copy()
            store.loc[idx, f"{k}_{spec}"] = v
        curves[spec] = (t, scores, p)

    store.to_parquet(PROCESSED / "propensity_2023_2024.parquet", index=False, compression="zstd")
    report["runtime_min"] = round((time.time() - t_start) / 60, 1)
    (LOGS / "propensity.json").write_text(json.dumps(report, indent=2))
    plots(curves, report)
    print(f"\ndone in {report['runtime_min']} min")


def plots(curves, report):
    FIG.mkdir(parents=True, exist_ok=True)
    # calibration curves, main spec
    t, scores, _ = curves["all"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], color=eda.AXIS, linewidth=1)
    for name, col in (("lr", eda.UNPAID), ("gb", eda.PAID), ("gb_cal", eda.AQUA)):
        q = pd.qcut(scores[name], 20, labels=False, duplicates="drop")
        g = pd.DataFrame({"y": t, "p": scores[name], "q": q}).groupby("q").mean()
        mt = report["all"]["metrics"][name]
        ax.plot(g["p"], g["y"], color=col, marker="o", markersize=5, markeredgecolor=eda.SURFACE, markeredgewidth=1.2,
                label=f"{name.replace('_cal', ' + isotonic')}  AUC {mt['auc']:.3f}, ECE {mt['ece']:.3f}")
    ax.set_xlabel("predicted probability of being paid (out-of-fold)")
    ax.set_ylabel("observed share paid")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Calibration of the propensity models", pad=18)
    eda.subtitle(ax, "main specification; 20 equal-size bins; grey line = perfect calibration")
    fig.savefig(FIG / "p01_calibration.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # overlap for each spec (primary score)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    b = np.linspace(0, 1, 51)
    for ax, spec in zip(axes, ("all", "all_hist", "incident")):
        t, _, p = curves[spec]
        for flag, col, lab in ((False, eda.UNPAID, "unpaid"), (True, eda.PAID, "paid")):
            ax.hist(p[t == flag], bins=b, histtype="step", color=col, linewidth=2, density=True, label=lab)
        r = report[spec]
        ax.axvspan(0, 0.1, color=eda.GRID, alpha=0.6, linewidth=0)
        ax.axvspan(0.9, 1, color=eda.GRID, alpha=0.6, linewidth=0)
        ax.set_title(f"{spec}: AUC {r['metrics'][r['primary']]['auc']:.3f}, {r['pct_ps_0.1_0.9']}% in 0.1-0.9",
                     fontsize=10)
        ax.set_xlabel("propensity score")
    axes[0].set_ylabel("density")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "p02_overlap_by_spec.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
