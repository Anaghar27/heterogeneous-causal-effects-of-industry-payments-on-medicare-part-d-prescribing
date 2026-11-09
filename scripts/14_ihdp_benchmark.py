"""Stage 6a: validate the estimators on IHDP, where the true effect for every unit is known.

IHDP (Hill 2011), 10 semi-synthetic replications from the CEVAE repository, 747 units each.
Columns: treatment, y_factual, y_cfactual, mu0, mu1, x1..x25. True individual effect = mu1 - mu0.

Estimators (the ones planned for the CMS data):
  naive           difference in observed means
  IPW             inverse probability weighting with a logistic propensity model (clipped 0.01-0.99)
  DR-learner      econml DRLearner: boosting nuisance models, boosting final CATE model
  causal forest   econml CausalForestDML

Metrics, averaged over replications (each split 90% train / 10% test, as in the literature):
  ATE error   |estimated ATE - true ATE|
  sqrt PEHE   root mean squared error of the individual effects (in-sample and out-of-sample)

Raw data: data/external/ihdp/ (git-ignored). Outputs: data/logs/ihdp_benchmark.json,
reports/figures/model/b01_ihdp_benchmark.png, and a done marker for detached runs.
"""
import importlib
import json
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from econml.dml import CausalForestDML
from econml.dr import DRLearner
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from config import LOGS, PROJECT

warnings.filterwarnings("ignore")
eda = importlib.import_module("08_eda")
RAW = PROJECT / "data" / "external" / "ihdp"
FIG = PROJECT / "reports" / "figures" / "model"
DONE = LOGS / "ihdp_benchmark.done"
COLS = ["t", "y", "y_cf", "mu0", "mu1"] + [f"x{i}" for i in range(1, 26)]
N_REP = 10


def gbr():
    return GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)


def gbc():
    return GradientBoostingClassifier(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)


def ipw_ate(X, t, y):
    p = np.clip(LogisticRegression(max_iter=2000).fit(X, t).predict_proba(X)[:, 1], 0.01, 0.99)
    w1, w0 = t / p, (1 - t) / (1 - p)
    return np.sum(w1 * y) / w1.sum() - np.sum(w0 * y) / w0.sum()


def run_rep(i):
    d = pd.read_csv(RAW / f"ihdp_npci_{i}.csv", header=None, names=COLS)
    X = d[[f"x{k}" for k in range(1, 26)]].to_numpy()
    t, y = d["t"].to_numpy().astype(int), d["y"].to_numpy()
    ite = (d["mu1"] - d["mu0"]).to_numpy()
    tr, te = train_test_split(np.arange(len(d)), test_size=0.1, random_state=i)
    true_ate = ite.mean()
    rows = []

    def add(name, ate, cate_tr=None, cate_te=None):
        rows.append({"rep": i, "estimator": name, "ate_true": true_ate, "ate_est": ate,
                     "ate_error": abs(ate - true_ate),
                     "pehe_in": np.sqrt(np.mean((cate_tr - ite[tr]) ** 2)) if cate_tr is not None else np.nan,
                     "pehe_out": np.sqrt(np.mean((cate_te - ite[te]) ** 2)) if cate_te is not None else np.nan})

    naive = y[tr][t[tr] == 1].mean() - y[tr][t[tr] == 0].mean()
    add("naive", naive, np.full(len(tr), naive), np.full(len(te), naive))
    ipw = ipw_ate(X[tr], t[tr], y[tr])
    add("IPW", ipw, np.full(len(tr), ipw), np.full(len(te), ipw))

    # logistic propensity + clipping: with ~140 treated units a boosting propensity model produces
    # near-0/1 scores, and the DR pseudo-outcome divides by them (first run: ATE error 0.95)
    dr = DRLearner(model_propensity=LogisticRegression(max_iter=2000), model_regression=gbr(),
                   model_final=GradientBoostingRegressor(n_estimators=100, max_depth=2, learning_rate=0.05,
                                                         min_samples_leaf=20, random_state=0),
                   min_propensity=0.05, cv=3, random_state=0)
    dr.fit(y[tr], t[tr], X=X[tr])
    c_tr, c_te = dr.effect(X[tr]), dr.effect(X[te])
    add("DR-learner", float(dr.ate(X[tr])), c_tr, c_te)

    cf = CausalForestDML(model_y=gbr(), model_t=gbc(), discrete_treatment=True, n_estimators=1000,
                         min_samples_leaf=5, cv=3, random_state=0)
    cf.fit(y[tr], t[tr], X=X[tr])
    c_tr, c_te = cf.effect(X[tr]), cf.effect(X[te])
    add("causal forest", float(cf.ate(X[tr])), c_tr, c_te)
    return rows


def plot(summ):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, col, se, lab in ((a1, "ate_error", "ate_error_se", "ATE error (lower is better)"),
                             (a2, "pehe_out", "pehe_out_se", "out-of-sample sqrt PEHE (lower is better)")):
        v, e = summ[col].to_numpy()[::-1], summ[se].to_numpy()[::-1]
        ax.barh(summ.index[::-1], v, xerr=e, color=eda.PAID, height=0.55, error_kw={"ecolor": eda.INK2, "capsize": 3})
        for k in range(len(v)):  # label past the end of the error bar so it never overlaps
            ax.text(v[k] + e[k], k, f"  {v[k]:.2f}", va="center", color=eda.INK2, fontsize=9)
        ax.set_xlim(0, (v + e).max() * 1.25)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel(lab)
    a1.set_title("IHDP benchmark: error vs the known true effect (10 replications, mean +/- SE)", loc="left",
                 fontsize=11, fontweight="semibold")
    fig.tight_layout()
    fig.savefig(FIG / "b01_ihdp_benchmark.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    DONE.unlink(missing_ok=True)
    t0 = time.time()
    rows = []
    for i in range(1, N_REP + 1):
        rows += run_rep(i)
        print(f"replication {i}/{N_REP} done ({time.time() - t0:,.0f}s)", flush=True)
    r = pd.DataFrame(rows)
    order = ["naive", "IPW", "DR-learner", "causal forest"]
    summ = r.groupby("estimator").agg(ate_error=("ate_error", "mean"), ate_error_se=("ate_error", "sem"),
                                      pehe_in=("pehe_in", "mean"), pehe_out=("pehe_out", "mean"),
                                      pehe_out_se=("pehe_out", "sem")).loc[order]
    print(summ.round(3).to_string())

    plot(summ)

    out = {"replications": N_REP, "units": 747, "split": "90/10", "summary": summ.round(4).reset_index().to_dict(orient="records"),
           "per_rep": r.round(4).to_dict(orient="records"), "runtime_s": round(time.time() - t0)}
    (LOGS / "ihdp_benchmark.json").write_text(json.dumps(out, indent=2))
    DONE.write_text("ok")


if __name__ == "__main__":
    if "--plot-only" in sys.argv:  # redraw from the saved summary without refitting
        plot(pd.DataFrame(json.loads((LOGS / "ihdp_benchmark.json").read_text())["summary"]).set_index("estimator"))
        sys.exit(0)
    try:
        main()
    except Exception as e:  # noqa: BLE001
        DONE.write_text(f"failed: {e!r}")
        raise
    sys.exit(0)
