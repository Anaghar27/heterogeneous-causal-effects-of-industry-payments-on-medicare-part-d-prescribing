"""Stage 3b: covariate balance after weighting, overlap by specialty, and Layer A explainability.

Reads the scores and weights from 12_propensity.py. Success criterion (Report 1, section 2.7):
every covariate |SMD| < 0.1 after weighting.

Layer A (diagnostic only): SHAP on the propensity model says what drives WHO GETS PAID, i.e. the
shape of the confounding. It says nothing about who responds to payment (that is Layer B, stage 5).

Outputs: data/logs/balance.json; reports/figures/model/p03_love_plot.png, p04_overlap_by_specialty.png,
p05_propensity_shap.png
"""
import importlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from config import LOGS, PROCESSED, PROJECT

eda = importlib.import_module("08_eda")
prop = importlib.import_module("12_propensity")
FIG = PROJECT / "reports" / "figures" / "model"
SPECS = ("all", "all_hist", "incident")
SCHEMES = {"unweighted": None, "IPW (trimmed)": "w_ate", "ATT": "w_att", "overlap": "w_ato"}
SHAP_N = 50_000


def wsmd(x, t, w):
    """weighted standardized mean difference; pooled SD from the unweighted groups (fixed denominator)."""
    m1, m0 = np.average(x[t], weights=w[t]), np.average(x[~t], weights=w[~t])
    sd = np.sqrt((x[t].var() + x[~t].var()) / 2)
    return 0.0 if sd == 0 else (m1 - m0) / sd


def covariates(m, X, spec):
    """every feature as a numeric column: numeric + flags + one indicator per category level."""
    num = eda.NUM + eda.FLAGS + (prop.HIST_NUM + prop.HIST_FLAGS if spec == "all_hist" else [])
    cols = {eda.LABEL.get(c, c): X[c].to_numpy(float) for c in num}
    for c in eda.CAT:
        for lvl in sorted(m[c].astype(str).unique()):
            cols[f"{c}: {eda.LABEL.get(lvl, lvl)}"] = (m[c].astype(str) == lvl).to_numpy(float)
    return pd.DataFrame(cols)


def ess(w):
    return float(w.sum() ** 2 / (w ** 2).sum())


def main():
    m = pd.read_parquet(PROCESSED / "model_2023_2024.parquet")
    ps = pd.read_parquet(PROCESSED / "propensity_2023_2024.parquet")
    assert (ps["NPI"].to_numpy() == m["NPI"].to_numpy()).all()
    X = prop.design(m)
    out = {}
    love = None

    for spec in SPECS:
        rows = ps[f"in_{spec}"].to_numpy()
        keep = ps[f"keep_{spec}"].to_numpy()
        C = covariates(m.loc[rows].reset_index(drop=True), X.loc[rows].reset_index(drop=True), spec)
        t = ps.loc[rows, "treated"].to_numpy()
        res, summ = {}, {}
        for label, wcol in SCHEMES.items():
            if wcol is None:
                w, sel = np.ones(rows.sum()), np.ones(rows.sum(), bool)
            else:
                w = ps.loc[rows, f"{wcol}_{spec}"].to_numpy()
                sel = keep[rows] if wcol == "w_ate" else np.ones(rows.sum(), bool)
            s = np.array([wsmd(C[c].to_numpy()[sel], t[sel], w[sel]) for c in C.columns])
            res[label] = s
            summ[label] = {"max_abs_smd": round(float(np.abs(s).max()), 4),
                           "n_above_0.1": int((np.abs(s) >= 0.1).sum()),
                           "n_above_0.05": int((np.abs(s) >= 0.05).sum()),
                           "worst": C.columns[int(np.abs(s).argmax())],
                           "n": int(sel.sum()),
                           "ess_paid": round(ess(w[sel][t[sel]])), "ess_unpaid": round(ess(w[sel][~t[sel]]))}
        bal = pd.DataFrame(res, index=C.columns)
        out[spec] = {"n_covariates": len(C.columns), "summary": summ,
                     "pass_all_below_0.1": {k: v["n_above_0.1"] == 0 for k, v in summ.items()}}
        print(f"\n== {spec} ({len(C.columns)} covariates)")
        print(pd.DataFrame(summ).T.to_string())
        if spec == "all":
            love = bal

        # overlap by specialty (primary score)
        p = ps.loc[rows, f"ps_{spec}"].to_numpy()
        sp = pd.DataFrame({"specialty": m.loc[rows, "specialty"].to_numpy(), "p": p, "t": t, "keep": keep[rows]})
        g = sp.groupby("specialty").agg(n=("p", "size"), pct_paid=("t", "mean"),
                                         pct_in_0_1_0_9=("p", lambda s: ((s > 0.1) & (s < 0.9)).mean()),
                                         pct_trimmed=("keep", lambda s: 1 - s.mean()))
        g[["pct_paid", "pct_in_0_1_0_9", "pct_trimmed"]] *= 100
        out[spec]["overlap_by_specialty"] = g.sort_values("n", ascending=False).round(2).reset_index().to_dict(orient="records")
        if spec == "all":
            ov = g

    plots(love, ov)
    out["layer_a_shap"] = shap_layer_a(m, X)
    (LOGS / "balance.json").write_text(json.dumps(out, indent=2, default=str))


def plots(bal, ov):
    FIG.mkdir(parents=True, exist_ok=True)
    top = bal.reindex(bal["unweighted"].abs().sort_values(ascending=False).index).head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    y = np.arange(len(top))
    ax.axvspan(-0.1, 0.1, color=eda.GRID, alpha=0.6, linewidth=0)
    ax.axvline(0, color=eda.AXIS, linewidth=0.8)
    for col, colr in (("unweighted", eda.UNPAID), ("IPW (trimmed)", eda.PAID), ("overlap", eda.AQUA)):
        ax.scatter(top[col], y, s=38, color=colr, edgecolor=eda.SURFACE, linewidth=1.4, zorder=3, label=col)
    ax.set_yticks(y, top.index, fontsize=7.5, color=eda.INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("standardized mean difference (paid - unpaid)")
    ax.legend(loc="lower right", fontsize=8)
    mx = bal["IPW (trimmed)"].abs().max()
    ax.set_title("Weighting balances every covariate" if mx < 0.1 else "Some covariates remain imbalanced", pad=18)
    eda.subtitle(ax, f"25 most imbalanced of {len(bal)} covariates; max |SMD| after IPW {mx:.3f}; band = 0.1")
    fig.savefig(FIG / "p03_love_plot.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    o = ov.sort_values("n", ascending=False).head(20).sort_values("pct_in_0_1_0_9")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.barh([eda.LABEL.get(i, i) for i in o.index], o["pct_in_0_1_0_9"] / 100, color=eda.PAID, height=0.6)
    for i, v in enumerate(o["pct_in_0_1_0_9"]):
        ax.text(v / 100, i, f" {v:.0f}%", va="center", color=eda.INK2, fontsize=8)
    eda.pct_axis(ax, "x")
    ax.set_xlim(0, 1.08)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8, colors=eda.INK2)
    ax.set_title("Overlap by specialty", pad=18)
    eda.subtitle(ax, "share of prescribers with propensity between 0.1 and 0.9; 20 largest specialties")
    fig.savefig(FIG / "p04_overlap_by_specialty.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def shap_layer_a(m, X):
    """SHAP on a boosting propensity model (main spec): what drives who gets paid. Diagnostic only."""
    try:
        import shap
    except ImportError:
        return {"error": "shap not installed"}
    # categories are one-hot encoded: TreeExplainer does not attribute native categorical (bitset) splits
    # correctly, so specialty would look unimportant. Per-level SHAP values are summed back to the variable.
    num = eda.NUM + eda.FLAGS
    dummies = pd.get_dummies(X[eda.CAT].astype(str), prefix_sep="=", dtype=float)
    Z = pd.concat([X[num].reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)
    group = [c if c in num else c.split("=")[0] for c in Z.columns]
    t = m["treated"].to_numpy()
    rng = np.random.default_rng(0)
    fit_i = rng.choice(len(Z), 300_000, replace=False)
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=200,
                                        random_state=0).fit(Z.iloc[fit_i], t[fit_i])
    ex_i = rng.choice(len(Z), SHAP_N, replace=False)
    try:
        sv = shap.TreeExplainer(gb).shap_values(Z.iloc[ex_i])
    except Exception as e:  # noqa: BLE001
        return {"error": f"TreeExplainer failed: {e}"}
    sv = sv[1] if isinstance(sv, list) else sv
    grouped = pd.DataFrame(sv, columns=Z.columns).T.groupby(group).sum().T  # SHAP is additive within a variable
    imp = grouped.abs().mean(axis=0)
    imp.index = [eda.LABEL.get(c, c) for c in imp.index]
    imp = imp.sort_values()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp.index, imp.values, color=eda.PAID, height=0.6)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8, colors=eda.INK2)
    ax.set_xlabel("mean |SHAP value| (log-odds of being paid)")
    ax.set_title("Layer A: what drives who gets paid", pad=18)
    eda.subtitle(ax, f"SHAP on a boosting propensity model, {SHAP_N:,} prescribers; diagnostic, not effect modifiers")
    fig.savefig(FIG / "p05_propensity_shap.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {"mean_abs_shap": imp.sort_values(ascending=False).round(4).to_dict()}


if __name__ == "__main__":
    main()
