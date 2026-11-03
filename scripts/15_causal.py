"""Stage 4-5: causal effect of any 2023 payment on 2024 brand share (ATE, ATT, CATE, subgroups, modifiers).

Usage:
  python scripts/15_causal.py dr       # Part A: AIPW / DR-learner on the incident cohort (+ `all` as secondary)
  python scripts/15_causal.py forest   # Part B: causal forest (econml CausalForestDML) on a subsample

Design (decided in stage 3):
  cohort    incident = no 2022 payment; newly paid in 2023 vs never paid (secondary: all prescribers)
  sample    informative outcome bounds only (~y_share_uninformative); ATE on the Crump-trimmed sample
  outcome   2024 brand share in percentage points (mid; repeated at lo / hi bounds)
  nuisance  out-of-fold propensity from 12_propensity.py (clipped 0.05-0.95, as validated on IHDP);
            outcome models cross-fitted on the SAME folds
  DR        AIPW pseudo-outcome phi = mu1 - mu0 + T(Y - mu1)/e - (1 - T)(Y - mu0)/(1 - e)
            ATE = mean(phi); CATE = cross-fitted boosting regression of phi on X (Kennedy DR-learner)
Each part writes data/logs/causal_<part>.json and a .done marker. Prescriber-level CATEs go to
data/processed (git-ignored); only aggregates are logged.
"""
import importlib
import json
import sys
import time
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from config import LOGS, PROCESSED, PROJECT

warnings.filterwarnings("ignore")
eda = importlib.import_module("08_eda")
prop = importlib.import_module("12_propensity")
FIG = PROJECT / "reports" / "figures" / "model"
CLIP = 0.05
FOREST_N = 200_000
SEED = 0


def hgb_reg():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.08, max_leaf_nodes=63, min_samples_leaf=200,
                                         l2_regularization=1.0, categorical_features="from_dtype", early_stopping=True,
                                         validation_fraction=0.1, n_iter_no_change=20, random_state=SEED)


def features(m):
    X = prop.design(m)[eda.NUM + eda.FLAGS + eda.CAT].copy()
    for c in eda.CAT:
        X[c] = X[c].astype("category")
    return X


def load(spec):
    m = pd.read_parquet(PROCESSED / "model_2023_2024.parquet")
    ps = pd.read_parquet(PROCESSED / "propensity_2023_2024.parquet")
    assert (ps["NPI"].to_numpy() == m["NPI"].to_numpy()).all()
    rows = ps[f"in_{spec}"].to_numpy() & ~m["y_share_uninformative"].to_numpy()
    m, ps = m.loc[rows].reset_index(drop=True), ps.loc[rows].reset_index(drop=True)
    return m, ps


def cross_fit_mu(X, t, y, fold):
    """out-of-fold outcome models for each arm (T-learner nuisance), on the propensity folds."""
    mu0, mu1 = np.empty(len(y)), np.empty(len(y))
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        for arm, mu in ((0, mu0), (1, mu1)):
            sel = tr & (t == arm)
            mu[te] = hgb_reg().fit(X[sel], y[sel]).predict(X[te])
    return mu0, mu1


def aipw(t, y, e, mu0, mu1):
    phi = mu1 - mu0 + t * (y - mu1) / e - (1 - t) * (y - mu0) / (1 - e)
    att_terms = (t * (y - mu0) - (1 - t) * e / (1 - e) * (y - mu0)) / t.mean()
    return phi, att_terms


def est(x):
    m, se = float(np.mean(x)), float(np.std(x, ddof=1) / np.sqrt(len(x)))
    return {"est": round(m, 4), "se": round(se, 4), "ci_lo": round(m - 1.96 * se, 4), "ci_hi": round(m + 1.96 * se, 4)}


def gate(phi, groups, min_n=2000):
    d = pd.DataFrame({"phi": phi, "g": groups}).groupby("g")["phi"].agg(["mean", "std", "size"])
    d = d[d["size"] >= min_n]
    d["se"] = d["std"] / np.sqrt(d["size"])
    d["ci_lo"], d["ci_hi"] = d["mean"] - 1.96 * d["se"], d["mean"] + 1.96 * d["se"]
    return d.drop(columns="std").rename(columns={"mean": "effect_pts", "size": "n"})


def part_dr():
    out = {}
    for spec in ("incident", "all"):
        t0 = time.time()
        m, ps = load(spec)
        X = features(m)
        t = m["treated"].to_numpy().astype(float)
        e = np.clip(ps[f"ps_{spec}"].to_numpy(), CLIP, 1 - CLIP)
        fold = ps[f"fold_{spec}"].to_numpy()
        keep = ps[f"keep_{spec}"].to_numpy()
        res = {"n": int(len(m)), "n_trimmed_sample": int(keep.sum()), "pct_paid": round(100 * t.mean(), 2)}
        phis = {}
        for v in (("lo", "mid", "hi") if spec == "incident" else ("mid",)):  # bounds for the main cohort only
            col = "y_brand_share" if v == "mid" else f"y_brand_share_{v}"
            y = 100 * m[col].to_numpy()
            mu0, mu1 = cross_fit_mu(X, t, y, fold)
            phi, att = aipw(t, y, e, mu0, mu1)
            phis[v] = phi
            res[f"ate_{v}"] = est(phi[keep])
            res[f"ate_{v}_untrimmed"] = est(phi)
            res[f"att_{v}"] = est(att)
            res[f"naive_{v}"] = round(float(y[t == 1].mean() - y[t == 0].mean()), 4)
            print(f"  {spec} {v}: ATE {res[f'ate_{v}']['est']:.3f} (SE {res[f'ate_{v}']['se']:.3f})  "
                  f"ATT {res[f'att_{v}']['est']:.3f}  naive {res[f'naive_{v}']:.3f}  [{time.time() - t0:,.0f}s]",
                  flush=True)
        out[spec] = res
        if spec == "incident":
            cate_block(m, X, phis["mid"], keep, fold, out)
        (LOGS / "causal_dr.json").write_text(json.dumps(out, indent=2, default=str))  # checkpoint per spec
    plot_ate(out)


def cate_block(m, X, phi, keep, fold, out):
    """DR-learner CATE, heterogeneity check, subgroup effects (GATE) and best linear projection."""
    tau = np.empty(len(phi))
    for k in np.unique(fold):  # cross-fitted final stage: each CATE predicted by a model that never saw the unit
        tr, te = fold != k, fold == k
        tau[te] = hgb_reg().fit(X[tr], phi[tr]).predict(X[te])
    q = pd.qcut(tau, 5, labels=[f"Q{i}" for i in range(1, 6)])
    calib = gate(phi, q, min_n=1)  # mean pseudo-outcome within predicted-CATE quintiles
    calib["mean_predicted"] = pd.Series(tau).groupby(q, observed=True).mean()
    q1, q5 = phi[q == "Q1"], phi[q == "Q5"]
    diff = q5.mean() - q1.mean()
    diff_se = np.sqrt(q5.var(ddof=1) / len(q5) + q1.var(ddof=1) / len(q1))

    vol = pd.qcut(m["x_log_clms"].fillna(m["x_log_clms"].median()), 3, labels=["low", "mid", "high"])
    subgroups = {
        "specialty": gate(phi, m["specialty"].to_numpy()).sort_values("n", ascending=False).head(20),
        "region": gate(phi, m["region"].to_numpy()),
        "rural_urban": gate(phi, m["ruca_group"].to_numpy()),
        "volume_tertile": gate(phi, vol.astype(str).to_numpy()),
        "baseline_brand_share_tertile": gate(phi, pd.qcut(m["x_brand_share_22"], 3, labels=["low", "mid", "high"])
                                             .astype(object).fillna("new in 2024").astype(str).to_numpy()),
    }

    # best linear projection: phi on standardized candidate modifiers, robust SE (Chernozhukov et al.)
    Z = pd.DataFrame({eda.LABEL[c]: X[c].astype(float) for c in eda.NUM})
    Z = (Z - Z.mean()) / Z.std()
    for sp in m["specialty"].value_counts().head(8).index[1:]:  # largest specialty is the reference
        Z[f"specialty: {eda.LABEL.get(sp, sp)}"] = (m["specialty"] == sp).astype(float).to_numpy()
    for r in ("Northeast", "South", "West"):
        Z[f"region: {r}"] = (m["region"] == r).astype(float).to_numpy()
    Z["rural/small town"] = m["ruca_group"].isin(["rural", "small_town"]).astype(float).to_numpy()
    fit = sm.OLS(phi[keep], sm.add_constant(Z[keep])).fit(cov_type="HC1")
    blp = pd.DataFrame({"coef_pts": fit.params, "se": fit.bse, "p": fit.pvalues}).drop(index="const")
    blp["ci_lo"], blp["ci_hi"] = blp["coef_pts"] - 1.96 * blp["se"], blp["coef_pts"] + 1.96 * blp["se"]

    out["incident"]["cate"] = {
        "mean": round(float(tau.mean()), 4), "sd": round(float(tau.std()), 4),
        "quantiles": {f"p{int(100 * p)}": round(float(np.quantile(tau, p)), 4) for p in (0.05, 0.25, 0.5, 0.75, 0.95)},
        "pct_negative": round(100 * float((tau < 0).mean()), 2),
        "calibration_by_quintile": calib.round(4).reset_index().rename(columns={"index": "quintile"}).to_dict(orient="records"),
        "q5_minus_q1": {"est": round(float(diff), 4), "se": round(float(diff_se), 4),
                        "z": round(float(diff / diff_se), 2)},
    }
    out["incident"]["subgroups"] = {k: v.round(4).reset_index().rename(columns={"index": "group", "g": "group"})
                                    .to_dict(orient="records") for k, v in subgroups.items()}
    out["incident"]["blp"] = blp.round(4).reset_index().rename(columns={"index": "modifier"}).to_dict(orient="records")
    pd.DataFrame({"NPI": m["NPI"], "phi": phi, "cate_dr": tau, "keep": keep}).to_parquet(
        PROCESSED / "cate_incident.parquet", index=False, compression="zstd")
    plot_cate(tau, calib, subgroups["specialty"], blp)


def part_forest():
    from econml.dml import CausalForestDML
    t0 = time.time()
    m, ps = load("incident")
    keep = ps["keep_incident"].to_numpy()
    m, ps = m.loc[keep].reset_index(drop=True), ps.loc[keep].reset_index(drop=True)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(m), min(FOREST_N, len(m)), replace=False)
    m, ps = m.iloc[idx].reset_index(drop=True), ps.iloc[idx].reset_index(drop=True)
    Xd = prop.design(m)
    num = Xd[eda.NUM + eda.FLAGS].astype(float)
    dummies = pd.get_dummies(m[eda.CAT].astype(str), prefix_sep="=", dtype=float)
    Xf = pd.concat([num, dummies], axis=1)
    t = m["treated"].to_numpy().astype(int)
    y = 100 * m["y_brand_share"].to_numpy()
    cf = CausalForestDML(
        model_y=HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, min_samples_leaf=200, random_state=SEED),
        model_t=HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, min_samples_leaf=200, random_state=SEED),
        discrete_treatment=True, n_estimators=400, min_samples_leaf=100, max_samples=0.3, cv=3,
        n_jobs=-1, random_state=SEED)
    cf.fit(y, t, X=Xf.to_numpy())
    print(f"  forest fit {time.time() - t0:,.0f}s", flush=True)
    tau = cf.effect(Xf.to_numpy())
    ate = cf.ate_inference(Xf.to_numpy())
    imp = pd.Series(cf.feature_importances_, index=Xf.columns)
    group = [c if c in num.columns else c.split("=")[0] for c in Xf.columns]
    imp_g = imp.groupby(group).sum().sort_values(ascending=False)
    imp_g.index = [eda.LABEL.get(c, c) for c in imp_g.index]
    sp = pd.DataFrame({"tau": tau, "g": m["specialty"].to_numpy()}).groupby("g")["tau"].agg(["mean", "size"])
    sp = sp[sp["size"] >= 1000].sort_values("size", ascending=False).head(20)
    ci = ate.conf_int_mean()
    out = {"n": int(len(m)), "ate": round(float(ate.mean_point), 4), "ate_ci": [round(float(ci[0]), 4), round(float(ci[1]), 4)],
           "cate_sd": round(float(tau.std()), 4),
           "cate_quantiles": {f"p{int(100 * p)}": round(float(np.quantile(tau, p)), 4) for p in (0.05, 0.25, 0.5, 0.75, 0.95)},
           "importance": imp_g.round(4).to_dict(),
           "specialty_mean_cate": sp.round(4).reset_index().rename(columns={"g": "specialty"}).to_dict(orient="records"),
           "runtime_s": round(time.time() - t0)}
    (LOGS / "causal_forest.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: v for k, v in out.items() if k not in ("specialty_mean_cate",)}, indent=1))


def plot_ate(out):
    FIG.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec, lab in (("incident", "newly paid vs never paid"), ("all", "all prescribers")):
        r = out[spec]
        rows += [(f"{lab}: naive", r["naive_mid"], np.nan, np.nan),
                 (f"{lab}: ATE (AIPW)", r["ate_mid"]["est"], r["ate_mid"]["ci_lo"], r["ate_mid"]["ci_hi"]),
                 (f"{lab}: ATT (AIPW)", r["att_mid"]["est"], r["att_mid"]["ci_lo"], r["att_mid"]["ci_hi"])]
    d = pd.DataFrame(rows, columns=["label", "est", "lo", "hi"])[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    y = np.arange(len(d))
    col = [eda.AXIS if "naive" in s else eda.PAID for s in d["label"]]
    ax.scatter(d["est"], y, s=50, color=col, edgecolor=eda.SURFACE, linewidth=1.5, zorder=3)
    ax.hlines(y, d["lo"], d["hi"], color=eda.INK2, linewidth=1.5)
    for k, (v, hi) in enumerate(zip(d["est"], d["hi"])):
        ax.text(np.nanmax([v, hi]), k, f"  {v:+.2f}", va="center", color=eda.INK2, fontsize=9)
    ax.axvline(0, color=eda.AXIS, linewidth=0.8)
    ax.set_yticks(y, d["label"], fontsize=8.5, color=eda.INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("effect on 2024 brand share (percentage points), 95% CI")
    ax.set_title("Doubly robust estimates of the payment effect", pad=18)
    eda.subtitle(ax, "grey = unadjusted difference; blue = AIPW with cross-fitted nuisance models")
    ax.set_xlim(min(0, d["est"].min()) - 0.3, d["est"].max() * 1.25 + 0.3)
    fig.savefig(FIG / "c01_ate.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_cate(tau, calib, spec_gate, blp):
    FIG.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    lo, hi = np.quantile(tau, [0.005, 0.995])
    a1.hist(np.clip(tau, lo, hi), bins=60, color=eda.PAID)
    a1.axvline(0, color=eda.AXIS, linewidth=0.8)
    a1.axvline(tau.mean(), color=eda.UNPAID, linewidth=1.5)
    a1.set_xlabel("estimated effect per prescriber (points)")
    a1.set_ylabel("prescribers")
    a1.set_title("Spread of individual effects (DR-learner)", fontsize=11, loc="left")
    x = np.arange(len(calib))
    a2.bar(x, calib["effect_pts"], color=eda.PAID, width=0.55, yerr=1.96 * calib["se"],
           error_kw={"ecolor": eda.INK2, "capsize": 3})
    a2.plot(x, calib["mean_predicted"], color=eda.UNPAID, marker="o", markersize=5, label="predicted")
    a2.set_xticks(x, calib.index.astype(str))
    a2.axhline(0, color=eda.AXIS, linewidth=0.8)
    a2.set_xlabel("quintile of predicted effect")
    a2.set_ylabel("effect (points)")
    a2.legend(loc="upper left", fontsize=8)
    a2.set_title("Is the heterogeneity real? (observed vs predicted)", fontsize=11, loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "c02_cate.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    g = spec_gate.sort_values("effect_pts")
    fig, ax = plt.subplots(figsize=(7, 6))
    y = np.arange(len(g))
    ax.hlines(y, g["ci_lo"], g["ci_hi"], color=eda.INK2, linewidth=1.5)
    ax.scatter(g["effect_pts"], y, s=40, color=eda.PAID, edgecolor=eda.SURFACE, linewidth=1.4, zorder=3)
    ax.axvline(0, color=eda.AXIS, linewidth=0.8)
    ax.set_yticks(y, [eda.LABEL.get(i, i) for i in g.index], fontsize=8, color=eda.INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("effect on 2024 brand share (points), 95% CI")
    ax.set_title("Effect of payment by specialty", pad=18)
    eda.subtitle(ax, "DR group averages, newly paid vs never paid; 20 largest specialties")
    fig.savefig(FIG / "c03_effect_by_specialty.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    b = blp.sort_values("coef_pts")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    y = np.arange(len(b))
    sig = b["p"] < 0.05
    ax.hlines(y, b["ci_lo"], b["ci_hi"], color=eda.INK2, linewidth=1.5)
    ax.scatter(b["coef_pts"], y, s=40, color=np.where(sig, eda.PAID, eda.AXIS), edgecolor=eda.SURFACE, linewidth=1.4, zorder=3)
    ax.axvline(0, color=eda.AXIS, linewidth=0.8)
    ax.set_yticks(y, b.index, fontsize=8, color=eda.INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("change in effect (points) per SD, or vs reference group")
    ax.set_title("Layer B: which traits modify the effect", pad=18)
    eda.subtitle(ax, "best linear projection of the CATE; blue = p < 0.05; reference: largest specialty, Midwest, urban")
    fig.savefig(FIG / "c04_effect_modifiers.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    part = sys.argv[1]
    done = LOGS / f"causal_{part}.done"
    done.unlink(missing_ok=True)
    try:
        {"dr": part_dr, "forest": part_forest}[part]()
        done.write_text("ok")
    except Exception as e:  # noqa: BLE001
        done.write_text(f"failed: {e!r}")
        raise
