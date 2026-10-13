"""Step 10: extended EDA for the report - outcome drivers, treatment separability, and propensity matching.

Writes figures to reports/report/figures/ and numbers to reports/report/extended_stats.json (git-ignored).
Matching here is descriptive (1:1 nearest neighbour on the logit propensity, exact on specialty, with
replacement, caliper 0.2 SD); the causal model comes later.
"""
import importlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.neighbors import NearestNeighbors

from config import PROCESSED, PROJECT

eda = importlib.import_module("08_eda")  # shared style, palette, helpers
OUT = PROJECT / "reports" / "report"
FIG = OUT / "figures"
PAID, UNPAID, INK2, MUTED, GRID, AXIS, SURFACE = eda.PAID, eda.UNPAID, eda.INK2, eda.MUTED, eda.GRID, eda.AXIS, eda.SURFACE
stats = {}


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def short(s):
    return eda.LABEL.get(s, s)


def paid_unpaid_lines(ax, d, x, y, q=10):
    d = d.assign(_bin=pd.qcut(d[x], q, duplicates="drop"))
    for flag, col, lab in [(False, UNPAID, "unpaid"), (True, PAID, "paid")]:
        g = d[d["treated"] == flag].groupby("_bin", observed=True).agg(x=(x, "median"), y=(y, "mean"))
        ax.plot(g["x"], g["y"], color=col, marker="o", markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.2,
                label=lab)


def drivers(m):
    """How brand share relates to specialty, location, size and patient mix (informative-bound rows only)."""
    d = m[~m["y_share_uninformative"]]
    stats["drivers_n"] = len(d)

    # specialty dumbbell
    top = d["specialty"].value_counts().head(15).index
    s = d[d["specialty"].isin(top)].groupby(["specialty", "treated"])["y_brand_share"].mean().unstack()
    s.columns = ["unpaid", "paid"]
    s["gap"] = s["paid"] - s["unpaid"]
    s = s.sort_values("unpaid")
    fig, ax = plt.subplots(figsize=(7, 5.5))
    y = np.arange(len(s))
    ax.hlines(y, s[["unpaid", "paid"]].min(axis=1), s[["unpaid", "paid"]].max(axis=1), color=AXIS, linewidth=2)
    ax.scatter(s["unpaid"], y, s=45, color=UNPAID, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="unpaid")
    ax.scatter(s["paid"], y, s=45, color=PAID, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="paid")
    ax.set_yticks(y, [short(i) for i in s.index], fontsize=8, color=INK2)
    ax.grid(axis="y", visible=False)
    eda.pct_axis(ax, "x")
    ax.set_xlabel("mean brand share, 2024")
    ax.legend(loc="lower right")
    ax.set_title(f"Specialty sets the level; paid > unpaid in {(s['gap'] > 0).sum()} of {len(s)} specialties", pad=18)
    eda.subtitle(ax, "15 largest specialties, unadjusted")
    save(fig, "e01_share_by_specialty")
    stats["specialty"] = (100 * s).round(2).reset_index().to_dict(orient="records")

    # state: % paid vs mean brand share
    st = d.groupby("state").agg(n=("NPI", "size"), pct_paid=("treated", "mean"), share=("y_brand_share", "mean"))
    st = st[st["n"] >= 1000]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.scatter(st["pct_paid"], st["share"], s=np.clip(st["n"] / 600, 12, 160), color=PAID, alpha=0.75,
               edgecolor=SURFACE, linewidth=1.2)
    r = st[["pct_paid", "share"]].corr().iloc[0, 1]
    for k in pd.concat([st.nlargest(3, "share"), st.nsmallest(3, "share"), st.nlargest(2, "pct_paid"),
                        st.nsmallest(2, "pct_paid")]).index.unique():
        ax.annotate(k, (st.loc[k, "pct_paid"], st.loc[k, "share"]), xytext=(4, 3), textcoords="offset points",
                    fontsize=8, color=INK2)
    eda.pct_axis(ax, "x")
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.set_xlabel("share of prescribers paid in 2023")
    ax.set_ylabel("mean brand share, 2024")
    ax.set_title("State payment rates barely relate to state brand share" if abs(r) < 0.2 else (f"States with more paid prescribers prescribe {'more' if r > 0 else 'less'} brand"), pad=18)
    eda.subtitle(ax, f"one dot per state (size = prescribers); correlation {r:.2f}; unadjusted")
    save(fig, "e02_state_paid_vs_share")
    stats["state_corr"] = round(float(r), 3)
    stats["state_top_share"] = (100 * st.nlargest(5, "share")[["pct_paid", "share"]]).round(1).reset_index().to_dict(orient="records")
    stats["state_bottom_share"] = (100 * st.nsmallest(5, "share")[["pct_paid", "share"]]).round(1).reset_index().to_dict(orient="records")
    loc = []
    for col in ("region", "ruca_group"):
        g = d.groupby([col, "treated"])["y_brand_share"].mean().unstack()
        g.columns = ["unpaid", "paid"]
        g = 100 * g
        g["gap_pts"] = g["paid"] - g["unpaid"]
        g["n"] = d[col].value_counts()
        loc.append(g.round(2).reset_index().rename(columns={col: "group"}).assign(var=col))
    stats["location"] = pd.concat(loc).to_dict(orient="records")

    # small multiples: size and patient mix
    panels = [("x_log_clms", "2022 total claims (log)"), ("x_risk_score", "patient risk score"),
              ("x_bene_age", "patient mean age"), ("x22_share_dual", "share dual-eligible")]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6))
    for ax, (x, lab) in zip(axes.flat, panels):
        paid_unpaid_lines(ax, d.dropna(subset=[x]), x, "y_brand_share")
        ax.set_xlabel(lab, fontsize=9)
        eda.pct_axis(ax)
        ax.tick_params(labelsize=8)
    axes[0, 0].set_ylabel("mean brand share 2024")
    axes[1, 0].set_ylabel("mean brand share 2024")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Brand share by deciles of size and patient mix", x=0.06, ha="left", fontweight="semibold")
    fig.tight_layout()
    save(fig, "e03_share_by_size_and_patients")

    # persistence 2022 -> 2024
    dd = d.dropna(subset=["x_brand_share_22"])
    stats["persistence_corr"] = round(float(dd[["x_brand_share_22", "y_brand_share"]].corr().iloc[0, 1]), 3)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    lim = dd["x_brand_share_22"].quantile(0.99)
    ax.plot([0, lim], [0, lim], color=AXIS, linewidth=1)
    paid_unpaid_lines(ax, dd, "x_brand_share_22", "y_brand_share", q=20)
    eda.pct_axis(ax, "x")
    eda.pct_axis(ax)
    ax.set_xlabel("brand share 2022 (20 bins)")
    ax.set_ylabel("mean brand share 2024")
    ax.legend(loc="upper left")
    ax.set_title("Prescribing habits persist: 2022 predicts 2024", pad=18)
    eda.subtitle(ax, f"correlation {stats['persistence_corr']:.2f}; grey line = no change")
    save(fig, "e04_persistence")

    # correlation heatmap among numeric covariates, treatment dose and outcome
    cols = eda.NUM + ["pay_log1p_usd", "y_brand_share"]
    c = d[cols].corr()
    names = [eda.LABEL.get(x, {"pay_log1p_usd": "log payment $", "y_brand_share": "brand share 2024"}.get(x, x)) for x in cols]
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(c, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols)), names, rotation=60, ha="right", fontsize=8, color=INK2)
    ax.set_yticks(range(len(cols)), names, fontsize=8, color=INK2)
    ax.grid(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = c.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="white" if abs(v) > 0.55 else "#0b0b0b")
    fig.colorbar(im, ax=ax, shrink=0.7, label="Pearson correlation")
    ax.set_title("Correlations: covariates, payment dose and outcome", pad=12)
    save(fig, "e05_correlations")
    stats["corr_with_outcome"] = c["y_brand_share"].drop("y_brand_share").round(3).sort_values().to_dict()
    stats["corr_with_dose"] = c["pay_log1p_usd"].drop("pay_log1p_usd").round(3).sort_values().to_dict()


def separability(m, X, t, fit):
    """How well can covariates separate paid from unpaid? (univariate and model-based)"""
    hold = ~fit
    rows = []
    for col in eda.NUM:
        a = roc_auc_score(t[hold], X.loc[hold, col])
        rows.append({"feature": eda.LABEL[col], "AUC": max(a, 1 - a)})
    rate = pd.Series(t[fit]).groupby(m.loc[fit, "specialty"].to_numpy()).mean()  # target-encode on fit rows only
    a = roc_auc_score(t[hold], m.loc[hold, "specialty"].map(rate).fillna(t[fit].mean()))
    rows.append({"feature": "specialty", "AUC": a})
    for col in ("region", "ruca_group", "state"):
        rr = pd.Series(t[fit]).groupby(m.loc[fit, col].to_numpy()).mean()
        a = roc_auc_score(t[hold], m.loc[hold, col].map(rr).fillna(t[fit].mean()))
        rows.append({"feature": col.replace("ruca_group", "rural/urban (RUCA)"), "AUC": max(a, 1 - a)})
    u = pd.DataFrame(rows).sort_values("AUC")

    lr = eda.make_pipeline(eda.ColumnTransformer([
        ("num", eda.StandardScaler(), eda.NUM + eda.FLAGS),
        ("cat", eda.OneHotEncoder(handle_unknown="ignore", min_frequency=50), eda.CAT)]),
        eda.LogisticRegression(max_iter=1000)).fit(X[fit], t[fit])
    p_lr = lr.predict_proba(X)[:, 1]
    Xg = X.copy()
    for col in eda.CAT:
        Xg[col] = Xg[col].astype("category")
    gb = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1, categorical_features="from_dtype",
                                        random_state=0).fit(Xg[fit], t[fit])
    p_gb = gb.predict_proba(Xg)[:, 1]
    auc_lr, auc_gb = roc_auc_score(t[hold], p_lr[hold]), roc_auc_score(t[hold], p_gb[hold])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.2, 1]})
    a1.barh(u["feature"], u["AUC"] - 0.5, left=0.5, color=PAID, height=0.6)
    for i, v in enumerate(u["AUC"]):
        a1.text(v, i, f" {v:.2f}", va="center", color=INK2, fontsize=8)
    a1.set_xlim(0.5, max(0.75, u["AUC"].max() + 0.05))
    a1.grid(axis="y", visible=False)
    a1.tick_params(axis="y", labelsize=8, colors=INK2)
    a1.set_xlabel("AUC of this feature alone (0.5 = no separation)")
    a1.set_title("Each feature alone separates weakly", fontsize=11)
    for p, col, lab in [(p_lr, UNPAID, f"logistic  AUC {auc_lr:.3f}"), (p_gb, PAID, f"gradient boosting  AUC {auc_gb:.3f}")]:
        fpr, tpr, _ = roc_curve(t[hold], p[hold])
        a2.plot(fpr, tpr, color=col, label=lab)
    a2.plot([0, 1], [0, 1], color=AXIS, linewidth=1)
    a2.set_xlabel("false positive rate")
    a2.set_ylabel("true positive rate")
    a2.legend(loc="lower right", fontsize=8)
    a2.set_title("Together: moderate, not perfect, separation", fontsize=11)
    fig.tight_layout()
    save(fig, "e06_separability")

    def ov(p):
        return {"pct_ps_0.1_0.9": round(100 * float(((p > 0.1) & (p < 0.9)).mean()), 1),
                "pct_ps_lt_0.05": round(100 * float((p < 0.05).mean()), 2),
                "pct_ps_gt_0.95": round(100 * float((p > 0.95).mean()), 2)}
    spec_ov = (pd.DataFrame({"specialty": m["specialty"], "p": p_gb, "t": t})
               .groupby("specialty").agg(n=("p", "size"), pct_paid=("t", "mean"),
                                         pct_in_0_1_0_9=("p", lambda s: ((s > 0.1) & (s < 0.9)).mean())))
    spec_ov = spec_ov.sort_values("n", ascending=False).head(15)
    stats["separability"] = {
        "univariate": u.round(3).to_dict(orient="records"), "auc_logistic": round(auc_lr, 4),
        "auc_gbm": round(auc_gb, 4), "overlap_logistic": ov(p_lr), "overlap_gbm": ov(p_gb),
        "overlap_by_specialty": (100 * spec_ov[["pct_paid", "pct_in_0_1_0_9"]]).round(1).join(spec_ov["n"])
                                .reset_index().to_dict(orient="records")}
    return p_lr


def matching(m, X, t, ps):
    """1:1 NN matching on logit(ps), exact on specialty, with replacement, caliper 0.2 SD."""
    lg = np.log(ps / (1 - ps))
    cal = 0.2 * lg.std()
    tid, cid = [], []
    for sp, idx in pd.Series(np.arange(len(m))).groupby(m["specialty"].to_numpy()):
        idx = idx.to_numpy()
        ti, ci = idx[t[idx]], idx[~t[idx]]
        if len(ti) == 0 or len(ci) == 0:
            continue
        nn = NearestNeighbors(n_neighbors=1).fit(lg[ci].reshape(-1, 1))
        dist, j = nn.kneighbors(lg[ti].reshape(-1, 1))
        ok = dist[:, 0] <= cal
        tid.append(ti[ok])
        cid.append(ci[j[ok, 0]])
    tid, cid = np.concatenate(tid), np.concatenate(cid)
    n_t = int(t.sum())
    w = np.zeros(len(m))
    w[tid] = 1
    np.add.at(w, cid, 1)

    rows = []
    for col in eda.NUM + ["new_in_2024"]:
        x = X[col].to_numpy(float)
        sel = w > 0
        rows.append({"covariate": eda.LABEL[col], "unadjusted": eda.smd(x, t),
                     "matched": eda.smd(x[sel], t[sel], w[sel])})
    for sp in m["specialty"].value_counts().head(6).index:
        x = (m["specialty"] == sp).to_numpy(float)
        sel = w > 0
        rows.append({"covariate": f"specialty: {short(sp)}", "unadjusted": eda.smd(x, t), "matched": eda.smd(x[sel], t[sel], w[sel])})
    bal = pd.DataFrame(rows).set_index("covariate")
    bal = bal.reindex(bal["unadjusted"].abs().sort_values().index)

    fig, ax = plt.subplots(figsize=(7, 5))
    yy = np.arange(len(bal))
    ax.axvspan(-0.1, 0.1, color=GRID, alpha=0.6, linewidth=0)
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.scatter(bal["unadjusted"], yy, s=40, color=UNPAID, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="unadjusted")
    ax.scatter(bal["matched"], yy, s=40, color=PAID, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="matched")
    ax.set_yticks(yy, bal.index, fontsize=8, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("standardized mean difference (paid - unpaid)")
    ax.legend(loc="lower right")
    ax.set_title("Matching within specialty balances every covariate", pad=18)
    eda.subtitle(ax, f"{len(tid):,} of {n_t:,} paid matched ({len(tid) / n_t:.1%}); "
                     f"{len(np.unique(cid)):,} distinct unpaid controls")
    save(fig, "e07_matched_balance")

    res = []
    for outcome, label in [("y_brand_share", "brand share 2024"), ("y_change_brand_share", "change in share 2022-24")]:
        for v in ("lo", "mid", "hi"):
            if outcome == "y_brand_share":
                col = "y_brand_share" if v == "mid" else f"y_brand_share_{v}"
                y = m[col].to_numpy()
            else:
                pre = m["x_brand_share_22"] if v == "mid" else m[f"x_brand_share_22_{v}"]
                cur = m["y_brand_share"] if v == "mid" else m[f"y_brand_share_{v}"]
                y = (cur - pre).to_numpy()
            ok = ~np.isnan(y[tid]) & ~np.isnan(y[cid]) & ~m["y_share_uninformative"].to_numpy()[tid] \
                 & ~m["y_share_uninformative"].to_numpy()[cid]
            diff = y[tid][ok] - y[cid][ok]
            res.append({"outcome": label, "bounds": v, "pairs": int(ok.sum()), "ATT_pts": 100 * diff.mean(),
                        "se_pts": 100 * diff.std() / np.sqrt(ok.sum())})
    att = pd.DataFrame(res)

    raw = 100 * (m.loc[t & ~m["y_share_uninformative"], "y_brand_share"].mean()
                 - m.loc[~t & ~m["y_share_uninformative"], "y_brand_share"].mean())
    mid = att[att["bounds"] == "mid"].set_index("outcome")
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels = ["raw difference\n(brand share 2024)", "matched\n(brand share 2024)", "matched\n(change 2022-24)"]
    vals = [raw, mid.loc["brand share 2024", "ATT_pts"], mid.loc["change in share 2022-24", "ATT_pts"]]
    errs = [0, 1.96 * mid.loc["brand share 2024", "se_pts"], 1.96 * mid.loc["change in share 2022-24", "se_pts"]]
    ax.barh(labels[::-1], vals[::-1], xerr=errs[::-1], color=[PAID, PAID, AXIS][::-1], height=0.5,
            error_kw={"ecolor": INK2, "elinewidth": 1.2, "capsize": 4})
    for i, v in enumerate(vals[::-1]):
        ax.text(v, i, f"  +{v:.2f} pts", va="center", color=INK2, fontsize=9)
    ax.axvline(0, color=AXIS, linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.tick_params(axis="y", labelsize=8, colors=INK2)
    ax.set_xlim(0, max(vals) * 1.35)
    ax.set_xlabel("paid minus unpaid (percentage points)")
    ax.set_title(f"Matching shrinks the raw gap from {vals[0]:.1f} to {vals[1]:.1f} points", pad=18)
    eda.subtitle(ax, "1:1 matched on propensity within specialty; excludes uninformative bounds; 95% CI (approx.)")
    save(fig, "e08_matched_att")
    stats["matching"] = {"caliper_logit": round(float(cal), 4), "treated": n_t, "matched_treated": int(len(tid)),
                         "pct_matched": round(100 * len(tid) / n_t, 2), "distinct_controls": int(len(np.unique(cid))),
                         "max_abs_smd_unadjusted": round(float(bal["unadjusted"].abs().max()), 3),
                         "max_abs_smd_matched": round(float(bal["matched"].abs().max()), 3),
                         "balance": bal.round(3).reset_index().to_dict(orient="records"),
                         "raw_diff_pts": round(float(raw), 3), "att": att.round(3).to_dict(orient="records")}


def payment_type(m):
    """Newly paid in 2023 (no 2022 payment) by dominant payment type vs never paid: raw change in share."""
    d = m[~m["new_in_2024"] & ~m["paid_2022"] & ~m["y_share_uninformative"]].copy()
    big = d[["usd_speaking", "usd_consulting"]].sum(axis=1)
    d["type"] = np.select([~d["treated"], d["pay_food_only"], big > 0.5 * d["pay_usd"]],
                          ["never paid", "food only", "mostly speaking/consulting"], "other mix")
    g = d.groupby("type").agg(n=("NPI", "size"), median_usd=("pay_usd", "median"),
                              share22=("x_brand_share_22", "mean"), share24=("y_brand_share", "mean"),
                              change=("y_change_brand_share", "mean"))
    g[["share22", "share24", "change"]] *= 100
    g = g.loc[["never paid", "food only", "other mix", "mostly speaking/consulting"]]
    stats["payment_type"] = g.round(3).reset_index().to_dict(orient="records")


def main():
    m = pd.read_parquet(PROCESSED / "model_2023_2024.parquet")
    t = m["treated"].to_numpy()
    X = eda.design(m)
    rng = np.random.default_rng(0)
    fit = np.zeros(len(m), bool)
    fit[rng.choice(len(m), 400_000, replace=False)] = True
    drivers(m)
    ps = np.clip(separability(m, X, t, fit), 0.01, 0.99)
    matching(m, X, t, ps)
    payment_type(m)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "extended_stats.json").write_text(json.dumps(stats, indent=2, default=str))
    print(json.dumps({k: v for k, v in stats.items() if k in ("separability", "matching", "payment_type", "state_corr",
                                                          "persistence_corr", "corr_with_outcome")}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
