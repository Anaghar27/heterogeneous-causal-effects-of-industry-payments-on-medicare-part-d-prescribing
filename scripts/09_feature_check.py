"""Step 9: do the unused Part D 2022 columns (race mix, plan mix, drug-class mix, ...) matter as confounders?

For each candidate block, measured on 2022 (pre-treatment) data:
  1. treatment prediction: held-out AUC gain over the current covariates
  2. outcome prediction:   R^2 gain for the 2022->2024 change in brand share (never-paid prescribers only)
  3. balance:              |SMD| paid vs unpaid, before and after the CURRENT propensity weights
  4. estimate shift:       change in the newly-paid effect (screen design from step 6) when the block is added
Aggregates only.
"""
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import LOGS, PROCESSED, partd_raw

BASE_NUM = ["x_log_clms", "x_log_benes", "x_log1p_drug_cost_per_clm", "x_bene_age", "x_risk_score", "x_brand_share_22",
            "x22_share_female", "x22_share_dual", "x22_share_under65", "x22_opioid_rate"]
BASE_FLAGS = ["new_in_2024", "x_benes_suppressed", "x22_share_female_missing", "x22_share_dual_missing",
              "x22_share_under65_missing", "x22_opioid_rate_missing"]
CAT = ["specialty", "region", "ruca_group"]
RAW = ["Prscrbr_NPI", "Tot_Clms", "Tot_Benes", "Tot_30day_Fills", "Tot_Day_Suply", "GE65_Tot_Clms", "MAPD_Tot_Clms",
       "LIS_Tot_Clms", "Antbtc_Tot_Clms", "Antpsyct_GE65_Tot_Clms", "Opioid_LA_Prscrbr_Rate",
       "Bene_Race_Wht_Cnt", "Bene_Race_Black_Cnt", "Bene_Race_Api_Cnt", "Bene_Race_Hspnc_Cnt", "Bene_Race_Natind_Cnt",
       "Bene_Race_Othr_Cnt"]
BLOCKS = {
    "race mix": ["c_share_white", "c_share_black", "c_share_api", "c_share_hispanic", "c_share_natind", "c_share_race_other"],
    "plan / income mix": ["c_share_mapd_clms", "c_share_lis_clms", "c_share_ge65_clms"],
    "drug-class mix": ["c_share_antibiotic_clms", "c_share_antipsych_clms", "c_opioid_la_rate"],
    "fill pattern": ["c_fills_per_clm", "c_days_per_clm"],
}


def candidates():
    r = pd.read_csv(partd_raw(2022), usecols=RAW, dtype={"Prscrbr_NPI": str})
    c = pd.DataFrame({"NPI": r["Prscrbr_NPI"].str.strip()})
    for src, name in [("Wht", "white"), ("Black", "black"), ("Api", "api"), ("Hspnc", "hispanic"),
                      ("Natind", "natind"), ("Othr", "race_other")]:
        c[f"c_share_{name}"] = r[f"Bene_Race_{src}_Cnt"] / r["Tot_Benes"]
    c["c_share_mapd_clms"] = r["MAPD_Tot_Clms"] / r["Tot_Clms"]
    c["c_share_lis_clms"] = r["LIS_Tot_Clms"] / r["Tot_Clms"]
    c["c_share_ge65_clms"] = r["GE65_Tot_Clms"] / r["Tot_Clms"]
    c["c_share_antibiotic_clms"] = r["Antbtc_Tot_Clms"] / r["Tot_Clms"]
    c["c_share_antipsych_clms"] = r["Antpsyct_GE65_Tot_Clms"] / r["Tot_Clms"]
    c["c_opioid_la_rate"] = r["Opioid_LA_Prscrbr_Rate"]
    c["c_fills_per_clm"] = r["Tot_30day_Fills"] / r["Tot_Clms"]
    c["c_days_per_clm"] = r["Tot_Day_Suply"] / r["Tot_Clms"]
    return c


def prep(d, extra):
    """median-fill numeric columns and add a missing flag for each (suppression)."""
    X = d[BASE_NUM + BASE_FLAGS + CAT].copy()
    for col in BASE_NUM:
        X[col] = X[col].fillna(X[col].median())
    for col in BASE_FLAGS:
        X[col] = X[col].astype(float)
    flags = []
    for col in extra:
        X[col] = d[col].fillna(d[col].median())
        if d[col].isna().any():
            X[f"{col}_na"] = d[col].isna().astype(float)
            flags.append(f"{col}_na")
    return X, BASE_NUM + BASE_FLAGS + extra + flags


def ps_model(num):
    pre = ColumnTransformer([("num", StandardScaler(), num),
                             ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=50), CAT)])
    return make_pipeline(pre, LogisticRegression(max_iter=1000))


def smd(x, t, w):
    m1, m0 = np.average(x[t], weights=w[t]), np.average(x[~t], weights=w[~t])
    v1, v0 = np.average((x[t] - m1) ** 2, weights=w[t]), np.average((x[~t] - m0) ** 2, weights=w[~t])
    return (m1 - m0) / np.sqrt((v1 + v0) / 2 + 1e-12)


def effect(d, cols):
    """newly paid 2023 vs never paid, change in brand share 2022->2024 (points); step-6 screen design."""
    s = d[d["in_screen"]].copy()
    rhs = " + ".join(["newly_paid_2023"] + cols + ["C(ruca_group)", "C(specialty)", "C(state)"])
    m = smf.ols(f"dy ~ {rhs}", data=s).fit(cov_type="HC1")
    return m.params["newly_paid_2023[T.True]"], m.bse["newly_paid_2023[T.True]"]


def main():
    d = pd.read_parquet(PROCESSED / "model_2023_2024.parquet")
    d = d.merge(candidates(), on="NPI", how="left", validate="one_to_one")
    t = d["treated"].to_numpy()
    rng = np.random.default_rng(0)
    fit = np.zeros(len(d), bool)
    fit[rng.choice(len(d), 400_000, replace=False)] = True

    # current propensity weights (the baseline the extra features must beat)
    Xb, numb = prep(d, [])
    mb = ps_model(numb).fit(Xb[fit], t[fit])
    ps = np.clip(mb.predict_proba(Xb)[:, 1], 0.01, 0.99)
    auc_base = roc_auc_score(t[~fit], ps[~fit])
    w = np.where(t, 1 / ps, 1 / (1 - ps))

    # outcome model among never-paid, prescribers with a 2022 baseline (pure 'what predicts brand trend')
    d["dy"] = 100 * d["y_change_brand_share"]
    ctrl = (~d["treated"] & ~d["new_in_2024"] & ~d["y_share_uninformative"]).to_numpy()
    def r2(X, num):  # noqa: E301
        pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), CAT)])
        idx = np.flatnonzero(ctrl)
        tr, te = idx[rng.random(len(idx)) < 0.7], None
        te = np.setdiff1d(idx, tr)
        m = make_pipeline(pre, LinearRegression()).fit(X.iloc[tr], d["dy"].iloc[tr])
        return r2_score(d["dy"].iloc[te], m.predict(X.iloc[te]))
    r2_base = r2(Xb, numb)

    # screen sample for the estimate-shift check
    d["in_screen"] = (~d["new_in_2024"] & ~d["paid_2022"] & ~d["y_share_uninformative"]).to_numpy()
    for col in sum(BLOCKS.values(), []):
        d[f"{col}_f"] = d[col].fillna(d[col].median())
        d[f"{col}_na"] = d[col].isna().astype(float)
    base_rhs = ["x_brand_share_22", "x_log_clms", "x_log_benes", "x_risk_score", "x_bene_age", "x_log1p_drug_cost_per_clm"]
    est0, se0 = effect(d, base_rhs)

    rows, smds = [], []
    for name, cols in list(BLOCKS.items()) + [("all four blocks", sum(BLOCKS.values(), []))]:
        X, num = prep(d, cols)
        m = ps_model(num).fit(X[fit], t[fit])
        auc = roc_auc_score(t[~fit], m.predict_proba(X[~fit])[:, 1])
        est, se = effect(d, base_rhs + [f"{c}_f" for c in cols] + [f"{c}_na" for c in cols])
        rows.append({"block": name, "n_features": len(cols),
                     "pct_missing_avg": round(100 * d[cols].isna().mean().mean(), 1),
                     "AUC_gain": round(auc - auc_base, 4), "outcome_R2_gain": round(r2(X, num) - r2_base, 4),
                     "effect_pts": round(est, 3), "effect_shift_pts": round(est - est0, 3),
                     "effect_shift_pct": round(100 * (est - est0) / est0, 1)})
        if name != "all four blocks":
            for c in cols:
                x = d[c].fillna(d[c].median()).to_numpy(float)
                smds.append({"block": name, "feature": c, "SMD_raw": round(smd(x, t, np.ones(len(x))), 3),
                             "SMD_after_current_weights": round(smd(x, t, w), 3),
                             "pct_missing": round(100 * d[c].isna().mean(), 1)})

    res = pd.DataFrame(rows).set_index("block")
    sm = pd.DataFrame(smds).set_index("feature")
    print(f"baseline: propensity AUC {auc_base:.4f}; outcome R2 {r2_base:.4f}; "
          f"newly-paid effect {est0:.3f} pts (SE {se0:.3f}), n={int(d['in_screen'].sum()):,}\n")
    print(res.to_string())
    print("\nper-feature balance (|SMD| >= 0.1 = imbalanced):")
    print(sm.to_string())
    (LOGS / "feature_check.json").write_text(json.dumps({
        "baseline": {"auc": auc_base, "outcome_r2": r2_base, "effect_pts": est0, "effect_se": se0},
        "blocks": res.reset_index().to_dict(orient="records"),
        "balance": sm.reset_index().to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
