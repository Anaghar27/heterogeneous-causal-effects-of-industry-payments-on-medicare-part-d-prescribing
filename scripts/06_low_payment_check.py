"""Exploratory check: do LOW payments move brand prescribing? (decides whether a treatment cutoff is needed)

Design (difference-in-differences style, one cohort):
  sample  : individual prescribers in Part D 2022 AND 2024, with NO payments in 2022
  groups  : never paid (2022, 2023) vs newly paid in 2023, by 2023 amount
  outcome : change in brand share, 2022 (pre) -> 2024 (post)
  adjust  : specialty + state fixed effects, RUCA, and 2022 (pre-treatment) volume, beneficiaries,
            risk score, age, baseline brand share
  bounds  : re-estimated with suppressed brand counts at their low and high bounds
This is a screening regression, not the final causal (ATE/CATE) analysis.
"""
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config import LOGS, PROCESSED

BINS = [0, 25, 100, 1000, np.inf]
LABELS = ["usd_1_25", "usd_25_100", "usd_100_1k", "usd_gt_1k"]


def build():
    a = pd.read_parquet(PROCESSED / "analytic_2023_2024.parquet")
    b = pd.read_parquet(PROCESSED / "partd_by_provider_2022.parquet",
                        columns=["NPI", "Prscrbr_Ent_Cd", "Prscrbr_State_Abrvtn", "Prscrbr_RUCA",
                                 "Bene_Avg_Age", "Bene_Avg_Risk_Scre", "brand_share_lo", "brand_share_hi"])
    b = b.rename(columns=lambda c: c if c == "NPI" else f"b22_{c}")
    d = a[a["in_partd_2022"] & ~a["paid_2022"]].merge(b, on="NPI", how="inner", validate="one_to_one")
    d = d[d["b22_Prscrbr_Ent_Cd"].eq("I") & d["base22_Prscrbr_Type"].notna()].copy()
    d["group"] = "never_paid"
    paid = d["treated"]
    d.loc[paid, "group"] = pd.cut(d.loc[paid, "total_payment_usd"], BINS, labels=LABELS).astype(str)
    d["group"] = pd.Categorical(d["group"], ["never_paid"] + LABELS)
    d["spec"] = d["base22_Prscrbr_Type"]
    d["state"] = d["b22_Prscrbr_State_Abrvtn"].fillna("NA")
    d["ruca"] = pd.to_numeric(d["b22_Prscrbr_RUCA"], errors="coerce").round().fillna(-1).astype(int).astype(str)
    d["log_clms22"] = np.log(d["base22_Tot_Clms"])
    d["log_benes22"] = np.log(d["base22_Tot_Benes_imp"])
    d["risk22"] = d["b22_Bene_Avg_Risk_Scre"]
    d["age22"] = d["b22_Bene_Avg_Age"]
    for v in ("lo", "mid", "hi"):
        pre = d["base22_brand_share_mid"] if v == "mid" else d[f"b22_brand_share_{v}"]
        d[f"share22_{v}"] = pre
        d[f"dshare_{v}"] = 100 * (d[f"brand_share_{v}"] - pre)  # percentage points
    # specialties too small to estimate within
    big = d["spec"].map(d["spec"].value_counts()) >= 500
    return d[big].dropna(subset=["risk22", "age22"])


def fit(d, v):
    f = (f"dshare_{v} ~ C(group, Treatment('never_paid')) + share22_{v} + log_clms22 + log_benes22"
         " + risk22 + age22 + C(ruca) + C(spec) + C(state)")
    m = smf.ols(f, data=d).fit(cov_type="HC1")
    rows = []
    for g in LABELS:
        k = f"C(group, Treatment('never_paid'))[T.{g}]"
        lo, hi = m.conf_int().loc[k]
        rows.append({"bounds": v, "group": g, "effect_pts": m.params[k], "ci_lo": lo, "ci_hi": hi, "p": m.pvalues[k]})
    return rows


def main():
    d = build()
    print(f"sample: {len(d):,} prescribers, {d['spec'].nunique()} specialties")
    desc = d.groupby("group", observed=True).agg(
        n=("NPI", "size"), median_usd=("total_payment_usd", "median"),
        pct_food_only=("food_bev_usd", lambda s: np.nan),
        share22=("share22_mid", "mean"), share24=("brand_share_mid", "mean"), dshare=("dshare_mid", "mean"),
        clms22=("base22_Tot_Clms", "median"))
    food_only = (d["food_bev_usd"] == d["total_payment_usd"]) & d["treated"]
    desc["pct_food_only"] = 100 * food_only.groupby(d["group"], observed=True).mean()
    desc["share22"] *= 100
    desc["share24"] *= 100
    print("\nRaw (unadjusted) by group; shares in %, change in points:")
    print(desc.round(2).to_string())

    res = pd.DataFrame([r for v in ("lo", "mid", "hi") for r in fit(d, v)])
    print("\nAdjusted change in brand share vs never paid (percentage points, 95% CI, robust SE):")
    print(res.pivot(index="group", columns="bounds", values="effect_pts").loc[LABELS, ["lo", "mid", "hi"]].round(3).to_string())
    mid = res[res["bounds"] == "mid"].set_index("group").loc[LABELS]
    print("\nmid bounds detail:")
    print(mid[["effect_pts", "ci_lo", "ci_hi", "p"]].round(4).to_string())
    (LOGS / "low_payment_check.json").write_text(json.dumps(
        {"n": len(d), "groups": desc.round(4).reset_index().astype({"group": str}).to_dict(orient="records"),
         "estimates": res.round(5).to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
