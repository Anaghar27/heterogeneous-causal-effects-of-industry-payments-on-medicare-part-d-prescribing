"""Step 7: clean the 2023 -> 2024 cohort and engineer model features.

Output: data/processed/model_2023_2024.parquet (one row per prescriber)
        data/logs/clean_features.json (exclusion flow + feature summary; aggregates only)

Covariates (X) are PRE-treatment: measured in Part D 2022, or the 2024 value only where
2022 is missing (flagged by `new_in_2024`). Outcomes (Y) are Part D 2024. Treatment (T) is
Open Payments 2023.
"""
import json

import numpy as np
import pandas as pd

from config import INTERIM, LOGS, PROCESSED, partd_raw

MIN_PAID_RATE = 0.01  # specialties paid less than this are (near-)ineligible for payments -> no overlap
MIN_SPEC_N = 500  # smaller specialties fold into "Other"
LOW_VOLUME_CLMS = 50
WIDE_BOUND = 0.10  # brand-share bound width above this = outcome too uncertain to use as a share

REGION = {
    "Northeast": "CT ME MA NH RI VT NJ NY PA",
    "Midwest": "IL IN MI OH WI IA KS MN MO NE ND SD",
    "South": "DE DC FL GA MD NC SC VA WV AL KY MS TN AR LA OK TX",
    "West": "AZ CO ID MT NV NM UT WY AK CA HI OR WA",
}
STATE_REGION = {s: r for r, ss in REGION.items() for s in ss.split()}

NATURE_GROUP = {
    "Food and Beverage": "food",
    "Travel and Lodging": "travel",
    "Consulting Fee": "consulting",
    "Education": "education",
    "Gift": "other",
    "Royalty or License": "royalty",
    "Honoraria": "speaking",
    "Compensation for services other than consulting, including serving as faculty or as a speaker at a venue "
    "other than a continuing education program": "speaking",
    "Compensation for serving as faculty or as a speaker for a medical education program": "speaking",
}
EXTRA_2022 = ["Prscrbr_NPI", "Tot_Benes", "Bene_Feml_Cnt", "Bene_Dual_Cnt", "Bene_Age_LT_65_Cnt", "Opioid_Prscrbr_Rate"]


def ruca_group(x):
    r = pd.to_numeric(x, errors="coerce")
    return pd.cut(r, [0, 3.99, 6.99, 9.99, 10.99], labels=["metro", "micropolitan", "small_town", "rural"]) \
        .astype(str).replace("nan", "unknown")


def nature_usd(year, npis):
    t = pd.read_parquet(INTERIM / f"payments_rows_{year}.parquet", columns=["npi", "nature", "amount_usd"])
    t["grp"] = t["nature"].map(NATURE_GROUP).fillna("other")
    w = t.pivot_table(index="npi", columns="grp", values="amount_usd", aggfunc="sum", fill_value=0.0)
    w = w.add_prefix("usd_").reindex(npis, fill_value=0.0)
    return w.reset_index(names="NPI")


def extra_2022():
    e = pd.read_csv(partd_raw(2022), usecols=EXTRA_2022, dtype={"Prscrbr_NPI": str})
    e = e.rename(columns={"Prscrbr_NPI": "NPI"})
    out = pd.DataFrame({"NPI": e["NPI"].str.strip()})
    for c, name in [("Bene_Feml_Cnt", "female"), ("Bene_Dual_Cnt", "dual"), ("Bene_Age_LT_65_Cnt", "under65")]:
        out[f"x22_share_{name}"] = e[c] / e["Tot_Benes"]
    out["x22_opioid_rate"] = e["Opioid_Prscrbr_Rate"]
    return out


def main():
    a = pd.read_parquet(PROCESSED / "analytic_2023_2024.parquet")
    p22 = pd.read_parquet(PROCESSED / "partd_by_provider_2022.parquet",
                          columns=["NPI", "Prscrbr_Ent_Cd", "Prscrbr_State_Abrvtn", "Prscrbr_RUCA",
                                   "Bene_Avg_Age", "Bene_Avg_Risk_Scre", "brand_share_lo", "brand_share_hi"])
    p22 = p22.rename(columns={c: f"p22_{c}" for c in p22.columns if c != "NPI"})
    d = a.merge(p22, on="NPI", how="left", validate="one_to_one")
    flow = [{"step": "Part D 2024 prescribers", "n": len(d), "pct_treated": round(100 * d["treated"].mean(), 1)}]

    def drop(mask, why):
        nonlocal d
        flow.append({"step": why, "removed": int(mask.sum()), "removed_treated": int((mask & d["treated"]).sum())})
        d = d[~mask].copy()
        flow[-1].update({"n": len(d), "pct_treated": round(100 * d["treated"].mean(), 1)})

    drop(d["Prscrbr_Ent_Cd"].ne("I"), "organizations (entity code O)")

    # pre-treatment specialty: 2022 where available, else 2024
    d["specialty_raw"] = d["base22_Prscrbr_Type"].fillna(d["Prscrbr_Type"])
    drop(d["specialty_raw"].isna(), "missing specialty")
    rate = d.groupby("specialty_raw")["treated"].agg(["mean", "size"])
    inelig = rate[(rate["mean"] < MIN_PAID_RATE) & (rate["size"] >= 50)].index
    tiny_inelig = rate[(rate["size"] < 50) & (rate["mean"] == 0)].index
    drop(d["specialty_raw"].isin(inelig.union(tiny_inelig)),
         f"specialties with <{MIN_PAID_RATE:.0%} paid (no treatment overlap)")
    flow[-1]["specialties"] = sorted(inelig.union(tiny_inelig).tolist())

    counts = d["specialty_raw"].value_counts()
    d["specialty"] = d["specialty_raw"].where(d["specialty_raw"].map(counts) >= MIN_SPEC_N, "Other")
    spec_change = d["base22_Prscrbr_Type"].notna() & d["base22_Prscrbr_Type"].ne(d["Prscrbr_Type"])

    # --- pre-treatment covariates
    d["new_in_2024"] = ~d["in_partd_2022"]
    state = d["p22_Prscrbr_State_Abrvtn"].fillna(d["Prscrbr_State_Abrvtn"])
    d["state"] = state
    d["region"] = state.map(STATE_REGION).fillna("territory_or_other")
    d["ruca_group"] = ruca_group(d["p22_Prscrbr_RUCA"].fillna(d["Prscrbr_RUCA"]))
    d["is_dentist"] = d["specialty_raw"].eq("Dentist")
    pre = lambda b22, y24: d[b22].where(d["in_partd_2022"], d[y24])  # noqa: E731
    d["x_log_clms"] = np.log(pre("base22_Tot_Clms", "Tot_Clms"))
    d["x_log_benes"] = np.log(pre("base22_Tot_Benes_imp", "Tot_Benes_imp"))
    d["x_benes_suppressed"] = pre("base22_benes_suppressed", "benes_suppressed").astype(bool)
    d["x_log1p_drug_cost_per_clm"] = np.log1p(pre("base22_Tot_Drug_Cst", "Tot_Drug_Cst") / pre("base22_Tot_Clms", "Tot_Clms"))
    d["x_bene_age"] = d["p22_Bene_Avg_Age"].fillna(d["Bene_Avg_Age"])
    d["x_risk_score"] = d["p22_Bene_Avg_Risk_Scre"].fillna(d["Bene_Avg_Risk_Scre"])
    # baseline outcome: only meaningful pre-treatment for prescribers present in 2022
    d["x_brand_share_22"] = d["base22_brand_share_mid"]
    d["x_brand_share_22_lo"] = d["p22_brand_share_lo"]
    d["x_brand_share_22_hi"] = d["p22_brand_share_hi"]
    d["x_brand_imputed_22"] = d["base22_brnd_imputed"].astype("boolean")
    d = d.merge(extra_2022(), on="NPI", how="left", validate="one_to_one")
    for c in ["x22_share_female", "x22_share_dual", "x22_share_under65", "x22_opioid_rate"]:
        d[f"{c}_missing"] = d[c].isna()

    # --- treatment features (2023)
    d["pay_usd"] = d["total_payment_usd"]
    d["pay_log1p_usd"] = np.log1p(d["pay_usd"])
    d["pay_group"] = pd.cut(d["pay_usd"], [-1, 0, 25, 100, 1000, np.inf],
                            labels=["$0", "$1-25", "$25-100", "$100-1k", ">$1k"])
    d["pay_food_only"] = d["treated"] & (d["food_bev_usd"] >= d["pay_usd"] - 1e-9)
    d = d.merge(nature_usd(2023, d["NPI"]), on="NPI", how="left", validate="one_to_one")

    # --- outcomes (2024) and quality flags
    d["y_brand_share"] = d["brand_share_mid"]
    d["y_brand_share_lo"] = d["brand_share_lo"]
    d["y_brand_share_hi"] = d["brand_share_hi"]
    d["y_log1p_brand_clms"] = np.log1p(d["Brnd_Tot_Clms_mid"])
    d["y_change_brand_share"] = d["y_brand_share"] - d["x_brand_share_22"]
    d["y_share_bound_width"] = d["y_brand_share_hi"] - d["y_brand_share_lo"]
    d["y_share_uninformative"] = d["y_share_bound_width"] > WIDE_BOUND
    d["low_volume"] = d["Tot_Clms"].lt(LOW_VOLUME_CLMS) | d["x_log_clms"].lt(np.log(LOW_VOLUME_CLMS))

    keep = (["NPI", "treated", "pay_usd", "pay_log1p_usd", "pay_group", "payment_count", "n_distinct_manufacturers",
             "pay_food_only", "food_bev_usd", "newly_paid_2023", "paid_2022", "total_payment_usd_2022"]
            + [c for c in d.columns if c.startswith("usd_")]
            + ["specialty", "specialty_raw", "is_dentist", "state", "region", "ruca_group", "new_in_2024"]
            + [c for c in d.columns if c.startswith(("x_", "x22_"))]
            + ["y_brand_share", "y_brand_share_lo", "y_brand_share_hi", "y_log1p_brand_clms", "y_change_brand_share",
               "Brnd_Tot_Clms_mid", "Brnd_Tot_Clms_lo", "Brnd_Tot_Clms_hi", "brnd_imputed",
               "Brnd_Tot_Drug_Cst", "Brnd_Tot_Drug_Cst_lo", "Brnd_Tot_Drug_Cst_hi",
               "Tot_Clms", "Tot_Drug_Cst", "low_volume", "y_share_bound_width", "y_share_uninformative"])
    m = d[keep].reset_index(drop=True)
    assert m["NPI"].is_unique
    m.to_parquet(PROCESSED / "model_2023_2024.parquet", index=False, compression="zstd")

    summary = {
        "flow": flow,
        "final_rows": len(m),
        "pct_treated": round(100 * m["treated"].mean(), 2),
        "n_specialties_model": int(m["specialty"].nunique()),
        "specialty_changed_2022_2024": int(spec_change.sum()),
        "pct_new_in_2024": round(100 * m["new_in_2024"].mean(), 2),
        "pct_low_volume": round(100 * m["low_volume"].mean(), 2),
        "pct_share_uninformative": round(100 * m["y_share_uninformative"].mean(), 2),
        "pct_dentist": round(100 * m["is_dentist"].mean(), 2),
        "missing_pct": {c: round(100 * float(m[c].isna().mean()), 2) for c in m.columns if m[c].isna().any()},
        "columns": list(m.columns),
    }
    (LOGS / "clean_features.json").write_text(json.dumps(summary, indent=2, default=str))
    print(pd.DataFrame(flow).to_string(index=False))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("flow", "columns")}, indent=2, default=str))


if __name__ == "__main__":
    main()
