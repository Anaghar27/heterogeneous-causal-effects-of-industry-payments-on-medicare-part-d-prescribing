"""Step 4: link Part D (left, the prescriber universe) to payments (right) on NPI.

Builds:
  analytic_2023_2024.parquet  payments 2023 -> Part D 2024, + Part D 2022 baseline, + 2022 payment history
  analytic_2022_2023.parquet  payments 2022 -> Part D 2023 (second cohort, no baseline year available)
A prescriber with no payment row is UNPAID (control), not missing.
"""
import json

import pandas as pd

from config import LOGS, PROCESSED

SPEC_COLS = ["NPI", "treated", "total_payment_usd", "payment_count", "Prscrbr_Type",
             "Tot_Clms", "Tot_Benes", "Brnd_Tot_Clms", "Brnd_Tot_Drug_Cst"]
BASE_COLS = ["Prscrbr_Type", "Tot_Clms", "Tot_Drug_Cst", "Tot_Benes_imp", "benes_suppressed",
             "Brnd_Tot_Clms_lo", "Brnd_Tot_Clms_hi", "Brnd_Tot_Clms_mid", "brnd_imputed", "brand_share_mid"]


def load(name):
    d = pd.read_parquet(PROCESSED / name)
    d["NPI"] = d["NPI"].astype("string").str.strip()
    assert d["NPI"].is_unique, f"duplicate NPIs in {name}"
    return d


def link(partd_year, pay_year):
    partd = load(f"partd_by_provider_{partd_year}.parquet")
    pay = load(f"payments_by_npi_{pay_year}.parquet")
    a = partd.merge(pay, on="NPI", how="left", validate="one_to_one", indicator="_m")
    a["has_payment_record"] = a.pop("_m").eq("both")
    for c in ("total_payment_usd", "payment_count", "n_payments_reported", "food_bev_usd",
              "food_bev_count", "max_single_payment_usd", "n_distinct_manufacturers"):
        a[c] = a[c].fillna(0)
    for c in ("payment_count", "n_payments_reported", "food_bev_count", "n_distinct_manufacturers"):
        a[c] = a[c].astype("int64")
    a["op_is_physician"] = a["op_is_physician"].astype("boolean")
    a["treated"] = a["total_payment_usd"] > 0
    stats = {
        "left": f"partd_by_provider_{partd_year}", "right": f"payments_by_npi_{pay_year}",
        "partd_rows": len(partd), "payment_npis": len(pay),
        "payment_npis_matched": int(pay["NPI"].isin(partd["NPI"]).sum()),
        "payment_usd_matched_share": round(float(
            pay.loc[pay["NPI"].isin(partd["NPI"]), "total_payment_usd"].sum() / pay["total_payment_usd"].sum()), 4),
    }
    return a, stats


def diagnostics(a, stats, label):
    n = len(a)
    s = dict(stats)
    s.update({
        "rows": n,
        "duplicate_npi": int(a["NPI"].duplicated().sum()),
        "pct_treated": round(100 * a["treated"].mean(), 2),
        "pct_unpaid": round(100 * (~a["treated"]).mean(), 2),
        "payment_record_but_total_usd_le_0": int((a["has_payment_record"] & ~a["treated"]).sum()),
        "nulls": {c: int(a[c].isna().sum()) for c in
                  ["NPI", "Prscrbr_Type", "Tot_Clms", "Brnd_Tot_Clms", "Brnd_Tot_Clms_mid", "Tot_Benes"]},
    })
    # raw (confounded, NOT causal) comparison
    raw = a.groupby("treated").agg(
        n=("NPI", "size"),
        mean_tot_clms=("Tot_Clms", "mean"),
        mean_brnd_clms_mid=("Brnd_Tot_Clms_mid", "mean"),
        median_brnd_clms_mid=("Brnd_Tot_Clms_mid", "median"),
        mean_brand_share_mid=("brand_share_mid", "mean"),
        pct_brnd_suppressed=("brnd_imputed", "mean"),
    ).round(3)
    raw["pct_brnd_suppressed"] = (100 * raw["pct_brnd_suppressed"]).round(1)
    s["raw_comparison"] = raw.reset_index().to_dict(orient="records")
    s["raw_brand_share_higher_in_treated"] = bool(raw.loc[True, "mean_brand_share_mid"] > raw.loc[False, "mean_brand_share_mid"])
    top = a["Prscrbr_Type"].value_counts().head(10).index
    s["pct_treated_top10_specialties"] = (100 * a[a["Prscrbr_Type"].isin(top)].groupby("Prscrbr_Type")["treated"]
                                          .mean()).round(1).sort_values(ascending=False).to_dict()
    print(f"\n===== {label} =====")
    print(json.dumps({k: v for k, v in s.items() if k not in ("raw_comparison",)}, indent=2, default=str))
    print("\nRaw comparison (confounded, not causal):")
    print(raw.to_string())
    return s


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    out = {}

    # --- main cohort: payments 2023 -> Part D 2024, baseline Part D 2022, history payments 2022
    a, st = link(2024, 2023)
    base = load("partd_by_provider_2022.parquet")[["NPI"] + BASE_COLS]
    base = base.rename(columns={c: f"base22_{c}" for c in BASE_COLS})
    a = a.merge(base, on="NPI", how="left", validate="one_to_one", indicator="_b")
    a["in_partd_2022"] = a.pop("_b").eq("both")
    p22 = load("payments_by_npi_2022.parquet")[["NPI", "total_payment_usd", "payment_count"]]
    p22 = p22.rename(columns={"total_payment_usd": "total_payment_usd_2022", "payment_count": "payment_count_2022"})
    a = a.merge(p22, on="NPI", how="left", validate="one_to_one", indicator="_p")
    a["paid_2022"] = a.pop("_p").eq("both")
    a[["total_payment_usd_2022", "payment_count_2022"]] = a[["total_payment_usd_2022", "payment_count_2022"]].fillna(0)
    a["payment_count_2022"] = a["payment_count_2022"].astype("int64")
    a["newly_paid_2023"] = a["treated"] & ~a["paid_2022"]
    a = a[SPEC_COLS + [c for c in a.columns if c not in SPEC_COLS]]
    a.to_parquet(PROCESSED / "analytic_2023_2024.parquet", index=False, compression="zstd")
    s = diagnostics(a, st, "analytic_2023_2024")
    s.update({
        "pct_in_partd_2022": round(100 * a["in_partd_2022"].mean(), 2),
        "pct_paid_2022": round(100 * a["paid_2022"].mean(), 2),
        "treatment_x_history": pd.crosstab(a["paid_2022"], a["treated"]).rename_axis(index="paid_2022", columns="treated_2023")
                               .to_dict(),
    })
    print("\npaid_2022 x treated_2023:\n", pd.crosstab(a["paid_2022"], a["treated"], margins=True).to_string())
    print(f"in Part D 2022: {s['pct_in_partd_2022']}%")
    s["out_bytes"] = (PROCESSED / "analytic_2023_2024.parquet").stat().st_size
    out["analytic_2023_2024"] = s

    # --- second cohort: payments 2022 -> Part D 2023
    b, st = link(2023, 2022)
    b = b[SPEC_COLS + [c for c in b.columns if c not in SPEC_COLS]]
    b.to_parquet(PROCESSED / "analytic_2022_2023.parquet", index=False, compression="zstd")
    s = diagnostics(b, st, "analytic_2022_2023")
    s["out_bytes"] = (PROCESSED / "analytic_2022_2023.parquet").stat().st_size
    out["analytic_2022_2023"] = s

    (LOGS / "link.json").write_text(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
