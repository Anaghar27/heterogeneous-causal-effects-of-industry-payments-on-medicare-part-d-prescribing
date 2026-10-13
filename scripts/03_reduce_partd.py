"""Step 3: reduce Medicare Part D Prescribers - by Provider to one row per NPI, with suppression bounds.

Usage: python scripts/03_reduce_partd.py 2024   (or: all)

CMS suppression in this file:
  '*' = cell suppressed because it is 1-10 claims
  '#' = counter-suppressed so a '*' cell cannot be back-calculated (any value >= 0)
  Tot_Benes blank = 1-10 beneficiaries
Identity Tot = Brnd + Gnrc + Othr (claims and cost) lets us bound a suppressed brand cell.
"""
import json
import sys

import numpy as np
import pandas as pd

from config import LOGS, PARTD_YEARS, PROCESSED, clean_npi, partd_raw

KEEP = ["Prscrbr_NPI", "Prscrbr_Type", "Prscrbr_Ent_Cd", "Prscrbr_State_Abrvtn", "Prscrbr_RUCA",
        "Tot_Clms", "Tot_Drug_Cst", "Tot_Benes", "Bene_Avg_Age", "Bene_Avg_Risk_Scre",
        "Brnd_Sprsn_Flag", "Brnd_Tot_Clms", "Brnd_Tot_Drug_Cst",
        "Gnrc_Sprsn_Flag", "Gnrc_Tot_Clms", "Gnrc_Tot_Drug_Cst",
        "Othr_Sprsn_Flag", "Othr_Tot_Clms", "Othr_Tot_Drug_Cst"]


def cell_bounds(value, flag, star_lo, star_hi):
    """Bounds for one suppressed cell: reported -> exact, '*' -> [star_lo, star_hi], '#' -> [0, inf)."""
    lo = value.copy()
    hi = value.copy()
    lo[flag.eq("*")], hi[flag.eq("*")] = star_lo, star_hi
    lo[flag.eq("#")], hi[flag.eq("#")] = 0.0, np.inf
    unknown = value.isna() & flag.isna()  # blank without a flag: treat as unbounded
    lo[unknown], hi[unknown] = 0.0, np.inf
    return lo, hi


def brand_bounds(d, kind, star_lo, star_hi):
    tot = d[f"Tot_{kind}"]
    b_lo, b_hi = cell_bounds(d[f"Brnd_Tot_{kind}"], d["Brnd_Sprsn_Flag"], star_lo, star_hi)
    g_lo, g_hi = cell_bounds(d[f"Gnrc_Tot_{kind}"], d["Gnrc_Sprsn_Flag"], star_lo, star_hi)
    o_lo, o_hi = cell_bounds(d[f"Othr_Tot_{kind}"], d["Othr_Sprsn_Flag"], star_lo, star_hi)
    lo = np.maximum(b_lo, tot - g_hi - o_hi).clip(lower=0)
    hi = np.minimum(b_hi, tot - g_lo - o_lo)
    return lo, hi


def main(year):
    src = partd_raw(year)
    d = pd.read_csv(src, usecols=KEEP, dtype={"Prscrbr_NPI": str, "Prscrbr_RUCA": str,
                                              "Brnd_Sprsn_Flag": str, "Gnrc_Sprsn_Flag": str, "Othr_Sprsn_Flag": str})
    st = {"source_file": src.name, "source_bytes": src.stat().st_size, "rows_raw": len(d)}
    d.insert(0, "NPI", clean_npi(d.pop("Prscrbr_NPI")))
    st["bad_npi"] = int(d["NPI"].isna().sum())
    st["duplicate_npi"] = int(d["NPI"].duplicated().sum())
    assert st["bad_npi"] == 0 and st["duplicate_npi"] == 0, st
    assert d["Tot_Clms"].notna().all() and d["Tot_Drug_Cst"].notna().all()
    st["brnd_flag"] = {str(k): int(v) for k, v in d["Brnd_Sprsn_Flag"].value_counts(dropna=False).items()}

    # verify the identities on fully reported rows before using them
    full = d[["Brnd_Tot_Clms", "Gnrc_Tot_Clms", "Othr_Tot_Clms"]].notna().all(axis=1)
    st["identity_clms_holds"] = float(np.isclose(
        d.loc[full, "Tot_Clms"], d.loc[full, ["Brnd_Tot_Clms", "Gnrc_Tot_Clms", "Othr_Tot_Clms"]].sum(axis=1)).mean())
    cst_sum = d.loc[full, ["Brnd_Tot_Drug_Cst", "Gnrc_Tot_Drug_Cst", "Othr_Tot_Drug_Cst"]].sum(axis=1)
    st["identity_cost_holds_within_1usd"] = float(((d.loc[full, "Tot_Drug_Cst"] - cst_sum).abs() <= 1.0).mean())

    lo, hi = brand_bounds(d, "Clms", 1, 10)
    assert (lo <= hi + 1e-9).all() and np.isfinite(hi).all()
    d["Brnd_Tot_Clms_lo"], d["Brnd_Tot_Clms_hi"] = lo, hi
    d["Brnd_Tot_Clms_mid"] = (lo + hi) / 2
    d["brnd_imputed"] = d["Brnd_Tot_Clms"].isna()
    d["brand_share_mid"] = d["Brnd_Tot_Clms_mid"] / d["Tot_Clms"]
    d["brand_share_lo"] = d["Brnd_Tot_Clms_lo"] / d["Tot_Clms"]
    d["brand_share_hi"] = d["Brnd_Tot_Clms_hi"] / d["Tot_Clms"]

    # cost: a '*' claims cell has unknown cost, so star bounds are [0, inf)
    clo, chi = brand_bounds(d, "Drug_Cst", 0.0, np.inf)
    d["Brnd_Tot_Drug_Cst_lo"], d["Brnd_Tot_Drug_Cst_hi"] = clo.round(2), chi.round(2)

    d["benes_suppressed"] = d["Tot_Benes"].isna()
    d["Tot_Benes_imp"] = d["Tot_Benes"].fillna(5.5)

    imp = d["brnd_imputed"]
    width = d["Brnd_Tot_Clms_hi"] - d["Brnd_Tot_Clms_lo"]
    st.update({
        "brnd_imputed_share": round(float(imp.mean()), 4),
        "brnd_clms_bound_width_imputed": {k: round(float(v), 2) for k, v in width[imp].describe().items()},
        "brnd_cost_exact_among_imputed": round(float((d["Brnd_Tot_Drug_Cst_hi"] - d["Brnd_Tot_Drug_Cst_lo"])[imp].le(1).mean()), 4),
        "benes_suppressed_share": round(float(d["benes_suppressed"].mean()), 4),
        "null_specialty": int(d["Prscrbr_Type"].isna().sum()),
        "rows_out": len(d),
    })
    d = d.drop(columns=["Gnrc_Sprsn_Flag", "Othr_Sprsn_Flag", "Othr_Tot_Clms", "Othr_Tot_Drug_Cst"])
    d["NPI"] = d["NPI"].astype("string")
    out = PROCESSED / f"partd_by_provider_{year}.parquet"
    d.to_parquet(out, index=False, compression="zstd")
    st.update({"out_file": out.name, "out_bytes": out.stat().st_size, "columns": list(d.columns)})
    (LOGS / f"partd_{year}.json").write_text(json.dumps(st, indent=2))
    print(json.dumps({k: v for k, v in st.items() if k != "columns"}, indent=2))


if __name__ == "__main__":
    PROCESSED.mkdir(parents=True, exist_ok=True)
    LOGS.mkdir(parents=True, exist_ok=True)
    years = PARTD_YEARS if sys.argv[1] == "all" else [int(sys.argv[1])]
    for y in years:
        print(f"== Part D {y}")
        main(y)
