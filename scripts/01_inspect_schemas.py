"""Step 1: inspect schemas of the raw CMS files (first 100,000 rows only)."""
import sys
from pathlib import Path

import pandas as pd

DOWNLOADS = Path.home() / "Downloads"
PAYMENTS = DOWNLOADS / "OP_DTL_GNRL_PGYR2023_P06302026_06032026.csv"
PARTD = DOWNLOADS / "Medicare_Part_D_Prescribers_by_Provider_2024.zip"
NROWS = 100_000

pd.set_option("display.max_rows", 500)
pd.set_option("display.width", 200)


def inspect(label, path):
    print(f"\n{'=' * 80}\n{label}: {path}\n{'=' * 80}")
    df = pd.read_csv(path, nrows=NROWS, low_memory=False)
    print(f"shape (first {NROWS:,} rows): {df.shape}")
    summary = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "non_null": df.notna().sum(),
        "example": df.apply(lambda s: s.dropna().iloc[0] if s.notna().any() else None),
    })
    print(summary.to_string())
    return df


if __name__ == "__main__":
    pay = inspect("OPEN PAYMENTS GENERAL 2023", PAYMENTS)
    partd = inspect("PART D BY PROVIDER 2024", PARTD)

    # Quick checks relevant to the proposed mapping
    npi_cols = [c for c in pay.columns if "npi" in c.lower()]
    print("\nPayments NPI-like columns:", npi_cols)
    for c in npi_cols:
        print(f"  {c}: non-null={pay[c].notna().sum():,}, unique={pay[c].nunique():,}")
    for c in ["Covered_Recipient_Type", "Nature_of_Payment_or_Transfer_of_Value"]:
        if c in pay.columns:
            print(f"\n{c} value counts:\n{pay[c].value_counts(dropna=False).to_string()}")
    print("\nPart D NPI unique in sample:", partd.iloc[:, 0].nunique(), "of", len(partd))
    sys.exit(0)
