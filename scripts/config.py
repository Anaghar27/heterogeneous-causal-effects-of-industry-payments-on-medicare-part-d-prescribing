"""Shared paths and settings for the data pipeline."""
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RAW = Path.home() / "Downloads"  # raw CMS files: read-only, never written to
INTERIM = PROJECT / "data" / "interim"  # slim row-level payments (re-aggregation without the 8 GB raw)
PROCESSED = PROJECT / "data" / "processed"  # analysis-ready, one row per NPI
LOGS = PROJECT / "data" / "logs"

PAYMENT_YEARS = (2022, 2023)
PARTD_YEARS = (2022, 2023, 2024)


def payments_raw(year):
    return RAW / f"OP_DTL_GNRL_PGYR{year}_P06302026_06032026.csv"


def partd_raw(year):
    return RAW / f"Medicare_Part_D_Prescribers_by_Provider_{year}.zip"


def clean_npi(s):
    """Return NPI as a 10-digit string; anything else becomes <NA>."""
    s = s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    return s.where(s.str.fullmatch(r"\d{10}"))
