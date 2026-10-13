"""Step 2: reduce Open Payments General Payments to (a) slim row-level parquet and (b) one row per NPI.

Usage: python scripts/02_reduce_payments.py 2023 [--nrows N]
Reads the raw CSV in chunks; never loads the whole file.
"""
import argparse
import json
import time

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from config import INTERIM, LOGS, PROCESSED, clean_npi, payments_raw

CHUNKSIZE = 1_000_000
KEEP_TYPES = {"Covered Recipient Physician", "Covered Recipient Non-Physician Practitioner"}

# raw column -> slim name
COLS = {
    "Covered_Recipient_NPI": "npi",
    "Covered_Recipient_Type": "recipient_type",
    "Covered_Recipient_Specialty_1": "op_specialty",
    "Total_Amount_of_Payment_USDollars": "amount_usd",
    "Number_of_Payments_Included_in_Total_Amount": "n_payments_in_record",
    "Date_of_Payment": "payment_date",
    "Nature_of_Payment_or_Transfer_of_Value": "nature",
    "Form_of_Payment_or_Transfer_of_Value": "form",
    "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_ID": "mfr_id",
    "Applicable_Manufacturer_or_Applicable_GPO_Making_Payment_Name": "mfr_name",
    "Indicate_Drug_or_Biological_or_Device_or_Medical_Supply_1": "product_type_1",
    "Product_Category_or_Therapeutic_Area_1": "product_category_1",
    "Name_of_Drug_or_Biological_or_Device_or_Medical_Supply_1": "product_name_1",
    "Associated_Drug_or_Biological_NDC_1": "ndc_1",
    "Dispute_Status_for_Publication": "disputed",
    "Change_Type": "change_type",
    "Record_ID": "record_id",
    "Program_Year": "program_year",
}
STR_COLS = [c for c in COLS.values() if c not in {"amount_usd", "n_payments_in_record", "payment_date", "record_id", "program_year"}]
SCHEMA = pa.schema(
    [(c, pa.string()) for c in ["npi", "recipient_type", "op_specialty"]]
    + [("amount_usd", pa.float64()), ("n_payments_in_record", pa.int32()), ("payment_date", pa.date32())]
    + [(c, pa.string()) for c in ["nature", "form", "mfr_id", "mfr_name", "product_type_1",
                                  "product_category_1", "product_name_1", "ndc_1", "disputed", "change_type"]]
    + [("record_id", pa.int64()), ("program_year", pa.int16())]
)


def add_counts(acc, s):
    for k, v in s.value_counts(dropna=False).items():
        k = "<NA>" if pd.isna(k) else str(k)
        acc[k] = acc.get(k, 0) + int(v)


def main(year, nrows=None):
    src = payments_raw(year)
    for d in (INTERIM, PROCESSED, LOGS):
        d.mkdir(parents=True, exist_ok=True)
    sfx = "_test" if nrows else ""
    slim_path = INTERIM / f"payments_rows_{year}{sfx}.parquet"
    agg_path = PROCESSED / f"payments_by_npi_{year}{sfx}.parquet"
    t0 = time.time()

    stats = {"source_file": src.name, "source_bytes": src.stat().st_size, "rows_raw": 0,
             "recipient_type": {}, "program_year": {}, "change_type": {}, "nature_raw": {},
             "dropped_non_clinician": 0, "dropped_bad_npi": 0, "unparsed_dates": 0,
             "negative_amounts": 0, "rows_kept": 0, "usd_kept": 0.0}
    partials, pairs = [], []
    tmp_path = slim_path.with_suffix(".parquet.partial")
    writer = pq.ParquetWriter(tmp_path, SCHEMA, compression="zstd")

    reader = pd.read_csv(src, usecols=list(COLS), dtype=str, chunksize=CHUNKSIZE, nrows=nrows,
                         keep_default_na=True)
    for i, ch in enumerate(reader):
        ch = ch.rename(columns=COLS)[list(COLS.values())]
        stats["rows_raw"] += len(ch)
        add_counts(stats["recipient_type"], ch["recipient_type"])
        add_counts(stats["program_year"], ch["program_year"])
        add_counts(stats["change_type"], ch["change_type"])

        clin = ch["recipient_type"].isin(KEEP_TYPES)
        stats["dropped_non_clinician"] += int((~clin).sum())
        ch = ch[clin].copy()
        ch["npi"] = clean_npi(ch["npi"])
        bad = ch["npi"].isna()
        stats["dropped_bad_npi"] += int(bad.sum())
        ch = ch[~bad]

        ch["amount_usd"] = pd.to_numeric(ch["amount_usd"], errors="raise")
        ch["n_payments_in_record"] = pd.to_numeric(ch["n_payments_in_record"], errors="raise").astype("int32")
        ch["record_id"] = pd.to_numeric(ch["record_id"], errors="raise").astype("int64")
        ch["program_year"] = pd.to_numeric(ch["program_year"], errors="raise").astype("int16")
        dates = pd.to_datetime(ch["payment_date"], format="%m/%d/%Y", errors="coerce")
        stats["unparsed_dates"] += int(dates.isna().sum())
        ch["payment_date"] = dates.dt.date
        stats["negative_amounts"] += int((ch["amount_usd"] < 0).sum())
        add_counts(stats["nature_raw"], ch["nature"])
        stats["rows_kept"] += len(ch)
        stats["usd_kept"] += float(ch["amount_usd"].sum())

        writer.write_table(pa.Table.from_pandas(ch, schema=SCHEMA, preserve_index=False))

        food = ch["nature"].eq("Food and Beverage")
        g = ch.assign(
            food_usd=ch["amount_usd"].where(food, 0.0),
            food_n=food.astype("int64"),
            is_physician=ch["recipient_type"].eq("Covered Recipient Physician"),
        ).groupby("npi")
        partials.append(pd.DataFrame({
            "total_payment_usd": g["amount_usd"].sum(),
            "payment_count": g.size(),
            "n_payments_reported": g["n_payments_in_record"].sum(),
            "food_bev_usd": g["food_usd"].sum(),
            "food_bev_count": g["food_n"].sum(),
            "max_single_payment_usd": g["amount_usd"].max(),
            "first_payment_date": g["payment_date"].min(),
            "last_payment_date": g["payment_date"].max(),
            "op_is_physician": g["is_physician"].max(),
        }))
        pairs.append(ch[["npi", "mfr_id"]].drop_duplicates())
        print(f"  chunk {i + 1}: {stats['rows_raw']:,} rows read, {time.time() - t0:,.0f}s", flush=True)
    writer.close()
    tmp_path.replace(slim_path)  # only a fully written file gets the final name

    p = pd.concat(partials)
    g = p.groupby(level=0)
    agg = pd.DataFrame({
        "total_payment_usd": g["total_payment_usd"].sum(),
        "payment_count": g["payment_count"].sum(),
        "n_payments_reported": g["n_payments_reported"].sum(),
        "food_bev_usd": g["food_bev_usd"].sum(),
        "food_bev_count": g["food_bev_count"].sum(),
        "max_single_payment_usd": g["max_single_payment_usd"].max(),
        "first_payment_date": g["first_payment_date"].min(),
        "last_payment_date": g["last_payment_date"].max(),
        "op_is_physician": g["op_is_physician"].max(),
    })
    mfr = pd.concat(pairs).drop_duplicates().groupby("npi").size().rename("n_distinct_manufacturers")
    agg = agg.join(mfr).rename_axis("NPI").reset_index()
    agg["NPI"] = agg["NPI"].astype("string")
    for c in ("first_payment_date", "last_payment_date"):
        agg[c] = pd.to_datetime(agg[c])
    agg.to_parquet(agg_path, index=False, compression="zstd")

    assert agg["NPI"].is_unique
    assert abs(agg["total_payment_usd"].sum() - stats["usd_kept"]) < 1e-3 * max(1.0, stats["usd_kept"])
    assert agg["payment_count"].sum() == stats["rows_kept"]
    stats.update({"npis_out": len(agg), "slim_rows_file": slim_path.name, "slim_bytes": slim_path.stat().st_size,
                  "agg_file": agg_path.name, "agg_bytes": agg_path.stat().st_size, "seconds": round(time.time() - t0)})
    (LOGS / f"payments_{year}{sfx}.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items() if k != "nature_raw"}, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("year", type=int)
    ap.add_argument("--nrows", type=int, default=None, help="test on the first N rows only")
    a = ap.parse_args()
    main(a.year, a.nrows)
