"""Step 5: record raw-file checksums and row counts at each step in data/README.md.

Run after steps 2-4 and BEFORE deleting any raw file: the SHA-256 is how a later
re-download can be checked against the exact version used here.
"""
import hashlib
import json
import sys

import numpy
import pandas
import pyarrow
import pyarrow.parquet as pq

from config import INTERIM, LOGS, PARTD_YEARS, PAYMENT_YEARS, PROCESSED, PROJECT, partd_raw, payments_raw

SOURCES = {
    "payments": "https://openpaymentsdata.cms.gov (Datasets > General Payment Data, program year {y})",
    "partd": "https://data.cms.gov (Medicare Part D Prescribers - by Provider, data year {y})",
}


def sha256(path, block=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(block):
            h.update(b)
    return h.hexdigest()


def mb(n):
    return f"{n / 1e6:,.1f} MB"


def main():
    checks_path = LOGS / "raw_checksums.json"
    checks = json.loads(checks_path.read_text()) if checks_path.exists() else {}
    raw = [("payments", y, payments_raw(y)) for y in PAYMENT_YEARS] + [("partd", y, partd_raw(y)) for y in PARTD_YEARS]
    for kind, y, p in raw:
        if p.exists() and p.name not in checks:  # keep recorded hashes after raw files are deleted
            print(f"hashing {p.name} ...", flush=True)
            checks[p.name] = {"bytes": p.stat().st_size, "sha256": sha256(p)}
    checks_path.write_text(json.dumps(checks, indent=2))

    L = ["# Data", "",
         "Raw CMS files are read from `~/Downloads` and are never modified. Only the small derived files below are",
         "kept in the project. `data/interim/` and `data/processed/` are git-ignored; regenerate them with:", "",
         "```", "pip install -r requirements.txt",
         *[f"python scripts/02_reduce_payments.py {y}" for y in PAYMENT_YEARS],
         "python scripts/03_reduce_partd.py all", "python scripts/04_link.py", "python scripts/05_manifest.py", "```", "",
         f"Built with Python {sys.version.split()[0]}, pandas {pandas.__version__}, pyarrow {pyarrow.__version__}, "
         f"numpy {numpy.__version__}.", "",
         "## Raw inputs", "", "| File | Size | SHA-256 | Source |", "|---|---|---|---|"]
    for kind, y, p in raw:
        c = checks.get(p.name, {})
        L.append(f"| `{p.name}` | {mb(c.get('bytes', 0))} | `{c.get('sha256', 'n/a')}` | {SOURCES[kind].format(y=y)} |")

    L += ["", "## Step 2: Open Payments General Payments -> one row per NPI", "",
          "Kept physicians and non-physician practitioners; dropped teaching hospitals and rows without a valid 10-digit NPI.",
          "`payment_count` = number of payment records; `total_payment_usd` = sum of `Total_Amount_of_Payment_USDollars`.", "",
          "| Year | Raw rows | Dropped: teaching hospital | Dropped: no/invalid NPI | Rows kept | USD kept | NPIs out | Slim rows file | NPI file |",
          "|---|---|---|---|---|---|---|---|---|"]
    for y in PAYMENT_YEARS:
        s = json.loads((LOGS / f"payments_{y}.json").read_text())
        L.append(f"| {y} | {s['rows_raw']:,} | {s['dropped_non_clinician']:,} | {s['dropped_bad_npi']:,} | {s['rows_kept']:,} | "
                 f"${s['usd_kept']:,.0f} | {s['npis_out']:,} | `interim/{s['slim_rows_file']}` ({mb(s['slim_bytes'])}) | "
                 f"`processed/{s['agg_file']}` ({mb(s['agg_bytes'])}) |")

    L += ["", "## Step 3: Part D Prescribers by Provider -> one row per NPI", "",
          "Blank brand cells are CMS suppression (`*` = 1-10 claims, `#` = counter-suppressed), never zero. Using",
          "Tot = Brand + Generic + Other (verified exactly on fully reported rows), each blank brand cell gets",
          "`Brnd_Tot_Clms_lo/_hi` bounds and a midpoint `Brnd_Tot_Clms_mid`; `brnd_imputed` flags these rows.",
          "Blank `Tot_Benes` means 1-10; `Tot_Benes_imp` = 5.5 with flag `benes_suppressed`.", "",
          "| Year | Rows (= NPIs) | Brand suppressed | Mean bound width (claims) | Tot_Benes suppressed | Output |",
          "|---|---|---|---|---|---|"]
    for y in PARTD_YEARS:
        s = json.loads((LOGS / f"partd_{y}.json").read_text())
        L.append(f"| {y} | {s['rows_out']:,} | {100 * s['brnd_imputed_share']:.1f}% | "
                 f"{s['brnd_clms_bound_width_imputed']['mean']:.1f} | {100 * s['benes_suppressed_share']:.1f}% | "
                 f"`processed/{s['out_file']}` ({mb(s['out_bytes'])}) |")

    link = json.loads((LOGS / "link.json").read_text())
    L += ["", "## Step 4: link (Part D = left/universe, payments = right, key = NPI string)", "",
          "No payment row = unpaid control. `treated` = `total_payment_usd > 0`.", "",
          "| Output | Rows | Treated | Unpaid | Payment NPIs found in Part D | Payment $ found in Part D | Size |",
          "|---|---|---|---|---|---|---|"]
    for name, s in link.items():
        L.append(f"| `processed/{name}.parquet` | {s['rows']:,} | {s['pct_treated']}% | {s['pct_unpaid']}% | "
                 f"{s['payment_npis_matched']:,} of {s['payment_npis']:,} | {100 * s['payment_usd_matched_share']:.1f}% | "
                 f"{mb(s['out_bytes'])} |")
    m = link["analytic_2023_2024"]
    L += ["", f"`analytic_2023_2024` also carries the Part D 2022 baseline (`base22_*`; {m['pct_in_partd_2022']}% of prescribers "
          f"present in 2022) and 2022 payment history (`paid_2022`, `newly_paid_2023`; {m['pct_paid_2022']}% paid in 2022).", ""]

    L += ["## Files", "", "| File | Rows | Columns |", "|---|---|---|"]
    for d in (INTERIM, PROCESSED):
        for f in sorted(d.glob("*.parquet")):
            if f.stem.endswith("_test"):
                continue
            md = pq.read_metadata(f)
            L.append(f"| `{f.relative_to(PROJECT / 'data').as_posix()}` | {md.num_rows:,} | {md.num_columns} |")
    (PROJECT / "data" / "README.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
