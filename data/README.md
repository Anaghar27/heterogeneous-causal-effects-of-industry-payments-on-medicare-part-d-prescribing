# Data

Raw CMS files are read from `~/Downloads` and are never modified. Only the small derived files below are
kept in the project. `data/interim/` and `data/processed/` are git-ignored; regenerate them with:

```
pip install -r requirements.txt
python scripts/02_reduce_payments.py 2022
python scripts/02_reduce_payments.py 2023
python scripts/03_reduce_partd.py all
python scripts/04_link.py
python scripts/05_manifest.py
```

Built with Python 3.10.10, pandas 2.3.3, pyarrow 23.0.1, numpy 2.2.6.

## Raw inputs

| File | Size | SHA-256 | Source |
|---|---|---|---|
| `OP_DTL_GNRL_PGYR2022_P06302026_06032026.csv` | 7,439.2 MB | `6a3922a922af978db4d1d98c4d74db5bc3f34bc56b359854811e9ad7edc0d5ef` | https://openpaymentsdata.cms.gov (Datasets > General Payment Data, program year 2022) |
| `OP_DTL_GNRL_PGYR2023_P06302026_06032026.csv` | 8,231.9 MB | `b42cca116c8c1873f987bcd7a80931125409716a63a398ebfbf99c9de2537f79` | https://openpaymentsdata.cms.gov (Datasets > General Payment Data, program year 2023) |
| `Medicare_Part_D_Prescribers_by_Provider_2022.zip` | 240.0 MB | `b2bf4e7369cad384fcb32f52a2ad3a8ff6cd1b29f5c7a75640138b7ddd4fb011` | https://data.cms.gov (Medicare Part D Prescribers - by Provider, data year 2022) |
| `Medicare_Part_D_Prescribers_by_Provider_2023.zip` | 247.3 MB | `0396d20d656870f9b47a73eb6b58d9cdd6772c211d2fd400e424aed086f8ee57` | https://data.cms.gov (Medicare Part D Prescribers - by Provider, data year 2023) |
| `Medicare_Part_D_Prescribers_by_Provider_2024.zip` | 256.5 MB | `09ff4c9bc4adea357c3d5fce842ecabce1d33db523cbd7eb1da45f8c994cdc52` | https://data.cms.gov (Medicare Part D Prescribers - by Provider, data year 2024) |

## Step 2: Open Payments General Payments -> one row per NPI

Kept physicians and non-physician practitioners; dropped teaching hospitals and rows without a valid 10-digit NPI.
`payment_count` = number of payment records; `total_payment_usd` = sum of `Total_Amount_of_Payment_USDollars`.

| Year | Raw rows | Dropped: teaching hospital | Dropped: no/invalid NPI | Rows kept | USD kept | NPIs out | Slim rows file | NPI file |
|---|---|---|---|---|---|---|---|---|
| 2022 | 13,322,266 | 39,164 | 12,564 | 13,270,538 | $2,715,376,797 | 863,747 | `interim/payments_rows_2022.parquet` (193.6 MB) | `processed/payments_by_npi_2022.parquet` (15.8 MB) |
| 2023 | 14,734,121 | 32,700 | 12,867 | 14,688,554 | $2,514,715,918 | 935,088 | `interim/payments_rows_2023.parquet` (225.1 MB) | `processed/payments_by_npi_2023.parquet` (17.3 MB) |

## Step 3: Part D Prescribers by Provider -> one row per NPI

Blank brand cells are CMS suppression (`*` = 1-10 claims, `#` = counter-suppressed), never zero. Using
Tot = Brand + Generic + Other (verified exactly on fully reported rows), each blank brand cell gets
`Brnd_Tot_Clms_lo/_hi` bounds and a midpoint `Brnd_Tot_Clms_mid`; `brnd_imputed` flags these rows.
Blank `Tot_Benes` means 1-10; `Tot_Benes_imp` = 5.5 with flag `benes_suppressed`.

| Year | Rows (= NPIs) | Brand suppressed | Mean bound width (claims) | Tot_Benes suppressed | Output |
|---|---|---|---|---|---|
| 2022 | 1,332,309 | 45.8% | 5.5 | 10.8% | `processed/partd_by_provider_2022.parquet` (88.7 MB) |
| 2023 | 1,380,665 | 45.0% | 5.5 | 10.4% | `processed/partd_by_provider_2023.parquet` (91.6 MB) |
| 2024 | 1,416,883 | 43.9% | 5.5 | 10.3% | `processed/partd_by_provider_2024.parquet` (94.1 MB) |

## Step 4: link (Part D = left/universe, payments = right, key = NPI string)

No payment row = unpaid control. `treated` = `total_payment_usd > 0`.

| Output | Rows | Treated | Unpaid | Payment NPIs found in Part D | Payment $ found in Part D | Size |
|---|---|---|---|---|---|---|
| `processed/analytic_2023_2024.parquet` | 1,416,883 | 49.76% | 50.24% | 705,088 of 935,088 | 75.3% | 137.5 MB |
| `processed/analytic_2022_2023.parquet` | 1,380,665 | 47.65% | 52.35% | 657,843 of 863,747 | 74.5% | 105.9 MB |

`analytic_2023_2024` also carries the Part D 2022 baseline (`base22_*`; 83.48% of prescribers present in 2022) and 2022 payment history (`paid_2022`, `newly_paid_2023`; 45.59% paid in 2022).

## Files

| File | Rows | Columns |
|---|---|---|
| `interim/payments_rows_2022.parquet` | 13,270,538 | 18 |
| `interim/payments_rows_2023.parquet` | 14,688,554 | 18 |
| `processed/analytic_2022_2023.parquet` | 1,380,665 | 38 |
| `processed/analytic_2023_2024.parquet` | 1,416,883 | 53 |
| `processed/partd_by_provider_2022.parquet` | 1,332,309 | 26 |
| `processed/partd_by_provider_2023.parquet` | 1,380,665 | 26 |
| `processed/partd_by_provider_2024.parquet` | 1,416,883 | 26 |
| `processed/payments_by_npi_2022.parquet` | 863,747 | 11 |
| `processed/payments_by_npi_2023.parquet` | 935,088 | 11 |
