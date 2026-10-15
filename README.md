# For Whom Do Industry Payments Change Prescribing? Heterogeneous Causal Effects of Pharmaceutical Payments on Brand Name Prescribing in Medicare Part D

Pharmaceutical manufacturers routinely make payments to physicians, such as sponsored meals, speaking fees, and consulting fees, and these payments are publicly reported through the Centers for Medicare & Medicaid Services (CMS) Open Payments program. Prior studies have shown that physicians who receive such payments prescribe more brand name drugs [1, 2]. This association, however, mixes the effect of the payment with the fact that manufacturers target high volume specialists who would prescribe more brand name drugs regardless. A single average effect also hides the question that matters most to payers and regulators: for which prescribers do payments change behavior the most? Our domain problem is therefore to provide an honest, segmented estimate of how much industry payments shift brand name prescribing in Medicare Part D. Our analytic problem is to estimate, from confounded observational data, both the average treatment effect (ATE) and the conditional average treatment effect (CATE) of receiving at least one industry payment on subsequent brand name prescribing, and to identify the prescriber characteristics that modify that effect.

We will link two public CMS datasets at the prescriber level using each prescriber's National Provider Identifier (NPI): Open Payments, which records industry payments by type, amount, and associated drug [3], and the Medicare Part D Prescribers files, which record prescribing volume and cost by drug [4]. In nontechnical terms, we will first model which prescribers receive payments, reweight the data so that paid and unpaid prescribers look alike on specialty, region, experience, and prior prescribing, and then use causal machine learning methods to estimate how much the payment changes prescribing for each type of prescriber. Whereas prior work has largely reported average associations, our contribution is a validated causal estimate for each subgroup with explicit robustness bounds against unmeasured confounding. All analysis is conducted at the population level, and we make no claims about individual prescribers.

**DS5500 Data Science Capstone, Group 16:** Anagha Raveendra, Rohan Chandrashekhar, Sri Sai Charan Yarlagadda. Sponsor: Prof. Kylie Bemis.

## Project status

| Milestone | Dates | Status |
|---|---|---|
| Week 1: data pipeline, NPI linkage, cohort, EDA on real CMS data | Oct 12-23 | Complete |
| Weeks 2-3: calibrated propensity models, weighting, overlap diagnostics, IHDP benchmark | Oct 26 - Nov 6 | Next |
| Later: doubly robust ATE and CATE estimation, sensitivity analysis | | Planned |

## Study design

| Year | Role | Definition |
|---|---|---|
| 2022 | Baseline | All covariates measured before payment (Part D 2022) |
| 2023 | Treatment | Any Open Payments General payment > $0; payment amount used as a dose |
| 2024 | Outcome | Brand-name share of the prescriber's Part D claims |

A second cohort (payments 2022 -> prescribing 2023) is also built, without a baseline year.

## Data

All inputs are public CMS files. Download them into `~/Downloads` with the names below (the scripts read them from there and never modify them). `scripts/config.py` holds the exact file names.

| Dataset | Years | File(s) | Source |
|---|---|---|---|
| Open Payments, General Payment Data | 2022, 2023 | `OP_DTL_GNRL_PGYR{year}_P06302026_06032026.csv` (about 7-8 GB each) | https://openpaymentsdata.cms.gov |
| Medicare Part D Prescribers - by Provider | 2022, 2023, 2024 | `Medicare_Part_D_Prescribers_by_Provider_{year}.zip` (about 250 MB each) | https://data.cms.gov |

Use the **by Provider** Part D file (one row per prescriber), not *by Provider and Drug*. CMS republishes Open Payments as records are corrected; the SHA-256 checksums of the exact files used are in [data/README.md](data/README.md), so a re-download can be verified.

## How to run

```
pip install -r requirements.txt
python scripts/01_inspect_schemas.py          # optional: print raw schemas
python scripts/02_reduce_payments.py 2022
python scripts/02_reduce_payments.py 2023     # each ~12 min, ~1.5 GB RAM (chunked)
python scripts/03_reduce_partd.py all
python scripts/04_link.py
python scripts/05_manifest.py                 # run before deleting any raw file
python scripts/06_low_payment_check.py
python scripts/07_clean_features.py
python scripts/08_eda.py
python scripts/09_feature_check.py
python scripts/10_eda_extended.py
```

Built with Python 3.10; package versions are pinned in [requirements.txt](requirements.txt).

## Pipeline

| Script | What it does | Main output |
|---|---|---|
| `01_inspect_schemas.py` | Reads the first 100k rows of each raw file and prints columns and types | console |
| `02_reduce_payments.py` | Reads each Open Payments CSV in 1M-row chunks; keeps physicians and non-physician practitioners; aggregates to one row per NPI | `data/processed/payments_by_npi_{year}.parquet`, slim row-level `data/interim/payments_rows_{year}.parquet` |
| `03_reduce_partd.py` | Keeps the needed Part D columns; bounds suppressed brand counts | `data/processed/partd_by_provider_{year}.parquet` |
| `04_link.py` | Joins Part D (left) to payments (right) on NPI; adds 2022 baseline and payment history | `data/processed/analytic_2023_2024.parquet`, `analytic_2022_2023.parquet` |
| `05_manifest.py` | Records raw-file checksums and row counts at each step | [data/README.md](data/README.md) |
| `06_low_payment_check.py` | Screens whether small payments move prescribing (dose-response) | `data/logs/low_payment_check.json` |
| `07_clean_features.py` | Applies exclusions and builds pre-treatment features, treatment and outcome columns | `data/processed/model_2023_2024.parquet` (**main modeling table**) |
| `08_eda.py` | Exploratory analysis | [reports/eda_summary.md](reports/eda_summary.md), [reports/figures/](reports/figures/) |
| `09_feature_check.py` | Tests whether unused Part D columns (race, plan mix, drug class, fills) matter as confounders | `data/logs/feature_check.json` |
| `10_eda_extended.py` | Outcome drivers, treatment separability, propensity-score matching | `reports/report/` (local) |
| `11_build_report.py` | Builds a local PDF summary of the steps above | `reports/report/` (local) |

`data/interim/`, `data/processed/` and `reports/report/` are git-ignored and are regenerated by the scripts. Only code, aggregate logs and aggregate figures are committed.

## Key design decisions

- **Join key.** NPIs are read as 10-digit text in both files. Numeric parsing turns `1234567890` into `1234567890.0` and silently breaks the join.
- **Universe.** Part D is the left table, so it defines who could prescribe. A prescriber with no payment row is unpaid (control), not missing.
- **Recipients.** Physicians and non-physician practitioners (NPs, PAs) are kept; teaching hospitals and rows without a valid NPI are dropped.
- **Treatment.** Any 2023 payment > $0. A screen (script 06) showed that even small payments are associated with a small rise in brand share, growing with the amount, so no cutoff is applied; the amount is modeled as a dose.
- **Suppressed counts.** CMS hides counts of 1-10. Because Total = Brand + Generic + Other holds exactly, each hidden brand count is bounded (at most 9 claims wide). Each prescriber gets low, middle and high values; the middle is the main outcome and results are repeated at both bounds. Prescribers whose share bounds are wider than 10 points are flagged (`y_share_uninformative`) and excluded from share outcomes.
- **Exclusions.** Organizations, missing specialty, and prescriber types paid less than 1% of the time (pharmacists, therapists, aides), for whom no treated comparison exists.
- **Covariates.** All measured in 2022, before treatment: practice size, cost per claim, patient age, risk, sex and dual-eligible mix, 2022 brand share, opioid rate, specialty (73 groups), region and rural/urban (RUCA). 2024 totals are not used as covariates because they are measured after treatment.

## Week 1 results (descriptive, not yet causal)

| | |
|---|---|
| Payment records read (2022 + 2023) | 28.1M; $5.2B kept for physicians and practitioners |
| Final cohort | **1,383,580 prescribers**, 51.0% paid in 2023 |
| Typical payment | median $210 per paid prescriber; 77% received food only |
| Paid rate by specialty | 14% (trainees) to 83% (orthopedic surgery) |
| Persistence of prescribing | 2022 brand share predicts 2024 (r = 0.81) |
| Propensity model | AUC 0.755 (logistic), 0.782 (gradient boosting); 94.9% of prescribers between 0.1 and 0.9 |
| Paid minus unpaid brand share, 2024 | **3.9 points raw, 0.6 points after matching** within specialty |
| Dose-response (newly paid vs never paid) | +0.17 ($1-25), +0.37 ($25-100), +0.79 ($100-1k), +1.00 points (>$1k) |

Most of the raw gap reflects who is paid (specialty, practice size, prior prescribing). A small association that grows with the payment amount remains. Full tables and figures are in [reports/eda_summary.md](reports/eda_summary.md).

## Limitations

- **Unmeasured confounding.** Manufacturers may target prescribers already about to adopt new brand-name drugs; a sensitivity analysis is planned [14].
- **Suppression.** Results must be reported at the low, middle and high bounds; brand cost cannot be bounded usefully and is a secondary outcome only.
- **No baseline for new prescribers.** 16% of prescribers are new in 2024 and lack 2022 data.
- **Experience.** Years in practice is not in these files; it would need the NPPES registry.

## Next steps (Oct 26 - Nov 6)

1. Calibrated propensity models (logistic and gradient boosting) with trimming of extreme scores.
2. Inverse probability weighting with every covariate SMD below 0.1, shown in a love plot.
3. Overlap diagnostics by specialty, focusing on trainees, cardiology and ophthalmology.
4. Benchmark the estimation pipeline on IHDP, where true effects are known [11, 12].

## References
[1] DeJong C, Aguilar T, Tseng CW, Lin GA, Boscardin WJ, Dudley RA. Pharmaceutical industry sponsored meals and physician prescribing patterns for Medicare beneficiaries. JAMA Internal Medicine, 176(8), 1114–1122, 2016.
[2] Yeh JS, Franklin JM, Avorn J, Landon J, Kesselheim AS. Association of industry payments to physicians with the prescribing of brand name statins in Massachusetts. JAMA Internal Medicine, 176(6), 763–768, 2016.
[3] Centers for Medicare & Medicaid Services. Open Payments Data, General Payment Data, Program Years 2022 and 2023. https://openpaymentsdata.cms.gov
[4] Centers for Medicare & Medicaid Services. Medicare Part D Prescribers by Provider, 2022, 2023 and 2024. https://data.cms.gov
[5] Battocchi K, Dillon E, Hei M, Lewis G, Oka P, Oprescu M, Syrgkanis V. EconML: A Python package for ML based heterogeneous treatment effects estimation. PyWhy, 2019.
[6] Wager S, Athey S. Estimation and inference of heterogeneous treatment effects using random forests. Journal of the American Statistical Association, 113(523), 1228–1242, 2018.
[7] Kennedy EH. Towards optimal doubly robust estimation of heterogeneous causal effects. Electronic Journal of Statistics, 17(2), 3008–3049, 2023.
[8] Künzel SR, Sekhon JS, Bickel PJ, Yu B. Metalearners for estimating heterogeneous treatment effects using machine learning. PNAS, 116(10), 4156–4165, 2019.
[9] van der Laan MJ, Rubin D. Targeted maximum likelihood learning. International Journal of Biostatistics, 2(1), 2006.
[10] Semenova V, Chernozhukov V. Debiased machine learning of conditional average treatment effects and other causal functions. The Econometrics Journal, 24(2), 264–289, 2021.
[11] Hill JL. Bayesian nonparametric modeling for causal inference. Journal of Computational and Graphical Statistics, 20(1), 217–240, 2011.
[12] Dorie V, Hill J, Shalit U, Scott M, Cervone D. Automated versus do it yourself methods for causal inference: Lessons learned from a data analysis competition. Statistical Science, 34(1), 43–68, 2019.
[13] Louizos C, Shalit U, Mooij JM, Sontag D, Zemel R, Welling M. Causal effect inference with deep latent variable models. NeurIPS, 2017.
[14] Cinelli C, Hazlett C. Making sense of sensitivity: Extending omitted variable bias. Journal of the Royal Statistical Society, Series B, 82(1), 39–67, 2020.
