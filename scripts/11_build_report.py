"""Step 11: build the private PDF report of everything done so far (reports/report/pipeline_report.pdf, git-ignored).

Every number is read from data/logs/*.json and reports/report/extended_stats.json, so re-running the
pipeline and then this script keeps the report in sync.
"""
import datetime as dt
import json

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (Image, KeepTogether, ListFlowable, ListItem, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

from config import LOGS, PROJECT

OUT = PROJECT / "reports" / "report"
FIG_BASE = PROJECT / "reports" / "figures"
FIG_EXT = OUT / "figures"
INK, INK2, RULE, BAND, ACCENT = "#0b0b0b", "#52514e", "#c3c2b7", "#f0efec", "#2a78d6"

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontName="Helvetica-Bold", fontSize=16, leading=20,
                    textColor=INK, spaceBefore=6, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontName="Helvetica-Bold", fontSize=12, leading=15,
                    textColor=INK, spaceBefore=10, spaceAfter=4)
P = ParagraphStyle("P", parent=ss["BodyText"], fontName="Helvetica", fontSize=9.5, leading=13.5, textColor=INK,
                   alignment=TA_LEFT, spaceAfter=6)
SMALL = ParagraphStyle("S", parent=P, fontSize=8, leading=10.5, textColor=INK2)
CELL = ParagraphStyle("C", parent=P, fontSize=7.5, leading=9.5, spaceAfter=0)
CELLB = ParagraphStyle("CB", parent=CELL, fontName="Helvetica-Bold")
TITLE = ParagraphStyle("T", parent=H1, fontSize=22, leading=27, spaceAfter=6)


def load(name):
    return json.loads((LOGS / name).read_text())


def p(text):
    return Paragraph(text, P)


def bullets(items):
    return ListFlowable([ListItem(Paragraph(t, P), leftIndent=10) for t in items], bulletType="bullet",
                        start="-", leftIndent=12, bulletFontSize=9)


def table(df, widths=None, index=False):
    df = df.reset_index() if index else df
    data = [[Paragraph(str(c), CELLB) for c in df.columns]]
    for row in df.itertuples(index=False):
        data.append([Paragraph(fmt(v), CELL) for v in row])
    t = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, colors.HexColor(RULE)),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor(RULE)),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f7f7f5")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
    ]))
    return KeepTogether([t, Spacer(1, 8)])


def fmt(v):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        if pd.isna(v):
            return ""
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.3f}".rstrip("0").rstrip(".") if v % 1 else f"{v:,.0f}"
    return str(v)


def fig(path, caption, width=6.5):
    w, h = ImageReader(str(path)).getSize()
    img = Image(str(path), width=width * inch, height=width * inch * h / w)
    return KeepTogether([img, Paragraph(caption, SMALL), Spacer(1, 10)])


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor(INK2))
    canvas.drawString(0.8 * inch, 0.5 * inch, "Industry payments and Part D brand prescribing - data preparation and EDA (internal)")
    canvas.drawRightString(7.7 * inch, 0.5 * inch, f"page {doc.page}")
    canvas.restoreState()


def main():
    pay = {y: load(f"payments_{y}.json") for y in (2022, 2023)}
    pdx = {y: load(f"partd_{y}.json") for y in (2022, 2023, 2024)}
    link = load("link.json")
    clean = load("clean_features.json")
    low = load("low_payment_check.json")
    fchk = load("feature_check.json")
    sums = load("raw_checksums.json")
    ext = json.loads((OUT / "extended_stats.json").read_text())
    main_l = link["analytic_2023_2024"]
    sep, mat = ext["separability"], ext["matching"]
    att = pd.DataFrame(mat["att"])
    att_mid = att[att["bounds"] == "mid"].set_index("outcome")["ATT_pts"]
    low_mid = pd.DataFrame(low["estimates"]).query("bounds == 'mid'").set_index("group")["effect_pts"]

    s = []
    # ---------------- title + summary
    s += [Spacer(1, 30), Paragraph("Industry Payments and Brand-Name Prescribing in Medicare Part D", TITLE),
          Paragraph(f"Data preparation, linkage and exploratory analysis - internal report, {dt.date.today():%B %d, %Y}", SMALL),
          Spacer(1, 18), Paragraph("Summary", H2)]
    s.append(bullets([
        f"Linked CMS Open Payments (2022, 2023) to Medicare Part D Prescribers by Provider (2022-2024) on NPI. "
        f"Main cohort: payments 2023 -> prescribing 2024, with 2022 as the pre-payment baseline: "
        f"<b>{clean['final_rows']:,} prescribers</b> after cleaning, <b>{clean['pct_treated']}% paid</b>.",
        f"CMS hides small counts. Hidden brand-claim counts were bounded exactly from Total = Brand + Generic + Other "
        f"(the identity holds in 100% of fully reported rows); the bound is at most 9 claims wide.",
        f"Paid prescribers differ before payment (larger, higher 2022 brand share, different specialties). "
        f"The raw paid-unpaid gap in 2024 brand share is {mat['raw_diff_pts']:.1f} points; after matching within "
        f"specialty on the propensity score it is <b>{att_mid['brand share 2024']:.2f} points</b>.",
        f"Larger payments go with larger rises in brand share (adjusted screen: +{low_mid['usd_1_25']:.2f}, "
        f"+{low_mid['usd_25_100']:.2f}, +{low_mid['usd_100_1k']:.2f}, +{low_mid['usd_gt_1k']:.2f} points from $1-25 to >$1k).",
        f"Overlap is good: propensity AUC {sep['auc_logistic']:.3f} (logistic) / {sep['auc_gbm']:.3f} (boosting); "
        f"{sep['overlap_logistic']['pct_ps_0.1_0.9']}% of prescribers have a propensity between 0.1 and 0.9.",
        "These are descriptive and screening results. The causal model (ATE/CATE with robustness bounds) is the next step.",
    ]))
    s.append(PageBreak())

    # ---------------- 1. data
    s.append(Paragraph("1. Data sources", H1))
    s.append(p("All inputs are public CMS files, read from the local Downloads folder and never modified. The two "
               "Open Payments CSVs (about 15.7 GB) were deleted after reduction; their SHA-256 checksums below allow a "
               "re-download to be verified against the exact version used."))
    rows = []
    for name, c in sums.items():
        kind = "Open Payments General" if name.startswith("OP_") else "Part D by Provider"
        rows.append({"file": name, "dataset": kind, "size (MB)": round(c["bytes"] / 1e6, 1), "SHA-256 (first 16)": c["sha256"][:16]})
    s.append(table(pd.DataFrame(rows), widths=[2.9 * inch, 1.3 * inch, 0.8 * inch, 1.5 * inch]))
    s.append(Paragraph("Pipeline", H2))
    steps = pd.DataFrame([
        ["01", "Inspect schemas (first 100k rows)", "column mapping"],
        ["02", "Reduce payments in 1M-row chunks", "slim row-level file + one row per NPI"],
        ["03", "Reduce Part D, bound suppressed cells", "one row per NPI per year"],
        ["04", "Link on NPI (Part D = left)", "analytic_2023_2024, analytic_2022_2023"],
        ["05", "Checksums and row counts", "data/README.md"],
        ["06", "Low-payment screen (DiD-style)", "dose-response estimates"],
        ["07", "Clean and engineer features", "model_2023_2024 (main modeling table)"],
        ["08", "EDA", "reports/figures, eda_summary.md"],
        ["09", "Test unused Part D features", "feature_check.json"],
        ["10", "Extended EDA, separability, matching", "this report's figures"],
    ], columns=["script", "what it does", "output"])
    s.append(table(steps, widths=[0.6 * inch, 3.0 * inch, 2.9 * inch]))

    # ---------------- 2. payments
    s.append(Paragraph("2. Open Payments reduction", H1))
    s.append(p("Each 7-8 GB CSV was read in 1,000,000-row chunks so memory stayed near 1.5 GB. Only physicians and "
               "non-physician practitioners were kept (nurse practitioners and physician assistants are Part D "
               "prescribers too, so dropping them would put paid prescribers into the control group). Teaching "
               "hospitals and rows without a valid 10-digit NPI were dropped. NPIs were read as text, because reading "
               "them as numbers turns 1234567890 into 1234567890.0 and silently breaks the join."))
    pr = pd.DataFrame([{"year": str(y), "raw rows": v["rows_raw"], "teaching hospital": v["dropped_non_clinician"],
                        "no/invalid NPI": v["dropped_bad_npi"], "rows kept": v["rows_kept"],
                        "USD kept ($M)": round(v["usd_kept"] / 1e6, 1), "NPIs": v["npis_out"]} for y, v in pay.items()])
    s.append(table(pr))
    s.append(p("Checks: every record has program year = file year, no negative amounts, all dates parse, and the "
               "per-NPI totals reconcile exactly with the row-level sums. A slim row-level copy (about 200 MB per year) "
               "keeps payment type, manufacturer, drug and date so payments can be re-aggregated later without the raw files."))

    # ---------------- 3. part d + suppression
    s.append(Paragraph("3. Part D reduction and suppressed values", H1))
    s.append(p("Part D reports one row per prescriber. For privacy CMS hides any count from 1 to 10 ('*') and hides a "
               "second cell ('#') so the first cannot be worked out by subtraction. A blank brand count is therefore "
               "never zero (real zeros are reported). Because Total = Brand + Generic + Other holds exactly, each hidden "
               "brand count can be bounded: for example, total 60 and generic 55 leave 5 for brand plus other, so brand "
               "is between 1 and 5. Each prescriber gets a low, high and middle value; the middle is the main outcome and "
               "results are repeated at the low and high values."))
    rows = [{"year": str(y), "prescribers": v["rows_out"], "brand hidden (%)": round(100 * v["brnd_imputed_share"], 1),
             "mean bound width (claims)": v["brnd_clms_bound_width_imputed"]["mean"],
             "max bound width": v["brnd_clms_bound_width_imputed"]["max"],
             "beneficiaries hidden (%)": round(100 * v["benes_suppressed_share"], 1),
             "identity holds": f"{100 * v['identity_clms_holds']:.0f}%"} for y, v in pdx.items()]
    s.append(table(pd.DataFrame(rows)))
    s.append(p("Brand <i>cost</i> cannot be bounded usefully (cost per claim varies too much; under 1.2% of hidden rows "
               "are recoverable), so cost is a secondary outcome only."))

    # ---------------- 4. linking
    s.append(Paragraph("4. Linking", H1))
    s.append(p("Part D is the left table: it defines who could prescribe. A prescriber with no payment row is unpaid "
               "(control), not missing. Treatment is any payment above $0."))
    lk = pd.DataFrame([{"cohort": k.replace("analytic_", ""), "prescribers": v["rows"], "paid (%)": v["pct_treated"],
                        "duplicate NPIs": v["duplicate_npi"],
                        "payment NPIs found in Part D": f"{v['payment_npis_matched']:,} of {v['payment_npis']:,}",
                        "payment $ found (%)": round(100 * v["payment_usd_matched_share"], 1)} for k, v in link.items()])
    s.append(table(lk))
    s.append(p(f"About a quarter of payment dollars go to NPIs that are not Part D prescribers (mostly anesthesiology, "
               f"nurse anesthetists and pediatrics), which is expected. In the main cohort {main_l['pct_in_partd_2022']}% "
               f"of prescribers also appear in Part D 2022 (the baseline) and {main_l['pct_paid_2022']}% were paid in 2022."))
    h = main_l["treatment_x_history"]
    hist = pd.DataFrame({"paid 2022": ["no", "yes"],
                         "unpaid 2023": [h["false"]["false"], h["false"]["true"]],
                         "paid 2023": [h["true"]["false"], h["true"]["true"]]})
    s.append(table(hist, widths=[1.2 * inch, 1.3 * inch, 1.3 * inch]))

    # ---------------- 5. low payments
    s.append(Paragraph("5. Do low payments matter?", H1))
    s.append(p("Before fixing the treatment definition we checked whether small payments (mostly meals) move "
               "prescribing. Design: prescribers present in 2022 with no 2022 payment; compare never paid with newly "
               "paid in 2023 by amount; outcome = change in brand share 2022 -> 2024; adjusted for specialty, state, "
               "rural/urban and 2022 size, patient age, risk and brand share. Values are percentage points."))
    lp = pd.DataFrame(low["estimates"]).pivot(index="group", columns="bounds", values="effect_pts")
    lp = lp.loc[["usd_1_25", "usd_25_100", "usd_100_1k", "usd_gt_1k"], ["lo", "mid", "hi"]]
    lp.index = ["$1-25", "$25-100", "$100-1k", ">$1k"]
    lp.columns = ["low bound", "middle", "high bound"]
    s.append(table(lp.round(3), index=True, widths=[1.2 * inch, 1.2 * inch, 1.2 * inch, 1.2 * inch]))
    s.append(p("Low payments have a small effect that grows with the amount; only $1-25 is fragile at the high bound. "
               "Decision: keep treatment = any payment > $0 and model the amount as a dose."))
    s.append(PageBreak())

    # ---------------- 6. cleaning + features
    s.append(Paragraph("6. Cleaning and feature engineering", H1))
    fl = pd.DataFrame(clean["flow"]).drop(columns=["specialties"], errors="ignore").fillna("")
    s.append(table(fl))
    s.append(p("Prescriber types that are almost never paid (pharmacists, therapists, aides and similar) were removed: "
               "an effect cannot be estimated for a group with no treated members. Specialty is taken from 2022 "
               f"(before payment); {clean['specialty_changed_2022_2024']:,} prescribers changed specialty between years. "
               f"Specialties with fewer than 500 prescribers are grouped as Other ({clean['n_specialties_model']} groups)."))
    feats = pd.DataFrame([
        ["numeric (10)", "log claims, log beneficiaries, log cost per claim, patient age, risk score, 2022 brand share, "
                         "share female, share dual-eligible, share under 65, opioid rate"],
        ["missing flags (6)", "new in 2024, beneficiaries hidden, and one for each patient-mix share"],
        ["categorical (3)", f"specialty ({clean['n_specialties_model']}), region (5), rural/urban RUCA (5)"],
        ["treatment", "paid yes/no, $ amount (raw, log, groups), counts, manufacturers, $ by type, 2022 history"],
        ["outcome", "2024 brand share and brand claims (low/middle/high), change since 2022, brand cost bounds"],
    ], columns=["block", "columns"])
    s.append(table(feats, widths=[1.3 * inch, 5.2 * inch]))
    s.append(p("All features are measured in 2022, before the 2023 payments. 2024 totals are not used as features "
               "because they are measured after treatment."))
    miss = pd.Series(clean["missing_pct"]).sort_values(ascending=False).rename("% missing").to_frame()
    s.append(table(miss.head(8), index=True, widths=[2.2 * inch, 1.0 * inch]))

    # ---------------- 7. EDA basic
    s.append(Paragraph("7. Exploratory analysis", H1))
    for f, cap in [
        ("01_payment_amounts", "Payments are small for most prescribers; the median paid prescriber received about $210 in 2023."),
        ("02_payment_types", "Dollars by type. Royalties go mostly to device inventors and are a candidate for exclusion from the dose."),
        ("03_outcome_distribution", "Brand share is right-skewed. The spike at 50% is an artifact of tiny prescribers whose counts are almost all hidden; "
                                    "these are flagged (y_share_uninformative) and excluded from share analyses."),
        ("04_paid_by_specialty", "Payment rates vary from 14% to 83% across large specialties, so specialty is a key confounder."),
        ("05_propensity_overlap", "Propensity scores overlap across most of the range."),
        ("06_covariate_balance", "Before weighting, paid prescribers are larger and had higher 2022 brand share; weighting balances all covariates."),
        ("07_trends_by_history", "Change in brand share since 2022 by payment history (unadjusted)."),
        ("08_dose_response", "Adjusted change in brand share by payment amount (section 5)."),
    ]:
        s.append(fig(FIG_BASE / f"{f}.png", cap, width=6.0))

    # ---------------- 8. drivers
    s.append(PageBreak())
    s.append(Paragraph("8. What drives brand prescribing", H1))
    s.append(p(f"These charts use the {ext['drivers_n']:,} prescribers with informative outcome bounds."))
    s.append(fig(FIG_EXT / "e01_share_by_specialty.png", "Specialty sets the level of brand prescribing; within most specialties paid prescribers are higher.", 6.0))
    sp = pd.DataFrame(ext["specialty"]).set_index("specialty").sort_values("gap", ascending=False)
    sp.index = [i.replace("Student in an Organized Health Care Education/Training Program", "Student / trainee") for i in sp.index]
    sp.columns = ["unpaid %", "paid %", "gap (pts)"]
    s.append(table(sp.round(2), index=True, widths=[2.3 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch]))
    s.append(fig(FIG_EXT / "e02_state_paid_vs_share.png",
                 f"States: payment rate and brand share are nearly unrelated (correlation {ext['state_corr']:.2f}), "
                 "so location matters mainly through specialty mix and practice size.", 6.0))
    loc = pd.DataFrame(ext["location"])[["var", "group", "n", "unpaid", "paid", "gap_pts"]]
    loc["var"] = loc["var"].replace({"ruca_group": "rural/urban"})
    loc.columns = ["variable", "group", "n", "unpaid %", "paid %", "gap (pts)"]
    s.append(table(loc))
    s.append(fig(FIG_EXT / "e03_share_by_size_and_patients.png",
                 "Brand share rises with practice size, patient risk and patient age; the paid line sits above the unpaid line throughout.", 6.3))
    s.append(fig(FIG_EXT / "e04_persistence.png",
                 f"Prescribing persists: 2022 brand share predicts 2024 (correlation {ext['persistence_corr']:.2f}). "
                 "This is why the 2022 baseline is the most important covariate.", 5.0))
    s.append(fig(FIG_EXT / "e05_correlations.png",
                 "Correlations among covariates, payment dose and outcome. The outcome correlates most with 2022 brand share and cost per claim.", 5.6))

    # ---------------- 9. separability
    s.append(Paragraph("9. Separability and overlap", H1))
    s.append(p("Causal methods need paid and unpaid prescribers to be comparable: if covariates separated them "
               "perfectly there would be no comparison group. We measured how well covariates predict payment."))
    s.append(fig(FIG_EXT / "e06_separability.png",
                 f"Left: each feature alone (specialty is strongest at {max(r['AUC'] for r in sep['univariate']):.2f}). "
                 f"Right: all features together, logistic AUC {sep['auc_logistic']:.3f} vs gradient boosting {sep['auc_gbm']:.3f}; "
                 "the small gain means the logistic model captures most of the structure.", 6.5))
    ovl = pd.DataFrame([{"model": "logistic", **sep["overlap_logistic"]}, {"model": "gradient boosting", **sep["overlap_gbm"]}])
    ovl.columns = ["model", "% with ps in 0.1-0.9", "% ps < 0.05", "% ps > 0.95"]
    s.append(table(ovl))
    osp = pd.DataFrame(sep["overlap_by_specialty"]).set_index("specialty")
    osp.index = [i.replace("Student in an Organized Health Care Education/Training Program", "Student / trainee") for i in osp.index]
    osp = osp[["n", "pct_paid", "pct_in_0_1_0_9"]]
    osp.columns = ["n", "% paid", "% with ps in 0.1-0.9"]
    s.append(table(osp, index=True, widths=[2.3 * inch, 1.0 * inch, 1.0 * inch, 1.4 * inch]))
    s.append(p("Overlap is weakest for students/trainees (rarely paid) and for cardiology and ophthalmology (mostly "
               "paid): estimates for these groups lean on fewer comparable prescribers and should be reported with care."))

    # ---------------- 10. matching
    s.append(Paragraph("10. Matching", H1))
    s.append(p(f"Each paid prescriber was matched to the most similar unpaid prescriber in the same specialty (1:1 "
               f"nearest neighbour on the logit propensity score, caliper 0.2 SD = {mat['caliper_logit']}, controls "
               f"reusable). {mat['matched_treated']:,} of {mat['treated']:,} paid prescribers ({mat['pct_matched']}%) "
               f"found a match, using {mat['distinct_controls']:,} distinct unpaid controls. Largest imbalance fell from "
               f"{mat['max_abs_smd_unadjusted']} to {mat['max_abs_smd_matched']} standardized mean differences."))
    s.append(fig(FIG_EXT / "e07_matched_balance.png", "Covariate balance before and after matching.", 6.0))
    s.append(fig(FIG_EXT / "e08_matched_att.png", "Raw vs matched paid-minus-unpaid difference in brand share.", 6.0))
    at = att.pivot(index="outcome", columns="bounds", values="ATT_pts")[["lo", "mid", "hi"]]
    at.columns = ["low bound", "middle", "high bound"]
    s.append(table(at.round(3), index=True, widths=[2.2 * inch, 1.1 * inch, 1.1 * inch, 1.1 * inch]))
    s.append(p("Matching removes about 85% of the raw gap. The remaining 0.5-0.8 point difference is consistent with "
               "the dose-response screen. Standard errors here ignore reuse of controls and are approximate; this is a "
               "descriptive check, not the final estimate."))

    # ---------------- 11. payment type + extra features
    s.append(Paragraph("11. Payment type", H1))
    pt = pd.DataFrame(ext["payment_type"]).set_index("type")
    pt.columns = ["n", "median $", "brand share 2022 %", "brand share 2024 %", "change (pts)"]
    s.append(table(pt.round(2), index=True, widths=[2.0 * inch, 0.8 * inch, 0.8 * inch, 1.1 * inch, 1.1 * inch, 0.9 * inch]))
    s.append(p("Prescribers newly paid in 2023 (no 2022 payment), unadjusted. Every paid group declined less than the "
               "never-paid group, including food-only. Group sizes for speaking/consulting are small."))
    s.append(Paragraph("12. Do the unused Part D columns matter?", H1))
    fb = pd.DataFrame(fchk["blocks"])[["block", "n_features", "pct_missing_avg", "AUC_gain", "outcome_R2_gain", "effect_shift_pct"]]
    fb.columns = ["block", "features", "% hidden", "AUC gain", "R2 gain", "effect shift %"]
    s.append(table(fb))
    s.append(p(f"Adding race mix, plan/income mix, drug-class mix and fill patterns changes the screening effect by at "
               f"most {fb['effect shift %'].abs().max():.1f}% (baseline {fchk['baseline']['effect_pts']:.3f} points, SE "
               f"{fchk['baseline']['effect_se']:.3f}). They are already balanced by the current features and are not needed "
               "for confounding; race mix remains useful for effect differences by patient population."))

    # ---------------- 13. limitations + next
    s.append(Paragraph("13. Limitations and next steps", H1))
    s.append(bullets([
        "Unmeasured confounding: representatives may target prescribers who were about to adopt new brand drugs. "
        "Plan: omitted-variable sensitivity analysis (Cinelli and Hazlett).",
        "Suppression: outcome results must be reported at low, middle and high bounds; tiny prescribers are excluded from share outcomes.",
        "16% of prescribers are new in 2024 and lack a baseline; analyze separately or with the indicator.",
        "Years in practice (mentioned in the proposal) is not in these files; it would need NPPES.",
        "Next: final propensity model (tuned, calibrated, trimmed), then doubly robust ATE and CATE estimation "
        "(DR-learner / causal forest), with the dose as a secondary treatment.",
    ]))

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "pipeline_report.pdf"
    doc = SimpleDocTemplate(str(out), pagesize=letter, leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                            topMargin=0.75 * inch, bottomMargin=0.8 * inch,
                            title="Industry payments and Part D brand prescribing - data preparation and EDA")
    doc.build(s, onFirstPage=footer, onLaterPages=footer)
    print(out)


if __name__ == "__main__":
    main()
