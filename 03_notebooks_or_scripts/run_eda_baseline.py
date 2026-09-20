"""
run_eda_baseline.py

SAS821S · T18 - Insurance Claims Fraud, Identity Abuse and Data-Leakage Analytics

Runs baseline exploratory data analysis (EDA) across all four linked data
sources, satisfying Implementation Plan Section 8.1:
    "Exploratory analysis, visual baselines and initial security hypotheses."

For each dataset this produces:
    - row/column counts and missing-value summary
    - descriptive statistics for key numeric fields
    - 2+ meaningful visualisations (charter requires "two or more meaningful
      security visualisations and a baseline of normal behaviour")

INPUTS (all already generated/linked in prior scripts):
    02_data/processed/claims_identity_kyc_linked.csv
    02_data/processed/staff_access_investigation_linked.csv
    02_data/processed/document_metadata_ocr_text_linked.csv
    02_data/raw/portal_authentication_device_data.csv

OUTPUTS:
    08_outputs/eda_charts/*.png          <- the actual chart images
    08_outputs/baseline_eda_summary.md   <- narrative summary, ready to paste
                                             into the Implementation Plan /
                                             Documentation report
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display in this environment - write straight to file
import matplotlib.pyplot as plt

CLAIMS_IN = "02_data/processed/claims_identity_kyc_linked.csv"
STAFF_IN = "02_data/processed/staff_access_investigation_linked.csv"
DOCS_IN = "02_data/processed/document_metadata_ocr_text_linked.csv"
AUTH_IN = "02_data/raw/portal_authentication_device_data.csv"

CHART_DIR = "08_outputs/eda_charts"
SUMMARY_OUT = "08_outputs/baseline_eda_summary.md"
os.makedirs(CHART_DIR, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.3})


def missing_value_report(df, name):
    miss = df.isnull().sum()
    miss = miss[miss > 0]
    pct = (miss / len(df) * 100).round(2)
    if len(miss) == 0:
        return f"- **{name}**: no missing values in any column.\n"
    lines = [f"- **{name}**: missing values found in {len(miss)} column(s):"]
    for col in miss.index:
        lines.append(f"  - `{col}`: {miss[col]} missing ({pct[col]}%)")
    return "\n".join(lines) + "\n"


def savefig(fig, filename):
    path = os.path.join(CHART_DIR, filename)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")
    return path


def eda_claims(md):
    df = pd.read_csv(CLAIMS_IN)
    md.append("## 1. Claims and Identity/KYC Records\n")
    md.append(f"Rows: {len(df):,} | Columns: {len(df.columns)} | "
              f"Unique customers: {df['Customer_ID'].nunique():,}\n")
    md.append(missing_value_report(df, "claims_identity_kyc_linked.csv"))

    md.append("\n**Key descriptive statistics:**\n")
    stats = df[["Claim_Amount", "Premium_Amount", "Coverage_Amount",
                "Credit_Score", "Fraud_Risk_Score"]].describe().round(2)
    md.append(stats.to_markdown() + "\n")

    fraud_rate = (df["Fraud_Flag"] == "Yes").mean()
    md.append(f"\nOverall fraud rate: **{fraud_rate:.1%}** of claims flagged fraudulent.\n")

    # Chart 1: Fraud flag distribution
    fig, ax = plt.subplots(figsize=(5, 4))
    df["Fraud_Flag"].value_counts().plot(kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_title("Claim volume by Fraud_Flag")
    ax.set_xlabel("Fraud_Flag")
    ax.set_ylabel("Number of claims")
    savefig(fig, "01_fraud_flag_distribution.png")

    # Chart 2: Claim amount distribution by fraud flag
    fig, ax = plt.subplots(figsize=(6, 4))
    df.boxplot(column="Claim_Amount", by="Fraud_Flag", ax=ax)
    ax.set_title("Claim_Amount by Fraud_Flag")
    plt.suptitle("")
    savefig(fig, "02_claim_amount_by_fraud.png")

    # Chart 3: Identity change rate by fraud flag (core security hypothesis)
    fig, ax = plt.subplots(figsize=(5, 4))
    df.groupby("Fraud_Flag")["Identity_Changed_Flag"].mean().plot(
        kind="bar", ax=ax, color=["#55A868", "#C44E52"])
    ax.set_title("Identity/KYC change rate by Fraud_Flag")
    ax.set_ylabel("Proportion with identity change")
    savefig(fig, "03_identity_change_by_fraud.png")

    md.append("\n**Charts:** `01_fraud_flag_distribution.png`, "
              "`02_claim_amount_by_fraud.png`, `03_identity_change_by_fraud.png`\n")
    md.append("\n**Baseline hypothesis (H1):** identity/KYC changes are markedly more "
              "common among fraudulent claims than legitimate ones, supporting the "
              "problem statement's core assumption.\n")
    return df


def eda_staff(md):
    df = pd.read_csv(STAFF_IN)
    md.append("\n## 2. Staff Access and Investigation Records\n")
    md.append(f"Rows: {len(df):,} | Columns: {len(df.columns)}\n")
    md.append(missing_value_report(df, "staff_access_investigation_linked.csv"))

    risk_rate = df["Insider_Risk_Label"].mean()
    md.append(f"\nStaff flagged Insider_Risk_Label=1: **{df['Insider_Risk_Label'].sum()}** "
              f"of {len(df)} agents ({risk_rate:.1%}).\n")

    fig, ax = plt.subplots(figsize=(5, 4))
    df["Insider_Risk_Label"].value_counts().sort_index().plot(
        kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_title("Staff by Insider_Risk_Label")
    ax.set_xlabel("Insider_Risk_Label (0=normal, 1=flagged)")
    savefig(fig, "04_insider_risk_distribution.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    df.boxplot(column="fraud_rate_handled", by="Insider_Risk_Label", ax=ax)
    ax.set_title("Fraud rate of claims handled, by Insider_Risk_Label")
    plt.suptitle("")
    savefig(fig, "05_fraud_exposure_by_insider_risk.png")

    md.append("\n**Charts:** `04_insider_risk_distribution.png`, "
              "`05_fraud_exposure_by_insider_risk.png`\n")
    md.append("\n**Baseline hypothesis (H2):** agents flagged as insider-risk tend to "
              "have handled a higher proportion of fraudulent claims than other agents.\n")
    return df


def eda_documents(md):
    df = pd.read_csv(DOCS_IN)
    md.append("\n## 3. Document Metadata and OCR/Text Data\n")
    md.append(f"Rows: {len(df):,} | Columns: {len(df.columns)}\n")
    md.append(missing_value_report(df, "document_metadata_ocr_text_linked.csv"))

    fig, ax = plt.subplots(figsize=(5, 4))
    df["document_similarity_flag"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).plot(
        kind="bar", ax=ax, color=["#55A868", "#DD8452", "#C44E52"])
    ax.set_title("Document similarity flag distribution")
    savefig(fig, "06_document_similarity_distribution.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    df["document_type"].value_counts().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title("Document volume by type")
    savefig(fig, "07_document_type_volume.png")

    high_linked = df[(df["document_similarity_flag"] == "HIGH")]["linked_identity_change"].mean()
    md.append(f"\nAmong HIGH-similarity documents, {high_linked:.1%} are linked to a claim "
              f"with an identity/KYC change - suggesting duplicated/templated document "
              f"content clusters around identity-change cases.\n")
    md.append("\n**Charts:** `06_document_similarity_distribution.png`, "
              "`07_document_type_volume.png`\n")
    return df


def eda_auth(md):
    df = pd.read_csv(AUTH_IN)
    md.append("\n## 4. Portal Authentication and Device Data\n")
    md.append(f"Rows: {len(df):,} | Columns: {len(df.columns)} | "
              f"Unique customers: {df['Customer_ID'].nunique():,}\n")
    md.append(missing_value_report(df, "portal_authentication_device_data.csv"))

    fig, ax = plt.subplots(figsize=(5, 4))
    df["new_device_flag"].value_counts().sort_index().plot(
        kind="bar", ax=ax, color=["#4C72B0", "#C44E52"])
    ax.set_xticklabels(["Usual device", "New device"], rotation=0)
    ax.set_title("Login volume: usual vs new device")
    savefig(fig, "08_new_device_logins.png")

    fig, ax = plt.subplots(figsize=(5, 4))
    df["authentication_result"].value_counts().plot(kind="bar", ax=ax, color="#55A868")
    ax.set_title("Authentication outcome distribution")
    plt.xticks(rotation=20, ha="right")
    savefig(fig, "09_authentication_outcomes.png")

    off_hours_rate = df["off_hours_login_flag"].mean()
    md.append(f"\nOff-hours logins (22:00-06:00): {off_hours_rate:.1%} of all login events.\n")
    md.append("\n**Charts:** `08_new_device_logins.png`, `09_authentication_outcomes.png`\n")
    return df


def main():
    md = ["# Baseline EDA Summary — T18 Insurance Claims Fraud & Identity Risk Analytics\n",
          "Generated automatically from the four linked data sources. Charts are saved "
          "in `08_outputs/eda_charts/`.\n"]

    print("Running EDA on Claims...")
    eda_claims(md)
    print("Running EDA on Staff...")
    eda_staff(md)
    print("Running EDA on Documents...")
    eda_documents(md)
    print("Running EDA on Portal Authentication...")
    eda_auth(md)

    with open(SUMMARY_OUT, "w") as f:
        f.write("\n".join(md))
    print(f"\nWrote summary to {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
