"""
build_dashboard.py

SAS821S · T18 - Insurance Claims Fraud, Identity Abuse and Data-Leakage Analytics

PURPOSE
-------
Satisfies Charter C10 "Decision-support prototype": presents findings through
a functional, analyst-facing view that supports investigation decisions.

Builds ONE self-contained HTML file (charts embedded as base64 images, no
external files or server needed) summarising:
    - key project metrics (KPI tiles)
    - the supervised + unsupervised model results
    - the top high-risk investigation cases
    - the top anomalous customers
    - the simulation scenario comparison
    - flagged suspicious documents

INPUT:  outputs from all prior scripts (models, simulation, NLP, timelines)
OUTPUT: 07_dashboard_or_prototype/dashboard.html
"""

import os
import base64
import pandas as pd

CHART_DIR = "08_outputs/model_charts"
EDA_CHART_DIR = "08_outputs/eda_charts"
OUT_DIR = "07_dashboard_or_prototype"
os.makedirs(OUT_DIR, exist_ok=True)

CLAIMS_IN = "02_data/processed/claims_identity_kyc_linked.csv"
ANOMALY_IN = "04_models/anomaly_model_customer_scores.csv"
CASES_IN = "08_outputs/incident_timelines/all_cases_summary.csv"
SIM_IN = "05_simulation/simulation_results.csv"
SUSPICIOUS_DOCS_IN = "06_text_mining/suspicious_document_flags.csv"
ADVERSARIAL_IN = "04_models/adversarial_test_results.csv"


def img_to_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def img_tag(path, alt="", width="100%"):
    if not os.path.exists(path):
        return f"<p><em>Chart not found: {path}</em></p>"
    b64 = img_to_base64(path)
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="width:{width}; max-width:600px; border:1px solid #ddd; border-radius:6px;">'


def df_to_html_table(df, max_rows=10):
    return df.head(max_rows).to_html(index=False, classes="data-table", border=0)


def main():
    claims = pd.read_csv(CLAIMS_IN)
    anomaly = pd.read_csv(ANOMALY_IN)
    cases = pd.read_csv(CASES_IN)
    sim = pd.read_csv(SIM_IN)
    susp_docs = pd.read_csv(SUSPICIOUS_DOCS_IN)
    adversarial = pd.read_csv(ADVERSARIAL_IN)

    total_claims = len(claims)
    fraud_rate = (claims["Fraud_Flag"] == "Yes").mean()
    n_anomalous = int(anomaly["is_anomalous"].sum())
    n_customers = len(anomaly)
    n_cases = len(cases)

    sim_summary = sim.groupby("scenario").agg(
        mean_net_benefit=("net_benefit", "mean"),
        mean_pct_caught=("pct_of_total_fraud_caught", "mean"),
        n_reviewed=("n_claims_reviewed", "first"),
    ).round(0)

    top_anomalous = anomaly.sort_values("anomaly_score", ascending=False).head(10)[
        ["Customer_ID", "anomaly_score", "new_device_rate", "unusual_location_rate", "failed_auth_rate"]
    ].round(3)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>T18 Fraud & Identity Risk Analytics - Decision Support Dashboard</title>
<style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6f8; margin: 0; padding: 0; color: #1a1a1a; }}
    header {{ background: #14213d; color: white; padding: 24px 32px; }}
    header h1 {{ margin: 0; font-size: 22px; }}
    header p {{ margin: 4px 0 0; color: #c9d2e0; font-size: 13px; }}
    .container {{ max-width: 1200px; margin: 0 auto; padding: 24px 32px; }}
    .kpi-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin-bottom: 28px; }}
    .kpi-card {{ background: white; border-radius: 8px; padding: 16px 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); flex: 1; min-width: 160px; }}
    .kpi-card .value {{ font-size: 26px; font-weight: 700; color: #14213d; }}
    .kpi-card .label {{ font-size: 12px; color: #666; text-transform: uppercase; letter-spacing: 0.4px; }}
    .kpi-card.alert .value {{ color: #c44e52; }}
    section {{ background: white; border-radius: 8px; padding: 20px 24px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
    section h2 {{ margin-top: 0; font-size: 17px; color: #14213d; border-bottom: 2px solid #eee; padding-bottom: 8px; }}
    .chart-row {{ display: flex; gap: 16px; flex-wrap: wrap; }}
    .chart-row > div {{ flex: 1; min-width: 280px; }}
    table.data-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    table.data-table th {{ background: #14213d; color: white; padding: 8px 10px; text-align: left; }}
    table.data-table td {{ padding: 7px 10px; border-bottom: 1px solid #eee; }}
    table.data-table tr:nth-child(even) {{ background: #f9fafb; }}
    .recommendation {{ background: #eaf3ea; border-left: 4px solid #55a868; padding: 14px 18px; border-radius: 4px; font-size: 14px; }}
    footer {{ text-align: center; color: #999; font-size: 12px; padding: 20px; }}
</style>
</head>
<body>
<header>
    <h1>T18 &middot; Insurance Claims Fraud, Identity Abuse and Data-Leakage Analytics</h1>
    <p>Decision-support prototype &middot; Members: Shanon Chiwara (Data &amp; Modelling) &middot; Ephath Shimhanda (Security Engineering &amp; Intelligence)</p>
</header>

<div class="container">

    <div class="kpi-row">
        <div class="kpi-card"><div class="value">{total_claims:,}</div><div class="label">Total Claims</div></div>
        <div class="kpi-card alert"><div class="value">{fraud_rate:.1%}</div><div class="label">Overall Fraud Rate</div></div>
        <div class="kpi-card"><div class="value">{n_customers:,}</div><div class="label">Unique Customers</div></div>
        <div class="kpi-card alert"><div class="value">{n_anomalous:,}</div><div class="label">Anomalous Customers Flagged</div></div>
        <div class="kpi-card alert"><div class="value">{n_cases}</div><div class="label">High-Risk Investigation Cases</div></div>
    </div>

    <section>
        <h2>1. Baseline: Identity Change &amp; Fraud Relationship</h2>
        <div class="chart-row">
            <div>{img_tag(f"{EDA_CHART_DIR}/01_fraud_flag_distribution.png", "Fraud flag distribution")}</div>
            <div>{img_tag(f"{EDA_CHART_DIR}/03_identity_change_by_fraud.png", "Identity change rate by fraud")}</div>
            <div>{img_tag(f"{EDA_CHART_DIR}/08_new_device_logins.png", "New device logins")}</div>
        </div>
    </section>

    <section>
        <h2>2. Supervised Fraud Classification (Logistic Regression + Random Forest)</h2>
        <div class="chart-row">
            <div>{img_tag(f"{CHART_DIR}/12_roc_curve_comparison.png", "ROC curve comparison")}</div>
            <div>{img_tag(f"{CHART_DIR}/13_feature_importance_rf.png", "Feature importance")}</div>
        </div>
    </section>

    <section>
        <h2>3. Unsupervised Anomaly Detection (Isolation Forest, Customer Login Behaviour)</h2>
        <p>Model trained with <strong>no fraud label</strong> - flagged customers independently show a far higher fraud rate, validating the approach.</p>
        <div class="chart-row">
            <div>{img_tag(f"{CHART_DIR}/15_anomaly_validation_vs_fraud.png", "Anomaly validation")}</div>
        </div>
        <h3 style="font-size:14px; margin-top:18px;">Top 10 Most Anomalous Customers (investigation priority)</h3>
        {df_to_html_table(top_anomalous)}
    </section>

    <section>
        <h2>4. Incident Investigation - High-Risk Cases</h2>
        <p>Cases selected where three independent signals agree: Fraud_Flag = Yes, identity/KYC change occurred, AND the customer was flagged anomalous by the unsupervised model.</p>
        {df_to_html_table(cases[["case_number", "Claim_ID", "Customer_ID", "Fraud_Risk_Score", "Claim_Amount", "n_events"]], max_rows=12)}
        <p style="font-size:12px; color:#666;">Full chronological timeline for each case is available in <code>08_outputs/incident_timelines/</code></p>
    </section>

    <section>
        <h2>5. Control Scenario Simulation (Monte Carlo, 1,000 iterations/scenario)</h2>
        <div class="chart-row">
            <div>{img_tag(f"{CHART_DIR}/16_simulation_comparison.png", "Simulation comparison")}</div>
        </div>
        {df_to_html_table(sim_summary.reset_index())}
        <div class="recommendation">
            <strong>Recommendation:</strong> Scenario B (identity-change trigger) captures the most total fraud value but reviews 17.9% of all claims. Scenario C (combined identity-change + anomaly-flag trigger) is far more efficient per case reviewed, reviewing only 2.6% of claims - the better choice where investigator capacity is limited.
        </div>
    </section>

    <section>
        <h2>6. Text Mining - Suspicious Document Flags</h2>
        <div class="chart-row">
            <div>{img_tag(f"{CHART_DIR}/18_top_keywords_by_type.png", "Top keywords by document type")}</div>
            <div>{img_tag(f"{EDA_CHART_DIR}/06_document_similarity_distribution.png", "Document similarity distribution")}</div>
        </div>
        <h3 style="font-size:14px; margin-top:18px;">Documents flagged for suspicious terms (top 10)</h3>
        {df_to_html_table(susp_docs)}
    </section>

    <section>
        <h2>7. Adversarial &amp; Robustness Testing</h2>
        <p>The trained Random Forest model was attacked three ways, without retraining: withholding its dominant feature, disguising real fraud claims to look legitimate, and inflating claim amounts to simulate drift.</p>
        <div class="chart-row">
            <div>{img_tag(f"{CHART_DIR}/19_adversarial_test_comparison.png", "Adversarial test comparison")}</div>
        </div>
        {df_to_html_table(adversarial[["scenario", "precision", "recall", "f1", "roc_auc", "n_samples"]].round(3), max_rows=10)}
        <div class="recommendation">
            <strong>Finding:</strong> Recall collapses from 0.412 to 0.000 when Identity_Changed_Flag is withheld or concealed — the model has a genuine single point of failure. ROC-AUC is more stable under claim-amount drift (0.723 &rarr; 0.715). Mitigation: the independent unsupervised anomaly model (Section 3) does not rely on this feature and remains effective against exactly this evasion strategy.
        </div>
    </section>

</div>
<footer>Generated automatically as part of the T18 Capstone Implementation Plan (Milestone 2) &middot; SAS821S Security Analytics &middot; NUST 2026</footer>
</body>
</html>
"""

    out_path = os.path.join(OUT_DIR, "dashboard.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Dashboard written to {out_path}")
    print(f"File size: {os.path.getsize(out_path) / 1024:.0f} KB")


if __name__ == "__main__":
    main()
