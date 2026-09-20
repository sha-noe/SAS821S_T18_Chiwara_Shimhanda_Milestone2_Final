"""
unsupervised_anomaly_model.py

SAS821S · T18 - Insurance Claims Fraud, Identity Abuse and Data-Leakage Analytics
Member A (Shanon) - ML and anomaly modelling (Charter RACI: A/R)

PURPOSE
-------
Satisfies Charter C4's second required method: "an unsupervised, anomaly or
behavioural method with a clear threshold and analyst interpretation."
Matches the Charter's stated approach: "Clustering and anomaly detection
using claimant, device and access-behaviour data" (User Behaviour Analytics).

APPROACH
--------
Aggregates portal_authentication_device_data.csv from individual login
events up to ONE ROW PER CUSTOMER (behavioural profile), then runs Isolation
Forest to flag customers whose overall login behaviour is anomalous relative
to the population - e.g. unusually high new-device rate, high failed-login
rate, frequent off-hours access. This is genuine UBA (Session 7), distinct
from the supervised model, which works at the claim level.

INPUT:  02_data/raw/portal_authentication_device_data.csv
        02_data/processed/claims_identity_kyc_linked.csv (for validation only)
OUTPUT: 04_models/anomaly_model_customer_scores.csv
        04_models/anomaly_model_report.txt
        08_outputs/model_charts/14_anomaly_score_distribution.png
        08_outputs/model_charts/15_anomaly_validation_vs_fraud.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

SEED = 42
np.random.seed(SEED)

AUTH_IN = "02_data/raw/portal_authentication_device_data.csv"
CLAIMS_IN = "02_data/processed/claims_identity_kyc_linked.csv"
MODEL_OUT_DIR = "04_models"
CHART_DIR = "08_outputs/model_charts"
os.makedirs(MODEL_OUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

REPORT_LINES = []


def log(msg):
    print(msg)
    REPORT_LINES.append(msg)


def savefig(fig, filename):
    path = os.path.join(CHART_DIR, filename)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


def main():
    log("=" * 70)
    log("UNSUPERVISED ANOMALY DETECTION - Customer Login Behaviour (UBA)")
    log("Method: Isolation Forest")
    log("=" * 70)

    auth = pd.read_csv(AUTH_IN)
    log(f"Loaded {len(auth)} login events for {auth['Customer_ID'].nunique()} customers")

    # --- Step 1: build a behavioural profile PER CUSTOMER ---------------------
    profile = auth.groupby("Customer_ID").agg(
        total_logins=("log_id", "count"),
        new_device_rate=("new_device_flag", "mean"),
        unusual_location_rate=("location_unusual_flag", "mean"),
        off_hours_rate=("off_hours_login_flag", "mean"),
        avg_failed_attempts=("failed_attempts_before_success", "mean"),
        failed_auth_rate=("authentication_result",
                           lambda s: (s == "Failed then Success").mean()),
        distinct_devices=("device_id", "nunique"),
    ).reset_index()

    log(f"Built behavioural profiles for {len(profile)} customers")
    log("\nProfile feature summary:")
    log(profile.describe().round(3).to_string())

    feature_cols = ["total_logins", "new_device_rate", "unusual_location_rate",
                     "off_hours_rate", "avg_failed_attempts", "failed_auth_rate",
                     "distinct_devices"]
    X = profile[feature_cols]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # --- Step 2: Isolation Forest -----------------------------------------------
    # contamination=0.08 -> we expect roughly 8% of customers to show
    # genuinely anomalous access behaviour; this threshold is documented
    # here (not hidden) and can be tuned by an analyst reviewing results.
    contamination = 0.08
    iso = IsolationForest(n_estimators=300, contamination=contamination,
                           random_state=SEED, n_jobs=-1)
    profile["anomaly_label"] = iso.fit_predict(X_scaled)  # -1 = anomaly, 1 = normal
    profile["anomaly_score"] = -iso.score_samples(X_scaled)  # higher = more anomalous
    profile["is_anomalous"] = (profile["anomaly_label"] == -1).astype(int)

    n_flagged = profile["is_anomalous"].sum()
    log(f"\nFlagged {n_flagged} of {len(profile)} customers as anomalous "
        f"({n_flagged/len(profile):.1%}, threshold=contamination {contamination})")

    profile_out = profile.drop(columns=["anomaly_label"]).sort_values(
        "anomaly_score", ascending=False)
    profile_out.to_csv(os.path.join(MODEL_OUT_DIR, "anomaly_model_customer_scores.csv"),
                        index=False)
    log(f"Saved per-customer anomaly scores to 04_models/anomaly_model_customer_scores.csv")

    # --- Chart: anomaly score distribution -------------------------------------
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(profile.loc[profile["is_anomalous"] == 0, "anomaly_score"], bins=40,
            alpha=0.6, label="Normal", color="#4C72B0")
    ax.hist(profile.loc[profile["is_anomalous"] == 1, "anomaly_score"], bins=40,
            alpha=0.8, label="Anomalous", color="#C44E52")
    ax.set_xlabel("Anomaly score (higher = more unusual)")
    ax.set_ylabel("Number of customers")
    ax.set_title("Isolation Forest anomaly score distribution")
    ax.legend()
    savefig(fig, "14_anomaly_score_distribution.png")

    # --- Step 3: validation - do anomalous customers actually overlap with ---
    #     known fraud/identity-change claims? (independent check, since the
    #     model was trained with NO fraud label at all - pure unsupervised)
    claims = pd.read_csv(CLAIMS_IN)
    cust_risk = claims.groupby("Customer_ID").agg(
        any_fraud=("Fraud_Flag", lambda s: (s == "Yes").any()),
        any_identity_change=("Identity_Changed_Flag", "max")
    ).reset_index()

    merged = profile.merge(cust_risk, on="Customer_ID", how="left")
    fraud_rate_anom = merged.loc[merged["is_anomalous"] == 1, "any_fraud"].mean()
    fraud_rate_normal = merged.loc[merged["is_anomalous"] == 0, "any_fraud"].mean()
    identity_rate_anom = merged.loc[merged["is_anomalous"] == 1, "any_identity_change"].mean()
    identity_rate_normal = merged.loc[merged["is_anomalous"] == 0, "any_identity_change"].mean()

    log("\n--- Independent validation (model never saw Fraud_Flag) ---")
    log(f"Fraud rate among ANOMALOUS customers:  {fraud_rate_anom:.1%}")
    log(f"Fraud rate among NORMAL customers:     {fraud_rate_normal:.1%}")
    log(f"Identity-change rate among ANOMALOUS:  {identity_rate_anom:.1%}")
    log(f"Identity-change rate among NORMAL:     {identity_rate_normal:.1%}")

    fig, ax = plt.subplots(1, 2, figsize=(9, 4))
    merged.groupby("is_anomalous")["any_fraud"].mean().plot(
        kind="bar", ax=ax[0], color=["#4C72B0", "#C44E52"])
    ax[0].set_title("Fraud rate: anomalous vs normal")
    ax[0].set_xticklabels(["Normal", "Anomalous"], rotation=0)
    merged.groupby("is_anomalous")["any_identity_change"].mean().plot(
        kind="bar", ax=ax[1], color=["#4C72B0", "#C44E52"])
    ax[1].set_title("Identity-change rate: anomalous vs normal")
    ax[1].set_xticklabels(["Normal", "Anomalous"], rotation=0)
    savefig(fig, "15_anomaly_validation_vs_fraud.png")

    log("\n--- Top 10 most anomalous customers (candidates for investigation) ---")
    log(profile_out.head(10)[["Customer_ID", "anomaly_score", "new_device_rate",
                               "unusual_location_rate", "failed_auth_rate"]].to_string(index=False))

    with open(os.path.join(MODEL_OUT_DIR, "anomaly_model_report.txt"), "w") as f:
        f.write("\n".join(REPORT_LINES))
    log(f"\nFull report written to {MODEL_OUT_DIR}/anomaly_model_report.txt")


if __name__ == "__main__":
    main()
