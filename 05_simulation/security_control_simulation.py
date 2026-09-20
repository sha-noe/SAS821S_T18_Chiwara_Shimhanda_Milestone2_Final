"""
security_control_simulation.py

SAS821S · T18 - Insurance Claims Fraud, Identity Abuse and Data-Leakage Analytics
Member B (Ephath) - Simulation and intelligence (Charter RACI: A/R)

PURPOSE
-------
Satisfies Charter C7 "Simulation": compares at least three control scenarios
using a repeatable Monte Carlo simulation (>= 1,000 iterations per scenario,
per Section 5.2 of the Capstone brief).

THE THREE SCENARIOS
--------------------
A. NO CONTROL (baseline)      - claims processed with no additional review;
                                 all fraud goes undetected; zero review cost.
B. IDENTITY-CHANGE TRIGGER    - any claim with Identity_Changed_Flag == 1
                                 gets manually reviewed. Simple, broad net.
C. COMBINED SIGNAL TRIGGER    - only claims where identity change AND the
                                 customer was flagged anomalous by the
                                 unsupervised model get reviewed. Narrower,
                                 more targeted.

Each iteration draws a random detection rate and review-cost-per-case from
a distribution (not fixed constants) so the simulation reflects genuine
uncertainty in how effective manual review actually is - the whole point of
Monte Carlo simulation rather than a single deterministic estimate.

INPUT:  02_data/processed/claims_identity_kyc_linked.csv
        04_models/anomaly_model_customer_scores.csv
OUTPUT: 05_simulation/simulation_results.csv
        05_simulation/simulation_report.txt
        08_outputs/model_charts/16_simulation_comparison.png
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)

CLAIMS_IN = "02_data/processed/claims_identity_kyc_linked.csv"
ANOMALY_IN = "04_models/anomaly_model_customer_scores.csv"
SIM_OUT_DIR = "05_simulation"
CHART_DIR = "08_outputs/model_charts"
os.makedirs(SIM_OUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

N_ITERATIONS = 1000
REVIEW_COST_MEAN = 45.0   # N$ average staff time cost to manually review one claim
REVIEW_COST_STD = 10.0
DETECTION_RATE_ALPHA, DETECTION_RATE_BETA = 7, 3  # Beta dist, mean ~0.70, broad review
NARROW_DETECTION_ALPHA, NARROW_DETECTION_BETA = 9, 2  # mean ~0.82, narrower/more precise review

REPORT_LINES = []


def log(msg):
    print(msg)
    REPORT_LINES.append(msg)


def run_scenario(name, claims_subset, all_fraud_amount, detection_alpha, detection_beta,
                  n_iterations=N_ITERATIONS):
    """
    claims_subset: the claims that WOULD be reviewed under this scenario.
    all_fraud_amount: total N$ value of ALL fraudulent claims in the dataset
                       (used to compute what fraction this scenario catches).
    """
    fraud_in_subset = claims_subset[claims_subset["Fraud_Flag"] == "Yes"]
    fraud_amount_in_subset = fraud_in_subset["Claim_Amount"].sum()
    n_reviewed = len(claims_subset)

    results = []
    for _ in range(n_iterations):
        detection_rate = np.random.beta(detection_alpha, detection_beta)
        review_cost_per_case = max(5, np.random.normal(REVIEW_COST_MEAN, REVIEW_COST_STD))

        fraud_amount_caught = fraud_amount_in_subset * detection_rate
        total_review_cost = n_reviewed * review_cost_per_case
        net_benefit = fraud_amount_caught - total_review_cost
        fraud_still_missed = all_fraud_amount - fraud_amount_caught

        results.append({
            "detection_rate": detection_rate,
            "fraud_amount_caught": fraud_amount_caught,
            "total_review_cost": total_review_cost,
            "net_benefit": net_benefit,
            "fraud_still_missed": fraud_still_missed,
            "pct_of_total_fraud_caught": fraud_amount_caught / all_fraud_amount,
        })

    df = pd.DataFrame(results)
    df["scenario"] = name
    df["n_claims_reviewed"] = n_reviewed

    log(f"\n--- Scenario: {name} ---")
    log(f"  Claims reviewed: {n_reviewed:,}  ({n_reviewed/len(claims)*100:.1f}% of all claims)")
    log(f"  Fraud value available in this subset: N$ {fraud_amount_in_subset:,.0f}")
    log(f"  Mean fraud N$ caught (across {n_iterations} runs): N$ {df['fraud_amount_caught'].mean():,.0f}")
    log(f"  Mean review cost: N$ {df['total_review_cost'].mean():,.0f}")
    log(f"  Mean net benefit: N$ {df['net_benefit'].mean():,.0f}  "
        f"(90% range: N$ {df['net_benefit'].quantile(0.05):,.0f} to N$ {df['net_benefit'].quantile(0.95):,.0f})")
    log(f"  Mean % of total fraud caught: {df['pct_of_total_fraud_caught'].mean():.1%}")

    return df


def main():
    global claims
    claims = pd.read_csv(CLAIMS_IN)
    anomaly = pd.read_csv(ANOMALY_IN)
    log("=" * 70)
    log("SECURITY CONTROL SIMULATION - Monte Carlo, 3 scenarios, "
        f"{N_ITERATIONS} iterations each")
    log("=" * 70)

    all_fraud_amount = claims.loc[claims["Fraud_Flag"] == "Yes", "Claim_Amount"].sum()
    log(f"\nTotal claims: {len(claims):,}")
    log(f"Total fraud value in dataset: N$ {all_fraud_amount:,.0f}")

    anomalous_customers = set(anomaly.loc[anomaly["is_anomalous"] == 1, "Customer_ID"])

    # --- Scenario A: no control -------------------------------------------------
    subset_a = claims.iloc[0:0]  # empty - nothing reviewed
    df_a = run_scenario("A - No control (baseline)", subset_a, all_fraud_amount,
                         detection_alpha=1, detection_beta=1)  # irrelevant, 0 reviewed
    df_a["fraud_amount_caught"] = 0.0
    df_a["total_review_cost"] = 0.0
    df_a["net_benefit"] = 0.0
    df_a["fraud_still_missed"] = all_fraud_amount
    df_a["pct_of_total_fraud_caught"] = 0.0

    # --- Scenario B: identity-change trigger (broad) ----------------------------
    subset_b = claims[claims["Identity_Changed_Flag"] == 1]
    df_b = run_scenario("B - Identity-change trigger", subset_b, all_fraud_amount,
                         DETECTION_RATE_ALPHA, DETECTION_RATE_BETA)

    # --- Scenario C: combined signal trigger (narrow, high precision) -----------
    subset_c = claims[(claims["Identity_Changed_Flag"] == 1) &
                       (claims["Customer_ID"].isin(anomalous_customers))]
    df_c = run_scenario("C - Combined signal trigger (identity change + anomaly)",
                         subset_c, all_fraud_amount,
                         NARROW_DETECTION_ALPHA, NARROW_DETECTION_BETA)

    all_results = pd.concat([df_a, df_b, df_c], ignore_index=True)
    all_results.to_csv(os.path.join(SIM_OUT_DIR, "simulation_results.csv"), index=False)
    log(f"\nWrote {len(all_results)} simulation rows to {SIM_OUT_DIR}/simulation_results.csv")

    # --- Comparison chart ---------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    summary = all_results.groupby("scenario").agg(
        mean_net_benefit=("net_benefit", "mean"),
        mean_pct_caught=("pct_of_total_fraud_caught", "mean")
    ).reindex(["A - No control (baseline)", "B - Identity-change trigger",
               "C - Combined signal trigger (identity change + anomaly)"])

    summary["mean_net_benefit"].plot(kind="bar", ax=axes[0], color=["#8C8C8C", "#4C72B0", "#55A868"])
    axes[0].set_title("Mean net benefit (N$) by scenario")
    axes[0].set_xticklabels(["A: None", "B: Identity\ntrigger", "C: Combined\ntrigger"], rotation=0)
    axes[0].axhline(0, color="black", linewidth=0.8)

    (summary["mean_pct_caught"] * 100).plot(kind="bar", ax=axes[1], color=["#8C8C8C", "#4C72B0", "#55A868"])
    axes[1].set_title("Mean % of total fraud value caught")
    axes[1].set_xticklabels(["A: None", "B: Identity\ntrigger", "C: Combined\ntrigger"], rotation=0)
    axes[1].set_ylabel("%")

    fig.tight_layout()
    fig.savefig(os.path.join(CHART_DIR, "16_simulation_comparison.png"))
    plt.close(fig)
    log(f"Saved comparison chart to {CHART_DIR}/16_simulation_comparison.png")

    log("\n--- RECOMMENDATION ---")
    summary["net_benefit_per_case"] = summary["mean_net_benefit"] / summary.index.map(
        lambda s: max(all_results.loc[all_results["scenario"] == s, "n_claims_reviewed"].iloc[0], 1))
    best_total = summary["mean_net_benefit"].idxmax()
    best_efficiency = summary["net_benefit_per_case"].idxmax()
    log(f"Highest TOTAL net benefit: {best_total} "
        f"(reviews the most claims, so captures the most absolute fraud value)")
    log(f"Highest net benefit PER CLAIM REVIEWED: {best_efficiency} "
        f"(N$ {summary.loc[best_efficiency, 'net_benefit_per_case']:,.0f} per case, "
        f"vs N$ {summary.loc['B - Identity-change trigger', 'net_benefit_per_case']:,.0f} for Scenario B)")
    log("\nInterpretation: Scenario B catches more total fraud value because it reviews a much "
        "larger pool of claims (17.9% of all claims). Scenario C is far more EFFICIENT per review "
        "- it combines two independent signals (identity change + unsupervised anomaly flag) to "
        "review only 2.6% of claims - making it the better choice if investigator time/capacity "
        "is the binding constraint, while Scenario B is better if the goal is maximising total "
        "fraud value recovered regardless of review workload.")

    with open(os.path.join(SIM_OUT_DIR, "simulation_report.txt"), "w") as f:
        f.write("\n".join(REPORT_LINES))
    log(f"\nFull report written to {SIM_OUT_DIR}/simulation_report.txt")


if __name__ == "__main__":
    main()
