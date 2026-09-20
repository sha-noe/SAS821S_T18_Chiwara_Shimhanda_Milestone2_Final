"""
adversarial_robustness_testing.py

SAS821S · T18 - Insurance Claims Fraud, Identity Abuse and Data-Leakage Analytics

PURPOSE
-------
Satisfies Charter Objective 3 ("Test how well the fraud detection method
performs when given unusual or manipulated data") and Implementation Plan
Section 11 ("At least three adversarial cases").

Loads the ALREADY-TRAINED Random Forest model (04_models/supervised_fraud_model.py
must be run first) and attacks it three distinct ways, without retraining,
measuring how much performance degrades on each:

  TEST 1 - FEATURE MASKING
      Withhold Identity_Changed_Flag at inference (set to 0 for every test
      row). Identity_Changed_Flag was found to be the dominant feature
      (44.7% importance) in Section 5 - this tests what happens if that
      single signal becomes unavailable or is deliberately concealed.

  TEST 2 - SYNTHETIC EVASION
      Take the test set's actual fraud cases and disguise them: conceal the
      identity change AND push the enrichment features toward
      "legitimate-looking" values (no address change, no past claims,
      witness present, police report filed, high safety rating). Tests
      whether a claim engineered to look clean can evade detection.

  TEST 3 - DISTRIBUTIONAL DRIFT
      Inflate Claim_Amount and Premium_Amount by 35% across the whole test
      set (simulating claim-value inflation since the model was trained),
      without retraining. Tests whether the model's decision boundary is
      stable under realistic real-world drift.

INPUT:  04_models/fraud_model_random_forest.joblib
        02_data/processed/claims_identity_kyc_linked.csv
OUTPUT: 04_models/adversarial_test_report.txt
        04_models/adversarial_test_results.csv
        08_outputs/model_charts/19_adversarial_test_comparison.png
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

SEED = 42
np.random.seed(SEED)

MODEL_IN = "04_models/fraud_model_random_forest.joblib"
CLAIMS_IN = "02_data/processed/claims_identity_kyc_linked.csv"
MODEL_OUT_DIR = "04_models"
CHART_DIR = "08_outputs/model_charts"
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


def rebuild_identical_test_set(artifact):
    """
    Reproduces EXACTLY the same cleaning + train/test split used in
    supervised_fraud_model.py (same SEED, same steps, same order), so the
    test set used for adversarial testing is provably identical to the one
    the model was evaluated on in Section 5 - not a different, easier
    (or harder) sample.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    df = pd.read_csv(CLAIMS_IN)
    numeric_features = artifact["numeric_features"]
    categorical_features = artifact["categorical_features"]

    df_model = df[numeric_features + categorical_features + ["Fraud_Flag"]].copy()
    bad_mask = df_model["Enriched_Witness_Present"] == "*"
    df_model["Enriched_Witness_Present"] = df_model["Enriched_Witness_Present"].replace("*", np.nan)
    df_model["Enriched_Witness_Present"] = pd.to_numeric(df_model["Enriched_Witness_Present"])
    mode_val = df_model["Enriched_Witness_Present"].mode()[0]
    df_model["Enriched_Witness_Present"] = df_model["Enriched_Witness_Present"].fillna(mode_val)
    df_model = df_model.dropna()

    le_target = LabelEncoder()
    le_target.fit(artifact["label_classes"])
    y = le_target.transform(df_model["Fraud_Flag"])

    X_num = df_model[numeric_features]
    X_cat = pd.get_dummies(df_model[categorical_features], drop_first=True)
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.reindex(columns=artifact["feature_names"], fill_value=0)  # align to training columns exactly

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y)

    return X_test, y_test, le_target


def evaluate(model, X, y, label):
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    metrics = {
        "scenario": label,
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, y_proba) if len(set(y)) > 1 else float("nan"),
        "n_samples": len(y),
    }
    return metrics


def main():
    log("=" * 70)
    log("ADVERSARIAL AND ROBUSTNESS TESTING")
    log("Attacking the already-trained Random Forest model (no retraining)")
    log("=" * 70)

    artifact = joblib.load(MODEL_IN)
    model = artifact["model"]
    log(f"Loaded trained model with {len(artifact['feature_names'])} features")

    X_test, y_test, le_target = rebuild_identical_test_set(artifact)
    log(f"Rebuilt identical test set: {len(X_test)} rows "
        f"(same SEED, same cleaning, same split as Section 5)")

    results = []

    # --- BASELINE (no manipulation) --------------------------------------------
    baseline = evaluate(model, X_test, y_test, "Baseline (no manipulation)")
    results.append(baseline)
    log(f"\n--- BASELINE ---")
    log(f"  Precision={baseline['precision']:.3f}  Recall={baseline['recall']:.3f}  "
        f"F1={baseline['f1']:.3f}  ROC-AUC={baseline['roc_auc']:.3f}")

    # --- TEST 1: FEATURE MASKING ------------------------------------------------
    log(f"\n--- TEST 1: FEATURE MASKING (Identity_Changed_Flag withheld) ---")
    X_masked = X_test.copy()
    X_masked["Identity_Changed_Flag"] = 0
    masked_result = evaluate(model, X_masked, y_test, "Test 1: Feature masking")
    results.append(masked_result)
    recall_drop = baseline["recall"] - masked_result["recall"]
    log(f"  Precision={masked_result['precision']:.3f}  Recall={masked_result['recall']:.3f}  "
        f"F1={masked_result['f1']:.3f}  ROC-AUC={masked_result['roc_auc']:.3f}")
    log(f"  Recall dropped by {recall_drop:.3f} ({recall_drop/baseline['recall']*100:.1f}% relative decrease) "
        f"when the dominant feature is withheld.")
    log(f"  INTERPRETATION: {'Model is heavily reliant on a single feature - a real robustness concern.' if recall_drop > 0.05 else 'Model degrades gracefully - other features partially compensate.'}")

    # --- TEST 2: SYNTHETIC EVASION ----------------------------------------------
    log(f"\n--- TEST 2: SYNTHETIC EVASION (disguised fraud claims) ---")
    fraud_class_idx = list(le_target.classes_).index("Yes")
    fraud_mask = (y_test == fraud_class_idx)
    X_fraud = X_test[fraud_mask].copy()
    log(f"  {len(X_fraud)} actual fraud claims in test set selected for disguise")

    X_evasive = X_fraud.copy()
    X_evasive["Identity_Changed_Flag"] = 0
    if "Enriched_Address_Change_Flag" in X_evasive.columns:
        X_evasive["Enriched_Address_Change_Flag"] = 0
    if "Enriched_Past_Claims_Count" in X_evasive.columns:
        X_evasive["Enriched_Past_Claims_Count"] = 0
    if "Enriched_Witness_Present" in X_evasive.columns:
        X_evasive["Enriched_Witness_Present"] = 1
    if "Enriched_Police_Report_Filed" in X_evasive.columns:
        X_evasive["Enriched_Police_Report_Filed"] = 1
    if "Enriched_Safety_Rating" in X_evasive.columns:
        X_evasive["Enriched_Safety_Rating"] = X_test["Enriched_Safety_Rating"].quantile(0.85)

    y_pred_evasive = model.predict(X_evasive)
    caught_after = (y_pred_evasive == fraud_class_idx).mean()

    results.append({
        "scenario": "Test 2: Synthetic evasion",
        "precision": float("nan"), "recall": caught_after, "f1": float("nan"),
        "roc_auc": float("nan"), "n_samples": len(X_evasive),
    })
    log(f"  Disguised fraud claims still caught: {caught_after:.1%} "
        f"(all {len(X_fraud)} rows here are true fraud by construction)")
    log(f"  INTERPRETATION: {'Model shows meaningful vulnerability to disguised claims - detection drops substantially.' if caught_after < 0.7 else 'Model remains reasonably robust even against disguised fraud claims.'}")

    # --- TEST 3: DISTRIBUTIONAL DRIFT --------------------------------------------
    log(f"\n--- TEST 3: DISTRIBUTIONAL DRIFT (+35% claim/premium inflation) ---")
    X_drift = X_test.copy()
    X_drift["Claim_Amount"] = X_drift["Claim_Amount"] * 1.35
    X_drift["Premium_Amount"] = X_drift["Premium_Amount"] * 1.35
    drift_result = evaluate(model, X_drift, y_test, "Test 3: Distributional drift")
    results.append(drift_result)
    auc_drop = baseline["roc_auc"] - drift_result["roc_auc"]
    log(f"  Precision={drift_result['precision']:.3f}  Recall={drift_result['recall']:.3f}  "
        f"F1={drift_result['f1']:.3f}  ROC-AUC={drift_result['roc_auc']:.3f}")
    log(f"  ROC-AUC changed by {auc_drop:+.3f} under +35% claim/premium inflation.")
    log(f"  INTERPRETATION: {'Meaningful performance decay under drift - would need periodic retraining in production.' if abs(auc_drop) > 0.03 else 'Model is stable under this level of drift - no immediate retraining need.'}")

    # --- Save results -------------------------------------------------------------
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(MODEL_OUT_DIR, "adversarial_test_results.csv"), index=False)
    log(f"\nSaved results to {MODEL_OUT_DIR}/adversarial_test_results.csv")

    # --- Chart: recall/AUC comparison across scenarios (where applicable) -------
    chart_data = results_df[results_df["scenario"].isin(
        ["Baseline (no manipulation)", "Test 1: Feature masking", "Test 3: Distributional drift"])]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(chart_data))
    width = 0.35
    ax.bar(x - width/2, chart_data["recall"], width, label="Recall", color="#4C72B0")
    ax.bar(x + width/2, chart_data["roc_auc"], width, label="ROC-AUC", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(["Baseline", "Test 1:\nFeature masking", "Test 3:\nDistributional drift"], fontsize=9)
    ax.set_title("Model robustness under adversarial conditions")
    ax.legend()
    ax.set_ylim(0, 1)
    savefig(fig, "19_adversarial_test_comparison.png")

    # --- Overall summary ------------------------------------------------------------
    log("\n" + "=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"Test 1 (feature masking):      recall fell from {baseline['recall']:.3f} to {masked_result['recall']:.3f}")
    log(f"Test 2 (synthetic evasion):    {caught_after:.1%} of disguised fraud claims still caught")
    log(f"Test 3 (distributional drift): ROC-AUC changed from {baseline['roc_auc']:.3f} to {drift_result['roc_auc']:.3f}")
    log("\nOverall robustness verdict: the model relies heavily on Identity_Changed_Flag "
        "(consistent with its 44.7% feature importance in Section 5), which is both its "
        "greatest strength (a genuine, strong signal) and its main vulnerability (a single "
        "point of failure an adversary could target by concealing identity changes). "
        "Recommended mitigation: maintain the unsupervised anomaly model (Section 6) as a "
        "parallel, independent detection layer, since it does not depend on this feature "
        "and remains a check against exactly this evasion strategy.")

    with open(os.path.join(MODEL_OUT_DIR, "adversarial_test_report.txt"), "w") as f:
        f.write("\n".join(REPORT_LINES))
    log(f"\nFull report written to {MODEL_OUT_DIR}/adversarial_test_report.txt")


if __name__ == "__main__":
    main()
