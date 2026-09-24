"""
supervised_fraud_model.py

SAS821S · T18 - Insurance Claims Fraud, Identity Abuse and Data-Leakage Analytics
Member A (Shanon) - ML and anomaly modelling

PURPOSE
-------
Satisfies Charter C4 "Machine learning" and the Implementation Plan's
"Machine-learning and anomaly methods" section: trains and evaluates a
supervised fraud classifier, per the Charter's stated method choice
(Logistic Regression and Random Forest), using confusion matrix, precision,
recall, F1-score and ROC-AUC as required by Section 5.2 of the Capstone brief.

INPUT:  02_data/processed/claims_identity_kyc_linked.csv
OUTPUT: 04_models/fraud_model_metrics.txt
        04_models/fraud_model_feature_importance.csv
        08_outputs/model_charts/10_confusion_matrix_logreg.png
        08_outputs/model_charts/11_confusion_matrix_rf.png
        08_outputs/model_charts/12_roc_curve_comparison.png
        08_outputs/model_charts/13_feature_importance_rf.png
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (confusion_matrix, classification_report, roc_auc_score,
                              roc_curve, precision_score, recall_score, f1_score,
                              ConfusionMatrixDisplay)

SEED = 42
np.random.seed(SEED)

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
    log("SUPERVISED FRAUD CLASSIFICATION - Logistic Regression + Random Forest")
    log("=" * 70)

    df = pd.read_csv(CLAIMS_IN)
    log(f"Loaded {len(df)} claims")

    # --- Feature selection ---------------------------------------------------
    # Use fields genuinely available at claim-review time; exclude anything
    # that would leak the outcome (Fraud_Risk_Score is itself a strong
    # proxy computed downstream, kept but documented as a candidate leak).
    numeric_features = [
        "Age", "Premium_Amount", "Coverage_Amount", "Claim_Amount",
        "Credit_Score", "Identity_Changed_Flag", "Enriched_Past_Claims_Count",
        "Enriched_Witness_Present", "Enriched_Police_Report_Filed",
        "Enriched_Safety_Rating", "Enriched_Days_Open",
        "Enriched_Form_Defects_Count", "Enriched_Address_Change_Flag",
    ]
    categorical_features = ["Gender", "Region", "Policy_Type", "Claim_Type",
                             "Enriched_Claim_Channel"]

    df_model = df[numeric_features + categorical_features + ["Fraud_Flag"]].copy()

    # --- Data quality fix: insurance_fraud_data.csv (the source of the
    #     Enriched_* columns) uses '*' as its own missing-value marker for
    #     Enriched_Witness_Present. 340 of 50,000 rows (0.68%) affected.
    #     Documented cleaning step: treat '*' as missing, then impute with
    #     the column mode (0 = no witness), since 0.68% is too small to
    #     justify dropping rows and losing other valid feature values.
    n_before = len(df_model)
    bad_mask = df_model["Enriched_Witness_Present"] == "*"
    log(f"Data quality: found {bad_mask.sum()} rows ({bad_mask.mean():.2%}) with '*' "
        f"placeholder in Enriched_Witness_Present - imputing with column mode")
    df_model["Enriched_Witness_Present"] = df_model["Enriched_Witness_Present"].replace("*", np.nan)
    df_model["Enriched_Witness_Present"] = pd.to_numeric(df_model["Enriched_Witness_Present"])
    mode_val = df_model["Enriched_Witness_Present"].mode()[0]
    df_model["Enriched_Witness_Present"] = df_model["Enriched_Witness_Present"].fillna(mode_val)

    df_model = df_model.dropna()
    log(f"After cleaning: {len(df_model)} rows remain ({n_before - len(df_model)} dropped for other missing values)")

    le_target = LabelEncoder()
    y = le_target.fit_transform(df_model["Fraud_Flag"])  # Yes=1, No=0 (check mapping)
    log(f"Target classes: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

    X_num = df_model[numeric_features]
    X_cat = pd.get_dummies(df_model[categorical_features], drop_first=True)
    X = pd.concat([X_num, X_cat], axis=1)
    feature_names = X.columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=SEED, stratify=y)
    log(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Model 1: Logistic Regression ----------------------------------------
    log("\n--- Logistic Regression ---")
    logreg = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
    logreg.fit(X_train_scaled, y_train)
    y_pred_lr = logreg.predict(X_test_scaled)
    y_proba_lr = logreg.predict_proba(X_test_scaled)[:, 1]

    log(classification_report(y_test, y_pred_lr, target_names=le_target.classes_))
    lr_auc = roc_auc_score(y_test, y_proba_lr)
    log(f"ROC-AUC: {lr_auc:.4f}")

    cm_lr = confusion_matrix(y_test, y_pred_lr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm_lr, display_labels=le_target.classes_).plot(ax=ax, cmap="Blues")
    ax.set_title("Logistic Regression - Confusion Matrix")
    savefig(fig, "10_confusion_matrix_logreg.png")

    # --- Model 2: Random Forest ------------------------------------------------
    log("\n--- Random Forest ---")
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, random_state=SEED,
                                 class_weight="balanced", n_jobs=-1)
    rf.fit(X_train, y_train)  # tree models don't need scaling

        artifact = {
        "model": rf,
        "feature_names": feature_names,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "label_classes": le_target.classes_,
    }
    joblib.dump(artifact, os.path.join(MODEL_OUT_DIR, "fraud_model_random_forest.joblib"))
    log(f"\nSaved trained model artifact to {MODEL_OUT_DIR}/fraud_model_random_forest.joblib")


    y_pred_rf = rf.predict(X_test)
    y_proba_rf = rf.predict_proba(X_test)[:, 1]

    log(classification_report(y_test, y_pred_rf, target_names=le_target.classes_))
    rf_auc = roc_auc_score(y_test, y_proba_rf)
    log(f"ROC-AUC: {rf_auc:.4f}")

    cm_rf = confusion_matrix(y_test, y_pred_rf)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(cm_rf, display_labels=le_target.classes_).plot(ax=ax, cmap="Greens")
    ax.set_title("Random Forest - Confusion Matrix")
    savefig(fig, "11_confusion_matrix_rf.png")

    # --- ROC curve comparison ---------------------------------------------------
    fig, ax = plt.subplots(figsize=(6, 5))
    for name, proba, auc in [("Logistic Regression", y_proba_lr, lr_auc),
                              ("Random Forest", y_proba_rf, rf_auc)]:
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="Random guess")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend()
    savefig(fig, "12_roc_curve_comparison.png")

    # --- Feature importance (Random Forest) --------------------------------
    importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    importances.head(15).to_csv(os.path.join(MODEL_OUT_DIR, "fraud_model_feature_importance.csv"))

    fig, ax = plt.subplots(figsize=(7, 6))
    importances.head(12).sort_values().plot(kind="barh", ax=ax, color="#4C72B0")
    ax.set_title("Top 12 Feature Importances - Random Forest")
    savefig(fig, "13_feature_importance_rf.png")

    log("\nTop 5 most important features (Random Forest):")
    for feat, imp in importances.head(5).items():
        log(f"  {feat}: {imp:.4f}")

    # --- Summary comparison table ---------------------------------------------
    summary = pd.DataFrame({
        "Model": ["Logistic Regression", "Random Forest"],
        "Precision": [precision_score(y_test, y_pred_lr), precision_score(y_test, y_pred_rf)],
        "Recall": [recall_score(y_test, y_pred_lr), recall_score(y_test, y_pred_rf)],
        "F1_Score": [f1_score(y_test, y_pred_lr), f1_score(y_test, y_pred_rf)],
        "ROC_AUC": [lr_auc, rf_auc],
    }).round(4)
    log("\n--- Final comparison ---")
    log(summary.to_string(index=False))

    with open(os.path.join(MODEL_OUT_DIR, "fraud_model_metrics.txt"), "w") as f:
        f.write("\n".join(REPORT_LINES))
    log(f"\nFull report written to {MODEL_OUT_DIR}/fraud_model_metrics.txt")


if __name__ == "__main__":
    main()
