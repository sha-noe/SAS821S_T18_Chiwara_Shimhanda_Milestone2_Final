"""
nlp_text_mining.py

SAS821S · T18 - Insurance Claims Fraud, Identity Abuse and Data-Leakage Analytics
Member B (Ephath) - Text mining/intelligence workflow (Charter RACI)

PURPOSE
-------
Satisfies Charter C8 "Text mining/NLP": analyse unstructured security text
and extract patterns, indicators, entities, categories, topics or sentiment.
Uses a genuine classification task (not just word counts): predicting
document_type from OCR text via TF-IDF + Logistic Regression, which counts
as the "classification, topic analysis... or another NLP technique"
required by Section 3 (Session 9) of the Capstone brief.

INPUT:  02_data/processed/document_metadata_ocr_text_linked.csv
OUTPUT: 06_text_mining/nlp_report.txt
        06_text_mining/keyword_frequency_by_type.csv
        06_text_mining/suspicious_document_flags.csv
        08_outputs/model_charts/17_document_classification_confusion.png
        08_outputs/model_charts/18_top_keywords_by_type.png
"""

import os
import re
from collections import Counter

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, confusion_matrix

SEED = 42
np.random.seed(SEED)

DOCS_IN = "02_data/processed/document_metadata_ocr_text_linked.csv"
OUT_DIR = "06_text_mining"
CHART_DIR = "08_outputs/model_charts"
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)

STOPWORDS = set("""a an the of to in on for and or is are was were be been being
this that these those with by from at as it its it's their his her he she
signed date verified reason relocation attached separately business""".split())

SUSPICIOUS_TERMS = ["override", "flagged", "unusual", "investigation", "unscheduled",
                     "bypassed", "discrepancy"]

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


def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 2]
    return tokens


def main():
    log("=" * 70)
    log("TEXT MINING / NLP - Document OCR Text")
    log("=" * 70)

    docs = pd.read_csv(DOCS_IN)
    log(f"Loaded {len(docs)} documents")

    # --- Step 1: preprocessing + keyword frequency per document type -----------
    docs["tokens"] = docs["ocr_text"].apply(preprocess)
    docs["token_count"] = docs["tokens"].apply(len)

    keyword_rows = []
    for doc_type, group in docs.groupby("document_type"):
        all_tokens = [t for tokens in group["tokens"] for t in tokens]
        top_words = Counter(all_tokens).most_common(10)
        for word, count in top_words:
            keyword_rows.append({"document_type": doc_type, "keyword": word, "frequency": count})

    keyword_df = pd.DataFrame(keyword_rows)
    keyword_df.to_csv(os.path.join(OUT_DIR, "keyword_frequency_by_type.csv"), index=False)
    log(f"\nSaved keyword frequency table to {OUT_DIR}/keyword_frequency_by_type.csv")

    # --- Chart: top keywords for the 2 most document-heavy types ---------------
    top_types = docs["document_type"].value_counts().head(2).index.tolist()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, dtype in zip(axes, top_types):
        sub = keyword_df[keyword_df["document_type"] == dtype].sort_values("frequency")
        ax.barh(sub["keyword"], sub["frequency"], color="#4C72B0")
        ax.set_title(f"Top keywords: {dtype}")
    savefig(fig, "18_top_keywords_by_type.png")

    # --- Step 2: suspicious-term flagging (indicator extraction) ---------------
    def count_suspicious(tokens):
        return sum(1 for t in tokens if t in SUSPICIOUS_TERMS)

    docs["suspicious_term_count"] = docs["tokens"].apply(count_suspicious)
    docs["suspicious_term_flag"] = (docs["suspicious_term_count"] > 0).astype(int)

    flagged = docs[docs["suspicious_term_flag"] == 1][
        ["document_id", "Customer_ID", "Claim_ID", "document_type",
         "suspicious_term_count", "document_similarity_flag"]
    ].sort_values("suspicious_term_count", ascending=False)
    flagged.to_csv(os.path.join(OUT_DIR, "suspicious_document_flags.csv"), index=False)
    log(f"\nFlagged {len(flagged)} of {len(docs)} documents containing suspicious terms "
        f"({', '.join(SUSPICIOUS_TERMS)})")
    log(f"Saved to {OUT_DIR}/suspicious_document_flags.csv")

    # --- Step 3: classification task - predict document_type from OCR text ----
    # This is the genuine NLP "classification" deliverable required by the
    # brief: TF-IDF vectorisation + Logistic Regression, evaluated properly.
    log("\n--- Document type classification (TF-IDF + Logistic Regression) ---")
    X_text = docs["ocr_text"]
    y = docs["document_type"]

    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.25, random_state=SEED, stratify=y)

    vectorizer = TfidfVectorizer(max_features=300, stop_words="english", ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=SEED)
    clf.fit(X_train_tfidf, y_train)
    y_pred = clf.predict(X_test_tfidf)

    log(classification_report(y_test, y_pred))
    log("\nNOTE - documented limitation: accuracy is very high (near 100%) because the "
        "OCR text in this dataset is synthetically generated from a small number of fixed "
        "templates per document type, giving each type a distinctive, noise-free vocabulary. "
        "Real scanned OCR text would include misreads, inconsistent formatting and overlapping "
        "vocabulary across document types, making this a genuinely harder classification problem "
        "in production. This result demonstrates the pipeline works correctly, not that "
        "document-type classification is a solved problem in general.")

    labels = sorted(y.unique())
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    ConfusionMatrixDisplay(cm, display_labels=labels).plot(ax=ax, cmap="Purples", xticks_rotation=45)
    ax.set_title("Document type classification - Confusion Matrix")
    savefig(fig, "17_document_classification_confusion.png")

    # --- Step 4: cross-check - do suspicious-term documents overlap with -------
    #     HIGH similarity flags (independent corroborating signal)?
    overlap = docs[(docs["suspicious_term_flag"] == 1) &
                    (docs["document_similarity_flag"] == "HIGH")]
    log(f"\nDocuments with BOTH a suspicious term AND a HIGH similarity flag: "
        f"{len(overlap)} (independent corroborating signals)")

    with open(os.path.join(OUT_DIR, "nlp_report.txt"), "w") as f:
        f.write("\n".join(REPORT_LINES))
    log(f"\nFull report written to {OUT_DIR}/nlp_report.txt")


if __name__ == "__main__":
    main()
