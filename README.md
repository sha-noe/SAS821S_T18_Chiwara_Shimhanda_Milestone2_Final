# T18 — Insurance Claims Fraud, Identity Abuse and Data-Leakage Analytics

**SAS821S · Security Analytics · Capstone Project · Semester 2, 2026**
**Group members:** Shanon Chiwara (223127744) — Data and Modelling Lead · Ephath Shimhanda (216069076) — Security Engineering and Intelligence Lead

## 1. What this project does

An insurer has identified suspicious claims submitted after identity-profile changes, unusual document uploads and staff overrides. This project builds a decision-support pipeline that:

- Links four data sources (claims/KYC, portal authentication/device, staff access, document/OCR text)
- Trains a supervised fraud classifier (Logistic Regression + Random Forest)
- Runs an unsupervised anomaly-detection model on customer login behaviour (Isolation Forest)
- Reconstructs chronological incident timelines for high-risk cases
- Simulates 3 fraud-control scenarios (Monte Carlo)
- Mines the OCR document text for suspicious patterns and classifies document type
- Presents all findings in a single analyst-facing dashboard (`dashboard.html`)

## 2. Requirements

- Python 3.10 or later
- Install dependencies:
  ```
  pip install -r requirements.txt
  ```

## 3. Folder structure

```
GROUP_T18_INSURANCE_FRAUD/
├── 01_charter/                  Project Charter (Milestone 1)
├── 02_data/
│   ├── raw/                     Original + generated source data
│   └── processed/               Linked/engineered datasets
├── 03_notebooks_or_scripts/     Data linking and generation scripts
├── 04_models/                   Supervised + unsupervised ML scripts and outputs
├── 05_simulation/               Control-scenario Monte Carlo simulation
├── 06_text_mining/              NLP/text-mining on OCR document text
├── 07_dashboard_or_prototype/   The analyst dashboard (dashboard.html)
├── 08_outputs/                  Charts, EDA summary, incident timelines
├── 09_documentation/            Implementation Plan, recommendations
├── README.md                    This file
└── requirements.txt
```

## 4. How to reproduce everything, in order

The scripts have dependencies on each other's output — **run them in this exact order** from the project root folder:

```
# 1. Generate the document/OCR text source
python 03_notebooks_or_scripts/generate_document_text_source.py

# 2. Link the four core data sources together (claims, staff, insider-threat, documents)
python 03_notebooks_or_scripts/link_and_engineer_sources.py

# 3. Generate the portal authentication/device login data
python 03_notebooks_or_scripts/generate_portal_auth_device_data.py

# 4. Run baseline EDA (produces charts + summary in 08_outputs/)
python 03_notebooks_or_scripts/run_eda_baseline.py

# 5. Train the supervised fraud model
python 04_models/supervised_fraud_model.py

# 6. Train the unsupervised anomaly model
python 04_models/unsupervised_anomaly_model.py

# 7. Run adversarial/robustness testing on the trained model
python 04_models/adversarial_robustness_testing.py

# 8. Build incident investigation timelines (depends on steps 5 & 6 output)
python 03_notebooks_or_scripts/incident_timeline_investigation.py

# 9. Run the control-scenario simulation
python 05_simulation/security_control_simulation.py

# 10. Run NLP/text mining on the document data
python 06_text_mining/nlp_text_mining.py

# 11. Build the analyst dashboard (pulls together all outputs above)
python 07_dashboard_or_prototype/build_dashboard.py
```

After step 11, open `07_dashboard_or_prototype/dashboard.html` directly in any browser — no server required.

## 5. Data sources and provenance

| Source | Origin | Notes |
|---|---|---|
| Insurance claims | Uploaded real-structure dataset (50,000 rows) | Enriched with synthetic Identity/KYC change fields per Charter Section 6 |
| Employee/staff roster | Uploaded dataset (1,200 agents) | Bank account fields dropped (not needed, ethical control) |
| Insider-threat behavioural data | Uploaded dataset (118,614 rows) | Sampled and linked to staff via engineered correlation with fraud exposure |
| Auto-fraud indicator dataset | Uploaded dataset (12,002 rows) | Used only to enrich claims features (address change, past claims, etc.), matched conditionally on fraud outcome — not a real join |
| Portal authentication/device logs | Fully synthetic (generated) | No real source existed; built to match the Charter's planned 4th data source |
| Document metadata/OCR text | Fully synthetic (generated) | Templated OCR text; documented limitation — lacks real scanning noise |

None of the original files shared a common ID key. Where synthetic linkage was used instead of a real join, this is documented in `02_data/processed/linkage_summary.txt` and in the relevant script's docstring.

## 6. Known limitations

- Document OCR text is template-generated, not real scanned OCR — the NLP classification task is therefore easier than a real-world equivalent (see `06_text_mining/nlp_report.txt` for the full caveat).
- Staff access timestamps in incident timelines are anchored to a claim the staff member handled, not guaranteed to be the specific claim displayed, since the staff table stores one access record per agent rather than a full multi-claim access log.
- The auto-fraud enrichment features (`Enriched_*` columns) come from a synthetic conditional match on fraud outcome, not a real per-claim join — documented in `link_and_engineer_sources.py`.

## 7. Responsible AI disclosure

Generative AI (Claude, Anthropic) was used to assist with code generation, data engineering scripts, and documentation drafting, per course policy Section 13. All outputs were reviewed, tested, and validated by the group before inclusion. Model results, correlations, and reported figures were independently verified by re-running scripts and inspecting outputs, not taken on faith.
