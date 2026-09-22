# Data Inventory, Provenance and Cleaning Log
### T18 - Insurance Claims Fraud, Identity Abuse and Data-Leakage Analytics

Satisfies Capstone requirements C2/C3 and the Milestone 2/3 "data inventory" and "cleaning/transformation log". Field-level definitions are in `data_dictionary.csv`; computed quality checks are in `../08_outputs/data_quality_report.md` (regenerate both with `03_notebooks_or_scripts/build_data_documentation.py`).

## 1. Inventory

| # | File | Rows x cols | Role in the project | Origin | Handling |
|---|---|---|---|---|---|
| 1 | `raw/insurance_claims_dataset.csv` | 50,000 x 16 | Primary claims table (source of `Fraud_Flag`) | Source file supplied to the group | Synthetic/de-identified; project use only |
| 2 | `raw/employee_data.csv` | 1,200 x 10 | Claims-agent roster | Source file supplied to the group | Synthetic; bank columns dropped in processing |
| 3 | `raw/insider_threat_clean_dataset.csv` | 118,614 x 22 | Pool of insider-behaviour profiles attached to agents | Source file supplied to the group | Contains sensitive attributes; four are never used in scoring |
| 4 | `raw/insurance_fraud_data.csv` | 12,002 x 29 | Feature enrichment for claims (8 columns carried over) | Source file supplied to the group | Contains `*` placeholders and impossible values (see section 4) |
| 5 | `raw/portal_authentication_device_data.csv` | 42,597 x 12 | Identity/access activity source (portal logins, devices) | **Generated** by `generate_portal_auth_device_data.py` | Fully synthetic |
| 6 | `raw/document_metadata_ocr_text.csv` | 220 x 10 | First-generation document text source | **Generated** by `generate_document_text_source.py` | Superseded by file 9 |
| 7 | `processed/claims_identity_kyc_linked.csv` | 50,000 x 29 | Analysis table for models, simulation, intelligence | **Derived** by `link_and_engineer_sources.py` | Synthetic linkage - not observed facts |
| 8 | `processed/staff_access_investigation_linked.csv` | 1,200 x 37 | Staff access, override and leakage-behaviour table | **Derived** by `link_and_engineer_sources.py` | Use Staff_ID in shared outputs |
| 9 | `processed/document_metadata_ocr_text_linked.csv` | 220 x 12 | Unstructured text source (NLP) keyed to real claim/customer IDs | **Derived** by `link_and_engineer_sources.py` | Synthetic text, synthetic names and ID-style numbers |

The four analytical sources used in the models are: claims + identity/KYC (7), portal authentication/device (5), staff access and investigation (8), and documents/OCR text (9). Combined volume 94,017 records; 220 text records.

### Source origin and licence - ACTION FOR THE GROUP
The repository does not record where files 1-4 were downloaded from or under what licence. Milestone 3 (Section 9.2) requires "source, licence/provenance" for every data source. **Before submitting, complete this table:**

| File | Publisher / download URL | Licence / permission | Date obtained |
|---|---|---|---|
| `insurance_claims_dataset.csv` | *[group to fill in]* | *[group to fill in]* | *[group to fill in]* |
| `employee_data.csv` | *[group to fill in]* | *[group to fill in]* | *[group to fill in]* |
| `insider_threat_clean_dataset.csv` | *[group to fill in]* | *[group to fill in]* | *[group to fill in]* |
| `insurance_fraud_data.csv` | *[group to fill in]* | *[group to fill in]* | *[group to fill in]* |

## 2. Generation record (what is real and what is planted)

All generation uses seed 42 (`random`, `numpy`, `Faker`). Data files are shipped, so results reproduce without re-running the generators; Faker is only needed to regenerate.

| Signal | How it was created | Parameters (from the scripts) | Observed in the data |
|---|---|---|---|
| Identity/KYC change | Bernoulli draw conditional on `Fraud_Flag` (`link_and_engineer_sources.py`) | 0.42 for fraud claims, 0.10 otherwise; KYC date 0-21 days before the claim | 41% of fraud vs 10% of non-fraud claims |
| Auto-fraud enrichment (8 `Enriched_*` columns) | Rows sampled from `insurance_fraud_data.csv` **matched on the fraud label**, not by a shared key | Fraud claims draw from the dataset's fraud rows, others from its non-fraud rows | Weak per-feature separation, preserved from the source |
| Agent handling | Weighted random assignment of claims to agents | Exponential weights | 1,169 of 1,200 agents handle at least one claim |
| Insider behaviour profile per agent | One profile row attached per agent | 35% chance of a "malicious" profile for agents in the top quartile of fraud exposure with 5+ claims; 4% otherwise | 133 of 1,200 agents labelled at-risk |
| Override activity, investigation outcome, access time | Generated per agent, correlated with the risk label | See script | One record per agent, not a full access log |
| Portal logins | Generated per customer from the claims table | New-device login 55% after a KYC change (6% background); unusual location 35% for elevated-fraud-risk customers (else 0); failed-first-attempt 28% (fraud-linked) vs 6% | New-device login rate 48% for identity-changed customers vs 6% for others; failed-then-success 27% vs 6% for fraud-linked vs other customers |
| Documents | Template text with synthetic names; 35% of documents drawn from identity-changed claims; 18% chance that a batch becomes a duplicate group of 2-3 documents | See script | 220 documents, 31 duplicate groups, each spanning 2-3 customers |

**Consequence for interpretation.** The relationships between identity change, device behaviour, staff overrides, duplicate documents and the fraud label exist because the generators put them there. The analysis therefore demonstrates that the pipeline *recovers planted patterns*, and how sensitive the models are to them. It does not establish real-world fraud rates or drivers. This limitation is stated in the README, the dashboard banner and every intelligence brief.

## 3. Cleaning and transformation log

| Step | Where | What was done | Records affected |
|---|---|---|---|
| T1 | `link_and_engineer_sources.py` | Bank routing/account columns dropped from the agent roster (ethical control) | 1,200 rows, 2 columns |
| T2 | same | `AGENT_ID` -> `Staff_ID`, `AGENT_NAME` -> `Staff_Name` | 1,200 |
| T3 | same | Added `Claim_Date`, `Handled_By_Staff_ID`, `Identity_Changed_Flag`, `KYC_Update_Date`, `Days_KYC_To_Claim` | 50,000 |
| T4 | same | Eight auto-fraud fields carried in as `Enriched_*` (conditional match on fraud label) | 50,000 |
| T5 | same | Per-agent `claims_handled`, `fraud_claims_handled`, `fraud_rate_handled` computed from claims; insider profile attached; `is_malicious` renamed `Insider_Risk_Label` | 1,200 |
| T6 | same | Document IDs re-keyed to real `Claim_ID`/`Customer_ID`; similarity score (difflib ratio) and flag (HIGH >= 0.90, MEDIUM >= 0.70) computed | 220 |
| T7 | `supervised_fraud_model.py` | `*` placeholder in `Enriched_Witness_Present` treated as missing and imputed with the column mode (0 = no witness); rows retained | 340 (0.68%) |
| T8 | same | Label-proxy fields `Fraud_Risk_Score` and `Claim_Status` excluded from features; identifiers excluded; categorical fields one-hot encoded; numeric fields scaled for logistic regression only | all |
| T9 | `unsupervised_anomaly_model.py` | 42,597 login events aggregated to 18,409 per-customer behavioural profiles; features standardised | 18,409 |
| T10 | `intelligence_products.py` | Staff referred to by `Staff_ID` only; sensitive personal attributes excluded from all scores | all outputs |
| T11 | file hygiene, 20 Sep 2026 | Removed a stray space from `insurance_claims_dataset .csv` so the linking script's path resolves | 1 file |

## 4. Known data-quality issues left as they are
- `insurance_fraud_data.csv` has `*` placeholders in `marital_status`, `claim_date`, `claim_day_of_week`, `age_of_vehicle` and `injury_claim`, impossible values (`age_of_driver` up to 278, negative `annual_income`, `zip_code` 0) and 8 missing labels. None of these columns is carried into the claims table, so they do not affect any model. Counts are in `../08_outputs/data_quality_report.md`.
- `late_exit_flag` in the insider data is constant 0 and carries no information.
- The agent roster has US-style addresses and states alongside Namibian claim data - another sign the roster is a generic synthetic file.

## 5. Privacy and handling
- No real customer, staff or credential data is intended to be present. Names, addresses and ID-number-style strings (which include SSN-format values) in the text and roster files are synthetic. If any of these files turns out to come from a real organisation, stop and ask the facilitator before submitting.
- The processed staff table still contains `Staff_Name` and address fields, although the Charter (Section 6) promises staff identities will be anonymised. All shared outputs use `Staff_ID` only; hashing or dropping names and addresses from the processed table is backlog item B-06.
- Do not put raw OCR text, names or ID-style numbers on presentation slides or in public repositories.
- Sensitive attributes in the insider dataset (`has_criminal_record`, `has_medical_history`, `has_foreign_citizenship`, `employee_origin_country`, `hostility_country_level`) are never used in a model or score.
