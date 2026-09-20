# Baseline EDA Summary — T18 Insurance Claims Fraud & Identity Risk Analytics

Generated automatically from the four linked data sources. Charts are saved in `08_outputs/eda_charts/`.

## 1. Claims and Identity/KYC Records

Rows: 50,000 | Columns: 29 | Unique customers: 18,409

- **claims_identity_kyc_linked.csv**: missing values found in 2 column(s):
  - `KYC_Update_Date`: 41034 missing (82.07%)
  - `Days_KYC_To_Claim`: 41034 missing (82.07%)


**Key descriptive statistics:**

|       |   Claim_Amount |   Premium_Amount |   Coverage_Amount |   Credit_Score |   Fraud_Risk_Score |
|:------|---------------:|-----------------:|------------------:|---------------:|-------------------:|
| count |       50000    |         50000    |          50000    |       50000    |           50000    |
| mean  |       25086.7  |          2602.55 |          52691    |         575.93 |              49.85 |
| std   |       14362.8  |          1387.69 |          27516.1  |         159.28 |              28.87 |
| min   |         501.24 |           200.05 |           5001.71 |         300    |               0    |
| 25%   |       12545    |          1401    |          28896.9  |         438    |              24.88 |
| 50%   |       25059.1  |          2599.5  |          52643.9  |         577    |              49.6  |
| 75%   |       37500.6  |          3806.92 |          76676.1  |         714    |              74.86 |
| max   |       49999.8  |          4999.86 |          99996.5  |         849    |             100    |


Overall fraud rate: **24.9%** of claims flagged fraudulent.


**Charts:** `01_fraud_flag_distribution.png`, `02_claim_amount_by_fraud.png`, `03_identity_change_by_fraud.png`


**Baseline hypothesis (H1):** identity/KYC changes are markedly more common among fraudulent claims than legitimate ones, supporting the problem statement's core assumption.


## 2. Staff Access and Investigation Records

Rows: 1,200 | Columns: 37

- **staff_access_investigation_linked.csv**: missing values found in 3 column(s):
  - `ADDRESS_LINE2`: 1021 missing (85.08%)
  - `CITY`: 7 missing (0.58%)
  - `Record_Accessed`: 31 missing (2.58%)


Staff flagged Insider_Risk_Label=1: **133** of 1200 agents (11.1%).


**Charts:** `04_insider_risk_distribution.png`, `05_fraud_exposure_by_insider_risk.png`


**Baseline hypothesis (H2):** agents flagged as insider-risk tend to have handled a higher proportion of fraudulent claims than other agents.


## 3. Document Metadata and OCR/Text Data

Rows: 220 | Columns: 12

- **document_metadata_ocr_text_linked.csv**: missing values found in 1 column(s):
  - `duplicate_group_id`: 143 missing (65.0%)


Among HIGH-similarity documents, 32.1% are linked to a claim with an identity/KYC change - suggesting duplicated/templated document content clusters around identity-change cases.


**Charts:** `06_document_similarity_distribution.png`, `07_document_type_volume.png`


## 4. Portal Authentication and Device Data

Rows: 42,597 | Columns: 12 | Unique customers: 18,409

- **portal_authentication_device_data.csv**: no missing values in any column.


Off-hours logins (22:00-06:00): 12.1% of all login events.


**Charts:** `08_new_device_logins.png`, `09_authentication_outcomes.png`
