"""
link_and_engineer_sources.py

SAS821S · T18 - Insurance Claims Fraud, Identity Abuse and Data-Leakage Analytics

PURPOSE
-------
The three uploaded files do NOT share a common key out of the box:
    - insurance_claims_dataset.csv   (Claim_ID, Customer_ID, Policy_Number, ...)
    - employee_data.csv              (AGENT_ID, ...)
    - insider_threat_clean_dataset.csv  (no ID column at all)

This script engineers the missing links so all sources connect through two
keys: Customer_ID (claimant) and Staff_ID (the agent/employee who handled a
claim) - matching the entity-linkage requirement in the Charter (Section 6 /
Section 7 "Security intelligence") and the Implementation Plan's
"Investigation, access analytics and intelligence" section.

It also adds the specific fields the Charter promised but the raw files don't
contain (Claim_Date, Identity_Changed_Flag, KYC_Update_Date, Override_Activity,
Investigation_Outcome, etc.), and realigns the previously generated document
text source so its Customer_ID values are drawn from the real claims dataset
instead of standalone fake IDs - so a document can genuinely be traced back to
a claim and an agent.

INPUTS  (expected in /mnt/user-data/uploads/)
    insurance_claims_dataset.csv
    employee_data.csv
    insider_threat_clean_dataset.csv

OUTPUTS (written to 02_data/processed/)
    claims_identity_kyc_linked.csv
    staff_access_investigation_linked.csv
    document_metadata_ocr_text_linked.csv
    linkage_summary.txt   <- join validation report, read this first

Run:
    python3 link_and_engineer_sources.py
"""

import os
import random
import difflib
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from faker import Faker

fake = Faker()
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
Faker.seed(SEED)

UPLOAD_DIR = "02_data/raw"
OUT_DIR = "02_data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

CLAIMS_IN = os.path.join(UPLOAD_DIR, "insurance_claims_dataset.csv")
EMP_IN = os.path.join(UPLOAD_DIR, "employee_data.csv")
INSIDER_IN = os.path.join(UPLOAD_DIR, "insider_threat_clean_dataset.csv")

CLAIMS_OUT = os.path.join(OUT_DIR, "claims_identity_kyc_linked.csv")
STAFF_OUT = os.path.join(OUT_DIR, "staff_access_investigation_linked.csv")
DOCS_OUT = os.path.join(OUT_DIR, "document_metadata_ocr_text_linked.csv")
SUMMARY_OUT = os.path.join(OUT_DIR, "linkage_summary.txt")


def log(lines, msg):
    print(msg)
    lines.append(msg)


def random_date(days_back_start=730, days_back_end=0):
    start = datetime.now() - timedelta(days=days_back_start)
    end = datetime.now() - timedelta(days=days_back_end)
    delta = (end - start).days
    return start + timedelta(days=random.randint(0, max(delta, 1)))


# =============================================================================
# STEP 1 - CLAIMS + IDENTITY/KYC: engineer the fields the charter promised
# =============================================================================
def build_claims(emp_ids, summary):
    claims = pd.read_csv(CLAIMS_IN)
    n = len(claims)
    log(summary, f"[1] Loaded {n} claims, {claims['Customer_ID'].nunique()} unique customers")

    # --- Claim_Date: charter lists this as a key field; source file lacks it
    claims["Claim_Date"] = [random_date() for _ in range(n)]

    # --- Assign the agent who handled each claim (this is the join key to
    #     the staff table). Not uniform-random: real agents handle very
    #     different volumes, so draw from a skewed distribution.
    weights = np.random.exponential(scale=1.0, size=len(emp_ids))
    weights = weights / weights.sum()
    claims["Handled_By_Staff_ID"] = np.random.choice(emp_ids, size=n, p=weights)

    # --- Identity/KYC change fields. Deliberately correlated with Fraud_Flag
    #     so the "identity change precedes suspicious claim" pattern your
    #     problem statement describes is actually present in the data, not
    #     just asserted in prose.
    is_fraud = (claims["Fraud_Flag"] == "Yes").values
    base_rate = 0.10          # background rate of identity/KYC changes
    fraud_rate = 0.42         # elevated rate among fraudulent claims
    change_prob = np.where(is_fraud, fraud_rate, base_rate)
    claims["Identity_Changed_Flag"] = (np.random.random(n) < change_prob).astype(int)

    kyc_dates = []
    for changed, claim_date in zip(claims["Identity_Changed_Flag"], claims["Claim_Date"]):
        if changed:
            # KYC change happens shortly BEFORE the claim (0-21 days prior) -
            # this is the specific suspicious pattern flagged in the charter
            kyc_dates.append(claim_date - timedelta(days=random.randint(0, 21)))
        else:
            kyc_dates.append(pd.NaT)
    claims["KYC_Update_Date"] = kyc_dates

    claims["Days_KYC_To_Claim"] = (
        pd.to_datetime(claims["Claim_Date"]) - pd.to_datetime(claims["KYC_Update_Date"])
    ).dt.days

    claims.to_csv(CLAIMS_OUT, index=False)
    log(summary, f"[1] Wrote {CLAIMS_OUT} ({len(claims)} rows, {len(claims.columns)} cols)")
    log(summary, f"    Identity_Changed_Flag rate: fraud={claims.loc[is_fraud,'Identity_Changed_Flag'].mean():.2f}"
                  f" vs non-fraud={claims.loc[~is_fraud,'Identity_Changed_Flag'].mean():.2f}")
    return claims


# =============================================================================
# STEP 1B - ENRICH CLAIMS with real fraud-indicator features from
#           insurance_fraud_data.csv (a separate auto-insurance fraud
#           dataset with no shared ID). Since there is no natural key,
#           rows are matched CONDITIONALLY on Fraud_Flag: a fraudulent claim
#           is enriched using a randomly sampled row from that dataset's
#           genuinely-fraudulent cases, and a non-fraud claim from its
#           genuinely non-fraudulent cases. This is a documented synthetic
#           linkage (not a real join) that preserves the real feature/target
#           correlation from the source dataset rather than adding pure
#           noise. Only columns NOT already present in the claims table are
#           carried over (Age/Gender already exist there).
# =============================================================================
FRAUD_ENRICH_IN = os.path.join(UPLOAD_DIR, "insurance_fraud_data.csv")

ENRICH_COLUMN_MAP = {
    "address_change": "Enriched_Address_Change_Flag",
    "past_num_of_claims": "Enriched_Past_Claims_Count",
    "witness_present": "Enriched_Witness_Present",
    "police_report": "Enriched_Police_Report_Filed",
    "safety_rating": "Enriched_Safety_Rating",
    "days open": "Enriched_Days_Open",
    "form defects": "Enriched_Form_Defects_Count",
    "channel": "Enriched_Claim_Channel",
}


def enrich_claims_with_fraud_indicators(claims, summary):
    fd = pd.read_csv(FRAUD_ENRICH_IN)
    fd.columns = [c.strip() for c in fd.columns]
    log(summary, f"[1b] Loaded {len(fd)} rows from insurance_fraud_data.csv for feature enrichment "
                  f"(no shared key with claims - matched conditionally on Fraud_Flag instead)")

    fraud_pool = fd[fd["fraud reported"] == "Y"].reset_index(drop=True)
    clean_pool = fd[fd["fraud reported"] == "N"].reset_index(drop=True)

    src_cols = list(ENRICH_COLUMN_MAP.keys())
    is_fraud = (claims["Fraud_Flag"] == "Yes").values

    n_fraud = int(is_fraud.sum())
    n_clean = int((~is_fraud).sum())

    fraud_sample = fraud_pool[src_cols].sample(n=n_fraud, replace=True, random_state=SEED).reset_index(drop=True)
    clean_sample = clean_pool[src_cols].sample(n=n_clean, replace=True, random_state=SEED).reset_index(drop=True)

    enriched = pd.DataFrame(index=claims.index, columns=src_cols, dtype=object)
    enriched.loc[is_fraud, src_cols] = fraud_sample.values
    enriched.loc[~is_fraud, src_cols] = clean_sample.values
    enriched = enriched.rename(columns=ENRICH_COLUMN_MAP)

    claims = pd.concat([claims.reset_index(drop=True), enriched.reset_index(drop=True)], axis=1)

    claims.to_csv(CLAIMS_OUT, index=False)
    log(summary, f"[1b] Re-wrote {CLAIMS_OUT} with {len(ENRICH_COLUMN_MAP)} enrichment columns "
                  f"({len(claims.columns)} cols total)")
    log(summary, "     Enrichment columns added: " + ", ".join(ENRICH_COLUMN_MAP.values()))
    fraud_addr_rate = claims.loc[is_fraud, "Enriched_Address_Change_Flag"].mean()
    clean_addr_rate = claims.loc[~is_fraud, "Enriched_Address_Change_Flag"].mean()
    log(summary, f"     Enriched_Address_Change_Flag rate: fraud={fraud_addr_rate:.2f} vs non-fraud={clean_addr_rate:.2f}"
                  f" (signal preserved from source dataset)")
    return claims


# =============================================================================
# STEP 2 - STAFF ACCESS & INVESTIGATION: link employee roster to behavioural
#          data and to the claims each agent actually handled
# =============================================================================
def build_staff(claims, summary):
    emp = pd.read_csv(EMP_IN)
    log(summary, f"[2] Loaded {len(emp)} employees/agents")

    # Ethical control per Charter Section 11: drop raw bank account/routing
    # numbers even though this is a public/synthetic dataset - they serve no
    # analytical purpose here and match the charter's "no real customer/
    # sensitive info retained" commitment.
    emp = emp.drop(columns=["EMP_ROUTING_NUMBER", "EMP_ACCT_NUMBER"], errors="ignore")
    emp = emp.rename(columns={"AGENT_ID": "Staff_ID", "AGENT_NAME": "Staff_Name"})

    # --- Per-agent fraud exposure, computed from the claims they handled.
    #     This is the genuine cross-source signal: agents who handled a
    #     disproportionate share of fraudulent claims are more likely to be
    #     linked to an insider-threat behavioural profile below.
    agent_stats = (
        claims.groupby("Handled_By_Staff_ID")
        .agg(claims_handled=("Claim_ID", "count"),
             fraud_claims_handled=("Fraud_Flag", lambda s: (s == "Yes").sum()))
        .reset_index()
        .rename(columns={"Handled_By_Staff_ID": "Staff_ID"})
    )
    agent_stats["fraud_rate_handled"] = (
        agent_stats["fraud_claims_handled"] / agent_stats["claims_handled"]
    )
    emp = emp.merge(agent_stats, on="Staff_ID", how="left")
    emp[["claims_handled", "fraud_claims_handled", "fraud_rate_handled"]] = emp[
        ["claims_handled", "fraud_claims_handled", "fraud_rate_handled"]
    ].fillna(0)

    # --- Attach an insider-threat behavioural profile to each agent.
    #     118,614 behavioural rows vs ~1,200 agents: sample WITHOUT the
    #     is_malicious label leaking directly from fraud_rate (that would be
    #     circular). Instead, agents in the top quartile of fraud_rate_handled
    #     are given a higher chance of being sampled from the is_malicious=1
    #     pool - a soft, realistic correlation rather than a hard rule.
    insider = pd.read_csv(INSIDER_IN)
    log(summary, f"[2] Loaded {len(insider)} insider-threat behavioural records "
                  f"({insider['is_malicious'].sum()} malicious)")

    malicious_pool = insider[insider["is_malicious"] == 1].reset_index(drop=True)
    normal_pool = insider[insider["is_malicious"] == 0].reset_index(drop=True)

    high_risk_threshold = emp["fraud_rate_handled"].quantile(0.75)

    profiles = []
    for _, row in emp.iterrows():
        elevated = row["fraud_rate_handled"] >= high_risk_threshold and row["claims_handled"] >= 5
        draw_malicious = random.random() < (0.35 if elevated else 0.04)
        pool = malicious_pool if draw_malicious and len(malicious_pool) else normal_pool
        profiles.append(pool.sample(n=1, random_state=random.randint(0, 10_000)).iloc[0])

    profile_df = pd.DataFrame(profiles).reset_index(drop=True)
    emp = pd.concat([emp.reset_index(drop=True), profile_df], axis=1)
    emp = emp.rename(columns={"is_malicious": "Insider_Risk_Label"})

    # --- Fields the Charter promised for this source that neither raw file
    #     has: Access_Time, Record_Accessed, Override_Activity, Investigation
    #     Outcome. Override_Activity/Investigation_Outcome are correlated
    #     with Insider_Risk_Label so they're not pure noise.
    access_times, records_accessed, overrides, outcomes = [], [], [], []
    claims_by_staff = claims.groupby("Handled_By_Staff_ID")["Claim_ID"].apply(list).to_dict()

    for _, row in emp.iterrows():
        sid = row["Staff_ID"]
        handled = claims_by_staff.get(sid, [])
        access_times.append(random_date(days_back_start=180))
        records_accessed.append(random.choice(handled) if handled else "N/A")

        if row["Insider_Risk_Label"] == 1:
            overrides.append(random.choice(["Approved above authority limit",
                                             "Bypassed secondary verification",
                                             "Unscheduled record access", "No override recorded"]))
            outcomes.append(random.choices(
                ["Confirmed - insider risk", "Under investigation", "Unsubstantiated"],
                weights=[0.45, 0.35, 0.20])[0])
        else:
            overrides.append(random.choices(["No override recorded", "Approved above authority limit"],
                                             weights=[0.9, 0.1])[0])
            outcomes.append(random.choices(["No findings", "Unsubstantiated"],
                                            weights=[0.85, 0.15])[0])

    emp["Access_Time"] = access_times
    emp["Record_Accessed"] = records_accessed
    emp["Override_Activity"] = overrides
    emp["Investigation_Outcome"] = outcomes

    emp.to_csv(STAFF_OUT, index=False)
    log(summary, f"[2] Wrote {STAFF_OUT} ({len(emp)} rows, {len(emp.columns)} cols)")
    log(summary, f"    Insider_Risk_Label positives: {emp['Insider_Risk_Label'].sum()} of {len(emp)} agents")
    return emp


# =============================================================================
# STEP 3 - DOCUMENT / OCR TEXT SOURCE: realign Customer_ID to the real claims
#          dataset so documents genuinely link to claims and claimants
# =============================================================================
FILE_TYPES = ["PDF", "JPG", "PNG", "DOCX"]
TEMPLATES = [
    ("claim_form",
     "INSURANCE CLAIM FORM. Claimant: {name}. Policy Number: {policy}. "
     "Date of Incident: {date}. Description: Vehicle damage sustained in a "
     "collision on {road}. Estimated repair cost: N$ {amount}. Signed: {name}."),
    ("id_document",
     "NATIONAL IDENTITY DOCUMENT. Full Name: {name}. ID Number: {idnum}. "
     "Date of Birth: {dob}. Nationality: Namibian. Issued by Ministry of "
     "Home Affairs. Address updated on {date}."),
    ("kyc_update",
     "KYC UPDATE FORM. Customer: {name}. Previous Address: {old_addr}. "
     "New Address: {new_addr}. Reason for change: relocation. Verified by "
     "staff member on {date}."),
    ("investigation_note",
     "INVESTIGATION NOTE. Case relates to claimant {name}. Flagged for "
     "review due to unusual document upload pattern following identity "
     "change on {date}. Staff override recorded by {staff}."),
]
FILLER_SENTENCES = [
    "Handled by branch office during standard working hours.",
    "Customer contacted via telephone for confirmation.",
    "Supporting documents received by email attachment.",
    "Additional photographic evidence submitted separately.",
    "Follow-up scheduled with the claims handler within 5 working days.",
]


def build_ocr_text(key, template, name, staff_name, boilerplate_only=False):
    base = template.format(
        name=name, policy=f"POL-{random.randint(100000,999999)}",
        date=random_date().strftime("%Y-%m-%d"), road=fake.street_name(),
        amount=f"{random.randint(500,85000):,}", idnum=fake.ssn(),
        dob=fake.date_of_birth(minimum_age=18, maximum_age=75).isoformat(),
        old_addr=fake.address().replace("\n", ", "),
        new_addr=fake.address().replace("\n", ", "), staff=staff_name,
    )
    if boilerplate_only:
        return base
    filler = " ".join(random.sample(FILLER_SENTENCES, random.randint(1, 2)))
    return f"{base} {filler}"


def similarity(a, b):
    return round(difflib.SequenceMatcher(None, a, b).ratio(), 3)


def build_documents(claims, staff, summary, n_records=220):
    # Bias document generation toward customers who had an identity change
    # AND toward staff with an insider-risk flag, so the resulting document
    # set actually overlaps with the suspicious claims/staff you'll be
    # investigating - not a random, disconnected sample.
    flagged_claims = claims[claims["Identity_Changed_Flag"] == 1]
    normal_claims = claims[claims["Identity_Changed_Flag"] == 0]
    risky_staff_ids = set(staff.loc[staff["Insider_Risk_Label"] == 1, "Staff_ID"])

    n_flagged_docs = int(n_records * 0.35)
    n_normal_docs = n_records - n_flagged_docs

    sample_flagged = flagged_claims.sample(n=min(n_flagged_docs, len(flagged_claims)),
                                            random_state=SEED, replace=len(flagged_claims) < n_flagged_docs)
    sample_normal = normal_claims.sample(n=min(n_normal_docs, len(normal_claims)),
                                          random_state=SEED, replace=False)
    chosen = pd.concat([sample_flagged, sample_normal]).reset_index(drop=True)

    records = []
    duplicate_group_counter = 0
    doc_counter = 1
    i = 0
    while i < len(chosen):
        make_dup_group = random.random() < 0.18 and (len(chosen) - i) >= 2
        group_size = random.randint(2, 3) if make_dup_group else 1
        group_size = min(group_size, len(chosen) - i)
        group_id = ""
        key, template = random.choice(TEMPLATES)
        name = fake.name()
        staff_id = random.choice(list(risky_staff_ids)) if (make_dup_group and risky_staff_ids) else None
        staff_name = staff.loc[staff["Staff_ID"] == staff_id, "Staff_Name"].values
        staff_name = staff_name[0] if len(staff_name) else fake.name()

        if make_dup_group:
            duplicate_group_counter += 1
            group_id = f"DUPGRP-{duplicate_group_counter:03d}"
        base_text = build_ocr_text(key, template, name, staff_name, boilerplate_only=make_dup_group)

        for _ in range(group_size):
            row = chosen.iloc[i]
            text = base_text if make_dup_group and random.random() < 0.5 else build_ocr_text(
                key, template, name, staff_name, boilerplate_only=make_dup_group)
            records.append({
                "document_id": f"DOC-{doc_counter:05d}",
                "Customer_ID": row["Customer_ID"],
                "Claim_ID": row["Claim_ID"],
                "upload_time": random_date(days_back_start=180).strftime("%Y-%m-%d %H:%M:%S"),
                "file_type": random.choice(FILE_TYPES),
                "file_size_kb": random.randint(80, 4500),
                "document_type": key,
                "ocr_text": text,
                "duplicate_group_id": group_id,
                "linked_identity_change": row["Identity_Changed_Flag"],
            })
            doc_counter += 1
            i += 1

    docs = pd.DataFrame(records)

    # similarity scoring (same-type peer comparison, calibrated threshold)
    scores, flags = [], []
    for idx, r in docs.iterrows():
        if r["duplicate_group_id"]:
            peers = docs[(docs["duplicate_group_id"] == r["duplicate_group_id"]) & (docs.index != idx)]["ocr_text"]
        else:
            same_type = docs[(docs["document_type"] == r["document_type"]) & (docs.index != idx)]["ocr_text"]
            peers = same_type.sample(n=min(6, len(same_type)), random_state=idx) if len(same_type) else pd.Series([])
        best = max((similarity(r["ocr_text"], p) for p in peers), default=0.0)
        scores.append(best)
        flags.append("HIGH" if best >= 0.90 else ("MEDIUM" if best >= 0.70 else "LOW"))
    docs["document_similarity_score"] = scores
    docs["document_similarity_flag"] = flags

    docs.to_csv(DOCS_OUT, index=False)
    log(summary, f"[3] Wrote {DOCS_OUT} ({len(docs)} rows, {len(docs.columns)} cols)")
    log(summary, f"    Documents linked to identity-changed claims: {docs['linked_identity_change'].sum()} of {len(docs)}")
    log(summary, f"    HIGH similarity flags: {(docs['document_similarity_flag']=='HIGH').sum()}")
    return docs


# =============================================================================
# STEP 4 - VALIDATE THE JOINS ACTUALLY WORK
# =============================================================================
def validate_joins(claims, staff, docs, summary):
    log(summary, "\n[4] JOIN VALIDATION")

    merged_cs = claims.merge(staff, left_on="Handled_By_Staff_ID", right_on="Staff_ID", how="left")
    match_rate = merged_cs["Staff_ID"].notna().mean()
    log(summary, f"    Claims -> Staff join match rate: {match_rate:.1%}")

    merged_cd = claims.merge(docs, on="Claim_ID", how="inner")
    log(summary, f"    Claims <-> Documents join: {len(merged_cd)} matching rows "
                  f"(documents were generated FROM a claims sample, so this confirms the key works)")

    same_cust = docs.merge(claims[["Customer_ID"]].drop_duplicates(), on="Customer_ID", how="inner")
    log(summary, f"    Documents -> Claims Customer_ID coverage: {len(same_cust)} of {len(docs)} documents "
                  f"resolve to a real Customer_ID in the claims table")

    log(summary, "\n    Usable join keys:")
    log(summary, "      Customer_ID   : claims_identity_kyc_linked.csv <-> document_metadata_ocr_text_linked.csv")
    log(summary, "      Claim_ID      : claims_identity_kyc_linked.csv <-> document_metadata_ocr_text_linked.csv")
    log(summary, "      Staff_ID      : staff_access_investigation_linked.csv <-> claims_identity_kyc_linked.csv (via Handled_By_Staff_ID)")


def main():
    summary = []
    log(summary, "=" * 70)
    log(summary, "T18 DATA LINKAGE - build report")
    log(summary, "=" * 70)

    emp_ids_preview = pd.read_csv(EMP_IN)["AGENT_ID"].tolist()
    claims = build_claims(emp_ids_preview, summary)
    claims = enrich_claims_with_fraud_indicators(claims, summary)
    staff = build_staff(claims, summary)
    docs = build_documents(claims, staff, summary)
    validate_joins(claims, staff, docs, summary)

    with open(SUMMARY_OUT, "w") as f:
        f.write("\n".join(summary))
    log(summary, f"\nSummary written to {SUMMARY_OUT}")


if __name__ == "__main__":
    main()
