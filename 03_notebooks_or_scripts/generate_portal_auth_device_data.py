"""
generate_portal_auth_device_data.py

SAS821S · T18 - Insurance Claims Fraud, Identity Abuse and Data-Leakage Analytics

Generates the 4th planned data source from the Charter (Section 6):
    "Portal Authentication and Device Data"
    Key fields: User ID, Login Time, IP Address, Device ID, Authentication
    Result, Location/Network Information.

This was the one source not yet built. It links to the existing claims table
via Customer_ID (the "User ID" logging into the claimant portal).

DESIGN LOGIC (why the numbers aren't random)
---------------------------------------------
Your Charter's problem statement specifically describes: identity/KYC change
-> new device login -> possible fraud. So this generator builds that pattern
in deliberately:

  - Every customer gets 1-3 "usual" devices established over time.
  - Customers whose claims show Identity_Changed_Flag == 1 have an elevated
    chance of logging in from a brand-new, previously-unseen device shortly
    AFTER their KYC_Update_Date - this is the exact suspicious sequence your
    problem statement describes, made detectable rather than just asserted.
  - Customers linked to a Fraud_Flag == 'Yes' claim get an elevated rate of
    failed login attempts before success (credential-stuffing-like pattern)
    and a higher chance of an unusual login location relative to their
    established "home" region.
  - Ordinary customers get realistic but unremarkable login behaviour.

INPUT:  02_data/processed/claims_identity_kyc_linked.csv
OUTPUT: 02_data/raw/portal_authentication_device_data.csv
"""

import os
import random
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
from faker import Faker

fake = Faker()
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
Faker.seed(SEED)

CLAIMS_IN = "02_data/processed/claims_identity_kyc_linked.csv"
OUT_DIR = "02_data/raw"
OUT_FILE = os.path.join(OUT_DIR, "portal_authentication_device_data.csv")

os.makedirs(OUT_DIR, exist_ok=True)

REGION_CITY_MAP = {
    "North": ["Ondangwa", "Oshakati", "Tsumeb"],
    "South": ["Keetmanshoop", "Luderitz", "Karasburg"],
    "East": ["Gobabis", "Rundu", "Katima Mulilo"],
    "West": ["Swakopmund", "Walvis Bay", "Henties Bay"],
}
NETWORK_TYPES = ["Home Wi-Fi", "Mobile Data", "Public Wi-Fi", "VPN"]


def random_login_time(days_back=180):
    dt = datetime.now() - timedelta(days=random.uniform(0, days_back))
    # Off-hours logins (22:00-05:00) are rarer but not impossible
    if random.random() < 0.12:
        hour = random.choice(list(range(22, 24)) + list(range(0, 6)))
    else:
        hour = random.randint(6, 21)
    return dt.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))


def make_device_id():
    return f"DEV-{fake.hexify(text='^^^^^^^^').upper()}"


def make_ip():
    return fake.ipv4_public()


def build_auth_data(claims):
    # One row per (customer, claim identity-risk) so we know which customers
    # are "elevated risk" for login-pattern purposes.
    customer_risk = (
        claims.groupby("Customer_ID")
        .agg(any_identity_change=("Identity_Changed_Flag", "max"),
             any_fraud=("Fraud_Flag", lambda s: (s == "Yes").any()),
             kyc_date=("KYC_Update_Date", "max"),
             region=("Region", "first"))
        .reset_index()
    )
    customer_risk["kyc_date"] = pd.to_datetime(customer_risk["kyc_date"], errors="coerce")

    records = []
    log_counter = 1

    for _, row in customer_risk.iterrows():
        cid = row["Customer_ID"]
        elevated_device_risk = bool(row["any_identity_change"])
        elevated_fraud_risk = bool(row["any_fraud"])
        home_region = row["region"] if row["region"] in REGION_CITY_MAP else random.choice(list(REGION_CITY_MAP))

        # Establish this customer's "usual" devices (1-3) up front
        usual_devices = [make_device_id() for _ in range(random.randint(1, 3))]

        # Number of login events for this customer (most customers log in a
        # handful of times; a few are much more active)
        n_events = max(1, int(np.random.poisson(lam=2.2)))

        for _ in range(n_events):
            login_time = random_login_time()

            # --- Device selection -------------------------------------------------
            use_new_device = False
            if elevated_device_risk and row["kyc_date"] is not pd.NaT and pd.notna(row["kyc_date"]):
                # After a KYC change, elevated chance of a brand-new device
                if login_time >= row["kyc_date"].to_pydatetime() and random.random() < 0.55:
                    use_new_device = True
            elif random.random() < 0.06:
                # background rate: anyone might occasionally use a new device
                use_new_device = True

            if use_new_device:
                device_id = make_device_id()
                new_device_flag = 1
            else:
                device_id = random.choice(usual_devices)
                new_device_flag = 0

            # --- Location / network -------------------------------------------------
            unusual_location = elevated_fraud_risk and random.random() < 0.35
            if unusual_location:
                city = random.choice([c for cities in REGION_CITY_MAP.values() for c in cities
                                       if c not in REGION_CITY_MAP[home_region]])
                network_type = random.choices(NETWORK_TYPES, weights=[0.15, 0.25, 0.35, 0.25])[0]
            else:
                city = random.choice(REGION_CITY_MAP[home_region])
                network_type = random.choices(NETWORK_TYPES, weights=[0.55, 0.30, 0.10, 0.05])[0]

            # --- Authentication outcome (credential-stuffing-like pattern) ---------
            fail_prob = 0.28 if elevated_fraud_risk else 0.06
            failed_first = random.random() < fail_prob
            auth_result = "Failed then Success" if failed_first else "Success"
            failed_attempts = random.randint(1, 4) if failed_first else 0

            records.append({
                "log_id": f"AUTH-{log_counter:06d}",
                "Customer_ID": cid,
                "login_time": login_time.strftime("%Y-%m-%d %H:%M:%S"),
                "device_id": device_id,
                "new_device_flag": new_device_flag,
                "ip_address": make_ip(),
                "location_city": city,
                "location_unusual_flag": int(unusual_location),
                "network_type": network_type,
                "authentication_result": auth_result,
                "failed_attempts_before_success": failed_attempts,
                "off_hours_login_flag": int(login_time.hour >= 22 or login_time.hour < 6),
            })
            log_counter += 1

    return pd.DataFrame(records)


def main():
    claims = pd.read_csv(CLAIMS_IN)
    print(f"Loaded {len(claims)} claims covering {claims['Customer_ID'].nunique()} customers")

    auth = build_auth_data(claims)
    auth.to_csv(OUT_FILE, index=False)

    print(f"\nWrote {len(auth)} login events to {OUT_FILE}")
    print(f"Columns: {list(auth.columns)}")

    # --- Validation: confirm the intended correlations are actually present ---
    merged = auth.merge(
        claims.groupby("Customer_ID").agg(
            any_identity_change=("Identity_Changed_Flag", "max"),
            any_fraud=("Fraud_Flag", lambda s: (s == "Yes").any())
        ).reset_index(),
        on="Customer_ID", how="left"
    )
    print("\n--- Validation ---")
    print("New-device login rate:")
    print(merged.groupby("any_identity_change")["new_device_flag"].mean().round(3))
    print("\nUnusual-location login rate:")
    print(merged.groupby("any_fraud")["location_unusual_flag"].mean().round(3))
    print("\nFailed-then-success auth rate:")
    print(merged.groupby("any_fraud")["authentication_result"]
          .apply(lambda s: (s == "Failed then Success").mean()).round(3))


if __name__ == "__main__":
    main()
