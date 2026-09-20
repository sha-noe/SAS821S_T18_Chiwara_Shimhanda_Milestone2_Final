# Incident Response Recommendations
## T18 - Insurance Claims Fraud, Identity Abuse and Data-Leakage Analytics

Generated from 12 high-risk cases where three independent signals
agreed: the claims-level Fraud_Flag, an identity/KYC change, and the
unsupervised anomaly model (which never saw the fraud label).

## Operational recommendations (for claims/fraud investigators)
1. **Flag any claim preceded by a KYC/identity change within 21 days** for
   secondary review before payout - this single condition accounted for the
   strongest signal in both the supervised and unsupervised models.
2. **Treat a new, unrecognised device login following an identity change as
   a hard trigger** for identity re-verification, not just a soft risk
   factor - 12 of 12 top-risk cases in this sample showed
   this exact sequence.
3. **Cross-reference the handling staff member's Insider_Risk_Label** before
   final approval on any claim already flagged by the above two rules.

## Executive-level recommendation (for management)
- Consider a policy control requiring mandatory secondary verification for
  any claim submitted within 21 days of an identity/KYC update - this is a
  low-cost control given how concentrated the fraud signal is around this
  window (see `08_outputs/eda_charts/03_identity_change_by_fraud.png`).

## Individual case files
See `08_outputs/incident_timelines/` for the full chronological timeline of
each of the 12 cases, including affected entities and a
per-case recommended action.
