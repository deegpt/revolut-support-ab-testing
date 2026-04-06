#!/usr/bin/env python
# coding: utf-8

# # Notebook 01 — Synthetic Dataset Generation
# **Project:** Revolut Support A/B Test — Triage Screen vs Chatbot-First Flow
# **Author:** Deepak Gupta | Operations Data Analyst
#
# ## Purpose
# This notebook generates a realistic 120,000-row synthetic dataset simulating a
# 14-day A/B experiment at Revolut. The data is designed to mirror real operational
# support analytics characteristics:
# - Log-normal resolution times (right-skewed, as real support data is)
# - Poisson-distributed message counts (discrete counts per session)
# - Stratified account type and region distributions matching Revolut's user base
# - Injected treatment effect: ~5 pp reduction in escalation rate for Variant B
#
# ## Why Synthetic?
# Portfolio projects cannot use real company data. Synthetic data generated with
# realistic statistical properties allows us to demonstrate the same analytical
# methods that would apply to real data, while maintaining ethical boundaries.

# ─── Imports ───
import numpy as np
import pandas as pd

# Set seed for full reproducibility
np.random.seed(42)

# ─── Configuration ───
N = 120_000
n_control = 60_000
n_variant  = 60_000

issue_categories = ['Card Blocked', 'FX Dispute', 'KYC Pending', 'Transfer Failed', 'Account Locked']
issue_weights    = [0.30, 0.20, 0.18, 0.22, 0.10]

account_types    = ['Standard', 'Premium', 'Metal']
account_weights  = [0.65, 0.25, 0.10]

regions          = ['UK', 'EU', 'APAC', 'IN']
region_weights   = [0.35, 0.30, 0.20, 0.15]

# ─── Generator Function ───
def generate_group(n, group_name, escalation_rate_base, resolution_time_mean, csat_mean):
    """
    Generates a synthetic user group for the A/B experiment.

    Parameters
    ----------
    n                     : int   — number of users
    group_name            : str   — 'control' or 'variant'
    escalation_rate_base  : float — baseline probability of escalating to live agent
    resolution_time_mean  : float — mean resolution time in minutes (log-normal)
    csat_mean             : float — mean CSAT score (normal, clipped to 1-5)

    Design decisions:
    - escalation_rate adjusted by account_type: Metal users (-3pp) are more self-sufficient
    - resolution_time drawn from log-normal to simulate realistic right-skewed distributions
    - csat_score drawn from normal then rounded to integer 1-5
    - messages_sent drawn from Poisson (discrete count data)
    - ticket_abandoned is a guardrail metric: variant has slight increase to flag risk
    - user_tenure_months uses exponential distribution (many new users, fewer long-tenured)
    """
    df = pd.DataFrame()
    df["user_id"]       = [f"USR-{group_name[:1].upper()}-{str(i).zfill(6)}" for i in range(n)]
    df["group"]         = group_name
    df["account_type"]  = np.random.choice(account_types, size=n, p=account_weights)
    df["issue_category"]= np.random.choice(issue_categories, size=n, p=issue_weights)
    df["region"]        = np.random.choice(regions, size=n, p=region_weights)
    df["experiment_day"]= np.random.randint(1, 15, size=n)

    # Account-type adjusted escalation probability
    escalation_adj = np.where(df["account_type"] == "Metal",   -0.03,
                     np.where(df["account_type"] == "Premium", -0.01, 0.0))
    esc_prob = np.clip(escalation_rate_base + escalation_adj, 0.05, 0.95)
    df["escalated_to_agent"] = np.array([np.random.binomial(1, p) for p in esc_prob])

    # Resolution time: log-normal, clipped to realistic range [2, 300] minutes
    df["resolution_time_mins"] = np.round(
        np.random.lognormal(mean=np.log(resolution_time_mean), sigma=0.6, size=n), 1
    ).clip(2, 300)

    # CSAT: normally distributed, rounded and clipped to ordinal 1-5 scale
    df["csat_score"] = np.clip(
        np.round(np.random.normal(csat_mean, 0.8, size=n)).astype(int), 1, 5
    )

    # Messages per session: Poisson (variant needs fewer turns due to triage screen)
    msg_mean = 4.5 if group_name == "control" else 3.8
    df["messages_sent"] = np.clip(
        np.round(np.random.poisson(msg_mean, size=n)).astype(int), 1, 20
    )

    # Ticket abandonment: guardrail metric — slight increase in variant
    abandon_rate = 0.07 if group_name == "control" else 0.075
    df["ticket_abandoned"] = np.random.binomial(1, abandon_rate, size=n)

    df["session_channel"]    = "Chatbot-First" if group_name == "control" else "Triage-Screen"
    df["user_tenure_months"] = np.clip(
        np.round(np.random.exponential(18, size=n)).astype(int), 1, 120
    )
    return df


# ─── Generate Groups ───
# Control: baseline escalation 35%, avg resolution 42 mins, avg CSAT 3.2
df_control = generate_group(n_control, "control",
                             escalation_rate_base=0.35,
                             resolution_time_mean=42,
                             csat_mean=3.2)

# Variant: lower escalation 30%, faster resolution 33 mins, higher CSAT 3.6
# Treatment effect injected: ~5 pp escalation reduction, 9 mins faster, +0.4 CSAT
df_variant = generate_group(n_variant, "variant",
                             escalation_rate_base=0.30,
                             resolution_time_mean=33,
                             csat_mean=3.6)

# ─── Combine and Shuffle ───
df = pd.concat([df_control, df_variant], ignore_index=True)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# ─── Save ───
import os
os.makedirs("data", exist_ok=True)
df.to_csv("data/revolut_ab_test_dataset.csv", index=False)

# ─── Summary ───
print("Dataset shape:", df.shape)
print("\nGroup distribution:")
print(df["group"].value_counts())
print("\nColumn dtypes:")
print(df.dtypes)
print("\nSample rows:")
print(df.head(3).to_string())
print("\n✅ Dataset saved to data/revolut_ab_test_dataset.csv")
