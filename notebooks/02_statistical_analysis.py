#!/usr/bin/env python
# coding: utf-8

# # Notebook 02 — Full Statistical Analysis
# **Project:** Revolut Support A/B Test
# **Author:** Deepak Gupta | Operations Data Analyst
#
# ## Tests Performed
# 1. Two-Proportion Z-Test        — Primary metric (escalation rate)
# 2. 95% Confidence Intervals     — Primary metric difference and individual rates
# 3. Cohen's h Effect Size        — Standardised practical significance measure
# 4. Power Analysis               — Validates sample size adequacy
# 5. Mann-Whitney U Test          — Resolution time (non-normal distribution)
# 6. Mann-Whitney U Test          — CSAT score (ordinal, non-normal)
# 7. Guardrail Z-Test             — Ticket abandonment (risk check)
# 8. Segment Analysis             — Account type and issue category breakdowns
#
# ## Why Each Method Was Chosen
# See inline comments per section for full statistical rationale.

# ─── Imports ───
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.power import zt_ind_solve_power
import warnings
warnings.filterwarnings("ignore")

# ─── Load Data ───
df    = pd.read_csv("data/revolut_ab_test_dataset.csv")
ctrl  = df[df["group"] == "control"]
var   = df[df["group"] == "variant"]
n_c, n_v = len(ctrl), len(var)

# ═══════════════════════════════════════════════════════════════
# 1. TWO-PROPORTION Z-TEST — Primary Metric: Escalation Rate
# ═══════════════════════════════════════════════════════════════
#
# Why this test?
# The escalation metric is binary (0/1). Comparing two independent proportions
# at large sample sizes (n=60,000 each) is the textbook use case for the
# two-proportion z-test. The normal approximation to the binomial holds because
# both n*p and n*(1-p) >> 5 for all groups.
#
# H0: p_variant >= p_control (variant does NOT reduce escalations)
# H1: p_variant <  p_control (variant DOES reduce escalations)
# One-sided test (we only care about reduction, not increase)

esc_c = ctrl["escalated_to_agent"].sum()
esc_v = var["escalated_to_agent"].sum()
p_c   = esc_c / n_c
p_v   = esc_v / n_v
lift_pp  = p_v - p_c
lift_rel = lift_pp / p_c * 100

z_stat, p_val = proportions_ztest([esc_v, esc_c], [n_v, n_c], alternative="smaller")

print("=" * 60)
print("1. TWO-PROPORTION Z-TEST — Escalation Rate")
print(f"   Control  : {p_c:.4f} ({p_c*100:.2f}%)  n={n_c:,}")
print(f"   Variant  : {p_v:.4f} ({p_v*100:.2f}%)  n={n_v:,}")
print(f"   Absolute Lift: {lift_pp*100:.2f} pp  |  Relative: {lift_rel:.1f}%")
print(f"   Z-statistic : {z_stat:.4f}  |  p-value: {p_val:.8f}")
print(f"   Decision: {'REJECT H0' if p_val < 0.05 else 'FAIL TO REJECT H0'}")

# ═══════════════════════════════════════════════════════════════
# 2. CONFIDENCE INTERVALS (95%)
# ═══════════════════════════════════════════════════════════════
#
# A p-value alone tells us the probability of seeing this result under H0,
# but not the magnitude. The CI gives a range of plausible true values for
# the treatment effect, which is what business stakeholders need for decisions.

ci_lo_c, ci_hi_c = proportion_confint(esc_c, n_c, method="normal")
ci_lo_v, ci_hi_v = proportion_confint(esc_v, n_v, method="normal")
pooled_se        = ((p_c*(1-p_c)/n_c) + (p_v*(1-p_v)/n_v)) ** 0.5
diff_ci_lo       = lift_pp - 1.96 * pooled_se
diff_ci_hi       = lift_pp + 1.96 * pooled_se

print("\n2. CONFIDENCE INTERVALS (95%)")
print(f"   Control CI   : [{ci_lo_c*100:.2f}%, {ci_hi_c*100:.2f}%]")
print(f"   Variant CI   : [{ci_lo_v*100:.2f}%, {ci_hi_v*100:.2f}%]")
print(f"   Difference CI: [{diff_ci_lo*100:.2f}pp, {diff_ci_hi*100:.2f}pp]")

# ═══════════════════════════════════════════════════════════════
# 3. COHEN'S h — Effect Size
# ═══════════════════════════════════════════════════════════════
#
# With n=60,000 per group, even a trivial 0.1 pp difference would be
# statistically significant. Effect size standardises the difference
# so we can assess practical (business) significance independently of n.
# Cohen's h uses arcsine transformation to normalise proportion differences.
# Benchmarks: h=0.2 small, h=0.5 medium, h=0.8 large (Cohen 1988)

h = 2 * (np.arcsin(np.sqrt(p_v)) - np.arcsin(np.sqrt(p_c)))
effect_label = "small" if abs(h) < 0.2 else ("medium" if abs(h) < 0.5 else "large")

print(f"\n3. COHEN'S h EFFECT SIZE")
print(f"   h = {h:.4f} ({effect_label} effect)")
print(f"   Note: 'small' at Revolut's 50M+ user scale = millions of deflected tickets")

# ═══════════════════════════════════════════════════════════════
# 4. POWER ANALYSIS
# ═══════════════════════════════════════════════════════════════
#
# Power (1 - β) = probability of correctly detecting a true effect.
# Standard threshold: 80%. Reporting min required n shows design rigour.

achieved_power = zt_ind_solve_power(effect_size=abs(h), nobs1=n_c, alpha=0.05, ratio=1, alternative="two-sided")
required_n     = zt_ind_solve_power(effect_size=abs(h), power=0.80, alpha=0.05, ratio=1, alternative="two-sided")

print(f"\n4. POWER ANALYSIS")
print(f"   Achieved power          : {achieved_power:.4f} ({achieved_power*100:.1f}%)")
print(f"   Min N/group for 80% pwr : {required_n:.0f}")
print(f"   Our N/group             : {n_c:,} — significantly overpowered")

# ═══════════════════════════════════════════════════════════════
# 5. MANN-WHITNEY U — Resolution Time
# ═══════════════════════════════════════════════════════════════
#
# Resolution time is log-normally distributed (long right tail).
# t-test assumes normality — violated here. Mann-Whitney U is a
# non-parametric rank-based test making no distributional assumptions.
# H1: variant resolution times < control (one-sided)

u_stat, mwu_p = mannwhitneyu(var["resolution_time_mins"], ctrl["resolution_time_mins"], alternative="less")
med_c = ctrl["resolution_time_mins"].median()
med_v = var["resolution_time_mins"].median()

print(f"\n5. MANN-WHITNEY U — Resolution Time")
print(f"   Median Control : {med_c:.1f} mins  |  Median Variant: {med_v:.1f} mins")
print(f"   Delta Median   : {med_v - med_c:.1f} mins ({(med_v-med_c)/med_c*100:.1f}%)")
print(f"   U-stat: {u_stat:,.0f}  |  p-value: {mwu_p:.8f}")

# ═══════════════════════════════════════════════════════════════
# 6. MANN-WHITNEY U — CSAT Score
# ═══════════════════════════════════════════════════════════════
#
# CSAT is ordinal (1-5). The difference between 3→4 is not necessarily
# equal to 4→5. Mann-Whitney U works on ranks, appropriate for ordinal data.
# H1: variant CSAT > control (one-sided)

u_csat, mwu_csat_p = mannwhitneyu(var["csat_score"], ctrl["csat_score"], alternative="greater")

print(f"\n6. MANN-WHITNEY U — CSAT Score")
print(f"   Mean Control : {ctrl['csat_score'].mean():.3f}  |  Mean Variant: {var['csat_score'].mean():.3f}")
print(f"   U-stat: {u_csat:,.0f}  |  p-value: {mwu_csat_p:.8f}")

# ═══════════════════════════════════════════════════════════════
# 7. GUARDRAIL — Ticket Abandonment Rate
# ═══════════════════════════════════════════════════════════════
#
# If users are simply abandoning tickets instead of escalating, the primary
# metric improvement is a false signal. This guardrail test explicitly checks
# whether triage screen friction drove users to give up. Any significant
# increase is an alert that must be investigated before full rollout.

ab_c   = ctrl["ticket_abandoned"].sum()
ab_v   = var["ticket_abandoned"].sum()
p_ab_c = ab_c / n_c
p_ab_v = ab_v / n_v
z_ab, p_ab = proportions_ztest([ab_v, ab_c], [n_v, n_c], alternative="larger")

print(f"\n7. GUARDRAIL — Ticket Abandonment")
print(f"   Control: {p_ab_c*100:.2f}%  |  Variant: {p_ab_v*100:.2f}%")
print(f"   p-value: {p_ab:.4f}  |  Status: {'ALERT' if p_ab < 0.05 else 'Safe'}")

# ═══════════════════════════════════════════════════════════════
# 8. SEGMENT ANALYSIS
# ═══════════════════════════════════════════════════════════════
#
# Aggregate results can mask important subgroup effects (heterogeneous
# treatment effects). Segment analysis detects whether the variant
# works for all user types or only specific cohorts.

print("\n8. SEGMENT ANALYSIS")
print("\n   By Account Type:")
seg_a = df.groupby(["group", "account_type"])["escalated_to_agent"].mean().unstack("group") * 100
seg_a["lift_pp"] = seg_a["variant"] - seg_a["control"]
print(seg_a.round(2).to_string())

print("\n   By Issue Category:")
seg_i = df.groupby(["group", "issue_category"])["escalated_to_agent"].mean().unstack("group") * 100
seg_i["lift_pp"] = seg_i["variant"] - seg_i["control"]
print(seg_i.round(2).to_string())

print("\n" + "=" * 60)
print("FINAL DECISION")
print("  PRIMARY : SHIP     — Escalations reduced significantly")
print("  SECONDARY: SHIP    — Resolution faster, CSAT higher")
print("  GUARDRAIL: MONITOR — Abandonment slightly elevated")
print("  RECOMMENDATION: Phased rollout with abandonment follow-up prompt")
