#!/usr/bin/env python
# coding: utf-8

# # Notebook 03 — Visualizations
# **Project:** Revolut Support A/B Test
# **Author:** Deepak Gupta | Operations Data Analyst
#
# ## Charts Generated
# 1. Escalation Rate Bar Chart with 95% CI  — Primary metric comparison
# 2. Resolution Time Distribution           — Overlapping histograms (log-normal)
# 3. CSAT Score Distribution               — Grouped bar chart by score
# 4. Segment by Account Type               — Escalation rate breakdown
# 5. Daily Trend Line Chart                — 14-day stability check
# 6. Segment by Issue Category             — Escalation rate breakdown
#
# Design philosophy: Clean, minimal, data-forward aesthetic aligned with
# Revolut's product design language. No decorative noise.

# ─── Imports ───
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os

os.makedirs("visuals", exist_ok=True)

# ─── Load Data ───
df   = pd.read_csv("data/revolut_ab_test_dataset.csv")
ctrl = df[df["group"] == "control"]
var  = df[df["group"] == "variant"]

# ─────────────────────────────────────────────────────────────
# CHART 1: Escalation Rate (Primary Metric)
# WHY: This is the headline result. Error bars show 95% CI so readers
#      can immediately assess statistical confidence visually.
# ─────────────────────────────────────────────────────────────

means = [34.55, 29.56]
ci_lo = [34.17, 29.19]
ci_hi = [34.93, 29.92]

fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=["Control (A)", "Variant (B)"], y=means,
    error_y=dict(type="data", array=[m-l for m, l in zip(means, ci_lo)], visible=True),
    text=[f"{m:.2f}%<br>CI: [{l:.2f}%, {h:.2f}%]" for m, l, h in zip(means, ci_lo, ci_hi)],
    textposition="outside"
))
fig1.update_layout(title="Escalation Rate Drops -4.99pp in Variant | n=120k | p<0.001")
fig1.update_xaxes(title_text="Test Group")
fig1.update_yaxes(title_text="Escalation Rate %", dtick=5, range=[0, 39])
fig1.write_image("visuals/01_escalation_rate.png")
print("Chart 1 saved: visuals/01_escalation_rate.png")

# ─────────────────────────────────────────────────────────────
# CHART 2: Resolution Time Distribution
# WHY: Overlapping histogram reveals the leftward shift. The visible
#      right-skew justifies Mann-Whitney U over a t-test.
# ─────────────────────────────────────────────────────────────

fig2 = go.Figure()
fig2.add_trace(go.Histogram(
    x=ctrl["resolution_time_mins"].clip(0, 200), name="Control",
    opacity=0.6, nbinsx=60, histnorm="probability density"
))
fig2.add_trace(go.Histogram(
    x=var["resolution_time_mins"].clip(0, 200), name="Variant",
    opacity=0.6, nbinsx=60, histnorm="probability density"
))
fig2.update_layout(barmode="overlay", title="Variant Resolves Issues 8.8 Mins Faster (Median) | MWU p<0.001")
fig2.update_xaxes(title_text="Resolution Mins")
fig2.update_yaxes(title_text="Density")
fig2.write_image("visuals/02_resolution_time.png")
print("Chart 2 saved: visuals/02_resolution_time.png")

# ─────────────────────────────────────────────────────────────
# CHART 3: CSAT Score Distribution
# WHY: Score-by-score comparison shows where satisfaction improved.
#      Validates that faster resolution translates to happier users.
# ─────────────────────────────────────────────────────────────

csat_c = ctrl["csat_score"].value_counts(normalize=True).sort_index() * 100
csat_v = var["csat_score"].value_counts(normalize=True).sort_index() * 100

fig3 = go.Figure()
fig3.add_trace(go.Bar(name="Control", x=csat_c.index.astype(str), y=csat_c.values))
fig3.add_trace(go.Bar(name="Variant", x=csat_v.index.astype(str), y=csat_v.values))
fig3.update_layout(barmode="group", title="Variant Improves CSAT | Mean 3.20 -> 3.59 | MWU p<0.001")
fig3.update_xaxes(title_text="CSAT Score")
fig3.update_yaxes(title_text="Users %")
fig3.write_image("visuals/03_csat_distribution.png")
print("Chart 3 saved: visuals/03_csat_distribution.png")

# ─────────────────────────────────────────────────────────────
# CHART 4: Segment by Account Type
# WHY: Validates the variant works for all account types (no HTE risk).
#      Metal users benefit most — premium users engage with help articles.
# ─────────────────────────────────────────────────────────────

seg_df = pd.DataFrame({
    "Account Type": ["Metal", "Premium", "Standard"],
    "Control":      [32.47, 33.97, 35.09],
    "Variant":      [26.81, 29.59, 29.96],
})

fig4 = go.Figure()
fig4.add_trace(go.Bar(name="Control", x=seg_df["Account Type"], y=seg_df["Control"]))
fig4.add_trace(go.Bar(name="Variant", x=seg_df["Account Type"], y=seg_df["Variant"]))
fig4.update_layout(barmode="group", title="Variant Reduces Escalations Across All Account Types")
fig4.update_xaxes(title_text="Account Type")
fig4.update_yaxes(title_text="Escalation Rate %", dtick=5, range=[0, 40])
fig4.write_image("visuals/04_segment_account_type.png")
print("Chart 4 saved: visuals/04_segment_account_type.png")

# ─────────────────────────────────────────────────────────────
# CHART 5: Daily Escalation Trend
# WHY: Critical stability check. A flat gap = genuine behavioral change.
#      Converging lines would signal a novelty effect — not durable improvement.
# ─────────────────────────────────────────────────────────────

daily   = df.groupby(["experiment_day", "group"])["escalated_to_agent"].mean().reset_index()
daily["rate_pct"] = daily["escalated_to_agent"] * 100

fig5 = go.Figure()
fig5.add_trace(go.Scatter(
    x=daily[daily["group"]=="control"]["experiment_day"],
    y=daily[daily["group"]=="control"]["rate_pct"],
    mode="lines+markers", name="Control (A)", line=dict(width=2.5)
))
fig5.add_trace(go.Scatter(
    x=daily[daily["group"]=="variant"]["experiment_day"],
    y=daily[daily["group"]=="variant"]["rate_pct"],
    mode="lines+markers", name="Variant (B)", line=dict(width=2.5)
))
fig5.update_layout(title="Variant Consistently Lower — Stable Gap Over 14 Days")
fig5.update_xaxes(title_text="Experiment Day", dtick=1)
fig5.update_yaxes(title_text="Escalation Rate %", dtick=2, range=[26, 40])
fig5.write_image("visuals/05_daily_trend.png")
print("Chart 5 saved: visuals/05_daily_trend.png")

# ─────────────────────────────────────────────────────────────
# CHART 6: Segment by Issue Category
# WHY: Identifies which issue types benefit most from the redesign.
#      Transfer Failed shows largest lift — payment help articles are
#      most effective at enabling self-service.
# ─────────────────────────────────────────────────────────────

issue_df = pd.DataFrame({
    "Issue":   ["Account Locked", "Card Blocked", "FX Dispute", "KYC Pending", "Transfer Failed"],
    "Control": [35.05, 34.44, 34.49, 34.42, 34.63],
    "Variant": [30.07, 29.73, 29.57, 29.46, 29.16],
})

fig6 = go.Figure()
fig6.add_trace(go.Bar(name="Control", x=issue_df["Issue"], y=issue_df["Control"]))
fig6.add_trace(go.Bar(name="Variant", x=issue_df["Issue"], y=issue_df["Variant"]))
fig6.update_layout(barmode="group", title="Variant Reduces Escalations Across All Issue Types")
fig6.update_xaxes(title_text="Issue Category")
fig6.update_yaxes(title_text="Escalation Rate %", dtick=5, range=[0, 40])
fig6.write_image("visuals/06_segment_issue_category.png")
print("Chart 6 saved: visuals/06_segment_issue_category.png")

print("\n✅ All 6 charts generated in visuals/")
