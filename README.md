# 🏦 Revolut Support A/B Testing — Operations Analytics Portfolio Project

> **Role:** Junior / Operations Data Analyst @ Revolut Support Analytics Team  
> **Experiment:** Chatbot-First Flow (Control A) vs. Redesigned Triage Screen (Variant B)  
> **Dataset:** 120,000 synthetic users | 60k control / 60k variant | 14-day experiment  
> **Stack:** Python · Pandas · SciPy · Statsmodels · Plotly

---

## 📌 Project Overview

This project simulates a real-world A/B test that an Operations Data Analyst at Revolut would be asked to run after identifying a decline in First-Contact Resolution (FCR) rates through SLA monitoring dashboards.

The investigation found that users receiving an **automated chatbot response first** were escalating to live agents at a 30–35% higher rate than expected. The Product team proposed a fix — a **redesigned triage screen** showing contextual help articles before offering chat. This A/B test answers: *does the new flow reduce escalations without hurting user satisfaction or increasing abandonment?*

---

## 📂 Repository Structure

```
revolut-support-ab-testing/
│
├── README.md                          ← This file
│
├── data/
│   └── revolut_ab_test_dataset.csv    ← 120k row synthetic dataset (13 columns)
│
├── notebooks/
│   ├── 01_dataset_generation.py       ← Synthetic data generation logic
│   ├── 02_statistical_analysis.py     ← Full statistical test suite
│   └── 03_visualizations.py           ← All Plotly chart code
│
└── visuals/
    ├── 01_escalation_rate.png         ← Primary metric: escalation rate bar chart
    ├── 02_resolution_time.png         ← Resolution time distribution
    ├── 03_csat_distribution.png       ← CSAT score distribution
    ├── 04_segment_account_type.png    ← Escalation by account type
    ├── 05_daily_trend.png             ← Daily escalation trend (14 days)
    └── 06_segment_issue_category.png  ← Escalation by issue category
```

---

## 🧪 Experiment Design

| Attribute | Detail |
|-----------|--------|
| **Hypothesis** | Users shown the redesigned triage screen (Variant B) will have a lower live-agent escalation rate and faster resolution compared to the chatbot-first flow (Control A) |
| **Primary Metric** | Live-agent escalation rate (binary: escalated vs. not) |
| **Secondary Metrics** | Resolution time (minutes), CSAT score (1–5) |
| **Guardrail Metric** | Ticket abandonment rate |
| **Experiment Duration** | 14 days |
| **Sample Size** | 60,000 users per group (120,000 total) |
| **Significance Level** | α = 0.05 (one-sided for primary, two-sided for guardrail) |

---

## 📊 Statistical Results Summary

### Primary Metric — Escalation Rate

| Group | Escalation Rate | 95% Confidence Interval |
|-------|----------------|------------------------|
| Control (A) — Chatbot-First | 34.55% | [34.17%, 34.93%] |
| Variant (B) — Triage Screen | 29.56% | [29.19%, 29.92%] |
| **Absolute Lift** | **−4.99 pp** | **[−5.52 pp, −4.47 pp]** |
| Relative Lift | −14.5% | — |
| Z-statistic | −18.53 | — |
| p-value | < 0.000001 | ✅ Statistically significant |
| Cohen's h | −0.107 | Small effect (meaningful at scale) |

### Secondary Metrics

| Metric | Control | Variant | Δ | Test | p-value |
|--------|---------|---------|---|------|---------|
| Resolution Time (median) | 42.0 mins | 33.2 mins | −8.8 mins | Mann-Whitney U | < 0.001 |
| CSAT Score (mean) | 3.20 | 3.59 | +0.39 | Mann-Whitney U | < 0.001 |

### Guardrail Metric

| Metric | Control | Variant | p-value | Status |
|--------|---------|---------|---------|--------|
| Ticket Abandonment | 7.01% | 7.31% | 0.023 | ⚠️ Monitor |

### Power Analysis

| Parameter | Value |
|-----------|-------|
| Effect size (Cohen's h) | 0.107 |
| Achieved power | 100% |
| Minimum N/group for 80% power | 1,369 |

---

## 📈 Visualizations

### 1. Escalation Rate — Primary Metric
![Escalation Rate](visuals/01_escalation_rate.png)

### 2. Resolution Time Distribution
![Resolution Time](visuals/02_resolution_time.png)

### 3. CSAT Score Distribution
![CSAT Distribution](visuals/03_csat_distribution.png)

### 4. Escalation Rate by Account Type
![Account Segment](visuals/04_segment_account_type.png)

### 5. Daily Escalation Rate Trend (14 Days)
![Daily Trend](visuals/05_daily_trend.png)

### 6. Escalation Rate by Issue Category
![Issue Segment](visuals/06_segment_issue_category.png)

---

## 🔍 Key Insights

1. **Ship Variant B** — the redesigned triage screen produces a statistically significant, practically meaningful 4.99 pp reduction in live-agent escalations
2. **Fastest resolution for Transfer Failed** — the issue category with the highest baseline escalation shows the strongest improvement (−5.47 pp), suggesting triage screen help articles are most effective for payment-related issues
3. **Metal users benefit most** — premium users engage more carefully with contextual help articles, achieving the largest relative reduction (−5.66 pp)
4. **⚠️ Abandonment flag requires investigation** — ticket abandonment increased slightly from 7.01% to 7.31% (p = 0.023). Before full rollout, Product should investigate whether these abandoned users have unresolved issues or genuinely self-served
5. **Effect size is intentionally small but valuable** — Cohen's h = 0.107 at Revolut's scale of 50M+ users represents massive cost savings and agent capacity freed for complex cases

---

## 💡 Methods & Rationale

### Two-Proportion Z-Test (Primary Metric)
Used for comparing two binary conversion rates (escalated vs. not). Appropriate because n > 30 per group, outcomes are independent Bernoulli trials, and the Central Limit Theorem ensures normal approximation of proportions at n=60,000.

### 95% Confidence Intervals
Provide a range of plausible true differences rather than a single point estimate, enabling business decision-making even under uncertainty.

### Cohen's h (Effect Size)
Standardised measure of difference between two proportions that does not depend on sample size. Prevents the mistake of treating a statistically significant result as practically significant when the sample is very large.

### Power Analysis
Validates that the sample size is sufficient to detect the observed effect. An achieved power of 100% means the experiment design is sound; reporting the minimum required N (1,369) demonstrates analytical rigour.

### Mann-Whitney U Test (Secondary Metrics)
Used for resolution time and CSAT score because both distributions are non-normal (resolution time is log-normally distributed; CSAT is ordinal 1–5). Mann-Whitney is a non-parametric rank-based test that makes no distributional assumptions.

---

## 🏗️ Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| `user_id` | String | Unique user identifier |
| `group` | String | `control` or `variant` |
| `account_type` | Categorical | Standard (65%), Premium (25%), Metal (10%) |
| `issue_category` | Categorical | Card Blocked, FX Dispute, KYC Pending, Transfer Failed, Account Locked |
| `region` | Categorical | UK (35%), EU (30%), APAC (20%), IN (15%) |
| `experiment_day` | Integer | Day 1–14 of experiment |
| `escalated_to_agent` | Binary | **Primary metric** — 1 = escalated to live agent |
| `resolution_time_mins` | Float | Log-normal distributed; clipped [2, 300] mins |
| `csat_score` | Integer (1–5) | Post-resolution satisfaction score |
| `messages_sent` | Integer | Chat turns per session (Poisson distributed) |
| `ticket_abandoned` | Binary | **Guardrail** — 1 = user dropped off without resolution |
| `session_channel` | String | `Chatbot-First` vs `Triage-Screen` |
| `user_tenure_months` | Integer | User tenure, exponential distribution (mean 18 months) |

---

## 🚀 How to Run

```bash
# Clone the repo
git clone https://github.com/deegpt/revolut-support-ab-testing.git
cd revolut-support-ab-testing

# Install dependencies
pip install pandas numpy scipy statsmodels plotly

# Run in order
python notebooks/01_dataset_generation.py
python notebooks/02_statistical_analysis.py
python notebooks/03_visualizations.py
```

---

## 👤 Author

**Deepak Gupta** · Data Analyst @ Revolut  
[GitHub](https://github.com/deegpt) · [Portfolio](https://deegpt.github.io/deegpt2.github.io/)

---

*This project uses entirely synthetic data generated for portfolio and educational purposes. No real Revolut user data is used.*
