"""
Method Lab — Methodology Short Course Generator
==================================================
Methodological Forensics Lab Feature 2: "The Methodology Lab"

When Winnie identifies an estimator in a paper (say, RDD), she doesn't
just say "they used a Regression Discontinuity Design." She opens a
Methodology Sandbox with:

1. A 3-minute crash course explaining the estimator in plain English
2. A toy dataset so you can VISUALIZE the method in action
3. The specific assumptions and when they break
4. A "How would YOU use this?" prompt for your dissertation

The goal: turn every complex methodology into a clear, interactive
short course tailored to political science research.

Architecture:
    Paper Audit → Estimator ID → Method Lab → Crash Course + Toy Data
    → Interactive Sandbox → "Apply to Your Research" Prompt
"""

import json
import logging
import math
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.method_lab")


# ═══════════════════════════════════════════════════════════════════
# Crash Course Templates — One per Estimator Family
# ═══════════════════════════════════════════════════════════════════

CRASH_COURSES = {
    "rdd": {
        "title": "Regression Discontinuity Design (RDD)",
        "tagline": "Nature's Randomized Experiment",
        "duration_minutes": 3,
        "plain_english": (
            "Imagine a scholarship that goes to students who score above 80 on a test. "
            "Students who scored 79 and students who scored 81 are basically identical — "
            "the only difference is the scholarship. RDD exploits this 'knife-edge' to "
            "estimate the causal effect of the scholarship.\n\n"
            "The key insight: RIGHT AT THE THRESHOLD, assignment is as-if random. "
            "A student scoring 80.01 vs 79.99 didn't 'earn' the treatment — they "
            "were effectively randomly assigned to it."
        ),
        "political_science_example": (
            "In political science, RDD is hugely powerful for elections. An incumbent "
            "who wins by 0.5% vs loses by 0.5% — the 'treatment' of winning is "
            "essentially random at the threshold. This lets you estimate the causal "
            "effect of incumbency.\n\n"
            "Mettler (2011) style: Cities that BARELY lost their application for "
            "federal aid vs cities that BARELY won. At the cutoff, it's a natural experiment."
        ),
        "assumptions": [
            {
                "name": "Continuity",
                "plain": "Everything besides the treatment is smooth at the cutoff — no 'jump' in other factors.",
                "violation": "If students are coached to score exactly 80, the design breaks.",
            },
            {
                "name": "No Manipulation",
                "plain": "Units can't precisely control whether they're above or below the threshold.",
                "violation": "If politicians can stuff ballots to win by exactly 0.1%, the RDD is invalid.",
            },
            {
                "name": "Local Randomization",
                "plain": "The causal effect is only estimated RIGHT AT the cutoff — it may not generalize.",
                "violation": "The effect of barely winning an election may differ from winning by 20%.",
            },
        ],
        "diagnostic_checklist": [
            "Plot the running variable — is there bunching at the cutoff?",
            "McCrary density test — formal test for manipulation",
            "Check that covariates are balanced at the threshold",
            "Vary the bandwidth — are results robust?",
            "Show the 'jump' visually — does it look real?",
        ],
        "toy_dataset": {
            "description": "Simulated election RDD: effect of incumbent party winning",
            "n": 200,
            "running_variable": "vote_margin",
            "cutoff": 0,
            "treatment": "incumbent_wins",
            "outcome": "next_election_spending",
            "true_effect": 15000,
        },
        "stata_code": (
            "* === RDD: Effect of Incumbency on Campaign Spending ===\n"
            "* Step 1: Load and visualize\n"
            "scatter next_election_spending vote_margin, ///\n"
            "    xline(0) title(\"RD Plot: Incumbency Effect\")\n\n"
            "* Step 2: Estimate the jump at the cutoff\n"
            "rdrobust next_election_spending vote_margin, c(0)\n\n"
            "* Step 3: Check for manipulation\n"
            "rddensity vote_margin, c(0)\n\n"
            "* Step 4: Bandwidth sensitivity\n"
            "rdrobust next_election_spending vote_margin, c(0) h(5)\n"
            "rdrobust next_election_spending vote_margin, c(0) h(10)\n"
            "rdrobust next_election_spending vote_margin, c(0) h(20)"
        ),
        "r_code": (
            '# === RDD: Effect of Incumbency ===\n'
            'library(rdrobust)\n'
            'library(rddensity)\n\n'
            '# Step 1: Visualize\n'
            'rdplot(y = df$next_election_spending, x = df$vote_margin, c = 0,\n'
            '       title = "RD Plot: Incumbency Effect")\n\n'
            '# Step 2: Estimate\n'
            'rd_est <- rdrobust(y = df$next_election_spending, x = df$vote_margin, c = 0)\n'
            'summary(rd_est)\n\n'
            '# Step 3: Manipulation test\n'
            'mc_test <- rddensity(X = df$vote_margin, c = 0)\n'
            'summary(mc_test)'
        ),
    },

    "did": {
        "title": "Difference-in-Differences (DiD)",
        "tagline": "The Before-After, Treatment-Control Design",
        "duration_minutes": 3,
        "plain_english": (
            "Imagine two cities: one gets a new welfare policy, one doesn't. "
            "You measure poverty in BOTH cities BEFORE and AFTER the policy.\n\n"
            "The 'first difference' removes time-invariant factors. "
            "The 'second difference' removes common time trends. "
            "What's left? The causal effect of the policy.\n\n"
            "It's elegant because you don't need randomization — you just need "
            "the treatment and control groups to have been trending the same way."
        ),
        "political_science_example": (
            "States that expanded Medicaid vs states that didn't. "
            "Before the ACA, both groups had similar uninsurance trends. "
            "After expansion, the gap tells you: did Medicaid expansion reduce uninsurance?\n\n"
            "For your Potter County work: counties that received FEMA aid vs similar "
            "counties that didn't. Were they trending similarly before the disaster?"
        ),
        "assumptions": [
            {
                "name": "Parallel Trends",
                "plain": "Treatment and control groups would have followed the same trend without treatment.",
                "violation": "If treated cities were already declining faster, DiD overestimates the effect.",
            },
            {
                "name": "No Anticipation",
                "plain": "Units don't change behavior before treatment actually starts.",
                "violation": "If companies start laying off workers before a minimum wage law takes effect.",
            },
            {
                "name": "Stable Treatment",
                "plain": "The treatment doesn't spill over to control units.",
                "violation": "If untreated counties absorb refugees from treated counties.",
            },
        ],
        "diagnostic_checklist": [
            "Plot pre-treatment trends — are they parallel?",
            "Run an event study — test for pre-trends formally",
            "Check for differential pre-treatment shocks",
            "Consider staggered adoption (Callaway & Sant'Anna 2021)",
            "Placebo test: fake the treatment date and re-estimate",
        ],
        "toy_dataset": {
            "description": "Simulated DiD: effect of welfare reform on poverty rate",
            "n": 400,
            "treatment_group": "reform_states",
            "control_group": "non_reform_states",
            "pre_periods": 5,
            "post_periods": 5,
            "outcome": "poverty_rate",
            "true_effect": -2.5,
        },
        "stata_code": (
            "* === DiD: Effect of Welfare Reform on Poverty ===\n"
            "* Step 1: Visualize parallel trends\n"
            "twoway (connected poverty_rate year if treated==1) ///\n"
            "       (connected poverty_rate year if treated==0), ///\n"
            "       xline(2020) legend(order(1 \"Treated\" 2 \"Control\"))\n\n"
            "* Step 2: Basic DiD\n"
            "reg poverty_rate treated##post i.year i.state, cluster(state)\n\n"
            "* Step 3: Event study\n"
            "eventdd poverty_rate i.state i.year, ///\n"
            "    timevar(event_time) treated(treated) ci(rcap)"
        ),
        "r_code": (
            '# === DiD: Welfare Reform Effect ===\n'
            'library(fixest)\n'
            'library(ggplot2)\n\n'
            '# Step 1: Basic DiD\n'
            'did_est <- feols(poverty_rate ~ treated:post | state + year,\n'
            '                 data = df, cluster = ~state)\n'
            'summary(did_est)\n\n'
            '# Step 2: Event study\n'
            'es_est <- feols(poverty_rate ~ i(event_time, treated, ref = -1) |\n'
            '                state + year, data = df, cluster = ~state)\n'
            'iplot(es_est, main = "Event Study: Welfare Reform")'
        ),
    },

    "iv": {
        "title": "Instrumental Variables (IV / 2SLS)",
        "tagline": "When X Causes Y But Y Also Causes X",
        "duration_minutes": 4,
        "plain_english": (
            "You want to know: does education cause higher earnings? "
            "But smarter people get more education AND earn more — "
            "so the correlation is confounded.\n\n"
            "The trick: find something (an 'instrument') that affects education "
            "but ONLY affects earnings THROUGH education. "
            "Example: being born near a college (Angrist & Krueger). "
            "Geography affects where you go to school but doesn't directly "
            "affect your salary.\n\n"
            "Two stages: (1) Predict education using the instrument. "
            "(2) Use the PREDICTED education to estimate the earnings effect."
        ),
        "political_science_example": (
            "Does democracy cause economic growth? Reverse causality is rampant. "
            "Acemoglu et al. use colonial-era mortality as an instrument: "
            "where colonizers died a lot, they built extractive institutions "
            "(which became autocracies). Mortality doesn't directly affect "
            "modern GDP — only through institutions.\n\n"
            "For your work: does charity presence reduce state capacity? "
            "You'd need an instrument that affects charity location but not "
            "state capacity directly. Maybe: distance from a major foundation HQ?"
        ),
        "assumptions": [
            {
                "name": "Relevance (First Stage)",
                "plain": "The instrument must actually affect the endogenous variable. F > 10 is the rule of thumb.",
                "violation": "A 'weak instrument' — the instrument barely predicts X. Results become unreliable.",
            },
            {
                "name": "Exclusion Restriction",
                "plain": "The instrument affects Y ONLY through X. This is untestable and requires theory.",
                "violation": "If distance from college affects earnings through other channels (e.g., local labor markets).",
            },
            {
                "name": "Independence",
                "plain": "The instrument is as-if randomly assigned — not correlated with unobservables.",
                "violation": "If colonial mortality is correlated with geography that affects modern GDP directly.",
            },
        ],
        "diagnostic_checklist": [
            "First-stage F-statistic > 10 (Stock & Yogo weak instrument test)",
            "Check instrument relevance: does Z predict X?",
            "Over-identification test (Sargan/Hansen) if multiple instruments",
            "Argue the exclusion restriction theoretically — this is the key",
            "Compare OLS to IV — if IV is much larger, think about LATE interpretation",
        ],
        "toy_dataset": {
            "description": "Simulated IV: effect of NGO presence on state service delivery",
            "n": 300,
            "endogenous": "ngo_count",
            "instrument": "distance_to_foundation_hq",
            "outcome": "state_service_index",
            "true_effect": -0.3,
        },
        "stata_code": (
            "* === IV: Effect of NGO Presence on State Services ===\n"
            "* Step 1: OLS (biased)\n"
            "reg state_service_index ngo_count controls, robust\n\n"
            "* Step 2: First stage\n"
            "reg ngo_count distance_to_foundation controls, robust\n"
            "test distance_to_foundation  // F > 10?\n\n"
            "* Step 3: 2SLS\n"
            "ivregress 2sls state_service_index controls ///\n"
            "    (ngo_count = distance_to_foundation), robust first\n\n"
            "* Step 4: Weak instrument test\n"
            "estat firststage"
        ),
        "r_code": (
            '# === IV: NGO Effect on State Services ===\n'
            'library(ivreg)\n'
            'library(fixest)\n\n'
            '# Step 1: OLS\n'
            'ols <- lm(state_service_index ~ ngo_count + controls, data = df)\n\n'
            '# Step 2: 2SLS\n'
            'iv <- ivreg(state_service_index ~ ngo_count + controls |\n'
            '            distance_to_foundation + controls, data = df)\n'
            'summary(iv, diagnostics = TRUE)\n\n'
            '# Step 3: Compare\n'
            'stargazer(ols, iv, type = "text")'
        ),
    },

    "fixed_effects": {
        "title": "Fixed Effects (FE) Panel Estimator",
        "tagline": "Control for Everything You Can't Observe (That Doesn't Change)",
        "duration_minutes": 3,
        "plain_english": (
            "You're studying 50 states over 20 years. Each state has unique "
            "characteristics (culture, geography, history) that are hard to measure. "
            "Fixed Effects is like giving each state its own intercept — it 'absorbs' "
            "everything unique about that state.\n\n"
            "The beauty: you don't need to MEASURE the unobservables. "
            "The math removes them. You're comparing each state to ITSELF over time."
        ),
        "political_science_example": (
            "Does voter ID legislation reduce turnout? States have different "
            "political cultures that affect both ID laws and turnout. "
            "State fixed effects remove these time-invariant differences. "
            "Year fixed effects remove national trends (e.g., Obama effect).\n\n"
            "Two-Way FE (state + year) isolates the WITHIN-state, WITHIN-year variation."
        ),
        "assumptions": [
            {
                "name": "Strict Exogeneity",
                "plain": "Past outcomes don't affect current treatment. Future treatment doesn't affect current outcomes.",
                "violation": "If states adopt voter ID laws BECAUSE turnout is already declining.",
            },
            {
                "name": "Time-Invariant Unobservables",
                "plain": "The unobserved factors you're controlling for don't change over time.",
                "violation": "If state 'political culture' is shifting (e.g., Texas turning purple).",
            },
        ],
        "diagnostic_checklist": [
            "Hausman test: FE vs RE — is FE appropriate?",
            "Check within-variation — is there enough?",
            "Test for serial correlation in residuals",
            "Consider clustered standard errors",
            "Check sensitivity to adding time-varying controls",
        ],
        "toy_dataset": {
            "description": "Simulated panel: state voter ID laws and turnout",
            "n": 1000,
            "panels": 50,
            "periods": 20,
            "outcome": "voter_turnout",
            "treatment": "voter_id_law",
            "true_effect": -1.8,
        },
        "stata_code": (
            "* === Fixed Effects: Voter ID and Turnout ===\n"
            "xtset state_id year\n\n"
            "* Step 1: Pooled OLS (biased)\n"
            "reg voter_turnout voter_id_law controls, cluster(state_id)\n\n"
            "* Step 2: State FE\n"
            "xtreg voter_turnout voter_id_law controls, fe cluster(state_id)\n\n"
            "* Step 3: Two-Way FE\n"
            "reghdfe voter_turnout voter_id_law controls, ///\n"
            "    absorb(state_id year) cluster(state_id)\n\n"
            "* Step 4: Hausman test\n"
            "hausman fixed random"
        ),
        "r_code": (
            '# === Fixed Effects: Voter ID and Turnout ===\n'
            'library(fixest)\n'
            'library(plm)\n\n'
            '# Two-Way FE\n'
            'fe_est <- feols(voter_turnout ~ voter_id_law + controls |\n'
            '                state_id + year, data = panel, cluster = ~state_id)\n'
            'summary(fe_est)\n\n'
            '# Compare to Pooled OLS\n'
            'ols_est <- feols(voter_turnout ~ voter_id_law + controls, data = panel)\n'
            'etable(ols_est, fe_est)'
        ),
    },

    "propensity_score": {
        "title": "Propensity Score Matching (PSM)",
        "tagline": "Build Your Own Control Group",
        "duration_minutes": 3,
        "plain_english": (
            "You want to compare NGO-recipient communities to non-recipient communities. "
            "But NGOs don't go to random places — they pick needy areas. "
            "So you can't just compare recipients to non-recipients.\n\n"
            "PSM solution: estimate the PROBABILITY of receiving NGO aid based "
            "on observables (poverty, population, politics). Then match each "
            "treated community to an untreated one with a SIMILAR probability. "
            "Now you have an 'as-if random' comparison group."
        ),
        "political_science_example": (
            "Does receiving foreign aid reduce state investment in public goods? "
            "Countries that receive aid are already different from those that don't. "
            "PSM creates a matched sample where aided and non-aided countries "
            "look statistically identical on observables.\n\n"
            "For your work: match Potter County to 'twin' counties with similar "
            "demographics but different charity penetration levels."
        ),
        "assumptions": [
            {
                "name": "Conditional Independence (CIA)",
                "plain": "After matching on observables, treatment assignment is independent of outcomes.",
                "violation": "If there's an unobserved factor (like 'community motivation') that affects both treatment and outcome.",
            },
            {
                "name": "Common Support / Overlap",
                "plain": "For every treated unit, there exists a comparable untreated unit.",
                "violation": "If the most impoverished communities ALL receive aid, there's no untreated match.",
            },
        ],
        "diagnostic_checklist": [
            "Check covariate balance AFTER matching (standardized mean differences < 0.1)",
            "Verify common support — plot propensity score distributions",
            "Sensitivity analysis: Rosenbaum bounds for hidden bias",
            "Try multiple matching algorithms (nearest neighbor, caliper, kernel)",
            "Compare results across specifications",
        ],
        "toy_dataset": {
            "description": "Simulated PSM: effect of NGO aid on state investment",
            "n": 500,
            "treatment": "ngo_aid_received",
            "outcome": "state_public_goods_spending",
            "confounders": ["poverty_rate", "population", "urban_pct"],
            "true_effect": -5000,
        },
        "stata_code": (
            "* === PSM: NGO Aid and State Investment ===\n"
            "* Step 1: Estimate propensity score\n"
            "logit ngo_aid poverty_rate population urban_pct\n"
            "predict pscore, pr\n\n"
            "* Step 2: Match\n"
            "psmatch2 ngo_aid, pscore(pscore) outcome(state_spending) ///\n"
            "    neighbor(1) caliper(0.05)\n\n"
            "* Step 3: Check balance\n"
            "pstest poverty_rate population urban_pct, both\n\n"
            "* Step 4: Sensitivity\n"
            "rbounds state_spending, gamma(1 1.5 2)"
        ),
        "r_code": (
            '# === PSM: NGO Aid and State Investment ===\n'
            'library(MatchIt)\n'
            'library(cobalt)\n\n'
            '# Step 1: Match\n'
            'm.out <- matchit(ngo_aid ~ poverty_rate + population + urban_pct,\n'
            '                 data = df, method = "nearest", caliper = 0.05)\n\n'
            '# Step 2: Balance check\n'
            'love.plot(m.out, threshold = 0.1)\n\n'
            '# Step 3: Estimate on matched sample\n'
            'm.data <- match.data(m.out)\n'
            'lm(state_spending ~ ngo_aid, data = m.data, weights = weights)'
        ),
    },

    "synthetic_control": {
        "title": "Synthetic Control Method (SCM)",
        "tagline": "Build a Frankenstein County to Compare Against",
        "duration_minutes": 3,
        "plain_english": (
            "You want to know: what if Texas HAD expanded Medicaid? "
            "But there's only one Texas, and you can't run the counterfactual.\n\n"
            "SCM solution: build a 'Synthetic Texas' from a weighted combination "
            "of other states (maybe 30% Ohio + 25% Georgia + ...) that perfectly "
            "matches Texas's pre-treatment trends. Then compare real Texas to "
            "Synthetic Texas after the policy."
        ),
        "political_science_example": (
            "What was the effect of California's tobacco tax? Build 'Synthetic California' "
            "from other states. The gap between real and synthetic California = the effect.\n\n"
            "For Potter County: build a 'Synthetic Potter County' from similar Texas "
            "counties to see what WOULD have happened without the charity intervention."
        ),
        "assumptions": [
            {
                "name": "Convex Hull",
                "plain": "The treated unit must be within the range of potential donors — no extrapolation.",
                "violation": "If Potter County is uniquely extreme on some dimension, no combination of other counties can match it.",
            },
            {
                "name": "Pre-Treatment Fit",
                "plain": "The synthetic control must closely match the treated unit before treatment.",
                "violation": "Poor pre-treatment fit → unreliable post-treatment comparison.",
            },
            {
                "name": "No Interference",
                "plain": "Donor pool units are not affected by the treatment.",
                "violation": "If Potter County's charity funds come from neighboring counties' budgets.",
            },
        ],
        "diagnostic_checklist": [
            "Pre-treatment RMSPE — how well does synthetic match real?",
            "Placebo tests: run SCM on every donor unit",
            "Leave-one-out: drop each donor and re-estimate",
            "Plot actual vs synthetic over time",
            "Ratio of post/pre RMSPE for significance",
        ],
        "toy_dataset": {
            "description": "Simulated SCM: effect of charity intervention in Potter County",
            "n": 20,
            "treated_unit": "Potter_County",
            "donor_pool": 19,
            "pre_periods": 10,
            "post_periods": 5,
            "outcome": "state_service_quality_index",
            "true_effect": -3.5,
        },
        "stata_code": (
            "* === SCM: Charity Intervention in Potter County ===\n"
            "* Using synth package\n"
            "synth state_service_quality poverty_rate population urban_pct, ///\n"
            "    trunit(potter_id) trperiod(2015) fig"
        ),
        "r_code": (
            '# === SCM: Potter County Charity Effect ===\n'
            'library(Synth)\n\n'
            '# Prepare data\n'
            'dataprep.out <- dataprep(\n'
            '  foo = panel_data,\n'
            '  predictors = c("poverty_rate", "population", "urban_pct"),\n'
            '  dependent = "state_service_quality",\n'
            '  unit.variable = "county_id",\n'
            '  time.variable = "year",\n'
            '  treatment.identifier = potter_id,\n'
            '  controls.identifier = donor_ids,\n'
            '  time.predictors.prior = 2005:2014,\n'
            '  time.optimize.ssr = 2005:2014\n'
            ')\n\n'
            'synth.out <- synth(dataprep.out)\n'
            'path.plot(synth.out, dataprep.out)'
        ),
    },

    "qca": {
        "title": "Qualitative Comparative Analysis (QCA)",
        "tagline": "Not Variables — Configurations",
        "duration_minutes": 3,
        "plain_english": (
            "Traditional statistics ask: 'Does X cause Y, holding Z constant?' "
            "QCA asks: 'What COMBINATIONS of conditions produce the outcome?'\n\n"
            "It's set theory, not regression. A country might need BOTH "
            "strong institutions AND low inequality to achieve democracy — "
            "neither alone is sufficient. QCA finds these configurations."
        ),
        "political_science_example": (
            "What combination of factors produces successful welfare states? "
            "Maybe it's (strong unions + left government + small country) OR "
            "(corporatist culture + high GDP). QCA finds multiple 'paths' "
            "to the same outcome — equifinality.\n\n"
            "For your work: what configurations of factors produce 'state withdrawal' "
            "in counties? High charity + low capacity + rural? Or urban + fiscal crisis + corruption?"
        ),
        "assumptions": [
            {
                "name": "Equifinality",
                "plain": "Multiple different paths can lead to the same outcome.",
                "violation": "Not really a 'violation' — it's a feature. But you must accept that there's no single 'cause.'",
            },
            {
                "name": "Conjunctural Causation",
                "plain": "Conditions work in COMBINATION, not isolation.",
                "violation": "If you interpret each condition as an independent 'variable' rather than part of a configuration.",
            },
        ],
        "diagnostic_checklist": [
            "Calibrate set membership thresholds carefully",
            "Check truth table for contradictory configurations",
            "Report both parsimonious and intermediate solutions",
            "Consistency and coverage scores",
            "Sensitivity: vary calibration thresholds",
        ],
        "toy_dataset": {
            "description": "Simulated fsQCA: conditions for state withdrawal",
            "n": 30,
            "conditions": ["charity_density", "fiscal_stress", "rural_pct", "political_competition"],
            "outcome": "state_service_withdrawal",
        },
        "stata_code": "* QCA is typically done in R or dedicated software (fsQCA/QCA package)",
        "r_code": (
            '# === fsQCA: Conditions for State Withdrawal ===\n'
            'library(QCA)\n\n'
            '# Calibrate fuzzy sets\n'
            'df$fs_charity <- calibrate(df$charity_density,\n'
            '                           type = "fuzzy",\n'
            '                           thresholds = c(2, 5, 10))\n\n'
            '# Truth table\n'
            'tt <- truthTable(df, outcome = "state_withdrawal",\n'
            '                 conditions = c("fs_charity", "fs_fiscal",\n'
            '                                "fs_rural", "fs_competition"),\n'
            '                 incl.cut = 0.8)\n\n'
            '# Minimize\n'
            'solution <- minimize(tt, details = TRUE)\n'
            'print(solution)'
        ),
    },

    "bayesian": {
        "title": "Bayesian Inference",
        "tagline": "Update Your Beliefs With Evidence",
        "duration_minutes": 4,
        "plain_english": (
            "Frequentist stats: 'If we repeated this experiment 1000 times, "
            "95% of intervals would contain the true effect.'\n\n"
            "Bayesian stats: 'Given the data and my prior knowledge, there's "
            "a 95% probability the effect is between X and Y.'\n\n"
            "You START with a prior belief (from theory or past research), "
            "COMBINE it with new data, and GET a posterior belief. "
            "The more data, the less the prior matters."
        ),
        "political_science_example": (
            "You believe (from literature) that NGOs slightly reduce state capacity "
            "(your prior). You collect new data from Texas. Bayesian inference combines "
            "your prior with the new evidence to give a POSTERIOR estimate.\n\n"
            "If your data says NGOs help and the literature says they hurt, "
            "the posterior will be somewhere in between — weighted by sample sizes."
        ),
        "assumptions": [
            {
                "name": "Prior Specification",
                "plain": "You must choose a prior distribution. This is both a strength (you encode expertise) and a criticism.",
                "violation": "A poorly chosen prior can dominate with small samples. Always do sensitivity analysis.",
            },
            {
                "name": "Likelihood Model",
                "plain": "The data-generating process must be correctly specified.",
                "violation": "If you assume normality but the data is heavily skewed.",
            },
        ],
        "diagnostic_checklist": [
            "Prior sensitivity analysis: try informative vs weakly informative priors",
            "Check MCMC convergence: R-hat, effective sample size, trace plots",
            "Posterior predictive checks: does the model fit the data?",
            "Compare to frequentist results — large differences suggest model misspecification",
            "Report credible intervals, not confidence intervals",
        ],
        "toy_dataset": {
            "description": "Simulated Bayesian: NGO effect with informative prior",
            "n": 150,
            "prior_mean": -0.2,
            "prior_sd": 0.5,
            "true_effect": -0.3,
        },
        "stata_code": (
            "* === Bayesian Regression ===\n"
            "bayesmh state_capacity ngo_count controls, ///\n"
            "    likelihood(normal({var})) ///\n"
            "    prior({state_capacity:ngo_count}, normal(-0.2, 0.5)) ///\n"
            "    mcmcsize(10000) burnin(2000)"
        ),
        "r_code": (
            '# === Bayesian: NGO Effect ===\n'
            'library(brms)\n\n'
            '# Specify priors\n'
            'my_priors <- c(\n'
            '  prior(normal(-0.2, 0.5), class = "b", coef = "ngo_count"),\n'
            '  prior(normal(0, 1), class = "b")  # weakly informative for controls\n'
            ')\n\n'
            '# Fit model\n'
            'fit <- brm(state_capacity ~ ngo_count + controls,\n'
            '           data = df, prior = my_priors,\n'
            '           iter = 4000, chains = 4)\n\n'
            'summary(fit)\n'
            'plot(fit)  # trace + posterior plots'
        ),
    },

    "multilevel": {
        "title": "Multilevel / Hierarchical Linear Model (HLM)",
        "tagline": "Students in Classrooms in Schools in Districts",
        "duration_minutes": 3,
        "plain_english": (
            "Your data is NESTED: voters within counties within states. "
            "Observations within the same county are correlated — they share "
            "a local economy, a county government, etc.\n\n"
            "HLM handles this by modeling variation at MULTIPLE LEVELS: "
            "individual-level and county-level. It lets each county have its "
            "own intercept (and optionally, its own slope)."
        ),
        "political_science_example": (
            "Voter turnout depends on BOTH individual factors (education, age) "
            "AND county factors (polling place accessibility, party competition). "
            "HLM separates these. How much of turnout variation is BETWEEN counties "
            "vs WITHIN counties? The ICC (Intraclass Correlation) tells you."
        ),
        "assumptions": [
            {
                "name": "Nested Structure",
                "plain": "Units must be hierarchically nested (students in schools, not cross-classified).",
                "violation": "If students attend multiple schools (cross-classified), you need a different model.",
            },
            {
                "name": "Random Effects Distribution",
                "plain": "Group-level effects are assumed normally distributed.",
                "violation": "If some groups are extreme outliers, the normal assumption may not hold.",
            },
        ],
        "diagnostic_checklist": [
            "Calculate ICC — is multilevel modeling warranted?",
            "Check residual normality at each level",
            "Compare random intercept vs random slope models",
            "Test for cross-level interactions",
            "Verify sufficient group-level sample size (30+ groups)",
        ],
        "toy_dataset": {
            "description": "Simulated HLM: turnout in voters nested within counties",
            "n": 2000,
            "groups": 100,
            "outcome": "voter_turnout",
            "level1_predictors": ["education", "age", "income"],
            "level2_predictors": ["county_polling_access", "county_competition"],
        },
        "stata_code": (
            "* === HLM: Voter Turnout ===\n"
            "mixed voter_turnout education age income ///\n"
            "    county_polling_access county_competition ///\n"
            "    || county_id:, mle\n\n"
            "estat icc  // Intraclass correlation"
        ),
        "r_code": (
            '# === HLM: Voter Turnout ===\n'
            'library(lme4)\n\n'
            '# Random intercept model\n'
            'hlm <- lmer(voter_turnout ~ education + age + income +\n'
            '            county_polling_access + county_competition +\n'
            '            (1 | county_id), data = df)\n\n'
            'summary(hlm)\n'
            'performance::icc(hlm)  # ICC'
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════
# Toy Dataset Generator — Learn By Touching Data
# ═══════════════════════════════════════════════════════════════════

def generate_toy_dataset(estimator_id: str, seed: int = 42) -> dict:
    """Generate a toy dataset for hands-on methodology learning.

    "I've pulled a 'Toy Dataset' from the Bolt so you can visualize
    the 'jump' at the threshold right now."
    """
    random.seed(seed)
    spec = CRASH_COURSES.get(estimator_id, {}).get("toy_dataset")
    if not spec:
        return {"error": f"No toy dataset for estimator: {estimator_id}"}

    n = spec.get("n", 200)

    if estimator_id == "rdd":
        # Running variable centered at cutoff
        running = [random.gauss(0, 10) for _ in range(n)]
        treatment = [1 if r >= 0 else 0 for r in running]
        noise = [random.gauss(0, 5) for _ in range(n)]
        outcome = [50 + spec["true_effect"] * t + 2 * r + e
                    for r, t, e in zip(running, treatment, noise)]
        return {
            "type": "rdd",
            "columns": ["vote_margin", "incumbent_wins", "next_election_spending"],
            "data": [
                {"vote_margin": round(r, 2),
                 "incumbent_wins": t,
                 "next_election_spending": round(o, 2)}
                for r, t, o in zip(running, treatment, outcome)
            ],
            "n": n,
            "true_effect": spec["true_effect"],
        }

    elif estimator_id == "did":
        data = []
        for i in range(n // 2):
            state = i
            treated = 1 if i < n // 4 else 0
            for year in range(-spec.get("pre_periods", 5),
                              spec.get("post_periods", 5) + 1):
                post = 1 if year >= 0 else 0
                y = (20 + random.gauss(0, 2) - 0.3 * year
                     + spec["true_effect"] * treated * post)
                data.append({
                    "state_id": state,
                    "year": 2020 + year,
                    "treated": treated,
                    "post": post,
                    "poverty_rate": round(y, 2),
                })
        return {"type": "did", "data": data, "n": len(data),
                "true_effect": spec["true_effect"]}

    elif estimator_id == "iv":
        data = []
        for i in range(n):
            z = random.gauss(50, 20)  # instrument
            u = random.gauss(0, 1)    # unobservable
            x = 10 - 0.1 * z + u + random.gauss(0, 0.5)  # endogenous
            y = 5 + spec["true_effect"] * x + 0.5 * u + random.gauss(0, 1)
            data.append({
                "distance_to_foundation": round(z, 1),
                "ngo_count": round(max(0, x), 1),
                "state_service_index": round(y, 2),
            })
        return {"type": "iv", "data": data, "n": n,
                "true_effect": spec["true_effect"]}

    # Generic fallback
    return {"type": estimator_id, "n": n, "data": [],
            "note": "Detailed toy dataset generator for this estimator is in development."}


# ═══════════════════════════════════════════════════════════════════
# The Method Lab — The Complete Interactive Experience
# ═══════════════════════════════════════════════════════════════════

class MethodLab:
    """The Methodology Lab: turn every methodology into a crash course.

    When a Forensic Audit identifies an estimator, the Method Lab
    generates:
    1. Plain-English explanation (no jargon)
    2. Political science example (tailored to YOUR field)
    3. Toy dataset you can visualize RIGHT NOW
    4. Stata + R code you can run immediately
    5. Assumption checklist and diagnostic tests
    6. "Apply to Your Research" prompt

    Usage:
        lab = MethodLab()
        course = lab.generate_short_course("rdd")
    """

    def __init__(self):
        self._completed_courses: list[str] = []
        self._usage_log: list[dict] = []

    def generate_short_course(self, estimator_id: str,
                                paper_context: str = "",
                                user_topic: str = "") -> dict:
        """Generate a complete short course for an estimator.

        This is the "3-minute crash course" from the narrative.
        """
        course_data = CRASH_COURSES.get(estimator_id)
        if not course_data:
            return {"error": f"No crash course for: {estimator_id}",
                    "available": list(CRASH_COURSES.keys())}

        # Generate toy dataset
        toy_data = generate_toy_dataset(estimator_id)

        # Build the "Apply to Your Research" prompt
        apply_prompt = self._build_application_prompt(
            estimator_id, course_data, user_topic or "state capacity and charitable organizations"
        )

        course = {
            "title": course_data["title"],
            "tagline": course_data["tagline"],
            "duration_minutes": course_data["duration_minutes"],

            # The Crash Course
            "explanation": {
                "plain_english": course_data["plain_english"],
                "political_science_example": course_data["political_science_example"],
            },

            # Assumptions & Diagnostics
            "methodology": {
                "assumptions": course_data["assumptions"],
                "diagnostic_checklist": course_data["diagnostic_checklist"],
            },

            # Code — Ready to Run
            "code": {
                "stata": course_data["stata_code"],
                "r": course_data["r_code"],
            },

            # Toy Dataset — Touch the Data
            "toy_dataset": toy_data,

            # Apply to Your Research
            "application_prompt": apply_prompt,

            # Paper Context (if from a forensic audit)
            "paper_context": paper_context[:500] if paper_context else "",
        }

        # Log usage
        self._completed_courses.append(estimator_id)
        self._usage_log.append({
            "estimator": estimator_id,
            "timestamp": time.time(),
            "with_context": bool(paper_context),
        })

        return course

    def _build_application_prompt(self, estimator_id: str,
                                    course_data: dict,
                                    user_topic: str) -> str:
        """Build a "How would YOU use this?" prompt."""
        prompts = {
            "rdd": (
                f"🎓 **Apply RDD to Your Research:**\n\n"
                f"Think about your topic: *{user_topic}*.\n\n"
                f"1. What is your 'threshold'? Is there a cutoff score, "
                f"election margin, or eligibility boundary?\n"
                f"2. Can units manipulate their position at the threshold?\n"
                f"3. What is the 'running variable' (the continuous score)?\n"
                f"4. Would the effect at the cutoff generalize to your population?\n\n"
                f"**Try this**: Plot your running variable's density near the "
                f"cutoff. If there's bunching, the design may be invalid."
            ),
            "did": (
                f"🎓 **Apply DiD to Your Research:**\n\n"
                f"Think about your topic: *{user_topic}*.\n\n"
                f"1. What is the 'treatment' event? When did it happen?\n"
                f"2. Who got treated and who didn't?\n"
                f"3. Were the groups trending similarly BEFORE the treatment?\n"
                f"4. Could anything else have changed at the same time?\n\n"
                f"**Try this**: Plot your outcome variable for treatment and "
                f"control groups over time. Do the pre-treatment lines look parallel?"
            ),
            "iv": (
                f"🎓 **Apply IV to Your Research:**\n\n"
                f"Think about your topic: *{user_topic}*.\n\n"
                f"1. What is your endogeneity problem? What causes X and Y simultaneously?\n"
                f"2. What instrument could plausibly affect X but NOT Y directly?\n"
                f"3. Can you defend the exclusion restriction with theory?\n"
                f"4. Is your first-stage F > 10?\n\n"
                f"**Try this**: Write down three potential instruments. For each, "
                f"list every channel through which it could affect Y. If there's "
                f"only ONE channel (through X), you have a candidate."
            ),
        }
        return prompts.get(estimator_id,
            f"🎓 **Apply {course_data['title']} to Your Research:**\n\n"
            f"Consider how this methodology could address the causal questions "
            f"in your work on *{user_topic}*. What assumptions would need to hold? "
            f"What data would you need?"
        )

    def adjust(self, coefficients: dict) -> dict:
        """Adjust sandbox coefficients and return updated significance.

        Called by /api/forensic/sandbox to let users tweak model params
        and immediately see how p-values and CIs change.
        """
        if not coefficients:
            return {"status": "no_change", "coefficients": {}, "sandbox": self.get_sandbox_state()}
        # Simulate coefficient adjustment
        import math
        adjusted = {}
        for var, val in coefficients.items():
            try:
                coef = float(val)
            except (TypeError, ValueError):
                coef = 0.0
            se = max(abs(coef) * 0.3, 0.01)
            t_stat = coef / se if se else 0
            p_val = 2 * (1 - min(0.9999, 0.5 + 0.5 * math.erf(abs(t_stat) / math.sqrt(2))))
            adjusted[var] = {
                "coefficient": round(coef, 4),
                "std_error": round(se, 4),
                "t_statistic": round(t_stat, 3),
                "p_value": round(p_val, 4),
                "significant": p_val < 0.05,
            }
        return {
            "status": "adjusted",
            "variables": adjusted,
            "n_adjusted": len(adjusted),
        }

    def get_sandbox_state(self) -> dict:
        """Return current sandbox state for the forensic workbench panel."""
        return {
            "available_methods": list(CRASH_COURSES.keys()),
            "completed_courses": list(set(self._completed_courses)),
            "total_sessions": len(self._usage_log),
        }

    def get_available_courses(self) -> list[dict]:
        """List all available crash courses."""
        return [
            {
                "id": eid,
                "title": data["title"],
                "tagline": data["tagline"],
                "duration": data["duration_minutes"],
                "completed": eid in self._completed_courses,
            }
            for eid, data in CRASH_COURSES.items()
        ]

    @property
    def status(self) -> dict:
        return {
            "available_courses": len(CRASH_COURSES),
            "completed": len(set(self._completed_courses)),
            "total_sessions": len(self._usage_log),
        }


# Global instance
method_lab = MethodLab()
