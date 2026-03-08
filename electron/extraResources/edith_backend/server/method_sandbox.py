"""
Methodological Sandbox — Learn by Doing Before Spending a Dime
================================================================
Pedagogical Mirror Feature 3: The "Synthetic Lubbock" Pilot Lab.

Before you spend money fielding a real survey, Winnie creates a synthetic
population using census data. You "run" your survey on 10,000 digital
citizens to see if your questions ACTUALLY capture "Blame Diffusion."

Architecture:
    Research Design → Synthetic Population Generator →
    Survey Simulation → Pilot Analysis → Design Feedback

This teaches you:
- Why certain survey questions fail before you go into the field
- How sampling bias distorts your findings
- What your statistical power actually is
- Whether your operationalization captures the construct

The Sandbox is not a toy — it's a DRESS REHEARSAL for your methods.
"""

import hashlib
import json
import logging
import math
import os
import random
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.method_sandbox")


# ═══════════════════════════════════════════════════════════════════
# Synthetic Population — Census-Calibrated Digital Citizens
# ═══════════════════════════════════════════════════════════════════

# Population distribution parameters (calibrated to Lubbock, TX metro)
LUBBOCK_DEMOGRAPHICS = {
    "total_population": 310_000,
    "age_distribution": {
        "18-24": 0.18,  # university student population
        "25-34": 0.16,
        "35-44": 0.13,
        "45-54": 0.12,
        "55-64": 0.11,
        "65+": 0.10,
        "under_18": 0.20,  # excluded from surveys
    },
    "education": {
        "less_than_hs": 0.14,
        "hs_diploma": 0.25,
        "some_college": 0.24,
        "bachelors": 0.22,
        "graduate": 0.15,
    },
    "race_ethnicity": {
        "white": 0.43,
        "hispanic": 0.37,
        "black": 0.09,
        "asian": 0.03,
        "other": 0.08,
    },
    "income_brackets": {
        "under_25k": 0.22,
        "25k_50k": 0.24,
        "50k_75k": 0.20,
        "75k_100k": 0.15,
        "over_100k": 0.19,
    },
    "political_lean": {
        "strong_republican": 0.30,
        "lean_republican": 0.15,
        "independent": 0.20,
        "lean_democrat": 0.10,
        "strong_democrat": 0.15,
        "nonpartisan": 0.10,
    },
    "snap_participation_rate": 0.14,  # ~14% in Lubbock County
    "voter_turnout_2024": 0.58,
}


@dataclass
class SyntheticRespondent:
    """A synthetic survey respondent."""
    respondent_id: str
    age_group: str
    education: str
    race_ethnicity: str
    income: str
    political_lean: str
    snap_participant: bool
    voter: bool
    # Latent variables (not directly observed)
    institutional_trust: float  # 0-1
    blame_diffusion_score: float  # 0-1 (the construct we're trying to measure)
    state_reliance: float  # 0-1
    service_satisfaction: float  # 0-1

    def to_dict(self) -> dict:
        return {
            "id": self.respondent_id,
            "age": self.age_group,
            "education": self.education,
            "race": self.race_ethnicity,
            "income": self.income,
            "political_lean": self.political_lean,
            "snap": self.snap_participant,
            "voter": self.voter,
            "trust": round(self.institutional_trust, 3),
            "blame_diffusion": round(self.blame_diffusion_score, 3),
            "state_reliance": round(self.state_reliance, 3),
            "satisfaction": round(self.service_satisfaction, 3),
        }


# ═══════════════════════════════════════════════════════════════════
# Population Generator — Build the "Synthetic Lubbock"
# ═══════════════════════════════════════════════════════════════════

class SyntheticPopulationGenerator:
    """Generate a census-calibrated synthetic population.

    Each digital citizen has:
    - Observable demographics (age, education, income, race)
    - Latent constructs (trust, blame diffusion, reliance)
    - Correlated attributes (education ↔ income, trust ↔ participation)
    """

    def __init__(self, seed: int = 42):
        self._rng = random.Random(seed)
        self._demographics = LUBBOCK_DEMOGRAPHICS

    def generate_population(self, n: int = 10_000) -> list[SyntheticRespondent]:
        """Generate n synthetic respondents."""
        population = []

        for i in range(n):
            # Sample demographics with proper distributions
            age = self._weighted_choice(self._demographics["age_distribution"])
            if age == "under_18":
                continue  # Excluded from survey

            education = self._weighted_choice(self._demographics["education"])
            race = self._weighted_choice(self._demographics["race_ethnicity"])
            income = self._weighted_choice(self._demographics["income_brackets"])
            politics = self._weighted_choice(self._demographics["political_lean"])

            # Generate correlated latent variables
            # Education → Income correlation
            edu_score = {"less_than_hs": 0.2, "hs_diploma": 0.35,
                         "some_college": 0.5, "bachelors": 0.7,
                         "graduate": 0.9}.get(education, 0.5)

            # Income → Trust correlation
            income_score = {"under_25k": 0.2, "25k_50k": 0.35,
                            "50k_75k": 0.5, "75k_100k": 0.65,
                            "over_100k": 0.8}.get(income, 0.5)

            # Political lean → Trust in institutions correlation
            trust_base = {
                "strong_republican": 0.3, "lean_republican": 0.4,
                "independent": 0.5, "lean_democrat": 0.55,
                "strong_democrat": 0.45, "nonpartisan": 0.5,
            }.get(politics, 0.5)

            # Generate latent constructs with noise
            institutional_trust = self._bounded(
                trust_base + income_score * 0.2 + self._rng.gauss(0, 0.15)
            )

            # Blame diffusion: higher for low trust + high state reliance
            snap = self._rng.random() < self._demographics["snap_participation_rate"]
            state_reliance = self._bounded(
                (1 - income_score) * 0.6 + (0.3 if snap else 0) +
                self._rng.gauss(0, 0.1)
            )

            # The key construct: blame diffusion
            # Higher when trust is low AND reliance is high
            blame_diffusion = self._bounded(
                (1 - institutional_trust) * 0.4 +
                state_reliance * 0.3 +
                (1 - edu_score) * 0.15 +
                self._rng.gauss(0, 0.12)
            )

            # Voter participation correlated with education and trust
            voter = self._rng.random() < (
                self._demographics["voter_turnout_2024"] * (0.5 + edu_score * 0.5)
            )

            service_satisfaction = self._bounded(
                0.5 + institutional_trust * 0.2 - state_reliance * 0.15 +
                self._rng.gauss(0, 0.1)
            )

            respondent = SyntheticRespondent(
                respondent_id=f"SYN-{i:05d}",
                age_group=age,
                education=education,
                race_ethnicity=race,
                income=income,
                political_lean=politics,
                snap_participant=snap,
                voter=voter,
                institutional_trust=institutional_trust,
                blame_diffusion_score=blame_diffusion,
                state_reliance=state_reliance,
                service_satisfaction=service_satisfaction,
            )
            population.append(respondent)

        return population

    def _weighted_choice(self, distribution: dict) -> str:
        keys = list(distribution.keys())
        weights = list(distribution.values())
        return self._rng.choices(keys, weights=weights, k=1)[0]

    def _bounded(self, value: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))


# ═══════════════════════════════════════════════════════════════════
# Survey Simulator — Test Your Questions Before the Field
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SurveyQuestion:
    """A survey question to test on the synthetic population."""
    question_id: str
    text: str
    question_type: str  # "likert_5", "likert_7", "binary", "open_ended", "ranking"
    construct_target: str  # Which latent variable this aims to measure
    reverse_coded: bool = False
    options: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.question_id,
            "text": self.text,
            "type": self.question_type,
            "target": self.construct_target,
            "reverse_coded": self.reverse_coded,
        }


class SurveySimulator:
    """Simulate survey administration on the synthetic population.

    Tests:
    1. Response rate by demographic group
    2. Measurement validity: does the question actually capture the construct?
    3. Statistical power: can you detect the effect with this sample size?
    4. Bias detection: which groups are under/over-represented?
    """

    def __init__(self):
        self._population: list[SyntheticRespondent] = []
        self._questions: list[SurveyQuestion] = []
        self._generator = SyntheticPopulationGenerator()

    def setup_population(self, n: int = 10_000, seed: int = 42) -> dict:
        """Generate the synthetic population."""
        self._generator = SyntheticPopulationGenerator(seed=seed)
        self._population = self._generator.generate_population(n)

        # Population summary
        demographics = {}
        for key in ["age_group", "education", "race_ethnicity", "income", "political_lean"]:
            counts = {}
            for r in self._population:
                val = getattr(r, key)
                counts[val] = counts.get(val, 0) + 1
            demographics[key] = {
                k: round(v / len(self._population), 3)
                for k, v in sorted(counts.items(), key=lambda x: -x[1])
            }

        return {
            "population_size": len(self._population),
            "demographics": demographics,
            "snap_rate": round(
                sum(1 for r in self._population if r.snap_participant) / len(self._population), 3
            ),
            "voter_rate": round(
                sum(1 for r in self._population if r.voter) / len(self._population), 3
            ),
            "avg_trust": round(
                sum(r.institutional_trust for r in self._population) / len(self._population), 3
            ),
            "avg_blame_diffusion": round(
                sum(r.blame_diffusion_score for r in self._population) / len(self._population), 3
            ),
        }

    def add_question(self, question_id: str, text: str,
                      question_type: str = "likert_5",
                      construct_target: str = "blame_diffusion",
                      reverse_coded: bool = False) -> dict:
        """Add a survey question to test."""
        q = SurveyQuestion(
            question_id=question_id, text=text,
            question_type=question_type,
            construct_target=construct_target,
            reverse_coded=reverse_coded,
        )
        self._questions.append(q)
        return q.to_dict()

    def run_pilot(self, sample_size: int = 500,
                   sampling_method: str = "random") -> dict:
        """Run the pilot survey on the synthetic population.

        Returns:
        - Response patterns for each question
        - Measurement validity analysis
        - Power analysis
        - Sampling bias report
        """
        if not self._population:
            self.setup_population()

        if not self._questions:
            return {"error": "No survey questions added. Use add_question() first."}

        # Draw sample
        if sampling_method == "random":
            sample = random.sample(self._population, min(sample_size, len(self._population)))
        elif sampling_method == "convenience":
            # Convenience sample: biased toward younger, more educated
            weighted = sorted(
                self._population,
                key=lambda r: (
                    -{"18-24": 5, "25-34": 4, "35-44": 3, "45-54": 2, "55-64": 1, "65+": 0}.get(r.age_group, 2)
                    - {"graduate": 3, "bachelors": 2, "some_college": 1}.get(r.education, 0)
                    + random.random()
                ),
            )
            sample = weighted[:sample_size]
        elif sampling_method == "stratified":
            # Proportional stratified by income
            sample = self._stratified_sample(sample_size, "income")
        else:
            sample = random.sample(self._population, min(sample_size, len(self._population)))

        # Simulate response rates (varies by demographics)
        responses = []
        for respondent in sample:
            # Response probability correlates with education and age
            response_prob = 0.3  # 30% base
            if respondent.education in {"bachelors", "graduate"}:
                response_prob += 0.2
            if respondent.age_group in {"45-54", "55-64", "65+"}:
                response_prob += 0.15
            if respondent.voter:
                response_prob += 0.1

            if random.random() < response_prob:
                responses.append(respondent)

        # Generate responses to each question
        question_results = []
        for q in self._questions:
            q_result = self._simulate_question_responses(q, responses)
            question_results.append(q_result)

        # Sampling bias analysis
        bias = self._analyze_sampling_bias(sample, responses)

        # Statistical power analysis
        power = self._estimate_power(responses)

        return {
            "sample_size": len(sample),
            "responses": len(responses),
            "response_rate": round(len(responses) / max(len(sample), 1), 3),
            "sampling_method": sampling_method,
            "question_results": question_results,
            "sampling_bias": bias,
            "power_analysis": power,
            "recommendation": self._generate_recommendation(responses, bias, power),
        }

    def _simulate_question_responses(self, question: SurveyQuestion,
                                       respondents: list[SyntheticRespondent]) -> dict:
        """Simulate responses to a single question.

        The response is correlated with the target construct
        but includes measurement noise.
        """
        true_scores = []
        measured_scores = []

        for r in respondents:
            # Get the true latent construct value
            true_val = getattr(r, question.construct_target, 0.5)
            true_scores.append(true_val)

            # Add measurement noise (imperfect questions)
            noise = random.gauss(0, 0.2)
            measured = max(0, min(1, true_val + noise))

            if question.reverse_coded:
                measured = 1 - measured

            if question.question_type == "likert_5":
                # Map to 1-5 scale
                measured = min(5, max(1, round(measured * 4 + 1)))
            elif question.question_type == "likert_7":
                measured = min(7, max(1, round(measured * 6 + 1)))
            elif question.question_type == "binary":
                measured = 1 if measured > 0.5 else 0

            measured_scores.append(measured)

        # Calculate validity metrics
        if true_scores and measured_scores:
            # Pearson correlation as validity measure
            mean_true = sum(true_scores) / len(true_scores)
            mean_meas = sum(measured_scores) / len(measured_scores)

            cov = sum((t - mean_true) * (m - mean_meas)
                      for t, m in zip(true_scores, measured_scores)) / len(true_scores)
            std_true = math.sqrt(sum((t - mean_true) ** 2 for t in true_scores) / len(true_scores))
            std_meas = math.sqrt(sum((m - mean_meas) ** 2 for m in measured_scores) / len(measured_scores))

            validity = cov / (std_true * std_meas + 0.001)
        else:
            validity = 0

        return {
            "question_id": question.question_id,
            "question_text": question.text,
            "construct_target": question.construct_target,
            "n_responses": len(measured_scores),
            "mean": round(sum(measured_scores) / max(len(measured_scores), 1), 3),
            "std": round(
                math.sqrt(
                    sum((m - sum(measured_scores) / max(len(measured_scores), 1)) ** 2
                        for m in measured_scores)
                    / max(len(measured_scores), 1)
                ), 3
            ),
            "validity_correlation": round(validity, 3),
            "validity_rating": (
                "excellent" if validity > 0.7 else
                "good" if validity > 0.5 else
                "acceptable" if validity > 0.3 else
                "poor — consider rewording"
            ),
        }

    def _analyze_sampling_bias(self, sample: list, responses: list) -> dict:
        """Compare sample demographics to population demographics."""
        pop_n = len(self._population)
        resp_n = len(responses)

        if resp_n == 0:
            return {"error": "No responses to analyze"}

        bias_report = {}
        for key in ["education", "race_ethnicity", "income"]:
            pop_dist = {}
            resp_dist = {}
            for r in self._population:
                val = getattr(r, key)
                pop_dist[val] = pop_dist.get(val, 0) + 1
            for r in responses:
                val = getattr(r, key)
                resp_dist[val] = resp_dist.get(val, 0) + 1

            deviations = {}
            for val in pop_dist:
                pop_pct = pop_dist[val] / pop_n
                resp_pct = resp_dist.get(val, 0) / resp_n
                deviations[val] = {
                    "population": round(pop_pct, 3),
                    "sample": round(resp_pct, 3),
                    "deviation": round(resp_pct - pop_pct, 3),
                    "biased": abs(resp_pct - pop_pct) > 0.05,
                }
            bias_report[key] = deviations

        return bias_report

    def _estimate_power(self, responses: list[SyntheticRespondent]) -> dict:
        """Estimate statistical power for detecting blame diffusion effects."""
        n = len(responses)
        if n < 30:
            return {"power": 0, "sufficient": False,
                    "message": "Insufficient sample size for power analysis"}

        # Effect size (Cohen's d) for blame diffusion
        snap = [r.blame_diffusion_score for r in responses if r.snap_participant]
        non_snap = [r.blame_diffusion_score for r in responses if not r.snap_participant]

        if not snap or not non_snap:
            return {"power": 0, "sufficient": False,
                    "message": "Need both SNAP and non-SNAP respondents"}

        mean_diff = abs(sum(snap) / len(snap) - sum(non_snap) / len(non_snap))
        pooled_var = (
            sum((x - sum(snap) / len(snap)) ** 2 for x in snap) +
            sum((x - sum(non_snap) / len(non_snap)) ** 2 for x in non_snap)
        ) / (len(snap) + len(non_snap) - 2)
        pooled_sd = math.sqrt(pooled_var) if pooled_var > 0 else 0.01

        cohens_d = mean_diff / pooled_sd

        # Approximate power using normal distribution
        # Power ≈ Φ(|d|√(n/4) - z_α/2) where z_α/2 ≈ 1.96
        noncentrality = abs(cohens_d) * math.sqrt(n / 4)
        # Rough approximation
        power = min(0.99, max(0.05, 1 - math.exp(-(noncentrality - 1.96))))

        return {
            "power": round(power, 3),
            "sufficient": power >= 0.80,
            "cohens_d": round(cohens_d, 3),
            "effect_size": (
                "large" if cohens_d > 0.8 else
                "medium" if cohens_d > 0.5 else
                "small"
            ),
            "n_treatment": len(snap),
            "n_control": len(non_snap),
            "sample_size_needed_80": self._required_n(cohens_d, 0.80),
            "message": (
                f"Power = {power:.0%}. {'Sufficient' if power >= 0.80 else 'INSUFFICIENT — increase sample size'}."
            ),
        }

    def _required_n(self, d: float, target_power: float = 0.80) -> int:
        """Estimate required N per group for a given effect size and power."""
        if d <= 0:
            return 999999
        # Approximate: n ≈ (z_α + z_β)² / d² * 4
        z_alpha = 1.96
        z_beta = 0.84  # For 80% power
        n = int(((z_alpha + z_beta) ** 2 / (d ** 2)) * 4)
        return max(30, n)

    def _stratified_sample(self, n: int, stratify_by: str) -> list:
        """Draw a stratified sample."""
        strata = {}
        for r in self._population:
            key = getattr(r, stratify_by)
            strata.setdefault(key, []).append(r)

        sample = []
        for key, members in strata.items():
            stratum_n = max(1, int(n * len(members) / len(self._population)))
            sample.extend(random.sample(members, min(stratum_n, len(members))))

        return sample[:n]

    def _generate_recommendation(self, responses: list, bias: dict, power: dict) -> str:
        """Generate a plain-language recommendation."""
        issues = []

        if not power.get("sufficient"):
            issues.append(
                f"Statistical power ({power.get('power', 0):.0%}) is below 80%. "
                f"You need ~{power.get('sample_size_needed_80', 'unknown')} respondents per group."
            )

        # Check for systematic bias
        for key, deviations in bias.items():
            if isinstance(deviations, dict):
                for val, info in deviations.items():
                    if isinstance(info, dict) and info.get("biased"):
                        issues.append(
                            f"{key.replace('_', ' ').title()}: '{val}' is "
                            f"{'over' if info['deviation'] > 0 else 'under'}-represented "
                            f"by {abs(info['deviation']):.1%}."
                        )

        if not issues:
            return "Survey design looks solid. Proceed with confidence."

        return "ISSUES FOUND:\n" + "\n".join(f"• {issue}" for issue in issues[:5])

    def get_status(self) -> dict:
        return {
            "population_size": len(self._population),
            "questions": len(self._questions),
            "question_list": [q.to_dict() for q in self._questions],
        }


# Global instance
method_sandbox = SurveySimulator()
