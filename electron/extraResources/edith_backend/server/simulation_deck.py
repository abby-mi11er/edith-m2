"""
Simulation Deck — Extreme PolSci Engine
=========================================
The "Virtual University" that turns the M4 + Bolt into a
generative social science laboratory.

Three Capabilities:
  1. Agent-Based Modeling — 10,000 synthetic agents in "Synthetic Texas"
  2. Game Theory Designer — PSRO, Nash equilibria, payoff matrices
  3. Synthetic Controls — Counterfactual county twins for causal impact

Hardware Requirements:
  - 3,100 MB/s sustained throughput (Oyen Bolt U34)
  - M4 Neural Engine for agent LLM inference
  - Thermal endurance for 72-hour simulation runs
"""

import copy
import hashlib
import json
import logging
import math
import os
import random
import re
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

log = logging.getLogger("edith.simulation")


# ═══════════════════════════════════════════════════════════════════
# §1: AGENT-BASED MODELING — "Synthetic Texas"
# 10,000 agents grounded in census data, interacting over time
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SyntheticAgent:
    """A single synthetic citizen in the simulation."""
    agent_id: int
    county: str = "lubbock"
    income: float = 0.0
    education: float = 0.0       # 0-20 years
    age: int = 35
    party_id: float = 0.0        # -1 (strong D) to +1 (strong R)
    snap_enrolled: bool = False
    charity_user: bool = False
    voter: bool = True
    policy_awareness: float = 0.5  # 0-1, how "visible" is gov to them
    social_network: list = field(default_factory=list)
    anger: float = 0.0           # 0-1, political frustration
    opinion_history: list = field(default_factory=list)

    def update_opinion(self, shock_effect: float, neighbor_influence: float):
        """Update agent's opinion based on shock + social contagion."""
        # Direct shock effect (e.g., SNAP cut)
        self.anger = max(0, min(1, self.anger + shock_effect))

        # Submerged state effect: low awareness dampens reaction
        if self.policy_awareness < 0.3:
            self.anger *= 0.7  # Can't be angry about what you don't see

        # Social contagion from neighbors
        self.anger = max(0, min(1,
            self.anger * 0.7 + neighbor_influence * 0.3
        ))

        # Update party identification drift
        if self.anger > 0.6 and self.snap_enrolled:
            self.party_id -= 0.01  # Angry SNAP users drift left
        elif self.anger > 0.6 and self.charity_user:
            self.party_id += 0.005  # Angry charity users drift right

        self.party_id = max(-1, min(1, self.party_id))

        # Voter mobilization: very angry or very calm people vote
        self.voter = self.anger > 0.4 or self.anger < 0.1

        self.opinion_history.append({
            "anger": round(self.anger, 3),
            "party_id": round(self.party_id, 3),
            "voter": self.voter,
        })


class AgentBasedModel:
    """Synthetic Society simulation.

    Populate with N agents grounded in census data,
    introduce shocks, and observe emergent behavior.
    §IMP-3.5: Stores per-month snapshots for time scrubbing.
    """

    def __init__(self, n_agents: int = 10000, county: str = "lubbock"):
        self.n_agents = n_agents
        self.county = county
        self.agents: list[SyntheticAgent] = []
        self.month = 0
        self.history: list[dict] = []
        self.shocks: list[dict] = []
        self._lock = threading.Lock()
        # §IMP-3.5: Per-month snapshots for replay/time scrubbing
        self.snapshots: list[dict] = []

    def populate(self, census_data: dict = None) -> dict:
        """Create N synthetic agents grounded in census distributions.

        §IMP-3.1: Loads real census data from VAULT CSV if available.
        Falls back to synthetic distributions.
        """
        t0 = time.time()
        self.agents = []

        # §IMP-3.1: Try loading real census data from VAULT
        if not census_data:
            census_data = self._load_census_csv()

        # Default census distributions (Lubbock County, TX approximation)
        cd = census_data or {
            "median_income": 52000,
            "income_std": 28000,
            "median_education": 13.5,
            "education_std": 3.0,
            "median_age": 33,
            "age_std": 15,
            "snap_rate": 0.12,
            "charity_usage": 0.08,
            "party_lean": 0.25,  # slightly R
            "voter_reg_rate": 0.65,
        }

        for i in range(self.n_agents):
            income = max(0, random.gauss(cd["median_income"], cd["income_std"]))
            education = max(0, min(20, random.gauss(cd["median_education"], cd["education_std"])))
            age = max(18, min(95, int(random.gauss(cd["median_age"], cd["age_std"]))))

            # SNAP enrollment inversely correlated with income
            snap_prob = cd["snap_rate"] * (1 + max(0, (30000 - income) / 30000))
            snap = random.random() < min(snap_prob, 0.5)

            # Charity usage (mild positive correlation with rural, negative with income)
            charity_prob = cd["charity_usage"] * (1 + max(0, (40000 - income) / 40000))
            charity = random.random() < min(charity_prob, 0.3)

            # Party ID: income + education effects
            party_base = cd["party_lean"]
            party_base += (income - cd["median_income"]) / (cd["income_std"] * 6)
            party_base -= (education - cd["median_education"]) / (cd["education_std"] * 4)
            party_id = max(-1, min(1, random.gauss(party_base, 0.3)))

            # Policy awareness: education + income + age effects
            awareness = min(1, max(0,
                0.3 + education / 40 + (income / cd["median_income"]) * 0.1
                + (1 if snap else 0) * (-0.15)  # Submerged state effect
            ))

            agent = SyntheticAgent(
                agent_id=i,
                county=self.county,
                income=round(income, 2),
                education=round(education, 1),
                age=age,
                party_id=round(party_id, 3),
                snap_enrolled=snap,
                charity_user=charity,
                voter=random.random() < cd["voter_reg_rate"],
                policy_awareness=round(awareness, 3),
                anger=round(random.uniform(0, 0.3), 3),
            )
            self.agents.append(agent)

        # Build social networks (each agent knows ~20 neighbors)
        self._build_social_network()

        elapsed = time.time() - t0
        return {
            "agents_created": self.n_agents,
            "county": self.county,
            "census_source": "csv" if cd != census_data else "synthetic",
            "elapsed_ms": round(elapsed * 1000, 1),
            "demographics": self._compute_aggregate(),
        }

    @staticmethod
    def _load_census_csv() -> dict:
        """§IMP-3.1: Load real census data from VAULT/ARTEFACTS/census_*.csv."""
        import csv as _csv
        data_root = os.environ.get("EDITH_DATA_ROOT", "")
        if not data_root:
            return None
        artefacts = os.path.join(data_root, "ARTEFACTS")
        for fname in ["census_lubbock.csv", "census_data.csv", "acs_data.csv"]:
            path = os.path.join(artefacts, fname)
            if os.path.exists(path):
                try:
                    with open(path) as f:
                        reader = _csv.DictReader(f)
                        rows = list(reader)
                    if rows:
                        # Extract key variables, average across rows
                        incomes = [float(r.get("median_income", r.get("income", 52000))) for r in rows]
                        result = {
                            "median_income": statistics.mean(incomes),
                            "income_std": statistics.stdev(incomes) if len(incomes) > 1 else 28000,
                            "snap_rate": float(rows[0].get("snap_rate", 0.12)),
                            "voter_reg_rate": float(rows[0].get("voter_reg_rate", 0.65)),
                        }
                        log.info(f"§CENSUS: Loaded real data from {fname} ({len(rows)} rows)")
                        return result
                except Exception as e:
                    log.warning(f"§CENSUS: Failed to load {fname}: {e}")
        return None

    def _build_social_network(self, avg_connections: int = 20):
        """Build social network with homophily (similar agents connect)."""
        for agent in self.agents:
            # Prefer agents with similar income and party_id
            candidates = random.sample(
                self.agents,
                min(100, self.n_agents)
            )
            scored = []
            for c in candidates:
                if c.agent_id == agent.agent_id:
                    continue
                similarity = (
                    1.0 - abs(agent.party_id - c.party_id) / 2
                    + (1.0 - abs(agent.income - c.income) / 100000) * 0.5
                )
                scored.append((c.agent_id, similarity))

            scored.sort(key=lambda x: -x[1])
            agent.social_network = [s[0] for s in scored[:avg_connections]]

    def introduce_shock(
        self,
        shock_type: str,
        magnitude: float,
        description: str = "",
        target_filter: dict = None,
    ) -> dict:
        """Introduce a policy shock into the simulation.

        Shock types: snap_cut, charity_closure, income_shock,
                     information_campaign, election_event
        """
        shock = {
            "type": shock_type,
            "magnitude": magnitude,
            "description": description or f"{shock_type} (magnitude: {magnitude})",
            "month_introduced": self.month,
            "agents_affected": 0,
        }

        affected = 0
        for agent in self.agents:
            # Check if agent matches target filter
            if target_filter:
                if target_filter.get("snap_only") and not agent.snap_enrolled:
                    continue
                if target_filter.get("charity_only") and not agent.charity_user:
                    continue
                min_income = target_filter.get("min_income", 0)
                max_income = target_filter.get("max_income", float("inf"))
                if not (min_income <= agent.income <= max_income):
                    continue

            if shock_type == "snap_cut":
                if agent.snap_enrolled:
                    agent.anger += magnitude * 0.5
                    agent.income -= agent.income * magnitude * 0.1
                    affected += 1
                elif agent.policy_awareness < 0.4:
                    pass  # Submerged state: unaware citizens don't react
                else:
                    agent.anger += magnitude * 0.05

            elif shock_type == "charity_closure":
                if agent.charity_user:
                    agent.charity_user = False
                    agent.anger += magnitude * 0.4
                    # Charity loss may increase SNAP dependency
                    if random.random() < 0.3:
                        agent.snap_enrolled = True
                    affected += 1

            elif shock_type == "income_shock":
                agent.income *= (1 - magnitude)
                if agent.income < 25000 and not agent.snap_enrolled:
                    if random.random() < 0.2:
                        agent.snap_enrolled = True
                affected += 1

            elif shock_type == "information_campaign":
                agent.policy_awareness = min(1, agent.policy_awareness + magnitude * 0.3)
                affected += 1

            elif shock_type == "election_event":
                agent.anger += magnitude * random.uniform(-0.1, 0.2)
                affected += 1

        shock["agents_affected"] = affected
        self.shocks.append(shock)
        return shock

    def simulate_month(self) -> dict:
        """Advance the simulation by one month.

        §IMP-3.5: Takes a snapshot for replay/time-scrubbing.
        """
        self.month += 1
        t0 = time.time()

        # Every agent updates based on shock + neighbors
        for agent in self.agents:
            # Calculate neighbor influence
            neighbor_angers = []
            for nid in agent.social_network:
                if 0 <= nid < len(self.agents):
                    neighbor_angers.append(self.agents[nid].anger)

            avg_neighbor = (
                statistics.mean(neighbor_angers) if neighbor_angers else 0
            )

            # Decay: anger naturally subsides without reinforcement
            decay = -0.02

            agent.update_opinion(decay, avg_neighbor)

        # Record aggregate state
        agg = self._compute_aggregate()
        agg["month"] = self.month
        agg["elapsed_ms"] = round((time.time() - t0) * 1000, 1)
        self.history.append(agg)

        # §IMP-3.5: Store per-month snapshot for time scrubbing
        self.snapshots.append({
            "month": self.month,
            "aggregate": agg,
            "distributions": {
                "anger": [round(a.anger, 3) for a in self.agents[:100]],  # Sample for perf
                "party_id": [round(a.party_id, 3) for a in self.agents[:100]],
            },
            "shocks_active": len(self.shocks),
        })

        return agg

    def simulate_months(self, n_months: int = 12) -> dict:
        """Run a full simulation for N months."""
        t0 = time.time()
        monthly_results = []

        for _ in range(n_months):
            result = self.simulate_month()
            monthly_results.append(result)

        # Compute trajectory
        anger_trajectory = [r["mean_anger"] for r in monthly_results]
        turnout_trajectory = [r["voter_turnout"] for r in monthly_results]
        party_trajectory = [r["mean_party_id"] for r in monthly_results]

        return {
            "months_simulated": n_months,
            "total_elapsed_ms": round((time.time() - t0) * 1000, 1),
            "final_state": monthly_results[-1],
            "trajectories": {
                "anger": anger_trajectory,
                "turnout": turnout_trajectory,
                "party_id": party_trajectory,
            },
            "shocks_applied": len(self.shocks),
            "monthly_results": monthly_results,
            # §IMP-3.5: Include snapshots for time-scrubbing UI
            "snapshots": self.snapshots,
        }

    def _compute_aggregate(self) -> dict:
        """Compute aggregate statistics across all agents."""
        angers = [a.anger for a in self.agents]
        pi = [a.party_id for a in self.agents]
        incomes = [a.income for a in self.agents]
        voters = sum(1 for a in self.agents if a.voter)
        snap = sum(1 for a in self.agents if a.snap_enrolled)
        charity = sum(1 for a in self.agents if a.charity_user)

        return {
            "mean_anger": round(statistics.mean(angers), 4),
            "anger_std": round(statistics.stdev(angers), 4) if len(angers) > 1 else 0,
            "mean_party_id": round(statistics.mean(pi), 4),
            "voter_turnout": round(voters / max(len(self.agents), 1), 4),
            "snap_rate": round(snap / max(len(self.agents), 1), 4),
            "charity_rate": round(charity / max(len(self.agents), 1), 4),
            "mean_income": round(statistics.mean(incomes), 2),
            "total_agents": len(self.agents),
        }

    def get_distribution(self, variable: str = "anger") -> dict:
        """Get the full distribution of a variable across agents."""
        values = []
        for a in self.agents:
            if variable == "anger":
                values.append(a.anger)
            elif variable == "party_id":
                values.append(a.party_id)
            elif variable == "income":
                values.append(a.income)
            elif variable == "policy_awareness":
                values.append(a.policy_awareness)

        if not values:
            return {"error": f"Unknown variable: {variable}"}

        sorted_v = sorted(values)
        n = len(sorted_v)
        return {
            "variable": variable,
            "n": n,
            "mean": round(statistics.mean(values), 4),
            "median": round(sorted_v[n // 2], 4),
            "std": round(statistics.stdev(values), 4) if n > 1 else 0,
            "p10": round(sorted_v[int(n * 0.1)], 4),
            "p25": round(sorted_v[int(n * 0.25)], 4),
            "p75": round(sorted_v[int(n * 0.75)], 4),
            "p90": round(sorted_v[int(n * 0.9)], 4),
            "min": round(sorted_v[0], 4),
            "max": round(sorted_v[-1], 4),
        }


# ═══════════════════════════════════════════════════════════════════
# §2: GAME THEORY DESIGNER — PSRO + Nash Equilibrium
# "She hands you a Payoff Matrix and a Nash Equilibrium."
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Strategy:
    """A strategy in a game-theoretic model."""
    name: str
    description: str = ""
    payoff_function: str = ""  # Symbolic representation


class GameTheoryDesigner:
    """Automated Formal Modeling with PSRO.

    Discovers Nash Equilibria for political science games.
    """

    def __init__(self):
        self.players: list[str] = []
        self.strategies: dict[str, list[Strategy]] = {}
        self.payoff_matrix: dict = {}
        self.equilibria: list[dict] = []

    def define_game(
        self,
        players: list[str],
        strategies: dict[str, list[str]],
        description: str = "",
    ) -> dict:
        """Define a game with players and their strategy sets."""
        self.players = players
        self.strategies = {
            p: [Strategy(name=s) for s in strats]
            for p, strats in strategies.items()
        }
        return {
            "game_defined": True,
            "players": players,
            "strategies": {p: [s.name for s in sl] for p, sl in self.strategies.items()},
            "description": description,
        }

    def compute_payoff_matrix(
        self,
        payoff_rules: dict = None,
        model_chain: list[str] = None,
    ) -> dict:
        """Compute the payoff matrix for all strategy combinations.

        Can use LLM to generate payoffs for complex political games.
        """
        model_chain = model_chain or [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]

        if len(self.players) != 2:
            return {"error": "Currently supports 2-player games"}

        p1, p2 = self.players
        s1 = [s.name for s in self.strategies[p1]]
        s2 = [s.name for s in self.strategies[p2]]

        matrix = {}
        for a in s1:
            for b in s2:
                key = f"{a}/{b}"
                if payoff_rules and key in payoff_rules:
                    matrix[key] = payoff_rules[key]
                else:
                    # Generate payoffs via LLM
                    try:
                        from server.backend_logic import generate_text_via_chain
                        prompt = (
                            f"GAME: {p1} vs {p2}\n"
                            f"Strategy pair: {p1} plays '{a}', {p2} plays '{b}'\n"
                            f"Assign realistic payoffs (0-10) for each player.\n"
                            f"Output ONLY as JSON: {{\"p1\": X, \"p2\": Y}}"
                        )
                        text, _ = generate_text_via_chain(
                            prompt, model_chain,
                            system_instruction="You are a game theory expert. Output valid JSON only.",
                            temperature=0.1,
                        )
                        parsed = json.loads(re.search(r'\{.*\}', text).group())
                        matrix[key] = {"p1": parsed.get("p1", 5), "p2": parsed.get("p2", 5)}
                    except Exception:
                        matrix[key] = {
                            "p1": random.randint(1, 10),
                            "p2": random.randint(1, 10),
                        }

        self.payoff_matrix = matrix
        return {
            "matrix": matrix,
            "players": self.players,
            "strategies": {p1: s1, p2: s2},
            # §IMP-3.10: Formatted payoff table for frontend display
            "formatted_table": self._format_payoff_table(matrix, p1, p2, s1, s2),
        }

    def _format_payoff_table(self, matrix, p1, p2, s1, s2) -> dict:
        """§IMP-3.10: Format payoff matrix as a structured table for UI rendering."""
        headers = [f"{p2}: {s}" for s in s2]
        rows = []
        for a in s1:
            row = {"label": f"{p1}: {a}", "cells": []}
            for b in s2:
                key = f"{a}/{b}"
                payoffs = matrix.get(key, {"p1": 0, "p2": 0})
                row["cells"].append({
                    "p1": payoffs.get("p1", 0),
                    "p2": payoffs.get("p2", 0),
                    "display": f"({payoffs.get('p1', 0)}, {payoffs.get('p2', 0)})",
                })
            rows.append(row)
        return {"headers": headers, "rows": rows, "p1_label": p1, "p2_label": p2}

    def find_nash_equilibria(self) -> dict:
        """Find all pure-strategy Nash Equilibria in the payoff matrix."""
        if not self.payoff_matrix:
            return {"error": "Compute payoff matrix first"}

        p1, p2 = self.players
        s1 = [s.name for s in self.strategies[p1]]
        s2 = [s.name for s in self.strategies[p2]]

        equilibria = []

        for a in s1:
            for b in s2:
                key = f"{a}/{b}"
                current = self.payoff_matrix.get(key, {})
                is_nash = True

                # Check if p1 can profitably deviate
                for alt_a in s1:
                    if alt_a == a:
                        continue
                    alt_key = f"{alt_a}/{b}"
                    alt_payoff = self.payoff_matrix.get(alt_key, {})
                    if alt_payoff.get("p1", 0) > current.get("p1", 0):
                        is_nash = False
                        break

                if not is_nash:
                    continue

                # Check if p2 can profitably deviate
                for alt_b in s2:
                    if alt_b == b:
                        continue
                    alt_key = f"{a}/{alt_b}"
                    alt_payoff = self.payoff_matrix.get(alt_key, {})
                    if alt_payoff.get("p2", 0) > current.get("p2", 0):
                        is_nash = False
                        break

                if is_nash:
                    equilibria.append({
                        "strategies": {p1: a, p2: b},
                        "payoffs": current,
                        "type": "pure_strategy",
                    })

        self.equilibria = equilibria
        return {
            "equilibria": equilibria,
            "count": len(equilibria),
            "has_dominant_strategy": self._check_dominant(s1, s2),
        }

    def find_mixed_nash(self) -> dict:
        """Find mixed-strategy Nash Equilibrium for 2x2 games."""
        p1, p2 = self.players
        s1 = [s.name for s in self.strategies[p1]]
        s2 = [s.name for s in self.strategies[p2]]

        if len(s1) != 2 or len(s2) != 2:
            return {"error": "Mixed strategy solver requires 2x2 game"}

        a, b = s1
        c, d = s2

        def _pay(entry, key):
            """Get payoff from entry, trying 'p1'/'p2' or actual player names."""
            if isinstance(entry, dict):
                return entry.get(key, entry.get(p1 if key == "p1" else p2, 0))
            return 0

        # Extract payoffs
        ac = self.payoff_matrix.get(f"{a}/{c}", {})
        ad = self.payoff_matrix.get(f"{a}/{d}", {})
        bc = self.payoff_matrix.get(f"{b}/{c}", {})
        bd = self.payoff_matrix.get(f"{b}/{d}", {})

        # P2's mixing probability (makes P1 indifferent)
        denom1 = (_pay(ac, "p1") - _pay(ad, "p1")) - (_pay(bc, "p1") - _pay(bd, "p1"))
        if abs(denom1) < 1e-10:
            return {"error": "No interior mixed equilibrium exists"}
        q = (_pay(bd, "p1") - _pay(ad, "p1")) / denom1

        # P1's mixing probability (makes P2 indifferent)
        denom2 = (_pay(ac, "p2") - _pay(bc, "p2")) - (_pay(ad, "p2") - _pay(bd, "p2"))
        if abs(denom2) < 1e-10:
            return {"error": "No interior mixed equilibrium exists"}
        p = (_pay(bd, "p2") - _pay(bc, "p2")) / denom2

        if not (0 <= p <= 1 and 0 <= q <= 1):
            return {"error": "Mixed equilibrium out of probability bounds"}

        return {
            "mixed_equilibrium": {
                p1: {a: round(p, 4), b: round(1 - p, 4)},
                p2: {c: round(q, 4), d: round(1 - q, 4)},
            },
            "expected_payoffs": {
                p1: round(p * q * _pay(ac, "p1") + p * (1-q) * _pay(ad, "p1")
                     + (1-p) * q * _pay(bc, "p1") + (1-p) * (1-q) * _pay(bd, "p1"), 4),
                p2: round(p * q * _pay(ac, "p2") + p * (1-q) * _pay(ad, "p2")
                     + (1-p) * q * _pay(bc, "p2") + (1-p) * (1-q) * _pay(bd, "p2"), 4),
            },
            "type": "mixed_strategy",
        }

    def psro_self_play(self, rounds: int = 1000) -> dict:
        """Policy-Space Response Oracle — self-play to discover equilibria.

        Agents play the game thousands of times, learning best responses.
        """
        if not self.payoff_matrix:
            return {"error": "Define payoff matrix first"}

        p1, p2 = self.players
        s1 = [s.name for s in self.strategies[p1]]
        s2 = [s.name for s in self.strategies[p2]]

        # Initialize uniform strategy weights
        weights1 = {s: 1.0 / len(s1) for s in s1}
        weights2 = {s: 1.0 / len(s2) for s in s2}

        history = []
        learning_rate = 0.05

        for r in range(rounds):
            # Sample strategies
            a = random.choices(s1, weights=[weights1[s] for s in s1])[0]
            b = random.choices(s2, weights=[weights2[s] for s in s2])[0]

            key = f"{a}/{b}"
            payoffs = self.payoff_matrix.get(key, {"p1": 0, "p2": 0})

            # Update weights (fictitious play)
            for s in s1:
                expected = sum(
                    weights2[s2_s] * self.payoff_matrix.get(f"{s}/{s2_s}", {}).get("p1", 0)
                    for s2_s in s2
                )
                weights1[s] += learning_rate * expected
            for s in s2:
                expected = sum(
                    weights1[s1_s] * self.payoff_matrix.get(f"{s1_s}/{s}", {}).get("p2", 0)
                    for s1_s in s1
                )
                weights2[s] += learning_rate * expected

            # Normalize
            total1 = sum(weights1.values())
            total2 = sum(weights2.values())
            weights1 = {s: w / total1 for s, w in weights1.items()}
            weights2 = {s: w / total2 for s, w in weights2.items()}

            if r % (rounds // 10) == 0:
                history.append({
                    "round": r,
                    "weights_p1": dict(weights1),
                    "weights_p2": dict(weights2),
                })

        # Find dominant strategy from weights
        best1 = max(weights1, key=weights1.get)
        best2 = max(weights2, key=weights2.get)

        return {
            "rounds": rounds,
            "converged_strategy": {
                p1: {s: round(w, 4) for s, w in weights1.items()},
                p2: {s: round(w, 4) for s, w in weights2.items()},
            },
            "dominant": {p1: best1, p2: best2},
            "history": history,
        }

    def _check_dominant(self, s1: list, s2: list) -> dict:
        p1, p2 = self.players
        dominant = {}
        for player_strats, opp_strats, pi_key, player in [
            (s1, s2, "p1", p1), (s2, s1, "p2", p2)
        ]:
            for s in player_strats:
                is_dominant = True
                for alt in player_strats:
                    if alt == s:
                        continue
                    for opp in opp_strats:
                        k1 = f"{s}/{opp}" if pi_key == "p1" else f"{opp}/{s}"
                        k2 = f"{alt}/{opp}" if pi_key == "p1" else f"{opp}/{alt}"
                        if self.payoff_matrix.get(k1, {}).get(pi_key, 0) <= \
                           self.payoff_matrix.get(k2, {}).get(pi_key, 0):
                            is_dominant = False
                            break
                    if not is_dominant:
                        break
                if is_dominant:
                    dominant[player] = s
        return dominant

    def analyze_political_game(
        self,
        description: str,
        model_chain: list[str] = None,
    ) -> dict:
        """Auto-design a game from a political science description.

        "If a charity provides SNAP-like benefits, but the GOP takes credit..."
        """
        model_chain = model_chain or [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]

        prompt = (
            f"POLITICAL PUZZLE: {description}\n\n"
            f"Design a 2-player game theory model:\n"
            f"1. Who are the 2 players?\n"
            f"2. What are each player's 2-3 strategies?\n"
            f"3. What are the payoffs for each strategy combination? (1-10 scale)\n\n"
            f"Output as JSON:\n"
            f'{{"players": ["P1", "P2"], '
            f'"strategies": {{"P1": ["s1", "s2"], "P2": ["s1", "s2"]}}, '
            f'"payoffs": {{"s1/s1": {{"p1": X, "p2": Y}}, ...}}, '
            f'"reasoning": "explanation"}}'
        )

        try:
            from server.backend_logic import generate_text_via_chain
            text, model = generate_text_via_chain(
                prompt, model_chain,
                system_instruction=(
                    "You are a formal political theory expert. Design rigorous "
                    "game-theoretic models with realistic payoffs. Output valid JSON."
                ),
                temperature=0.2,
            )
            parsed = json.loads(re.search(r'\{.*\}', text, re.DOTALL).group())

            # Auto-populate the designer
            self.define_game(parsed["players"], parsed["strategies"])
            self.payoff_matrix = parsed.get("payoffs", {})

            # Find equilibria
            nash = self.find_nash_equilibria()
            psro = self.psro_self_play(500)

            return {
                "game": parsed,
                "nash_equilibria": nash,
                "psro_result": psro,
                "model": model,
            }
        except Exception as e:
            return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# §3: SYNTHETIC CONTROLS — Counterfactual County Twins
# "What would Lubbock look like without those charities?"
# ═══════════════════════════════════════════════════════════════════

class SyntheticControl:
    """Synthetic Control Method implementation.

    Finds "Mathematical Twins" for treated units and constructs
    counterfactual timelines.
    """

    def __init__(self):
        self.treated: str = ""
        self.donor_pool: list[str] = []
        self.weights: dict[str, float] = {}
        self.pre_treatment: dict = {}
        self.post_treatment: dict = {}

    def build_synthetic(
        self,
        treated_unit: str,
        donor_units: list[str],
        covariates: dict[str, dict],
        outcome_var: str = "voter_turnout",
    ) -> dict:
        """Find optimal donor weights to construct a synthetic control.

        covariates: {unit_name: {var1: value, var2: value, ...}}
        """
        self.treated = treated_unit
        self.donor_pool = donor_units

        treated_data = covariates.get(treated_unit, {})
        if not treated_data:
            return {"error": f"No data for treated unit: {treated_unit}"}

        # Compute similarity weights using inverse distance
        raw_weights = {}
        for donor in donor_units:
            donor_data = covariates.get(donor, {})
            if not donor_data:
                continue

            # Euclidean distance across all covariates
            distance = 0
            count = 0
            for var in treated_data:
                if var in donor_data and var != outcome_var:
                    t_val = float(treated_data[var])
                    d_val = float(donor_data[var])
                    # Normalize by treated value to handle scale differences
                    if abs(t_val) > 0.001:
                        distance += ((t_val - d_val) / abs(t_val)) ** 2
                    else:
                        distance += (t_val - d_val) ** 2
                    count += 1

            if count > 0:
                distance = math.sqrt(distance / count)
                raw_weights[donor] = 1.0 / max(distance, 0.001)

        # Normalize weights
        total = sum(raw_weights.values())
        self.weights = {
            d: round(w / total, 4)
            for d, w in sorted(raw_weights.items(), key=lambda x: -x[1])
        }

        # Construct synthetic outcome
        synthetic_outcome = sum(
            self.weights[d] * float(covariates.get(d, {}).get(outcome_var, 0))
            for d in self.weights
        )

        actual = float(treated_data.get(outcome_var, 0))
        gap = actual - synthetic_outcome

        return {
            "treated": treated_unit,
            "synthetic_outcome": round(synthetic_outcome, 4),
            "actual_outcome": round(actual, 4),
            "treatment_effect": round(gap, 4),
            "weights": dict(list(self.weights.items())[:10]),
            "top_donors": [
                {"unit": d, "weight": w}
                for d, w in list(self.weights.items())[:5]
            ],
            "donor_pool_size": len(self.weights),
        }

    def compute_placebo_tests(
        self,
        covariates: dict[str, dict],
        outcome_var: str = "voter_turnout",
    ) -> dict:
        """Run placebo tests: treat every donor as if it were the treated unit.

        If the treatment effect for the real treated unit is extreme
        relative to placebos, we have causal evidence.
        """
        # Real treatment effect
        real_result = self.build_synthetic(
            self.treated, self.donor_pool, covariates, outcome_var
        )
        real_effect = real_result.get("treatment_effect", 0)

        # Placebo effects
        placebo_effects = []
        for donor in self.donor_pool[:20]:  # Limit for speed
            remaining = [d for d in self.donor_pool if d != donor]
            remaining.append(self.treated)

            sc = SyntheticControl()
            result = sc.build_synthetic(donor, remaining, covariates, outcome_var)
            placebo_effects.append({
                "unit": donor,
                "effect": result.get("treatment_effect", 0),
            })

        # Compute p-value
        more_extreme = sum(
            1 for p in placebo_effects
            if abs(p["effect"]) >= abs(real_effect)
        )
        p_value = (more_extreme + 1) / (len(placebo_effects) + 1)

        return {
            "treated_unit": self.treated,
            "treatment_effect": round(real_effect, 4),
            "placebo_count": len(placebo_effects),
            "p_value": round(p_value, 4),
            "significant": p_value < 0.1,
            "rank": more_extreme + 1,
            "interpretation": (
                f"Effect rank: {more_extreme + 1}/{len(placebo_effects) + 1}. "
                f"{'Significant — causal evidence.' if p_value < 0.1 else 'Not significant — could be noise.'}"
            ),
            "placebo_effects": sorted(placebo_effects, key=lambda x: -abs(x["effect"]))[:5],
        }

    def counterfactual_timeline(
        self,
        treated_unit: str,
        time_series: dict[str, dict[str, float]],
        treatment_period: int,
    ) -> dict:
        """Build a counterfactual timeline.

        time_series: {unit: {period: outcome_value}}
        """
        treated_ts = time_series.get(treated_unit, {})
        if not treated_ts:
            return {"error": f"No time series for {treated_unit}"}

        # Get all donors
        donors = [u for u in time_series if u != treated_unit]

        # Use pre-treatment periods to fit weights
        pre_periods = [p for p in sorted(treated_ts.keys(), key=int) if int(p) < treatment_period]
        post_periods = [p for p in sorted(treated_ts.keys(), key=int) if int(p) >= treatment_period]

        # Build covariates from pre-period means
        covariates = {}
        for unit in [treated_unit] + donors:
            ts = time_series.get(unit, {})
            pre_values = [ts.get(p, 0) for p in pre_periods]
            covariates[unit] = {
                "pre_mean": statistics.mean(pre_values) if pre_values else 0,
                "pre_std": statistics.stdev(pre_values) if len(pre_values) > 1 else 0,
                "pre_trend": (pre_values[-1] - pre_values[0]) / max(len(pre_values), 1) if pre_values else 0,
            }

        # Build synthetic from pre-treatment fit
        sc = SyntheticControl()
        sc.build_synthetic(treated_unit, donors, covariates, "pre_mean")

        # Project synthetic into post-treatment
        actual_line = []
        synthetic_line = []
        gaps = []

        for p in sorted(treated_ts.keys(), key=int):
            actual_val = treated_ts.get(p, 0)
            synthetic_val = sum(
                sc.weights.get(d, 0) * time_series.get(d, {}).get(p, 0)
                for d in donors
            )
            gap = actual_val - synthetic_val

            actual_line.append({"period": int(p), "value": round(actual_val, 4)})
            synthetic_line.append({"period": int(p), "value": round(synthetic_val, 4)})
            if int(p) >= treatment_period:
                gaps.append(round(gap, 4))

        avg_gap = statistics.mean(gaps) if gaps else 0

        return {
            "treated_unit": treated_unit,
            "treatment_period": treatment_period,
            "actual_timeline": actual_line,
            "synthetic_timeline": synthetic_line,
            "post_treatment_gaps": gaps,
            "average_treatment_effect": round(avg_gap, 4),
            "weights": dict(list(sc.weights.items())[:5]),
        }


# ═══════════════════════════════════════════════════════════════════
# SIMULATION DECK — unified interface
# ═══════════════════════════════════════════════════════════════════

# Module-level singletons
_active_abm: Optional[AgentBasedModel] = None
_game_designer = GameTheoryDesigner()
_synthetic_control = SyntheticControl()


def create_simulation(
    sim_type: str,
    n_agents: int = 10000,
    county: str = "lubbock",
) -> dict:
    """Create a new simulation."""
    global _active_abm
    if sim_type == "abm":
        _active_abm = AgentBasedModel(n_agents, county)
        result = _active_abm.populate()
        return {"type": "abm", **result}
    return {"error": f"Unknown sim type: {sim_type}"}


def get_active_simulation() -> dict:
    """Get the current active simulation status."""
    if _active_abm:
        return {
            "type": "abm",
            "agents": _active_abm.n_agents,
            "month": _active_abm.month,
            "shocks": len(_active_abm.shocks),
            "current_state": _active_abm._compute_aggregate(),
        }
    return {"status": "no_active_simulation"}


# ═══════════════════════════════════════════════════════════════════
# SCENE 5: WAR GAME EXPORT — Research Memo to Bolt
# ═══════════════════════════════════════════════════════════════════

def export_war_game_memo(
    simulation_results: dict = None,
    game_theory_results: dict = None,
    description: str = "",
    model_chain: list[str] = None,
) -> dict:
    """Scene 5: 'You hit Export. A professional Research Memo is written to your Bolt.'

    Generates a full research memo from simulation + game theory results:
    1. LLM-written narrative with academic prose
    2. Simulation data (per-month snapshots, agent distributions)
    3. Nash equilibria and strategic analysis
    4. County-level predictions and policy implications
    5. Saved to ARTEFACTS/war_games/ on the Bolt

    Args:
        simulation_results: ABM simulation output (or uses active sim)
        game_theory_results: Nash equilibria etc (optional)
        description: Human-readable description of the war game
        model_chain: LLM models for memo generation

    Returns: dict with memo path, narrative text, and raw data reference.
    """
    import os
    import json
    from datetime import datetime

    model_chain = model_chain or [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]
    t0 = time.time()

    # Use active simulation if none provided
    if simulation_results is None and _active_abm:
        simulation_results = {
            "agents": _active_abm.n_agents,
            "county": _active_abm.county,
            "month": _active_abm.month,
            "shocks": _active_abm.shocks,
            "aggregate": _active_abm._compute_aggregate(),
            "snapshots": _active_abm.snapshots[-6:],  # Last 6 months
        }

    if not simulation_results:
        return {"error": "No simulation results to export"}

    # Build data summary for LLM
    data_summary = json.dumps(simulation_results, indent=2, default=str)[:3000]
    game_summary = ""
    if game_theory_results:
        game_summary = json.dumps(game_theory_results, indent=2, default=str)[:2000]

    # Generate narrative via LLM
    prompt = (
        f"Generate a 2-page Research Memo based on this policy simulation.\n\n"
        f"DESCRIPTION: {description or 'Policy shock simulation'}\n\n"
        f"SIMULATION DATA:\n{data_summary}\n\n"
    )
    if game_summary:
        prompt += f"GAME THEORY RESULTS:\n{game_summary}\n\n"

    prompt += (
        f"FORMAT:\n"
        f"# Research Memo: [Generated Title]\n"
        f"## Executive Summary (3-4 sentences)\n"
        f"## Simulation Design\n"
        f"- Agents, shocks applied, time horizon\n"
        f"## Key Findings\n"
        f"- 3-4 numbered findings with data support\n"
        f"## Strategic Implications\n"
        f"- Nash equilibria interpretation (if available)\n"
        f"- Predicted electoral consequences\n"
        f"## Policy Recommendations\n"
        f"- 2-3 actionable recommendations\n"
        f"## Methodological Notes\n"
        f"- Limitations and assumptions\n\n"
        f"Use academic prose with citations where appropriate. "
        f"Reference specific numbers from the simulation data."
    )

    narrative = "[Memo generation requires LLM access]"
    model_used = ""
    try:
        from server.backend_logic import generate_text_via_chain
        text, model_used = generate_text_via_chain(
            prompt, model_chain,
            system_instruction=(
                "You are E.D.I.T.H., writing a professional research memo "
                "for a PhD political scientist. Use precise language and "
                "reference simulation parameters explicitly."
            ),
            temperature=0.3,
        )
        narrative = text.strip()
    except Exception as e:
        log.error(f"§SIM: Memo generation failed: {e}")

    # Save to Bolt
    data_root = os.environ.get("EDITH_DATA_ROOT", ".")
    memo_dir = os.path.join(data_root, "ARTEFACTS", "war_games")
    os.makedirs(memo_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    memo_path = os.path.join(memo_dir, f"memo_{timestamp}.json")

    memo_package = {
        "generated": datetime.now().isoformat(),
        "description": description,
        "narrative": narrative,
        "model": model_used,
        "simulation": simulation_results,
        "game_theory": game_theory_results or {},
        "export_metadata": {
            "source": "simulation_deck.export_war_game_memo",
            "version": "1.0",
        },
    }

    try:
        with open(memo_path, "w") as f:
            json.dump(memo_package, f, indent=2, default=str)
        log.info(f"§SIM: War game memo exported → {memo_path}")
    except Exception as e:
        log.error(f"§SIM: Failed to save memo: {e}")
        memo_path = ""

    elapsed = time.time() - t0
    return {
        "status": "exported",
        "path": memo_path,
        "narrative_length": len(narrative),
        "has_game_theory": bool(game_theory_results),
        "model": model_used,
        "elapsed_s": round(elapsed, 2),
        "narrative_preview": narrative[:500] + "..." if len(narrative) > 500 else narrative,
    }
