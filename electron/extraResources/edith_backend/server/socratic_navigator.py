"""
Socratic Navigator — Adversarial Intellectual Training
========================================================
Winnie is not "too helpful." She is intellectually adversarial.

Instead of summarizing a paper, she:
1. Identifies the 3 weakest points in the author's argument
2. Cross-references against your Potter County data and Ancestral Knowledge
3. Forces you to defend your logic before it becomes a "Verified Link"
4. Launches methodology sandboxes so you master the math, not just the definition
5. Runs "Committee of Sages" mode — Mettler, Aldrich, Kim debate in parallel

Five Pedagogical Engines:
  A. Socratic Needle — Active interrogation that finds contradictions
  B. Methodology Sandbox — Toy datasets + real-time coefficient exploration
  C. Ancestral Pedagogics — Teaches through your 2023 exams / Arkansas notes
  D. Ontology Mapper — Shows where your dissertation lives in the gap
  E. Committee of Sages — Three parallel debate personas
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.socratic_navigator")


# ═══════════════════════════════════════════════════════════════════
# Dissertation Ontology — Abby's Intellectual Map
# ═══════════════════════════════════════════════════════════════════

DISSERTATION_MAP = {
    "core_argument": (
        "Charitable organizations in rural Texas substitute for state welfare "
        "functions, diffusing political blame and reducing state capacity pressure."
    ),
    "chapters": {
        1: {"title": "Introduction: The Puzzle of Charity Substitution",
            "core_claim": "When charities fill welfare gaps, voters misattribute service provision"},
        2: {"title": "Theory: Administrative Burden, Blame Diffusion, and the Submerged State",
            "core_claim": "Charities create an 'accountability blur' between state and non-state actors"},
        3: {"title": "State Capacity and Service Delivery in Potter County",
            "core_claim": "Local state capacity gaps predict nonprofit density"},
        4: {"title": "Charity as Governance: The Substitution Mechanism",
            "core_claim": "Charities provide quasi-governmental services, reducing political accountability"},
        5: {"title": "Data, Methods, and the Lubbock Case",
            "core_claim": "Mixed-methods design using DiD/IV on local admin data + field interviews"},
        6: {"title": "Findings: Blame Without a Name",
            "core_claim": "Charity density correlates with lower electoral accountability scores"},
        7: {"title": "Conclusion: Implications for Democratic Governance",
            "core_claim": "The submerged charity state weakens democratic feedback loops"},
    },
    "key_scholars": {
        "Mettler": {
            "framework": "Submerged State",
            "key_claim": "Government policies are hidden from public view, reducing democratic engagement",
            "weak_points": [
                "Focuses on tax expenditures, not direct service substitution",
                "Does not account for rural/urban capacity differences",
                "Limited engagement with non-state providers as 'submerging' agents",
            ],
        },
        "Aldrich": {
            "framework": "Institutional Design and Democratic Accountability",
            "key_claim": "Institutional structures determine patterns of accountability",
            "weak_points": [
                "Largely theoretical — limited empirical testing in welfare contexts",
                "Assumes rational voters with full information",
                "Does not address the 'charity blur' in accountability chains",
            ],
        },
        "Moynihan": {
            "framework": "Administrative Burden Theory",
            "key_claim": "Learning, compliance, and psychological costs reduce welfare take-up",
            "weak_points": [
                "Focuses on federal programs — unclear applicability to local charities",
                "Does not address whether charities reduce or increase burden",
                "Limited theorization of burden as a political strategy",
            ],
        },
        "Lipsky": {
            "framework": "Street-Level Bureaucracy",
            "key_claim": "Frontline workers shape policy through discretionary decisions",
            "weak_points": [
                "Written in 1980 — does not account for nonprofit service delivery",
                "Assumes workers are state employees, not charity volunteers",
                "Limited engagement with technology's impact on discretion",
            ],
        },
    },
    "ancestral_knowledge": {
        "2024_comp_exams": {
            "topics": ["institutional friction", "path dependence", "welfare state theory",
                       "federalism", "administrative burden"],
            "key_insight": "Institutional friction is the mechanism through which charities 'absorb' blame",
        },
        "2023_seminars": {
            "topics": ["principal-agent problems", "moral hazard", "adverse selection",
                       "delegation theory", "political accountability"],
            "key_insight": "Charity substitution creates a classic principal-agent problem between voters and the state",
        },
        "arkansas_undergrad": {
            "topics": ["rural governance", "social capital", "community organizations",
                       "Southern politics", "food insecurity"],
            "key_insight": "Rural communities rely on informal networks that formal data misses",
        },
    },
}


# ═══════════════════════════════════════════════════════════════════
# Socratic Needle — Active Interrogation Engine
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SocraticChallenge:
    """A challenge that forces the student to defend their logic."""
    challenge_id: str
    challenge_type: str  # "contradiction", "mechanism", "evidence", "scope", "assumption"
    question: str
    context: str
    scholar_reference: str
    difficulty: str  # "warm-up", "seminar", "defense"
    related_chapter: int
    user_response: str = ""
    resolved: bool = False

    def to_dict(self) -> dict:
        return {
            "id": self.challenge_id,
            "type": self.challenge_type,
            "question": self.question,
            "context": self.context[:200],
            "difficulty": self.difficulty,
            "chapter": self.related_chapter,
            "resolved": self.resolved,
        }


class SocraticNeedle:
    """The adversarial interrogation engine.

    "She identifies the three weakest points in the author's argument
    and asks you to reconcile them."
    """

    def __init__(self):
        self._challenges: list[SocraticChallenge] = []
        self._resolved: list[SocraticChallenge] = []

    def interrogate(self, text: str, source: str = "") -> list[SocraticChallenge]:
        """Generate Socratic challenges from a reading passage.

        Returns 3–5 challenges that force the student to think critically.
        """
        text_lower = text.lower()
        challenges = []

        # 1. Find assumption challenges
        assumption_signals = [
            (r"assum\w+", "What evidence supports this assumption? Is it testable?"),
            (r"given that", "Is this 'given' actually established, or are you assuming it?"),
            (r"it follows that", "Does it necessarily follow? What are the alternative explanations?"),
        ]
        for pattern, template in assumption_signals:
            if re.search(pattern, text_lower):
                # Find the sentence containing the assumption
                for sent in re.split(r'(?<=[.!?])\s+', text):
                    if re.search(pattern, sent.lower()):
                        challenges.append(SocraticChallenge(
                            challenge_id=f"soc_{hashlib.md5(sent[:50].encode()).hexdigest()[:8]}",
                            challenge_type="assumption",
                            question=template,
                            context=sent.strip()[:200],
                            scholar_reference=self._find_relevant_scholar(sent),
                            difficulty="seminar",
                            related_chapter=self._match_chapter(sent),
                        ))
                        break

        # 2. Find causal claim challenges
        causal_signals = ["causes", "leads to", "results in", "produces", "reduces",
                          "increases", "affects", "determines"]
        for signal in causal_signals:
            if signal in text_lower:
                for sent in re.split(r'(?<=[.!?])\s+', text):
                    if signal in sent.lower():
                        mechanism_q = (
                            f"You claim that something '{signal}' something else. "
                            f"What is the specific institutional mechanism? "
                            f"Can you diagram the causal chain?"
                        )
                        challenges.append(SocraticChallenge(
                            challenge_id=f"soc_{hashlib.md5(sent[:50].encode()).hexdigest()[:8]}",
                            challenge_type="mechanism",
                            question=mechanism_q,
                            context=sent.strip()[:200],
                            scholar_reference=self._find_relevant_scholar(sent),
                            difficulty="defense",
                            related_chapter=self._match_chapter(sent),
                        ))
                        break

        # 3. Cross-reference with Ancestral Knowledge
        for period, knowledge in DISSERTATION_MAP["ancestral_knowledge"].items():
            for topic in knowledge["topics"]:
                if topic in text_lower:
                    challenges.append(SocraticChallenge(
                        challenge_id=f"soc_anc_{topic[:10]}",
                        challenge_type="contradiction",
                        question=(
                            f"Your {period.replace('_', ' ')} work on '{topic}' claimed: "
                            f"'{knowledge['key_insight']}'. "
                            f"Does this new reading support or challenge that position? Be specific."
                        ),
                        context=f"Cross-reference with {period}",
                        scholar_reference=period,
                        difficulty="seminar",
                        related_chapter=self._match_chapter(topic),
                    ))

        # 4. Scholar-specific challenges
        for scholar, data in DISSERTATION_MAP["key_scholars"].items():
            if scholar.lower() in text_lower:
                weak_point = data["weak_points"][0]
                challenges.append(SocraticChallenge(
                    challenge_id=f"soc_sch_{scholar[:6]}",
                    challenge_type="evidence",
                    question=(
                        f"{scholar}'s '{data['framework']}' framework has a known weakness: "
                        f"'{weak_point}'. Does this reading address that weakness, "
                        f"or does it inherit the same blind spot?"
                    ),
                    context=f"Scholar critique: {scholar}",
                    scholar_reference=scholar,
                    difficulty="defense",
                    related_chapter=2,
                ))

        # Deduplicate and limit
        seen_types = set()
        unique = []
        for c in challenges:
            key = f"{c.challenge_type}_{c.related_chapter}"
            if key not in seen_types:
                seen_types.add(key)
                unique.append(c)

        self._challenges.extend(unique[:5])
        return unique[:5]

    def resolve_challenge(self, challenge_id: str, response: str) -> dict:
        """Student resolves a challenge with their defense."""
        for c in self._challenges:
            if c.challenge_id == challenge_id:
                c.user_response = response
                c.resolved = True
                self._resolved.append(c)
                self._challenges.remove(c)
                return {
                    "resolved": True,
                    "challenge": c.challenge_type,
                    "verdict": self._evaluate_response(c, response),
                }
        return {"resolved": False, "error": "Challenge not found"}

    def _evaluate_response(self, challenge: SocraticChallenge, response: str) -> dict:
        """Evaluate whether the student's response is adequate."""
        resp_lower = response.lower()
        score = 0.5  # Start neutral

        # Check for mechanism language
        if any(w in resp_lower for w in ["mechanism", "because", "through", "via", "process"]):
            score += 0.15

        # Check for evidence citations
        if any(w in resp_lower for w in ["data", "evidence", "regression", "table", "figure"]):
            score += 0.15

        # Check for nuance
        if any(w in resp_lower for w in ["however", "although", "but", "limitation", "caveat"]):
            score += 0.1

        # Penalize vagueness
        if any(w in resp_lower for w in ["clearly", "obviously", "everyone knows"]):
            score -= 0.1

        score = max(0.0, min(1.0, score))

        if score >= 0.7:
            verdict = "✓ VERIFIED — This is now a confirmed theoretical link."
        elif score >= 0.5:
            verdict = "⚠ PARTIAL — You need stronger evidence or a clearer mechanism."
        else:
            verdict = "✗ INSUFFICIENT — Revisit the literature and try again."

        return {"score": round(score, 2), "verdict": verdict}

    def _find_relevant_scholar(self, text: str) -> str:
        text_lower = text.lower()
        for scholar in DISSERTATION_MAP["key_scholars"]:
            if scholar.lower() in text_lower:
                return scholar
        return ""

    def _match_chapter(self, text: str) -> int:
        text_lower = text.lower()
        chapter_signals = {
            1: ["research question", "puzzle", "introduction"],
            2: ["theory", "framework", "mechanism", "burden"],
            3: ["state capacity", "potter county", "service delivery"],
            4: ["charity", "nonprofit", "substitution", "governance"],
            5: ["data", "method", "regression", "variable", "lubbock"],
            6: ["finding", "result", "coefficient", "evidence"],
            7: ["implication", "conclusion", "future"],
        }
        best_ch, best_hits = 2, 0  # Default to theory chapter
        for ch, signals in chapter_signals.items():
            hits = sum(1 for s in signals if s in text_lower)
            if hits > best_hits:
                best_ch, best_hits = ch, hits
        return best_ch

    @property
    def status(self) -> dict:
        return {
            "active_challenges": len(self._challenges),
            "resolved": len(self._resolved),
        }


# ═══════════════════════════════════════════════════════════════════
# Methodology Sandbox — Applied Learning Engine
# ═══════════════════════════════════════════════════════════════════

class MethodologySandbox:
    """Real-time methodology exploration with toy datasets.

    "She pulls a toy dataset and says, 'If we shift the bandwidth
    by 5%, watch how the significance disappears.'"
    """

    SANDBOX_TEMPLATES = {
        "did": {
            "title": "Difference-in-Differences Sandbox",
            "instruction": (
                "Look at the screen. I've plotted the treatment and control groups "
                "over time. The parallel trends assumption holds pre-treatment. "
                "Now watch: if I add a pre-existing time trend to the treatment group, "
                "the DiD estimate becomes biased upward. This is why you must always "
                "check for differential pre-trends."
            ),
            "stata_code": (
                "* DiD Sandbox: Charity Density and Voter Turnout\n"
                "* Generate the data\n"
                "clear\nset obs 200\ngen county = mod(_n-1, 10) + 1\n"
                "gen post = _n > 100\ngen treatment = county <= 5\n"
                "gen y = 50 + 5*treatment + 10*post + 7*treatment*post + rnormal(0,3)\n"
                "* The DiD Estimator\n"
                "reg y treatment##post, cluster(county)\n"
                "* Check: Does adding county FE change the estimate?\n"
                "xtreg y treatment##post, fe cluster(county)\n"
            ),
            "key_question": (
                "The treatment*post coefficient is your causal estimate. "
                "But what happens if you add county-specific time trends? "
                "Try: `gen county_trend = county * post` and re-run."
            ),
        },
        "rdd": {
            "title": "Regression Discontinuity Sandbox",
            "instruction": (
                "I've created a sharp RDD at the poverty threshold. "
                "Notice how the jump at the cutoff disappears as you widen "
                "the bandwidth. This is the bias-variance tradeoff in RDD."
            ),
            "stata_code": (
                "* RDD Sandbox: Program Eligibility and Charity Use\n"
                "clear\nset obs 500\ngen income = runiform(0, 100)\n"
                "gen eligible = income < 50\n"
                "gen charity_use = 30 - 0.3*income + 15*eligible + rnormal(0,5)\n"
                "* Sharp RDD\n"
                "rd charity_use income, z0(50) bwidth(10)\n"
                "* Sensitivity: Try bwidth(5) and bwidth(20)\n"
            ),
            "key_question": (
                "What happens to the local average treatment effect "
                "when you change the bandwidth from 10 to 5 to 20? "
                "Why does this matter for your Potter County analysis?"
            ),
        },
        "iv": {
            "title": "Instrumental Variables Sandbox",
            "instruction": (
                "I've simulated an endogeneity problem. OLS is biased because "
                "charity_density is correlated with unobserved 'community need.' "
                "The instrument (distance_to_church) corrects this."
            ),
            "stata_code": (
                "* IV Sandbox: Charity Density and Political Accountability\n"
                "clear\nset obs 300\ngen need = rnormal(0,1)\n"
                "gen distance = runiform(0, 50)\n"
                "gen charity_density = 20 - 0.5*distance + 2*need + rnormal(0,2)\n"
                "gen accountability = 70 - 0.8*charity_density + 3*need + rnormal(0,5)\n"
                "* Biased OLS\nreg accountability charity_density\n"
                "* IV Correction\nivregress 2sls accountability (charity_density = distance)\n"
                "* First-stage F: Check instrument strength\n"
                "estat firststage\n"
            ),
            "key_question": (
                "Compare the OLS and IV coefficients. The OLS is biased "
                "toward zero — why? And is distance_to_church a valid "
                "instrument? What's the exclusion restriction?"
            ),
        },
    }

    def launch_sandbox(self, method: str) -> dict:
        """Launch a methodology sandbox."""
        method_lower = method.lower().replace("-", "").replace(" ", "")

        # Map common names
        name_map = {
            "did": "did", "differenceindifferences": "did",
            "rdd": "rdd", "regressiondiscontinuity": "rdd",
            "iv": "iv", "instrumentalvariable": "iv", "2sls": "iv",
        }
        method_key = name_map.get(method_lower, method_lower)

        template = self.SANDBOX_TEMPLATES.get(method_key)
        if not template:
            return {
                "error": f"No sandbox for '{method}'",
                "available": list(self.SANDBOX_TEMPLATES.keys()),
            }

        return {
            "sandbox": method_key,
            "title": template["title"],
            "instruction": template["instruction"],
            "stata_code": template["stata_code"],
            "key_question": template["key_question"],
            "launched": True,
        }


# ═══════════════════════════════════════════════════════════════════
# Committee of Sages — Multi-Persona Debate Engine
# ═══════════════════════════════════════════════════════════════════

SAGE_PERSONAS = {
    "mettler": {
        "name": "Suzanne Mettler",
        "framework": "Submerged State",
        "style": "methodical, evidence-driven, focused on visibility of government",
        "opening": "Let me push back on this from the perspective of policy visibility...",
        "critique_lens": [
            "Is the government's role sufficiently visible in this analysis?",
            "Are you accounting for the 'submersion' of policy benefits?",
            "How do citizens perceive the source of these services?",
        ],
    },
    "aldrich": {
        "name": "John Aldrich",
        "framework": "Institutional Design and Party Politics",
        "style": "rigorous, institutionalist, challenges descriptive claims",
        "opening": "Your argument is descriptive. What is the specific institutional mechanism?",
        "critique_lens": [
            "What institutional incentives drive this behavior?",
            "Is this a rational response to institutional constraints?",
            "How does party competition factor into your model?",
        ],
    },
    "kim": {
        "name": "Soo Yeon Kim",
        "framework": "International Political Economy",
        "style": "quantitative, comparative, pushes for generalizability",
        "opening": "Interesting case study. But does this generalize beyond Potter County?",
        "critique_lens": [
            "What is the external validity of this finding?",
            "Have you considered the comparative cases?",
            "Is your identification strategy robust to alternative specifications?",
        ],
    },
}


class CommitteeOfSages:
    """Three parallel debate personas that stress-test your arguments.

    "They debate in your spatial audio. You listen until you understand
    the nuance well enough to interject and settle the debate."
    """

    def __init__(self):
        self._debate_history: list[dict] = []

    def convene(self, topic: str, student_claim: str = "") -> dict:
        """Convene the Committee of Sages on a topic."""
        t0 = time.time()
        responses = {}

        for sage_id, persona in SAGE_PERSONAS.items():
            critique = self._generate_sage_response(persona, topic, student_claim)
            responses[sage_id] = critique

        # Find consensus and disagreement
        all_questions = []
        for sage_id, resp in responses.items():
            all_questions.extend(resp.get("questions", []))

        unique_concerns = list(set(all_questions))

        debate = {
            "topic": topic[:200],
            "student_claim": student_claim[:200] if student_claim else "",
            "sages": responses,
            "consensus_concerns": unique_concerns[:5],
            "debate_verdict": self._render_verdict(responses),
            "elapsed_ms": round((time.time() - t0) * 1000, 1),
        }

        self._debate_history.append(debate)
        return debate

    def _generate_sage_response(self, persona: dict,
                                   topic: str, claim: str) -> dict:
        """Generate a sage's critique of a topic/claim."""
        topic_lower = topic.lower()

        # Select relevant critique lenses
        relevant_critiques = []
        for lens in persona["critique_lens"]:
            relevant_critiques.append(lens)

        # Find specific weaknesses based on topic
        concerns = []
        name = persona["name"]

        if "charity" in topic_lower or "nonprofit" in topic_lower:
            concerns.append(f"{name}: Are these organizations truly 'substituting' for the state, or are they complementary?")
        if "blame" in topic_lower or "accountability" in topic_lower:
            concerns.append(f"{name}: Can you operationalize 'blame diffusion'? What would you measure?")
        if "state capacity" in topic_lower:
            concerns.append(f"{name}: How do you disentangle 'low capacity' from 'strategic withdrawal'?")
        if "cartel" in topic_lower or "mexico" in topic_lower:
            concerns.append(f"{name}: Interesting comparative case. But are cartels and food banks really comparable governance actors?")

        return {
            "sage": name,
            "framework": persona["framework"],
            "opening": persona["opening"],
            "style": persona["style"],
            "questions": relevant_critiques[:2],
            "specific_concerns": concerns[:2],
        }

    def _render_verdict(self, responses: dict) -> str:
        """Render a committee verdict."""
        total_concerns = sum(
            len(r.get("specific_concerns", []))
            for r in responses.values()
        )

        if total_concerns == 0:
            return "PASS — The committee has no immediate objections. Proceed with caution."
        elif total_concerns <= 3:
            return f"CONDITIONAL — {total_concerns} concerns raised. Address before proceeding."
        return f"REVISE — {total_concerns} concerns. Significant revision needed."

    @property
    def status(self) -> dict:
        return {
            "debates_held": len(self._debate_history),
            "sages": list(SAGE_PERSONAS.keys()),
        }


# ═══════════════════════════════════════════════════════════════════
# Ontology Mapper — "Where Does Your Dissertation Live?"
# ═══════════════════════════════════════════════════════════════════

class OntologyMapper:
    """Maps readings to the theoretical gap where your dissertation lives.

    "She highlights a node: 'Do you see the gap between Welfare Politics
    and Non-State Provision? That gap is where your dissertation lives.'"
    """

    SUBFIELD_CLUSTERS = {
        "welfare_politics": {
            "core_scholars": ["Pierson", "Esping-Andersen", "Hacker", "Mettler", "Campbell"],
            "key_concepts": ["welfare state", "social policy", "redistribution", "retrenchment"],
        },
        "non_state_provision": {
            "core_scholars": ["Salamon", "Smith", "Lipsky", "Boris"],
            "key_concepts": ["nonprofit", "third sector", "voluntary", "philanthropy", "charity"],
        },
        "latino_politics": {
            "core_scholars": ["Fraga", "Garcia Bedolla", "Barreto", "Pantoja"],
            "key_concepts": ["Latino voters", "immigration", "descriptive representation"],
        },
        "state_capacity": {
            "core_scholars": ["Fukuyama", "Mann", "Tilly", "Besley", "Persson"],
            "key_concepts": ["state building", "extractive", "infrastructural power", "capacity"],
        },
        "criminal_governance": {
            "core_scholars": ["Lessing", "Arias", "Durán-Martínez", "Magaloni"],
            "key_concepts": ["cartel", "narco", "criminal governance", "protection racket"],
        },
    }

    def locate_in_ontology(self, text: str) -> dict:
        """Map a reading to the theoretical space."""
        text_lower = text.lower()
        cluster_scores = {}

        for cluster_name, cluster_data in self.SUBFIELD_CLUSTERS.items():
            score = 0
            # Scholar mentions
            for scholar in cluster_data["core_scholars"]:
                if scholar.lower() in text_lower:
                    score += 2
            # Concept mentions
            for concept in cluster_data["key_concepts"]:
                if concept in text_lower:
                    score += 1

            if score > 0:
                cluster_scores[cluster_name] = score

        if not cluster_scores:
            return {"location": "unmapped", "message": "This reading doesn't clearly map to known clusters."}

        # Sort by relevance
        ranked = sorted(cluster_scores.items(), key=lambda x: -x[1])
        primary = ranked[0][0]
        secondary = ranked[1][0] if len(ranked) > 1 else None

        # Find the gap
        gap_analysis = None
        if primary and secondary:
            gap_analysis = (
                f"This reading bridges '{primary.replace('_', ' ')}' and "
                f"'{secondary.replace('_', ' ')}'. Your dissertation lives "
                f"in exactly this gap. Pay attention to the mechanism that "
                f"connects these two subfields."
            )
        elif primary == "welfare_politics":
            gap_analysis = (
                "This is core welfare politics. Your contribution is showing how "
                "this applies when nonprofits — not the state — deliver the services."
            )
        elif primary == "non_state_provision":
            gap_analysis = (
                "Core nonprofit literature. Your contribution is the political "
                "accountability angle: what happens to democratic feedback when "
                "charities replace the state?"
            )

        return {
            "primary_cluster": primary,
            "secondary_cluster": secondary,
            "cluster_scores": cluster_scores,
            "gap_analysis": gap_analysis,
            "dissertation_relevance": "high" if len(ranked) >= 2 else "moderate",
        }


# ═══════════════════════════════════════════════════════════════════
# The Socratic Navigator — Master Controller
# ═══════════════════════════════════════════════════════════════════

class SocraticNavigator:
    """The full Socratic instruction engine.

    Usage:
        nav = SocraticNavigator()

        # Process a reading through the Socratic gauntlet
        result = nav.process_reading("text of the reading", source="Mettler 2011")

        # Launch a methodology sandbox
        sandbox = nav.sandbox("DiD")

        # Convene the Committee
        debate = nav.convene_committee("charity substitution and accountability")

        # Resolve a Socratic challenge
        nav.resolve("challenge_id", "My defense is...")
    """

    def __init__(self):
        self.needle = SocraticNeedle()
        self.sandbox = MethodologySandbox()
        self.committee = CommitteeOfSages()
        self.ontology = OntologyMapper()

    def process_reading(self, text: str, source: str = "") -> dict:
        """Process a reading through the full Socratic pipeline.

        1. Map it in the ontology
        2. Generate Socratic challenges
        3. Check for methodology that needs a sandbox
        4. Return the complete learning plan
        """
        result = {
            "source": source,
            "length": len(text.split()),
        }

        # 1. Ontology mapping
        result["ontology"] = self.ontology.locate_in_ontology(text)

        # 2. Socratic challenges
        challenges = self.needle.interrogate(text, source)
        result["challenges"] = [c.to_dict() for c in challenges]
        result["challenge_count"] = len(challenges)

        # 3. Methodology detection → sandbox launch
        method_signals = {
            "did": ["difference-in-difference", "diff-in-diff", "DiD", "parallel trends"],
            "rdd": ["regression discontinuity", "RDD", "cutoff", "bandwidth"],
            "iv": ["instrumental variable", "IV", "2SLS", "endogenous"],
        }
        methods_found = []
        text_lower = text.lower()
        for method, signals in method_signals.items():
            if any(s.lower() in text_lower for s in signals):
                methods_found.append(method)
                result.setdefault("sandboxes", []).append(
                    self.sandbox.launch_sandbox(method)
                )

        result["methods_detected"] = methods_found

        # 4. Key scholars detected
        scholars_found = []
        for scholar in DISSERTATION_MAP["key_scholars"]:
            if scholar.lower() in text_lower:
                scholars_found.append(scholar)
        result["scholars_detected"] = scholars_found

        # 5. Chapter relevance
        chapter_hits = {}
        for ch_num, ch_data in DISSERTATION_MAP["chapters"].items():
            title_words = set(ch_data["title"].lower().split())
            hits = len(title_words & set(text_lower.split()))
            if hits > 2:
                chapter_hits[ch_num] = ch_data["title"]
        result["relevant_chapters"] = chapter_hits

        return result

    def launch_sandbox(self, method: str) -> dict:
        """Launch a methodology sandbox."""
        return self.sandbox.launch_sandbox(method)

    def convene_committee(self, topic: str, claim: str = "") -> dict:
        """Convene the Committee of Sages."""
        return self.committee.convene(topic, claim)

    def resolve(self, challenge_id: str, response: str) -> dict:
        """Resolve a Socratic challenge with your defense."""
        return self.needle.resolve_challenge(challenge_id, response)

    @property
    def status(self) -> dict:
        return {
            "needle": self.needle.status,
            "committee": self.committee.status,
        }


# Global instance
socratic_navigator = SocraticNavigator()
