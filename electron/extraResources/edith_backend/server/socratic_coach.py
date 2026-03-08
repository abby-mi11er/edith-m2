"""
Adversarial Socratic Coach — The 24/7 R1-Level Mentor
=======================================================
Pedagogical Mirror Feature 1: Active Epistemology.

Winnie NEVER gives you the "easy" answer. Instead, she:
1. Gives you a Partial Truth — enough to orient, not enough to satisfy
2. Asks a Socratic question that forces you to find the flaw
3. Escalates through Bloom's Taxonomy until you reach "Creating"
4. Tracks your cognitive growth over time

The Coach doesn't care about your data.
She cares about your THEORETICAL RIGOR.

Architecture:
    User Query → Bloom Level Assessment → Partial Truth Generator →
    Socratic Question Engine → Cognitive Growth Tracker

This mimics the Oral Defense / Comprehensive Exam environment.
Every day with Winnie is a dress rehearsal for the VIVA.
"""

# ⚠️ SUPERSEDED BY: server/socratic_navigator.py
# socratic_navigator.py is the newer, richer implementation with 5 engines.
# This module is kept for any legacy imports.


import hashlib
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.socratic_coach")


# ═══════════════════════════════════════════════════════════════════
# Bloom's Taxonomy — The Cognitive Ladder
# ═══════════════════════════════════════════════════════════════════

BLOOM_LEVELS = {
    1: {
        "name": "Remember",
        "description": "Recall facts and basic concepts",
        "verbs": ["define", "list", "recall", "identify", "name", "describe"],
        "challenge": "Can you recall the core definition without looking it up?",
        "coaching_tone": "foundational",
    },
    2: {
        "name": "Understand",
        "description": "Explain ideas or concepts in your own words",
        "verbs": ["explain", "summarize", "paraphrase", "classify", "discuss"],
        "challenge": "Now explain this to a first-year who has never seen it.",
        "coaching_tone": "clarifying",
    },
    3: {
        "name": "Apply",
        "description": "Use information in new situations",
        "verbs": ["apply", "demonstrate", "solve", "use", "implement", "execute"],
        "challenge": "How would you apply this in a different institutional context?",
        "coaching_tone": "practical",
    },
    4: {
        "name": "Analyze",
        "description": "Draw connections among ideas",
        "verbs": ["analyze", "differentiate", "examine", "compare", "contrast", "deconstruct"],
        "challenge": "What assumption is this framework hiding from you?",
        "coaching_tone": "probing",
    },
    5: {
        "name": "Evaluate",
        "description": "Justify a stand or decision",
        "verbs": ["evaluate", "argue", "defend", "judge", "critique", "assess"],
        "challenge": "A reviewer says this is methodologically flawed. Defend it.",
        "coaching_tone": "adversarial",
    },
    6: {
        "name": "Create",
        "description": "Produce new or original work",
        "verbs": ["create", "design", "construct", "develop", "formulate", "propose", "synthesize"],
        "challenge": "Propose a theoretical mechanism that nobody has articulated before.",
        "coaching_tone": "generative",
    },
}


# ═══════════════════════════════════════════════════════════════════
# Socratic Question Templates — The Art of the Follow-Up
# ═══════════════════════════════════════════════════════════════════

SOCRATIC_TEMPLATES = {
    "clarification": [
        "What do you mean by '{term}'? How does that definition differ from {author}'s?",
        "Can you say that more precisely? What's the boundary condition of that claim?",
        "You used the word '{term}.' In Mettler's framework, that means something different. Which do you mean?",
    ],
    "assumption_probe": [
        "What are you assuming about the causal pathway here?",
        "Is there a hidden variable you're not accounting for? What if {confounder} mediates this?",
        "Your argument assumes {assumption}. What if that doesn't hold in {context}?",
    ],
    "evidence_challenge": [
        "What evidence would DISPROVE this claim? Can you articulate the null?",
        "You cite {author}. But {counter_author} found the opposite. How do you reconcile this?",
        "If this is true, what should we observe in the data? And do we?",
    ],
    "perspective_shift": [
        "You're looking at this from a {paradigm} lens. What does it look like from {alt_paradigm}?",
        "A rational choice theorist would argue the opposite. How do you respond?",
        "Let's steelman the critique. What's the BEST argument against your position?",
    ],
    "implication_probe": [
        "If your theory is correct, what are the downstream implications for {topic}?",
        "Who would disagree with this conclusion, and why would they be right to?",
        "Your argument leads to a troubling implication. Do you see it?",
    ],
    "method_challenge": [
        "Why this method and not {alternative_method}? Justify the epistemological choice.",
        "Your design assumes {assumption}. That's a strong assumption. Defend it.",
        "If you can't randomize, how do you identify the causal effect? Walk me through the backdoor criterion.",
    ],
}


# ═══════════════════════════════════════════════════════════════════
# Cognitive Growth Tracker — Your Bloom's Level Over Time
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CognitiveProfile:
    """Track a student's cognitive growth across Bloom's levels."""
    sessions: int = 0
    bloom_distribution: dict = field(default_factory=lambda: {
        1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0
    })
    topics_mastered: list[str] = field(default_factory=list)
    topics_struggling: list[str] = field(default_factory=list)
    defense_readiness: float = 0.0  # 0-1
    last_session: str = ""
    streak_days: int = 0

    @property
    def dominant_level(self) -> int:
        if not any(self.bloom_distribution.values()):
            return 1
        return max(self.bloom_distribution, key=self.bloom_distribution.get)

    @property
    def growth_score(self) -> float:
        """0-100 score reflecting weighted Bloom's distribution."""
        total = sum(self.bloom_distribution.values()) or 1
        weighted = sum(
            level * count / total
            for level, count in self.bloom_distribution.items()
        )
        return round((weighted / 6) * 100, 1)

    def to_dict(self) -> dict:
        return {
            "sessions": self.sessions,
            "bloom_distribution": self.bloom_distribution,
            "dominant_level": self.dominant_level,
            "dominant_level_name": BLOOM_LEVELS[self.dominant_level]["name"],
            "growth_score": self.growth_score,
            "defense_readiness": round(self.defense_readiness, 2),
            "topics_mastered": self.topics_mastered[-10:],
            "topics_struggling": self.topics_struggling[-5:],
            "streak_days": self.streak_days,
        }


# ═══════════════════════════════════════════════════════════════════
# Socratic Engine — The Core Logic
# ═══════════════════════════════════════════════════════════════════

class SocraticCoach:
    """The Adversarial Socratic Coach — never gives the easy answer.

    Modes:
    - "socratic": Default — Partial truths + forcing questions
    - "defense": Oral defense simulation — aggressive cross-examination
    - "mentor": Gentler — scaffolded learning with encouragement
    - "comps": Comprehensive exam mode — tests breadth and depth
    """

    def __init__(self, data_root: str = ""):
        self._data_root = data_root or os.environ.get("EDITH_APP_DATA_DIR", "")
        self._profile = CognitiveProfile()
        self._mode = "socratic"
        self._current_bloom = 1
        self._conversation_depth = 0
        self._load_profile()

    def _profile_path(self) -> Path:
        return Path(self._data_root or ".") / "VAULT" / "PEDAGOGY" / "cognitive_profile.json"

    def _load_profile(self):
        path = self._profile_path()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                self._profile = CognitiveProfile(
                    sessions=data.get("sessions", 0),
                    bloom_distribution=data.get("bloom_distribution", {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}),
                    topics_mastered=data.get("topics_mastered", []),
                    topics_struggling=data.get("topics_struggling", []),
                    defense_readiness=data.get("defense_readiness", 0.0),
                    last_session=data.get("last_session", ""),
                    streak_days=data.get("streak_days", 0),
                )
            except Exception:
                pass

    def _save_profile(self):
        path = self._profile_path()
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self._profile.to_dict(), indent=2))
        except Exception as e:
            log.warning(f"Failed to save cognitive profile: {e}")

    def set_mode(self, mode: str) -> dict:
        """Switch coaching mode."""
        valid = {"socratic", "defense", "mentor", "comps"}
        if mode not in valid:
            return {"error": f"Invalid mode. Choose from: {valid}"}
        self._mode = mode
        return {"mode": self._mode, "description": self._mode_description()}

    def _mode_description(self) -> str:
        descriptions = {
            "socratic": "Partial truths + forcing questions. You'll work for every answer.",
            "defense": "Oral defense simulation. I will cross-examine your claims.",
            "mentor": "Scaffolded learning with encouragement, but still no easy answers.",
            "comps": "Comprehensive exam mode. Breadth AND depth. Show me everything.",
        }
        return descriptions.get(self._mode, "")

    # ──────────────────────────────────────────────────────────────
    # Core: Assess the Bloom Level of a Query
    # ──────────────────────────────────────────────────────────────

    def assess_bloom_level(self, query: str) -> int:
        """Determine the Bloom's Taxonomy level of the user's question.

        Level 1 (Remember): "What is voting calculus?"
        Level 3 (Apply): "How would voting calculus apply in a midterm?"
        Level 5 (Evaluate): "Why is voting calculus insufficient for explaining..."
        Level 6 (Create): "What new framework could integrate voting calculus with..."
        """
        q_lower = query.lower()

        # Score each level based on verb presence and complexity
        scores = {}
        for level, info in BLOOM_LEVELS.items():
            verb_matches = sum(1 for v in info["verbs"] if v in q_lower)
            scores[level] = verb_matches

        # Complexity heuristics
        word_count = len(query.split())
        question_marks = query.count("?")

        # Multi-part questions suggest higher-order thinking
        if question_marks > 1 or word_count > 30:
            scores[4] = scores.get(4, 0) + 1
            scores[5] = scores.get(5, 0) + 1

        # Conditional language suggests analysis/evaluation
        conditional = sum(1 for w in ["if", "however", "although", "despite", "whereas"]
                         if w in q_lower)
        scores[4] = scores.get(4, 0) + conditional
        scores[5] = scores.get(5, 0) + conditional

        # Theory-building language suggests creation
        if any(w in q_lower for w in ["propose", "novel", "new framework", "integrate",
                                       "reconcile", "bridge", "original"]):
            scores[6] = scores.get(6, 0) + 2

        # Find the highest matching level
        best_level = max(scores, key=scores.get)
        if scores[best_level] == 0:
            # No strong signal — default based on complexity
            if word_count < 8:
                return 1
            elif word_count < 20:
                return 2
            else:
                return 3

        return best_level

    # ──────────────────────────────────────────────────────────────
    # Core: Generate the Socratic Response
    # ──────────────────────────────────────────────────────────────

    def generate_pedagogical_prompt(
        self, query: str, sources: list[dict] | None = None
    ) -> dict:
        """Transform a user query into a pedagogical interaction.

        Instead of just answering, the Coach:
        1. Assesses the Bloom level of the question
        2. Provides a Partial Truth (enough to orient)
        3. Generates a Socratic follow-up question
        4. Suggests the next Bloom level to push toward

        Returns a dict with the modified system prompt and instructions.
        """
        bloom_level = self.assess_bloom_level(query)
        self._current_bloom = bloom_level
        target_bloom = min(6, bloom_level + 1)  # Always push one level up

        # Track in profile
        self._profile.bloom_distribution[bloom_level] = (
            self._profile.bloom_distribution.get(bloom_level, 0) + 1
        )
        self._profile.sessions += 1
        self._conversation_depth += 1

        # Extract key terms for the Socratic templates
        key_terms = self._extract_key_terms(query)

        # Build the coaching system prompt
        bloom_info = BLOOM_LEVELS[bloom_level]
        target_info = BLOOM_LEVELS[target_bloom]

        # Mode-specific intensity
        if self._mode == "defense":
            intensity = self._build_defense_prompt(query, bloom_level, key_terms)
        elif self._mode == "comps":
            intensity = self._build_comps_prompt(query, bloom_level, key_terms)
        elif self._mode == "mentor":
            intensity = self._build_mentor_prompt(query, bloom_level, key_terms)
        else:
            intensity = self._build_socratic_prompt(query, bloom_level, key_terms)

        system_prompt = (
            f"You are the Dissertation Coach — an adversarial Socratic mentor.\n"
            f"CURRENT MODE: {self._mode.upper()}\n"
            f"STUDENT'S BLOOM LEVEL: {bloom_info['name']} (Level {bloom_level})\n"
            f"TARGET BLOOM LEVEL: {target_info['name']} (Level {target_bloom})\n\n"
            f"RULES:\n"
            f"1. NEVER give the complete answer. Give a PARTIAL TRUTH — enough to orient, not enough to satisfy.\n"
            f"2. After your partial truth, ask a SOCRATIC QUESTION that forces the student to identify\n"
            f"   the theoretical flaw, hidden assumption, or missing variable.\n"
            f"3. Push the student from '{bloom_info['name']}' toward '{target_info['name']}'.\n"
            f"4. Use the coaching tone: '{bloom_info['coaching_tone']}'\n"
            f"5. {target_info['challenge']}\n\n"
            f"{intensity}\n\n"
            f"REMEMBER: You are preparing this student for the VIVA. "
            f"Every easy answer you give is a disservice. "
            f"A good mentor makes the student do the hard thinking.\n"
        )

        # Update defense readiness
        if bloom_level >= 4:
            self._profile.defense_readiness = min(
                1.0, self._profile.defense_readiness + 0.02
            )

        self._save_profile()

        return {
            "system_prompt": system_prompt,
            "bloom_level": bloom_level,
            "bloom_name": bloom_info["name"],
            "target_level": target_bloom,
            "target_name": target_info["name"],
            "mode": self._mode,
            "coaching_tone": bloom_info["coaching_tone"],
            "key_terms": key_terms,
            "conversation_depth": self._conversation_depth,
            "defense_readiness": round(self._profile.defense_readiness, 2),
        }

    def _build_socratic_prompt(self, query: str, bloom: int, terms: list[str]) -> str:
        term_str = terms[0] if terms else "this concept"
        return (
            f"SOCRATIC STRATEGY:\n"
            f"- Start with: 'That's a good question, but it reveals an assumption...'\n"
            f"- Give 60% of the answer — the part they could find in any textbook.\n"
            f"- Then ask: 'But what happens to {term_str} when we remove the assumption of...?'\n"
            f"- If they answer well, escalate to: 'Now defend that against Aldrich's critique.'\n"
        )

    def _build_defense_prompt(self, query: str, bloom: int, terms: list[str]) -> str:
        return (
            f"DEFENSE SIMULATION STRATEGY:\n"
            f"- You are a skeptical committee member. Your job is to stress-test.\n"
            f"- Open with: 'I'm not convinced. Walk me through your reasoning step by step.'\n"
            f"- After each student response, identify the weakest link and probe it.\n"
            f"- Ask: 'What is the identifying assumption here? Why should I believe the exclusion restriction holds?'\n"
            f"- If the student is vague, say: 'Be more precise. What EXACTLY is the causal mechanism?'\n"
            f"- End with: 'If a reviewer said [X], how would you respond in your revisions?'\n"
        )

    def _build_comps_prompt(self, query: str, bloom: int, terms: list[str]) -> str:
        return (
            f"COMPREHENSIVE EXAM STRATEGY:\n"
            f"- Test BREADTH: 'Before we go deeper, situate this within the broader literature.'\n"
            f"- Test DEPTH: 'Now pick the most important argument and take it to its logical conclusion.'\n"
            f"- Test CONNECTIONS: 'How does this relate to {terms[0] if terms else 'the core debate'} "
            f"in the adjacent subfield?'\n"
            f"- Test GAPS: 'What question does this literature NOT answer? Why hasn't anyone addressed it?'\n"
            f"- Final challenge: 'Write me an abstract for the paper you SHOULD write next.'\n"
        )

    def _build_mentor_prompt(self, query: str, bloom: int, terms: list[str]) -> str:
        return (
            f"MENTOR STRATEGY:\n"
            f"- Start with encouragement: 'Good instinct to ask about this.'\n"
            f"- Provide scaffolding: break the concept into 3 smaller pieces.\n"
            f"- After explaining, ask: 'Which of these three pieces is most relevant to YOUR argument?'\n"
            f"- Guide, don't lecture: 'You're close. What if you thought about it from the {terms[0] if terms else 'other'} perspective?'\n"
        )

    def _extract_key_terms(self, query: str) -> list[str]:
        """Extract academic key terms from the query."""
        # Remove stopwords and short words
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "being", "have", "has", "had", "do", "does", "did", "will",
                      "would", "could", "should", "may", "might", "shall", "can",
                      "for", "and", "nor", "but", "or", "yet", "so", "at", "by",
                      "in", "of", "on", "to", "up", "it", "its", "with", "from",
                      "this", "that", "these", "those", "what", "how", "why", "when",
                      "where", "who", "which", "about", "between", "into", "through"}

        words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
        terms = [w for w in words if w not in stopwords]

        # Prioritize capitalized terms from original query
        caps = re.findall(r'\b[A-Z][a-z]{2,}\b', query)
        terms = caps + [t for t in terms if t.lower() not in [c.lower() for c in caps]]

        return terms[:5]

    # ──────────────────────────────────────────────────────────────
    # Progress & Analytics
    # ──────────────────────────────────────────────────────────────

    def get_profile(self) -> dict:
        """Return the student's cognitive growth profile."""
        return self._profile.to_dict()

    def get_defense_readiness(self) -> dict:
        """Assess readiness for the oral defense / VIVA.

        Based on:
        - Bloom distribution (should be weighted toward 4-6)
        - Diversity of topics engaged
        - Consistency (streak)
        """
        profile = self._profile

        # Calculate component scores
        higher_order = sum(
            profile.bloom_distribution.get(l, 0) for l in [4, 5, 6]
        )
        total = sum(profile.bloom_distribution.values()) or 1
        higher_order_ratio = higher_order / total

        readiness = {
            "overall": round(profile.defense_readiness, 2),
            "higher_order_ratio": round(higher_order_ratio, 2),
            "sessions_completed": profile.sessions,
            "dominant_level": BLOOM_LEVELS[profile.dominant_level]["name"],
            "streak_days": profile.streak_days,
        }

        # Generate recommendation
        if profile.defense_readiness >= 0.8:
            readiness["recommendation"] = (
                "You're showing strong analytical and evaluative thinking. "
                "Focus on synthesizing novel arguments to reach Create level consistently."
            )
        elif profile.defense_readiness >= 0.5:
            readiness["recommendation"] = (
                "Good progress. You're comfortable with analysis. "
                "Practice defending your methodological choices under pressure."
            )
        elif profile.defense_readiness >= 0.2:
            readiness["recommendation"] = (
                "Building foundation. Spend more time on 'Why' questions — "
                "evaluate and judge, don't just apply."
            )
        else:
            readiness["recommendation"] = (
                "Early stages. That's okay. Focus on understanding the core debates "
                "before attempting to evaluate them."
            )

        return readiness

    def reset_session(self):
        """Reset the conversation depth for a new session."""
        self._conversation_depth = 0

    def get_bloom_summary(self) -> dict:
        """Return a Bloom's Taxonomy summary for the UI visualization."""
        return {
            "levels": {
                str(level): {
                    "name": info["name"],
                    "description": info["description"],
                    "count": self._profile.bloom_distribution.get(level, 0),
                    "is_dominant": level == self._profile.dominant_level,
                }
                for level, info in BLOOM_LEVELS.items()
            },
            "growth_score": self._profile.growth_score,
        }


# Global instance
socratic_coach = SocraticCoach()
