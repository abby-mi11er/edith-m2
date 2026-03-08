"""
Lit Locator — Theoretical Atlas & Theory Bridge Mapper
========================================================
Methodological Forensics Lab Feature 3: Where Does This Paper Live?

You look at the 3D Atlas. The paper isn't just a dot; it's a Gravity Well.
Winnie draws "Causal Lines" backward to its ancestors (papers it cites)
and "Influence Lines" forward to the papers that cite it.

This module builds the THEORETICAL MAP:
1. Ancestry — which theories does this paper build on?
2. Lineage — which papers cite this one?
3. Gap Detection — what space does this paper claim to fill?
4. Bridge Mapping — how does it connect two theoretical traditions?
5. Dissertation Fit — where does it plug into YOUR work?

Architecture:
    Forensic Audit → Theory Extraction → Ancestry Graph →
    Forward Citation Trace → Gap Detection → Bridge Mapping →
    "Plug into Your Dissertation" Recommendation
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.lit_locator")


# ═══════════════════════════════════════════════════════════════════
# Theoretical Tradition Registry — The Canonical Lineages
# ═══════════════════════════════════════════════════════════════════

THEORETICAL_TRADITIONS = {
    "principal_agent": {
        "label": "Principal-Agent Theory",
        "founders": ["Jensen & Meckling (1976)", "Moe (1984)", "Miller (2005)"],
        "core_concepts": ["moral hazard", "adverse selection", "monitoring", "shirking",
                           "information asymmetry", "incentive alignment"],
        "key_tensions": ["How do principals monitor agents they can't observe?",
                          "When do agents pursue their own goals?"],
        "adjacent_traditions": ["institutional_choice", "public_choice", "bureaucracy_politics"],
        "political_science_application": (
            "Government contracts with nonprofits to deliver services, "
            "creating a classic P-A problem: how does the state (principal) "
            "monitor the charity (agent)?"
        ),
    },
    "administrative_burden": {
        "label": "Administrative Burden Theory",
        "founders": ["Moynihan et al. (2015)", "Herd & Moynihan (2018)", "Heinrich (2016)"],
        "core_concepts": ["learning costs", "compliance costs", "psychological costs",
                           "take-up rates", "hassle", "ordeal mechanisms"],
        "key_tensions": ["Are burdens intentional gatekeeping or bureaucratic inertia?",
                          "Who bears the costs? (Usually the most vulnerable.)"],
        "adjacent_traditions": ["policy_feedback", "street_level_bureaucracy", "welfare_state"],
        "political_science_application": (
            "When charities 'fill gaps' in state services, they may also create "
            "new administrative burdens: different applications, different eligibility "
            "rules, different reporting requirements."
        ),
    },
    "state_capacity": {
        "label": "State Capacity Theory",
        "founders": ["Mann (1984)", "Tilly (1990)", "Fukuyama (2004)", "Besley & Persson (2011)"],
        "core_concepts": ["infrastructural power", "extractive capacity", "regulatory capacity",
                           "coercive capacity", "administrative reach"],
        "key_tensions": ["Is state capacity zero-sum with private provision?",
                          "Does capacity grow through crisis or through bureaucratic investment?"],
        "adjacent_traditions": ["institutional_choice", "welfare_state", "development_economics"],
        "political_science_application": (
            "The central question: does charitable provision SUBSTITUTE for state "
            "capacity (crowding out) or COMPLEMENT it (building on shared infrastructure)?"
        ),
    },
    "policy_feedback": {
        "label": "Policy Feedback Theory",
        "founders": ["Pierson (1993)", "Mettler (2002)", "Soss (1999)", "Campbell (2003)"],
        "core_concepts": ["lock-in", "path dependence", "increasing returns",
                           "resource/interpretive effects", "participatory effects"],
        "key_tensions": ["Do policies create their own constituencies?",
                          "How do submerged policies affect democratic engagement?"],
        "adjacent_traditions": ["administrative_burden", "welfare_state", "political_behavior"],
        "political_science_application": (
            "Once charities assume welfare functions, they create stakeholders: "
            "donors, staff, and beneficiaries who resist state re-entry. "
            "The policy feeds back to prevent its own reversal."
        ),
    },
    "street_level_bureaucracy": {
        "label": "Street-Level Bureaucracy",
        "founders": ["Lipsky (1980)", "Maynard-Mooney & Musheno (2003)"],
        "core_concepts": ["discretion", "coping mechanisms", "rationing", "creaming",
                           "rubber-stamping", "citizen-agent encounters"],
        "key_tensions": ["When does discretion help vs. hurt citizens?",
                          "How do frontline workers balance efficiency and justice?"],
        "adjacent_traditions": ["administrative_burden", "principal_agent", "public_management"],
        "political_science_application": (
            "Charity caseworkers operate like street-level bureaucrats — "
            "they exercise discretion over who gets help, but without the same "
            "democratic accountability mechanisms."
        ),
    },
    "welfare_state": {
        "label": "Welfare State Theory",
        "founders": ["Esping-Andersen (1990)", "Pierson (1994)", "Skocpol (1992)"],
        "core_concepts": ["decommodification", "welfare regimes", "liberal/conservative/social democratic",
                           "retrenchment", "welfare chauvinism"],
        "key_tensions": ["Why do some states provide more than others?",
                          "Is the welfare state in permanent crisis?"],
        "adjacent_traditions": ["state_capacity", "policy_feedback", "political_economy"],
        "political_science_application": (
            "The US as a 'liberal' welfare state relies heavily on private providers. "
            "Your dissertation examines the consequences: what happens when the "
            "'private' pillar becomes the dominant service provider in a locality?"
        ),
    },
    "institutional_choice": {
        "label": "Institutional Choice / Rational Design",
        "founders": ["Ostrom (1990)", "North (1990)", "Williamson (1985)"],
        "core_concepts": ["transaction costs", "collective action", "commons governance",
                           "nested enterprises", "polycentricity", "credible commitment"],
        "key_tensions": ["Why do inefficient institutions persist?",
                          "When does decentralization work vs. fragment?"],
        "adjacent_traditions": ["principal_agent", "public_choice", "state_capacity"],
        "political_science_application": (
            "Why do governments choose to delegate to charities rather than build "
            "state capacity? Transaction cost theory suggests: when monitoring is "
            "cheap and tasks are specific."
        ),
    },
    "democratic_erosion": {
        "label": "Democratic Erosion / Backsliding",
        "founders": ["Levitsky & Ziblatt (2018)", "Bermeo (2016)", "Coppedge et al. (2008)"],
        "core_concepts": ["executive aggrandizement", "institutional decay", "norm erosion",
                           "competitive authoritarianism", "democratic deconsolidation"],
        "key_tensions": ["Is democratic erosion gradual or sudden?",
                          "Can democracies die without anyone noticing?"],
        "adjacent_traditions": ["state_capacity", "institutional_choice", "political_behavior"],
        "political_science_application": (
            "When charities replace state services, is this a form of 'quiet' "
            "democratic erosion? Citizens lose the ability to vote on service levels "
            "because decisions are made by private boards, not elected officials."
        ),
    },
}


# ═══════════════════════════════════════════════════════════════════
# Gap Detection — Where Is the Void This Paper Claims to Fill?
# ═══════════════════════════════════════════════════════════════════

GAP_INDICATORS = [
    r"(?:little|less|scant)\s+(?:is known|research|attention|has been|work)",
    r"(?:gap|lacuna|void|silence)\s+in\s+the\s+literature",
    r"(?:no|few)\s+(?:studies|scholars|researchers)\s+have\s+(?:examined|explored|investigated)",
    r"(?:remains|remain)\s+(?:under-?explored|under-?studied|under-?theorized|unclear)",
    r"(?:despite|notwithstanding)\s+[^.]+(?:little|no)\s+(?:attention|research)",
    r"(?:this\s+(?:paper|study|article))\s+(?:fills?|addresses|contributes|bridges)\s+(?:this|the|a)\s+gap",
    r"(?:extends?|build|advance)\s+(?:the|this|our)\s+(?:literature|understanding|knowledge)",
    r"(?:novel|new|original)\s+contribution",
    r"(?:first\s+)(?:to\s+)?(?:examine|study|analyze|investigate|demonstrate)",
]


@dataclass
class TheoryNode:
    """A node in the theoretical ancestry graph."""
    theory_id: str
    label: str
    role: str  # "ancestor", "sibling", "descendant", "bridge"
    strength: float  # 0-1, how strongly the paper connects to this theory
    evidence: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "theory": self.label,
            "id": self.theory_id,
            "role": self.role,
            "strength": round(self.strength, 2),
            "evidence": self.evidence[:3],
        }


@dataclass
class TheoryBridge:
    """A bridge between two theoretical traditions."""
    source_theory: str
    target_theory: str
    bridge_type: str  # "synthesis", "extension", "critique", "application"
    description: str

    def to_dict(self) -> dict:
        return {
            "from": self.source_theory,
            "to": self.target_theory,
            "type": self.bridge_type,
            "description": self.description,
        }


# ═══════════════════════════════════════════════════════════════════
# The Lit Locator Engine
# ═══════════════════════════════════════════════════════════════════

class LitLocator:
    """Map a paper's position in the theoretical landscape.

    "Winnie highlights the specific 'Theory Bridge.' 'Abby, this paper's
    broader contribution is that it bridges Principal-Agent Theory with
    Administrative Burden.'"

    Usage:
        locator = LitLocator()
        location = locator.locate_paper(text, title, citations)
    """

    def __init__(self):
        self._atlas: list[dict] = []  # Papers already located
        self._connections: list[dict] = []

    def locate_paper(self, text: str, title: str = "",
                      citations: list[dict] = None,
                      user_dissertation_topic: str = "") -> dict:
        """Perform a complete theoretical location analysis.

        Returns the paper's position in the theoretical landscape:
        where it came from, what gap it fills, and how it connects
        to your dissertation.
        """
        t0 = time.time()

        # 1. Identify theoretical ancestry
        ancestry = self._map_ancestry(text)

        # 2. Detect gaps the paper claims to fill
        gaps = self._detect_gaps(text)

        # 3. Map theory bridges
        bridges = self._find_bridges(ancestry)

        # 4. Trace citation lineage
        lineage = self._trace_lineage(citations or [])

        # 5. Generate dissertation fit assessment
        diss_fit = self._assess_dissertation_fit(
            ancestry, bridges, gaps,
            user_dissertation_topic or "state capacity, charitable organizations, and welfare provision"
        )

        location = {
            "title": title,
            "ancestry": {
                "primary_tradition": ancestry[0].to_dict() if ancestry else None,
                "all_traditions": [a.to_dict() for a in ancestry],
                "theoretical_density": len(ancestry),
            },
            "gaps": {
                "claimed_gaps": gaps,
                "fills_existing_gap": len(gaps) > 0,
            },
            "bridges": {
                "theory_bridges": [b.to_dict() for b in bridges],
                "is_bridge_paper": len(bridges) > 0,
            },
            "lineage": lineage,
            "dissertation_fit": diss_fit,
            "elapsed_seconds": round(time.time() - t0, 3),
        }

        # Add to atlas
        self._atlas.append({
            "title": title,
            "traditions": [a.theory_id for a in ancestry],
            "bridges": [(b.source_theory, b.target_theory) for b in bridges],
            "timestamp": time.time(),
        })

        return location

    def _map_ancestry(self, text: str) -> list[TheoryNode]:
        """Map the paper's theoretical ancestry."""
        nodes = []
        text_lower = text.lower()

        for tid, tradition in THEORETICAL_TRADITIONS.items():
            # Score: how many core concepts appear?
            concept_hits = [
                c for c in tradition["core_concepts"]
                if c.lower() in text_lower
            ]
            # Also check for founder citations
            founder_hits = [
                f for f in tradition["founders"]
                if any(part.lower() in text_lower for part in f.split()[:1])
            ]

            strength = min(1.0, (len(concept_hits) * 0.15 + len(founder_hits) * 0.25))

            if strength > 0.2:
                evidence = concept_hits[:3] + founder_hits[:2]
                # Classify role
                if strength > 0.6:
                    role = "ancestor"  # Primary theoretical parent
                elif strength > 0.3:
                    role = "sibling"   # Related but not central
                else:
                    role = "distant"   # Tangentially related

                nodes.append(TheoryNode(
                    theory_id=tid,
                    label=tradition["label"],
                    role=role,
                    strength=strength,
                    evidence=evidence,
                ))

        nodes.sort(key=lambda n: n.strength, reverse=True)
        return nodes

    def _detect_gaps(self, text: str) -> list[dict]:
        """Detect the gaps the paper claims to fill."""
        gaps = []
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            for pattern in GAP_INDICATORS:
                if re.search(pattern, sentence, re.IGNORECASE):
                    gaps.append({
                        "claim": sentence.strip()[:300],
                        "type": "explicit_gap_claim",
                    })
                    break

        return gaps[:10]  # Cap at 10 gap claims

    def _find_bridges(self, ancestry: list[TheoryNode]) -> list[TheoryBridge]:
        """Find theory bridges — papers that connect two traditions."""
        bridges = []
        # If the paper draws on 2+ traditions, it's bridging them
        ancestors = [n for n in ancestry if n.role == "ancestor"]

        for i, source in enumerate(ancestors):
            for target in ancestors[i + 1:]:
                # Check if these traditions are in each other's "adjacent" list
                source_data = THEORETICAL_TRADITIONS.get(source.theory_id, {})
                target_data = THEORETICAL_TRADITIONS.get(target.theory_id, {})

                if target.theory_id in source_data.get("adjacent_traditions", []):
                    bridge_type = "synthesis"
                else:
                    bridge_type = "novel_bridge"

                bridges.append(TheoryBridge(
                    source_theory=source.label,
                    target_theory=target.label,
                    bridge_type=bridge_type,
                    description=(
                        f"This paper connects {source.label} with {target.label}, "
                        f"drawing on concepts from both traditions."
                    ),
                ))

        return bridges

    def _trace_lineage(self, citations: list[dict]) -> dict:
        """Trace the citation lineage of the paper."""
        if not citations:
            return {"traced": False, "reason": "no_citations_provided"}

        # Group citations by decade
        by_decade = {}
        for cite in citations:
            year = cite.get("year", 0)
            if year:
                decade = (year // 10) * 10
                by_decade.setdefault(decade, []).append(cite)

        # Find the oldest citation (intellectual origin)
        all_years = [c.get("year", 9999) for c in citations if c.get("year")]
        earliest = min(all_years) if all_years else None
        latest = max(all_years) if all_years else None

        return {
            "traced": True,
            "total_citations": len(citations),
            "citations_in_vault": sum(1 for c in citations if c.get("in_vault")),
            "earliest_citation_year": earliest,
            "latest_citation_year": latest,
            "by_decade": {
                str(decade): len(cites)
                for decade, cites in sorted(by_decade.items())
            },
            "intellectual_depth": latest - earliest if earliest and latest else 0,
        }

    def _assess_dissertation_fit(self, ancestry: list[TheoryNode],
                                   bridges: list[TheoryBridge],
                                   gaps: list[dict],
                                   dissertation_topic: str) -> dict:
        """Assess how this paper fits into the user's dissertation.

        "Abby, this paper's broader contribution is that it bridges
        Principal-Agent Theory with Administrative Burden. It fills the
        gap I highlighted in your Atlas last week."
        """
        # Check overlap with dissertation-relevant theories
        dissertation_theories = {
            "principal_agent", "administrative_burden", "state_capacity",
            "policy_feedback", "welfare_state", "street_level_bureaucracy",
        }

        relevant_nodes = [
            n for n in ancestry
            if n.theory_id in dissertation_theories
        ]

        # Generate fit assessment
        if len(relevant_nodes) >= 2:
            fit_level = "essential"
            recommendation = (
                f"This paper directly engages with {len(relevant_nodes)} theories "
                f"central to your dissertation on {dissertation_topic}. "
                f"Consider integrating its framework into your literature review."
            )
        elif len(relevant_nodes) == 1:
            fit_level = "relevant"
            recommendation = (
                f"This paper engages with {relevant_nodes[0].label}, which is "
                f"relevant to your work. It may provide useful citations or "
                f"framework extensions."
            )
        else:
            fit_level = "peripheral"
            recommendation = (
                f"This paper's theoretical framework is peripheral to your "
                f"dissertation. However, check if its METHODS could be applied "
                f"to your research questions."
            )

        # Check if any bridge connects dissertation theories
        diss_bridges = [
            b for b in bridges
            if any(t in b.source_theory.lower() or t in b.target_theory.lower()
                   for t in ["principal", "administrative", "state capacity",
                             "welfare", "policy feedback"])
        ]

        if diss_bridges:
            recommendation += (
                f"\n\n🌉 THEORY BRIDGE ALERT: This paper bridges "
                f"{diss_bridges[0].source_theory} with {diss_bridges[0].target_theory}. "
                f"This is exactly the kind of connection your dissertation needs."
            )

        return {
            "fit_level": fit_level,
            "relevant_traditions": [n.to_dict() for n in relevant_nodes],
            "recommendation": recommendation,
            "bridges_to_your_work": [b.to_dict() for b in diss_bridges],
        }

    def get_atlas_summary(self) -> dict:
        """Summarize all papers located in the atlas."""
        tradition_counts = {}
        for entry in self._atlas:
            for t in entry.get("traditions", []):
                tradition_counts[t] = tradition_counts.get(t, 0) + 1

        return {
            "papers_mapped": len(self._atlas),
            "tradition_distribution": tradition_counts,
            "most_common_tradition": (
                max(tradition_counts, key=tradition_counts.get)
                if tradition_counts else None
            ),
        }


# Global instance
lit_locator = LitLocator()
