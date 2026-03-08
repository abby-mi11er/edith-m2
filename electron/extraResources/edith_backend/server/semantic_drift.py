"""
Semantic Drift Detection — The Genealogy of Ideas
====================================================
Pedagogical Mirror Feature 4: Diachronic Word Embeddings.

"Accountability" means something different in 1996 vs. 2026.
"Non-State Provision" shifted from a "Market Efficiency" argument
in the 90s to a "Democratic Erosion" argument in the 2020s.

This module tracks how academic terms drift in meaning across
your 1TB vault, producing a "Genealogy of Ideas" that helps you
place your work precisely in the historical arc of the literature.

Architecture:
    Corpus → Era Partitioning → Per-Era Embeddings →
    Drift Vectors → Semantic Timelines → Context Windows

The key insight: If you use "accountability" without understanding
its semantic history, you're citing the word, not the concept.
"""

import hashlib
import json
import logging
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.semantic_drift")


# ═══════════════════════════════════════════════════════════════════
# Era Definitions — How we partition academic time
# ═══════════════════════════════════════════════════════════════════

ACADEMIC_ERAS = {
    "pre_reagan": {"start": 1960, "end": 1980, "label": "Pre-Reagan Era",
                   "paradigm": "Great Society / Keynesian Consensus"},
    "reagan_revolution": {"start": 1981, "end": 1992, "label": "Reagan Revolution",
                          "paradigm": "Devolution / New Federalism"},
    "clinton_welfare": {"start": 1993, "end": 2000, "label": "Clinton Welfare Reform",
                        "paradigm": "Third Way / Welfare-to-Work"},
    "bush_gwot": {"start": 2001, "end": 2008, "label": "Bush GWOT Era",
                  "paradigm": "Security State / Faith-Based Initiatives"},
    "obama_aca": {"start": 2009, "end": 2016, "label": "Obama/ACA Era",
                  "paradigm": "Universal Coverage / Nudge Theory"},
    "trump_disruption": {"start": 2017, "end": 2020, "label": "Trump Disruption",
                         "paradigm": "Administrative Deconstruction"},
    "post_covid": {"start": 2021, "end": 2026, "label": "Post-COVID Reckoning",
                   "paradigm": "State Capacity Debate / Democratic Erosion"},
}


# ═══════════════════════════════════════════════════════════════════
# Term Tracking — The Core Data Structure
# ═══════════════════════════════════════════════════════════════════

@dataclass
class TermSnapshot:
    """A snapshot of a term's meaning in a specific era."""
    term: str
    era: str
    year_range: tuple[int, int]
    contexts: list[str]  # Excerpt sentences where the term appears
    co_occurring_terms: list[str]  # Words that frequently appear alongside
    paradigm_alignment: str  # Which paradigm this usage aligns with
    frequency: int  # How often the term appears in this era
    sentiment_polarity: float  # -1 (critical) to +1 (celebratory)

    def to_dict(self) -> dict:
        return {
            "term": self.term,
            "era": self.era,
            "year_range": list(self.year_range),
            "context_count": len(self.contexts),
            "sample_contexts": self.contexts[:3],
            "co_occurring": self.co_occurring_terms[:10],
            "paradigm": self.paradigm_alignment,
            "frequency": self.frequency,
            "sentiment": round(self.sentiment_polarity, 3),
        }


@dataclass
class DriftVector:
    """Measures how a term's meaning shifted between two eras."""
    term: str
    era_from: str
    era_to: str
    drift_magnitude: float  # 0 (stable) to 1 (completely changed)
    drift_direction: str  # Description of the shift
    from_paradigm: str
    to_paradigm: str
    evidence_contexts: list[str]  # Examples showing the shift
    co_occurrence_shift: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "term": self.term,
            "from": self.era_from,
            "to": self.era_to,
            "magnitude": round(self.drift_magnitude, 3),
            "direction": self.drift_direction,
            "from_paradigm": self.from_paradigm,
            "to_paradigm": self.to_paradigm,
            "evidence": self.evidence_contexts[:3],
            "co_occurrence_shift": {
                k: round(v, 3)
                for k, v in sorted(
                    self.co_occurrence_shift.items(),
                    key=lambda x: abs(x[1]), reverse=True
                )[:10]
            },
        }


# ═══════════════════════════════════════════════════════════════════
# Semantic Drift Engine — The Core
# ═══════════════════════════════════════════════════════════════════

class SemanticDriftEngine:
    """Track how academic terms change meaning across eras.

    Core capabilities:
    1. Term Archaeology: How did "accountability" mean in 1996?
    2. Drift Detection: When did the meaning shift?
    3. Genealogy Map: The full lifecycle of a concept
    4. Placement Guide: Where does YOUR usage fit historically?
    """

    def __init__(self, data_root: str = ""):
        self._data_root = data_root or os.environ.get("EDITH_APP_DATA_DIR", "")
        self._term_registry: dict[str, dict[str, TermSnapshot]] = {}  # term -> {era -> snapshot}
        self._drift_cache: dict[str, list[DriftVector]] = {}  # term -> [drifts]
        self._corpus_stats: dict = {"documents_processed": 0, "eras_covered": set()}

    # ──────────────────────────────────────────────────────────────
    # Register terms from the corpus
    # ──────────────────────────────────────────────────────────────

    def register_term_usage(
        self,
        term: str,
        context: str,
        year: int,
        source_author: str = "",
        source_title: str = "",
    ) -> dict:
        """Register a term usage from the corpus.

        Call this for each occurrence of a tracked term as documents
        are indexed. The engine builds up era-partitioned profiles.
        """
        term_lower = term.lower()
        era = self._year_to_era(year)
        if not era:
            return {"status": "skipped", "reason": f"Year {year} outside tracked eras"}

        era_info = ACADEMIC_ERAS[era]

        if term_lower not in self._term_registry:
            self._term_registry[term_lower] = {}

        if era not in self._term_registry[term_lower]:
            self._term_registry[term_lower][era] = TermSnapshot(
                term=term_lower,
                era=era,
                year_range=(era_info["start"], era_info["end"]),
                contexts=[],
                co_occurring_terms=[],
                paradigm_alignment=era_info["paradigm"],
                frequency=0,
                sentiment_polarity=0.0,
            )

        snapshot = self._term_registry[term_lower][era]

        # Add context (keep top 20 examples)
        if len(snapshot.contexts) < 20:
            attribution = f"[{source_author}, {year}]" if source_author else f"[{year}]"
            snapshot.contexts.append(f"{attribution}: ...{context[:200]}...")

        # Update frequency
        snapshot.frequency += 1

        # Extract co-occurring terms
        co_terms = self._extract_co_occurring(context, term_lower)
        snapshot.co_occurring_terms = list(
            set(snapshot.co_occurring_terms + co_terms)
        )[:30]

        # Update sentiment (simple lexicon-based)
        snapshot.sentiment_polarity = self._estimate_sentiment(
            [c for c in snapshot.contexts[-10:]]
        )

        self._corpus_stats["eras_covered"].add(era)

        return {
            "status": "registered",
            "term": term_lower,
            "era": era,
            "frequency": snapshot.frequency,
        }

    def _year_to_era(self, year: int) -> Optional[str]:
        for era_key, era_info in ACADEMIC_ERAS.items():
            if era_info["start"] <= year <= era_info["end"]:
                return era_key
        return None

    def _extract_co_occurring(self, context: str, target_term: str) -> list[str]:
        """Extract significant co-occurring terms from context."""
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                      "have", "has", "had", "do", "does", "did", "will", "would",
                      "could", "should", "for", "and", "nor", "but", "or", "yet",
                      "so", "at", "by", "in", "of", "on", "to", "up", "it",
                      "its", "with", "from", "this", "that", "not", "no",
                      "which", "who", "whom", "what", "how", "when", "where"}

        words = re.findall(r'\b[a-z]{4,}\b', context.lower())
        significant = [w for w in words if w not in stopwords and w != target_term]
        return significant[:10]

    def _estimate_sentiment(self, contexts: list[str]) -> float:
        """Simple lexicon-based sentiment on academic text."""
        if not contexts:
            return 0.0

        positive = {"effective", "successful", "improved", "strengthened", "enhanced",
                     "democratic", "equitable", "transparent", "efficient", "innovative",
                     "progress", "achievement", "benefit", "empowerment"}
        negative = {"failure", "erosion", "decline", "dysfunction", "inequality",
                     "corruption", "crisis", "fragmented", "inadequate", "problematic",
                     "privatization", "retrenchment", "deficiency", "undermine"}

        text = " ".join(contexts).lower()
        pos = sum(1 for w in positive if w in text)
        neg = sum(1 for w in negative if w in text)
        total = pos + neg or 1
        return (pos - neg) / total

    # ──────────────────────────────────────────────────────────────
    # Drift Analysis — The Core Algorithm
    # ──────────────────────────────────────────────────────────────

    def analyze_drift(self, term: str) -> list[DriftVector]:
        """Analyze how a term's meaning drifted across all eras.

        Returns a list of DriftVectors showing the shift between
        each consecutive era where the term appears.
        """
        term_lower = term.lower()
        if term_lower not in self._term_registry:
            return []

        eras = self._term_registry[term_lower]
        # Sort eras chronologically
        era_order = [e for e in ACADEMIC_ERAS.keys() if e in eras]

        drifts = []
        for i in range(len(era_order) - 1):
            era_from = era_order[i]
            era_to = era_order[i + 1]
            snap_from = eras[era_from]
            snap_to = eras[era_to]

            drift = self._compute_drift(snap_from, snap_to)
            drifts.append(drift)

        self._drift_cache[term_lower] = drifts
        return drifts

    def _compute_drift(self, snap_from: TermSnapshot,
                        snap_to: TermSnapshot) -> DriftVector:
        """Compute the drift between two era snapshots.

        Drift is measured by:
        1. Co-occurrence shift (what words surround the term)
        2. Sentiment shift (positive/neutral/negative usage)
        3. Paradigm change (underlying framework shift)
        """
        # Co-occurrence shift: Jaccard distance
        set_from = set(snap_from.co_occurring_terms)
        set_to = set(snap_to.co_occurring_terms)
        union = set_from | set_to
        intersection = set_from & set_to
        jaccard_distance = 1 - (len(intersection) / max(len(union), 1))

        # Sentiment shift
        sentiment_shift = abs(snap_to.sentiment_polarity - snap_from.sentiment_polarity)

        # Paradigm shift (binary: same or different)
        paradigm_shift = 0.3 if snap_from.paradigm_alignment != snap_to.paradigm_alignment else 0.0

        # Combined drift magnitude
        drift_magnitude = min(1.0, (
            jaccard_distance * 0.5 +
            sentiment_shift * 0.25 +
            paradigm_shift * 0.25
        ))

        # Determine drift direction
        if sentiment_shift > 0.3:
            if snap_to.sentiment_polarity > snap_from.sentiment_polarity:
                direction = f"Shifted toward more positive/celebratory usage"
            else:
                direction = f"Shifted toward more critical/negative usage"
        elif jaccard_distance > 0.5:
            # Look at what appeared and disappeared
            appeared = set_to - set_from
            disappeared = set_from - set_to
            direction = (
                f"Context shifted. New associations: {', '.join(list(appeared)[:5])}. "
                f"Lost associations: {', '.join(list(disappeared)[:5])}"
            )
        else:
            direction = "Relatively stable meaning with minor contextual shifts"

        # Co-occurrence shift details
        co_shift = {}
        for term in appeared if 'appeared' in dir() else (set_to - set_from):
            co_shift[term] = 1.0  # Appeared
        for term in disappeared if 'disappeared' in dir() else (set_from - set_to):
            co_shift[term] = -1.0  # Disappeared

        return DriftVector(
            term=snap_from.term,
            era_from=snap_from.era,
            era_to=snap_to.era,
            drift_magnitude=drift_magnitude,
            drift_direction=direction,
            from_paradigm=snap_from.paradigm_alignment,
            to_paradigm=snap_to.paradigm_alignment,
            evidence_contexts=(
                snap_from.contexts[:2] + snap_to.contexts[:2]
            ),
            co_occurrence_shift=co_shift,
        )

    # ──────────────────────────────────────────────────────────────
    # Term Genealogy — Full Lifecycle View
    # ──────────────────────────────────────────────────────────────

    def get_term_genealogy(self, term: str) -> dict:
        """Get the complete genealogy of a term across all eras.

        This is the "History of Ideas" view: how a concept was born,
        evolved, was contested, and where it stands today.
        """
        term_lower = term.lower()
        if term_lower not in self._term_registry:
            return {"term": term, "found": False, "eras": {}}

        eras = self._term_registry[term_lower]
        drifts = self.analyze_drift(term)

        # Build timeline
        timeline = []
        era_order = [e for e in ACADEMIC_ERAS.keys() if e in eras]
        for era_key in era_order:
            snap = eras[era_key]
            era_info = ACADEMIC_ERAS[era_key]
            timeline.append({
                "era": era_info["label"],
                "years": f"{era_info['start']}-{era_info['end']}",
                "paradigm": era_info["paradigm"],
                "frequency": snap.frequency,
                "sentiment": round(snap.sentiment_polarity, 3),
                "top_associations": snap.co_occurring_terms[:5],
                "sample_context": snap.contexts[0] if snap.contexts else "",
            })

        # Find the biggest drift
        biggest_drift = max(drifts, key=lambda d: d.drift_magnitude) if drifts else None

        # Determine the current meaning
        latest_era = era_order[-1] if era_order else None
        current_usage = eras[latest_era] if latest_era else None

        return {
            "term": term_lower,
            "found": True,
            "total_eras": len(eras),
            "total_occurrences": sum(s.frequency for s in eras.values()),
            "timeline": timeline,
            "drifts": [d.to_dict() for d in drifts],
            "biggest_shift": biggest_drift.to_dict() if biggest_drift else None,
            "current_meaning": {
                "era": ACADEMIC_ERAS[latest_era]["label"] if latest_era else "",
                "paradigm": current_usage.paradigm_alignment if current_usage else "",
                "top_associations": current_usage.co_occurring_terms[:5] if current_usage else [],
                "sentiment": round(current_usage.sentiment_polarity, 3) if current_usage else 0,
            } if current_usage else {},
        }

    # ──────────────────────────────────────────────────────────────
    # Placement Guide — Where Does YOUR Usage Fit?
    # ──────────────────────────────────────────────────────────────

    def assess_usage_placement(self, term: str, user_context: str) -> dict:
        """Assess where the user's usage of a term falls historically.

        Helps the student understand: "Am I using 'accountability' in
        the 1996 sense or the 2024 sense? And does it matter?"
        """
        term_lower = term.lower()
        if term_lower not in self._term_registry:
            return {"error": f"No drift data for '{term}'"}

        # Extract user's co-occurring terms
        user_co_terms = set(self._extract_co_occurring(user_context, term_lower))

        # Find which era the user's usage most resembles
        best_era = ""
        best_overlap = 0

        eras = self._term_registry[term_lower]
        for era_key, snapshot in eras.items():
            era_co_terms = set(snapshot.co_occurring_terms)
            overlap = len(user_co_terms & era_co_terms)
            if overlap > best_overlap:
                best_overlap = overlap
                best_era = era_key

        era_info = ACADEMIC_ERAS.get(best_era, {})

        # Check if this is dated
        latest_era = list(ACADEMIC_ERAS.keys())[-1]
        is_dated = best_era != latest_era and best_era != ""

        return {
            "term": term_lower,
            "your_usage_resembles": era_info.get("label", "Unknown"),
            "era_years": f"{era_info.get('start', '')}-{era_info.get('end', '')}",
            "paradigm": era_info.get("paradigm", ""),
            "is_dated": is_dated,
            "overlap_terms": list(user_co_terms & set(eras.get(best_era, TermSnapshot(
                term="", era="", year_range=(0, 0), contexts=[], co_occurring_terms=[],
                paradigm_alignment="", frequency=0, sentiment_polarity=0
            )).co_occurring_terms))[:5],
            "recommendation": (
                f"Your usage of '{term}' aligns with how it was used in the "
                f"{era_info.get('label', 'earlier')} era ({era_info.get('paradigm', '')})."
                + (" Consider whether you intend this historical meaning or the "
                   "current usage, which has shifted." if is_dated else
                   " This aligns with contemporary usage.")
            ),
        }

    # ──────────────────────────────────────────────────────────────
    # Pre-Built Concept Trackers — Key Terms in the Subfield
    # ──────────────────────────────────────────────────────────────

    def get_tracked_terms(self) -> list[str]:
        """Return all terms currently being tracked."""
        return list(self._term_registry.keys())

    def get_drift_summary(self) -> dict:
        """Return a summary of all tracked terms and their drift status."""
        summaries = {}
        for term in self._term_registry:
            drifts = self.analyze_drift(term)
            if drifts:
                max_drift = max(d.drift_magnitude for d in drifts)
                summaries[term] = {
                    "eras_present": len(self._term_registry[term]),
                    "max_drift": round(max_drift, 3),
                    "drift_status": (
                        "highly_drifted" if max_drift > 0.6 else
                        "moderately_drifted" if max_drift > 0.3 else
                        "stable"
                    ),
                }
        return summaries

    def generate_teaching_prompt(self, term: str) -> str:
        """Generate a teaching prompt about a term's semantic drift.

        Used by the Socratic Coach to help students understand
        the genealogy of their key concepts.
        """
        genealogy = self.get_term_genealogy(term)
        if not genealogy.get("found"):
            return f"No semantic history found for '{term}'."

        timeline = genealogy.get("timeline", [])
        biggest = genealogy.get("biggest_shift")

        prompt = (
            f"📚 SEMANTIC HISTORY OF '{term.upper()}'\n\n"
            f"This term appears in {genealogy['total_eras']} academic eras "
            f"with {genealogy['total_occurrences']} total occurrences.\n\n"
        )

        for t in timeline:
            prompt += (
                f"▸ {t['era']} ({t['years']}): Used {t['frequency']}x. "
                f"Paradigm: {t['paradigm']}. "
                f"Top associations: {', '.join(t['top_associations'][:3])}\n"
            )

        if biggest:
            prompt += (
                f"\n⚠️ BIGGEST SHIFT: {biggest['from']} → {biggest['to']}\n"
                f"   {biggest['direction']}\n"
                f"   Drift magnitude: {biggest['magnitude']:.0%}\n"
            )

        prompt += (
            f"\nQUESTION FOR YOU: When you use '{term}' in your dissertation, "
            f"which era's meaning do you intend? And does your committee know that?"
        )

        return prompt

    def save_registry(self) -> dict:
        """Save the term registry to disk."""
        if not self._data_root:
            return {"error": "No data root set"}

        save_path = Path(self._data_root) / "VAULT" / "PEDAGOGY" / "semantic_drift.json"
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                term: {era: snap.to_dict() for era, snap in eras.items()}
                for term, eras in self._term_registry.items()
            }
            save_path.write_text(json.dumps(data, indent=2))
            return {"status": "saved", "terms": len(data), "path": str(save_path)}
        except Exception as e:
            return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# Pre-Configured Political Science Terms
# ═══════════════════════════════════════════════════════════════════

POLITICAL_SCIENCE_TERMS = [
    "accountability", "devolution", "federalism", "privatization",
    "welfare", "governance", "state capacity", "bureaucracy",
    "public provision", "non-state provision", "blame diffusion",
    "democratic erosion", "institutional trust", "civic engagement",
    "policy feedback", "path dependence", "social contract",
    "administrative state", "regulatory capture", "fiscal federalism",
]


# Global instance
semantic_drift = SemanticDriftEngine()
