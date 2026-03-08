"""
Reasoning & Audit Enhancements — Improvements to Winnie's reasoning and hallucination detection.

Implements:
  5.1   Model-specific prompt engineering
  5.3   Confidence calibration (multi-signal scoring)
  5.5   Answer length control integration
  5.10  Reasoning trace (show planning + evidence mapping)
  6.4   Per-paragraph confidence bands (green/yellow/red)
  6.7   Source freshness check
  6.8   Cross-source contradiction detection
  6.9   Recursive audit (re-audit after corrections)
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field

log = logging.getLogger("edith.reasoning_enhancements")


# ---------------------------------------------------------------------------
# 5.1: Model-Specific Prompt Engineering
# ---------------------------------------------------------------------------

MODEL_PROMPT_STYLES = {
    "openai_ft": {
        "style": "concise",
        "system_prefix": "",
        "instruction_format": "direct",
        "notes": "Fine-tuned model responds best to concise, direct instructions",
        "wrap_sources": lambda sources: "\n".join(
            f"[S{i+1}] {s.get('title','')}: {s.get('snippet','')[:400]}"
            for i, s in enumerate(sources)
        ),
    },
    "gemini_flash": {
        "style": "detailed",
        "system_prefix": "You are an expert research assistant. ",
        "instruction_format": "structured",
        "notes": "Gemini responds well to detailed, structured prompts with clear sections",
        "wrap_sources": lambda sources: "\n---\n".join(
            f"### Source {i+1}: {s.get('title','')}\n"
            f"**File**: {s.get('filename','unknown')}\n"
            f"**Section**: {s.get('section','')}\n"
            f"**Content**: {s.get('snippet','')[:600]}"
            for i, s in enumerate(sources)
        ),
    },
    "gemini_lite": {
        "style": "concise",
        "system_prefix": "",
        "instruction_format": "direct",
        "notes": "Lite model: keep prompts short to fit smaller context",
        "wrap_sources": lambda sources: "\n".join(
            f"[S{i+1}] {s.get('snippet','')[:300]}"
            for i, s in enumerate(sources[:5])
        ),
    },
}


def format_sources_for_model(model_key: str, sources: list[dict]) -> str:
    """Format sources optimally for the target model."""
    style = get_prompt_style(model_key)
    formatter = style.get("wrap_sources")
    if formatter:
        return formatter(sources)
    return "\n".join(f"[S{i+1}] {s.get('snippet','')}" for i, s in enumerate(sources))


# ---------------------------------------------------------------------------
# 5.3: Confidence Calibration
# ---------------------------------------------------------------------------

@dataclass
class ConfidenceSignals:
    """Multi-signal confidence assessment for an answer."""
    citation_coverage: float = 0.0       # % of sources cited
    source_relevance_avg: float = 0.0    # Average source relevance score
    audit_result: str = "pending"         # clean / corrected / skipped
    answer_length_ratio: float = 0.0      # Generated vs. max allowed
    claim_count: int = 0                  # Number of factual claims made
    supported_claims: int = 0             # Claims supported by sources
    source_count: int = 0                 # Number of sources available
    model_used: str = ""                  # Which model generated the answer

    @property
    def calibrated_score(self) -> float:
        """
        Compute a calibrated confidence score (0.0 to 1.0).

        Weights:
          - Citation coverage: 30%
          - Source relevance: 20%
          - Audit result: 25%
          - Claim support ratio: 25%
        """
        audit_score = {
            "clean": 1.0,
            "corrected": 0.6,
            "skipped": 0.3,
            "pending": 0.5,
            "failed": 0.2,
        }.get(self.audit_result, 0.5)

        claim_support = (
            self.supported_claims / max(1, self.claim_count)
            if self.claim_count > 0 else 0.7  # Default if we can't count claims
        )

        score = (
            0.30 * min(1.0, self.citation_coverage) +
            0.20 * min(1.0, self.source_relevance_avg) +
            0.25 * audit_score +
            0.25 * claim_support
        )
        return round(min(1.0, max(0.0, score)), 3)

    @property
    def level(self) -> str:
        """Human-readable confidence level."""
        score = self.calibrated_score
        if score >= 0.8:
            return "high"
        elif score >= 0.5:
            return "medium"
        else:
            return "low"

    def as_dict(self) -> dict:
        return {
            "calibrated_score": self.calibrated_score,
            "level": self.level,
            "citation_coverage": round(self.citation_coverage, 3),
            "source_relevance_avg": round(self.source_relevance_avg, 3),
            "audit_result": self.audit_result,
            "claim_support_ratio": round(
                self.supported_claims / max(1, self.claim_count), 3
            ) if self.claim_count else None,
            "source_count": self.source_count,
            "model_used": self.model_used,
        }


# ---------------------------------------------------------------------------
# 5.10: Reasoning Trace
# ---------------------------------------------------------------------------

@dataclass
class ReasoningTrace:
    """Captures the reasoning process for transparency."""
    query: str = ""
    depth: str = ""
    started_at: float = field(default_factory=time.time)

    # Retrieval phase
    queries_used: list[str] = field(default_factory=list)
    sources_retrieved: int = 0
    sources_after_filter: int = 0
    retrieval_method: str = ""
    retrieval_time_ms: float = 0.0

    # Planning phase
    answer_plan: list[str] = field(default_factory=list)
    evidence_gaps: list[str] = field(default_factory=list)

    # Generation phase
    model_used: str = ""
    fallback_used: bool = False
    tokens_generated: int = 0
    generation_time_ms: float = 0.0

    # Audit phase
    audit_result: str = ""
    corrections: list[dict] = field(default_factory=list)
    audit_time_ms: float = 0.0

    # Evidence mapping: which source supports which claim
    evidence_map: list[dict] = field(default_factory=list)

    @property
    def total_time_ms(self) -> float:
        return (time.time() - self.started_at) * 1000

    def add_evidence_link(self, claim: str, source_idx: int, confidence: float = 1.0):
        """Record that a source supports a specific claim."""
        self.evidence_map.append({
            "claim": claim[:200],
            "source": f"S{source_idx}",
            "confidence": round(confidence, 2),
        })

    def as_dict(self) -> dict:
        return {
            "query": self.query[:200],
            "depth": self.depth,
            "total_time_ms": round(self.total_time_ms, 1),
            "retrieval": {
                "queries": self.queries_used,
                "retrieved": self.sources_retrieved,
                "after_filter": self.sources_after_filter,
                "method": self.retrieval_method,
                "time_ms": round(self.retrieval_time_ms, 1),
            },
            "planning": {
                "outline": self.answer_plan,
                "gaps": self.evidence_gaps,
            },
            "generation": {
                "model": self.model_used,
                "fallback": self.fallback_used,
                "tokens": self.tokens_generated,
                "time_ms": round(self.generation_time_ms, 1),
            },
            "audit": {
                "result": self.audit_result,
                "corrections": len(self.corrections),
                "time_ms": round(self.audit_time_ms, 1),
            },
            "evidence_map": self.evidence_map[:20],  # Limit for response size
        }


# ---------------------------------------------------------------------------
# 6.4: Per-Paragraph Confidence Bands
# ---------------------------------------------------------------------------

def assign_paragraph_confidence(
    answer: str,
    sources: list[dict],
    citation_pattern: str = r"\[S(\d+)\]",
) -> list[dict]:
    """
    Assign confidence levels to each paragraph of an answer.

    Returns list of {text, citations, confidence, level} per paragraph.

    Level mapping:
      green  = well-supported (multiple citations, sources match)
      yellow = partially supported (1 citation or weak match)
      red    = ungrounded (no citations)
    """
    paragraphs = [p.strip() for p in answer.split("\n\n") if p.strip()]
    results = []

    for para in paragraphs:
        # Extract citation references
        citations = re.findall(citation_pattern, para)
        citation_indices = [int(c) for c in citations if c.isdigit()]

        # Check if cited sources actually exist
        valid_citations = [
            idx for idx in citation_indices
            if 0 < idx <= len(sources)
        ]

        # Determine confidence level
        if len(valid_citations) >= 2:
            level = "green"
            confidence = 0.9
        elif len(valid_citations) == 1:
            level = "yellow"
            confidence = 0.6
        elif citation_indices and not valid_citations:
            # Cites sources that don't exist -> hallucinated citation
            level = "red"
            confidence = 0.1
        else:
            # No citations at all
            # Check if it's a summary/transition paragraph (acceptable without citations)
            is_transition = len(para.split()) < 15 or para.startswith(("In summary", "Overall", "To conclude"))
            if is_transition:
                level = "yellow"
                confidence = 0.5
            else:
                level = "red"
                confidence = 0.2

        results.append({
            "text": para,
            "citations": valid_citations,
            "confidence": confidence,
            "level": level,
        })

    return results


# ---------------------------------------------------------------------------
# 6.7: Source Freshness Check
# ---------------------------------------------------------------------------

def check_source_freshness(
    query: str,
    sources: list[dict],
    freshness_cutoff_year: int | None = None,
) -> list[dict]:
    """
    Flag sources that may be outdated for the given query.

    Returns list of {source_idx, year, warning} for stale sources.
    """
    current_year = time.localtime().tm_year
    cutoff = freshness_cutoff_year or (current_year - 5)

    # Only check freshness for queries about current state
    query_lower = query.lower()
    needs_freshness = any(w in query_lower for w in [
        "current", "recent", "latest", "nowadays", "today",
        "2024", "2025", "2026", "trend", "contemporary",
    ])

    if not needs_freshness:
        return []

    stale = []
    for i, source in enumerate(sources):
        year = source.get("year") or source.get("pub_year")
        if not year:
            continue
        try:
            year = int(str(year)[:4])
        except (ValueError, TypeError):
            continue

        if year < cutoff:
            stale.append({
                "source_idx": i + 1,
                "year": year,
                "warning": f"Source S{i+1} ({year}) may be outdated for this query about current trends",
            })

    return stale


# ---------------------------------------------------------------------------
# 6.8: Cross-Source Contradiction Detection
# ---------------------------------------------------------------------------

def detect_contradictions(sources: list[dict | str]) -> list[dict]:
    """
    Detect potential contradictions between sources using keyword heuristics.

    Looks for:
    - Opposing claims about the same entity
    - Conflicting statistics
    - Different conclusions on the same topic
    """
    contradictions = []

    # Normalize mixed source inputs (dicts from UI/API or raw strings)
    normalized_sources: list[dict] = []
    for source in sources or []:
        if isinstance(source, str):
            normalized_sources.append({"snippet": source})
        elif isinstance(source, dict):
            snippet = (
                source.get("snippet")
                or source.get("text")
                or source.get("content")
                or ""
            )
            normalized_sources.append({**source, "snippet": snippet})

    # Extract key claims from each source
    source_claims: list[list[str]] = []
    for source in normalized_sources:
        text = source.get("snippet", "")
        # Split into sentences
        sentences = re.split(r"[.!?]+", text)
        claims = [s.strip() for s in sentences if len(s.strip()) > 20]
        source_claims.append(claims)

    # Compare pairs of sources for contradictions
    negation_pairs = [
        (r"\bincreases?\b", r"\bdecreases?\b"),
        (r"\bhigher\b", r"\blower\b"),
        (r"\bmore\b", r"\bless\b"),
        (r"\bpositive\b", r"\bnegative\b"),
        (r"\bsignificant\b", r"\binsignificant\b"),
        (r"\bsupport\b", r"\boppose\b"),
        (r"\bconfirms?\b", r"\bcontradicts?\b"),
        (r"\bcorrelat", r"\bno (?:significant )?(?:correlation|relationship)\b"),
    ]

    for i in range(len(normalized_sources)):
        for j in range(i + 1, len(normalized_sources)):
            for claim_a in source_claims[i][:5]:  # Limit comparisons
                for claim_b in source_claims[j][:5]:
                    # Check for shared topic (at least 3 common words)
                    words_a = set(re.findall(r"\b\w{4,}\b", claim_a.lower()))
                    words_b = set(re.findall(r"\b\w{4,}\b", claim_b.lower()))
                    common = words_a & words_b

                    if len(common) < 3:
                        continue

                    # Check for negation patterns
                    for pos_pattern, neg_pattern in negation_pairs:
                        a_has_pos = bool(re.search(pos_pattern, claim_a, re.I))
                        a_has_neg = bool(re.search(neg_pattern, claim_a, re.I))
                        b_has_pos = bool(re.search(pos_pattern, claim_b, re.I))
                        b_has_neg = bool(re.search(neg_pattern, claim_b, re.I))

                        if (a_has_pos and b_has_neg) or (a_has_neg and b_has_pos):
                            contradictions.append({
                                "source_a": i + 1,
                                "source_b": j + 1,
                                "claim_a": claim_a[:200],
                                "claim_b": claim_b[:200],
                                "shared_topic": list(common)[:5],
                                "type": "opposing_claims",
                            })
                            break  # One contradiction per pair is enough

    return contradictions[:5]  # Limit to top 5


# ---------------------------------------------------------------------------
# 6.9: Recursive Audit
# ---------------------------------------------------------------------------

@dataclass
class AuditState:
    """Tracks audit state across recursive passes."""
    max_passes: int = 2
    current_pass: int = 0
    original_answer: str = ""
    current_answer: str = ""
    all_corrections: list[dict] = field(default_factory=list)
    is_clean: bool = False

    @property
    def should_re_audit(self) -> bool:
        """Check if we should run another audit pass."""
        return (
            not self.is_clean
            and self.current_pass < self.max_passes
            and len(self.all_corrections) > 0
        )

    def record_pass(self, is_clean: bool, corrections: list[dict], corrected_answer: str):
        """Record the result of an audit pass."""
        self.current_pass += 1
        self.is_clean = is_clean
        self.all_corrections.extend(corrections)
        if corrected_answer:
            self.current_answer = corrected_answer

    def as_dict(self) -> dict:
        return {
            "passes": self.current_pass,
            "is_clean": self.is_clean,
            "total_corrections": len(self.all_corrections),
            "corrections": self.all_corrections,
        }
