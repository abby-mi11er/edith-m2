"""
Retrieval Enhancements — R1/R3/R9
====================================
  - Reciprocal Rank Fusion (RRF) for multi-signal scoring
  - Temporal weighting for recency bias
  - Query intent routing (methods/theory/empirical)
"""
from __future__ import annotations

import logging
import re

log = logging.getLogger("edith.retrieval_enhance")


# ═══════════════════════════════════════════════════════════════════
# R1: Reciprocal Rank Fusion (RRF)
# ═══════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════
# R3: Temporal Weighting
# ═══════════════════════════════════════════════════════════════════

def apply_temporal_weight(
    sources: list[dict],
    decay_rate: float = 0.05,
    reference_year: int = 2026,
    boost_recent_years: int = 5,
) -> list[dict]:
    """Apply recency weighting to source scores.

    Recent papers (last 5 years) get a boost.
    Older papers get a decay based on age.

    Args:
        sources: list of source dicts (must have metadata.year or metadata.date)
        decay_rate: per-year decay factor for old papers
        reference_year: current year
        boost_recent_years: papers within this window get a 1.0 + bonus

    Returns:
        Sources with _temporal_weight added to metadata
    """
    for source in sources:
        meta = source.get("metadata", {})
        year = meta.get("year") or meta.get("publication_year")

        if not year:
            # Try to parse from date string
            date_str = meta.get("date", "") or meta.get("publication_date", "")
            year_match = re.search(r"(19|20)\d{2}", str(date_str))
            if year_match:
                year = int(year_match.group())

        if year:
            try:
                year = int(year)
                age = reference_year - year

                if age <= 0:
                    weight = 1.2  # Future/current year
                elif age <= boost_recent_years:
                    weight = 1.0 + (boost_recent_years - age) * 0.04  # 1.0 to 1.2
                else:
                    weight = max(0.5, 1.0 - (age - boost_recent_years) * decay_rate)

                source["_temporal_weight"] = round(weight, 3)
            except (ValueError, TypeError):
                source["_temporal_weight"] = 1.0
        else:
            source["_temporal_weight"] = 1.0

    return sources


def rerank_by_temporal_weight(sources: list[dict], time_factor: float = 0.15) -> list[dict]:
    """Re-rank sources incorporating temporal weight.

    Args:
        sources: sources with existing scores and _temporal_weight
        time_factor: how much temporal weight influences final ranking (0-1)

    Returns:
        Re-ranked sources
    """
    sources = apply_temporal_weight(sources)

    for s in sources:
        base_score = s.get("relevance_score") or s.get("_rrf_score") or s.get("score", 0.5)
        temporal = s.get("_temporal_weight", 1.0)
        s["_final_score"] = base_score * (1 - time_factor) + base_score * temporal * time_factor

    sources.sort(key=lambda s: s.get("_final_score", 0), reverse=True)
    return sources


# ═══════════════════════════════════════════════════════════════════
# R9: Query Intent Routing
# ═══════════════════════════════════════════════════════════════════

# Intent patterns → retrieval parameter tuning
_INTENT_PROFILES = {
    "methodology": {
        "patterns": [
            r"\b(how|method|approach|technique|design|instrument|measure|survey|experiment|analysis|procedure)\b",
            r"\b(RCT|regression|case\s+study|ethnograph|meta-analysis|mixed\s+methods)\b",
        ],
        "bm25_weight": 0.45,     # Methods use precise terminology
        "diversity_lambda": 0.4,  # Less diversity — precision matters
        "rerank_boost": ["methodology", "research design", "methods section"],
    },
    "theoretical": {
        "patterns": [
            r"\b(theory|framework|model|paradigm|perspective|concept|argue|posit|hypothesis)\b",
            r"\b(constructivist|realist|institutionalist|rational\s+choice|neo-?liberal)\b",
        ],
        "bm25_weight": 0.30,     # Theory is more semantic
        "diversity_lambda": 0.7,  # Want diverse theoretical perspectives
        "rerank_boost": ["theory", "framework", "conceptual"],
    },
    "empirical": {
        "patterns": [
            r"\b(evidence|data|findings?|results?|show|demonstrat|correlat|significant|outcome)\b",
            r"\b(study|studies|sample|population|N\s*=|p\s*[<>])\b",
        ],
        "bm25_weight": 0.35,
        "diversity_lambda": 0.5,
        "rerank_boost": ["findings", "results", "empirical"],
    },
    "comparative": {
        "patterns": [
            r"\b(compar|contrast|differ|similar|versus|vs\.?|between|across)\b",
            r"\b(more|less|greater|smaller|better|worse)\b.*\bthan\b",
        ],
        "bm25_weight": 0.30,
        "diversity_lambda": 0.8,  # Maximum diversity for perspectives
        "rerank_boost": ["comparison", "contrast", "difference"],
    },
}



# ══════════════════════════════════════════════════════════════
# Merged from retrieval_improvements.py
# ══════════════════════════════════════════════════════════════

#!/usr/bin/env python3
"""
Retrieval Engine Improvements Module
=====================================
Enhancements for server/chroma_backend.py:
  2.1  Contextual reranking with surrounding chunks
  2.2  Retrieval quality telemetry
  2.3  Adaptive top_k based on query complexity
  2.4  Pre-computed document summaries
  2.5  Query decomposition for multi-part questions
  2.6  Temporal weighting for recency
  2.7  Retrieval cache with semantic dedup
  2.8  Confidence calibration on retrieval scores
  2.9  Cross-collection retrieval
  2.10 SPLADE sparse retrieval integration (stub)
"""

# ⚠️ OVERLAPS: server/retrieval_enhancements.py provides RRF + temporal weighting
# This module provides contextual reranking, telemetry, and query decomposition.
# The temporal weighting here duplicates retrieval_enhancements.apply_temporal_weight.



import hashlib
import json
import re
import time as _time
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# 2.1  Contextual Reranking — expand each result with neighbors
# ---------------------------------------------------------------------------

def expand_with_neighbors(
    results: list[dict],
    collection,
    window: int = 1,
) -> list[dict]:
    """
    For each result chunk, fetch its previous/next neighbors from the same
    document to provide richer context for reranking and display.
    """
    expanded = []
    for r in results:
        sha256 = r.get("metadata", {}).get("sha256", "")
        chunk_idx = r.get("metadata", {}).get("chunk", -1)
        doc_text = r.get("document", "")

        # Fetch neighbors
        neighbor_texts = []
        if sha256 and chunk_idx >= 0 and collection:
            for offset in range(-window, window + 1):
                if offset == 0:
                    continue
                neighbor_id = f"{sha256}:{chunk_idx + offset}"
                try:
                    result = collection.get(ids=[neighbor_id], include=["documents"])
                    docs = result.get("documents", [])
                    if docs and docs[0]:
                        neighbor_texts.append(docs[0])
                except Exception:
                    pass

        expanded_doc = "\n\n".join(
            [t for t in neighbor_texts[:window]] + [doc_text] +
            [t for t in neighbor_texts[window:]]
        )

        r_copy = dict(r)
        r_copy["expanded_document"] = expanded_doc[:8000]
        r_copy["neighbor_count"] = len(neighbor_texts)
        expanded.append(r_copy)

    return expanded


# ---------------------------------------------------------------------------
# 2.2  Retrieval Quality Telemetry
# ---------------------------------------------------------------------------

class RetrievalTelemetry:
    """Track retrieval performance metrics for each query."""

    def __init__(self, log_path: Optional[Path] = None):
        self.log_path = log_path
        self.queries: list[dict] = []

    def record(
        self,
        query: str,
        results_count: int,
        top_score: float = 0.0,
        avg_score: float = 0.0,
        latency_ms: float = 0.0,
        strategy: str = "",
        reranked: bool = False,
        hyde_used: bool = False,
        bm25_used: bool = False,
        user_feedback: Optional[str] = None,
        cache_hit: bool = False,
    ):
        entry = {
            "timestamp": _time.strftime("%Y-%m-%dT%H:%M:%S"),
            "query": query[:200],
            "results_count": results_count,
            "top_score": round(top_score, 4),
            "avg_score": round(avg_score, 4),
            "latency_ms": round(latency_ms, 1),
            "strategy": strategy,
            "reranked": reranked,
            "hyde_used": hyde_used,
            "bm25_used": bm25_used,
            "user_feedback": user_feedback,
            "cache_hit": cache_hit,
        }
        self.queries.append(entry)

        if self.log_path:
            try:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(entry) + "\n")
            except Exception:
                pass

    @property
    def summary(self) -> dict:
        if not self.queries:
            return {"total_queries": 0}
        latencies = [q["latency_ms"] for q in self.queries if q["latency_ms"] > 0]
        scores = [q["top_score"] for q in self.queries if q["top_score"] > 0]
        return {
            "total_queries": len(self.queries),
            "avg_latency_ms": round(sum(latencies) / max(len(latencies), 1), 1),
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 1),
            "avg_top_score": round(sum(scores) / max(len(scores), 1), 4),
            "cache_hit_rate": round(
                sum(1 for q in self.queries if q.get("cache_hit")) / max(len(self.queries), 1), 3
            ),
        }


# ---------------------------------------------------------------------------
# 2.3  Adaptive top_k Based on Query Complexity
# ---------------------------------------------------------------------------

def adaptive_top_k(
    query: str,
    base_k: int = 10,
    min_k: int = 5,
    max_k: int = 30,
) -> int:
    """
    Dynamically adjust top_k based on query complexity:
    - Simple factual queries → fewer results (tighter focus)
    - Complex multi-faceted queries → more results (broader coverage)
    """
    words = query.split()
    word_count = len(words)

    # Question word analysis
    question_words = {"what", "who", "when", "where", "why", "how"}
    q_count = sum(1 for w in words if w.lower().rstrip("?") in question_words)

    # Multi-part indicators
    multi_part_words = {"and", "or", "also", "both", "versus", "vs", "compare", "contrast",
                        "difference", "between", "relationship", "relate"}
    multi_count = sum(1 for w in words if w.lower() in multi_part_words)

    # Specificity indicators (proper nouns, years, numbers)
    specific_count = sum(1 for w in words if w[0].isupper() or re.match(r"\d{4}", w))

    # Compute adjustment
    complexity_score = 0
    complexity_score += min(word_count / 10, 2.0)  # Longer = more complex
    complexity_score += multi_count * 1.5           # Multi-part = more results
    complexity_score += max(0, q_count - 1) * 0.5   # Multiple questions = more
    complexity_score -= specific_count * 0.3         # Specific = fewer needed

    adjusted_k = int(base_k + complexity_score * 3)
    return max(min_k, min(adjusted_k, max_k))


# ---------------------------------------------------------------------------
# 2.4  Pre-computed Document Summaries (for result previews)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# 2.5  Query Decomposition for Multi-Part Questions
# ---------------------------------------------------------------------------

def decompose_query(query: str) -> list[str]:
    """
    Split a complex multi-part query into sub-queries.
    Rules-based (no LLM call for speed).
    """
    # Already simple
    words = query.split()
    if len(words) <= 5:
        return [query]

    sub_queries = []

    # Split on explicit conjunctions
    parts = re.split(r"\band\b|\bor\b|\balso\b|;|\?(?=\s)|,\s*(?=\w{3,})", query, flags=re.I)
    parts = [p.strip().rstrip("?.,") for p in parts if len(p.strip()) > 10]

    if len(parts) > 1:
        sub_queries = parts
    else:
        # Try splitting on comparison patterns
        compare_match = re.search(
            r"(.*?)\b(versus|vs\.?|compared?\s+to|differ(?:ence|ent)\s+(?:from|between))\b(.*)",
            query, re.I
        )
        if compare_match:
            part1 = compare_match.group(1).strip()
            part2 = compare_match.group(3).strip()
            if part1 and part2:
                sub_queries = [part1, part2, query]  # Include original for context
            else:
                sub_queries = [query]
        else:
            sub_queries = [query]

    return sub_queries[:5]  # Cap at 5 sub-queries


# ---------------------------------------------------------------------------
# 2.6  Temporal Weighting
# NOTE: Canonical implementation is in retrieval_enhancements.py.
#       This re-export preserves backward compatibility.
# ---------------------------------------------------------------------------

from server.retrieval_enhancements import apply_temporal_weight as apply_temporal_weight  # noqa: F401


# ---------------------------------------------------------------------------
# 2.7  Retrieval Cache with Semantic Dedup
# ---------------------------------------------------------------------------

class RetrievalCache:
    """
    LRU-style cache for retrieval results.
    Uses query hash for exact match, with TTL expiration.
    """

    def __init__(self, max_size: int = 500, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl = ttl_seconds
        self._cache: dict[str, dict] = {}

    def _query_hash(self, query: str, top_k: int) -> str:
        key = f"{query.lower().strip()}:{top_k}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def get(self, query: str, top_k: int) -> Optional[list[dict]]:
        """Return cached results if available and not expired."""
        h = self._query_hash(query, top_k)
        entry = self._cache.get(h)
        if not entry:
            return None
        if _time.time() - entry["timestamp"] > self.ttl:
            del self._cache[h]
            return None
        return entry["results"]

    def put(self, query: str, top_k: int, results: list[dict]):
        """Cache retrieval results."""
        if len(self._cache) >= self.max_size:
            # Evict oldest
            oldest_key = min(self._cache, key=lambda k: self._cache[k]["timestamp"])
            del self._cache[oldest_key]

        h = self._query_hash(query, top_k)
        self._cache[h] = {
            "results": results,
            "timestamp": _time.time(),
            "query": query[:200],
        }

    def clear(self):
        self._cache.clear()

    @property
    def stats(self) -> dict:
        return {
            "size": len(self._cache),
            "max_size": self.max_size,
            "ttl": self.ttl,
        }


# ---------------------------------------------------------------------------
# 2.8  Confidence Calibration
# ---------------------------------------------------------------------------

def calibrate_confidence(
    results: list[dict],
    min_threshold: float = 0.3,
    high_confidence: float = 0.7,
) -> list[dict]:
    """
    Map raw retrieval scores to calibrated confidence levels.
    Adds a `confidence` field: 'high', 'medium', 'low', or 'uncertain'.
    """
    if not results:
        return results

    scores = [r.get("score", 0) for r in results]
    max_score = max(scores) if scores else 1.0
    min_score = min(scores) if scores else 0.0
    score_range = max_score - min_score if max_score > min_score else 1.0

    calibrated = []
    for r in results:
        r_copy = dict(r)
        score = r_copy.get("score", 0.0)

        # Normalize to 0-1 range
        normalized = (score - min_score) / score_range if score_range > 0 else 0.5

        if normalized >= high_confidence:
            r_copy["confidence"] = "high"
        elif normalized >= min_threshold:
            r_copy["confidence"] = "medium"
        elif normalized >= min_threshold * 0.5:
            r_copy["confidence"] = "low"
        else:
            r_copy["confidence"] = "uncertain"

        r_copy["confidence_score"] = round(normalized, 4)
        calibrated.append(r_copy)

    return calibrated


# ---------------------------------------------------------------------------
# 2.9  Cross-Collection Retrieval
# ---------------------------------------------------------------------------

try:
    from server.chroma_backend import retrieve_cross_collection as cross_collection_retrieve
except ImportError:
    def cross_collection_retrieve(*args, **kwargs):
        """Fallback stub when chroma_backend is unavailable."""
        return []

# ---------------------------------------------------------------------------
# 2.10  SPLADE Sparse Retrieval (Stub — requires splade model)
# ---------------------------------------------------------------------------

class SPLADERetriever:
    """
    SPLADE sparse retrieval stub.
    Full implementation requires: pip install transformers torch
    and downloading a SPLADE model checkpoint.
    """

    def __init__(self, model_name: str = "naver/splade-cocondenser-ensembledistil"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> bool:
        """Lazy-load the SPLADE model."""
        if self._loaded:
            return True
        try:
            from transformers import AutoModelForMaskedLM, AutoTokenizer  # type: ignore
            import torch  # type: ignore
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForMaskedLM.from_pretrained(self.model_name)
            self.model.eval()
            self._loaded = True
            return True
        except Exception:
            return False

    def encode(self, text: str) -> dict[str, float]:
        """Encode text into a sparse SPLADE vector (term weights)."""
        if not self._loaded:
            return {}
        try:
            import torch  # type: ignore
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                output = self.model(**tokens)
            logits = output.logits
            weights = torch.max(torch.log1p(torch.relu(logits)), dim=1).values[0]
            non_zero = torch.nonzero(weights).squeeze(-1)
            sparse = {}
            for idx in non_zero:
                token = self.tokenizer.decode([idx.item()])
                weight = weights[idx.item()].item()
                if weight > 0.1:
                    sparse[token.strip()] = round(weight, 4)
            return sparse
        except Exception:
            return {}

    def score(self, query_sparse: dict, doc_sparse: dict) -> float:
        """Compute dot product between query and document sparse vectors."""
        score = 0.0
        for term, q_weight in query_sparse.items():
            if term in doc_sparse:
                score += q_weight * doc_sparse[term]
        return score


# ---------------------------------------------------------------------------
# Integration: create retrieval context for server startup
# ---------------------------------------------------------------------------

def create_retrieval_context(app_state: Path) -> dict:
    """
    Create all retrieval improvement objects.
    Called during server startup.
    """
    telemetry = RetrievalTelemetry(
        log_path=app_state / "edith_retrieval_telemetry.jsonl"
    )
    cache = RetrievalCache(max_size=500, ttl_seconds=3600)

    return {
        "telemetry": telemetry,
        "cache": cache,
        "decompose_query": decompose_query,
        "adaptive_top_k": adaptive_top_k,
        "expand_with_neighbors": expand_with_neighbors,
        "apply_temporal_weight": apply_temporal_weight,
        "calibrate_confidence": calibrate_confidence,
        "cross_collection_retrieve": cross_collection_retrieve,
    }
