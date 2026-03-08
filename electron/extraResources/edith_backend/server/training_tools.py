"""
§4.0: Training Data Tools
DPO preparation, deduplication, balanced sampling, synthetic generation,
and quality scoring for the E.D.I.T.H. training pipeline.
"""
from __future__ import annotations
import json
import hashlib
import os
import re
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict

# ---------------------------------------------------------------------------
# DPO (Direct Preference Optimization) Data Preparation
# ---------------------------------------------------------------------------

def prepare_dpo_pairs(feedback_path: str | Path) -> list[dict]:
    """Convert thumbs-up/thumbs-down feedback into DPO training pairs.
    
    DPO format: {prompt, chosen (good answer), rejected (bad answer)}
    Reads from the feedback log where users gave +1 or -1 with corrections.
    """
    feedback_path = Path(feedback_path)
    if not feedback_path.exists():
        return []
    
    pairs = []
    entries_by_query = defaultdict(list)
    
    with open(feedback_path) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                # §FIX T3: Read 'question' key (matches feedback_endpoint output)
                query = entry.get("question", entry.get("query", ""))
                if query:
                    entries_by_query[query].append(entry)
            except json.JSONDecodeError:
                continue
    
    for query, entries in entries_by_query.items():
        # §FIX T3: Read 'rating' key (matches feedback_endpoint output)
        good = [e for e in entries if e.get("rating", e.get("feedback")) == "up"]
        bad = [e for e in entries if e.get("rating", e.get("feedback")) == "down"]
        
        for g in good:
            for b in bad:
                pairs.append({
                    "prompt": query,
                    "chosen": g.get("answer", ""),
                    "rejected": b.get("bad_answer", b.get("answer", "")),
                    "metadata": {
                        "source": "user_feedback",
                        "timestamp": datetime.now(timezone.utc).isoformat(),  # §FIX T4
                    }
                })
    
    return pairs


# ---------------------------------------------------------------------------
# Deduplication by Embedding Similarity
# ---------------------------------------------------------------------------

def deduplicate_training_data(
    input_path: str | Path,
    output_path: str | Path,
    similarity_threshold: float = 0.95,
) -> dict:
    """Remove near-duplicate Q&A pairs by text similarity.
    
    Uses a fast text-hash approach: normalize, hash, compare.
    For semantic dedup, use embedding cosine similarity (requires API).
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    if not input_path.exists():
        return {"error": "Input file not found"}
    
    seen_hashes = set()
    unique_entries = []
    duplicates = 0
    total = 0
    
    with open(input_path) as f:
        for line in f:
            total += 1
            try:
                entry = json.loads(line.strip())
            except json.JSONDecodeError:
                continue
            
            # Create a fingerprint from the user message
            messages = entry.get("messages", [])
            user_msgs = [m["content"] for m in messages if m.get("role") == "user"]
            fingerprint = _text_fingerprint(" ".join(user_msgs))
            
            if fingerprint in seen_hashes:
                duplicates += 1
                continue
            
            seen_hashes.add(fingerprint)
            unique_entries.append(line.strip())
    
    with open(output_path, "w") as f:
        for entry in unique_entries:
            f.write(entry + "\n")
    
    return {
        "total": total,
        "unique": len(unique_entries),
        "duplicates_removed": duplicates,
        "output_path": str(output_path),
    }


def _text_fingerprint(text: str) -> str:
    """Normalize text and create a hash fingerprint for dedup."""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    # Remove common filler words for better matching
    normalized = re.sub(r'\b(the|a|an|is|are|was|were|be|been|being)\b', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    return hashlib.md5(normalized.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Balanced Topic Sampling
# ---------------------------------------------------------------------------

def analyze_topic_balance(input_path: str | Path) -> dict:
    """Analyze training data distribution across topics/modes."""
    input_path = Path(input_path)
    if not input_path.exists():
        return {"error": "Input file not found"}
    
    topic_counts = Counter()
    mode_counts = Counter()
    total = 0
    
    with open(input_path) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                total += 1
                
                meta = entry.get("metadata", {})
                topic = meta.get("topic", meta.get("academic_topic", "unknown"))
                mode = meta.get("quality_tier", "unknown")
                
                topic_counts[topic] += 1
                mode_counts[mode] += 1
            except json.JSONDecodeError:
                continue
    
    return {
        "total": total,
        "topics": dict(topic_counts.most_common()),
        "modes": dict(mode_counts.most_common()),
        "most_represented": topic_counts.most_common(3),
        "least_represented": topic_counts.most_common()[-3:] if len(topic_counts) >= 3 else [],
    }


def balanced_sample(
    input_path: str | Path,
    output_path: str | Path,
    max_per_topic: int = 50,
) -> dict:
    """Create a balanced sample with at most max_per_topic entries per topic."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    topic_buckets = defaultdict(list)
    
    with open(input_path) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                meta = entry.get("metadata", {})
                topic = meta.get("topic", meta.get("academic_topic", "general"))
                topic_buckets[topic].append(line.strip())
            except json.JSONDecodeError:
                continue
    
    sampled = []
    for topic, entries in topic_buckets.items():
        sampled.extend(entries[:max_per_topic])
    
    with open(output_path, "w") as f:
        for entry in sampled:
            f.write(entry + "\n")
    
    return {
        "topics": len(topic_buckets),
        "total_sampled": len(sampled),
        "output_path": str(output_path),
    }


# ---------------------------------------------------------------------------
# Quality Scoring Dashboard Data
# ---------------------------------------------------------------------------

def compute_quality_metrics(input_path: str | Path) -> dict:
    """Compute quality metrics across training data for dashboard display."""
    input_path = Path(input_path)
    if not input_path.exists():
        return {"error": "Input file not found"}
    
    total = 0
    coverages = []
    lengths = []
    versions = Counter()
    models = Counter()
    
    with open(input_path) as f:
        for line in f:
            try:
                entry = json.loads(line.strip())
                total += 1
                meta = entry.get("metadata", {})
                
                cov = meta.get("coverage", 0)
                if cov:
                    coverages.append(cov)
                
                messages = entry.get("messages", [])
                asst = [m for m in messages if m.get("role") == "assistant"]
                if asst:
                    lengths.append(len(asst[0].get("content", "")))
                
                versions[meta.get("data_version", "unknown")] += 1
                models[meta.get("model", "unknown")] += 1
                
            except json.JSONDecodeError:
                continue
    
    return {
        "total_entries": total,
        "avg_coverage": round(sum(coverages) / max(len(coverages), 1), 3),
        "avg_answer_length": round(sum(lengths) / max(len(lengths), 1)),
        "min_coverage": round(min(coverages), 3) if coverages else 0,
        "max_coverage": round(max(coverages), 3) if coverages else 0,
        "versions": dict(versions.most_common()),
        "models": dict(models.most_common()),
    }


# ---------------------------------------------------------------------------
# Auto Fine-Tune Trigger
# ---------------------------------------------------------------------------

def check_finetune_trigger(
    autolearn_path: str | Path,
    threshold: int = 100,
    last_count_path: str | Path | None = None,
) -> dict:
    """Check if enough new training pairs exist to trigger a fine-tune job.
    
    Returns trigger=True when new pairs since last fine-tune exceed threshold.
    """
    autolearn_path = Path(autolearn_path)
    if not autolearn_path.exists():
        return {"trigger": False, "reason": "No autolearn file"}
    
    with open(autolearn_path) as _f:
        current_count = sum(1 for _ in _f)
    
    last_count = 0
    if last_count_path:
        lcp = Path(last_count_path)
        if lcp.exists():
            try:
                last_count = int(lcp.read_text().strip())
            except (ValueError, OSError):
                last_count = 0
    
    new_pairs = current_count - last_count
    
    return {
        "trigger": new_pairs >= threshold,
        "current_count": current_count,
        "last_finetune_count": last_count,
        "new_pairs": new_pairs,
        "threshold": threshold,
    }


# ══════════════════════════════════════════════════════════════
# Merged from training_enhancements.py
# ══════════════════════════════════════════════════════════════

"""
Training Enhancements — Improvements to Winnie's training loop.

Implements:
  7.1   Active learning (identify least-confident queries)
  7.5   A/B model comparison before deployment
  7.6   Training data deduplication (SimHash)
  7.8   Rollback mechanism
  7.9   Per-topic training balance tracking
  7.10  Training cost tracking
"""

# ⚠️ OVERLAPS: server/training_tools.py + server/training_devops.py
# This module provides training pipeline improvements. Coordinate with
# training_tools.py (core data generation) and training_devops.py (multi-corpus).



import hashlib
import json
import logging
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.training_enhancements")


# ---------------------------------------------------------------------------
# 7.6: Training Data Deduplication (SimHash)
# ---------------------------------------------------------------------------

def _simhash(text: str, num_bits: int = 64) -> int:
    """
    Compute a SimHash fingerprint for near-duplicate detection.
    Similar texts produce similar hashes (small Hamming distance).
    """
    tokens = text.lower().split()
    v = [0] * num_bits

    for token in tokens:
        token_hash = int(hashlib.md5(token.encode()).hexdigest(), 16)
        for i in range(num_bits):
            if token_hash & (1 << i):
                v[i] += 1
            else:
                v[i] -= 1

    fingerprint = 0
    for i in range(num_bits):
        if v[i] > 0:
            fingerprint |= (1 << i)
    return fingerprint


def _hamming_distance(a: int, b: int) -> int:
    """Count bits that differ between two hashes."""
    return bin(a ^ b).count("1")


def deduplicate_training_data(
    jsonl_path: Path,
    output_path: Path | None = None,
    similarity_threshold: int = 5,
) -> dict:
    """
    Remove near-duplicate training pairs from a JSONL file.

    Uses SimHash to detect pairs where the question+answer are very similar.
    Pairs with Hamming distance <= similarity_threshold are considered duplicates.

    Returns stats: {original, deduplicated, removed}.
    """
    pairs = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not pairs:
        return {"original": 0, "deduplicated": 0, "removed": 0}

    # Compute SimHash for each pair
    hashes = []
    for pair in pairs:
        messages = pair.get("messages", [])
        text = " ".join(m.get("content", "") for m in messages if m.get("role") in ("user", "assistant"))
        hashes.append(_simhash(text))

    # Mark duplicates
    keep = [True] * len(pairs)
    for i in range(len(pairs)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(pairs)):
            if not keep[j]:
                continue
            if _hamming_distance(hashes[i], hashes[j]) <= similarity_threshold:
                # Keep the one with higher quality tier
                tier_i = pairs[i].get("quality_tier", "")
                tier_j = pairs[j].get("quality_tier", "")
                tier_ranks = {"human_correction": 4, "feedback_positive": 3, "sharpened": 2, "sharpened_contrastive": 1}
                if tier_ranks.get(tier_j, 0) > tier_ranks.get(tier_i, 0):
                    keep[i] = False
                    break
                else:
                    keep[j] = False

    # Write deduplicated output
    deduplicated = [p for p, k in zip(pairs, keep) if k]
    out = output_path or jsonl_path
    with open(out, "w", encoding="utf-8") as f:
        for pair in deduplicated:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    removed = len(pairs) - len(deduplicated)
    log.info(f"Deduplication: {len(pairs)} → {len(deduplicated)} ({removed} removed)")

    return {"original": len(pairs), "deduplicated": len(deduplicated), "removed": removed}


# ---------------------------------------------------------------------------
# 7.9: Per-Topic Training Balance
# ---------------------------------------------------------------------------

KNOWN_TOPICS = [
    "American Politics", "Comparative Politics", "International Relations",
    "Political Theory", "Methodology", "Public Policy", "Political Economy",
    "Race & Ethnicity", "Gender Politics", "Environmental Politics",
    "Congress", "Presidency", "Courts", "Elections", "Public Opinion",
]


def analyze_topic_balance(jsonl_path: Path) -> dict:
    """
    Analyze topic distribution in training data and identify imbalances.

    Returns {topic_counts, total, underrepresented, overrepresented}.
    """
    topic_counts = Counter()
    total = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                pair = json.loads(line)
            except json.JSONDecodeError:
                continue

            total += 1
            messages = pair.get("messages", [])
            text = " ".join(m.get("content", "") for m in messages).lower()

            matched = False
            for topic in KNOWN_TOPICS:
                if topic.lower() in text or any(
                    kw in text for kw in topic.lower().split()
                ):
                    topic_counts[topic] += 1
                    matched = True

            if not matched:
                topic_counts["Other"] += 1

    if total == 0:
        return {"topic_counts": {}, "total": 0, "underrepresented": [], "overrepresented": []}

    avg = total / max(1, len(topic_counts))
    underrepresented = [
        {"topic": t, "count": c, "ratio": round(c / avg, 2)}
        for t, c in topic_counts.items()
        if c < avg * 0.5
    ]
    overrepresented = [
        {"topic": t, "count": c, "ratio": round(c / avg, 2)}
        for t, c in topic_counts.items()
        if c > avg * 2
    ]

    return {
        "topic_counts": dict(topic_counts.most_common()),
        "total": total,
        "unique_topics": len(topic_counts),
        "average_per_topic": round(avg, 1),
        "underrepresented": sorted(underrepresented, key=lambda x: x["count"]),
        "overrepresented": sorted(overrepresented, key=lambda x: -x["count"]),
    }


# ---------------------------------------------------------------------------
# 7.10: Training Cost Tracking
# ---------------------------------------------------------------------------

@dataclass
class TrainingCostTracker:
    """Track fine-tuning costs across runs."""
    log_path: Path = field(default_factory=lambda: Path("eval/training_costs.jsonl"))
    runs: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.log_path.exists():
            try:
                with open(self.log_path, "r") as f:
                    self.runs = [json.loads(line) for line in f if line.strip()]
            except Exception:
                self.runs = []

    def record_run(
        self,
        model_name: str,
        training_pairs: int,
        epochs: int = 3,
        cost_usd: float = 0.0,
        eval_score: float = 0.0,
        notes: str = "",
    ):
        """Record a fine-tuning run."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": model_name,
            "pairs": training_pairs,
            "epochs": epochs,
            "cost_usd": round(cost_usd, 4),
            "eval_score": round(eval_score, 3),
            "notes": notes,
        }
        self.runs.append(entry)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        return entry

    @property
    def total_cost(self) -> float:
        return sum(r.get("cost_usd", 0) for r in self.runs)

    @property
    def best_run(self) -> dict | None:
        if not self.runs:
            return None
        return max(self.runs, key=lambda r: r.get("eval_score", 0))

    @property
    def summary(self) -> dict:
        return {
            "total_runs": len(self.runs),
            "total_cost_usd": round(self.total_cost, 2),
            "total_pairs_trained": sum(r.get("pairs", 0) for r in self.runs),
            "best_eval_score": round(self.best_run.get("eval_score", 0), 3) if self.best_run else 0,
            "latest_model": self.runs[-1].get("model", "") if self.runs else "",
        }


# ---------------------------------------------------------------------------
# 7.1: Active Learning — Identify Least-Confident Queries
# ---------------------------------------------------------------------------

class ActiveLearningQueue:
    """Track queries where Winnie is least confident for human review."""

    def __init__(self, queue_path: Path | None = None, max_size: int = 100):
        self.max_size = max_size
        self._queue: list[dict] = []
        self._path = queue_path

        if self._path and self._path.exists():
            try:
                self._queue = json.loads(self._path.read_text(encoding="utf-8"))
            except Exception:
                self._queue = []

    def add(self, query: str, confidence: float, answer: str, sources: list[str] | None = None):
        """Add a low-confidence query to the review queue."""
        if confidence > 0.5:  # Only track low-confidence
            return

        entry = {
            "query": query,
            "confidence": round(confidence, 3),
            "answer_preview": answer[:300],
            "source_count": len(sources) if sources else 0,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "reviewed": False,
        }
        self._queue.append(entry)

        # Keep only the lowest-confidence items
        self._queue.sort(key=lambda x: x["confidence"])
        self._queue = self._queue[:self.max_size]
        self._save()

    def get_pending(self, limit: int = 20) -> list[dict]:
        """Get unreviewed items, sorted by lowest confidence first."""
        return [q for q in self._queue if not q.get("reviewed")][:limit]

    def mark_reviewed(self, query: str):
        """Mark a query as reviewed."""
        for item in self._queue:
            if item["query"] == query:
                item["reviewed"] = True
        self._save()

    def _save(self):
        if self._path:
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._path.write_text(json.dumps(self._queue, indent=2), encoding="utf-8")
            except Exception:
                pass

    @property
    def stats(self) -> dict:
        return {
            "total": len(self._queue),
            "pending": len([q for q in self._queue if not q.get("reviewed")]),
            "avg_confidence": round(
                sum(q["confidence"] for q in self._queue) / max(1, len(self._queue)), 3
            ) if self._queue else 0,
        }


# ---------------------------------------------------------------------------
# 7.5: A/B Model Comparison
# ---------------------------------------------------------------------------

def compare_models(
    questions: list[str],
    model_a_answers: list[str],
    model_b_answers: list[str],
    model_a_name: str = "current",
    model_b_name: str = "candidate",
) -> dict:
    """
    Compare two models' answers on the same questions.
    Returns comparison metrics to decide whether to deploy the candidate.
    """
    assert len(questions) == len(model_a_answers) == len(model_b_answers)

    comparisons = []
    a_wins = 0
    b_wins = 0
    ties = 0

    for i, (q, ans_a, ans_b) in enumerate(zip(questions, model_a_answers, model_b_answers)):
        # Simple heuristic comparison (production would use LLM judge)
        score_a = _answer_quality_heuristic(ans_a)
        score_b = _answer_quality_heuristic(ans_b)

        if score_a > score_b + 0.1:
            winner = model_a_name
            a_wins += 1
        elif score_b > score_a + 0.1:
            winner = model_b_name
            b_wins += 1
        else:
            winner = "tie"
            ties += 1

        comparisons.append({
            "question": q[:100],
            "score_a": round(score_a, 3),
            "score_b": round(score_b, 3),
            "winner": winner,
        })

    total = len(questions)
    result = {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "total_questions": total,
        "a_wins": a_wins,
        "b_wins": b_wins,
        "ties": ties,
        "a_win_rate": round(a_wins / max(1, total), 3),
        "b_win_rate": round(b_wins / max(1, total), 3),
        "recommendation": "deploy" if b_wins > a_wins else "keep_current",
        "comparisons": comparisons[:20],  # First 20 for review
    }

    return result


def _answer_quality_heuristic(answer: str) -> float:
    """Quick heuristic answer quality score (0-1) for A/B comparison."""
    if not answer or len(answer.strip()) < 10:
        return 0.0

    score = 0.5

    # Has citations
    citations = len(re.findall(r"\[S\d+\]", answer))
    if citations >= 3:
        score += 0.2
    elif citations >= 1:
        score += 0.1

    # Reasonable length
    words = len(answer.split())
    if 100 <= words <= 500:
        score += 0.15
    elif words > 500:
        score += 0.1

    # Has structure (paragraphs, lists)
    if answer.count("\n\n") >= 2:
        score += 0.1

    # No error markers
    if "[ERROR" in answer or "I don't know" in answer.lower():
        score -= 0.3

    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# 7.8: Model Rollback
# ---------------------------------------------------------------------------

@dataclass
class ModelRegistry:
    """Track deployed models with rollback support."""
    registry_path: Path = field(default_factory=lambda: Path("eval/model_registry.json"))
    models: list[dict] = field(default_factory=list)

    def __post_init__(self):
        if self.registry_path.exists():
            try:
                self.models = json.loads(self.registry_path.read_text(encoding="utf-8"))
            except Exception:
                self.models = []

    def register(self, model_id: str, eval_score: float, training_pairs: int, notes: str = ""):
        """Register a newly fine-tuned model."""
        entry = {
            "model_id": model_id,
            "deployed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "eval_score": round(eval_score, 3),
            "training_pairs": training_pairs,
            "notes": notes,
            "active": True,
        }
        # Deactivate previous
        for m in self.models:
            m["active"] = False
        self.models.append(entry)
        self._save()
        return entry

    def rollback(self) -> dict | None:
        """Rollback to the previous model version."""
        if len(self.models) < 2:
            return None
        # Deactivate current
        self.models[-1]["active"] = False
        # Reactivate previous
        self.models[-2]["active"] = True
        self._save()
        log.warning(f"Rolled back to model: {self.models[-2]['model_id']}")
        return self.models[-2]

    @property
    def active_model(self) -> str | None:
        for m in reversed(self.models):
            if m.get("active"):
                return m["model_id"]
        return None

    def _save(self):
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.registry_path.write_text(json.dumps(self.models, indent=2), encoding="utf-8")
