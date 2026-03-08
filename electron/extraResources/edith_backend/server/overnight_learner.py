"""
Overnight Learner — E.D.I.T.H.'s Daily Consolidation Engine
=============================================================
"Winnie gets smarter while you sleep."

Runs overnight (or on-demand) to:
  1. Consolidate the day's interactions
  2. Update the research profile
  3. Identify weak spots (low-confidence topics)
  4. Generate a morning briefing for the next session
"""
import json
import logging
import os
import time
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger("edith.overnight_learner")


def _get_data_root() -> Path:
    """Get the writable data root."""
    root = os.environ.get("EDITH_DATA_ROOT", "")
    if root:
        p = Path(root)
        if p.exists():
            # Check writability
            try:
                test_dir = p / ".edith"
                test_dir.mkdir(parents=True, exist_ok=True)
                return p
            except OSError:
                pass
    return Path.home()


# ═══════════════════════════════════════════════════════════════════
# §1: INTERACTION CONSOLIDATION
# Read autolearn + DPO data, extract patterns
# ═══════════════════════════════════════════════════════════════════

def consolidate_interactions(since_hours: int = 24) -> dict:
    """Analyze recent interactions and extract learning signals.

    Returns:
        {
            "total_interactions": int,
            "topics_discussed": {topic: count},
            "methods_used": [str],
            "weak_spots": [{topic, confidence, reason}],
            "strong_spots": [{topic, confidence}],
            "tools_used": {tool: count},
            "avg_confidence": float,
            "training_pairs_generated": int,
        }
    """
    root = _get_data_root()
    cutoff = datetime.now() - timedelta(hours=since_hours)
    cutoff_iso = cutoff.isoformat()

    topics = {}
    methods = set()
    confidences = []
    tools = {}
    total = 0
    training_pairs = 0

    # Read autolearn data
    autolearn_path = root / "autolearn.jsonl"
    if autolearn_path.exists():
        try:
            with open(autolearn_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        meta = entry.get("metadata", {})
                        ts = meta.get("timestamp", "")
                        if ts >= cutoff_iso:
                            total += 1
                            training_pairs += 1
                            coverage = meta.get("coverage", 0)
                            confidences.append(coverage)

                            # Extract topic from user message
                            msgs = entry.get("messages", [])
                            for m in msgs:
                                if m.get("role") == "user":
                                    topic = _extract_topic_keywords(m.get("content", ""))
                                    for t in topic:
                                        topics[t] = topics.get(t, 0) + 1
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass

    # Read DPO negatives
    dpo_path = root / "training_data" / "dpo_negatives.jsonl"
    if dpo_path.exists():
        try:
            with open(dpo_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        ts = entry.get("timestamp", "")
                        if ts >= cutoff_iso:
                            total += 1
                            conf = entry.get("source_count", 0) / 5.0
                            confidences.append(conf)
                    except json.JSONDecodeError:
                        continue
        except OSError:
            pass

    # Read research profile for tool usage
    try:
        from server.research_profile import get_profile
        profile = get_profile()
        tools = profile.get_profile().get("tool_usage", {})
        recent_queries = profile.get_profile().get("recent_queries", [])
        for q in recent_queries:
            if q.get("ts", "") >= cutoff_iso:
                total += 1
    except Exception:
        pass

    # Identify weak and strong spots
    weak_spots = []
    strong_spots = []
    for topic, count in sorted(topics.items(), key=lambda x: x[1], reverse=True):
        # Weak = discussed often but low confidence
        if count >= 2:
            strong_spots.append({"topic": topic, "count": count})
        else:
            weak_spots.append({"topic": topic, "count": count, "reason": "low frequency"})

    avg_conf = sum(confidences) / max(len(confidences), 1)

    return {
        "total_interactions": total,
        "topics_discussed": topics,
        "methods_used": sorted(methods),
        "weak_spots": weak_spots[:10],
        "strong_spots": strong_spots[:10],
        "tools_used": tools,
        "avg_confidence": round(avg_conf, 3),
        "training_pairs_generated": training_pairs,
        "period_hours": since_hours,
        "consolidated_at": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════════════════════════
# §2: PROFILE UPDATE
# Push consolidated insights into the research profile
# ═══════════════════════════════════════════════════════════════════

def update_profile_from_consolidation(consolidation: dict):
    """Push consolidated learning into the research profile."""
    try:
        from server.research_profile import get_profile
        profile = get_profile()

        # Record frequently discussed topics
        for topic, count in consolidation.get("topics_discussed", {}).items():
            for _ in range(count):
                profile.record_topic(topic)

        # Record methods
        for method in consolidation.get("methods_used", []):
            profile.record_method(method)

        profile.force_save()
        log.info("Overnight: profile updated from consolidation")
    except Exception as e:
        log.warning(f"Overnight: profile update failed: {e}")


# ═══════════════════════════════════════════════════════════════════
# §3: MORNING BRIEFING
# Generate a personalized briefing for the next session
# ═══════════════════════════════════════════════════════════════════

def generate_briefing(consolidation: dict = None) -> dict:
    """Generate a morning briefing based on recent activity.

    Returns a structured briefing with:
      - Summary of yesterday's work
      - Suggested next steps
      - Weak spots to revisit
      - Papers to read
      - Methods to practice
    """
    if consolidation is None:
        consolidation = consolidate_interactions(since_hours=24)

    # Get profile context
    profile_ctx = ""
    try:
        from server.research_profile import get_profile
        profile_ctx = get_profile().get_context_string()
    except Exception:
        pass

    # Build briefing sections
    sections = []

    # Yesterday's summary
    total = consolidation.get("total_interactions", 0)
    topics = consolidation.get("topics_discussed", {})
    avg_conf = consolidation.get("avg_confidence", 0)

    if total > 0:
        top_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5]
        topic_str = ", ".join(f"{t} ({c}x)" for t, c in top_topics) if top_topics else "general research"
        sections.append({
            "title": "Yesterday's Work",
            "content": f"{total} interactions across {len(topics)} topics. "
                       f"Focus areas: {topic_str}. "
                       f"Average confidence: {avg_conf:.0%}.",
        })

    # Weak spots
    weak = consolidation.get("weak_spots", [])
    if weak:
        weak_str = ", ".join(w["topic"] for w in weak[:5])
        sections.append({
            "title": "Areas to Strengthen",
            "content": f"Consider revisiting: {weak_str}. "
                       f"These topics had limited source coverage.",
        })

    # Strong spots
    strong = consolidation.get("strong_spots", [])
    if strong:
        strong_str = ", ".join(s["topic"] for s in strong[:3])
        sections.append({
            "title": "Your Strengths",
            "content": f"Deep knowledge in: {strong_str}.",
        })

    # Suggested next steps (LLM-generated if available)
    suggestions = []
    if topics:
        top_topic = max(topics, key=topics.get)
        suggestions.append(f"Continue exploring '{top_topic}' — you were making progress")
    if weak:
        suggestions.append(f"Review the literature on '{weak[0]['topic']}' to build confidence")
    if avg_conf < 0.5:
        suggestions.append("Index more papers in your library to improve source coverage")

    if suggestions:
        sections.append({
            "title": "Suggested Next Steps",
            "content": "\n".join(f"• {s}" for s in suggestions),
        })

    # Training data status
    pairs = consolidation.get("training_pairs_generated", 0)
    if pairs > 0:
        sections.append({
            "title": "Learning Progress",
            "content": f"Generated {pairs} training pairs. "
                       f"Winnie is learning from your interactions.",
        })

    # Save briefing to disk
    briefing = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "generated_at": datetime.now().isoformat(),
        "sections": sections,
        "consolidation_summary": {
            "total_interactions": total,
            "topic_count": len(topics),
            "avg_confidence": avg_conf,
            "training_pairs": pairs,
        },
    }

    try:
        briefing_dir = _get_data_root() / ".edith" / "briefings"
        briefing_dir.mkdir(parents=True, exist_ok=True)
        briefing_path = briefing_dir / f"briefing_{briefing['date']}.json"
        tmp = briefing_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(briefing, f, indent=2)
        tmp.rename(briefing_path)
    except OSError as e:
        log.warning(f"Could not save briefing: {e}")

    return briefing


# ═══════════════════════════════════════════════════════════════════
# §4: FULL OVERNIGHT RUN
# ═══════════════════════════════════════════════════════════════════

def run_overnight(since_hours: int = 24) -> dict:
    """Run the full overnight learning loop.

    1. Consolidate interactions
    2. Update profile
    3. Generate briefing
    """
    t0 = time.time()
    log.info("§OVERNIGHT: Starting learning loop...")

    consolidation = consolidate_interactions(since_hours=since_hours)
    update_profile_from_consolidation(consolidation)
    briefing = generate_briefing(consolidation)

    elapsed = round(time.time() - t0, 2)
    log.info(f"§OVERNIGHT: Complete in {elapsed}s — "
             f"{consolidation['total_interactions']} interactions, "
             f"{consolidation['training_pairs_generated']} training pairs")

    return {
        "consolidation": consolidation,
        "briefing": briefing,
        "elapsed_s": elapsed,
    }


# ═══════════════════════════════════════════════════════════════════
# §5: TOPIC EXTRACTION HELPER
# ═══════════════════════════════════════════════════════════════════

def _extract_topic_keywords(text: str) -> list[str]:
    """Extract topic keywords from text."""
    import re
    # Remove common question words
    text = text.lower().strip()
    for w in ["what is", "how does", "why does", "can you", "please",
              "explain", "analyze", "describe", "compare"]:
        text = text.replace(w, "")

    # Extract multi-word phrases
    words = re.findall(r"\b[a-z]{3,}\b", text)
    # Filter out common stopwords
    stops = {"the", "and", "for", "are", "but", "not", "you", "all", "can",
             "had", "her", "was", "one", "our", "out", "has", "how", "its",
             "may", "they", "this", "that", "with", "will", "each", "from",
             "have", "been", "some", "about", "which", "their", "would", "there"}
    keywords = [w for w in words if w not in stops]

    # Return the most meaningful phrases (bigrams)
    if len(keywords) >= 2:
        bigrams = [f"{keywords[i]} {keywords[i+1]}" for i in range(len(keywords)-1)]
        return bigrams[:3]
    return keywords[:3]
