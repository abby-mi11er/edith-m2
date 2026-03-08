"""
Edith Session Memory — Cross-session conversation context.

Remembers key topics, document references, and decisions from past sessions.
Loads recent session summaries into the system prompt so Edith can say
"Last time you asked about Northweingast, I found it in your IR Pro-Sem folder."

§CE-13: Conversation summarization — auto-compress long chats
§CE-14: Topic segmentation — detect topic shifts within sessions
§CE-15: Memory priority tagging — pin important messages
"""

import json
import os
import re
import time
import logging
from pathlib import Path
from datetime import datetime
from collections import defaultdict

log = logging.getLogger("edith.session_memory")

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)
except ImportError:
    pass

APP_STATE = Path(
    os.environ.get("EDITH_APP_DATA_DIR", str(Path(__file__).resolve().parent.parent))
).expanduser().resolve()

MEMORY_FILE = APP_STATE / "session_summaries.jsonl"
PINNED_FILE = APP_STATE / "pinned_memories.json"
MAX_SUMMARIES_IN_CONTEXT = 5
MAX_SUMMARY_CHARS = 800


def save_session_summary(
    chat_id: str,
    messages: list,
    model: str = "",
    source_mode: str = "",
):
    """Summarize a chat session and append to session memory."""
    if not messages or len(messages) < 2:
        return

    # Extract key info from the conversation
    user_msgs = [m for m in messages if m.get("role") == "user"]
    asst_msgs = [m for m in messages if m.get("role") == "assistant"]

    if not user_msgs:
        return

    # Extract topics from user messages
    topics = []
    for m in user_msgs[:10]:  # Cap at first 10 messages
        text = (m.get("text") or m.get("content") or "")[:200]
        if text:
            topics.append(text)

    # Extract referenced documents
    doc_refs = set()
    for m in asst_msgs:
        sources = m.get("sources") or []
        for s in sources:
            if isinstance(s, dict):
                fname = s.get("filename") or s.get("source") or ""
                if fname:
                    doc_refs.add(Path(fname).name)

    # §CE-14: Detect topic segments within the session
    segments = detect_topic_segments(messages)

    # Build summary
    topic_summary = " | ".join(topics[:5])
    docs_list = list(doc_refs)[:10]

    entry = {
        "chat_id": str(chat_id),
        "timestamp": datetime.now().isoformat(),
        "turn_count": len(messages),
        "topics": topic_summary[:MAX_SUMMARY_CHARS],
        "documents_referenced": docs_list,
        "source_mode": source_mode,
        "model": model,
        "topic_segments": segments,  # §CE-14
    }

    try:
        MEMORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with MEMORY_FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        # Auto-prune to prevent unbounded growth
        prune_old_summaries()
    except Exception:
        pass


def load_recent_summaries(n: int = MAX_SUMMARIES_IN_CONTEXT) -> list:
    """Load the N most recent session summaries."""
    if not MEMORY_FILE.exists():
        return []
    try:
        lines = MEMORY_FILE.read_text(encoding="utf-8").strip().split("\n")
        summaries = []
        for line in lines[-n:]:
            if line.strip():
                try:
                    summaries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return summaries
    except Exception:
        return []


def format_memory_context() -> str:
    """Format recent session summaries as context for the system prompt.

    Now includes pinned memories (§CE-15) at the top for priority.
    """
    # §CE-15: Load pinned memories first
    pinned = load_pinned_memories()
    summaries = load_recent_summaries()

    if not summaries and not pinned:
        return ""

    lines = ["## Research Memory (for context continuity)"]

    # Pinned memories first
    if pinned:
        lines.append("\n### Pinned (important)")
        for p in pinned[:5]:
            ts = p.get("timestamp", "")[:16].replace("T", " ")
            text = p.get("text", "")[:200]
            lines.append(f"- **[PINNED {ts}]** {text}")

    # Recent sessions
    if summaries:
        lines.append("\n### Recent Sessions")
        for s in summaries:
            ts = s.get("timestamp", "")[:16].replace("T", " ")
            topics = s.get("topics", "")
            docs = s.get("documents_referenced", [])
            turns = s.get("turn_count", 0)
            mode = s.get("source_mode", "")

            entry = f"- **{ts}** ({turns} turns, {mode}): {topics}"
            if docs:
                entry += f"\n  Referenced: {', '.join(docs[:5])}"
            lines.append(entry)

    lines.append("")
    lines.append(
        "Use the above context to provide continuity. "
        "If the user's current question relates to a previous session topic, "
        "reference it naturally (e.g., 'Building on our earlier discussion of X...')."
    )
    return "\n".join(lines)


def prune_old_summaries(keep_last: int = 50):
    """Keep only the most recent N summaries."""
    if not MEMORY_FILE.exists():
        return
    try:
        lines = MEMORY_FILE.read_text(encoding="utf-8").strip().split("\n")
        if len(lines) > keep_last:
            MEMORY_FILE.write_text(
                "\n".join(lines[-keep_last:]) + "\n",
                encoding="utf-8",
            )
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════
# §CE-13: Conversation Summarization — Auto-compress long chats
# ═══════════════════════════════════════════════════════════════════

def summarize_conversation(messages: list, max_summary_tokens: int = 200) -> str:
    """Auto-summarize long conversations into a concise digest.

    Used to compress conversation history that exceeds the context window.
    Extracts key questions, decisions, and conclusions.

    Returns a 3-5 sentence summary suitable for injection into context.
    """
    if not messages:
        return ""

    user_msgs = [
        (m.get("text") or m.get("content") or "")[:150]
        for m in messages if m.get("role") == "user"
    ]
    asst_msgs = [
        (m.get("text") or m.get("content") or "")[:150]
        for m in messages if m.get("role") == "assistant"
    ]

    # Extract the core topics from user messages
    questions = user_msgs[:5]
    conclusions = asst_msgs[-2:] if asst_msgs else []

    parts = []
    if questions:
        parts.append(f"The user asked about: {'; '.join(questions[:3])}")
    if conclusions:
        parts.append(f"Key conclusions: {'; '.join(conclusions[:2])}")
    parts.append(f"Total: {len(messages)} messages exchanged.")

    summary = " ".join(parts)

    # Truncate to target token budget (rough: 4 chars per token)
    max_chars = max_summary_tokens * 4
    if len(summary) > max_chars:
        summary = summary[:max_chars] + "..."

    return summary


def compress_history(messages: list, max_recent: int = 6) -> list:
    """Compress conversation history for context window efficiency.

    Keeps the most recent N messages verbatim. Earlier messages are
    replaced with a summary message.

    Returns a new message list that fits more information in less context.
    """
    if len(messages) <= max_recent:
        return messages

    # Split into old and recent
    old_messages = messages[:-max_recent]
    recent_messages = messages[-max_recent:]

    # Summarize the old part
    summary_text = summarize_conversation(old_messages)

    # Create a synthetic summary message
    summary_msg = {
        "role": "system",
        "content": f"[Previous conversation summary: {summary_text}]",
    }

    return [summary_msg] + recent_messages


# ═══════════════════════════════════════════════════════════════════
# §M2-2: Sliding Window Attention — RAM-Aware Context Management
# ═══════════════════════════════════════════════════════════════════

# Rough token estimate: ~4 chars per token
_TOKEN_THRESHOLD = int(os.environ.get("SLIDING_WINDOW_TOKENS", "2048"))


def _estimate_tokens(messages: list) -> int:
    """Estimate total tokens across all messages."""
    return sum(len(m.get("text") or m.get("content") or "") for m in messages) // 4


def _detect_ram_gb() -> int:
    """Detect system RAM in GB."""
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=3,
        )
        return round(int(result.stdout.strip()) / (1024**3))
    except Exception:
        return 16  # optimistic fallback


def sliding_window_compress(messages: list, max_recent: int = 5) -> list:
    """§M2-2: Hardware-aware sliding window compression.

    On 8GB machines, when context exceeds 2048 tokens:
    - Keeps the system prompt (first message if role=system)
    - Keeps the last `max_recent` messages verbatim
    - Drops all middle messages entirely (no summary — saves tokens)

    On >=16GB machines, falls back to standard compress_history()
    which preserves a summary of dropped messages.

    Returns a compressed message list with static memory footprint.
    """
    max_memory = int(os.environ.get("MAX_MEMORY_GB", "0"))
    ram_gb = max_memory if max_memory else _detect_ram_gb()

    token_count = _estimate_tokens(messages)

    if ram_gb > 8 or token_count <= _TOKEN_THRESHOLD:
        # High-memory machine or small context: use standard compress
        return compress_history(messages, max_recent=max_recent + 1)

    # === M2 Sliding Window: keep edges, drop middle ===
    if len(messages) <= max_recent + 1:
        return messages

    # Preserve system prompt if present
    has_system = messages[0].get("role") == "system" if messages else False
    system_msgs = [messages[0]] if has_system else []

    # Keep last N messages
    recent = messages[-max_recent:]

    # Build a minimal bridge marker
    dropped_count = len(messages) - len(system_msgs) - len(recent)
    bridge = {
        "role": "system",
        "content": f"[{dropped_count} earlier messages omitted to save memory]",
    }

    log.info(f"§M2-2: Sliding window active — kept {len(system_msgs) + len(recent)} "
             f"messages, dropped {dropped_count} (tokens: {token_count} → ~{_estimate_tokens(system_msgs + [bridge] + recent)})")

    return system_msgs + [bridge] + recent


# ═══════════════════════════════════════════════════════════════════
# §CE-14: Topic Segmentation — Detect topic shifts within sessions
# ═══════════════════════════════════════════════════════════════════

def detect_topic_segments(messages: list) -> list[dict]:
    """Detect when the conversation shifts topics.

    Returns a list of topic segments with start/end indices and labels.
    This lets the memory system store segments separately for cleaner retrieval.
    """
    if not messages:
        return []

    segments = []
    current_segment_start = 0
    current_keywords: set[str] = set()

    for i, msg in enumerate(messages):
        text = (msg.get("text") or msg.get("content") or "").lower()
        if not text or msg.get("role") != "user":
            continue

        # Extract keywords (simple: words > 5 chars, not common words)
        stopwords = {"about", "their", "these", "would", "could", "should",
                     "there", "where", "which", "think", "really", "being"}
        words = set(
            w for w in re.findall(r'\b[a-z]{5,}\b', text)
            if w not in stopwords
        )

        if current_keywords:
            # Calculate overlap
            overlap = len(words & current_keywords) / max(len(current_keywords), 1)
            if overlap < 0.15 and len(words) > 2:
                # Topic shift detected
                segments.append({
                    "start": current_segment_start,
                    "end": i - 1,
                    "keywords": list(current_keywords)[:5],
                })
                current_segment_start = i
                current_keywords = words
            else:
                current_keywords.update(words)
        else:
            current_keywords = words

    # Final segment
    if current_keywords:
        segments.append({
            "start": current_segment_start,
            "end": len(messages) - 1,
            "keywords": list(current_keywords)[:5],
        })

    return segments


# ═══════════════════════════════════════════════════════════════════
# §CE-15: Memory Priority Tagging — Pin important messages
# ═══════════════════════════════════════════════════════════════════

def pin_memory(text: str, category: str = "general") -> dict:
    """Pin an important piece of information for permanent recall.

    Pinned memories are always included in context, even as
    other conversation history fades. The user can say
    "Remember this forever" and it gets pinned.
    """
    entry = {
        "text": text[:500],
        "category": category,
        "timestamp": datetime.now().isoformat(),
        "id": f"pin_{int(time.time())}",
    }

    pinned = load_pinned_memories()
    pinned.append(entry)

    # Keep at most 50 pinned memories
    if len(pinned) > 50:
        pinned = pinned[-50:]

    try:
        PINNED_FILE.parent.mkdir(parents=True, exist_ok=True)
        PINNED_FILE.write_text(
            json.dumps(pinned, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    except Exception:
        log.warning("Failed to save pinned memory")

    return entry


def unpin_memory(pin_id: str) -> bool:
    """Remove a pinned memory by its ID."""
    pinned = load_pinned_memories()
    before = len(pinned)
    pinned = [p for p in pinned if p.get("id") != pin_id]
    if len(pinned) == before:
        return False

    try:
        PINNED_FILE.write_text(
            json.dumps(pinned, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return True
    except Exception:
        return False


def load_pinned_memories() -> list:
    """Load all pinned memories."""
    if not PINNED_FILE.exists():
        return []
    try:
        return json.loads(PINNED_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def search_memory(query: str, limit: int = 10) -> list[dict]:
    """Search across session summaries and pinned memories.

    §CE-15: Simple keyword search across all stored memories.
    Returns matching entries ranked by recency.
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())
    results = []

    # Search pinned memories
    for p in load_pinned_memories():
        text = p.get("text", "").lower()
        if any(w in text for w in query_words):
            results.append({**p, "_type": "pinned", "_score": 2.0})

    # Search session summaries
    for s in load_recent_summaries(50):
        topics = s.get("topics", "").lower()
        if any(w in topics for w in query_words):
            results.append({**s, "_type": "session", "_score": 1.0})

    # Sort by score then recency
    results.sort(key=lambda x: (x.get("_score", 0), x.get("timestamp", "")), reverse=True)
    return results[:limit]
