"""
Research Profile — Persistent Memory for E.D.I.T.H.
=====================================================
"Winnie remembers your dissertation."

File-backed research profile that persists across server restarts.
Tracks user preferences, research history, writing style, and
committee feedback — so every tool can personalize its output.
"""
import json
import logging
import os
import threading
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

log = logging.getLogger("edith.research_profile")


class ResearchProfile:
    """Persistent, file-backed research profile.

    Stores:
      - Preferred methods (DiD, RDD, multi-level, etc.)
      - Research topics and their frequency
      - Committee feedback history
      - Writing style traits
      - Tool usage patterns
      - Citation preferences
    """

    def __init__(self, profile_path: str = None):
        self._lock = threading.RLock()

        # Default to EDITH_DATA_ROOT or ~/.edith/
        if profile_path:
            self._path = Path(profile_path)
        else:
            data_root = os.environ.get("EDITH_DATA_ROOT", "")
            if data_root:
                candidate = Path(data_root) / ".edith" / "research_profile.json"
                # Check if parent is writable
                try:
                    candidate.parent.mkdir(parents=True, exist_ok=True)
                    self._path = candidate
                except OSError:
                    # External drive not writable — fall back to home
                    self._path = Path.home() / ".edith" / "research_profile.json"
            else:
                self._path = Path.home() / ".edith" / "research_profile.json"

        self._profile = self._load()
        self._dirty = False
        self._last_save = time.time()

    def _load(self) -> dict:
        """Load profile from disk, or initialize empty."""
        if self._path.exists():
            try:
                with open(self._path, "r") as f:
                    data = json.load(f)
                log.info(f"Research profile loaded from {self._path}")
                return data
            except (json.JSONDecodeError, OSError) as e:
                log.warning(f"Profile load failed: {e}, starting fresh")
        return self._empty_profile()

    def _empty_profile(self) -> dict:
        return {
            "version": 1,
            "created": datetime.now().isoformat(),
            "last_updated": datetime.now().isoformat(),

            # Research interests
            "topics": {},           # topic -> {"count": N, "last_seen": ts}
            "methods": {},          # method -> {"count": N, "preferred": bool}
            "fields": [],           # e.g. ["political science", "public policy"]

            # Preferences
            "preferred_language": "stata",      # stata/r/python
            "citation_style": "APSA",
            "rigor_level": "doctoral",
            "writing_traits": [],               # ["precise", "hedged", "evidence-first"]

            # History
            "recent_queries": [],               # last 50 queries
            "committee_feedback": [],           # advisor notes
            "tool_usage": {},                   # tool_name -> count

            # Context
            "dissertation_topic": "",
            "advisor_name": "",
            "university": "",
            "datasets": [],                     # known datasets
        }

    def _save(self):
        """Save to disk (debounced — max once per 30s)."""
        with self._lock:
            if not self._dirty:
                return
            # Debounce
            if time.time() - self._last_save < 30:
                return
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                self._profile["last_updated"] = datetime.now().isoformat()
                tmp = self._path.with_suffix(".tmp")
                with open(tmp, "w") as f:
                    json.dump(self._profile, f, indent=2, default=str)
                tmp.rename(self._path)
                self._dirty = False
                self._last_save = time.time()
                log.debug("Research profile saved")
            except OSError as e:
                log.warning(f"Profile save failed: {e}")

    def force_save(self):
        """Force immediate save (bypass debounce)."""
        with self._lock:
            self._dirty = True
            self._last_save = 0
            self._save()

    # ── Topic tracking ────────────────────────────────────────────

    def record_topic(self, topic: str):
        """Record a research topic interaction."""
        with self._lock:
            t = topic.lower().strip()
            if t not in self._profile["topics"]:
                self._profile["topics"][t] = {"count": 0, "first_seen": datetime.now().isoformat()}
            self._profile["topics"][t]["count"] += 1
            self._profile["topics"][t]["last_seen"] = datetime.now().isoformat()
            self._dirty = True
            self._save()

    def get_top_topics(self, n: int = 10) -> list[dict]:
        """Get most-researched topics."""
        topics = self._profile.get("topics", {})
        sorted_topics = sorted(topics.items(), key=lambda x: x[1]["count"], reverse=True)
        return [{"topic": t, **info} for t, info in sorted_topics[:n]]

    # ── Method tracking ───────────────────────────────────────────

    def record_method(self, method: str):
        """Record use of a research method."""
        with self._lock:
            m = method.lower().strip()
            if m not in self._profile["methods"]:
                self._profile["methods"][m] = {"count": 0, "preferred": False}
            self._profile["methods"][m]["count"] += 1
            self._dirty = True
            self._save()

    def get_preferred_methods(self) -> list[str]:
        """Get methods sorted by usage."""
        methods = self._profile.get("methods", {})
        return sorted(methods.keys(), key=lambda m: methods[m]["count"], reverse=True)

    # ── Tool usage tracking ───────────────────────────────────────

    def record_tool_use(self, tool_name: str):
        """Track which tools the user uses most."""
        with self._lock:
            usage = self._profile.setdefault("tool_usage", {})
            usage[tool_name] = usage.get(tool_name, 0) + 1
            self._dirty = True
            self._save()

    # ── Query history ─────────────────────────────────────────────

    def record_query(self, query: str, intent: str = ""):
        """Record a research query."""
        with self._lock:
            queries = self._profile.setdefault("recent_queries", [])
            queries.append({
                "query": query[:300],
                "intent": intent,
                "ts": datetime.now().isoformat(),
            })
            # Keep last 50
            if len(queries) > 50:
                self._profile["recent_queries"] = queries[-50:]
            self._dirty = True
            self._save()

    # ── Committee feedback ────────────────────────────────────────

    def add_feedback(self, feedback: str, source: str = "advisor"):
        """Record committee/advisor feedback."""
        with self._lock:
            fb_list = self._profile.setdefault("committee_feedback", [])
            fb_list.append({
                "feedback": feedback[:500],
                "source": source,
                "ts": datetime.now().isoformat(),
            })
            if len(fb_list) > 100:
                self._profile["committee_feedback"] = fb_list[-100:]
            self._dirty = True
            self._save()

    # ── Context string for LLM prompts ────────────────────────────

    def get_context_string(self) -> str:
        """Generate a context string to inject into LLM prompts.

        This is the key function — it gives every tool personalized context.
        """
        parts = []

        # Dissertation topic
        topic = self._profile.get("dissertation_topic", "")
        if topic:
            parts.append(f"Dissertation topic: {topic}")

        # Preferred methods
        methods = self.get_preferred_methods()[:5]
        if methods:
            parts.append(f"Preferred methods: {', '.join(methods)}")

        # Top research areas
        top = self.get_top_topics(5)
        if top:
            areas = [t["topic"] for t in top]
            parts.append(f"Research areas: {', '.join(areas)}")

        # Language preference
        lang = self._profile.get("preferred_language", "")
        if lang:
            parts.append(f"Preferred language: {lang}")

        # Writing style
        traits = self._profile.get("writing_traits", [])
        if traits:
            parts.append(f"Writing style: {', '.join(traits)}")

        # Recent feedback
        feedback = self._profile.get("committee_feedback", [])[-3:]
        if feedback:
            fb_str = " | ".join(f["feedback"][:80] for f in feedback)
            parts.append(f"Recent committee feedback: {fb_str}")

        if not parts:
            return ""
        return "RESEARCHER PROFILE:\n" + "\n".join(f"  - {p}" for p in parts)

    # ── Full profile access ───────────────────────────────────────

    def get_profile(self) -> dict:
        """Return full profile for API."""
        with self._lock:
            return dict(self._profile)

    def update_profile(self, updates: dict) -> dict:
        """Update profile fields."""
        with self._lock:
            for key in ["dissertation_topic", "advisor_name", "university",
                        "preferred_language", "citation_style", "rigor_level",
                        "writing_traits", "fields", "datasets"]:
                if key in updates:
                    self._profile[key] = updates[key]
            self._dirty = True
            self._last_save = 0  # force save
            self._save()
            return dict(self._profile)


# ── Global singleton ──────────────────────────────────────────────
_profile: Optional[ResearchProfile] = None
_profile_lock = threading.Lock()


def get_profile() -> ResearchProfile:
    """Get or create the global research profile (thread-safe)."""
    global _profile
    if _profile is None:
        with _profile_lock:
            if _profile is None:  # double-checked locking
                _profile = ResearchProfile()
    return _profile
