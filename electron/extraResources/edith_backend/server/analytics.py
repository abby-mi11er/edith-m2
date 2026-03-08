"""
Analytics Engine — Research Velocity, Topic Evolution, Session Intelligence
==============================================================================
Batch 5 — CE-51/52/53: Know thyself (as a researcher).

This module tracks:
- How fast you're producing dissertation output (research velocity)
- How your research topics evolve over time (topic trajectory)
- Session intelligence (peak productivity hours, mode usage patterns)
"""

import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("edith.analytics")


# ═══════════════════════════════════════════════════════════════════
# CE-51: Research Velocity Tracker
# ═══════════════════════════════════════════════════════════════════

@dataclass
class ResearchSession:
    """A single research session."""
    session_id: str
    start_time: float
    end_time: float = 0
    queries: int = 0
    sources_retrieved: int = 0
    words_generated: int = 0
    modes_used: list[str] = field(default_factory=list)
    topics: list[str] = field(default_factory=list)

    @property
    def duration_minutes(self) -> float:
        end = self.end_time or time.time()
        return (end - self.start_time) / 60

    def to_dict(self) -> dict:
        return {
            "id": self.session_id,
            "duration_min": round(self.duration_minutes, 1),
            "queries": self.queries,
            "sources": self.sources_retrieved,
            "words": self.words_generated,
            "modes": self.modes_used,
            "topics": self.topics,
        }


class ResearchVelocityTracker:
    """Track research output velocity over time.

    Metrics:
    - Queries per hour (throughput)
    - Sources per query (depth)
    - Words generated per session (output)
    - Mode distribution (what tools you're using)
    """

    def __init__(self):
        self._sessions: list[ResearchSession] = []
        self._current: ResearchSession | None = None
        self._daily_stats: dict[str, dict] = {}

    def start_session(self, session_id: str = "") -> str:
        session_id = session_id or f"RS-{int(time.time())}"
        self._current = ResearchSession(session_id=session_id, start_time=time.time())
        return session_id

    def record_query(self, mode: str = "chat", topic: str = "",
                      sources_count: int = 0, words: int = 0):
        if not self._current:
            self.start_session()

        self._current.queries += 1
        self._current.sources_retrieved += sources_count
        self._current.words_generated += words
        if mode and mode not in self._current.modes_used:
            self._current.modes_used.append(mode)
        if topic and topic not in self._current.topics:
            self._current.topics.append(topic[:50])

        # Update daily stats
        day = time.strftime("%Y-%m-%d")
        if day not in self._daily_stats:
            self._daily_stats[day] = {"queries": 0, "sources": 0, "words": 0}
        self._daily_stats[day]["queries"] += 1
        self._daily_stats[day]["sources"] += sources_count
        self._daily_stats[day]["words"] += words

    def end_session(self):
        if self._current:
            self._current.end_time = time.time()
            self._sessions.append(self._current)
            self._sessions = self._sessions[-100:]  # Keep last 100
            self._current = None

    def get_velocity(self) -> dict:
        """Calculate current research velocity metrics."""
        if not self._daily_stats:
            return {"status": "no_data"}

        days = sorted(self._daily_stats.keys())
        recent = days[-7:]  # Last 7 days

        recent_stats = [self._daily_stats[d] for d in recent if d in self._daily_stats]
        if not recent_stats:
            return {"status": "no_recent_data"}

        avg_queries = sum(s["queries"] for s in recent_stats) / len(recent_stats)
        avg_sources = sum(s["sources"] for s in recent_stats) / len(recent_stats)
        avg_words = sum(s["words"] for s in recent_stats) / len(recent_stats)

        # Trend (compare last 3 days to previous 4)
        if len(recent_stats) >= 5:
            recent_avg = sum(s["queries"] for s in recent_stats[-3:]) / 3
            older_avg = sum(s["queries"] for s in recent_stats[:-3]) / max(len(recent_stats) - 3, 1)
            trend = "accelerating" if recent_avg > older_avg * 1.2 else (
                "decelerating" if recent_avg < older_avg * 0.8 else "steady"
            )
        else:
            trend = "insufficient_data"

        return {
            "avg_daily_queries": round(avg_queries, 1),
            "avg_daily_sources": round(avg_sources, 1),
            "avg_daily_words": round(avg_words),
            "trend": trend,
            "active_days": len(days),
            "total_queries": sum(s["queries"] for s in self._daily_stats.values()),
            "total_words": sum(s["words"] for s in self._daily_stats.values()),
        }

    @property
    def status(self) -> dict:
        return {
            "current_session": self._current.to_dict() if self._current else None,
            "total_sessions": len(self._sessions),
            "velocity": self.get_velocity(),
        }


# ═══════════════════════════════════════════════════════════════════
# CE-52: Topic Evolution Tracker
# ═══════════════════════════════════════════════════════════════════

class TopicEvolutionTracker:
    """Track how your research focus evolves over time.

    Shows you:
    - Which topics you're spending the most time on
    - How your focus has shifted week over week
    - Gaps in your coverage
    """

    def __init__(self):
        self._topic_history: list[dict] = []  # [{timestamp, topic, mode}]
        self._topic_counts: dict[str, int] = {}

    def record_topic(self, topic: str, mode: str = "chat"):
        topic_lower = topic.lower().strip()
        if not topic_lower:
            return

        self._topic_history.append({
            "timestamp": time.time(),
            "topic": topic_lower,
            "mode": mode,
        })
        self._topic_counts[topic_lower] = self._topic_counts.get(topic_lower, 0) + 1
        self._topic_history = self._topic_history[-500:]

    def get_focus_distribution(self, days: int = 7) -> dict:
        """What proportion of your time goes to each topic?"""
        cutoff = time.time() - days * 86400
        recent = [t for t in self._topic_history if t["timestamp"] > cutoff]
        if not recent:
            return {"status": "no_data"}

        counts = {}
        for t in recent:
            counts[t["topic"]] = counts.get(t["topic"], 0) + 1

        total = sum(counts.values())
        return {
            topic: round(count / total, 3)
            for topic, count in sorted(counts.items(), key=lambda x: -x[1])
        }

    def get_topic_trajectory(self) -> list[dict]:
        """Show how your focus shifted over time."""
        if not self._topic_history:
            return []

        # Group by week
        weeks: dict[str, dict[str, int]] = {}
        for entry in self._topic_history:
            week = time.strftime("%Y-W%W", time.localtime(entry["timestamp"]))
            if week not in weeks:
                weeks[week] = {}
            topic = entry["topic"]
            weeks[week][topic] = weeks[week].get(topic, 0) + 1

        return [
            {"week": week, "topics": counts}
            for week, counts in sorted(weeks.items())
        ]

    @property
    def status(self) -> dict:
        return {
            "total_observations": len(self._topic_history),
            "unique_topics": len(self._topic_counts),
            "top_topics": sorted(
                self._topic_counts.items(), key=lambda x: -x[1]
            )[:10],
            "focus_distribution": self.get_focus_distribution(),
        }


# ═══════════════════════════════════════════════════════════════════
# CE-53: Session Intelligence — Peak Productivity Detection
# ═══════════════════════════════════════════════════════════════════

class SessionIntelligence:
    """Detect your peak productivity patterns.

    Answers: When are you most productive? What mode
    generates the most output? What's your optimal session length?
    """

    def __init__(self):
        self._hourly_activity: dict[int, dict] = {}  # hour → {queries, words, sources}
        self._mode_productivity: dict[str, dict] = {}  # mode → {queries, words}

    def record(self, hour: int, mode: str = "chat", words: int = 0, sources: int = 0):
        if hour not in self._hourly_activity:
            self._hourly_activity[hour] = {"queries": 0, "words": 0, "sources": 0, "sessions": 0}
        self._hourly_activity[hour]["queries"] += 1
        self._hourly_activity[hour]["words"] += words
        self._hourly_activity[hour]["sources"] += sources

        if mode not in self._mode_productivity:
            self._mode_productivity[mode] = {"queries": 0, "words": 0}
        self._mode_productivity[mode]["queries"] += 1
        self._mode_productivity[mode]["words"] += words

    def get_peak_hours(self, top_n: int = 3) -> list[dict]:
        """Find your most productive hours."""
        if not self._hourly_activity:
            return []

        ranked = sorted(
            self._hourly_activity.items(),
            key=lambda x: x[1]["queries"],
            reverse=True,
        )
        return [
            {"hour": h, "label": f"{h:02d}:00-{h+1:02d}:00", **stats}
            for h, stats in ranked[:top_n]
        ]

    def get_mode_efficiency(self) -> dict:
        """Which mode generates the most output per query?"""
        efficiency = {}
        for mode, stats in self._mode_productivity.items():
            queries = stats["queries"]
            words = stats["words"]
            efficiency[mode] = {
                "queries": queries,
                "total_words": words,
                "words_per_query": round(words / max(queries, 1)),
            }
        return efficiency

    @property
    def status(self) -> dict:
        return {
            "peak_hours": self.get_peak_hours(),
            "mode_efficiency": self.get_mode_efficiency(),
            "total_hours_tracked": len(self._hourly_activity),
        }


# Global instances
velocity_tracker = ResearchVelocityTracker()
topic_tracker = TopicEvolutionTracker()
session_intelligence = SessionIntelligence()
