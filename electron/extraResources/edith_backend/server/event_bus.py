"""
E.D.I.T.H. Event Bus — The Unified Nervous System
=====================================================
Replaces 7 ad-hoc bridges, 9 stream classes, and manual try/except wiring
with a single pub/sub system. Every module publishes events, subscribers react.

The entire cross-module intelligence is readable in ONE file.

Event Domains:
  research.*    — paper indexing, discovery, citations, lit review
  analysis.*    — sniper, causal, forensic, method lab
  pedagogy.*    — socratic chamber, mastery tracking, concept maps, scaffolding
  training.*    — DPO pairs, feedback, sharpening, fine-tuning
  system.*      — thermal, idle, health, metrics
  export.*      — latex, overleaf, notion, peer review
  mission.*     — mission lifecycle (created, step, complete, failed)
"""

import asyncio
import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

log = logging.getLogger("edith.bus")


# ═══════════════════════════════════════════════════════════════════
# Event Data Model
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Event:
    """An event flowing through the nervous system."""
    name: str               # e.g. "paper.indexed", "sniper.weakness"
    data: dict              # payload
    source: str = ""        # which module emitted it
    timestamp: float = field(default_factory=time.time)
    id: str = ""            # auto-generated

    def __post_init__(self):
        if not self.id:
            self.id = f"evt-{int(self.timestamp*1000) % 10_000_000:07d}"

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp,
        }


# Type for subscriber callbacks
Subscriber = Callable[[Event], Coroutine[Any, Any, None]]


# ═══════════════════════════════════════════════════════════════════
# The Bus
# ═══════════════════════════════════════════════════════════════════

class EventBus:
    """
    The single nervous system for E.D.I.T.H.
    
    Usage:
        bus = EventBus()
        
        # Subscribe (usually at startup)
        bus.on("paper.indexed", handle_subconscious_links)
        bus.on("paper.indexed", handle_kg_injection)
        bus.on("sniper.weakness", handle_dpo_logging)
        bus.on("concept.struggled", handle_spaced_rep_card)
        
        # Emit (from any handler)
        await bus.emit("paper.indexed", {"title": "...", "id": "..."}, source="indexer")
    
    Subscribers are always non-blocking. A failing subscriber never breaks
    the emitter. Events are logged for replay and debugging.
    """

    def __init__(self, history_limit: int = 500):
        self._subscribers: dict[str, list[Subscriber]] = defaultdict(list)
        self._wildcard_subscribers: list[Subscriber] = []
        self._history: list[Event] = []
        self._history_limit = history_limit
        self._stats: dict[str, int] = defaultdict(int)
        self._error_count: int = 0

    # ── Subscribe ──

    def on(self, event_name: str, callback: Subscriber, name: str = "") -> str:
        """
        Subscribe to an event.
        
        Args:
            event_name: Event to listen for. Use "*" for all events.
            callback: Async function(event) to call.
            name: Optional human-readable name for debugging.
        
        Returns:
            Subscriber ID for unsubscribe.
        """
        if not name:
            name = getattr(callback, "__name__", str(callback))

        if event_name == "*":
            self._wildcard_subscribers.append(callback)
            log.debug(f"§BUS: Wildcard subscriber registered: {name}")
        else:
            self._subscribers[event_name].append(callback)
            log.debug(f"§BUS: Subscriber '{name}' registered for '{event_name}'")

        return f"{event_name}:{name}"

    def off(self, event_name: str, callback: Subscriber):
        """Unsubscribe from an event."""
        if event_name == "*":
            self._wildcard_subscribers = [s for s in self._wildcard_subscribers if s != callback]
        elif event_name in self._subscribers:
            self._subscribers[event_name] = [s for s in self._subscribers[event_name] if s != callback]

    # ── Emit ──

    async def emit(self, event_name: str, data: dict = None, source: str = ""):
        """
        Emit an event. All matching subscribers fire asynchronously.
        
        Subscribers NEVER block the emitter. Failures are logged but
        don't propagate. This is the core contract that makes the 
        nervous system safe.
        """
        event = Event(name=event_name, data=data or {}, source=source)

        # Record history
        self._history.append(event)
        if len(self._history) > self._history_limit:
            self._history = self._history[-self._history_limit:]
        self._stats[event_name] += 1

        # Gather all matching subscribers
        handlers = list(self._subscribers.get(event_name, []))

        # Prefix matching: "paper.indexed" also triggers "paper.*" subscribers
        prefix = event_name.rsplit(".", 1)[0] + ".*" if "." in event_name else ""
        if prefix and prefix in self._subscribers:
            handlers.extend(self._subscribers[prefix])

        # Wildcard subscribers
        handlers.extend(self._wildcard_subscribers)

        if not handlers:
            return event

        # Fire all subscribers concurrently, non-blocking
        for handler in handlers:
            try:
                asyncio.ensure_future(self._safe_call(handler, event))
            except Exception as e:
                self._error_count += 1
                log.debug(f"§BUS: Failed to schedule handler for '{event_name}': {e}")

        log.info(f"§BUS: {event_name} → {len(handlers)} subscribers (source={source})")
        return event

    async def _safe_call(self, handler: Subscriber, event: Event):
        """Call a subscriber safely — never propagate errors."""
        try:
            result = handler(event)
            if asyncio.iscoroutine(result) or asyncio.isfuture(result):
                await asyncio.wait_for(result, timeout=30.0)
        except asyncio.TimeoutError:
            name = getattr(handler, "__name__", "?")
            log.warning(f"§BUS: Handler '{name}' timed out for '{event.name}'")
            self._error_count += 1
        except Exception as e:
            name = getattr(handler, "__name__", "?")
            log.warning(f"§BUS: Handler '{name}' failed for '{event.name}': {e}")
            self._error_count += 1

    # ── Query ──

    def history(self, event_name: str = "", limit: int = 50) -> list[dict]:
        """Get event history, optionally filtered by name."""
        events = self._history
        if event_name:
            events = [e for e in events if e.name == event_name or e.name.startswith(event_name.rstrip("*"))]
        return [e.to_dict() for e in events[-limit:]]

    def replay(self, event_name: str, limit: int = 10) -> list[Event]:
        """Get raw events for replay (used by late-joining subscribers)."""
        events = [e for e in self._history if e.name == event_name]
        return events[-limit:]

    @property
    def status(self) -> dict:
        """Full bus status for the Cockpit dashboard."""
        return {
            "total_events": sum(self._stats.values()),
            "unique_event_types": len(self._stats),
            "subscriber_count": sum(len(v) for v in self._subscribers.values()) + len(self._wildcard_subscribers),
            "error_count": self._error_count,
            "history_size": len(self._history),
            "top_events": dict(sorted(self._stats.items(), key=lambda x: -x[1])[:15]),
            "subscribers": {
                name: len(handlers) for name, handlers in self._subscribers.items()
            },
        }

    @property
    def wiring_map(self) -> dict:
        """Human-readable map of all event→subscriber wiring."""
        wiring = {}
        for event_name, handlers in self._subscribers.items():
            wiring[event_name] = [getattr(h, "__name__", str(h)) for h in handlers]
        if self._wildcard_subscribers:
            wiring["*"] = [getattr(h, "__name__", str(h)) for h in self._wildcard_subscribers]
        return wiring


# ═══════════════════════════════════════════════════════════════════
# Global Instance
# ═══════════════════════════════════════════════════════════════════

bus = EventBus()


# ═══════════════════════════════════════════════════════════════════
# Subscriber Handlers — The Entire Nervous System in One Place
# ═══════════════════════════════════════════════════════════════════
#
# RESEARCH DOMAIN
# ────────────────

async def _on_paper_indexed(event: Event):
    """paper.indexed → Subconscious links + KG injection."""
    from server.subconscious_streams import subconscious_memory
    await subconscious_memory.on_paper_indexed(event.data, event.data.get("chunks", []))

    # Also inject into Knowledge Graph
    import os
    data_root = os.environ.get("EDITH_DATA_ROOT", "")
    if data_root and event.data.get("title"):
        kg_path = Path(data_root) / "knowledge_graph" / "auto_nodes.json"
        try:
            kg_path.parent.mkdir(parents=True, exist_ok=True)
            existing = json.loads(kg_path.read_text()) if kg_path.exists() else []
            existing.append({
                "id": event.data.get("id", event.data.get("title", "")[:50]),
                "label": event.data.get("title", ""),
                "type": "paper",
                "metadata": {k: v for k, v in event.data.items()
                            if k in ("author", "year", "tags", "type")},
                "source": "event_bus",
            })
            kg_path.write_text(json.dumps(existing[-500:], indent=2))
        except Exception:
            pass


async def _on_discovery_results(event: Event):
    """discovery.results → Auto-generate flashcards for top papers."""
    import os
    data_root = os.environ.get("EDITH_DATA_ROOT", "")
    papers = event.data.get("papers", [])
    if not papers or not data_root:
        return
    cards = []
    for p in papers[:5]:
        if isinstance(p, dict) and p.get("title"):
            cards.append({
                "front": f"What is the main finding of: {p['title'][:80]}?",
                "back": p.get("abstract", p.get("title", ""))[:200],
                "source": "auto_discovery",
                "created_at": time.time(),
            })
    if cards:
        card_path = Path(data_root) / "training" / "auto_flashcards.json"
        card_path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(card_path.read_text()) if card_path.exists() else []
        existing.extend(cards)
        card_path.write_text(json.dumps(existing[-200:], indent=2))


# ANALYSIS DOMAIN
# ────────────────

async def _on_sniper_weakness(event: Event):
    """sniper.weakness → Log DPO training pair."""
    import os
    data_root = os.environ.get("EDITH_DATA_ROOT", "")
    if not data_root:
        return
    weakness = event.data.get("description", "")
    fix = event.data.get("suggestion", "")
    if weakness:
        pair_path = Path(data_root) / "training" / "dpo_pairs.json"
        pair_path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(pair_path.read_text()) if pair_path.exists() else []
        existing.append({
            "prompt": f"Evaluate this claim: {event.data.get('original_claim', '')}",
            "rejected": weakness,
            "chosen": fix or f"This claim has a weakness: {weakness}",
            "source": "sniper_bridge",
            "timestamp": time.time(),
        })
        pair_path.write_text(json.dumps(existing[-500:], indent=2))


async def _on_method_winner(event: Event):
    """method.winner → Auto-generate Stata code via VibeCoder."""
    from server.subconscious_streams import bridge_method_lab_to_vibe_coder
    await bridge_method_lab_to_vibe_coder(event.data)


# PEDAGOGY DOMAIN
# ────────────────

async def _on_concept_struggled(event: Event):
    """concept.struggled → Auto-generate SpacedRep cards for weak concepts."""
    import os
    data_root = os.environ.get("EDITH_DATA_ROOT", "")
    if not data_root:
        return
    concept = event.data.get("concept", "")
    context = event.data.get("context", "")
    if concept:
        card_path = Path(data_root) / "training" / "auto_flashcards.json"
        card_path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(card_path.read_text()) if card_path.exists() else []
        existing.append({
            "front": f"Explain this concept: {concept}",
            "back": context or f"Review your notes on {concept}",
            "source": "socratic_struggle",
            "priority": "high",
            "created_at": time.time(),
        })
        card_path.write_text(json.dumps(existing[-200:], indent=2))


async def _on_mastery_updated(event: Event):
    """mastery.updated → Update the Learning Progress HUD."""
    import os
    data_root = os.environ.get("EDITH_DATA_ROOT", "")
    if not data_root:
        return
    topic = event.data.get("topic", "")
    status = event.data.get("status", "developing")  # mastered | developing | review_needed
    hud_path = Path(data_root) / "pedagogy" / "mastery_hud.json"
    hud_path.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(hud_path.read_text()) if hud_path.exists() else {}
    existing[topic] = {
        "status": status,
        "updated_at": time.time(),
        "taught_by": event.data.get("taught_by", ""),
    }
    hud_path.write_text(json.dumps(existing, indent=2))


async def _on_paper_deconstructed(event: Event):
    """paper.deconstructed → Feed concept map + mastery tracker."""
    concepts = event.data.get("concepts", [])
    for concept in concepts[:10]:
        name = concept if isinstance(concept, str) else concept.get("name", "")
        if name:
            await bus.emit("mastery.updated", {
                "topic": name,
                "status": "developing",
                "taught_by": "deconstructionist",
            }, source="pedagogy")


# TRAINING DOMAIN
# ────────────────

async def _on_feedback_negative(event: Event):
    """feedback.negative → Run contradiction detection on downvoted response."""
    import os
    data_root = os.environ.get("EDITH_DATA_ROOT", "")
    if not data_root:
        return
    response = event.data.get("response_text", "")
    sources = event.data.get("sources", [])
    if response:
        contra_path = Path(data_root) / "contradictions" / "auto_detected.json"
        contra_path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(contra_path.read_text()) if contra_path.exists() else []
        existing.append({
            "response": response[:500],
            "sources": [s.get("title", str(s)[:50]) for s in sources[:5]] if isinstance(sources, list) else [],
            "detected_at": time.time(),
            "trigger": "thumbs_down",
        })
        contra_path.write_text(json.dumps(existing[-200:], indent=2))


async def _on_committee_consensus(event: Event):
    """committee.consensus → Log as high-confidence DPO training pair."""
    import os
    data_root = os.environ.get("EDITH_DATA_ROOT", "")
    if not data_root:
        return
    consensus = event.data.get("consensus", "")
    question = event.data.get("question", "")
    if consensus and question:
        pair_path = Path(data_root) / "training" / "dpo_pairs.json"
        pair_path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(pair_path.read_text()) if pair_path.exists() else []
        existing.append({
            "prompt": question,
            "chosen": consensus,
            "rejected": "",
            "source": "committee_consensus",
            "confidence": "high",
            "timestamp": time.time(),
        })
        pair_path.write_text(json.dumps(existing[-500:], indent=2))


async def _on_spacedrep_weakness(event: Event):
    """spacedrep.weakness → Auto-queue deep dives on weak topics."""
    from server.subconscious_streams import bridge_spaced_rep_to_deep_dive
    await bridge_spaced_rep_to_deep_dive(event.data.get("cards", []))


# SYSTEM DOMAIN
# ────────────────

async def _on_anomaly_detected(event: Event):
    """anomaly.detected → Auto-trigger Doctor heal."""
    import os
    data_root = os.environ.get("EDITH_DATA_ROOT", "")
    severity = event.data.get("severity", 0)
    if severity >= 7:
        # Auto-clear caches for high-severity anomalies
        cache_dir = Path(data_root) / "cache" if data_root else None
        if cache_dir and cache_dir.exists():
            cleared = 0
            for f in cache_dir.glob("*.json"):
                try:
                    if time.time() - f.stat().st_mtime > 3600:
                        f.unlink()
                        cleared += 1
                except Exception:
                    pass
            log.info(f"§BUS: anomaly.detected (severity={severity}) → cleared {cleared} cache files")


async def _on_morning_brief(event: Event):
    """morning.brief → Feed priority items into active learning queue."""
    import os
    data_root = os.environ.get("EDITH_DATA_ROOT", "")
    if not data_root:
        return
    priorities = event.data.get("priorities", [])
    if priorities:
        queue_path = Path(data_root) / "training" / "active_learning_queue.json"
        queue_path.parent.mkdir(parents=True, exist_ok=True)
        existing = json.loads(queue_path.read_text()) if queue_path.exists() else []
        for item in priorities[:10]:
            existing.append({
                "item": item if isinstance(item, str) else json.dumps(item)[:200],
                "source": "morning_brief",
                "urgency": "high",
                "queued_at": time.time(),
            })
        queue_path.write_text(json.dumps(existing[-100:], indent=2))


async def _on_export_started(event: Event):
    """export.started → Trigger Socratic Pre-Defense Briefing."""
    from server.subconscious_streams import socratic_review
    content = event.data.get("content", "")
    if content and len(content) > 100:
        briefing = await socratic_review.pre_defense_briefing(content[:5000])
        # Store briefing for the export result to pick up
        import os
        data_root = os.environ.get("EDITH_DATA_ROOT", "")
        if data_root:
            brief_path = Path(data_root) / "export" / "last_briefing.json"
            brief_path.parent.mkdir(parents=True, exist_ok=True)
            brief_path.write_text(json.dumps(briefing, indent=2, default=str))


async def _on_training_pair_logged(event: Event):
    """training.pair_logged → Track DPO pair accumulation, alert when finetune threshold nears."""
    import os
    data_root = os.environ.get("EDITH_DATA_ROOT", "")
    if not data_root:
        return
    tracker_path = Path(data_root) / "training" / "pair_accumulator.json"
    tracker_path.parent.mkdir(parents=True, exist_ok=True)
    tracker = json.loads(tracker_path.read_text()) if tracker_path.exists() else {"count": 0, "since_last_finetune": 0}
    tracker["count"] += 1
    tracker["since_last_finetune"] += 1
    tracker["last_source"] = event.data.get("source", "unknown")
    tracker["last_at"] = time.time()
    tracker_path.write_text(json.dumps(tracker, indent=2))
    # Log milestone
    if tracker["since_last_finetune"] in (50, 75, 100):
        log.info(f"§BUS: {tracker['since_last_finetune']} DPO pairs accumulated — "
                 f"{'FINETUNE READY!' if tracker['since_last_finetune'] >= 100 else 'approaching threshold'}")


# ═══════════════════════════════════════════════════════════════════
# AUTOPILOT DOMAIN — 6 Auto-Triggers for Full Autonomy
# ═══════════════════════════════════════════════════════════════════

async def _on_autopilot_classify(event: Event):
    """paper.indexed → Auto-detect method + academic topic."""
    import os, re as _re
    text = event.data.get("text", event.data.get("abstract", ""))
    sha256 = event.data.get("sha256", event.data.get("id", ""))
    if not text or not sha256:
        return
    # Skip if already classified
    if event.data.get("method") and event.data.get("academic_topic"):
        return
    text_lower = text[:3000].lower()
    method_patterns = {
        "Difference-in-Differences": ["difference.in.difference", "diff.in.diff", "parallel trends"],
        "Instrumental Variables": ["instrumental variable", " iv ", "2sls", "two.stage least"],
        "Regression Discontinuity": ["regression discontinuity", "rdd"],
        "Synthetic Control": ["synthetic control", "donor pool"],
        "Fixed Effects": ["fixed effect", "within estimat"],
        "OLS": ["ordinary least squares", " ols "],
        "Event Study": ["event study", "pre.trend"],
        "Logit": ["logistic regression", "logit"],
        "Probit": ["probit model", "probit regression"],
        "Propensity Score": ["propensity score", "psm"],
    }
    detected_method = ""
    for method, patterns in method_patterns.items():
        if any(_re.search(p, text_lower) for p in patterns):
            detected_method = method
            break

    topic_patterns = {
        "Labor Economics": ["labor market", "wages", "employment", "unemployment"],
        "Health Economics": ["health", "mortality", "hospital", "insurance"],
        "Public Policy": ["policy", "welfare", "snap", "government program"],
        "Development Economics": ["developing countr", "poverty", "microfinance"],
        "Education Economics": ["education", "school", "student", "teacher"],
        "Environmental Economics": ["climate", "pollution", "carbon", "emissions"],
        "Finance": ["stock", "return", "portfolio", "asset pric"],
        "Trade": ["trade", "tariff", "export", "import"],
        "Urban Economics": ["housing", "rent", "urban", "city"],
    }
    detected_topic = ""
    for topic, patterns in topic_patterns.items():
        if sum(1 for p in patterns if p in text_lower) >= 2:
            detected_topic = topic
            break

    if detected_method or detected_topic:
        chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "")
        if chroma_dir:
            try:
                from server.chroma_backend import _get_client
                col = _get_client(chroma_dir).get_or_create_collection(
                    "edith_corpus", metadata={"hnsw:space": "cosine"})
                results = col.get(where={"sha256": sha256}, limit=1)
                if results["ids"]:
                    update = dict(event.data.get("metadata", {}))
                    if detected_method:
                        update["method"] = detected_method
                    if detected_topic:
                        update["academic_topic"] = detected_topic
                    update["auto_classified"] = True
                    col.update(ids=results["ids"], metadatas=[update])
                    log.info(f"§AUTOPILOT: Classified {sha256[:12]} → {detected_method}/{detected_topic}")
            except Exception as e:
                log.debug(f"§AUTOPILOT: Classification failed: {e}")


async def _on_autopilot_relate(event: Event):
    """paper.indexed → Find and log similar papers."""
    import os
    text = event.data.get("text", event.data.get("abstract", ""))
    sha256 = event.data.get("sha256", event.data.get("id", ""))
    title = event.data.get("title", "")
    if not text or not sha256:
        return
    chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "")
    if not chroma_dir:
        return
    try:
        from server.chroma_backend import _get_client
        col = _get_client(chroma_dir).get_or_create_collection(
            "edith_corpus", metadata={"hnsw:space": "cosine"})
        results = col.query(query_texts=[text[:500]], n_results=5,
                           include=["metadatas", "distances"])
        similar = []
        for meta, dist in zip(results.get("metadatas", [[]])[0],
                             results.get("distances", [[]])[0]):
            if meta and meta.get("sha256") != sha256 and dist < 0.5:
                similar.append({"sha256": meta.get("sha256", ""),
                              "title": meta.get("title", ""), "distance": round(dist, 3)})
        if similar:
            data_root = os.environ.get("EDITH_DATA_ROOT", os.environ.get("DATA_ROOT", ""))
            if data_root:
                rel_path = Path(data_root) / "paper_relations.jsonl"
                with open(str(rel_path), 'a') as f:
                    f.write(json.dumps({"source": sha256, "title": title,
                                       "similar": similar, "timestamp": time.time()}) + "\n")
                log.info(f"§AUTOPILOT: {len(similar)} similar papers for {sha256[:12]}")
    except Exception as e:
        log.debug(f"§AUTOPILOT: Similarity failed: {e}")


async def _on_autopilot_reading_init(event: Event):
    """paper.indexed → Set initial reading progress to 'unread'."""
    sha256 = event.data.get("sha256", event.data.get("id", ""))
    if not sha256:
        return
    try:
        from server.routes.flywheel import _reading_progress, _save_reading_progress
        if sha256 not in _reading_progress:
            _reading_progress[sha256] = {"status": "unread", "updated_at": time.time(), "notes_count": 0}
            _save_reading_progress()
    except Exception:
        pass


async def _on_autopilot_reading_focused(event: Event):
    """paper.focused → Mark as 'reading' when user views a paper."""
    sha256 = event.data.get("sha256", "")
    if not sha256:
        return
    try:
        from server.routes.flywheel import _reading_progress, _save_reading_progress
        current = _reading_progress.get(sha256, {})
        if current.get("status") != "done":
            _reading_progress[sha256] = {"status": "reading", "updated_at": time.time(),
                                        "notes_count": current.get("notes_count", 0)}
            _save_reading_progress()
            log.info(f"§AUTOPILOT: {sha256[:12]} → reading")
    except Exception:
        pass


async def _on_autopilot_reading_done(event: Event):
    """paper.deconstructed → Mark as 'done' after forensic breakdown."""
    sha256 = event.data.get("sha256", "")
    if not sha256:
        return
    try:
        from server.routes.flywheel import _reading_progress, _save_reading_progress
        _reading_progress[sha256] = {"status": "done", "updated_at": time.time(),
                                    "notes_count": _reading_progress.get(sha256, {}).get("notes_count", 0)}
        _save_reading_progress()
        log.info(f"§AUTOPILOT: {sha256[:12]} → done")
    except Exception:
        pass


async def _on_autopilot_training_capture(event: Event):
    """chat.response → Save good turns as Winnie training data."""
    import os
    if not event.data.get("is_clean", False) or event.data.get("coverage", 0) < 0.1:
        return
    query = event.data.get("query", "")
    response = event.data.get("response", "")
    if not query or not response:
        return
    data_root = os.environ.get("EDITH_DATA_ROOT", os.environ.get("DATA_ROOT", ""))
    if not data_root:
        return
    try:
        train_path = Path(data_root) / "edith_master_train.jsonl"
        train_path.parent.mkdir(parents=True, exist_ok=True)
        topics = set()
        for s in (event.data.get("sources", []) or [])[:5]:
            if isinstance(s, dict):
                t = s.get("academic_topic", s.get("topic", ""))
                if t:
                    topics.add(t)
        entry = {
            "messages": [{"role": "user", "content": query[:2000]},
                        {"role": "assistant", "content": response[:4000]}],
            "source": "chat_autopilot", "model": event.data.get("model", ""),
            "coverage": round(event.data.get("coverage", 0), 2), "timestamp": time.time(),
        }
        if topics:
            entry["topic"] = list(topics)[0]
        with open(str(train_path), 'a') as f:
            f.write(json.dumps(entry) + "\n")
        log.info(f"§AUTOPILOT: Captured training pair (topic={entry.get('topic', 'general')})")
    except Exception as e:
        log.debug(f"§AUTOPILOT: Training capture failed: {e}")


async def _on_autopilot_method_suggest(event: Event):
    """chat.response → Detect RQ and log detected research questions."""
    import os, re as _re
    query = event.data.get("query", "")
    if not query:
        return
    query_lower = query.lower()
    rq_signals = [
        r"effect\s+of\s+\w+\s+on", r"impact\s+of\s+\w+\s+on",
        r"how\s+does\s+\w+\s+affect", r"does\s+\w+\s+cause",
        r"causal.*?(?:effect|impact)", r"identify\s+(?:the\s+)?(?:effect|impact)",
    ]
    if not any(_re.search(p, query_lower) for p in rq_signals):
        return
    # Log to RQ evolution tracker
    try:
        from server.routes.flywheel_advanced import _rq_history, _save_rq_history
        _rq_history.append({
            "question": query, "context": "auto-detected from chat",
            "timestamp": time.time(), "date": time.strftime("%Y-%m-%d %H:%M"),
            "version": len(_rq_history) + 1, "auto": True,
        })
        _save_rq_history()
        log.info(f"§AUTOPILOT: Detected RQ → logged to evolution tracker")
    except Exception:
        pass


async def _on_autopilot_export_review(event: Event):
    """export.started → Run pre-export peer review checks."""
    import os, re as _re
    text = event.data.get("content", "")
    if not text or len(text) < 100:
        return
    text_lower = text.lower()
    warnings = []
    checks = [
        (["identification", "causal", "instrument", "exogen"],
         "No identification strategy discussed", "major"),
        (["robustness", "sensitivity", "placebo", "falsification"],
         "No robustness checks mentioned", "major"),
        (["cluster", "heterosked", "robust standard"],
         "Standard errors may not be properly specified", "moderate"),
        (["external validity", "generalizab"],
         "Consider discussing external validity", "minor"),
    ]
    for keywords, warning, severity in checks:
        if not any(kw in text_lower for kw in keywords):
            warnings.append({"message": warning, "severity": severity})
    citations = len(_re.findall(r'\(\d{4}\)', text))
    if citations < 5:
        warnings.append({"message": f"Only {citations} citations", "severity": "moderate"})
    major = sum(1 for w in warnings if w["severity"] == "major")
    # Save review result so export endpoint can pick it up
    data_root = os.environ.get("EDITH_DATA_ROOT", os.environ.get("DATA_ROOT", ""))
    if data_root and warnings:
        review_path = Path(data_root) / "export" / "pre_export_review.json"
        review_path.parent.mkdir(parents=True, exist_ok=True)
        review_path.write_text(json.dumps({
            "warnings": warnings, "major_issues": major,
            "ready": major == 0, "timestamp": time.time(),
        }, indent=2))
        log.info(f"§AUTOPILOT: Pre-export review: {major} major, {len(warnings)} total issues")


# ═══════════════════════════════════════════════════════════════════
# Registration — Wire Everything at Startup
# ═══════════════════════════════════════════════════════════════════

def register_all_subscribers():
    """
    Register all event subscribers. Call once at app startup.
    
    This function IS the complete wiring diagram of E.D.I.T.H.'s
    cross-module intelligence. If you want to know how modules talk
    to each other, read this function.
    """
    # ── Research ──
    bus.on("paper.indexed",          _on_paper_indexed,         name="subconscious+kg")
    bus.on("discovery.results",      _on_discovery_results,     name="auto_flashcards")

    # ── Analysis ──
    bus.on("sniper.weakness",        _on_sniper_weakness,       name="dpo_logger")
    bus.on("method.winner",          _on_method_winner,         name="vibe_coder")

    # ── Pedagogy ──
    bus.on("concept.struggled",      _on_concept_struggled,     name="spaced_rep_card")
    bus.on("mastery.updated",        _on_mastery_updated,       name="mastery_hud")
    bus.on("paper.deconstructed",    _on_paper_deconstructed,   name="concept_tracker")

    # ── Training ──
    bus.on("feedback.negative",      _on_feedback_negative,     name="contradiction_detect")
    bus.on("committee.consensus",    _on_committee_consensus,   name="consensus_dpo")
    bus.on("spacedrep.weakness",     _on_spacedrep_weakness,    name="deep_dive_queue")

    # ── System ──
    bus.on("anomaly.detected",       _on_anomaly_detected,      name="auto_heal")
    bus.on("morning.brief",          _on_morning_brief,         name="active_learning")
    bus.on("export.started",         _on_export_started,        name="socratic_review")

    # ── Training (new) ──
    bus.on("training.pair_logged",    _on_training_pair_logged,  name="pair_accumulator")

    # ── Autopilot: 6 Auto-Triggers ──
    bus.on("paper.indexed",          _on_autopilot_classify,      name="auto_classify")
    bus.on("paper.indexed",          _on_autopilot_relate,        name="auto_relate")
    bus.on("paper.indexed",          _on_autopilot_reading_init,  name="reading_init")
    bus.on("paper.focused",          _on_autopilot_reading_focused, name="reading_focused")
    bus.on("paper.deconstructed",    _on_autopilot_reading_done,  name="reading_done")
    bus.on("chat.response",          _on_autopilot_training_capture, name="training_capture")
    bus.on("chat.response",          _on_autopilot_method_suggest, name="method_suggest")
    bus.on("export.started",         _on_autopilot_export_review, name="export_review")

    log.info(f"§BUS: {bus.status['subscriber_count']} subscribers registered across "
             f"{len(bus._subscribers)} event types")


# ═══════════════════════════════════════════════════════════════════
# Event Catalog — Documentation of All Known Events
# ═══════════════════════════════════════════════════════════════════

EVENT_CATALOG = {
    # Research
    "paper.indexed":        "A paper was added to Chroma. Data: title, id, sha256, abstract, text, chunks, metadata",
    "paper.focused":        "User viewed/focused a paper in any panel. Data: sha256, title, source_panel",
    "paper.deconstructed":  "ForensicWorkbench finished dissecting a paper. Data: sha256, concepts, assumptions, formulas",
    "discovery.results":    "OpenAlex/Scholar search returned papers. Data: papers[], query",
    "literature.gap":       "A gap was found in the knowledge graph. Data: gap_description, between_nodes",

    # Analysis
    "sniper.weakness":      "Methodological Sniper found a flaw. Data: description, suggestion, original_claim",
    "sniper.defended":      "Self-defense audit completed. Data: verdict, fix_suggestions[]",
    "causal.dag_updated":   "Causal DAG was modified. Data: nodes[], edges[], variable",
    "method.winner":        "Method Lab comparison finished. Data: name, variables, score",
    "simulation.complete":  "Monte Carlo simulation finished. Data: results, n_iterations",

    # Chat
    "chat.response":        "Chat response generated. Data: query, response, model, sources, is_clean, coverage",

    # Pedagogy
    "concept.struggled":    "User struggled to explain a concept in Socratic session. Data: concept, context",
    "concept.mastered":     "User demonstrated mastery of a concept. Data: concept, evidence",
    "mastery.updated":      "Learning HUD status changed. Data: topic, status, taught_by",
    "socratic.session":     "Socratic Chamber session started/ended. Data: paper_id, duration, struggles[]",
    "scaffold.provided":    "A pedagogical clue was given to help the user. Data: concept, clue_type",

    # Training
    "feedback.negative":    "User gave thumbs down. Data: response_text, sources, question",
    "feedback.positive":    "User gave thumbs up. Data: response_text, question",
    "committee.consensus":  "3+ models agreed on an answer. Data: consensus, question, models",
    "spacedrep.weakness":   "Flashcard performance dropped. Data: cards[], topic",
    "training.pair_logged": "A new DPO pair was logged. Data: prompt, chosen, rejected",
    "training.sharpen_started": "Overnight sharpening loop started. Data: n_questions, dry_run",
    "training.sharpen_completed": "Overnight sharpening loop finished. Data: questions, agreements, training_pairs_saved, finetune_trigger",

    # System
    "system.thermal":       "Thermal state changed. Data: state, cpu_percent, memory_percent",
    "system.idle":          "User has been idle. Data: idle_minutes",
    "anomaly.detected":     "Health anomaly found. Data: severity, component, description",
    "morning.brief":        "Morning briefing generated. Data: priorities[], insights[]",

    # Export
    "export.started":       "An export to LaTeX/Word/Overleaf began. Data: format, content, title",
    "export.completed":     "Export finished. Data: format, path, briefing",

    # Mission
    "mission.created":      "A new mission was created. Data: mission_id, template, question",
    "mission.step":         "A mission step completed. Data: mission_id, step_name, status",
    "mission.completed":    "A mission finished all steps. Data: mission_id, results",
    "mission.failed":       "A mission step failed. Data: mission_id, step_name, error",
}
