"""
Citadel Connectome Master — The Sovereign Brain's Master DNA
===============================================================
The single controller that binds EVERY module into one recursive
metabolic loop. This is the final "citadel_connectome_master.py."

The Recursive Metabolic Loop:
  Library → Forensic Lab → Method Lab → Cockpit →
  Auto-Annotator → Notion Mirror → Atlas → Chat → Dream → repeat

Module Map:
  ┌─ Sensory Thalamus ─────────────────────────────┐
  │  notion_bridge.py   ← Notion API               │
  │  forensic_audit.py  ← PDF ingestion             │
  │  memory_pinning.py  ← Bolt handshake            │
  └──────────┬──────────────────────────────────────┘
             ▼
  ┌─ Hippocampus (Bolt 1TB) ────────────────────────┐
  │  graph_vector_engine.py ← Entity + Vector store  │
  │  chroma_backend.py      ← Semantic search        │
  │  lit_locator.py         ← Theoretical Atlas      │
  └──────────┬──────────────────────────────────────┘
             ▼
  ┌─ Prefrontal Cortex ─────────────────────────────┐
  │  citadel_neural_net.py ← 3 Agents               │
  │  connectome.py         ← Logic Audit             │
  │  synapse_bridge.py     ← Truth Maintenance       │
  │  method_lab.py         ← Crash Courses           │
  └──────────┬──────────────────────────────────────┘
             ▼
  ┌─ REM Engine ────────────────────────────────────┐
  │  dream_engine.py       ← Overnight synthesis     │
  │  shadow_drafter.py     ← Background drafting     │
  │  auto_annotator.py     ← Stata/R translation     │
  └──────────┬──────────────────────────────────────┘
             ▼
  ┌─ Output Layer ──────────────────────────────────┐
  │  neural_health_hud.py  ← Visual dashboard        │
  │  notion_bridge.py      ← Push back to Notion     │
  └─────────────────────────────────────────────────┘
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.connectome_master")


class SovereignBrain:
    """The Master Connectome: one controller to bind them all.

    This is the single entry point for the entire Citadel.
    Every morning, you run:
        brain = SovereignBrain()
        brain.ignite()

    And the entire system comes alive.

    Usage:
        brain = SovereignBrain()

        # Morning boot — full ignition sequence
        brain.ignite()

        # Process a thought — full metabolic cycle
        result = brain.metabolize("SNAP diffusion masks state capacity")

        # Deep-click on a syllabus week
        result = brain.deep_click(week=7)

        # Forensic audit a paper
        result = brain.audit_paper("/path/to/paper.pdf")

        # Evening shutdown
        brain.sovereign_collapse()
    """

    def __init__(self, bolt_path: str = ""):
        try:
            from server.vault_config import VAULT_ROOT
            _default = str(VAULT_ROOT)
        except ImportError:
            _default = os.environ.get("EDITH_DATA_ROOT", ".")
        self._bolt_path = bolt_path or _default
        self._state = "dormant"
        self._boot_time: float = 0
        self._cycle_count: int = 0
        self._modules: dict = {}

    # ─── Module Loader ────────────────────────────────────────────

    def _load(self, name: str):
        """Lazy-load a module. Returns None if unavailable."""
        if name in self._modules:
            return self._modules[name]

        loaders = {
            "brain": lambda: __import__("server.citadel_neural_net", fromlist=["edith_brain"]).edith_brain,
            "connectome": lambda: __import__("server.connectome", fromlist=["connectome"]).connectome,
            "synapse": lambda: __import__("server.synapse_bridge", fromlist=["synapse_bridge"]).synapse_bridge,
            "forensics": lambda: __import__("server.forensic_audit", fromlist=["forensic_orchestrator"]).forensic_orchestrator,
            "method_lab": lambda: __import__("server.method_lab", fromlist=["MethodLab"]).MethodLab(),
            "graph": lambda: __import__("server.graph_vector_engine", fromlist=["graph_engine"]).graph_engine,
            "dream": lambda: __import__("server.dream_engine", fromlist=["dream_engine"]).dream_engine,
            "shadow": lambda: __import__("server.shadow_drafter", fromlist=["shadow_drafter"]).shadow_drafter,
            "notion": lambda: __import__("server.notion_bridge", fromlist=["notion_bridge"]).notion_bridge,
            "annotator": lambda: __import__("server.auto_annotator", fromlist=["auto_annotator"]).auto_annotator,
            "pinner": lambda: __import__("server.memory_pinning", fromlist=["memory_pinner"]).memory_pinner,
            "hud": lambda: __import__("server.neural_health_hud", fromlist=["neural_hud"]).neural_hud,
        }

        try:
            self._modules[name] = loaders[name]()
            return self._modules[name]
        except Exception as e:
            log.debug(f"§MASTER: Module '{name}' unavailable: {e}")
            return None

    # ═══════════════════════════════════════════════════════════════
    # IGNITION — The Morning Boot
    # ═══════════════════════════════════════════════════════════════

    def ignite(self) -> dict:
        """The full morning ignition sequence.

        "You walk into your office in Lubbock and plug in the Bolt.
        The HUD wakes up, glowing cerulean."
        """
        t0 = time.time()
        self._boot_time = t0
        checks = []

        # 1. Hardware Handshake — Verify Bolt
        bolt_ok = Path(self._bolt_path).exists()
        checks.append(("Bolt SSD", "connected" if bolt_ok else "DISCONNECTED"))

        if not bolt_ok:
            self._state = "error"
            return {"state": "error", "reason": "Bolt not connected", "checks": checks}

        # 2. Memory Pinning — Pin Ancestral Knowledge to M4 RAM
        pinner = self._load("pinner")
        if pinner:
            try:
                pin_result = pinner.auto_pin_on_connect()
                checks.append(("Memory Pinning", pin_result.get("action", "done")))
            except Exception:
                checks.append(("Memory Pinning", "skipped"))
        else:
            checks.append(("Memory Pinning", "module offline"))

        # 3. Boot the Neural Net (4 Lobes)
        brain = self._load("brain")
        if brain:
            brain_result = brain.boot()
            checks.append(("Neural Net", brain_result.get("state", "unknown")))
        else:
            checks.append(("Neural Net", "offline"))

        # 4. Initialize Knowledge Graph
        graph = self._load("graph")
        if graph:
            stats = graph.get_graph_stats()
            checks.append(("Knowledge Graph",
                           f"{stats['total_entities']} entities, {stats['total_relationships']} relationships"))
        else:
            checks.append(("Knowledge Graph", "offline"))

        # 5. Sync Notion Mirror
        notion = self._load("notion")
        if notion:
            checks.append(("Notion Bridge", f"{notion.status.get('pages_mirrored', 0)} pages mirrored"))
        else:
            checks.append(("Notion Bridge", "offline"))

        # 6. Auto-audit new PDFs
        forensics = self._load("forensics")
        if forensics:
            audit_result = forensics.auto_audit_on_connect()
            new_papers = audit_result.get("new_papers", 0)
            checks.append(("Forensic Ingestion", f"{new_papers} new papers detected"))
        else:
            checks.append(("Forensic Ingestion", "offline"))

        # 7. Scan for new Stata output
        annotator = self._load("annotator")
        if annotator:
            scan = annotator.scan_and_annotate()
            checks.append(("Auto-Annotator", f"{scan.get('annotated', 0)} new annotations"))
        else:
            checks.append(("Auto-Annotator", "offline"))

        # 8. Load Dream Engine results from last night
        dream = self._load("dream")
        if dream:
            checks.append(("Dream Engine", f"{dream.status.get('total_bridges_found', 0)} bridges found"))
        else:
            checks.append(("Dream Engine", "offline"))

        # 9. Initialize Synapse Bridge
        synapse = self._load("synapse")
        if synapse:
            weak = synapse.status.get("weak_claims", 0)
            checks.append(("Truth Maintenance", f"{weak} weak claims" if weak else "all claims strong"))
        else:
            checks.append(("Truth Maintenance", "offline"))

        # 10. Generate HUD
        hud = self._load("hud")
        if hud:
            checks.append(("Neural HUD", "rendering"))
        else:
            checks.append(("Neural HUD", "offline"))

        self._state = "active"
        elapsed = time.time() - t0

        # Build the greeting
        greeting = self._morning_greeting(checks)

        return {
            "state": "active",
            "checks": [{"system": c[0], "status": c[1]} for c in checks],
            "elapsed_seconds": round(elapsed, 2),
            "greeting": greeting,
        }

    def _morning_greeting(self, checks: list) -> str:
        """Generate the Winnie morning greeting."""
        online = sum(1 for _, status in checks if status not in ("offline", "DISCONNECTED", "module offline"))
        total = len(checks)

        greeting = f"Good morning. {online}/{total} systems online.\n\n"

        # Mention new papers
        for name, status in checks:
            if name == "Forensic Ingestion" and "new papers" in status:
                count = status.split()[0]
                if count != "0":
                    greeting += f"📋 I found {count} new papers to audit.\n"

        # Mention dream results
        for name, status in checks:
            if name == "Dream Engine" and "bridges" in status:
                count = status.split()[0]
                if count != "0":
                    greeting += f"🌉 I found {count} hidden bridges while you slept.\n"

        # Mention weak claims
        for name, status in checks:
            if name == "Truth Maintenance" and "weak" in status:
                greeting += f"⚠️ {status} — review before your committee meeting.\n"

        greeting += "\nShall we ignite the Cockpit?"
        return greeting

    # ═══════════════════════════════════════════════════════════════
    # METABOLIZE — The Core Intellectual Heartbeat
    # ═══════════════════════════════════════════════════════════════

    def metabolize(self, stimulus: str) -> dict:
        """The core metabolic cycle: Intake → Audit → Map → Proof → Store.

        Every thought passes through every lobe.
        Every interaction strengthens the synaptic weights.
        """
        if self._state != "active":
            self.ignite()

        t0 = time.time()
        result = {"stimulus": stimulus[:200], "lobes": {}}

        # LOBE 1: Prefrontal Cortex — Multi-agent analysis
        brain = self._load("brain")
        if brain:
            think_result = brain.think(stimulus)
            result["lobes"]["cortex"] = {
                "concepts_linked": think_result.get("concepts_linked", 0),
                "agents": think_result.get("lobes", {}).get("cortex", {}),
            }

        # LOBE 2: Connectome — Logic Audit + Causal Proof
        conn = self._load("connectome")
        if conn:
            flow = conn.metabolic_flow(stimulus)
            result["lobes"]["connectome"] = {
                "consistent": flow.get("audit", {}).get("consistent", True),
                "confidence": flow.get("audit", {}).get("confidence", 0),
                "proofs": len(flow.get("proofs", [])),
                "hud_action": flow.get("hud_action", "none"),
                "future_project": flow.get("future_project"),
            }

        # LOBE 3: Synapse Bridge — Connect thought to data
        synapse = self._load("synapse")
        if synapse:
            bridge = synapse.bridge_thought_to_data(stimulus)
            result["lobes"]["synapse"] = bridge.to_dict()

            # Check voice
            voice = synapse.check_voice(stimulus)
            if voice.get("feedback"):
                result["voice_alert"] = voice["feedback"]

        # LOBE 4: Shadow Drafter — Add to background draft
        shadow = self._load("shadow")
        if shadow:
            shadow.add_highlight(
                text=stimulus,
                source_title="Live Session",
                category="",  # Auto-classify
            )
            result["lobes"]["shadow"] = shadow.status

        # LOBE 5: Graph Engine — Update knowledge graph
        graph = self._load("graph")
        if graph:
            # Record note path if concepts are linked
            concepts = result.get("lobes", {}).get("cortex", {}).get("agents", {})
            theorist = concepts.get("theorist", {})
            fits = theorist.get("chapter_fits", [])
            if len(fits) >= 2:
                graph.record_note_path(
                    fits[0].get("chapter", ""),
                    fits[1].get("chapter", ""),
                    context=stimulus[:200],
                )

        self._cycle_count += 1
        elapsed = time.time() - t0

        result["cycle"] = self._cycle_count
        result["elapsed_ms"] = round(elapsed * 1000, 1)
        result["metabolized"] = True

        return result

    # ═══════════════════════════════════════════════════════════════
    # DEEP-CLICK — The Recursive Drill-Down
    # ═══════════════════════════════════════════════════════════════

    def deep_click(self, topic: str = "", week: int = 0) -> dict:
        """The Deep-Click: recursive drill-down into a topic.

        Pull readings, sync Notion, prime the Cockpit, generate crash courses.
        """
        result = {"topic": topic or f"Week {week}", "layers": {}}

        # Layer 1: Pull related literature from vault
        graph = self._load("graph")
        if graph:
            query_result = graph.query(topic or f"syllabus week {week}")
            result["layers"]["literature"] = {
                "vector_matches": len(query_result.get("vector_results", [])),
                "graph_entities": len(query_result.get("graph_entities", [])),
            }

        # Layer 2: Recall Notion notes on this topic
        notion = self._load("notion")
        if notion:
            recalls = notion.recall(topic or f"week {week}")
            result["layers"]["notion_recall"] = {
                "notes_found": len(recalls),
                "titles": [r["title"] for r in recalls[:5]],
            }

        # Layer 3: Generate methodology crash course if needed
        lab = self._load("method_lab")
        if lab and topic:
            # Check if topic mentions a method
            method_map = {
                "rdd": "rdd", "regression discontinuity": "rdd",
                "iv": "iv", "instrumental variable": "iv",
                "did": "did", "difference-in-difference": "did",
                "fixed effects": "fe", "panel": "fe",
                "qca": "qca", "qualitative comparative": "qca",
                "psm": "psm", "propensity score": "psm",
            }
            for keyword, method_id in method_map.items():
                if keyword in topic.lower():
                    course = lab.generate_short_course(method_id)
                    if "error" not in course:
                        result["layers"]["crash_course"] = {
                            "method": method_id,
                            "available": True,
                        }
                    break

        # Layer 4: Prime the workspace
        result["layers"]["cockpit"] = {"status": "primed", "topic": topic}

        # Metabolize the deep-click itself
        self.metabolize(f"Deep-click on {topic or f'Week {week}'}")

        return result

    # ═══════════════════════════════════════════════════════════════
    # AUDIT PAPER — Full Forensic Pipeline
    # ═══════════════════════════════════════════════════════════════

    def audit_paper(self, pdf_path: str, **kwargs) -> dict:
        """Full forensic audit on a paper.

        "Winnie, perform a Full Forensic Audit on this text."
        """
        forensics = self._load("forensics")
        if not forensics:
            return {"error": "Forensics module offline"}

        # Run the full pipeline
        result = forensics.full_pipeline(pdf_path, **kwargs)

        # Ingest into knowledge graph
        graph = self._load("graph")
        if graph:
            graph.ingest_document(pdf_path)

        # Metabolize the audit
        title = result.get("audit", {}).get("paper", {}).get("title", "")
        if title:
            self.metabolize(f"Forensic audit of: {title}")

        return result

    # ═══════════════════════════════════════════════════════════════
    # DREAM — Overnight Synthesis
    # ═══════════════════════════════════════════════════════════════

    def dream(self) -> dict:
        """Enter the REM cycle for overnight synthesis."""
        result = {}

        # Run dream engine
        dream = self._load("dream")
        if dream:
            result["dreams"] = dream.dream()

        # Consolidate neural net memory
        brain = self._load("brain")
        if brain:
            result["consolidation"] = brain.consolidate()

        # Generate shadow drafts from today's highlights
        shadow = self._load("shadow")
        if shadow:
            draft = shadow.generate_draft(
                title=f"Shadow Draft: {time.strftime('%B %d, %Y')}",
            )
            if draft.word_count > 0:
                save_result = shadow.save_draft(draft)
                result["shadow_draft"] = {
                    "words": draft.word_count,
                    "saved": save_result,
                }

        # Cross-pollination audit
        graph = self._load("graph")
        if graph:
            conflicts = graph.cross_pollination_audit()
            result["contradictions"] = conflicts

        return result

    # ═══════════════════════════════════════════════════════════════
    # SOVEREIGN COLLAPSE — Evening Shutdown
    # ═══════════════════════════════════════════════════════════════

    def sovereign_collapse(self) -> dict:
        """Sovereign Collapse: archive, dream, and shut down.

        "You say 'Winnie, archive and Dream.' The HUD vanishes.
        The M4 is empty. You carry your brain in your pocket."
        """
        result = {"actions": []}

        # 1. Generate final shadow draft
        shadow = self._load("shadow")
        if shadow and shadow.status.get("highlights", 0) > 0:
            draft = shadow.generate_draft(
                title=f"End-of-Day Summary: {time.strftime('%B %d, %Y')}",
            )
            shadow.save_draft(draft)
            result["actions"].append(f"Shadow draft saved ({draft.word_count} words)")

        # 2. Run dream cycle
        dream_result = self.dream()
        bridges = dream_result.get("dreams", {}).get("bridges_discovered", 0)
        result["actions"].append(f"Dream cycle complete ({bridges} bridges found)")

        # 3. Save all neural state
        brain = self._load("brain")
        if brain:
            shutdown = brain.shutdown()
            result["actions"].append(
                f"Neural net saved ({shutdown.get('connections_saved', 0)} connections)"
            )

        # 4. Save knowledge graph
        graph = self._load("graph")
        if graph:
            graph._save_graph()
            result["actions"].append("Knowledge graph persisted to Bolt")

        # 5. Save synapse state
        synapse = self._load("synapse")
        if synapse:
            synapse._save_state()
            result["actions"].append("Synapse bridge state saved")

        self._state = "dormant"
        uptime = time.time() - self._boot_time if self._boot_time else 0

        result["state"] = "dormant"
        result["uptime_minutes"] = round(uptime / 60, 1)
        result["cycles_completed"] = self._cycle_count
        result["message"] = (
            "The Mac is empty. Your brain is on the Bolt. "
            f"Good night. ({self._cycle_count} thoughts metabolized today.)"
        )

        return result

    # ═══════════════════════════════════════════════════════════════
    # HUD — Render the Neural Health Dashboard
    # ═══════════════════════════════════════════════════════════════

    def render_hud(self) -> str:
        """Render the Neural Health HUD as HTML."""
        hud = self._load("hud")
        if hud:
            return hud.render_html()
        return "<html><body><h1>HUD Offline</h1></body></html>"

    def hud_snapshot(self) -> dict:
        """Get HUD data as JSON."""
        hud = self._load("hud")
        if hud:
            return hud.snapshot()
        return {"state": "offline"}

    @property
    def status(self) -> dict:
        return {
            "state": self._state,
            "cycles": self._cycle_count,
            "modules_loaded": list(self._modules.keys()),
            "bolt": Path(self._bolt_path).exists(),
        }


# ═══════════════════════════════════════════════════════════════════
# The Global Instance — The One Brain
# ═══════════════════════════════════════════════════════════════════

sovereign_brain = SovereignBrain()


# Quick API for use from chat/API routes
def ignite() -> dict:
    """Morning ignition."""
    return sovereign_brain.ignite()

def metabolize(thought: str) -> dict:
    """Process a thought through the full brain."""
    return sovereign_brain.metabolize(thought)

def deep_click(topic: str = "", week: int = 0) -> dict:
    """Deep-click on a topic or syllabus week."""
    return sovereign_brain.deep_click(topic=topic, week=week)

def audit(pdf_path: str) -> dict:
    """Full forensic audit on a paper."""
    return sovereign_brain.audit_paper(pdf_path)

def dream() -> dict:
    """Overnight synthesis."""
    return sovereign_brain.dream()

def collapse() -> dict:
    """Evening shutdown."""
    return sovereign_brain.sovereign_collapse()

def hud() -> str:
    """Render the HUD."""
    return sovereign_brain.render_hud()
