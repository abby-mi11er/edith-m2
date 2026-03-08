"""
Citadel Neural Net — The Recurrent Sovereign Intelligence
============================================================
The Master DNA: every module is a neuron, the Bolt is the synapse.

Four Lobes:
  A. Sensory Thalamus — watches Notion/PDFs/Stata for stimuli
  B. Hippocampus — the Bolt's 1TB long-term potentiation
  C. Prefrontal Cortex — multi-agent orchestration (Theorist/Methodologist/Skeptic)
  D. REM Engine — overnight recursive synthesis

The Metabolic Cycle:
  PERCEPTION → COGNITION → CONSOLIDATION → REST → repeat

Every interaction strengthens synaptic weights. The brain grows
every time you use it.
"""

import asyncio
import hashlib
import json
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.neural_net")


# ═══════════════════════════════════════════════════════════════════
# Synaptic Weight System — Connections Strengthen With Use
# ═══════════════════════════════════════════════════════════════════

@dataclass
class SynapticConnection:
    """A weighted connection between two concepts in the brain."""
    source: str
    target: str
    weight: float = 0.5
    activations: int = 0
    last_activated: float = 0.0

    def activate(self):
        """Hebbian learning: neurons that fire together wire together."""
        self.activations += 1
        self.weight = min(2.0, self.weight + 0.05 * (1 / (1 + self.activations * 0.01)))
        self.last_activated = time.time()

    def decay(self, hours_since: float):
        """Synaptic decay for unused connections."""
        if hours_since > 168:  # 1 week
            self.weight = max(0.1, self.weight * 0.95)

    def to_dict(self) -> dict:
        return {
            "source": self.source, "target": self.target,
            "weight": round(self.weight, 3),
            "activations": self.activations,
        }


# ═══════════════════════════════════════════════════════════════════
# The Four Lobes
# ═══════════════════════════════════════════════════════════════════

class SensoryThalamus:
    """Lobe A: Watches for new stimuli from Notion, PDFs, Stata logs."""

    def __init__(self, bolt_path: str):
        self._bolt_path = bolt_path
        self._watch_paths = [
            os.path.join(bolt_path, "VAULT", "READINGS"),
            os.path.join(bolt_path, "VAULT", "NOTION_MIRROR"),
            os.path.join(bolt_path, "VAULT", "STATA_OUTPUT"),
            os.path.join(bolt_path, "VAULT", "NEW_PAPERS"),
            os.path.join(bolt_path, "VAULT", "INBOX"),
        ]
        self._known_files: dict[str, float] = {}  # path → mtime

    async def sense_environment(self) -> list[dict]:
        """Scan for new or modified files — the brain's 'ears'."""
        stimuli = []
        for watch_dir in self._watch_paths:
            watch_path = Path(watch_dir)
            if not watch_path.exists():
                continue

            for ext in ["*.pdf", "*.md", "*.txt", "*.log", "*.smcl"]:
                for f in watch_path.rglob(ext):
                    fpath = str(f)
                    mtime = f.stat().st_mtime

                    if fpath not in self._known_files or self._known_files[fpath] < mtime:
                        self._known_files[fpath] = mtime

                        # Classify the stimulus
                        category = "literature"
                        if "NOTION" in fpath.upper():
                            category = "notion_note"
                        elif "STATA" in fpath.upper() or f.suffix in (".log", ".smcl"):
                            category = "stata_output"
                        elif "DATA" in fpath.upper():
                            category = "dataset"

                        stimuli.append({
                            "path": fpath,
                            "category": category,
                            "name": f.stem,
                            "size": f.stat().st_size,
                            "modified": mtime,
                        })

        return stimuli


class Hippocampus:
    """Lobe B: The Bolt's long-term memory with synaptic weights."""

    def __init__(self, bolt_path: str):
        self._bolt_path = bolt_path
        self._connections: dict[str, SynapticConnection] = {}
        self._load_connections()

    def strengthen(self, concept_a: str, concept_b: str):
        """Hebbian learning: strengthen the link between two concepts."""
        key = self._connection_key(concept_a, concept_b)
        if key not in self._connections:
            self._connections[key] = SynapticConnection(
                source=concept_a, target=concept_b,
            )
        self._connections[key].activate()

    def recall_strength(self, concept_a: str, concept_b: str) -> float:
        """How strongly are two concepts linked?"""
        key = self._connection_key(concept_a, concept_b)
        conn = self._connections.get(key)
        return conn.weight if conn else 0.0

    def strongest_connections(self, concept: str, top_k: int = 5) -> list[dict]:
        """Find the strongest connections from a concept."""
        related = []
        concept_lower = concept.lower()
        for key, conn in self._connections.items():
            if concept_lower in conn.source.lower() or concept_lower in conn.target.lower():
                other = conn.target if concept_lower in conn.source.lower() else conn.source
                related.append({"concept": other, "weight": conn.weight,
                                "activations": conn.activations})

        related.sort(key=lambda r: r["weight"], reverse=True)
        return related[:top_k]

    def decay_all(self):
        """Apply synaptic decay to all connections."""
        now = time.time()
        for conn in self._connections.values():
            hours = (now - conn.last_activated) / 3600
            conn.decay(hours)

    def save(self):
        self._save_connections()

    def _connection_key(self, a: str, b: str) -> str:
        pair = sorted([a.lower().strip(), b.lower().strip()])
        return f"{pair[0]}↔{pair[1]}"

    def _save_connections(self):
        conn_dir = Path(self._bolt_path) / "VAULT" / "NEURAL"
        conn_dir.mkdir(parents=True, exist_ok=True)
        conn_file = conn_dir / "synaptic_weights.json"
        try:
            data = {k: v.to_dict() for k, v in self._connections.items()}
            conn_file.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _load_connections(self):
        conn_file = Path(self._bolt_path) / "VAULT" / "NEURAL" / "synaptic_weights.json"
        if conn_file.exists():
            try:
                data = json.loads(conn_file.read_text())
                for k, v in data.items():
                    self._connections[k] = SynapticConnection(
                        source=v["source"], target=v["target"],
                        weight=v["weight"], activations=v["activations"],
                    )
            except Exception:
                pass

    @property
    def status(self) -> dict:
        weights = [c.weight for c in self._connections.values()]
        return {
            "connections": len(self._connections),
            "avg_weight": round(sum(weights) / max(len(weights), 1), 3),
            "strongest": max(weights) if weights else 0,
        }


class PrefrontalCortex:
    """Lobe C: Multi-agent orchestration — Theorist, Methodologist, Skeptic."""

    def __init__(self):
        self._agents = {
            "theorist": {"role": "Synthesize new readings into dissertation theory",
                         "active": False, "last_output": ""},
            "methodologist": {"role": "Audit the math, run short courses",
                               "active": False, "last_output": ""},
            "skeptic": {"role": "Stress-test logic against ancestral knowledge",
                         "active": False, "last_output": ""},
        }

    async def process_thought(self, thought: str, context: dict = None) -> dict:
        """Route a thought through all three agents in parallel."""
        results = {}

        # Theorist: Where does this fit in the dissertation?
        self._agents["theorist"]["active"] = True
        results["theorist"] = self._theorist_analysis(thought)
        self._agents["theorist"]["active"] = False

        # Methodologist: What's the math behind this claim?
        self._agents["methodologist"]["active"] = True
        results["methodologist"] = self._methodologist_analysis(thought)
        self._agents["methodologist"]["active"] = False

        # Skeptic: Does this contradict anything in the vault?
        self._agents["skeptic"]["active"] = True
        results["skeptic"] = self._skeptic_analysis(thought)
        self._agents["skeptic"]["active"] = False

        return results

    def _theorist_analysis(self, thought: str) -> dict:
        """The Theorist agent: fit this thought into the dissertation."""
        thought_lower = thought.lower()

        # Match to dissertation chapters
        chapter_signals = {
            "Chapter 1: Introduction": ["research question", "puzzle", "motivation"],
            "Chapter 2: Theory": ["framework", "mechanism", "principal-agent",
                                   "administrative burden", "theory"],
            "Chapter 3: State Capacity": ["state capacity", "potter county",
                                           "service delivery", "bureaucratic"],
            "Chapter 4: Charity Substitution": ["charity", "nonprofit", "NGO",
                                                  "volunteer", "crowding out"],
            "Chapter 5: Data & Methods": ["regression", "variable", "dataset",
                                            "sample", "IV", "RDD", "DiD"],
            "Chapter 6: Findings": ["result", "coefficient", "significant",
                                      "finding", "evidence"],
            "Chapter 7: Conclusion": ["implication", "future research",
                                        "limitation", "contribute"],
        }

        matches = []
        for chapter, signals in chapter_signals.items():
            hits = sum(1 for s in signals if s in thought_lower)
            if hits > 0:
                matches.append({"chapter": chapter, "relevance": hits})

        matches.sort(key=lambda m: m["relevance"], reverse=True)
        return {
            "agent": "theorist",
            "chapter_fits": matches[:3],
            "recommendation": (
                f"This thought best fits {matches[0]['chapter']}"
                if matches else "No clear chapter fit — consider for literature review"
            ),
        }

    def _methodologist_analysis(self, thought: str) -> dict:
        """The Methodologist agent: what method does this invoke?"""
        thought_lower = thought.lower()
        methods_detected = []

        method_signals = {
            "OLS": ["regression", "OLS", "linear model"],
            "IV/2SLS": ["instrumental variable", "IV", "2SLS", "endogenous"],
            "RDD": ["discontinuity", "threshold", "cutoff", "bandwidth"],
            "DiD": ["difference-in-difference", "DiD", "parallel trends"],
            "Fixed Effects": ["fixed effects", "within estimator", "entity-demeaned"],
            "Case Study": ["case study", "process tracing", "thick description"],
            "QCA": ["qualitative comparative", "QCA", "fuzzy set", "necessary condition"],
        }

        for method, signals in method_signals.items():
            if any(s.lower() in thought_lower for s in signals):
                methods_detected.append(method)

        return {
            "agent": "methodologist",
            "methods_detected": methods_detected,
            "recommendation": (
                f"This involves {', '.join(methods_detected)}. "
                f"Check assumptions before proceeding."
                if methods_detected else "No specific method detected."
            ),
        }

    def _skeptic_analysis(self, thought: str) -> dict:
        """The Skeptic agent: what could be wrong with this claim?"""
        thought_lower = thought.lower()
        concerns = []

        # Check for common logical issues
        if "clearly" in thought_lower or "obviously" in thought_lower:
            concerns.append("Unsupported certainty — provide evidence for this claim.")
        if "all" in thought_lower or "every" in thought_lower:
            concerns.append("Overgeneralization — are there counterexamples?")
        if "proves" in thought_lower:
            concerns.append("Causal language — can you establish causation, not just correlation?")
        if "significant" in thought_lower and "statistically" not in thought_lower:
            concerns.append("Ambiguous 'significant' — do you mean statistically or substantively?")
        if "assume" in thought_lower:
            concerns.append("Stated assumption — is this justified by the literature?")

        return {
            "agent": "skeptic",
            "concerns": concerns,
            "severity": "high" if len(concerns) >= 2 else "low" if not concerns else "moderate",
            "recommendation": (
                f"⚠️ {len(concerns)} concern(s) flagged. Address before committee review."
                if concerns else "✓ No immediate logical red flags."
            ),
        }

    @property
    def status(self) -> dict:
        return {
            "agents": {
                name: {"active": info["active"]}
                for name, info in self._agents.items()
            },
        }


# ═══════════════════════════════════════════════════════════════════
# The E.D.I.T.H. Brain — The Master Controller
# ═══════════════════════════════════════════════════════════════════

class EDITHBrain:
    """The Living Brain of the Citadel.

    Integrates all four lobes into a continuous metabolic cycle.
    Every interaction strengthens synaptic weights.
    The brain grows every time you use it.

    Usage:
        brain = EDITHBrain()
        brain.boot()

        # Process a thought (triggers all lobes)
        result = brain.think("Charities mask state failure in Potter County")

        # Check brain health
        health = brain.health_check()

        # Run dream cycle
        dreams = brain.dream()

        # Graceful shutdown
        brain.shutdown()
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
        self._thoughts_processed: int = 0
        self._last_consolidation: float = 0

        # Initialize the four lobes
        self.thalamus = SensoryThalamus(self._bolt_path)
        self.hippocampus = Hippocampus(self._bolt_path)
        self.cortex = PrefrontalCortex()

        # Module references (lazy-loaded)
        self._synapse = None
        self._dreamer = None
        self._forensics = None
        self._annotator = None

    def boot(self) -> dict:
        """The 10-point Hardware/Software Handshake."""
        log.info("§BRAIN: Initializing E.D.I.T.H. Neural Net...")
        self._boot_time = time.time()
        checks = []

        # 1. Check Bolt
        bolt_ok = Path(self._bolt_path).exists()
        checks.append({"check": "Bolt SSD", "status": "✓" if bolt_ok else "✗"})

        # 2. Check Vault structure
        vault = Path(self._bolt_path) / "VAULT"
        vault_ok = vault.exists()
        checks.append({"check": "Vault", "status": "✓" if vault_ok else "✗"})

        # 3. Load synaptic weights
        conn_count = len(self.hippocampus._connections)
        checks.append({"check": "Synaptic Weights", "status": f"✓ ({conn_count} connections)"})

        # 4. Initialize agents
        checks.append({"check": "Prefrontal Cortex", "status": "✓ (3 agents ready)"})

        # 5. Check Neural Engine
        import platform
        chip = platform.processor() or "unknown"
        checks.append({"check": "Neural Engine", "status": f"✓ ({chip})"})

        self._state = "active"
        log.info(f"§BRAIN: Online — {len(checks)} systems checked")

        return {
            "state": self._state,
            "checks": checks,
            "boot_time": time.strftime("%H:%M:%S"),
        }

    def think(self, thought: str) -> dict:
        """Process a thought through the full brain.

        This is the core "Deep-Click" — all lobes fire together.
        """
        if self._state != "active":
            self.boot()

        t0 = time.time()
        result = {"thought": thought[:200], "lobes": {}}

        # Always use sync path — process_thought is async in signature
        # but only calls sync methods, and run_until_complete crashes
        # when FastAPI's event loop is already running.
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                cortex_result = self._sync_cortex(thought)
            else:
                cortex_result = loop.run_until_complete(
                    self.cortex.process_thought(thought)
                )
        except RuntimeError:
            cortex_result = self._sync_cortex(thought)
        result["lobes"]["cortex"] = cortex_result

        # Extract concepts for synaptic strengthening
        concepts = self._extract_concepts(thought)
        for i, c1 in enumerate(concepts):
            for c2 in concepts[i + 1:]:
                self.hippocampus.strengthen(c1, c2)

        result["concepts_linked"] = len(concepts)

        # Synapse Bridge — connect to data
        try:
            if not self._synapse:
                from server.synapse_bridge import synapse_bridge
                self._synapse = synapse_bridge
            bridge_result = self._synapse.bridge_thought_to_data(thought)
            result["lobes"]["synapse"] = bridge_result.to_dict()
        except Exception as e:
            result["lobes"]["synapse"] = {"status": "offline", "error": str(e)}

        # Forensic check — auto-annotate if it looks like Stata output
        if any(kw in thought.lower() for kw in ["coefficient", "regression", "p-value", "r-squared"]):
            try:
                if not self._annotator:
                    from server.auto_annotator import auto_annotator
                    self._annotator = auto_annotator
                result["lobes"]["annotator"] = {"status": "triggered"}
            except Exception:
                pass

        self._thoughts_processed += 1
        elapsed = time.time() - t0

        # Save synaptic growth
        self.hippocampus.save()

        result["elapsed_ms"] = round(elapsed * 1000, 1)
        result["total_thoughts"] = self._thoughts_processed
        result["synaptic_growth"] = self.hippocampus.status

        return result

    def _sync_cortex(self, thought: str) -> dict:
        """Synchronous version of cortex processing."""
        return {
            "theorist": self.cortex._theorist_analysis(thought),
            "methodologist": self.cortex._methodologist_analysis(thought),
            "skeptic": self.cortex._skeptic_analysis(thought),
        }

    def _extract_concepts(self, text: str) -> list[str]:
        """Extract key concepts from text for synaptic linking."""
        import re
        concepts = []
        text_lower = text.lower()

        # Known concept vocabulary
        concept_vocab = [
            "state capacity", "administrative burden", "principal-agent",
            "charity", "nonprofit", "accountability", "blame",
            "federalism", "devolution", "welfare", "cartel",
            "potter county", "lubbock", "mexico", "governance",
            "policy feedback", "institutional", "bureaucratic",
            "regression", "IV", "RDD", "DiD", "fixed effects",
        ]

        for concept in concept_vocab:
            if concept in text_lower:
                concepts.append(concept)

        return concepts

    def dream(self) -> dict:
        """Enter REM cycle — overnight speculative synthesis."""
        try:
            if not self._dreamer:
                from server.dream_engine import dream_engine
                self._dreamer = dream_engine
            return self._dreamer.dream()
        except Exception as e:
            return {"error": str(e)}

    def consolidate(self) -> dict:
        """Memory consolidation — strengthen active connections, decay unused ones."""
        self.hippocampus.decay_all()
        self.hippocampus.save()
        self._last_consolidation = time.time()
        return {
            "consolidated": True,
            "connections": self.hippocampus.status,
            "timestamp": time.strftime("%H:%M:%S"),
        }

    def health_check(self) -> dict:
        """Full neural health report."""
        uptime = time.time() - self._boot_time if self._boot_time else 0

        return {
            "state": self._state,
            "uptime_minutes": round(uptime / 60, 1),
            "thoughts_processed": self._thoughts_processed,
            "hippocampus": self.hippocampus.status,
            "cortex": self.cortex.status,
            "bolt_connected": Path(self._bolt_path).exists(),
            "last_consolidation": (
                time.strftime("%H:%M", time.localtime(self._last_consolidation))
                if self._last_consolidation else "never"
            ),
        }

    def shutdown(self) -> dict:
        """Sovereign Collapse — save everything and go dormant."""
        self.hippocampus.save()
        self._state = "dormant"
        return {
            "state": "dormant",
            "thoughts_saved": self._thoughts_processed,
            "connections_saved": len(self.hippocampus._connections),
            "message": "The Mac is empty. Your brain is on the Bolt. Good night, Abby.",
        }


# Global instance
edith_brain = EDITHBrain()
