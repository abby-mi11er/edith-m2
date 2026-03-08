"""
Server State — Centralized Lazy-Loading State Management
=========================================================
Replaces the ~100 mutable globals scattered across main.py with a single
thread-safe state object. Modules load lazily on first access instead of
eagerly at startup, which:
  1. Eliminates import-time bottleneck (134 modules → load on demand)
  2. Makes multi-worker uvicorn safe (no shared mutable globals)
  3. Provides clean dependency injection for route handlers

Usage in route files:
    from server.server_state import state
    he = state.hybrid_engine  # lazily loaded on first access
    if state.is_available("mlx_inference"):
        result = state.mlx_inference.generate(prompt)
"""

import importlib
import logging
import threading
import time
from typing import Any, Dict, Optional

log = logging.getLogger("edith.state")


class _LazyModule:
    """Descriptor that lazily imports a module attribute on first access."""
    __slots__ = ("_module_path", "_attr_name", "_lock", "_value", "_loaded")

    def __init__(self, module_path: str, attr_name: str = ""):
        self._module_path = module_path
        self._attr_name = attr_name
        self._lock = threading.Lock()
        self._value = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return self._value
        with self._lock:
            if self._loaded:
                return self._value
            try:
                mod = importlib.import_module(self._module_path)
                if self._attr_name:
                    self._value = getattr(mod, self._attr_name)
                else:
                    self._value = mod
                self._loaded = True
                log.info(f"§LAZY: Loaded {self._module_path}"
                         f"{'.' + self._attr_name if self._attr_name else ''}")
            except Exception as e:
                log.warning(f"§LAZY: Failed to load {self._module_path}: {e}")
                self._value = None
                self._loaded = True  # Don't retry on every request
        return self._value

    @property
    def available(self) -> bool:
        if not self._loaded:
            self.load()
        return self._value is not None

    def reset(self):
        """Reset for testing."""
        with self._lock:
            self._value = None
            self._loaded = False


class ServerState:
    """Centralized, thread-safe server state with lazy module loading.

    Provides three access patterns:
      1. state.module_name → lazy-loads and caches the module/attribute
      2. state.is_available("module_name") → check if loadable
      3. state.config → immutable server config (set once at startup)
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._config: Dict[str, Any] = {}
        self._runtime: Dict[str, Any] = {}
        self._worker_available = False
        self._startup_complete = False
        self._startup_time = 0.0

        # ── Lazy module registry ──────────────────────────────────
        # Heavy compute modules (candidates for worker offloading)
        self._modules: Dict[str, _LazyModule] = {
            # MLX / Neural Engine
            "mlx_inference": _LazyModule("server.mlx_inference"),
            "mlx_embeddings": _LazyModule("server.mlx_embeddings"),

            # Hybrid / Cognitive
            "hybrid_engine": _LazyModule("server.hybrid_engine", "hybrid_engine"),
            "tone_generator": _LazyModule("server.hybrid_engine", "tone_generator"),
            "cognitive_engine": _LazyModule("server.cognitive_engine"),
            "focus_mode_engage": _LazyModule("server.cognitive_engine", "engage_focus_mode"),
            "focus_mode_disengage": _LazyModule("server.cognitive_engine", "disengage_focus_mode"),

            # Citadel / Theme
            "citadel_theme": _LazyModule("server.citadel_theme", "citadel_theme"),
            "citadel_boot": _LazyModule("server.citadel_boot"),
            "vector_mapping": _LazyModule("server.vector_mapping"),
            "atlas_lod": _LazyModule("server.vector_mapping", "atlas_lod"),

            # Orchestration
            "memory_scaler": _LazyModule("server.memory_scaler"),
            "prefetcher": _LazyModule("server.prefetcher", "prefetcher"),
            "deep_dive": _LazyModule("server.deep_dive", "deep_dive_engine"),
            "peer_review": _LazyModule("server.peer_review"),
            "shadow_discovery": _LazyModule("server.shadow_discovery"),
            "vibe_coder": _LazyModule("server.vibe_coder"),
            "auto_maintenance": _LazyModule("server.auto_maintenance", "maintenance"),
            "hw_monitor": _LazyModule("server.hw_monitor"),
            "storage_manager": _LazyModule("server.storage_manager"),

            # Infrastructure
            "infrastructure": _LazyModule("server.infrastructure"),
            "security_hardening": _LazyModule("server.security"),

            # Citadel brain modules
            "citadel_neural_net": _LazyModule("server.citadel_neural_net", "EDITHBrain"),
            "dream_engine": _LazyModule("server.dream_engine", "DreamEngine"),
            "synapse_bridge": _LazyModule("server.synapse_bridge", "SynapseBridge"),

            # Forensic lab
            "paper_deconstructor": _LazyModule("server.paper_deconstructor", "PaperDeconstructor"),
            "method_lab": _LazyModule("server.method_lab", "MethodLab"),
            "lit_locator": _LazyModule("server.lit_locator", "LitLocator"),
            "forensic_audit": _LazyModule("server.forensic_audit", "ForensicAuditOrchestrator"),
            "auto_annotator": _LazyModule("server.auto_annotator", "AutoAnnotator"),

            # Discovery / Knowledge
            "knowledge_graph": _LazyModule("server.knowledge_graph", "KnowledgeGraph"),
            "scholarly_repos": _LazyModule("server.scholarly_repositories", "ScholarlyRepositories"),
            "discovery_mode": _LazyModule("server.discovery_mode"),

            # Completions
            "completions": _LazyModule("server.completions"),
            "monte_carlo": _LazyModule("server.completions", "monte_carlo"),
            "recommendation_engine": _LazyModule("server.completions", "recommendation_engine"),
            "study_session": _LazyModule("server.completions", "study_session"),

            # Pedagogy / Index
            "index_pedagogy": _LazyModule("server.index_pedagogy"),
            "pedagogy": _LazyModule("server.pedagogy"),

            # Causal / Guardrails
            "grounded_guardrails": _LazyModule("server.grounded_guardrails"),
            "causal_engine": _LazyModule("server.causal_engine", "CausalEngine"),
            "simulation_deck": _LazyModule("server.simulation_deck", "SimulationDeck"),
            "oracle_engine": _LazyModule("server.oracle_engine", "OracleEngine"),

            # Jarvis / Agents
            "jarvis_layer": _LazyModule("server.jarvis_layer"),
            "antigravity_engine": _LazyModule("server.antigravity_engine"),

            # Socratic / Navigator
            "socratic_navigator": _LazyModule("server.socratic_navigator"),
            "metabolic_monitor": _LazyModule("server.metabolic_monitor"),
            "neural_health_hud": _LazyModule("server.neural_health_hud", "NeuralHealthHUD"),
            "citadel_bridge": _LazyModule("server.citadel_bridge"),
            "connectome_master": _LazyModule("server.citadel_connectome_master"),
            "operational_rhythm": _LazyModule("server.operational_rhythm"),

            # Training
            "training_enhancements": _LazyModule("server.training_tools"),
            "training_devops": _LazyModule("server.training_devops", "MultiCorpusManager"),

            # Indexing
            "indexing_enhancements": _LazyModule("server.indexing_enhancements"),

            # Export / Notes
            "export_academic": _LazyModule("server.export_academic"),
            "export_notes": _LazyModule("server.export_notes"),
            "citation_formatter": _LazyModule("server.citation_formatter"),

            # Reasoning
            "reasoning_enhancements": _LazyModule("server.reasoning_enhancements"),

            # Memory
            "memory_enhancements": _LazyModule("server.memory_pinning"),

            # Security / RBAC
            "security_features": _LazyModule("server.security"),
            "rbac": _LazyModule("server.rbac"),
            "shared_mode": _LazyModule("server.shared_mode"),
            "tls_config": _LazyModule("server.tls_config"),

            # Desktop
            "desktop_features": _LazyModule("server.desktop_features"),

            # Connectors
            "connectors": _LazyModule("pipelines.connectors"),

            # Vault watcher
            "vault_watcher": _LazyModule("server.vault_watcher"),
        }

    def __getattr__(self, name: str) -> Any:
        """Allow state.module_name to lazily load modules."""
        if name.startswith("_"):
            raise AttributeError(name)
        modules = object.__getattribute__(self, "_modules")
        if name in modules:
            return modules[name].load()
        raise AttributeError(f"ServerState has no module '{name}'")

    def is_available(self, name: str) -> bool:
        """Check if a module is loadable without failing."""
        if name not in self._modules:
            return False
        return self._modules[name].available

    def get_lazy(self, name: str, fallback: Any = None) -> Any:
        """Get a lazily-loaded module, with a fallback if unavailable."""
        if name not in self._modules:
            return fallback
        result = self._modules[name].load()
        return result if result is not None else fallback

    # ── Config (immutable after init) ─────────────────────────────

    def set_config(self, **kwargs):
        """Set configuration values (called once at startup)."""
        self._config.update(kwargs)

    def get_config(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    @property
    def config(self) -> Dict[str, Any]:
        return dict(self._config)

    # ── Runtime state (mutable, thread-safe) ──────────────────────

    def set_runtime(self, key: str, value: Any):
        with self._lock:
            self._runtime[key] = value

    def get_runtime(self, key: str, default: Any = None) -> Any:
        return self._runtime.get(key, default)

    # ── Startup tracking ──────────────────────────────────────────

    def mark_startup_complete(self):
        self._startup_complete = True
        self._startup_time = time.time()
        log.info("§STATE: Server startup complete — all modules available on demand")

    @property
    def startup_complete(self) -> bool:
        return self._startup_complete

    # ── Worker integration ────────────────────────────────────────

    @property
    def worker_available(self) -> bool:
        return self._worker_available

    def set_worker_available(self, available: bool):
        self._worker_available = available

    # ── Warmup (background, non-blocking) ─────────────────────────

    def warmup_modules(self, module_names: list):
        """Pre-load a list of modules in a background thread.

        Unlike the old _load_orchestration_modules, this:
        - Doesn't set globals
        - Doesn't block the event loop
        - Logs but doesn't crash on failures
        """
        def _warmup():
            t0 = time.time()
            loaded = 0
            for name in module_names:
                if name in self._modules:
                    try:
                        self._modules[name].load()
                        loaded += 1
                    except Exception:
                        pass  # Already logged by _LazyModule.load()
            elapsed = time.time() - t0
            log.info(f"§STATE: Warmup complete — {loaded}/{len(module_names)} "
                     f"modules in {elapsed:.1f}s")
            self.mark_startup_complete()

        thread = threading.Thread(target=_warmup, daemon=True, name="state-warmup")
        thread.start()
        return thread

    # ── Status for diagnostics ────────────────────────────────────

    def status(self) -> Dict[str, Any]:
        """Return state summary for /api/status."""
        loaded = sum(1 for m in self._modules.values() if m._loaded and m._value is not None)
        total = len(self._modules)
        return {
            "modules_loaded": loaded,
            "modules_total": total,
            "modules_pct": round(loaded / total * 100, 1) if total else 0,
            "startup_complete": self._startup_complete,
            "worker_available": self._worker_available,
            "config_keys": list(self._config.keys()),
        }


# ── Singleton ─────────────────────────────────────────────────────
state = ServerState()


# ── Shared state used by route modules ──────────────────────────────
# Moved from main.py to break circular imports.

import threading

# Library cache
library_cache: dict = {}
library_cache_ts: float = 0.0
library_building: bool = False
library_lock = threading.Lock()

# Route metrics
route_call_counts: dict = {}
route_call_start: dict = {}

# Training autolearn lock
autolearn_lock = threading.Lock()

# Server start time
import time as _time
server_start_time: float = _time.time()
