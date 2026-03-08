"""Route module registry for E.D.I.T.H."""
from server.routes import notes, openalex

# §FIX W1: Complete list of all registered route modules
# (matches registration at bottom of main.py)
ROUTER_MODULES = [
    # Phase 1: Core routes (register via .register(app))
    "brain",
    "chat",
    "doctor",
    "export",
    "indexing",
    "library",
    "pipelines",
    "reasoning",
    "research",
    "search",
    "security",
    "system",
    "training",
    # Phase 2: §ORCH-7 extracted domain routes
    "orchestration",
    "cognitive",
    "causal",
    "jarvis",
    "antigravity",
    "integrations",
    "intelligence",
    # Phase 3: Master Connector List 2026
    "connectors_hub",
    # Phase 4: Methodological Sniper
    "sniper",
]
