"""
Handler Dependency Map — Guide for incremental extraction from main.py

main.py currently houses all 89 handler functions (4,170 lines).
This module documents which handlers can be safely extracted into
their route modules, and which are too coupled to move right now.

MIGRATION STATUS
================

§ORCH-7 EXTRACTION COMPLETE (85 endpoints relocated):

  ✅ orchestration.py — 18 endpoints
     Deep Dive (2), Peer Review (1), Tutor (1), Explain Term (1),
     Shadow Scan (1), Vibe Coding (4), Maintenance (2)
     → Lazy-loaded: deep_dive_engine, committee, shadow_variable,
       vibe_coder, auto_maintenance

  ✅ cognitive.py — 13 endpoints
     Graph Retrieve (1), Persona (4), Peer Review (1), Discover (1),
     Cross-Language (1), Socratic (2), Difficulty (1), Spaced Rep (4)
     → Lazy-loaded: cognitive_engine, socratic_navigator, spaced_repetition

  ✅ causal.py — 21 endpoints
     Guardrails (4), Causal Engine (7), Simulation Deck (10)
     → Lazy-loaded: grounded_guardrails, causal_engine, simulation_deck

  ✅ jarvis.py — 23 endpoints
     Ambient Watcher (4), Overnight Sandbox (4), Approval Gate (3),
     Thought Streams (2), Portable Env (2), Oracle Engine (8)
     → Lazy-loaded: jarvis_layer, oracle_engine

  ✅ antigravity.py — 10 endpoints
     Tab-to-Intent (1), Artifact Plan (1), Self-Heal (1),
     Research Memo (1), Dispatch (1), Agents (1),
     Thought Chain (1), Skill (3)
     → Lazy-loaded: antigravity_engine

REMAINING IN main.py (Cockpit, Infrastructure, Security inline blocks):
  - cockpit_atlas, cockpit_topology, cockpit_clusters, cockpit_status
    → uses: CLUSTER_CENTROIDS, CLUSTER_COLORS (module-level dicts)
  - Infrastructure (5), Security (8), Tools (10), Rhythm (7)
    → Already delegated via getattr in existing route modules

COUPLED (depend on 5+ module-level globals — extract after dependency injection):
  - chat_endpoint, chat_stream_endpoint
  - library_endpoint, library_sources_endpoint, _scan_library_sources
  - upload_file, serve_file

MIGRATION PATH
==============
1. ✅ Route modules exist with getattr() delegation (Phase 1)
2. ✅ Extract EXTRACTABLE handlers into route modules directly (Phase 2 — §ORCH-7)
3. ⬜ Create a ServerContext class to hold shared state
4. ⬜ Inject ServerContext into COUPLED handlers
5. ⬜ Move COUPLED handlers to route modules via dependency injection
"""

# This module is documentation only — no executable code.
# See server/routes/*.py for the current routing implementation.
HANDLER_COUNT = 89
EXTRACTED = 85
REMAINING = 4  # Cockpit only; Infra/Security/Tools/Rhythm use getattr delegation
