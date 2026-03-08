"""
E.D.I.T.H. Citadel Server Package
===================================
Core server modules for the Citadel research assistant.

Canonical modules (prefer these for new code):
  - security.py           — Core middleware, auth, rate limiting
  - security_hardening.py — AES encryption, anomaly detection, Bolt handshake
  - security_features.py  — Student tiers, exam lockdown, plagiarism detection
  - memory_pinning.py     — M4 Unified Memory + Bolt SSD mmap pinning
  - session_memory.py     — Cross-session conversation context
  - memory_enhancements.py — Entity extraction, pinned memories, memory search
  - retrieval_enhancements.py — RRF, temporal weighting, query intent routing
  - socratic_navigator.py — Adversarial training (5 engines, Committee of Sages)

Overlapping modules (kept for backward compat, see deprecation banners):
  - security_improvements.py  → use security.py
  - security_enhancements.py  → use security.py or security_hardening.py
  - memory_improvements.py    → use memory_enhancements.py
  - retrieval_improvements.py → use retrieval_enhancements.py
  - socratic_coach.py         → use socratic_navigator.py
  - training_enhancements.py  → use training_tools.py or training_devops.py
"""
