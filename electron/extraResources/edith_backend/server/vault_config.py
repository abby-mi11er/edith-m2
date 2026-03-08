"""
vault_config.py — The Sovereign Vault Path Registry
====================================================
Single source of truth for ALL data paths in EDITH.
Every module that reads/writes research data imports from here.

The Vault can live on:
  - Oyen Bolt SSD:     /Volumes/OYEN_BOLT/Vault
  - Portable drive:    /Volumes/edith drive/Vault
  - Local fallback:    ~/.edith_cache

Set VAULT_ROOT in .env to point to your drive.
"""
import os
from pathlib import Path

# ── Detect vault location ──────────────────────────────────
_env_root = os.getenv("VAULT_ROOT", "")
_env_data_root = os.getenv("EDITH_DATA_ROOT", "")
_candidates = [
    Path(_env_root) if _env_root else None,
    Path(_env_data_root) / "Vault" if _env_data_root else None,
    Path.home() / ".edith_cache",
]

VAULT_ROOT: Path = Path.home() / ".edith_cache"  # fallback
for candidate in _candidates:
    if candidate and candidate.exists():
        VAULT_ROOT = candidate
        break

BOLT_MOUNTED = VAULT_ROOT != Path.home() / ".edith_cache"

# ── Corpus (the Living Library) ────────────────────────────
CORPUS_DIR    = VAULT_ROOT / "Corpus" / "Vault"
ARCHIVES_DIR  = VAULT_ROOT / "Corpus" / "Archives"
DISCOVERY_DIR = VAULT_ROOT / "Corpus" / "Discovery"

# ── Connectome (the Semantic Core) ─────────────────────────
VECTORS_DIR   = VAULT_ROOT / "Connectome" / "Vectors"
GRAPH_DIR     = VAULT_ROOT / "Connectome" / "Graph"
SNAPSHOTS_DIR = VAULT_ROOT / "Connectome" / "Snapshots"

# ── Forge (Self-Improvement) ──────────────────────────────
TRAINING_DIR  = VAULT_ROOT / "Forge" / "Fine_Tuning"
RUNS_DIR      = VAULT_ROOT / "Forge" / "Runs"
AUDIT_DIR     = VAULT_ROOT / "Forge" / "Audit"

# ── Database paths ─────────────────────────────────────────
VAULT_DB      = VAULT_ROOT / "vault.db"
MEMORY_DB     = VAULT_ROOT / "edith_memory.sqlite3"
FEEDBACK_DB   = VAULT_ROOT / "feedback.sqlite3"

# ── Ensure all directories exist ──────────────────────────
def ensure_vault_dirs():
    """Create the full vault directory tree if it doesn't exist."""
    for d in [CORPUS_DIR, ARCHIVES_DIR, DISCOVERY_DIR,
              VECTORS_DIR, GRAPH_DIR, SNAPSHOTS_DIR,
              TRAINING_DIR, RUNS_DIR, AUDIT_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    # §M4-1: Apply SQLite pragmas after dirs are ready
    configure_sqlite_pragmas()


# ── §M4-1: SQLite WAL + Memory-Mapped I/O ─────────────────────────
import sqlite3
import logging

_vault_log = logging.getLogger("edith.vault")


def configure_sqlite_pragmas():
    """§M4-1: Configure SQLite for hardware-aware performance.

    WAL mode: prevents M4 from locking the database during
    overnight jobs if the Bolt is later plugged into the M2.

    mmap_size: lets M4 treat the Oyen Bolt SSD as virtual RAM,
    speeding up Forensic Audits and Knowledge Graph traversals.
    M4 gets 256MB mmap; M2 gets 32MB.
    """
    mmap_mb = int(os.getenv("SQLITE_MMAP_MB", "64"))
    mmap_bytes = mmap_mb * 1024 * 1024

    for db_path in [VAULT_DB, MEMORY_DB, FEEDBACK_DB]:
        if not db_path.exists():
            continue
        try:
            # §FIX B2: Context manager prevents connection leak on PRAGMA failure
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.cursor()
                cursor.execute("PRAGMA journal_mode=WAL;")
                # §SEC: int() ensures mmap_bytes is numeric — prevents injection via env var
                cursor.execute("PRAGMA mmap_size=?;", (int(mmap_bytes),))
                cursor.execute("PRAGMA cache_size=?;", (-int(mmap_mb * 1024),))
                conn.commit()
            _vault_log.info(f"§M4-1: {db_path.name} → WAL + mmap={mmap_mb}MB")
        except Exception as e:
            _vault_log.warning(f"§M4-1: Failed to configure {db_path.name}: {e}")

def get_vault_status() -> dict:
    """Return vault health for the HUD."""
    return {
        "root": str(VAULT_ROOT),
        "mounted": BOLT_MOUNTED,
        "corpus_files": len(list(CORPUS_DIR.glob("**/*"))) if CORPUS_DIR.exists() else 0,
        "vectors_exists": VECTORS_DIR.exists(),
        "snapshots": len(list(SNAPSHOTS_DIR.glob("*.json"))) if SNAPSHOTS_DIR.exists() else 0,
    }
