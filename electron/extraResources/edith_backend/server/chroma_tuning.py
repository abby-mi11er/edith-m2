"""
ChromaDB IOPS Saturation — NVMe/Thunderbolt Tuning
====================================================
Tunes ChromaDB's underlying SQLite for maximum IOPS on NVMe/Thunderbolt
drives. These pragmas turn ChromaDB from "reading through a straw" into
"drinking from a firehose."

Default ChromaDB SQLite settings vs. our M4+Thunderbolt tuning:

  journal_mode:  delete   → WAL    (concurrent reads during writes)
  synchronous:   FULL(2)  → NORMAL (2x write throughput, safe with WAL)
  cache_size:    -2000    → -65536 (64MB in-memory cache vs 2MB)
  mmap_size:     0        → 268MB  (memory-mapped I/O for NVMe)
  page_size:     4096     → 4096   (matches NVMe sector size)
  temp_store:    default  → MEMORY (temp tables in RAM)
  busy_timeout:  5000     → 30000  (longer timeout for parallel queries)
"""

import logging
import os
import sqlite3

log = logging.getLogger("edith.chroma_tune")


# Tuning profiles keyed by compute profile mode
_PROFILES = {
    "committee": {
        # M4 + Thunderbolt: maximum throughput
        "journal_mode": "WAL",
        "synchronous": "NORMAL",      # Safe with WAL, 2x write throughput
        "cache_size": -65536,          # 64MB cache (vs default 2MB)
        "mmap_size": 268435456,        # 256MB memory-mapped I/O
        "temp_store": "MEMORY",        # Temp tables in unified memory
        "busy_timeout": 60000,         # 60s timeout — indexer holds write locks for extended periods
        "wal_autocheckpoint": 1000,    # Checkpoint every 1000 pages
    },
    "focus_enhanced": {
        # M4 + USB or M2 + Thunderbolt: moderate tuning
        "journal_mode": "WAL",
        "synchronous": "NORMAL",
        "cache_size": -32768,          # 32MB cache
        "mmap_size": 134217728,        # 128MB mmap
        "temp_store": "MEMORY",
        "busy_timeout": 45000,         # 45s — tolerates indexer contention
    },
    "focus": {
        # M2 + USB/Internal: conservative tuning
        "journal_mode": "WAL",
        "synchronous": "NORMAL",
        "cache_size": -16384,          # 16MB cache (8x default)
        "mmap_size": 67108864,         # 64MB mmap
        "temp_store": "MEMORY",
        "busy_timeout": 30000,         # 30s — tolerates indexer contention
    },
}

# §SEC: Whitelist of allowed PRAGMA names — defense in depth
_ALLOWED_PRAGMAS = frozenset({
    "journal_mode", "synchronous", "cache_size", "mmap_size",
    "temp_store", "busy_timeout", "wal_autocheckpoint", "page_size",
})


def get_tuning_profile(mode: str = "") -> dict:
    """Get the SQLite tuning profile for the given compute mode."""
    if not mode:
        try:
            from server.backend_logic import get_compute_profile
            mode = get_compute_profile().get("mode", "focus")
        except Exception:
            mode = "focus"
    return _PROFILES.get(mode, _PROFILES["focus"])


def tune_chroma_db(chroma_dir: str, mode: str = "") -> dict:
    """Apply IOPS-saturating pragmas to ChromaDB's SQLite database.

    Args:
        chroma_dir: Path to ChromaDB directory
        mode: Compute profile mode (auto-detected if empty)

    Returns:
        dict with before/after settings and applied pragmas
    """
    profile = get_tuning_profile(mode)
    results = {"applied": [], "before": {}, "after": {}, "mode": mode}

    # Find the SQLite file(s) in the chroma directory
    sqlite_files = []
    for root, dirs, files in os.walk(chroma_dir):
        for f in files:
            if f.endswith(".sqlite3"):
                sqlite_files.append(os.path.join(root, f))

    if not sqlite_files:
        log.warning(f"§IOPS: No SQLite files found in {chroma_dir}")
        return results

    for db_path in sqlite_files:
        try:
            conn = sqlite3.connect(db_path)
            db_name = os.path.basename(db_path)

            # Read current settings
            before = {}
            for pragma in profile:
                if pragma not in _ALLOWED_PRAGMAS:
                    continue
                try:
                    cur = conn.execute(f"PRAGMA {pragma}")
                    val = cur.fetchone()
                    before[pragma] = val[0] if val else None
                except Exception:
                    before[pragma] = None
            results["before"][db_name] = before

            # Apply tuning pragmas
            # §SEC: 'profile' dict is hardcoded (not user input) — no injection risk
            applied = []
            for pragma, value in profile.items():
                if pragma not in _ALLOWED_PRAGMAS:
                    continue
                try:
                    conn.execute(f"PRAGMA {pragma} = {value}")
                    applied.append(f"{pragma}={value}")
                except Exception as e:
                    log.debug(f"§IOPS: PRAGMA {pragma} failed on {db_name}: {e}")
            conn.commit()

            # Verify settings took effect
            after = {}
            for pragma in profile:
                if pragma not in _ALLOWED_PRAGMAS:
                    continue
                try:
                    cur = conn.execute(f"PRAGMA {pragma}")
                    val = cur.fetchone()
                    after[pragma] = val[0] if val else None
                except Exception:
                    after[pragma] = None
            results["after"][db_name] = after

            conn.close()
            results["applied"].extend(applied)

            log.info(f"§IOPS: Tuned {db_name}: {', '.join(applied)}")
        except Exception as e:
            log.error(f"§IOPS: Failed to tune {db_path}: {e}")

    return results


def log_tuning_report(results: dict):
    """Log a human-readable tuning report."""
    if not results.get("applied"):
        return

    for db_name in results.get("before", {}):
        before = results["before"][db_name]
        after = results["after"].get(db_name, {})
        changes = []
        for key in before:
            if before[key] != after.get(key):
                changes.append(f"  {key}: {before[key]} → {after[key]}")
        if changes:
            log.info(f"§IOPS: {db_name} tuning applied:\n" + "\n".join(changes))
