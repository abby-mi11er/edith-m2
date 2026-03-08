"""
Doctor — E.D.I.T.H. Diagnostics & Health Checks
==================================================
§ORCH-6: Bolt I/O Speed Test — benchmark the Soul's connection
§ORCH-7: main.py handler decoupling assessment

The Doctor runs at boot and on-demand to verify:
1. Bolt read/write throughput (detects cable/port degradation)
2. SQLite integrity across all 3 databases
3. ChromaDB collection health
4. System resource utilization
"""

import json
import logging
import os
import time
import tempfile
from pathlib import Path

log = logging.getLogger("edith.doctor")


# ═══════════════════════════════════════════════════════════════════
# §ORCH-6: Bolt I/O Speed Test
# ═══════════════════════════════════════════════════════════════════

def bolt_io_benchmark(
    test_size_mb: int = 10,
    vault_root: str = "",
) -> dict:
    """§ORCH-6: Benchmark the Oyen Bolt's read/write throughput.

    Since the Bolt is the "Soul," USB-C cable degradation or a dusty
    port can silently tank vector search latency. This test writes
    and reads a temporary file to measure actual throughput.

    Expected baselines:
        Thunderbolt 4:  ~2,800 MB/s read, ~2,200 MB/s write
        USB-C 3.2:      ~900 MB/s read, ~800 MB/s write
        USB-C 3.0:      ~400 MB/s read, ~350 MB/s write
        Degraded cable: <100 MB/s ← triggers WARNING

    Returns dict with read/write MB/s and health assessment.
    """
    vault_root = vault_root or os.environ.get("VAULT_ROOT", "")
    if not vault_root or not Path(vault_root).exists():
        return {"status": "skipped", "reason": "No VAULT_ROOT found"}

    test_bytes = test_size_mb * 1024 * 1024
    test_data = os.urandom(test_bytes)
    test_path = Path(vault_root) / ".edith_io_test.tmp"

    results = {
        "vault_root": vault_root,
        "test_size_mb": test_size_mb,
    }

    # === Write Test ===
    try:
        t0 = time.perf_counter()
        test_path.write_bytes(test_data)
        write_elapsed = time.perf_counter() - t0
        write_mbps = test_size_mb / max(write_elapsed, 0.001)
        results["write_mbps"] = round(write_mbps, 1)
        results["write_ms"] = round(write_elapsed * 1000, 1)
    except Exception as e:
        results["write_error"] = str(e)
        write_mbps = 0

    # === Read Test ===
    try:
        t0 = time.perf_counter()
        _ = test_path.read_bytes()
        read_elapsed = time.perf_counter() - t0
        read_mbps = test_size_mb / max(read_elapsed, 0.001)
        results["read_mbps"] = round(read_mbps, 1)
        results["read_ms"] = round(read_elapsed * 1000, 1)
    except Exception as e:
        results["read_error"] = str(e)
        read_mbps = 0

    # === Cleanup ===
    try:
        test_path.unlink(missing_ok=True)
    except Exception:
        pass

    # === Health Assessment ===
    if read_mbps > 1000:
        health = "excellent"
        connection = "Thunderbolt"
    elif read_mbps > 400:
        health = "good"
        connection = "USB-C 3.2"
    elif read_mbps > 100:
        health = "fair"
        connection = "USB-C 3.0"
    else:
        health = "degraded"
        connection = "WARNING: Cable/port issue"

    results["health"] = health
    results["estimated_connection"] = connection

    if health == "degraded":
        log.warning(f"§ORCH-6: Bolt I/O DEGRADED — Read: {read_mbps:.0f} MB/s, "
                    f"Write: {write_mbps:.0f} MB/s. Check cable/port!")
    else:
        log.info(f"§ORCH-6: Bolt I/O {health} — Read: {read_mbps:.0f} MB/s, "
                 f"Write: {write_mbps:.0f} MB/s ({connection})")

    return results


# ═══════════════════════════════════════════════════════════════════
# SQLite Integrity Check
# ═══════════════════════════════════════════════════════════════════

def check_sqlite_integrity() -> dict:
    """Check integrity of all E.D.I.T.H. SQLite databases."""
    try:
        from server.vault_config import VAULT_DB, MEMORY_DB, FEEDBACK_DB
    except ImportError:
        return {"status": "skipped", "reason": "vault_config unavailable"}

    import sqlite3
    import contextlib
    results = {}

    for name, db_path in [("vault", VAULT_DB), ("memory", MEMORY_DB), ("feedback", FEEDBACK_DB)]:
        if not db_path.exists():
            results[name] = {"status": "missing"}
            continue

        try:
            with contextlib.closing(sqlite3.connect(str(db_path))) as conn:
                cursor = conn.cursor()

                # Integrity check
                cursor.execute("PRAGMA integrity_check;")
                integrity = cursor.fetchone()[0]

                # WAL mode check
                cursor.execute("PRAGMA journal_mode;")
                journal = cursor.fetchone()[0]

                # Page count
                cursor.execute("PRAGMA page_count;")
                pages = cursor.fetchone()[0]

                cursor.execute("PRAGMA page_size;")
                page_size = cursor.fetchone()[0]

                size_mb = round((pages * page_size) / (1024 * 1024), 1)

            results[name] = {
                "status": "ok" if integrity == "ok" else "corrupt",
                "integrity": integrity,
                "journal_mode": journal,
                "size_mb": size_mb,
            }
        except Exception as e:
            results[name] = {"status": "error", "error": str(e)}

    return results


# ═══════════════════════════════════════════════════════════════════
# ChromaDB Collection Health
# ═══════════════════════════════════════════════════════════════════

def check_chroma_health(chroma_dir: str = "") -> dict:
    """Check ChromaDB collection statistics and health."""
    if not chroma_dir:
        chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "")
    if not chroma_dir:
        try:
            from server.vault_config import VECTORS_DIR
            chroma_dir = str(VECTORS_DIR)
        except ImportError:
            return {"status": "skipped", "reason": "No chroma_dir"}

    if not Path(chroma_dir).exists():
        return {"status": "missing", "path": chroma_dir}

    try:
        from server.chroma_backend import _get_client
        client = _get_client(chroma_dir)
        collections = client.list_collections()

        col_info = []
        total_docs = 0
        for col in collections:
            count = col.count()
            total_docs += count
            col_info.append({
                "name": col.name,
                "documents": count,
            })

        # Storage size
        chroma_size_mb = 0
        for f in Path(chroma_dir).rglob("*"):
            if f.is_file():
                chroma_size_mb += f.stat().st_size
        chroma_size_mb = round(chroma_size_mb / (1024 * 1024), 1)

        return {
            "status": "healthy",
            "collections": col_info,
            "total_documents": total_docs,
            "storage_mb": chroma_size_mb,
        }
    except Exception as e:
        return {"status": "error", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# Full Diagnostic Report
# ═══════════════════════════════════════════════════════════════════

def run_full_diagnostic() -> dict:
    """Run all diagnostic checks and return a combined report.

    Call at boot or on-demand from the Doctor panel.
    """
    t0 = time.time()

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "bolt_io": bolt_io_benchmark(),
        "sqlite": check_sqlite_integrity(),
        "chroma": check_chroma_health(),
    }

    # System info
    try:
        import subprocess
        mem = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True, text=True, timeout=3,
        )
        ram_gb = round(int(mem.stdout.strip()) / (1024**3))
        report["system"] = {"ram_gb": ram_gb}
    except Exception:
        report["system"] = {}

    report["elapsed_ms"] = round((time.time() - t0) * 1000, 1)

    # Overall health
    bolt_health = report["bolt_io"].get("health", "unknown")
    sqlite_ok = all(
        v.get("status") in ("ok", "missing")
        for v in report["sqlite"].values()
        if isinstance(v, dict)
    )
    chroma_ok = report["chroma"].get("status") in ("healthy", "skipped", "missing")

    if bolt_health in ("excellent", "good") and sqlite_ok and chroma_ok:
        report["overall"] = "HEALTHY"
    elif bolt_health == "degraded":
        report["overall"] = "WARNING — Bolt I/O degraded"
    elif not sqlite_ok:
        report["overall"] = "WARNING — SQLite integrity issue"
    else:
        report["overall"] = "FAIR"

    log.info(f"§DOCTOR: Diagnostic complete in {report['elapsed_ms']}ms — "
             f"{report['overall']}")

    return report
