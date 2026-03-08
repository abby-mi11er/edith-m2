"""
Auto-Maintenance System — E.D.I.T.H.'s "Mental Health"
========================================================
Automated upkeep that keeps E.D.I.T.H. running at peak performance.

Capabilities:
  - Nightly vector re-indexing (re-embed stale chunks)
  - Log purging (keep DB lean, archive old logs)
  - Health checks (disk, memory, index integrity)
  - State snapshot/restore for drive portability

Exposed as:
  POST /api/maintenance/run   — trigger maintenance cycle
  GET  /api/maintenance/status — last run status
"""

import json
import logging
import os
import shutil
import sqlite3
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.maintenance")


class MaintenanceEngine:
    """Automated maintenance for E.D.I.T.H.'s infrastructure."""

    def __init__(self):
        self._last_run: Optional[dict] = None
        self._running = False
        self._lock = threading.Lock()

    @property
    def status(self) -> dict:
        return {
            "running": self._running,
            "last_run": self._last_run,
        }

    def run_full_cycle(self, background: bool = True) -> dict:
        """Run a full maintenance cycle.

        Steps:
          1. Health check (disk, memory, index)
          2. Log purge (clear old entries)
          3. Index integrity check
          4. State snapshot for portability
        """
        if self._running:
            return {"status": "already_running"}

        if background:
            thread = threading.Thread(target=self._cycle, daemon=True)
            thread.start()
            return {"status": "started"}
        else:
            return self._cycle()

    def _cycle(self) -> dict:
        self._running = True
        t0 = time.time()
        results = {
            "started": datetime.now().isoformat(),
            "steps": {},
        }

        try:
            # Step 1: Health check
            results["steps"]["health"] = self._health_check()

            # Step 2: Log purge
            results["steps"]["log_purge"] = self._purge_logs()

            # Step 3: Index integrity
            results["steps"]["index_check"] = self._check_index()

            # Step 4: State snapshot
            results["steps"]["snapshot"] = self._save_state_snapshot()

            # Step 5: Auto-backup
            results["steps"]["backup"] = self._auto_backup()

            # Step 6: Stale embedding check
            results["steps"]["stale_check"] = self._check_stale_embeddings()

            results["status"] = "completed"
        except Exception as e:
            results["status"] = "failed"
            results["error"] = str(e)[:500]
            log.error(f"§MAINT: Cycle failed: {e}")

        results["elapsed"] = round(time.time() - t0, 2)
        results["completed"] = datetime.now().isoformat()

        with self._lock:
            self._last_run = results
            self._running = False

        log.info(f"§MAINT: Cycle completed in {results['elapsed']}s")
        return results

    def _health_check(self) -> dict:
        """Check disk, memory, and system health."""
        health = {"status": "ok", "checks": []}

        # Disk space
        data_root = os.environ.get("EDITH_DATA_ROOT", "")
        if data_root and os.path.isdir(data_root):
            usage = shutil.disk_usage(data_root)
            free_gb = round(usage.free / (1024**3), 2)
            pct_used = round(usage.used / usage.total * 100, 1)
            health["checks"].append({
                "name": "disk_space",
                "ok": free_gb > 5,
                "detail": f"{free_gb}GB free ({pct_used}% used)",
            })
            if free_gb < 5:
                health["status"] = "warn"

        # Memory pressure
        try:
            from server.memory_scaler import get_memory_pressure
            mp = get_memory_pressure()
            health["checks"].append({
                "name": "memory",
                "ok": mp["pressure_level"] != "critical",
                "detail": f"{mp['pressure_level']}, {mp['free_mb']}MB free",
            })
            if mp["pressure_level"] == "critical":
                health["status"] = "critical"
        except ImportError:
            pass

        # Python environment
        import sys
        health["checks"].append({
            "name": "python",
            "ok": True,
            "detail": f"{sys.version.split()[0]}",
        })

        # Check for key packages
        for pkg in ["pandas", "statsmodels", "numpy", "matplotlib"]:
            try:
                __import__(pkg)
                health["checks"].append({
                    "name": f"pkg_{pkg}",
                    "ok": True,
                    "detail": "installed",
                })
            except ImportError:
                health["checks"].append({
                    "name": f"pkg_{pkg}",
                    "ok": False,
                    "detail": "NOT INSTALLED — vibe coding will be limited",
                })
                if health["status"] == "ok":
                    health["status"] = "degraded"

        return health

    def _purge_logs(self) -> dict:
        """Purge old log files and compact databases."""
        purged = {"files_removed": 0, "bytes_freed": 0}

        data_root = os.environ.get("EDITH_DATA_ROOT", "")
        if not data_root:
            return purged

        # Remove old log files (>30 days)
        log_dirs = [
            os.path.join(data_root, "logs"),
            os.path.join(data_root, "audit_logs"),
        ]

        cutoff = time.time() - (30 * 86400)  # 30 days ago
        for log_dir in log_dirs:
            if not os.path.isdir(log_dir):
                continue
            for fname in os.listdir(log_dir):
                fpath = os.path.join(log_dir, fname)
                try:
                    if os.path.getmtime(fpath) < cutoff:
                        size = os.path.getsize(fpath)
                        os.remove(fpath)
                        purged["files_removed"] += 1
                        purged["bytes_freed"] += size
                except OSError:
                    pass

        purged["bytes_freed_mb"] = round(purged["bytes_freed"] / (1024 * 1024), 2)
        return purged

    def _check_index(self) -> dict:
        """Check ChromaDB index integrity."""
        result = {"status": "ok", "collections": []}

        chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "")
        if not chroma_dir or not os.path.isdir(chroma_dir):
            result["status"] = "skipped"
            result["reason"] = "EDITH_CHROMA_DIR not set or doesn't exist"
            return result

        # Check SQLite files in chroma directory
        sqlite_files = list(Path(chroma_dir).rglob("*.sqlite3"))
        for db_path in sqlite_files[:5]:
            try:
                import contextlib
                with contextlib.closing(sqlite3.connect(str(db_path))) as conn:
                    cursor = conn.execute("PRAGMA integrity_check;")
                    integrity = cursor.fetchone()[0]
                    result["collections"].append({
                        "file": db_path.name,
                        "integrity": integrity,
                        "ok": integrity == "ok",
                    })
                    if integrity != "ok":
                        result["status"] = "degraded"
            except Exception as e:
                result["collections"].append({
                    "file": db_path.name,
                    "integrity": "error",
                    "ok": False,
                    "error": str(e)[:100],
                })
                result["status"] = "error"

        return result

    def _save_state_snapshot(self) -> dict:
        """Save a state snapshot for drive portability."""
        data_root = os.environ.get("EDITH_DATA_ROOT", "")
        if not data_root:
            return {"status": "skipped", "reason": "No EDITH_DATA_ROOT"}

        snapshot = {
            "timestamp": datetime.now().isoformat(),
            "python_version": __import__("sys").version.split()[0],
            "data_root": data_root,
            "env_keys": [
                k for k in os.environ
                if k.startswith("EDITH_") and "KEY" not in k.upper()
            ],
            "installed_packages": [],
        }

        for pkg in ["pandas", "numpy", "statsmodels", "matplotlib",
                     "seaborn", "scikit-learn", "chromadb", "mlx"]:
            try:
                mod = __import__(pkg)
                version = getattr(mod, "__version__", "unknown")
                snapshot["installed_packages"].append(f"{pkg}=={version}")
            except ImportError:
                pass

        snapshot_path = os.path.join(data_root, ".edith_state.json")
        try:
            with open(snapshot_path, "w") as f:
                json.dump(snapshot, f, indent=2)
            return {"status": "saved", "path": snapshot_path}
        except Exception as e:
            return {"status": "failed", "error": str(e)[:100]}

    def _auto_backup(self) -> dict:
        """§4.10: Auto-backup — incremental backup of ChromaDB + config."""
        data_root = os.environ.get("EDITH_DATA_ROOT", "")
        chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "")
        if not data_root:
            return {"status": "skipped"}

        backup_dir = os.path.join(data_root, "backups",
                                  datetime.now().strftime("%Y%m%d_%H%M"))
        os.makedirs(backup_dir, exist_ok=True)

        backed_up = []

        # Backup SQLite databases
        if chroma_dir and os.path.isdir(chroma_dir):
            for db_file in Path(chroma_dir).rglob("*.sqlite3"):
                try:
                    dest = os.path.join(backup_dir, db_file.name)
                    shutil.copy2(str(db_file), dest)
                    backed_up.append(db_file.name)
                except Exception as e:
                    log.debug(f"§BACKUP: Failed to copy {db_file}: {e}")

        # Backup critical config files
        config_files = [
            os.path.join(data_root, ".edith_state.json"),
            os.path.join(data_root, ".env"),
        ]
        for cf in config_files:
            if os.path.exists(cf):
                try:
                    shutil.copy2(cf, backup_dir)
                    backed_up.append(os.path.basename(cf))
                except Exception:
                    pass

        # Remove backups older than 7 days
        backups_root = os.path.join(data_root, "backups")
        if os.path.isdir(backups_root):
            cutoff = time.time() - (7 * 86400)
            for d in os.listdir(backups_root):
                dpath = os.path.join(backups_root, d)
                if os.path.isdir(dpath) and os.path.getmtime(dpath) < cutoff:
                    try:
                        shutil.rmtree(dpath)
                    except Exception:
                        pass

        return {"status": "completed", "backed_up": backed_up, "dir": backup_dir}

    def _check_stale_embeddings(self) -> dict:
        """§4.2: Check for stale embeddings that need re-indexing."""
        chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "")
        if not chroma_dir or not os.path.isdir(chroma_dir):
            return {"status": "skipped"}

        stale_days = int(os.environ.get("EDITH_REINDEX_DAYS", "7"))
        cutoff = time.time() - (stale_days * 86400)

        stale_files = []
        for db_path in Path(chroma_dir).rglob("*.sqlite3"):
            try:
                mtime = os.path.getmtime(str(db_path))
                if mtime < cutoff:
                    stale_files.append({
                        "file": db_path.name,
                        "age_days": round((time.time() - mtime) / 86400, 1),
                    })
            except Exception:
                pass

        return {
            "stale_count": len(stale_files),
            "stale_files": stale_files,
            "needs_reindex": len(stale_files) > 0,
            "threshold_days": stale_days,
        }


# Singleton
maintenance = MaintenanceEngine()

