"""
Operational Rhythm — Scheduled Maintenance, Auto-Backup, Health Pulse
=======================================================================
Batch 4 — CE-48/49/50: Keep the Citadel running without babysitting.

The system that watches the system. Runs maintenance tasks
on schedule, backs up critical data, and sends health pulses.
"""

import json
import logging
import os
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path

log = logging.getLogger("edith.rhythm")


@dataclass
class ScheduledTask:
    """A recurring maintenance task."""
    name: str
    interval_seconds: int
    last_run: float = 0.0
    run_count: int = 0
    last_status: str = "never_run"
    enabled: bool = True

    @property
    def next_run(self) -> float:
        return self.last_run + self.interval_seconds

    @property
    def overdue(self) -> bool:
        return time.time() > self.next_run and self.last_run > 0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "interval_s": self.interval_seconds,
            "run_count": self.run_count,
            "last_status": self.last_status,
            "overdue": self.overdue,
            "enabled": self.enabled,
        }


class OperationalRhythm:
    """Scheduled maintenance and auto-backup manager.

    CE-48: Scheduled maintenance — ChromaDB optimization, cache cleanup
    CE-49: Auto-backup — Bolt SSD snapshots on schedule
    CE-50: Health pulse — periodic system health check
    """

    def __init__(self, data_root: str = ""):
        self._data_root = data_root or os.environ.get("EDITH_APP_DATA_DIR", "")
        self._tasks: dict[str, ScheduledTask] = {
            "cache_cleanup": ScheduledTask("cache_cleanup", 3600),  # hourly
            "chroma_optimize": ScheduledTask("chroma_optimize", 86400),  # daily
            "backup_config": ScheduledTask("backup_config", 43200),  # 12 hours
            "health_pulse": ScheduledTask("health_pulse", 300),  # 5 minutes
            "log_rotation": ScheduledTask("log_rotation", 86400),  # daily
        }
        self._health_history: list[dict] = []
        self._running = False
        self._thread: threading.Thread | None = None

    def check_due_tasks(self) -> list[str]:
        """Check which tasks are due to run."""
        now = time.time()
        due = []
        for name, task in self._tasks.items():
            if task.enabled and now > task.next_run:
                due.append(name)
        return due

    def run_task(self, task_name: str) -> dict:
        """Run a specific maintenance task."""
        task = self._tasks.get(task_name)
        if not task:
            return {"error": f"Unknown task: {task_name}"}

        try:
            if task_name == "cache_cleanup":
                result = self._cleanup_caches()
            elif task_name == "chroma_optimize":
                result = self._optimize_chroma()
            elif task_name == "backup_config":
                result = self._backup_config()
            elif task_name == "health_pulse":
                result = self._health_pulse()
            elif task_name == "log_rotation":
                result = self._rotate_logs()
            else:
                result = {"status": "unknown_task"}

            task.last_run = time.time()
            task.run_count += 1
            task.last_status = "success"
            return result
        except Exception as e:
            task.last_status = f"failed: {str(e)[:100]}"
            return {"error": str(e)}

    def _cleanup_caches(self) -> dict:
        """Clean stale cache entries."""
        cleaned = 0
        cache_dir = Path(self._data_root) / "CACHE"
        if cache_dir.exists():
            cutoff = time.time() - 86400  # 24 hours
            for p in cache_dir.glob("*.json"):
                if p.stat().st_mtime < cutoff:
                    p.unlink(missing_ok=True)
                    cleaned += 1
        return {"cleaned_files": cleaned}

    def _optimize_chroma(self) -> dict:
        """Trigger ChromaDB optimization (compaction)."""
        # ChromaDB handles optimization internally, but we log it
        return {"status": "optimization_requested", "note": "ChromaDB handles compaction automatically"}

    def _backup_config(self) -> dict:
        """Backup configuration files to a timestamped directory."""
        backup_dir = Path(self._data_root) / "BACKUPS" / time.strftime("%Y%m%d_%H%M")
        backup_dir.mkdir(parents=True, exist_ok=True)

        backed_up = []
        config_files = [".env", "config.json", "personas/"]
        for cf in config_files:
            src = Path(self._data_root) / cf
            if src.exists():
                if src.is_file():
                    (backup_dir / src.name).write_bytes(src.read_bytes())
                    backed_up.append(cf)

        return {"backup_dir": str(backup_dir), "files": backed_up}

    def _health_pulse(self) -> dict:
        """Quick system health check."""
        import shutil

        pulse = {
            "timestamp": time.time(),
            "disk_free_gb": round(shutil.disk_usage("/").free / (1024 ** 3), 1),
            "data_dir_exists": Path(self._data_root).exists() if self._data_root else False,
        }

        # Check Bolt SSD if configured
        try:
            from server.vault_config import VAULT_ROOT
            bolt_path = str(VAULT_ROOT)
        except ImportError:
            bolt_path = os.environ.get("EDITH_DATA_ROOT", "")
        if bolt_path and Path(bolt_path).exists():
            bolt_usage = shutil.disk_usage(bolt_path)
            pulse["bolt_free_gb"] = round(bolt_usage.free / (1024 ** 3), 1)
            pulse["bolt_used_pct"] = round(bolt_usage.used / bolt_usage.total * 100, 1)

        self._health_history.append(pulse)
        self._health_history = self._health_history[-288:]  # Keep 24 hours at 5-min intervals

        return pulse

    def _rotate_logs(self) -> dict:
        """Rotate old log files."""
        rotated = 0
        log_dir = Path(self._data_root) / "LOGS"
        if log_dir.exists():
            cutoff = time.time() - 604800  # 7 days
            for p in log_dir.glob("*.log"):
                if p.stat().st_mtime < cutoff:
                    p.unlink(missing_ok=True)
                    rotated += 1
        return {"rotated": rotated}

    @property
    def status(self) -> dict:
        return {
            "tasks": {name: t.to_dict() for name, t in self._tasks.items()},
            "due_now": self.check_due_tasks(),
            "health_history_length": len(self._health_history),
            "latest_pulse": self._health_history[-1] if self._health_history else None,
        }


# Global instance
operational_rhythm = OperationalRhythm()
