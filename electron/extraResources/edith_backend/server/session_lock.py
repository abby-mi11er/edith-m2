"""
Session Lock — Prevent M4/M2 Brain-Split
==========================================
When E.D.I.T.H. ignites on a machine, it writes a lock.pid to the
Bolt drive. If another machine tries to open E.D.I.T.H. while the
lock is active, users get a warning: "Soul is active on another node."

Lock file: {EDITH_DATA_ROOT}/.edith_session.lock

Usage:
    from server.session_lock import SessionLock

    lock = SessionLock()

    # On boot:
    result = lock.acquire()
    if result["locked_by_other"]:
        warn_user(result["owner"])

    # On shutdown:
    lock.release()
"""
from __future__ import annotations

import json
import logging
import os
import platform
import threading
import time
from pathlib import Path
from server.utils import atomic_write_json

log = logging.getLogger("edith.session_lock")


class SessionLock:
    """Bolt-level session lock to prevent concurrent machine access."""

    LOCK_FILE = ".edith_session.lock"
    STALE_THRESHOLD_SECONDS = 3600  # 1 hour without heartbeat = stale

    def __init__(self, data_root: str | None = None):
        root = data_root or os.environ.get("EDITH_DATA_ROOT", "")
        self._root = Path(root) if root else None
        self._lock_path: Path | None = None
        self._heartbeat_interval = 60  # seconds
        self._heartbeat_thread: threading.Thread | None = None
        self._stop_heartbeat = threading.Event()

        if self._root and self._root.exists():
            self._lock_path = self._root / self.LOCK_FILE

    @property
    def available(self) -> bool:
        return self._lock_path is not None

    def _machine_id(self) -> dict:
        """Build a machine identity fingerprint."""
        return {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "system": platform.system(),
            "pid": os.getpid(),
            "user": os.environ.get("USER", os.environ.get("USERNAME", "unknown")),
        }

    def _is_stale(self, lock_data: dict) -> bool:
        """Check if an existing lock is stale (owner crashed/closed without releasing)."""
        heartbeat = lock_data.get("last_heartbeat", lock_data.get("acquired_at", 0))
        age = time.time() - heartbeat
        return age > self.STALE_THRESHOLD_SECONDS

    def acquire(self, force: bool = False) -> dict:
        """Try to acquire the session lock.

        Returns:
            dict with keys:
                acquired: bool — whether we got the lock
                locked_by_other: bool — whether another machine holds it
                owner: dict — info about the current lock holder
                message: str — human-readable status
        """
        if not self.available:
            return {"acquired": True, "locked_by_other": False, "owner": None,
                    "message": "No data root configured — running without lock"}

        # Check for existing lock
        if self._lock_path.exists() and not force:
            try:
                existing = json.loads(self._lock_path.read_text())
            except Exception:
                existing = {}

            owner_host = existing.get("hostname", "unknown")
            my_host = platform.node()

            # Same machine? Re-acquire
            if owner_host == my_host:
                log.info(f"[Lock] Re-acquiring lock (same machine: {my_host})")
            elif self._is_stale(existing):
                log.warning(f"[Lock] Stale lock from {owner_host} — overriding")
            else:
                # Active lock from another machine
                return {
                    "acquired": False,
                    "locked_by_other": True,
                    "owner": existing,
                    "message": f"Soul is currently active on '{owner_host}'. "
                               f"Force takeover with force=True.",
                }

        # Acquire the lock
        lock_data = {
            **self._machine_id(),
            "acquired_at": time.time(),
            "last_heartbeat": time.time(),
        }
        try:
            atomic_write_json(self._lock_path, lock_data)
            log.info(f"[Lock] Session lock acquired on {lock_data['hostname']}")
            self._start_heartbeat_thread()
        except Exception as e:
            log.error(f"[Lock] Failed to write lock: {e}")
            return {
                "acquired": False,
                "locked_by_other": False,
                "owner": None,
                "message": f"Failed to write lock file: {e}",
            }

        return {
            "acquired": True,
            "locked_by_other": False,
            "owner": lock_data,
            "message": f"Session lock acquired on {lock_data['hostname']}",
        }

    # §REFACTOR: _atomic_write consolidated into server.utils.atomic_write_json

    def heartbeat(self):
        """Update the lock timestamp to prove we're still alive."""
        if not self.available or not self._lock_path or not self._lock_path.exists():
            return
        try:
            data = json.loads(self._lock_path.read_text())
            if data.get("hostname") == platform.node():
                data["last_heartbeat"] = time.time()
                atomic_write_json(self._lock_path, data)
        except Exception:
            pass

    def _start_heartbeat_thread(self):
        """Start background thread that heartbeats every 60s."""
        self._stop_heartbeat.clear()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            return  # already running

        def _loop():
            while not self._stop_heartbeat.wait(self._heartbeat_interval):
                self.heartbeat()

        self._heartbeat_thread = threading.Thread(
            target=_loop, daemon=True, name="edith-session-heartbeat"
        )
        self._heartbeat_thread.start()
        log.info(f"[Lock] Heartbeat thread started (interval={self._heartbeat_interval}s)")

    def release(self):
        """Release the session lock (called on clean shutdown)."""
        # Stop heartbeat thread first
        self._stop_heartbeat.set()
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=2)

        if not self.available or not self._lock_path:
            return
        if self._lock_path.exists():
            try:
                data = json.loads(self._lock_path.read_text())
                if data.get("hostname") == platform.node():
                    self._lock_path.unlink()
                    log.info("[Lock] Session lock released")
                else:
                    log.warning("[Lock] Lock belongs to another machine — not releasing")
            except Exception as e:
                log.error(f"[Lock] Failed to release lock: {e}")

    def status(self) -> dict:
        """Get current lock status for the NeuralHUD."""
        if not self.available:
            return {"available": False}
        if not self._lock_path.exists():
            return {"available": True, "locked": False}
        try:
            data = json.loads(self._lock_path.read_text())
            return {
                "available": True,
                "locked": True,
                "stale": self._is_stale(data),
                "owner": data.get("hostname"),
                "is_us": data.get("hostname") == platform.node(),
                "age_seconds": round(time.time() - data.get("acquired_at", 0)),
            }
        except Exception:
            return {"available": True, "locked": False}
