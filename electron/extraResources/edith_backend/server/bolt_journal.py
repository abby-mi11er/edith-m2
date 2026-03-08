"""
Bolt Journal Safety — Pre-Weld Snapshots & Corruption Recovery
===============================================================
Ensures data integrity when the Oyen Bolt is ejected mid-write.

Before every state weld (write to vault.db or session state), a
pre-weld snapshot is created. If the drive is pulled mid-write,
the next ignition detects corruption and rolls back automatically.

Usage:
    from server.bolt_journal import BoltJournal
    journal = BoltJournal("/Volumes/OyenBolt/EDITH_Data")

    # Before writing:
    journal.begin_transaction("save_session")
    write_to_vault(...)
    journal.commit_transaction()

    # On boot:
    journal.recover_if_needed()
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import time
from pathlib import Path

log = logging.getLogger("edith.bolt_journal")


class BoltJournal:
    """Write-ahead journal for Bolt drive operations."""

    JOURNAL_DIR = ".edith_journal"
    SENTINEL_FILE = "pending_weld.json"

    def __init__(self, data_root: str | None = None):
        root = data_root or os.environ.get("EDITH_DATA_ROOT", "")
        self._root = Path(root) if root else None
        self._journal_dir: Path | None = None
        self._active_tx: dict | None = None

        if self._root and self._root.exists():
            self._journal_dir = self._root / self.JOURNAL_DIR
            self._journal_dir.mkdir(exist_ok=True)

    @property
    def available(self) -> bool:
        return self._journal_dir is not None and self._journal_dir.exists()

    def begin_transaction(self, operation: str, targets: list[str] | None = None):
        """Create a pre-weld snapshot before a critical write.

        Args:
            operation: human label like "save_session" or "sync_vault"
            targets: list of file paths being written to
        """
        if not self.available:
            return

        tx_id = f"{int(time.time() * 1000)}_{operation}"
        sentinel = self._journal_dir / self.SENTINEL_FILE
        snapshot_dir = self._journal_dir / tx_id

        # Create snapshot of target files
        snapshot_dir.mkdir(exist_ok=True)
        backed_up = []
        for idx, target in enumerate(targets or []):
            src = Path(target)
            if src.exists() and src.is_file():
                # Use idx prefix to avoid collision when multiple targets share basename
                dst = snapshot_dir / f"{idx:03d}_{src.name}"
                try:
                    shutil.copy2(src, dst)
                    backed_up.append({"src": str(src), "snapshot": str(dst)})
                except Exception as e:
                    log.warning(f"Journal backup failed for {src}: {e}")

        # Write sentinel (marks transaction as in-progress)
        # §ATOMIC: Write to .tmp then rename — prevents corruption on drive pull
        self._active_tx = {
            "tx_id": tx_id,
            "operation": operation,
            "started_at": time.time(),
            "targets": targets or [],
            "backed_up": backed_up,
        }
        tmp_sentinel = sentinel.parent / (sentinel.name + ".tmp")
        tmp_sentinel.write_text(json.dumps(self._active_tx, indent=2))
        os.replace(str(tmp_sentinel), str(sentinel))
        log.info(f"[Journal] Transaction started: {operation} ({len(backed_up)} files backed up)")

    def commit_transaction(self):
        """Mark the current transaction as successfully completed."""
        if not self.available or not self._active_tx:
            return

        sentinel = self._journal_dir / self.SENTINEL_FILE
        tx_id = self._active_tx.get("tx_id", "")

        # Remove sentinel (no longer in-progress)
        if sentinel.exists():
            sentinel.unlink()

        # Clean up snapshot (write succeeded, no need to keep)
        snapshot_dir = self._journal_dir / tx_id
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir, ignore_errors=True)

        log.info(f"[Journal] Transaction committed: {self._active_tx.get('operation')}")
        self._active_tx = None

    def rollback_transaction(self):
        """Roll back to the pre-weld snapshot."""
        if not self.available or not self._active_tx:
            return {"rolled_back": False, "reason": "no active transaction"}

        backed_up = self._active_tx.get("backed_up", [])
        restored = []
        for entry in backed_up:
            snapshot = Path(entry["snapshot"])
            original = Path(entry["src"])
            if snapshot.exists():
                try:
                    shutil.copy2(snapshot, original)
                    restored.append(str(original))
                except Exception as e:
                    log.error(f"[Journal] Rollback failed for {original}: {e}")

        # Clean up
        sentinel = self._journal_dir / self.SENTINEL_FILE
        if sentinel.exists():
            sentinel.unlink()
        tx_id = self._active_tx.get("tx_id", "")
        snapshot_dir = self._journal_dir / tx_id
        if snapshot_dir.exists():
            shutil.rmtree(snapshot_dir, ignore_errors=True)

        log.warning(f"[Journal] Rolled back: {self._active_tx.get('operation')} ({len(restored)} files restored)")
        self._active_tx = None
        return {"rolled_back": True, "restored": restored}

    def recover_if_needed(self) -> dict:
        """Check for incomplete transactions on boot and auto-recover.

        Called during Citadel Ignition. If a sentinel file exists,
        it means the Bolt was pulled mid-write. Roll back automatically.
        """
        if not self.available:
            return {"status": "unavailable"}

        sentinel = self._journal_dir / self.SENTINEL_FILE
        if not sentinel.exists():
            return {"status": "clean", "message": "No pending transactions"}

        # Stale sentinel found — drive was pulled mid-write
        try:
            pending = json.loads(sentinel.read_text())
        except Exception:
            sentinel.unlink()
            return {"status": "cleaned", "message": "Corrupt sentinel removed"}

        log.warning(f"[Journal] RECOVERY: Found incomplete transaction: {pending.get('operation')}")
        self._active_tx = pending

        result = self.rollback_transaction()
        return {
            "status": "recovered",
            "operation": pending.get("operation"),
            "rolled_back": result.get("restored", []),
            "message": f"Auto-recovered from interrupted '{pending.get('operation')}' — "
                       f"{len(result.get('restored', []))} files restored to pre-weld state",
        }

    def status(self) -> dict:
        """Get journal status for the NeuralHUD."""
        return {
            "available": self.available,
            "active_transaction": self._active_tx is not None,
            "journal_dir": str(self._journal_dir) if self._journal_dir else None,
        }
