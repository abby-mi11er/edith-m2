"""
Vault File Watcher — Auto-index new documents when they land in the Vault.

Uses Python's watchdog library to monitor the Vault folder for new files.
When a new PDF, DOCX, TXT, or other indexable file appears, it queues
a background re-index of that specific file via chroma_index.py --single-file.

Usage:
    from server.vault_watcher import VaultWatcher
    watcher = VaultWatcher(vault_root="/path/to/vault")
    watcher.start()  # non-blocking, runs in background thread
    watcher.stop()
"""

import os
import sys
import time
import logging
import subprocess
import threading
from pathlib import Path

log = logging.getLogger("vault_watcher")

# File extensions that should trigger indexing
INDEXABLE_EXTENSIONS = {
    ".pdf", ".docx", ".doc", ".txt", ".md", ".rtf",
    ".xlsx", ".xls", ".csv", ".dta", ".dbf",
    ".tex", ".bib", ".json", ".jsonl",
}

# Minimum file size to avoid indexing empty/temp files
MIN_FILE_SIZE = 100  # bytes

# Debounce: wait this many seconds after last event before indexing
DEBOUNCE_SECONDS = 5


class VaultWatcher:
    """Watch the Vault folder for new files and auto-index them."""

    def __init__(self, vault_root: str, chroma_script: str | None = None):
        self.vault_root = Path(vault_root)
        self.chroma_script = chroma_script or str(
            Path(__file__).parent.parent / "scripts" / "chroma_index.py"
        )
        self._pending: set[str] = set()
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._observer = None
        self._running = False

    def start(self):
        """Start watching the Vault folder (non-blocking)."""
        try:
            from watchdog.observers import Observer
            from watchdog.events import FileSystemEventHandler, FileCreatedEvent, FileModifiedEvent

            watcher_self = self

            class _Handler(FileSystemEventHandler):
                def on_created(self, event):
                    if not event.is_directory:
                        watcher_self._on_file_event(event.src_path)

                def on_modified(self, event):
                    if not event.is_directory:
                        watcher_self._on_file_event(event.src_path)

            self._observer = Observer()
            self._observer.schedule(_Handler(), str(self.vault_root), recursive=True)
            self._observer.daemon = True
            self._observer.start()
            self._running = True
            log.info(f"§WATCH: Vault watcher started on {self.vault_root}")
        except ImportError:
            log.warning("§WATCH: watchdog not installed — vault watcher disabled (pip install watchdog)")
        except Exception as e:
            log.warning(f"§WATCH: Could not start vault watcher: {e}")

    def stop(self):
        """Stop watching."""
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._running = False
            log.info("§WATCH: Vault watcher stopped")

    def _on_file_event(self, filepath: str):
        """Called when a file is created or modified in the Vault."""
        path = Path(filepath)

        # Skip non-indexable files
        if path.suffix.lower() not in INDEXABLE_EXTENSIONS:
            return

        # Skip temp files, hidden files, and small files
        if path.name.startswith(".") or path.name.startswith("~"):
            return

        try:
            if path.stat().st_size < MIN_FILE_SIZE:
                return
        except OSError:
            return

        # Add to pending queue and debounce
        with self._lock:
            self._pending.add(str(path))
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(DEBOUNCE_SECONDS, self._process_pending)
            self._timer.daemon = True
            self._timer.start()

    def _process_pending(self):
        """Process all pending files after debounce period."""
        with self._lock:
            files = list(self._pending)
            self._pending.clear()

        if not files:
            return

        log.info(f"§WATCH: Auto-indexing {len(files)} new file(s)...")

        for filepath in files:
            try:
                result = subprocess.run(
                    [sys.executable, self.chroma_script, "--single-file", filepath],
                    capture_output=True,
                    text=True,
                    timeout=300,
                    cwd=str(Path(self.chroma_script).parent.parent),
                )
                if result.returncode == 0:
                    log.info(f"§WATCH: Indexed {Path(filepath).name}")
                else:
                    log.warning(f"§WATCH: Failed to index {Path(filepath).name}: {result.stderr[:200]}")
            except subprocess.TimeoutExpired:
                log.warning(f"§WATCH: Timeout indexing {Path(filepath).name}")
            except Exception as e:
                log.warning(f"§WATCH: Error indexing {Path(filepath).name}: {e}")

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def status(self) -> dict:
        return {
            "running": self._running,
            "vault_root": str(self.vault_root),
            "pending_files": len(self._pending),
        }
