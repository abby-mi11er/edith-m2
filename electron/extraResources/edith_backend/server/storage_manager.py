"""
Storage Management — P1/P3
=============================
  - Incremental backup (rsync to secondary location)
  - Automatic index repair on corruption detection
"""

import logging
import os
import subprocess
import shutil
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.storage")


# ═══════════════════════════════════════════════════════════════════
# P1: Incremental Backup
# ═══════════════════════════════════════════════════════════════════

class BackupManager:
    """Incremental backup of ChromaDB and critical data using rsync."""

    def __init__(self, data_root: str = "", backup_dir: str = ""):
        self._data_root = data_root or os.environ.get("EDITH_DATA_ROOT", "")
        self._backup_dir = backup_dir or os.path.expanduser(
            "~/Library/Application Support/Edith/backups"
        )
        self._last_backup: Optional[float] = None
        self._backup_count = 0

    def run_backup(self, dry_run: bool = False) -> dict:
        """Run incremental backup using rsync.

        Args:
            dry_run: if True, only show what would be copied

        Returns:
            dict with status, files_copied, elapsed
        """
        if not self._data_root or not os.path.isdir(self._data_root):
            return {"status": "error", "error": "Data root not available"}

        # Ensure backup dir exists
        os.makedirs(self._backup_dir, exist_ok=True)

        # Timestamp-based backup folder
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        target = os.path.join(self._backup_dir, f"backup_{timestamp}")

        # Use rsync for incremental copy
        # --link-dest reuses unchanged files from the latest backup (dedup)
        latest_link = os.path.join(self._backup_dir, "latest")

        cmd = [
            "rsync", "-a", "--delete",
            "--exclude", "*.log",
            "--exclude", "__pycache__",
            "--exclude", ".venv",
            "--exclude", "node_modules",
        ]

        if os.path.islink(latest_link):
            cmd.extend(["--link-dest", os.readlink(latest_link)])

        if dry_run:
            cmd.append("--dry-run")
            cmd.append("-v")

        cmd.extend([
            self._data_root + "/",
            target + "/",
        ])

        try:
            t0 = time.time()
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            elapsed = time.time() - t0

            if result.returncode == 0:
                if not dry_run:
                    # Update 'latest' symlink
                    if os.path.islink(latest_link):
                        os.unlink(latest_link)
                    os.symlink(target, latest_link)
                    self._last_backup = time.time()
                    self._backup_count += 1

                lines = result.stdout.strip().split("\n") if result.stdout else []
                log.info(f"§BACKUP: {'Dry run' if dry_run else 'Completed'} "
                         f"in {elapsed:.1f}s ({len(lines)} items)")
                return {
                    "status": "ok",
                    "target": target,
                    "elapsed": f"{elapsed:.1f}s",
                    "items": len(lines),
                    "dry_run": dry_run,
                }
            else:
                log.error(f"§BACKUP: rsync failed: {result.stderr}")
                return {"status": "error", "error": result.stderr[:200]}
        except subprocess.TimeoutExpired:
            return {"status": "error", "error": "Backup timed out (10min)"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def list_backups(self) -> list[dict]:
        """List available backups."""
        if not os.path.isdir(self._backup_dir):
            return []

        backups = []
        for entry in sorted(os.listdir(self._backup_dir), reverse=True):
            if entry.startswith("backup_"):
                path = os.path.join(self._backup_dir, entry)
                if os.path.isdir(path):
                    try:
                        size = sum(
                            f.stat().st_size
                            for f in Path(path).rglob("*")
                            if f.is_file()
                        )
                        backups.append({
                            "name": entry,
                            "path": path,
                            "size_mb": round(size / (1024 * 1024), 1),
                            "timestamp": entry.replace("backup_", ""),
                        })
                    except Exception:
                        backups.append({"name": entry, "path": path})

        return backups[:10]  # Last 10

    @property
    def status(self) -> dict:
        return {
            "last_backup": time.strftime(
                "%Y-%m-%d %H:%M", time.localtime(self._last_backup)
            ) if self._last_backup else "never",
            "backup_count": self._backup_count,
            "backup_dir": self._backup_dir,
        }


# Singleton
backup_manager = BackupManager()


# ═══════════════════════════════════════════════════════════════════
# P3: Automatic Index Repair
# ═══════════════════════════════════════════════════════════════════

def check_and_repair_index(chroma_dir: str) -> dict:
    """Check ChromaDB index integrity and attempt repair if corrupted.

    Repair strategy:
    1. Try to open and count each collection
    2. If a collection fails, try to delete and recreate it
    3. If the entire DB fails, offer to rebuild from source PDFs

    Returns:
        dict with status, collections checked, and any repairs performed
    """
    if not chroma_dir or not os.path.isdir(chroma_dir):
        return {"status": "skip", "reason": "No chroma directory"}

    try:
        import chromadb
    except ImportError:
        return {"status": "skip", "reason": "chromadb not installed"}

    results = {
        "status": "ok",
        "collections": {},
        "repairs": [],
    }

    try:
        client = chromadb.PersistentClient(path=chroma_dir)
        collections = client.list_collections()

        for coll in collections:
            try:
                c = client.get_collection(coll.name)
                count = c.count()
                results["collections"][coll.name] = {
                    "status": "ok",
                    "count": count,
                }
            except Exception as e:
                error_msg = str(e)
                results["collections"][coll.name] = {
                    "status": "error",
                    "error": error_msg[:200],
                }
                results["status"] = "degraded"

                # Attempt repair: delete corrupted collection
                try:
                    log.warning(f"§REPAIR: Attempting to repair {coll.name}")
                    client.delete_collection(coll.name)
                    # Recreate empty collection
                    client.create_collection(coll.name)
                    results["repairs"].append({
                        "collection": coll.name,
                        "action": "recreated_empty",
                        "note": "Needs re-indexing from source documents",
                    })
                    log.info(f"§REPAIR: {coll.name} recreated (needs re-index)")
                except Exception as repair_err:
                    results["repairs"].append({
                        "collection": coll.name,
                        "action": "repair_failed",
                        "error": str(repair_err)[:200],
                    })
                    log.error(f"§REPAIR: Failed to repair {coll.name}: {repair_err}")

    except Exception as e:
        results["status"] = "critical"
        results["error"] = str(e)[:200]
        log.error(f"§REPAIR: ChromaDB client failed: {e}")

    return results


def get_reindex_command(collection_name: str, data_root: str = "") -> str:
    """Generate the CLI command to re-index a collection."""
    data_root = data_root or os.environ.get("EDITH_DATA_ROOT", "")
    project_root = os.environ.get("EDITH_PROJECT_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return (
        f"cd '{project_root}' && "
        f"python3 chroma_index.py "
        f"--collection {collection_name} "
        f"--data-root '{data_root}'"
    )
