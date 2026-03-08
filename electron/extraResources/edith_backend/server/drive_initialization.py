"""
Drive Initialization — Sovereign Soul Setup
================================================
When a user plugs in the Bolt drive, this module creates the
standard E.D.I.T.H. directory structure, writes the anchor file,
and validates the drive is ready for operation.

Architecture:
    Mac (Engine)  →  fast SSD, app code, binaries, venv
    Bolt (Soul)   →  personal data, vectors, training, vault

The app WILL NOT BOOT without the anchor file on the Bolt.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import time
import uuid
from pathlib import Path

log = logging.getLogger("edith.drive_init")

# ── Bolt Drive Directory Structure ──────────────────────────────────
# Maps exactly to the Sovereign Soul spec
SOUL_DIRECTORIES = [
    ".vault",                   # keys.enc, session.bin (AES-256)
    "ChromaDB",                 # vector.db, index/ — Long-term Memory
    "Library",                  # Full Mendeley/Zotero mirror
    "Library/PDFs",             # Paper PDFs
    "Brain",                    # Winnie training logs
    "Stata_Work",               # Sensitive census/welfare data
    "Stata_Work/datasets",      # Raw datasets
    "Stata_Work/mle_logs",      # MLE analysis logs
    "Connectome",               # 3D library map
    "Connectome/Snapshots",     # StateWeld session snapshots
    "exports",                  # BibTeX, RIS, PDF exports
    "backups",                  # Automated backups
]

ANCHOR_FILE = ".edith_anchor"
BOLT_VOLUME_NAME = "Edith Bolt"
DEFAULT_BOLT_PATH = f"/Volumes/{BOLT_VOLUME_NAME}"


def detect_drives() -> list[dict]:
    """Detect available external drives for Soul initialization.

    Returns a list of drives with path, name, size, and anchor status.
    """
    drives = []
    volumes = Path("/Volumes")
    if not volumes.exists():
        return drives

    for vol in volumes.iterdir():
        if not vol.is_dir():
            continue
        name = vol.name
        # Skip system volumes
        if name in ("Macintosh HD", "Macintosh HD - Data", "Recovery", "Preboot", "VM"):
            continue
        try:
            stat = os.statvfs(str(vol))
            total_gb = round((stat.f_blocks * stat.f_frsize) / (1024 ** 3), 1)
            free_gb = round((stat.f_bavail * stat.f_frsize) / (1024 ** 3), 1)
            drives.append({
                "path": str(vol),
                "name": name,
                "total_gb": total_gb,
                "free_gb": free_gb,
                "has_anchor": (vol / ANCHOR_FILE).exists(),
                "is_bolt": name == BOLT_VOLUME_NAME or (vol / ".edith_drive_marker").exists(),
            })
        except OSError:
            continue

    return drives


def find_bolt() -> str | None:
    """Find the Bolt drive path if mounted."""
    bolt = Path(DEFAULT_BOLT_PATH)
    if bolt.exists() and bolt.is_dir():
        return str(bolt)
    # Fallback: scan for anchor on any volume
    for drive in detect_drives():
        if drive["has_anchor"]:
            return drive["path"]
    return None


def initialize_soul_drive(drive_path: str, owner: str = "") -> dict:
    """Initialize a new Bolt drive as an E.D.I.T.H. Soul.

    Creates the standard directory structure and writes the anchor file.

    Args:
        drive_path: Absolute path to the drive root (e.g., /Volumes/OyenBolt)
        owner: Optional owner name for the anchor

    Returns:
        dict with status, soul_id, and directories created
    """
    drive = Path(drive_path).resolve()
    if not drive.exists() or not drive.is_dir():
        return {"status": "error", "message": f"Drive not found: {drive_path}"}

    # §SEC: Path validation — only allow /Volumes/ or explicit EDITH_DATA_ROOT
    allowed_root = os.environ.get("EDITH_DATA_ROOT", "")
    if not (str(drive).startswith("/Volumes/") or
            (allowed_root and str(drive).startswith(allowed_root))):
        return {"status": "error",
                "message": f"Security: drive must be under /Volumes/. Got: {drive}"}

    # Check if already initialized
    anchor = drive / ANCHOR_FILE
    if anchor.exists():
        try:
            existing = json.loads(anchor.read_text())
            return {
                "status": "already_initialized",
                "soul_id": existing.get("soul_id"),
                "data_root": str(drive),
                "message": f"Drive already initialized as Soul '{existing.get('soul_id', '')[:8]}'",
            }
        except Exception:
            pass

    # Create directory structure
    created = []
    for subdir in SOUL_DIRECTORIES:
        target = drive / subdir
        target.mkdir(parents=True, exist_ok=True)
        created.append(subdir)

    # Write anchor file
    soul_id = str(uuid.uuid4())
    anchor_data = {
        "soul_id": soul_id,
        "version": "2.0.0",
        "created_at": time.time(),
        "created_by": owner or os.environ.get("USER", "unknown"),
        "hostname": platform.node(),
        "machine": platform.machine(),
        "data_root": str(drive),
        "directories": SOUL_DIRECTORIES,
    }

    try:
        anchor.write_text(json.dumps(anchor_data, indent=2))
    except Exception as e:
        return {"status": "error", "message": f"Failed to write anchor: {e}"}

    log.info(f"[DriveInit] Soul initialized: {soul_id[:8]} on {drive_path}")
    log.info(f"[DriveInit] Created {len(created)} directories")

    return {
        "status": "initialized",
        "soul_id": soul_id,
        "data_root": str(drive),
        "directories_created": created,
        "message": f"Soul '{soul_id[:8]}' initialized on {drive.name}",
    }


def verify_soul_drive(drive_path: str) -> dict:
    """Verify an existing Soul drive is intact.

    Checks anchor file, directory structure, and ChromaDB presence.
    """
    drive = Path(drive_path).resolve()

    # §SEC: Path validation
    allowed_root = os.environ.get("EDITH_DATA_ROOT", "")
    if not (str(drive).startswith("/Volumes/") or
            (allowed_root and str(drive).startswith(allowed_root))):
        return {"status": "error", "valid": False,
                "message": f"Security: drive must be under /Volumes/. Got: {drive}"}

    anchor = drive / ANCHOR_FILE
    if not anchor.exists():
        return {"status": "not_initialized", "valid": False}

    try:
        anchor_data = json.loads(anchor.read_text())
    except Exception:
        return {"status": "corrupt_anchor", "valid": False}

    missing = []
    for subdir in SOUL_DIRECTORIES:
        if not (drive / subdir).exists():
            missing.append(subdir)

    has_chroma = (drive / "ChromaDB" / "chroma.sqlite3").exists()
    has_vault = (drive / ".vault" / "keys.enc").exists()
    has_brain = any((drive / "Brain").glob("*.jsonl")) if (drive / "Brain").exists() else False

    return {
        "status": "valid" if not missing else "incomplete",
        "valid": len(missing) == 0,
        "soul_id": anchor_data.get("soul_id"),
        "data_root": str(drive),
        "missing_dirs": missing,
        "has_vectors": has_chroma,
        "has_vault": has_vault,
        "has_brain": has_brain,
        "created_at": anchor_data.get("created_at"),
        "created_by": anchor_data.get("created_by"),
    }


def get_bolt_env_mapping(bolt_path: str) -> dict[str, str]:
    """Return environment variable mappings for a mounted Bolt drive.

    These get set during the ignition sequence so the backend
    reads/writes data to the Bolt instead of the local Mac.
    """
    return {
        "EDITH_DATA_ROOT": bolt_path,
        "CHROMA_DB_PATH": f"{bolt_path}/ChromaDB",
        "EDITH_LIBRARY_PATH": f"{bolt_path}/Library",
        "EDITH_BRAIN_PATH": f"{bolt_path}/Brain",
        "EDITH_VAULT_PATH": f"{bolt_path}/.vault",
        "STATA_DATA": f"{bolt_path}/Stata_Work",
        "EDITH_CONNECTOME_PATH": f"{bolt_path}/Connectome",
        "EDITH_EXPORTS_PATH": f"{bolt_path}/exports",
    }
