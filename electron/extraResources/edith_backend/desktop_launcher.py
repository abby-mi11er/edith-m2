#!/usr/bin/env python3
"""
desktop_launcher.py — Electron desktop launcher for E.D.I.T.H.

This script is invoked by electron/main.js to start the FastAPI backend.
It is designed to be spawned as a subprocess with the following env vars:
    - EDITH_PORT: port to bind (default 8001)
    - EDITH_OPEN_BROWSER: set to "false" to suppress browser auto-open
    - EDITH_DATA_ROOT: root directory for data files
    - GOOGLE_API_KEY: Gemini API key
    - EDITH_DESKTOP_MODE: "electron" when launched from Electron shell
"""

import logging
import os
import sqlite3
import sys

# Ensure project root is on sys.path for imports
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
log = logging.getLogger("edith.launcher")


def _is_sqlite_ok(sqlite_path: str) -> bool:
    """Return True only if sqlite_path is a readable SQLite database."""
    try:
        conn = sqlite3.connect(sqlite_path)
        try:
            conn.execute("PRAGMA schema_version;").fetchone()
        finally:
            conn.close()
        return True
    except Exception:
        return False


def _is_candidate_usable(path: str) -> bool:
    """A candidate directory is usable when it exists and has no broken sqlite file."""
    if not os.path.isdir(path):
        return False
    sqlite_path = os.path.join(path, "chroma.sqlite3")
    if not os.path.exists(sqlite_path):
        return True
    return _is_sqlite_ok(sqlite_path)


def _resolve_chroma_dir() -> str:
    """Pick a valid Chroma path, even if .env points at a missing Bolt mount."""
    configured = (os.environ.get("EDITH_CHROMA_DIR") or "").strip()
    if configured and _is_candidate_usable(configured):
        return configured

    candidates = [
        "/Volumes/Edith Bolt/ChromaDB",
        "/Volumes/Edith Bolt/chroma",
    ]

    data_root = (os.environ.get("EDITH_DATA_ROOT") or "").strip()
    if data_root:
        candidates.extend([
            os.path.join(data_root, "ChromaDB"),
            os.path.join(data_root, "chroma"),
        ])

    candidates.extend([
        os.path.join(ROOT_DIR, "chroma"),
        os.path.join(ROOT_DIR, "chroma_store"),
        os.path.join(ROOT_DIR, "vectors"),
        os.path.expanduser("~/Library/Application Support/Edith/chroma"),
    ])

    # Prefer a path that already has a valid Chroma sqlite database.
    for candidate in candidates:
        sqlite_path = os.path.join(candidate, "chroma.sqlite3")
        if os.path.isdir(candidate) and os.path.exists(sqlite_path) and _is_sqlite_ok(sqlite_path):
            return candidate

    # Fallback to an existing usable directory.
    for candidate in candidates:
        if _is_candidate_usable(candidate):
            return candidate

    # Last resort: create a fresh local project directory.
    fallback = os.path.join(ROOT_DIR, "chroma_store")
    os.makedirs(fallback, exist_ok=True)
    return fallback


def main():
    port = int(os.environ.get("EDITH_PORT", "8001"))
    host = os.environ.get("EDITH_HOST", "127.0.0.1")
    open_browser = os.environ.get("EDITH_OPEN_BROWSER", "true").lower() != "false"

    log.info(f"E.D.I.T.H. desktop launcher starting on {host}:{port}")
    log.info(f"  Data root: {os.environ.get('EDITH_DATA_ROOT', '(default)')}")
    log.info(f"  Desktop mode: {os.environ.get('EDITH_DESKTOP_MODE', 'standalone')}")
    log.info(f"  Python: {sys.executable} ({sys.version_info.major}.{sys.version_info.minor})")

    # Default to production when launched from Electron (packaged app)
    if os.environ.get("EDITH_DESKTOP_MODE") == "electron":
        os.environ.setdefault("EDITH_ENV", "production")
    log.info(f"  Environment: {os.environ.get('EDITH_ENV', 'development')}")

    resolved_chroma = _resolve_chroma_dir()
    configured_chroma = (os.environ.get("EDITH_CHROMA_DIR") or "").strip()
    if configured_chroma and not _is_candidate_usable(configured_chroma):
        log.warning(f"  Chroma dir unavailable or invalid: {configured_chroma} -> using {resolved_chroma}")
    elif not configured_chroma:
        log.info(f"  Chroma dir: {resolved_chroma}")

    os.environ["EDITH_CHROMA_DIR"] = resolved_chroma
    os.environ.setdefault("CHROMA_DB_PATH", resolved_chroma)

    try:
        import uvicorn
    except ImportError:
        log.error("uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

    # Optionally open browser (suppressed when launched from Electron)
    if open_browser:
        import threading
        import webbrowser
        threading.Timer(2.0, lambda: webbrowser.open(f"http://{host}:{port}")).start()

    uvicorn.run(
        "server.main:app",
        host=host,
        port=port,
        log_level="info",
        timeout_keep_alive=30,
        # Disable reload in production / Electron
        reload=False,
    )


if __name__ == "__main__":
    main()
