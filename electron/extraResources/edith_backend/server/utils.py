"""
Shared utilities for E.D.I.T.H. server modules.
"""

import json
import os
import logging
from pathlib import Path

log = logging.getLogger("edith.utils")


def atomic_write_json(path: Path, data: dict, indent: int = 2) -> None:
    """Write JSON atomically using write-then-rename.

    Writes to a .tmp file first, then uses os.replace() (which is
    atomic on POSIX) to swap it into place. This prevents corruption
    if the process is killed or the Bolt drive is pulled mid-write.

    Args:
        path: Final destination path.
        data: Dict to serialize as JSON.
        indent: JSON indentation (default 2).
    """
    tmp = path.parent / (path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(data, indent=indent, default=str))
        os.replace(str(tmp), str(path))
    except Exception:
        # Clean up temp file on failure
        try:
            tmp.unlink(missing_ok=True)
        except Exception:
            pass
        raise


def _normalized_app_mode() -> str:
    """Return one of: production, development, test."""
    raw = (os.environ.get("EDITH_APP_MODE") or os.environ.get("EDITH_ENV") or "development").strip().lower()
    aliases = {
        "prod": "production",
        "production": "production",
        "dev": "development",
        "development": "development",
        "test": "test",
        "testing": "test",
    }
    return aliases.get(raw, "development")


def preflight_check() -> dict:
    """Pre-launch validation: storage, retrieval, credentials, and deps.

    Production mode is strict (missing paths/deps are blocking).
    Development/test mode is permissive (warn loudly, keep booting).
    """
    issues = []
    warnings = []
    mode = _normalized_app_mode()
    strict = mode == "production"

    def _record_path_problem(msg: str):
        if strict:
            issues.append(msg)
        else:
            warnings.append(msg)

    # 1. Check Bolt / EDITH_DATA_ROOT
    data_root = os.environ.get("EDITH_DATA_ROOT", "")
    if data_root:
        p = Path(data_root)
        if not p.exists():
            _record_path_problem(f"EDITH_DATA_ROOT={data_root} does not exist")
        elif not p.is_dir():
            _record_path_problem(f"EDITH_DATA_ROOT={data_root} is not a directory")
    elif strict:
        issues.append("EDITH_DATA_ROOT not set in production mode")
    else:
        warnings.append("EDITH_DATA_ROOT not set — using local storage")

    # 2. Check ChromaDB
    chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "")
    if chroma_dir:
        cp = Path(chroma_dir)
        if not cp.exists():
            _record_path_problem(f"EDITH_CHROMA_DIR={chroma_dir} does not exist")
        elif not cp.is_dir():
            _record_path_problem(f"EDITH_CHROMA_DIR={chroma_dir} is not a directory")
        else:
            sqlite_files = list(cp.glob("*.sqlite3"))
            if not sqlite_files:
                warnings.append(f"EDITH_CHROMA_DIR={chroma_dir} has no .sqlite3 files")
    elif strict:
        issues.append("EDITH_CHROMA_DIR not set in production mode")
    else:
        warnings.append("EDITH_CHROMA_DIR not set — retrieval may be limited")

    # 3. Check API keys
    if not os.environ.get("OPENAI_API_KEY", ""):
        warnings.append("OPENAI_API_KEY not set — OpenAI inference will fail")
    if not os.environ.get("GOOGLE_API_KEY", "") and not os.environ.get("GEMINI_API_KEY", ""):
        warnings.append("GOOGLE_API_KEY not set — Gemini inference will fail")

    # 4. Check Python deps
    for pkg in ["fastapi", "uvicorn"]:
        try:
            __import__(pkg)
        except ImportError:
            issues.append(f"Missing required package: {pkg}")

    try:
        __import__("chromadb")
    except ImportError:
        if strict:
            issues.append("Missing required package: chromadb")
        else:
            warnings.append("chromadb not installed — retrieval endpoints may be unavailable")

    status = "ready" if not issues else "blocked"
    result = {"status": status, "mode": mode, "issues": issues, "warnings": warnings}

    for issue in issues:
        log.error(f"§PREFLIGHT FAIL: {issue}")
    for warn in warnings:
        log.warning(f"§PREFLIGHT WARN: {warn}")
    if not issues:
        log.info(f"§PREFLIGHT ({mode}): checks passed with {len(warnings)} warning(s)")

    return result
