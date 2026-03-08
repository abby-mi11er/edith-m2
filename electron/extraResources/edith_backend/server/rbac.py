"""
Role-Based Access Control (RBAC) for Edith server.

Roles:
- admin:  Full access — settings, data management, all queries
- editor: Can query, view sources, submit feedback, use all tools
- viewer: Read-only — can query and view results, no feedback/settings
- guest:  Minimal — health check only, no queries

Role assignment is stored in a local JSON file alongside config.
§4.3: Thread-safe user file writes.
§4.9: Admin user management API helpers.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, Request

# ---------------------------------------------------------------------------
# Role definitions
# ---------------------------------------------------------------------------

ROLES = {
    "admin": {
        "description": "Full access to all features",
        "permissions": frozenset({
            "chat", "chat_stream", "feedback", "test_query", "test_retrieve",
            "literature_review", "research_design", "file_access", "library",
            "settings", "index", "status", "doctor", "export", "admin",
        }),
    },
    # §FIX V3: Explicit 'local' role for localhost Electron app connections
    # (security.py returns role='local' for 127.0.0.1/::1)
    "local": {
        "description": "Local machine access (Electron app)",
        "permissions": frozenset({
            "chat", "chat_stream", "feedback", "test_query", "test_retrieve",
            "literature_review", "research_design", "file_access", "library",
            "settings", "index", "status", "doctor", "export", "admin",
        }),
    },
    "editor": {
        "description": "Query, feedback, and tool access",
        "permissions": frozenset({
            "chat", "chat_stream", "feedback", "test_query", "test_retrieve",
            "literature_review", "research_design", "file_access", "library",
            "status", "export",
        }),
    },
    "viewer": {
        "description": "Read-only query and view access",
        "permissions": frozenset({
            "chat", "chat_stream", "file_access", "library", "status",
        }),
    },
    "guest": {
        "description": "Health check only",
        "permissions": frozenset({
            "status",
        }),
    },
}

DEFAULT_ROLE = os.environ.get("EDITH_RBAC_DEFAULT_ROLE", "viewer")

# ---------------------------------------------------------------------------
# Route → permission mapping
# ---------------------------------------------------------------------------

_ROUTE_PERMISSIONS: dict[str, str] = {
    "/chat": "chat",
    "/chat/stream": "chat_stream",
    "/api/feedback": "feedback",
    "/api/test-query": "test_query",
    "/api/test-retrieve": "test_retrieve",
    "/api/literature-review": "literature_review",
    "/api/research-design": "research_design",
    "/api/file": "file_access",
    "/api/library": "library",
    "/api/library/progress": "library",
    "/api/library/detail": "library",
    "/status": "status",
    "/api/status": "status",
    "/api/index": "index",
    "/api/doctor": "doctor",
    "/api/export": "export",
    "/api/settings": "settings",
    "/api/admin": "admin",
}

# ---------------------------------------------------------------------------
# User store (simple local JSON) — thread-safe (§4.3)
# ---------------------------------------------------------------------------

_USERS_FILE: Optional[Path] = None
_USERS: dict[str, dict] = {}  # token_hash → {"name": ..., "role": ...}
_USERS_LOCK = threading.Lock()


def init_rbac(users_file: str = "") -> None:
    """Load user→role mappings from a JSON file.

    Also auto-registers the admin user from EDITH_ACCESS_PASSWORD.
    """
    global _USERS_FILE, _USERS, DEFAULT_ROLE
    if not users_file:
        app_data = os.environ.get(
            "EDITH_APP_DATA_DIR",
            str(Path(__file__).parent.parent),
        )
        _USERS_FILE = Path(app_data) / "edith_users.json"
    else:
        _USERS_FILE = Path(users_file)

    with _USERS_LOCK:
        if _USERS_FILE.exists():
            try:
                data = json.loads(_USERS_FILE.read_text(encoding="utf-8"))
                if "users" in data and isinstance(data["users"], dict):
                    _USERS.update(data["users"])
                else:
                    _USERS.update(
                        {k: v for k, v in data.items()
                         if isinstance(v, dict) and "role" in v}
                    )
                if "default_role" in data and data["default_role"] in ROLES:
                    DEFAULT_ROLE = data["default_role"]
            except Exception:
                pass

        # Auto-register admin from EDITH_ACCESS_PASSWORD
        password = os.environ.get("EDITH_ACCESS_PASSWORD", "")
        if password:
            bearer_token = hashlib.sha256(
                (password + ":edith-api-token-salt").encode()
            ).hexdigest()
            admin_hash = hashlib.sha256(bearer_token.encode()).hexdigest()
            _USERS[admin_hash] = {"name": "admin", "role": "admin"}


def get_user_role(token_hash: str) -> str:
    """Look up the role for a given token hash."""
    with _USERS_LOCK:
        user = _USERS.get(token_hash)
    if user and isinstance(user.get("role"), str):
        role = user["role"]
        if role in ROLES:
            return role
    return DEFAULT_ROLE


def add_user(name: str, token_hash: str, role: str = "editor") -> None:
    """Add a user→role mapping and persist to disk (§4.3: thread-safe)."""
    if role not in ROLES:
        raise ValueError(f"Invalid role: {role}. Must be one of {list(ROLES.keys())}")
    with _USERS_LOCK:
        _USERS[token_hash] = {"name": name, "role": role}
        _save_users_locked()


def remove_user(token_hash: str) -> bool:
    """Remove a user by token hash.  Returns True if removed."""
    with _USERS_LOCK:
        if token_hash in _USERS:
            del _USERS[token_hash]
            _save_users_locked()
            return True
    return False


def list_users() -> list[dict]:
    """Return a list of users (§4.9: admin API helper).  Hashes are truncated."""
    with _USERS_LOCK:
        return [
            {"hash_prefix": h[:12], "name": u.get("name", ""), "role": u.get("role", "")}
            for h, u in _USERS.items()
        ]


def _save_users_locked() -> None:
    """Persist user mappings to disk.  Must be called with _USERS_LOCK held.

    §4.3: Atomic write via tempfile + rename to prevent corruption.
    """
    if not _USERS_FILE:
        return
    _USERS_FILE.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps({"users": _USERS, "default_role": DEFAULT_ROLE}, indent=2)
    # Atomic write: write to temp file, then rename
    fd, tmp_path = tempfile.mkstemp(
        dir=str(_USERS_FILE.parent), suffix=".tmp"
    )
    closed = False
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        closed = True
        os.replace(tmp_path, str(_USERS_FILE))
    except Exception:
        if not closed:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


# Keep old name for backwards compat
_save_users = lambda: None  # no-op — use add_user() which calls _save_users_locked()


# ---------------------------------------------------------------------------
# RBAC enforcement dependency
# ---------------------------------------------------------------------------


def check_permission(request: Request, token_hash: str = "", explicit_role: str = "") -> None:
    """Check if the current user has permission for this route."""
    path = request.url.path
    permission = None
    for route_prefix, perm in _ROUTE_PERMISSIONS.items():
        if path == route_prefix or path.startswith(route_prefix + "/"):
            permission = perm
            break

    if permission is None:
        return

    role = explicit_role if explicit_role in ROLES else get_user_role(token_hash)
    allowed = ROLES.get(role, {}).get("permissions", frozenset())

    if permission not in allowed:
        raise HTTPException(
            status_code=403,
            detail=f"Role '{role}' does not have '{permission}' permission",
        )
