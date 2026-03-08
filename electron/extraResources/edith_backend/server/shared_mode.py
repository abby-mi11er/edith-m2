"""
Edith Shared Library Mode — Allow multiple users to search the same library.

When EDITH_SHARED_MODE=true:
- App binds to 0.0.0.0 instead of 127.0.0.1
- Shared users authenticate via token
- Shared users get read-only search access (no settings/training)

Usage:
    Set EDITH_SHARED_MODE=true in .env
    Set EDITH_SHARED_TOKENS=token1,token2,token3 in .env
"""

import os
import hashlib
import secrets
from functools import wraps

SHARED_MODE = os.environ.get("EDITH_SHARED_MODE", "false").lower() == "true"
SHARED_TOKENS = set(
    t.strip() for t in os.environ.get("EDITH_SHARED_TOKENS", "").split(",") if t.strip()
)
OWNER_TOKEN = os.environ.get("EDITH_OWNER_TOKEN", "").strip()


def is_shared_mode() -> bool:
    """Check if shared mode is enabled."""
    return SHARED_MODE


def generate_share_token() -> str:
    """Generate a new share token."""
    return secrets.token_urlsafe(24)


def validate_share_token(token: str) -> dict:
    """Validate a share token.

    Returns: {"valid": bool, "role": "owner"|"reader"|"invalid"}
    """
    if not token:
        return {"valid": False, "role": "invalid"}

    token = token.strip()

    # Check if it's the owner token
    if OWNER_TOKEN and token == OWNER_TOKEN:
        return {"valid": True, "role": "owner"}

    # Check if it's a valid shared token
    if token in SHARED_TOKENS:
        return {"valid": True, "role": "reader"}

    # Check hashed tokens (for stored tokens)
    token_hash = hashlib.sha256(token.encode()).hexdigest()
    if token_hash in SHARED_TOKENS:
        return {"valid": True, "role": "reader"}

    return {"valid": False, "role": "invalid"}


def get_shared_permissions(role: str) -> dict:
    """Get permissions for a given role.

    Returns dict of feature -> enabled.
    """
    if role == "owner":
        return {
            "search": True,
            "chat": True,
            "settings": True,
            "training": True,
            "export": True,
            "feedback": True,
            "indexing": True,
        }
    elif role == "reader":
        return {
            "search": True,
            "chat": True,
            "settings": False,
            "training": False,
            "export": True,
            "feedback": True,
            "indexing": False,
        }
    return {
        "search": False,
        "chat": False,
        "settings": False,
        "training": False,
        "export": False,
        "feedback": False,
        "indexing": False,
    }


def get_streamlit_server_args() -> dict:
    """Get Streamlit server configuration for shared mode.

    Returns args to pass to streamlit run.
    """
    if SHARED_MODE:
        return {
            "server.address": "0.0.0.0",
            "server.port": int(os.environ.get("EDITH_SHARED_PORT", "8501")),
            "server.headless": True,
        }
    return {
        "server.address": "127.0.0.1",
    }
