"""
E.D.I.T.H. Configuration — All env vars, model configs, and paths
=================================================================
Extracted from main.py to centralize configuration.
Every config value lives here. Nothing is scattered.
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

log = logging.getLogger("edith.config")

# ── Paths ──
ROOT_DIR = Path(__file__).parent.parent
load_dotenv(ROOT_DIR / ".env")

# ── API Keys ──
API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DATA_ROOT = os.environ.get("EDITH_DATA_ROOT")

if os.environ.get("GOOGLE_API_KEY") and os.environ.get("GEMINI_API_KEY"):
    log.info("Both GOOGLE_API_KEY and GEMINI_API_KEY are set. Using GOOGLE_API_KEY.")

# ── Model Configuration ──
OPENAI_FT_MODEL = os.environ.get("WINNIE_OPENAI_MODEL",
    os.environ.get("OPENAI_BASE_MODEL", ""))
DEFAULT_MODEL = os.environ.get("EDITH_MODEL", "gemini-2.5-flash")
ORACLE_MODEL = os.environ.get("EDITH_ORACLE_MODEL", "gemini-2.5-pro")
_fallback_raw = os.environ.get("EDITH_MODEL_FALLBACKS", "gemini-2.5-flash,gemini-2.0-flash")
FALLBACK_MODELS = [m.strip() for m in _fallback_raw.split(",") if m.strip()]
EMBED_MODEL = os.environ.get("EDITH_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

if OPENAI_FT_MODEL and OPENAI_API_KEY:
    log.info(f"Winnie model active: {OPENAI_FT_MODEL} (primary, domain-trained)")
else:
    log.info(f"Winnie not available — using Gemini-only chain")

# ── ChromaDB Configuration ──

def _find_chroma_dir() -> str:
    """Auto-discover ChromaDB directory — prefer CITADEL/Bolt over internal SSD."""
    env_dir = os.environ.get("EDITH_CHROMA_DIR") or os.environ.get("CHROMA_DB_PATH")
    if env_dir and os.path.isdir(env_dir):
        return env_dir
    _dr = os.environ.get("EDITH_DATA_ROOT", "")
    if _dr:
        # drive_initialization.py creates "ChromaDB/" (capital C)
        for sub in ("ChromaDB", "chroma"):
            dr_chroma = os.path.join(_dr, sub)
            if os.path.isdir(dr_chroma):
                return dr_chroma
    try:
        from server.vault_config import VAULT_ROOT
        citadel_chroma = str(VAULT_ROOT / "edith_data" / "chroma")
    except ImportError:
        citadel_chroma = ""
    if citadel_chroma and os.path.isdir(citadel_chroma):
        return citadel_chroma
    try:
        from server.vault_config import VECTORS_DIR
        candidates = [str(ROOT_DIR / "chroma"), str(VECTORS_DIR)]
    except ImportError:
        candidates = [str(ROOT_DIR / "chroma")]
    for d in candidates:
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "chroma.sqlite3")):
            return d
    _legacy_dirs = [
        os.path.expanduser("~/edith_data/chromadb"),
        os.path.expanduser("~/Library/Application Support/Edith/chroma"),
    ]
    for d in _legacy_dirs:
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "chroma.sqlite3")):
            log.warning(f"§LEAK: ChromaDB on internal SSD at {d} — migrate to CITADEL drive")
            return d
    return str(ROOT_DIR / "chroma")


def _find_collection_name() -> str:
    env_name = os.environ.get("EDITH_CHROMA_COLLECTION")
    if env_name:
        return env_name
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        colls = client.list_collections()
        if colls:
            best = max(colls, key=lambda c: c.count())
            log.info(f"Auto-detected ChromaDB collection: {best.name} ({best.count()} chunks)")
            return best.name
    except Exception as _exc:
        log.warning(f"Suppressed exception: {_exc}")
    return "edith_docs_sections"


CHROMA_DIR = _find_chroma_dir()
CHROMA_COLLECTION = _find_collection_name()

# ── Retrieval Configuration ──
RETRIEVAL_BACKEND = os.environ.get("EDITH_RETRIEVAL_BACKEND", "chroma").lower()
GOOGLE_STORE_ID = os.environ.get("EDITH_GOOGLE_STORE_ID", "")

try:
    from server.google_retrieval import google_retrieval_available
    USE_GOOGLE_RETRIEVAL = (
        RETRIEVAL_BACKEND == "google" and
        google_retrieval_available(API_KEY or "", GOOGLE_STORE_ID)
    )
except ImportError:
    USE_GOOGLE_RETRIEVAL = False

if USE_GOOGLE_RETRIEVAL:
    log.info(f"Retrieval backend: Google File Search (store: {GOOGLE_STORE_ID})")
else:
    log.info(f"Retrieval backend: Chroma (dir: {CHROMA_DIR})")

# ── Model Chain Builder ──

def build_model_chain(requested: str) -> list:
    """Build a model chain from the requested model + configured fallbacks."""
    chain = [requested]
    for fb in FALLBACK_MODELS:
        if fb not in chain:
            chain.append(fb)
    if DEFAULT_MODEL not in chain:
        chain.append(DEFAULT_MODEL)
    return chain
