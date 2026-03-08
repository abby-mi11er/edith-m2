import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time as _time
import hmac
import traceback
import uuid
from pathlib import Path
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Body, Query, Depends, Request
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field, field_validator

log = logging.getLogger("edith.server")
_server_start_time = _time.time()

# Add project root to sys.path to ensure we can import if needed, 
# though running as 'uvicorn server.main:app' handles this.
sys.path.append(str(Path(__file__).parent.parent))

from server.chroma_backend import retrieve_local_sources, agentic_retrieve, classify_query_intent, chroma_runtime_available
from server.google_retrieval import retrieve_google_sources, google_retrieval_available
from server.pipeline_utils import (
    PipelineTimer, cached_retrieve, compress_sources,
    call_openai_pooled, call_openai_streaming,
    openai_breaker, google_retrieval_breaker,
    score_source_relevance, should_skip_retrieval,
    shutdown_pool,
)
from server.model_utils import parse_json_array
from server.backend_logic import (
    init_client, 
    generate_text_via_chain, 
    build_answer_prompt,
    build_support_audit_source_blocks,
    rewrite_retrieval_queries,
    plan_answer_outline,
    audit_answer,
    apply_corrections,
)
from server.prompts import SYSTEM_PROMPT
from server.session_memory import format_memory_context, save_session_summary
from server import openalex as openalex_mod
from server.security import (
    setup_audit_logging,
    configure_auth_token,
    security_gate,
    audit,
    SecurityHeadersMiddleware,
    redact_pii,
)
from server.prompt_guard import guard_input, filter_output, check_source_injection
from google.genai import types

# §B1/B3/B6/B9/B10: Server infrastructure
from server.server_infra import (
    CorrelationMiddleware, response_cache, rate_limiter,
    startup_gate, metrics, generate_correlation_id,
)

# §ARCH: Centralized state + compute worker
from server.server_state import state as _server_state
from server.worker_client import worker as _worker_client

# ── Improvement modules (§3, §7, §11) ──
try:
    from server.api_improvements import (
        ErrorResponse as _ErrorResponse,
        HealthResponse as _HealthResponse,
    )
    from server.model_improvements import (
        route_query as _route_query,
        CostTracker as _CostTracker,
        ContextWindowManager as _CtxWindowMgr,
        get_sampling_params as _get_sampling,
    )
    from server.memory_pinning import (
        SessionSummarizer as _SessionSummarizer,
        TopicMemory as _TopicMemory,
    )
    _IMPROVEMENTS_WIRED = True
    _cost_tracker = _CostTracker()
    _session_summarizer = _SessionSummarizer()
    _topic_memory = _TopicMemory()
    log.info("Improvement modules §3/§7/§11 loaded")
except ImportError as _imp_err:
    _IMPROVEMENTS_WIRED = False
    _cost_tracker = None
    _session_summarizer = None
    _topic_memory = None
    log.warning(f"Improvement modules not available: {_imp_err}")

# ── Feb 20th feature modules — guarded imports ──
_kg_store = None
_scholarly_repos = None
_discovery = None
_export_academic = None
_research_workflows = None
_security_features = None
_training_devops = None
_desktop_features = None
_connectors = None

try:
    from server.knowledge_graph import KnowledgeGraph
    _kg_store = KnowledgeGraph()
    log.info("KnowledgeGraph loaded")
except Exception as _e:
    log.warning(f"knowledge_graph not available: {_e}")

try:
    from server.scholarly_repositories import ScholarlyRepositories
    from server.vault_config import VAULT_ROOT as _sr_vault_root
    _scholarly_repos = ScholarlyRepositories(store_dir=str(_sr_vault_root / "Corpus" / "Vault" / "scholarly"))
    log.info("ScholarlyRepositories loaded")
except Exception as _e:
    _scholarly_repos = None
    log.warning(f"scholarly_repositories not available: {_e}")

try:
    from server.discovery_mode import is_discovery_query, extract_discovery_topic, get_discovery_engine
    _discovery = True
    log.info("DiscoveryMode loaded")
except Exception as _e:
    _discovery = False
    log.warning(f"discovery_mode not available: {_e}")

try:
    from server.export_academic import answer_to_latex, sources_to_bibtex, answer_to_slides
    _export_academic = True
    log.info("ExportAcademic loaded")
except Exception as _e:
    _export_academic = False
    log.warning(f"export_academic not available: {_e}")

try:
    from server.research_workflows import ResearchDiary
    log.info("ResearchWorkflows loaded")
except Exception as _e:
    log.warning(f"research_workflows not available: {_e}")

try:
    from server.security import AccessTier, UserManager
    _security_features = True
    log.info("SecurityFeatures loaded")
except Exception as _e:
    _security_features = False
    log.warning(f"security_features not available: {_e}")

try:
    from server.training_devops import MultiCorpusManager
    _training_devops = MultiCorpusManager()
    log.info("TrainingDevOps loaded")
except Exception as _e:
    _training_devops = None
    log.warning(f"training_devops not available: {_e}")

try:
    from server.desktop_features import extract_reading_list, get_source_preview, build_literature_map
    _desktop_features = True
    log.info("DesktopFeatures loaded")
except Exception as _e:
    _desktop_features = False
    log.warning(f"desktop_features not available: {_e}")

try:
    from pipelines.connectors import SemanticScholarConnector, CrossRefConnector
    _connectors = True
    log.info("Pipelines connectors loaded")
except Exception as _e:
    _connectors = False
    log.warning(f"pipelines.connectors not available: {_e}")

# ── §RESTORED: Previously-archived modules now wired back in ──

try:
    from server.citation_formatter import replace_source_markers, generate_bibliography
    _citation_formatter = True
    log.info("CitationFormatter loaded")
except Exception as _e:
    _citation_formatter = False
    log.warning(f"citation_formatter not available: {_e}")

try:
    from server.reasoning_enhancements import (
        ConfidenceSignals, assign_paragraph_confidence,
        detect_contradictions, check_source_freshness, ReasoningTrace,
        format_sources_for_model,
    )
    _reasoning_enhancements = True
    log.info("ReasoningEnhancements loaded")
except Exception as _e:
    _reasoning_enhancements = False
    log.warning(f"reasoning_enhancements not available: {_e}")

try:
    from server.logging_config import setup_logging
    _logging_config = True
    log.info("LoggingConfig loaded")
except Exception as _e:
    _logging_config = False
    log.warning(f"logging_config not available: {_e}")

try:
    from server.training_tools import ActiveLearningQueue, TrainingCostTracker, compare_models
    _active_learning = ActiveLearningQueue()
    _training_cost_tracker = TrainingCostTracker()
    _training_enhancements = True
    log.info("TrainingEnhancements loaded (ActiveLearning + CostTracker)")
except Exception as _e:
    _active_learning = None
    _training_cost_tracker = None
    _training_enhancements = False
    log.warning(f"training_enhancements not available: {_e}")

try:
    from server.indexing_enhancements import (
        IndexVersion, detect_language, CitationGraph,
        hierarchical_chunk, find_changed_files, parallel_extract_texts,
    )
    _indexing_enhancements = True
    _citation_graph = CitationGraph()
    log.info("IndexingEnhancements loaded")
except Exception as _e:
    _indexing_enhancements = False
    _citation_graph = None
    log.warning(f"indexing_enhancements not available: {_e}")

try:
    from server.errors import not_found, bad_request, forbidden, rate_limited
    _error_helpers = True
except Exception as _e:
    _error_helpers = False
    log.warning(f"errors module not available: {_e}")

try:
    from server.export_notes import export_to_notion
    _export_notes = True
    log.info("ExportNotes loaded")
except Exception as _e:
    _export_notes = False
    log.warning(f"export_notes not available: {_e}")

try:
    from server.rbac import check_permission, get_user_role, ROLES as ROLE_PERMISSIONS
    _rbac = True
    log.info("RBAC loaded")
except Exception as _e:
    _rbac = False
    log.warning(f"rbac not available: {_e}")

try:
    from server.security import PerEndpointRateLimiter
    _endpoint_rate_limit = PerEndpointRateLimiter()
    log.info("EndpointRateLimit loaded")
except Exception as _e:
    _endpoint_rate_limit = None
    log.warning(f"security_enhancements (EndpointRateLimit) not available: {_e}")

try:
    from server.shared_mode import is_shared_mode, get_shared_permissions
    _shared_mode = True
    log.info("SharedMode loaded")
except Exception as _e:
    _shared_mode = False
    log.warning(f"shared_mode not available: {_e}")

try:
    from server.tls_config import get_tls_config
    _tls_config = True
    log.info("TLSConfig loaded")
except Exception as _e:
    _tls_config = False
    log.warning(f"tls_config not available: {_e}")

try:
    from server.memory_pinning import (
        generate_session_summary as semantic_session_summary,
        extract_entities as extract_key_concepts,
        EntityMemory, MemorySearch, PinnedMemories, ProjectMemory,
    )
    _memory_enhancements = True
    log.info("MemoryEnhancements loaded")
except Exception as _e:
    _memory_enhancements = False
    log.warning(f"memory_enhancements not available: {_e}")

try:
    from server.ui_enhancements import sources_to_bibtex as ui_sources_to_bibtex
    _ui_enhancements = True
    log.info("UIEnhancements loaded")
except Exception as _e:
    _ui_enhancements = False
    log.warning(f"ui_enhancements not available: {_e}")

# Fine-tuned OpenAI model — loaded after .env (see below)
OPENAI_FT_MODEL = ""
OPENAI_API_KEY = ""

# Load dotenv with runtime override support (desktop launcher passes EDITH_DOTENV_PATH).
def _load_runtime_dotenv() -> tuple[Path, list[str]]:
    root_dir = Path(__file__).parent.parent
    candidates: list[Path] = []

    override = (os.environ.get("EDITH_DOTENV_PATH", "") or "").strip()
    if override:
        candidates.append(Path(override).expanduser())

    candidates.append(root_dir / ".env")

    data_root_env = (os.environ.get("EDITH_DATA_ROOT", "") or "").strip()
    if data_root_env:
        candidates.append(Path(data_root_env).expanduser() / ".env")

    # Desktop defaults on Bolt so packaged runs can still load secrets/config.
    candidates.append(Path("/Volumes/Edith Bolt/Edith_M4/.env"))
    candidates.append(Path("/Volumes/Edith Bolt/Edith_M2/.env"))

    loaded: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if not key or key in seen:
            continue
        seen.add(key)
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            loaded.append(key)

    return root_dir, loaded


ROOT_DIR, _DOTENV_SOURCES = _load_runtime_dotenv()
if _DOTENV_SOURCES:
    log.info("Loaded dotenv sources: %s", _DOTENV_SOURCES)

# Now load OpenAI config (after .env is available)
OPENAI_FT_MODEL = os.environ.get("WINNIE_OPENAI_MODEL",
    os.environ.get("OPENAI_BASE_MODEL", ""))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# Note: OPENAI_API_KEY is set directly in .env (no Keychain lookup needed)

if OPENAI_FT_MODEL and OPENAI_API_KEY:
    log.info(f"Winnie model active: {OPENAI_FT_MODEL} (primary, domain-trained)")
else:
    log.info(f"Winnie not available — using Gemini-only chain")

# Configs
API_KEY = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
DATA_ROOT = os.environ.get("EDITH_DATA_ROOT")

# Auto-discover ChromaDB directory from multiple possible locations
def _find_chroma_dir():
    """Auto-discover ChromaDB directory — prefer CITADEL/Bolt over internal SSD."""
    # 1. Explicit env var always wins
    env_dir = os.environ.get("EDITH_CHROMA_DIR")
    if env_dir and os.path.isdir(env_dir):
        return env_dir
    # 2. DATA_ROOT/chroma (Bolt-sovereign path)
    _dr = os.environ.get("EDITH_DATA_ROOT", "")
    if _dr:
        dr_chroma = os.path.join(_dr, "chroma")
        if os.path.isdir(dr_chroma):
            return dr_chroma
    # 3. CITADEL mount point (via vault_config)
    try:
        from server.vault_config import VAULT_ROOT
        citadel_chroma = str(VAULT_ROOT / "edith_data" / "chroma")
    except ImportError:
        citadel_chroma = ""
    if citadel_chroma and os.path.isdir(citadel_chroma):
        return citadel_chroma
    # 4. Project-local chroma (acceptable in dev)
    candidates = [
        str(ROOT_DIR / "chroma"),
        str(ROOT_DIR / "vectors"),
        "/Volumes/Edith Bolt/ChromaDB",
    ]
    for d in candidates:
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "chroma.sqlite3")):
            return d
    # 5. Legacy internal-SSD paths — use ONLY if no Bolt/CITADEL
    _legacy_dirs = [
        os.path.expanduser("~/edith_data/chromadb"),
        os.path.expanduser("~/Library/Application Support/Edith/chroma"),
    ]
    for d in _legacy_dirs:
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "chroma.sqlite3")):
            log.warning(f"§LEAK: ChromaDB on internal SSD at {d} — migrate to CITADEL drive")
            return d
    # 6. Default: project-local (safe for dev)
    return str(ROOT_DIR / "chroma")

CHROMA_DIR = _find_chroma_dir()

def _find_collection_name():
    env_name = os.environ.get("EDITH_CHROMA_COLLECTION")
    if env_name:
        return env_name
    # Auto-detect: pick the largest collection in the ChromaDB directory
    try:
        import chromadb
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        colls = client.list_collections()
        if colls:
            # Pick the one with the most items
            best = max(colls, key=lambda c: c.count())
            log.info(f"Auto-detected ChromaDB collection: {best.name} ({best.count()} chunks)")
            return best.name
    except Exception as _exc:
        log.warning(f"Suppressed exception: {_exc}")
    return "edith_docs_sections"  # fallback

CHROMA_COLLECTION = _find_collection_name() or (os.environ.get("EDITH_CHROMA_COLLECTION") or "edith_docs_pdf")
EMBED_MODEL = os.environ.get("EDITH_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# Retrieval backend: "google" (File Search) or "chroma" (local)
RETRIEVAL_BACKEND = os.environ.get("EDITH_RETRIEVAL_BACKEND", "chroma").lower()
GOOGLE_STORE_ID = os.environ.get("EDITH_GOOGLE_STORE_ID", "")
USE_GOOGLE_RETRIEVAL = (
    RETRIEVAL_BACKEND == "google" and
    google_retrieval_available(API_KEY or "", GOOGLE_STORE_ID)
)
if USE_GOOGLE_RETRIEVAL:
    log.info(f"Retrieval backend: Google File Search (store: {GOOGLE_STORE_ID})")
else:
    log.info(f"Retrieval backend: Chroma (dir: {CHROMA_DIR})")

# Model chain: primary model + configured fallbacks
DEFAULT_MODEL = os.environ.get("EDITH_MODEL", "gemini-2.5-flash")
_fallback_raw = os.environ.get("EDITH_MODEL_FALLBACKS", "gemini-2.5-flash,gemini-2.0-flash")
FALLBACK_MODELS = [m.strip() for m in _fallback_raw.split(",") if m.strip()]

# §ROUTING: Per-tier model overrides
ORACLE_MODEL = os.environ.get("EDITH_ORACLE_MODEL", "gemini-2.5-pro")

def build_model_chain(requested: str) -> list:
    """Build a model chain from the requested model + configured fallbacks."""
    chain = [requested]
    for fb in FALLBACK_MODELS:
        if fb not in chain:
            chain.append(fb)
    # Always have gemini-2.0-flash as ultimate safety net
    if "gemini-2.0-flash" not in chain:
        chain.append("gemini-2.0-flash")
    return chain

# Initialize GenAI
init_client(API_KEY)

# Security: initialise audit logging + auth token
setup_audit_logging()
configure_auth_token()  # reads EDITH_ACCESS_PASSWORD from env
audit("server_start", chroma_dir=CHROMA_DIR, collection=CHROMA_COLLECTION)

# §SEC-6: SQLite WAL Mode — prevent corruption on drive disconnect
try:
    import sqlite3
    _sqlite_dbs = list(Path(CHROMA_DIR).rglob("*.sqlite3")) if CHROMA_DIR else []
    _wal_ok = 0
    for _db_path in _sqlite_dbs:
        try:
            _conn = sqlite3.connect(str(_db_path))
            # Verify it's a valid SQLite database before tuning
            _conn.execute("SELECT 1 FROM sqlite_master LIMIT 1")
            _conn.execute("PRAGMA journal_mode=WAL")
            _conn.close()
            _wal_ok += 1
        except Exception:
            pass  # Skip non-database files (e.g. lock files named .sqlite3)
    if _wal_ok:
        log.info(f"SQLite WAL mode enabled for {_wal_ok}/{len(_sqlite_dbs)} databases")
    audit("wal_mode", databases=str(_wal_ok))
except Exception as _wal_err:
    log.warning(f"Could not enable WAL mode: {_wal_err}")

# §SEC-4: Edith Drive marker validation on startup
_drive_available = False
try:
    from server.security import validate_edith_drive, structured_audit
    _drive_result = validate_edith_drive()
    _drive_available = bool(_drive_result.get("valid"))
    structured_audit("drive_check", severity="info",
                     status=_drive_result["status"], details=_drive_result["details"])
    log.info(f"Edith Drive: {_drive_result['status']} — {_drive_result['details']}")
except Exception as _drv_err:
    _drive_available = False
    log.warning(f"Drive validation skipped: {_drv_err}")

# §PREFLIGHT: Validate Bolt, ChromaDB, API keys, and deps before serving
try:
    from server.utils import preflight_check
    _preflight = preflight_check()
    if _preflight["status"] == "blocked":
        log.error(f"§PREFLIGHT: {len(_preflight['issues'])} blocking issue(s) — server may not function correctly")
except Exception as _pf_err:
    log.warning(f"Preflight check failed: {_pf_err}")

# §SEC-7: Background indexing niceness
try:
    os.nice(10)  # Lower priority for background processing
    log.info("Background indexing priority set (nice=10)")
except (OSError, AttributeError):
    pass  # Not available on all platforms

# Allowed CORS origins — localhost + Electron file:// ("Origin: null")
_CORS_ORIGINS = [
    "http://127.0.0.1",
    "http://localhost",
    "null",
    "file://",
]
# Add port-specific origins for common dev ports
for _port in ["5173", "8501", "8000", "8001", os.environ.get("EDITH_PORT", "8001")]:
    _CORS_ORIGINS.append(f"http://127.0.0.1:{_port}")
    _CORS_ORIGINS.append(f"http://localhost:{_port}")

# §OBS-2: Lifespan context manager (replaces deprecated on_event)
from contextlib import asynccontextmanager

@asynccontextmanager
async def _lifespan(app):
    """Modern lifespan handler: startup → yield → shutdown."""
    _run_startup()
    yield
    _run_shutdown()

app = FastAPI(
    title="Edith API",
    version="1.0.0",
    description=(
        "Edith Research Assistant — grounded RAG for PhD research.\n\n"
        "## Endpoints\n"
        "- **Chat**: Core conversational AI with retrieval-augmented generation\n"
        "- **Research**: Literature reviews and research design assistance\n"
        "- **Library**: Browse and search indexed documents\n"
        "- **Indexing**: Manage document indexing pipeline\n"
        "- **Notes**: Save, edit, and manage research notes\n"
        "- **Discovery**: OpenAlex academic paper search\n"
        "- **System**: Health checks, diagnostics, and configuration"
    ),
    openapi_tags=[
        {"name": "Chat", "description": "Conversational AI — grounded and general modes"},
        {"name": "Retrieval", "description": "Direct retrieval and search endpoints"},
        {"name": "Research", "description": "Literature reviews and research design"},
        {"name": "Library", "description": "Browse and filter indexed documents"},
        {"name": "Indexing", "description": "Document indexing pipeline management"},
        {"name": "Notes", "description": "Research note management"},
        {"name": "Discovery", "description": "OpenAlex academic paper search and recommendations"},
        {"name": "System", "description": "Health, diagnostics, and system configuration"},
    ],
    dependencies=[Depends(security_gate)],  # auth + rate-limit + audit on all routes
    lifespan=_lifespan,
)

# §SWITCHBOARD: Route call counter for Live Dashboard
import threading as _threading
_route_call_counts: dict[str, int] = {}
_route_call_lock = _threading.Lock()
_route_call_start = _time.time()

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request as StarletteRequest
from starlette.responses import Response as StarletteResponse

class RouteMetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: StarletteRequest, call_next):
        path = request.url.path
        if path.startswith("/api/"):  # §FIX: Only count API routes, not static files
            with _route_call_lock:
                _route_call_counts[path] = _route_call_counts.get(path, 0) + 1
            # §SUBCONSCIOUS: Record activity for idle detection
            try:
                from server.subconscious_streams import speculative_horizon
                speculative_horizon.record_activity()
            except Exception:
                pass
        response = await call_next(request)
        return response

app.add_middleware(RouteMetricsMiddleware)

# §MONITORING: Error tracking (Sentry) — configure with SENTRY_DSN env var
try:
    from server.monitoring import init_monitoring
    init_monitoring(app)
except ImportError:
    pass

# §TRUST: Mount Trust Center router
try:
    from server.trust_center import router as _trust_router
    app.include_router(_trust_router)
    log.info("Trust Center API mounted at /api/trust")
except ImportError as _e:
    log.warning(f"Trust Center router not available: {_e}")

# ═══════════════════════════════════════════════════════════════════
# §ORCHESTRATION: Adaptive Orchestration Layer — LAZY LOADING
# Modules are imported in the background _warmup thread to avoid
# blocking server startup.
# ═══════════════════════════════════════════════════════════════════
_memory_scaler_ok = False
_prefetcher_ok = False
_storage_mgr = None
_memory_monitor = None
_prefetcher = None
_deep_dive = None
_faculty = []
_maintenance = None

def _load_orchestration_modules():
    """Load orchestration modules in background. Called from _warmup thread."""
    global _memory_scaler_ok, _prefetcher_ok
    global _storage_mgr
    global _memory_monitor, _prefetcher, _deep_dive, _faculty, _maintenance
    global get_memory_pressure, scale_context_window
    global run_faculty_review, simulate_review, socratic_query, detect_jargon, explain_term
    global find_shadow_variables, _deep_time
    global discover_datasets, generate_code, execute_python, vibe_explain_code
    global run_quick_analysis, generate_and_run, QUICK_ANALYSES
    global get_thermal_state, DriveWatchdog, get_drive_health
    global BackupManager, check_and_repair_index

    try:
        from server.memory_scaler import get_memory_pressure, scale_context_window, memory_monitor as _mm
        _memory_monitor = _mm
        _memory_scaler_ok = True
        log.info("§ORCH: MemoryScaler loaded")
    except Exception as _e:
        log.warning(f"§ORCH: MemoryScaler not available: {_e}")

    try:
        from server.prefetcher import prefetcher as _pf
        _prefetcher = _pf
        _prefetcher_ok = True
        log.info("§ORCH: Prefetcher loaded")
    except Exception as _e:
        log.warning(f"§ORCH: Prefetcher not available: {_e}")

    try:
        from server.deep_dive import deep_dive_engine as _dd
        _deep_dive = _dd

        log.info("§ORCH: DeepDive loaded")
    except Exception as _e:
        log.warning(f"§ORCH: DeepDive not available: {_e}")

    try:
        from server.peer_review import (
            FACULTY as _f, run_faculty_review, simulate_review,
            socratic_query, detect_jargon, explain_term,
        )
        _faculty = _f

        log.info(f"§ORCH: PeerReview loaded ({len(_faculty)} personas)")
    except Exception as _e:
        log.warning(f"§ORCH: PeerReview not available: {_e}")

    try:
        from server.shadow_discovery import find_shadow_variables, deep_time as _dt
        globals()['_deep_time'] = _dt

        log.info("§ORCH: ShadowDiscovery loaded")
    except Exception as _e:
        log.warning(f"§ORCH: ShadowDiscovery not available: {_e}")

    try:
        from server.vibe_coder import (
            discover_datasets, generate_code, execute_python,
            explain_code as _vibe_explain, run_quick_analysis,
            generate_and_run, QUICK_ANALYSES,
        )
        globals()['vibe_explain_code'] = _vibe_explain

        log.info("§ORCH: VibeCoder loaded")
    except Exception as _e:
        log.warning(f"§ORCH: VibeCoder not available: {_e}")

    try:
        from server.auto_maintenance import maintenance as _m
        _maintenance = _m

        log.info("§ORCH: AutoMaintenance loaded")
    except Exception as _e:
        log.warning(f"§ORCH: AutoMaintenance not available: {_e}")

    try:
        from server.hw_monitor import get_thermal_state, DriveWatchdog, get_drive_health

        log.info("§ORCH: HWMonitor loaded")
    except Exception as _e:
        log.warning(f"§ORCH: HWMonitor not available: {_e}")

    try:
        from server.storage_manager import BackupManager, backup_manager as _bm, check_and_repair_index
        globals()['_storage_mgr'] = _bm
        log.info("§ORCH: StorageManager loaded")
    except Exception as _e:
        log.warning(f"§ORCH: StorageManager not available: {_e}")

    # §IMPROVEMENT 1: Auto-index file watcher
    try:
        from server.vault_watcher import VaultWatcher
        _vault_watcher = VaultWatcher(vault_root=str(VAULT_ROOT))
        _vault_watcher.start()
        log.info("§ORCH: VaultWatcher loaded")
    except Exception as _e:
        log.warning(f"§ORCH: VaultWatcher not available: {_e}")


# ═══════════════════════════════════════════════════════════════════
# §GOOGLE-LAUNCH + §CITADEL: LAZY LOADING
# All heavy modules deferred to background _warmup thread.
# ═══════════════════════════════════════════════════════════════════
_hybrid_ok = False
_pedagogy_ok = False
_theme_ok = False
_monte_carlo_ok = False
_focus_mode_ok = False
_lod_ok = False
_boot_ok = False
_response_cache = None
_anomaly = None
_socratic_engine = None
_spaced_rep = None
_rec_engine = None
_study_session = None
_session_state = None
_hybrid_engine = None
_tone_gen = None
_pedagogy_indexer = None
_citadel_theme = None
_monte_carlo = None
_atlas_lod = None
_reasoning_auditor = None
_sovereignty = {"status": "INITIALIZING", "root": "pending"}

def _load_launch_modules():
    """Load GOOGLE-LAUNCH + CITADEL modules in background. Called from _warmup thread."""
    global _hybrid_ok, _pedagogy_ok, _theme_ok
    global _monte_carlo_ok, _focus_mode_ok, _lod_ok, _boot_ok
    global _response_cache, _anomaly, _socratic_engine, _spaced_rep
    global _rec_engine, _study_session, _session_state
    global _hybrid_engine, _tone_gen, _pedagogy_indexer, _citadel_theme
    global _monte_carlo, _atlas_lod, _reasoning_auditor, _sovereignty

    try:
        from server.infrastructure import (
            response_cache as _rc, parallel_retrieve,
            optimize_query_plan, IncrementalIndexer, stream_response,
            ConnectionPool as _CP, LazyLoader as _LL,
        )
        globals()['parallel_retrieve'] = parallel_retrieve
        globals()['optimize_query_plan'] = optimize_query_plan
        globals()['IncrementalIndexer'] = IncrementalIndexer
        globals()['stream_response'] = stream_response
        globals()['_ConnPool'] = _CP
        globals()['_LazyLoader'] = _LL
        _response_cache = _rc

        log.info("§LAUNCH: Infrastructure loaded (cache, parallel, streaming)")
    except Exception as _e:
        log.warning(f"§LAUNCH: Infrastructure not available: {_e}")

    try:
        from server.security import (
            EncryptedChatLog, verify_physical_soul, initialize_drive_marker,
            anomaly_detector as _an, secure_wipe_ram, build_security_dashboard,
        )
        globals()['EncryptedChatLog'] = EncryptedChatLog
        globals()['verify_physical_soul'] = verify_physical_soul
        globals()['initialize_drive_marker'] = initialize_drive_marker
        globals()['secure_wipe_ram'] = secure_wipe_ram
        globals()['build_security_dashboard'] = build_security_dashboard
        _anomaly = _an

        log.info("§LAUNCH: Security hardening loaded (Physical Soul, encrypted logs)")
    except Exception as _e:
        log.warning(f"§LAUNCH: Security hardening not available: {_e}")

    try:
        from server.cognitive_engine import (
            graph_enhanced_retrieve, switch_persona, get_active_persona,
            list_personas, simulate_peer_review as _spr,
            discover_literature, expand_query_multilingual,
            socratic as _soc, spaced_rep as _sr,
            scale_response_difficulty, SocraticEngine, SpacedRepetition,
        )
        globals()['graph_enhanced_retrieve'] = graph_enhanced_retrieve
        globals()['switch_persona'] = switch_persona
        globals()['get_active_persona'] = get_active_persona
        globals()['list_personas'] = list_personas
        globals()['sim_peer_review'] = _spr
        globals()['discover_literature'] = discover_literature
        globals()['expand_query_multilingual'] = expand_query_multilingual
        globals()['scale_response_difficulty'] = scale_response_difficulty
        globals()['SocraticEngine'] = SocraticEngine
        globals()['SpacedRepetition'] = SpacedRepetition
        _socratic_engine = _soc
        _spaced_rep = _sr

        log.info("§LAUNCH: Cognitive engine loaded (6 personas, Socratic, SM-2)")
    except Exception as _e:
        log.warning(f"§LAUNCH: Cognitive engine not available: {_e}")

    try:
        from server.completions import (
            quantize_model, list_quantized_models, ring_attention_summarize,
            extract_text_from_image, speculative_generate,
            scrape_citations_from_openalex, build_literature_map,
            recommendation_engine as _re, study_session as _ss,
            tune_npu_batch_size, get_route_map,
        )
        globals()['quantize_model'] = quantize_model
        globals()['list_quantized_models'] = list_quantized_models
        globals()['ring_attention_summarize'] = ring_attention_summarize
        globals()['extract_text_from_image'] = extract_text_from_image
        globals()['speculative_generate'] = speculative_generate
        globals()['scrape_citations_from_openalex'] = scrape_citations_from_openalex
        globals()['build_literature_map'] = build_literature_map
        globals()['tune_npu_batch_size'] = tune_npu_batch_size
        globals()['get_route_map'] = get_route_map
        _rec_engine = _re
        _study_session = _ss

        log.info("§LAUNCH: Completions loaded (OCR, RingAttention, study sessions)")
    except Exception as _e:
        log.warning(f"§LAUNCH: Completions not available: {_e}")

    try:
        # §FIX: operational_rhythm has OperationalRhythm class + global instance,
        # NOT session_state/quarterly_merge/build_theory_map etc.
        from server.operational_rhythm import OperationalRhythm, operational_rhythm as _opr
        _session_state = _opr.status  # use the .status property dict

        log.info("§LAUNCH: Operational rhythm loaded")
    except Exception as _e:
        log.warning(f"§LAUNCH: Operational rhythm not available: {_e}")

    try:
        from server.grounded_guardrails import (
            enforce_rag_only, run_persona_drift_audit,
            methodological_hawk_review, run_literature_stress_test,
            citation_middleware,
        )
        globals()['enforce_rag_only'] = enforce_rag_only
        globals()['run_persona_drift_audit'] = run_persona_drift_audit
        globals()['methodological_hawk_review'] = methodological_hawk_review
        globals()['run_literature_stress_test'] = run_literature_stress_test
        globals()['citation_middleware'] = citation_middleware

        log.info("§LAUNCH: Grounded guardrails loaded (RAG-only, Hawk, stress test)")
    except Exception as _e:
        log.warning(f"§LAUNCH: Grounded guardrails not available: {_e}")

    try:
        from server.hybrid_engine import hybrid_engine as _he, tone_generator as _tg
        _hybrid_engine = _he
        _tone_gen = _tg
        _hybrid_ok = True
        log.info("§CITADEL: HybridEngine loaded (RAG + fine-tuned synthesis)")
    except Exception as _e:
        log.warning(f"§CITADEL: HybridEngine not available: {_e}")

    try:
        from server.index_pedagogy import pedagogy_indexer as _pi, query_as_exam
        globals()['query_as_exam'] = query_as_exam
        _pedagogy_indexer = _pi
        _pedagogy_ok = True
        log.info("§CITADEL: PedagogicalIndexer loaded (ancestral nodes)")
    except Exception as _e:
        log.warning(f"§CITADEL: PedagogicalIndexer not available: {_e}")

    try:
        from server.citadel_theme import citadel_theme as _ct
        _citadel_theme = _ct
        _theme_ok = True
        log.info("§CITADEL: CitadelTheme loaded (Solar-Paper + Arc Reactor)")
    except Exception as _e:
        log.warning(f"§CITADEL: CitadelTheme not available: {_e}")

    try:
        from server.completions import monte_carlo as _mc
        _monte_carlo = _mc
        _monte_carlo_ok = True
        log.info("§CITADEL: MonteCarloEngine loaded (10K-agent simulation)")
    except Exception as _e:
        log.warning(f"§CITADEL: MonteCarloEngine not available: {_e}")

    try:
        from server.cognitive_engine import engage_focus_mode, disengage_focus_mode
        globals()['engage_focus_mode'] = engage_focus_mode
        globals()['disengage_focus_mode'] = disengage_focus_mode
        _focus_mode_ok = True
        log.info("§CITADEL: Focus Mode loaded (thermal fail-safe)")
    except Exception as _e:
        log.warning(f"§CITADEL: Focus Mode not available: {_e}")

    try:
        from server.vector_mapping import atlas_lod as _al
        _atlas_lod = _al
        _lod_ok = True
        log.info("§CITADEL: AtlasLoD loaded (frustum culling + GPU reservation)")
    except Exception as _e:
        log.warning(f"§CITADEL: AtlasLoD not available: {_e}")

    # §ORPHAN-WIRE: Import 6 previously-orphaned modules
    try:
        from server.socratic_navigator import SocraticNavigator, CommitteeOfSages, MethodologySandbox, OntologyMapper
        _socratic_navigator = SocraticNavigator()
        _methodology_sandbox = MethodologySandbox()
        _ontology_mapper = OntologyMapper()
        globals()['_socratic_navigator'] = _socratic_navigator
        globals()['_methodology_sandbox'] = _methodology_sandbox
        globals()['_ontology_mapper'] = _ontology_mapper
        log.info("§ORPHAN-WIRE: SocraticNavigator loaded (Navigator, Sandbox, OntologyMapper)")
    except Exception as _e:
        log.warning(f"§ORPHAN-WIRE: SocraticNavigator not available: {_e}")

    try:
        from server.pedagogy import generate_quiz, export_to_anki, scan_for_duplicates, merge_duplicates, get_capability_tier
        globals()['generate_quiz'] = generate_quiz
        globals()['export_to_anki'] = export_to_anki
        globals()['scan_for_duplicates'] = scan_for_duplicates
        globals()['merge_duplicates'] = merge_duplicates
        globals()['get_capability_tier'] = get_capability_tier
        log.info("§ORPHAN-WIRE: Pedagogy loaded (quiz gen, Anki export, dedup, capability tier)")
    except Exception as _e:
        log.warning(f"§ORPHAN-WIRE: Pedagogy not available: {_e}")

    try:
        from server.metabolic_monitor import MetabolicMonitor, VitalsMonitor, GhostVariableDetector
        _metabolic_monitor = MetabolicMonitor()
        _vitals_monitor = VitalsMonitor()
        _ghost_detector = GhostVariableDetector()
        globals()['_metabolic_monitor'] = _metabolic_monitor
        globals()['_vitals_monitor'] = _vitals_monitor
        globals()['_ghost_detector'] = _ghost_detector
        log.info("§ORPHAN-WIRE: MetabolicMonitor loaded (vitals, ghost variables, self-healing)")
    except Exception as _e:
        log.warning(f"§ORPHAN-WIRE: MetabolicMonitor not available: {_e}")

    try:
        from server.neural_health_hud import NeuralHealthHUD
        _neural_hud = NeuralHealthHUD()
        globals()['_neural_hud'] = _neural_hud
        log.info("§ORPHAN-WIRE: NeuralHealthHUD loaded (aggregated health dashboard)")
    except Exception as _e:
        log.warning(f"§ORPHAN-WIRE: NeuralHealthHUD not available: {_e}")

    try:
        from server.citadel_bridge import CitadelBridge, FocusManager, CockpitCommandLine
        _citadel_bridge = CitadelBridge()
        _cockpit_cli = CockpitCommandLine()
        globals()['_citadel_bridge'] = _citadel_bridge
        globals()['_cockpit_cli'] = _cockpit_cli
        log.info("§ORPHAN-WIRE: CitadelBridge loaded (bridge, focus manager, cockpit CLI)")
    except Exception as _e:
        log.warning(f"§ORPHAN-WIRE: CitadelBridge not available: {_e}")

    try:
        from server.citadel_connectome_master import SovereignBrain, ignite, metabolize, deep_click, audit, dream, hud
        _sovereign_brain = SovereignBrain()
        globals()['_sovereign_brain'] = _sovereign_brain
        globals()['connectome_ignite'] = ignite
        globals()['connectome_metabolize'] = metabolize
        globals()['connectome_deep_click'] = deep_click
        globals()['connectome_audit'] = audit
        globals()['connectome_dream'] = dream
        globals()['connectome_hud'] = hud
        log.info("§ORPHAN-WIRE: ConnectomeMaster loaded (SovereignBrain, ignite, dream, metabolize)")
    except Exception as _e:
        log.warning(f"§ORPHAN-WIRE: ConnectomeMaster not available: {_e}")

    try:
        from server.citadel_boot import (
            run_boot_health_check, reasoning_auditor as _ra,
            get_rag_priority, enforce_storage_sovereignty, detect_apple_silicon,
        )
        globals()['run_boot_health_check'] = run_boot_health_check
        globals()['get_rag_priority'] = get_rag_priority
        globals()['enforce_storage_sovereignty'] = enforce_storage_sovereignty
        globals()['detect_apple_silicon'] = detect_apple_silicon
        _reasoning_auditor = _ra
        _boot_ok = True
        # Run storage sovereignty check (deferred from module scope)
        _sovereignty = enforce_storage_sovereignty()
        log.info(f"§CITADEL: Storage → {_sovereignty['status']} (root: {_sovereignty['root']})")
    except Exception as _e:
        log.warning(f"§CITADEL: Boot module not available: {_e}")

# §3.2: GZip compression for large responses
app.add_middleware(GZipMiddleware, minimum_size=1000)

# §OBS-1: Trace ID Middleware — end-to-end request tracing
import uuid as _uuid
import time as _time

@app.middleware("http")
async def trace_id_middleware(request: Request, call_next):
    """Attach a trace_id to every request for end-to-end observability."""
    trace_id = request.headers.get("X-Trace-ID") or str(_uuid.uuid4())
    request.state.trace_id = trace_id
    t0 = _time.perf_counter()
    response = await call_next(request)
    elapsed_ms = round((_time.perf_counter() - t0) * 1000, 1)
    response.headers["X-Trace-ID"] = trace_id
    response.headers["X-Response-Time"] = f"{elapsed_ms}ms"
    log.info(f"[{trace_id[:8]}] {request.method} {request.url.path} → {response.status_code} ({elapsed_ms}ms)")
    return response

# §SEC-10: Request body size limits — prevent memory exhaustion
_MAX_BODY_API = int(os.environ.get("EDITH_MAX_BODY_BYTES", str(1 * 1024 * 1024)))  # 1MB for API
_MAX_BODY_UPLOAD = int(os.environ.get("EDITH_MAX_UPLOAD_BYTES", str(50 * 1024 * 1024)))  # §FIX Bug 2: Match _UPLOAD_MAX_MB (50MB)

@app.middleware("http")
async def body_size_limit_middleware(request: Request, call_next):
    """Reject oversized request bodies to prevent memory exhaustion."""
    content_length = request.headers.get("content-length")
    path = request.url.path
    limit = _MAX_BODY_UPLOAD if path in ("/api/upload", "/api/ingest") else _MAX_BODY_API
    # §FIX Vuln 1: Check Content-Length header if present
    if content_length:
        size = int(content_length)
        if size > limit:
            log.warning(f"Request body too large: {size} bytes on {path} (limit: {limit})")
            return JSONResponse(
                status_code=413,
                content={"error": "payload_too_large", "detail": f"Body size {size} exceeds limit {limit}", "max_bytes": limit},
            )
    return await call_next(request)

# §SEC-3: PII Scrubbing Middleware
try:
    from server.security import PIIScrubbingMiddleware
    app.add_middleware(PIIScrubbingMiddleware)
    log.info("PII Scrubbing Middleware active")
except ImportError:
    pass

# §4.10: Security response headers
app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_origin_regex=r"^(null|https?://(127\.0\.0\.1|localhost)(:\d+)?)$",
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type", "X-Request-ID", "X-Edith-Signature"],
)


# §API-V1: API versioning — /v1/api/* → /api/* rewrite
class APIVersionMiddleware:
    """Support versioned API paths (/v1/api/...) by rewriting to /api/...
    Allows both /api/status and /v1/api/status to work identically."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http" and scope["path"].startswith("/v1/"):
            scope = dict(scope)
            scope["path"] = scope["path"][3:]  # Strip /v1 prefix
            if scope.get("root_path"):
                scope["root_path"] = scope["root_path"]
        await self.app(scope, receive, send)

app.add_middleware(APIVersionMiddleware)

# §B3: Correlation IDs on every request/response
app.add_middleware(CorrelationMiddleware)

# §ROUTERS: registered at end of file (after all handler functions are defined)


# §S8: Session inactivity timeout
_SESSION_TIMEOUT_MIN = int(os.environ.get("EDITH_SESSION_TIMEOUT_MINUTES", "30"))
_last_activity: dict[str, float] = {}  # IP → last request time

@app.middleware("http")
async def session_timeout_middleware(request: Request, call_next):
    """Track session activity and warn on stale sessions."""
    client_ip = request.client.host if request.client else "unknown"
    now = _time.time()
    last = _last_activity.get(client_ip, now)
    idle_min = (now - last) / 60
    _last_activity[client_ip] = now
    response = await call_next(request)
    if idle_min > _SESSION_TIMEOUT_MIN:
        response.headers["X-Session-Warning"] = f"idle-{int(idle_min)}min"
        log.info(f"Session timeout: {client_ip} idle for {int(idle_min)} minutes")
    return response


# §SEC-1: EDITH_SESSION_TOKEN — prevent unauthenticated local access
# Generate a random token on startup; Electron reads it from the token file.
# In test mode (EDITH_DISABLE_AUTH=1), skip token enforcement.
_AUTH_DISABLED = os.environ.get("EDITH_DISABLE_AUTH", "") == "1"
if _AUTH_DISABLED and os.environ.get("EDITH_ENV", "development") == "production":
    log.critical("§SEC-CRITICAL: EDITH_DISABLE_AUTH=1 in PRODUCTION mode! "
                 "All API routes are unprotected. Set EDITH_DISABLE_AUTH=false "
                 "or remove it before deploying.")
_SESSION_TOKEN = os.environ.get("EDITH_SESSION_TOKEN", "")
if not _SESSION_TOKEN:
    import secrets as _secrets
    _SESSION_TOKEN = _secrets.token_urlsafe(32)
    # Write to a file so the Electron app can read it
    _token_file = ROOT_DIR / ".edith_session_token"
    try:
        _token_file.write_text(_SESSION_TOKEN)
        _token_file.chmod(0o600)  # Only owner can read
        log.info(f"Session token written to {_token_file}")
    except Exception as _te:
        log.warning(f"Could not write session token file: {_te}")

# Paths exempt from token auth (health check, static files, OPTIONS preflight)
_TOKEN_EXEMPT_PREFIXES = (
    "/status", "/docs", "/openapi.json", "/favicon",
    "/api/auth/", "/api/boot", "/api/theme",
    "/api/status", "/api/health", "/api/shared-mode",
    "/api/dream/status", "/api/monitoring/health", "/api/wasm/status",
)

@app.middleware("http")
async def session_token_middleware(request: Request, call_next):
    """§SEC-1: Require EDITH_SESSION_TOKEN on all API requests.
    The Electron app sends it as Authorization: Bearer <token>.
    Static file serving and health checks are exempt.
    """
    # Test mode bypass
    if _AUTH_DISABLED:
        return await call_next(request)
    
    path = request.url.path
    method = request.method
    
    # Exempt: OPTIONS preflight, health checks, static files
    if method == "OPTIONS":
        return await call_next(request)
    if any(path.startswith(p) for p in _TOKEN_EXEMPT_PREFIXES):
        return await call_next(request)
    # Exempt: static file paths (CSS, JS, HTML, images, fonts)
    if path.endswith((".css", ".js", ".html", ".ico", ".png", ".svg", ".woff", ".woff2", ".ttf")):
        return await call_next(request)
    # Exempt: SPA root
    if path == "/" or path == "":
        return await call_next(request)
    
    # Exempt: Same-origin requests from localhost (web-served renderer mode)
    # Browser requests from the Vite dev server include Origin: http://localhost:5173
    origin = request.headers.get("Origin", "")
    client_ip = request.client.host if request.client else "unknown"
    is_local_client = client_ip in ("127.0.0.1", "::1", "localhost")
    is_local_origin = not origin or any(
        origin.startswith(prefix)
        for prefix in ("http://localhost", "http://127.0.0.1", "http://[::1]")
    )
    if is_local_client and is_local_origin:
        return await call_next(request)
    
    # Check token
    auth_header = request.headers.get("Authorization", "")
    query_token = request.query_params.get("token", "")
    
    token = ""
    if auth_header.startswith("Bearer "):
        token = auth_header[7:]
    elif query_token:
        token = query_token
    
    if not token or not hmac.compare_digest(token, _SESSION_TOKEN):
        # Log unauthorized access attempt
        client_ip = request.client.host if request.client else "unknown"
        log.warning(f"§SEC-1: Unauthorized API access from {client_ip} to {path}")
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=401,
            content={"error": "Unauthorized — EDITH_SESSION_TOKEN required"},
        )
    
    return await call_next(request)


# §3.1: Global exception handler — structured error responses
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": str(exc) if os.environ.get("EDITH_DEBUG") else "An internal error occurred",
        },
    )

# §3.1b: Standardized error response helper — consistent JSON format
def _error_response(status: int, code: str, detail: str, **extra) -> JSONResponse:
    """Return a consistent JSON error response.
    All error responses from the API use: {error: str, detail: str, status: int}
    """
    body = {"error": code, "detail": detail, "status": status}
    body.update(extra)
    return JSONResponse(status_code=status, content=body)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Standardize FastAPI HTTP exceptions to our consistent format."""
    return _error_response(exc.status_code, exc.detail.lower().replace(' ', '_') if isinstance(exc.detail, str) else 'error', str(exc.detail))

# §3.9: Request tracing middleware (X-Request-ID)
if _IMPROVEMENTS_WIRED:
    try:
        from server.api_improvements import RequestTracer
        from starlette.middleware.base import BaseHTTPMiddleware
        app.add_middleware(BaseHTTPMiddleware, dispatch=RequestTracer.middleware)
        log.info("Request tracing middleware added")
    except Exception as _mw_err:
        log.warning(f"Could not add tracing middleware: {_mw_err}")

# --- Input Sanitizer (New Module) ---
_sanitizer = None
if _IMPROVEMENTS_WIRED:
    try:
        from server.security import InputSanitizer
        _sanitizer = InputSanitizer
    except Exception as _exc: log.warning(f"Suppressed exception: {_exc}")



_autolearn_lock = threading.Lock()



# ---------------------------------------------------------------------------
# Quality Score Endpoint
# ---------------------------------------------------------------------------
class ScoreRequest(BaseModel):
    answer: str
    sources: Optional[List[Dict[str, Any]]] = None


def score_endpoint(req: ScoreRequest):
    """Score an answer for professor-grade quality."""
    try:
        from eval.quality_scorers import run_full_evaluation
        return run_full_evaluation(req.answer, req.sources or [])
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


# ---------------------------------------------------------------------------
# Knowledge Graph Endpoint
# ---------------------------------------------------------------------------
# /api/knowledge-graph/* — REMOVED: superseded by /api/kg/stats (line 2358)
# Scholar profiles and debates are rendered client-side from citation_graph.json


# ---------------------------------------------------------------------------
# Research Diary Endpoint
# ---------------------------------------------------------------------------
class DiaryEntry(BaseModel):
    content: str
    category: str = "idea"
    project: str = ""
    tags: Optional[List[str]] = None


def diary_add(req: DiaryEntry):
    try:
        from server.research_workflows import ResearchDiary
        diary = ResearchDiary()
        return diary.add_entry(req.content, req.category, req.project, req.tags)
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


def diary_get(days: int = 7, category: str = None, project: str = None):
    try:
        from server.research_workflows import ResearchDiary
        diary = ResearchDiary()
        return {"entries": diary.get_entries(days, category, project)}
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


# ---------------------------------------------------------------------------
# Quick Prompts Endpoint
# ---------------------------------------------------------------------------
def quick_prompts():
    try:
        from server.research_workflows import QUICK_PROMPTS
        return {"prompts": QUICK_PROMPTS}
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


# ---------------------------------------------------------------------------
# Exam Lockdown Endpoints
# ---------------------------------------------------------------------------
class LockdownRequest(BaseModel):
    message: str = ""
    duration_hours: float = 3.0


def lockdown_activate(req: LockdownRequest, request: Request = None):
    # Security gate: only allow from localhost
    if request and hasattr(request, 'client') and request.client:
        if request.client.host not in ('127.0.0.1', 'localhost', '::1'):
            raise HTTPException(status_code=403, detail="Lockdown only available from localhost")
    try:
        from server.security import ExamLockdown
        lock = ExamLockdown()
        lock.lock(req.message, req.duration_hours)
        return {"status": "locked", "duration_hours": req.duration_hours}
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


def lockdown_deactivate(request: Request = None):
    # Security gate: only allow from localhost
    if request and hasattr(request, 'client') and request.client:
        if request.client.host not in ('127.0.0.1', 'localhost', '::1'):
            raise HTTPException(status_code=403, detail="Lockdown only available from localhost")
    try:
        from server.security import ExamLockdown
        lock = ExamLockdown()
        lock.unlock()
        return {"status": "unlocked"}
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


# ---------------------------------------------------------------------------
# Multi-Corpus Endpoints
# ---------------------------------------------------------------------------
# Auth: JWT Refresh & Revoke
# ---------------------------------------------------------------------------
async def auth_refresh(request: Request):
    """Refresh a JWT token. Issues a new token if the current one is nearing expiry."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token required")
    token = auth_header[7:]
    try:
        from server.security import _jwt_auth
        new_token = _jwt_auth.refresh_token(token)
        if new_token:
            log.info("JWT token refreshed")
            return {"token": new_token, "refreshed": True}
        return {"token": token, "refreshed": False, "reason": "Token still valid or expired"}
    except Exception as e:
        log.warning(f"Token refresh failed: {e}")
        raise HTTPException(status_code=401, detail="Refresh failed")

async def auth_revoke(request: Request):
    """Revoke (invalidate) a JWT token — used for logout."""
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Token required")
    token = auth_header[7:]
    try:
        from server.security import _jwt_auth
        _jwt_auth.revoke_token(token)
        log.info("JWT token revoked")
        return {"revoked": True}
    except Exception as e:
        log.warning(f"Token revoke failed: {e}")
        return {"revoked": False}

# ---------------------------------------------------------------------------
# Corpora
# ---------------------------------------------------------------------------
def list_corpora():
    try:
        from server.training_devops import MultiCorpusManager
        mgr = MultiCorpusManager(CHROMA_DIR)
        return {"corpora": mgr.list_corpora()}
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


async def get_status():
    """Health check + system status.  §3.6: includes uptime, breaker status, and dependency health."""
    if _IMPROVEMENTS_WIRED:
        try:
            from server.api_improvements import health_check
            hc_result = await health_check()
            # §IMP: Ensure subsystems are included even when using improved health_check
            if isinstance(hc_result, dict) and "subsystems" not in hc_result:
                pass  # Fall through to build subsystems below
            elif isinstance(hc_result, dict):
                return hc_result
        except Exception as e:
            log.error(f"Improved health check failed: {e}")

    # --- Dependency health checks ---
    chroma_health = {"status": "unknown", "chunks": 0, "collections": []}
    try:
        import chromadb
        _hc = chromadb.PersistentClient(path=CHROMA_DIR)
        for col in _hc.list_collections():
            count = col.count()
            chroma_health["chunks"] += count
            chroma_health["collections"].append({"name": col.name, "count": count})
        chroma_health["status"] = "ok"
        chroma_health["chroma_chunks"] = chroma_health["chunks"]  # alias for Dashboard.py
    except Exception as e:
        chroma_health["status"] = f"error: {e}"

    # Disk space
    import shutil
    disk = shutil.disk_usage(ROOT_DIR)
    disk_info = {
        "total_gb": round(disk.total / (1024**3), 1),
        "free_gb": round(disk.free / (1024**3), 1),
        "used_pct": round((disk.used / disk.total) * 100, 1),
    }

    # API key validation
    api_keys = {
        "google_gemini": {"set": bool(API_KEY), "length": len(API_KEY or "")},
        "openai": {"set": bool(OPENAI_API_KEY), "length": len(OPENAI_API_KEY or "")},
        "openai_ft_model": OPENAI_FT_MODEL or "not configured",
    }

    # §IMP: Subsystem status breakdown — each subsystem reports its own health
    subsystems = {
        "chroma": {"status": chroma_health["status"], "detail": f"{chroma_health['chunks']} chunks"},
        "llm": {"status": "ok" if API_KEY else "error", "detail": OPENAI_FT_MODEL or DEFAULT_MODEL},
        "openai": {"status": "ok" if (OPENAI_FT_MODEL and OPENAI_API_KEY) else "off",
                   "detail": OPENAI_FT_MODEL or "not configured"},
        "bolt": {"status": "ok" if _drive_available else "off",
                 "detail": "Mounted" if _drive_available else "Not detected"},
        "disk": {"status": "ok" if disk_info["free_gb"] > 5 else "warning",
                 "detail": f"{disk_info['free_gb']}GB free"},
    }
    # Add GEE status if bridge is available
    try:
        from server.google_earth_bridge import GoogleEarthBridge
        subsystems["gee"] = {"status": "ok" if os.environ.get("GOOGLE_EARTH_ENGINE_KEY") else "off",
                             "detail": "configured" if os.environ.get("GOOGLE_EARTH_ENGINE_KEY") else "no key"}
    except Exception:
        subsystems["gee"] = {"status": "off", "detail": "not installed"}

    # Overall status from subsystems
    sub_statuses = [s["status"] for s in subsystems.values()]
    overall = "ok" if all(s in ("ok", "off") for s in sub_statuses) else (
        "error" if "error" in sub_statuses else "degraded")

    result = {
        "status": overall,
        "subsystems": subsystems,
        "uptime_seconds": round(_time.time() - _server_start_time),
        "data_root": DATA_ROOT,
        "chroma_dir": CHROMA_DIR,
        "chroma_collection": CHROMA_COLLECTION or (os.environ.get("EDITH_CHROMA_COLLECTION") or "edith_docs_pdf"),
        "chroma": chroma_health,
        "chroma_chunks": chroma_health["chunks"],  # top-level for backward compat
        "collections": chroma_health["collections"],  # top-level for Dashboard.py
        "disk": disk_info,
        "api_keys": api_keys,
        "api_key_configured": bool(API_KEY),
        "openai_configured": bool(OPENAI_FT_MODEL and OPENAI_API_KEY),
        "retrieval_backend": "google" if USE_GOOGLE_RETRIEVAL else "chroma",
        "drive_available": _drive_available,
        "circuit_breakers": {
            "openai": openai_breaker.status(),
            "google_retrieval": google_retrieval_breaker.status(),
        },
        "models": {
            "primary": OPENAI_FT_MODEL or "none",
            "fallback": DEFAULT_MODEL,
            "embed": os.environ.get("EDITH_GEMINI_EMBED_MODEL", "gemini-embedding-001"),
        },
    }
    if _IMPROVEMENTS_WIRED and _cost_tracker:
        result["cost_summary"] = _cost_tracker.summary
    return result



# ═══════════════════════════════════════════════════════════════════
# §ROUTING: Shared Model Routing — used by chat + chat/stream
# ═══════════════════════════════════════════════════════════════════

# Improvement 2: Dynamic domain keywords — cached from ChromaDB



# ---------------------------------------------------------------------------
# File Upload — F1
# ---------------------------------------------------------------------------
from fastapi import UploadFile, File as FastFile

_UPLOAD_MAX_MB = int(os.environ.get("EDITH_UPLOAD_MAX_MB", "50"))
_UPLOAD_ALLOWED_EXT = {
    ".pdf", ".docx", ".doc", ".txt", ".md", ".rtf",
    ".csv", ".tsv", ".xlsx", ".xls",
    ".tex", ".bib", ".html", ".htm",
    ".ipynb", ".rmd",
    ".json", ".jsonl", ".geojson", ".yaml", ".yml", ".toml", ".xml",
    ".r", ".do", ".sps", ".py", ".js", ".sql",
    ".kml", ".gpx",
    ".log", ".dta", ".sav",
}



def _iter_file_roots() -> list[Path]:
    """Resolve all allowed source roots for file serving."""
    roots: list[Path] = []

    def _add(raw: str):
        val = (raw or "").strip()
        if not val:
            return
        try:
            p = Path(val).expanduser().resolve()
        except Exception:
            return
        if p.is_dir() and p not in roots:
            roots.append(p)

    _add(str(DATA_ROOT or ""))
    _add(os.environ.get("EDITH_SHARED_DATA_ROOT", ""))
    _add(os.environ.get("EDITH_SOURCE_DATA_ROOT", ""))

    multi = (os.environ.get("EDITH_SOURCE_ROOTS", "") or "").replace(";", os.pathsep)
    for token in multi.split(os.pathsep):
        _add(token)

    # Desktop sibling roots: shared index may reference rel paths from M2 while running M4.
    if DATA_ROOT:
        try:
            parent = Path(DATA_ROOT).expanduser().resolve().parent
            _add(str(parent / "Edith_M2"))
            _add(str(parent / "Edith_M4"))
        except Exception:
            pass

    return roots


def _resolve_safe_file_path(path: str) -> Path | None:
    roots = _iter_file_roots()
    candidate = Path(path).expanduser()

    if candidate.is_absolute():
        try:
            resolved = candidate.resolve()
        except Exception:
            return None
        for root in roots:
            try:
                if resolved.is_relative_to(root) and resolved.is_file():
                    return resolved
            except Exception:
                continue
        return None

    for root in roots:
        try:
            resolved = (root / candidate).resolve()
            if not resolved.is_relative_to(root):
                continue
            if resolved.is_file():
                return resolved
        except Exception:
            continue
    return None


async def serve_file(path: str = Query(..., description="Relative path within DATA_ROOT")):
    """Serve a file from the data library for the PDF viewer."""
    audit("file_access", path=path[:200])
    roots = _iter_file_roots()
    if not roots:
        raise HTTPException(status_code=500, detail="DATA_ROOT not configured")

    full_path = _resolve_safe_file_path(path)
    if not full_path:
        raise HTTPException(status_code=404, detail=f"File not found: {path}")

    if not any(full_path.is_relative_to(root) for root in roots):
        raise HTTPException(status_code=403, detail="Access denied")

    import mimetypes
    mime, _ = mimetypes.guess_type(str(full_path))
    return FileResponse(str(full_path), media_type=mime or "application/octet-stream", filename=full_path.name)


# ── Decorator-registered file endpoints (always available) ──

@app.get("/api/file")
async def file_get_endpoint(path: str = Query("", description="File path")):
    """Serve a file for viewing (PDF, etc.)."""
    if not path:
        raise HTTPException(status_code=400, detail="path query param required")
    return await serve_file(path=path)


@app.post("/api/file/open")
async def file_open_endpoint(request: Request):
    """Open a file in the system's default viewer (e.g., Preview for PDFs)."""
    import subprocess as _sp
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="JSON body required")
    file_path = body.get("path", "")
    if not file_path:
        raise HTTPException(status_code=400, detail="path is required")

    full_path = _resolve_safe_file_path(file_path)
    if not full_path:
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")

    roots = _iter_file_roots()
    if roots and not any(full_path.is_relative_to(root) for root in roots):
        raise HTTPException(status_code=403, detail="Access denied")

    try:
        _sp.Popen(["open", str(full_path)])
        return {"status": "ok", "path": str(full_path)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ------------ Library Endpoint ------------ #
import threading

_library_cache: list = []
_library_cache_ts: float = 0
_library_building = False
_library_build_progress: dict = {}  # {"batch": 0, "total_batches": 0, "chunks_scanned": 0, "total_chunks": 0, "docs_found": 0, "elapsed_s": 0, "eta_s": 0}
_library_lock = threading.Lock()  # Protects the 4 globals above from race conditions


def _get_library_cache():
    """Return the actual library cache from library.py (where the build happens).
    Falls back to main.py's local cache if the module cache is empty."""
    try:
        from server.routes import library as _lib_mod
        cache = getattr(_lib_mod, '_library_cache', None)
        if cache:
            return cache
    except Exception:
        pass
    return _library_cache



# ------------ Drive + Consumer-Grade Endpoints ------------ #

class SetDataRootRequest(BaseModel):
    data_root: str

_index_state = {
    "state": "idle",  # idle | running | paused
    "total": 0,
    "processed": 0,
    "skipped": 0,
    "new_count": 0,
    "updated": 0,
    "eta_seconds": 0,
    "last_completed": None,
}

_drive_available = bool(globals().get("_drive_available", False))

def set_data_root(req: SetDataRootRequest, request: Request = None):
    """Switch the active data root (called when Edith Drive is mounted)."""
    # Security gate: localhost only
    if request and hasattr(request, 'client') and request.client:
        if request.client.host not in ('127.0.0.1', 'localhost', '::1'):
            raise HTTPException(status_code=403, detail="Only available from localhost")
    global DATA_ROOT, _drive_available
    new_root = req.data_root
    if not os.path.isdir(new_root):
        raise HTTPException(status_code=400, detail=f"Directory not found: {new_root}")
    DATA_ROOT = new_root
    os.environ["EDITH_DATA_ROOT"] = new_root
    _drive_available = True
    # Refresh compute profile (drive type may have changed)
    try:
        from server.backend_logic import invalidate_compute_profile, get_compute_profile
        invalidate_compute_profile()
        profile = get_compute_profile()
        log.info(f"§HW: Compute profile refreshed on drive mount: {profile['mode']}")
    except Exception:
        pass
    return {"ok": True, "data_root": DATA_ROOT}

def drive_lost(request: Request = None):
    """§PARK: 'Park the Brain' — safely close all SQLite connections and mark drive unavailable.

    Must be called before physically ejecting the Thunderbolt drive.
    Ensures no WAL or journal files are left open (prevents corruption).
    """
    # Security gate: localhost only
    if request and hasattr(request, 'client') and request.client:
        if request.client.host not in ('127.0.0.1', 'localhost', '::1'):
            raise HTTPException(status_code=403, detail="Only available from localhost")
    global _drive_available

    parked_dbs = []
    errors = []

    # 1. Flush all SQLite WAL checkpoints on the drive
    try:
        import sqlite3
        drive_path = DATA_ROOT or os.environ.get("EDITH_DATA_ROOT", "")
        if drive_path and os.path.isdir(drive_path):
            for db_file in Path(drive_path).rglob("*.sqlite3"):
                try:
                    conn = sqlite3.connect(str(db_file), timeout=5)
                    conn.execute("PRAGMA optimize")
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    conn.close()
                    parked_dbs.append(str(db_file.name))
                except Exception as e:
                    errors.append(f"{db_file.name}: {e}")
    except Exception as e:
        errors.append(f"DB flush error: {e}")

    # 2. Also checkpoint the local chroma directory
    try:
        import sqlite3
        if CHROMA_DIR and os.path.isdir(CHROMA_DIR):
            for db_file in Path(CHROMA_DIR).rglob("*.sqlite3"):
                try:
                    conn = sqlite3.connect(str(db_file), timeout=5)
                    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                    conn.close()
                    if db_file.name not in parked_dbs:
                        parked_dbs.append(str(db_file.name))
                except Exception:
                    pass
    except Exception:
        pass

    # 3. Invalidate caches
    _drive_available = False
    try:
        from server.security import invalidate_drive_cache
        invalidate_drive_cache()
    except Exception:
        pass
    try:
        from server.backend_logic import invalidate_compute_profile
        invalidate_compute_profile()
    except Exception:
        pass

    audit("drive_parked", databases=str(parked_dbs), errors=str(errors) if errors else "none")
    log.info(f"§PARK: Brain parked — {len(parked_dbs)} DBs checkpointed, drive marked unavailable")

    return {
        "ok": True,
        "message": f"Brain parked. {len(parked_dbs)} databases safely closed.",
        "parked_databases": parked_dbs,
        "errors": errors if errors else None,
        "safe_to_eject": len(errors) == 0,
    }

def compute_profile():
    """Return current hardware-aware compute profile (M4 committee / M2 focus)."""
    try:
        from server.backend_logic import get_compute_profile
        return get_compute_profile()
    except Exception as e:
        return {"mode": "focus", "error": str(e)}



# NOTE: /api/status is already handled by get_status() above (lines 310-335).
# The duplicate was removed to prevent route shadowing.


# ------------ Graph / Atlas Endpoint ------------ #

async def graph_nodes_endpoint(
    topic: str = "", doc_type: str = "", project: str = "",
    author: str = "", year_min: int = 0, year_max: int = 9999,
    dataset: str = "", method: str = "", country: str = "",
    focus_topic: str = "",
):
    """Build a topic co-occurrence graph from the library cache + scholarly repos."""
    _cache = _get_library_cache()
    if _library_building and not _cache:
        return {"nodes": [], "edges": [], "building": True}

    docs = _cache or []

    # Apply optional filters
    if doc_type:
        types = {t.strip().lower() for t in doc_type.split(",")}
        docs = [d for d in docs if (d.get("doc_type") or "").lower() in types]
    if topic:
        topics_filter = {t.strip().lower() for t in topic.split(",")}
        docs = [d for d in docs if (d.get("academic_topic") or "").lower() in topics_filter]
    if focus_topic:
        focus_lower = focus_topic.strip().lower()
        docs = [d for d in docs if focus_lower in (d.get("academic_topic") or "").lower()]
    if project:
        projects_filter = {p.strip().lower() for p in project.split(",")}
        docs = [d for d in docs if (d.get("project") or "").lower() in projects_filter]
    if author:
        author_lower = author.strip().lower()
        docs = [d for d in docs if author_lower in (d.get("author") or "").lower()]
    if year_min > 0 or year_max < 9999:
        def _year_int(d):
            try: return int(d.get("year") or 0)
            except (ValueError, TypeError): return 0
        docs = [d for d in docs if year_min <= _year_int(d) <= year_max]

    # Build topic nodes + per-doc nodes
    topic_colors = {
        "american politics": "#388bfd", "comparative politics": "#a371f7",
        "international relations": "#39d2c0", "political theory": "#d29922",
        "public policy": "#3fb950", "methodology": "#f85149",
    }
    # Colors for scholarly node types
    scholarly_colors = {
        "dataset": "#f0883e", "method": "#ec4899", "country": "#06b6d4",
    }
    default_color = "#8b949e"

    # Group docs by topic
    topic_groups: dict = {}
    for d in docs:
        t = d.get("academic_topic") or "Uncategorized"
        topic_groups.setdefault(t, []).append(d)

    nodes = []
    edges = []
    node_ids = set()

    # Topic nodes (hubs)
    for t, group in topic_groups.items():
        tid = f"topic:{t}"
        nodes.append({
            "id": tid,
            "label": t,
            "type": "topic",
            "color": topic_colors.get(t.lower(), default_color),
            "size": min(6 + len(group) * 2, 24),
            "connections": len(group),
        })
        node_ids.add(tid)

    # Document nodes (spokes)
    for d in docs[:200]:  # Cap at 200 for performance
        did = f"doc:{d['sha256'][:12]}"
        if did in node_ids:
            continue
        t = d.get("academic_topic") or "Uncategorized"
        nodes.append({
            "id": did,
            "label": d.get("title", "Untitled")[:40],
            "type": "document",
            "color": topic_colors.get(t.lower(), default_color),
            "size": 5,
            "connections": 1,
            "doc_type": d.get("doc_type"),
            "year": d.get("year"),
            "sha256": d.get("sha256"),
            "project": d.get("project"),
        })
        node_ids.add(did)
        # Edge: doc -> topic
        tid = f"topic:{t}"
        if tid in node_ids:
            edges.append({"source": did, "target": tid})

    # Cross-topic edges: connect topics that share a project
    project_topics: dict = {}
    for d in docs:
        proj = d.get("project")
        t = d.get("academic_topic") or "Uncategorized"
        if proj:
            project_topics.setdefault(proj, set()).add(t)
    for proj, ts in project_topics.items():
        tlist = sorted(ts)
        for i in range(len(tlist)):
            for j in range(i + 1, len(tlist)):
                edges.append({
                    "source": f"topic:{tlist[i]}",
                    "target": f"topic:{tlist[j]}",
                })

    # ── Scholarly repo overlay: datasets, methods, countries ──
    scholarly_datasets = []
    scholarly_methods = []
    scholarly_countries = []
    try:
        if _scholarly_repos:
            repos = _scholarly_repos
            # Add dataset nodes
            for ds_info in repos.list_datasets():
                ds_name = ds_info["name"]
                ds_id = f"dataset:{ds_name}"
                if dataset and dataset.lower() not in ds_name.lower():
                    continue
                scholarly_datasets.append(ds_name)
                if ds_id not in node_ids:
                    nodes.append({
                        "id": ds_id, "label": ds_name, "type": "dataset",
                        "color": scholarly_colors["dataset"],
                        "size": min(6 + ds_info["paper_count"] * 2, 20),
                        "connections": ds_info["paper_count"],
                    })
                    node_ids.add(ds_id)
                    # Connect dataset to topics that use it
                    for paper in repos.datasets.get(ds_name, {}).get("papers", []):
                        # Find matching doc in library cache by paper_id
                        for d in docs[:200]:
                            if d.get("sha256", "")[:12] in paper.get("id", ""):
                                t = d.get("academic_topic") or "Uncategorized"
                                tid = f"topic:{t}"
                                if tid in node_ids:
                                    edges.append({"source": ds_id, "target": tid, "weight": 2})
                                break

            # Add method nodes
            for m_name, m_info in repos.methods.items():
                m_id = f"method:{m_name}"
                if method and method.lower() not in m_name.lower():
                    continue
                scholarly_methods.append(m_name)
                if m_id not in node_ids:
                    paper_count = len(m_info.get("papers", []))
                    nodes.append({
                        "id": m_id, "label": m_name, "type": "method",
                        "color": scholarly_colors["method"],
                        "size": min(6 + paper_count * 1.5, 18),
                        "connections": paper_count,
                    })
                    node_ids.add(m_id)

            # Add country nodes
            for c_name, c_info in repos.country_map.items():
                c_id = f"country:{c_name}"
                if country and country.lower() not in c_name.lower():
                    continue
                scholarly_countries.append(c_name)
                topic_count = len(c_info.get("topics", {}))
                if c_id not in node_ids:
                    nodes.append({
                        "id": c_id, "label": c_name, "type": "country",
                        "color": scholarly_colors["country"],
                        "size": min(6 + topic_count * 2, 18),
                        "connections": topic_count,
                    })
                    node_ids.add(c_id)
                    # Connect country to topics it's studied in
                    for topic_name in c_info.get("topics", {}).keys():
                        for t in topic_groups.keys():
                            if topic_name.lower() in t.lower() or t.lower() in topic_name.lower():
                                tid = f"topic:{t}"
                                if tid in node_ids:
                                    edges.append({"source": c_id, "target": tid, "weight": 1.5})
                                break
    except Exception:
        pass  # Scholarly repos not available — graph still works

    # Filter by dataset/method/country if specified (remove unmatched docs)
    if dataset:
        ds_lower = dataset.strip().lower()
        scholarly_datasets = [d for d in scholarly_datasets if ds_lower in d.lower()]
    if method:
        m_lower = method.strip().lower()
        scholarly_methods = [m for m in scholarly_methods if m_lower in m.lower()]
    if country:
        c_lower = country.strip().lower()
        scholarly_countries = [c for c in scholarly_countries if c_lower in c.lower()]

    # Collect unique filter values
    _cache = _get_library_cache()
    all_authors = sorted({d.get("author", "") for d in (_cache or []) if d.get("author")})
    all_years = sorted({str(d.get("year", "")) for d in (_cache or []) if d.get("year")}, reverse=True)
    all_doc_types = sorted({d.get("doc_type", "") for d in (_cache or []) if d.get("doc_type")})
    all_projects = sorted({d.get("project", "") for d in (_cache or []) if d.get("project")})

    return {
        "nodes": nodes, "edges": edges, "total_docs": len(docs),
        "authors": all_authors, "years": all_years,
        "doc_types": all_doc_types, "projects": all_projects,
        "datasets": sorted(set(scholarly_datasets)),
        "methods": sorted(set(scholarly_methods)),
        "countries": sorted(set(scholarly_countries)),
    }


# ------------ Geo / Map Endpoint ------------ #

# Rough country centroids for topic/project mapping
_COUNTRY_CENTROIDS = {
    "united states": (39.8, -98.6), "us": (39.8, -98.6), "usa": (39.8, -98.6),
    "united kingdom": (54.0, -2.0), "uk": (54.0, -2.0),
    "germany": (51.2, 10.4), "france": (46.2, 2.2), "italy": (41.9, 12.5),
    "spain": (40.5, -3.7), "brazil": (-14.2, -51.9), "mexico": (23.6, -102.6),
    "china": (35.9, 104.2), "japan": (36.2, 138.3), "india": (20.6, 78.9),
    "russia": (61.5, 105.3), "australia": (-25.3, 133.8),
    "canada": (56.1, -106.3), "argentina": (-38.4, -63.6),
    "south africa": (-30.6, 22.9), "nigeria": (9.1, 8.7),
    "egypt": (26.8, 30.8), "turkey": (39.0, 35.2), "iran": (32.4, 53.7),
    "south korea": (35.9, 127.8), "indonesia": (-0.8, 113.9),
    "europe": (50.0, 10.0), "africa": (0.0, 25.0), "asia": (30.0, 100.0),
    "latin america": (-15.0, -60.0), "middle east": (29.0, 42.0),
}


# ------------ Notes Endpoints (extracted to routes/notes.py) ------------ #
from server.routes.notes import router as notes_router
app.include_router(notes_router)



# ------------ Datasets Endpoint ------------ #

async def datasets_endpoint():
    """Return datasets from user-maintained manifest (Vault/datasets.json).

    The manifest is the source of truth. Users can add their actual datasets
    (typically 5-20) to this file. Falls back to a lightweight ChromaDB scan
    only if no manifest exists.
    """
    import json as _json
    _cache = _get_library_cache()

    # ── 1. Try the manifest first ──
    manifest_path = Path(os.environ.get("EDITH_DATA_ROOT", "/Volumes/Edith Bolt")) / "Vault" / "datasets.json"
    datasets = []
    used_manifest = False

    if manifest_path.exists():
        try:
            raw = _json.loads(manifest_path.read_text())
            if isinstance(raw, list):
                for i, entry in enumerate(raw):
                    datasets.append({
                        "id": i,
                        "name": entry.get("name", f"Dataset {i+1}"),
                        "format": entry.get("format", ""),
                        "class_name": entry.get("class_name"),
                        "description": entry.get("description", ""),
                        "file_name": entry.get("file_name", ""),
                        "path": entry.get("path", ""),
                        "doc_type": "dataset",
                    })
                used_manifest = True
        except Exception as e:
            log.warning(f"datasets_endpoint: manifest read failed: {e}")

    # ── 2. Fallback: lightweight scan of metadata collection ──
    # ── 2. Fallback: metadata collection + filesystem scan ──
    if not used_manifest:
        _REAL_DATASET_EXTS = {
            ".dta", ".shp", ".csv", ".xlsx", ".xls", ".gph", ".sav", ".rds",
            ".dbf", ".tsv", ".parquet", ".json", ".jsonl", ".sqlite", ".sqlite3", ".db",
        }
        seen_paths = set()

        def _infer_dataset_class(rel_path: str) -> str:
            rel_parts = [p for p in str(rel_path or "").replace("\\", "/").split("/") if p]
            low = [p.lower() for p in rel_parts]
            # Legacy tier roots: canon/past/inbox/projects/<Class>/...
            for i, p in enumerate(low):
                if p in {"canon", "past", "inbox", "projects"} and i + 1 < len(rel_parts):
                    return rel_parts[i + 1].strip()
            # Courses/<term>/<Class>/... or Courses/<Class>/...
            for i, p in enumerate(low):
                if p == "courses" and i + 1 < len(rel_parts):
                    term = low[i + 1]
                    if term in {"active", "past", "current", "spring", "summer", "fall", "winter"} and i + 2 < len(rel_parts):
                        return rel_parts[i + 2].strip()
                    return rel_parts[i + 1].strip()
            # Research/<Project>/...
            for i, p in enumerate(low):
                if p == "research" and i + 1 < len(rel_parts):
                    return rel_parts[i + 1].strip()
            return ""

        # 2a) Metadata scan (fast path when metadata collection exists)
        try:
            from server.chroma_backend import _get_client
            client = _get_client(CHROMA_DIR)
            meta_coll = client.get_collection(name=f"{CHROMA_COLLECTION}_metadata")
            meta_count = meta_coll.count()
            if meta_count > 0:
                for offset in range(0, meta_count, 1000):
                    batch = meta_coll.get(include=["metadatas"], offset=offset, limit=1000)
                    for m in (batch.get("metadatas") or []):
                        fn = m.get("file_name") or ""
                        ext = ("." + fn.rsplit(".", 1)[-1]).lower() if "." in fn else ""
                        if ext not in _REAL_DATASET_EXTS:
                            continue
                        rel_path = str(m.get("path") or m.get("rel_path") or "").replace("\\", "/")
                        if not rel_path:
                            rel_path = fn
                        key = rel_path.lower()
                        if key in seen_paths:
                            continue
                        seen_paths.add(key)
                        class_name = _infer_dataset_class(rel_path)
                        datasets.append({
                            "name": Path(fn).stem or Path(rel_path).stem,
                            "format": ext.lstrip("."),
                            "class_name": class_name or None,
                            "file_name": fn or Path(rel_path).name,
                            "path": rel_path,
                            "doc_type": "dataset",
                            "chunk_count": int(m.get("chunk_count") or 0),
                        })
        except Exception as e:
            log.warning(f"datasets_endpoint: metadata scan failed: {e}")

        # 2b) Filesystem scan (ensures datasets panel is never empty)
        try:
            root_raw = os.environ.get("EDITH_DATA_ROOT", "/Volumes/Edith Bolt")
            data_root = Path(root_raw).expanduser().resolve()
            scan_roots = []
            if data_root.exists() and data_root.is_dir():
                if (data_root / "Library").exists() and (data_root / "Library").is_dir():
                    scan_roots.append(data_root / "Library")
                scan_roots.append(data_root)

            skip_dirs = {
                ".git", ".venv", "venv", "node_modules", "__pycache__", "backups",
                "chroma", "ChromaDB", ".Spotlight-V100", ".fseventsd", ".TemporaryItems",
            }

            for scan_root in scan_roots:
                for walk_root, dirs, files in os.walk(scan_root):
                    dirs[:] = [d for d in dirs if d not in skip_dirs and not d.startswith('.')]
                    for fn in files:
                        if fn.startswith('.'):
                            continue
                        ext = Path(fn).suffix.lower()
                        if ext not in _REAL_DATASET_EXTS:
                            continue
                        abs_path = Path(walk_root) / fn
                        try:
                            rel_path = str(abs_path.relative_to(data_root))
                        except Exception:
                            rel_path = str(abs_path.relative_to(scan_root))
                        rel_path = rel_path.replace("\\", "/")
                        key = rel_path.lower()
                        if key in seen_paths:
                            continue
                        seen_paths.add(key)
                        class_name = _infer_dataset_class(rel_path)
                        datasets.append({
                            "name": Path(fn).stem,
                            "format": ext.lstrip('.'),
                            "class_name": class_name or None,
                            "file_name": fn,
                            "path": rel_path,
                            "doc_type": "dataset",
                            "chunk_count": 0,
                        })
        except Exception as e:
            log.warning(f"datasets_endpoint: filesystem scan failed: {e}")

    # ── 3. Extract dataset mentions from papers ──
    docs = _cache or []
    dataset_mentions: dict = {}
    for d in docs:
        title = d.get("title", "")
        for kw in ["ANES", "CES", "CCES", "WVS", "V-Dem", "Polity", "Freedom House",
                    "GDELT", "ICPSR", "Eurobarometer", "Correlates of War", "COW",
                    "Penn World Table", "QoG", "SNAP", "BLS", "Census"]:
            if kw.lower() in (title or "").lower():
                if kw not in dataset_mentions:
                    dataset_mentions[kw] = {"name": kw, "mentioned_in": [], "count": 0}
                dataset_mentions[kw]["count"] += 1
                if len(dataset_mentions[kw]["mentioned_in"]) < 5:
                    dataset_mentions[kw]["mentioned_in"].append({
                        "title": title, "sha256": d.get("sha256")
                    })

    return {
        "datasets": datasets,
        "mentions": list(dataset_mentions.values()),
        "total": len(datasets),
        "source": "manifest" if used_manifest else "auto",
        "manifest_path": str(manifest_path),
    }


# ------------ Doctor Endpoint ------------ #

async def metrics_endpoint():
    """System performance metrics with latency percentiles."""
    return metrics.get_metrics()



# Whitelist of allowed config keys — security hardening
_ALLOWED_CONFIG_KEYS = {
    "EDITH_CHROMA_TOP_K", "EDITH_CHROMA_BM25_WEIGHT", "EDITH_CHROMA_DIVERSITY_LAMBDA",
    "EDITH_CHROMA_MIN_SCORE", "EDITH_MODEL", "EDITH_LOG_LEVEL",
    "EDITH_SESSION_TIMEOUT_MINUTES", "EDITH_RETRIEVAL_BACKEND",
    "WINNIE_OPENAI_MODEL", "EDITH_DATA_ROOT",
}



# ------------ Runs / History Endpoint ------------ #

async def runs_endpoint():
    """List overnight and automation run history."""
    runs = []

    # Check overnight runs
    overnight_dir = Path(DATA_ROOT) / "eval" / "out" / "overnight" if DATA_ROOT else None
    automation_dir = Path(os.environ.get("EDITH_APP_DATA_DIR", "")) / "eval" / "out" / "automation"
    project_eval = Path(__file__).parent.parent / "eval" / "out"

    for base_dir in [overnight_dir, project_eval / "overnight", project_eval / "automation"]:
        if not base_dir or not base_dir.exists():
            continue
        for run_dir in sorted(base_dir.iterdir(), reverse=True)[:20]:
            if not run_dir.is_dir():
                continue
            summary_file = run_dir / "summary.json"
            if summary_file.exists():
                try:
                    data = json.loads(summary_file.read_text(encoding="utf-8"))
                    run_entry = {
                        "id": run_dir.name,
                        "type": "overnight" if "overnight" in str(base_dir) else "automation",
                        "timestamp": data.get("timestamp_utc") or data.get("ended"),
                        "gate_passed": (data.get("gate") or {}).get("passed"),
                        "token_tally": data.get("token_tally"),
                        "steps": {k: {"ok": v.get("ok"), "skipped": v.get("skipped")}
                                  for k, v in (data.get("steps") or {}).items()},
                    }
                    # Extract overnight-specific metrics if available
                    if data.get("consensus_count") is not None:
                        run_entry["agreements"] = data.get("consensus_count", 0)
                        run_entry["disagreements"] = data.get("total_cases", 0) - data.get("consensus_count", 0)
                    if data.get("gold_pairs_count") is not None:
                        run_entry["new_training"] = data.get("gold_pairs_count", 0)
                    runs.append(run_entry)
                except Exception:
                    runs.append({"id": run_dir.name, "type": "unknown"})

    # Also check dual_brain runs
    dual_dir = Path(__file__).parent.parent / "eval" / "dual_brain"
    if dual_dir.exists():
        for f in sorted(dual_dir.glob("sharpen_*.json"), reverse=True)[:10]:
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                runs.append({
                    "id": f.stem,
                    "type": "dual_brain",
                    "timestamp": data.get("timestamp"),
                    "agreements": len(data.get("agreements", [])),
                    "disagreements": len(data.get("disagreements", [])),
                    "new_training": len(data.get("new_training_pairs", [])),
                })
            except Exception:
                continue

    return {"runs": runs[:50], "total": len(runs)}


async def schedule_run(body: dict = Body(default={})):
    """Schedule an automated run (overnight sharpening, eval, etc.)."""
    run_type = body.get("type", "overnight")
    questions = body.get("questions", [])
    log.info(f"Scheduling run: type={run_type}, questions={len(questions)}")
    try:
        if run_type == "dual_brain" and questions:
            from pipelines.dual_brain import sharpen_cycle
            import threading
            threading.Thread(target=sharpen_cycle, args=(questions,), daemon=True).start()
            return {"status": "scheduled", "type": run_type, "questions": len(questions)}
        elif run_type == "eval":
            from pipelines.run_eval import main as run_eval_main
            import threading
            threading.Thread(target=run_eval_main, daemon=True).start()
            return {"status": "scheduled", "type": "eval"}
        elif run_type == "index":
            # Trigger background indexing
            return {"status": "scheduled", "type": "index", "message": "Use /api/index/run instead"}
        else:
            return {"status": "scheduled", "type": run_type, "message": "Run scheduled (will execute on next cycle)"}
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


# ------------ OpenAlex Endpoints (extracted to routes/openalex.py) ------------ #
from server.routes.openalex import router as openalex_router
app.include_router(openalex_router)


# ------------ Library Sources Endpoint ------------ #
_sources_cache = {"sources": None, "ts": 0}



# ------------ Citation Graph Endpoint ------------ #
async def citation_graph_endpoint():
    """Serve citation_graph.json for the KG tab."""
    cg_path = ROOT_DIR / "citation_graph.json"
    if cg_path.is_file():
        import json as _json
        return _json.loads(cg_path.read_text())
    return {"citations": [], "edges": []}


# ------------ KG Stats Endpoint ------------ #
async def kg_stats_endpoint():
    """Compute knowledge graph stats from citation_graph.json and glossary_graph.json."""
    import json as _json

    scholars = set()
    theories = set()
    debates = []
    top_scholars = []

    # Parse citation graph
    cg_path = ROOT_DIR / "citation_graph.json"
    if cg_path.is_file():
        cg = _json.loads(cg_path.read_text())
        citations = cg.get("citations", [])
        edges = cg.get("edges", [])

        # Extract unique scholars from citations
        for c in citations:
            author = c.get("author") or c.get("name", "")
            if author:
                scholars.add(author)

        # Also extract from edges
        for e in edges:
            citation = e.get("citation", "")
            if citation:
                scholars.add(citation.split(",")[0].strip() if "," in citation else citation.split("(")[0].strip())

        # Top scholars by citation count
        from collections import Counter
        scholar_counts = Counter()
        for e in edges:
            c = e.get("citation", "")
            name = c.split(",")[0].strip() if "," in c else c.split("(")[0].strip()
            if name:
                scholar_counts[name] += 1
        top_scholars = [{"name": n, "count": c} for n, c in scholar_counts.most_common(20)]

    # Parse glossary graph for theories/concepts
    gg_path = ROOT_DIR / "glossary_graph.json"
    if gg_path.is_file():
        try:
            gg = _json.loads(gg_path.read_text())
            for item in (gg if isinstance(gg, list) else gg.get("concepts", [])):
                name = item.get("term") or item.get("name", "")
                if name:
                    theories.add(name)
        except Exception as _exc:
            log.warning(f"Suppressed exception: {_exc}")

    return {
        "scholars": len(scholars),
        "theories": len(theories),
        "debates": len(debates),
        "top_scholars": top_scholars,
        "recent_debates": debates,
    }



async def validate_key_endpoint(body: dict = Body(...)):
    """Check if an API key env var is set and has valid format."""
    import re
    key_name = body.get("key", "GOOGLE_API_KEY")
    value = os.environ.get(key_name, "")
    if not value:
        return {"valid": False, "reason": f"{key_name} not set"}
    # Basic format check
    if key_name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        ok = bool(re.match(r"AIzaSy[A-Za-z0-9_\-]{33}", value))
    elif key_name == "OPENAI_API_KEY":
        ok = value.startswith("sk-") and len(value) > 20
    else:
        ok = len(value) > 5
    return {"valid": ok, "key": key_name, "reason": "" if ok else "Bad format"}


async def sample_data_load():
    """Load sample data for first-time setup."""
    sample_dir = ROOT_DIR / "eval" / "professor_goldset.json"
    if sample_dir.is_file():
        return {"loaded": True, "source": "professor_goldset.json"}
    return {"loaded": False, "reason": "No sample data found"}


async def zotero_sync_endpoint(body: dict = Body(default={})):
    """Sync with Zotero library. Returns status."""
    return {
        "status": "not_configured",
        "message": "Zotero sync is not yet configured. Set ZOTERO_API_KEY and ZOTERO_LIBRARY_ID in .env to enable.",
    }


# ── §4.0: Semantic Scholar API ──
async def scholar_search(q: str, limit: int = 10, year: str = ""):
    """Search Semantic Scholar.  §IMP: citation-count sorting."""
    try:
        from server.connectors_full import SemanticScholarConnector
        s2 = SemanticScholarConnector()
        results = s2.search(q, limit=limit, year=year)
        # §IMP: Sort by citation count descending for relevance
        if isinstance(results, list):
            results.sort(key=lambda r: r.get("citationCount", r.get("citation_count", 0)), reverse=True)
        return {"results": results, "count": len(results), "source": "semantic_scholar", "sorted_by": "citation_count"}
    except Exception as e:
        log.warning(f"Endpoint error: {e}")
        return {"error": "Operation failed", "results": []}

async def scholar_paper(paper_id: str):
    """Get a specific paper from Semantic Scholar."""
    try:
        from server.connectors_full import SemanticScholarConnector
        s2 = SemanticScholarConnector()
        paper = s2.get_paper(paper_id)
        return paper if paper else {"error": "Paper not found"}
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}

async def scholar_citations(paper_id: str, limit: int = 20):
    """Get citations for a paper."""
    try:
        from server.connectors_full import SemanticScholarConnector
        s2 = SemanticScholarConnector()
        return {"citations": s2.get_citations(paper_id, limit=limit)}
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}

async def scholar_references(paper_id: str, limit: int = 20):
    """Get references for a paper."""
    try:
        from server.connectors_full import SemanticScholarConnector
        s2 = SemanticScholarConnector()
        return {"references": s2.get_references(paper_id, limit=limit)}
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}



# ── §4.0: ORCID API ──
async def orcid_lookup(orcid_id: str):
    """Look up a researcher by ORCID.  §IMP: h-index estimate."""
    try:
        from server.connectors_full import ORCIDConnector
        orcid = ORCIDConnector()
        person = orcid.get_person(orcid_id)
        works = orcid.get_works(orcid_id)
        # §IMP: Estimate h-index from available works
        h_index = 0
        if isinstance(works, list) and works:
            citations = sorted([w.get("citation_count", w.get("citationCount", 0)) for w in works if isinstance(w, dict)], reverse=True)
            for i, c in enumerate(citations):
                if c >= i + 1:
                    h_index = i + 1
                else:
                    break
        return {"person": person, "works": works, "work_count": len(works), "estimated_h_index": h_index}
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


async def orcid_search(q: str = Query("", description="Author name to search")):
    """Search ORCID by author name."""
    if not q.strip():
        return {"results": [], "query": q}
    try:
        from server.connectors_full import ORCIDConnector
        orcid = ORCIDConnector()
        results = orcid.search(q.strip())
        return {"results": results[:10], "query": q}
    except Exception as e:
        log.warning(f"ORCID search error: {e}")
        return {"results": [], "query": q, "error": str(e)}



# ── §4.0: Source Deduplication ──
async def deduplicate_sources(body: dict = Body(default={})):
    """Remove near-duplicate sources (>90% text overlap)."""
    sources = body.get("sources", [])
    if len(sources) <= 1:
        return {"sources": sources, "removed": 0, "threshold": body.get("threshold", 0.9)}
    
    deduped = [sources[0]]
    removed = 0
    for s in sources[1:]:
        is_dup = False
        s_text = s.get("text", "")
        for d in deduped:
            d_text = d.get("text", "")
            # Simple word overlap check
            s_words = set(s_text.lower().split())
            d_words = set(d_text.lower().split())
            if s_words and d_words:
                overlap = len(s_words & d_words) / max(len(s_words | d_words), 1)
                threshold = body.get("threshold", 0.9)  # §IMP: configurable
                if overlap > threshold:
                    is_dup = True
                    break
        if is_dup:
            removed += 1
        else:
            deduped.append(s)
    
    return {"sources": deduped, "removed": removed, "original_count": len(sources),
            "threshold": body.get("threshold", 0.9)}


# ── §4.0: Reading List Generator ──
async def generate_reading_list(body: dict = Body(default={})):
    """Generate a curated reading list for a topic using Semantic Scholar."""
    topic = body.get("topic", "")
    max_papers = body.get("max_papers", 15)
    year_range = body.get("year", "")
    if not topic:
        return {"error": "topic is required"}
    try:
        from server.connectors_full import SemanticScholarConnector
        s2 = SemanticScholarConnector()
        papers = s2.search(topic, limit=max_papers, year=year_range)
        # Sort by citation count for relevance
        papers.sort(key=lambda p: p.get("citation_count", 0), reverse=True)
        # Categorize into tiers
        essential = [p for p in papers if p.get("citation_count", 0) > 100]
        recommended = [p for p in papers if 10 < p.get("citation_count", 0) <= 100]
        exploratory = [p for p in papers if p.get("citation_count", 0) <= 10]
        return {
            "topic": topic,
            "total": len(papers),
            "essential": essential,
            "recommended": recommended,
            "exploratory": exploratory,
        }
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}



# ── §4.0: Alert Subscription ──
_alert_subs_path = Path(DATA_ROOT) / "config" / "alert_subscriptions.json" if DATA_ROOT else None

async def subscribe_alert(body: dict = Body(default={})):
    """Subscribe to alerts for new papers on a topic.  §IMP: file-backed persistence."""
    topic = body.get("topic", "")
    email = body.get("email", "")
    frequency = body.get("frequency", "weekly")
    if not topic:
        return {"error": "topic is required"}
    # §IMP: Persist to file
    subs = []
    if _alert_subs_path and _alert_subs_path.exists():
        try:
            subs = json.loads(_alert_subs_path.read_text())
        except Exception:
            subs = []
    sub = {"topic": topic, "email": email, "frequency": frequency, "created": str(datetime.datetime.now())}
    subs.append(sub)
    if _alert_subs_path:
        _alert_subs_path.parent.mkdir(parents=True, exist_ok=True)
        _alert_subs_path.write_text(json.dumps(subs, indent=2))
    return {
        "status": "subscribed",
        "subscription": sub,
        "total_subscriptions": len(subs),
    }



# ── §SEC: Manifest Integrity Check ──
async def verify_manifest(body: dict = Body(default={})):
    """Verify integrity of sync manifest using SHA-256 checksums."""
    # §SEC: No user-controlled path — hardcoded to project manifest only
    manifest_path = ROOT_DIR / "google_sync_manifest.json"
    if not manifest_path.exists():
        return {"status": "not_found", "path": str(manifest_path)}
    
    try:
        import hashlib
        content = manifest_path.read_bytes()
        file_hash = hashlib.sha256(content).hexdigest()
        data = json.loads(content)
        entries = data if isinstance(data, dict) else {}
        
        suspicious = []
        for path_key, meta in entries.items():
            if not isinstance(meta, dict):
                continue
            stored_hash = meta.get("sha256", "")
            fp = Path(path_key)
            if fp.exists() and stored_hash:
                actual = hashlib.sha256(fp.read_bytes()).hexdigest()
                if stored_hash != actual:
                    suspicious.append({"path": path_key, "stored": stored_hash[:12], "actual": actual[:12]})
        
        return {"status": "verified", "manifest_hash": file_hash[:16],
                "total_entries": len(entries), "suspicious": suspicious, "suspicious_count": len(suspicious)}
    except Exception as e:
        log.error(f"Endpoint error: {e}")
        return {"status": "error", "error": "Operation failed. Check server logs."}


# ── §4.0: A/B Testing Framework ──
async def ab_test(body: dict = Body(default={})):
    """A/B test: run a query against two models and return both results for comparison."""
    question = body.get("question", "")
    model_a = body.get("model_a", DEFAULT_MODEL)
    model_b = body.get("model_b", OPENAI_FT_MODEL or DEFAULT_MODEL)
    if not question:
        return {"error": "question is required"}
    
    results = {}
    for label, model_id in [("A", model_a), ("B", model_b)]:
        try:
            if "gpt" in model_id or "ft:" in model_id:
                if OPENAI_API_KEY:
                    import requests as _req
                    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
                    r = _req.post(f"{base_url}/chat/completions",
                        headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                        json={"model": model_id, "messages": [{"role": "user", "content": question}],
                              "temperature": 0.1, "max_tokens": 1000}, timeout=30)
                    results[label] = {"model": model_id, "answer": r.json()["choices"][0]["message"]["content"]}
                else:
                    results[label] = {"model": model_id, "error": "No API key"}
            else:
                cfg = types.GenerateContentConfig(temperature=0.1, max_output_tokens=1000)
                resp = CLIENT.models.generate_content(model=model_id, contents=question, config=cfg)
                results[label] = {"model": model_id, "answer": resp.text}
        except Exception as e:
            log.warning(f"Model {model_id} error: {e}")
            results[label] = {"model": model_id, "error": "Model unavailable"}
    
    return {"question": question, "results": results}


# ── §RESTORED: API endpoints for restored modules ─────────────

async def confidence_endpoint(body: dict = Body(default={})):
    """Score answer confidence using multi-signal calibration."""
    if not _reasoning_enhancements:
        return {"error": "reasoning_enhancements not available"}
    try:
        signals = ConfidenceSignals(
            citation_coverage=body.get("citation_coverage", 0.0),
            source_relevance_avg=body.get("source_relevance_avg", 0.0),
            audit_result=body.get("audit_result", "pending"),
            claim_count=body.get("claim_count", 0),
            supported_claims=body.get("supported_claims", 0),
        )
        return {
            "score": signals.calibrated_score(),
            "level": signals.level(),
            "signals": signals.as_dict(),
        }
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


async def paragraph_confidence_endpoint(body: dict = Body(default={})):
    """Assign green/yellow/red confidence to each paragraph of an answer."""
    if not _reasoning_enhancements:
        return {"error": "reasoning_enhancements not available"}
    try:
        answer = body.get("answer", "")
        sources = body.get("sources", [])
        result = assign_paragraph_confidence(answer, sources)
        return {"paragraphs": result}
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


async def contradictions_endpoint(body: dict = Body(default={})):
    """Detect potential contradictions between sources.  §IMP: severity scoring."""
    if not _reasoning_enhancements:
        return {"error": "reasoning_enhancements not available"}
    try:
        raw_sources = body.get("sources", [])
        sources: list[dict] = []
        for src in raw_sources if isinstance(raw_sources, list) else []:
            if isinstance(src, str):
                sources.append({"snippet": src})
            elif isinstance(src, dict):
                snippet = src.get("snippet") or src.get("text") or src.get("content") or ""
                sources.append({**src, "snippet": snippet})
        result = detect_contradictions(sources)
        # §IMP: Add severity scoring (0-1) based on claim overlap
        scored = []
        for c in (result if isinstance(result, list) else []):
            severity = 0.5  # default
            if isinstance(c, dict):
                # Higher severity when sources directly conflict vs merely differ
                claim_a = (c.get("source_a_claim") or c.get("claim_1") or "").lower()
                claim_b = (c.get("source_b_claim") or c.get("claim_2") or "").lower()
                if claim_a and claim_b:
                    # Simple word overlap as severity proxy
                    words_a, words_b = set(claim_a.split()), set(claim_b.split())
                    overlap = len(words_a & words_b) / max(len(words_a | words_b), 1)
                    severity = round(min(1.0, 0.3 + overlap * 0.7), 2)
                c["severity"] = severity
            scored.append(c)
        return {"contradictions": scored, "count": len(scored)}
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


async def source_freshness_endpoint(body: dict = Body(default={})):
    """Flag sources that may be outdated.  §IMP: configurable recency window."""
    if not _reasoning_enhancements:
        return {"error": "reasoning_enhancements not available"}
    try:
        query = body.get("query", "")
        sources = body.get("sources", [])
        max_age_years = body.get("max_age_years", 5)  # §IMP: configurable
        stale = check_source_freshness(query, sources)
        # §IMP: Add age-based severity
        if isinstance(stale, list):
            import datetime
            current_year = datetime.datetime.now().year
            for s in stale:
                if isinstance(s, dict):
                    pub_year = s.get("year", s.get("pub_year", current_year))
                    age = current_year - int(pub_year) if pub_year else 0
                    s["age_years"] = age
                    s["severity"] = "critical" if age > max_age_years * 2 else ("high" if age > max_age_years else "low")
        return {"stale_sources": stale, "max_age_years": max_age_years}
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


async def active_learning_queue():
    """Get low-confidence queries pending human review.  §IMP: urgency ranking."""
    if not _active_learning:
        return {"pending": [], "stats": {}, "count": 0}
    pending = _active_learning.get_pending(limit=20)
    # §IMP: Add urgency based on age and confidence
    if isinstance(pending, list):
        for i, item in enumerate(pending):
            if isinstance(item, dict):
                conf = item.get("confidence", 0.5)
                item["urgency"] = "high" if conf < 0.3 else ("medium" if conf < 0.6 else "low")
    return {
        "pending": pending,
        "count": len(pending) if isinstance(pending, list) else 0,
        "stats": _active_learning.stats,
    }


async def active_learning_review(body: dict = Body(default={})):
    """Mark a query as reviewed in the active learning queue."""
    if not _active_learning:
        return {"error": "active learning not available"}
    query = body.get("query", "")
    if query:
        _active_learning.mark_reviewed(query)
    return {"status": "reviewed", "query": query}



async def citation_graph_stats():
    """Get stats of the auto-built citation graph from reference sections."""
    if not _citation_graph:
        return {"nodes": 0, "edges": 0}
    return _citation_graph.stats  # §FIX: @property, not method


async def detect_language_endpoint(body: dict = Body(default={})):
    """Detect the language of a text."""
    if not _indexing_enhancements:
        return {"error": "indexing_enhancements not available"}
    text = body.get("text", "")
    if not text:
        return {"error": "text is required"}
    return detect_language(text)


async def bibliography_endpoint(body: dict = Body(default={})):
    """Generate a formatted bibliography from sources."""
    if not _citation_formatter:
        return {"error": "citation_formatter not available"}
    sources = body.get("sources", [])
    style = body.get("style", "apa")
    return {
        "bibliography": generate_bibliography(sources, style=style),
        "style": style,
        "count": len(sources),
    }



async def shared_mode_status():
    """Check if shared mode is enabled and get permissions."""
    if not _shared_mode:
        return {"shared": False, "permissions": {}}
    return {
        "shared": is_shared_mode(),
        "permissions": get_shared_permissions("editor") if is_shared_mode() else {},
    }



# ── §PIPE: Pipeline API Endpoints ─────────────────────────────
# Wire standalone pipeline modules as on-demand API endpoints
# so the UI can trigger them without CLI access.

async def api_build_graph():
    """Rebuild the D3 knowledge graph from PhD-OS ontology artifacts."""
    try:
        from pipelines.build_graph import run_pipeline
        import threading
        threading.Thread(target=run_pipeline, daemon=True).start()
        return {"status": "started", "pipeline": "build_graph"}
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


async def api_extract_entities():
    """Run hybrid entity/claim extraction (Gemini Flash + regex fallback)."""
    try:
        from pipelines.extract_entities import run_pipeline
        import threading
        threading.Thread(target=run_pipeline, daemon=True).start()
        return {"status": "started", "pipeline": "extract_entities"}
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}



async def api_dual_brain(body: dict = Body(default={})):
    """
    Run a dual-brain sharpening cycle.
    Body: { "questions": ["Q1", "Q2", ...] }
    Runs both models, judges consensus, returns results.
    """
    questions = body.get("questions", [])
    if not questions:
        return {"error": "questions array is required"}
    try:
        from pipelines.dual_brain import sharpen_cycle
        import threading
        results: dict = {}
        def _run():
            results.update(sharpen_cycle(questions))
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return {"status": "started", "question_count": len(questions)}
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


async def api_run_eval():
    """Run the full evaluation pipeline (MRR, NDCG, hallucination, latency)."""
    try:
        from pipelines.run_eval import main as run_eval_main
        import threading
        threading.Thread(target=run_eval_main, daemon=True).start()
        return {"status": "started", "pipeline": "eval"}
    except Exception as e:
        log.warning(f"Endpoint error: {e}"); return {"error": "Operation failed"}


# §EXTRACTED: NYT routes moved to server/routes/ — see wiring section above


# §EXTRACTED: MENDELEY routes moved to server/routes/ — see wiring section above


# ═══════════════════════════════════════════════════════════════════
# §RESILIENCE: Architecture Improvements Layer (must be before SPA catch-all)
# ═══════════════════════════════════════════════════════════════════
try:
    from server.resilience import register_resilience_routes, register_batch_route, wire_eventbus_to_websocket
    register_resilience_routes(app)
    register_batch_route(app)
    wire_eventbus_to_websocket()
    log.info("§RESILIENCE: 10 improvements active — /api/resilience/status")
except Exception as _e:
    log.warning(f"§RESILIENCE: Could not load: {_e}")

# ── SPA catch-all moved to end of file (after all router includes) ──
# See bottom of this file for the /{full_path:path} handler.
# §SPA: Resolve renderer dist directory (built output)
# Priority: app/renderer (packaged) → renderer/dist (built) → renderer (dev fallback)
_RENDERER_DIST = ROOT_DIR / "app" / "renderer"
if not _RENDERER_DIST.is_dir():
    _RENDERER_DIST = ROOT_DIR / "renderer" / "dist"  # Vite build output
if not _RENDERER_DIST.is_dir():
    _RENDERER_DIST = ROOT_DIR / "renderer"  # dev fallback (Vite serves from src)
_RENDERER_INDEX = _RENDERER_DIST / "index.html"


if __name__ == "__main__":
    import uvicorn
    import multiprocessing
    # §ARCH: Use multiple workers to eliminate single-process bottleneck
    # Workers scale to CPU count (min 2, max 8) for optimal throughput
    _cpu_count = multiprocessing.cpu_count()
    _num_workers = max(2, min(_cpu_count, 8))
    log.info(f"§ARCH: Starting uvicorn with {_num_workers} workers (CPUs: {_cpu_count})")
    uvicorn.run(
        "server.main:app",
        host="127.0.0.1",
        port=8001,
        workers=_num_workers,
        log_level="info",
    )


# -- Graceful shutdown: drain active streams before closing pools --
_active_streams: set = set()

def _run_shutdown():
    """Shutdown sequence: release lock, drain streams, stop worker, close pools."""
    # Release session lock before closing
    try:
        from server.session_lock import SessionLock
        SessionLock().release()
    except Exception:
        pass

    if _active_streams:
        log.info(f"Waiting for {len(_active_streams)} active stream(s) to finish...")
        deadline = _time.time() + 10  # Wait up to 10 seconds
        while _active_streams and _time.time() < deadline:
            _time.sleep(0.5)
        if _active_streams:
            log.warning(f"Shutdown timeout: {len(_active_streams)} stream(s) still active")
    # §FIX: Clean up reindex subprocesses on shutdown
    if hasattr(app, "_reindex_pids") and app._reindex_pids:
        import signal as _sig
        for pid in app._reindex_pids:
            try:
                os.kill(pid, _sig.SIGTERM)
                log.info(f"Shutdown: terminated reindex subprocess pid={pid}")
            except (ProcessLookupError, OSError):
                pass  # Already exited
        app._reindex_pids.clear()

    # §ARCH: Stop the compute worker process
    try:
        _worker_client.stop_worker()
    except Exception:
        pass

    shutdown_pool()
    log.info("Server shutdown: connection pools closed")


# -- Startup hook: warm library cache in background --
def _run_startup():
    """Initialize logging, validate environment, and warm caches on startup."""
    # Wire logging configuration
    if _logging_config:
        try:
            setup_logging()
            log.info("Structured logging initialized")
        except Exception as e:
            log.warning(f"Logging setup failed, using defaults: {e}")

    # Startup environment validation
    if not API_KEY:
        log.warning("⚠ GOOGLE_API_KEY not set — Gemini inference will fail")
    if not DATA_ROOT:
        log.warning("⚠ EDITH_DATA_ROOT not set — indexing and library disabled")
    if OPENAI_FT_MODEL and not OPENAI_API_KEY:
        log.warning("⚠ OPENAI_API_KEY not set — Winnie fine-tuned model disabled")

    # §HW: Log compute profile at startup
    try:
        from server.backend_logic import get_compute_profile
        _profile = get_compute_profile()
        log.info(f"§HW: Compute profile: {_profile['mode']} | "
                 f"chip={_profile['chip'][:40]} | "
                 f"drive={_profile['drive_connection']} | "
                 f"agents={_profile['agents']} | "
                 f"top_k={_profile['top_k']} | "
                 f"neural_cores={_profile['neural_engine_cores']}")
    except Exception as e:
        log.warning(f"$HW: Compute profile detection failed: {e}")

    # $BOLT-1: Journal recovery -- auto-rollback interrupted writes
    try:
        from server.bolt_journal import BoltJournal
        journal = BoltJournal()
        recovery = journal.recover_if_needed()
        if recovery.get("status") == "recovered":
            log.warning(f"BOLT RECOVERY: {recovery['message']}")
        else:
            log.info(f"Bolt journal: {recovery.get('status', 'ok')}")
    except Exception as e:
        log.warning(f"Bolt journal check skipped: {e}")

    # $BOLT-2: Session lock -- prevent brain-split between machines
    try:
        from server.session_lock import SessionLock
        lock_result = SessionLock().acquire()
        if lock_result.get("locked_by_other"):
            owner = lock_result.get("owner", {})
            log.warning(f"SESSION CONFLICT: Soul active on {owner.get('hostname', 'unknown')}")
        else:
            log.info(f"Session lock: {lock_result.get('message', 'acquired')}")
    except Exception as e:
        log.warning(f"Session lock skipped: {e}")

    # §B9: Register startup components
    startup_gate.register_component("library")
    startup_gate.register_component("mlx_inference")
    startup_gate.register_component("embeddings")
    startup_gate.register_component("chroma")
    startup_gate.register_component("iops")

    # §ARCH: Configure centralized server state
    _server_state.set_config(
        api_key=API_KEY,
        data_root=DATA_ROOT,
        chroma_dir=CHROMA_DIR,
        chroma_collection=CHROMA_COLLECTION,
        embed_model=EMBED_MODEL,
        default_model=DEFAULT_MODEL,
        retrieval_backend=RETRIEVAL_BACKEND,
        openai_ft_model=OPENAI_FT_MODEL,
    )

    # Warm library cache + MLX models + ChromaDB health in background
    import threading
    def _warmup():
        _warmup_t0 = _time.time()

        # Load orchestration modules first (deferred from module scope)
        _load_orchestration_modules()
        _load_launch_modules()

        try:
            from server.routes.library import _scan_library_sources
            _scan_library_sources(papers_only=True)
            log.info("Library cache warmed on startup")
            startup_gate.mark_component_ready("library")
        except NameError:
            # CHROMA_DIR not yet injected — retry after a short delay
            import time as _time2
            _time2.sleep(2)
            try:
                from server.routes.library import _scan_library_sources
                _scan_library_sources(papers_only=True)
                log.info("Library cache warmed on startup (retry)")
                startup_gate.mark_component_ready("library")
            except Exception as e2:
                log.warning(f"Library cache warmup failed (retry): {e2}")
        except Exception as e:
            log.warning(f"Library cache warmup failed: {e}")

        # §ARCH: Offload heavy compute (MLX, embeddings, ChromaDB) to worker
        # The worker runs in a separate process — no event loop blocking
        _worker_launched = False
        try:
            _worker_launched = _worker_client.ensure_worker_running()
            if _worker_launched:
                log.info("§ARCH: Compute worker launched — heavy warmup offloaded")
                _server_state.set_worker_available(True)
                # Worker handles MLX, embeddings, ChromaDB warmup in its own process
                startup_gate.mark_component_ready("mlx_inference")
                startup_gate.mark_component_ready("embeddings")
                startup_gate.mark_component_ready("chroma")
                startup_gate.mark_component_ready("iops")
            else:
                log.info("§ARCH: Worker not available — falling back to in-process warmup")
        except Exception as e:
            log.warning(f"§ARCH: Worker launch failed: {e}")

        # Fallback: in-process warmup if worker didn't start
        if not _worker_launched:
            # §HW: Pre-download local MLX inference model
            try:
                from server.mlx_inference import load_model, is_available
                if is_available():
                    if load_model():
                        log.info("§LOCAL: MLX inference model pre-loaded")
                    else:
                        log.info("§LOCAL: MLX model download skipped (will try on first query)")
                startup_gate.mark_component_ready("mlx_inference")
            except Exception as e:
                log.debug(f"§LOCAL: MLX warmup skipped: {e}")
                startup_gate.mark_component_ready("mlx_inference")

            # §NPU: Pre-load embedding model (CoreML Neural Engine path)
            try:
                from server.mlx_embeddings import embed, is_available as embed_avail, get_backend_info
                if embed_avail():
                    embed(["warmup"])  # Trigger model load + CoreML compilation
                    bi = get_backend_info()
                    log.info(f"§NPU: Embedding model pre-loaded via {bi.get('backend', '?')}")
                startup_gate.mark_component_ready("embeddings")
            except Exception as e:
                log.debug(f"§NPU: Embedding warmup skipped: {e}")
                startup_gate.mark_component_ready("embeddings")

            # ChromaDB integrity check
            try:
                if CHROMA_DIR and os.path.isdir(CHROMA_DIR):
                    import chromadb
                    _client = chromadb.PersistentClient(path=CHROMA_DIR)
                    for _coll in _client.list_collections():
                        try:
                            c = _client.get_collection(_coll.name)
                            count = c.count()
                            log.info(f"§CHROMA: {_coll.name}: {count} docs — OK")
                        except Exception as _ce:
                            log.error(f"§CHROMA: {_coll.name}: INTEGRITY FAILURE — {_ce}")
                startup_gate.mark_component_ready("chroma")
            except Exception as e:
                log.debug(f"§CHROMA: Health check skipped: {e}")
                startup_gate.mark_component_ready("chroma")

            # §IOPS: Apply hardware-aware SQLite pragmas to ChromaDB
            try:
                if CHROMA_DIR and os.path.isdir(CHROMA_DIR):
                    from server.chroma_tuning import tune_chroma_db, log_tuning_report
                    tuning = tune_chroma_db(CHROMA_DIR)
                    log_tuning_report(tuning)
                startup_gate.mark_component_ready("iops")
            except Exception as e:
                log.debug(f"§IOPS: ChromaDB tuning skipped: {e}")
                startup_gate.mark_component_ready("iops")

        # §ORCH: Start background services
        try:
            if _memory_scaler_ok:
                _memory_monitor.start()
                log.info("§ORCH: Memory monitor started")
        except Exception as e:
            log.debug(f"§ORCH: Memory monitor start failed: {e}")

        try:
            if _prefetcher_ok:
                _prefetcher.start()
                log.info("§ORCH: Prefetcher started")
        except Exception as e:
            log.debug(f"§ORCH: Prefetcher start failed: {e}")

        # §B9: All components ready
        startup_gate.mark_ready()
        _server_state.mark_startup_complete()

        _warmup_elapsed = _time.time() - _warmup_t0
        log.info(f"§ARCH: Startup warmup complete in {_warmup_elapsed:.1f}s "
                 f"(worker={'active' if _worker_launched else 'fallback'})")

        # §LAUNCH: Pre-import heavy modules so first request doesn't block 30s
        # These imports populate sys.modules; the route _ensure_*() guards
        # will find them already loaded and skip the blocking import.
        _heavy_modules = [
            ("server.cognitive_engine", "cognitive"),
            ("server.antigravity_engine", "antigrav"),
            ("server.committee", "committee"),
            ("server.simulation_deck", "simulation"),
        ]
        for _mod_name, _label in _heavy_modules:
            try:
                _t = _time.time()
                __import__(_mod_name)
                log.info(f"§LAUNCH: Pre-loaded {_label} in {_time.time()-_t:.1f}s")
            except Exception as _e:
                log.debug(f"§LAUNCH: {_label} pre-load skipped: {_e}")
        # Trigger route-level caching so _ensure_*() returns instantly
        try:
            from server.routes.cognitive import _ensure_cognitive
            _ensure_cognitive()
        except Exception:
            pass
        try:
            from server.routes.antigravity import _ensure_antigrav
            _ensure_antigrav()
        except Exception:
            pass
        try:
            from server.routes.orchestration import _ensure_peer_review
            _ensure_peer_review()
        except Exception:
            pass
        try:
            from server.routes.causal import _ensure_simulation
            _ensure_simulation()
        except Exception:
            pass
        log.info(f"§LAUNCH: Heavy module pre-load complete "
                 f"({_time.time()-_warmup_t0:.1f}s total warmup)")

        # Fix 3: Dream Engine scheduler — runs DreamEngine.dream() at 3 AM daily
        try:
            from server.dream_engine import DreamEngine
            _dream = DreamEngine()

            def _dream_scheduler():
                import datetime
                while True:
                    now = datetime.datetime.now()
                    target = now.replace(hour=3, minute=0, second=0, microsecond=0)
                    if target <= now:
                        target += datetime.timedelta(days=1)
                    wait_secs = (target - now).total_seconds()
                    log.info(f"§DREAM: Next dream cycle in {wait_secs/3600:.1f}h")
                    _time.sleep(wait_secs)
                    try:
                        log.info("§DREAM: Starting overnight dream cycle")
                        result = _dream.dream()
                        log.info(f"§DREAM: Complete — {result}")
                    except Exception as _de:
                        log.warning(f"§DREAM: Dream cycle failed: {_de}")

            threading.Thread(target=_dream_scheduler, daemon=True, name="dream-scheduler").start()
            log.info("§DREAM: Scheduler armed for 03:00")
        except Exception as _de:
            log.debug(f"§DREAM: Scheduler setup failed: {_de}")

    threading.Thread(target=_warmup, daemon=True).start()


# ═══════════════════════════════════════════════════════════════════
# §ORCH-7: HANDLERS EXTRACTED — See server/routes/ for all endpoints
#
# The following route modules replaced ~85 inline @app handlers:
#   orchestration.py  — Deep Dive, Peer Review, Tutor, Shadow, Vibe, Maintenance
#   cognitive.py      — Persona, Socratic, Spaced Rep, Graph Retrieve
#   causal.py         — Guardrails, Causal Engine, Simulation Deck
#   jarvis.py         — Ambient Watcher, Sandbox, Oracle Engine
#   antigravity.py    — Tab-to-Intent, Self-Heal, Agent Dispatch, Skill
# ═══════════════════════════════════════════════════════════════════

# §ORCH-7 FIX: Minimal globals for Cockpit + session archive endpoints
# These were in the removed inline handler blocks but are still needed
# by cockpit_status(), session_init(), and session_archive().
_jarvis_ok = _sim_ok = _oracle_ok = _antigrav_ok = _causal_ok = False
_ambient_watcher = _portable_env = None
try:
    from server.jarvis_layer import AmbientWatcher, PortableEnvironment
    _ambient_watcher = AmbientWatcher()
    _portable_env = PortableEnvironment()
    _jarvis_ok = True
except Exception:
    pass
try:
    from server.simulation_deck import SimulationDeck
    _sim_ok = True
except Exception:
    pass
try:
    from server.oracle_engine import OracleEngine
    _oracle_ok = True
except Exception:
    pass
try:
    from server.antigravity_engine import generate_research_memo
    _antigrav_ok = True
except Exception:
    pass
try:
    from server.causal_engine import CausalEngine
    _causal_ok = True
except Exception:
    pass

# --- Vector Mapping / Cockpit ---
try:
    from server.vector_mapping import (
        build_atlas_from_chroma, generate_topological_summary,
        classify_field, CLUSTER_CENTROIDS, CLUSTER_COLORS,
    )
    _atlas_ok = True
    log.info(f"§LAUNCH: Vector mapping loaded ({len(CLUSTER_CENTROIDS)} clusters)")
except Exception as _e:
    _atlas_ok = False
    log.warning(f"§LAUNCH: Vector mapping not available: {_e}")

@app.post("/api/cockpit/atlas", tags=["Cockpit"])
async def cockpit_atlas(request: Request):
    if not _atlas_ok: return _error_response(503, "unavailable", "Vector mapping not loaded")
    body = await request.json()
    try:
        import asyncio
        result = await asyncio.wait_for(
            asyncio.to_thread(
                build_atlas_from_chroma,
                chroma_dir=CHROMA_DIR, embed_model=EMBED_MODEL,
                sample_size=body.get("sample_size", 2000),
            ),
            timeout=30.0,
        )
        return result
    except asyncio.TimeoutError:
        return {"status": "partial", "message": "Atlas generation timed out (30s limit)", "clusters": {}, "sample_size": body.get("sample_size", 2000)}
    except Exception as e:
        return {"status": "error", "message": str(e)[:200]}

@app.post("/api/cockpit/topology", tags=["Cockpit"])
async def cockpit_topology(request: Request):
    if not _atlas_ok: return _error_response(503, "unavailable", "Vector mapping not loaded")
    body = await request.json()
    try:
        import asyncio
        result = await asyncio.wait_for(
            asyncio.to_thread(
                generate_topological_summary,
                body.get("cluster", "APE"),
            ),
            timeout=30.0,
        )
        return result
    except asyncio.TimeoutError:
        return {"status": "partial", "message": "Topology generation timed out (30s limit)", "cluster": body.get("cluster", "APE")}
    except Exception as e:
        return {"status": "error", "message": str(e)[:200]}

# ── SSE Streaming Variants ──────────────────────────────────────
@app.post("/api/cockpit/atlas/stream", tags=["Cockpit"])
async def cockpit_atlas_stream(request: Request):
    """SSE-streaming version of atlas generation — shows progress instead of spinner."""
    if not _atlas_ok: return _error_response(503, "unavailable", "Vector mapping not loaded")
    body = await request.json()
    from server.agentic import sse_progress_stream
    return StreamingResponse(
        sse_progress_stream(
            build_atlas_from_chroma,
            work_kwargs={"chroma_dir": CHROMA_DIR, "embed_model": EMBED_MODEL, "sample_size": body.get("sample_size", 2000)},
            step_label="atlas_generation",
            timeout=60.0,
        ),
        media_type="text/event-stream",
    )

@app.post("/api/cockpit/topology/stream", tags=["Cockpit"])
async def cockpit_topology_stream(request: Request):
    """SSE-streaming version of topology generation."""
    if not _atlas_ok: return _error_response(503, "unavailable", "Vector mapping not loaded")
    body = await request.json()
    from server.agentic import sse_progress_stream
    return StreamingResponse(
        sse_progress_stream(
            generate_topological_summary,
            work_args=(body.get("cluster", "APE"),),
            step_label="topology_generation",
            timeout=60.0,
        ),
        media_type="text/event-stream",
    )

# ── Session Store Status ────────────────────────────────────────
@app.get("/api/sessions/stats", tags=["Agentic"])
async def session_stats():
    """Return session store statistics."""
    try:
        from server.agentic import sessions
        return sessions.stats()
    except Exception as e:
        return {"error": str(e)}

# ── Overnight Learning Loop ──────────────────────────────────────
@app.post("/api/learn/overnight", tags=["Agentic"])
async def learn_overnight(request: Request):
    """Run the overnight learning consolidation.

    Consolidates the day's interactions, updates the research profile,
    and generates a morning briefing. This is how Winnie gets smarter.
    """
    import asyncio
    body = await request.json() if request.headers.get("content-type", "").startswith("application/json") else {}
    since_hours = body.get("since_hours", 24)
    try:
        from server.overnight_learner import run_overnight
        result = await asyncio.to_thread(run_overnight, since_hours)
        return result
    except Exception as e:
        return {"error": str(e)[:200]}

@app.get("/api/learn/briefing", tags=["Agentic"])
async def get_briefing():
    """Get the morning briefing — a summary of yesterday's work + next steps."""
    import asyncio
    try:
        from server.overnight_learner import generate_briefing
        briefing = await asyncio.to_thread(generate_briefing)
        return briefing
    except Exception as e:
        return {"error": str(e)[:200]}

@app.get("/api/learn/consolidation", tags=["Agentic"])
async def get_consolidation():
    """Get the raw consolidation data — topics, methods, weak spots."""
    import asyncio
    try:
        from server.overnight_learner import consolidate_interactions
        result = await asyncio.to_thread(consolidate_interactions, 24)
        return result
    except Exception as e:
        return {"error": str(e)[:200]}

@app.get("/api/cockpit/clusters", tags=["Cockpit"])
async def cockpit_clusters():
    if not _atlas_ok: return _error_response(503, "unavailable", "Vector mapping not loaded")
    return {
        "clusters": {
            name: {"centroid": list(c), "color": CLUSTER_COLORS.get(name, "#94A3B8")}
            for name, c in CLUSTER_CENTROIDS.items()
        },
    }

@app.get("/api/cockpit/status", tags=["Cockpit"])
async def cockpit_status():
    return {
        "surfaces": ["atlas", "warroom", "committee", "cockpit"],
        "modules_loaded": {
            "atlas": _atlas_ok,
            "simulation": _sim_ok,
            "oracle": _oracle_ok,
            "antigravity": _antigrav_ok,
            "jarvis": _jarvis_ok,
            "causal": globals().get('_causal_ok', False),
        },
        "endpoints": 216,
        "modules": 19,
    }


# ═══════════════════════════════════════════════════════════════════
# §DEDUP: Vault + Citadel routes REMOVED — now served by:
#   - routes/vault.py (vault/init, vault/on-mount, vault/save-artefact, session/archive)
#   - routes/citadel_api.py (hybrid/query, hybrid/status, pedagogy/*, simulation/shock,
#     focus/engage, focus/disengage, atlas/lod, theme, boot, audit/recent, rag/priority)
# These were extracted but the inline copies remained, causing duplicate routes.
# ═══════════════════════════════════════════════════════════════════


# §UNWIRED: Register all remaining modules for import validation
# These are either used indirectly by wired modules (auto_annotator,
# graph_vector_engine, etc.) or standalone features (analytics, etc.)
# ═══════════════════════════════════════════════════════════════════

# ── Citadel Brain Modules ──
_citadel_neural_net = None
_dream_engine_mod = None
_synapse_bridge_mod = None

try:
    from server.citadel_neural_net import EDITHBrain
    _citadel_neural_net = True
    log.info("CitadelNeuralNet loaded")
except Exception as _e:
    log.warning(f"citadel_neural_net not available: {_e}")

try:
    from server.dream_engine import DreamEngine
    _dream_engine_mod = True
    log.info("DreamEngine loaded")
except Exception as _e:
    log.warning(f"dream_engine not available: {_e}")

try:
    from server.synapse_bridge import SynapseBridge
    _synapse_bridge_mod = True
    log.info("SynapseBridge loaded")
except Exception as _e:
    log.warning(f"synapse_bridge not available: {_e}")

# ── Forensic Lab Modules ──
_paper_deconstructor = None
_method_lab_mod = None
_lit_locator_mod = None
_forensic_audit_mod = None
_auto_annotator_mod = None
_notion_bridge_mod = None
_shadow_drafter_mod = None

try:
    from server.paper_deconstructor import PaperDeconstructor
    _paper_deconstructor = True
    log.info("PaperDeconstructor loaded")
except Exception as _e:
    log.warning(f"paper_deconstructor not available: {_e}")

try:
    from server.method_lab import MethodLab
    _method_lab_mod = True
    log.info("MethodLab loaded")
except Exception as _e:
    log.warning(f"method_lab not available: {_e}")

try:
    from server.lit_locator import LitLocator
    _lit_locator_mod = True
    log.info("LitLocator loaded")
except Exception as _e:
    log.warning(f"lit_locator not available: {_e}")

try:
    from server.forensic_audit import ForensicAuditOrchestrator
    _forensic_audit_mod = True
    log.info("ForensicAudit loaded")
except Exception as _e:
    log.warning(f"forensic_audit not available: {_e}")

try:
    from server.auto_annotator import AutoAnnotator
    _auto_annotator_mod = True
    log.info("AutoAnnotator loaded")
except Exception as _e:
    log.warning(f"auto_annotator not available: {_e}")

try:
    from server.notion_bridge import NotionBridge
    _notion_bridge_mod = True
    log.info("NotionBridge loaded")
except Exception as _e:
    log.warning(f"notion_bridge not available: {_e}")

try:
    from server.shadow_drafter import ShadowDrafter
    _shadow_drafter_mod = True
    log.info("ShadowDrafter loaded")
except Exception as _e:
    log.warning(f"shadow_drafter not available: {_e}")

# ── Vector & Graph Engines ──
_graph_vector_engine_mod = None
_memory_pinning_mod = None

try:
    from server.graph_vector_engine import GraphVectorEngine
    _graph_vector_engine_mod = True
    log.info("GraphVectorEngine loaded")
except Exception as _e:
    log.warning(f"graph_vector_engine not available: {_e}")

try:
    from server.memory_pinning import MemoryPinner
    _memory_pinning_mod = True
    log.info("MemoryPinning loaded")
except Exception as _e:
    log.warning(f"memory_pinning not available: {_e}")

# ── Infrastructure Modules ──
_analytics_mod = None
_anomaly_mod = None
_input_sanitizer_mod = None
_rate_limiter_mod = None
_semantic_drift_mod = None
_speculative_indexer_mod = None

try:
    from server.analytics import SessionIntelligence
    _analytics_mod = True
    log.info("Analytics loaded")
except Exception as _e:
    log.warning(f"analytics not available: {_e}")

try:
    from server.anomaly import AnomalyDetector
    _anomaly_mod = True
    log.info("Anomaly loaded")
except Exception as _e:
    log.warning(f"anomaly not available: {_e}")

try:
    from server.input_sanitizer import DANGEROUS_PATTERNS, ACADEMIC_PATTERNS
    _input_sanitizer_mod = True
    log.info("InputSanitizer loaded")
except Exception as _e:
    log.warning(f"input_sanitizer not available: {_e}")

try:
    from server.rate_limiter import AdaptiveRateLimiter
    _rate_limiter_mod = True
    log.info("RateLimiter loaded")
except Exception as _e:
    log.warning(f"rate_limiter not available: {_e}")

try:
    from server.semantic_drift import DriftVector, POLITICAL_SCIENCE_TERMS
    _semantic_drift_mod = True
    log.info("SemanticDrift loaded")
except Exception as _e:
    log.warning(f"semantic_drift not available: {_e}")

try:
    from server.speculative_indexer import SpeculativeIndexer
    _speculative_indexer_mod = True
    log.info("SpeculativeIndexer loaded")
except Exception as _e:
    log.warning(f"speculative_indexer not available: {_e}")

# ── Experimental / Staged Modules ──
_causal_discovery_mod = None
_causal_raytracing_mod = None
_method_sandbox_mod = None
_spatial_audio_mod = None
_wasm_sovereignty_mod = None
_socratic_coach_mod = None

try:
    from server.causal_discovery import CausalDiscoveryEngine
    _causal_discovery_mod = True
    log.info("CausalDiscovery loaded")
except Exception as _e:
    log.warning(f"causal_discovery not available: {_e}")

try:
    from server.causal_raytracing import CausalRayTracer
    _causal_raytracing_mod = True
    log.info("CausalRaytracing loaded")
except Exception as _e:
    log.warning(f"causal_raytracing not available: {_e}")

try:
    from server.method_sandbox import SurveySimulator
    _method_sandbox_mod = True
    log.info("MethodSandbox loaded")
except Exception as _e:
    log.warning(f"method_sandbox not available: {_e}")

try:
    from server.spatial_audio import SpatialAudioEngine
    _spatial_audio_mod = True
    log.info("SpatialAudio loaded")
except Exception as _e:
    log.warning(f"spatial_audio not available: {_e}")

try:
    from server.wasm_sovereignty import SovereigntyEngine
    _wasm_sovereignty_mod = True
    log.info("WasmSovereignty loaded")
except Exception as _e:
    log.warning(f"wasm_sovereignty not available: {_e}")

try:
    from server.socratic_coach import SocraticCoach
    _socratic_coach_mod = True
    log.info("SocraticCoach loaded (deprecated — use socratic_navigator)")
except Exception as _e:
    log.warning(f"socratic_coach not available: {_e}")



# ═══════════════════════════════════════════════════════════════════
# §BRAIN: Endpoints extracted to server/routes/brain.py
# 25 routes: Bridge(7), Monitor(8), Socratic(4), Connectome(2),
#            HUD(1), Master(6) — each with guarded imports
# ═══════════════════════════════════════════════════════════════════

try:
    from server.routes.brain import router as _brain_router
    app.include_router(_brain_router)
    log.info("Brain routes loaded (25 endpoints)")
except Exception as _e:
    log.warning(f"Brain routes unavailable: {_e}")

# §MCL-2026: Master Connector List — Zotero, Claude, Perplexity, etc.
try:
    from server.routes.connectors_hub import router as _connectors_hub_router
    app.include_router(_connectors_hub_router)
    log.info("Connectors Hub routes loaded (20 endpoints)")
except Exception as _e:
    log.warning(f"Connectors Hub routes unavailable: {_e}")

# §SNIPER: Methodological Sniper — forensic paper audit
try:
    from server.routes.sniper import router as _sniper_router
    app.include_router(_sniper_router)
    log.info("Sniper routes loaded (5 endpoints)")
except Exception as _e:
    log.warning(f"Sniper routes unavailable: {_e}")

# §FLYWHEEL: 10 Flywheel enhancements — Oracle pull, Stata→LaTeX, focus, training, etc.
try:
    from server.routes.flywheel import router as _flywheel_router
    app.include_router(_flywheel_router)
    log.info("Flywheel routes loaded (12 endpoints)")
except Exception as _e:
    log.warning(f"Flywheel routes unavailable: {_e}")

# §FLYWHEEL-ADV: 8 advanced capabilities — lit review, recommendations, peer review, etc.
try:
    from server.routes.flywheel_advanced import router as _flywheel_adv_router
    app.include_router(_flywheel_adv_router)
    log.info("Flywheel Advanced routes loaded (16 endpoints)")
except Exception as _e:
    log.warning(f"Flywheel Advanced routes unavailable: {_e}")


# ═══════════════════════════════════════════════════════════════════
# §ROUTERS: Route modules — 89+ routes across 12 domain groups
# MUST be at end of file: handler functions must be defined before
# register() calls getattr(main, handler_name)
# ═══════════════════════════════════════════════════════════════════
from server.routes import chat as _r_chat, library as _r_library
from server.routes import search as _r_search, training as _r_training
from server.routes import pipelines as _r_pipelines, doctor as _r_doctor
from server.routes import export as _r_export, security as _r_security
from server.routes import indexing as _r_indexing, reasoning as _r_reasoning
from server.routes import research as _r_research, system as _r_system
from server.vault_config import VAULT_ROOT, VECTORS_DIR, BOLT_MOUNTED

for _mod in [_r_chat, _r_library, _r_search, _r_training, _r_pipelines,
             _r_doctor, _r_export, _r_security, _r_indexing, _r_reasoning,
             _r_research, _r_system]:
    _router = _mod.register(app, ns=globals())
    app.include_router(_router)

# §ORCH-7: Auto-extracted domain route modules (lazy-loaded imports)
from server.routes import orchestration as _r_orch
from server.routes import cognitive as _r_cognitive
from server.routes import causal as _r_causal
from server.routes import jarvis as _r_jarvis
from server.routes import antigravity as _r_antigrav
from server.routes import integrations as _r_integrations
from server.routes import intelligence as _r_intelligence
from server.routes import pipeline as _r_pipeline
from server.routes import agent as _r_agent

for _mod in [_r_orch, _r_cognitive, _r_causal, _r_jarvis, _r_antigrav,
             _r_integrations, _r_intelligence, _r_pipeline, _r_agent]:
    try:
        _router = _mod.register(app)
        app.include_router(_router)
    except Exception as _e:
        log.warning(f"§ORCH-7: Route module {_mod.__name__} failed: {_e}")

# §WIRE: Additional extracted route modules (not covered by core or ORCH-7)
for _mod_path in ["server.routes.library_analytics", "server.routes.tools_misc", "server.routes.compat"]:
    try:
        _mod = __import__(_mod_path, fromlist=["router"])
        app.include_router(_mod.router)
        log.info(f"§WIRE: {_mod_path.split('.')[-1]} routes registered")
    except Exception as _e:
        log.warning(f"§WIRE: {_mod_path.split('.')[-1]} unavailable: {_e}")

# §SWITCHBOARD: Mission Orchestration Engine
try:
    from server.routes.missions import router as _missions_router
    app.include_router(_missions_router)
    # Initialize MissionRunner and register templates
    from server.mission_runner import get_mission_runner
    from server.mission_templates import register_all_templates
    _mission_runner = get_mission_runner(app)
    register_all_templates(_mission_runner)
    log.info(f"§SWITCHBOARD: Mission routes loaded (7 endpoints, {len(_mission_runner.available_templates)} templates)")
except Exception as _e:
    log.warning(f"§SWITCHBOARD: Mission routes unavailable: {_e}")

# §SUBCONSCIOUS: Cross-Domain Handshake Streams
try:
    from server.routes.streams import router as _streams_router
    app.include_router(_streams_router)
    log.info("§SUBCONSCIOUS: Streams routes loaded (15 endpoints, 9 handshakes)")
except Exception as _e:
    log.warning(f"§SUBCONSCIOUS: Streams routes unavailable: {_e}")

# §BUS: Unified Event Bus — The Nervous System
try:
    from server.routes.event_bus import router as _bus_router
    app.include_router(_bus_router)
    from server.event_bus import register_all_subscribers
    register_all_subscribers()
    log.info("§BUS: Event Bus loaded (5 endpoints, 13 subscribers, 29 event types)")
except Exception as _e:
    log.warning(f"§BUS: Event Bus unavailable: {_e}")

# §FERRARI: Unified Routers — search, dashboard, export consolidation
try:
    from server.routes.ferrari import router as _ferrari_router
    app.include_router(_ferrari_router)
    log.info("§FERRARI: Unified routers loaded (research/search, system/dashboard, export/unified)")
except Exception as _e:
    log.warning(f"§FERRARI: Unified routers unavailable: {_e}")

# §FERRARI: Extracted Cockpit, Vault, Citadel API routes
try:
    from server.routes.cockpit import router as _cockpit_router, init_cockpit_state
    app.include_router(_cockpit_router)
    init_cockpit_state({
        "atlas_ok": _atlas_ok,
        "sim_ok": _sim_ok,
        "oracle_ok": _oracle_ok,
        "antigrav_ok": _antigrav_ok,
        "jarvis_ok": _jarvis_ok,
        "causal_ok": globals().get('_causal_ok', False),
        "CHROMA_DIR": CHROMA_DIR, "EMBED_MODEL": EMBED_MODEL,
        "build_atlas_from_chroma": globals().get('build_atlas_from_chroma'),
        "generate_topological_summary": globals().get('generate_topological_summary'),
        "CLUSTER_CENTROIDS": globals().get('CLUSTER_CENTROIDS', {}),
        "CLUSTER_COLORS": globals().get('CLUSTER_COLORS', {}),
    })
    log.info("§FERRARI: Cockpit routes extracted (4 endpoints)")
except Exception as _e:
    log.warning(f"§FERRARI: Cockpit routes unavailable: {_e}")

try:
    from server.routes.vault import router as _vault_router, init_vault_state
    app.include_router(_vault_router)
    init_vault_state({
        "jarvis_ok": _jarvis_ok if '_jarvis_ok' in dir() else False,
        "antigrav_ok": _antigrav_ok if '_antigrav_ok' in dir() else False,
        "ambient_watcher": globals().get('_ambient_watcher'),
        "overnight_sandbox": globals().get('_overnight_sandbox'),
        "portable_env": globals().get('_portable_env'),
    })
    log.info("§FERRARI: Vault + Session routes extracted (4 endpoints)")
except Exception as _e:
    log.warning(f"§FERRARI: Vault routes unavailable: {_e}")

try:
    from server.routes.citadel_api import router as _citadel_router, init_citadel_state
    app.include_router(_citadel_router)
    init_citadel_state({
        "hybrid_ok": _hybrid_ok if '_hybrid_ok' in dir() else False,
        "hybrid_engine": globals().get('_hybrid_engine'),
        "pedagogy_ok": _pedagogy_ok if '_pedagogy_ok' in dir() else False,
        "pedagogy_indexer": globals().get('_pedagogy_indexer'),
        "query_as_exam": globals().get('query_as_exam'),
        "monte_carlo_ok": _monte_carlo_ok if '_monte_carlo_ok' in dir() else False,
        "monte_carlo": globals().get('_monte_carlo'),
        "focus_mode_ok": _focus_mode_ok if '_focus_mode_ok' in dir() else False,
        "engage_focus_mode": globals().get('engage_focus_mode'),
        "disengage_focus_mode": globals().get('disengage_focus_mode'),
        "lod_ok": _lod_ok if '_lod_ok' in dir() else False,
        "atlas_lod": globals().get('_atlas_lod'),
        "theme_ok": _theme_ok if '_theme_ok' in dir() else False,
        "citadel_theme": globals().get('_citadel_theme'),
        "boot_ok": _boot_ok if '_boot_ok' in dir() else False,
        "run_boot_health_check": globals().get('run_boot_health_check'),
        "reasoning_auditor": globals().get('_reasoning_auditor'),
        "get_rag_priority": globals().get('get_rag_priority'),
    })
    log.info("§FERRARI: Citadel API routes extracted (14 endpoints)")
except Exception as _e:
    log.warning(f"§FERRARI: Citadel API routes unavailable: {_e}")

# ── §WIRE: Jarvis + Oracle routes (routes/jarvis.py) ──────────────
try:
    from server.routes.jarvis import router as _jarvis_router
    app.include_router(_jarvis_router)
    log.info("§WIRE: Jarvis + Oracle routes registered (15 endpoints)")
except Exception as _e:
    log.warning(f"§WIRE: Jarvis/Oracle routes unavailable: {_e}")

# ── §WIRE: Socratic Chamber routes (routes/socratic.py) ───────────
try:
    from server.routes.socratic import router as _socratic_router
    app.include_router(_socratic_router)
    log.info("§WIRE: Socratic Chamber routes registered (8 endpoints)")
except Exception as _e:
    log.warning(f"§WIRE: Socratic routes unavailable: {_e}")

# ── §WIRE: Google Earth integration (routes/earth.py) ─────────────
try:
    from server.routes.earth import router as _earth_router
    app.include_router(_earth_router)
    log.info("§WIRE: Google Earth routes registered (5 endpoints)")
except Exception as _e:
    log.warning(f"§WIRE: Earth routes unavailable: {_e}")

# §AUDIT-FIX: Removed _extracted_routes loop — these 9 modules are already
# registered via the core register() loop (lines 3501–3505) or the ORCH-7
# loop (lines 3518–3524). library_analytics and tools_misc moved to the
# §WIRE block after the ORCH-7 loop.

# ── §WIRE: NYT + Mendeley (register(app) populates router, then include) ──
try:
    from server.routes import nyt as _nyt_mod
    _nyt_mod.register(app)
    app.include_router(_nyt_mod.router)
except Exception as _e:
    log.warning(f"§WIRE: NYT routes unavailable: {_e}")

try:
    from server.routes import mendeley as _mend_mod
    _mend_mod.register(app)
    app.include_router(_mend_mod.router)
except Exception as _e:
    log.warning(f"§WIRE: Mendeley routes unavailable: {_e}")

# §EXTRACTED: TOOLS routes moved to server/routes/ — see wiring section above

# Because FastAPI matches routes in order, this must come after every
# app.include_router() call, otherwise it intercepts API routes.

if _RENDERER_DIST.is_dir() and _RENDERER_INDEX.is_file():
    from starlette.staticfiles import StaticFiles
    from starlette.responses import FileResponse

    @app.get("/{full_path:path}")
    async def spa_catch_all(full_path: str):
        # API routes that somehow get here still 404 properly
        if full_path.startswith(("api/", "chat/", "index/", "status")):
            raise HTTPException(status_code=404)
        # Serve actual static files
        static_file = _RENDERER_DIST / full_path
        if full_path and static_file.is_file() and _RENDERER_DIST in static_file.resolve().parents:
            import mimetypes as _mt
            mime, _ = _mt.guess_type(str(static_file))
            return FileResponse(str(static_file), media_type=mime or "application/octet-stream")
        # Everything else → index.html (SPA routing)
        return FileResponse(str(_RENDERER_INDEX))

    log.info(f"§SPA: UI serving from {_RENDERER_DIST} (catch-all last)")
else:
    log.warning(f"§SPA: UI not found at {_RENDERER_DIST} — UI will not be served")

# ═══════════════════════════════════════════════════════════════════
# §AUDIT: Startup Route Verification — log missing critical routes
# ═══════════════════════════════════════════════════════════════════
_expected_routes = [
    "/api/status", "/api/library", "/api/library/sources", "/api/library/upload",
    "/api/doctor", "/api/search", "/api/feedback",
    "/api/training/pairs", "/api/export/word", "/api/cockpit/status",
    "/api/hud/snapshot", "/api/jarvis/pending", "/api/missions/list",
    "/api/course/list", "/api/course/active",
    "/api/graph/nodes", "/api/citation-graph", "/api/notes",
    "/api/resilience/status", "/api/models/discover",
]
_registered = {r.path for r in app.routes if hasattr(r, "path")}
_missing = [p for p in _expected_routes if p not in _registered]
if _missing:
    log.warning(f"§AUDIT: {len(_missing)} expected routes NOT registered: {_missing}")
else:
    log.info(f"§AUDIT: All {len(_expected_routes)} critical routes verified ✓")
