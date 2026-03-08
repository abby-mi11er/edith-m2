import os
import sys
import time
import csv
import json
import base64
import re
import html
import hmac
import hashlib
import signal
import uuid
import tempfile
import subprocess
import sqlite3
import shutil
import io
import zipfile
from pathlib import Path
from datetime import datetime
from urllib.parse import quote as url_quote, urlparse
from collections import Counter, defaultdict

import streamlit as st
import streamlit.components.v1 as components
from dotenv import load_dotenv
from google import genai
from google.genai import types
try:
    from chroma_backend import (
        chroma_runtime_available,
        retrieve_local_sources,
        format_local_context,
        merge_sources,
    )
except Exception:
    chroma_runtime_available = None
    retrieve_local_sources = None
    format_local_context = None
    merge_sources = None
try:
    from cryptography.fernet import Fernet, InvalidToken
except Exception:
    Fernet = None
    InvalidToken = Exception

# Load .env from override, local project, cwd, or user app support.
APP_HOME_DEFAULT = Path.home() / "Library" / "Application Support" / "Edith"
DOTENV_CANDIDATES = []
dotenv_override = os.environ.get("EDITH_DOTENV_PATH")
if dotenv_override:
    DOTENV_CANDIDATES.append(Path(dotenv_override).expanduser())
DOTENV_CANDIDATES.extend(
    [
        Path(__file__).parent / ".env",
        Path.cwd() / ".env",
        APP_HOME_DEFAULT / ".env",
    ]
)
seen = set()
for env_path in DOTENV_CANDIDATES:
    key = str(env_path)
    if key in seen:
        continue
    seen.add(key)
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)

ENV_TARGET_PATH = Path(dotenv_override).expanduser() if dotenv_override else (APP_HOME_DEFAULT / ".env")

API_KEY = os.environ.get("GOOGLE_API_KEY")
STORE_ID = os.environ.get("EDITH_STORE_ID")
VAULT_ID = os.environ.get("EDITH_VAULT_ID", "").strip()
STORE_MAIN = os.environ.get("EDITH_STORE_MAIN", "").strip()
DATA_ROOT = os.environ.get("EDITH_DATA_ROOT", "")
MODEL_OVERRIDE = os.environ.get("EDITH_MODEL", "").strip()
MODEL_PROFILE_DEFAULT = os.environ.get("EDITH_MODEL_PROFILE", "latest").strip().lower()
MODEL_FALLBACKS_ENV = os.environ.get("EDITH_MODEL_FALLBACKS", "").strip()
ALLOW_PREVIEW_MODELS = os.environ.get("EDITH_ALLOW_PREVIEW_MODELS", "true").lower() == "true"
RETRIEVAL_BACKEND_DEFAULT = os.environ.get("EDITH_RETRIEVAL_BACKEND", "chroma").strip().lower()
CHROMA_DIR = os.environ.get("EDITH_CHROMA_DIR", str(Path(os.environ.get("EDITH_APP_DATA_DIR", str(Path(__file__).parent))).expanduser() / "chroma"))
CHROMA_COLLECTION = os.environ.get("EDITH_CHROMA_COLLECTION", "edith_docs").strip() or "edith_docs"
EMBED_MODEL = os.environ.get("EDITH_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()
try:
    CHROMA_TOP_K = int(os.environ.get("EDITH_CHROMA_TOP_K", "8"))
except ValueError:
    CHROMA_TOP_K = 8
try:
    CHROMA_POOL_MULTIPLIER = int(os.environ.get("EDITH_CHROMA_POOL_MULTIPLIER", "4"))
except ValueError:
    CHROMA_POOL_MULTIPLIER = 4
try:
    CHROMA_DIVERSITY_LAMBDA = float(os.environ.get("EDITH_CHROMA_DIVERSITY_LAMBDA", "0.65"))
except ValueError:
    CHROMA_DIVERSITY_LAMBDA = 0.65
try:
    CHROMA_BM25_WEIGHT = float(os.environ.get("EDITH_CHROMA_BM25_WEIGHT", "0.35"))
except ValueError:
    CHROMA_BM25_WEIGHT = 0.35
CHROMA_RERANK_MODEL = os.environ.get("EDITH_CHROMA_RERANK_MODEL", "BAAI/bge-reranker-base").strip()
CHROMA_RERANK_ENABLED_DEFAULT = os.environ.get("EDITH_CHROMA_RERANK", "true").lower() == "true"
CHROMA_FORCE_RERANK_FILES_ONLY = os.environ.get("EDITH_CHROMA_FORCE_RERANK_FILES_ONLY", "true").lower() == "true"
CHROMA_SECTION_FILTER_DEFAULT = os.environ.get("EDITH_SECTION_FILTER", "").strip()
CHROMA_DOC_TYPE_FILTER_DEFAULT = os.environ.get("EDITH_DOC_TYPE_FILTER", "").strip()
try:
    CHROMA_FAMILY_CAP = int(os.environ.get("EDITH_CHROMA_FAMILY_CAP", "2"))
except ValueError:
    CHROMA_FAMILY_CAP = 2
try:
    CHROMA_RERANK_TOP_N = int(os.environ.get("EDITH_CHROMA_RERANK_TOP_N", "14"))
except ValueError:
    CHROMA_RERANK_TOP_N = 14
SOURCE_MODE_DEFAULT = os.environ.get("EDITH_SOURCE_MODE", "Files only")
HYBRID_POLICY_DEFAULT = os.environ.get("EDITH_HYBRID_FILE_POLICY", "require_files")
PASSWORD = os.environ.get("EDITH_APP_PASSWORD", "")
PASSWORD_HASH = os.environ.get("EDITH_APP_PASSWORD_HASH", "").strip().lower()
REQUIRE_PASSWORD = os.environ.get("EDITH_REQUIRE_PASSWORD", "true").lower() == "true"
MAX_TURNS = int(os.environ.get("EDITH_CHAT_TURNS", "12"))
STREAMING_DEFAULT = os.environ.get("EDITH_STREAMING_DEFAULT", "true").lower() == "true"
CHAT_ENCRYPTION_ENABLED = os.environ.get("EDITH_CHAT_ENCRYPTION", "true").lower() == "true"
CHAT_ENCRYPTION_KEY = os.environ.get("EDITH_CHAT_ENCRYPTION_KEY", "").strip()
ALLOW_WEB_TOOLS = os.environ.get("EDITH_ALLOW_WEB_TOOLS", "false").lower() == "true"
WEB_DOMAIN_ALLOWLIST_DEFAULT = os.environ.get(
    "EDITH_WEB_DOMAIN_ALLOWLIST",
    ".gov,.edu,nature.com,science.org,arxiv.org,nih.gov,who.int,cdc.gov,oecd.org,worldbank.org",
).strip()
WEB_DOMAIN_ALLOWLIST_ENABLED_DEFAULT = (
    os.environ.get("EDITH_WEB_DOMAIN_ALLOWLIST_ENABLED", "true").lower() == "true"
)
CLOUD_INDEX_OPT_IN = os.environ.get("EDITH_CLOUD_INDEX_OPT_IN", "false").lower() == "true"
EXPORT_REDACT_DEFAULT = os.environ.get("EDITH_EXPORT_REDACT_SENSITIVE", "true").lower() == "true"
try:
    AUTO_LOCK_MINUTES = int(os.environ.get("EDITH_AUTO_LOCK_MINUTES", "20"))
except ValueError:
    AUTO_LOCK_MINUTES = 20
REQUIRE_CITATIONS = os.environ.get("EDITH_REQUIRE_CITATIONS", "true").lower() == "true"
SFT_REDACT_PII = os.environ.get("EDITH_SFT_REDACT_PII", "true").lower() == "true"
SFT_REDACT_TOKEN = (os.environ.get("EDITH_SFT_REDACT_TOKEN", "[REDACTED]") or "[REDACTED]").strip() or "[REDACTED]"
QUERY_REWRITE_DEFAULT = os.environ.get("EDITH_QUERY_REWRITE", "true").lower() == "true"
SUPPORT_AUDIT_DEFAULT = os.environ.get("EDITH_SUPPORT_AUDIT", "true").lower() == "true"
CONFIDENCE_ROUTING_DEFAULT = os.environ.get("EDITH_CONFIDENCE_ROUTING", "true").lower() == "true"
try:
    MAX_SNIPPET_CHARS = int(os.environ.get("EDITH_MAX_SOURCE_SNIPPET_CHARS", "500"))
except ValueError:
    MAX_SNIPPET_CHARS = 500
if MAX_SNIPPET_CHARS < 120:
    MAX_SNIPPET_CHARS = 120
try:
    QUERY_REWRITE_MAX = int(os.environ.get("EDITH_QUERY_REWRITE_MAX", "3"))
except ValueError:
    QUERY_REWRITE_MAX = 3
if QUERY_REWRITE_MAX < 1:
    QUERY_REWRITE_MAX = 1
if QUERY_REWRITE_MAX > 5:
    QUERY_REWRITE_MAX = 5
try:
    SUPPORT_AUDIT_MAX_SOURCES = int(os.environ.get("EDITH_SUPPORT_AUDIT_MAX_SOURCES", "8"))
except ValueError:
    SUPPORT_AUDIT_MAX_SOURCES = 8
if SUPPORT_AUDIT_MAX_SOURCES < 1:
    SUPPORT_AUDIT_MAX_SOURCES = 1
if SUPPORT_AUDIT_MAX_SOURCES > 20:
    SUPPORT_AUDIT_MAX_SOURCES = 20
try:
    CONFIDENCE_LOW_THRESHOLD = float(os.environ.get("EDITH_CONFIDENCE_LOW_THRESHOLD", "0.45"))
except ValueError:
    CONFIDENCE_LOW_THRESHOLD = 0.45
if CONFIDENCE_LOW_THRESHOLD < 0.0:
    CONFIDENCE_LOW_THRESHOLD = 0.0
if CONFIDENCE_LOW_THRESHOLD > 1.0:
    CONFIDENCE_LOW_THRESHOLD = 1.0
try:
    INLINE_CITATION_MAX = int(os.environ.get("EDITH_INLINE_CITATION_MAX", "8"))
except ValueError:
    INLINE_CITATION_MAX = 8
if INLINE_CITATION_MAX < 1:
    INLINE_CITATION_MAX = 1
if INLINE_CITATION_MAX > 20:
    INLINE_CITATION_MAX = 20
SENTENCE_PROVENANCE_DEFAULT = os.environ.get("EDITH_SENTENCE_PROVENANCE", "true").lower() == "true"
STRICT_SENTENCE_TAGS_DEFAULT = os.environ.get("EDITH_STRICT_SENTENCE_TAGS", "true").lower() == "true"
PRODUCTION_TEMPLATE_DEFAULT = os.environ.get("EDITH_PRODUCTION_TEMPLATE", "true").lower() == "true"
QUOTE_FIRST_RECALL_DEFAULT = os.environ.get("EDITH_QUOTE_FIRST_RECALL", "true").lower() == "true"
LOG_REDACT_ENABLED = os.environ.get("EDITH_LOG_REDACT", "true").lower() == "true"
CONTRADICTION_CHECK_DEFAULT = os.environ.get("EDITH_CONTRADICTION_CHECK", "true").lower() == "true"
AUTO_ADD_FEEDBACK_CASES = os.environ.get("EDITH_AUTO_ADD_FEEDBACK_CASES", "true").lower() == "true"
MULTI_PASS_DEFAULT = os.environ.get("EDITH_MULTI_PASS", "true").lower() == "true"
RECURSIVE_CONTROLLER_DEFAULT = os.environ.get("EDITH_RECURSIVE_CONTROLLER", "true").lower() == "true"
ACTION_APPROVAL_DEFAULT = os.environ.get("EDITH_ACTION_APPROVAL", "true").lower() == "true"
CONTEXT_PACKING_DEFAULT = os.environ.get("EDITH_CONTEXT_PACKING", "true").lower() == "true"
DISTILL_RETRIEVAL_QUERY_DEFAULT = os.environ.get("EDITH_DISTILL_RETRIEVAL_QUERY", "true").lower() == "true"
NEXT_QUESTIONS_DEFAULT = os.environ.get("EDITH_NEXT_QUESTIONS", "true").lower() == "true"
RESEARCHER_MODE_DEFAULT = os.environ.get("EDITH_RESEARCHER_MODE", "true").lower() == "true"
AGENT_ACTIONS_ENABLED_DEFAULT = os.environ.get("EDITH_AGENT_ACTIONS_ENABLED", "false").lower() == "true"
TOOL_ALLOWLIST_RAW = os.environ.get(
    "EDITH_TOOL_ALLOWLIST",
    "file_search,google_search,google_search_retrieval",
).strip()
WORKSPACE_ALLOWLIST_RAW = os.environ.get("EDITH_WORKSPACE_ALLOWLIST", "").strip()
RUN_LEDGER_ENABLED = os.environ.get("EDITH_RUN_LEDGER", "true").lower() == "true"
RUN_LEDGER_ENCRYPT = os.environ.get("EDITH_RUN_LEDGER_ENCRYPT", "true").lower() == "true"
RUN_LEDGER_INCLUDE_TEXT = os.environ.get("EDITH_RUN_LEDGER_INCLUDE_TEXT", "false").lower() == "true"
try:
    CHAT_RETENTION_DAYS = int(os.environ.get("EDITH_CHAT_RETENTION_DAYS", "0"))
except ValueError:
    CHAT_RETENTION_DAYS = 0
try:
    CHUNK_SIZE_TOKENS = int(os.environ.get("EDITH_CHUNK_SIZE_TOKENS", "250"))
except ValueError:
    CHUNK_SIZE_TOKENS = 250
try:
    CHUNK_OVERLAP_TOKENS = int(os.environ.get("EDITH_CHUNK_OVERLAP_TOKENS", "30"))
except ValueError:
    CHUNK_OVERLAP_TOKENS = 30
try:
    MAX_FILE_MB = int(os.environ.get("EDITH_MAX_FILE_MB", "50"))
except ValueError:
    MAX_FILE_MB = 50
try:
    REINDEX_TIMEOUT_SECONDS = int(os.environ.get("EDITH_REINDEX_TIMEOUT_SECONDS", "1800"))
except ValueError:
    REINDEX_TIMEOUT_SECONDS = 1800
try:
    MAX_QUERY_CHARS = int(os.environ.get("EDITH_MAX_QUERY_CHARS", "4000"))
except ValueError:
    MAX_QUERY_CHARS = 4000
try:
    RECURSIVE_CONTROLLER_MAX_DEPTH = int(os.environ.get("EDITH_RECURSIVE_MAX_DEPTH", "2"))
except ValueError:
    RECURSIVE_CONTROLLER_MAX_DEPTH = 2
try:
    RECURSIVE_CONTROLLER_BATCH_SIZE = int(os.environ.get("EDITH_RECURSIVE_BATCH_SIZE", "6"))
except ValueError:
    RECURSIVE_CONTROLLER_BATCH_SIZE = 6
try:
    RECURSIVE_CONTROLLER_MAX_BATCHES = int(os.environ.get("EDITH_RECURSIVE_MAX_BATCHES", "6"))
except ValueError:
    RECURSIVE_CONTROLLER_MAX_BATCHES = 6
try:
    RECURSIVE_CONTROLLER_MIN_SOURCES = int(os.environ.get("EDITH_RECURSIVE_MIN_SOURCES", "14"))
except ValueError:
    RECURSIVE_CONTROLLER_MIN_SOURCES = 14
try:
    RECURSIVE_CONTROLLER_MAX_CALLS = int(os.environ.get("EDITH_RECURSIVE_MAX_CALLS", "18"))
except ValueError:
    RECURSIVE_CONTROLLER_MAX_CALLS = 18
if RECURSIVE_CONTROLLER_MAX_DEPTH < 1:
    RECURSIVE_CONTROLLER_MAX_DEPTH = 1
if RECURSIVE_CONTROLLER_MAX_DEPTH > 4:
    RECURSIVE_CONTROLLER_MAX_DEPTH = 4
if RECURSIVE_CONTROLLER_BATCH_SIZE < 2:
    RECURSIVE_CONTROLLER_BATCH_SIZE = 2
if RECURSIVE_CONTROLLER_BATCH_SIZE > 12:
    RECURSIVE_CONTROLLER_BATCH_SIZE = 12
if RECURSIVE_CONTROLLER_MAX_BATCHES < 2:
    RECURSIVE_CONTROLLER_MAX_BATCHES = 2
if RECURSIVE_CONTROLLER_MAX_BATCHES > 12:
    RECURSIVE_CONTROLLER_MAX_BATCHES = 12
if RECURSIVE_CONTROLLER_MIN_SOURCES < 4:
    RECURSIVE_CONTROLLER_MIN_SOURCES = 4
if RECURSIVE_CONTROLLER_MIN_SOURCES > 80:
    RECURSIVE_CONTROLLER_MIN_SOURCES = 80
if RECURSIVE_CONTROLLER_MAX_CALLS < 4:
    RECURSIVE_CONTROLLER_MAX_CALLS = 4
if RECURSIVE_CONTROLLER_MAX_CALLS > 48:
    RECURSIVE_CONTROLLER_MAX_CALLS = 48
RATE_LIMIT_ENABLED = os.environ.get("EDITH_RATE_LIMIT_ENABLED", "true").lower() == "true"
try:
    RATE_LIMIT_CHAT_MAX = int(os.environ.get("EDITH_RATE_LIMIT_CHAT_MAX", "24"))
except ValueError:
    RATE_LIMIT_CHAT_MAX = 24
try:
    RATE_LIMIT_CHAT_WINDOW_SECONDS = int(os.environ.get("EDITH_RATE_LIMIT_CHAT_WINDOW_SECONDS", "60"))
except ValueError:
    RATE_LIMIT_CHAT_WINDOW_SECONDS = 60
try:
    RATE_LIMIT_MUTATION_MAX = int(os.environ.get("EDITH_RATE_LIMIT_MUTATION_MAX", "12"))
except ValueError:
    RATE_LIMIT_MUTATION_MAX = 12
try:
    RATE_LIMIT_MUTATION_WINDOW_SECONDS = int(
        os.environ.get("EDITH_RATE_LIMIT_MUTATION_WINDOW_SECONDS", "300")
    )
except ValueError:
    RATE_LIMIT_MUTATION_WINDOW_SECONDS = 300
REQUIRE_HTTPS_WEB_SOURCES = os.environ.get("EDITH_REQUIRE_HTTPS_WEB_SOURCES", "true").lower() == "true"
OAUTH_REQUIRED = os.environ.get("EDITH_OAUTH_REQUIRED", "false").lower() == "true"
OAUTH_HEADER = os.environ.get("EDITH_OAUTH_HEADER", "X-Forwarded-Email").strip() or "X-Forwarded-Email"
OAUTH_TRUSTED_EMAILS_RAW = os.environ.get("EDITH_OAUTH_TRUSTED_EMAILS", "").strip()
RBAC_DEFAULT_ROLE = os.environ.get("EDITH_RBAC_DEFAULT_ROLE", "admin").strip().lower() or "admin"
RBAC_EMAIL_ROLES_RAW = os.environ.get("EDITH_RBAC_EMAIL_ROLES", "").strip()

if STORE_ID and not STORE_ID.startswith("fileSearchStores/"):
    STORE_ID = f"fileSearchStores/{STORE_ID}"
if VAULT_ID and not VAULT_ID.startswith("fileSearchStores/"):
    VAULT_ID = f"fileSearchStores/{VAULT_ID}"
if STORE_MAIN and not STORE_MAIN.startswith("fileSearchStores/"):
    STORE_MAIN = f"fileSearchStores/{STORE_MAIN}"
if MAX_TURNS < 2:
    MAX_TURNS = 2
if CHUNK_SIZE_TOKENS < 50:
    CHUNK_SIZE_TOKENS = 50
if CHUNK_OVERLAP_TOKENS < 0:
    CHUNK_OVERLAP_TOKENS = 0
if CHUNK_OVERLAP_TOKENS >= CHUNK_SIZE_TOKENS:
    CHUNK_OVERLAP_TOKENS = max(0, CHUNK_SIZE_TOKENS // 5)
if RETRIEVAL_BACKEND_DEFAULT not in ("google", "chroma"):
    RETRIEVAL_BACKEND_DEFAULT = "google"
if CHROMA_TOP_K < 1:
    CHROMA_TOP_K = 1
if CHROMA_TOP_K > 30:
    CHROMA_TOP_K = 30
if CHROMA_POOL_MULTIPLIER < 1:
    CHROMA_POOL_MULTIPLIER = 1
if CHROMA_POOL_MULTIPLIER > 12:
    CHROMA_POOL_MULTIPLIER = 12
if CHROMA_DIVERSITY_LAMBDA < 0.0:
    CHROMA_DIVERSITY_LAMBDA = 0.0
if CHROMA_DIVERSITY_LAMBDA > 1.0:
    CHROMA_DIVERSITY_LAMBDA = 1.0
if CHROMA_BM25_WEIGHT < 0.0:
    CHROMA_BM25_WEIGHT = 0.0
if CHROMA_BM25_WEIGHT > 1.0:
    CHROMA_BM25_WEIGHT = 1.0
if CHROMA_RERANK_TOP_N < 1:
    CHROMA_RERANK_TOP_N = 1
if CHROMA_RERANK_TOP_N > 40:
    CHROMA_RERANK_TOP_N = 40
if CHROMA_FAMILY_CAP < 1:
    CHROMA_FAMILY_CAP = 1
if CHROMA_FAMILY_CAP > 8:
    CHROMA_FAMILY_CAP = 8
if MAX_FILE_MB < 1:
    MAX_FILE_MB = 1
if MAX_FILE_MB > 1024:
    MAX_FILE_MB = 1024
if REINDEX_TIMEOUT_SECONDS < 60:
    REINDEX_TIMEOUT_SECONDS = 60
if REINDEX_TIMEOUT_SECONDS > 43200:
    REINDEX_TIMEOUT_SECONDS = 43200
if MAX_QUERY_CHARS < 200:
    MAX_QUERY_CHARS = 200
if MAX_QUERY_CHARS > 20000:
    MAX_QUERY_CHARS = 20000
if RATE_LIMIT_CHAT_MAX < 1:
    RATE_LIMIT_CHAT_MAX = 1
if RATE_LIMIT_CHAT_MAX > 500:
    RATE_LIMIT_CHAT_MAX = 500
if RATE_LIMIT_CHAT_WINDOW_SECONDS < 5:
    RATE_LIMIT_CHAT_WINDOW_SECONDS = 5
if RATE_LIMIT_CHAT_WINDOW_SECONDS > 3600:
    RATE_LIMIT_CHAT_WINDOW_SECONDS = 3600
if RATE_LIMIT_MUTATION_MAX < 1:
    RATE_LIMIT_MUTATION_MAX = 1
if RATE_LIMIT_MUTATION_MAX > 200:
    RATE_LIMIT_MUTATION_MAX = 200
if RATE_LIMIT_MUTATION_WINDOW_SECONDS < 10:
    RATE_LIMIT_MUTATION_WINDOW_SECONDS = 10
if RATE_LIMIT_MUTATION_WINDOW_SECONDS > 86400:
    RATE_LIMIT_MUTATION_WINDOW_SECONDS = 86400
if AUTO_LOCK_MINUTES < 1:
    AUTO_LOCK_MINUTES = 1
if AUTO_LOCK_MINUTES > 480:
    AUTO_LOCK_MINUTES = 480

client = genai.Client(api_key=API_KEY) if API_KEY else None

SYSTEM = (
    "You are a file-only research assistant. "
    "Use only information supported by retrieved sources. "
    "If the answer is not in the sources, reply: 'Not found in sources.'"
)

UPLOAD_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".txt", ".md", ".rtf", ".odt", ".tex",
    ".csv", ".tsv", ".xlsx", ".xls", ".json", ".jsonl",
}

CHUNK = {"max_tokens_per_chunk": CHUNK_SIZE_TOKENS, "max_overlap_tokens": CHUNK_OVERLAP_TOKENS}
IS_FROZEN_APP = bool(getattr(sys, "frozen", False))
DESKTOP_MODE = os.environ.get("EDITH_DESKTOP_MODE", "").strip().lower() == "electron"
APP_STATE_DIR = Path(
    os.environ.get("EDITH_APP_DATA_DIR", str(Path(__file__).parent))
).expanduser()
APP_STATE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_REPORT = APP_STATE_DIR / "edith_index_report.csv"
INDEX_STATUS_PATH = APP_STATE_DIR / "index_status.json"
INDEX_DB_PATH = APP_STATE_DIR / "edith_index.sqlite3"
WATCH_PID_PATH = APP_STATE_DIR / ".edith_watch.pid"
WATCH_LOG_PATH = APP_STATE_DIR / ".edith_watch.log"
CHAT_DIR = APP_STATE_DIR / "chat_history"
CHAT_KEY_PATH = CHAT_DIR / ".chat.key"
RUN_LEDGER_PATH = Path(os.environ.get("EDITH_RUN_LEDGER_PATH", str(APP_STATE_DIR / "run_ledger.jsonl"))).expanduser()
FEEDBACK_DB_PATH = APP_STATE_DIR / "feedback.sqlite3"
RETRIEVAL_PROFILE_PATH = APP_STATE_DIR / "retrieval_profile.json"
GLOSSARY_GRAPH_PATH = APP_STATE_DIR / "glossary_graph.json"
CITATION_GRAPH_PATH = APP_STATE_DIR / "citation_graph.json"
CHAPTER_ANCHORS_PATH = APP_STATE_DIR / "chapter_anchors.json"
CLAIM_INVENTORY_PATH = APP_STATE_DIR / "claim_inventory.json"
EXPERIMENT_LEDGER_PATH = APP_STATE_DIR / "experiment_ledger.json"
INDEX_HEALTH_REPORT_PATH = APP_STATE_DIR / "index_health_report.json"
BIBLIOGRAPHY_DB_PATH = APP_STATE_DIR / "bibliography_db.json"
ENTITY_TIMELINE_PATH = APP_STATE_DIR / "entity_timeline.json"
SNAPSHOT_DIR = APP_STATE_DIR / "snapshots"
WEB_CACHE_PATH = APP_STATE_DIR / "web_cache.json"
TAL_TOKENS_PATH = APP_STATE_DIR / "tal_capability_tokens.json"
TAL_AUDIT_PATH = APP_STATE_DIR / "tal_audit.jsonl"
RESEARCH_NOTEBOOK_PATH = APP_STATE_DIR / "research_notebook.jsonl"
try:
    WEB_CACHE_MAX_ENTRIES = int(os.environ.get("EDITH_WEB_CACHE_MAX_ENTRIES", "1200"))
except ValueError:
    WEB_CACHE_MAX_ENTRIES = 1200
try:
    TAL_WEB_ONCE_TTL_SECONDS = int(os.environ.get("EDITH_TAL_WEB_ONCE_TTL_SECONDS", "900"))
except ValueError:
    TAL_WEB_ONCE_TTL_SECONDS = 900
try:
    TAL_WEB_CHAT_TTL_SECONDS = int(os.environ.get("EDITH_TAL_WEB_CHAT_TTL_SECONDS", "28800"))
except ValueError:
    TAL_WEB_CHAT_TTL_SECONDS = 28800
try:
    TAL_WEB_CHAT_MAX_CALLS = int(os.environ.get("EDITH_TAL_WEB_CHAT_MAX_CALLS", "40"))
except ValueError:
    TAL_WEB_CHAT_MAX_CALLS = 40
if WEB_CACHE_MAX_ENTRIES < 100:
    WEB_CACHE_MAX_ENTRIES = 100
if TAL_WEB_ONCE_TTL_SECONDS < 60:
    TAL_WEB_ONCE_TTL_SECONDS = 60
if TAL_WEB_ONCE_TTL_SECONDS > 86400:
    TAL_WEB_ONCE_TTL_SECONDS = 86400
if TAL_WEB_CHAT_TTL_SECONDS < 300:
    TAL_WEB_CHAT_TTL_SECONDS = 300
if TAL_WEB_CHAT_TTL_SECONDS > 604800:
    TAL_WEB_CHAT_TTL_SECONDS = 604800
if TAL_WEB_CHAT_MAX_CALLS < 1:
    TAL_WEB_CHAT_MAX_CALLS = 1
if TAL_WEB_CHAT_MAX_CALLS > 200:
    TAL_WEB_CHAT_MAX_CALLS = 200

QUICK_PROMPTS = [
    "Summarize the most important themes in my files.",
    "List the key documents related to a specific topic.",
    "Create a short literature-style summary from my documents.",
    "Find any disagreements or contradictions across my files.",
]
ANSWER_TYPE_OPTIONS = ["Explain", "Compare", "Critique", "Draft", "Flashcards", "Table"]
SOURCE_QUALITY_OPTIONS = ["Precise", "Balanced", "Exhaustive"]
ANSWER_TYPE_HINTS = {
    "Explain": "",
    "Compare": "Answer type: compare relevant sources side-by-side and call out key differences with citations.",
    "Critique": "Answer type: provide a critical appraisal with strengths, limitations, assumptions, and evidence quality.",
    "Draft": "Answer type: produce a polished draft response suitable for academic writing, preserving grounded citations.",
    "Flashcards": "Answer type: produce concise study flashcards (Q/A or cloze style) with source-backed facts.",
    "Table": "Answer type: present the answer as a structured table with rows grounded in cited evidence.",
}
PRESET_QUICK_PROMPTS = {
    "study": [
        "Explain this concept in plain language, then in academic language with citations.",
        "Create a quick quiz from my files on this topic.",
        "List common confusions and key distinctions for this topic with sources.",
        "Generate 15 Anki flashcards from the strongest evidence in my files.",
    ],
    "paper breakdown": [
        "Break down this paper by section and extract thesis, method, data, findings, and limitations with citations.",
        "Create a compact methods and data table for this paper with citations.",
        "Write a one-paragraph summary of this paper's contribution with citations.",
        "Generate 10 discussion questions grounded in this paper.",
    ],
    "synthesis": [
        "Synthesize what my files say about this topic: agreements, disagreements, gaps, and next questions.",
        "Compare 2-5 key sources on this topic by theory, method, data, and findings.",
        "Find contradictions across my sources and cite both sides.",
        "Build a literature review outline with citations and key sources.",
    ],
    "academic writing": [
        "Rewrite my previous answer into thesis-style academic prose with careful hedging and citations.",
        "Tighten the argument structure and keep all citation labels.",
        "Create a sectioned outline for writing this into a paper chapter.",
        "Draft a related-work paragraph with bundled citations.",
    ],
    "production": [
        "From sources, list assumptions and constraints. Then provide a clearly labeled proposed plan and risks/unknowns.",
        "Generate a production checklist grounded in sources and mark unsupported items clearly.",
        "Identify what evidence is missing for a high-confidence production recommendation.",
        "Create a concise decision memo with citations and explicit uncertainty labels.",
    ],
}
PRESET_ACTION_PROMPTS = {
    "study": [
        ("Make flashcards", "Generate 20 Anki flashcards (mix of basic and cloze) from the strongest evidence in my files."),
        ("Quiz me", "Create a 10-question quiz from my files on this topic. Include answer key and citations."),
        ("Give examples", "Provide 5 concrete examples grounded in my sources, each with a citation."),
        ("Compare concepts", "Compare the two most relevant concepts for this topic and cite the best supporting sources."),
    ],
    "paper breakdown": [
        ("Extract methods table", "Extract a methods and data table for this paper with citations."),
        ("1-par summary", "Write one paragraph on this paper's core contribution with citations."),
        ("Discussion questions", "Generate 10 discussion questions for this paper with brief evidence hints."),
        ("Paper flashcards", "Generate 15 grounded flashcards from this paper's thesis, method, and findings."),
    ],
    "synthesis": [
        ("Build lit review", "Draft a literature review outline from my sources with themes, disagreements, and gaps."),
        ("Compare papers", "Compare 2-5 key sources by theory, method, data, and findings with citations."),
        ("Find contradictions", "Identify contradictions across my sources and cite both sides."),
        ("Read next", "Recommend what to read next from my library based on current gaps."),
    ],
    "academic writing": [
        ("Rewrite academic", "Rewrite your previous answer in stronger academic prose while preserving citation labels."),
        ("Add citations", "Strengthen citation coverage sentence-by-sentence and keep claims source-grounded."),
        ("Tighten argument", "Tighten argument flow and remove weak or unsupported claims."),
        ("Make outline", "Create a thesis-style outline from your previous answer with section headings and citations."),
    ],
}
MODE_PRESETS = [
    "Custom",
    "Grounded (strict)",
    "Study",
    "Paper Breakdown",
    "Synthesis",
    "Academic Writing",
    "Production",
]
SOURCE_MODES = ["Files only", "Web only", "Files + Web"]
HYBRID_FILE_POLICIES = ["flexible", "prefer_files", "require_files"]
RETRIEVAL_BACKEND_LABELS = {
    "Google File Search": "google",
    "Local Chroma": "chroma",
}
MODEL_PROFILES = ["latest", "balanced", "stable", "fast"]
MODEL_PROFILE_CHAINS = {
    "latest": [
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ],
    "balanced": [
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ],
    "stable": [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
        "gemini-2.5-flash-lite",
    ],
    "fast": [
        "gemini-2.5-flash-lite",
        "gemini-2.5-flash",
        "gemini-3-flash-preview",
    ],
}

CHAT_DIR.mkdir(parents=True, exist_ok=True)


def prompt_examples_for_preset(name: str):
    key = clean_text(name or "").strip().lower()
    return PRESET_QUICK_PROMPTS.get(key) or QUICK_PROMPTS


def action_prompts_for_preset(name: str):
    key = clean_text(name or "").strip().lower()
    return PRESET_ACTION_PROMPTS.get(key) or []


def answer_type_hint(value: str):
    key = clean_text(value or "").strip().title()
    return ANSWER_TYPE_HINTS.get(key, "")


def system_prompt_for_mode(source_mode: str, hybrid_policy: str = "require_files", require_citations: bool = True):
    if require_citations:
        base = (
            "Use only information supported by retrieved sources. "
            "If support is missing, reply exactly: Not found in sources. "
            "When source metadata includes author/year, reference them by name in the answer."
        )
    else:
        base = (
            "Prefer information supported by retrieved sources. "
            "When support is incomplete, explicitly label uncertainty. "
            "When source metadata includes author/year, reference them by name in the answer."
        )
    if source_mode == "Files only":
        return "You are a file-only research assistant. " + base
    if source_mode == "Web only":
        return "You are a web-research assistant. Use web sources only. " + base
    if hybrid_policy == "require_files":
        return (
            "You are a hybrid research assistant using both files and web sources. "
            "Every final answer must include support from at least one file source. "
            + base
        )
    if hybrid_policy == "prefer_files":
        return (
            "You are a hybrid research assistant using both files and web sources. "
            "Prefer file sources over web sources whenever possible. "
            + base
        )
    return "You are a hybrid research assistant using both files and web sources. " + base


def parse_csv_list(value: str):
    if not value:
        return []
    out = []
    for part in value.split(","):
        p = part.strip()
        if p:
            out.append(p)
    return out


def preset_defaults(name: str):
    preset = (name or "").strip().lower()
    if preset == "grounded (strict)":
        return {
            "source_mode": "Files only",
            "strict_citations": True,
            "hybrid_policy": "require_files",
            "query_rewrite_on": True,
            "support_audit_on": True,
            "sentence_provenance_on": True,
            "strict_sentence_tags_on": True,
            "confidence_routing_on": True,
            "researcher_mode": False,
            "verbosity_level": "standard",
            "writing_style": "academic",
            "include_methods_table": False,
            "include_limitations": True,
            "next_questions_on": True,
        }
    if preset == "study":
        return {
            "source_mode": "Files only",
            "strict_citations": True,
            "hybrid_policy": "require_files",
            "query_rewrite_on": True,
            "support_audit_on": True,
            "sentence_provenance_on": False,
            "strict_sentence_tags_on": False,
            "confidence_routing_on": True,
            "researcher_mode": False,
            "verbosity_level": "standard",
            "writing_style": "plain",
            "include_methods_table": False,
            "include_limitations": True,
            "next_questions_on": True,
        }
    if preset == "paper breakdown":
        return {
            "source_mode": "Files only",
            "strict_citations": True,
            "hybrid_policy": "require_files",
            "query_rewrite_on": True,
            "support_audit_on": True,
            "sentence_provenance_on": True,
            "strict_sentence_tags_on": True,
            "confidence_routing_on": True,
            "researcher_mode": True,
            "verbosity_level": "deep",
            "writing_style": "academic",
            "include_methods_table": True,
            "include_limitations": True,
            "next_questions_on": True,
        }
    if preset == "synthesis":
        return {
            "source_mode": "Files only",
            "strict_citations": True,
            "hybrid_policy": "prefer_files",
            "query_rewrite_on": True,
            "support_audit_on": True,
            "sentence_provenance_on": True,
            "strict_sentence_tags_on": True,
            "confidence_routing_on": True,
            "researcher_mode": True,
            "verbosity_level": "deep",
            "writing_style": "academic",
            "include_methods_table": True,
            "include_limitations": True,
            "next_questions_on": True,
        }
    if preset == "academic writing":
        return {
            "source_mode": "Files only",
            "strict_citations": True,
            "hybrid_policy": "prefer_files",
            "query_rewrite_on": True,
            "support_audit_on": True,
            "sentence_provenance_on": True,
            "strict_sentence_tags_on": True,
            "confidence_routing_on": True,
            "researcher_mode": True,
            "verbosity_level": "deep",
            "writing_style": "academic",
            "include_methods_table": False,
            "include_limitations": True,
            "next_questions_on": True,
        }
    if preset == "production":
        return {
            "source_mode": "Files only",
            "strict_citations": True,
            "hybrid_policy": "require_files",
            "query_rewrite_on": True,
            "support_audit_on": True,
            "sentence_provenance_on": True,
            "strict_sentence_tags_on": True,
            "confidence_routing_on": True,
            "researcher_mode": True,
            "verbosity_level": "deep",
            "writing_style": "academic",
            "include_methods_table": True,
            "include_limitations": True,
            "next_questions_on": True,
        }
    return {}


STORE_ID_VALUE_RE = re.compile(r"^[A-Za-z0-9._-]+$")
GOOGLE_API_KEY_RE = re.compile(r"^AIza[0-9A-Za-z_-]{20,}$")
SAFE_UPLOAD_NAME_RE = re.compile(r"[^A-Za-z0-9._ -]+")
ROLE_NAMES = ("viewer", "editor", "admin")
ROLE_PERMISSIONS = {
    "viewer": {
        "chat.ask",
        "library.view",
        "export.read",
    },
    "editor": {
        "chat.ask",
        "library.view",
        "export.read",
        "web.search",
        "files.upload",
        "index.run",
        "watcher.manage",
        "vault.sync",
        "vault.list",
        "privacy.update",
    },
    "admin": {
        "chat.ask",
        "library.view",
        "export.read",
        "web.search",
        "files.upload",
        "index.run",
        "watcher.manage",
        "vault.sync",
        "vault.list",
        "privacy.update",
        "settings.connection",
        "data.delete",
        "data.reset",
        "tal.manage",
    },
}


def clean_env_scalar(value: str, max_len: int = 4096):
    raw = str(value or "")
    raw = raw.replace("\x00", "").replace("\r", " ").replace("\n", " ").strip()
    if len(raw) > max_len:
        raw = raw[:max_len]
    return raw


def normalize_store_id(value: str):
    s = clean_env_scalar(value, max_len=512)
    if not s:
        return ""
    raw_id = s.split("/", 1)[1] if s.startswith("fileSearchStores/") else s
    if not STORE_ID_VALUE_RE.fullmatch(raw_id):
        return ""
    return f"fileSearchStores/{raw_id}"


def friendly_store_display(value: str):
    sid = normalize_store_id(value)
    if not sid:
        return "not set"
    core = sid.split("/", 1)[1]
    if len(core) <= 28:
        return core
    return f"{core[:12]}...{core[-8:]}"


def valid_google_api_key_format(value: str):
    token = clean_env_scalar(value, max_len=6000)
    if not token:
        return False
    return bool(GOOGLE_API_KEY_RE.fullmatch(token))


def normalize_data_root_path(value: str):
    text = clean_env_scalar(value, max_len=4096)
    if not text:
        return ""
    try:
        return str(Path(text).expanduser().resolve())
    except Exception:
        return text


def normalize_role_name(value: str):
    role = clean_env_scalar(value, max_len=32).lower()
    if role in ROLE_NAMES:
        return role
    return "admin"


def parse_oauth_trusted_emails(raw: str):
    out = []
    for token in str(raw or "").split(","):
        email = clean_env_scalar(token, max_len=320).lower()
        if email and "@" in email and email not in out:
            out.append(email)
    return out


def parse_rbac_email_roles(raw: str):
    mapping = {}
    for token in str(raw or "").split(","):
        part = clean_env_scalar(token, max_len=400)
        if not part or ":" not in part:
            continue
        email_raw, role_raw = part.split(":", 1)
        email = clean_env_scalar(email_raw, max_len=320).lower()
        role = normalize_role_name(role_raw)
        if email and "@" in email:
            mapping[email] = role
    return mapping


def oauth_header_value(header_name: str):
    try:
        headers = st.context.headers
    except Exception:
        headers = {}
    target = clean_env_scalar(header_name, max_len=120).lower()
    if not target:
        return ""
    for key, value in dict(headers or {}).items():
        if str(key).lower() != target:
            continue
        if isinstance(value, (list, tuple)):
            value = value[0] if value else ""
        return clean_env_scalar(value, max_len=320).lower()
    return ""


def resolve_oauth_identity():
    if not OAUTH_REQUIRED:
        return {
            "required": False,
            "ok": True,
            "email": "",
            "reason": "",
        }
    email = oauth_header_value(OAUTH_HEADER)
    if not email:
        return {
            "required": True,
            "ok": False,
            "email": "",
            "reason": f"Missing OAuth identity header: {OAUTH_HEADER}",
        }
    trusted = parse_oauth_trusted_emails(OAUTH_TRUSTED_EMAILS_RAW)
    if trusted and email not in trusted:
        return {
            "required": True,
            "ok": False,
            "email": email,
            "reason": "OAuth identity is not in EDITH_OAUTH_TRUSTED_EMAILS.",
        }
    return {
        "required": True,
        "ok": True,
        "email": email,
        "reason": "",
    }


def resolve_session_role(email: str):
    role_map = parse_rbac_email_roles(RBAC_EMAIL_ROLES_RAW)
    email_l = clean_env_scalar(email, max_len=320).lower()
    if email_l and email_l in role_map:
        return role_map[email_l]
    return normalize_role_name(RBAC_DEFAULT_ROLE)


def role_has_permission(role: str, permission: str):
    role_norm = normalize_role_name(role)
    perms = ROLE_PERMISSIONS.get(role_norm, ROLE_PERMISSIONS["viewer"])
    return permission in perms


def consume_rate_limit(action: str, limit_count: int, window_seconds: int):
    if not RATE_LIMIT_ENABLED:
        return {"allowed": True, "remaining": max(0, int(limit_count)), "retry_after_s": 0}
    now_ts = float(time.time())
    key = f"rate_limit_{clean_env_scalar(action, max_len=64).lower()}"
    history = st.session_state.get(key, [])
    valid = []
    for item in history:
        try:
            ts = float(item)
        except Exception:
            continue
        if (now_ts - ts) < float(window_seconds):
            valid.append(ts)
    allowed = len(valid) < int(limit_count)
    retry_after = 0
    if allowed:
        valid.append(now_ts)
    elif valid:
        retry_after = max(1, int(float(window_seconds) - (now_ts - valid[0])))
    st.session_state[key] = valid
    remaining = max(0, int(limit_count) - len(valid))
    return {"allowed": allowed, "remaining": remaining, "retry_after_s": retry_after}


def enforce_rate_limit(action: str, limit_count: int, window_seconds: int, label: str):
    state = consume_rate_limit(action, limit_count, window_seconds)
    if state.get("allowed"):
        return True
    retry_after = int(state.get("retry_after_s") or 0)
    if retry_after > 0:
        st.warning(f"{label} is temporarily rate-limited. Try again in {retry_after}s.")
    else:
        st.warning(f"{label} is temporarily rate-limited.")
    return False


def env_render(value: str):
    s = (value or "").strip()
    if not s:
        return ""
    if re.search(r'[\s#"]', s):
        escaped = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return s


def upsert_env_values(path: Path, updates):
    path.parent.mkdir(parents=True, exist_ok=True)
    existing_lines = []
    if path.exists():
        try:
            existing_lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        except Exception:
            existing_lines = []

    remaining = dict(updates)
    out_lines = []
    for line in existing_lines:
        stripped = line.lstrip()
        if stripped.startswith("#") or "=" not in line:
            out_lines.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in remaining:
            out_lines.append(f"{key}={env_render(str(remaining.pop(key)))}\n")
        else:
            out_lines.append(line)

    for key, value in remaining.items():
        out_lines.append(f"{key}={env_render(str(value))}\n")

    path.write_text("".join(out_lines), encoding="utf-8")
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


def setup_required():
    needs = []
    api_value = (API_KEY or "").strip()
    if not api_value:
        needs.append("Google API key")
    elif not valid_google_api_key_format(api_value):
        needs.append("Google API key (format invalid)")
    if REQUIRE_PASSWORD and not ((PASSWORD or "").strip() or (PASSWORD_HASH or "").strip()):
        needs.append("Edith password")
    backend = (RETRIEVAL_BACKEND_DEFAULT or "google").strip().lower()
    require_store_fields = backend == "google"
    if require_store_fields:
        if not normalize_store_id(STORE_ID):
            needs.append("Edith storage id")
        if not normalize_store_id(VAULT_ID):
            needs.append("Edith vault id")
    return needs


def hash_password_pbkdf2(password: str, iterations: int = 260000):
    salt_hex = os.urandom(16).hex()
    digest_hex = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        bytes.fromhex(salt_hex),
        int(iterations),
    ).hex()
    return f"pbkdf2_sha256${int(iterations)}${salt_hex}${digest_hex}"


def verify_password_hash(candidate: str, stored_hash: str):
    value = (stored_hash or "").strip()
    if not value:
        return False
    if value.startswith("pbkdf2_sha256$"):
        parts = value.split("$")
        if len(parts) != 4:
            return False
        _, iter_text, salt_hex, digest_hex = parts
        try:
            iterations = int(iter_text)
            if iterations < 100000 or iterations > 1000000:
                return False
            salt = bytes.fromhex(salt_hex)
            expected = hashlib.pbkdf2_hmac(
                "sha256",
                candidate.encode("utf-8"),
                salt,
                iterations,
            ).hex()
        except (ValueError, TypeError):
            return False
        return hmac.compare_digest(expected, digest_hex.lower())
    digest = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
    return hmac.compare_digest(digest, value.lower())


def save_connection_settings(api_key_value: str, vault_value: str, store_value: str, password_value: str = "", data_root_value: str = ""):
    safe_api_key = clean_env_scalar(api_key_value, max_len=6000)
    safe_password = clean_env_scalar(password_value, max_len=512)
    safe_data_root = normalize_data_root_path(data_root_value)
    if safe_api_key and (not valid_google_api_key_format(safe_api_key)):
        raise ValueError("Google API key format is invalid.")
    updates = {
        "GOOGLE_API_KEY": safe_api_key,
        "EDITH_VAULT_ID": normalize_store_id(vault_value),
        "EDITH_STORE_MAIN": normalize_store_id(store_value),
        "EDITH_STORE_ID": normalize_store_id(store_value),
        "EDITH_REQUIRE_PASSWORD": "true",
    }
    if safe_data_root:
        updates["EDITH_DATA_ROOT"] = safe_data_root
    if safe_password:
        updates["EDITH_APP_PASSWORD_HASH"] = hash_password_pbkdf2(safe_password)
        updates["EDITH_APP_PASSWORD"] = ""

    upsert_env_values(ENV_TARGET_PATH, updates)
    for key, value in updates.items():
        os.environ[key] = str(value)
    return updates


def persist_runtime_policy_settings(**kwargs):
    updates = {}
    if "allow_web_tools" in kwargs:
        updates["EDITH_ALLOW_WEB_TOOLS"] = "true" if kwargs.get("allow_web_tools") else "false"
    if "cloud_index_opt_in" in kwargs:
        updates["EDITH_CLOUD_INDEX_OPT_IN"] = "true" if kwargs.get("cloud_index_opt_in") else "false"
    if "web_domain_allowlist_enabled" in kwargs:
        updates["EDITH_WEB_DOMAIN_ALLOWLIST_ENABLED"] = (
            "true" if kwargs.get("web_domain_allowlist_enabled") else "false"
        )
    if "web_domain_allowlist" in kwargs:
        updates["EDITH_WEB_DOMAIN_ALLOWLIST"] = clean_env_scalar(
            kwargs.get("web_domain_allowlist") or "",
            max_len=2000,
        )
    if "export_redact_sensitive" in kwargs:
        updates["EDITH_EXPORT_REDACT_SENSITIVE"] = (
            "true" if kwargs.get("export_redact_sensitive") else "false"
        )
    if "auto_lock_minutes" in kwargs:
        try:
            mins = int(kwargs.get("auto_lock_minutes"))
        except Exception:
            mins = AUTO_LOCK_MINUTES
        mins = max(1, min(480, mins))
        updates["EDITH_AUTO_LOCK_MINUTES"] = str(mins)
    if not updates:
        return {}
    upsert_env_values(ENV_TARGET_PATH, updates)
    for key, value in updates.items():
        os.environ[key] = str(value)
    return updates


def stable_hash(text: str):
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def clamp_int(value, low: int, high: int):
    try:
        v = int(value)
    except Exception:
        v = low
    return max(low, min(high, v))


def clamp_float(value, low: float, high: float):
    try:
        v = float(value)
    except Exception:
        v = low
    return max(low, min(high, v))


PII_PATTERNS = [
    (re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b"), "email"),
    (re.compile(r"\b(?:\+?1[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"), "phone"),
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "ssn"),
    (re.compile(r"\b(?:\d[ -]*?){13,16}\b"), "card"),
    (re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"), "api_key"),
    (re.compile(r"\bAIza[0-9A-Za-z\\-_]{20,}\b"), "api_key"),
]


def redact_pii_text(text: str, enabled: bool = True, replacement: str = "[REDACTED]"):
    s = str(text or "")
    if not enabled or not s:
        return s
    out = s
    for rx, _kind in PII_PATTERNS:
        out = rx.sub(replacement, out)
    return out


def sanitize_for_logs(text: str):
    return redact_pii_text(text, enabled=LOG_REDACT_ENABLED, replacement="[REDACTED]")


def now_iso():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def default_retrieval_profile():
    return {
        "top_k": clamp_int(CHROMA_TOP_K, 4, 20),
        "bm25_weight": clamp_float(CHROMA_BM25_WEIGHT, 0.05, 0.95),
        "diversity_lambda": clamp_float(CHROMA_DIVERSITY_LAMBDA, 0.1, 0.95),
        "rerank_top_n": clamp_int(CHROMA_RERANK_TOP_N, 6, 40),
        "rerank_on": bool(CHROMA_RERANK_ENABLED_DEFAULT),
        "updated_at": now_iso(),
    }


def normalize_retrieval_profile(raw):
    base = default_retrieval_profile()
    if not isinstance(raw, dict):
        return base
    base["top_k"] = clamp_int(raw.get("top_k", base["top_k"]), 4, 20)
    base["bm25_weight"] = round(clamp_float(raw.get("bm25_weight", base["bm25_weight"]), 0.05, 0.95), 3)
    base["diversity_lambda"] = round(clamp_float(raw.get("diversity_lambda", base["diversity_lambda"]), 0.1, 0.95), 3)
    base["rerank_top_n"] = clamp_int(raw.get("rerank_top_n", base["rerank_top_n"]), 6, 40)
    if base["rerank_top_n"] < base["top_k"]:
        base["rerank_top_n"] = max(base["top_k"], 6)
    base["rerank_on"] = bool(raw.get("rerank_on", base["rerank_on"]))
    base["updated_at"] = str(raw.get("updated_at") or now_iso())
    return base


def load_retrieval_profile():
    if not RETRIEVAL_PROFILE_PATH.exists():
        return default_retrieval_profile()
    try:
        payload = json.loads(RETRIEVAL_PROFILE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return default_retrieval_profile()
    return normalize_retrieval_profile(payload)


def save_retrieval_profile(profile: dict):
    data = normalize_retrieval_profile(profile)
    data["updated_at"] = now_iso()
    RETRIEVAL_PROFILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    RETRIEVAL_PROFILE_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(RETRIEVAL_PROFILE_PATH, 0o600)
    except OSError:
        pass
    return data


def source_key(source):
    uri = (source.get("uri") or "").strip()
    title = (source.get("title") or "").strip()
    chunk = source.get("chunk")
    page = source.get("page")
    if uri:
        base = uri
    elif title:
        base = title
    else:
        base = (source.get("snippet") or "")[:80]
    extras = []
    if chunk is not None:
        extras.append(f"chunk={chunk}")
    if page is not None:
        extras.append(f"page={page}")
    if extras:
        return base + "|" + "|".join(extras)
    return base


def summarize_sources_for_ledger(sources, limit=20):
    out = []
    for s in (sources or [])[:limit]:
        title = sanitize_for_logs(s.get("title"))
        uri = sanitize_for_logs(s.get("uri"))
        section = sanitize_for_logs(s.get("section_heading"))
        markers = sanitize_for_logs(s.get("figure_table_markers"))
        eq_markers = sanitize_for_logs(s.get("equation_markers"))
        doc_type = sanitize_for_logs(s.get("doc_type"))
        version_stage = sanitize_for_logs(s.get("version_stage"))
        stitch_span = sanitize_for_logs(s.get("stitch_span"))
        out.append(
            {
                "key": source_key(s),
                "title": title,
                "uri": uri,
                "source_type": s.get("source_type"),
                "score": s.get("score"),
                "chunk": s.get("chunk"),
                "page": s.get("page"),
                "section_heading": section,
                "doc_type": doc_type,
                "version_stage": version_stage,
                "doc_family": s.get("doc_family"),
                "figure_table_markers": markers,
                "equation_markers": eq_markers,
                "stitch_span": stitch_span,
                "sha256": s.get("sha256"),
            }
        )
    return out


def collect_doc_hashes(sources):
    hashes = []
    for s in sources or []:
        sha = (s.get("sha256") or "").strip()
        if sha:
            hashes.append(sha)
        else:
            hashes.append(stable_hash(source_key(s)))
    seen = set()
    out = []
    for h in hashes:
        if h in seen:
            continue
        seen.add(h)
        out.append(h)
    return out


def safe_rel_to_data_root(rel_path: str):
    rel = str(rel_path or "").strip()
    if not rel or rel.startswith("http://") or rel.startswith("https://"):
        return None
    root = Path(DATA_ROOT).expanduser().resolve() if DATA_ROOT else None
    if not root:
        return None
    p = Path(rel)
    if p.is_absolute():
        abs_path = p.resolve()
    else:
        abs_path = (root / rel).resolve()
    try:
        if not abs_path.is_relative_to(root):
            return None
    except Exception:
        if not str(abs_path).startswith(str(root)):
            return None
    return abs_path if abs_path.exists() else None


def source_open_uri(source):
    rel = (source.get("rel_path") or source.get("uri") or "").strip()
    abs_path = safe_rel_to_data_root(rel)
    if not abs_path:
        return ""
    uri = "file://" + url_quote(str(abs_path))
    page = source.get("page")
    try:
        if page:
            page_i = int(page)
            if page_i > 0:
                uri += f"#page={page_i}"
    except Exception:
        pass
    return uri


def parse_csv_tokens(raw: str):
    out = []
    for part in str(raw or "").split(","):
        token = part.strip()
        if token and token not in out:
            out.append(token)
    return out


def tal_load_tokens():
    if not TAL_TOKENS_PATH.exists():
        return {"tokens": []}
    try:
        payload = json.loads(TAL_TOKENS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"tokens": []}
    if not isinstance(payload, dict):
        return {"tokens": []}
    toks = payload.get("tokens")
    if not isinstance(toks, list):
        toks = []
    return {"tokens": toks}


def tal_save_tokens(payload):
    data = payload if isinstance(payload, dict) else {"tokens": []}
    data.setdefault("tokens", [])
    TAL_TOKENS_PATH.parent.mkdir(parents=True, exist_ok=True)
    TAL_TOKENS_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try:
        os.chmod(TAL_TOKENS_PATH, 0o600)
    except OSError:
        pass


def tal_prune_tokens(payload):
    now_ts = float(time.time())
    tokens = []
    for tok in (payload or {}).get("tokens", []):
        if not isinstance(tok, dict):
            continue
        try:
            expires_at = float(tok.get("expires_at", 0.0))
        except Exception:
            expires_at = 0.0
        if expires_at <= now_ts:
            continue
        try:
            remaining = int(tok.get("remaining_calls", 0))
        except Exception:
            remaining = 0
        if remaining <= 0:
            continue
        tokens.append(tok)
    return {"tokens": tokens}


def tal_log_event(event: str, details: dict | None = None):
    payload = {
        "ts": now_iso(),
        "event": clean_text(event),
        "details": details or {},
        "user_role": clean_text(st.session_state.get("user_role", RBAC_DEFAULT_ROLE)),
        "user_email": clean_text(st.session_state.get("user_email", "")),
        "chat_id": clean_text(st.session_state.get("active_chat_id", "")),
    }
    TAL_AUDIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        with TAL_AUDIT_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    except Exception:
        pass


def tal_issue_capability(tool: str, scope: str, ttl_seconds: int, max_calls: int, chat_id: str = ""):
    tool_name = clean_text(tool).lower()
    if not tool_name:
        return ""
    ttl = max(60, min(604800, int(ttl_seconds)))
    calls = max(1, min(200, int(max_calls)))
    chat_scope = clean_text(chat_id)
    payload = tal_prune_tokens(tal_load_tokens())
    token_id = uuid.uuid4().hex
    token = {
        "id": token_id,
        "tool": tool_name,
        "scope": clean_text(scope) or "once",
        "chat_id": chat_scope,
        "issued_at": now_iso(),
        "expires_at": float(time.time() + ttl),
        "remaining_calls": calls,
    }
    payload["tokens"].append(token)
    tal_save_tokens(payload)
    tal_log_event(
        "capability_issued",
        {
            "tool": tool_name,
            "scope": token["scope"],
            "chat_id": chat_scope,
            "ttl_seconds": ttl,
            "max_calls": calls,
        },
    )
    return token_id


def tal_find_capability(tool: str, chat_id: str = "", consume: bool = False):
    tool_name = clean_text(tool).lower()
    chat_scope = clean_text(chat_id)
    payload = tal_prune_tokens(tal_load_tokens())
    tokens = payload.get("tokens", [])
    pick_idx = None
    for idx, tok in enumerate(tokens):
        if clean_text(tok.get("tool")).lower() != tool_name:
            continue
        tok_chat = clean_text(tok.get("chat_id"))
        if tok_chat and chat_scope and tok_chat != chat_scope:
            continue
        if tok_chat and (not chat_scope):
            continue
        pick_idx = idx
        break
    selected = tokens[pick_idx] if pick_idx is not None else None
    if selected and consume:
        try:
            selected["remaining_calls"] = int(selected.get("remaining_calls", 0)) - 1
        except Exception:
            selected["remaining_calls"] = 0
        payload = tal_prune_tokens(payload)
        tal_save_tokens(payload)
        tal_log_event(
            "capability_consumed",
            {
                "tool": tool_name,
                "chat_id": chat_scope,
                "token_id": clean_text(selected.get("id")),
                "remaining_calls": int(selected.get("remaining_calls", 0)),
            },
        )
    elif consume:
        tal_log_event(
            "capability_missing",
            {
                "tool": tool_name,
                "chat_id": chat_scope,
            },
        )
    return selected


def tal_web_capability_active(consume: bool = False):
    chat_id = clean_text(st.session_state.get("active_chat_id", ""))
    tok = tal_find_capability("web.search", chat_id=chat_id, consume=consume)
    if tok:
        return True
    tok = tal_find_capability("web.search", chat_id="", consume=consume)
    return bool(tok)


def tal_web_capability_status():
    loaded = tal_load_tokens()
    payload = tal_prune_tokens(loaded)
    if len((loaded or {}).get("tokens", [])) != len((payload or {}).get("tokens", [])):
        tal_save_tokens(payload)
    chat_id = clean_text(st.session_state.get("active_chat_id", ""))
    once = 0
    chat = 0
    for tok in payload.get("tokens", []):
        if clean_text(tok.get("tool")).lower() != "web.search":
            continue
        tok_chat = clean_text(tok.get("chat_id"))
        if tok_chat and tok_chat != chat_id:
            continue
        if tok_chat:
            chat += int(tok.get("remaining_calls", 0) or 0)
        else:
            once += int(tok.get("remaining_calls", 0) or 0)
    return {"once": once, "chat": chat, "active": (once + chat) > 0}


def tal_grant_web_once():
    return tal_issue_capability(
        tool="web.search",
        scope="once",
        ttl_seconds=TAL_WEB_ONCE_TTL_SECONDS,
        max_calls=1,
        chat_id="",
    )


def tal_grant_web_for_chat():
    chat_id = clean_text(st.session_state.get("active_chat_id", ""))
    return tal_issue_capability(
        tool="web.search",
        scope="chat",
        ttl_seconds=TAL_WEB_CHAT_TTL_SECONDS,
        max_calls=TAL_WEB_CHAT_MAX_CALLS,
        chat_id=chat_id,
    )


def tal_consume_web_if_used(sources):
    if ALLOW_WEB_TOOLS:
        return
    has_web = any(
        isinstance(s, dict) and clean_text(s.get("source_type")).lower() == "web"
        for s in (sources or [])
    )
    if has_web:
        tal_web_capability_active(consume=True)


def tal_clear_tokens():
    tal_save_tokens({"tokens": []})
    tal_log_event("capabilities_cleared", {})


def normalized_allowlist(raw: str):
    vals = []
    for token in parse_csv_tokens(raw):
        norm = token.strip().lower()
        if norm and norm not in vals:
            vals.append(norm)
    return vals


def tool_allowed(tool_name: str):
    tname = str(tool_name or "").strip().lower()
    role_now = normalize_role_name(st.session_state.get("user_role", RBAC_DEFAULT_ROLE))

    if tname in {"google_search", "google_search_retrieval"}:
        if not role_has_permission(role_now, "web.search"):
            tal_log_event("tool_denied", {"tool": tname, "reason": "rbac"})
            return False
        source_mode_now = clean_text(st.session_state.get("source_mode", SOURCE_MODE_DEFAULT))
        if source_mode_now not in ("Files + Web", "Web only"):
            tal_log_event("tool_denied", {"tool": tname, "reason": "source_mode"})
            return False
        if not (ALLOW_WEB_TOOLS or tal_web_capability_active(consume=False)):
            tal_log_event("tool_denied", {"tool": tname, "reason": "capability_missing"})
            return False

    allow = normalized_allowlist(TOOL_ALLOWLIST_RAW)
    if not allow:
        if tname in {"google_search", "google_search_retrieval"}:
            tal_log_event("tool_allowed", {"tool": tname, "reason": "policy"})
        return True
    ok = tname in allow
    if tname in {"google_search", "google_search_retrieval"}:
        tal_log_event("tool_allowed" if ok else "tool_denied", {"tool": tname, "reason": "allowlist"})
    return ok


def workspace_allowed(project_filter: str):
    allow = normalized_allowlist(WORKSPACE_ALLOWLIST_RAW)
    if not allow:
        return True
    project = str(project_filter or "All").strip().lower()
    if project in {"all", "*"}:
        return "all" in allow or "*" in allow
    return project in allow


def parse_domain_allowlist(raw: str):
    out = []
    for token in parse_csv_tokens(raw):
        dom = token.strip().lower()
        if dom.startswith("https://"):
            dom = dom[8:]
        if dom.startswith("http://"):
            dom = dom[7:]
        dom = dom.strip("/")
        if dom and dom not in out:
            out.append(dom)
    return out


def domain_allowed_by_allowlist(domain: str, allowlist):
    dom = clean_text(domain or "").lower()
    if not dom:
        return False
    allowed = [clean_text(x).lower() for x in (allowlist or []) if clean_text(x)]
    if not allowed:
        return True
    for item in allowed:
        if item.startswith("."):
            if dom.endswith(item):
                return True
            continue
        if dom == item or dom.endswith("." + item):
            return True
    return False


def apply_web_domain_policy(sources, enabled: bool, raw_allowlist: str):
    source_list = list(sources or [])
    if not enabled:
        return source_list, []
    allow = parse_domain_allowlist(raw_allowlist)
    if not allow:
        return source_list, []
    kept = []
    blocked_domains = []
    for src in source_list:
        if not isinstance(src, dict):
            continue
        if clean_text(src.get("source_type") or "").lower() != "web":
            kept.append(src)
            continue
        uri = clean_text(src.get("uri") or "")
        if REQUIRE_HTTPS_WEB_SOURCES and uri.lower().startswith("http://"):
            dom = clean_text(src.get("web_domain") or web_source_domain(uri)).lower() or "insecure-http"
            if dom not in blocked_domains:
                blocked_domains.append(dom)
            continue
        dom = clean_text(src.get("web_domain") or web_source_domain(src.get("uri") or "")).lower()
        if domain_allowed_by_allowlist(dom, allow):
            kept.append(src)
        elif dom:
            if dom not in blocked_domains:
                blocked_domains.append(dom)
    return kept, blocked_domains


SENSITIVE_CATEGORY_KEYWORDS = {
    "medical": ["medical", "diagnosis", "patient", "clinic", "hospital", "health", "phi", "hipaa"],
    "student_records": ["student", "grade", "transcript", "ferpa", "course roster", "admissions"],
    "government_ids": ["ssn", "social security", "passport", "driver license", "tax id"],
    "financial": ["bank", "account number", "routing", "credit card", "payment"],
}


def detect_sensitive_library_categories(index_rows):
    text_bits = []
    for row in (index_rows or [])[:800]:
        if not isinstance(row, dict):
            continue
        text_bits.append(clean_text(row.get("title") or ""))
        text_bits.append(clean_text(row.get("rel_path") or ""))
        text_bits.append(clean_text(row.get("file_name") or ""))
        text_bits.append(clean_text(row.get("tag") or ""))
    blob = " ".join(text_bits).lower()
    found = []
    if not blob:
        return found
    for category, keys in SENSITIVE_CATEGORY_KEYWORDS.items():
        if any(k in blob for k in keys):
            found.append(category)
    return found


def clean_text(text: str):
    s = str(text or "")
    s = s.replace("\x00", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def is_probably_system_id(text: str) -> bool:
    s = clean_text(text)
    if not s or len(s) < 10:
        return False
    if s.startswith("fileSearchStores/"):
        return True
    if " " in s:
        return False
    if "." in s:
        return False
    if not re.fullmatch(r"[A-Za-z0-9_-]+", s):
        return False
    if re.fullmatch(r"[a-z0-9]{10,}", s):
        return True
    letters = [ch for ch in s if ch.isalpha()]
    digits = [ch for ch in s if ch.isdigit()]
    if not letters:
        return True
    if not digits:
        return False
    vowel_count = sum(1 for ch in letters if ch.lower() in "aeiou")
    return vowel_count <= 1


def classify_query_intent(text: str):
    q = (text or "").strip().lower()
    if not q:
        return "general"
    if any(
        k in q
        for k in (
            "what do you know about",
            "tell me about",
            "give me an overview of",
            "overview of",
            "what do my files say about",
            "what does my library say about",
            "main themes in",
            "big picture of",
        )
    ):
        return "overview"
    if any(k in q for k in ("what did i mean by", "define ", "definition", "what is ")):
        return "definition"
    if any(k in q for k in ("where did i say", "where is", "where mentioned", "quote where")):
        return "where_mentioned"
    if any(k in q for k in ("compare", "difference", "vs ", "versus", "contrast")):
        return "compare"
    if any(k in q for k in ("production", "deploy", "deployment", "monitoring", "rollout", "ops")):
        return "production"
    if any(k in q for k in ("equation", "math", "proof", "pseudocode", "algorithm", "function", "code", "sql")):
        return "code_math"
    return "general"


def intent_retrieval_overrides(intent: str):
    cfg = {
        "quote_first": False,
        "force_production_template": False,
        "overview_mode": False,
        "top_k_delta": 0,
        "rerank_top_n_delta": 0,
        "doc_type_filter": "",
        "require_equations": False,
    }
    i = str(intent or "").strip().lower()
    if i == "overview":
        cfg["overview_mode"] = True
        cfg["top_k_delta"] = 4
        cfg["rerank_top_n_delta"] = 8
    elif i in {"definition", "where_mentioned"}:
        cfg["quote_first"] = True
        cfg["top_k_delta"] = -1
    elif i == "compare":
        cfg["top_k_delta"] = 2
        cfg["rerank_top_n_delta"] = 4
    elif i == "production":
        cfg["force_production_template"] = True
        cfg["top_k_delta"] = 1
    elif i == "code_math":
        cfg["doc_type_filter"] = "code,data_table,note,paper"
        cfg["require_equations"] = True
        cfg["top_k_delta"] = 1
    return cfg


def run_record_label(rec):
    ts = rec.get("timestamp", "")
    rid = rec.get("run_id", "")
    q = (rec.get("query") or "").strip().replace("\n", " ")
    if len(q) > 60:
        q = q[:57] + "..."
    return f"{ts} | {rid[-6:]} | {q}"


def load_recent_run_records(limit=40):
    if not RUN_LEDGER_ENABLED or not RUN_LEDGER_PATH.exists():
        return []
    rows = []
    try:
        with RUN_LEDGER_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                if "enc" in payload:
                    token = str(payload.get("enc") or "").strip()
                    if not token or CHAT_CIPHER is None:
                        continue
                    try:
                        raw = CHAT_CIPHER.decrypt(token.encode("utf-8"))
                        payload = json.loads(raw.decode("utf-8"))
                    except Exception:
                        continue
                rows.append(payload)
    except Exception:
        return []
    rows = [r for r in rows if isinstance(r, dict) and r.get("run_id")]
    rows.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return rows[:limit]


def append_run_record(record):
    if not RUN_LEDGER_ENABLED:
        return
    if RUN_LEDGER_ENCRYPT and CHAT_CIPHER is None:
        return
    try:
        RUN_LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = record
        if RUN_LEDGER_ENCRYPT:
            token = CHAT_CIPHER.encrypt(json.dumps(record, ensure_ascii=False).encode("utf-8")).decode("utf-8")
            payload = {"enc": token}
        with RUN_LEDGER_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        os.chmod(RUN_LEDGER_PATH, 0o600)
    except OSError:
        pass


def feedback_db_connect():
    FEEDBACK_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(FEEDBACK_DB_PATH)
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS feedback_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at TEXT NOT NULL,
            run_id TEXT,
            chat_id TEXT,
            feedback_type TEXT NOT NULL,
            value INTEGER NOT NULL,
            query_hash TEXT,
            source_count INTEGER,
            file_source_count INTEGER,
            settings_json TEXT,
            source_keys_json TEXT,
            note TEXT
        )
        """
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_feedback_events_created_at ON feedback_events(created_at)"
    )
    con.execute(
        "CREATE INDEX IF NOT EXISTS idx_feedback_events_run_id ON feedback_events(run_id)"
    )
    con.commit()
    return con


def init_feedback_store():
    try:
        con = feedback_db_connect()
        con.close()
    except Exception:
        pass


def source_hint_terms_for_eval(sources, limit: int = 3):
    hints = []
    for s in sources or []:
        raw = str(s.get("title") or s.get("uri") or "").strip()
        if not raw:
            continue
        tokens = query_keywords(raw, limit=3)
        if tokens:
            hints.append(tokens[0].lower())
        if len(hints) >= limit:
            break
    out = []
    for h in hints:
        if h and h not in out:
            out.append(h)
    return out[:limit]


def auto_append_eval_case_from_feedback(message: dict, feedback_type: str):
    if not AUTO_ADD_FEEDBACK_CASES:
        return {"ok": False, "reason": "disabled"}
    query = str(message.get("query") or "").strip()
    if not query:
        return {"ok": False, "reason": "missing_query"}

    root = Path(__file__).parent
    cases = root / "eval" / "cases.jsonl"
    cases.parent.mkdir(parents=True, exist_ok=True)

    qhash = stable_hash(query.lower())
    existing = set()
    if cases.exists():
        try:
            with cases.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        row = json.loads(line)
                    except Exception:
                        continue
                    rq = str((row or {}).get("query") or "").strip().lower()
                    if rq:
                        existing.add(stable_hash(rq))
        except Exception:
            existing = set()
    if qhash in existing:
        return {"ok": False, "reason": "duplicate"}

    expected_refusal = feedback_type in {"should_refuse", "bad_citation"}
    hints = []
    if not expected_refusal:
        hints = source_hint_terms_for_eval(message.get("sources") or [], limit=3)
    case = {
        "id": f"auto_fb_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{qhash[:6]}",
        "query": query,
        "expected_refusal": bool(expected_refusal),
        "expected_sources": hints,
        "origin_feedback": feedback_type,
    }
    try:
        with cases.open("a", encoding="utf-8") as f:
            f.write(json.dumps(case, ensure_ascii=False) + "\n")
    except Exception as e:
        return {"ok": False, "reason": str(e)}
    return {"ok": True, "case_id": case["id"]}


def record_feedback_event(message: dict, feedback_type: str, value: int, note: str = ""):
    if not isinstance(message, dict):
        return False, "Missing message payload."
    payload_sources = [s for s in (message.get("sources") or []) if isinstance(s, dict)]
    settings_snapshot = message.get("settings") if isinstance(message.get("settings"), dict) else {}
    try:
        con = feedback_db_connect()
        con.execute(
            """
            INSERT INTO feedback_events (
                created_at, run_id, chat_id, feedback_type, value, query_hash,
                source_count, file_source_count, settings_json, source_keys_json, note
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                now_iso(),
                str(message.get("run_id") or ""),
                str(st.session_state.get("active_chat_id") or ""),
                str(feedback_type or "").strip()[:40],
                int(value),
                stable_hash(str(message.get("query") or "")),
                int(len(payload_sources)),
                int(sum(1 for s in payload_sources if s.get("source_type") == "file")),
                json.dumps(settings_snapshot, ensure_ascii=False),
                json.dumps([source_key(s) for s in payload_sources], ensure_ascii=False),
                clean_env_scalar(note, max_len=280),
            ),
        )
        con.commit()
        con.close()
        if feedback_type in {"answer", "sources", "should_refuse", "bad_citation"} and int(value) < 0:
            auto_append_eval_case_from_feedback(message, feedback_type=feedback_type)
        return True, "Feedback saved."
    except Exception as e:
        return False, f"Feedback save failed: {e}"


def load_feedback_summary(days: int = 60):
    out = {
        "answer_good": 0,
        "answer_bad": 0,
        "sources_good": 0,
        "sources_bad": 0,
        "missing_source": 0,
        "should_refuse": 0,
        "bad_citation": 0,
        "notes": [],
        "rows": 0,
    }
    try:
        con = feedback_db_connect()
        cur = con.cursor()
        horizon = None
        if int(days) > 0:
            horizon_ts = time.time() - (int(days) * 86400)
            horizon = datetime.fromtimestamp(horizon_ts).strftime("%Y-%m-%dT%H:%M:%S")
            cur.execute(
                """
                SELECT feedback_type, value, note
                FROM feedback_events
                WHERE created_at >= ?
                ORDER BY created_at DESC
                LIMIT 2000
                """,
                (horizon,),
            )
        else:
            cur.execute(
                """
                SELECT feedback_type, value, note
                FROM feedback_events
                ORDER BY created_at DESC
                LIMIT 2000
                """
            )
        rows = cur.fetchall()
        con.close()
    except Exception:
        return out

    for fb_type, value, note in rows:
        key = str(fb_type or "").strip().lower()
        val = int(value or 0)
        if key == "answer":
            if val > 0:
                out["answer_good"] += 1
            else:
                out["answer_bad"] += 1
        elif key == "sources":
            if val > 0:
                out["sources_good"] += 1
            else:
                out["sources_bad"] += 1
        elif key == "missing_source":
            out["missing_source"] += 1
            n = (note or "").strip()
            if n:
                out["notes"].append(n)
        elif key == "should_refuse":
            out["should_refuse"] += 1
        elif key == "bad_citation":
            out["bad_citation"] += 1
    out["rows"] = len(rows)
    if out["notes"]:
        out["notes"] = out["notes"][:8]
    return out


def load_positive_feedback_run_ids():
    try:
        con = feedback_db_connect()
        cur = con.cursor()
        cur.execute(
            """
            SELECT
                run_id,
                SUM(CASE WHEN feedback_type='answer' AND value>0 THEN 1 ELSE 0 END) AS good_answer,
                SUM(CASE WHEN feedback_type='answer' AND value<0 THEN 1 ELSE 0 END) AS bad_answer,
                SUM(CASE WHEN feedback_type='should_refuse' THEN 1 ELSE 0 END) AS should_refuse,
                SUM(CASE WHEN feedback_type='sources' AND value<0 THEN 1 ELSE 0 END) AS bad_sources,
                SUM(CASE WHEN feedback_type='bad_citation' THEN 1 ELSE 0 END) AS bad_citation
            FROM feedback_events
            WHERE run_id IS NOT NULL AND run_id <> ''
            GROUP BY run_id
            """
        )
        rows = cur.fetchall()
        con.close()
    except Exception:
        return set()

    allowed = set()
    for run_id, good_answer, bad_answer, should_refuse, bad_sources, bad_citation in rows:
        if int(good_answer or 0) < 1:
            continue
        if int(bad_answer or 0) > 0:
            continue
        if int(should_refuse or 0) > 0:
            continue
        if int(bad_sources or 0) > 1:
            continue
        if int(bad_citation or 0) > 0:
            continue
        allowed.add(str(run_id))
    return allowed


def export_feedback_failures_to_eval(path: Path, limit: int = 30, append_to_cases: Path | None = None):
    try:
        con = feedback_db_connect()
        cur = con.cursor()
        cur.execute(
            """
            SELECT run_id, feedback_type, note
            FROM feedback_events
            WHERE (feedback_type='answer' AND value<0)
               OR (feedback_type='sources' AND value<0)
               OR feedback_type='should_refuse'
               OR feedback_type='bad_citation'
            ORDER BY created_at DESC
            LIMIT 400
            """
        )
        rows = cur.fetchall()
        con.close()
    except Exception:
        return {"written": 0, "path": str(path), "error": "feedback_query_failed"}

    run_map = {}
    for rec in load_recent_run_records(limit=800):
        run_id = str(rec.get("run_id") or "").strip()
        query = str(rec.get("query") or "").strip()
        if run_id and query and run_id not in run_map:
            run_map[run_id] = query

    payloads = []
    seen = set()
    for run_id, feedback_type, note in rows:
        rid = str(run_id or "").strip()
        q = run_map.get(rid, "").strip()
        if not q:
            continue
        key = stable_hash(q.lower())
        if key in seen:
            continue
        seen.add(key)
        expected_refusal = feedback_type in {"should_refuse", "bad_citation"}
        item = {
            "id": f"feedback_{len(payloads)+1:03d}",
            "query": q,
            "expected_refusal": bool(expected_refusal),
            "expected_sources": [],
            "origin_feedback": str(feedback_type),
        }
        n = str(note or "").strip()
        if n:
            item["note"] = n[:180]
        payloads.append(item)
        if len(payloads) >= max(1, int(limit)):
            break

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in payloads:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    appended = 0
    if append_to_cases:
        target = Path(append_to_cases).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        existing_hashes = set()
        if target.exists():
            try:
                with target.open("r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        try:
                            row = json.loads(line)
                        except Exception:
                            continue
                        q = str((row or {}).get("query") or "").strip().lower()
                        if q:
                            existing_hashes.add(stable_hash(q))
            except Exception:
                existing_hashes = set()
        append_rows = []
        for row in payloads:
            q = str(row.get("query") or "").strip().lower()
            if not q:
                continue
            h = stable_hash(q)
            if h in existing_hashes:
                continue
            existing_hashes.add(h)
            append_rows.append(row)
        if append_rows:
            with target.open("a", encoding="utf-8") as f:
                for row in append_rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            appended = len(append_rows)

    return {"written": len(payloads), "path": str(path), "appended": appended, "append_target": str(append_to_cases) if append_to_cases else ""}


def propose_retrieval_candidate_from_feedback(current_profile: dict, summary: dict):
    current = normalize_retrieval_profile(current_profile or {})
    candidate = dict(current)
    reasons = []

    answer_good = int(summary.get("answer_good", 0))
    answer_bad = int(summary.get("answer_bad", 0))
    sources_good = int(summary.get("sources_good", 0))
    sources_bad = int(summary.get("sources_bad", 0))
    missing_source = int(summary.get("missing_source", 0))
    should_refuse = int(summary.get("should_refuse", 0))
    bad_citation = int(summary.get("bad_citation", 0))

    total_answer = max(1, answer_good + answer_bad)
    total_sources = max(1, sources_good + sources_bad)
    bad_rate = answer_bad / float(total_answer)
    bad_sources_rate = sources_bad / float(total_sources)

    if bad_rate >= 0.45:
        candidate["top_k"] += 2
        candidate["rerank_top_n"] += 3
        reasons.append("High bad-answer rate; increased retrieval depth and rerank pool.")
    elif bad_rate <= 0.15 and bad_sources_rate <= 0.2:
        candidate["top_k"] -= 1
        reasons.append("Stable positive feedback; reduced top_k for lower latency.")

    if bad_sources_rate >= 0.35:
        candidate["bm25_weight"] += 0.10
        candidate["diversity_lambda"] -= 0.05
        candidate["rerank_on"] = True
        reasons.append("Frequent wrong-source feedback; boosted lexical precision and reranking.")

    if missing_source >= 3:
        candidate["top_k"] += 1
        candidate["bm25_weight"] += 0.05
        reasons.append("Missing-source reports detected; widened retrieval with slight lexical bias.")
    if should_refuse >= 2:
        candidate["bm25_weight"] += 0.05
        candidate["rerank_on"] = True
        reasons.append("Frequent 'should have refused' feedback; tightened lexical grounding and reranking.")
    if bad_citation >= 2:
        candidate["rerank_on"] = True
        candidate["rerank_top_n"] += 2
        reasons.append("Frequent bad-citation feedback; expanded rerank pool for better support alignment.")

    candidate = normalize_retrieval_profile(candidate)
    changed = any(candidate.get(k) != current.get(k) for k in ("top_k", "bm25_weight", "diversity_lambda", "rerank_top_n", "rerank_on"))
    if not reasons:
        reasons.append("Insufficient feedback signal; kept profile unchanged.")
    return candidate, reasons, changed


def parse_json_object_from_text(text: str):
    raw = str(text or "").strip()
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except Exception:
        pass
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except Exception:
            return {}
    return {}


def run_eval_with_profile(
    profile: dict,
    tag: str,
    cases_relpath: str = "eval/cases.jsonl",
    report_suffix: str = "",
    env_overrides: dict | None = None,
):
    root = Path(__file__).parent
    cases = root / cases_relpath
    if not cases.exists():
        return {"ok": False, "error": f"{cases_relpath} not found."}
    safe_suffix = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(report_suffix or ""))
    out_report = root / "eval" / "out" / f"report_{tag}{safe_suffix}.html"
    out_report.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    p = normalize_retrieval_profile(profile)
    env["EDITH_CHROMA_TOP_K"] = str(p.get("top_k"))
    env["EDITH_CHROMA_BM25_WEIGHT"] = str(p.get("bm25_weight"))
    env["EDITH_CHROMA_DIVERSITY_LAMBDA"] = str(p.get("diversity_lambda"))
    env["EDITH_CHROMA_RERANK_TOP_N"] = str(p.get("rerank_top_n"))
    env["EDITH_CHROMA_RERANK"] = "true" if p.get("rerank_on") else "false"
    if isinstance(env_overrides, dict):
        for k, v in env_overrides.items():
            env[str(k)] = str(v)
    cmd = [
        sys.executable,
        str(root / "eval" / "run.py"),
        "--mode",
        "Files only",
        "--backend",
        "chroma",
        "--report",
        str(out_report),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(root),
            env=env,
            capture_output=True,
            text=True,
            timeout=1800,
        )
    except Exception as e:
        return {"ok": False, "error": str(e)}
    payload = parse_json_object_from_text(proc.stdout)
    summary = payload.get("summary") if isinstance(payload, dict) else {}
    if not isinstance(summary, dict):
        return {"ok": False, "error": proc.stderr.strip() or "Eval summary parse failed."}
    return {"ok": proc.returncode == 0, "summary": summary, "report": str(payload.get("report") or out_report), "stderr": proc.stderr.strip()}


def eval_gate_candidate_profile(current_profile: dict, candidate_profile: dict):
    baseline = run_eval_with_profile(current_profile, tag="baseline")
    challenger = run_eval_with_profile(candidate_profile, tag="candidate")
    baseline_trap = run_eval_with_profile(
        current_profile,
        tag="baseline",
        cases_relpath="eval/hallucination_traps.jsonl",
        report_suffix="_trap",
    )
    challenger_trap = run_eval_with_profile(
        candidate_profile,
        tag="candidate",
        cases_relpath="eval/hallucination_traps.jsonl",
        report_suffix="_trap",
    )
    result = {
        "baseline": baseline,
        "candidate": challenger,
        "baseline_trap": baseline_trap,
        "candidate_trap": challenger_trap,
        "passed": False,
        "reasons": [],
    }
    if not baseline.get("ok"):
        result["reasons"].append(f"Baseline eval failed: {baseline.get('error') or baseline.get('stderr') or 'unknown'}")
        return result
    if not challenger.get("ok"):
        result["reasons"].append(f"Candidate eval failed: {challenger.get('error') or challenger.get('stderr') or 'unknown'}")
        return result

    b = baseline.get("summary", {})
    c = challenger.get("summary", {})
    b_prec = float(b.get("citation_precision") or 0.0)
    c_prec = float(c.get("citation_precision") or 0.0)
    b_ref = float(b.get("refusal_accuracy") or 0.0)
    c_ref = float(c.get("refusal_accuracy") or 0.0)
    b_p95 = float(b.get("latency_p95") or 0.0)
    c_p95 = float(c.get("latency_p95") or 0.0)

    precision_ok = c_prec >= (b_prec - 0.01)
    refusal_ok = c_ref >= (b_ref - 0.01)
    latency_ok = c_p95 <= (b_p95 * 1.25 + 0.05)
    improved = (c_prec > b_prec + 0.005) or (c_ref > b_ref + 0.005) or (c_p95 < b_p95 - 0.05)

    if not precision_ok:
        result["reasons"].append("Candidate citation precision regressed beyond tolerance.")
    if not refusal_ok:
        result["reasons"].append("Candidate refusal correctness regressed beyond tolerance.")
    if not latency_ok:
        result["reasons"].append("Candidate latency regression exceeded threshold.")
    if not improved:
        result["reasons"].append("Candidate did not improve key metrics.")

    trap_enforced = baseline_trap.get("ok") and challenger_trap.get("ok")
    trap_ok = True
    if trap_enforced:
        bt = baseline_trap.get("summary", {})
        ct = challenger_trap.get("summary", {})
        b_trap_ref = float(bt.get("refusal_accuracy") or 0.0)
        c_trap_ref = float(ct.get("refusal_accuracy") or 0.0)
        trap_ok = c_trap_ref >= max(0.98, b_trap_ref - 0.01)
        if not trap_ok:
            result["reasons"].append(
                "Candidate trap refusal accuracy regressed or is below minimum threshold."
            )
    else:
        result["reasons"].append("Trap suite unavailable; skipped trap gate.")

    result["passed"] = precision_ok and refusal_ok and latency_ok and improved and trap_ok
    return result


def run_ab_eval_variants(current_profile: dict):
    base = normalize_retrieval_profile(current_profile or {})
    variants = [
        {
            "name": "A_current",
            "profile": dict(base),
            "env": {},
        },
        {
            "name": "B_rerank_off",
            "profile": normalize_retrieval_profile({**base, "rerank_on": False}),
            "env": {},
        },
        {
            "name": "C_deep_rerank",
            "profile": normalize_retrieval_profile(
                {
                    **base,
                    "top_k": min(20, int(base.get("top_k", 8)) + 2),
                    "rerank_on": True,
                    "rerank_top_n": min(40, int(base.get("rerank_top_n", 14)) + 6),
                }
            ),
            "env": {},
        },
        {
            "name": "D_no_rewrite",
            "profile": dict(base),
            "env": {"EDITH_QUERY_REWRITE": "false"},
        },
        {
            "name": "E_rewrite_on",
            "profile": dict(base),
            "env": {"EDITH_QUERY_REWRITE": "true"},
        },
    ]
    rows = []
    for v in variants:
        main = run_eval_with_profile(
            v["profile"],
            tag=f"ab_{v['name']}",
            env_overrides=v.get("env") or {},
        )
        trap = run_eval_with_profile(
            v["profile"],
            tag=f"ab_{v['name']}",
            cases_relpath="eval/hallucination_traps.jsonl",
            report_suffix="_trap",
            env_overrides=v.get("env") or {},
        )
        main_s = (main.get("summary") or {}) if isinstance(main, dict) else {}
        trap_s = (trap.get("summary") or {}) if isinstance(trap, dict) else {}
        precision = float(main_s.get("citation_precision") or 0.0)
        refusal = float(main_s.get("refusal_accuracy") or 0.0)
        p95 = float(main_s.get("latency_p95") or 999.0)
        trap_refusal = float(trap_s.get("refusal_accuracy") or 0.0)
        score = round((precision * 0.45) + (refusal * 0.35) + (trap_refusal * 0.15) - (min(120.0, p95) / 1200.0), 4)
        rows.append(
            {
                "variant": v["name"],
                "ok": bool(main.get("ok")),
                "trap_ok": bool(trap.get("ok")),
                "citation_precision": round(precision, 4),
                "refusal_accuracy": round(refusal, 4),
                "trap_refusal_accuracy": round(trap_refusal, 4),
                "latency_p95": round(p95, 3),
                "score": score,
                "env_overrides": v.get("env") or {},
                "profile": v["profile"],
                "report": main.get("report"),
                "trap_report": trap.get("report"),
            }
        )
    rows.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return rows


def choose_replay_value(value, allowed_values, fallback):
    if value in allowed_values:
        return value
    return fallback


def source_overlap_ratio(old_keys, new_keys):
    a = set(old_keys or [])
    b = set(new_keys or [])
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a.intersection(b))
    union = len(a.union(b))
    return inter / float(union) if union else 0.0


def normalize_source_keys_for_compare(record):
    keys = []
    if isinstance(record, dict):
        raw_keys = record.get("source_keys") or []
        if isinstance(raw_keys, list):
            for k in raw_keys:
                if isinstance(k, str) and k.strip():
                    keys.append(k.strip())
        if not keys:
            raw_sources = record.get("sources") or []
            if isinstance(raw_sources, list):
                for row in raw_sources:
                    if isinstance(row, dict):
                        k = (row.get("key") or "").strip()
                        if k:
                            keys.append(k)
    return keys


def compute_replay_compare(previous_record, answer_text, sources, used_model):
    if not isinstance(previous_record, dict):
        return None
    old_keys = normalize_source_keys_for_compare(previous_record)
    new_keys = [source_key(s) for s in (sources or []) if isinstance(s, dict)]
    old_answer_hash = (previous_record.get("answer_hash") or "").strip()
    new_answer_hash = stable_hash(answer_text or "")
    compare = {
        "replay_of": previous_record.get("run_id"),
        "source_overlap": round(source_overlap_ratio(old_keys, new_keys), 3),
        "source_count_delta": len(new_keys) - len(old_keys),
        "model_changed": (previous_record.get("model_used") or "") != (used_model or ""),
    }
    if old_answer_hash:
        compare["answer_changed"] = old_answer_hash != new_answer_hash
    else:
        compare["answer_changed"] = None
    return compare


def build_run_record(
    run_id,
    query,
    answer_text,
    sources,
    model_used,
    model_chain,
    settings,
    status,
    meta,
    duration_ms,
    replay_payload=None,
    replay_compare=None,
):
    query_clean = sanitize_for_logs(query or "")
    answer_clean = sanitize_for_logs(answer_text or "")
    source_summary = summarize_sources_for_ledger(sources or [])
    source_keys = [x.get("key") for x in source_summary if isinstance(x, dict) and x.get("key")]
    record = {
        "timestamp": now_iso(),
        "run_id": run_id,
        "query": query_clean,
        "query_hash": stable_hash(query or ""),
        "answer_hash": stable_hash(answer_text or ""),
        "status": status,
        "duration_ms": int(max(duration_ms, 0)),
        "model_used": model_used,
        "model_chain": list(model_chain or []),
        "source_count": len(sources or []),
        "source_keys": source_keys,
        "doc_hashes": collect_doc_hashes(sources or []),
        "sources": source_summary,
        "settings": settings or {},
        "meta": meta or {},
    }
    if RUN_LEDGER_INCLUDE_TEXT:
        record["answer_preview"] = answer_clean[:400]
    if isinstance(replay_payload, dict) and replay_payload.get("run_id"):
        record["replay_of"] = replay_payload.get("run_id")
    if replay_compare:
        record["replay_compare"] = replay_compare
    return record


def is_preview_model(name: str) -> bool:
    n = (name or "").lower()
    return ("preview" in n) or ("exp" in n)


def normalize_model_name(name: str) -> str:
    if not name:
        return ""
    return name.replace("models/", "").strip()


def model_sort_key(name: str):
    n = (name or "").lower()
    m = re.search(r"gemini-(\d+)(?:[.-](\d+))?", n)
    major = int(m.group(1)) if m else 0
    minor = int(m.group(2)) if m and m.group(2) else 0
    family = 0
    if "pro" in n:
        family = 3
    elif "flash" in n and "lite" not in n:
        family = 2
    elif "lite" in n:
        family = 1
    preview = 1 if is_preview_model(n) else 0
    return (major, minor, preview, family, n)


@st.cache_data(show_spinner=False, ttl=1800)
def list_available_models(api_key_fingerprint: str):
    del api_key_fingerprint
    if client is None:
        return []
    names = []
    try:
        for m in client.models.list():
            methods = (
                getattr(m, "supported_generation_methods", None)
                or getattr(m, "supportedGenerationMethods", None)
                or getattr(m, "supported_actions", None)
                or []
            )
            if methods and "generateContent" not in methods:
                continue
            name = normalize_model_name(getattr(m, "name", ""))
            if name:
                names.append(name)
    except Exception:
        return []
    deduped = []
    seen = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    deduped.sort(key=model_sort_key, reverse=True)
    return deduped


def choose_latest_available_model(available_models, allow_preview: bool):
    cands = []
    for name in available_models or []:
        low = name.lower()
        if not low.startswith("gemini-"):
            continue
        if "embedding" in low:
            continue
        if (not allow_preview) and is_preview_model(low):
            continue
        cands.append(name)
    if not cands:
        return ""
    cands.sort(key=model_sort_key, reverse=True)
    return cands[0]


def dedupe_keep_order(values):
    out = []
    seen = set()
    for value in values:
        v = normalize_model_name(value)
        if not v or v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def resolve_model_chain(
    profile: str,
    override_model: str,
    fallback_csv: str,
    available_models,
    allow_preview: bool,
):
    profile = (profile or "latest").lower()
    if profile not in MODEL_PROFILES:
        profile = "latest"

    chain = []
    if override_model:
        chain.append(override_model)

    configured_fallbacks = parse_csv_list(fallback_csv)
    if configured_fallbacks:
        chain.extend(configured_fallbacks)
    else:
        chain.extend(MODEL_PROFILE_CHAINS.get(profile, MODEL_PROFILE_CHAINS["latest"]))
        if profile == "latest":
            dynamic_latest = choose_latest_available_model(available_models, allow_preview=allow_preview)
            if dynamic_latest:
                chain.insert(0, dynamic_latest)

    if not allow_preview:
        chain = [m for m in chain if not is_preview_model(m)]

    chain = dedupe_keep_order(chain)
    if not chain:
        chain = ["gemini-2.5-flash"]

    if available_models:
        available_set = set(available_models)
        known = [m for m in chain if m in available_set]
        unknown = [m for m in chain if m not in available_set]
        chain = known + unknown
    return chain


def is_retryable_model_error(exc: Exception):
    msg = str(exc).lower()
    retry_markers = [
        "429",
        "500",
        "503",
        "timeout",
        "deadline",
        "temporar",
        "unavailable",
        "rate",
        "quota",
        "resource exhausted",
        "model not found",
        "404",
    ]
    return any(marker in msg for marker in retry_markers)


def generate_with_model_fallback(contents, cfg, model_chain):
    if client is None:
        raise RuntimeError("Google API key is not configured.")
    last_error = None
    for model_name in model_chain:
        try:
            resp = client.models.generate_content(model=model_name, contents=contents, config=cfg)
            return resp, model_name
        except Exception as e:
            last_error = e
            if not is_retryable_model_error(e):
                raise
            continue
    if last_error:
        raise last_error
    raise RuntimeError("No model candidates configured.")


def parse_json_object(text: str):
    if not text:
        return None
    raw = text.strip()
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    if fenced:
        raw = fenced.group(1)
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except Exception:
            pass

    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    blob = m.group(0)
    try:
        obj = json.loads(blob)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def parse_json_array(text: str):
    if not text:
        return None
    raw = text.strip()
    try:
        arr = json.loads(raw)
        return arr if isinstance(arr, list) else None
    except Exception:
        pass

    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.S | re.I)
    if fenced:
        raw = fenced.group(1)
        try:
            arr = json.loads(raw)
            return arr if isinstance(arr, list) else None
        except Exception:
            pass

    m = re.search(r"\[.*\]", text, flags=re.S)
    if not m:
        return None
    blob = m.group(0)
    try:
        arr = json.loads(blob)
        return arr if isinstance(arr, list) else None
    except Exception:
        return None


def sanitize_flashcard_tag(tag: str):
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "_", clean_text(tag or "").strip().lower())
    return cleaned.strip("_")[:40]


def build_flashcard_source_block(sources, index_map=None, max_sources: int = 10, max_chars: int = 280):
    rows = []
    for idx, src in enumerate((sources or [])[:max_sources], start=1):
        if not isinstance(src, dict):
            continue
        cite = normalized_source_citation(src, index_map=index_map or {})
        snippet = clean_text(src.get("snippet") or "")
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."
        if snippet:
            rows.append(f"[S{idx}] {cite}\nsnippet: {snippet}")
        else:
            rows.append(f"[S{idx}] {cite}")
    return "\n\n".join(rows)


def generate_anki_flashcards(
    question: str,
    answer_text: str,
    sources,
    model_chain,
    card_type: str = "Both",
    card_count: int = 15,
    focus: str = "Concepts",
    index_map=None,
):
    if not sources:
        return [], {"ok": False, "error": "No sources available for flashcard generation."}

    count = max(1, min(int(card_count), 80))
    mode = (card_type or "Both").strip().lower()
    if mode not in {"basic", "cloze", "both"}:
        mode = "both"
    focus_label = clean_text(focus or "Concepts")
    source_block = build_flashcard_source_block(sources, index_map=index_map or {})
    prompt = (
        "Generate Anki-ready flashcards from GROUNDED SOURCES.\n"
        "Use only evidence supported by SOURCES.\n"
        "Return STRICT JSON array only. No markdown.\n\n"
        "Schema per item:\n"
        "{"
        "\"type\":\"basic|cloze\","
        "\"front\":\"\","
        "\"back\":\"\","
        "\"text\":\"\","
        "\"tags\":[\"\"],"
        "\"citation\":\"[S#] ...\""
        "}\n\n"
        f"Target card count: {count}\n"
        f"Card mode: {mode}\n"
        f"Focus: {focus_label}\n"
        "Rules:\n"
        "- Basic cards: fill front/back. Leave text empty.\n"
        "- Cloze cards: fill text using {{c1::...}} cloze syntax. Use back for short explanation.\n"
        "- Keep each card concise and factual.\n"
        "- Include at least one citation in citation field.\n"
        "- tags should include: edith, flashcards, and one focus tag.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"ANSWER:\n{answer_text}\n\n"
        f"SOURCES:\n{source_block}\n"
    )
    try:
        text, used_model = generate_text_via_chain(prompt, model_chain, temperature=0.1)
    except Exception as e:
        return [], {"ok": False, "error": str(e)}

    payload = parse_json_array(text)
    if payload is None:
        obj = parse_json_object(text) or {}
        payload = obj.get("cards") if isinstance(obj, dict) else None
    if not isinstance(payload, list):
        return [], {"ok": False, "error": "Model did not return valid flashcard JSON."}

    cards = []
    focus_tag = sanitize_flashcard_tag(focus_label) or "concepts"
    for row in payload:
        if not isinstance(row, dict):
            continue
        ctype = clean_text(row.get("type") or "").lower()
        if ctype not in {"basic", "cloze"}:
            ctype = "cloze" if "{{c1::" in clean_text(row.get("text") or "") else "basic"
        front = clean_text(row.get("front") or "")
        back = clean_text(row.get("back") or "")
        text_field = clean_text(row.get("text") or "")
        citation = clean_text(row.get("citation") or "")
        tags_raw = row.get("tags") or []
        tags = []
        if isinstance(tags_raw, list):
            tags = [sanitize_flashcard_tag(x) for x in tags_raw if sanitize_flashcard_tag(x)]
        elif isinstance(tags_raw, str):
            tags = [sanitize_flashcard_tag(x) for x in re.split(r"[,\s]+", tags_raw) if sanitize_flashcard_tag(x)]
        tags = dedupe_keep_order(["edith", "flashcards", focus_tag] + tags)

        if ctype == "basic":
            if not front or not back:
                continue
        else:
            if not text_field:
                continue
            if "{{c1::" not in text_field:
                text_field = f"{{{{c1::{text_field}}}}}"

        cards.append(
            {
                "NoteType": "Cloze" if ctype == "cloze" else "Basic",
                "Front": front,
                "Back": back,
                "Text": text_field,
                "Tags": " ".join(tags),
                "Citation": citation,
                "Focus": focus_label,
            }
        )
        if len(cards) >= count:
            break

    if not cards:
        return [], {"ok": False, "error": "No valid flashcards could be generated."}
    return cards, {"ok": True, "model": used_model, "count": len(cards)}


def flashcards_to_delimited_text(cards, delimiter: str = ","):
    out = io.StringIO()
    fields = ["NoteType", "Front", "Back", "Text", "Tags", "Citation", "Focus"]
    writer = csv.DictWriter(out, fieldnames=fields, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()
    for card in cards or []:
        writer.writerow({k: clean_text((card or {}).get(k) or "") for k in fields})
    return out.getvalue()


def generate_methods_data_rows(question: str, sources, model_chain, max_rows: int = 12, index_map=None):
    blocks = build_flashcard_source_block(sources, index_map=index_map or {}, max_sources=14, max_chars=360)
    prompt = (
        "Extract a methods and data table from SOURCES.\n"
        "Return STRICT JSON array only. No markdown.\n"
        "Each row schema:\n"
        "{"
        "\"paper\":\"\","
        "\"design\":\"\","
        "\"data\":\"\","
        "\"sample\":\"\","
        "\"dv\":\"\","
        "\"ivs\":\"\","
        "\"estimator\":\"\","
        "\"ses\":\"\","
        "\"key_finding\":\"\","
        "\"limitations\":\"\","
        "\"citation\":\"[S#] ...\""
        "}\n"
        f"Maximum rows: {max(1, min(int(max_rows), 30))}\n\n"
        f"QUESTION:\n{question}\n\n"
        f"SOURCES:\n{blocks}\n"
    )
    try:
        text, used_model = generate_text_via_chain(prompt, model_chain, temperature=0.0)
    except Exception as e:
        return [], {"ok": False, "error": str(e)}
    payload = parse_json_array(text)
    if payload is None:
        obj = parse_json_object(text) or {}
        payload = obj.get("rows") if isinstance(obj, dict) else None
    if not isinstance(payload, list):
        return [], {"ok": False, "error": "Model did not return valid methods JSON."}

    fields = [
        "paper",
        "design",
        "data",
        "sample",
        "dv",
        "ivs",
        "estimator",
        "ses",
        "key_finding",
        "limitations",
        "citation",
    ]
    rows = []
    for row in payload[: max(1, min(int(max_rows), 30))]:
        if not isinstance(row, dict):
            continue
        clean_row = {k: clean_text(row.get(k) or "") for k in fields}
        if not clean_row["paper"] and not clean_row["key_finding"]:
            continue
        rows.append(clean_row)
    if not rows:
        return [], {"ok": False, "error": "No valid methods rows generated."}
    return rows, {"ok": True, "model": used_model, "count": len(rows)}


def methods_rows_to_csv(rows):
    out = io.StringIO()
    fields = [
        "paper",
        "design",
        "data",
        "sample",
        "dv",
        "ivs",
        "estimator",
        "ses",
        "key_finding",
        "limitations",
        "citation",
    ]
    writer = csv.DictWriter(out, fieldnames=fields, quoting=csv.QUOTE_MINIMAL)
    writer.writeheader()
    for row in rows or []:
        writer.writerow({k: clean_text((row or {}).get(k) or "") for k in fields})
    return out.getvalue()


def build_theme_clusters(sources, index_map=None, max_clusters: int = 6):
    buckets = {}
    for src in sources or []:
        if not isinstance(src, dict):
            continue
        info = enrich_source(src, index_map or {})
        label = (
            clean_text(info.get("section_heading") or "")
            or clean_text(info.get("doc_type") or "")
            or clean_text(info.get("project") or "")
            or clean_text(info.get("tag") or "")
            or clean_text(info.get("author") or "")
            or "general"
        )
        if len(label) > 48:
            label = label[:48].rstrip() + "..."
        key = label.lower()
        row = buckets.setdefault(
            key,
            {"label": label or "general", "count": 0, "sources": set()},
        )
        row["count"] += 1
        sk = source_key(src)
        if sk:
            row["sources"].add(sk)
    out = []
    for b in buckets.values():
        out.append(
            {
                "label": b.get("label") or "general",
                "count": int(b.get("count") or 0),
                "unique_sources": len(b.get("sources") or []),
            }
        )
    out.sort(key=lambda x: (int(x.get("count", 0)), int(x.get("unique_sources", 0))), reverse=True)
    return out[: max(1, min(int(max_clusters), 12))]


def normalize_query_list(user_query: str, values):
    out = []
    seen = set()
    for value in values or []:
        q = re.sub(r"\s+", " ", str(value or "")).strip()
        if len(q) < 8:
            continue
        key = q.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(q)
        if len(out) >= QUERY_REWRITE_MAX:
            break
    if user_query.strip().lower() not in seen:
        out.insert(0, user_query.strip())
    if not out:
        out = [user_query.strip()]
    return out[:QUERY_REWRITE_MAX]


def heuristic_query_variants(user_query: str):
    base = (user_query or "").strip()
    if not base:
        return []
    variants = [base]

    keywords = query_keywords(base, limit=6)
    if keywords:
        variants.append(" ".join(keywords[:4]))
        variants.append("definition of " + " ".join(keywords[:3]))

    acronyms = re.findall(r"\b[A-Z]{2,}\b", user_query or "")
    for ac in acronyms[:2]:
        variants.append(f"{base} {ac} full form definition")

    return normalize_query_list(base, variants)


def merge_query_variants(user_query: str, model_variants):
    combined = []
    combined.extend(heuristic_query_variants(user_query))
    combined.extend(model_variants or [])
    return normalize_query_list(user_query, combined)


def rewrite_query_variants(user_query: str, source_mode: str, project_filter: str, tag_filter: str, model_chain):
    prompt = (
        "Rewrite the user question into retrieval-friendly query variants.\n"
        "Return strict JSON only: {\"queries\": [\"...\", \"...\"]}\n"
        f"- Include up to {QUERY_REWRITE_MAX} concise variants.\n"
        "- Include one exact-keyword variant.\n"
        "- Include one acronym expansion variant when acronyms are present.\n"
        "- Include one definition-style variant when useful.\n"
        "- Preserve names, years, and key entities.\n"
        "- Do not add facts.\n\n"
        f"Source mode: {source_mode}\n"
        f"Project filter: {project_filter}\n"
        f"Tag filter: {tag_filter}\n"
        f"Question: {user_query}\n"
    )
    cfg = types.GenerateContentConfig(temperature=0.0)
    try:
        resp, model_used = generate_with_model_fallback(prompt, cfg, model_chain)
        text = get_text(resp)
        data = parse_json_object(text) or {}
        queries = merge_query_variants(user_query, data.get("queries"))
        return queries, {"used": True, "model": model_used, "queries": queries}
    except Exception as e:
        heur = heuristic_query_variants(user_query)
        return heur, {"used": False, "error": str(e), "queries": heur}


def with_retrieval_hints(user_query: str, query_variants):
    variants = normalize_query_list(user_query, query_variants)
    if len(variants) <= 1:
        return user_query
    lines = "\n".join(f"- {q}" for q in variants)
    return (
        f"{user_query}\n\n"
        "INTERNAL_RETRIEVAL_HINTS (for search only, do not quote directly):\n"
        f"{lines}"
    )


def is_production_query(text: str):
    q = (text or "").lower()
    markers = [
        "production",
        "deploy",
        "deployment",
        "monitoring",
        "checklist",
        "risk",
        "risks",
        "assumption",
        "rollout",
    ]
    return any(m in q for m in markers)


def is_action_request(text: str):
    q = (text or "").lower()
    markers = [
        "delete",
        "deploy",
        "rewrite",
        "drop table",
        "run this command",
        "script to",
        "execute",
        "overwrite",
        "reset",
        "migrate",
        "launchctl",
        "systemctl",
    ]
    return any(m in q for m in markers)


def append_action_approval_template(prompt_text: str):
    block = (
        "\n\nACTION SAFETY RULE (mandatory):\n"
        "If the request implies operational risk (delete, deploy, rewrite, migration, command/script execution):\n"
        "1) Provide a high-level plan only.\n"
        "2) Do NOT output executable commands/scripts yet.\n"
        "3) End with: Approval required: reply 'APPROVE ACTION' to generate final commands.\n"
    )
    return (prompt_text or "").rstrip() + block


def enforce_action_approval_output(answer_text: str):
    text = (answer_text or "").strip()
    if not text:
        return text, False
    risky = False
    if "```" in text:
        risky = True
    if re.search(r"(?im)^\s*(rm|sudo|git|pip|python|curl|wget|brew|launchctl|systemctl|docker|kubectl)\b", text):
        risky = True
    if re.search(r"(?i)\b(drop table|truncate|delete from|chmod 777)\b", text):
        risky = True
    if not risky:
        return text, False

    body, cites = split_answer_and_citations(text)
    plan_lines = []
    for line in re.split(r"\n+", body):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("```"):
            continue
        if re.search(r"(?im)^\s*(rm|sudo|git|pip|python|curl|wget|brew|launchctl|systemctl|docker|kubectl)\b", stripped):
            continue
        plan_lines.append(stripped)
        if len(plan_lines) >= 8:
            break
    if not plan_lines:
        plan_lines = ["- Review scope and impacted files/services.", "- Validate backups/rollback.", "- Run a dry-run before execution."]
    safe = "Action requires approval before executable commands.\n\nPlan:\n"
    safe += "\n".join(f"- {ln.lstrip('- ').strip()}" for ln in plan_lines[:8])
    safe += "\n\nApproval required: reply 'APPROVE ACTION' to generate final commands."
    if cites:
        safe += f"\n\n{cites}"
    return safe, True


def append_production_template(prompt_text: str):
    block = (
        "\n\nRESPONSE FORMAT (mandatory):\n"
        "1) What sources say (cite each factual claim).\n"
        "2) Assumptions (cite where possible).\n"
        "3) Proposed plan (label clearly as proposal, not source fact).\n"
        "4) Risks and unknowns (if unsupported, say Not found in sources.)."
    )
    return (prompt_text or "").rstrip() + block


def is_quote_first_query(text: str):
    q = (text or "").lower()
    markers = [
        "where did i say",
        "exact quote",
        "quote the",
        "verbatim",
        "exact wording",
        "where is",
    ]
    return any(m in q for m in markers)


def append_quote_first_template(prompt_text: str):
    block = (
        "\n\nRESPONSE FORMAT (quote-first recall):\n"
        "- Return short direct quotes first.\n"
        "- For each quote include [S#] and page/section when available.\n"
        "- Include author/year when available in source metadata.\n"
        "- Do not paraphrase unless explicitly asked."
    )
    return (prompt_text or "").rstrip() + block


def append_response_style_template(
    prompt_text: str,
    verbosity_level: str,
    writing_style: str,
    include_methods_table: bool,
    include_limitations: bool,
):
    verbosity = clean_text(verbosity_level or "standard").lower()
    style = clean_text(writing_style or "academic").lower()
    if verbosity not in {"concise", "standard", "deep"}:
        verbosity = "standard"
    if style not in {"academic", "plain"}:
        style = "academic"

    detail_rule = {
        "concise": "Keep to 1 short paragraph and 3 bullets max.",
        "standard": "Use 1 short paragraph and 4-6 bullets.",
        "deep": "Use 2 short paragraphs and 6-10 bullets when evidence supports it.",
    }[verbosity]
    style_rule = (
        "Use precise academic language and explicit assumptions."
        if style == "academic"
        else "Use plain language with minimal jargon."
    )
    extras = []
    if include_methods_table:
        extras.append("Include a compact methods/data table when multiple studies are cited.")
    if include_limitations:
        extras.append("Include a short limitations section grounded in the evidence.")
    extras_line = ("\n- " + "\n- ".join(extras)) if extras else ""
    block = (
        "\n\nRESPONSE STYLE:\n"
        f"- {detail_rule}\n"
        f"- {style_rule}"
        f"{extras_line}"
    )
    return (prompt_text or "").rstrip() + block


def append_mode_preset_template(prompt_text: str, mode_preset: str):
    preset = clean_text(mode_preset or "").strip().lower()
    if preset == "study":
        block = (
            "\n\nSTUDY MODE FORMAT:\n"
            "1) Explain the concept in plain language.\n"
            "2) Give an academic phrasing of the same concept.\n"
            "3) Add common confusions/distinctions.\n"
            "4) End with 2-3 quiz questions.\n"
            "Keep claims source-grounded and cite key statements."
        )
        return (prompt_text or "").rstrip() + block
    if preset == "paper breakdown":
        block = (
            "\n\nPAPER BREAKDOWN FORMAT (mandatory):\n"
            "1) Section-by-section outline.\n"
            "2) Main thesis and contribution.\n"
            "3) Theory/mechanism.\n"
            "4) Methods and data (table if possible).\n"
            "5) Findings.\n"
            "6) Limitations and threats to validity.\n"
            "7) Why it matters in the literature.\n"
            "Include short supporting snippets and citations."
        )
        return (prompt_text or "").rstrip() + block
    if preset == "synthesis":
        block = (
            "\n\nSYNTHESIS MODE FORMAT:\n"
            "1) Themes of agreement.\n"
            "2) Disagreements and likely reasons.\n"
            "3) Gaps/open questions.\n"
            "4) Related-work outline.\n"
            "5) Key sources list.\n"
            "Bundle citations on major claims."
        )
        return (prompt_text or "").rstrip() + block
    if preset == "academic writing":
        block = (
            "\n\nACADEMIC WRITING MODE FORMAT:\n"
            "1) Use thesis-style prose with clear signposting.\n"
            "2) Use careful hedging (e.g., suggests, indicates) when appropriate.\n"
            "3) Keep argument flow tight and explicit.\n"
            "4) Preserve citation labels on factual claims."
        )
        return (prompt_text or "").rstrip() + block
    return prompt_text


def apply_answer_mode_templates(
    prompt_text: str,
    production_template_run: bool,
    quote_first_run: bool,
    mode_preset: str = "Custom",
    answer_type: str = "Explain",
    verbosity_level: str = "standard",
    writing_style: str = "academic",
    include_methods_table: bool = False,
    include_limitations: bool = True,
    action_approval_run: bool = False,
):
    out = append_mode_preset_template(prompt_text or "", mode_preset=mode_preset)
    out = append_response_style_template(
        out,
        verbosity_level=verbosity_level,
        writing_style=writing_style,
        include_methods_table=include_methods_table,
        include_limitations=include_limitations,
    )
    if quote_first_run and QUOTE_FIRST_RECALL_DEFAULT:
        out = append_quote_first_template(out)
    if production_template_run:
        out = append_production_template(out)
    if action_approval_run and ACTION_APPROVAL_DEFAULT:
        out = append_action_approval_template(out)
    hint = answer_type_hint(answer_type)
    if hint:
        out = (out or "").rstrip() + "\n\nANSWER TYPE:\n- " + hint
    return out


def strip_library_command(query_text: str):
    q = clean_text(query_text or "").strip()
    low = q.lower()
    if low.startswith("/library "):
        return q[9:].strip(), True
    if low == "/library":
        return "", True
    if low.startswith("/jarvis "):
        return q[8:].strip(), True
    if low == "/jarvis":
        return "", True
    if low.startswith("ask the library:"):
        return q.split(":", 1)[1].strip(), True
    if low.startswith("ask library:"):
        return q.split(":", 1)[1].strip(), True
    return q, False


def append_library_meta_template(prompt_text: str):
    block = (
        "\n\nASK THE LIBRARY MODE:\n"
        "Return a structured library brief with these sections:\n"
        "1) Top 12 sources (title + author/year when available).\n"
        "2) Three schools of thought/themes.\n"
        "3) Consensus vs disagreement.\n"
        "4) Methods distribution (high-level).\n"
        "5) Where evidence is thin.\n"
        "Keep claims source-grounded and preserve citation labels."
    )
    return (prompt_text or "").rstrip() + block


METHOD_TOKEN_PATTERNS = {
    "difference-in-differences": [
        re.compile(r"\bdifference[-\s]?in[-\s]?differences?\b", re.IGNORECASE),
        re.compile(r"\bdiff[-\s]?in[-\s]?diff\b", re.IGNORECASE),
        re.compile(r"\bdid\b\s+(design|model|estimate|analysis)", re.IGNORECASE),
    ],
    "regression discontinuity": [
        re.compile(r"\bregression discontinuity\b", re.IGNORECASE),
        re.compile(r"\brdd\b", re.IGNORECASE),
    ],
    "survey analysis": [
        re.compile(r"\bsurvey\b", re.IGNORECASE),
        re.compile(r"\bquestionnaire\b", re.IGNORECASE),
        re.compile(r"\bpoll(ing)?\b", re.IGNORECASE),
    ],
    "experiment": [
        re.compile(r"\bexperiment(al)?\b", re.IGNORECASE),
        re.compile(r"\brandomi[sz]ed\b", re.IGNORECASE),
        re.compile(r"\brct\b", re.IGNORECASE),
    ],
    "panel model": [
        re.compile(r"\bpanel (data|model)\b", re.IGNORECASE),
        re.compile(r"\bfixed effects?\b", re.IGNORECASE),
        re.compile(r"\brandom effects?\b", re.IGNORECASE),
        re.compile(r"\bwithin estimator\b", re.IGNORECASE),
    ],
    "instrumental variables": [
        re.compile(r"\binstrumental variables?\b", re.IGNORECASE),
        re.compile(r"\b2sls\b", re.IGNORECASE),
        re.compile(r"\biv (strategy|estimate|regression|design)\b", re.IGNORECASE),
    ],
}


def infer_method_tokens(text_blob: str):
    txt = clean_text(text_blob or "")
    found = []
    for label, patterns in METHOD_TOKEN_PATTERNS.items():
        if any(p.search(txt) for p in patterns):
            found.append(label)
    return found


def build_research_map_snapshot(
    focus: str,
    bibliography_rows,
    claim_rows,
    notebook_rows,
    citation_edges,
    experiment_rows,
):
    term = clean_text(focus or "").strip().lower()
    if not term:
        return {"papers": [], "notes": [], "claims": [], "methods": [], "disagreements": []}

    papers = []
    method_counts = defaultdict(int)
    for row in bibliography_rows or []:
        if not isinstance(row, dict):
            continue
        title = clean_text(row.get("title") or "")
        authors_raw = row.get("authors") or []
        if isinstance(authors_raw, (list, tuple, set)):
            authors = [clean_text(x) for x in authors_raw if clean_text(x)]
        else:
            one_author = clean_text(authors_raw)
            authors = [one_author] if one_author else []
        year = clean_text(row.get("year") or "")
        keywords_raw = row.get("keywords") or []
        if isinstance(keywords_raw, (list, tuple, set)):
            keywords = [clean_text(x) for x in keywords_raw if clean_text(x)]
        else:
            one_keyword = clean_text(keywords_raw)
            keywords = [one_keyword] if one_keyword else []
        hay = " ".join([title, " ".join(authors), " ".join(keywords)]).lower()
        if term not in hay:
            continue
        papers.append(
            {
                "title": title or "untitled",
                "author": clean_text(authors[0] if authors else "Unknown"),
                "year": year or "n.d.",
                "keywords": [k for k in keywords if k][:8],
            }
        )
        for token in infer_method_tokens(" ".join(keywords + [title])):
            method_counts[token] += 1
        if len(papers) >= 20:
            break

    notes = []
    for row in notebook_rows or []:
        if not isinstance(row, dict):
            continue
        title = clean_text(row.get("title") or "")
        body = clean_text(row.get("text") or "")
        q = clean_text(row.get("query") or "")
        hay = f"{title} {body} {q}".lower()
        if term in hay:
            notes.append(
                {
                    "title": title or "note",
                    "kind": clean_text(row.get("kind") or "note"),
                    "project": clean_text(row.get("project") or "All"),
                }
            )
        if len(notes) >= 20:
            break

    claims = []
    for row in claim_rows or []:
        if not isinstance(row, dict):
            continue
        claim_text = clean_text(row.get("claim") or "")
        if term in claim_text.lower():
            claims.append(
                {
                    "claim": claim_text,
                    "citations": [clean_text(x) for x in (row.get("support_citations") or [])][:6],
                }
            )
        if len(claims) >= 20:
            break

    for row in experiment_rows or []:
        if not isinstance(row, dict):
            continue
        blob = " ".join(
            [
                clean_text(row.get("experiment") or ""),
                clean_text(row.get("result") or ""),
                clean_text(row.get("source_doc") or ""),
            ]
        )
        if term not in blob.lower():
            continue
        for token in infer_method_tokens(blob):
            method_counts[token] += 1

    disagreements = []
    edge_groups = defaultdict(set)
    for edge in citation_edges or []:
        if not isinstance(edge, dict):
            continue
        doc = clean_text(edge.get("source_doc") or "")
        cit = clean_text(edge.get("citation") or "")
        if not doc or not cit:
            continue
        if term in doc.lower() or term in cit.lower():
            edge_groups[doc].add(cit)
    sorted_edges = sorted(edge_groups.items(), key=lambda item: len(item[1]), reverse=True)
    for doc, cites in sorted_edges[:12]:
        if len(cites) >= 2:
            disagreements.append({"doc": doc, "citations": sorted(list(cites))[:6]})

    methods = sorted(
        [{"method": k, "count": v} for k, v in method_counts.items()],
        key=lambda x: x["count"],
        reverse=True,
    )
    return {
        "papers": papers[:12],
        "notes": notes[:12],
        "claims": claims[:12],
        "methods": methods[:10],
        "disagreements": disagreements[:10],
    }


DATASET_TOKEN_PATTERNS = {
    "ANES": re.compile(r"\bANES\b", re.IGNORECASE),
    "CCES": re.compile(r"\bCCES\b", re.IGNORECASE),
    "CPS": re.compile(r"\bCPS\b", re.IGNORECASE),
    "ACS": re.compile(r"\bACS\b", re.IGNORECASE),
    "GSS": re.compile(r"\bGSS\b", re.IGNORECASE),
    "Census": re.compile(r"\bCensus\b", re.IGNORECASE),
    "Panel Study": re.compile(r"\bpanel study\b", re.IGNORECASE),
}


def infer_dataset_tokens(text_blob: str):
    txt = clean_text(text_blob or "")
    found = []
    for label, pattern in DATASET_TOKEN_PATTERNS.items():
        if pattern.search(txt):
            found.append(label)
    return found


def build_network_snapshot(
    focus: str,
    bibliography_rows,
    citation_edges,
    glossary_nodes,
    glossary_edges,
    claim_rows,
    experiment_rows,
):
    term = clean_text(focus or "").strip().lower()

    # Citation network (paper/doc hubs + shared-citation edges).
    doc_to_citations = defaultdict(set)
    for edge in citation_edges or []:
        if not isinstance(edge, dict):
            continue
        doc = clean_text(edge.get("source_doc") or "")
        cit = clean_text(edge.get("citation") or "")
        if not doc or not cit:
            continue
        if term and term not in f"{doc} {cit}".lower():
            continue
        doc_to_citations[doc].add(cit)
    citation_nodes = []
    for doc, cites in doc_to_citations.items():
        citation_nodes.append({"doc": doc, "citation_count": len(cites)})
    citation_nodes.sort(key=lambda x: x["citation_count"], reverse=True)
    citation_nodes = citation_nodes[:24]

    shared_counter = Counter()
    citation_to_docs = defaultdict(list)
    for doc, cites in doc_to_citations.items():
        for cit in cites:
            citation_to_docs[cit].append(doc)
    for docs in citation_to_docs.values():
        uniq = sorted(set(docs))
        if len(uniq) < 2:
            continue
        cap = min(len(uniq), 16)
        for i in range(cap):
            for j in range(i + 1, cap):
                shared_counter[(uniq[i], uniq[j])] += 1
    citation_shared_edges = [
        {"source": a, "target": b, "weight": int(w)}
        for (a, b), w in shared_counter.most_common(40)
    ]

    citation_papers = []
    for row in bibliography_rows or []:
        if not isinstance(row, dict):
            continue
        title = clean_text(row.get("title") or "")
        authors_raw = row.get("authors") or []
        if isinstance(authors_raw, (list, tuple, set)):
            authors = [clean_text(x) for x in authors_raw if clean_text(x)]
        else:
            one_author = clean_text(authors_raw)
            authors = [one_author] if one_author else []
        lead = authors[0] if authors else "Unknown"
        year = clean_text(row.get("year") or "") or "n.d."
        mentions = int(row.get("citation_mentions") or 0)
        source_docs = row.get("source_docs") or []
        keywords = row.get("keywords") or []
        hay = " ".join(
            [
                title,
                " ".join(authors),
                " ".join(str(x) for x in source_docs),
                " ".join(str(x) for x in keywords),
            ]
        ).lower()
        if term and term not in hay:
            continue
        citation_papers.append(
            {
                "title": title or "untitled",
                "lead_author": lead,
                "year": year,
                "mentions": mentions,
            }
        )
    citation_papers.sort(key=lambda x: x["mentions"], reverse=True)
    citation_papers = citation_papers[:20]

    # Co-authorship network.
    author_counts = Counter()
    coauthor_counter = Counter()
    for row in bibliography_rows or []:
        if not isinstance(row, dict):
            continue
        title = clean_text(row.get("title") or "")
        authors_raw = row.get("authors") or []
        if isinstance(authors_raw, (list, tuple, set)):
            authors = [clean_text(x) for x in authors_raw if clean_text(x)]
        else:
            one_author = clean_text(authors_raw)
            authors = [one_author] if one_author else []
        keywords = [clean_text(x) for x in (row.get("keywords") or []) if clean_text(x)]
        hay = " ".join([title, " ".join(authors), " ".join(keywords)]).lower()
        if term and term not in hay:
            continue
        uniq_authors = []
        seen_authors = set()
        for a in authors:
            key = a.lower()
            if key in seen_authors:
                continue
            seen_authors.add(key)
            uniq_authors.append(a)
        for a in uniq_authors:
            author_counts[a] += 1
        cap = min(len(uniq_authors), 12)
        for i in range(cap):
            for j in range(i + 1, cap):
                a = uniq_authors[i]
                b = uniq_authors[j]
                edge_key = tuple(sorted([a, b]))
                coauthor_counter[edge_key] += 1
    coauthor_nodes = [{"author": k, "paper_count": int(v)} for k, v in author_counts.most_common(30)]
    coauthor_edges = [
        {"author_a": a, "author_b": b, "papers_together": int(w)}
        for (a, b), w in coauthor_counter.most_common(40)
    ]

    # Concept network.
    concept_nodes = []
    for node in glossary_nodes or []:
        if not isinstance(node, dict):
            continue
        term_name = clean_text(node.get("term") or "")
        if not term_name:
            continue
        if term and term not in term_name.lower():
            continue
        concept_nodes.append({"term": term_name, "mentions": int(node.get("mentions") or 0)})
    concept_nodes.sort(key=lambda x: x["mentions"], reverse=True)
    concept_nodes = concept_nodes[:24]
    concept_edges = []
    for edge in glossary_edges or []:
        if not isinstance(edge, dict):
            continue
        a = clean_text(edge.get("a") or "")
        b = clean_text(edge.get("b") or "")
        w = int(edge.get("weight") or 0)
        if not a or not b or w <= 0:
            continue
        if term and term not in f"{a} {b}".lower():
            continue
        concept_edges.append({"source": a, "target": b, "weight": w})
    concept_edges.sort(key=lambda x: x["weight"], reverse=True)
    concept_edges = concept_edges[:40]

    # Methods ↔ data ↔ findings network rows.
    mdf_rows = []
    mdf_counter = Counter()
    for row in experiment_rows or []:
        if not isinstance(row, dict):
            continue
        doc = clean_text(row.get("source_doc") or "")
        chapter = clean_text(row.get("chapter") or "")
        result = clean_text(row.get("result") or "")
        exp_name = clean_text(row.get("experiment") or "")
        blob = " ".join([doc, chapter, result, exp_name])
        if term and term not in blob.lower():
            continue
        methods = infer_method_tokens(blob) or ["unspecified method"]
        datasets = infer_dataset_tokens(blob) or ["unspecified dataset"]
        for m in methods[:3]:
            for d in datasets[:3]:
                mdf_counter[(m, d)] += 1
                mdf_rows.append(
                    {
                        "method": m,
                        "dataset": d,
                        "finding": result[:180] if result else "(no explicit finding text)",
                        "source_doc": doc or "(unknown doc)",
                    }
                )
        if len(mdf_rows) >= 120:
            break
    mdf_pairs = [{"method": m, "dataset": d, "count": int(c)} for (m, d), c in mdf_counter.most_common(40)]

    # Claims support network.
    claim_nodes = []
    claim_edges = []
    for idx, row in enumerate(claim_rows or []):
        if not isinstance(row, dict):
            continue
        claim_text = clean_text(row.get("claim") or "")
        where = row.get("where") or {}
        doc = clean_text(where.get("doc") or "")
        chapter = clean_text(where.get("chapter") or "")
        citations = [clean_text(x) for x in (row.get("support_citations") or []) if clean_text(x)]
        if term and term not in f"{claim_text} {doc} {chapter}".lower():
            continue
        claim_id = f"C{idx+1}"
        claim_nodes.append(
            {
                "claim_id": claim_id,
                "claim": claim_text[:220],
                "doc": doc or "(unknown)",
                "chapter": chapter or "General",
                "support_strength": len(citations),
            }
        )
        for cit in citations[:8]:
            claim_edges.append({"claim_id": claim_id, "supports_from": cit})
        if len(claim_nodes) >= 40:
            break
    claim_nodes.sort(key=lambda x: x["support_strength"], reverse=True)

    return {
        "focus": term,
        "citation_network": {
            "paper_hubs": citation_papers,
            "doc_nodes": citation_nodes,
            "shared_edges": citation_shared_edges,
        },
        "coauthor_network": {
            "author_nodes": coauthor_nodes,
            "coauthor_edges": coauthor_edges,
        },
        "concept_network": {
            "concept_nodes": concept_nodes,
            "concept_edges": concept_edges,
        },
        "method_data_finding_network": {
            "pairs": mdf_pairs,
            "rows": mdf_rows[:60],
        },
        "claims_network": {
            "claim_nodes": claim_nodes[:24],
            "support_edges": claim_edges[:120],
        },
    }


def generate_text_via_chain(prompt_text: str, model_chain, system_instruction: str = "", temperature: float = 0.1):
    cfg = types.GenerateContentConfig(
        temperature=float(temperature),
        system_instruction=system_instruction or None,
    )
    resp, used_model = generate_with_model_fallback(prompt_text, cfg, model_chain)
    return (get_text(resp) or "").strip(), used_model


def plan_answer_outline(question: str, sources, model_chain):
    blocks = build_support_audit_source_blocks(sources)
    if not blocks:
        return {"used": False, "outline": [], "missing_evidence": []}
    prompt = (
        "You are planning a grounded answer.\n"
        "Using only SOURCES, produce strict JSON:\n"
        "{\"outline\":[\"...\"],\"missing_evidence\":[\"...\"],\"must_quote\":true|false}\n"
        "- outline: 3-7 concise answer bullets.\n"
        "- missing_evidence: evidence gaps required for high-confidence answer.\n"
        "- must_quote: true for definition/where-mentioned style questions.\n"
        "Do not include markdown.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"SOURCES:\n{blocks}"
    )
    try:
        text, used_model = generate_text_via_chain(prompt, model_chain, temperature=0.0)
        data = parse_json_object(text) or {}
        outline = [str(x).strip() for x in (data.get("outline") or []) if str(x).strip()][:8]
        missing = [str(x).strip() for x in (data.get("missing_evidence") or []) if str(x).strip()][:6]
        return {
            "used": True,
            "model": used_model,
            "outline": outline,
            "missing_evidence": missing,
            "must_quote": bool(data.get("must_quote")),
        }
    except Exception as e:
        return {"used": False, "error": str(e), "outline": [], "missing_evidence": []}


def build_multi_pass_prompt(question: str, sources, plan_meta: dict | None = None):
    blocks = build_support_audit_source_blocks(sources)
    outline = (plan_meta or {}).get("outline") or []
    missing = (plan_meta or {}).get("missing_evidence") or []
    prompt = (
        "Answer ONLY from SOURCES.\n"
        "If unsupported, say exactly: Not found in sources.\n"
        "Preserve citation labels [S#] on factual claims.\n\n"
        "Use author names and years when present in source metadata.\n\n"
        f"QUESTION:\n{question}\n\n"
    )
    if outline:
        prompt += "OUTLINE (from planning pass):\n" + "\n".join(f"- {x}" for x in outline) + "\n\n"
    if missing:
        prompt += (
            "MISSING EVIDENCE (do not invent these):\n"
            + "\n".join(f"- {x}" for x in missing)
            + "\n\n"
        )
    prompt += f"SOURCES:\n{blocks}"
    return prompt


def rewrite_overview_answer(question: str, draft_answer: str, sources, model_chain):
    if not sources:
        return draft_answer, {"enabled": True, "used": False, "reason": "no_sources"}
    blocks = build_support_audit_source_blocks((sources or [])[:10])
    prompt = (
        "Rewrite the draft as a grounded library-overview answer.\n"
        "Use only SOURCES.\n"
        "Keep it concise and source-faithful.\n"
        "Do not overstate claims.\n"
        "Required format:\n"
        "1) What your files say (2-4 short bullets, each with [S#]).\n"
        "2) Limits (1 short bullet about evidence gaps).\n"
        "If only one source supports the topic, state that clearly.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"DRAFT:\n{draft_answer}\n\n"
        f"SOURCES:\n{blocks}"
    )
    try:
        text, used_model = generate_text_via_chain(prompt, model_chain, temperature=0.1)
        if text.strip():
            return text.strip(), {"enabled": True, "used": True, "model": used_model}
        return draft_answer, {"enabled": True, "used": False, "reason": "empty_rewrite"}
    except Exception as e:
        return draft_answer, {"enabled": True, "used": False, "error": str(e)}


def detect_source_conflicts(question: str, sources, model_chain):
    file_sources = [s for s in (sources or []) if s.get("source_type") == "file"]
    if len(file_sources) < 2:
        return {"ran": False, "conflicts": []}
    blocks = build_support_audit_source_blocks(file_sources[:10])
    prompt = (
        "Find contradictions across SOURCES for the QUESTION.\n"
        "Return strict JSON only:\n"
        "{\"conflicts\":[{\"claim\":\"...\",\"source_a\":\"S#\",\"source_b\":\"S#\",\"reason\":\"...\"}]}\n"
        "If none, return {\"conflicts\":[]}.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"SOURCES:\n{blocks}"
    )
    try:
        text, used_model = generate_text_via_chain(prompt, model_chain, temperature=0.0)
        data = parse_json_object(text) or {}
        conflicts = []
        for row in (data.get("conflicts") or []):
            if not isinstance(row, dict):
                continue
            claim = str(row.get("claim") or "").strip()
            if not claim:
                continue
            conflicts.append(
                {
                    "claim": claim[:320],
                    "source_a": str(row.get("source_a") or "").strip()[:12],
                    "source_b": str(row.get("source_b") or "").strip()[:12],
                    "reason": str(row.get("reason") or "").strip()[:400],
                }
            )
        return {"ran": True, "model": used_model, "conflicts": conflicts[:8]}
    except Exception as e:
        return {"ran": False, "error": str(e), "conflicts": []}


def suggested_missing_artifacts(query: str, max_items: int = 3):
    keys = query_keywords(query, limit=6)
    suggestions = []
    if keys:
        key = "_".join(keys[:3])
        suggestions.append(f"{key}_method_notes.md")
        suggestions.append(f"{key}_results_table.csv")
        suggestions.append(f"{key}_assumptions.md")
    return suggestions[:max_items]


def build_support_audit_source_blocks(sources):
    blocks = []
    for i, s in enumerate((sources or [])[:SUPPORT_AUDIT_MAX_SOURCES], start=1):
        label = (s.get("title") or s.get("uri") or f"source_{i}").strip()
        src_type = (s.get("source_type") or "file").strip()
        snippet = (s.get("snippet") or "").strip()
        if snippet:
            snippet = snippet[:MAX_SNIPPET_CHARS]
        block = f"[S{i}] type={src_type} title={label}\n"
        if s.get("uri"):
            block += f"uri={s.get('uri')}\n"
        if snippet:
            block += f"snippet={snippet}\n"
        blocks.append(block.strip())
    return "\n\n".join(blocks)


def soften_overclaim_language(sentence: str) -> str:
    text = clean_text(sentence)
    if not text:
        return sentence
    rewrites = [
        (r"\b(unimportant|irrelevant)\b", "less central in this source"),
        (r"\b(proves?|definitively)\b", "suggests"),
        (r"\b(always|never)\b", "often"),
        (r"\b(all|none)\b", "many"),
    ]
    out = text
    for pattern, replacement in rewrites:
        out = re.sub(pattern, replacement, out, flags=re.IGNORECASE)
    return out


def audit_answer_support(question: str, draft_answer: str, sources, model_chain):
    del question
    del model_chain
    if not sources:
        return "Not found in sources.", {"ran": True, "verdict": "unsupported", "reason": "no_sources"}

    body, citation_block = split_answer_and_citations(draft_answer)
    claims = [s for s in split_sentences(body) if sentence_is_substantive(s)]
    if not claims:
        return draft_answer, {
            "ran": True,
            "verdict": "supported",
            "reason": "no_substantive_claims",
            "claim_count": 0,
            "coverage_ratio": 1.0,
        }

    supported_claims = []
    unsupported_claims = []
    revised_sentences = []
    for claim in claims:
        src_idx, match_score = match_sentence_to_source(claim, sources)
        if src_idx and match_score >= 0.18:
            supported_claims.append(
                {
                    "claim": claim,
                    "source_index": src_idx,
                    "match_score": round(match_score, 3),
                }
            )
            softened = soften_overclaim_language(claim)
            if re.search(r"\[S\d+\]", claim):
                revised_sentences.append(softened)
            else:
                revised_sentences.append(f"{softened} [S{src_idx}]")
        else:
            unsupported_claims.append({"claim": claim, "match_score": round(match_score, 3) if match_score else 0.0})

    coverage = len(supported_claims) / float(max(1, len(claims)))
    if not supported_claims or coverage < 0.45:
        return "Not found in sources.", {
            "ran": True,
            "verdict": "unsupported",
            "reason": "claim_coverage_too_low",
            "claim_count": len(claims),
            "supported_count": len(supported_claims),
            "unsupported_count": len(unsupported_claims),
            "coverage_ratio": round(coverage, 3),
            "unsupported_claims": unsupported_claims[:10],
        }

    if unsupported_claims:
        revised = " ".join(revised_sentences).strip()
        if citation_block:
            revised = f"{revised}\n\n{citation_block}".strip()
        return revised, {
            "ran": True,
            "verdict": "partial",
            "reason": "removed_unsupported_claims",
            "claim_count": len(claims),
            "supported_count": len(supported_claims),
            "unsupported_count": len(unsupported_claims),
            "coverage_ratio": round(coverage, 3),
            "unsupported_claims": unsupported_claims[:10],
        }

    return draft_answer, {
        "ran": True,
        "verdict": "supported",
        "reason": "all_claims_supported",
        "claim_count": len(claims),
        "supported_count": len(supported_claims),
        "unsupported_count": 0,
        "coverage_ratio": 1.0,
    }


def source_identity(source):
    uri = (source.get("uri") or "").strip()
    title = (source.get("title") or "").strip()
    return uri or title


def score_retrieval_confidence(sources, source_mode: str, hybrid_policy: str):
    srcs = sources or []
    if not srcs:
        return 0.0, {
            "source_count": 0,
            "distinct_sources": 0,
            "distinct_files": 0,
            "snippet_ratio": 0.0,
            "file_ratio": 0.0,
        }

    distinct_sources = len({source_identity(s) for s in srcs if source_identity(s)})
    file_sources = [s for s in srcs if s.get("source_type") == "file"]
    web_sources = [s for s in srcs if s.get("source_type") == "web"]
    distinct_files = len({source_identity(s) for s in file_sources if source_identity(s)})
    snippet_ratio = sum(1 for s in srcs if (s.get("snippet") or "").strip()) / float(len(srcs))
    file_ratio = len(file_sources) / float(len(srcs))
    numeric_scores = [float(s.get("score")) for s in srcs if isinstance(s.get("score"), (int, float))]
    score_quality = 0.0
    if numeric_scores:
        top = sorted(numeric_scores, reverse=True)[: min(4, len(numeric_scores))]
        score_quality = sum(top) / float(len(top))
        score_quality = max(0.0, min(1.0, score_quality))

    # Mode-aware confidence. File evidence is weighted highest when strict file grounding is expected.
    if source_mode == "Web only":
        mode_score = min(1.0, len(web_sources) / 3.0)
    elif source_mode == "Files only" or hybrid_policy == "require_files":
        mode_score = min(1.0, len(file_sources) / 3.0)
    else:
        mode_score = min(1.0, len(srcs) / 4.0)

    diversity_score = min(1.0, distinct_sources / 3.0)
    file_diversity_score = min(1.0, distinct_files / 2.0) if file_sources else 0.0
    score = (
        (0.30 * mode_score)
        + (0.20 * diversity_score)
        + (0.15 * snippet_ratio)
        + (0.20 * file_diversity_score)
        + (0.15 * score_quality)
    )
    score = max(0.0, min(1.0, score))

    return score, {
        "source_count": len(srcs),
        "distinct_sources": distinct_sources,
        "distinct_files": distinct_files,
        "snippet_ratio": round(snippet_ratio, 3),
        "file_ratio": round(file_ratio, 3),
        "score_quality": round(score_quality, 3),
    }


def choose_reasoning_model(model_chain, current_model: str):
    # Prefer a non-preview Pro model for heavy reasoning, then preview Pro.
    preferred = []
    fallback = []
    for m in model_chain:
        low = (m or "").lower()
        if "pro" not in low:
            continue
        if m == current_model:
            continue
        if "preview" in low:
            fallback.append(m)
        else:
            preferred.append(m)
    if preferred:
        return preferred[0]
    if fallback:
        return fallback[0]
    return ""


def run_with_specific_model(contents, cfg, model_name: str):
    if client is None:
        raise RuntimeError("Google API key is not configured.")
    resp = client.models.generate_content(model=model_name, contents=contents, config=cfg)
    return resp, model_name


def build_source_ref_maps(sources):
    index_to_typed = {}
    token_to_index = {}
    file_n = 0
    web_n = 0
    for idx, src in enumerate((sources or []), start=1):
        stype = (src.get("source_type") or "file").strip().lower()
        if stype == "web":
            web_n += 1
            typed = f"W{web_n}"
        else:
            file_n += 1
            typed = f"F{file_n}"
        index_to_typed[idx] = typed
        token_to_index[typed] = idx
        token_to_index[f"S{idx}"] = idx
    return {
        "index_to_typed": index_to_typed,
        "token_to_index": token_to_index,
    }


def source_ref_label(source_index: int, sources) -> str:
    maps = build_source_ref_maps(sources)
    return maps.get("index_to_typed", {}).get(int(source_index), f"S{int(source_index)}")


def build_inline_citations(sources, index_map=None):
    maps = build_source_ref_maps(sources)
    lines = []
    for i, s in enumerate((sources or [])[:INLINE_CITATION_MAX], start=1):
        token = maps.get("index_to_typed", {}).get(i, f"S{i}")
        label = normalized_source_citation(s, index_map=index_map)
        lines.append(f"[{token}] {label}")
    return "Citations:\n" + "\n".join(lines) if lines else ""


def ensure_inline_citations(answer_text: str, sources, index_map=None):
    text = (answer_text or "").strip()
    if not text:
        return text
    if text.strip().lower() == "not found in sources.":
        return text
    if not sources:
        return text
    if re.search(r"(?im)^citations:\s*$", text):
        return text
    block = build_inline_citations(sources, index_map=index_map)
    if not block:
        return text
    return text.rstrip() + "\n\n" + block


def validate_citation_refs(answer_text: str, sources):
    maps = build_source_ref_maps(sources)
    refs = set()
    invalid_tokens = []
    for m in re.finditer(r"\[(S|F|W)(\d+)\]", answer_text or ""):
        token = f"{m.group(1)}{m.group(2)}"
        idx = maps.get("token_to_index", {}).get(token)
        if idx is None:
            invalid_tokens.append(token)
            continue
        refs.add(int(idx))
    if not refs:
        return {"ok": False, "reason": "missing_refs", "refs": []}
    max_allowed = len((sources or []))
    invalid = sorted([r for r in refs if r < 1 or r > max_allowed])
    if invalid or invalid_tokens:
        return {
            "ok": False,
            "reason": "invalid_refs",
            "refs": sorted(refs),
            "invalid_refs": invalid,
            "invalid_tokens": invalid_tokens,
        }
    return {"ok": True, "reason": "ok", "refs": sorted(refs)}


def split_answer_and_citations(answer_text: str):
    text = (answer_text or "").strip()
    m = re.search(r"(?im)\n\s*citations:\s*\n", text)
    if not m:
        return text, ""
    body = text[: m.start()].rstrip()
    cites = text[m.start():].strip()
    return body, cites


def build_answer_pack_markdown(message: dict, query: str, index_map):
    if not isinstance(message, dict):
        return ""
    answer = str(message.get("text") or "").strip()
    if not answer:
        return ""
    body, citations_block = split_answer_and_citations(answer)
    lines = []
    lines.append("# Edith Answer Pack")
    lines.append("")
    lines.append("## Question")
    lines.append(query or "Unknown query")
    lines.append("")
    lines.append("## Answer")
    lines.append(body or answer)
    lines.append("")
    srcs = [s for s in (message.get("sources") or []) if isinstance(s, dict)]
    if srcs:
        lines.append("## Evidence Snippets")
        for idx, s in enumerate(srcs, start=1):
            label = s.get("title") or s.get("uri") or f"source_{idx}"
            page = s.get("page")
            section = (s.get("section_heading") or "").strip()
            snippet = (s.get("snippet") or "").strip()
            if len(snippet) > 700:
                snippet = snippet[:700].rstrip() + "..."
            meta = []
            if page:
                meta.append(f"p.{page}")
            if section:
                meta.append(f"section={section}")
            if s.get("chunk") is not None:
                meta.append(f"chunk={s.get('chunk')}")
            lines.append(f"- [S{idx}] {label}" + (f" ({', '.join(meta)})" if meta else ""))
            if snippet:
                lines.append(f"  - \"{snippet}\"")
        lines.append("")
        lines.append("## Bibliography (normalized)")
        for idx, s in enumerate(srcs, start=1):
            info = enrich_source(s, index_map or {})
            author = info.get("author") or "Unknown"
            year = info.get("year") or "n.d."
            title = info.get("title") or (s.get("title") or s.get("uri") or f"source_{idx}")
            lines.append(f"- [S{idx}] {author} ({year}). {title}.")
        lines.append("")
    if citations_block:
        lines.append("## Answer Citations Block")
        lines.append(citations_block)
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def find_assistant_query(messages, assistant_idx: int):
    if assistant_idx < 0 or assistant_idx >= len(messages or []):
        return ""
    msg = messages[assistant_idx] or {}
    direct = (msg.get("query") or "").strip()
    if direct:
        return direct
    for i in range(assistant_idx - 1, -1, -1):
        m = messages[i] or {}
        if m.get("role") == "user":
            return (m.get("text") or "").strip()
    return ""


def latest_assistant_message_with_sources(messages):
    for m in reversed(messages or []):
        if m.get("role") == "assistant" and m.get("sources"):
            return m
    return None


def format_sources_for_training(sources, max_sources: int = 6, max_chars: int = 500):
    rows = []
    for idx, src in enumerate((sources or [])[:max_sources], start=1):
        if not isinstance(src, dict):
            continue
        title = redact_pii_text((src.get("title") or src.get("uri") or f"source_{idx}").strip(), enabled=SFT_REDACT_PII, replacement=SFT_REDACT_TOKEN)
        uri = redact_pii_text((src.get("uri") or "").strip(), enabled=SFT_REDACT_PII, replacement=SFT_REDACT_TOKEN)
        snippet = redact_pii_text((src.get("snippet") or "").strip(), enabled=SFT_REDACT_PII, replacement=SFT_REDACT_TOKEN)
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars].rstrip() + "..."
        meta_bits = []
        if src.get("page") is not None:
            meta_bits.append(f"page={src.get('page')}")
        if src.get("chunk") is not None:
            meta_bits.append(f"chunk={src.get('chunk')}")
        meta_line = f" ({', '.join(meta_bits)})" if meta_bits else ""
        row = f"[S{idx}] {title}{meta_line}"
        if uri:
            row += f"\nuri={uri}"
        if snippet:
            row += f"\nsnippet={snippet}"
        rows.append(row)
    return "\n\n".join(rows)


def build_sft_example(query: str, answer: str, sources, require_citations: bool):
    q = redact_pii_text((query or "").strip(), enabled=SFT_REDACT_PII, replacement=SFT_REDACT_TOKEN)
    a = redact_pii_text((answer or "").strip(), enabled=SFT_REDACT_PII, replacement=SFT_REDACT_TOKEN)
    if not q or not a:
        return None
    source_block = format_sources_for_training(sources)
    strict_line = (
        "If facts are unsupported by SOURCES, answer exactly: Not found in sources."
        if require_citations
        else "Prefer SOURCES, but if insufficient, state uncertainty."
    )
    system_text = (
        "You are Edith, a grounded research assistant.\n"
        f"{strict_line}\n"
        "Keep answers concise and preserve citation labels like [S1], [S2] when evidence is used."
    )
    user_text = (
        f"QUESTION:\n{q}\n\n"
        "SOURCES:\n"
        f"{source_block if source_block else '(none)'}"
    )
    return {
        "messages": [
            {"role": "system", "content": system_text},
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": a},
        ]
    }


def collect_sft_examples_from_saved_chats(
    max_examples: int = 600,
    include_refusals: bool = True,
    only_positive_feedback: bool = False,
):
    examples = []
    allowed_run_ids = load_positive_feedback_run_ids() if only_positive_feedback else None
    chats = list_saved_chats()
    for chat in chats:
        data = load_chat(chat.get("id", ""))
        if not data:
            continue
        msgs = data.get("messages", []) or []
        for idx, msg in enumerate(msgs):
            if msg.get("role") != "assistant":
                continue
            answer = (msg.get("text") or "").strip()
            if not answer:
                continue
            if isinstance(allowed_run_ids, set):
                run_id = str(msg.get("run_id") or "").strip()
                if not run_id or run_id not in allowed_run_ids:
                    continue
            query = find_assistant_query(msgs, idx)
            sources = msg.get("sources") or []
            is_refusal = answer.strip().lower() == "not found in sources."
            if not sources and not (include_refusals and is_refusal):
                continue
            ex = build_sft_example(
                query=query,
                answer=answer,
                sources=sources,
                require_citations=True,
            )
            if ex:
                examples.append(ex)
                if len(examples) >= max_examples:
                    return examples
    return examples


def split_train_val_examples(examples, val_ratio: float = 0.1):
    ordered = []
    for ex in examples or []:
        try:
            key = stable_hash(json.dumps(ex, sort_keys=True, ensure_ascii=False))
        except Exception:
            key = stable_hash(str(ex))
        ordered.append((key, ex))
    ordered.sort(key=lambda x: x[0])
    val_every = max(2, int(round(1.0 / max(0.001, float(val_ratio)))))
    train = []
    val = []
    for i, (_, ex) in enumerate(ordered):
        if (i + 1) % val_every == 0:
            val.append(ex)
        else:
            train.append(ex)
    if not train and val:
        train.append(val.pop(0))
    return train, val


def examples_to_jsonl(examples):
    lines = []
    for ex in examples or []:
        lines.append(json.dumps(ex, ensure_ascii=False))
    return "\n".join(lines) + ("\n" if lines else "")


def query_keywords(query: str, limit: int = 6):
    words = re.findall(r"[A-Za-z0-9]{3,}", (query or "").lower())
    out = []
    for w in words:
        if w in out:
            continue
        out.append(w)
        if len(out) >= limit:
            break
    return out


def query_requests_fresh_web(query: str):
    q = (query or "").lower()
    markers = [
        "latest",
        "today",
        "current",
        "news",
        "recent",
        "updated",
        "update",
        "this week",
        "this month",
        "as of",
    ]
    return any(x in q for x in markers)


def query_fingerprint_terms(query: str, limit: int = 10):
    toks = re.findall(r"[A-Za-z0-9]{3,}", (query or "").lower())
    out = []
    for t in toks:
        if t not in out:
            out.append(t)
        if len(out) >= limit:
            break
    return out


def compact_snippet_for_query(snippet: str, query: str, max_sentences: int = 4):
    text = clean_text(snippet or "")
    if not text:
        return ""
    sentences = split_sentences(text)
    if not sentences:
        return text[:600]
    qterms = set(query_fingerprint_terms(query, limit=10))
    picked = []
    used = set()
    for sent in sentences:
        key = sent.lower()
        if key in used:
            continue
        stoks = set(re.findall(r"[A-Za-z0-9]{3,}", sent.lower()))
        overlap = len(stoks.intersection(qterms))
        if overlap > 0:
            picked.append(sent)
            used.add(key)
        if len(picked) >= max_sentences:
            break
    if not picked:
        picked = sentences[:max_sentences]
    compact = " ".join(picked)
    return compact[:700]


def pack_sources_for_context(query: str, sources, max_sources: int = 10):
    packed = []
    seen = set()
    qterms = set(query_fingerprint_terms(query, limit=10))
    for src in sources or []:
        if not isinstance(src, dict):
            continue
        snippet = compact_snippet_for_query(src.get("snippet") or "", query)
        if not snippet:
            continue
        stoks = set(re.findall(r"[A-Za-z0-9]{3,}", snippet.lower()))
        overlap = len(stoks.intersection(qterms))
        sig = stable_hash(snippet.lower()[:420])
        if sig in seen:
            continue
        seen.add(sig)
        row = dict(src)
        row["snippet"] = snippet
        row["context_overlap"] = overlap
        packed.append(row)
    packed.sort(key=lambda x: (float(x.get("score") or 0.0), int(x.get("context_overlap") or 0)), reverse=True)
    return packed[: max(2, int(max_sources))]


SECTION_HINTS = {
    "abstract": ("abstract", "summary"),
    "introduction": ("introduction", "background", "theory"),
    "methods": ("method", "design", "data", "sample", "measurement"),
    "results": ("result", "finding", "analysis", "evidence"),
    "discussion": ("discussion", "implication", "conclusion"),
    "limitations": ("limitation", "caveat", "future work"),
}


def source_doc_identity(source):
    s = source or {}
    return clean_text(
        s.get("rel_path")
        or s.get("uri")
        or s.get("title")
        or s.get("file_name")
        or s.get("sha256")
        or ""
    ).lower()


def extract_section_bucket(source):
    s = source or {}
    text = " ".join(
        [
            clean_text(s.get("section_heading") or ""),
            clean_text(s.get("doc_type") or ""),
            clean_text(s.get("snippet") or "")[:220],
        ]
    ).lower()
    for bucket, markers in SECTION_HINTS.items():
        if any(m in text for m in markers):
            return bucket
    return ""


def section_boost_for_intent(source, intent: str):
    section = extract_section_bucket(source)
    intent = clean_text(intent or "").lower()
    if not section:
        return 0.0
    wanted = set()
    if intent in {"overview", "general"}:
        wanted = {"abstract", "introduction", "results", "discussion"}
    elif intent in {"compare"}:
        wanted = {"methods", "results", "discussion"}
    elif intent in {"definition", "where_mentioned"}:
        wanted = {"abstract", "introduction"}
    elif intent in {"production"}:
        wanted = {"methods", "results", "limitations"}
    elif intent in {"code_math"}:
        wanted = {"methods", "results"}
    if section in wanted:
        return 0.06
    if section in {"limitations"} and intent in {"overview", "general", "production"}:
        return 0.03
    return 0.0


def apply_section_boosts(sources, intent: str):
    out = []
    for s in (sources or []):
        row = dict(s)
        base = float(row.get("score") or 0.0)
        boost = section_boost_for_intent(row, intent)
        row["section_boost"] = round(boost, 4)
        row["score"] = round(base + boost, 4)
        out.append(row)
    out.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return out


def select_breadth_depth_sources(
    sources,
    intent: str,
    max_docs: int = 10,
    chunks_per_doc: int = 4,
    max_total: int = 28,
):
    boosted = apply_section_boosts(sources or [], intent)
    if not boosted:
        return []

    grouped = defaultdict(list)
    for row in boosted:
        grouped[source_doc_identity(row)].append(row)

    docs = []
    for doc_id, rows in grouped.items():
        if not doc_id:
            continue
        rows.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        docs.append((doc_id, rows))
    docs.sort(key=lambda x: float((x[1][0] or {}).get("score") or 0.0), reverse=True)
    docs = docs[: max(1, int(max_docs))]

    picked = []
    seen_chunk = set()
    for _, rows in docs:
        kept = 0
        for row in rows:
            sig = stable_hash(
                "|".join(
                    [
                        clean_text(row.get("rel_path") or row.get("uri") or ""),
                        str(row.get("chunk") if row.get("chunk") is not None else ""),
                        clean_text(row.get("snippet") or "")[:240],
                    ]
                )
            )
            if sig in seen_chunk:
                continue
            seen_chunk.add(sig)
            picked.append(row)
            kept += 1
            if kept >= max(1, int(chunks_per_doc)):
                break
        if len(picked) >= max(2, int(max_total)):
            break

    picked.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return picked[: max(2, int(max_total))]


def evidence_sufficiency_report(sources):
    rows = list(sources or [])
    if not rows:
        return {
            "unique_docs": 0,
            "section_coverage": 0.0,
            "redundancy": 1.0,
            "sufficient": False,
            "reason": "no_sources",
        }
    docs = set()
    sections = set()
    snippets = set()
    for s in rows:
        doc_id = source_doc_identity(s)
        if doc_id:
            docs.add(doc_id)
        bucket = extract_section_bucket(s)
        if bucket:
            sections.add(bucket)
        snip = clean_text((s or {}).get("snippet") or "")
        if snip:
            snippets.add(stable_hash(snip[:320].lower()))
    unique_docs = len(docs)
    section_coverage = len(sections) / float(max(1, len(SECTION_HINTS)))
    unique_ratio = len(snippets) / float(max(1, len(rows)))
    redundancy = max(0.0, 1.0 - unique_ratio)
    sufficient = unique_docs >= 2 and (section_coverage >= 0.2 or unique_ratio >= 0.55)
    reason = "ok" if sufficient else "thin_evidence"
    return {
        "unique_docs": int(unique_docs),
        "section_coverage": round(section_coverage, 3),
        "redundancy": round(redundancy, 3),
        "sufficient": bool(sufficient),
        "reason": reason,
    }


def build_research_evidence_cards(sources, max_docs: int = 8, max_snippets_per_doc: int = 4):
    grouped = defaultdict(list)
    for idx, src in enumerate((sources or []), start=1):
        if not isinstance(src, dict):
            continue
        doc_id = source_doc_identity(src) or f"doc_{idx}"
        grouped[doc_id].append((idx, src))

    ranked_docs = []
    for doc_id, rows in grouped.items():
        rows.sort(key=lambda x: float((x[1] or {}).get("score") or 0.0), reverse=True)
        ranked_docs.append((doc_id, rows))
    ranked_docs.sort(key=lambda x: float((x[1][0][1] or {}).get("score") or 0.0), reverse=True)

    cards = []
    for _, rows in ranked_docs[: max(1, int(max_docs))]:
        first = rows[0][1]
        title = derive_title_from_source(first, {})
        author = clean_text(first.get("author") or "")
        year = clean_text(first.get("year") or "")
        meta = []
        if author or year:
            meta.append((author or "Unknown author") + (f" ({year})" if year else ""))
        if first.get("doc_type"):
            meta.append(clean_text(first.get("doc_type") or ""))
        snippets = []
        citations = []
        for idx, src in rows[: max(1, int(max_snippets_per_doc))]:
            citations.append(f"S{idx}")
            clip = compact_snippet_for_query(src.get("snippet") or "", title, max_sentences=2)
            if clip:
                snippets.append(clip)
        cards.append(
            {
                "title": title,
                "meta": " | ".join([m for m in meta if m]),
                "citations": citations,
                "snippets": snippets,
            }
        )
    return cards


def build_research_synthesis_prompt(question: str, sources, cards):
    source_blocks = build_support_audit_source_blocks((sources or [])[:14])
    card_lines = []
    for i, card in enumerate(cards or [], start=1):
        cits = ",".join(card.get("citations") or [])
        title = clean_text(card.get("title") or f"Document {i}")
        meta = clean_text(card.get("meta") or "")
        snippets = card.get("snippets") or []
        card_lines.append(f"[D{i}] {title}" + (f" | {meta}" if meta else ""))
        card_lines.append(f"citations={cits}")
        for sn in snippets[:3]:
            card_lines.append(f"- {sn}")
        card_lines.append("")
    card_text = "\n".join(card_lines).strip() or "(no cards)"
    return (
        "Write a PhD-level grounded synthesis using only SOURCES and EVIDENCE_CARDS.\n"
        "Rules:\n"
        "- Use cautious, source-faithful wording.\n"
        "- Cite factual claims with [S#].\n"
        "- Do not invent studies, methods, or findings.\n"
        "- If evidence is thin, say so explicitly.\n\n"
        "Output format:\n"
        "1) What your files say (3-6 bullets).\n"
        "2) Research synthesis (agreements/disagreements/gaps).\n"
        "3) Evidence limits (1-2 bullets).\n\n"
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE_CARDS:\n{card_text}\n\n"
        f"SOURCES:\n{source_blocks}\n"
    )


def interleave_sources_by_document(sources, max_total: int = 36):
    grouped = defaultdict(list)
    for row in (sources or []):
        if not isinstance(row, dict):
            continue
        doc_id = source_doc_identity(row)
        if not doc_id:
            doc_id = stable_hash(clean_text(row.get("snippet") or "")[:120])
        grouped[doc_id].append(dict(row))
    buckets = []
    for doc_id, rows in grouped.items():
        rows.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
        buckets.append({"doc_id": doc_id, "rows": rows})
    buckets.sort(key=lambda b: float((b["rows"][0] or {}).get("score") or 0.0), reverse=True)
    interleaved = []
    while len(interleaved) < max(1, int(max_total)):
        advanced = False
        for bucket in buckets:
            rows = bucket.get("rows") or []
            if rows:
                interleaved.append(rows.pop(0))
                advanced = True
                if len(interleaved) >= max(1, int(max_total)):
                    break
        if not advanced:
            break
    if interleaved:
        return interleaved
    ranked = [dict(x) for x in (sources or []) if isinstance(x, dict)]
    ranked.sort(key=lambda x: float(x.get("score") or 0.0), reverse=True)
    return ranked[: max(1, int(max_total))]


def split_sources_for_recursive_controller(sources, batch_size: int = 6, max_batches: int = 6):
    size = max(2, int(batch_size))
    limit = max(2, int(max_batches)) * size
    ordered = interleave_sources_by_document(sources, max_total=limit)
    out = []
    for i in range(0, len(ordered), size):
        batch = ordered[i : i + size]
        if batch:
            out.append(batch)
        if len(out) >= max(2, int(max_batches)):
            break
    return out


def build_recursive_leaf_prompt(question: str, sources, depth: int, batch_index: int, total_batches: int):
    source_blocks = build_support_audit_source_blocks(sources)
    return (
        f"Recursive controller v1 — map pass (depth {int(depth)}, batch {int(batch_index)}/{int(total_batches)}).\n"
        "Use only SOURCES.\n"
        "Write concise evidence notes with these sections:\n"
        "1) Evidence findings (3-5 bullets).\n"
        "2) Methods/data signals (1-3 bullets).\n"
        "3) Limits/gaps (1-2 bullets).\n"
        "Preserve [S#] labels where available.\n"
        "Do not invent studies or claims.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"SOURCES:\n{source_blocks}\n"
    )


def build_recursive_merge_prompt(question: str, notes, depth: int):
    blocks = []
    for i, note in enumerate(notes or [], start=1):
        txt = clean_text((note or {}).get("text") or "")
        if not txt:
            continue
        if len(txt) > 1500:
            txt = txt[:1500].rstrip() + "..."
        blocks.append(f"[N{i}]\n{txt}")
    joined = "\n\n".join(blocks).strip() or "(no notes)"
    return (
        f"Recursive controller v1 — reduce pass (depth {int(depth)}).\n"
        "Merge NOTES into one grounded synthesis note.\n"
        "Keep the same sections:\n"
        "1) Evidence findings\n"
        "2) Methods/data signals\n"
        "3) Limits/gaps\n"
        "Do not add external facts.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"NOTES:\n{joined}\n"
    )


def build_recursive_final_prompt(question: str, merged_note: str, sources):
    source_blocks = build_support_audit_source_blocks((sources or [])[:14])
    note_text = clean_text(merged_note or "")
    if len(note_text) > 3000:
        note_text = note_text[:3000].rstrip() + "..."
    return (
        "Recursive controller v1 — final synthesis pass.\n"
        "Use MERGED_NOTE and SOURCES only.\n"
        "If evidence is insufficient, answer exactly: Not found in sources.\n"
        "Otherwise produce:\n"
        "1) What your files say (3-6 bullets)\n"
        "2) Agreements/disagreements (2-5 bullets)\n"
        "3) Thin evidence areas (1-2 bullets)\n"
        "Keep source labels [S#] where possible.\n"
        "Use cautious language.\n\n"
        f"QUESTION:\n{question}\n\n"
        f"MERGED_NOTE:\n{note_text}\n\n"
        f"SOURCES:\n{source_blocks}\n"
    )


def run_recursive_controller_v1(
    question: str,
    sources,
    model_chain,
    source_mode: str,
    hybrid_policy: str,
    strict_citations: bool,
    max_depth: int = 2,
    batch_size: int = 6,
    max_batches: int = 6,
    max_calls: int = 18,
):
    meta = {
        "enabled": True,
        "used": False,
        "calls": 0,
        "call_budget": max(4, int(max_calls)),
        "leaf_batches": 0,
        "depth_used": 0,
        "models": [],
    }
    rows = [dict(x) for x in (sources or []) if isinstance(x, dict)]
    if len(rows) < max(4, int(RECURSIVE_CONTROLLER_MIN_SOURCES)):
        meta["reason"] = "insufficient_sources"
        return "", meta
    batches = split_sources_for_recursive_controller(
        rows,
        batch_size=max(2, int(batch_size)),
        max_batches=max(2, int(max_batches)),
    )
    if len(batches) < 2:
        meta["reason"] = "single_batch"
        return "", meta

    def _record_model(name: str):
        nm = clean_text(name or "")
        if nm and nm not in meta["models"]:
            meta["models"].append(nm)

    def _call_budget_available():
        return int(meta.get("calls", 0)) < int(meta.get("call_budget", 0))

    leaf_notes = []
    max_depth = clamp_int(max_depth, 1, 4)
    try:
        for i, batch in enumerate(batches, start=1):
            if not _call_budget_available():
                meta["reason"] = "call_budget_reached"
                meta["budget_exhausted"] = True
                break
            leaf_prompt = build_recursive_leaf_prompt(
                question=question,
                sources=batch,
                depth=1,
                batch_index=i,
                total_batches=len(batches),
            )
            leaf_text, leaf_model = generate_text_via_chain(
                leaf_prompt,
                model_chain,
                system_instruction=system_prompt_for_mode(source_mode, hybrid_policy, strict_citations),
                temperature=0.1,
            )
            meta["calls"] += 1
            _record_model(leaf_model)
            if leaf_text.strip():
                leaf_notes.append({"depth": 1, "batch": i, "text": leaf_text.strip(), "source_count": len(batch)})
        meta["leaf_batches"] = len(leaf_notes)
        if len(leaf_notes) < 2:
            meta["reason"] = "leaf_generation_thin"
            return "", meta

        current_notes = list(leaf_notes)
        depth = 1
        while len(current_notes) > 1 and depth < max_depth:
            next_notes = []
            pairs = [current_notes[i : i + 2] for i in range(0, len(current_notes), 2)]
            for grp in pairs:
                if len(grp) == 1:
                    next_notes.append(grp[0])
                    continue
                if not _call_budget_available():
                    meta["reason"] = "call_budget_reached"
                    meta["budget_exhausted"] = True
                    next_notes.extend(grp)
                    continue
                merge_prompt = build_recursive_merge_prompt(
                    question=question,
                    notes=grp,
                    depth=depth + 1,
                )
                merge_text, merge_model = generate_text_via_chain(
                    merge_prompt,
                    model_chain,
                    system_instruction=system_prompt_for_mode(source_mode, hybrid_policy, strict_citations),
                    temperature=0.1,
                )
                meta["calls"] += 1
                _record_model(merge_model)
                merged = clean_text(merge_text)
                if not merged:
                    merged = "\n\n".join(clean_text(x.get("text") or "") for x in grp if clean_text(x.get("text") or ""))
                next_notes.append({"depth": depth + 1, "text": merged, "source_count": sum(int(x.get("source_count") or 0) for x in grp)})
            current_notes = next_notes
            depth += 1

        merged_note = clean_text((current_notes[0] or {}).get("text") or "")
        if not merged_note:
            meta["reason"] = "empty_merged_note"
            return "", meta

        if not _call_budget_available():
            meta["reason"] = "call_budget_reached"
            meta["budget_exhausted"] = True
            return "", meta

        final_prompt = build_recursive_final_prompt(question=question, merged_note=merged_note, sources=rows)
        final_text, final_model = generate_text_via_chain(
            final_prompt,
            model_chain,
            system_instruction=system_prompt_for_mode(source_mode, hybrid_policy, strict_citations),
            temperature=0.12,
        )
        meta["calls"] += 1
        _record_model(final_model)
        meta["depth_used"] = depth
        if clean_text(final_text):
            meta["used"] = True
            meta["model"] = clean_text(final_model or "")
            return final_text.strip(), meta
        meta["reason"] = "empty_final_text"
        return "", meta
    except Exception as e:
        meta["error"] = str(e)
        return "", meta


def append_key_sources_section(answer_text: str, sources, index_map=None, max_items: int = 12):
    text = clean_text(answer_text or "")
    if not text:
        return answer_text
    if len(sources or []) <= max(8, INLINE_CITATION_MAX):
        return answer_text
    if re.search(r"(?im)^key sources:\s*$", text):
        return answer_text
    lines = []
    for i, src in enumerate((sources or [])[: max(1, int(max_items))], start=1):
        citation = normalized_source_citation(src, index_map=index_map)
        if not citation:
            continue
        lines.append(f"- [S{i}] {citation}")
    if not lines:
        return answer_text
    return (answer_text.rstrip() + "\n\nKey sources:\n" + "\n".join(lines)).strip()


def apply_doc_scope_filter(sources, scoped_doc_path: str):
    scope = clean_text(scoped_doc_path or "").lower()
    if not scope:
        return list(sources or [])
    scope_name = Path(scope).name
    out = []
    for src in (sources or []):
        if not isinstance(src, dict):
            continue
        rel = clean_text(src.get("rel_path") or src.get("uri") or "").lower()
        title = clean_text(src.get("title") or src.get("file_name") or "").lower()
        if scope in rel or scope_name in rel or scope_name in title:
            out.append(src)
    return out


def distill_retrieval_query(user_query: str, history_messages, source_mode: str, model_chain):
    history_messages = history_messages or []
    recent_user = []
    for msg in reversed(history_messages):
        if msg.get("role") == "user":
            txt = clean_text(msg.get("text") or "")
            if txt:
                recent_user.append(txt)
            if len(recent_user) >= 3:
                break
    recent_user.reverse()
    if len(recent_user) < 2:
        return user_query, {"used": False, "reason": "short_history", "query": user_query}

    prompt = (
        "Distill retrieval intent from conversation.\n"
        "Return strict JSON only: {\"query\":\"...\", \"constraints\":[\"...\"], \"terms\":[\"...\"]}\n"
        "- Keep entities, years, acronyms, and constraints.\n"
        "- Query must be <= 180 characters.\n"
        "- Do not add facts.\n\n"
        f"Source mode: {source_mode}\n"
        "Recent user messages:\n"
        + "\n".join(f"- {x}" for x in recent_user[-3:])
        + f"\n\nCurrent question:\n{user_query}\n"
    )
    try:
        text, model_used = generate_text_via_chain(prompt, model_chain, temperature=0.0)
        data = parse_json_object(text) or {}
        distill = clean_text(data.get("query") or user_query)
        if not distill:
            distill = user_query
        constraints = [clean_text(x) for x in (data.get("constraints") or []) if clean_text(x)][:6]
        terms = [clean_text(x) for x in (data.get("terms") or []) if clean_text(x)][:8]
        return distill, {
            "used": True,
            "model": model_used,
            "query": distill,
            "constraints": constraints,
            "terms": terms,
        }
    except Exception as e:
        fallback = clean_text(user_query or "")
        if recent_user:
            fallback = clean_text(f"{recent_user[-1]} {user_query}")
        return fallback, {"used": False, "error": str(e), "query": fallback}


def stoplight_status(gate_msg: str, confidence_score: float, coverage_stats: dict, strict_citations: bool):
    if gate_msg:
        return {"level": "red", "label": "Red | Missing evidence", "reason": gate_msg}
    coverage = float((coverage_stats or {}).get("coverage_ratio") or 0.0)
    audit_unsupported = int((coverage_stats or {}).get("audit_unsupported_count") or 0)
    if audit_unsupported > 0:
        return {
            "level": "yellow",
            "label": "Yellow | Partial support",
            "reason": f"{audit_unsupported} claim(s) removed by support audit",
        }
    if strict_citations and coverage < 0.55:
        return {"level": "red", "label": "Red | Low support coverage", "reason": f"Coverage {coverage:.2f}"}
    if confidence_score < CONFIDENCE_LOW_THRESHOLD or coverage < 0.85:
        return {"level": "yellow", "label": "Yellow | Partial support", "reason": f"score={confidence_score:.2f}, coverage={coverage:.2f}"}
    return {"level": "green", "label": "Green | Fully supported", "reason": f"score={confidence_score:.2f}, coverage={coverage:.2f}"}


def render_stoplight_badge(stoplight: dict):
    level = html.escape(str((stoplight or {}).get("level") or "yellow"))
    label = html.escape(str((stoplight or {}).get("label") or "Yellow | Partial support"))
    st.markdown(f"<div class='stoplight {level}'>{label}</div>", unsafe_allow_html=True)


def render_pipeline_progress(placeholder, step_key: str):
    if placeholder is None:
        return
    steps = [
        ("retrieve", "Retrieving sources"),
        ("extract", "Extracting evidence"),
        ("synthesize", "Synthesizing"),
        ("write", "Writing answer"),
        ("verify", "Verifying support"),
    ]
    order = {k: i for i, (k, _) in enumerate(steps)}
    current = order.get(step_key, 0)
    chips = []
    for idx, (_, label) in enumerate(steps):
        if idx < current:
            chips.append(f"<span class='status-pill'>Done: {html.escape(label)}</span>")
        elif idx == current:
            chips.append(f"<span class='status-pill'>Now: {html.escape(label)}</span>")
        else:
            chips.append(f"<span class='status-pill'>{html.escape(label)}</span>")
    placeholder.markdown("<div class='source-chip-row'>" + "".join(chips) + "</div>", unsafe_allow_html=True)


def build_decision_summary(
    query_intent: str,
    source_mode: str,
    researcher_mode: bool,
    source_count: int,
    file_source_count: int,
):
    mode = "research synthesis pipeline" if researcher_mode else "grounded response pipeline"
    scope = "files only" if source_mode == "Files only" else "files/web hybrid"
    return (
        f"Decision summary: using {mode} ({scope}), "
        f"intent={query_intent}, evidence={file_source_count} file source(s), "
        f"{source_count} total source(s)."
    )


def friendly_model_tier(model_name: str):
    name = clean_text(model_name or "").lower()
    if any(x in name for x in ("flash", "mini", "lite")):
        return "Fast"
    if any(x in name for x in ("pro", "o3", "o4", "reason")):
        return "Strong"
    return "Balanced"


def build_next_question_prompt(question: str, sources):
    blocks = build_support_audit_source_blocks((sources or [])[:8])
    return (
        "Generate grounded follow-up questions.\n"
        "Return strict JSON only: {\"questions\":[\"...\",\"...\",\"...\"]}\n"
        "- Exactly 3 follow-up questions.\n"
        "- Must be answerable from SOURCES.\n"
        "- Keep each question short (max 14 words).\n"
        "- Use action-oriented wording.\n"
        "- Reference specific sections, limitations, or results when possible.\n"
        "- No generic prompts, no preamble.\n\n"
        f"ORIGINAL QUESTION:\n{question}\n\nSOURCES:\n{blocks}\n"
    )


def generate_grounded_followups(question: str, sources, model_chain):
    if not sources:
        return []
    try:
        text, _ = generate_text_via_chain(build_next_question_prompt(question, sources), model_chain, temperature=0.0)
        data = parse_json_object(text) or {}
        qs = []
        for item in (data.get("questions") or []):
            q = clean_text(item)
            if not q:
                continue
            if q not in qs:
                qs.append(q)
            if len(qs) >= 3:
                break
        return qs
    except Exception:
        return []

def nearest_library_matches(query: str, rows, limit: int = 5):
    keys = query_keywords(query, limit=10)
    if not keys or not rows:
        return []
    scored = []
    for row in rows:
        hay = " ".join(
            [
                str(row.get("title_guess", "")),
                str(row.get("author_guess", "")),
                str(row.get("year_guess", "")),
                str(row.get("rel_path", "")),
                str(row.get("project", "")),
                str(row.get("tag", "")),
            ]
        ).lower()
        if not hay.strip():
            continue
        hits = [k for k in keys if k in hay]
        if not hits:
            continue
        score = len(hits) / float(len(keys))
        scored.append((score, hits, row))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for score, hits, row in scored[:limit]:
        item = dict(row)
        item["_match_score"] = round(score, 3)
        item["_match_terms"] = hits
        out.append(item)
    return out


def sentence_tokens(text: str):
    return set(re.findall(r"[A-Za-z0-9]{4,}", (text or "").lower()))


def split_sentences(text: str):
    raw = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    out = []
    for s in raw:
        part = s.strip()
        if part:
            out.append(part)
    return out


def sentence_is_substantive(sentence: str):
    s = (sentence or "").strip()
    if len(s) < 25:
        return False
    if re.match(r"^\[S\d+\]", s):
        return False
    if s.lower().startswith("not found in sources"):
        return False
    words = re.findall(r"[A-Za-z0-9]{3,}", s)
    return len(words) >= 4


def match_sentence_to_source(sentence: str, sources):
    stoks = sentence_tokens(sentence)
    if not stoks:
        return None, 0.0
    best_idx = None
    best_score = 0.0
    for idx, src in enumerate(sources or [], start=1):
        snippet = (src.get("snippet") or "").strip()
        if not snippet:
            continue
        ttoks = sentence_tokens(snippet)
        if not ttoks:
            continue
        inter = len(stoks.intersection(ttoks))
        if inter <= 1:
            continue
        score = inter / float(max(1, len(stoks)))
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx, best_score


def apply_sentence_provenance(answer_text: str, sources, strict_mode: bool):
    body, citation_block = split_answer_and_citations(answer_text)
    sentences = split_sentences(body)
    if not sentences:
        return answer_text, [], 0

    rows = []
    rebuilt = []
    unsupported = 0

    for sentence in sentences:
        row = {"sentence": sentence, "supported": False}
        rendered = sentence

        if re.search(r"\[S\d+\]", sentence):
            m = re.search(r"\[S(\d+)\]", sentence)
            ref_idx = int(m.group(1)) if m else None
            source = (sources or [])[ref_idx - 1] if ref_idx and 1 <= ref_idx <= len(sources or []) else None
            row.update(
                {
                    "supported": bool(source),
                    "source_index": ref_idx if source else None,
                    "source_title": (source or {}).get("title"),
                    "source_snippet": (source or {}).get("snippet"),
                }
            )
            if not source and sentence_is_substantive(sentence):
                unsupported += 1
        elif sentence_is_substantive(sentence):
            src_idx, match_score = match_sentence_to_source(sentence, sources)
            if src_idx:
                source = (sources or [])[src_idx - 1]
                rendered = f"{sentence} [S{src_idx}]"
                row.update(
                    {
                        "supported": True,
                        "source_index": src_idx,
                        "source_title": source.get("title"),
                        "source_snippet": source.get("snippet"),
                        "match_score": round(match_score, 3),
                    }
                )
            else:
                unsupported += 1
                row.update({"supported": False, "source_index": None, "source_title": None, "source_snippet": None})
        else:
            row.update({"supported": True, "source_index": None, "source_title": None, "source_snippet": None})

        rebuilt.append(rendered)
        rows.append(row)

    if strict_mode and unsupported > 0:
        return "Not found in sources.", rows, unsupported

    new_body = " ".join(rebuilt).strip()
    if citation_block:
        new_text = f"{new_body}\n\n{citation_block}".strip()
    else:
        new_text = new_body
    return new_text, rows, unsupported


def citation_coverage_stats(provenance_rows):
    substantive = 0
    supported = 0
    for row in provenance_rows or []:
        sentence = row.get("sentence") or ""
        if not sentence_is_substantive(sentence):
            continue
        substantive += 1
        if row.get("supported"):
            supported += 1
    if substantive <= 0:
        return {"substantive_sentences": 0, "supported_sentences": 0, "coverage_ratio": 1.0}
    return {
        "substantive_sentences": substantive,
        "supported_sentences": supported,
        "coverage_ratio": round(supported / float(substantive), 3),
    }


def verify_password(candidate: str):
    if PASSWORD_HASH:
        return verify_password_hash(candidate, PASSWORD_HASH)
    if PASSWORD:
        return hmac.compare_digest(candidate, PASSWORD)
    return not REQUIRE_PASSWORD


def init_chat_cipher():
    if not CHAT_ENCRYPTION_ENABLED:
        return None
    if Fernet is None:
        return None
    key = CHAT_ENCRYPTION_KEY
    if not key:
        if CHAT_KEY_PATH.exists():
            try:
                key = CHAT_KEY_PATH.read_text().strip()
            except Exception:
                key = ""
        if not key:
            key = Fernet.generate_key().decode()
            try:
                CHAT_KEY_PATH.write_text(key)
                os.chmod(CHAT_KEY_PATH, 0o600)
            except Exception:
                pass
    try:
        return Fernet(key.encode())
    except Exception:
        return None


CHAT_CIPHER = init_chat_cipher()


def encrypt_chat_bytes(raw: bytes):
    if CHAT_CIPHER:
        return CHAT_CIPHER.encrypt(raw)
    return raw


def decrypt_chat_bytes(raw: bytes):
    if CHAT_CIPHER:
        return CHAT_CIPHER.decrypt(raw)
    return raw


def to_role(role: str) -> str:
    return "model" if role == "assistant" else "user"


def build_contents(history):
    contents = []
    for m in history:
        contents.append(
            types.Content(
                role=to_role(m["role"]),
                parts=[types.Part(text=m["text"])],
            )
        )
    return contents


def get_text(resp) -> str:
    text = getattr(resp, "text", None)
    if text:
        return text.strip()

    candidates = getattr(resp, "candidates", []) or []
    if not candidates:
        return ""

    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) if content else None
    if not parts:
        return ""

    chunks = []
    for p in parts:
        t = getattr(p, "text", None)
        if t:
            chunks.append(t)

    return "".join(chunks).strip()


def web_source_domain(uri: str) -> str:
    try:
        host = (urlparse(uri or "").netloc or "").lower().strip()
    except Exception:
        host = ""
    if host.startswith("www."):
        host = host[4:]
    return host


def classify_web_source(uri: str):
    domain = web_source_domain(uri)
    if not domain:
        return {"domain": "", "score": 0.25, "tier": "unknown"}
    high_markers = (
        ".gov",
        ".edu",
        "nature.com",
        "science.org",
        "jamanetwork.com",
        "thelancet.com",
        "nejm.org",
        "wiley.com",
        "springer.com",
        "ieee.org",
        "acm.org",
        "arxiv.org",
        "who.int",
        "cdc.gov",
        "oecd.org",
        "un.org",
    )
    medium_markers = (
        "wikipedia.org",
        "reuters.com",
        "apnews.com",
        "bbc.com",
        "nytimes.com",
        "wsj.com",
        "ft.com",
        "nber.org",
        "imf.org",
        "worldbank.org",
    )
    score = 0.45
    tier = "medium"
    if domain.endswith(".gov") or domain.endswith(".edu") or any(m in domain for m in high_markers):
        score = 0.92
        tier = "high"
    elif any(m in domain for m in medium_markers):
        score = 0.74
        tier = "medium"
    elif any(x in domain for x in ("blog", "medium.com", "substack.com")):
        score = 0.36
        tier = "low"
    return {"domain": domain, "score": score, "tier": tier}


def load_web_cache():
    if not WEB_CACHE_PATH.exists():
        return {}
    try:
        payload = json.loads(WEB_CACHE_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def save_web_cache(cache):
    payload = cache if isinstance(cache, dict) else {}
    try:
        WEB_CACHE_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass


def update_web_cache_from_sources(sources):
    web_sources = [s for s in (sources or []) if isinstance(s, dict) and (s.get("source_type") == "web")]
    if not web_sources:
        return {"updated": 0, "size": len(load_web_cache())}
    cache = load_web_cache()
    now = now_iso()
    updated = 0
    for src in web_sources:
        uri = clean_text(src.get("uri") or "")
        if not uri:
            continue
        trust = classify_web_source(uri)
        old = cache.get(uri) if isinstance(cache.get(uri), dict) else {}
        cache[uri] = {
            "uri": uri,
            "title": clean_text(src.get("title") or old.get("title") or ""),
            "snippet": clean_text(src.get("snippet") or old.get("snippet") or "")[:520],
            "author": clean_text(src.get("author") or old.get("author") or ""),
            "year": clean_text(src.get("year") or old.get("year") or ""),
            "domain": trust.get("domain") or clean_text(src.get("web_domain") or old.get("domain") or ""),
            "trust_tier": trust.get("tier") or clean_text(src.get("web_trust_tier") or old.get("trust_tier") or ""),
            "trust_score": float(src.get("web_trust_score") or trust.get("score") or old.get("trust_score") or 0.0),
            "first_seen": clean_text(old.get("first_seen") or now),
            "last_seen": now,
            "accessed_at": clean_text(src.get("accessed_at") or now),
        }
        updated += 1
    if len(cache) > WEB_CACHE_MAX_ENTRIES:
        rows = []
        for key, val in cache.items():
            if isinstance(val, dict):
                rows.append((clean_text(val.get("last_seen") or ""), key, val))
        rows.sort(key=lambda x: x[0], reverse=True)
        keep = rows[:WEB_CACHE_MAX_ENTRIES]
        cache = {k: v for _, k, v in keep}
    save_web_cache(cache)
    return {"updated": updated, "size": len(cache)}


def clear_web_cache():
    save_web_cache({})


def _metadata_scalar(value):
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        txt = clean_text(str(value))
        return txt
    for attr in ("string_value", "stringValue", "number_value", "numberValue", "bool_value", "boolValue"):
        v = getattr(value, attr, None)
        if v is not None:
            return clean_text(str(v))
    if isinstance(value, dict):
        for key in ("string_value", "stringValue", "number_value", "numberValue", "bool_value", "boolValue", "value"):
            if key in value and value.get(key) is not None:
                return clean_text(str(value.get(key)))
    return ""


def _metadata_to_dict(raw):
    out = {}
    if raw is None:
        return out
    if isinstance(raw, dict):
        for k, v in raw.items():
            key = clean_text(str(k)).lower()
            val = _metadata_scalar(v)
            if key and val:
                out[key] = val
        return out
    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                key = clean_text(str(item.get("key") or item.get("name") or "")).lower()
                val = _metadata_scalar(item.get("value"))
                if not val:
                    val = _metadata_scalar(item)
            else:
                key = clean_text(str(getattr(item, "key", None) or getattr(item, "name", None) or "")).lower()
                val = _metadata_scalar(getattr(item, "value", None) or item)
            if key and val:
                out[key] = val
        return out
    return out


def _extract_source_metadata(obj):
    merged = {}
    if obj is None:
        return merged
    for field in (
        "custom_metadata",
        "customMetadata",
        "metadata",
        "document_metadata",
        "documentMetadata",
        "attributes",
    ):
        raw = getattr(obj, field, None)
        if raw is None:
            continue
        meta = _metadata_to_dict(raw)
        for k, v in meta.items():
            if k and v and k not in merged:
                merged[k] = v
    return merged


def _merge_source_row(dst: dict, src: dict):
    for key in (
        "title",
        "file_name",
        "display_name",
        "uri",
        "source_type",
        "author",
        "year",
        "rel_path",
        "project",
        "tag",
        "doc_type",
        "version_stage",
        "section_heading",
        "citation_source",
        "vault_export_id",
        "vault_export_date",
        "vault_custodian",
        "vault_matter_name",
        "web_domain",
        "web_trust_tier",
        "accessed_at",
    ):
        if not dst.get(key) and src.get(key):
            dst[key] = src.get(key)
    left = clean_text(dst.get("snippet") or "")
    right = clean_text(src.get("snippet") or "")
    if len(right) > len(left):
        dst["snippet"] = right[:MAX_SNIPPET_CHARS]
    for num_key in ("page", "chunk", "score", "vector_score", "bm25_score", "rerank_score"):
        if dst.get(num_key) is None and src.get(num_key) is not None:
            dst[num_key] = src.get(num_key)
    if dst.get("web_trust_score") is None and src.get("web_trust_score") is not None:
        dst["web_trust_score"] = src.get("web_trust_score")
    return dst


def extract_sources(resp):
    sources = []
    candidates = getattr(resp, "candidates", []) or []
    if not candidates:
        return sources

    cand = candidates[0]

    cm = getattr(cand, "citation_metadata", None) or getattr(cand, "citationMetadata", None)
    if cm:
        citation_sources = (
            getattr(cm, "citation_sources", None)
            or getattr(cm, "citationSources", None)
            or []
        )
        for s in citation_sources:
            uri = getattr(s, "uri", None) or getattr(s, "url", None)
            title = getattr(s, "title", None)
            source_type = "web" if (uri or "").startswith("http") else "file"
            meta = _extract_source_metadata(s)
            row = {"title": title, "uri": uri, "snippet": None, "source_type": source_type}
            for key in (
                "author",
                "year",
                "rel_path",
                "file_name",
                "display_name",
                "project",
                "tag",
                "doc_type",
                "version_stage",
                "section_heading",
                "citation_source",
                "vault_export_id",
                "vault_export_date",
                "vault_custodian",
                "vault_matter_name",
            ):
                if meta.get(key):
                    row[key] = meta.get(key)
            if not row.get("title") and meta.get("title"):
                row["title"] = meta.get("title")
            if source_type == "web":
                trust = classify_web_source(uri or "")
                row["web_domain"] = trust.get("domain")
                row["web_trust_tier"] = trust.get("tier")
                row["web_trust_score"] = trust.get("score")
                row["accessed_at"] = now_iso()
            sources.append(row)

    gm = getattr(cand, "grounding_metadata", None) or getattr(cand, "groundingMetadata", None)
    if gm:
        chunks = (
            getattr(gm, "grounding_chunks", None)
            or getattr(gm, "groundingChunks", None)
            or []
        )
        for ch in chunks:
            for field in ("retrieved_context", "retrievedContext", "web", "document"):
                rc = getattr(ch, field, None)
                if not rc:
                    continue
                uri = getattr(rc, "uri", None) or getattr(rc, "url", None)
                title = (
                    getattr(rc, "title", None)
                    or getattr(rc, "displayName", None)
                    or getattr(rc, "display_name", None)
                )
                snippet = (
                    getattr(rc, "text", None)
                    or getattr(rc, "snippet", None)
                    or getattr(rc, "content", None)
                )
                if snippet:
                    snippet = snippet[:MAX_SNIPPET_CHARS]
                source_type = "web" if field == "web" or (uri or "").startswith("http") else "file"
                meta = _extract_source_metadata(rc)
                row = {
                    "title": title,
                    "uri": uri,
                    "snippet": snippet,
                    "source_type": source_type,
                }
                for key in (
                    "author",
                    "year",
                    "rel_path",
                    "file_name",
                    "display_name",
                    "project",
                    "tag",
                    "doc_type",
                    "version_stage",
                    "section_heading",
                    "citation_source",
                    "vault_export_id",
                    "vault_export_date",
                    "vault_custodian",
                    "vault_matter_name",
                ):
                    if meta.get(key):
                        row[key] = meta.get(key)
                if not row.get("title") and meta.get("title"):
                    row["title"] = meta.get("title")
                for num_field in ("page", "chunk", "score", "vector_score", "bm25_score", "rerank_score"):
                    raw_num = getattr(rc, num_field, None)
                    if raw_num is not None:
                        row[num_field] = raw_num
                if source_type == "web":
                    trust = classify_web_source(uri or "")
                    row["web_domain"] = trust.get("domain")
                    row["web_trust_tier"] = trust.get("tier")
                    row["web_trust_score"] = trust.get("score")
                    row["accessed_at"] = now_iso()
                sources.append(row)

    # de-dupe
    deduped = {}
    order = []
    for s in sources:
        key = (s.get("uri") or s.get("title") or "").strip()
        if not key:
            continue
        if key not in deduped:
            deduped[key] = dict(s)
            order.append(key)
        else:
            deduped[key] = _merge_source_row(deduped[key], s)
    return [deduped[k] for k in order]


def build_reasoning_rows(query: str, sources, index_map=None, limit: int = 3):
    rows = []
    index_map = index_map or {}
    q_terms = set(query_keywords(query, limit=8))
    for idx, source in enumerate((sources or []), start=1):
        if not isinstance(source, dict):
            continue
        info = enrich_source(source, index_map)
        if should_hide_source_by_default(source, info):
            continue
        label = derive_title_from_source(source, info)
        author = clean_text(source.get("author") or info.get("author") or "")
        year = clean_text(source.get("year") or info.get("year") or "")
        score = source.get("score")
        score_txt = ""
        if isinstance(score, (int, float)):
            score_txt = f"{float(score):.2f}"
        clip = compact_snippet_for_query(source.get("snippet") or "", query, max_sentences=2)
        if not clip:
            clip = clean_text(source.get("snippet") or "")[:320]
        stoks = set(re.findall(r"[A-Za-z0-9]{3,}", clip.lower()))
        overlap = len(stoks.intersection(q_terms)) if q_terms else 0
        reason_bits = []
        if score_txt:
            reason_bits.append(f"high relevance score ({score_txt})")
        if overlap:
            reason_bits.append(f"matches {overlap} query term{'s' if overlap != 1 else ''}")
        page = source.get("page")
        if page:
            reason_bits.append(f"evidence on page {page}")
        if author or year:
            who = author or "Unknown author"
            when = year or "n.d."
            reason_bits.append(f"{who} ({when})")
        reason = ", ".join(reason_bits[:3]) if reason_bits else "selected as direct supporting evidence"
        sort_score = float(score) if isinstance(score, (int, float)) else 0.0
        rows.append(
            {
                "index": idx,
                "label": label,
                "reason": reason,
                "clip": clip,
                "sort_score": sort_score + (0.03 * overlap),
            }
        )
    rows.sort(key=lambda x: x["sort_score"], reverse=True)
    return rows[: max(1, int(limit))]


def render_source_chips(sources, index_map=None, max_chips: int = 6):
    if not sources:
        return
    index_map = index_map or {}
    ref_maps = build_source_ref_maps(sources)
    chips = []
    for idx, src in enumerate(sources, start=1):
        if not isinstance(src, dict):
            continue
        info = enrich_source(src, index_map)
        if should_hide_source_by_default(src, info):
            continue
        token = ref_maps.get("index_to_typed", {}).get(idx, f"S{idx}")
        citation = normalized_source_citation(src, index_map=index_map)
        short = citation if len(citation) <= 64 else (citation[:61].rstrip() + "...")
        open_uri = source_open_uri(src) or clean_text(src.get("uri") or "")
        label = f"[{token}] {short}"
        if open_uri:
            chips.append(
                f"<a class='source-chip' href='{html.escape(open_uri)}' target='_blank'>{html.escape(label)}</a>"
            )
        else:
            chips.append(f"<span class='source-chip'>{html.escape(label)}</span>")
        if len(chips) >= max(1, int(max_chips)):
            break
    if chips:
        st.markdown("<div class='source-chip-row'>" + "".join(chips) + "</div>", unsafe_allow_html=True)


def render_reasoning_panel(sources, query: str = "", index_map=None):
    rows = build_reasoning_rows(query=query or "", sources=sources or [], index_map=index_map or {}, limit=3)
    if not rows:
        return
    ref_maps = build_source_ref_maps(sources or [])
    with st.expander("Reasoning", expanded=False):
        st.caption("How this answer was grounded in source evidence.")
        for row in rows:
            ref_token = ref_maps.get("index_to_typed", {}).get(int(row.get("index", 0)), f"S{int(row.get('index', 0))}")
            st.markdown(
                f"- **[{html.escape(ref_token)}] {html.escape(str(row.get('label') or 'source'))}** — "
                f"{html.escape(str(row.get('reason') or ''))}",
                unsafe_allow_html=True,
            )
            clip = clean_text(row.get("clip") or "")
            if clip:
                st.markdown(
                    f"<div class='source-card-quote'>{highlight_snippet(clip, query or '')}</div>",
                    unsafe_allow_html=True,
                )


def render_sources(sources, query="", audit_mode=False, index_map=None):
    if not sources:
        return
    index_map = index_map or {}
    ref_maps = build_source_ref_maps(sources)
    prepared = []
    for idx, s in enumerate(sources):
        info = enrich_source(s, index_map)
        hide_default = should_hide_source_by_default(s, info)
        if hide_default and not audit_mode:
            continue
        label = derive_title_from_source(s, info)
        prepared.append((idx + 1, s, info, label))
    hidden_count = max(0, len(sources) - len(prepared))
    if not prepared and not audit_mode:
        with st.expander("Evidence", expanded=False):
            if hidden_count:
                st.caption("All retrieved sources are system-style IDs. Turn on Audit to show all sources.")
            else:
                st.caption("No sources available.")
        with st.expander("Citations", expanded=False):
            if hidden_count:
                st.caption("All retrieved sources are system-style IDs. Turn on Audit to show all source entries.")
            else:
                st.caption("No citations available.")
        return

    with st.expander("Evidence", expanded=False):
        if hidden_count and not audit_mode:
            st.caption(f"{hidden_count} system-style source IDs hidden. Turn on Audit to show all sources.")
        for ref_idx, s, info, label in prepared:
            typed_ref = ref_maps.get("index_to_typed", {}).get(ref_idx, f"S{ref_idx}")
            src_type = s.get("source_type", "file")
            author = clean_text(s.get("author") or info.get("author") or "")
            year = clean_text(s.get("year") or info.get("year") or "")
            score = s.get("score")
            page = s.get("page")
            chunk = s.get("chunk")
            section = (s.get("section_heading") or "").strip()
            doc_type = (s.get("doc_type") or info.get("doc_type") or "").strip()
            version_stage = (s.get("version_stage") or info.get("version_stage") or "").strip()
            doc_family = (s.get("doc_family") or "").strip()
            figure_table_markers = (s.get("figure_table_markers") or "").strip()
            equation_markers = (s.get("equation_markers") or "").strip()
            stitch_span = (s.get("stitch_span") or "").strip()
            web_domain = clean_text(s.get("web_domain") or info.get("web_domain") or "")
            web_trust_tier = clean_text(s.get("web_trust_tier") or info.get("web_trust_tier") or "")
            accessed_at = clean_text(s.get("accessed_at") or info.get("accessed_at") or "")
            web_trust_score = s.get("web_trust_score")
            vault_export_id = clean_text(s.get("vault_export_id") or "")
            vault_export_date = clean_text(s.get("vault_export_date") or "")
            vault_custodian = clean_text(s.get("vault_custodian") or "")
            vault_matter_name = clean_text(s.get("vault_matter_name") or "")

            st.markdown("<div class='source-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='source-card-title'>[{html.escape(typed_ref)}] {html.escape(label)}</div>", unsafe_allow_html=True)
            meta_bits = [f"type={src_type}"]
            if author or year:
                who = author or "Unknown author"
                when = year or "n.d."
                meta_bits.insert(0, f"{who} ({when})")
            if isinstance(score, (int, float)):
                meta_bits.append(f"score={float(score):.3f}")
            if page:
                meta_bits.append(f"p.{page}")
            if chunk is not None:
                meta_bits.append(f"chunk {chunk}")
            if section:
                meta_bits.append(f"section: {section}")
            if doc_type:
                meta_bits.append(f"doc_type: {doc_type}")
            if version_stage and version_stage != "unknown":
                meta_bits.append(f"version: {version_stage}")
            if doc_family:
                meta_bits.append(f"family: {doc_family}")
            if vault_export_id:
                meta_bits.append(f"export: {vault_export_id}")
            if vault_export_date:
                meta_bits.append(f"export_date: {vault_export_date}")
            if vault_custodian:
                meta_bits.append(f"custodian: {vault_custodian}")
            if vault_matter_name:
                meta_bits.append(f"matter: {vault_matter_name}")
            if figure_table_markers:
                meta_bits.append(f"fig/table: {figure_table_markers}")
            if equation_markers:
                meta_bits.append(f"equation: {equation_markers}")
            if stitch_span:
                meta_bits.append(f"span: {stitch_span}")
            if web_domain:
                meta_bits.append(f"domain: {web_domain}")
            if web_trust_tier:
                meta_bits.append(f"trust: {web_trust_tier}")
            if isinstance(web_trust_score, (int, float)):
                meta_bits.append(f"trust_score={float(web_trust_score):.2f}")
            if accessed_at:
                meta_bits.append(f"accessed: {accessed_at[:10]}")
            safe_meta = [html.escape(str(bit)) for bit in meta_bits]
            st.markdown(f"<div class='source-card-meta'>{' | '.join(safe_meta)}</div>", unsafe_allow_html=True)

            snippet = s.get("snippet") or ""
            if snippet:
                clip = compact_snippet_for_query(snippet, query, max_sentences=3) if query else clean_text(snippet)[:420]
                if not clip:
                    clip = clean_text(snippet)[:420]
                highlighted = highlight_snippet(clip, query)
                st.markdown(f"<div class='source-card-quote'>{highlighted}</div>", unsafe_allow_html=True)
            if s.get("uri"):
                st.caption(s["uri"])
            open_uri = source_open_uri(s)
            if open_uri:
                st.markdown(f"[Open cited page]({open_uri})")

            citation = normalized_source_citation(s, index_map=index_map)
            copy_button(citation)

            if audit_mode:
                st.caption(f"Normalized citation: {citation}")
                if info.get("rel_path"):
                    st.caption(f"Library file: {info['rel_path']}")
                st.caption(f"Citation source: {info.get('citation_source', 'filename')}")

            st.markdown("</div>", unsafe_allow_html=True)

    with st.expander("Citations", expanded=False):
        if hidden_count and not audit_mode:
            st.caption(f"{hidden_count} system-style source IDs hidden. Turn on Audit to show all source entries.")
        for ref_idx, s, _, _ in prepared:
            typed_ref = ref_maps.get("index_to_typed", {}).get(ref_idx, f"S{ref_idx}")
            label = html.escape(normalized_source_citation(s, index_map=index_map))
            uri = html.escape((s.get("uri") or "").strip())
            if uri and uri != label:
                st.markdown(f"- [{html.escape(typed_ref)}] {label} ({uri})")
            else:
                st.markdown(f"- [{html.escape(typed_ref)}] {label}")


def render_web_usage_summary(sources):
    web_sources = [s for s in (sources or []) if isinstance(s, dict) and s.get("source_type") == "web"]
    if not web_sources:
        return
    domains = []
    for s in web_sources:
        dom = clean_text(s.get("web_domain") or web_source_domain(s.get("uri") or ""))
        if dom and dom not in domains:
            domains.append(dom)
    st.caption(
        f"Web sources used: {len(web_sources)}"
        + (f" across {len(domains)} domain(s)." if domains else ".")
    )
    with st.expander("Web sources used", expanded=False):
        for s in web_sources[:12]:
            uri = clean_text(s.get("uri") or "")
            domain = clean_text(s.get("web_domain") or web_source_domain(uri))
            trust = clean_text(s.get("web_trust_tier") or "")
            accessed = clean_text(s.get("accessed_at") or "")
            line = []
            if domain:
                line.append(domain)
            if trust:
                line.append(f"trust={trust}")
            if accessed:
                line.append(f"accessed={accessed[:10]}")
            if line:
                st.caption(" | ".join(line))
            if uri:
                st.markdown(f"- {uri}")


def render_sentence_provenance(rows, render_key: str, sources=None):
    if not rows:
        return
    ref_maps = build_source_ref_maps(sources or [])
    options = list(range(len(rows)))

    def _fmt(idx):
        row = rows[idx]
        prefix = "Supported" if row.get("supported") else "Unsupported"
        sentence = (row.get("sentence") or "").strip().replace("\n", " ")
        if len(sentence) > 90:
            sentence = sentence[:87] + "..."
        return f"{prefix}: {sentence}"

    with st.expander("Sentence provenance", expanded=False):
        pick = st.selectbox(
            "Choose sentence",
            options,
            format_func=_fmt,
            key=f"prov_pick_{render_key}",
        )
        row = rows[pick]
        st.markdown(row.get("sentence", ""))
        if row.get("supported") and row.get("source_snippet"):
            src_idx = row.get("source_index")
            if src_idx is None:
                token = "S?"
            else:
                token = ref_maps.get("index_to_typed", {}).get(int(src_idx), f"S{int(src_idx)}")
            label = row.get("source_title") or token
            st.caption(f"Support: [{token}] {label}")
            st.markdown(highlight_snippet(row.get("source_snippet", ""), row.get("sentence", "")), unsafe_allow_html=True)
        elif row.get("supported"):
            st.caption("Support: non-factual or already cited sentence.")
        else:
            st.caption("Support: not matched to a source snippet.")


def wait_op(op, poll_s=2):
    while not getattr(op, "done", False):
        time.sleep(poll_s)
        op = client.operations.get(op)
    return op


def upload_store_file(path: str, config: dict):
    for key in ("local_file_path", "file_path"):
        try:
            return client.file_search_stores.upload_to_file_search_store(
                file_search_store_name=STORE_ID,
                config=config,
                **{key: str(path)},
            )
        except TypeError as e:
            if key == "file_path":
                raise
            if "unexpected keyword argument" not in str(e).lower():
                raise
    return client.file_search_stores.import_file(
        file_search_store_name=STORE_ID,
        file_path=str(path),
        config=config,
    )


def sanitize_upload_filename(value: str):
    base = Path(str(value or "")).name.replace("\x00", "")
    cleaned = SAFE_UPLOAD_NAME_RE.sub("_", base).strip(" .")
    if not cleaned:
        cleaned = f"upload_{int(time.time())}.bin"
    if len(cleaned) > 240:
        stem = Path(cleaned).stem[:200]
        suffix = Path(cleaned).suffix[:32]
        cleaned = f"{stem}{suffix}" or f"upload_{int(time.time())}.bin"
    return cleaned


def upload_file(uploaded_file, project="", retrieval_backend="google"):
    if not uploaded_file:
        return
    if not enforce_rate_limit(
        action="mutation_upload",
        limit_count=RATE_LIMIT_MUTATION_MAX,
        window_seconds=RATE_LIMIT_MUTATION_WINDOW_SECONDS,
        label="Upload",
    ):
        return

    safe_name = sanitize_upload_filename(uploaded_file.name)
    suffix = Path(safe_name).suffix.lower()
    if suffix not in UPLOAD_EXTENSIONS:
        st.error("Unsupported file type.")
        return
    size_bytes = int(getattr(uploaded_file, "size", 0) or 0)
    max_bytes = MAX_FILE_MB * 1024 * 1024
    if size_bytes > max_bytes:
        st.error(f"File too large ({size_bytes / (1024*1024):.1f} MB). Max is {MAX_FILE_MB} MB.")
        return
    file_bytes = uploaded_file.getbuffer()
    if size_bytes <= 0:
        size_bytes = len(file_bytes)
        if size_bytes > max_bytes:
            st.error(f"File too large ({size_bytes / (1024*1024):.1f} MB). Max is {MAX_FILE_MB} MB.")
            return

    project = "" if project == "All" else project
    tag = infer_tag_from_filename(safe_name)
    rel_path = f"{project}/{safe_name}" if project else safe_name

    if retrieval_backend == "chroma":
        if not DATA_ROOT:
            st.sidebar.error("EDITH_DATA_ROOT is required for Local Chroma uploads.")
            return
        root = Path(DATA_ROOT).expanduser()
        target_dir = root / project if project else root
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe_name
        root_resolved = root.resolve()
        target_resolved = target_path.resolve()
        if root_resolved != target_resolved and root_resolved not in target_resolved.parents:
            st.sidebar.error("Upload path validation failed.")
            return
        if target_path.exists():
            stem = target_path.stem
            ext = target_path.suffix
            target_path = target_dir / f"{stem}_{int(time.time())}{ext}"
        try:
            target_path.write_bytes(file_bytes)
            st.sidebar.success(f"Saved locally: {target_path.name}")
            st.sidebar.caption("Run Reindex files to include this in Chroma.")
            if tag:
                st.sidebar.caption(f"Detected tag: {tag}")
            return
        except Exception as e:
            st.sidebar.error(f"Local save error: {e}")
            return

    cloud_upload_allowed = bool(st.session_state.get("cloud_index_opt_in", CLOUD_INDEX_OPT_IN))
    if not cloud_upload_allowed:
        st.sidebar.error("Cloud index uploads are off. Enable Cloud index uploads in Privacy and Data first.")
        return

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        op = upload_store_file(
            tmp_path,
            {
                "display_name": safe_name,
                "chunking_config": {"white_space_config": CHUNK},
                "custom_metadata": {
                    "rel_path": rel_path,
                    "project": project,
                    "tag": tag,
                },
            },
        )
        wait_op(op)
        st.sidebar.success(f"Stored: {safe_name}")
    except Exception as e:
        st.sidebar.error(f"Upload error: {e}")
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def run_reindex(retrieval_backend="google"):
    if not enforce_rate_limit(
        action="mutation_reindex",
        limit_count=RATE_LIMIT_MUTATION_MAX,
        window_seconds=RATE_LIMIT_MUTATION_WINDOW_SECONDS,
        label="Reindex",
    ):
        return 2, "Rate-limited. Retry reindex in a few moments."
    if retrieval_backend == "google" and not bool(st.session_state.get("cloud_index_opt_in", CLOUD_INDEX_OPT_IN)):
        return 2, "Cloud index uploads are disabled. Enable Cloud index uploads in Privacy and Data to continue."
    script_name = "index_files.py" if retrieval_backend == "google" else "chroma_index.py"
    script = Path(__file__).parent / script_name
    if not script.exists():
        return 1, f"{script_name} not found."

    env = os.environ.copy()
    env.setdefault("EDITH_APP_DATA_DIR", str(APP_STATE_DIR))
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
            env=env,
            timeout=REINDEX_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        parts = [p for p in [exc.stdout, exc.stderr] if p]
        tail = "\n".join(parts).strip()
        msg = f"Reindex timed out after {REINDEX_TIMEOUT_SECONDS} seconds."
        if tail:
            msg = f"{msg}\n\n{tail}"
        return 1, msg
    output = "\n".join([p for p in [result.stdout, result.stderr] if p]).strip() or "(No output)"
    return result.returncode, output


def load_index_status():
    if not INDEX_STATUS_PATH.exists():
        return {}
    try:
        payload = json.loads(INDEX_STATUS_PATH.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def save_index_status(code: int, output: str):
    payload = {
        "last_run_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "last_code": int(code),
        "last_ok": int(code) == 0,
        "last_error": "" if int(code) == 0 else (str(output or "")[:1000]),
    }
    try:
        INDEX_STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def parse_index_error_root(last_error: str):
    text = clean_text(last_error or "")
    if not text:
        return ""
    m = re.search(r"EDITH_DATA_ROOT not found(?: or not a directory)?:\s*(.+)", text)
    if m:
        return clean_text(m.group(1))
    return ""


def detect_library_mismatch(index_status: dict, data_root: str):
    root = clean_text(data_root or "")
    if not root:
        return ""
    err = clean_text((index_status or {}).get("last_error") or "")
    err_root = parse_index_error_root(err)
    if not err_root:
        return ""
    try:
        root_norm = str(Path(root).expanduser().resolve())
    except Exception:
        root_norm = root
    try:
        err_norm = str(Path(err_root).expanduser().resolve())
    except Exception:
        err_norm = err_root
    if root_norm != err_norm:
        return err_root
    return ""


def run_vault_sync(no_index: bool = False):
    if not enforce_rate_limit(
        action="mutation_vault_sync",
        limit_count=RATE_LIMIT_MUTATION_MAX,
        window_seconds=RATE_LIMIT_MUTATION_WINDOW_SECONDS,
        label="Vault sync",
    ):
        return 2, "Rate-limited. Retry vault sync in a few moments."
    if not bool(st.session_state.get("cloud_index_opt_in", CLOUD_INDEX_OPT_IN)):
        return 2, "Cloud index uploads are disabled. Enable Cloud index uploads in Privacy and Data to run Vault import."
    script_py = Path(__file__).parent / "scripts" / "sync_vault_exports.py"
    script_sh = Path(__file__).parent / "sync_vault_exports.sh"
    if script_py.exists():
        cmd = [sys.executable, str(script_py)]
    elif script_sh.exists():
        cmd = [str(script_sh)]
    else:
        return 1, "Vault export import script not found."
    if no_index:
        cmd.append("--no-index")

    env = os.environ.copy()
    env.setdefault("EDITH_APP_DATA_DIR", str(APP_STATE_DIR))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
            env=env,
            timeout=max(REINDEX_TIMEOUT_SECONDS, 300),
        )
    except subprocess.TimeoutExpired as exc:
        parts = [p for p in [exc.stdout, exc.stderr] if p]
        tail = "\n".join(parts).strip()
        msg = f"Vault export import timed out after {max(REINDEX_TIMEOUT_SECONDS, 300)} seconds."
        if tail:
            msg = f"{msg}\n\n{tail}"
        return 1, msg
    output = "\n".join([p for p in [result.stdout, result.stderr] if p]).strip() or "(No output)"
    return result.returncode, output


def run_vault_list(limit: int = 200, contains: str = ""):
    if not enforce_rate_limit(
        action="mutation_vault_list",
        limit_count=max(4, RATE_LIMIT_MUTATION_MAX),
        window_seconds=RATE_LIMIT_MUTATION_WINDOW_SECONDS,
        label="Vault inventory refresh",
    ):
        return 2, "Rate-limited. Retry vault inventory in a few moments."
    script = Path(__file__).parent / "scripts" / "list_vault_docs.py"
    if not script.exists():
        return 1, "Vault list script not found."
    cmd = [sys.executable, str(script), "--limit", str(max(1, int(limit)))]
    term = (contains or "").strip()
    if term:
        cmd.extend(["--contains", term])
    env = os.environ.copy()
    env.setdefault("EDITH_APP_DATA_DIR", str(APP_STATE_DIR))
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
            env=env,
            timeout=max(60, min(REINDEX_TIMEOUT_SECONDS, 600)),
        )
    except subprocess.TimeoutExpired as exc:
        parts = [p for p in [exc.stdout, exc.stderr] if p]
        tail = "\n".join(parts).strip()
        msg = "Vault list timed out."
        if tail:
            msg = f"{msg}\n\n{tail}"
        return 1, msg
    output = "\n".join([p for p in [result.stdout, result.stderr] if p]).strip() or "(No output)"
    return result.returncode, output


def parse_vault_list_output(output: str):
    rows = []
    total = None
    for line in (output or "").splitlines():
        ln = line.strip()
        if not ln:
            continue
        if ln.lower().startswith("shown:"):
            try:
                total = int(ln.split(":", 1)[1].strip())
            except Exception:
                total = None
            continue
        m = re.match(r"^(\d+)\.\s+(.*)$", ln)
        if not m:
            continue
        rows.append({"#": int(m.group(1)), "document": m.group(2).strip()})
    return rows, total


def load_last_vault_sync_summary():
    if not DATA_ROOT:
        return {}
    p = Path(DATA_ROOT).expanduser() / "vault_sync" / "_reports" / "last_sync_summary.json"
    if not p.exists():
        return {}
    try:
        obj = json.loads(p.read_text(encoding="utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def run_build_phd_indexes():
    if IS_FROZEN_APP:
        return 1, "PhD index build is disabled in packaged mode. Run scripts/build_phd_os_indexes.py from source."
    script = Path(__file__).parent / "scripts" / "build_phd_os_indexes.py"
    if not script.exists():
        return 1, "scripts/build_phd_os_indexes.py not found."
    env = os.environ.copy()
    env.setdefault("EDITH_APP_DATA_DIR", str(APP_STATE_DIR))
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
            env=env,
            timeout=REINDEX_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        parts = [p for p in [exc.stdout, exc.stderr] if p]
        tail = "\n".join(parts).strip()
        msg = f"PhD index build timed out after {REINDEX_TIMEOUT_SECONDS} seconds."
        if tail:
            msg = f"{msg}\n\n{tail}"
        return 1, msg
    output = "\n".join([p for p in [result.stdout, result.stderr] if p]).strip() or "(No output)"
    return result.returncode, output


def run_index_health_report():
    if IS_FROZEN_APP:
        return 1, "Index health report is disabled in packaged mode. Run scripts/index_health_report.py from source."
    script = Path(__file__).parent / "scripts" / "index_health_report.py"
    if not script.exists():
        return 1, "scripts/index_health_report.py not found."
    env = os.environ.copy()
    env.setdefault("EDITH_APP_DATA_DIR", str(APP_STATE_DIR))
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
            env=env,
            timeout=REINDEX_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        parts = [p for p in [exc.stdout, exc.stderr] if p]
        tail = "\n".join(parts).strip()
        msg = f"Index health report timed out after {REINDEX_TIMEOUT_SECONDS} seconds."
        if tail:
            msg = f"{msg}\n\n{tail}"
        return 1, msg
    output = "\n".join([p for p in [result.stdout, result.stderr] if p]).strip() or "(No output)"
    return result.returncode, output


def run_corpus_snapshot():
    if IS_FROZEN_APP:
        return 1, "Corpus snapshot is disabled in packaged mode. Run scripts/corpus_snapshot.py from source."
    script = Path(__file__).parent / "scripts" / "corpus_snapshot.py"
    if not script.exists():
        return 1, "scripts/corpus_snapshot.py not found."
    env = os.environ.copy()
    env.setdefault("EDITH_APP_DATA_DIR", str(APP_STATE_DIR))
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).parent),
            env=env,
            timeout=REINDEX_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired as exc:
        parts = [p for p in [exc.stdout, exc.stderr] if p]
        tail = "\n".join(parts).strip()
        msg = f"Corpus snapshot timed out after {REINDEX_TIMEOUT_SECONDS} seconds."
        if tail:
            msg = f"{msg}\n\n{tail}"
        return 1, msg
    output = "\n".join([p for p in [result.stdout, result.stderr] if p]).strip() or "(No output)"
    return result.returncode, output


def list_projects(data_root: str):
    if not data_root:
        return []
    root = Path(data_root).expanduser()
    if not root.exists():
        return []
    projects = []
    for p in root.iterdir():
        if p.is_dir() and not p.name.startswith("."):
            projects.append(p.name)
    return sorted(projects)


@st.cache_data(show_spinner=False)
def load_index_report(mtime: float):
    if not INDEX_REPORT.exists():
        return {}
    mapping = {}
    try:
        with INDEX_REPORT.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                name_key = (row.get("file_name") or "").strip()
                rel_key = (row.get("rel_path") or "").strip()
                if name_key:
                    key = name_key.lower()
                    if key not in mapping:
                        mapping[key] = row
                if rel_key:
                    key = rel_key.lower()
                    if key not in mapping:
                        mapping[key] = row
    except Exception:
        return {}
    return mapping


@st.cache_data(show_spinner=False)
def load_index_rows(mtime: float):
    if not INDEX_REPORT.exists():
        return []
    rows = []
    try:
        with INDEX_REPORT.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(
                    {
                        "title": row.get("title_guess", ""),
                        "author": row.get("author_guess", ""),
                        "year": row.get("year_guess", ""),
                        "doc_type": row.get("doc_type", ""),
                        "version_stage": row.get("version_stage", ""),
                        "ocr_used": row.get("ocr_used", ""),
                        "citation_source": row.get("citation_source", ""),
                        "vault_export_id": row.get("vault_export_id", ""),
                        "vault_export_date": row.get("vault_export_date", ""),
                        "vault_custodian": row.get("vault_custodian", ""),
                        "vault_matter_name": row.get("vault_matter_name", ""),
                        "project": row.get("project", ""),
                        "tag": row.get("tag", ""),
                        "file_name": row.get("file_name", ""),
                        "rel_path": row.get("rel_path", ""),
                    }
                )
    except Exception:
        return []
    return rows


@st.cache_data(show_spinner=False)
def estimate_index_queue(data_root: str, report_mtime: float):
    root = Path(data_root or "").expanduser()
    if not root.exists() or not root.is_dir():
        return {"total_files": 0, "indexed_docs": 0, "pending_files": 0}
    indexed_rel = set()
    if INDEX_REPORT.exists():
        try:
            with INDEX_REPORT.open("r", newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    rel = clean_text(row.get("rel_path") or "").lower()
                    if rel:
                        indexed_rel.add(rel)
        except Exception:
            indexed_rel = set()
    total_supported = 0
    pending = 0
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if fn.startswith("."):
                continue
            p = Path(dirpath) / fn
            if p.suffix.lower() not in UPLOAD_EXTENSIONS:
                continue
            total_supported += 1
            rel = clean_text(str(p.relative_to(root))).lower()
            if rel not in indexed_rel:
                pending += 1
    return {
        "total_files": int(total_supported),
        "indexed_docs": int(len(indexed_rel)),
        "pending_files": int(max(0, pending)),
    }


@st.cache_data(show_spinner=False)
def load_json_artifact(path_str: str, mtime: float):
    path = Path(path_str)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


def infer_citation_from_filename(name: str):
    if not name:
        return "", "", ""
    stem = Path(name).stem
    cleaned = re.sub(r"[_]+|[-]+", " ", stem)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    year_match = re.search(r"(19|20)\d{2}", cleaned)
    year = year_match.group(0) if year_match else ""
    author = ""
    title = cleaned
    if year_match:
        before, _, after = cleaned.partition(year)
        before = before.strip(" -_()")
        after = after.strip(" -_()")
        if before:
            author = before.split(",")[0].strip()
        if after:
            title = after
    return title, author, year


def infer_author_year_from_text(text: str):
    raw = clean_text(text or "")
    if not raw:
        return "", ""
    year_match = re.search(r"\b(19|20)\d{2}\b", raw)
    year = year_match.group(0) if year_match else ""
    author = ""
    # Common scholarly pattern: "Lastname, F. (2023)" or "Lastname et al., 2021"
    m = re.search(r"([A-Z][A-Za-z'`\-]+(?:\s+et al\.)?)\s*,?\s*\(?((19|20)\d{2})\)?", raw)
    if m:
        author = clean_text(m.group(1))
        if not year:
            year = clean_text(m.group(2))
    if not author:
        m2 = re.search(r"by\s+([A-Z][A-Za-z'`\-]+(?:\s+[A-Z][A-Za-z'`\-]+)?)", raw)
        if m2:
            author = clean_text(m2.group(1))
    return author, year


def infer_tag_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    m = re.search(r"#([A-Za-z0-9_-]+)", stem)
    if m:
        return m.group(1)
    m = re.search(r"\[([A-Za-z0-9_-]+)\]", stem)
    if m:
        return m.group(1)
    return ""


def derive_title_from_source(source, info: dict | None = None):
    src = source or {}
    info = info or {}
    title = clean_text(
        src.get("title")
        or src.get("display_name")
        or src.get("file_name")
        or info.get("title")
        or ""
    )
    uri = clean_text(src.get("uri") or "")
    rel_path = clean_text(src.get("rel_path") or info.get("rel_path") or "")
    snippet = clean_text(src.get("snippet") or "")

    # Prefer metadata title unless it looks like a system id.
    if not title or is_probably_system_id(title):
        candidate = ""
        if rel_path:
            candidate = Path(rel_path).name
        elif clean_text(src.get("file_name") or ""):
            candidate = clean_text(src.get("file_name") or "")
        elif uri and not uri.startswith("http"):
            candidate = Path(uri).name
        candidate = clean_text(candidate)
        if candidate:
            guessed_title, _, _ = infer_citation_from_filename(candidate)
            if guessed_title:
                title = guessed_title
            else:
                title = candidate
    if (not title or is_probably_system_id(title)) and snippet:
        first = snippet.split("\n", 1)[0].strip()
        if len(first) >= 12 and not is_probably_system_id(first):
            title = first[:120]

    if not title:
        fallback = clean_text(uri or "")
        if fallback and not is_probably_system_id(fallback):
            title = fallback
        else:
            title = "Source excerpt"
    if is_probably_system_id(title):
        title = "Source excerpt"
    return title


def should_hide_source_by_default(source, info: dict | None = None) -> bool:
    src = source or {}
    info = info or {}
    has_human_metadata = bool(
        clean_text(src.get("author") or info.get("author") or "")
        or clean_text(src.get("year") or info.get("year") or "")
        or clean_text(src.get("rel_path") or info.get("rel_path") or "")
    )
    if has_human_metadata:
        return False
    raw_title = clean_text(src.get("title") or "")
    if not raw_title or not is_probably_system_id(raw_title):
        return False
    recovered_title = derive_title_from_source(src, info)
    return is_probably_system_id(recovered_title)


def enrich_source(source, index_map):
    title = clean_text(source.get("title") or "")
    uri = (source.get("uri") or "").strip()
    direct_author = clean_text(source.get("author") or "")
    direct_year = clean_text(source.get("year") or "")
    direct_rel = clean_text(source.get("rel_path") or "")
    direct_doc_type = clean_text(source.get("doc_type") or "")
    direct_version = clean_text(source.get("version_stage") or "")
    direct_citation_source = clean_text(source.get("citation_source") or "")
    direct_web_domain = clean_text(source.get("web_domain") or "")
    direct_web_trust_tier = clean_text(source.get("web_trust_tier") or "")
    direct_accessed_at = clean_text(source.get("accessed_at") or "")
    if source.get("snippet") and (not direct_author or not direct_year):
        inferred_author, inferred_year = infer_author_year_from_text(source.get("snippet") or "")
        if not direct_author and inferred_author:
            direct_author = inferred_author
        if not direct_year and inferred_year:
            direct_year = inferred_year
    candidates = []
    if title:
        candidates.append(title)
    if direct_rel:
        candidates.append(direct_rel)
    if uri:
        candidates.append(uri)
        try:
            candidates.append(Path(uri).name)
        except Exception:
            pass

    row = None
    for c in candidates:
        key = c.lower()
        if key in index_map:
            row = index_map[key]
            break

    if row:
        return {
            "title": row.get("title_guess") or title,
            "author": row.get("author_guess") or direct_author or "",
            "year": row.get("year_guess") or direct_year or "",
            "doc_type": row.get("doc_type") or direct_doc_type or "",
            "version_stage": row.get("version_stage") or direct_version or "",
            "rel_path": row.get("rel_path") or direct_rel or "",
            "citation_source": row.get("citation_source") or direct_citation_source or "filename",
            "web_domain": direct_web_domain,
            "web_trust_tier": direct_web_trust_tier,
            "accessed_at": direct_accessed_at,
            "inferred": True,
        }

    t, a, y = infer_citation_from_filename(title or uri)
    return {
        "title": t or title,
        "author": direct_author or a,
        "year": direct_year or y,
        "doc_type": direct_doc_type,
        "version_stage": direct_version,
        "rel_path": direct_rel,
        "citation_source": direct_citation_source or "filename",
        "web_domain": direct_web_domain,
        "web_trust_tier": direct_web_trust_tier,
        "accessed_at": direct_accessed_at,
        "inferred": True,
    }


def citation_key_author_year(author: str, year: str):
    a = clean_text(author or "")
    y = clean_text(year or "")
    surname = ""
    if a:
        surname = re.split(r"[\s,;]+", a)[0]
    if surname and y:
        return f"{surname}{y}"
    if surname:
        return surname
    if y:
        return y
    return "Unknown"


def normalized_source_citation(source, index_map=None):
    info = {}
    if isinstance(index_map, dict):
        info = enrich_source(source or {}, index_map)
    src = source or {}
    title = derive_title_from_source(src, info)
    author = clean_text(src.get("author") or info.get("author") or "")
    year = clean_text(src.get("year") or info.get("year") or "")
    section = clean_text(src.get("section_heading") or "")
    page = src.get("page")
    bits = []
    if author and year:
        bits.append(f"{author} ({year})")
    elif author:
        bits.append(author)
    elif year:
        bits.append(year)
    bits.append(title)
    try:
        if page:
            bits.append(f"p. {int(page)}")
    except Exception:
        pass
    if section:
        bits.append(f"§{section}")
    return ", ".join([b for b in bits if b])


def highlight_snippet(snippet: str, query: str):
    if not snippet:
        return ""
    safe = html.escape(snippet)
    if not query:
        return safe
    words = re.findall(r"[A-Za-z0-9]{4,}", query.lower())
    unique = []
    for w in words:
        if w not in unique:
            unique.append(w)
    for w in unique[:6]:
        safe = re.sub(
            rf"(?i)\b{re.escape(w)}\b",
            r"<mark>\g<0></mark>",
            safe,
        )
    return safe


def copy_button(text: str):
    button_id = f"copy_{uuid.uuid4().hex}"
    payload = base64.b64encode((text or "").encode("utf-8")).decode("ascii")
    html_btn = f"""
<button id="{button_id}" class="copy-btn">Copy citation</button>
<script>
(() => {{
  const btn = document.getElementById("{button_id}");
  if (!btn) return;
  const encoded = "{payload}";
  btn.addEventListener("click", async () => {{
    try {{
      const decoded = atob(encoded);
      await navigator.clipboard.writeText(decoded);
    }} catch (_) {{}}
  }});
}})();
</script>
"""
    components.html(html_btn, height=32)


def is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def watcher_cmdline(pid: int) -> str:
    if pid <= 0:
        return ""
    try:
        proc = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True,
            text=True,
            check=False,
        )
        return (proc.stdout or "").strip()
    except Exception:
        return ""


def is_edith_watcher_pid(pid: int) -> bool:
    cmd = watcher_cmdline(pid).lower()
    if not cmd:
        return False
    return "watch_files.py" in cmd and "python" in cmd


def start_watcher(retrieval_backend="google"):
    script = Path(__file__).parent / "watch_files.py"
    if not script.exists():
        return False, "watch_files.py not found."
    if WATCH_PID_PATH.exists():
        try:
            pid = int(WATCH_PID_PATH.read_text().strip())
            if is_pid_running(pid):
                if is_edith_watcher_pid(pid):
                    return True, "Watcher already running."
                WATCH_PID_PATH.unlink(missing_ok=True)
        except Exception:
            pass
    log = WATCH_LOG_PATH.open("a")
    env = os.environ.copy()
    env.setdefault("EDITH_APP_DATA_DIR", str(APP_STATE_DIR))
    env["EDITH_RETRIEVAL_BACKEND"] = retrieval_backend
    proc = subprocess.Popen(
        [sys.executable, str(script)],
        cwd=str(Path(__file__).parent),
        stdout=log,
        stderr=log,
        start_new_session=True,
        env=env,
    )
    WATCH_PID_PATH.write_text(str(proc.pid))
    return True, f"Watcher started (pid {proc.pid})."


def stop_watcher():
    if not WATCH_PID_PATH.exists():
        return False, "Watcher not running."
    try:
        pid = int(WATCH_PID_PATH.read_text().strip())
    except Exception:
        WATCH_PID_PATH.unlink(missing_ok=True)
        return False, "Invalid watcher pid."
    if not is_pid_running(pid):
        WATCH_PID_PATH.unlink(missing_ok=True)
        return False, "Watcher pid was not running."
    if not is_edith_watcher_pid(pid):
        return False, "Refusing to kill: pid does not match Edith watcher."
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError:
        pass
    WATCH_PID_PATH.unlink(missing_ok=True)
    return True, "Watcher stopped."


def build_file_search_tool(project: str, tag: str):
    if not STORE_ID:
        return None
    kwargs = {"file_search_store_names": [STORE_ID]}
    metadata = {}
    if project and project != "All":
        metadata["project"] = project
    if tag:
        metadata["tag"] = tag
    if metadata:
        kwargs["metadata_filter"] = metadata
    return types.Tool(file_search=types.FileSearch(**kwargs))


def build_file_search_tool_unfiltered():
    if not STORE_ID:
        return None
    return types.Tool(file_search=types.FileSearch(file_search_store_names=[STORE_ID]))


def build_web_search_tool():
    try:
        return types.Tool(google_search=types.GoogleSearch())
    except Exception:
        pass
    try:
        return types.Tool(google_search_retrieval=types.GoogleSearchRetrieval())
    except Exception:
        return None


def build_tools(source_mode: str, project: str, tag: str, retrieval_backend: str = "google"):
    tools = []
    if (
        retrieval_backend == "google"
        and source_mode in ("Files only", "Files + Web")
        and tool_allowed("file_search")
    ):
        t = build_file_search_tool(project, tag)
        if t:
            tools.append(t)
    web_capability = tal_web_capability_status().get("active")
    if (ALLOW_WEB_TOOLS or web_capability) and source_mode in ("Web only", "Files + Web") and (
        tool_allowed("google_search") or tool_allowed("google_search_retrieval")
    ):
        web_tool = build_web_search_tool()
        if web_tool:
            tools.append(web_tool)
    return tools


def build_fallback_tools(source_mode: str, retrieval_backend: str = "google"):
    tools = []
    if (
        retrieval_backend == "google"
        and source_mode in ("Files only", "Files + Web")
        and tool_allowed("file_search")
    ):
        t = build_file_search_tool_unfiltered()
        if t:
            tools.append(t)
    web_capability = tal_web_capability_status().get("active")
    if (ALLOW_WEB_TOOLS or web_capability) and source_mode in ("Web only", "Files + Web") and (
        tool_allowed("google_search") or tool_allowed("google_search_retrieval")
    ):
        web_tool = build_web_search_tool()
        if web_tool:
            tools.append(web_tool)
    return tools


def has_file_sources(sources):
    return any((s.get("source_type") == "file") for s in sources)


def has_web_sources(sources):
    return any((s.get("source_type") == "web") for s in sources)


def source_gate_message(source_mode: str, sources, hybrid_policy: str = "require_files", require_citations: bool = True):
    if not require_citations:
        return None
    if source_mode == "Files only":
        return None if has_file_sources(sources) else "Not found in sources."
    if source_mode == "Web only":
        return None if has_web_sources(sources) else "Not found in sources."
    if hybrid_policy == "require_files":
        return None if has_file_sources(sources) else "Not found in sources."
    return None if sources else "Not found in sources."


def hybrid_policy_caption(hybrid_policy: str, sources):
    if hybrid_policy == "prefer_files" and not has_file_sources(sources):
        return "Hybrid policy note: no file evidence was retrieved, answer used web evidence."
    return ""


def trust_badge_text(source_mode: str, strict_citations: bool):
    web_enabled = source_mode in ("Web only", "Files + Web")
    if strict_citations and web_enabled:
        return "Grounded | Web enabled"
    if strict_citations:
        return "Grounded | Citations required"
    if web_enabled:
        return "Flexible | Web enabled"
    return "Flexible | General knowledge allowed"


def run_live_health_checks(active_model: str, retrieval_backend: str):
    checks = []
    if not client:
        checks.append({"name": "API client", "ok": False, "detail": "GOOGLE_API_KEY missing or invalid."})
        return checks

    vault_root = (os.getenv("EDITH_VAULT_EXPORT_DIR") or "").strip()
    if vault_root:
        p = Path(vault_root).expanduser()
        checks.append(
            {
                "name": "Vault export dir",
                "ok": p.exists() and p.is_dir(),
                "detail": f"{p}" if p.exists() else f"Missing: {p}",
            }
        )

    if retrieval_backend == "google":
        if not STORE_ID:
            checks.append({"name": "Store connectivity", "ok": False, "detail": "EDITH_STORE_ID not configured."})
        else:
            try:
                client.file_search_stores.get(name=STORE_ID)
                checks.append({"name": "Store connectivity", "ok": True, "detail": f"Connected to {STORE_ID}."})
            except Exception as e:
                checks.append({"name": "Store connectivity", "ok": False, "detail": f"{type(e).__name__}: {e}"})

    if active_model:
        try:
            resp = client.models.generate_content(
                model=active_model,
                contents="Respond with OK.",
                config=types.GenerateContentConfig(temperature=0.0),
            )
            text = (resp.text or "").strip()
            checks.append(
                {
                    "name": "Model reachability",
                    "ok": bool(text),
                    "detail": f"{active_model} responded." if text else f"{active_model} returned empty text.",
                }
            )
        except Exception as e:
            checks.append({"name": "Model reachability", "ok": False, "detail": f"{type(e).__name__}: {e}"})
    else:
        checks.append({"name": "Model reachability", "ok": False, "detail": "No active model selected."})

    oauth_state = resolve_oauth_identity()
    checks.append(
        {
            "name": "OAuth identity",
            "ok": bool(oauth_state.get("ok")),
            "detail": (
                (f"Header {OAUTH_HEADER} accepted." if oauth_state.get("required") else "Not required.")
                if oauth_state.get("ok")
                else (oauth_state.get("reason") or "OAuth check failed.")
            ),
        }
    )
    role_now = normalize_role_name(st.session_state.get("user_role", RBAC_DEFAULT_ROLE))
    checks.append(
        {
            "name": "RBAC role",
            "ok": True,
            "detail": f"Current role: {role_now}",
        }
    )
    checks.append(
        {
            "name": "Web transport policy",
            "ok": bool(REQUIRE_HTTPS_WEB_SOURCES),
            "detail": "HTTPS-only web sources enabled." if REQUIRE_HTTPS_WEB_SOURCES else "HTTPS-only web sources disabled.",
        }
    )
    checks.append(
        {
            "name": "Rate limiting",
            "ok": bool(RATE_LIMIT_ENABLED),
            "detail": (
                f"chat={RATE_LIMIT_CHAT_MAX}/{RATE_LIMIT_CHAT_WINDOW_SECONDS}s, "
                f"admin={RATE_LIMIT_MUTATION_MAX}/{RATE_LIMIT_MUTATION_WINDOW_SECONDS}s"
            )
            if RATE_LIMIT_ENABLED
            else "Rate limiting disabled.",
        }
    )

    return checks


def watcher_status():
    if not WATCH_PID_PATH.exists():
        return {"running": False, "detail": "Watcher pid file not found."}
    try:
        pid = int(WATCH_PID_PATH.read_text().strip())
    except Exception:
        return {"running": False, "detail": "Watcher pid file invalid."}
    if not is_pid_running(pid):
        return {"running": False, "detail": "Watcher process is not running."}
    if not is_edith_watcher_pid(pid):
        return {"running": False, "detail": "Watcher pid does not match Edith watcher process."}
    return {"running": True, "detail": f"Watcher running (pid {pid})."}


def run_health_doctor(active_model: str, retrieval_backend: str, source_mode: str, index_count: int):
    report = {"checks": [], "actions": []}
    missing = setup_required()
    report["checks"].append(
        {
            "name": "Setup completeness",
            "ok": len(missing) == 0,
            "detail": "Complete." if not missing else ("Missing: " + ", ".join(missing)),
        }
    )
    report["checks"].extend(run_live_health_checks(active_model=active_model, retrieval_backend=retrieval_backend))
    wstat = watcher_status()
    report["checks"].append({"name": "Auto-index watcher", "ok": bool(wstat.get("running")), "detail": wstat.get("detail", "")})
    needs_file_index = source_mode in ("Files only", "Files + Web")
    report["checks"].append(
        {
            "name": "Indexed files",
            "ok": (index_count > 0) or (not needs_file_index),
            "detail": f"{index_count} indexed documents." if needs_file_index else "Not required for current mode.",
        }
    )

    reindex_script = Path(__file__).parent / ("index_files.py" if retrieval_backend == "google" else "chroma_index.py")
    watcher_script = Path(__file__).parent / "watch_files.py"

    if needs_file_index and index_count == 0 and reindex_script.exists():
        code, output = run_reindex(retrieval_backend=retrieval_backend)
        save_index_status(code, output)
        report["actions"].append(
            {
                "name": "Reindex files",
                "ok": code == 0,
                "detail": "Reindex completed." if code == 0 else (output[:500] or "Reindex failed."),
            }
        )
    elif needs_file_index and index_count == 0:
        report["actions"].append(
            {
                "name": "Reindex files",
                "ok": False,
                "detail": "Indexer script not found.",
            }
        )

    if watcher_script.exists():
        wstat_after = watcher_status()
        if not wstat_after.get("running"):
            ok, msg = start_watcher(retrieval_backend=retrieval_backend)
            report["actions"].append({"name": "Start watcher", "ok": bool(ok), "detail": msg})
    else:
        report["actions"].append(
            {"name": "Start watcher", "ok": False, "detail": "Watcher script not found."}
        )
    return report


def render_not_found_help(source_mode: str, strict_citations: bool, query: str = "", index_rows=None):
    st.info("Not found in sources.")
    st.caption("Try one of these next steps:")
    role_now = normalize_role_name(st.session_state.get("user_role", RBAC_DEFAULT_ROLE))
    web_rbac_allowed = role_has_permission(role_now, "web.search")
    wants_fresh = query_requests_fresh_web(query)
    if wants_fresh and source_mode == "Files only" and ALLOW_WEB_TOOLS:
        st.caption("This looks like a current-events query. Files + Web mode may be needed.")
    hint_cols = st.columns(4)
    if hint_cols[0].button("Search library", key="nf_search_btn"):
        st.caption("Open the Library tab and search for likely source files.")
    if hint_cols[1].button("Relax strictness", key="nf_relax_btn"):
        st.session_state.strict_citations = False
    reindex_available = (Path(__file__).parent / ("index_files.py" if st.session_state.get("retrieval_backend", RETRIEVAL_BACKEND_DEFAULT) == "google" else "chroma_index.py")).exists()
    if hint_cols[2].button("Reindex now", key="nf_reindex_btn", disabled=not reindex_available):
        code, output = run_reindex(retrieval_backend=st.session_state.get("retrieval_backend", RETRIEVAL_BACKEND_DEFAULT))
        st.session_state.reindex_output = output
        st.session_state.reindex_code = code
        save_index_status(code, output)
    if source_mode == "Files only" and ALLOW_WEB_TOOLS and web_rbac_allowed:
        web_btn_label = "Try Files + Web now" if wants_fresh else "Enable web mode"
        if hint_cols[3].button(web_btn_label, key="nf_web_btn"):
            st.session_state.source_mode = "Files + Web"
    elif source_mode == "Files only" and (not ALLOW_WEB_TOOLS):
        if not web_rbac_allowed:
            hint_cols[3].caption("Web blocked by role policy")
        elif hint_cols[3].button("Allow web once", key="nf_allow_web_once_btn"):
            tal_grant_web_once()
            st.session_state.source_mode = "Files + Web"
            st.session_state.queued_prompt = query
            st.rerun()
        elif hint_cols[3].button("Turn on web access", key="nf_enable_web_access_btn"):
            if DESKTOP_MODE:
                st.session_state.desktop_setup_open_request = True
            else:
                persist_runtime_policy_settings(allow_web_tools=True)
            st.rerun()
    elif source_mode == "Files + Web":
        hint_cols[3].caption("Web already enabled")
    else:
        hint_cols[3].caption("Add files or switch mode")
    if source_mode == "Files only" and (not ALLOW_WEB_TOOLS) and web_rbac_allowed:
        consent_cols = st.columns(3)
        if consent_cols[0].button("Allow web this chat", key="nf_allow_web_chat_btn"):
            tal_grant_web_for_chat()
            st.session_state.source_mode = "Files + Web"
            st.session_state.queued_prompt = query
            st.rerun()
        if consent_cols[1].button("Keep files only", key="nf_keep_files_only_btn"):
            st.session_state.source_mode = "Files only"
        cap = tal_web_capability_status()
        if cap.get("active"):
            consent_cols[2].caption(f"Capability active: once={cap.get('once', 0)}, chat={cap.get('chat', 0)}")
        else:
            consent_cols[2].caption("No active web capability token")
    if strict_citations:
        st.caption("Strict mode is on. Edith refuses answers when evidence is weak.")
    action_cols = st.columns(3)
    if action_cols[0].button("Broaden search", key="nf_broaden_search_inline_btn"):
        st.session_state.chroma_top_k = 20
        st.session_state.chroma_rerank_top_n = max(24, int(st.session_state.get("chroma_rerank_top_n", 24)))
        st.session_state.queued_prompt = query
        st.rerun()
    if action_cols[1].button("Search web", key="nf_search_web_inline_btn"):
        if not web_rbac_allowed:
            st.caption("Web is blocked by role policy.")
        elif ALLOW_WEB_TOOLS or tal_web_capability_status().get("active"):
            st.session_state.source_mode = "Files + Web"
            st.session_state.queued_prompt = query
            st.rerun()
        else:
            st.caption("Web access is disabled in settings.")
    if action_cols[2].button("Show near-miss docs", key="nf_nearmiss_btn"):
        st.session_state.show_not_found_debug = True
    intent_guess = classify_query_intent(query)
    if intent_guess in {"overview", "general", "compare"}:
        if st.button("Broaden retrieval (20 docs)", key="nf_broaden_btn"):
            st.session_state.chroma_top_k = 20
            st.session_state.chroma_rerank_top_n = max(24, int(st.session_state.get("chroma_rerank_top_n", 24)))
            st.session_state.queued_prompt = query
            st.rerun()
    if st.button("Why didn't it find it?", key="nf_debug_button"):
        st.session_state.show_not_found_debug = not bool(st.session_state.get("show_not_found_debug", False))
    if st.session_state.get("show_not_found_debug"):
        meta = st.session_state.get("last_response_meta", {}) or {}
        with st.expander("Not-found diagnostics", expanded=True):
            stores = []
            if st.session_state.get("retrieval_backend", RETRIEVAL_BACKEND_DEFAULT) == "google":
                if STORE_ID:
                    stores.append(STORE_ID)
            else:
                stores.append(f"chroma::{CHROMA_COLLECTION}")
            st.caption("Searched stores: " + (", ".join(stores) if stores else "none"))
            qv = meta.get("query_variants") or []
            if qv:
                st.caption("Top queries")
                for qline in qv[:5]:
                    st.markdown(f"- `{qline}`")
            else:
                st.caption("Top queries: none captured")
            gate_reason = meta.get("gate_message") or ((meta.get("support_audit") or {}).get("reason"))
            if gate_reason:
                st.caption(f"Gate reason: {gate_reason}")

    keys = query_keywords(query)
    if keys:
        st.caption("Suggested keywords: " + ", ".join(keys))
        key_cols = st.columns(min(3, len(keys)))
        for i, kw in enumerate(keys[:3]):
            if key_cols[i].button(f"Retry: {kw}", key=f"nf_kw_retry_{i}"):
                st.session_state.queued_prompt = f"Find '{kw}' in my files and cite sources."
                st.rerun()
    near = nearest_library_matches(query, index_rows or [], limit=4)
    if near:
        st.caption("Closest indexed documents:")
        for row in near:
            title = html.escape(str(row.get("title_guess") or Path(row.get("rel_path", "")).name or "Untitled"))
            detail = " | ".join(
                [
                    html.escape(str(row.get("author_guess") or "author unknown")),
                    html.escape(str(row.get("year_guess") or "year unknown")),
                    html.escape(str(row.get("project") or "project unknown")),
                ]
            )
            rel_safe = html.escape(str(row.get("rel_path", "")))
            st.markdown(f"- **{title}** — `{rel_safe}`")
            st.caption(f"Match score: {row.get('_match_score', 0.0):.2f} | {detail}")
        primary = near[0]
        anchor = primary.get("title_guess") or Path(primary.get("rel_path", "")).name or ""
        if anchor and st.button("Ask about closest document", key="nf_closest_doc_btn"):
            st.session_state.queued_prompt = f'Summarize "{anchor}" from my files with citations.'
            st.rerun()


def render_thin_evidence_help(query: str, run_id: str, source_count: int):
    if source_count > 1:
        return
    st.warning("I only found one strongly relevant source for this broad question.")
    st.caption("You can broaden retrieval before finalizing your interpretation.")
    cols = st.columns(3)
    if cols[0].button("Broaden search (20 sources)", key=f"thin_broaden_{run_id}"):
        st.session_state.chroma_top_k = 20
        st.session_state.chroma_rerank_top_n = max(24, int(st.session_state.get("chroma_rerank_top_n", 24)))
        st.session_state.queued_prompt = query
        st.rerun()
    if cols[1].button("Answer from this source", key=f"thin_keep_{run_id}"):
        st.caption("Proceeding with the currently retrieved evidence.")
    if cols[2].button("Show source detail", key=f"thin_show_{run_id}"):
        st.session_state.audit_mode = True
        st.rerun()


def chat_file_path(chat_id: str) -> Path:
    suffix = ".json.enc" if CHAT_CIPHER else ".json"
    return CHAT_DIR / f"{chat_id}{suffix}"


def chat_candidate_paths(chat_id: str):
    return [
        CHAT_DIR / f"{chat_id}.json.enc",
        CHAT_DIR / f"{chat_id}.json",
    ]


def new_chat_state(title: str = "New chat") -> dict:
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "id": datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6],
        "title": title,
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }


def list_saved_chats():
    chat_map = {}
    for p in list(CHAT_DIR.glob("*.json")) + list(CHAT_DIR.glob("*.json.enc")):
        try:
            data = load_chat_from_path(p)
            if not data:
                continue
            cid = data.get("id", p.name.replace(".json.enc", "").replace(".json", ""))
            item = {
                "id": cid,
                "title": data.get("title", p.stem),
                "updated_at": data.get("updated_at", ""),
                "path": p,
            }
            old = chat_map.get(cid)
            if not old or item["updated_at"] >= old.get("updated_at", ""):
                chat_map[cid] = item
        except Exception:
            continue
    chats = list(chat_map.values())
    chats.sort(key=lambda c: c.get("updated_at", ""), reverse=True)
    return chats


def load_chat(chat_id: str):
    for p in chat_candidate_paths(chat_id):
        if not p.exists():
            continue
        data = load_chat_from_path(p)
        if data:
            return data
    return None


def load_chat_from_path(path: Path):
    try:
        raw = path.read_bytes()
    except Exception:
        return None
    if path.suffix == ".enc":
        try:
            payload = decrypt_chat_bytes(raw)
            return json.loads(payload.decode("utf-8"))
        except (InvalidToken, ValueError, UnicodeDecodeError):
            return None
    try:
        return json.loads(raw.decode("utf-8"))
    except Exception:
        return None


def save_chat(chat: dict):
    chat["updated_at"] = datetime.now().isoformat(timespec="seconds")
    p = chat_file_path(chat["id"])
    payload = json.dumps(chat, indent=2).encode("utf-8")
    data = encrypt_chat_bytes(payload)
    p.write_bytes(data)
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass


def append_notebook_line(entry: dict):
    RESEARCH_NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False)
    with RESEARCH_NOTEBOOK_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")
    try:
        os.chmod(RESEARCH_NOTEBOOK_PATH, 0o600)
    except OSError:
        pass


def load_notebook_entries(limit: int = 400):
    if not RESEARCH_NOTEBOOK_PATH.exists():
        return []
    rows = []
    try:
        with RESEARCH_NOTEBOOK_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    obj = json.loads(raw)
                    if isinstance(obj, dict):
                        rows.append(obj)
                except Exception:
                    continue
    except Exception:
        return []
    rows.sort(key=lambda x: clean_text(x.get("created_at") or ""), reverse=True)
    if limit > 0:
        return rows[: int(limit)]
    return rows


def write_notebook_markdown_entry(entry: dict):
    if not DATA_ROOT:
        return ""
    root = Path(DATA_ROOT).expanduser()
    if not root.exists() or not root.is_dir():
        return ""
    project = clean_text(entry.get("project") or "general").strip() or "general"
    project_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", project).strip("_") or "general"
    notes_dir = root / "edith_notes" / project_slug
    notes_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    kind = clean_text(entry.get("kind") or "note").strip().lower() or "note"
    safe_kind = re.sub(r"[^A-Za-z0-9._-]+", "_", kind).strip("_") or "note"
    out_path = notes_dir / f"{ts}_{safe_kind}.md"
    lines = [
        f"# {entry.get('title') or 'Research note'}",
        "",
        f"- Created: {entry.get('created_at') or now_iso()}",
        f"- Project: {project}",
        f"- Kind: {kind}",
        f"- Run ID: {clean_text(entry.get('run_id') or '')}",
        "",
        "## Note",
        clean_text(entry.get("text") or ""),
        "",
    ]
    sources = entry.get("sources") or []
    if sources:
        lines.append("## Sources")
        for s in sources[:20]:
            if not isinstance(s, dict):
                continue
            label = clean_text(s.get("label") or s.get("title") or s.get("uri") or "source")
            uri = clean_text(s.get("uri") or "")
            if uri:
                lines.append(f"- {label} ({uri})")
            else:
                lines.append(f"- {label}")
        lines.append("")
    out_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    return str(out_path)


def save_notebook_entry(kind: str, text: str, query: str = "", sources=None, project: str = "All", run_id: str = ""):
    body = clean_text(text or "").strip()
    if not body:
        return {"ok": False, "error": "Note text is empty."}
    kind_name = clean_text(kind or "note").strip().lower() or "note"
    src_rows = []
    for s in (sources or [])[:20]:
        if not isinstance(s, dict):
            continue
        src_rows.append(
            {
                "label": clean_text(s.get("title") or s.get("uri") or "source"),
                "title": clean_text(s.get("title") or ""),
                "uri": clean_text(s.get("uri") or ""),
                "page": s.get("page"),
                "chunk": s.get("chunk"),
            }
        )
    entry = {
        "id": f"note_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}",
        "created_at": now_iso(),
        "kind": kind_name,
        "title": f"{kind_name.title()} note",
        "project": clean_text(project or "All") or "All",
        "query": clean_text(query or ""),
        "run_id": clean_text(run_id or ""),
        "text": body,
        "sources": src_rows,
    }
    try:
        append_notebook_line(entry)
        note_path = write_notebook_markdown_entry(entry)
        return {"ok": True, "entry": entry, "path": note_path}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def notebook_entries_to_markdown(entries):
    lines = ["# Edith Research Notebook", ""]
    for row in entries or []:
        if not isinstance(row, dict):
            continue
        lines.append(f"## {clean_text(row.get('title') or 'Note')}")
        lines.append(f"- Created: {clean_text(row.get('created_at') or '')}")
        lines.append(f"- Project: {clean_text(row.get('project') or '')}")
        lines.append(f"- Kind: {clean_text(row.get('kind') or '')}")
        q = clean_text(row.get("query") or "")
        if q:
            lines.append(f"- Query: {q}")
        lines.append("")
        lines.append(clean_text(row.get("text") or ""))
        lines.append("")
        srcs = row.get("sources") or []
        if srcs:
            lines.append("Sources:")
            for s in srcs[:20]:
                if not isinstance(s, dict):
                    continue
                label = clean_text(s.get("label") or s.get("title") or "source")
                uri = clean_text(s.get("uri") or "")
                if uri:
                    lines.append(f"- {label} ({uri})")
                else:
                    lines.append(f"- {label}")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def apply_chat_retention_policy(retention_days: int):
    if retention_days <= 0:
        return 0
    removed = 0
    cutoff = datetime.now().timestamp() - (retention_days * 86400)
    for p in list(CHAT_DIR.glob("*.json")) + list(CHAT_DIR.glob("*.json.enc")):
        data = load_chat_from_path(p)
        if not data:
            continue
        updated = data.get("updated_at")
        if not updated:
            continue
        try:
            ts = datetime.fromisoformat(updated).timestamp()
        except ValueError:
            continue
        if ts < cutoff:
            try:
                p.unlink()
                removed += 1
            except OSError:
                pass
    return removed


def delete_chat_history():
    removed = 0
    for p in list(CHAT_DIR.glob("*.json")) + list(CHAT_DIR.glob("*.json.enc")):
        try:
            p.unlink()
            removed += 1
        except OSError:
            pass
    return removed


def delete_local_index_data():
    removed = {"files": 0, "dirs": 0}
    for p in [
        INDEX_REPORT,
        INDEX_STATUS_PATH,
        INDEX_DB_PATH,
        INDEX_HEALTH_REPORT_PATH,
        BIBLIOGRAPHY_DB_PATH,
        GLOSSARY_GRAPH_PATH,
        CITATION_GRAPH_PATH,
        CHAPTER_ANCHORS_PATH,
        CLAIM_INVENTORY_PATH,
        EXPERIMENT_LEDGER_PATH,
        ENTITY_TIMELINE_PATH,
        FEEDBACK_DB_PATH,
    ]:
        if p.exists():
            try:
                p.unlink()
                removed["files"] += 1
            except OSError:
                pass
    chroma_path = Path(CHROMA_DIR).expanduser()
    if chroma_path.exists() and chroma_path.is_dir():
        safe_to_delete = False
        try:
            resolved = chroma_path.resolve()
            app_root = APP_STATE_DIR.resolve()
            home = Path.home().resolve()
            if resolved not in (Path("/"), home) and len(resolved.parts) >= 4:
                if resolved == app_root or app_root in resolved.parents:
                    safe_to_delete = True
                elif "chroma" in resolved.name.lower():
                    safe_to_delete = True
        except Exception:
            safe_to_delete = False
        if safe_to_delete:
            try:
                shutil.rmtree(chroma_path)
                removed["dirs"] += 1
            except OSError:
                pass
        else:
            removed["skipped_unsafe_dir"] = str(chroma_path)
    return removed


def reset_local_app_state():
    preserved = {
        ".env",
        "config.json",
        "secrets.json",
    }
    removed = {"files": 0, "dirs": 0, "errors": 0}
    if not APP_STATE_DIR.exists():
        return removed
    for child in APP_STATE_DIR.iterdir():
        if child.name in preserved:
            continue
        try:
            if child.is_dir():
                shutil.rmtree(child)
                removed["dirs"] += 1
            else:
                child.unlink()
                removed["files"] += 1
        except OSError:
            removed["errors"] += 1
    return removed


def redacted_env_text(path: Path):
    try:
        raw = path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""
    secret_markers = {
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "EDITH_APP_PASSWORD",
        "EDITH_APP_PASSWORD_HASH",
        "EDITH_CHAT_ENCRYPTION_KEY",
        "NOTION_KEY",
    }
    out_lines = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            out_lines.append(line)
            continue
        key, _sep, value = line.partition("=")
        key_name = key.strip()
        if key_name in secret_markers or key_name.endswith("_KEY") or key_name.endswith("_TOKEN"):
            if value.strip():
                out_lines.append(f"{key_name}=[REDACTED]")
            else:
                out_lines.append(f"{key_name}=")
        else:
            out_lines.append(line)
    return "\n".join(out_lines) + ("\n" if raw.endswith("\n") else "")


def export_user_data_zip_bytes():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # Chats
        for p in list(CHAT_DIR.glob("*.json")) + list(CHAT_DIR.glob("*.json.enc")):
            try:
                zf.writestr(f"chat_history/{p.name}", p.read_bytes())
            except Exception:
                continue
        # Core state files
        state_files = [
            INDEX_REPORT,
            INDEX_STATUS_PATH,
            INDEX_DB_PATH,
            INDEX_HEALTH_REPORT_PATH,
            RUN_LEDGER_PATH,
            FEEDBACK_DB_PATH,
            RETRIEVAL_PROFILE_PATH,
            GLOSSARY_GRAPH_PATH,
            CITATION_GRAPH_PATH,
            CHAPTER_ANCHORS_PATH,
            CLAIM_INVENTORY_PATH,
            EXPERIMENT_LEDGER_PATH,
            BIBLIOGRAPHY_DB_PATH,
            ENTITY_TIMELINE_PATH,
            WEB_CACHE_PATH,
            TAL_TOKENS_PATH,
            TAL_AUDIT_PATH,
            RESEARCH_NOTEBOOK_PATH,
        ]
        for p in state_files:
            try:
                if p.exists():
                    zf.writestr(f"state/{p.name}", p.read_bytes())
            except Exception:
                continue
        if SNAPSHOT_DIR.exists():
            for p in SNAPSHOT_DIR.rglob("*"):
                if not p.is_file():
                    continue
                try:
                    rel = p.relative_to(APP_STATE_DIR)
                    zf.writestr(str(rel), p.read_bytes())
                except Exception:
                    continue
        try:
            if ENV_TARGET_PATH.exists():
                redacted = redacted_env_text(ENV_TARGET_PATH)
                if redacted:
                    zf.writestr("state/env.redacted", redacted.encode("utf-8"))
        except Exception:
            pass
    buffer.seek(0)
    return buffer.getvalue()


def auto_title_from_first_user_message(messages):
    for m in messages:
        if m.get("role") == "user":
            text = (m.get("text") or "").strip()
            if not text:
                continue
            if len(text) > 60:
                return text[:57] + "..."
            return text
    return "New chat"


def sanitize_export_text(text: str, redact_sensitive: bool = True):
    out = str(text or "")
    if not redact_sensitive:
        return out
    out = redact_pii_text(out, enabled=True, replacement="[REDACTED]")
    out = re.sub(r"file://\S+", "[FILE_PATH]", out)
    out = re.sub(r"/Users/[^\s)]+", "/Users/[PATH_REDACTED]", out)
    out = re.sub(r"[A-Za-z]:\\\\[^\s)]+", "[PATH_REDACTED]", out)
    return out


def sanitize_export_source_label(source: dict, redact_sensitive: bool = True):
    source = source or {}
    label = source.get("title") or source.get("uri") or "source"
    uri = source.get("uri") or ""
    if redact_sensitive:
        label = sanitize_export_text(label, redact_sensitive=True)
        uri = sanitize_export_text(uri, redact_sensitive=True)
    return label, uri


def chat_to_markdown(chat: dict, redact_sensitive: bool = True):
    lines = [
        f"# {sanitize_export_text(chat.get('title', 'Chat'), redact_sensitive=redact_sensitive)}",
        "",
        f"- Chat ID: {sanitize_export_text(chat.get('id', ''), redact_sensitive=redact_sensitive)}",
        f"- Updated: {sanitize_export_text(chat.get('updated_at', ''), redact_sensitive=redact_sensitive)}",
        "",
    ]
    for msg in chat.get("messages", []):
        role = msg.get("role", "")
        lines.append(f"## {role.title()}")
        lines.append("")
        lines.append(sanitize_export_text(msg.get("text", ""), redact_sensitive=redact_sensitive))
        lines.append("")
        sources = msg.get("sources") or []
        if sources:
            lines.append("Sources:")
            for s in sources:
                label, uri = sanitize_export_source_label(s, redact_sensitive=redact_sensitive)
                if uri:
                    lines.append(f"- {label} ({uri})")
                else:
                    lines.append(f"- {label}")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def chat_to_pdf_bytes(chat: dict, redact_sensitive: bool = True):
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas
    except Exception:
        return None

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    tmp_path = tmp.name
    tmp.close()

    c = canvas.Canvas(tmp_path, pagesize=letter)
    width, height = letter
    y = height - 40
    left = 40

    def draw_line(text):
        nonlocal y
        if y < 50:
            c.showPage()
            y = height - 40
        c.drawString(left, y, text[:140])
        y -= 14

    draw_line(sanitize_export_text(chat.get("title", "Chat"), redact_sensitive=redact_sensitive))
    draw_line(f"Chat ID: {sanitize_export_text(chat.get('id', ''), redact_sensitive=redact_sensitive)}")
    draw_line(f"Updated: {sanitize_export_text(chat.get('updated_at', ''), redact_sensitive=redact_sensitive)}")
    y -= 8

    for msg in chat.get("messages", []):
        draw_line(f"[{msg.get('role', '').upper()}]")
        safe_text = sanitize_export_text(msg.get("text", ""), redact_sensitive=redact_sensitive)
        for raw_line in (safe_text or "").splitlines():
            draw_line(raw_line if raw_line.strip() else " ")
        sources = msg.get("sources") or []
        if sources:
            draw_line("Sources:")
            for s in sources:
                label, _uri = sanitize_export_source_label(s, redact_sensitive=redact_sensitive)
                draw_line(f"- {label}")
        y -= 6

    c.save()
    with open(tmp_path, "rb") as f:
        data = f.read()
    try:
        os.remove(tmp_path)
    except OSError:
        pass
    return data


def filter_library_rows(rows, search="", project="All", author="", year="", tag="", doc_type="All"):
    filtered = []
    q = search.strip().lower()
    a = author.strip().lower()
    y = year.strip()
    t = tag.strip().lower()
    p = project.strip()
    d = doc_type.strip().lower()
    for r in rows:
        if p and p != "All" and (r.get("project") or "") != p:
            continue
        if a and a not in (r.get("author") or "").lower():
            continue
        if y and y != (r.get("year") or ""):
            continue
        if t and t != (r.get("tag") or "").lower():
            continue
        row_doc_type = (r.get("doc_type") or "").strip().lower()
        if d and d != "all" and d != row_doc_type:
            continue
        hay = " ".join(
            [
                r.get("title", ""),
                r.get("author", ""),
                r.get("year", ""),
                r.get("doc_type", ""),
                r.get("version_stage", ""),
                r.get("citation_source", ""),
                r.get("file_name", ""),
                r.get("rel_path", ""),
            ]
        ).lower()
        if q and q not in hay:
            continue
        filtered.append(r)
    return filtered


st.set_page_config(page_title="Edith", page_icon="Edith")

st.markdown(
    """
<style>
:root{
  --bg: #f6f6f8;
  --bg2: #efeff2;
  --card: rgba(255,255,255,0.94);
  --ink: #17181b;
  --muted: #5d6470;
  --accent: #0a84ff;
  --border: rgba(17,24,39,0.12);
  --border-soft: rgba(17,24,39,0.08);
  --shadow-soft: 0 8px 22px rgba(15, 23, 42, 0.06);
  --shadow-card: 0 12px 30px rgba(15, 23, 42, 0.08);
}

html, body, [class*="css"]{
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Helvetica, Arial, sans-serif;
  color: var(--ink);
}

.block-container{
  max-width: 1320px;
  padding-top: 1.25rem;
  padding-bottom: 2rem;
}

.stApp{
  background: radial-gradient(1200px 800px at 8% -12%, #ffffff 0%, var(--bg) 58%, var(--bg2) 100%);
}

h1, h2, h3{
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Helvetica, Arial, sans-serif;
  letter-spacing: -0.02em;
  color: #111827;
}

/* Sidebar */
[data-testid="stSidebar"]{
  background: linear-gradient(180deg, #fafbfc 0%, #f2f3f6 100%);
  border-right: 1px solid var(--border-soft);
}
[data-testid="stSidebar"] h3{
  letter-spacing: -0.01em;
  font-size: 1.03rem;
}

[data-testid="stSidebar"] .stCaption{
  color: var(--muted);
}

/* Buttons */
.stButton > button{
  border-radius: 12px !important;
  border: 1px solid var(--border-soft) !important;
  background: #ffffff !important;
  padding: 0.52rem 0.9rem !important;
  font-weight: 600 !important;
  color: #1f2937 !important;
  transition: all 160ms ease;
  min-height: 2.45rem;
  box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}
.stButton > button:hover{
  transform: translateY(-1px);
  box-shadow: var(--shadow-soft);
}
.stButton > button[kind="primary"]{
  background: linear-gradient(180deg, #1f8bff 0%, #0a84ff 100%) !important;
  color: #ffffff !important;
  border-color: rgba(10,132,255,0.65) !important;
}

/* Inputs */
[data-baseweb="input"] input,
[data-baseweb="textarea"] textarea{
  border-radius: 12px !important;
  border-color: var(--border-soft) !important;
  background: rgba(255,255,255,0.95) !important;
}
[data-baseweb="input"] input:focus,
[data-baseweb="textarea"] textarea:focus{
  border-color: rgba(10,132,255,0.52) !important;
  box-shadow: 0 0 0 4px rgba(10,132,255,0.10) !important;
}

/* Chat bubbles */
[data-testid="stChatMessage"]{
  border-radius: 16px !important;
  padding: 1rem 1.05rem !important;
  border: 1px solid var(--border-soft) !important;
  background: var(--card) !important;
  box-shadow: var(--shadow-card);
  animation: riseFade 220ms ease-out;
}
[data-testid="stChatMessage"][aria-label="Chat message from user"]{
  border-left: 3px solid rgba(10,132,255,0.78) !important;
}

/* Chat input */
[data-testid="stChatInput"] textarea{
  border-radius: 16px !important;
  border: 1px solid var(--border-soft) !important;
  background: rgba(255,255,255,0.95) !important;
  color: var(--ink) !important;
  padding: 0.85rem 1rem !important;
  font-size: 15px !important;
  box-shadow: var(--shadow-soft);
}
[data-testid="stChatInput"] textarea:focus{
  border: 1px solid rgba(10,132,255,0.65) !important;
  box-shadow: 0 0 0 4px rgba(10,132,255,0.15) !important;
}

/* File uploader */
[data-testid='stFileUploader']{
  background-color: rgba(255,255,255,0.88) !important;
  border: 1px dashed rgba(17,24,39,0.20) !important;
  border-radius: 16px !important;
  padding: 12px !important;
}

[data-testid="stExpander"]{
  border: 1px solid var(--border-soft);
  border-radius: 14px;
  background: rgba(255,255,255,0.62);
}

[data-testid="stTabs"] [role="tablist"]{
  gap: 0.45rem;
  border-bottom: 1px solid var(--border-soft);
  padding-bottom: 0.35rem;
}
[data-testid="stTabs"] [role="tab"]{
  border: 1px solid var(--border-soft);
  background: rgba(255,255,255,0.78);
  border-radius: 999px;
  padding: 0.28rem 0.9rem;
  font-weight: 600;
  color: #4b5563;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"]{
  color: #0a57b5;
  border-color: rgba(10,132,255,0.36);
  background: #ffffff;
  box-shadow: 0 2px 10px rgba(10,132,255,0.12);
}

[data-testid="stAlert"]{
  border-radius: 12px;
  border: 1px solid var(--border-soft);
}

mark{
  background: #ffe9a8;
  padding: 0.08em 0.2em;
  border-radius: 4px;
}
.copy-btn{
  border: 1px solid var(--border-soft);
  background: #ffffff;
  padding: 6px 10px;
  border-radius: 10px;
  font-size: 12px;
  cursor: pointer;
}
.copy-btn:hover{
  box-shadow: var(--shadow-soft);
}

.hero{
  padding: 1.1rem 1.25rem;
  border-radius: 16px;
  border: 1px solid var(--border-soft);
  background: linear-gradient(150deg, #ffffff 0%, #f7f8fc 100%);
  box-shadow: var(--shadow-card);
  margin-bottom: 1rem;
}
.hero-title{
  font-family: -apple-system, BlinkMacSystemFont, "SF Pro Display", "Segoe UI", Helvetica, Arial, sans-serif;
  font-size: 2rem;
  font-weight: 600;
}
.hero-sub{
  color: var(--muted);
  margin-top: 0.28rem;
}
.status-pill{
  display: inline-block;
  margin-top: 0.64rem;
  padding: 0.36rem 0.74rem;
  border-radius: 999px;
  border: 1px solid var(--border-soft);
  background: rgba(255,255,255,0.86);
  font-size: 12px;
  font-weight: 500;
}
.mode-pill{
  position: sticky;
  top: 0.35rem;
  z-index: 10;
  margin-bottom: 0.8rem;
  backdrop-filter: blur(4px);
}
.source-truth-card{
  margin: 0.15rem 0 0.8rem 0;
  padding: 0.72rem 0.86rem;
  border-radius: 14px;
  border: 1px solid var(--border-soft);
  background: rgba(255,255,255,0.92);
}
.source-truth-title{
  font-weight: 600;
  margin-bottom: 0.22rem;
}
.source-truth-meta{
  color: #4b5563;
  font-size: 12px;
  line-height: 1.45;
}
.stoplight{
  display: inline-block;
  margin-top: 0.45rem;
  margin-bottom: 0.5rem;
  padding: 0.30rem 0.62rem;
  border-radius: 999px;
  border: 1px solid var(--border);
  font-size: 12px;
  font-weight: 600;
}
.stoplight.green{
  background: rgba(40, 164, 98, 0.12);
  color: #126b42;
}
.stoplight.yellow{
  background: rgba(217, 152, 0, 0.14);
  color: #7a5700;
}
.stoplight.red{
  background: rgba(193, 61, 61, 0.12);
  color: #7f2121;
}
.followup-row{
  margin-top: 0.45rem;
}
.source-card{
  border: 1px solid var(--border-soft);
  border-radius: 14px;
  background: rgba(255,255,255,0.90);
  padding: 0.78rem 0.84rem;
  margin-bottom: 0.6rem;
  box-shadow: 0 1px 3px rgba(15,23,42,0.06);
}
.source-card-title{
  font-weight: 600;
  margin-bottom: 0.2rem;
}
.source-card-meta{
  color: #6b7280;
  font-size: 12px;
}
.source-card-quote{
  margin-top: 0.45rem;
  padding: 0.48rem 0.62rem;
  border-radius: 10px;
  background: rgba(246, 248, 252, 0.95);
  border: 1px solid rgba(15, 23, 42, 0.08);
  font-size: 0.94rem;
  line-height: 1.48;
}
.source-chip-row{
  display: flex;
  flex-wrap: wrap;
  gap: 0.4rem;
  margin: 0.25rem 0 0.6rem 0;
}
.source-chip{
  display: inline-block;
  padding: 0.23rem 0.56rem;
  border-radius: 999px;
  border: 1px solid rgba(15, 23, 42, 0.14);
  background: rgba(255,255,255,0.98);
  font-size: 12px;
  color: #1f2937 !important;
  text-decoration: none !important;
}
.source-chip:hover{
  background: rgba(246, 248, 252, 1.0);
}
.library-card{
  border: 1px solid var(--border-soft);
  border-radius: 14px;
  background: rgba(255,255,255,0.90);
  padding: 0.8rem 0.9rem;
  margin-bottom: 0.6rem;
  box-shadow: 0 1px 3px rgba(15,23,42,0.06);
}

/* ChatGPT-like interface overrides */
.stApp{
  background: #ffffff !important;
}
[data-testid="stSidebar"]{
  background: #f7f7f8 !important;
  border-right: 1px solid #ececf1 !important;
}
.block-container{
  max-width: 960px !important;
  padding-top: 1rem !important;
}
[data-testid="stChatMessage"]{
  border: none !important;
  box-shadow: none !important;
  border-radius: 14px !important;
  background: transparent !important;
  padding: 0.25rem 0.2rem !important;
}
[data-testid="stChatMessage"][aria-label="Chat message from user"]{
  background: #f4f4f4 !important;
  border: 1px solid #ececf1 !important;
  padding: 0.7rem 0.85rem !important;
}
[data-testid="stChatInput"]{
  position: sticky;
  bottom: 0.5rem;
  z-index: 20;
}
[data-testid="stChatInput"] textarea{
  border: 1px solid #d9d9e3 !important;
  border-radius: 26px !important;
  box-shadow: none !important;
  background: #ffffff !important;
}
[data-testid="stTabs"] [role="tab"]{
  border-radius: 10px !important;
  border-color: #ececf1 !important;
  box-shadow: none !important;
}
.hero{
  box-shadow: none !important;
  border: 1px solid #ececf1 !important;
  background: #ffffff !important;
}

@keyframes riseFade{
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""",
    unsafe_allow_html=True,
)

missing_setup_fields = setup_required()
if missing_setup_fields:
    st.markdown(
        """
<div class="hero">
  <div class="hero-title">Edith Setup</div>
  <div class="hero-sub">One-time secure configuration. Values are saved to your local Edith .env file.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.warning("Missing setup fields: " + ", ".join(missing_setup_fields))
    with st.form("first_run_setup", clear_on_submit=False):
        setup_backend = (RETRIEVAL_BACKEND_DEFAULT or "google").strip().lower()
        google_mode = setup_backend == "google"
        api_key_input = st.text_input(
            "Google API key",
            value="",
            type="password",
            help="Used for model and retrieval API calls.",
            placeholder="AIza...",
        )
        password_input = st.text_input(
            "Edith password",
            value="",
            type="password",
            help="Saved as PBKDF2-SHA256 hash in local config.",
        )
        password_confirm = st.text_input(
            "Confirm Edith password",
            value="",
            type="password",
        )
        vault_input = st.text_input(
            "Edith vault id",
            value=VAULT_ID or STORE_MAIN or STORE_ID or "",
            help=(
                "Required in Google retrieval mode. "
                "Example: fileSearchStores/edith-academic-vault-xxxx"
            ),
        )
        store_input = st.text_input(
            "Edith storage id",
            value=STORE_ID or STORE_MAIN or VAULT_ID or "",
            help=(
                "Required in Google retrieval mode. "
                "Store used by EDITH_STORE_ID for File Search."
            ),
        )
        data_root_input = st.text_input(
            "Library folder (optional)",
            value=DATA_ROOT or "",
            help="Used for local Chroma indexing and uploads.",
        )
        submitted = st.form_submit_button("Save and launch")

    st.caption(f"Config path: {ENV_TARGET_PATH}")
    if submitted:
        errors = []
        api_key_value = (api_key_input or "").strip() or (API_KEY or "").strip()
        pass_value = (password_input or "").strip()
        pass_confirm = (password_confirm or "").strip()
        vault_value = normalize_store_id(vault_input)
        store_value = normalize_store_id(store_input) or normalize_store_id(vault_input)
        data_root_value = normalize_data_root_path(data_root_input)

        if not api_key_value:
            errors.append("Google API key is required.")
        elif not valid_google_api_key_format(api_key_value):
            errors.append("Google API key format is invalid.")
        if not pass_value and not ((PASSWORD or "").strip() or (PASSWORD_HASH or "").strip()):
            errors.append("Edith password is required.")
        if pass_value and pass_value != pass_confirm:
            errors.append("Password confirmation does not match.")
        if google_mode and not vault_value:
            errors.append("Edith vault id is required in Google retrieval mode.")
        if google_mode and not store_value:
            errors.append("Edith storage id is required in Google retrieval mode.")

        if errors:
            for err in errors:
                st.error(err)
        else:
            try:
                save_connection_settings(
                    api_key_value=api_key_value,
                    vault_value=vault_value,
                    store_value=store_value,
                    password_value=pass_value,
                    data_root_value=data_root_value,
                )
            except Exception as exc:
                st.error(f"Could not save settings: {exc}")
            else:
                st.success("Saved. Reloading Edith...")
                st.rerun()

    st.stop()

if "auth_unlocked" not in st.session_state:
    st.session_state.auth_unlocked = not REQUIRE_PASSWORD
if "last_activity_ts" not in st.session_state:
    st.session_state.last_activity_ts = time.time()
if "lock_reason" not in st.session_state:
    st.session_state.lock_reason = ""

if REQUIRE_PASSWORD:
    now_ts = time.time()
    auto_lock_seconds = max(60, int(AUTO_LOCK_MINUTES * 60))
    if (
        st.session_state.auth_unlocked
        and auto_lock_seconds > 0
        and (now_ts - float(st.session_state.last_activity_ts)) > auto_lock_seconds
    ):
        st.session_state.auth_unlocked = False
        st.session_state.lock_reason = f"Session locked after {AUTO_LOCK_MINUTES} minutes of inactivity."

    if not st.session_state.auth_unlocked:
        st.markdown(
            """
<div class="hero">
  <div class="hero-title">Edith Locked</div>
  <div class="hero-sub">Enter your password to continue.</div>
</div>
""",
            unsafe_allow_html=True,
        )
        if st.session_state.lock_reason:
            st.caption(st.session_state.lock_reason)
        unlock_pw = st.text_input("Password", type="password", key="edith_unlock_password")
        unlock_col1, unlock_col2 = st.columns([0.2, 0.8])
        unlock_clicked = unlock_col1.button("Unlock", key="unlock_submit")
        if unlock_clicked:
            if verify_password(unlock_pw):
                st.session_state.auth_unlocked = True
                st.session_state.last_activity_ts = now_ts
                st.session_state.lock_reason = ""
                st.rerun()
            else:
                st.error("Password incorrect.")
        st.stop()
    st.session_state.last_activity_ts = now_ts
elif PASSWORD or PASSWORD_HASH:
    pw = st.sidebar.text_input("Password (optional)", type="password")
    if pw and not verify_password(pw):
        st.stop()

oauth_identity = resolve_oauth_identity()
if OAUTH_REQUIRED and not oauth_identity.get("ok"):
    st.markdown(
        """
<div class="hero">
  <div class="hero-title">OAuth Required</div>
  <div class="hero-sub">This deployment requires authenticated access before Edith can load.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.error(str(oauth_identity.get("reason") or "OAuth identity check failed."))
    st.stop()

active_user_email = clean_text(oauth_identity.get("email") or "")
active_user_role = resolve_session_role(active_user_email)
st.session_state.user_email = active_user_email
st.session_state.user_role = active_user_role
effective_user_role = normalize_role_name(st.session_state.get("user_role", active_user_role))
can_chat = role_has_permission(effective_user_role, "chat.ask")
can_upload_files = role_has_permission(effective_user_role, "files.upload")
can_run_index = role_has_permission(effective_user_role, "index.run")
can_manage_watcher = role_has_permission(effective_user_role, "watcher.manage")
can_sync_vault = role_has_permission(effective_user_role, "vault.sync")
can_list_vault = role_has_permission(effective_user_role, "vault.list")
can_update_privacy = role_has_permission(effective_user_role, "privacy.update")
can_update_connection = role_has_permission(effective_user_role, "settings.connection")
can_delete_data = role_has_permission(effective_user_role, "data.delete")
can_reset_data = role_has_permission(effective_user_role, "data.reset")
can_manage_tal = role_has_permission(effective_user_role, "tal.manage")

if "model_profile" not in st.session_state:
    default_profile = MODEL_PROFILE_DEFAULT if MODEL_PROFILE_DEFAULT in MODEL_PROFILES else "latest"
    st.session_state.model_profile = default_profile
if "model_override" not in st.session_state:
    st.session_state.model_override = MODEL_OVERRIDE
if "model_fallbacks" not in st.session_state:
    st.session_state.model_fallbacks = MODEL_FALLBACKS_ENV
if "retrieval_backend" not in st.session_state:
    st.session_state.retrieval_backend = RETRIEVAL_BACKEND_DEFAULT

api_fingerprint = hashlib.sha256((API_KEY or "").encode("utf-8")).hexdigest()[:12]
available_models = list_available_models(api_fingerprint)
model_chain = resolve_model_chain(
    profile=st.session_state.model_profile,
    override_model=st.session_state.model_override,
    fallback_csv=st.session_state.model_fallbacks,
    available_models=available_models,
    allow_preview=ALLOW_PREVIEW_MODELS,
)
ACTIVE_MODEL = model_chain[0]
chroma_ready = bool(chroma_runtime_available and chroma_runtime_available())
hero_model = html.escape(str(ACTIVE_MODEL))
if st.session_state.retrieval_backend == "chroma" and not chroma_ready:
    hero_backend = "chroma (unavailable)"
else:
    hero_backend = html.escape(str(st.session_state.retrieval_backend))
hero_role = html.escape(active_user_role.title())
if active_user_email:
    hero_identity = html.escape(active_user_email)
else:
    hero_identity = "local-session"
hero_auth = "OAuth" if OAUTH_REQUIRED else "Password"

st.markdown(
    f"""
<div class="hero">
  <div class="hero-title">Edith</div>
  <div class="hero-sub">ChatGPT for your world. Proof-linked, organized, and safe by default.</div>
  <div class="status-pill">Model: {hero_model} | Backend: {hero_backend} | Role: {hero_role} | Auth: {hero_auth} ({hero_identity})</div>
</div>
""",
    unsafe_allow_html=True,
)

st.session_state.setdefault("open_connection_settings", False)
st.session_state.setdefault("desktop_setup_open_request", False)

top_spacer, top_settings = st.columns([0.72, 0.28])
with top_settings:
    if st.button(
        "Settings",
        key="main_settings_btn",
        use_container_width=True,
        disabled=not can_update_connection,
    ):
        if DESKTOP_MODE:
            st.session_state.desktop_setup_open_request = True
        else:
            st.session_state.open_connection_settings = True
        st.rerun()

if DESKTOP_MODE and st.session_state.desktop_setup_open_request:
    components.html(
        """
<script>
(() => {
  try {
    const host = (window.parent && window.parent.edithDesktop) ? window.parent : window;
    if (host.edithDesktop && host.edithDesktop.openSetup) {
      host.edithDesktop.openSetup();
    }
  } catch (_) {}
})();
</script>
        """,
        height=0,
    )
    st.session_state.desktop_setup_open_request = False
    st.caption("Opening desktop settings...")

# Load index report for citations and library views
report_mtime = INDEX_REPORT.stat().st_mtime if INDEX_REPORT.exists() else 0.0
index_map = load_index_report(report_mtime)
index_rows = load_index_rows(report_mtime)
glossary_mtime = GLOSSARY_GRAPH_PATH.stat().st_mtime if GLOSSARY_GRAPH_PATH.exists() else 0.0
citation_graph_mtime = CITATION_GRAPH_PATH.stat().st_mtime if CITATION_GRAPH_PATH.exists() else 0.0
chapter_anchors_mtime = CHAPTER_ANCHORS_PATH.stat().st_mtime if CHAPTER_ANCHORS_PATH.exists() else 0.0
claim_inventory_mtime = CLAIM_INVENTORY_PATH.stat().st_mtime if CLAIM_INVENTORY_PATH.exists() else 0.0
experiment_ledger_mtime = EXPERIMENT_LEDGER_PATH.stat().st_mtime if EXPERIMENT_LEDGER_PATH.exists() else 0.0
index_health_mtime = INDEX_HEALTH_REPORT_PATH.stat().st_mtime if INDEX_HEALTH_REPORT_PATH.exists() else 0.0
bibliography_mtime = BIBLIOGRAPHY_DB_PATH.stat().st_mtime if BIBLIOGRAPHY_DB_PATH.exists() else 0.0
timeline_mtime = ENTITY_TIMELINE_PATH.stat().st_mtime if ENTITY_TIMELINE_PATH.exists() else 0.0
glossary_graph = load_json_artifact(str(GLOSSARY_GRAPH_PATH), glossary_mtime)
citation_graph = load_json_artifact(str(CITATION_GRAPH_PATH), citation_graph_mtime)
chapter_anchors = load_json_artifact(str(CHAPTER_ANCHORS_PATH), chapter_anchors_mtime)
claim_inventory = load_json_artifact(str(CLAIM_INVENTORY_PATH), claim_inventory_mtime)
experiment_ledger = load_json_artifact(str(EXPERIMENT_LEDGER_PATH), experiment_ledger_mtime)
index_health_report = load_json_artifact(str(INDEX_HEALTH_REPORT_PATH), index_health_mtime)
bibliography_db = load_json_artifact(str(BIBLIOGRAPHY_DB_PATH), bibliography_mtime)
entity_timeline = load_json_artifact(str(ENTITY_TIMELINE_PATH), timeline_mtime)
project_options = ["All"] + list_projects(DATA_ROOT)
active_retrieval_profile = load_retrieval_profile()
init_feedback_store()

if "queued_prompt" not in st.session_state:
    st.session_state.queued_prompt = None
if "show_not_found_debug" not in st.session_state:
    st.session_state.show_not_found_debug = False
if "replay_payload" not in st.session_state:
    st.session_state.replay_payload = None
if "source_mode" not in st.session_state:
    st.session_state.source_mode = SOURCE_MODE_DEFAULT if SOURCE_MODE_DEFAULT in SOURCE_MODES else "Files only"
if "mode_preset" not in st.session_state:
    st.session_state.mode_preset = "Grounded (strict)"
if "strict_citations" not in st.session_state:
    st.session_state.strict_citations = bool(REQUIRE_CITATIONS)
if "simple_ui" not in st.session_state:
    st.session_state.simple_ui = True
if "chatgpt_layout_bootstrap" not in st.session_state:
    # Default new sessions to the cleaner chat-first layout.
    st.session_state.simple_ui = True
    st.session_state.chatgpt_layout_bootstrap = True
if "cloud_index_opt_in" not in st.session_state:
    st.session_state.cloud_index_opt_in = bool(CLOUD_INDEX_OPT_IN)
if "web_domain_allowlist_enabled" not in st.session_state:
    st.session_state.web_domain_allowlist_enabled = bool(WEB_DOMAIN_ALLOWLIST_ENABLED_DEFAULT)
if "web_domain_allowlist" not in st.session_state:
    st.session_state.web_domain_allowlist = WEB_DOMAIN_ALLOWLIST_DEFAULT
if "export_redact_sensitive" not in st.session_state:
    st.session_state.export_redact_sensitive = bool(EXPORT_REDACT_DEFAULT)
if "export_redact_sensitive_data_controls" not in st.session_state:
    st.session_state.export_redact_sensitive_data_controls = bool(
        st.session_state.get("export_redact_sensitive", EXPORT_REDACT_DEFAULT)
    )
if "pending_export_redact_sensitive" in st.session_state:
    # Apply deferred sync before widgets are instantiated for this run.
    pending_export_redact = bool(st.session_state.pop("pending_export_redact_sensitive"))
    st.session_state.export_redact_sensitive = pending_export_redact
    st.session_state.export_redact_sensitive_data_controls = pending_export_redact
if "audit_mode" not in st.session_state:
    st.session_state.audit_mode = False
if "open_connection_settings" not in st.session_state:
    st.session_state.open_connection_settings = False
if "desktop_setup_open_request" not in st.session_state:
    st.session_state.desktop_setup_open_request = False
if "scoped_doc_path" not in st.session_state:
    st.session_state.scoped_doc_path = ""
if "inline_upload_open" not in st.session_state:
    st.session_state.inline_upload_open = False
if "stop_generation" not in st.session_state:
    st.session_state.stop_generation = False
if "chroma_rerank_on" not in st.session_state:
    st.session_state.chroma_rerank_on = bool(active_retrieval_profile.get("rerank_on", CHROMA_RERANK_ENABLED_DEFAULT))
if "chroma_diversity_lambda" not in st.session_state:
    st.session_state.chroma_diversity_lambda = float(active_retrieval_profile.get("diversity_lambda", CHROMA_DIVERSITY_LAMBDA))
if "chroma_bm25_weight" not in st.session_state:
    st.session_state.chroma_bm25_weight = float(active_retrieval_profile.get("bm25_weight", CHROMA_BM25_WEIGHT))
if "chroma_top_k" not in st.session_state:
    st.session_state.chroma_top_k = int(active_retrieval_profile.get("top_k", CHROMA_TOP_K))
if "chroma_rerank_top_n" not in st.session_state:
    st.session_state.chroma_rerank_top_n = int(active_retrieval_profile.get("rerank_top_n", CHROMA_RERANK_TOP_N))
if "section_filter" not in st.session_state:
    st.session_state.section_filter = CHROMA_SECTION_FILTER_DEFAULT
if "doc_type_filter" not in st.session_state:
    st.session_state.doc_type_filter = CHROMA_DOC_TYPE_FILTER_DEFAULT
if "context_packing_on" not in st.session_state:
    st.session_state.context_packing_on = bool(CONTEXT_PACKING_DEFAULT)
if "distill_query_on" not in st.session_state:
    st.session_state.distill_query_on = bool(DISTILL_RETRIEVAL_QUERY_DEFAULT)
if "next_questions_on" not in st.session_state:
    st.session_state.next_questions_on = bool(NEXT_QUESTIONS_DEFAULT)
if "researcher_mode" not in st.session_state:
    st.session_state.researcher_mode = bool(RESEARCHER_MODE_DEFAULT)
if "verbosity_level" not in st.session_state:
    st.session_state.verbosity_level = "standard"
if "writing_style" not in st.session_state:
    st.session_state.writing_style = "academic"
if "include_methods_table" not in st.session_state:
    st.session_state.include_methods_table = False
if "include_limitations" not in st.session_state:
    st.session_state.include_limitations = True
if "action_approval_on" not in st.session_state:
    st.session_state.action_approval_on = bool(ACTION_APPROVAL_DEFAULT)
if "action_outputs_enabled" not in st.session_state:
    st.session_state.action_outputs_enabled = bool(AGENT_ACTIONS_ENABLED_DEFAULT)
if "auto_scope_doc" not in st.session_state:
    st.session_state.auto_scope_doc = True
if "source_quality" not in st.session_state:
    st.session_state.source_quality = "Balanced"
if "answer_type" not in st.session_state:
    st.session_state.answer_type = "Explain"
if "methods_extract" not in st.session_state:
    st.session_state.methods_extract = {}
if "anki_flashcards_cards" not in st.session_state:
    st.session_state.anki_flashcards_cards = []
if "anki_flashcards_meta" not in st.session_state:
    st.session_state.anki_flashcards_meta = {}
if "reindex_output" not in st.session_state:
    st.session_state.reindex_output = ""
    st.session_state.reindex_code = None
if "vault_sync_output" not in st.session_state:
    st.session_state.vault_sync_output = ""
    st.session_state.vault_sync_code = None
if "vault_list_output" not in st.session_state:
    st.session_state.vault_list_output = ""
    st.session_state.vault_list_code = None
if "retention_cleaned_count" not in st.session_state:
    st.session_state.retention_cleaned_count = apply_chat_retention_policy(CHAT_RETENTION_DAYS)
if "last_response_meta" not in st.session_state:
    st.session_state.last_response_meta = {}
if "sft_export_train_jsonl" not in st.session_state:
    st.session_state.sft_export_train_jsonl = ""
if "sft_export_val_jsonl" not in st.session_state:
    st.session_state.sft_export_val_jsonl = ""
if "sft_export_summary" not in st.session_state:
    st.session_state.sft_export_summary = {}
if "feedback_summary" not in st.session_state:
    st.session_state.feedback_summary = load_feedback_summary()
if "tuning_candidate_profile" not in st.session_state:
    st.session_state.tuning_candidate_profile = {}
if "tuning_candidate_reasons" not in st.session_state:
    st.session_state.tuning_candidate_reasons = []
if "tuning_eval_result" not in st.session_state:
    st.session_state.tuning_eval_result = {}
if "ab_eval_rows" not in st.session_state:
    st.session_state.ab_eval_rows = []
if "doctor_report" not in st.session_state:
    st.session_state.doctor_report = {}
if "active_chat_id" not in st.session_state:
    saved = list_saved_chats()
    if saved:
        st.session_state.active_chat_id = saved[0]["id"]
    else:
        chat = new_chat_state()
        save_chat(chat)
        st.session_state.active_chat_id = chat["id"]
if "chat_data" not in st.session_state:
    chat = load_chat(st.session_state.active_chat_id)
    if not chat:
        chat = new_chat_state()
        save_chat(chat)
    st.session_state.chat_data = chat
if "msgs" not in st.session_state:
    st.session_state.msgs = st.session_state.chat_data.get("messages", [])


def persist_current_chat():
    st.session_state.chat_data["messages"] = st.session_state.msgs
    if st.session_state.chat_data.get("title", "New chat") == "New chat":
        st.session_state.chat_data["title"] = auto_title_from_first_user_message(st.session_state.msgs)
    save_chat(st.session_state.chat_data)


# Sidebar controls
with st.sidebar:
    st.markdown("### Workspace")
    st.caption(f"Access role: {effective_user_role}")
    if OAUTH_REQUIRED:
        st.caption("OAuth: required")
    if RATE_LIMIT_ENABLED:
        st.caption(
            f"Rate limits: chat {RATE_LIMIT_CHAT_MAX}/{RATE_LIMIT_CHAT_WINDOW_SECONDS}s, "
            f"admin {RATE_LIMIT_MUTATION_MAX}/{RATE_LIMIT_MUTATION_WINDOW_SECONDS}s"
        )
    data_root_path = Path(DATA_ROOT).expanduser() if DATA_ROOT else None
    if data_root_path and data_root_path.exists():
        st.caption(f"Library folder: `{data_root_path}`")
        st.markdown(f"[Open folder](file://{url_quote(str(data_root_path.resolve()))})")
    else:
        st.caption("Library folder: not configured")

    st.markdown("### Session")
    saved_chats = list_saved_chats()
    chat_search = st.text_input("Search chats", value="", key="chat_search_text", placeholder="Find a conversation")
    if chat_search.strip():
        needle = chat_search.strip().lower()
        saved_chats = [
            c
            for c in saved_chats
            if needle in (c.get("title", "").lower() + " " + c.get("updated_at", "").lower())
        ]
    labels = [f"{c['title']} | {c['updated_at']}" for c in saved_chats]
    ids = [c["id"] for c in saved_chats]
    if not ids:
        ids = [st.session_state.chat_data["id"]]
        labels = [f"{st.session_state.chat_data.get('title', 'New chat')} | current"]
    selected_id = st.selectbox(
        "Saved chats",
        ids,
        index=ids.index(st.session_state.active_chat_id) if st.session_state.active_chat_id in ids else 0,
        format_func=lambda x: labels[ids.index(x)],
    )
    if selected_id != st.session_state.active_chat_id:
        loaded = load_chat(selected_id)
        if loaded:
            st.session_state.active_chat_id = selected_id
            st.session_state.chat_data = loaded
            st.session_state.msgs = loaded.get("messages", [])
            st.rerun()
    if saved_chats:
        st.caption("Recent")
        for c in saved_chats[:5]:
            label = clean_text(c.get("title") or "Chat")
            if st.button(label[:42], key=f"chat_recent_{c.get('id')}"):
                loaded = load_chat(c.get("id"))
                if loaded:
                    st.session_state.active_chat_id = c.get("id")
                    st.session_state.chat_data = loaded
                    st.session_state.msgs = loaded.get("messages", [])
                    st.rerun()

    col_new, col_clear = st.columns(2)
    if col_new.button("New chat"):
        chat = new_chat_state()
        save_chat(chat)
        st.session_state.active_chat_id = chat["id"]
        st.session_state.chat_data = chat
        st.session_state.msgs = []
        st.rerun()
    if col_clear.button("Clear chat"):
        st.session_state.msgs = []
        st.session_state.chat_data["messages"] = []
        st.session_state.chat_data["title"] = "New chat"
        persist_current_chat()
        st.session_state.last_response_meta = {}
        st.rerun()

    if REQUIRE_PASSWORD:
        st.caption(f"App lock: On (auto-lock {AUTO_LOCK_MINUTES} min)")
        if st.button("Lock now", key="lock_now_btn"):
            st.session_state.auth_unlocked = False
            st.session_state.lock_reason = "Locked manually."
            st.rerun()

    simple_ui = st.toggle(
        "ChatGPT layout",
        value=bool(st.session_state.get("simple_ui", True)),
        key="simple_ui",
        help="Uses a cleaner, chat-first interface and hides most technical controls.",
    )
    if simple_ui:
        st.caption("Clean mode is on. Advanced diagnostics and tuning stay in expandable sections.")

    st.markdown("### Export")
    export_redact_sensitive = st.toggle(
        "Redact sensitive fields",
        value=bool(st.session_state.get("export_redact_sensitive", EXPORT_REDACT_DEFAULT)),
        key="export_redact_sensitive",
        help="Hides emails, IDs, and local file paths in exported chat files.",
    )
    export_chat = dict(st.session_state.chat_data)
    export_chat["messages"] = st.session_state.msgs
    export_md = chat_to_markdown(export_chat, redact_sensitive=export_redact_sensitive)
    st.download_button(
        "Download Markdown",
        data=export_md,
        file_name=f"{st.session_state.chat_data['id']}.md",
        mime="text/markdown",
    )
    pdf_bytes = chat_to_pdf_bytes(export_chat, redact_sensitive=export_redact_sensitive)
    if pdf_bytes:
        st.download_button(
            "Download PDF",
            data=pdf_bytes,
            file_name=f"{st.session_state.chat_data['id']}.pdf",
            mime="application/pdf",
        )
    else:
        st.caption("PDF export unavailable. Install `reportlab`.")

    with st.expander("Flashcards (Anki)", expanded=False):
        st.caption("Generate source-grounded flashcards from the latest cited answer.")
        latest_card_msg = latest_assistant_message_with_sources(st.session_state.get("msgs", []))
        fc_col1, fc_col2, fc_col3 = st.columns(3)
        anki_card_type = fc_col1.selectbox(
            "Card type",
            ["Basic", "Cloze", "Both"],
            index=2,
            key="anki_card_type",
        )
        anki_focus = fc_col2.selectbox(
            "Focus",
            ["Concepts", "Methods", "Findings", "Critiques", "Definitions"],
            index=0,
            key="anki_focus",
        )
        anki_count = fc_col3.selectbox("Count", [10, 15, 25, 50], index=1, key="anki_count")
        if latest_card_msg is None:
            st.caption("Generate at least one assistant answer with sources first.")

        if st.button(
            "Generate flashcards",
            key="generate_anki_flashcards_btn",
            disabled=(latest_card_msg is None) or (not can_chat),
        ):
            if enforce_rate_limit(
                "flashcards_generate",
                RATE_LIMIT_MUTATION_MAX,
                RATE_LIMIT_MUTATION_WINDOW_SECONDS,
                "Flashcard generation",
            ):
                msgs_now = st.session_state.get("msgs", [])
                source_question = ""
                for i in range(len(msgs_now) - 1, -1, -1):
                    m = msgs_now[i] or {}
                    if m.get("role") == "assistant" and m.get("sources"):
                        source_question = find_assistant_query(msgs_now, i)
                        break
                cards, meta = generate_anki_flashcards(
                    question=source_question,
                    answer_text=latest_card_msg.get("text", ""),
                    sources=latest_card_msg.get("sources") or [],
                    model_chain=model_chain,
                    card_type=anki_card_type,
                    card_count=int(anki_count),
                    focus=anki_focus,
                    index_map=index_map,
                )
                st.session_state.anki_flashcards_cards = cards
                st.session_state.anki_flashcards_meta = meta
                st.session_state.anki_flashcards_query = source_question
                if meta.get("ok"):
                    st.success(
                        f"Generated {int(meta.get('count', len(cards)))} flashcards using {meta.get('model', 'model')}"
                    )
                else:
                    st.error(meta.get("error") or "Flashcard generation failed.")

        cards = st.session_state.get("anki_flashcards_cards") or []
        if cards:
            preview = []
            for row in cards[:8]:
                preview.append(
                    {
                        "Type": row.get("NoteType"),
                        "Front": row.get("Front"),
                        "Text": row.get("Text"),
                        "Tags": row.get("Tags"),
                        "Citation": row.get("Citation"),
                    }
                )
            st.dataframe(preview, use_container_width=True, hide_index=True)
            csv_text = flashcards_to_delimited_text(cards, delimiter=",")
            tsv_text = flashcards_to_delimited_text(cards, delimiter="\t")
            stamp = datetime.now().strftime("%Y%m%d_%H%M")
            base_name = f"{st.session_state.chat_data['id']}_anki_{stamp}"
            dl_col1, dl_col2 = st.columns(2)
            dl_col1.download_button(
                "Download Anki CSV",
                data=csv_text,
                file_name=f"{base_name}.csv",
                mime="text/csv",
            )
            dl_col2.download_button(
                "Download Anki TSV",
                data=tsv_text,
                file_name=f"{base_name}.tsv",
                mime="text/tab-separated-values",
            )
            st.caption(
                "Anki import tip: Basic cards map to Front/Back. Cloze cards map Text to cloze Text and Back to Extra."
            )

    if not simple_ui:
        st.markdown("### Replay")
        st.caption("Use the Runs tab for replay and run diffs.")
        if st.button("Clear replay state"):
            st.session_state.replay_payload = None

        with st.expander("Advanced: Model Selection", expanded=False):
            if st.button("Refresh model catalog"):
                list_available_models.clear()
                st.rerun()

            profile_index = MODEL_PROFILES.index(st.session_state.model_profile) if st.session_state.model_profile in MODEL_PROFILES else 0
            st.session_state.model_profile = st.selectbox("Model profile", MODEL_PROFILES, index=profile_index)
            st.session_state.model_override = st.text_input(
                "Model override (optional)",
                value=st.session_state.model_override,
                placeholder="gemini-3-pro-preview",
            ).strip()
            st.session_state.model_fallbacks = st.text_input(
                "Fallback models (comma separated)",
                value=st.session_state.model_fallbacks,
                placeholder="gemini-3-flash-preview, gemini-2.5-pro",
            ).strip()

    model_chain = resolve_model_chain(
        profile=st.session_state.model_profile,
        override_model=st.session_state.model_override,
        fallback_csv=st.session_state.model_fallbacks,
        available_models=available_models,
        allow_preview=ALLOW_PREVIEW_MODELS,
    )
    ACTIVE_MODEL = model_chain[0]
    if available_models:
        st.caption(f"Models detected: {len(available_models)}")
    else:
        st.caption("Model catalog unavailable; using configured chain.")
    st.caption("Active: " + " -> ".join(model_chain[:3]))

    st.markdown("### Controls")
    preset_index = MODE_PRESETS.index(st.session_state.mode_preset) if st.session_state.mode_preset in MODE_PRESETS else 0
    mode_preset = st.selectbox("Preset behavior", MODE_PRESETS, index=preset_index, key="mode_preset")
    preset_cfg = preset_defaults(mode_preset)
    preset_locked = mode_preset != "Custom"
    if preset_cfg:
        st.session_state.source_mode = preset_cfg.get("source_mode", st.session_state.source_mode)
        st.session_state.strict_citations = bool(preset_cfg.get("strict_citations", st.session_state.strict_citations))
        if "researcher_mode" in preset_cfg:
            st.session_state.researcher_mode = bool(preset_cfg.get("researcher_mode"))
        if "verbosity_level" in preset_cfg:
            st.session_state.verbosity_level = clean_text(preset_cfg.get("verbosity_level")).lower()
        if "writing_style" in preset_cfg:
            st.session_state.writing_style = clean_text(preset_cfg.get("writing_style")).lower()
        if "include_methods_table" in preset_cfg:
            st.session_state.include_methods_table = bool(preset_cfg.get("include_methods_table"))
        if "include_limitations" in preset_cfg:
            st.session_state.include_limitations = bool(preset_cfg.get("include_limitations"))
        if "next_questions_on" in preset_cfg:
            st.session_state.next_questions_on = bool(preset_cfg.get("next_questions_on"))

    source_mode_options = SOURCE_MODES if ALLOW_WEB_TOOLS else ["Files only"]
    mode_default = st.session_state.source_mode if st.session_state.source_mode in source_mode_options else source_mode_options[0]
    source_mode = st.selectbox("Answer scope", source_mode_options, index=source_mode_options.index(mode_default), key="source_mode", disabled=preset_locked)
    strict_citations = st.toggle("Require citations", value=st.session_state.strict_citations, key="strict_citations", disabled=preset_locked)
    researcher_mode = st.toggle(
        "Researcher mode",
        value=bool(st.session_state.researcher_mode),
        key="researcher_mode",
        help="Breadth→depth retrieval and synthesis pipeline for literature-style answers.",
        disabled=preset_locked,
    )
    verbosity_level = st.selectbox(
        "Depth",
        ["concise", "standard", "deep"],
        index=["concise", "standard", "deep"].index(st.session_state.verbosity_level)
        if st.session_state.verbosity_level in {"concise", "standard", "deep"}
        else 1,
        key="verbosity_level",
        disabled=preset_locked,
    )
    writing_style = st.selectbox(
        "Style",
        ["academic", "plain"],
        index=0 if st.session_state.writing_style == "academic" else 1,
        key="writing_style",
        disabled=preset_locked,
    )
    include_methods_table = st.toggle(
        "Include methods table",
        value=bool(st.session_state.include_methods_table),
        key="include_methods_table",
        disabled=preset_locked,
    )
    include_limitations = st.toggle(
        "Include limitations",
        value=bool(st.session_state.include_limitations),
        key="include_limitations",
        disabled=preset_locked,
    )
    project_filter = st.selectbox("Project filter (optional)", project_options)

    default_hybrid = preset_cfg.get("hybrid_policy", HYBRID_POLICY_DEFAULT) if HYBRID_POLICY_DEFAULT in HYBRID_FILE_POLICIES else preset_cfg.get("hybrid_policy", "require_files")
    hybrid_policy = default_hybrid
    tag_filter = ""
    section_filter = st.session_state.get("section_filter", CHROMA_SECTION_FILTER_DEFAULT)
    doc_type_filter = st.session_state.get("doc_type_filter", CHROMA_DOC_TYPE_FILTER_DEFAULT)
    audit_mode = st.session_state.audit_mode
    query_rewrite_on = bool(preset_cfg.get("query_rewrite_on", QUERY_REWRITE_DEFAULT))
    support_audit_on = bool(preset_cfg.get("support_audit_on", SUPPORT_AUDIT_DEFAULT))
    confidence_routing_on = bool(preset_cfg.get("confidence_routing_on", CONFIDENCE_ROUTING_DEFAULT))
    multi_pass_on = bool(MULTI_PASS_DEFAULT)
    recursive_controller_on = bool(RECURSIVE_CONTROLLER_DEFAULT)
    contradiction_check_on = bool(CONTRADICTION_CHECK_DEFAULT)
    context_packing_on = bool(st.session_state.context_packing_on)
    distill_query_on = bool(st.session_state.distill_query_on)
    next_questions_on = bool(st.session_state.next_questions_on)
    researcher_mode_on = bool(st.session_state.researcher_mode)
    verbosity_level_on = clean_text(st.session_state.verbosity_level or "standard").lower()
    writing_style_on = clean_text(st.session_state.writing_style or "academic").lower()
    include_methods_table_on = bool(st.session_state.include_methods_table)
    include_limitations_on = bool(st.session_state.include_limitations)
    action_approval_on = bool(st.session_state.action_approval_on)
    action_outputs_enabled = bool(st.session_state.action_outputs_enabled)
    sentence_provenance_on = bool(preset_cfg.get("sentence_provenance_on", SENTENCE_PROVENANCE_DEFAULT))
    strict_sentence_tags_on = bool(preset_cfg.get("strict_sentence_tags_on", STRICT_SENTENCE_TAGS_DEFAULT))
    streaming = STREAMING_DEFAULT

    with st.expander("Advanced", expanded=False):
        backend_labels = list(RETRIEVAL_BACKEND_LABELS.keys())
        backend_keys = [RETRIEVAL_BACKEND_LABELS[x] for x in backend_labels]
        default_backend = st.session_state.retrieval_backend if st.session_state.retrieval_backend in backend_keys else "google"
        backend_index = backend_keys.index(default_backend)
        backend_label = st.selectbox("Retrieval backend", backend_labels, index=backend_index)
        retrieval_backend = RETRIEVAL_BACKEND_LABELS[backend_label]
        if retrieval_backend == "chroma" and not chroma_ready:
            st.caption("Local Chroma unavailable; install dependencies and restart.")
        st.session_state.retrieval_backend = retrieval_backend

        hybrid_policy = st.select_slider(
            "Hybrid file policy",
            options=HYBRID_FILE_POLICIES,
            value=default_hybrid,
            disabled=(source_mode != "Files + Web") or preset_locked,
        )
        tag_filter = st.text_input("Tag filter (from filename #tag)", value="").strip()
        section_filter = st.text_input(
            "Section filter (e.g., Methods)",
            value=st.session_state.get("section_filter", CHROMA_SECTION_FILTER_DEFAULT),
            help="Filters local file retrieval to matching section headings.",
            disabled=(st.session_state.retrieval_backend != "chroma"),
        ).strip()
        doc_type_filter = st.text_input(
            "Doc type filter (comma-separated)",
            value=st.session_state.get("doc_type_filter", CHROMA_DOC_TYPE_FILTER_DEFAULT),
            help="Example: thesis_chapter,paper,note,log,data_table",
            disabled=(st.session_state.retrieval_backend != "chroma"),
        ).strip()
        if tag_filter.startswith("#"):
            tag_filter = tag_filter[1:]
        if tag_filter.startswith("[") and tag_filter.endswith("]") and len(tag_filter) > 2:
            tag_filter = tag_filter[1:-1]
        audit_mode = st.toggle("Audit mode", value=st.session_state.audit_mode)
        st.session_state.audit_mode = audit_mode

        query_rewrite_on = st.toggle("Query rewrite (extra call)", value=query_rewrite_on, disabled=preset_locked)
        support_audit_on = st.toggle("Support audit (extra call)", value=support_audit_on, disabled=preset_locked)
        confidence_routing_on = st.toggle("Confidence routing (extra call)", value=confidence_routing_on, disabled=preset_locked)
        multi_pass_on = st.toggle("Multi-pass answering (extra call)", value=multi_pass_on)
        recursive_controller_on = st.toggle(
            "Recursive controller v1 (extra calls)",
            value=recursive_controller_on,
            help="Recursively maps and reduces large evidence sets for long-context synthesis.",
        )
        contradiction_check_on = st.toggle("Contradiction check (extra call)", value=contradiction_check_on)
        context_packing_on = st.toggle("Context packing", value=context_packing_on)
        distill_query_on = st.toggle("Conversation-to-query distillation", value=distill_query_on)
        next_questions_on = st.toggle("Grounded next questions", value=next_questions_on)
        action_approval_on = st.toggle("Action approval guardrail", value=action_approval_on)
        action_outputs_enabled = st.toggle(
            "Enable action outputs (write/modify)",
            value=action_outputs_enabled,
            help="Default read-only safety mode keeps this off.",
        )
        sentence_provenance_on = st.toggle("Sentence provenance", value=sentence_provenance_on, disabled=preset_locked)
        strict_sentence_tags_on = st.toggle("Strict sentence tags", value=strict_sentence_tags_on, disabled=preset_locked)
        streaming = st.toggle("Streaming (extra call)", value=STREAMING_DEFAULT)
        if st.session_state.retrieval_backend == "chroma":
            st.session_state.chroma_top_k = st.slider(
                "Top K (local retrieval)",
                min_value=4,
                max_value=20,
                value=int(st.session_state.chroma_top_k),
                step=1,
            )
            st.session_state.chroma_bm25_weight = st.slider(
                "Lexical weight (BM25)",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.chroma_bm25_weight),
                step=0.05,
                help="Higher values favor exact keyword matches.",
            )
            st.session_state.chroma_diversity_lambda = st.slider(
                "Diversity selector",
                min_value=0.0,
                max_value=1.0,
                value=float(st.session_state.chroma_diversity_lambda),
                step=0.05,
                help="Lower values reduce redundant chunks.",
            )
            st.session_state.chroma_rerank_on = st.toggle(
                "Cross-encoder rerank (slower, better quality)",
                value=st.session_state.chroma_rerank_on,
            )
            st.session_state.chroma_rerank_top_n = st.slider(
                "Rerank pool size",
                min_value=6,
                max_value=40,
                value=int(st.session_state.chroma_rerank_top_n),
                step=1,
                disabled=not st.session_state.chroma_rerank_on,
            )

    st.session_state.section_filter = section_filter
    st.session_state.doc_type_filter = doc_type_filter
    st.session_state.context_packing_on = bool(context_packing_on)
    st.session_state.distill_query_on = bool(distill_query_on)
    st.session_state.next_questions_on = bool(next_questions_on)
    st.session_state.action_approval_on = bool(action_approval_on)
    st.session_state.action_outputs_enabled = bool(action_outputs_enabled)
    if not st.session_state.action_outputs_enabled:
        st.caption("Action tools mode: read-only (default).")
    else:
        st.caption("Action tools mode: write/modify enabled for this session.")
    tool_allowlist_now = normalized_allowlist(TOOL_ALLOWLIST_RAW)
    if (not simple_ui) and tool_allowlist_now:
        st.caption("Tool allowlist: " + ", ".join(tool_allowlist_now))
    workspace_allowlist_now = normalized_allowlist(WORKSPACE_ALLOWLIST_RAW)
    if (not simple_ui) and workspace_allowlist_now:
        st.caption("Workspace allowlist: " + ", ".join(workspace_allowlist_now))

    retrieval_backend = st.session_state.retrieval_backend
    st.session_state.chroma_top_k = clamp_int(st.session_state.get("chroma_top_k", CHROMA_TOP_K), 4, 20)
    st.session_state.chroma_bm25_weight = clamp_float(st.session_state.get("chroma_bm25_weight", CHROMA_BM25_WEIGHT), 0.05, 0.95)
    st.session_state.chroma_diversity_lambda = clamp_float(st.session_state.get("chroma_diversity_lambda", CHROMA_DIVERSITY_LAMBDA), 0.1, 0.95)
    st.session_state.chroma_rerank_top_n = clamp_int(st.session_state.get("chroma_rerank_top_n", CHROMA_RERANK_TOP_N), 6, 40)
    if st.session_state.chroma_rerank_top_n < st.session_state.chroma_top_k:
        st.session_state.chroma_rerank_top_n = st.session_state.chroma_top_k
    if source_mode != "Files + Web":
        hybrid_policy = default_hybrid
    if source_mode == "Files only":
        if retrieval_backend == "chroma":
            st.caption("Queries restricted to local Chroma-indexed files.")
            if CHROMA_FORCE_RERANK_FILES_ONLY:
                st.caption("Two-stage retrieval is enforced: retrieve -> cross-encoder rerank.")
            if (st.session_state.get("section_filter") or "").strip():
                st.caption(f"Section filter active: {st.session_state.get('section_filter').strip()}")
            if (st.session_state.get("doc_type_filter") or "").strip():
                st.caption(f"Doc type filter active: {st.session_state.get('doc_type_filter').strip()}")
        else:
            st.caption("Queries restricted to your indexed files.")
        st.caption("Files-only mode enforces citation-only answers.")
        st.caption("Claim-level enforcement is active (unsupported factual sentences are removed or blocked).")
    elif source_mode == "Web only":
        st.caption("Queries restricted to web search sources.")
    else:
        if hybrid_policy == "require_files":
            st.caption("Hybrid mode requires at least one file source per answer.")
        elif hybrid_policy == "prefer_files":
            st.caption("Hybrid mode prefers files, but can answer from web when needed.")
        else:
            st.caption("Hybrid mode allows files and web without file requirement.")
    if not ALLOW_WEB_TOOLS:
        st.caption("Web retrieval is disabled.")
    if DATA_ROOT:
        st.caption(f"Library folder: {DATA_ROOT}")
    else:
        st.caption("Library folder: not set")

    st.markdown("### Quick Start")
    has_api_key = bool((API_KEY or "").strip())
    has_store_config = True if retrieval_backend != "google" else bool((STORE_ID or "").strip() and (VAULT_ID or "").strip())
    has_data_root = bool(DATA_ROOT and Path(DATA_ROOT).expanduser().exists())
    needs_index = source_mode in ("Files only", "Files + Web")
    has_index = bool(index_rows) if needs_index else True

    st.caption(f"{'OK' if has_api_key else 'Needs setup'}: API key")
    if retrieval_backend == "google":
        st.caption(f"{'OK' if has_store_config else 'Needs setup'}: Cloud index configuration")
    else:
        st.caption("OK: Local index configuration")
    st.caption(f"{'OK' if has_data_root else 'Needs setup'}: Library folder")
    st.caption(f"{'OK' if has_index else 'Needs action'}: Indexed documents ({len(index_rows)})")

    qa_col1, qa_col2 = st.columns(2)
    if qa_col1.button("Open settings", key="quick_open_settings", disabled=not can_update_connection):
        if DESKTOP_MODE:
            st.session_state.desktop_setup_open_request = True
        else:
            st.session_state.open_connection_settings = True
        st.rerun()
    if qa_col2.button(
        "Index now",
        key="quick_index_now",
        disabled=(
            (not can_run_index)
            or
            not (Path(__file__).parent / ("index_files.py" if retrieval_backend == "google" else "chroma_index.py")).exists()
            or (retrieval_backend == "google" and not bool(st.session_state.get("cloud_index_opt_in", CLOUD_INDEX_OPT_IN)))
        ),
    ):
        with st.spinner("Indexing..."):
            code, output = run_reindex(retrieval_backend=retrieval_backend)
        st.session_state.reindex_output = output
        st.session_state.reindex_code = code
        save_index_status(code, output)

    st.markdown("### Security")
    st.caption(f"App lock: {'On' if REQUIRE_PASSWORD else 'Off'}")
    st.caption(f"OAuth: {'required' if OAUTH_REQUIRED else 'not required'}")
    st.caption(f"RBAC role: {effective_user_role}")
    if REQUIRE_PASSWORD:
        st.caption(f"Auto-lock: {AUTO_LOCK_MINUTES} minutes")
    if DESKTOP_MODE:
        st.caption("Secrets storage: OS-encrypted desktop storage")
    else:
        st.caption("Secrets storage: local environment configuration")
    if CHAT_ENCRYPTION_ENABLED and CHAT_CIPHER:
        st.caption("Chat history encryption: enabled")
    elif CHAT_ENCRYPTION_ENABLED and not CHAT_CIPHER:
        st.caption("Chat history encryption requested, but cipher unavailable.")
    else:
        st.caption("Chat history encryption: disabled")
    if CHAT_RETENTION_DAYS > 0:
        st.caption(f"Retention policy: delete chats older than {CHAT_RETENTION_DAYS} days")
        if st.session_state.retention_cleaned_count:
            st.caption(f"Auto-cleaned {st.session_state.retention_cleaned_count} old chat(s) this session.")
    else:
        st.caption("Retention policy: disabled")
    if RUN_LEDGER_ENABLED:
        if RUN_LEDGER_ENCRYPT and CHAT_CIPHER:
            st.caption("Run ledger: enabled (encrypted)")
        elif RUN_LEDGER_ENCRYPT and not CHAT_CIPHER:
            st.caption("Run ledger: disabled (encryption requested but cipher unavailable)")
        else:
            st.caption("Run ledger: enabled (plain text)")
    else:
        st.caption("Run ledger: disabled")
    st.caption(f"Web HTTPS-only: {'enabled' if REQUIRE_HTTPS_WEB_SOURCES else 'disabled'}")
    cap_status = tal_web_capability_status()
    st.caption(
        "Web capability tokens: "
        f"once={int(cap_status.get('once', 0))}, chat={int(cap_status.get('chat', 0))}"
    )
    if can_manage_tal:
        if st.button("Revoke all capability tokens", key="revoke_tal_tokens_btn"):
            tal_clear_tokens()
            st.success("Capability tokens revoked.")
            st.rerun()
    st.caption(f"Require citations: {'enabled' if strict_citations else 'disabled'}")
    st.caption(f"Sentence provenance: {'enabled' if sentence_provenance_on else 'disabled'}")
    if not strict_citations and support_audit_on:
        st.caption("Support audit is active, but strict citation gate is disabled.")
    if confidence_routing_on:
        st.caption(f"Low-confidence threshold: {CONFIDENCE_LOW_THRESHOLD:.2f}")

    st.markdown("### Data controls")
    if not can_update_privacy:
        st.caption("RBAC: privacy and policy changes are read-only for your role.")
    policy_col1, policy_col2 = st.columns(2)
    web_enabled_local = policy_col1.toggle(
        "Web access",
        value=bool(ALLOW_WEB_TOOLS),
        key="policy_allow_web_tools",
        help="When off, Edith stays file-only and does not call web tools.",
        disabled=not can_update_privacy,
    )
    cloud_enabled_local = policy_col2.toggle(
        "Cloud index uploads",
        value=bool(st.session_state.get("cloud_index_opt_in", CLOUD_INDEX_OPT_IN)),
        key="cloud_index_opt_in",
        help="When on, indexing/upload actions can send file content to Google File Search store.",
        disabled=not can_update_privacy,
    )
    web_allowlist_enabled = st.toggle(
        "Web domain allowlist",
        value=bool(st.session_state.get("web_domain_allowlist_enabled", WEB_DOMAIN_ALLOWLIST_ENABLED_DEFAULT)),
        key="web_domain_allowlist_enabled",
        help="Restricts web sources to approved domains when web mode is enabled.",
        disabled=not can_update_privacy,
    )
    web_allowlist_raw = st.text_input(
        "Allowed web domains",
        value=st.session_state.get("web_domain_allowlist", WEB_DOMAIN_ALLOWLIST_DEFAULT),
        key="web_domain_allowlist",
        placeholder=".gov,.edu,nature.com,science.org",
        disabled=(not web_allowlist_enabled) or (not can_update_privacy),
    )
    export_redact_now = st.toggle(
        "Redact exports",
        value=bool(
            st.session_state.get(
                "export_redact_sensitive_data_controls",
                st.session_state.get("export_redact_sensitive", EXPORT_REDACT_DEFAULT),
            )
        ),
        key="export_redact_sensitive_data_controls",
        help="Applies redaction to Markdown/PDF exports by default.",
        disabled=not can_update_privacy,
    )
    auto_lock_mins_now = st.number_input(
        "Auto-lock minutes",
        min_value=1,
        max_value=480,
        value=int(AUTO_LOCK_MINUTES),
        step=1,
        disabled=not can_update_privacy,
    )
    if st.button("Save privacy settings", key="save_privacy_settings_btn", disabled=not can_update_privacy):
        if DESKTOP_MODE:
            persist_runtime_policy_settings(
                web_domain_allowlist_enabled=bool(web_allowlist_enabled),
                web_domain_allowlist=web_allowlist_raw,
                export_redact_sensitive=bool(export_redact_now),
                auto_lock_minutes=int(auto_lock_mins_now),
            )
            st.session_state.desktop_setup_open_request = True
            st.success("Saved local privacy preferences. Open desktop settings to update web/cloud toggles.")
        else:
            persist_runtime_policy_settings(
                allow_web_tools=bool(web_enabled_local),
                cloud_index_opt_in=bool(cloud_enabled_local),
                web_domain_allowlist_enabled=bool(web_allowlist_enabled),
                web_domain_allowlist=web_allowlist_raw,
                export_redact_sensitive=bool(export_redact_now),
                auto_lock_minutes=int(auto_lock_mins_now),
            )
            st.success("Privacy settings saved. Reloading…")
        st.session_state.pending_export_redact_sensitive = bool(export_redact_now)
        st.rerun()

    if web_enabled_local and strict_citations and source_mode in ("Files + Web", "Web only"):
        sensitive_hits = detect_sensitive_library_categories(index_rows)
        if sensitive_hits:
            labels = ", ".join(x.replace("_", " ") for x in sensitive_hits)
            st.warning(f"Sensitive categories detected in your library ({labels}). Consider keeping Web access off.")

    dc_col1, dc_col2 = st.columns(2)
    if dc_col1.button("Delete chat history", key="delete_chat_history_btn", disabled=not can_delete_data):
        removed = delete_chat_history()
        st.session_state.msgs = []
        st.session_state.chat_data = new_chat_state()
        save_chat(st.session_state.chat_data)
        st.session_state.active_chat_id = st.session_state.chat_data["id"]
        st.success(f"Deleted {removed} saved chat file(s).")
    if dc_col2.button("Delete local index", key="delete_local_index_btn", disabled=not can_delete_data):
        removed = delete_local_index_data()
        msg = f"Deleted local index artifacts: files={removed.get('files', 0)}, dirs={removed.get('dirs', 0)}"
        if removed.get("skipped_unsafe_dir"):
            msg += f". Skipped unsafe directory: {removed.get('skipped_unsafe_dir')}"
        st.success(msg)
    if st.button("Delete web cache", key="delete_web_cache_btn", disabled=not can_delete_data):
        clear_web_cache()
        st.success("Web cache deleted.")
    reset_col1, reset_col2 = st.columns(2)
    if reset_col1.button("Turn off web access", key="turn_off_web_access_btn", disabled=not can_update_privacy):
        if DESKTOP_MODE:
            st.session_state.desktop_setup_open_request = True
            st.info("Web access is managed in desktop settings.")
        else:
            persist_runtime_policy_settings(allow_web_tools=False)
            st.success("Web access turned off.")
        st.rerun()
    if reset_col2.button("Turn off cloud index", key="turn_off_cloud_index_btn", disabled=not can_update_privacy):
        if DESKTOP_MODE:
            st.session_state.desktop_setup_open_request = True
            st.info("Cloud index is managed in desktop settings.")
        else:
            persist_runtime_policy_settings(cloud_index_opt_in=False)
            st.success("Cloud index uploads turned off.")
        st.rerun()
    st.caption("Reset app clears local chats, index data, cache, and run history while preserving credentials.")
    if st.button("Reset app", key="reset_app_btn", disabled=not can_reset_data):
        removed = reset_local_app_state()
        st.session_state.msgs = []
        st.session_state.last_response_meta = {}
        st.success(
            f"App reset complete. Removed files={removed.get('files', 0)}, dirs={removed.get('dirs', 0)}."
        )
        st.rerun()
    data_export_zip = export_user_data_zip_bytes()
    st.download_button(
        "Export my data",
        data=data_export_zip,
        file_name=f"edith_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
        mime="application/zip",
        key="export_my_data_btn",
    )

    if DESKTOP_MODE:
        st.caption("Connection settings are managed by the Edith desktop setup screen.")
        if can_update_connection:
            components.html(
                """
<div style="padding: 0.2rem 0 0.1rem 0;">
  <button id="edith-open-setup-btn"
    style="border:1px solid rgba(0,0,0,0.14);border-radius:999px;padding:8px 14px;background:#fff;color:#1f2430;font-size:13px;font-weight:600;cursor:pointer;">
    Open Desktop Settings
  </button>
</div>
<script>
(() => {
  const btn = document.getElementById("edith-open-setup-btn");
  if (!btn) return;
  btn.addEventListener("click", async () => {
    try {
      const host = (window.parent && window.parent.edithDesktop) ? window.parent : window;
      if (host.edithDesktop && host.edithDesktop.openSetup) {
        await host.edithDesktop.openSetup();
      }
    } catch (_) {}
  });
})();
</script>
                """,
                height=48,
            )
        else:
            st.caption("RBAC: desktop setup changes are disabled for your role.")
        st.caption("Shortcut: app menu Settings or Command+,.")
    else:
        if not can_update_connection:
            st.caption("RBAC: connection settings are read-only for your role.")
        with st.expander("Connection Settings", expanded=bool(st.session_state.get("open_connection_settings", False))):
            st.caption("Update API/store credentials without editing files manually.")
            with st.form("connection_settings_form", clear_on_submit=False):
                api_key_update = st.text_input(
                    "Google API key (leave blank to keep current)",
                    value="",
                    type="password",
                    disabled=not can_update_connection,
                )
                vault_update = st.text_input(
                    "Edith vault id",
                    value=VAULT_ID or STORE_MAIN or STORE_ID or "",
                    disabled=not can_update_connection,
                )
                store_update = st.text_input(
                    "Edith storage id",
                    value=STORE_ID or STORE_MAIN or VAULT_ID or "",
                    disabled=not can_update_connection,
                )
                data_root_update = st.text_input(
                    "Library folder",
                    value=DATA_ROOT or "",
                    disabled=not can_update_connection,
                )
                pw_update = st.text_input(
                    "New Edith password (optional)",
                    value="",
                    type="password",
                    disabled=not can_update_connection,
                )
                pw_confirm = st.text_input(
                    "Confirm new password",
                    value="",
                    type="password",
                    disabled=not can_update_connection,
                )
                save_connection = st.form_submit_button(
                    "Save connection settings",
                    disabled=not can_update_connection,
                )

            if save_connection:
                api_final = (api_key_update or "").strip() or (API_KEY or "").strip()
                vault_final = normalize_store_id(vault_update)
                store_final = normalize_store_id(store_update) or vault_final
                data_root_final = normalize_data_root_path(data_root_update)
                google_mode = retrieval_backend == "google"
                errors = []
                if not api_final:
                    errors.append("Google API key is required.")
                elif not valid_google_api_key_format(api_final):
                    errors.append("Google API key format is invalid.")
                if google_mode and not vault_final:
                    errors.append("Edith vault id is required in Google retrieval mode.")
                if google_mode and not store_final:
                    errors.append("Edith storage id is required in Google retrieval mode.")
                if pw_update and pw_update != pw_confirm:
                    errors.append("Password confirmation does not match.")

                if errors:
                    for err in errors:
                        st.error(err)
                else:
                    try:
                        save_connection_settings(
                            api_key_value=api_final,
                            vault_value=vault_final,
                            store_value=store_final,
                            password_value=pw_update,
                            data_root_value=data_root_final,
                        )
                    except Exception as exc:
                        st.error(f"Could not save connection settings: {exc}")
                    else:
                        st.success("Connection settings saved. Reloading...")
                        st.rerun()
        if st.session_state.get("open_connection_settings"):
            st.session_state.open_connection_settings = False

    st.markdown("### Current Query")
    st.caption(f"Mode: {source_mode}")
    st.caption(f"Researcher mode: {'on' if researcher_mode else 'off'}")
    st.caption(f"Depth/style: {verbosity_level} / {writing_style}")
    if include_methods_table or include_limitations:
        extras = []
        if include_methods_table:
            extras.append("methods table")
        if include_limitations:
            extras.append("limitations")
        st.caption("Extras: " + ", ".join(extras))
    st.caption(f"Strict citations: {'on' if strict_citations else 'off'}")
    st.caption(f"Project filter: {project_filter}")
    if tag_filter:
        st.caption(f"Tag filter: {tag_filter}")

    st.markdown("### Health")
    static_checks = [
        ("API key", bool((API_KEY or "").strip()), "Configured"),
        (
            "OAuth identity",
            True if not OAUTH_REQUIRED else bool(active_user_email),
            "Required and present" if OAUTH_REQUIRED and active_user_email else ("Not required" if not OAUTH_REQUIRED else "Missing"),
        ),
        ("RBAC role", True, effective_user_role),
        (
            "Cloud index IDs",
            True if retrieval_backend != "google" else bool((STORE_ID or "").strip() and (VAULT_ID or "").strip()),
            "Configured for Google mode" if retrieval_backend == "google" else "Not required in Chroma mode",
        ),
        (
            "Retrieval runtime",
            True if retrieval_backend != "chroma" else bool(chroma_ready),
            "Chroma ready" if retrieval_backend == "chroma" else "Google File Search selected",
        ),
        (
            "Index coverage",
            bool(index_rows) if source_mode in ("Files only", "Files + Web") else True,
            f"{len(index_rows)} indexed documents",
        ),
        ("Active model", bool(ACTIVE_MODEL), ACTIVE_MODEL or "No active model"),
    ]
    for name, ok, detail in static_checks:
        status_txt = "OK" if ok else "Needs attention"
        st.caption(f"{status_txt}: {name} — {detail}")

    if "live_health_checks" not in st.session_state:
        st.session_state.live_health_checks = []
    if st.button("Run live health checks"):
        with st.spinner("Running checks..."):
            st.session_state.live_health_checks = run_live_health_checks(
                active_model=ACTIVE_MODEL,
                retrieval_backend=retrieval_backend,
            )

    if st.session_state.live_health_checks:
        for item in st.session_state.live_health_checks:
            status_txt = "OK" if item.get("ok") else "Fail"
            st.caption(f"{status_txt}: {item.get('name')} — {item.get('detail')}")

    st.markdown("### Knowledge Base")
    if not can_upload_files:
        st.caption("RBAC: uploads are disabled for your role.")
    cloud_upload_allowed = retrieval_backend != "google" or bool(
        st.session_state.get("cloud_index_opt_in", CLOUD_INDEX_OPT_IN)
    )
    if retrieval_backend == "google" and not cloud_upload_allowed:
        st.warning("Cloud index uploads are off. Turn on Cloud index uploads in Data controls to upload or reindex.")
    upload_project = st.selectbox("Assign project", project_options, key="upload_project")
    uf = st.file_uploader(
        "Upload a file",
        type=sorted({e.lstrip('.') for e in UPLOAD_EXTENSIONS}),
        disabled=(not cloud_upload_allowed) or (not can_upload_files),
    )
    if uf:
        upload_file(uf, project=upload_project, retrieval_backend=retrieval_backend)

    reindex_script_name = "index_files.py" if retrieval_backend == "google" else "chroma_index.py"
    reindex_available = (Path(__file__).parent / reindex_script_name).exists()
    if not reindex_available:
        st.caption(f"Reindex unavailable: `{reindex_script_name}` not found.")
    if not can_run_index:
        st.caption("RBAC: indexing is disabled for your role.")
    index_status = load_index_status()
    index_rows_count = len(index_rows)
    if index_status.get("last_run_at"):
        st.caption(f"Last index run: {index_status.get('last_run_at')}")
    elif INDEX_REPORT.exists():
        try:
            st.caption(
                "Last index run: "
                + datetime.fromtimestamp(INDEX_REPORT.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
            )
        except Exception:
            pass
    st.caption(f"Indexed documents: {index_rows_count}")
    queue_est = estimate_index_queue(DATA_ROOT, report_mtime) if DATA_ROOT else {"pending_files": 0, "total_files": 0}
    if queue_est.get("total_files", 0) > 0:
        st.caption(
            f"Index queue: pending={int(queue_est.get('pending_files', 0))} of "
            f"{int(queue_est.get('total_files', 0))} supported file(s)"
        )
    if index_status.get("last_error"):
        st.caption("Last index error:")
        st.code(str(index_status.get("last_error", ""))[:400], language="text")
    reindex_disabled = (
        (not can_run_index)
        or (not reindex_available)
        or (retrieval_backend == "google" and not cloud_upload_allowed)
    )
    if st.button("Reindex changed files only", disabled=reindex_disabled):
        with st.spinner("Indexing..."):
            code, output = run_reindex(retrieval_backend=retrieval_backend)
        st.session_state.reindex_output = output
        st.session_state.reindex_code = code
        save_index_status(code, output)

    if st.session_state.reindex_code is not None:
        if st.session_state.reindex_code == 0:
            st.success("Index complete")
        else:
            st.error("Index failed")
            out_low = (st.session_state.reindex_output or "").lower()
            if "not found" in out_low and "edith_data_root" in out_low:
                st.caption("Hint: set `EDITH_DATA_ROOT` in setup settings, then retry.")
            elif "permission" in out_low:
                st.caption("Hint: grant file access to the selected data folder and retry.")
            elif "chroma" in out_low and "install" in out_low:
                st.caption("Hint: install Chroma dependencies in your current environment.")
            elif "google_api_key" in out_low:
                st.caption("Hint: verify Google API key in desktop settings.")
        st.text_area("Indexer output", st.session_state.reindex_output, height=160)
    if INDEX_REPORT.exists() and st.button("Show index report"):
        try:
            with INDEX_REPORT.open("r", newline="") as f:
                rows = list(csv.DictReader(f))
            st.dataframe(rows[:200], use_container_width=True)
            if len(rows) > 200:
                st.caption("Showing first 200 rows.")
        except Exception:
            st.caption("Unable to read index report.")

    st.markdown("### Auto-indexing")
    if not can_manage_watcher:
        st.caption("RBAC: watcher management is disabled for your role.")
    watcher_running = False
    if WATCH_PID_PATH.exists():
        try:
            pid = int(WATCH_PID_PATH.read_text().strip())
            watcher_running = is_pid_running(pid)
        except Exception:
            watcher_running = False

    st.caption("Status: running" if watcher_running else "Status: stopped")
    col_a, col_b = st.columns(2)
    watcher_script_exists = (Path(__file__).parent / "watch_files.py").exists()
    if col_a.button("Start watcher", disabled=(not watcher_script_exists) or (not can_manage_watcher)):
        ok, msg = start_watcher(retrieval_backend=retrieval_backend)
        st.caption(msg)
    if col_b.button("Stop watcher", disabled=(not watcher_script_exists) or (not can_manage_watcher)):
        ok, msg = stop_watcher()
        st.caption(msg)
    if WATCH_LOG_PATH.exists() and st.button("Show watch log"):
        try:
            st.text_area("Watch log", WATCH_LOG_PATH.read_text(), height=160)
        except Exception:
            st.caption("Unable to read watch log.")

    st.markdown("### Import (Vault exports)")
    if not can_sync_vault:
        st.caption("RBAC: vault sync is disabled for your role.")
    vault_sync_available = (
        (Path(__file__).parent / "scripts" / "sync_vault_exports.py").exists()
        or (Path(__file__).parent / "sync_vault_exports.sh").exists()
    )
    if not vault_sync_available:
        st.caption("Vault import script not found.")
    vault_sync_disabled = (
        (not can_sync_vault)
        or (not vault_sync_available)
        or (not bool(st.session_state.get("cloud_index_opt_in", CLOUD_INDEX_OPT_IN)))
    )
    if retrieval_backend == "google" and not bool(st.session_state.get("cloud_index_opt_in", CLOUD_INDEX_OPT_IN)):
        st.caption("Cloud index uploads are off, so Vault import is paused.")
    if st.button("Sync Vault now", key="run_vault_sync", disabled=vault_sync_disabled):
        with st.spinner("Syncing Vault exports..."):
            code, output = run_vault_sync(no_index=False)
        st.session_state.vault_sync_code = code
        st.session_state.vault_sync_output = output
        if code == 0:
            st.success("Vault export import completed.")
        elif code == 2:
            st.warning("Vault export import completed with degraded checks.")
        else:
            st.error("Vault export import failed.")
        st.text_area("Vault import output", output, height=140)

    last_vault_sync = load_last_vault_sync_summary()
    totals = (last_vault_sync.get("totals") or {}) if isinstance(last_vault_sync, dict) else {}
    if totals:
        st.caption(
            "Last vault sync: "
            f"new={int(totals.get('files_new', 0))}, "
            f"updated={int(totals.get('files_updated', 0))}, "
            f"dedup={int(totals.get('files_deduped', 0))}, "
            f"failed_zips={int(totals.get('zip_failed', 0))}"
        )
        if last_vault_sync.get("degraded"):
            st.caption("Last vault sync status: degraded")
        else:
            st.caption("Last vault sync status: ok")

    with st.expander("Vault exports in folder", expanded=False):
        vault_list_script = Path(__file__).parent / "scripts" / "list_vault_docs.py"
        st.caption("Check what is already in your Google store before adding more files.")
        if not can_list_vault:
            st.caption("RBAC: vault inventory view is disabled for your role.")
        inv_col1, inv_col2 = st.columns([0.42, 0.58])
        vault_list_limit = inv_col1.number_input(
            "Max rows",
            min_value=25,
            max_value=1000,
            value=200,
            step=25,
            key="vault_list_limit",
        )
        vault_list_contains = inv_col2.text_input(
            "Filter (title/path contains)",
            value="",
            key="vault_list_contains",
            placeholder="e.g., Miller 2025, policy, methods",
        )
        if not vault_list_script.exists():
            st.caption("Vault list script not found.")
        if st.button(
            "List vault contents",
            key="run_vault_list",
            disabled=(not vault_list_script.exists()) or (not can_list_vault),
        ):
            with st.spinner("Reading vault contents..."):
                code, output = run_vault_list(limit=int(vault_list_limit), contains=vault_list_contains)
            st.session_state.vault_list_code = code
            st.session_state.vault_list_output = output

        if st.session_state.vault_list_code is not None:
            if st.session_state.vault_list_code == 0:
                rows, shown_total = parse_vault_list_output(st.session_state.vault_list_output)
                if rows:
                    st.dataframe(rows, use_container_width=True, hide_index=True)
                    if shown_total is not None:
                        st.caption(f"Showing {len(rows)} of {shown_total} listed documents.")
                    else:
                        st.caption(f"Showing {len(rows)} listed documents.")
                else:
                    st.caption("No documents matched this filter.")
            else:
                st.error("Vault listing failed.")
            st.text_area("Vault list output", st.session_state.vault_list_output, height=120)

    st.markdown("### Web cache")
    web_cache = load_web_cache()
    st.caption(f"Cached pages/snippets: {len(web_cache)}")
    if web_cache:
        latest_seen = ""
        for entry in web_cache.values():
            if not isinstance(entry, dict):
                continue
            ts = clean_text(entry.get("last_seen") or "")
            if ts and ts > latest_seen:
                latest_seen = ts
        if latest_seen:
            st.caption(f"Last web evidence seen: {latest_seen}")
    if st.button("Clear web cache", key="clear_web_cache_btn"):
        clear_web_cache()
        st.success("Web cache cleared.")
        st.rerun()

    if not simple_ui:
        with st.expander("Debug / last response metadata", expanded=False):
            meta = st.session_state.get("last_response_meta", {}) or {}
            st.json(meta)
            qv = meta.get("query_variants") or []
            if qv:
                st.caption("Rewritten retrieval queries")
                for qline in qv[:6]:
                    st.markdown(f"- `{qline}`")
            dist_q = (meta.get("query_distillation") or {}).get("query")
            if dist_q:
                st.caption(f"Distilled retrieval query: `{dist_q}`")
            gate_reason = meta.get("gate_message") or ((meta.get("support_audit") or {}).get("reason"))
            if gate_reason:
                st.caption(f"Gate/debug reason: {gate_reason}")
            last_assistant = latest_assistant_message_with_sources(st.session_state.get("msgs", []))
            if last_assistant and last_assistant.get("sources"):
                rows = []
                for s in (last_assistant.get("sources") or [])[:12]:
                    rows.append(
                        {
                            "title": s.get("title"),
                            "score": s.get("score"),
                            "vector": s.get("vector_score"),
                            "bm25": s.get("bm25_score"),
                            "rerank": s.get("rerank_score"),
                            "section": s.get("section_heading"),
                            "page": s.get("page"),
                            "file": s.get("uri"),
                        }
                    )
                if rows:
                    st.caption("Top retrieved chunks")
                    st.dataframe(rows, use_container_width=True)

simple_ui = bool(st.session_state.get("simple_ui", True))


badge = trust_badge_text(source_mode, strict_citations)
status_mode = html.escape(str(source_mode))
status_badge = html.escape(str(badge))
status_project = html.escape(str(project_filter))
status_research = "Researcher" if bool(researcher_mode) else "Standard"
scoped_doc_path = clean_text(st.session_state.get("scoped_doc_path") or "")
status_scope = html.escape(Path(scoped_doc_path).name if scoped_doc_path else "All")
status_quality = html.escape(str(st.session_state.get("source_quality", "Balanced")))
status_answer_type = html.escape(str(st.session_state.get("answer_type", "Explain")))
last_with_sources = latest_assistant_message_with_sources(st.session_state.get("msgs", []))
web_used_count = 0
if isinstance(last_with_sources, dict):
    web_used_count = sum(
        1
        for s in (last_with_sources.get("sources") or [])
        if isinstance(s, dict) and s.get("source_type") == "web"
    )
st.markdown(
    f"""
<div class="status-pill mode-pill">
  Mode: {status_mode} | {status_badge} | {status_research} | Quality: {status_quality} | Answer: {status_answer_type} | Project: {status_project} | Scope: {status_scope}
</div>
""",
    unsafe_allow_html=True,
)
if web_used_count > 0:
    st.caption(f"Web: {web_used_count} source(s) used in the last answer.")

index_status_top = load_index_status()
backend_name = "Google File Search" if retrieval_backend == "google" else "Local Chroma"
store_name = STORE_ID if retrieval_backend == "google" else CHROMA_COLLECTION
store_label = "Cloud index" if retrieval_backend == "google" else "Collection"
store_display = (
    friendly_store_display(store_name)
    if retrieval_backend == "google"
    else (clean_text(store_name) or "not set")
)
library_folder_display = DATA_ROOT or "(not set)"
queue_est_top = estimate_index_queue(DATA_ROOT, report_mtime) if DATA_ROOT else {
    "pending_files": 0,
    "total_files": 0,
    "indexed_docs": len(index_rows),
}
pending_top = int(queue_est_top.get("pending_files", 0))
total_supported_top = int(queue_est_top.get("total_files", 0))
last_index_at = clean_text(index_status_top.get("last_run_at") or "")
if not last_index_at and INDEX_REPORT.exists():
    try:
        last_index_at = datetime.fromtimestamp(INDEX_REPORT.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        last_index_at = ""
library_state = "Ready" if (len(index_rows) > 0) else "Needs indexing"
st.markdown(
    f"""
<div class="source-truth-card">
  <div class="source-truth-title">Active library</div>
  <div class="source-truth-meta">Library folder: {html.escape(str(library_folder_display))}</div>
  <div class="source-truth-meta">Backend: {html.escape(backend_name)} | {html.escape(store_label)}: {html.escape(str(store_display))}</div>
  <div class="source-truth-meta">Library: {library_state} | Indexed documents: {len(index_rows)} | Last index: {html.escape(last_index_at or 'never')}</div>
  <div class="source-truth-meta">Index queue: {pending_top} pending of {total_supported_top} supported files</div>
</div>
""",
    unsafe_allow_html=True,
)
if retrieval_backend == "google" and not bool(st.session_state.get("cloud_index_opt_in", CLOUD_INDEX_OPT_IN)):
    st.info("Cloud index uploads are off. Chat can still query existing store content, but upload/reindex actions are disabled.")
mismatch_root = detect_library_mismatch(index_status_top, DATA_ROOT)
if mismatch_root:
    st.warning("Index job is pointed at a different folder than the app.")
    st.caption(f"Indexer attempted: `{mismatch_root}`")
    if st.button("Fix folder mismatch", key="fix_folder_mismatch_btn"):
        if DESKTOP_MODE:
            st.session_state.desktop_setup_open_request = True
        else:
            st.session_state.open_connection_settings = True
        st.rerun()
lib_action_col1, lib_action_col2 = st.columns(2)
if lib_action_col1.button("Change library", key="library_change_btn"):
    if DESKTOP_MODE:
        st.session_state.desktop_setup_open_request = True
    else:
        st.session_state.open_connection_settings = True
    st.rerun()
if lib_action_col2.button("Fix library", key="library_fix_btn"):
    if DATA_ROOT and Path(DATA_ROOT).expanduser().exists():
        reindex_script = Path(__file__).parent / ("index_files.py" if retrieval_backend == "google" else "chroma_index.py")
        if reindex_script.exists():
            with st.spinner("Repairing library index..."):
                code, output = run_reindex(retrieval_backend=retrieval_backend)
            st.session_state.reindex_output = output
            st.session_state.reindex_code = code
            save_index_status(code, output)
            if code == 0:
                st.success("Library repaired and reindexed.")
            else:
                st.error("Library repair ran but indexing reported an error. Open Doctor for details.")
            st.rerun()
        else:
            st.error("Index script not found for the selected backend.")
    else:
        if DESKTOP_MODE:
            st.session_state.desktop_setup_open_request = True
        else:
            st.session_state.open_connection_settings = True
        st.warning("Library folder is not reachable. Update it in Settings.")
        st.rerun()
if retrieval_backend == "chroma":
    st.caption("Privacy: Your documents stay on-device in local Chroma mode.")
else:
    st.caption("Privacy: Indexed content is uploaded to your configured Google File Search store.")
if retrieval_backend == "google" and not client:
    st.warning("Google API is unavailable right now. Switch to local Chroma mode for offline use.")
elif retrieval_backend == "google" and chroma_ready:
    st.caption("Fallback available: local Chroma mode can be used if Google services are unavailable.")
if index_status_top.get("last_error"):
    st.warning("Index needs attention. Open Doctor for guided repair steps.")

quick_col1, quick_col2 = st.columns([0.55, 0.45])
with quick_col1:
    st.markdown(
        f"""
<div class="source-chip-row">
  <span class="source-chip">{html.escape(source_mode)}</span>
  <span class="source-chip">{'Strict citations' if strict_citations else 'Flexible citations'}</span>
  <span class="source-chip">Quality: {html.escape(str(st.session_state.get('source_quality', 'Balanced')))}</span>
  <span class="source-chip">Answer: {html.escape(str(st.session_state.get('answer_type', 'Explain')))}</span>
  <span class="source-chip">Project: {html.escape(str(project_filter))}</span>
  <span class="source-chip">Scope: {html.escape(Path(scoped_doc_path).name if scoped_doc_path else 'All')}</span>
</div>
""",
        unsafe_allow_html=True,
    )
with quick_col2:
    with st.popover("Command palette (⌘K)"):
        cmd = st.selectbox(
            "Quick action",
            [
                "Switch to Files only",
                "Switch to Files + Web",
                "Ask the Library brief",
                "Index now",
                "Open Library tab",
                "Export current chat (Markdown)",
                "Clear doc scope",
                "Open settings",
            ],
            key="cmd_palette_action",
        )
        if st.button("Run", key="cmd_palette_run"):
            if cmd == "Switch to Files only":
                st.session_state.source_mode = "Files only"
            elif cmd == "Switch to Files + Web" and ALLOW_WEB_TOOLS:
                st.session_state.source_mode = "Files + Web"
            elif cmd == "Ask the Library brief":
                st.session_state.queued_prompt = "/library Summarize the strongest schools of thought, disagreements, methods distribution, and thin-evidence gaps."
            elif cmd == "Index now":
                code, output = run_reindex(retrieval_backend=retrieval_backend)
                st.session_state.reindex_output = output
                st.session_state.reindex_code = code
                save_index_status(code, output)
            elif cmd == "Open Library tab":
                st.info("Open the Library tab above.")
            elif cmd == "Export current chat (Markdown)":
                st.info("Use the Download Markdown button in the sidebar Export section.")
            elif cmd == "Clear doc scope":
                st.session_state.scoped_doc_path = ""
            elif cmd == "Open settings":
                if DESKTOP_MODE:
                    st.session_state.desktop_setup_open_request = True
                else:
                    st.session_state.open_connection_settings = True
            st.rerun()

last_answer_for_drawer = latest_assistant_message_with_sources(st.session_state.get("msgs", []))
with st.popover("Details"):
    src_tab, ev_tab, run_tab, adv_tab = st.tabs(["Sources", "Evidence", "Run details", "Settings"])
    with src_tab:
        if last_answer_for_drawer and last_answer_for_drawer.get("sources"):
            render_sources(
                last_answer_for_drawer.get("sources") or [],
                query=last_answer_for_drawer.get("query", ""),
                audit_mode=True,
                index_map=index_map,
            )
        else:
            st.caption("No sources yet.")
    with ev_tab:
        if last_answer_for_drawer and last_answer_for_drawer.get("sources"):
            render_reasoning_panel(
                last_answer_for_drawer.get("sources") or [],
                query=last_answer_for_drawer.get("query", ""),
                index_map=index_map,
            )
            if last_answer_for_drawer.get("provenance"):
                render_sentence_provenance(
                    last_answer_for_drawer.get("provenance") or [],
                    render_key=f"drawer_{st.session_state.active_chat_id}",
                    sources=last_answer_for_drawer.get("sources") or [],
                )
        else:
            st.caption("No evidence yet.")
    with run_tab:
        st.json(st.session_state.get("last_response_meta", {}) or {})
    with adv_tab:
        st.caption("Advanced controls remain in the sidebar.")
        st.caption("Use this area when tuning retrieval or diagnostics.")

audit_mode = st.toggle(
    "Show sources and reasoning trail",
    value=st.session_state.audit_mode,
    key="audit_mode_main",
    help="Displays retrieved snippets, citations, and reasoning details.",
)
st.session_state.audit_mode = audit_mode

chat_tab, library_tab, notebook_tab, runs_tab, phd_tab, doctor_tab = st.tabs(
    ["Chat", "Library", "Notebook", "Runs", "PhD OS", "Doctor"]
)

with chat_tab:
    if source_mode in ("Files only", "Files + Web") and not index_rows:
        st.warning("No library indexed yet.")
        st.caption("Quick setup checklist")
        st.markdown("1. Confirm **Library folder** points to your files.")
        st.markdown("2. Add documents to that folder.")
        if retrieval_backend == "google" and not bool(st.session_state.get("cloud_index_opt_in", CLOUD_INDEX_OPT_IN)):
            st.markdown("3. Turn on **Cloud index uploads** in Data controls.")
            st.markdown("4. Click **Index now**.")
        else:
            st.markdown("3. Click **Index now**.")
        empty_col1, empty_col2 = st.columns(2)
        if empty_col1.button("Open settings", key="chat_empty_open_settings"):
            if DESKTOP_MODE:
                st.session_state.desktop_setup_open_request = True
            else:
                st.session_state.open_connection_settings = True
            st.rerun()
        chat_reindex_script = Path(__file__).parent / ("index_files.py" if retrieval_backend == "google" else "chroma_index.py")
        if empty_col2.button(
            "Index now",
            key="chat_index_now",
            disabled=(not chat_reindex_script.exists())
            or (retrieval_backend == "google" and not bool(st.session_state.get("cloud_index_opt_in", CLOUD_INDEX_OPT_IN))),
        ):
            with st.spinner("Indexing..."):
                code, output = run_reindex(retrieval_backend=retrieval_backend)
            st.session_state.reindex_output = output
            st.session_state.reindex_code = code
            save_index_status(code, output)
            st.rerun()
    elif source_mode in ("Files only", "Files + Web") and len(index_rows) < 20:
        st.info("Library is almost ready. Add 10–20 files for stronger retrieval quality.")
        st.caption("Next: add files -> index -> ask with citations.")

    st.markdown("#### Primary actions")
    chat_cloud_upload_allowed = retrieval_backend != "google" or bool(
        st.session_state.get("cloud_index_opt_in", CLOUD_INDEX_OPT_IN)
    )
    pa_col1, pa_col2, pa_col3, pa_col4 = st.columns(4)
    if pa_col1.button(
        "Add files",
        key="chat_primary_add_files",
        disabled=(retrieval_backend == "google" and not chat_cloud_upload_allowed),
    ):
        st.session_state.inline_upload_open = True
    reindex_script = Path(__file__).parent / ("index_files.py" if retrieval_backend == "google" else "chroma_index.py")
    if pa_col2.button(
        "Index now",
        key="chat_primary_index",
        disabled=(not reindex_script.exists()) or (retrieval_backend == "google" and not chat_cloud_upload_allowed),
    ):
        with st.spinner("Indexing..."):
            code, output = run_reindex(retrieval_backend=retrieval_backend)
        st.session_state.reindex_output = output
        st.session_state.reindex_code = code
        save_index_status(code, output)
        st.rerun()
    if pa_col3.button("Open library", key="chat_primary_open_library"):
        st.caption("Open the Library tab above.")
    watch_now = watcher_status()
    watcher_label = "Auto-indexing: On" if watch_now.get("running") else "Auto-indexing: Off"
    if pa_col4.button(watcher_label, key="chat_primary_toggle_watcher"):
        if watch_now.get("running"):
            ok, msg = stop_watcher()
        else:
            ok, msg = start_watcher(retrieval_backend=retrieval_backend)
        if ok:
            st.success(msg)
        else:
            st.error(msg)
        st.rerun()
    if st.session_state.get("inline_upload_open"):
        up_col1, up_col2 = st.columns([0.35, 0.65])
        inline_project = up_col1.selectbox("Project", project_options, key="inline_upload_project")
        inline_upload = up_col2.file_uploader(
            "Add file to library",
            type=sorted({e.lstrip('.') for e in UPLOAD_EXTENSIONS}),
            key="inline_upload_file",
            disabled=(retrieval_backend == "google" and not chat_cloud_upload_allowed),
        )
        if inline_upload:
            upload_file(inline_upload, project=inline_project, retrieval_backend=retrieval_backend)
    if retrieval_backend == "google" and not chat_cloud_upload_allowed:
        st.caption("Cloud index uploads are off. Enable them in Data controls if you want remote indexing.")

    mode_row1, mode_row2 = st.columns(2)
    st.session_state.source_quality = mode_row1.select_slider(
        "Source quality",
        options=SOURCE_QUALITY_OPTIONS,
        value=st.session_state.get("source_quality", "Balanced"),
        key="source_quality_selector",
        help="Precise uses fewer sources, Balanced is default, Exhaustive expands retrieval for synthesis.",
    )
    st.session_state.answer_type = mode_row2.selectbox(
        "Answer type",
        options=ANSWER_TYPE_OPTIONS,
        index=ANSWER_TYPE_OPTIONS.index(st.session_state.get("answer_type", "Explain"))
        if st.session_state.get("answer_type", "Explain") in ANSWER_TYPE_OPTIONS
        else 0,
        key="answer_type_selector",
        help="Controls how the response is shaped for this question.",
    )

    with st.expander("Example prompts", expanded=(not st.session_state.msgs)):
        active_examples = prompt_examples_for_preset(mode_preset)
        cols = st.columns(2)
        for i, text in enumerate(active_examples):
            col = cols[i % 2]
            if col.button(text, key=f"qp_demo_{i}"):
                st.session_state.queued_prompt = text
                st.rerun()
        mode_actions = action_prompts_for_preset(mode_preset)
        if mode_actions:
            st.caption("Mode quick actions")
            action_cols = st.columns(4)
            for i, (label, prompt_text) in enumerate(mode_actions):
                col = action_cols[i % 4]
                if col.button(label, key=f"mode_action_{i}"):
                    st.session_state.queued_prompt = prompt_text
                    st.rerun()

    for idx, m in enumerate(st.session_state.msgs):
        with st.chat_message(m["role"]):
            if m.get("role") == "assistant":
                ans_body, ans_cites = split_answer_and_citations(m.get("text", ""))
                with st.expander("Answer", expanded=True):
                    st.markdown(ans_body or m.get("text", ""))
                if ans_cites:
                    with st.expander("Answer citations", expanded=False):
                        st.markdown(ans_cites)
                assistant_query = find_assistant_query(st.session_state.msgs, idx)
                with st.expander("Actions", expanded=False):
                    action_key = f"{st.session_state.active_chat_id}_{idx}"
                    col_copy, col_regen, col_continue, col_open_sources, col_explain, col_short, col_detail, col_pdf, col_pack = st.columns(9)
                    with col_copy:
                        copy_button(ans_body or m.get("text", ""))
                    with col_regen:
                        if st.button("Regenerate", key=f"msg_regen_{action_key}", use_container_width=True):
                            if assistant_query:
                                st.session_state.queued_prompt = assistant_query
                                st.rerun()
                    with col_continue:
                        if st.button("Continue", key=f"msg_continue_{action_key}", use_container_width=True):
                            st.session_state.queued_prompt = "Continue your previous answer with additional grounded details and citations."
                            st.rerun()
                    with col_open_sources:
                        if st.button("Open sources", key=f"msg_open_sources_{action_key}", use_container_width=True):
                            st.session_state.audit_mode = True
                            st.rerun()
                    with col_explain:
                        if st.button("Explain citations", key=f"msg_explain_{action_key}", use_container_width=True):
                            if assistant_query:
                                st.session_state.queued_prompt = (
                                    f"Explain the citations in your previous answer to: {assistant_query}. "
                                    "Identify the strongest evidence for each citation."
                                )
                            else:
                                st.session_state.queued_prompt = (
                                    "Explain the citations in your previous answer and identify the strongest evidence."
                                )
                            st.rerun()
                    with col_short:
                        if st.button("Make shorter", key=f"msg_short_{action_key}", use_container_width=True):
                            st.session_state.queued_prompt = (
                                "Rewrite your previous answer in a shorter form while preserving citation labels."
                            )
                            st.rerun()
                    with col_detail:
                        if st.button("More detail", key=f"msg_detail_{action_key}", use_container_width=True):
                            st.session_state.queued_prompt = (
                                "Rewrite your previous answer with more detail while preserving citation labels."
                            )
                            st.rerun()
                    with col_pdf:
                        message_pdf = chat_to_pdf_bytes(
                            {
                                "id": m.get("run_id", f"message_{idx}"),
                                "title": "Edith Message Export",
                                "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "messages": [
                                    {"role": "user", "text": assistant_query or "Prompt unavailable"},
                                    {"role": "assistant", "text": m.get("text", "")},
                                ],
                            },
                            redact_sensitive=bool(st.session_state.get("export_redact_sensitive", EXPORT_REDACT_DEFAULT)),
                        )
                        if message_pdf:
                            st.download_button(
                                "Export PDF",
                                data=message_pdf,
                                file_name=f"{m.get('run_id', f'message_{idx}')}.pdf",
                                mime="application/pdf",
                                key=f"msg_pdf_{action_key}",
                                use_container_width=True,
                            )
                    with col_pack:
                        pack_md = build_answer_pack_markdown(m, assistant_query, index_map=index_map)
                        if pack_md:
                            st.download_button(
                                "Answer pack",
                                data=pack_md,
                                file_name=f"{m.get('run_id', f'message_{idx}')}_pack.md",
                                mime="text/markdown",
                                key=f"msg_pack_{action_key}",
                                use_container_width=True,
                            )
                    st.caption("Notebook")
                    nb1, nb2, nb3 = st.columns(3)
                    if nb1.button("Save summary", key=f"note_summary_{action_key}", use_container_width=True):
                        summary_text = clean_text(answer_body or m.get("text") or "")
                        if len(summary_text) > 1000:
                            summary_text = summary_text[:1000].rstrip() + "..."
                        res = save_notebook_entry(
                            kind="summary",
                            text=summary_text,
                            query=assistant_query,
                            sources=m.get("sources") or [],
                            project=project_filter,
                            run_id=m.get("run_id") or "",
                        )
                        if res.get("ok"):
                            p = clean_text(res.get("path") or "")
                            st.success("Summary saved." + (f" Note file: {p}" if p else ""))
                        else:
                            st.error(res.get("error") or "Failed to save summary.")
                    if nb2.button("Save claim", key=f"note_claim_{action_key}", use_container_width=True):
                        claim_text = clean_text(answer_body or m.get("text") or "")
                        claim_line = ""
                        for part in re.split(r"[\n\.]+", claim_text):
                            cand = clean_text(part)
                            if len(cand) >= 24:
                                claim_line = cand
                                break
                        claim_line = claim_line or claim_text[:240]
                        res = save_notebook_entry(
                            kind="claim",
                            text=claim_line,
                            query=assistant_query,
                            sources=m.get("sources") or [],
                            project=project_filter,
                            run_id=m.get("run_id") or "",
                        )
                        if res.get("ok"):
                            st.success("Claim saved.")
                        else:
                            st.error(res.get("error") or "Failed to save claim.")
                    if nb3.button("Save quote", key=f"note_quote_{action_key}", use_container_width=True):
                        quote_text = ""
                        for s in (m.get("sources") or []):
                            if isinstance(s, dict) and clean_text(s.get("snippet") or ""):
                                quote_text = clean_text(s.get("snippet") or "")
                                break
                        if not quote_text:
                            quote_text = clean_text(answer_body or m.get("text") or "")
                        if len(quote_text) > 500:
                            quote_text = quote_text[:500].rstrip() + "..."
                        res = save_notebook_entry(
                            kind="quote",
                            text=quote_text,
                            query=assistant_query,
                            sources=m.get("sources") or [],
                            project=project_filter,
                            run_id=m.get("run_id") or "",
                        )
                        if res.get("ok"):
                            st.success("Quote saved.")
                        else:
                            st.error(res.get("error") or "Failed to save quote.")

                    st.caption("Create artifact")
                    art1, art2, art3, art4, art5 = st.columns(5)
                    if art1.button("Lit review outline", key=f"art_outline_{action_key}", use_container_width=True):
                        st.session_state.queued_prompt = (
                            "Using your previous answer and its sources, draft a literature review outline with citations."
                        )
                        st.rerun()
                    if art2.button("Annotated bibliography", key=f"art_bib_{action_key}", use_container_width=True):
                        st.session_state.queued_prompt = (
                            "Create an annotated bibliography from your previous answer's sources. "
                            "For each source include 1-2 sentence annotation and citation."
                        )
                        st.rerun()
                    if art3.button("Slide bullets", key=f"art_slides_{action_key}", use_container_width=True):
                        st.session_state.queued_prompt = (
                            "Convert your previous answer into slide-ready bullets with citations and one limitations slide."
                        )
                        st.rerun()
                    if art4.button("Research memo", key=f"art_memo_{action_key}", use_container_width=True):
                        st.session_state.queued_prompt = (
                            "Write a research memo from your previous answer with sections: summary, evidence, disagreements, "
                            "limitations, and next steps. Keep citations."
                        )
                        st.rerun()
                    if art5.button("Ask-to-Write", key=f"art_write_{action_key}", use_container_width=True):
                        st.session_state.queued_prompt = (
                            "Draft a thesis-style paragraph from your previous answer using only supported evidence. "
                            "Then run a strict support audit, remove unsupported claims, and preserve citation labels."
                        )
                        st.rerun()
                    if st.button("Methods table CSV", key=f"art_methods_{action_key}", use_container_width=True):
                        if enforce_rate_limit(
                            "methods_extract",
                            RATE_LIMIT_MUTATION_MAX,
                            RATE_LIMIT_MUTATION_WINDOW_SECONDS,
                            "Methods extraction",
                        ):
                            rows, meta = generate_methods_data_rows(
                                question=assistant_query or "Extract methods and data from the previous answer.",
                                sources=m.get("sources") or [],
                                model_chain=model_chain,
                                max_rows=12,
                                index_map=index_map,
                            )
                            mt = st.session_state.get("methods_extract", {})
                            mt[action_key] = {"rows": rows, "meta": meta}
                            st.session_state.methods_extract = mt
                            if meta.get("ok"):
                                st.success(f"Extracted {int(meta.get('count', len(rows)))} methods rows.")
                            else:
                                st.error(meta.get("error") or "Methods extraction failed.")
                    methods_state = (st.session_state.get("methods_extract") or {}).get(action_key, {})
                    methods_rows = methods_state.get("rows") if isinstance(methods_state, dict) else []
                    if methods_rows:
                        st.dataframe(methods_rows, use_container_width=True, hide_index=True)
                        methods_csv = methods_rows_to_csv(methods_rows)
                        st.download_button(
                            "Download methods table CSV",
                            data=methods_csv,
                            file_name=f"{m.get('run_id', action_key)}_methods.csv",
                            mime="text/csv",
                            key=f"methods_csv_{action_key}",
                            use_container_width=True,
                        )
                    if m.get("approval_required"):
                        if st.button("Approve and generate commands", key=f"msg_approve_{action_key}", use_container_width=True):
                            if assistant_query:
                                st.session_state.queued_prompt = (
                                    "APPROVE ACTION. Generate executable commands/scripts for this prior request:\n"
                                    f"{assistant_query}\n"
                                    "Use cited evidence where relevant."
                                )
                            else:
                                st.session_state.queued_prompt = "APPROVE ACTION. Generate executable commands/scripts."
                            st.rerun()
                    st.caption("Feedback")
                    fb_col1, fb_col2, fb_col3, fb_col4 = st.columns(4)
                    with fb_col1:
                        if st.button("Good answer", key=f"fb_answer_good_{action_key}", use_container_width=True):
                            ok, msg_text = record_feedback_event(m, feedback_type="answer", value=1)
                            st.session_state.feedback_summary = load_feedback_summary()
                            if ok:
                                st.success(msg_text)
                            else:
                                st.error(msg_text)
                    with fb_col2:
                        if st.button("Bad answer", key=f"fb_answer_bad_{action_key}", use_container_width=True):
                            ok, msg_text = record_feedback_event(m, feedback_type="answer", value=-1)
                            st.session_state.feedback_summary = load_feedback_summary()
                            if ok:
                                st.success(msg_text)
                            else:
                                st.error(msg_text)
                    with fb_col3:
                        if st.button("Good sources", key=f"fb_sources_good_{action_key}", use_container_width=True):
                            ok, msg_text = record_feedback_event(m, feedback_type="sources", value=1)
                            st.session_state.feedback_summary = load_feedback_summary()
                            if ok:
                                st.success(msg_text)
                            else:
                                st.error(msg_text)
                    with fb_col4:
                        if st.button("Wrong sources", key=f"fb_sources_bad_{action_key}", use_container_width=True):
                            ok, msg_text = record_feedback_event(m, feedback_type="sources", value=-1)
                            st.session_state.feedback_summary = load_feedback_summary()
                            if ok:
                                st.success(msg_text)
                            else:
                                st.error(msg_text)
                    if st.button("Should have refused", key=f"fb_should_refuse_{action_key}", use_container_width=True):
                        ok, msg_text = record_feedback_event(m, feedback_type="should_refuse", value=-1)
                        st.session_state.feedback_summary = load_feedback_summary()
                        if ok:
                            st.success(msg_text)
                        else:
                            st.error(msg_text)
                    if st.button("Bad citation", key=f"fb_bad_citation_{action_key}", use_container_width=True):
                        ok, msg_text = record_feedback_event(m, feedback_type="bad_citation", value=-1)
                        st.session_state.feedback_summary = load_feedback_summary()
                        if ok:
                            st.success(msg_text)
                        else:
                            st.error(msg_text)
                    missing_note = st.text_input(
                        "Missing source note (optional)",
                        key=f"fb_missing_note_{action_key}",
                        value="",
                        placeholder="e.g., 2024 county SNAP report",
                    )
                    if st.button("Save missing source note", key=f"fb_missing_save_{action_key}"):
                        ok, msg_text = record_feedback_event(
                            m,
                            feedback_type="missing_source",
                            value=-1,
                            note=missing_note,
                        )
                        st.session_state.feedback_summary = load_feedback_summary()
                        if ok:
                            st.success(msg_text)
                        else:
                            st.error(msg_text)
            else:
                st.markdown(m["text"])
            if (not simple_ui) and m.get("model"):
                st.caption(f"Used: {friendly_model_tier(m['model'])} ({m['model']})")
            if (not simple_ui) and m.get("run_id"):
                st.caption(f"Run ID: {m['run_id']}")
            if (not simple_ui) and m.get("replay_compare"):
                cmp = m["replay_compare"]
                st.caption(
                    "Replay delta: "
                    f"source overlap {cmp.get('source_overlap', 0.0):.2f}, "
                    f"answer changed={cmp.get('answer_changed')}"
                )
            if m.get("stoplight"):
                render_stoplight_badge(m.get("stoplight") or {})
                reason = (m.get("stoplight") or {}).get("reason")
                if reason:
                    st.caption(str(reason))
            if m.get("sources"):
                render_source_chips(m["sources"], index_map=index_map, max_chips=8)
                render_reasoning_panel(
                    m["sources"],
                    query=m.get("query", ""),
                    index_map=index_map,
                )
                render_web_usage_summary(m["sources"])
                render_sources(
                    m["sources"],
                    query=m.get("query", ""),
                    audit_mode=audit_mode,
                    index_map=index_map,
                )
            next_qs = m.get("next_questions") or []
            if next_qs:
                st.caption("Next questions")
                qcols = st.columns(len(next_qs))
                for nq_i, nq in enumerate(next_qs):
                    if qcols[nq_i].button(str(nq), key=f"nextq_{stable_hash(str(m.get('run_id')) + str(nq_i))[:10]}"):
                        st.session_state.queued_prompt = str(nq)
                        st.rerun()
            msg_clusters = m.get("theme_clusters") or []
            if msg_clusters:
                with st.expander("Theme clusters", expanded=False):
                    for cluster in msg_clusters:
                        if not isinstance(cluster, dict):
                            continue
                        st.markdown(
                            f"- **{clean_text(cluster.get('label') or 'general')}** "
                            f"(hits={int(cluster.get('count') or 0)}, "
                            f"unique_sources={int(cluster.get('unique_sources') or 0)})"
                        )
            if (m.get("conflict_report") or {}).get("conflicts"):
                with st.expander("Conflict report", expanded=False):
                    for c in (m.get("conflict_report") or {}).get("conflicts", []):
                        claim = html.escape(str(c.get("claim") or ""))
                        reason = html.escape(str(c.get("reason") or ""))
                        sa = html.escape(str(c.get("source_a") or ""))
                        sb = html.escape(str(c.get("source_b") or ""))
                        st.markdown(f"- **{claim}**")
                        if sa or sb:
                            st.caption(f"Sources: {sa} vs {sb}")
                        if reason:
                            st.caption(reason)
            if m.get("provenance"):
                render_sentence_provenance(
                    m.get("provenance") or [],
                    render_key=f"{st.session_state.active_chat_id}_{idx}",
                    sources=m.get("sources") or [],
                )

    composer_cols = st.columns(5)
    if composer_cols[0].button("Attach", key="composer_attach_btn"):
        st.session_state.inline_upload_open = True
        st.rerun()
    if composer_cols[1].button("Regenerate", key="composer_regen_btn"):
        last_user = ""
        for msg in reversed(st.session_state.msgs):
            if msg.get("role") == "user":
                last_user = clean_text(msg.get("text") or "")
                break
        if last_user:
            st.session_state.queued_prompt = last_user
            st.rerun()
    if composer_cols[2].button("Continue", key="composer_continue_btn"):
        st.session_state.queued_prompt = "Continue your previous answer with additional grounded details and citations."
        st.rerun()
    if composer_cols[3].button("Stop", key="composer_stop_btn"):
        st.session_state.stop_generation = True
        st.info("Stopping current generation...")
    if composer_cols[4].button("Clear scope", key="composer_clear_scope_btn"):
        st.session_state.scoped_doc_path = ""
        st.rerun()

    if not can_chat:
        st.caption("RBAC: chat is read-only for your role.")
    q = st.chat_input("Ask a question about your files", disabled=not can_chat)
    if not q and st.session_state.queued_prompt:
        q = st.session_state.queued_prompt
        st.session_state.queued_prompt = None
    if q:
        raw_user_input = str(q).strip()
        q, library_command_mode = strip_library_command(raw_user_input)
        if library_command_mode and not q:
            q = "What are the strongest themes and evidence clusters in my library?"
        display_user_text = (
            f"Ask the library: {q}" if library_command_mode else raw_user_input
        )
        if len(q) > MAX_QUERY_CHARS:
            st.error(f"Query too long ({len(q)} chars). Limit is {MAX_QUERY_CHARS} chars.")
            st.stop()
        if not can_chat:
            st.error("Your role is not allowed to submit prompts.")
            st.stop()
        if not enforce_rate_limit(
            action="chat_prompt",
            limit_count=RATE_LIMIT_CHAT_MAX,
            window_seconds=RATE_LIMIT_CHAT_WINDOW_SECONDS,
            label="Chat requests",
        ):
            st.stop()
        st.session_state.stop_generation = False
        replay_payload = st.session_state.replay_payload if isinstance(st.session_state.replay_payload, dict) else None
        replay_active = bool(replay_payload and (replay_payload.get("query", "").strip() == q.strip()))
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        run_started = time.time()

        source_mode_run = source_mode
        hybrid_policy_run = hybrid_policy
        strict_citations_run = bool(strict_citations)
        retrieval_backend_run = retrieval_backend
        project_filter_run = project_filter
        tag_filter_run = tag_filter
        section_filter_run = section_filter
        doc_type_filter_run = doc_type_filter
        query_rewrite_on_run = query_rewrite_on
        support_audit_on_run = support_audit_on
        confidence_routing_on_run = confidence_routing_on
        multi_pass_on_run = multi_pass_on
        recursive_controller_on_run = recursive_controller_on
        contradiction_check_on_run = contradiction_check_on
        sentence_provenance_on_run = sentence_provenance_on
        strict_sentence_tags_on_run = strict_sentence_tags_on
        chroma_rerank_on_run = bool(st.session_state.chroma_rerank_on)
        chroma_top_k_run = int(st.session_state.get("chroma_top_k", CHROMA_TOP_K))
        chroma_rerank_top_n_run = int(st.session_state.get("chroma_rerank_top_n", CHROMA_RERANK_TOP_N))
        chroma_diversity_lambda_run = float(
            st.session_state.get("chroma_diversity_lambda", CHROMA_DIVERSITY_LAMBDA)
        )
        chroma_bm25_weight_run = float(st.session_state.get("chroma_bm25_weight", CHROMA_BM25_WEIGHT))
        model_chain_run = list(model_chain)
        stream_on_run = streaming
        context_packing_on_run = bool(context_packing_on)
        distill_query_on_run = bool(distill_query_on)
        next_questions_on_run = bool(next_questions_on)
        researcher_mode_run = bool(researcher_mode_on)
        mode_preset_run = mode_preset if mode_preset in MODE_PRESETS else "Custom"
        scoped_doc_path_run = clean_text(st.session_state.get("scoped_doc_path") or "")
        verbosity_level_run = verbosity_level_on if verbosity_level_on in {"concise", "standard", "deep"} else "standard"
        writing_style_run = writing_style_on if writing_style_on in {"academic", "plain"} else "academic"
        include_methods_table_run = bool(include_methods_table_on)
        include_limitations_run = bool(include_limitations_on)
        action_approval_on_run = bool(action_approval_on)
        action_outputs_enabled_run = bool(action_outputs_enabled)
        source_quality_run = (
            st.session_state.get("source_quality", "Balanced")
            if st.session_state.get("source_quality", "Balanced") in SOURCE_QUALITY_OPTIONS
            else "Balanced"
        )
        answer_type_run = (
            st.session_state.get("answer_type", "Explain")
            if st.session_state.get("answer_type", "Explain") in ANSWER_TYPE_OPTIONS
            else "Explain"
        )

        if replay_active:
            replay_settings = replay_payload.get("settings", {}) or {}
            source_mode_run = choose_replay_value(
                replay_settings.get("source_mode"), SOURCE_MODES, source_mode
            )
            hybrid_policy_run = choose_replay_value(
                replay_settings.get("hybrid_policy"), HYBRID_FILE_POLICIES, hybrid_policy
            )
            retrieval_backend_run = choose_replay_value(
                replay_settings.get("retrieval_backend"), ("google", "chroma"), retrieval_backend
            )
            if retrieval_backend_run == "chroma" and not chroma_ready:
                retrieval_backend_run = retrieval_backend
            project_filter_run = replay_settings.get("project_filter", project_filter) or project_filter
            tag_filter_run = replay_settings.get("tag_filter", tag_filter) or tag_filter
            section_filter_run = replay_settings.get("section_filter", section_filter) or section_filter
            doc_type_filter_run = replay_settings.get("doc_type_filter", doc_type_filter) or doc_type_filter
            st.session_state.section_filter = section_filter_run
            st.session_state.doc_type_filter = doc_type_filter_run
            query_rewrite_on_run = bool(replay_settings.get("query_rewrite_on", query_rewrite_on))
            support_audit_on_run = bool(replay_settings.get("support_audit_on", support_audit_on))
            confidence_routing_on_run = bool(replay_settings.get("confidence_routing_on", confidence_routing_on))
            multi_pass_on_run = bool(replay_settings.get("multi_pass", multi_pass_on))
            recursive_controller_on_run = bool(
                replay_settings.get("recursive_controller_on", recursive_controller_on)
            )
            contradiction_check_on_run = bool(
                replay_settings.get("contradiction_check", contradiction_check_on)
            )
            sentence_provenance_on_run = bool(replay_settings.get("sentence_provenance_on", sentence_provenance_on))
            strict_sentence_tags_on_run = bool(replay_settings.get("strict_sentence_tags_on", strict_sentence_tags_on))
            chroma_rerank_on_run = bool(replay_settings.get("chroma_rerank_on", chroma_rerank_on_run))
            chroma_top_k_run = int(replay_settings.get("chroma_top_k", chroma_top_k_run))
            chroma_rerank_top_n_run = int(replay_settings.get("chroma_rerank_top_n", chroma_rerank_top_n_run))
            context_packing_on_run = bool(replay_settings.get("context_packing_on", context_packing_on_run))
            distill_query_on_run = bool(replay_settings.get("distill_query_on", distill_query_on_run))
            next_questions_on_run = bool(replay_settings.get("next_questions_on", next_questions_on_run))
            researcher_mode_run = bool(replay_settings.get("researcher_mode", researcher_mode_run))
            scoped_doc_path_run = clean_text(replay_settings.get("scoped_doc_path", scoped_doc_path_run))
            verbosity_level_run = clean_text(replay_settings.get("verbosity_level", verbosity_level_run)).lower()
            writing_style_run = clean_text(replay_settings.get("writing_style", writing_style_run)).lower()
            include_methods_table_run = bool(
                replay_settings.get("include_methods_table", include_methods_table_run)
            )
            include_limitations_run = bool(
                replay_settings.get("include_limitations", include_limitations_run)
            )
            action_approval_on_run = bool(replay_settings.get("action_approval_on", action_approval_on_run))
            action_outputs_enabled_run = bool(
                replay_settings.get("action_outputs_enabled_on", action_outputs_enabled_run)
            )
            replay_mode = clean_text(replay_settings.get("mode_preset", mode_preset_run))
            if replay_mode in MODE_PRESETS:
                mode_preset_run = replay_mode
            replay_quality = clean_text(replay_settings.get("source_quality", source_quality_run)).title()
            if replay_quality in SOURCE_QUALITY_OPTIONS:
                source_quality_run = replay_quality
            replay_answer_type = clean_text(replay_settings.get("answer_type", answer_type_run)).title()
            if replay_answer_type in ANSWER_TYPE_OPTIONS:
                answer_type_run = replay_answer_type
            st.session_state.chroma_diversity_lambda = float(
                replay_settings.get("chroma_diversity_lambda", st.session_state.get("chroma_diversity_lambda", CHROMA_DIVERSITY_LAMBDA))
            )
            st.session_state.chroma_bm25_weight = float(
                replay_settings.get("chroma_bm25_weight", st.session_state.get("chroma_bm25_weight", CHROMA_BM25_WEIGHT))
            )
            chroma_diversity_lambda_run = float(
                replay_settings.get("chroma_diversity_lambda", chroma_diversity_lambda_run)
            )
            chroma_bm25_weight_run = float(
                replay_settings.get("chroma_bm25_weight", chroma_bm25_weight_run)
            )
            st.session_state.chroma_top_k = clamp_int(chroma_top_k_run, 4, 20)
            st.session_state.chroma_rerank_top_n = clamp_int(chroma_rerank_top_n_run, 6, 40)
            strict_citations_run = bool(replay_settings.get("require_citations", strict_citations_run))
            stream_on_run = bool(replay_settings.get("streaming", streaming))
            rec_chain = replay_payload.get("model_chain") or []
            if isinstance(rec_chain, list) and rec_chain:
                model_chain_run = [normalize_model_name(x) for x in rec_chain if normalize_model_name(x)]
            if not model_chain_run:
                model_chain_run = list(model_chain)
            st.caption(f"Replaying run {replay_payload.get('run_id', '')} with saved settings.")

        chroma_top_k_run = clamp_int(chroma_top_k_run, 4, 20)
        chroma_rerank_top_n_run = clamp_int(chroma_rerank_top_n_run, 6, 40)
        if chroma_rerank_top_n_run < chroma_top_k_run:
            chroma_rerank_top_n_run = chroma_top_k_run
        if source_mode_run == "Files only":
            strict_citations_run = True
            support_audit_on_run = True
        if (
            CHROMA_FORCE_RERANK_FILES_ONLY
            and retrieval_backend_run == "chroma"
            and source_mode_run == "Files only"
        ):
            chroma_rerank_on_run = True
        if source_mode_run == "Files only" and strict_citations_run:
            sentence_provenance_on_run = True
            strict_sentence_tags_on_run = True
        if verbosity_level_run not in {"concise", "standard", "deep"}:
            verbosity_level_run = "standard"
        if writing_style_run not in {"academic", "plain"}:
            writing_style_run = "academic"
        if researcher_mode_run:
            query_rewrite_on_run = True
            support_audit_on_run = True
            confidence_routing_on_run = True
            multi_pass_on_run = False
            recursive_controller_on_run = True
            chroma_rerank_on_run = True
            chroma_top_k_run = max(chroma_top_k_run, 14)
            chroma_rerank_top_n_run = max(chroma_rerank_top_n_run, 26)
        if library_command_mode:
            researcher_mode_run = True
            query_rewrite_on_run = True
            support_audit_on_run = True
            recursive_controller_on_run = True
            source_quality_run = "Exhaustive"
            answer_type_run = "Compare"
            chroma_rerank_on_run = True
            chroma_top_k_run = max(chroma_top_k_run, 16)
            chroma_rerank_top_n_run = max(chroma_rerank_top_n_run, min(40, chroma_top_k_run + 10))
            chroma_diversity_lambda_run = min(0.95, max(chroma_diversity_lambda_run, 0.78))
        if source_quality_run == "Precise":
            chroma_rerank_on_run = True
            chroma_top_k_run = min(chroma_top_k_run, 10)
            chroma_rerank_top_n_run = min(24, max(chroma_top_k_run + 4, 10))
            chroma_diversity_lambda_run = min(chroma_diversity_lambda_run, 0.72)
            chroma_bm25_weight_run = min(0.7, max(chroma_bm25_weight_run, 0.35))
        elif source_quality_run == "Exhaustive":
            chroma_rerank_on_run = True
            chroma_top_k_run = max(chroma_top_k_run, 16)
            chroma_rerank_top_n_run = max(chroma_rerank_top_n_run, min(40, chroma_top_k_run + 10))
            chroma_diversity_lambda_run = min(0.95, max(chroma_diversity_lambda_run, 0.78))
            chroma_bm25_weight_run = max(0.2, min(chroma_bm25_weight_run, 0.5))

        query_intent = classify_query_intent(q)
        intent_cfg = intent_retrieval_overrides(query_intent)
        overview_mode_run = bool(intent_cfg.get("overview_mode"))
        if intent_cfg.get("top_k_delta"):
            chroma_top_k_run = clamp_int(chroma_top_k_run + int(intent_cfg.get("top_k_delta", 0)), 4, 20)
        if intent_cfg.get("rerank_top_n_delta"):
            chroma_rerank_top_n_run = clamp_int(
                chroma_rerank_top_n_run + int(intent_cfg.get("rerank_top_n_delta", 0)),
                6,
                40,
            )
        if chroma_rerank_top_n_run < chroma_top_k_run:
            chroma_rerank_top_n_run = chroma_top_k_run
        if overview_mode_run:
            query_rewrite_on_run = True
            chroma_top_k_run = max(chroma_top_k_run, 12)
            chroma_rerank_top_n_run = max(chroma_rerank_top_n_run, min(40, chroma_top_k_run + 8))
            chroma_diversity_lambda_run = min(0.95, max(chroma_diversity_lambda_run, 0.72))
        if researcher_mode_run:
            chroma_diversity_lambda_run = min(0.95, max(chroma_diversity_lambda_run, 0.78))
            chroma_bm25_weight_run = min(0.6, max(chroma_bm25_weight_run, 0.32))
            chroma_top_k_run = max(chroma_top_k_run, 16)
            chroma_rerank_top_n_run = max(chroma_rerank_top_n_run, min(40, chroma_top_k_run + 10))
        intent_doc_filter = str(intent_cfg.get("doc_type_filter") or "").strip()
        if intent_doc_filter:
            merged_doc_filters = parse_csv_tokens(",".join([doc_type_filter_run, intent_doc_filter]))
            doc_type_filter_run = ",".join(merged_doc_filters)

        active_model_run = model_chain_run[0] if model_chain_run else ""
        production_template_run = bool(
            PRODUCTION_TEMPLATE_DEFAULT and (is_production_query(q) or bool(intent_cfg.get("force_production_template")))
        )
        quote_first_run = bool(
            QUOTE_FIRST_RECALL_DEFAULT and (is_quote_first_query(q) or bool(intent_cfg.get("quote_first")))
        )
        approved_action_phrase = "approve action" in (q or "").lower()
        action_request_detected = is_action_request(q)
        approval_bypass_blocked = bool(
            approved_action_phrase and action_request_detected and (not action_outputs_enabled_run)
        )
        action_approval_run = bool(
            action_approval_on_run
            and action_request_detected
            and not (approved_action_phrase and action_outputs_enabled_run)
        )
        require_equations_run = bool(intent_cfg.get("require_equations"))
        multi_pass_run = bool(multi_pass_on_run and source_mode_run in ("Files only", "Files + Web"))
        recursive_controller_run = bool(
            recursive_controller_on_run
            and source_mode_run in ("Files only", "Files + Web")
            and (
                library_command_mode
                or researcher_mode_run
                or query_intent in {"overview", "general", "compare"}
            )
        )
        contradiction_check_run = bool(contradiction_check_on_run and source_mode_run in ("Files only", "Files + Web"))

        run_settings = {
            "mode_preset": mode_preset_run,
            "source_quality": source_quality_run,
            "answer_type": answer_type_run,
            "library_command_mode": bool(library_command_mode),
            "query_intent": query_intent,
            "source_mode": source_mode_run,
            "hybrid_policy": hybrid_policy_run,
            "researcher_mode": bool(researcher_mode_run),
            "verbosity_level": verbosity_level_run,
            "writing_style": writing_style_run,
            "include_methods_table": bool(include_methods_table_run),
            "include_limitations": bool(include_limitations_run),
            "scoped_doc_path": scoped_doc_path_run,
            "retrieval_backend": retrieval_backend_run,
            "project_filter": project_filter_run,
            "tag_filter": tag_filter_run,
            "section_filter": section_filter_run,
            "doc_type_filter": doc_type_filter_run,
            "query_rewrite_on": bool(query_rewrite_on_run),
            "overview_mode": bool(overview_mode_run),
            "support_audit_on": bool(support_audit_on_run),
            "confidence_routing_on": bool(confidence_routing_on_run),
            "sentence_provenance_on": bool(sentence_provenance_on_run),
            "strict_sentence_tags_on": bool(strict_sentence_tags_on_run),
            "chroma_rerank_on": bool(chroma_rerank_on_run),
            "chroma_top_k": int(chroma_top_k_run),
            "chroma_rerank_top_n": int(chroma_rerank_top_n_run),
            "chroma_diversity_lambda": float(chroma_diversity_lambda_run),
            "chroma_bm25_weight": float(chroma_bm25_weight_run),
            "context_packing_on": bool(context_packing_on_run),
            "distill_query_on": bool(distill_query_on_run),
            "next_questions_on": bool(next_questions_on_run),
            "action_approval_on": bool(action_approval_on_run),
            "action_outputs_enabled_on": bool(action_outputs_enabled_run),
            "approval_bypass_blocked": bool(approval_bypass_blocked),
            "production_template": bool(production_template_run),
            "quote_first_recall": bool(quote_first_run),
            "action_approval_active": bool(action_approval_run),
            "multi_pass": bool(multi_pass_run),
            "recursive_controller_on": bool(recursive_controller_run),
            "contradiction_check": bool(contradiction_check_run),
            "require_equations": bool(require_equations_run),
            "streaming": bool(stream_on_run),
            "require_citations": bool(strict_citations_run),
            "cloud_index_opt_in": bool(st.session_state.get("cloud_index_opt_in", CLOUD_INDEX_OPT_IN)),
            "web_domain_allowlist_enabled": bool(
                st.session_state.get("web_domain_allowlist_enabled", WEB_DOMAIN_ALLOWLIST_ENABLED_DEFAULT)
            ),
            "web_domain_allowlist": clean_text(
                st.session_state.get("web_domain_allowlist", WEB_DOMAIN_ALLOWLIST_DEFAULT)
            ),
            "user_role": effective_user_role,
            "oauth_user": active_user_email,
        }
        web_allowlist_enabled_run = bool(
            st.session_state.get("web_domain_allowlist_enabled", WEB_DOMAIN_ALLOWLIST_ENABLED_DEFAULT)
        )
        web_allowlist_raw_run = clean_text(
            st.session_state.get("web_domain_allowlist", WEB_DOMAIN_ALLOWLIST_DEFAULT)
        )
        blocked_web_domains = []
        distill_meta = {"enabled": bool(distill_query_on_run), "used": False, "query": q}

        def finalize_assistant_turn(
            message_payload,
            meta_payload,
            status,
            sources_for_record=None,
            answer_text=None,
            model_used_for_record="",
            query_variants_for_record=None,
        ):
            msg = dict(message_payload or {})
            msg.setdefault("role", "assistant")
            msg["run_id"] = run_id
            msg.setdefault("sources", [])
            msg["settings"] = dict(run_settings)
            effective_sources = list(
                sources_for_record if sources_for_record is not None else (msg.get("sources") or [])
            )
            effective_answer = answer_text if answer_text is not None else msg.get("text", "")
            effective_model = model_used_for_record or msg.get("model") or active_model_run
            replay_compare = compute_replay_compare(
                replay_payload, effective_answer, effective_sources, effective_model
            ) if replay_active else None
            if replay_compare:
                msg["replay_compare"] = replay_compare

            meta = dict(meta_payload or {})
            meta["run_id"] = run_id
            meta["status"] = status
            if query_variants_for_record is not None and "query_variants" not in meta:
                meta["query_variants"] = list(query_variants_for_record)
            if replay_compare:
                meta["replay_compare"] = replay_compare

            duration_ms = int(max((time.time() - run_started) * 1000.0, 0))
            record = build_run_record(
                run_id=run_id,
                query=q,
                answer_text=effective_answer or "",
                sources=effective_sources,
                model_used=effective_model,
                model_chain=model_chain_run,
                settings=run_settings,
                status=status,
                meta=meta,
                duration_ms=duration_ms,
                replay_payload=replay_payload if replay_active else None,
                replay_compare=replay_compare,
            )
            append_run_record(record)
            tal_consume_web_if_used(effective_sources)
            update_web_cache_from_sources(effective_sources)
            st.session_state.msgs.append(msg)
            st.session_state.last_response_meta = meta
            persist_current_chat()
            if replay_active:
                st.session_state.replay_payload = None

        st.session_state.msgs.append({"role": "user", "text": display_user_text})
        persist_current_chat()
        with st.chat_message("user"):
            st.markdown(display_user_text)

        with st.chat_message("assistant"):
            if source_mode_run == "Files only" and ALLOW_WEB_TOOLS and query_requests_fresh_web(q):
                st.caption("This query looks time-sensitive. If files are incomplete, switch to Files + Web.")
            stage = st.empty()
            progress_bar = st.empty()
            stage.caption("Searching sources...")
            render_pipeline_progress(progress_bar, "retrieve")
            if source_mode_run in ("Files only", "Files + Web") and not workspace_allowed(project_filter_run):
                blocked_text = (
                    f"Workspace/project '{project_filter_run}' is not allowed by EDITH_WORKSPACE_ALLOWLIST."
                )
                stage.empty()
                st.error(blocked_text)
                blocked_stoplight = stoplight_status("Not found in sources.", 0.0, {"coverage_ratio": 0.0}, True)
                finalize_assistant_turn(
                    {
                        "role": "assistant",
                        "text": "Not found in sources.",
                        "sources": [],
                        "query": q,
                        "model": active_model_run,
                        "provenance": [],
                        "stoplight": blocked_stoplight,
                    },
                    {
                        "error": "workspace_allowlist_block",
                        "message": blocked_text,
                        "source_mode": source_mode_run,
                        "project_filter": project_filter_run,
                        "stoplight": blocked_stoplight,
                    },
                    status="gated_workspace_allowlist",
                    sources_for_record=[],
                    answer_text="Not found in sources.",
                    model_used_for_record=active_model_run,
                    query_variants_for_record=[q],
                )
                st.stop()
            history = st.session_state.msgs[-MAX_TURNS:]
            history_for_retrieval = [dict(m) for m in history]
            distilled_query = q
            distill_meta = {"enabled": distill_query_on_run, "used": False, "query": q}
            if distill_query_on_run:
                stage.caption("Distilling retrieval query...")
                render_pipeline_progress(progress_bar, "retrieve")
                distilled_query, distill_meta = distill_retrieval_query(
                    user_query=q,
                    history_messages=history,
                    source_mode=source_mode_run,
                    model_chain=model_chain_run,
                )
            rewrite_meta = {"enabled": query_rewrite_on_run, "used": False, "queries": [distilled_query]}
            query_variants = [distilled_query]
            if query_rewrite_on_run:
                stage.caption("Rewriting query...")
                render_pipeline_progress(progress_bar, "retrieve")
                query_variants, rewrite_meta = rewrite_query_variants(
                    user_query=distilled_query,
                    source_mode=source_mode_run,
                    project_filter=project_filter_run,
                    tag_filter=tag_filter_run,
                    model_chain=model_chain_run,
                )
            stage.caption(f"Intent routing: {query_intent.replace('_', ' ')}")

            local_sources = []
            local_context = ""
            evidence_report = {
                "unique_docs": 0,
                "section_coverage": 0.0,
                "redundancy": 1.0,
                "sufficient": False,
                "reason": "no_sources",
            }
            if retrieval_backend_run == "chroma" and source_mode_run in ("Files only", "Files + Web"):
                stage.caption("Retrieving local Chroma sources...")
                render_pipeline_progress(progress_bar, "retrieve")
                if not retrieve_local_sources:
                    st.error("Local Chroma backend unavailable.")
                    stage.empty()
                    finalize_assistant_turn(
                        {"role": "assistant", "text": "Local Chroma backend unavailable.", "sources": []},
                        {
                        "error": "chroma_backend_unavailable",
                        "source_mode": source_mode_run,
                        "project_filter": project_filter_run,
                        "tag_filter": tag_filter_run,
                        "section_filter": section_filter_run,
                        "doc_type_filter": doc_type_filter_run,
                        "query_rewrite": rewrite_meta,
                    "query_distillation": distill_meta,
                        "retrieval_backend": retrieval_backend_run,
                        },
                        status="error_chroma_unavailable",
                        sources_for_record=[],
                        answer_text="Local Chroma backend unavailable.",
                        model_used_for_record=active_model_run,
                        query_variants_for_record=query_variants,
                    )
                    st.stop()
                try:
                    local_sources = retrieve_local_sources(
                        queries=query_variants,
                        chroma_dir=CHROMA_DIR,
                        collection_name=CHROMA_COLLECTION,
                        embed_model=EMBED_MODEL,
                        top_k=chroma_top_k_run,
                        pool_multiplier=CHROMA_POOL_MULTIPLIER,
                        diversity_lambda=float(chroma_diversity_lambda_run),
                        bm25_weight=float(chroma_bm25_weight_run),
                        rerank_model=CHROMA_RERANK_MODEL if chroma_rerank_on_run else "",
                        rerank_top_n=chroma_rerank_top_n_run,
                        project=project_filter_run,
                        tag=tag_filter_run,
                        section_filter=section_filter_run,
                        doc_type_filter=doc_type_filter_run,
                        require_equations=require_equations_run,
                        family_cap=CHROMA_FAMILY_CAP,
                    )
                    if scoped_doc_path_run:
                        local_sources = apply_doc_scope_filter(local_sources, scoped_doc_path_run)
                    if researcher_mode_run:
                        local_sources = select_breadth_depth_sources(
                            local_sources,
                            intent=query_intent,
                            max_docs=10,
                            chunks_per_doc=5,
                            max_total=max(chroma_top_k_run, 24),
                        )
                    evidence_report = evidence_sufficiency_report(local_sources)
                except Exception as e:
                    st.error(f"Local Chroma retrieval failed: {e}")
                    stage.empty()
                    finalize_assistant_turn(
                        {"role": "assistant", "text": "Local Chroma retrieval failed.", "sources": []},
                        {
                        "error": str(e),
                        "source_mode": source_mode_run,
                        "project_filter": project_filter_run,
                        "tag_filter": tag_filter_run,
                        "section_filter": section_filter_run,
                        "doc_type_filter": doc_type_filter_run,
                        "query_rewrite": rewrite_meta,
                    "query_distillation": distill_meta,
                        "retrieval_backend": retrieval_backend_run,
                        },
                        status="error_chroma_retrieval",
                        sources_for_record=[],
                        answer_text="Local Chroma retrieval failed.",
                        model_used_for_record=active_model_run,
                        query_variants_for_record=query_variants,
                    )
                    st.stop()
                packed_local_sources = (
                    pack_sources_for_context(q, local_sources, max_sources=chroma_top_k_run)
                    if context_packing_on_run
                    else list(local_sources)
                )
                local_context = format_local_context(packed_local_sources) if format_local_context else ""

            if history_for_retrieval and history_for_retrieval[-1].get("role") == "user":
                if retrieval_backend_run == "google":
                    prompt_text = with_retrieval_hints(q, query_variants)
                    if library_command_mode:
                        prompt_text = append_library_meta_template(prompt_text)
                    if scoped_doc_path_run:
                        prompt_text += (
                            "\n\nDOCUMENT_SCOPE:\n"
                            f"- Restrict grounding to this document when possible: {scoped_doc_path_run}\n"
                        )
                    history_for_retrieval[-1]["text"] = apply_answer_mode_templates(
                        prompt_text,
                        production_template_run=production_template_run,
                        quote_first_run=quote_first_run,
                        mode_preset=mode_preset_run,
                        answer_type=answer_type_run,
                        verbosity_level=verbosity_level_run,
                        writing_style=writing_style_run,
                        include_methods_table=include_methods_table_run,
                        include_limitations=include_limitations_run,
                        action_approval_run=action_approval_run,
                    )
                elif local_context:
                    prompt_text = (
                        f"{q}\n\n"
                        "LOCAL_FILE_SOURCES:\n"
                        f"{local_context}\n\n"
                        "Use LOCAL_FILE_SOURCES for grounded claims. If evidence is missing, say it is not found."
                    )
                    if library_command_mode:
                        prompt_text = append_library_meta_template(prompt_text)
                    if scoped_doc_path_run:
                        prompt_text += (
                            "\nOnly use evidence from this scoped file when available:\n"
                            f"- {scoped_doc_path_run}\n"
                        )
                    history_for_retrieval[-1]["text"] = apply_answer_mode_templates(
                        prompt_text,
                        production_template_run=production_template_run,
                        quote_first_run=quote_first_run,
                        mode_preset=mode_preset_run,
                        answer_type=answer_type_run,
                        verbosity_level=verbosity_level_run,
                        writing_style=writing_style_run,
                        include_methods_table=include_methods_table_run,
                        include_limitations=include_limitations_run,
                        action_approval_run=action_approval_run,
                    )
            contents = build_contents(history_for_retrieval)

            tools = build_tools(source_mode_run, project_filter_run, tag_filter_run, retrieval_backend=retrieval_backend_run)
            if not tools and not (retrieval_backend_run == "chroma" and source_mode_run == "Files only"):
                st.error("No retrieval tools available for the selected mode.")
                stage.empty()
                finalize_assistant_turn(
                    {"role": "assistant", "text": "No retrieval tools available.", "sources": []},
                    {
                    "error": "no_tools",
                    "source_mode": source_mode_run,
                    "project_filter": project_filter_run,
                    "tag_filter": tag_filter_run,
                    "section_filter": section_filter_run,
                    "doc_type_filter": doc_type_filter_run,
                    "retrieval_backend": retrieval_backend_run,
                    },
                    status="error_no_tools",
                    sources_for_record=[],
                    answer_text="No retrieval tools available.",
                    model_used_for_record=active_model_run,
                    query_variants_for_record=query_variants,
                )
                st.stop()

            if source_mode_run == "Files only" and retrieval_backend_run == "chroma" and strict_citations_run and not local_sources:
                gate_msg = "Not found in sources."
                gate_stoplight = stoplight_status(gate_msg, 0.0, {"coverage_ratio": 0.0}, strict_citations_run)
                stage.empty()
                render_not_found_help(source_mode_run, strict_citations_run, query=q, index_rows=index_rows)
                finalize_assistant_turn(
                    {
                        "role": "assistant",
                        "text": gate_msg,
                        "sources": [],
                        "query": q,
                        "model": active_model_run,
                        "provenance": [],
                        "stoplight": gate_stoplight,
                    },
                    {
                    "source_mode": source_mode_run,
                    "researcher_mode": bool(researcher_mode_run),
                    "query_intent": query_intent,
                    "hybrid_policy": hybrid_policy_run,
                    "model": active_model_run,
                    "gate_message": gate_msg,
                    "source_count": 0,
                    "file_source_count": 0,
                    "web_source_count": 0,
                    "project_filter": project_filter_run,
                    "tag_filter": tag_filter_run,
                    "section_filter": section_filter_run,
                    "doc_type_filter": doc_type_filter_run,
                    "query_rewrite": rewrite_meta,
                    "query_distillation": distill_meta,
                    "retrieval_backend": retrieval_backend_run,
                    "stoplight": gate_stoplight,
                    "local_source_count": 0,
                    "remote_source_count": 0,
                    "evidence_sufficiency": evidence_report,
                    "sentence_provenance": {"enabled": sentence_provenance_on_run, "rows": 0, "unsupported": 0},
                    },
                    status="gated_local_no_sources",
                    sources_for_record=[],
                    answer_text=gate_msg,
                    model_used_for_record=active_model_run,
                    query_variants_for_record=query_variants,
                )
                st.stop()

            if tools:
                cfg = types.GenerateContentConfig(
                    tools=tools,
                    temperature=0.2,
                    system_instruction=system_prompt_for_mode(source_mode_run, hybrid_policy_run, strict_citations_run),
                )
            else:
                cfg = types.GenerateContentConfig(
                    temperature=0.2,
                    system_instruction=system_prompt_for_mode(source_mode_run, hybrid_policy_run, strict_citations_run),
                )

            try:
                stage.caption("Retrieving grounded context...")
                render_pipeline_progress(progress_bar, "synthesize")
                base_resp, used_model = generate_with_model_fallback(contents, cfg, model_chain_run)
            except Exception as e:
                if "metadata" in str(e).lower() or "filter" in str(e).lower():
                    tools = build_fallback_tools(source_mode_run, retrieval_backend=retrieval_backend_run)
                    if not tools and not (retrieval_backend_run == "chroma" and source_mode_run == "Files only"):
                        st.error("No fallback retrieval tools available.")
                        stage.empty()
                        finalize_assistant_turn(
                            {"role": "assistant", "text": "No fallback retrieval tools available.", "sources": []},
                            {
                            "error": "no_fallback_tools",
                            "source_mode": source_mode_run,
                            "project_filter": project_filter_run,
                            "tag_filter": tag_filter_run,
                            "section_filter": section_filter_run,
                            "doc_type_filter": doc_type_filter_run,
                            "query_rewrite": rewrite_meta,
                    "query_distillation": distill_meta,
                            "retrieval_backend": retrieval_backend_run,
                            },
                            status="error_no_fallback_tools",
                            sources_for_record=[],
                            answer_text="No fallback retrieval tools available.",
                            model_used_for_record=active_model_run,
                            query_variants_for_record=query_variants,
                        )
                        st.stop()
                    if tools:
                        cfg = types.GenerateContentConfig(
                            tools=tools,
                            temperature=0.2,
                            system_instruction=system_prompt_for_mode(source_mode_run, hybrid_policy_run, strict_citations_run),
                        )
                    else:
                        cfg = types.GenerateContentConfig(
                            temperature=0.2,
                            system_instruction=system_prompt_for_mode(source_mode_run, hybrid_policy_run, strict_citations_run),
                        )
                    base_resp, used_model = generate_with_model_fallback(contents, cfg, model_chain_run)
                else:
                    st.error(f"Request failed: {e}")
                    stage.empty()
                    finalize_assistant_turn(
                        {"role": "assistant", "text": "Request failed.", "sources": []},
                        {
                        "error": str(e),
                        "source_mode": source_mode_run,
                        "project_filter": project_filter_run,
                        "tag_filter": tag_filter_run,
                        "section_filter": section_filter_run,
                        "doc_type_filter": doc_type_filter_run,
                        "query_rewrite": rewrite_meta,
                    "query_distillation": distill_meta,
                        "retrieval_backend": retrieval_backend_run,
                        },
                        status="error_request_failed",
                        sources_for_record=[],
                        answer_text="Request failed.",
                        model_used_for_record=active_model_run,
                        query_variants_for_record=query_variants,
                    )
                    st.stop()

            base_text = get_text(base_resp)
            remote_sources = extract_sources(base_resp)
            remote_sources, blocked_now = apply_web_domain_policy(
                remote_sources,
                enabled=web_allowlist_enabled_run,
                raw_allowlist=web_allowlist_raw_run,
            )
            if scoped_doc_path_run and source_mode_run in ("Files only", "Files + Web"):
                remote_sources = apply_doc_scope_filter(remote_sources, scoped_doc_path_run)
            if blocked_now:
                for dom in blocked_now:
                    if dom not in blocked_web_domains:
                        blocked_web_domains.append(dom)
            sources = merge_sources(local_sources, remote_sources) if merge_sources else (local_sources + remote_sources)
            file_sources = [s for s in sources if isinstance(s, dict) and s.get("source_type") == "file"]
            if file_sources:
                evidence_report = evidence_sufficiency_report(file_sources)
            confidence_score, confidence_metrics = score_retrieval_confidence(sources, source_mode_run, hybrid_policy_run)
            routing_meta = {
                "enabled": confidence_routing_on_run,
                "ran": False,
                "confidence_score": round(confidence_score, 3),
                "threshold": CONFIDENCE_LOW_THRESHOLD,
            }

            if confidence_routing_on_run and confidence_score < CONFIDENCE_LOW_THRESHOLD:
                target_model = choose_reasoning_model(model_chain_run, used_model)
                if target_model:
                    stage.caption(f"Low confidence detected. Retrying with {target_model}...")
                    render_pipeline_progress(progress_bar, "synthesize")
                    try:
                        routed_resp, routed_model = run_with_specific_model(contents, cfg, target_model)
                        routed_text = get_text(routed_resp)
                        routed_remote_sources = extract_sources(routed_resp)
                        routed_remote_sources, routed_blocked = apply_web_domain_policy(
                            routed_remote_sources,
                            enabled=web_allowlist_enabled_run,
                            raw_allowlist=web_allowlist_raw_run,
                        )
                        if scoped_doc_path_run and source_mode_run in ("Files only", "Files + Web"):
                            routed_remote_sources = apply_doc_scope_filter(routed_remote_sources, scoped_doc_path_run)
                        if routed_blocked:
                            for dom in routed_blocked:
                                if dom not in blocked_web_domains:
                                    blocked_web_domains.append(dom)
                        routed_sources = (
                            merge_sources(local_sources, routed_remote_sources)
                            if merge_sources
                            else (local_sources + routed_remote_sources)
                        )
                        if routed_text:
                            base_resp = routed_resp
                            used_model = routed_model
                            base_text = routed_text
                            remote_sources = routed_remote_sources
                            if routed_sources:
                                sources = routed_sources
                            file_sources = [s for s in sources if isinstance(s, dict) and s.get("source_type") == "file"]
                            if file_sources:
                                evidence_report = evidence_sufficiency_report(file_sources)
                            confidence_score, confidence_metrics = score_retrieval_confidence(sources, source_mode_run, hybrid_policy_run)
                            routing_meta = {
                                "enabled": True,
                                "ran": True,
                                "target_model": target_model,
                                "confidence_score": round(confidence_score, 3),
                                "threshold": CONFIDENCE_LOW_THRESHOLD,
                            }
                    except Exception as e:
                        routing_meta = {
                            "enabled": True,
                            "ran": False,
                            "target_model": target_model,
                            "error": str(e),
                            "confidence_score": round(confidence_score, 3),
                            "threshold": CONFIDENCE_LOW_THRESHOLD,
                        }

            gate_msg = source_gate_message(
                source_mode_run,
                sources,
                hybrid_policy_run,
                require_citations=strict_citations_run,
            )
            if (
                not gate_msg
                and researcher_mode_run
                and source_mode_run in ("Files only", "Files + Web")
                and strict_citations_run
                and query_intent in {"overview", "general", "compare"}
                and int(evidence_report.get("unique_docs", 0)) < 2
            ):
                gate_msg = (
                    "Not found in sources.\n\n"
                    "I only found one relevant document for this broad question. "
                    "Broaden retrieval across your library or narrow the question."
                )
            multi_pass_meta = {"enabled": bool(multi_pass_run), "used": False}
            recursive_controller_meta = {"enabled": bool(recursive_controller_run), "used": False}
            low_confidence_gate_threshold = max(0.22, CONFIDENCE_LOW_THRESHOLD * 0.60)
            if (
                not gate_msg
                and source_mode_run == "Files only"
                and strict_citations_run
                and confidence_score < low_confidence_gate_threshold
            ):
                gate_msg = "Not found in sources."
                routing_meta["low_confidence_gate"] = {
                    "threshold": round(low_confidence_gate_threshold, 3),
                    "score": round(confidence_score, 3),
                }

            if gate_msg:
                gate_text = gate_msg
                if gate_msg.strip().lower().startswith("not found in sources.") and production_template_run:
                    missing = suggested_missing_artifacts(q, max_items=3)
                    if missing:
                        lines = "\n".join(f"- {m}" for m in missing)
                        gate_text = (
                            "Not found in sources.\n\n"
                            "Missing artifacts likely needed:\n"
                            f"{lines}"
                        )
                stage.empty()
                progress_bar.empty()
                gate_lower = gate_msg.strip().lower()
                if gate_lower.startswith("not found in sources."):
                    if gate_text.strip().lower() != "not found in sources.":
                        st.markdown(gate_text)
                    render_not_found_help(source_mode_run, strict_citations_run, query=q, index_rows=index_rows)
                else:
                    st.markdown(gate_text)
                st.caption(f"Used: {friendly_model_tier(used_model)} ({used_model})")
                gate_stoplight = stoplight_status(gate_msg, confidence_score, {"coverage_ratio": 0.0}, strict_citations_run)
                render_stoplight_badge(gate_stoplight)
                finalize_assistant_turn(
                    {
                        "role": "assistant",
                        "text": gate_text,
                        "sources": sources,
                        "query": q,
                        "model": used_model,
                        "provenance": [],
                        "stoplight": gate_stoplight,
                    },
                    {
                    "source_mode": source_mode_run,
                    "researcher_mode": bool(researcher_mode_run),
                    "query_intent": query_intent,
                    "hybrid_policy": hybrid_policy_run,
                    "model": used_model,
                    "gate_message": gate_text,
                    "source_count": len(sources),
                    "file_source_count": sum(1 for s in sources if s.get("source_type") == "file"),
                    "web_source_count": sum(1 for s in sources if s.get("source_type") == "web"),
                    "project_filter": project_filter_run,
                    "tag_filter": tag_filter_run,
                    "section_filter": section_filter_run,
                    "doc_type_filter": doc_type_filter_run,
                    "query_rewrite": rewrite_meta,
                    "query_distillation": distill_meta,
                    "multi_pass": multi_pass_meta,
                    "recursive_controller": recursive_controller_meta,
                    "support_audit": {"enabled": support_audit_on_run, "ran": False},
                    "conflict_report": {"ran": False, "conflicts": []},
                    "retrieval_confidence": confidence_metrics,
                    "evidence_sufficiency": evidence_report,
                    "confidence_routing": routing_meta,
                    "retrieval_backend": retrieval_backend_run,
                    "stoplight": gate_stoplight,
                    "local_source_count": len(local_sources),
                    "remote_source_count": len(remote_sources),
                    "web_domain_filter": {
                        "enabled": bool(web_allowlist_enabled_run),
                        "allowlist": web_allowlist_raw_run,
                        "blocked_domains": blocked_web_domains,
                    },
                    "sentence_provenance": {"enabled": sentence_provenance_on_run, "rows": 0, "unsupported": 0},
                    "citation_coverage": {"coverage_ratio": 0.0, "substantive_sentences": 0, "supported_sentences": 0},
                    },
                    status="gated_source_policy",
                    sources_for_record=sources,
                    answer_text=gate_text,
                    model_used_for_record=used_model,
                    query_variants_for_record=query_variants,
                )
            else:
                final_text = base_text or "(No text returned.)"
                plan_meta = {"used": False, "outline": [], "missing_evidence": []}
                overview_meta = {"enabled": bool(overview_mode_run), "used": False}
                research_synthesis_meta = {
                    "enabled": bool(researcher_mode_run),
                    "used": False,
                    "cards": 0,
                }
                if multi_pass_run and sources and final_text.strip().lower() != "not found in sources.":
                    stage.caption("Planning answer (pass 1)...")
                    render_pipeline_progress(progress_bar, "extract")
                    plan_meta = plan_answer_outline(q, sources, model_chain_run)
                    rewrite_quote_first = bool(quote_first_run or plan_meta.get("must_quote"))
                    pass2_prompt = apply_answer_mode_templates(
                        build_multi_pass_prompt(q, sources, plan_meta=plan_meta),
                        production_template_run=production_template_run,
                        quote_first_run=rewrite_quote_first,
                        mode_preset=mode_preset_run,
                        answer_type=answer_type_run,
                        verbosity_level=verbosity_level_run,
                        writing_style=writing_style_run,
                        include_methods_table=include_methods_table_run,
                        include_limitations=include_limitations_run,
                        action_approval_run=action_approval_run,
                    )
                    if library_command_mode:
                        pass2_prompt = append_library_meta_template(pass2_prompt)
                    stage.caption("Writing answer (pass 2)...")
                    render_pipeline_progress(progress_bar, "write")
                    try:
                        pass2_text, pass2_model = generate_text_via_chain(
                            pass2_prompt,
                            model_chain_run,
                            system_instruction=system_prompt_for_mode(
                                source_mode_run,
                                hybrid_policy_run,
                                strict_citations_run,
                            ),
                            temperature=0.15,
                        )
                        if pass2_text:
                            final_text = pass2_text
                            used_model = pass2_model or used_model
                            multi_pass_meta = {
                                "enabled": True,
                                "used": True,
                                "plan": plan_meta,
                            }
                    except Exception as e:
                        multi_pass_meta = {
                            "enabled": True,
                            "used": False,
                            "error": str(e),
                            "plan": plan_meta,
                        }
                if (
                    recursive_controller_run
                    and sources
                    and final_text.strip().lower() != "not found in sources."
                    and len(sources) >= int(RECURSIVE_CONTROLLER_MIN_SOURCES)
                ):
                    stage.caption("Recursive controller (map/reduce passes)...")
                    render_pipeline_progress(progress_bar, "synthesize")
                    recursive_text, recursive_controller_meta = run_recursive_controller_v1(
                        question=q,
                        sources=sources,
                        model_chain=model_chain_run,
                        source_mode=source_mode_run,
                        hybrid_policy=hybrid_policy_run,
                        strict_citations=bool(strict_citations_run),
                        max_depth=int(RECURSIVE_CONTROLLER_MAX_DEPTH),
                        batch_size=int(RECURSIVE_CONTROLLER_BATCH_SIZE),
                        max_batches=int(RECURSIVE_CONTROLLER_MAX_BATCHES),
                        max_calls=int(RECURSIVE_CONTROLLER_MAX_CALLS),
                    )
                    if recursive_controller_meta.get("used") and recursive_text:
                        final_text = recursive_text
                        used_model = recursive_controller_meta.get("model") or used_model
                if (
                    overview_mode_run
                    and sources
                    and final_text.strip().lower() != "not found in sources."
                    and not recursive_controller_meta.get("used")
                ):
                    stage.caption("Building library overview...")
                    render_pipeline_progress(progress_bar, "synthesize")
                    final_text, overview_meta = rewrite_overview_answer(
                        question=q,
                        draft_answer=final_text,
                        sources=sources,
                        model_chain=model_chain_run,
                    )
                if (
                    researcher_mode_run
                    and sources
                    and final_text.strip().lower() != "not found in sources."
                    and not recursive_controller_meta.get("used")
                ):
                    stage.caption("Building evidence cards...")
                    render_pipeline_progress(progress_bar, "extract")
                    cards = build_research_evidence_cards(
                        sources,
                        max_docs=8,
                        max_snippets_per_doc=4,
                    )
                    research_synthesis_meta["cards"] = len(cards)
                    if cards:
                        stage.caption("Synthesizing across sources...")
                        render_pipeline_progress(progress_bar, "synthesize")
                        try:
                            synth_text, synth_model = generate_text_via_chain(
                                build_research_synthesis_prompt(q, sources, cards),
                                model_chain_run,
                                system_instruction=system_prompt_for_mode(
                                    source_mode_run,
                                    hybrid_policy_run,
                                    strict_citations_run,
                                ),
                                temperature=0.1,
                            )
                            if synth_text:
                                final_text = synth_text
                                used_model = synth_model or used_model
                                research_synthesis_meta = {
                                    "enabled": True,
                                    "used": True,
                                    "cards": len(cards),
                                    "model": synth_model,
                                }
                        except Exception as e:
                            research_synthesis_meta["error"] = str(e)

                if (
                    stream_on_run
                    and not multi_pass_meta.get("used")
                    and not recursive_controller_meta.get("used")
                    and not research_synthesis_meta.get("used")
                    and hasattr(client.models, "generate_content_stream")
                ):
                    placeholder = st.empty()
                    acc = []
                    stage.caption("Drafting response...")
                    render_pipeline_progress(progress_bar, "write")
                    try:
                        for chunk in client.models.generate_content_stream(
                            model=used_model,
                            contents=contents,
                            config=cfg,
                        ):
                            if bool(st.session_state.get("stop_generation")):
                                break
                            t = get_text(chunk)
                            if t:
                                acc.append(t)
                                placeholder.markdown("".join(acc))
                        final_text = "".join(acc).strip() or final_text
                        if bool(st.session_state.get("stop_generation")):
                            st.caption("Generation stopped by user.")
                            st.session_state.stop_generation = False
                    except Exception:
                        pass
                else:
                    stage.caption("Drafting response...")
                    render_pipeline_progress(progress_bar, "write")

                audit_meta = {"enabled": support_audit_on_run, "ran": False}
                if support_audit_on_run and strict_citations_run and final_text and sources:
                    stage.caption("Auditing support coverage...")
                    render_pipeline_progress(progress_bar, "verify")
                    final_text, audit_meta = audit_answer_support(
                        question=q,
                        draft_answer=final_text,
                        sources=sources,
                        model_chain=model_chain_run,
                    )
                    if audit_meta.get("verdict") == "partial":
                        st.caption("Support audit adjusted unsupported claims.")
                        unsupported = (audit_meta.get("unsupported_claims") or [])
                        if unsupported:
                            first_claim = clean_text((unsupported[0] or {}).get("claim") or "")
                            if first_claim:
                                st.caption(f"Removed unsupported claim: {first_claim[:120]}")
                    elif audit_meta.get("verdict") == "unsupported":
                        st.caption("Support audit blocked unsupported answer.")

                citation_check = {"ok": True, "reason": "not_required"}
                if strict_citations_run and sources and final_text.strip().lower() != "not found in sources.":
                    final_text = ensure_inline_citations(final_text, sources, index_map=index_map)
                    citation_check = validate_citation_refs(final_text, sources)
                    if not citation_check.get("ok"):
                        final_text = "Not found in sources."
                if sources and final_text.strip().lower() != "not found in sources.":
                    final_text = append_key_sources_section(final_text, sources, index_map=index_map, max_items=12)

                approval_required = False
                if action_approval_run and final_text.strip().lower() != "not found in sources.":
                    final_text, approval_required = enforce_action_approval_output(final_text)

                provenance_rows = []
                unsupported_sentences = 0
                coverage_stats = {"coverage_ratio": 1.0, "substantive_sentences": 0, "supported_sentences": 0}
                if sentence_provenance_on_run and final_text.strip().lower() != "not found in sources.":
                    final_text, provenance_rows, unsupported_sentences = apply_sentence_provenance(
                        final_text,
                        sources,
                        strict_mode=bool(strict_citations_run and strict_sentence_tags_on_run),
                    )
                    coverage_stats = citation_coverage_stats(provenance_rows)
                coverage_stats["audit_unsupported_count"] = int(audit_meta.get("unsupported_count") or 0)

                conflict_report = {"ran": False, "conflicts": []}
                if contradiction_check_run and final_text.strip().lower() != "not found in sources.":
                    stage.caption("Checking contradictions...")
                    conflict_report = detect_source_conflicts(q, sources, model_chain_run)

                next_questions = []
                if next_questions_on_run and sources and final_text.strip().lower() != "not found in sources.":
                    stage.caption("Generating next questions...")
                    next_questions = generate_grounded_followups(q, sources, model_chain_run)
                theme_clusters = []
                if (
                    sources
                    and final_text.strip().lower() != "not found in sources."
                    and (query_intent in {"overview", "general", "compare"} or overview_mode_run or researcher_mode_run)
                ):
                    theme_clusters = build_theme_clusters(sources, index_map=index_map, max_clusters=6)

                refusal_text = final_text.strip().lower()
                is_not_found = refusal_text.startswith("not found in sources.")
                if is_not_found:
                    coverage_stats = {"coverage_ratio": 0.0, "substantive_sentences": 0, "supported_sentences": 0}
                    stoplight = stoplight_status("Not found in sources.", confidence_score, coverage_stats, strict_citations_run)
                else:
                    stoplight = stoplight_status("", confidence_score, coverage_stats, strict_citations_run)

                stage.empty()
                progress_bar.empty()
                if stream_on_run and "placeholder" in locals():
                    placeholder.empty()
                if is_not_found and refusal_text == "not found in sources.":
                    render_not_found_help(source_mode_run, strict_citations_run, query=q, index_rows=index_rows)
                else:
                    answer_body, answer_cites = split_answer_and_citations(final_text)
                    with st.expander("Answer", expanded=True):
                        st.markdown(answer_body or final_text)
                    if answer_cites:
                        with st.expander("Answer citations", expanded=False):
                            st.markdown(answer_cites)

                if not simple_ui:
                    if used_model != active_model_run:
                        st.caption(f"Model fallback used: {friendly_model_tier(used_model)} ({used_model})")
                    else:
                        st.caption(f"Used: {friendly_model_tier(used_model)} ({used_model})")
                    if query_rewrite_on_run and rewrite_meta.get("used"):
                        st.caption("Query rewrite used for retrieval.")
                    if distill_meta.get("used"):
                        st.caption("Conversation-to-query distillation used.")
                    if production_template_run:
                        st.caption("Production answer template applied.")
                    if quote_first_run:
                        st.caption("Quote-first recall mode applied.")
                    st.caption(f"Intent routing: {query_intent}")
                    if multi_pass_meta.get("used"):
                        st.caption("Multi-pass answering used (plan -> answer -> audit).")
                    if recursive_controller_meta.get("used"):
                        st.caption(
                            "Recursive controller v1 used "
                            f"(calls={int(recursive_controller_meta.get('calls', 0))}, "
                            f"leaf_batches={int(recursive_controller_meta.get('leaf_batches', 0))}, "
                            f"depth={int(recursive_controller_meta.get('depth_used', 0))})."
                        )
                    if research_synthesis_meta.get("used"):
                        st.caption(
                            f"Research synthesis used ({research_synthesis_meta.get('cards', 0)} evidence cards)."
                        )
                    if routing_meta.get("ran"):
                        st.caption("Confidence routing used stronger model.")
                    if researcher_mode_run:
                        st.caption(
                            "Evidence sufficiency: "
                            f"docs={evidence_report.get('unique_docs', 0)}, "
                            f"section_coverage={float(evidence_report.get('section_coverage', 0.0)):.2f}, "
                            f"redundancy={float(evidence_report.get('redundancy', 0.0)):.2f}"
                        )
                    if not is_not_found and coverage_stats.get("substantive_sentences", 0) > 0:
                        st.caption(
                            f"Citation coverage: {coverage_stats.get('supported_sentences', 0)}/"
                            f"{coverage_stats.get('substantive_sentences', 0)} "
                            f"({coverage_stats.get('coverage_ratio', 0.0):.2f})"
                        )
                render_stoplight_badge(stoplight)
                decision_line = build_decision_summary(
                    query_intent=query_intent,
                    source_mode=source_mode_run,
                    researcher_mode=researcher_mode_run,
                    source_count=len(sources),
                    file_source_count=sum(1 for s in sources if s.get("source_type") == "file"),
                )
                st.caption(decision_line)
                if stoplight.get("level") in {"yellow", "red"}:
                    if st.button("Show why this is weak", key=f"weak_reason_{run_id}"):
                        st.session_state.show_not_found_debug = True
                        st.caption(
                            "Weakness details: "
                            f"docs={evidence_report.get('unique_docs', 0)}, "
                            f"coverage={float(coverage_stats.get('coverage_ratio', 0.0)):.2f}, "
                            f"confidence={float(confidence_score):.2f}"
                        )
                if approval_required:
                    st.caption("Approval required before executable commands/scripts.")
                if approval_bypass_blocked:
                    st.caption("Action output request ignored: enable 'write/modify' action mode in Advanced first.")
                if (
                    source_mode_run in ("Files only", "Files + Web")
                    and (overview_mode_run or researcher_mode_run or query_intent in {"overview", "general", "compare"})
                ):
                    render_thin_evidence_help(q, run_id, int(evidence_report.get("unique_docs", 0)))
                render_source_chips(sources, index_map=index_map, max_chips=8)
                render_reasoning_panel(sources, query=q, index_map=index_map)
                render_web_usage_summary(sources)
                if blocked_web_domains and source_mode_run in ("Files + Web", "Web only"):
                    st.caption(
                        "Web domain filter blocked: "
                        + ", ".join(blocked_web_domains[:6])
                        + ("..." if len(blocked_web_domains) > 6 else "")
                    )
                render_sources(sources, query=q, audit_mode=audit_mode, index_map=index_map)
                if next_questions:
                    st.caption("Grounded next questions")
                    nq_cols = st.columns(len(next_questions))
                    for nq_i, nq in enumerate(next_questions):
                        if nq_cols[nq_i].button(str(nq), key=f"live_nextq_{run_id}_{nq_i}"):
                            st.session_state.queued_prompt = str(nq)
                            st.rerun()
                if theme_clusters:
                    with st.expander("Theme clusters", expanded=False):
                        for cluster in theme_clusters:
                            st.markdown(
                                f"- **{clean_text(cluster.get('label') or 'general')}** "
                                f"(hits={int(cluster.get('count') or 0)}, "
                                f"unique_sources={int(cluster.get('unique_sources') or 0)})"
                            )
                if conflict_report.get("ran") and conflict_report.get("conflicts"):
                    with st.expander("Conflict report", expanded=False):
                        for c in conflict_report.get("conflicts", []):
                            claim = html.escape(str(c.get("claim") or ""))
                            reason = html.escape(str(c.get("reason") or ""))
                            sa = html.escape(str(c.get("source_a") or ""))
                            sb = html.escape(str(c.get("source_b") or ""))
                            st.markdown(f"- **{claim}**")
                            if sa or sb:
                                st.caption(f"Sources: {sa} vs {sb}")
                            if reason:
                                st.caption(reason)
                if provenance_rows:
                    render_sentence_provenance(
                        provenance_rows,
                        render_key=f"new_{st.session_state.active_chat_id}_{len(st.session_state.msgs)}",
                        sources=sources,
                    )
                policy_note = hybrid_policy_caption(hybrid_policy_run, sources) if source_mode_run == "Files + Web" else ""
                if policy_note:
                    st.caption(policy_note)
                finalize_assistant_turn(
                    {
                        "role": "assistant",
                        "text": final_text,
                        "sources": sources,
                        "query": q,
                        "model": used_model,
                        "provenance": provenance_rows,
                        "conflict_report": conflict_report,
                        "next_questions": next_questions,
                        "theme_clusters": theme_clusters,
                        "stoplight": stoplight,
                        "approval_required": approval_required,
                    },
                    {
                    "source_mode": source_mode_run,
                    "researcher_mode": bool(researcher_mode_run),
                    "hybrid_policy": hybrid_policy_run,
                    "model": used_model,
                    "source_count": len(sources),
                    "file_source_count": sum(1 for s in sources if s.get("source_type") == "file"),
                    "web_source_count": sum(1 for s in sources if s.get("source_type") == "web"),
                    "project_filter": project_filter_run,
                    "tag_filter": tag_filter_run,
                    "section_filter": section_filter_run,
                    "doc_type_filter": doc_type_filter_run,
                    "gated": False,
                    "query_rewrite": rewrite_meta,
                    "query_distillation": distill_meta,
                    "query_variants": query_variants,
                    "multi_pass": multi_pass_meta,
                    "recursive_controller": recursive_controller_meta,
                    "overview": overview_meta,
                    "research_synthesis": research_synthesis_meta,
                    "support_audit": audit_meta,
                    "conflict_report": conflict_report,
                    "theme_clusters": theme_clusters,
                    "citation_check": citation_check,
                    "retrieval_confidence": confidence_metrics,
                    "retrieval_confidence_score": round(confidence_score, 3),
                    "evidence_sufficiency": evidence_report,
                    "confidence_routing": routing_meta,
                    "retrieval_backend": retrieval_backend_run,
                    "context_packing_on": bool(context_packing_on_run),
                    "next_questions_on": bool(next_questions_on_run),
                    "action_approval_on": bool(action_approval_run),
                    "action_outputs_enabled_on": bool(action_outputs_enabled_run),
                    "approval_bypass_blocked": bool(approval_bypass_blocked),
                    "approval_required": bool(approval_required),
                    "stoplight": stoplight,
                    "local_source_count": len(local_sources),
                    "remote_source_count": len(remote_sources),
                    "web_domain_filter": {
                        "enabled": bool(web_allowlist_enabled_run),
                        "allowlist": web_allowlist_raw_run,
                        "blocked_domains": blocked_web_domains,
                    },
                    "sentence_provenance": {
                        "enabled": sentence_provenance_on_run,
                        "strict_tags": bool(strict_citations_run and strict_sentence_tags_on_run),
                        "rows": len(provenance_rows),
                        "unsupported": unsupported_sentences,
                    },
                    "citation_coverage": coverage_stats,
                    },
                    status="ok",
                    sources_for_record=sources,
                    answer_text=final_text,
                    model_used_for_record=used_model,
                    query_variants_for_record=query_variants,
                )


with library_tab:
    st.markdown("### Library")
    st.caption("Browse/search indexed sources by title, author, year, project, and tag.")
    st.toggle(
        "Auto-scope chat when opening a document",
        value=bool(st.session_state.get("auto_scope_doc", True)),
        key="auto_scope_doc",
        help="When on, \"Ask about this doc\" keeps chat scoped to that document until cleared.",
    )
    if st.session_state.get("scoped_doc_path"):
        scope_name = Path(st.session_state.get("scoped_doc_path")).name
        scope_cols = st.columns([0.8, 0.2])
        scope_cols[0].caption(f"Scoped to document: {scope_name}")
        if scope_cols[1].button("Clear scope", key="library_clear_scope"):
            st.session_state.scoped_doc_path = ""
            st.rerun()

    browse_tab, search_tab = st.tabs(["Browse", "Search"])
    doc_type_options = ["All"] + sorted({(r.get("doc_type") or "").strip() for r in index_rows if (r.get("doc_type") or "").strip()})

    def render_library_rows(rows, key_prefix: str):
        if not rows:
            st.info("No indexed documents matched.")
            return
        max_cards = min(len(rows), 120)
        for i, row in enumerate(rows[:max_cards]):
            title = row.get("title") or row.get("file_name") or "Untitled"
            author = row.get("author") or "Unknown"
            year = row.get("year") or "n.d."
            rel_path = row.get("rel_path") or row.get("file_name") or ""
            project = row.get("project") or "none"
            tag = row.get("tag") or "none"
            doc_type = row.get("doc_type") or "document"
            version_stage = row.get("version_stage") or "unknown"
            indexed_at = clean_text(row.get("indexed_at") or row.get("updated_at") or "")
            ocr_used = str(row.get("ocr_used") or "").strip().lower() == "true"
            st.markdown(
                f"""
<div class="library-card">
  <div class="source-card-title">{html.escape(title)}</div>
  <div class="source-card-meta">{html.escape(author)} | {html.escape(year)} | type={html.escape(doc_type)} | version={html.escape(version_stage)}</div>
  <div class="source-card-meta">project={html.escape(project)} | tag={html.escape(tag)} | ocr={'yes' if ocr_used else 'no'}</div>
  <div class="source-card-meta">last indexed: {html.escape(indexed_at or 'unknown')}</div>
  <div class="source-card-meta">{html.escape(rel_path)}</div>
</div>
""",
                unsafe_allow_html=True,
            )
            act1, act2 = st.columns(2)
            if act1.button("Ask about this doc", key=f"{key_prefix}_ask_{i}_{stable_hash(rel_path)[:8]}"):
                if bool(st.session_state.get("auto_scope_doc", True)):
                    st.session_state.scoped_doc_path = rel_path
                st.session_state.queued_prompt = (
                    "Summarize this document with citations only:\n"
                    f"- file: {rel_path}\n"
                    f"- title: {title}\n"
                    "Then list 3 key claims and supporting evidence."
                )
                st.rerun()
            if act2.button("Scope chat to this doc", key=f"{key_prefix}_scope_{i}_{stable_hash(rel_path)[:8]}"):
                st.session_state.scoped_doc_path = rel_path
                st.success(f"Scoped to {Path(rel_path).name}")
        if len(rows) > max_cards:
            st.caption(f"Showing first {max_cards} results. Narrow filters to see more.")

    with browse_tab:
        st.caption("Browse recently indexed documents.")
        rows = list(index_rows)
        rows.sort(
            key=lambda r: clean_text((r or {}).get("indexed_at") or (r or {}).get("updated_at") or ""),
            reverse=True,
        )
        render_library_rows(rows, key_prefix="browse")

    with search_tab:
        lib_col1, lib_col2, lib_col3, lib_col4 = st.columns(4)
        lib_search = lib_col1.text_input("Search", key="library_search_query")
        lib_author = lib_col2.text_input("Author", key="library_search_author")
        lib_year = lib_col3.text_input("Year", max_chars=4, key="library_search_year")
        lib_doc_type = lib_col4.selectbox("Doc type", doc_type_options, key="lib_doc_type")
        lib_col5, lib_col6 = st.columns(2)
        lib_project = lib_col5.selectbox("Project", project_options, key="lib_project")
        lib_tag = lib_col6.text_input("Tag", key="lib_tag")
        if lib_tag.startswith("#"):
            lib_tag = lib_tag[1:]
        if lib_tag.startswith("[") and lib_tag.endswith("]") and len(lib_tag) > 2:
            lib_tag = lib_tag[1:-1]

        filtered_rows = filter_library_rows(
            index_rows,
            search=lib_search,
            project=lib_project,
            author=lib_author,
            year=lib_year,
            tag=lib_tag,
            doc_type=lib_doc_type,
        )
        st.caption(f"{len(filtered_rows)} document(s)")
        render_library_rows(filtered_rows, key_prefix="search")


with notebook_tab:
    st.markdown("### Research Notebook")
    st.caption("Save summaries, claims, and quotes. Notes are kept locally and can be indexed back into your library.")
    nb_rows = load_notebook_entries(limit=800)
    nb_col1, nb_col2, nb_col3 = st.columns(3)
    nb_project = nb_col1.selectbox("Project", ["All"] + project_options[1:], key="nb_project_filter")
    nb_kind = nb_col2.selectbox("Kind", ["All", "summary", "claim", "quote", "note"], key="nb_kind_filter")
    nb_search = nb_col3.text_input("Search", key="nb_search_filter")

    def _match_notebook(r):
        if not isinstance(r, dict):
            return False
        if nb_project != "All" and clean_text(r.get("project") or "") != nb_project:
            return False
        if nb_kind != "All" and clean_text(r.get("kind") or "").lower() != nb_kind:
            return False
        if nb_search.strip():
            hay = " ".join(
                [
                    clean_text(r.get("title") or ""),
                    clean_text(r.get("text") or ""),
                    clean_text(r.get("query") or ""),
                ]
            ).lower()
            if nb_search.strip().lower() not in hay:
                return False
        return True

    filtered_nb = [r for r in nb_rows if _match_notebook(r)]
    st.caption(f"{len(filtered_nb)} note(s)")
    if filtered_nb:
        export_nb_md = notebook_entries_to_markdown(filtered_nb)
        st.download_button(
            "Download Notebook Markdown",
            data=export_nb_md,
            file_name=f"edith_notebook_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
            mime="text/markdown",
            key="download_notebook_md_btn",
        )
    else:
        st.info("No notebook entries yet. Use Save summary/claim/quote from answer actions.")

    if DATA_ROOT:
        notes_path = Path(DATA_ROOT).expanduser() / "edith_notes"
        st.caption(f"Notebook folder: `{notes_path}`")
    if st.button("Index notebook notes now", key="index_notebook_now_btn", disabled=not can_run_index):
        if enforce_rate_limit(
            "index_notebook_now",
            RATE_LIMIT_MUTATION_MAX,
            RATE_LIMIT_MUTATION_WINDOW_SECONDS,
            "Notebook indexing",
        ):
            code, output = run_reindex(retrieval_backend=retrieval_backend)
            st.session_state.reindex_output = output
            st.session_state.reindex_code = code
            save_index_status(code, output)
            if code == 0:
                st.success("Notebook notes indexed.")
            else:
                st.error("Notebook indexing failed. Check Doctor tab.")

    for i, row in enumerate(filtered_nb[:120]):
        title = clean_text(row.get("title") or "Notebook entry")
        created = clean_text(row.get("created_at") or "")
        kind = clean_text(row.get("kind") or "note")
        project = clean_text(row.get("project") or "All")
        st.markdown(f"**{html.escape(title)}**")
        st.caption(f"{created} | kind={kind} | project={project}")
        with st.expander("View note", expanded=False):
            st.markdown(clean_text(row.get("text") or ""))
            q = clean_text(row.get("query") or "")
            if q:
                st.caption(f"From query: {q}")
            srcs = row.get("sources") or []
            if srcs:
                st.caption("Sources")
                for s in srcs[:8]:
                    if not isinstance(s, dict):
                        continue
                    label = clean_text(s.get("label") or s.get("title") or "source")
                    uri = clean_text(s.get("uri") or "")
                    st.markdown(f"- {label}" + (f" ({uri})" if uri else ""))


with runs_tab:
    st.markdown("### Runs")
    st.caption("Reproducibility log with replay and quick diffs.")

    st.markdown("### Feedback + Auto-Tune")
    st.caption("Capture answer/source quality and tune Chroma retrieval with an eval gate.")
    if st.button("Refresh feedback summary", key="refresh_feedback_summary"):
        st.session_state.feedback_summary = load_feedback_summary()
    fb = st.session_state.feedback_summary or load_feedback_summary()
    fb_col1, fb_col2, fb_col3, fb_col4, fb_col5, fb_col6, fb_col7 = st.columns(7)
    fb_col1.metric("Answer good", int(fb.get("answer_good", 0)))
    fb_col2.metric("Answer bad", int(fb.get("answer_bad", 0)))
    fb_col3.metric("Sources good", int(fb.get("sources_good", 0)))
    fb_col4.metric("Sources bad", int(fb.get("sources_bad", 0)))
    fb_col5.metric("Missing source", int(fb.get("missing_source", 0)))
    fb_col6.metric("Should refuse", int(fb.get("should_refuse", 0)))
    fb_col7.metric("Bad citation", int(fb.get("bad_citation", 0)))
    if fb.get("notes"):
        st.caption("Recent missing-source notes:")
        for note in fb.get("notes", [])[:5]:
            st.markdown(f"- {html.escape(str(note))}")

    with st.expander("Feedback -> Eval cases", expanded=False):
        st.caption("Convert negative feedback into regression/trap cases.")
        fe_col1, fe_col2 = st.columns(2)
        fail_limit = fe_col1.slider("Max extracted cases", min_value=10, max_value=120, value=40, step=5)
        append_main_cases = fe_col2.toggle("Append to eval/cases.jsonl", value=True)
        if st.button("Export failing feedback cases", key="export_feedback_eval_cases"):
            out_path = Path(__file__).parent / "eval" / "generated" / "feedback_failures.jsonl"
            append_path = (Path(__file__).parent / "eval" / "cases.jsonl") if append_main_cases else None
            result = export_feedback_failures_to_eval(out_path, limit=int(fail_limit), append_to_cases=append_path)
            if result.get("error"):
                st.error(f"Export failed: {result.get('error')}")
            else:
                st.success(
                    f"Exported {int(result.get('written', 0))} cases to {result.get('path')}"
                    + (
                        f"; appended {int(result.get('appended', 0))} new cases to eval/cases.jsonl"
                        if append_main_cases
                        else ""
                    )
                )

    current_profile = normalize_retrieval_profile(
        {
            "top_k": int(st.session_state.get("chroma_top_k", CHROMA_TOP_K)),
            "bm25_weight": float(st.session_state.get("chroma_bm25_weight", CHROMA_BM25_WEIGHT)),
            "diversity_lambda": float(st.session_state.get("chroma_diversity_lambda", CHROMA_DIVERSITY_LAMBDA)),
            "rerank_top_n": int(st.session_state.get("chroma_rerank_top_n", CHROMA_RERANK_TOP_N)),
            "rerank_on": bool(st.session_state.get("chroma_rerank_on", CHROMA_RERANK_ENABLED_DEFAULT)),
        }
    )
    auto_candidate, auto_reasons, auto_changed = propose_retrieval_candidate_from_feedback(current_profile, fb)
    if auto_changed and not st.session_state.tuning_candidate_profile:
        st.session_state.tuning_candidate_profile = auto_candidate
        st.session_state.tuning_candidate_reasons = auto_reasons
        st.session_state.tuning_eval_result = {}
    if auto_changed:
        st.caption("Auto-tune signal detected from feedback.")
    tune_cols = st.columns(2)
    if tune_cols[0].button("Generate candidate from feedback", key="gen_tune_candidate"):
        candidate, reasons, changed = propose_retrieval_candidate_from_feedback(current_profile, fb)
        st.session_state.tuning_candidate_profile = candidate
        st.session_state.tuning_candidate_reasons = reasons
        st.session_state.tuning_eval_result = {}
        if changed:
            st.success("Candidate retrieval profile generated.")
        else:
            st.info("No parameter changes recommended from current feedback.")

    candidate_profile = st.session_state.tuning_candidate_profile or {}
    if candidate_profile:
        st.caption("Candidate profile")
        st.json(candidate_profile)
        for reason in st.session_state.tuning_candidate_reasons or []:
            st.caption(f"- {reason}")
        if tune_cols[1].button("Run eval gate + promote if pass", key="run_tune_eval"):
            with st.spinner("Running baseline and candidate eval..."):
                gate = eval_gate_candidate_profile(current_profile, candidate_profile)
            st.session_state.tuning_eval_result = gate
            if gate.get("passed"):
                saved = save_retrieval_profile(candidate_profile)
                st.session_state.chroma_top_k = int(saved.get("top_k", st.session_state.chroma_top_k))
                st.session_state.chroma_bm25_weight = float(saved.get("bm25_weight", st.session_state.chroma_bm25_weight))
                st.session_state.chroma_diversity_lambda = float(saved.get("diversity_lambda", st.session_state.chroma_diversity_lambda))
                st.session_state.chroma_rerank_top_n = int(saved.get("rerank_top_n", st.session_state.chroma_rerank_top_n))
                st.session_state.chroma_rerank_on = bool(saved.get("rerank_on", st.session_state.chroma_rerank_on))
                st.success("Eval gate passed. Candidate promoted and saved.")
            else:
                st.warning("Eval gate blocked promotion.")

    gate_result = st.session_state.tuning_eval_result or {}
    if gate_result:
        st.caption(f"Eval gate: {'PASS' if gate_result.get('passed') else 'BLOCKED'}")
        if gate_result.get("reasons"):
            for reason in gate_result.get("reasons", []):
                st.caption(f"- {reason}")
        base_summary = ((gate_result.get("baseline") or {}).get("summary") or {})
        cand_summary = ((gate_result.get("candidate") or {}).get("summary") or {})
        base_trap_summary = ((gate_result.get("baseline_trap") or {}).get("summary") or {})
        cand_trap_summary = ((gate_result.get("candidate_trap") or {}).get("summary") or {})
        if base_summary and cand_summary:
            st.caption(
                "Baseline vs candidate: "
                f"precision {base_summary.get('citation_precision', 0.0):.3f}->{cand_summary.get('citation_precision', 0.0):.3f}, "
                f"refusal {base_summary.get('refusal_accuracy', 0.0):.3f}->{cand_summary.get('refusal_accuracy', 0.0):.3f}, "
                f"p95 latency {base_summary.get('latency_p95', 0.0):.2f}s->{cand_summary.get('latency_p95', 0.0):.2f}s"
            )
        if base_trap_summary and cand_trap_summary:
            st.caption(
                "Trap refusal accuracy: "
                f"{base_trap_summary.get('refusal_accuracy', 0.0):.3f}->"
                f"{cand_trap_summary.get('refusal_accuracy', 0.0):.3f}"
            )

    with st.expander("A/B eval lab", expanded=False):
        st.caption("Compare retrieval/rerank/rewrite variants against the eval suite.")
        if st.button("Run A/B variants", key="run_ab_variants"):
            with st.spinner("Running A/B evaluations..."):
                st.session_state.ab_eval_rows = run_ab_eval_variants(current_profile)
        ab_rows = st.session_state.get("ab_eval_rows") or []
        if ab_rows:
            st.dataframe(ab_rows, use_container_width=True, hide_index=True)
            best = ab_rows[0]
            st.caption(
                f"Best variant: {best.get('variant')} | score={best.get('score')} | "
                f"precision={best.get('citation_precision')} | refusal={best.get('refusal_accuracy')} | "
                f"trap_refusal={best.get('trap_refusal_accuracy')}"
            )

    st.markdown("### Fine-Tune Dataset")
    if SFT_REDACT_PII:
        st.caption(f"PII redaction is enabled for SFT export ({SFT_REDACT_TOKEN}).")
    else:
        st.caption("PII redaction is disabled for SFT export.")
    ft_col1, ft_col2, ft_col3 = st.columns(3)
    ft_max_examples = ft_col1.slider("Max examples", min_value=50, max_value=5000, value=600, step=50)
    ft_include_refusals = ft_col2.toggle("Include refusal examples", value=True)
    ft_positive_only = ft_col3.toggle("Only positive-feedback runs", value=True)
    if st.button("Build SFT dataset from saved chats"):
        examples = collect_sft_examples_from_saved_chats(
            max_examples=int(ft_max_examples),
            include_refusals=bool(ft_include_refusals),
            only_positive_feedback=bool(ft_positive_only),
        )
        train_rows, val_rows = split_train_val_examples(examples, val_ratio=0.1)
        train_jsonl = examples_to_jsonl(train_rows)
        val_jsonl = examples_to_jsonl(val_rows)
        st.session_state.sft_export_train_jsonl = train_jsonl
        st.session_state.sft_export_val_jsonl = val_jsonl
        st.session_state.sft_export_summary = {
            "examples_total": len(examples),
            "train_count": len(train_rows),
            "val_count": len(val_rows),
            "include_refusals": bool(ft_include_refusals),
            "only_positive_feedback": bool(ft_positive_only),
        }
        st.success(f"Built dataset with {len(examples)} examples.")

    sft_summary = st.session_state.sft_export_summary or {}
    if sft_summary:
        st.caption(
            f"Dataset: total={sft_summary.get('examples_total', 0)}, "
            f"train={sft_summary.get('train_count', 0)}, "
            f"val={sft_summary.get('val_count', 0)}"
        )
        if sft_summary.get("only_positive_feedback"):
            st.caption("Quality filter: only positively-reviewed runs included.")
        if st.session_state.sft_export_train_jsonl:
            st.download_button(
                "Download train.jsonl",
                data=st.session_state.sft_export_train_jsonl,
                file_name="edith_train.jsonl",
                mime="application/jsonl",
                key="download_sft_train",
            )
        if st.session_state.sft_export_val_jsonl:
            st.download_button(
                "Download val.jsonl",
                data=st.session_state.sft_export_val_jsonl,
                file_name="edith_val.jsonl",
                mime="application/jsonl",
                key="download_sft_val",
            )

    run_rows = load_recent_run_records(limit=120)
    if not run_rows:
        st.info("No runs saved yet.")
    else:
        latencies = [int(r.get("duration_ms") or 0) for r in run_rows]
        source_counts = [int(r.get("source_count") or 0) for r in run_rows]
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Runs", len(run_rows))
        k2.metric("Avg latency", f"{int(sum(latencies)/max(1,len(latencies)))} ms")
        k3.metric("Avg sources", f"{sum(source_counts)/max(1,len(source_counts)):.1f}")
        k4.metric("Latest model", run_rows[0].get("model_used") or "n/a")

        for idx, rec in enumerate(run_rows):
            run_id = rec.get("run_id", "")
            label = run_record_label(rec)
            with st.expander(label, expanded=(idx == 0)):
                st.caption(f"Status: {rec.get('status', 'unknown')}")
                st.caption(f"Model: {rec.get('model_used', '')}")
                st.caption(f"Sources: {rec.get('source_count', 0)}")
                st.caption(f"Duration: {rec.get('duration_ms', 0)} ms")
                query_preview = (rec.get("query") or "").strip()
                if query_preview:
                    st.markdown(f"**Query**: {query_preview}")

                cols = st.columns(2)
                if cols[0].button("Replay", key=f"runs_replay_{run_id}"):
                    st.session_state.replay_payload = rec
                    st.session_state.queued_prompt = rec.get("query", "")
                    st.rerun()

                previous = None
                for prior in run_rows[idx + 1 :]:
                    if prior.get("query_hash") == rec.get("query_hash"):
                        previous = prior
                        break
                if previous:
                    old_keys = set(normalize_source_keys_for_compare(previous))
                    new_keys = set(normalize_source_keys_for_compare(rec))
                    overlap = source_overlap_ratio(old_keys, new_keys)
                    changed = (previous.get("answer_hash") or "") != (rec.get("answer_hash") or "")
                    cols[1].caption(f"Diff vs last similar run: overlap={overlap:.2f}, answer_changed={changed}")
                else:
                    cols[1].caption("No prior comparable run.")


with phd_tab:
    st.markdown("### PhD OS")
    st.caption("Glossary/citation graph, chapter anchors, claim inventory, experiment ledger, and index-health diagnostics.")
    if st.button("Build / refresh PhD indexes", key="build_phd_indexes", disabled=IS_FROZEN_APP or (not can_run_index)):
        with st.spinner("Building glossary/citation/chapter/claim/experiment indexes..."):
            code, output = run_build_phd_indexes()
        if code == 0:
            st.success("PhD indexes updated.")
            st.rerun()
        else:
            st.error(output)
    if st.button("Run index health report", key="run_index_health", disabled=IS_FROZEN_APP or (not can_run_index)):
        with st.spinner("Computing index health..."):
            code, output = run_index_health_report()
        if code == 0:
            st.success("Index health report updated.")
            st.rerun()
        else:
            st.error(output)
    if st.button("Create corpus snapshot", key="create_corpus_snapshot", disabled=IS_FROZEN_APP or (not can_run_index)):
        with st.spinner("Building reproducible corpus snapshot..."):
            code, output = run_corpus_snapshot()
        if code == 0:
            st.success("Corpus snapshot created.")
            st.caption(output)
        else:
            st.error(output)

    g_nodes = glossary_graph.get("nodes") or []
    c_nodes = citation_graph.get("citations") or []
    chapters = (chapter_anchors.get("chapters") or {}) if isinstance(chapter_anchors, dict) else {}
    claim_rows = claim_inventory.get("claims") or []
    experiment_rows = experiment_ledger.get("experiments") or []
    biblio_rows = bibliography_db.get("records") or []
    timeline_events = entity_timeline.get("events") or []
    m1, m2, m3, m4, m5, m6, m7 = st.columns(7)
    m1.metric("Glossary terms", len(g_nodes))
    m2.metric("Citations", len(c_nodes))
    m3.metric("Chapter anchors", len(chapters))
    m4.metric("Claim inventory", len(claim_rows))
    m5.metric("Experiments", len(experiment_rows))
    m6.metric("Bibliography", len(biblio_rows))
    m7.metric("Timeline events", len(timeline_events))

    st.markdown("#### Research map")
    st.caption("Concepts ↔ papers ↔ methods ↔ notes ↔ claims")
    map_query = st.text_input("Explore a concept", key="phd_research_map_query", placeholder="e.g., voting context")
    if map_query.strip():
        notebook_rows = load_notebook_entries(limit=600)
        map_snapshot = build_research_map_snapshot(
            focus=map_query,
            bibliography_rows=biblio_rows,
            claim_rows=claim_rows,
            notebook_rows=notebook_rows,
            citation_edges=(citation_graph.get("edges") or []) if isinstance(citation_graph, dict) else [],
            experiment_rows=experiment_rows,
        )
        rm_col1, rm_col2 = st.columns(2)
        with rm_col1:
            st.markdown("**Top papers**")
            if map_snapshot.get("papers"):
                for row in map_snapshot.get("papers", [])[:8]:
                    st.markdown(
                        f"- {html.escape(str(row.get('title') or 'untitled'))} "
                        f"({html.escape(str(row.get('author') or 'Unknown'))}, {html.escape(str(row.get('year') or 'n.d.'))})"
                    )
            else:
                st.caption("No matching papers yet.")
            st.markdown("**Methods used most**")
            if map_snapshot.get("methods"):
                for row in map_snapshot.get("methods", [])[:6]:
                    st.markdown(f"- {html.escape(str(row.get('method') or 'method'))}: {int(row.get('count') or 0)}")
            else:
                st.caption("No method pattern matches yet.")
        with rm_col2:
            st.markdown("**Your notes**")
            if map_snapshot.get("notes"):
                for row in map_snapshot.get("notes", [])[:8]:
                    st.markdown(
                        f"- {html.escape(str(row.get('title') or 'note'))} "
                        f"({html.escape(str(row.get('kind') or 'note'))}, {html.escape(str(row.get('project') or 'All'))})"
                    )
            else:
                st.caption("No matching notes yet.")
            st.markdown("**Disagreement clusters**")
            if map_snapshot.get("disagreements"):
                for row in map_snapshot.get("disagreements", [])[:6]:
                    st.markdown(f"- {html.escape(str(row.get('doc') or 'source'))}")
                    cits = row.get("citations") or []
                    if cits:
                        st.caption(" | ".join(html.escape(str(c)) for c in cits[:4]))
            else:
                st.caption("No disagreement clusters surfaced for this concept.")
        if map_snapshot.get("claims"):
            with st.expander("Related claims", expanded=False):
                for row in map_snapshot.get("claims", [])[:10]:
                    st.markdown(f"- {html.escape(str(row.get('claim') or 'claim'))}")
                    cits = row.get("citations") or []
                    if cits:
                        st.caption("Supporting citations: " + " | ".join(html.escape(str(c)) for c in cits[:5]))
        ask_col1, ask_col2 = st.columns(2)
        if ask_col1.button("Ask the Library", key="phd_map_ask_library_btn"):
            st.session_state.queued_prompt = (
                f"/library What does my library say about '{map_query}'? "
                "Include schools of thought, disagreements, methods distribution, and thin evidence."
            )
            st.rerun()
        if ask_col2.button("Compare key sources", key="phd_map_compare_btn"):
            st.session_state.queued_prompt = (
                f"Compare key sources in my files about '{map_query}' by method, data, and findings with citations."
            )
            st.rerun()

    st.markdown("#### Network explorer")
    st.caption("Citation, co-authorship, concept, methods↔data↔findings, and claims-support views.")
    net_col1, net_col2 = st.columns([0.6, 0.4])
    network_focus = net_col1.text_input(
        "Topic filter (optional)",
        key="phd_network_focus",
        placeholder="e.g., turnout, linked fate, context effects",
    ).strip()
    network_type = net_col2.selectbox(
        "Network",
        [
            "Citation network",
            "Co-authorship network",
            "Concept network",
            "Methods ↔ Data ↔ Findings",
            "Claims support network",
        ],
        key="phd_network_type",
    )
    network_snapshot = build_network_snapshot(
        focus=network_focus,
        bibliography_rows=biblio_rows,
        citation_edges=(citation_graph.get("edges") or []) if isinstance(citation_graph, dict) else [],
        glossary_nodes=g_nodes,
        glossary_edges=(glossary_graph.get("edges") or []) if isinstance(glossary_graph, dict) else [],
        claim_rows=claim_rows,
        experiment_rows=experiment_rows,
    )
    if network_type == "Citation network":
        citation_view = network_snapshot.get("citation_network") or {}
        n1, n2, n3 = st.columns(3)
        n1.metric("Paper hubs", len(citation_view.get("paper_hubs") or []))
        n2.metric("Docs in network", len(citation_view.get("doc_nodes") or []))
        n3.metric("Shared citation edges", len(citation_view.get("shared_edges") or []))
        paper_hubs = citation_view.get("paper_hubs") or []
        if paper_hubs:
            st.markdown("**Top cited papers in your library**")
            st.dataframe(paper_hubs[:12], use_container_width=True, hide_index=True)
        shared_edges = citation_view.get("shared_edges") or []
        if shared_edges:
            st.markdown("**Strongest shared-citation links between documents**")
            st.dataframe(shared_edges[:12], use_container_width=True, hide_index=True)
    elif network_type == "Co-authorship network":
        coauthor_view = network_snapshot.get("coauthor_network") or {}
        n1, n2 = st.columns(2)
        n1.metric("Authors", len(coauthor_view.get("author_nodes") or []))
        n2.metric("Co-author edges", len(coauthor_view.get("coauthor_edges") or []))
        if coauthor_view.get("author_nodes"):
            st.markdown("**Most central authors**")
            st.dataframe(coauthor_view.get("author_nodes")[:15], use_container_width=True, hide_index=True)
        if coauthor_view.get("coauthor_edges"):
            st.markdown("**Frequent collaborations**")
            st.dataframe(coauthor_view.get("coauthor_edges")[:15], use_container_width=True, hide_index=True)
    elif network_type == "Concept network":
        concept_view = network_snapshot.get("concept_network") or {}
        n1, n2 = st.columns(2)
        n1.metric("Concept nodes", len(concept_view.get("concept_nodes") or []))
        n2.metric("Concept edges", len(concept_view.get("concept_edges") or []))
        if concept_view.get("concept_nodes"):
            st.markdown("**Top concepts**")
            st.dataframe(concept_view.get("concept_nodes")[:15], use_container_width=True, hide_index=True)
        if concept_view.get("concept_edges"):
            st.markdown("**Strongest concept links**")
            st.dataframe(concept_view.get("concept_edges")[:15], use_container_width=True, hide_index=True)
    elif network_type == "Methods ↔ Data ↔ Findings":
        mdf_view = network_snapshot.get("method_data_finding_network") or {}
        n1, n2 = st.columns(2)
        n1.metric("Method/data pairs", len(mdf_view.get("pairs") or []))
        n2.metric("Evidence rows", len(mdf_view.get("rows") or []))
        if mdf_view.get("pairs"):
            st.markdown("**Most common method/data links**")
            st.dataframe(mdf_view.get("pairs")[:15], use_container_width=True, hide_index=True)
        if mdf_view.get("rows"):
            st.markdown("**Evidence rows**")
            st.dataframe(mdf_view.get("rows")[:12], use_container_width=True, hide_index=True)
    else:
        claims_view = network_snapshot.get("claims_network") or {}
        n1, n2 = st.columns(2)
        n1.metric("Claims", len(claims_view.get("claim_nodes") or []))
        n2.metric("Support edges", len(claims_view.get("support_edges") or []))
        if claims_view.get("claim_nodes"):
            st.markdown("**Claims with support strength**")
            st.dataframe(claims_view.get("claim_nodes")[:15], use_container_width=True, hide_index=True)
        if claims_view.get("support_edges"):
            st.markdown("**Claim support links**")
            st.dataframe(claims_view.get("support_edges")[:15], use_container_width=True, hide_index=True)

    net_actions_col1, net_actions_col2 = st.columns(2)
    if net_actions_col1.button("Ask graph question", key="phd_network_ask_btn"):
        focus_term = network_focus or "my core topics"
        st.session_state.queued_prompt = (
            f"/library Using the network evidence in my files, summarize citation hubs, co-author clusters, "
            f"concept links, and claim support for '{focus_term}' with citations."
        )
        st.rerun()
    net_actions_col2.download_button(
        "Download network snapshot (JSON)",
        data=json.dumps(network_snapshot, indent=2, ensure_ascii=False),
        file_name="edith_network_snapshot.json",
        mime="application/json",
        key="phd_network_download_json",
    )

    st.markdown("#### Glossary lookup")
    term_query = st.text_input("Term or acronym", key="phd_term_query").strip().lower()
    if term_query and g_nodes:
        matches = []
        for n in g_nodes:
            term = str(n.get("term") or "")
            acronyms = [str(x) for x in (n.get("acronyms") or [])]
            blob = " ".join([term] + acronyms).lower()
            if term_query in blob:
                matches.append(n)
            if len(matches) >= 12:
                break
        if matches:
            for n in matches:
                term = n.get("term") or "term"
                definition = n.get("definition") or "(no extracted definition yet)"
                introduced = n.get("introduced_in") or "General"
                st.markdown(f"**{html.escape(term)}**")
                st.caption(f"Introduced in: {introduced}")
                st.markdown(html.escape(definition))
                syn = n.get("synonyms") or []
                acr = n.get("acronyms") or []
                if syn:
                    st.caption("Synonyms: " + ", ".join(html.escape(str(x)) for x in syn[:6]))
                if acr:
                    st.caption("Acronyms: " + ", ".join(html.escape(str(x)) for x in acr[:6]))
                if st.button(f"Ask: What did I mean by {term}?", key=f"ask_term_{stable_hash(term)[:8]}"):
                    st.session_state.queued_prompt = f"What did I mean by '{term}'? Quote exact definitions with citations."
                    st.rerun()
                st.markdown("---")
        else:
            st.info("No glossary matches yet.")

    st.markdown("#### Related-work linker")
    claim_q = st.text_input("Claim/topic to trace supporting sources", key="phd_claim_query").strip().lower()
    if claim_q and citation_graph:
        edges = citation_graph.get("edges") or []
        hit_edges = []
        for e in edges:
            doc = str(e.get("source_doc") or "")
            cit = str(e.get("citation") or "")
            if claim_q in doc.lower() or claim_q in cit.lower():
                hit_edges.append(e)
            if len(hit_edges) >= 60:
                break
        if hit_edges:
            grouped = defaultdict(list)
            for e in hit_edges:
                grouped[str(e.get("source_doc") or "unknown")].append(str(e.get("citation") or "citation"))
            for doc, cites in list(grouped.items())[:20]:
                st.markdown(f"- `{doc}`")
                st.caption("Citations: " + ", ".join(html.escape(c) for c in cites[:8]))
        else:
            st.info("No related-work links matched that query.")

    st.markdown("#### Bibliography database")
    b_col1, b_col2, b_col3, b_col4 = st.columns(4)
    b_topic = b_col1.text_input("Topic / keyword", key="phd_biblio_topic").strip().lower()
    b_y_from = b_col2.text_input("Year from", key="phd_biblio_year_from", max_chars=4).strip()
    b_y_to = b_col3.text_input("Year to", key="phd_biblio_year_to", max_chars=4).strip()
    b_cited_only = b_col4.toggle("Only cited", value=True, key="phd_biblio_cited_only")
    if biblio_rows:
        shown = 0
        try:
            y_from_i = int(b_y_from) if b_y_from else None
        except Exception:
            y_from_i = None
        try:
            y_to_i = int(b_y_to) if b_y_to else None
        except Exception:
            y_to_i = None
        for row in biblio_rows:
            year_txt = str(row.get("year") or "").strip()
            title = str(row.get("title") or "untitled")
            authors = row.get("authors") or []
            venue = str(row.get("venue") or "")
            keywords = row.get("keywords") or []
            mentions = int(row.get("citation_mentions") or 0)
            if b_cited_only and mentions <= 0:
                continue
            try:
                y_i = int(re.search(r"(19|20)\d{2}", year_txt).group(0)) if year_txt else None
            except Exception:
                y_i = None
            if y_from_i and (y_i is None or y_i < y_from_i):
                continue
            if y_to_i and (y_i is None or y_i > y_to_i):
                continue
            hay = " ".join(
                [
                    title,
                    " ".join(str(a) for a in authors),
                    venue,
                    " ".join(str(k) for k in keywords),
                ]
            ).lower()
            if b_topic and b_topic not in hay:
                continue
            lead = authors[0] if authors else "Unknown"
            st.markdown(f"- **{html.escape(title)}**")
            st.caption(f"{html.escape(str(lead))} | {html.escape(year_txt or 'n.d.')} | {html.escape(venue or 'unknown venue')}")
            st.caption(f"Citation mentions: {mentions}")
            if keywords:
                st.caption("Keywords: " + ", ".join(html.escape(str(k)) for k in keywords[:8]))
            shown += 1
            if shown >= 30:
                break
        if shown == 0:
            st.info("No bibliography records matched these filters.")
        elif len(biblio_rows) > shown:
            st.caption(f"Showing {shown} records. Narrow filters for more.")
    else:
        st.info("No bibliography database found yet. Build PhD indexes first.")

    st.markdown("#### Claim inventory")
    claim_filter = st.text_input("Find asserted claim", key="phd_claim_inventory_filter").strip().lower()
    if claim_rows:
        shown = 0
        for row in claim_rows:
            claim_text = str(row.get("claim") or "").strip()
            where = row.get("where") or {}
            doc = str(where.get("doc") or "")
            chapter = str(where.get("chapter") or "")
            if claim_filter:
                hay = f"{claim_text} {doc} {chapter}".lower()
                if claim_filter not in hay:
                    continue
            st.markdown(f"- **{html.escape(claim_text)}**")
            if doc or chapter:
                st.caption(f"{chapter} | {doc}".strip(" |"))
            caveats = row.get("caveats") or []
            if caveats:
                st.caption("Caveats: " + " | ".join(html.escape(str(c)) for c in caveats[:2]))
            support_cites = row.get("support_citations") or []
            if support_cites:
                st.caption("Supporting citations: " + " | ".join(html.escape(str(c)) for c in support_cites[:6]))
            shown += 1
            if shown >= 25:
                break
        if len(claim_rows) > shown:
            st.caption(f"Showing {shown} claims. Narrow filter for more.")
    else:
        st.info("No claim inventory found yet. Build PhD indexes first.")

    st.markdown("#### Experiment ledger")
    exp_filter = st.text_input("Experiment/run search", key="phd_experiment_filter").strip().lower()
    if experiment_rows:
        shown = 0
        for row in experiment_rows:
            name = str(row.get("experiment") or "experiment")
            result = str(row.get("result") or "").strip()
            doc = str(row.get("source_doc") or "")
            date_txt = str(row.get("date") or "")
            chapter = str(row.get("chapter") or "")
            hay = f"{name} {result} {doc} {chapter} {date_txt}".lower()
            if exp_filter and exp_filter not in hay:
                continue
            st.markdown(f"- **{html.escape(name)}**")
            meta_bits = [x for x in [chapter, doc, date_txt] if x]
            if meta_bits:
                st.caption(" | ".join(html.escape(x) for x in meta_bits))
            params = row.get("parameters") or []
            if params:
                short_params = ", ".join(
                    f"{html.escape(str(p.get('key') or 'k'))}={html.escape(str(p.get('value') or 'v'))}"
                    for p in params[:5]
                )
                st.caption("Params: " + short_params)
            if result:
                st.caption("Result: " + html.escape(result[:220]))
            shown += 1
            if shown >= 25:
                break
        if len(experiment_rows) > shown:
            st.caption(f"Showing {shown} experiment rows. Narrow filter for more.")
    else:
        st.info("No experiment ledger rows found yet.")

    st.markdown("#### Entity timeline")
    timeline_entities = entity_timeline.get("entities") or []
    timeline_events = entity_timeline.get("events") or []
    ent_query = st.text_input("Entity (dataset/system/collaborator/version)", key="phd_entity_query").strip().lower()
    if timeline_entities:
        shown = 0
        for ent in timeline_entities:
            name = str(ent.get("entity") or "")
            etype = str(ent.get("entity_type") or "entity")
            intro_year = str(ent.get("introduced_year") or "")
            intro_doc = str(ent.get("introduced_in") or "")
            if ent_query:
                hay = f"{name} {etype} {intro_year} {intro_doc}".lower()
                if ent_query not in hay:
                    continue
            st.markdown(f"- **{html.escape(name)}**")
            st.caption(
                f"type={html.escape(etype)} | introduced={html.escape(intro_year or 'unknown')} | "
                f"source={html.escape(intro_doc or 'unknown')} | events={int(ent.get('event_count') or 0)}"
            )
            if st.button(f"When did I introduce {name}?", key=f"entity_when_{stable_hash(name)[:10]}"):
                st.session_state.queued_prompt = (
                    f"When did I introduce '{name}'? Answer from my files only and cite exact evidence."
                )
                st.rerun()
            shown += 1
            if shown >= 25:
                break
        if shown == 0:
            st.info("No entity timeline matches found.")
    else:
        st.info("No entity timeline found yet. Build PhD indexes first.")

    st.markdown("#### Nightly index health")
    totals = (index_health_report.get("totals") or {}) if isinstance(index_health_report, dict) else {}
    if totals:
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("New files", int(totals.get("new_files", 0)))
        h2.metric("Missing metadata", int(totals.get("missing_metadata_docs", 0)))
        h3.metric("Duplicates", int(totals.get("duplicate_families", 0)))
        h4.metric("Never retrieved", int(totals.get("never_retrieved_docs", 0)))
    else:
        st.caption("No index health report found yet.")

    st.markdown("#### Reproducible snapshots")
    latest_snapshot = SNAPSHOT_DIR / "latest_snapshot.json"
    if latest_snapshot.exists():
        try:
            snap = json.loads(latest_snapshot.read_text(encoding="utf-8"))
        except Exception:
            snap = {}
        if isinstance(snap, dict) and snap:
            st.caption(f"Generated: {snap.get('generated_at', 'unknown')}")
            st.caption(f"Corpus hash: {snap.get('corpus_hash', '')}")
            if snap.get("chroma_hash"):
                st.caption(f"Chroma hash: {snap.get('chroma_hash')}")
            st.caption(f"Files: {int(snap.get('docs_file_count') or 0)}")
        else:
            st.caption("Latest snapshot file is unreadable.")
    else:
        st.caption("No corpus snapshot found yet.")

    st.markdown("#### Chapter ground-truth anchors")
    chapter_keys = sorted(list(chapters.keys()))
    if chapter_keys:
        selected_chapter = st.selectbox("Chapter", chapter_keys, key="phd_chapter_pick")
        anchor = chapters.get(selected_chapter) or {}
        for label, key in [
            ("Thesis", "thesis"),
            ("Contributions", "contributions"),
            ("Assumptions / Limitations", "assumptions_limitations"),
            ("Key Results", "key_results"),
            ("Open Questions", "open_questions"),
        ]:
            st.markdown(f"**{label}**")
            rows = anchor.get(key) or []
            if rows:
                for r in rows:
                    st.markdown(f"- {html.escape(str(r))}")
            else:
                st.caption("No extracted lines.")
    else:
        st.info("No chapter anchors found yet. Build PhD indexes first.")


with doctor_tab:
    st.markdown("### Health Doctor")
    st.caption("One-click diagnostics and safe fixes for index, watcher, and connectivity.")
    live_now = run_live_health_checks(
        active_model=ACTIVE_MODEL,
        retrieval_backend=st.session_state.get("retrieval_backend", RETRIEVAL_BACKEND_DEFAULT),
    )
    quick_cols = st.columns(4)
    quick_cols[0].metric("Indexed docs", len(index_rows))
    wstat = watcher_status()
    quick_cols[1].metric("Watcher", "Running" if wstat.get("running") else "Stopped")
    quick_cols[2].metric("Setup", "Complete" if not setup_required() else "Needs setup")
    quick_cols[3].metric("Model", ACTIVE_MODEL or "n/a")

    for item in live_now:
        status_txt = "OK" if item.get("ok") else "Fail"
        st.caption(f"{status_txt}: {item.get('name')} — {item.get('detail')}")

    if st.button("Run Health Doctor (check + safe fixes)", key="run_health_doctor", disabled=not can_run_index):
        with st.spinner("Running doctor..."):
            st.session_state.doctor_report = run_health_doctor(
                active_model=ACTIVE_MODEL,
                retrieval_backend=st.session_state.get("retrieval_backend", RETRIEVAL_BACKEND_DEFAULT),
                source_mode=st.session_state.get("source_mode", SOURCE_MODE_DEFAULT),
                index_count=len(index_rows),
            )

    doctor_report = st.session_state.doctor_report or {}
    if doctor_report:
        st.markdown("#### Doctor Report")
        for row in doctor_report.get("checks", []):
            status_txt = "OK" if row.get("ok") else "Fail"
            st.caption(f"{status_txt}: {row.get('name')} — {row.get('detail')}")
        if doctor_report.get("actions"):
            st.markdown("#### Fix Actions")
            for row in doctor_report.get("actions", []):
                status_txt = "OK" if row.get("ok") else "Fail"
                st.caption(f"{status_txt}: {row.get('name')} — {row.get('detail')}")
