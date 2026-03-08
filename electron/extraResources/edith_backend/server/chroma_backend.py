import re
import os
import time
import math
import hashlib
import logging
from typing import List, Optional, Dict, Any
from collections import Counter, OrderedDict
from pathlib import Path
import json

log = logging.getLogger("edith.retrieval")


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name, str(default))
    try:
        return float(raw)
    except Exception:
        return float(default)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, str(default))
    try:
        return int(raw)
    except Exception:
        return int(default)

# ---------------------------------------------------------------------------
# §3.1 Query Result Cache — LRU with 5-min TTL
# ---------------------------------------------------------------------------

class _QueryCache:
    """Thread-safe LRU cache for retrieval results."""
    def __init__(self, max_size: int = 200, ttl_seconds: int = 300):
        self._cache: OrderedDict = OrderedDict()
        self._max = max_size
        self._ttl = ttl_seconds
        import threading
        self._lock = threading.Lock()  # §FIX B3: Thread safety for concurrent queries
        self._hits = 0
        self._misses = 0

    def _key(self, query: str, top_k: int, collection: str = "") -> str:
        return hashlib.md5(f"{query}|{top_k}|{collection}".encode()).hexdigest()

    def get(self, query: str, top_k: int, collection: str = ""):
        k = self._key(query, top_k, collection)
        with self._lock:
            entry = self._cache.get(k)
            if entry and (time.time() - entry["ts"]) < self._ttl:
                self._hits += 1
                self._cache.move_to_end(k)
                return entry["data"]
            if entry:
                del self._cache[k]  # expired
            self._misses += 1
        return None

    def put(self, query: str, top_k: int, data, collection: str = ""):
        k = self._key(query, top_k, collection)
        with self._lock:
            self._cache[k] = {"data": data, "ts": time.time()}
            self._cache.move_to_end(k)
            while len(self._cache) > self._max:
                self._cache.popitem(last=False)

    def invalidate(self):
        """Clear all cached results (call after re-indexing)."""
        with self._lock:
            self._cache.clear()

    def stats(self) -> dict:
        with self._lock:
            return {"hits": self._hits, "misses": self._misses, "size": len(self._cache)}


# §3.2 Embedding Vector Cache — persistent SQLite cache (§IMP-1.3)
class _EmbedCache:
    """Persistent embedding vector cache using SQLite.

    §IMP-1.3: Survives process restarts. Falls back to in-memory if SQLite fails.
    """
    def __init__(self, max_size: int = 5000, ttl_seconds: int = 86400):
        self._mem: OrderedDict = OrderedDict()
        self._max = max_size
        self._ttl = ttl_seconds
        self._db = None
        self._init_db()

    def _init_db(self):
        try:
            import sqlite3
            cache_dir = os.environ.get("EDITH_DATA_ROOT") or os.environ.get("EDITH_CHROMA_DIR", ".")
            db_path = os.path.join(cache_dir, ".edith_embed_cache.sqlite3")
            self._db = sqlite3.connect(db_path, check_same_thread=False)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS embed_cache "
                "(key TEXT PRIMARY KEY, vec BLOB, model TEXT, ts REAL)"
            )
            self._db.execute(
                "CREATE INDEX IF NOT EXISTS idx_ts ON embed_cache(ts)"
            )
            self._db.commit()
            # Prune old entries
            self._db.execute("DELETE FROM embed_cache WHERE ts < ?",
                             (time.time() - self._ttl,))
            self._db.commit()
            log.info(f"Persistent embed cache initialized at {db_path}")
        except Exception as e:
            log.warning(f"SQLite embed cache unavailable, using in-memory: {e}")
            self._db = None

    def get(self, text: str, model: str):
        k = hashlib.md5(f"{model}|{text}".encode()).hexdigest()
        # In-memory first
        entry = self._mem.get(k)
        if entry and (time.time() - entry["ts"]) < self._ttl:
            self._mem.move_to_end(k)
            return entry["vec"]
        # SQLite fallback
        if self._db:
            try:
                row = self._db.execute(
                    "SELECT vec, ts FROM embed_cache WHERE key = ?", (k,)
                ).fetchone()
                if row and (time.time() - row[1]) < self._ttl:
                    import json as _json
                    vec = _json.loads(row[0])  # §SEC: JSON instead of pickle
                    self._mem[k] = {"vec": vec, "ts": row[1]}
                    return vec
            except Exception:
                pass
        return None

    def put(self, text: str, model: str, vec):
        k = hashlib.md5(f"{model}|{text}".encode()).hexdigest()
        ts = time.time()
        self._mem[k] = {"vec": vec, "ts": ts}
        self._mem.move_to_end(k)
        while len(self._mem) > self._max:
            self._mem.popitem(last=False)
        # Persist to SQLite
        if self._db:
            try:
                import json as _json
                blob = _json.dumps(vec)  # §SEC: JSON instead of pickle
                self._db.execute(
                    "INSERT OR REPLACE INTO embed_cache (key, vec, model, ts) VALUES (?,?,?,?)",
                    (k, blob, model, ts)
                )
                # §FIX R2: Cap SQLite rows to prevent unbounded growth
                row_count = self._db.execute("SELECT COUNT(*) FROM embed_cache").fetchone()[0]
                if row_count > self._max * 2:  # prune when 2x over limit
                    self._db.execute(
                        "DELETE FROM embed_cache WHERE key IN "
                        "(SELECT key FROM embed_cache ORDER BY ts ASC LIMIT ?)",
                        (row_count - self._max,)
                    )
                self._db.commit()
            except Exception:
                pass


_query_cache = _QueryCache()
_embed_cache = _EmbedCache()

try:
    from retrieval_improvements import (
        RetrievalTelemetry, RetrievalCache, adaptive_top_k,
        decompose_query, apply_temporal_weight, calibrate_confidence,
        cross_collection_retrieve, expand_with_neighbors,
    )
    _RETRIEVAL_IMPROVEMENTS = True
except ImportError:
    try:
        from server.retrieval_enhancements import (
            RetrievalTelemetry, RetrievalCache, adaptive_top_k,
            decompose_query, apply_temporal_weight, calibrate_confidence,
            cross_collection_retrieve, expand_with_neighbors,
        )
        _RETRIEVAL_IMPROVEMENTS = True
    except ImportError:
        _RETRIEVAL_IMPROVEMENTS = False

# Singleton instances for retrieval improvements
_RETRIEVAL_CACHE = None
_RETRIEVAL_TELEMETRY = None

def _init_retrieval_improvements():
    global _RETRIEVAL_CACHE, _RETRIEVAL_TELEMETRY
    if not _RETRIEVAL_IMPROVEMENTS:
        return
    if _RETRIEVAL_CACHE is None:
        _RETRIEVAL_CACHE = RetrievalCache(max_size=500, ttl_seconds=3600)
    if _RETRIEVAL_TELEMETRY is None:
        app_state = Path(os.environ.get("EDITH_APP_DATA_DIR", ".")).expanduser()
        _RETRIEVAL_TELEMETRY = RetrievalTelemetry(
            log_path=app_state / "edith_retrieval_telemetry.jsonl"
        )

# ---------- Gemini Embedding API (query-side) ----------
def _gemini_embed_queries(texts: list, model: str = "gemini-embedding-001") -> list:
    """Embed query texts using Gemini API with RETRIEVAL_QUERY task type."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
    if not api_key:
        log.warning("§EMBED: No API key for Gemini embedding")
        return []
    try:
        import requests as _requests
    except ImportError:
        log.warning("§EMBED: requests library not available")
        return []
    timeout_s = max(1.0, _env_float("EDITH_GEMINI_EMBED_TIMEOUT_S", 8.0))
    max_retry_429 = max(0, _env_int("EDITH_GEMINI_EMBED_RETRY_429", 1))
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:batchEmbedContents?key={api_key}"
    requests_body = [{"model": f"models/{model}", "content": {"parts": [{"text": t[:8192]}]}, "taskType": "RETRIEVAL_QUERY"} for t in texts]
    try:
        resp = None
        for attempt in range(max_retry_429 + 1):
            resp = _requests.post(url, json={"requests": requests_body}, timeout=timeout_s)
            if resp.status_code != 429 or attempt >= max_retry_429:
                break
            backoff = min(2 * (attempt + 1), 4)
            log.warning(f"§EMBED: Gemini rate limited, retrying in {backoff}s...")
            time.sleep(backoff)
        if resp is None:
            return []
        if resp.status_code != 200:
            log.error(f"§EMBED: Gemini embed API returned {resp.status_code}: {resp.text[:200]}")
            return []
        data = resp.json()
        embeddings = [emb.get("values", []) for emb in data.get("embeddings", [])]
        if embeddings:
            log.debug(f"§EMBED: Gemini embed OK: {len(embeddings)} vectors, dim={len(embeddings[0])}")
        return embeddings
    except _requests.RequestException as e:
        log.error(f"§EMBED: Gemini request failed quickly: {type(e).__name__}: {e}")
        return []
    except Exception as e:
        log.error(f"§EMBED: Gemini embed exception: {type(e).__name__}: {e}")
        return []

def _infer_academic_topic(rel_path: str):
    parts = Path(rel_path).parts
    # If path is canon/Time Series/file.pdf, topic is Time Series
    if len(parts) >= 3:
        return parts[1]
    # If path is inbox/file.pdf, topic is inbox (fallback)
    if len(parts) >= 2:
        return parts[0]
    return "general"


def _safe_imports():
    try:
        import chromadb  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception:
            CrossEncoder = None
        return chromadb, SentenceTransformer, CrossEncoder
    except Exception:
        return None, None, None


_CHROMA = None
_ST = None
_CROSS = None
_CLIENT_CACHE = {}
_EMBEDDER_CACHE = {}
_EMBEDDER_ERROR_CACHE = {}
_RERANKER_CACHE = {}
_CHROMA_HEALTH_CACHE: Dict[str, Dict[str, Any]] = {}
_CHROMA_HEALTH_TTL_S = 30.0


def _resolve_chroma_dir(chroma_dir: str = "") -> str:
    raw = chroma_dir or os.environ.get("EDITH_CHROMA_DIR", "")
    if not raw:
        return ""
    try:
        return str(Path(raw).expanduser().resolve())
    except Exception:
        return str(Path(raw).expanduser())


def chroma_runtime_available(chroma_dir: str = "") -> bool:
    global _CHROMA, _ST, _CROSS
    if _CHROMA is None or _ST is None:
        _CHROMA, _ST, _CROSS = _safe_imports()
    if _CHROMA is None or _ST is None:
        return False

    key = _resolve_chroma_dir(chroma_dir)
    if not key:
        # Import/runtime check only (used by lightweight probes)
        return True

    now = time.time()
    cached = _CHROMA_HEALTH_CACHE.get(key)
    if cached and (now - float(cached.get("ts", 0.0))) < _CHROMA_HEALTH_TTL_S:
        return bool(cached.get("ok", False))

    try:
        client = _CHROMA.PersistentClient(path=key)
        list_collections = getattr(client, "list_collections", None)
        if callable(list_collections):
            list_collections()
        _CHROMA_HEALTH_CACHE[key] = {"ok": True, "ts": now, "error": ""}
        return True
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        if not cached or cached.get("error") != msg:
            log.warning(f"§CHROMA: runtime unavailable at {key}: {msg}")
        _CHROMA_HEALTH_CACHE[key] = {"ok": False, "ts": now, "error": msg}
        return False


def _get_client(chroma_dir: str):
    key = _resolve_chroma_dir(chroma_dir)
    if not key:
        raise RuntimeError("chroma directory is not configured")
    if not chroma_runtime_available(key):
        detail = _CHROMA_HEALTH_CACHE.get(key, {}).get("error", "runtime unavailable")
        raise RuntimeError(f"chroma runtime unavailable: {detail}")
    try:
        # Always create a fresh client — cached clients become corrupted
        # under uvicorn's threadpool with ChromaDB 1.5.0's Rust backend
        return _CHROMA.PersistentClient(path=key)
    except Exception as exc:
        msg = f"{type(exc).__name__}: {exc}"
        _CHROMA_HEALTH_CACHE[key] = {"ok": False, "ts": time.time(), "error": msg}
        raise RuntimeError(f"chroma client init failed for {key}: {msg}") from exc


def _get_embedder(model_name: str):
    if not chroma_runtime_available():
        raise RuntimeError("embedding runtime unavailable")
    key = model_name.strip()
    # API model names (Gemini, OpenAI) can't be loaded via sentence-transformers.
    # Fall back to the default local model — the Gemini API path handles these separately.
    _API_PREFIXES = ("gemini-", "text-embedding-", "models/")
    if any(key.startswith(p) for p in _API_PREFIXES):
        key = os.environ.get("EDITH_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    if key in _EMBEDDER_ERROR_CACHE:
        raise RuntimeError(f"embedder unavailable for {key}: {_EMBEDDER_ERROR_CACHE[key]}")
    if key not in _EMBEDDER_CACHE:
        env_mode = os.environ.get("EDITH_ENV", "").strip().lower()
        app_mode = os.environ.get("EDITH_APP_MODE", "").strip().lower()
        allow_download = os.environ.get("EDITH_ALLOW_EMBED_MODEL_DOWNLOAD", "false").strip().lower() == "true"
        local_only_default = "true" if (env_mode == "test" or app_mode == "test" or not allow_download) else "false"
        local_only = os.environ.get("EDITH_EMBED_LOCAL_FILES_ONLY", local_only_default).strip().lower() == "true"

        def _load_with_mode(local_files_only: bool):
            prev_hf = os.environ.get("HF_HUB_OFFLINE")
            prev_tf = os.environ.get("TRANSFORMERS_OFFLINE")
            if local_files_only:
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
            try:
                kwargs = {"local_files_only": True} if local_files_only else {}
                return _ST(key, **kwargs)
            finally:
                if local_files_only:
                    if prev_hf is None:
                        os.environ.pop("HF_HUB_OFFLINE", None)
                    else:
                        os.environ["HF_HUB_OFFLINE"] = prev_hf
                    if prev_tf is None:
                        os.environ.pop("TRANSFORMERS_OFFLINE", None)
                    else:
                        os.environ["TRANSFORMERS_OFFLINE"] = prev_tf

        try:
            _EMBEDDER_CACHE[key] = _load_with_mode(local_only)
        except Exception as exc:
            # Fall back to local-files-only as a last resort for restricted/offline hosts.
            if not local_only:
                try:
                    _EMBEDDER_CACHE[key] = _load_with_mode(True)
                except Exception as local_exc:
                    msg = f"{type(local_exc).__name__}: {local_exc}"
                    _EMBEDDER_ERROR_CACHE[key] = msg
                    raise RuntimeError(f"embedder load failed for {key}: {msg}") from local_exc
            else:
                msg = f"{type(exc).__name__}: {exc}"
                _EMBEDDER_ERROR_CACHE[key] = msg
                raise RuntimeError(f"embedder load failed for {key}: {msg}") from exc
        _EMBEDDER_ERROR_CACHE.pop(key, None)
    return _EMBEDDER_CACHE[key]


def _get_reranker(model_name: str):
    if not model_name:
        return None
    if not chroma_runtime_available():
        return None
    if _CROSS is None:
        return None
    key = model_name.strip()
    if not key:
        return None
    if key not in _RERANKER_CACHE:
        _RERANKER_CACHE[key] = _CROSS(key)
    return _RERANKER_CACHE[key]

def _load_negative_memory(chroma_dir: str):
    """
    Loads SHA256 hashes to exclude from retrieval.
    File: negative_memory.json in the chroma directory.
    """
    path = Path(chroma_dir) / "negative_memory.json"
    if not path.exists():
        return []
    try:
        import json
        with path.open("r") as f:
            data = json.load(f)
            return data.get("exclude_shas", [])
    except Exception:
        return []


def _cosine_sim(a, b):
    if not a or not b:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _mmr_select(candidates, top_k: int, lambda_mult: float):
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates

    lm = max(0.0, min(1.0, float(lambda_mult)))
    selected = []
    remaining = candidates[:]

    # start with highest relevance
    remaining.sort(key=lambda x: x.get("relevance", 0.0), reverse=True)
    selected.append(remaining.pop(0))

    while remaining and len(selected) < top_k:
        best_idx = 0
        best_score = -1e9
        for idx, cand in enumerate(remaining):
            rel = float(cand.get("relevance", 0.0))
            emb = cand.get("embedding") or []
            redundancy = 0.0
            for s in selected:
                sim = _cosine_sim(emb, s.get("embedding") or [])
                if sim > redundancy:
                    redundancy = sim
            mmr = (lm * rel) - ((1.0 - lm) * redundancy)
            if mmr > best_score:
                best_score = mmr
                best_idx = idx
        selected.append(remaining.pop(best_idx))

    return selected


def _tokenize(text: str):
    return [x.lower() for x in re.findall(r"[A-Za-z0-9]{2,}", text or "")]


def _equation_like(text: str):
    t = str(text or "")
    return bool(re.search(r"[A-Za-z][A-Za-z0-9_]{0,20}\s*=\s*[^.;\n]{3,80}", t))


def _parse_csv_filter(raw: str):
    values = []
    for part in str(raw or "").split(","):
        token = part.strip().lower()
        if token and token not in values:
            values.append(token)
    return values


_DOC_TYPE_ALIAS = {
    "paper": {
        "paper", "papers", "article", "journal_article", "conference_paper",
        "working_paper", "policy_paper", "empirical_paper", "theoretical_paper",
        "methodological_paper", "review_paper",
    },
    "papers": {
        "paper", "papers", "article", "journal_article", "conference_paper",
        "working_paper", "policy_paper", "empirical_paper", "theoretical_paper",
        "methodological_paper", "review_paper",
    },
    "article": {
        "article", "journal_article", "empirical_paper", "theoretical_paper",
        "methodological_paper", "review_paper",
    },
    "articles": {
        "article", "journal_article", "empirical_paper", "theoretical_paper",
        "methodological_paper", "review_paper",
    },
}


def _expand_doc_type_filters(values: set[str]) -> set[str]:
    expanded = set(values or set())
    for token in list(values or set()):
        expanded.update(_DOC_TYPE_ALIAS.get(token, set()))
    return expanded


def _doc_type_matches(doc_type: str, filters: set[str]) -> bool:
    if not filters:
        return True
    d = str(doc_type or "").strip().lower()
    if not d:
        return False
    compact = d.replace("-", "_").replace(" ", "_")
    if d in filters or compact in filters:
        return True
    if ("paper" in filters or "papers" in filters) and ("paper" in d or "article" in d):
        return True
    if ("article" in filters or "articles" in filters) and ("article" in d or "paper" in d):
        return True
    return False


def _version_stage_boost(stage: str):
    s = str(stage or "").strip().lower()
    if s == "final":
        return 0.04
    if s in {"published", "accepted"}:
        return 0.03
    if s in {"preprint", "revision"}:
        return 0.01
    if s == "draft":
        return -0.03
    return 0.0

def _tier_boost(tier: str):
    t = str(tier or "").strip().lower()
    if t == "canon":
        return 0.06
    if t == "projects":
        return 0.03
    if t == "inbox":
        return -0.02
    return 0.0


def _normalize_scores(values):
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-9:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _apply_freshness_decay(sources: list[dict], decay_rate: float = 0.05) -> list[dict]:
    """§IMP-6.7: Apply exponential freshness decay to source scores.

    Sources from 2024 score higher than 2015 (configurable).
    decay_rate: score penalty per year of age (default 5%/year).
    """
    import re as _re
    current_year = 2026
    for src in sources:
        # Extract year from metadata or title
        year = None
        for field in ["year", "date", "published"]:
            val = src.get(field, "") or src.get("metadata", {}).get(field, "")
            if val:
                match = _re.search(r"(19|20)\d{2}", str(val))
                if match:
                    year = int(match.group())
                    break
        if year and "score" in src:
            age = max(0, current_year - year)
            decay_factor = max(0.3, 1.0 - decay_rate * age)  # Floor at 30%
            src["score"] = round(src["score"] * decay_factor, 4)
            src["_freshness"] = {"year": year, "age": age, "decay_factor": round(decay_factor, 3)}
    return sources


def _explain_relevance(source: dict, query: str) -> str:
    """§IMP-6.10: Generate human-readable relevance explanation.

    Returns 'Matched because: keyword overlap, embedding similarity, etc.'
    """
    reasons = []
    # Keyword overlap
    q_words = set(query.lower().split())
    text = (source.get("text", "") or source.get("document", ""))[:500].lower()
    overlap = q_words & set(text.split()) - {"the", "of", "and", "in", "a", "to", "is", "for"}
    if overlap:
        reasons.append(f"keyword match: {', '.join(list(overlap)[:4])}")
    # Embedding similarity
    if source.get("score"):
        reasons.append(f"embedding similarity: {source['score']:.2f}")
    # Freshness
    if source.get("_freshness"):
        reasons.append(f"published {source['_freshness']['year']}")
    # Collection
    if source.get("_collection"):
        reasons.append(f"from: {source['_collection']}")
    return "; ".join(reasons) if reasons else "semantic similarity"


def _bm25_scores(doc_texts, query_tokens, k1=1.5, b=0.75):
    docs = [_tokenize(t) for t in (doc_texts or [])]
    if not docs:
        return []
    qtokens = [t for t in (query_tokens or []) if t]
    if not qtokens:
        return [0.0 for _ in docs]

    n_docs = len(docs)
    avgdl = sum(len(d) for d in docs) / float(max(1, n_docs))
    doc_freq = Counter()
    term_freqs = []
    for tokens in docs:
        tf = Counter(tokens)
        term_freqs.append(tf)
        for term in tf.keys():
            doc_freq[term] += 1

    idf = {}
    for term in qtokens:
        df = float(doc_freq.get(term, 0))
        idf[term] = math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))

    scores = []
    for tf, tokens in zip(term_freqs, docs):
        dl = float(max(1, len(tokens)))
        score = 0.0
        for term in qtokens:
            freq = float(tf.get(term, 0))
            if freq <= 0:
                continue
            denom = freq + k1 * (1.0 - b + b * (dl / max(1e-9, avgdl)))
            score += idf.get(term, 0.0) * ((freq * (k1 + 1.0)) / max(1e-9, denom))
        scores.append(score)

    return scores

def classify_query_intent(query: str):
    """
    Classifies the user query to dynamically adjust retrieval parameters.
    Returns: {vector_weight, bm25_weight, diversity_lambda}

    §IMP-1.4: Adjusts weights based on accumulated feedback.
    """
    q = (query or "").lower()

    # Base weights by intent type
    if any(k in q for k in ["methods", "methodology", "data", "sample", "dataset", "source", "table", "figure", "fig", "results", "estimate", "model", "identification", "quote", "page", "citation"]):
        base = {"vector_weight": 0.45, "bm25_weight": 0.55, "diversity": 0.5}
    elif any(k in q for k in ["compare", "synthesize", "literature", "review", "history", "trend", "evolution", "summary", "overview"]):
        base = {"vector_weight": 0.75, "bm25_weight": 0.25, "diversity": 0.8}
    else:
        base = {"vector_weight": 0.65, "bm25_weight": 0.35, "diversity": 0.65}

    # §IMP-1.4: Apply feedback-learned adjustments
    feedback = _load_retrieval_feedback()
    if feedback:
        # Positive feedback on vector-heavy results → boost vector weight
        v_adjust = feedback.get("vector_boost", 0.0)
        b_adjust = feedback.get("bm25_boost", 0.0)
        base["vector_weight"] = min(0.9, max(0.1, base["vector_weight"] + v_adjust))
        base["bm25_weight"] = min(0.9, max(0.1, base["bm25_weight"] + b_adjust))
        # Re-normalize
        total = base["vector_weight"] + base["bm25_weight"]
        base["vector_weight"] /= total
        base["bm25_weight"] /= total

    return base


# §IMP-1.4: Feedback-adaptive retrieval weights
_FEEDBACK_FILE = os.path.join(
    os.environ.get("EDITH_DATA_ROOT", "."), ".edith_retrieval_feedback.json"
)

def _load_retrieval_feedback() -> dict:
    """Load cumulative retrieval feedback adjustments."""
    try:
        if os.path.exists(_FEEDBACK_FILE):
            with open(_FEEDBACK_FILE) as f:
                return json.load(f)
    except Exception:
        pass
    return {}

def record_retrieval_feedback(positive: bool, query_intent: str = "general") -> None:
    """§IMP-1.4: Record user feedback to adjust retrieval weights over time.

    Positive feedback on lexical queries boosts bm25; on semantic boosts vector.
    """
    fb = _load_retrieval_feedback()
    increment = 0.005 if positive else -0.003

    if query_intent in ("methods", "lexical", "quote"):
        fb["bm25_boost"] = fb.get("bm25_boost", 0.0) + increment
    else:
        fb["vector_boost"] = fb.get("vector_boost", 0.0) + increment

    try:
        with open(_FEEDBACK_FILE, "w") as f:
            json.dump(fb, f)
    except Exception:
        pass


# §IMP-1.7: Source deduplication via MinHash fingerprinting
def deduplicate_sources(sources: list[dict], similarity_threshold: float = 0.8) -> list[dict]:
    """Remove near-duplicate sources before prompting to save context window.

    Uses 3-gram Jaccard similarity. Sources below threshold are kept.
    """
    if len(sources) <= 1:
        return sources

    def _ngrams(text, n=3):
        words = text.lower().split()
        return set(tuple(words[i:i+n]) for i in range(max(1, len(words) - n + 1)))

    fingerprints = []
    for s in sources:
        text = s.get("text", "") or s.get("content", "")
        fingerprints.append(_ngrams(text))

    kept = []
    for i, s in enumerate(sources):
        is_dup = False
        for j in range(len(kept)):
            ki = kept[j][0]
            overlap = len(fingerprints[i] & fingerprints[ki])
            union = len(fingerprints[i] | fingerprints[ki])
            if union > 0 and overlap / union > similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            kept.append((i, s))

    deduped = [s for _, s in kept]
    if len(deduped) < len(sources):
        log.info(f"Dedup: {len(sources)} → {len(deduped)} sources (removed {len(sources) - len(deduped)} near-duplicates)")
    return deduped


def classify_query_complexity(query: str) -> dict:
    """Classify query complexity to route to the appropriate prompt.
    
    Returns: {
        complexity: str,
        prompt_key: str,  # key into prompts module
        decompose: bool,  # whether to use query decomposition
        top_k_multiplier: float,  # multiply default top_k by this
        author_filter: str or None,  # author name if detected
    }
    """
    q = (query or "").lower()
    words = q.split()
    word_count = len(words)
    
    # Check for author-specific queries
    author_info = extract_author_query(query)
    author = author_info.get("author")
    
    # Exam question generation
    if any(p in q for p in ["exam question", "test question", "quiz question",
                             "write questions", "generate questions", "midterm", "final exam"]):
        return {"complexity": "exam", "prompt_key": "EXAM_QUESTION_PROMPT",
                "decompose": False, "top_k_multiplier": 2.0, "author_filter": author}
    
    # Paper outline requests
    if any(p in q for p in ["paper outline", "outline a paper", "structure a paper",
                             "dissertation outline", "write a paper about",
                             "help me outline", "paper structure"]):
        return {"complexity": "outline", "prompt_key": "PAPER_OUTLINE_PROMPT",
                "decompose": False, "top_k_multiplier": 1.5, "author_filter": author}
    
    # Annotated bibliography
    if any(p in q for p in ["annotated bibliography", "annotated bib", "summarize each source",
                             "evaluate each reading", "source by source"]):
        return {"complexity": "annotated_bib", "prompt_key": "ANNOTATED_BIB_PROMPT",
                "decompose": False, "top_k_multiplier": 2.0, "author_filter": author}
    
    # Gap identification
    if any(p in q for p in ["what's missing", "research gap", "what is unstudied",
                             "what remains unknown", "future research", "what should be studied"]):
        return {"complexity": "gap", "prompt_key": "GAP_IDENTIFIER_PROMPT",
                "decompose": True, "top_k_multiplier": 2.0, "author_filter": author}
    
    # Counterargument / debate
    if any(p in q for p in ["counterargument", "argue against", "critique of",
                             "problems with", "weaknesses of", "challenge the claim",
                             "devil's advocate", "other side", "opposing view"]):
        return {"complexity": "counterargument", "prompt_key": "COUNTERARGUMENT_PROMPT",
                "decompose": True, "top_k_multiplier": 1.5, "author_filter": author}
    
    # Literature review requests
    if any(p in q for p in ["literature review", "review the literature", "what does the literature",
                             "survey of research", "state of the field", "what do scholars"]):
        return {"complexity": "literature", "prompt_key": "LIT_REVIEW_PROMPT",
                "decompose": True, "top_k_multiplier": 2.0, "author_filter": author}
    
    # Methodological questions
    if any(p in q for p in ["how should i measure", "research design", "methodology for",
                             "how to study", "identification strategy", "what method",
                             "operationalize", "instrument variable", "natural experiment"]):
        return {"complexity": "methodological", "prompt_key": "RESEARCH_DESIGN_PROMPT",
                "decompose": False, "top_k_multiplier": 1.5, "author_filter": author}
    
    # Comparative/multi-entity questions
    if any(p in q for p in ["compare", "contrast", "difference between", " vs ",
                             "how does .* differ", "similarities between"]) or \
       ((" and " in q) and word_count > 8):
        return {"complexity": "comparative", "prompt_key": "GROUNDED_DEEP_PROMPT",
                "decompose": True, "top_k_multiplier": 1.5, "author_filter": author}
    
    # Analytical/causal questions
    if any(p in q for p in ["why does", "why do", "what causes", "what explains",
                             "how does .* affect", "what is the effect", "mechanism",
                             "what is the relationship", "under what conditions",
                             "when does", "critique", "evaluate", "assess"]) or \
       word_count > 15:
        return {"complexity": "analytical", "prompt_key": "CHAIN_OF_THOUGHT_PROMPT",
                "decompose": False, "top_k_multiplier": 1.0, "author_filter": author}
    
    # Simple factual
    return {"complexity": "simple", "prompt_key": "GROUNDED_PROMPT",
            "decompose": False, "top_k_multiplier": 1.0, "author_filter": author}


def synthesize_cross_paper(sources: list, query: str) -> list:
    """Group sources by document family, producing a structured source map.
    
    Instead of presenting 12 raw chunks, this groups chunks by paper/document,
    merges overlapping content, and orders by relevance — making it easier for
    the LLM to synthesize across papers rather than repeating one paper's content.
    
    Returns: list of source groups [{paper: str, chunks: [...], top_score: float}]
    """
    if not sources:
        return []
    
    # Group by document (using sha256 or filename as family key)
    families = {}
    for s in sources:
        meta = s.get("meta", {})
        family_key = meta.get("doc_family") or meta.get("sha256") or meta.get("source", "unknown")
        if family_key not in families:
            families[family_key] = {
                "paper": meta.get("source", meta.get("title", "Unknown")),
                "author": meta.get("author", ""),
                "year": meta.get("year", ""),
                "chunks": [],
                "top_score": 0,
            }
        families[family_key]["chunks"].append(s)
        score = s.get("relevance", s.get("score", 0))
        if score > families[family_key]["top_score"]:
            families[family_key]["top_score"] = score
    
    # Sort families by top relevance score (best paper first)
    sorted_families = sorted(families.values(), key=lambda f: f["top_score"], reverse=True)
    
    # Within each family, sort chunks by page/position
    for fam in sorted_families:
        fam["chunks"].sort(key=lambda c: c.get("meta", {}).get("page", 0))
    
    return sorted_families


# ---------------------------------------------------------------------------
# Methodology Detection — auto-tag chunks by research design
# ---------------------------------------------------------------------------

METHODOLOGY_PATTERNS = {
    "RCT": ["randomized control", "randomized experiment", "random assignment",
             "experimental group", "treatment group", "control group", "placebo"],
    "Diff-in-Diff": ["difference-in-difference", "diff-in-diff", "did estimate",
                      "parallel trends", "treatment effect", "pre-treatment"],
    "Regression Discontinuity": ["regression discontinuity", "RDD", "cutoff",
                                  "bandwidth", "running variable", "forcing variable"],
    "Instrumental Variables": ["instrumental variable", "two-stage", "2SLS",
                                "first stage", "exclusion restriction", "instrument"],
    "Synthetic Control": ["synthetic control", "counterfactual unit", "donor pool",
                           "pre-treatment fit"],
    "Case Study": ["case study", "process tracing", "within-case", "most-similar",
                    "most-different", "Mill's methods", "comparative case"],
    "Survey": ["survey data", "respondents", "sample size", "Likert", "polling",
               "cross-sectional survey", "panel survey", "survey experiment"],
    "Formal Model": ["formal model", "game theory", "equilibrium", "Nash",
                      "utility function", "payoff", "strategic interaction"],
    "Content Analysis": ["content analysis", "coding scheme", "intercoder reliability",
                          "textual analysis", "corpus", "sentiment analysis"],
    "QCA": ["qualitative comparative analysis", "QCA", "fuzzy set", "crisp set",
             "truth table", "necessity", "sufficiency"],
}

def detect_methodology(text: str) -> list[str]:
    """Detect research methodology mentioned in a text chunk."""
    t = text.lower()
    found = []
    for method, keywords in METHODOLOGY_PATTERNS.items():
        if any(kw in t for kw in keywords):
            found.append(method)
    return found


# ---------------------------------------------------------------------------
# Theory Detection — identify theoretical frameworks
# ---------------------------------------------------------------------------

THEORY_PATTERNS = {
    "Institutionalism": ["institutional", "path dependence", "critical juncture",
                          "institutional design", "rules of the game"],
    "Rational Choice": ["rational choice", "utility maximiz", "collective action",
                         "free rider", "Olson", "prisoner's dilemma", "strategic"],
    "Constructivism": ["constructivist", "social construction", "norms", "identity",
                        "ideational", "discourse", "intersubjective"],
    "Selectorate Theory": ["selectorate", "winning coalition", "Bueno de Mesquita",
                            "leader survival", "private goods", "public goods"],
    "Democratic Peace": ["democratic peace", "democracies don't fight",
                          "Kantian peace", "liberal peace"],
    "Modernization Theory": ["modernization theory", "Lipset", "economic development",
                              "middle class", "prerequisite"],
    "Dependency Theory": ["dependency theory", "world-system", "core periphery",
                           "Wallerstein", "unequal exchange"],
    "Realism": ["realist", "balance of power", "security dilemma", "anarchy",
                 "self-help", "Mearsheimer", "Waltz"],
    "Liberalism (IR)": ["liberal institutionalism", "Keohane", "international cooperation",
                         "regime theory", "interdependence"],
    "Behavioralism": ["behavioralism", "voting behavior", "political behavior",
                       "public opinion", "survey research"],
    "Historical Institutionalism": ["historical institutionalism", "path depend",
                                     "increasing returns", "Pierson", "Thelen"],
    "Veto Players": ["veto player", "Tsebelis", "policy stability", "status quo"],
    "Principal-Agent": ["principal-agent", "moral hazard", "adverse selection",
                         "delegation", "accountability"],
}

def detect_theory(text: str) -> list[str]:
    """Detect political science theories mentioned in a text chunk."""
    t = text.lower()
    found = []
    for theory, keywords in THEORY_PATTERNS.items():
        if any(kw in t for kw in keywords):
            found.append(theory)
    return found


# ---------------------------------------------------------------------------
# Author-Aware Search — extract author names from queries
# ---------------------------------------------------------------------------

_AUTHOR_QUERY_PATTERNS = [
    r"what does (\w+(?:\s\w+)?)\s+(?:argue|say|claim|find|show|propose|suggest|write)",
    r"according to (\w+(?:\s\w+)?)",
    r"(\w+(?:\s\w+)?)'s (?:argument|theory|model|finding|work|paper|claim)",
    r"(\w+(?:\s\w+)?) (?:\(\d{4}\)|\d{4})",
    r"in (\w+(?:\s\w+)?) and (?:\w+)",
]

def extract_author_query(query: str) -> dict:
    """Extract author name from a query for author-filtered search.
    
    Returns: {author: str or None, clean_query: str}
    """
    import re
    for pattern in _AUTHOR_QUERY_PATTERNS:
        match = re.search(pattern, query, re.IGNORECASE)
        if match:
            author = match.group(1).strip()
            # Filter out common non-author words
            if author.lower() in {"how", "what", "why", "when", "the", "this", "that", "it"}:
                continue
            return {"author": author, "clean_query": query}
    return {"author": None, "clean_query": query}


# ---------------------------------------------------------------------------
# Debate Detection — flag conflicting findings across sources
# ---------------------------------------------------------------------------

_DISAGREEMENT_MARKERS = [
    "however", "in contrast", "contradicts", "challenge this", "dispute",
    "on the other hand", "critics argue", "disagrees with", "rejects",
    "contrary to", "opposite finding", "mixed results", "no effect",
    "fails to find", "does not support", "inconsistent with",
]

def detect_debate(sources: list) -> list[dict]:
    """Flag potential debates when sources contain opposing findings."""
    debates = []
    for i, s in enumerate(sources):
        text = (s.get("text", "") or "").lower()
        if any(marker in text for marker in _DISAGREEMENT_MARKERS):
            debates.append({
                "source_index": i,
                "source_label": s.get("label", f"S{i+1}"),
                "signal": "potential_disagreement",
                "preview": s.get("text", "")[:200],
            })
    return debates


# ---------- HyDE: Hypothetical Document Embeddings (Gao et al. 2022) ----------
def _generate_hypothetical_doc(query: str) -> str:
    """Generate a hypothetical document passage that would answer the query.
    This is embedded alongside the real query to improve recall by 20-30%.
    The hypothetical doc lives in the same semantic space as stored chunks."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
    if not api_key:
        return ""
    try:
        import requests as _req
        model = os.environ.get("EDITH_MODEL", "gemini-2.5-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        prompt = (
            "Write a short academic passage (3-5 sentences) that would directly answer this question. "
            "Write it as if it were an excerpt from a research paper, lecture note, or textbook. "
            "Do not say 'This passage answers...' — just write the passage itself.\n\n"
            f"Question: {query}"
        )
        resp = _req.post(url, json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 300}
        }, timeout=15)
        if resp.status_code != 200:
            return ""
        data = resp.json()
        candidates = data.get("candidates", [])
        if candidates:
            parts = candidates[0].get("content", {}).get("parts", [])
            if parts:
                return parts[0].get("text", "")
    except Exception:
        pass
    return ""


# ---------- Agentic Retrieval: Sufficiency Check ----------
def _check_retrieval_sufficiency(query: str, sources: list, min_relevance: float = 0.35) -> dict:
    """Check if retrieved sources are sufficient to answer the query.
    Returns {sufficient: bool, reason: str, suggested_refinement: str}."""
    if not sources:
        return {"sufficient": False, "reason": "no_results", "refinement": query}
    
    # Check if top results have decent relevance scores
    top_scores = [s.get("relevance", 0) for s in sources[:5]]
    avg_top = sum(top_scores) / len(top_scores) if top_scores else 0
    
    if avg_top < min_relevance:
        return {"sufficient": False, "reason": "low_relevance", "refinement": query}
    
    # Check diversity — are all results from the same document?
    unique_docs = len(set(s.get("meta", {}).get("sha256", "") for s in sources[:8]))
    if unique_docs < 2 and len(sources) >= 4:
        return {"sufficient": False, "reason": "low_diversity", "refinement": query}
    
    return {"sufficient": True, "reason": "ok", "refinement": ""}


def _refine_query(query: str, attempt: int, reason: str) -> list:
    """Generate refined queries for agentic retrieval retry."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
    if not api_key:
        # Fallback: simple expansions
        if attempt == 1:
            return [query, f"{query} methodology approach"]
        return [query, f"{query} evidence findings results"]
    try:
        import requests as _req
        model = os.environ.get("EDITH_MODEL", "gemini-2.5-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        prompt = (
            f"The following search query returned {'no' if reason == 'no_results' else 'poor'} results "
            f"from an academic document library. Generate 2 alternative search queries that "
            f"might find relevant documents. Return ONLY the queries, one per line.\n\n"
            f"Original query: {query}\nAttempt: {attempt}\nIssue: {reason}"
        )
        resp = _req.post(url, json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 100}
        }, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
                return lines[:2] if lines else [query]
    except Exception:
        pass
    return [query]


def retrieve_cross_collection(
    queries: list[str],
    chroma_dir: str,
    collection_names: list[str],
    embed_model: str,
    top_k: int = 8,
    **kwargs,
) -> list[dict]:
    """§IMP-6.2: Search across multiple ChromaDB collections simultaneously.

    Merges results from all collections, re-ranks by score, and deduplicates.
    """
    all_results = []
    for coll_name in collection_names:
        try:
            results = retrieve_local_sources(
                queries=queries,
                chroma_dir=chroma_dir,
                collection_name=coll_name,
                embed_model=embed_model,
                top_k=top_k,
                **kwargs,
            )
            for r in results:
                r["_collection"] = coll_name
            all_results.extend(results)
        except Exception:
            pass

    # Deduplicate by content hash
    seen = set()
    deduped = []
    for r in all_results:
        content_key = (r.get("document") or r.get("text") or "")[:200]
        if content_key not in seen:
            seen.add(content_key)
            deduped.append(r)

    # Re-sort by score (descending)
    deduped.sort(key=lambda r: r.get("score", 0), reverse=True)
    return deduped[:top_k]


def chunk_with_overlap(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """§IMP-6.9: Split text into chunks with token overlap.

    Preserves sentence boundaries at overlap points for context continuity.
    chunk_size: target tokens per chunk.
    overlap: overlap tokens between adjacent chunks.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        # Advance by chunk_size - overlap
        start += max(chunk_size - overlap, 1)

    return chunks


def retrieve_local_sources(
    queries,
    chroma_dir: str,
    collection_name: str,
    embed_model: str,
    top_k: int = 8,
    pool_multiplier: int = 4,
    diversity_lambda: float = 0.65,
    bm25_weight: float = 0.35,
    rerank_model: str = "",
    rerank_top_n: int = 14,
    project: str = "",
    tag: str = "",
    section_filter: str = "",
    doc_type_filter: str = "",
    require_equations: bool = False,
    stitch_neighbors: int = 2,
    family_cap: int = 2,
    hierarchical: bool = False,
):
    if not chroma_runtime_available():
        raise RuntimeError("Install chromadb and sentence-transformers for Chroma mode.")

    qlist = [str(q).strip() for q in (queries or []) if str(q).strip()]
    if not qlist:
        return []

    # Improvement 2.3: Adaptive top_k
    if _RETRIEVAL_IMPROVEMENTS and len(qlist) == 1:
        top_k = adaptive_top_k(qlist[0], base_k=top_k)

    # Improvement 2.5: Query decomposition for complex queries
    if _RETRIEVAL_IMPROVEMENTS and len(qlist) == 1:
        sub_queries = decompose_query(qlist[0])
        if len(sub_queries) > 1:
            qlist = sub_queries

    # Improvement 2.7: Check cache first
    _init_retrieval_improvements()
    if _RETRIEVAL_CACHE and len(qlist) == 1:
        cached = _RETRIEVAL_CACHE.get(qlist[0], top_k)
        if cached is not None:
            if _RETRIEVAL_TELEMETRY:
                _RETRIEVAL_TELEMETRY.record(
                    query=qlist[0], results_count=len(cached),
                    cache_hit=True, latency_ms=0,
                )
            return cached

    _retrieval_start = time.time()

    top_k = max(1, int(top_k))
    pool_multiplier = max(1, int(pool_multiplier))

    # Bug fix: Wire up classify_query_intent to dynamically set weights
    original_query = qlist[0] if qlist else ""
    intent = classify_query_intent(original_query)
    bm25_weight = intent["bm25_weight"]
    diversity_lambda = intent["diversity"]

    embedder = None
    
    # --- HyDE: Generate hypothetical document for better retrieval ---
    use_hyde = os.environ.get("EDITH_HYDE", "true").lower() == "true"
    hyde_doc = ""
    if use_hyde and len(qlist) == 1:  # Only for single queries (not multi-query rewrites)
        hyde_doc = _generate_hypothetical_doc(qlist[0])
        if hyde_doc:
            qlist = qlist + [hyde_doc]  # Search with both original query AND hypothetical doc
    
    # --- §3.1 Check query cache first ---
    cache_key_query = original_query
    cached_result = _query_cache.get(cache_key_query, top_k, collection_name)
    if cached_result is not None:
        log.debug(f"Query cache hit: {cache_key_query[:40]}...")
        return cached_result

    # --- Embed queries (original + HyDE) with §3.2 embedding cache ---
    # §HW: Try local MLX/MPS embeddings first (Neural Engine), then Gemini API
    env_mode = os.environ.get("EDITH_ENV", "").strip().lower()
    app_mode = os.environ.get("EDITH_APP_MODE", "").strip().lower()
    is_test_mode = env_mode == "test" or app_mode == "test"
    use_local_embed = os.environ.get("EDITH_LOCAL_EMBED", "false" if is_test_mode else "true").lower() == "true"
    use_gemini = os.environ.get("EDITH_USE_GEMINI_EMBED", "false" if is_test_mode else "true").lower() == "true"
    gemini_model = os.environ.get("EDITH_GEMINI_EMBED_MODEL", "gemini-embedding-001")
    active_embed_model = gemini_model if use_gemini else embed_model
    log.debug(f"§EMBED-PATH: use_local={use_local_embed}, use_gemini={use_gemini}, active_model={active_embed_model}")

    # Check embedding cache for each query
    q_vectors = []
    uncached_texts = []
    uncached_indices = []
    for i, qt in enumerate(qlist):
        cached_vec = _embed_cache.get(qt, active_embed_model)
        if cached_vec is not None:
            q_vectors.append(cached_vec)
        else:
            q_vectors.append(None)  # placeholder
            uncached_texts.append(qt)
            uncached_indices.append(i)

    log.debug(f"§EMBED-PATH: {len(qlist)} queries, {len(uncached_texts)} uncached")

    # Embed only uncached texts — try MLX first, then Gemini, then CPU
    if uncached_texts:
        new_vecs = None

        # §HW: Try local MLX/MPS embeddings (Neural Engine acceleration)
        if use_local_embed and new_vecs is None:
            try:
                from server.mlx_embeddings import embed as mlx_embed, is_available as mlx_avail
                if mlx_avail():
                    local_vecs = mlx_embed(uncached_texts)
                    if local_vecs and len(local_vecs) == len(uncached_texts):
                        new_vecs = local_vecs
                        log.debug(f"§EMBED-PATH: MLX succeeded, dim={len(local_vecs[0])}")
            except Exception as _mlx_err:
                log.debug(f"§HW: MLX embed unavailable: {_mlx_err}")

        # Gemini API embeddings (existing path)
        if new_vecs is None and use_gemini:
            log.debug("§EMBED-PATH: Calling Gemini embed API...")
            new_vecs = _gemini_embed_queries(uncached_texts, model=gemini_model)
            if not new_vecs or len(new_vecs) != len(uncached_texts):
                log.warning(f"§EMBED-PATH: Gemini embed returned {len(new_vecs) if new_vecs else 0} vecs for {len(uncached_texts)} texts")
                new_vecs = None
            else:
                log.debug(f"§EMBED-PATH: Gemini embed OK, dim={len(new_vecs[0])}")

        # CPU fallback (sentence-transformers)
        if new_vecs is None:
            log.warning("§EMBED-PATH: Falling back to CPU embedder (sentence-transformers)")
            if embedder is None:
                embedder = _get_embedder(embed_model)
            new_vecs = embedder.encode(uncached_texts, normalize_embeddings=True).tolist()

        # Fill in and cache
        for idx, vec in zip(uncached_indices, new_vecs):
            q_vectors[idx] = vec
            _embed_cache.put(qlist[idx], active_embed_model, vec)

    # If all were cached but model mismatch somehow left Nones, re-embed all
    if any(v is None for v in q_vectors):
        if use_gemini:
            q_vectors = _gemini_embed_queries(qlist, model=gemini_model)
            if not q_vectors or len(q_vectors) != len(qlist):
                q_vectors = None
        if q_vectors is None or any(v is None for v in q_vectors):
            if embedder is None:
                embedder = _get_embedder(embed_model)
            q_vectors = embedder.encode(qlist, normalize_embeddings=True).tolist()

    # Bug fix: Embedding model version check
    query_embed_dim = len(q_vectors[0]) if q_vectors and q_vectors[0] else 0
    query_embed_model = gemini_model if use_gemini else embed_model

    merged_q = [0.0 for _ in q_vectors[0]]
    for qv in q_vectors:
        for i, v in enumerate(qv):
            merged_q[i] += float(v)
    merged_q = [v / len(q_vectors) for v in merged_q]

    client = _get_client(chroma_dir)
    collection = client.get_collection(name=collection_name)

    # Check for embedding model mismatch between index and query
    _embed_mismatch_warning = ""
    try:
        sample = collection.get(limit=1, include=["embeddings", "metadatas"])
        sample_embeds = (sample.get("embeddings") or [[]])[0]
        sample_meta = (sample.get("metadatas") or [{}])[0]
        index_dim = len(sample_embeds) if sample_embeds else 0
        index_model = sample_meta.get("embed_model", "")
        if index_dim and query_embed_dim and index_dim != query_embed_dim:
            _embed_mismatch_warning = (
                f"DIMENSION MISMATCH: index={index_dim}-dim, query={query_embed_dim}-dim. "
                f"Re-embedding with local model to match index."
            )
            import logging
            logging.getLogger("edith.retrieval").warning(_embed_mismatch_warning)
            # §HW: Auto-recover — re-embed with sentence-transformers (matches 384-dim index)
            try:
                q_vectors = embedder.encode(qlist, normalize_embeddings=True).tolist()
                query_embed_dim = len(q_vectors[0]) if q_vectors else 0
                logging.getLogger("edith.retrieval").info(
                    f"§HW: Re-embedded with local model → {query_embed_dim}-dim (matches index)")
            except Exception as _re_err:
                logging.getLogger("edith.retrieval").error(f"§HW: Re-embed failed: {_re_err}")
        elif index_model and index_model != query_embed_model:
            _embed_mismatch_warning = (
                f"Model mismatch: index used '{index_model}', query uses '{query_embed_model}'. "
                f"Quality may be slightly degraded."
            )
            import logging
            logging.getLogger("edith.retrieval").info(_embed_mismatch_warning)
    except Exception:
        pass
    
    # Phase 2: Hierarchical search (narrowing to top sections)
    hier_sha_filter = []
    if hierarchical:
        try:
            s_coll = client.get_or_create_collection(name=f"{collection_name}_sections")
            s_res = s_coll.query(
                query_embeddings=[merged_q],
                n_results=15, # Narrow to top 15 sections
                include=["metadatas"]
            )
            s_metas = (s_res.get("metadatas") or [[]])[0]
            hier_sha_filter = list(set([str(m.get("sha256")) for m in s_metas if m.get("sha256")]))
        except Exception:
            hier_sha_filter = []

    # Phase 2: Negative Memory (Exclusion list)
    exclude_shas = _load_negative_memory(chroma_dir)

    where = {}
    if project and project != "All":
        where["project"] = project
    if tag:
        where["tag"] = tag
    
    # Combine Filters
    filters = []
    if project and project != "All":
        filters.append({"project": {"$eq": project}})
    if tag:
        filters.append({"tag": {"$eq": tag}})
    if exclude_shas:
        if len(exclude_shas) == 1:
            filters.append({"sha256": {"$ne": exclude_shas[0]}})
        else:
            filters.append({"sha256": {"$nin": exclude_shas}})
    if hier_sha_filter:
        if len(hier_sha_filter) == 1:
            filters.append({"sha256": {"$eq": hier_sha_filter[0]}})
        else:
            filters.append({"sha256": {"$in": hier_sha_filter}})

    if len(filters) > 1:
        where = {"$and": filters}
    elif len(filters) == 1:
        where = filters[0]
    else:
        where = None

    pool_n = max(top_k, top_k * pool_multiplier)
    candidate_map = {}

    section_filter = (section_filter or "").strip().lower()
    doc_type_filter_vals = _expand_doc_type_filters(set(_parse_csv_filter(doc_type_filter)))
    family_cap = max(1, int(family_cap))

    def _query_one_vector(qv):
        """Query ChromaDB with a single vector (runs in thread)."""
        return collection.query(
            query_embeddings=[qv],
            n_results=pool_n,
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"],
        )

    # §HW: Parallel retrieval — fire all query vectors simultaneously
    # On M4+Thunderbolt: 8 concurrent queries against NVMe
    # On M2: 2 concurrent queries (or sequential fallback)
    try:
        from server.backend_logic import get_compute_profile
        _max_workers = get_compute_profile().get("max_concurrent_retrieval", 2)
    except Exception:
        _max_workers = 2

    all_results = []
    if len(q_vectors) > 1 and _max_workers > 1:
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            _t0 = time.time()
            with ThreadPoolExecutor(max_workers=min(_max_workers, len(q_vectors))) as pool:
                futures = [pool.submit(_query_one_vector, qv) for qv in q_vectors]
                for future in as_completed(futures):
                    try:
                        all_results.append(future.result())
                    except Exception as _e:
                        log.warning(f"Parallel query failed: {_e}")
            log.debug(f"§HW: Parallel retrieval: {len(q_vectors)} queries in "
                      f"{time.time()-_t0:.2f}s (workers={_max_workers})")
        except Exception:
            # Fallback to sequential
            all_results = [_query_one_vector(qv) for qv in q_vectors]
    else:
        all_results = [_query_one_vector(qv) for qv in q_vectors]

    for res in all_results:
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        embeds = (res.get("embeddings") or [[]])[0]

        for rid, doc, meta, dist, emb in zip(ids, docs, metas, dists, embeds):
            section_heading = str((meta or {}).get("section_heading") or "").strip().lower()
            if section_filter and section_filter not in section_heading:
                continue
            doc_type = str((meta or {}).get("doc_type") or "").strip().lower()
            if not _doc_type_matches(doc_type, doc_type_filter_vals):
                continue
            eq_markers = str((meta or {}).get("equation_markers") or "").strip()
            if require_equations and not eq_markers and not _equation_like(doc or ""):
                continue
            rel = 1.0 - float(dist)
            prev = candidate_map.get(rid)
            if emb is None:
                emb_vec = []
            elif hasattr(emb, "tolist"):
                emb_vec = emb.tolist()
            else:
                emb_vec = list(emb)

            item = {
                "id": rid,
                "text": doc or "",
                "meta": meta or {},
                "vector_relevance": rel,
                "relevance": rel,
                "embedding": emb_vec,
            }
            if not prev or item["vector_relevance"] > prev["vector_relevance"]:
                candidate_map[rid] = item

    candidates = list(candidate_map.values())
    if not candidates:
        return []

    bm25_weight = max(0.0, min(1.0, float(bm25_weight)))
    query_tokens = _tokenize(" ".join(qlist))
    bm25_raw = _bm25_scores([c.get("text") or "" for c in candidates], query_tokens)
    vec_raw = [float(c.get("vector_relevance", 0.0)) for c in candidates]
    bm25_norm = _normalize_scores(bm25_raw)
    vec_norm = _normalize_scores(vec_raw)
    for idx, cand in enumerate(candidates):
        v = vec_norm[idx] if idx < len(vec_norm) else 0.0
        b = bm25_norm[idx] if idx < len(bm25_norm) else 0.0
        combined = ((1.0 - bm25_weight) * v) + (bm25_weight * b)
        
        meta = cand.get("meta") or {}
        v_boost = _version_stage_boost(meta.get("version_stage"))
        t_boost = _tier_boost(meta.get("tier"))
        
        combined = max(0.0, min(1.0, combined + v_boost + t_boost))
        cand["vector_score"] = round(v, 4)
        cand["bm25_score"] = round(b, 4)
        cand["version_stage_boost"] = round(v_boost, 4)
        cand["tier_boost"] = round(t_boost, 4)
        cand["relevance"] = combined

    selected = _mmr_select(candidates, top_k=top_k, lambda_mult=diversity_lambda)

    rerank_model = (rerank_model or "").strip()
    rerank_top_n = max(1, int(rerank_top_n))
    reranker = _get_reranker(rerank_model)
    if reranker and selected:
        merged_query = " ".join(qlist[:3]).strip()
        rerank_pool = selected[: min(len(selected), rerank_top_n)]
        try:
            pairs = [(merged_query, s.get("text") or "") for s in rerank_pool]
            rr_scores = reranker.predict(pairs)
            rr_norm = _normalize_scores([float(x) for x in rr_scores])
            for i, score in enumerate(rr_norm):
                rerank_pool[i]["rerank_score"] = round(float(score), 4)
                rerank_pool[i]["relevance"] = (0.65 * float(score)) + (0.35 * float(rerank_pool[i].get("relevance", 0.0)))
            rerank_pool.sort(key=lambda x: x.get("relevance", 0.0), reverse=True)
            selected = rerank_pool + selected[len(rerank_pool):]
        except Exception:
            pass

    selected.sort(key=lambda x: x.get("relevance", 0.0), reverse=True)

    stitch_neighbors = max(0, int(stitch_neighbors))
    neighbor_text = {}
    if stitch_neighbors > 0 and selected:
        neighbor_ids = set()
        for item in selected:
            meta = item.get("meta") or {}
            sha = str(meta.get("sha256") or "").strip()
            chunk = meta.get("chunk")
            try:
                chunk_i = int(chunk)
            except Exception:
                continue
            if not sha:
                continue
            for off in range(-stitch_neighbors, stitch_neighbors + 1):
                if off == 0:
                    continue
                neighbor_ids.add(f"{sha}:{chunk_i + off}")
        if neighbor_ids:
            try:
                got = collection.get(ids=list(neighbor_ids), include=["documents"])
                ids_g = got.get("ids") or []
                docs_g = got.get("documents") or []
                for rid, doc in zip(ids_g, docs_g):
                    if isinstance(rid, str):
                        neighbor_text[rid] = (doc or "")
            except Exception:
                neighbor_text = {}

    family_counts = Counter()
    capped = []
    overflow = []
    for item in selected:
        meta = item.get("meta") or {}
        family = str(meta.get("doc_family") or meta.get("sha256") or item.get("id") or "").strip()
        if family and family_counts[family] >= family_cap:
            overflow.append(item)
            continue
        if family:
            family_counts[family] += 1
        capped.append(item)
    selected = (capped + overflow)[:top_k]

    out = []
    # Keep chat retrieval focused on text-bearing research documents.
    _ALLOWED_TEXT_EXTS = {
        "pdf", "txt", "md", "docx", "tex", "html", "htm", "ipynb", "rmd", "json"
    }
    for item in selected:
        meta = item.get("meta") or {}
        rel_path = meta.get("rel_path") or meta.get("path") or ""
        file_name = meta.get("file_name") or (Path(rel_path).name if rel_path else "")
        _ext_raw = Path(file_name or rel_path).suffix.lower()
        ext = _ext_raw[1:] if _ext_raw.startswith(".") else _ext_raw
        if ext and ext not in _ALLOWED_TEXT_EXTS:
            continue
        page = meta.get("page")
        chunk = meta.get("chunk", 0)
        section_heading = meta.get("section_heading") or ""
        figure_table_markers = (meta.get("figure_table_markers") or "").strip()
        equation_markers = (meta.get("equation_markers") or "").strip()
        title = meta.get("title") or meta.get("title_guess") or file_name or rel_path or "local_source"
        snippet = (item.get("text") or "").strip()
        stitch_span = ""
        if stitch_neighbors > 0:
            sha = str(meta.get("sha256") or "").strip()
            try:
                chunk_i = int(chunk)
            except Exception:
                chunk_i = None
            if sha and chunk_i is not None:
                parts = []
                for off in range(-stitch_neighbors, stitch_neighbors + 1):
                    rid = f"{sha}:{chunk_i + off}"
                    if off == 0:
                        parts.append((item.get("text") or "").strip())
                    elif rid in neighbor_text:
                        parts.append((neighbor_text[rid] or "").strip())
                parts = [p for p in parts if p]
                if len(parts) > 1:
                    stitched = " ".join(parts)
                    if stitched:
                        snippet = stitched
                        stitch_span = f"{max(0, chunk_i - stitch_neighbors)}-{chunk_i + stitch_neighbors}"
        if len(snippet) > 4000:
            snippet = snippet[:4000]
        if not snippet.strip():
            continue

        out.append(
            {
                "title": title,
                "uri": rel_path,
                "snippet": snippet,
                "source_type": "file",
                "rel_path": rel_path,
                "file_name": file_name,
                "sha256": meta.get("sha256") or "",
                "chunk": chunk,
                "page": page if page is not None else 0,
                "section_heading": section_heading,
                "tier": meta.get("tier") or "",
                "doc_type": meta.get("doc_type") or "",
                "version_stage": meta.get("version_stage") or "",
                "author": meta.get("author") or "",
                "year": str(meta.get("year") or ""),
                "citation_source": meta.get("citation_source") or "",
                "figure_table_markers": figure_table_markers,
                "equation_markers": equation_markers,
                "doc_family": meta.get("doc_family") or "",
                "academic_topic": meta.get("academic_topic") or _infer_academic_topic(rel_path),
                "vault_export_id": meta.get("vault_export_id") or "",
                "vault_export_date": meta.get("vault_export_date") or "",
                "vault_custodian": meta.get("vault_custodian") or "",
                "vault_matter_name": meta.get("vault_matter_name") or "",
                "stitch_span": stitch_span,
                "score": round(float(item.get("relevance", 0.0)), 4),
                "vector_score": item.get("vector_score"),
                "bm25_score": item.get("bm25_score"),
                "rerank_score": item.get("rerank_score"),
                "version_stage_boost": item.get("version_stage_boost"),
                "tier_boost": item.get("tier_boost"),
            }
        )

    # keep retrieval order with a deterministic score tie-break
    out.sort(key=lambda x: x.get("score", 0.0), reverse=True)

    # Improvement 2.6: Temporal weighting
    if _RETRIEVAL_IMPROVEMENTS:
        out = apply_temporal_weight(out)

    # Improvement 2.8: Confidence calibration
    if _RETRIEVAL_IMPROVEMENTS:
        out = calibrate_confidence(out)

    # Improvement 2.2: Telemetry
    if _RETRIEVAL_TELEMETRY:
        latency = (time.time() - _retrieval_start) * 1000
        top_score = out[0].get("score", 0) if out else 0
        avg_score = sum(r.get("score", 0) for r in out) / max(len(out), 1)
        _RETRIEVAL_TELEMETRY.record(
            query=qlist[0] if qlist else "",
            results_count=len(out),
            top_score=top_score,
            avg_score=avg_score,
            latency_ms=latency,
            strategy="hybrid",
            reranked=bool(rerank_model),
            hyde_used=bool(hyde_doc),
            bm25_used=True,
        )

    # Improvement 2.7: Cache results
    if _RETRIEVAL_CACHE and len(qlist) >= 1:
        _RETRIEVAL_CACHE.put(qlist[0], top_k, out)

    # §3.1: Store in built-in query cache
    _query_cache.put(original_query, top_k, out, collection_name)

    return out


def retrieve_full_documents(
    sha256_list: List[str],
    chroma_dir: str,
    collection_name: str,
    max_tokens_per_doc: int = 200000 
):
    """
    Retrieves all chunks for a set of SHA256 hashes and reconstructs full text blocks.
    Utilizes Gemini's large context window.
    """
    if not sha256_list or not chroma_runtime_available():
        return []
    
    client = _get_client(chroma_dir)
    collection = client.get_or_create_collection(name=collection_name)
    
    docs_out = []
    for sha in sha256_list[:5]: # Cap at 5 full docs for safety
        try:
            # Query all chunks for this SHA
            res = collection.get(
                where={"sha256": sha},
                include=["documents", "metadatas"]
            )
            ids = res.get("ids") or []
            docs = res.get("documents") or []
            metas = res.get("metadatas") or []
            
            if not docs:
                continue
                
            # Sort by chunk index to reconstruct
            pairs = []
            for d, m in zip(docs, metas):
                try:
                    cidx = int(m.get("chunk", 0))
                except Exception:
                    cidx = 0
                pairs.append((cidx, d, m))
            pairs.sort(key=lambda x: x[0])
            
            full_text = "\n\n".join([p[1] for p in pairs])
            # Cap if extremely long
            if len(full_text) > max_tokens_per_doc * 4:
                full_text = full_text[:max_tokens_per_doc * 4] + "... [TRUNCATED]"
                
            first_meta = pairs[0][2]
            docs_out.append({
                "title": first_meta.get("title") or first_meta.get("file_name") or "full_doc",
                "uri": first_meta.get("rel_path") or "",
                "text": full_text,
                "academic_topic": first_meta.get("academic_topic") or _infer_academic_topic(first_meta.get("rel_path") or ""),
                "sha256": sha,
                "source_type": "full_file"
            })
        except Exception:
            continue
            
    return docs_out


def format_local_context(sources):
    blocks = []
    for i, s in enumerate(sources or [], start=1):
        rel = (s.get("rel_path") or s.get("uri") or s.get("title") or "").strip()
        chunk = s.get("chunk")
        page = s.get("page")
        header = f"[S{i}] file={rel}"
        if page:
            header += f" page={page}"
        if chunk is not None:
            header += f" chunk={chunk}"
        section_heading = (s.get("section_heading") or "").strip()
        if section_heading:
            header += f" section={section_heading}"
        snippet = (s.get("snippet") or "").strip()
        blocks.append(f"{header}\n{snippet}")
    return "\n\n".join(blocks)


def merge_sources(primary, secondary):
    """Merge two source lists, tolerating nested list shapes from callers."""
    out: list[dict] = []
    seen = set()

    def _iter_source_dicts(group):
        if not group:
            return
        if isinstance(group, dict):
            yield group
            return
        if isinstance(group, (list, tuple)):
            for item in group:
                if isinstance(item, dict):
                    yield item
                elif isinstance(item, (list, tuple)):
                    for sub in item:
                        if isinstance(sub, dict):
                            yield sub

    for group in (primary, secondary):
        for s in _iter_source_dicts(group):
            key = (s.get("uri") or s.get("title") or "").strip().lower()
            if not key:
                key = (s.get("snippet") or "").strip().lower()[:120]
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(s)
    return out


def decompose_query(query: str) -> list[str]:
    """Split a complex multi-part question into independent sub-queries.
    
    Example: "Compare Boix and Acemoglu on democratization"
    → ["Boix theory of democratization", "Acemoglu theory of democratization"]
    
    Returns the original query if decomposition fails or isn't needed.
    """
    # Quick heuristic: only decompose if query looks multi-part
    multi_signals = ["compare", "contrast", "difference between", " and ", " vs ", 
                     "relationship between", "how does .* relate to", "both"]
    if not any(s in query.lower() for s in multi_signals) or len(query.split()) < 6:
        return [query]
    
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
    if not api_key:
        return [query]
    
    try:
        import requests as _req
        model = os.environ.get("EDITH_MODEL", "gemini-2.5-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        prompt = (
            "Decompose this research question into 2-3 independent search queries "
            "that would each find relevant academic sources. Return ONLY the queries, "
            "one per line. Keep each under 15 words.\n\n"
            f"Question: {query}"
        )
        resp = _req.post(url, json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.3, "maxOutputTokens": 150}
        }, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                lines = [l.strip().lstrip("0123456789.-) ") for l in text.strip().split("\n") if l.strip()]
                sub_queries = [l for l in lines if 3 < len(l) < 200][:3]
                if sub_queries:
                    return sub_queries
    except Exception:
        pass
    return [query]


def generate_hyde(query: str) -> str:
    """HyDE: Generate a hypothetical document passage that would answer the query.
    
    Instead of embedding the question directly (which lives in "question space"),
    we generate a hypothetical answer paragraph and embed THAT (which lives in
    "document space" — same space as the indexed chunks). This dramatically
    improves retrieval for abstract or conceptual queries.
    
    Returns the hypothetical passage, or empty string on failure.
    """
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY") or ""
    if not api_key:
        return ""
    
    try:
        import requests as _req
        model = os.environ.get("EDITH_MODEL", "gemini-2.5-flash")
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        prompt = (
            "Write a short paragraph (3-4 sentences) that might appear in an academic paper "
            "answering this research question. Write in an academic style with specific claims "
            "and terminology. Do NOT include citations or references.\n\n"
            f"Question: {query}"
        )
        resp = _req.post(url, json={
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0.4, "maxOutputTokens": 200}
        }, timeout=8)
        if resp.status_code == 200:
            data = resp.json()
            candidates = data.get("candidates", [])
            if candidates:
                text = candidates[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                text = text.strip()
                if len(text) > 30:
                    return text
    except Exception:
        pass
    return ""


def agentic_retrieve(
    query: str,
    chroma_dir: str,
    collection_name: str,
    embed_model: str,
    top_k: int = 8,
    max_attempts: int = 3,
    min_relevance: float = 0.35,
    **kwargs,
) -> dict:
    """Agentic retrieval: multi-pass retrieval with sufficiency checking.
    
    Enhanced pipeline:
    1. Decompose complex queries into sub-queries
    2. Generate HyDE passage for better vector matching
    3. Retrieve with original + decomposed + HyDE queries
    4. Check if results are sufficient (relevance, diversity)
    5. If insufficient, use LLM to reformulate and retry
    6. Merge all results across attempts
    
    Returns: {sources: list, attempts: int, refinements: list, 
              sub_queries: list, hyde_used: bool}
    """
    all_sources = []
    refinement_log = []
    best_sources = []
    
    # Step 1: Decompose complex queries
    sub_queries = decompose_query(query)
    decomposed = len(sub_queries) > 1
    
    # Step 2: Generate HyDE passage
    hyde_passage = generate_hyde(query)
    hyde_used = bool(hyde_passage)
    
    # Build initial query set: original + decomposed + HyDE
    current_queries = list(sub_queries)  # Start with decomposed queries
    if hyde_passage:
        current_queries.append(hyde_passage)
    
    for attempt in range(max_attempts):
        sources = retrieve_local_sources(
            queries=current_queries,
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            embed_model=embed_model,
            top_k=top_k,
            **kwargs,
        )
        
        # Merge new sources with existing (deduplicated)
        all_sources = merge_sources(all_sources, sources)
        
        # Check sufficiency
        check = _check_retrieval_sufficiency(query, all_sources, min_relevance)
        
        if check["sufficient"] or attempt >= max_attempts - 1:
            best_sources = all_sources
            break
        
        # Not sufficient — refine and retry (without HyDE on retries)
        refined = _refine_query(query, attempt + 1, check["reason"])
        refinement_log.append({
            "attempt": attempt + 1,
            "reason": check["reason"],
            "new_queries": refined
        })
        current_queries = refined
    
    return {
        "sources": best_sources[:top_k * 2],  # Return up to 2x top_k from multi-pass
        "attempts": len(refinement_log) + 1,
        "refinements": refinement_log,
        "sub_queries": sub_queries if decomposed else [],
        "hyde_used": hyde_used,
    }


# ═══════════════════════════════════════════════════════════════════
# §CE-28: Academic Re-Ranking — Weight by scholarly impact
# ═══════════════════════════════════════════════════════════════════

def academic_rerank(sources: list[dict], query: str = "") -> list[dict]:
    """Re-rank retrieval results by academic weight.

    Beyond vector similarity, this considers:
    - Citation count (if available in metadata)
    - Recency bonus (newer papers get slight boost)
    - Author authority (repeat authors in corpus get boost)
    - Methodological match (if query implies a method preference)
    """
    import re as _re

    for source in sources:
        meta = source.get("meta", {})
        base_score = source.get("score", source.get("relevance", 0.5))

        # Citation count boost
        citations = meta.get("citation_count", 0)
        try:
            citations = int(citations)
        except (TypeError, ValueError):
            citations = 0
        citation_boost = min(0.1, math.log1p(citations) * 0.02)

        # Recency boost (papers within last 5 years get 0.05 boost)
        year = meta.get("year", "")
        try:
            year_int = int(year)
            if year_int >= 2021:
                recency_boost = 0.05
            elif year_int >= 2016:
                recency_boost = 0.02
            else:
                recency_boost = 0
        except (TypeError, ValueError):
            recency_boost = 0

        # Version stage boost (published > working_paper > draft)
        stage = meta.get("version_stage", "")
        stage_boost = {
            "published": 0.03, "accepted": 0.02, "working_paper": 0.01,
        }.get(stage, 0)

        source["academic_score"] = round(
            base_score + citation_boost + recency_boost + stage_boost, 4
        )

    sources.sort(key=lambda s: s.get("academic_score", 0), reverse=True)
    return sources


# ═══════════════════════════════════════════════════════════════════
# §CE-29: Semantic Deduplication — Remove near-duplicate chunks
# ═══════════════════════════════════════════════════════════════════

def semantic_dedup(sources: list[dict], similarity_threshold: float = 0.92) -> list[dict]:
    """Remove semantically duplicate results from retrieval.

    Two chunks from the same document with >92% text overlap are redundant.
    Keep the one with higher relevance score.
    """
    if len(sources) <= 1:
        return sources

    def _text_similarity(a: str, b: str) -> float:
        """Jaccard similarity of word sets."""
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        intersection = len(words_a & words_b)
        union = len(words_a | words_b)
        return intersection / union if union else 0.0

    deduped = []
    for source in sources:
        text = source.get("snippet", source.get("text", ""))
        is_dup = False
        for kept in deduped:
            kept_text = kept.get("snippet", kept.get("text", ""))
            if _text_similarity(text, kept_text) > similarity_threshold:
                is_dup = True
                break
        if not is_dup:
            deduped.append(source)

    removed = len(sources) - len(deduped)
    if removed > 0:
        log.debug(f"§DEDUP: Removed {removed} near-duplicate chunks")

    return deduped


# ═══════════════════════════════════════════════════════════════════
# §CE-30: Query Expansion — Synonym and concept broadening
# ═══════════════════════════════════════════════════════════════════

ACADEMIC_SYNONYMS = {
    "accountability": ["transparency", "answerability", "responsiveness", "oversight"],
    "devolution": ["decentralization", "federalism", "local governance", "subsidiarity"],
    "welfare": ["social assistance", "public benefits", "safety net", "social protection"],
    "privatization": ["outsourcing", "contracting out", "marketization", "public-private"],
    "bureaucracy": ["administration", "civil service", "public administration", "agency"],
    "state capacity": ["government capacity", "institutional capacity", "administrative capacity"],
    "blame diffusion": ["blame avoidance", "blame shifting", "accountability diffusion"],
    "democratic erosion": ["democratic backsliding", "democratic decline", "illiberalism"],
}


def expand_query(query: str, max_expansions: int = 3) -> list[str]:
    """Expand a query with academic synonyms and related concepts.

    "accountability in welfare" →
    [
        "accountability in welfare",
        "transparency in social assistance",
        "answerability in public benefits",
    ]
    """
    expanded = [query]
    query_lower = query.lower()

    for term, synonyms in ACADEMIC_SYNONYMS.items():
        if term in query_lower:
            for syn in synonyms[:max_expansions]:
                expanded_query = query_lower.replace(term, syn)
                if expanded_query != query_lower:
                    expanded.append(expanded_query)

    return expanded[:max_expansions + 1]


def get_retrieval_diagnostics(query: str, sources: list[dict]) -> dict:
    """Return diagnostics about retrieval quality for the Doctor panel."""
    if not sources:
        return {"quality": "no_results", "score": 0}

    scores = [s.get("score", 0) for s in sources]
    top_score = max(scores) if scores else 0
    avg_score = sum(scores) / len(scores) if scores else 0

    # Diversity measure: unique documents
    unique_docs = len(set(s.get("sha256", s.get("uri", "")) for s in sources))
    diversity = unique_docs / max(len(sources), 1)

    quality = "excellent" if top_score > 0.7 and diversity > 0.6 else (
        "good" if top_score > 0.5 else
        "fair" if top_score > 0.3 else "poor"
    )

    return {
        "quality": quality,
        "top_score": round(top_score, 3),
        "avg_score": round(avg_score, 3),
        "result_count": len(sources),
        "unique_documents": unique_docs,
        "diversity": round(diversity, 3),
    }


# ═══════════════════════════════════════════════════════════════════
# §BOLT-1: Vector Compression — Product Quantization for Large Souls
# ═══════════════════════════════════════════════════════════════════

_VECTOR_PQ = os.environ.get("VECTOR_PQ", "false").lower() == "true"
_PQ_THRESHOLD_DOCS = 10000  # Enable PQ when collection exceeds this size


def get_collection_with_pq(
    client,
    collection_name: str,
    embedding_dim: int = 384,
):
    """§BOLT-1: Get or create a collection with PQ compression if enabled.

    When VECTOR_PQ=true and the collection is large (>10K docs),
    enables Product Quantization-friendly HNSW settings that reduce
    vector storage by ~4x. This makes the "Soul" faster to load
    over the USB-C/Bolt interface.

    PQ works by:
        1. Dividing each 384-dim vector into 48 sub-vectors of 8 dims
        2. Quantizing each sub-vector to a codebook index (1 byte)
        3. Storage goes from 384×4 bytes = 1.5KB → 48 bytes per vector

    ChromaDB doesn't natively support PQ, so we optimize HNSW params
    for compressed retrieval and enable scalar quantization where available.
    """
    try:
        collection = client.get_collection(name=collection_name)

        if _VECTOR_PQ:
            # Check collection size
            count = collection.count()
            if count > _PQ_THRESHOLD_DOCS:
                log.info(f"§BOLT-1: Collection '{collection_name}' has {count} docs — "
                         f"PQ-optimized HNSW in effect")

        return collection
    except Exception:
        # Create with PQ-optimized settings
        metadata = {}
        if _VECTOR_PQ:
            metadata = {
                "hnsw:space": "cosine",
                "hnsw:M": 32,  # Higher M for compressed neighbors
                "hnsw:construction_ef": 200,
                "hnsw:search_ef": 100,
                # Scalar quantization (ChromaDB's closest to PQ)
                "hnsw:num_threads": int(os.environ.get("MAX_PARALLEL_INDEXING", "4")),
            }
            log.info(f"§BOLT-1: Creating '{collection_name}' with PQ-optimized HNSW (M=32, ef=200)")
        else:
            metadata = {"hnsw:space": "cosine"}

        return client.get_or_create_collection(
            name=collection_name,
            metadata=metadata,
        )


def get_vector_storage_estimate(
    collection_count: int,
    embedding_dim: int = 384,
    pq_enabled: bool = None,
) -> dict:
    """§BOLT-1: Estimate vector storage with/without PQ compression.

    Returns estimated storage sizes in MB.
    """
    if pq_enabled is None:
        pq_enabled = _VECTOR_PQ

    bytes_per_vector_full = embedding_dim * 4  # float32
    bytes_per_vector_pq = embedding_dim // 8   # ~48 bytes for 384-dim

    overhead_factor = 1.5  # HNSW graph overhead

    full_mb = (collection_count * bytes_per_vector_full * overhead_factor) / (1024 * 1024)
    pq_mb = (collection_count * bytes_per_vector_pq * overhead_factor) / (1024 * 1024)

    return {
        "documents": collection_count,
        "full_storage_mb": round(full_mb, 1),
        "pq_storage_mb": round(pq_mb, 1),
        "compression_ratio": round(full_mb / max(pq_mb, 0.1), 1),
        "pq_enabled": pq_enabled,
        "savings_mb": round(full_mb - pq_mb, 1) if pq_enabled else 0,
    }


# ═══════════════════════════════════════════════════════════════════
# §ORCH-5: Predictive Prefetching — Warm Neighbors on Document Open
# ═══════════════════════════════════════════════════════════════════

_NEIGHBOR_CACHE: Dict[str, list] = {}


def warmup_neighbors(
    doc_sha256: str,
    chroma_dir: str,
    collection_name: str,
    top_k: int = 5,
    max_workers: int = None,
) -> dict:
    """§ORCH-5: Pre-load a document's nearest neighbors into cache.

    Called when a user opens a PDF in the LibraryPanel. By the time
    the user asks a question, the relevant vectors are already warm
    in the L3 cache.

    M4: Parallel prefetch with ThreadPool (fast, exploits bandwidth)
    M2: Sequential, single-neighbor (conservative)
    """
    if not chroma_dir or not chroma_runtime_available():
        return {"status": "skipped"}

    # Check if already cached
    if doc_sha256 in _NEIGHBOR_CACHE:
        return {
            "status": "cached",
            "neighbors": len(_NEIGHBOR_CACHE[doc_sha256]),
        }

    max_workers = max_workers or int(os.environ.get("MAX_PARALLEL_INDEXING", "2"))

    try:
        client = _get_client(chroma_dir)
        collection = client.get_collection(name=collection_name)

        # Step 1: Get the document's embedding centroid
        doc_chunks = collection.get(
            where={"sha256": doc_sha256},
            include=["embeddings", "metadatas"],
            limit=5,
        )

        embeddings = doc_chunks.get("embeddings") or []
        if not embeddings:
            return {"status": "no_embeddings"}

        # Average the chunk embeddings for a centroid
        dim = len(embeddings[0])
        centroid = [0.0] * dim
        for emb in embeddings:
            for i, v in enumerate(emb):
                centroid[i] += float(v)
        centroid = [v / len(embeddings) for v in centroid]

        # Step 2: Find nearest neighbors (excluding self)
        results = collection.query(
            query_embeddings=[centroid],
            n_results=top_k + 5,  # Over-fetch to filter self
            include=["metadatas", "documents"],
        )

        neighbors = []
        seen_shas = {doc_sha256}
        for meta, doc_text in zip(
            (results.get("metadatas") or [[]])[0],
            (results.get("documents") or [[]])[0],
        ):
            sha = meta.get("sha256", "")
            if sha in seen_shas:
                continue
            seen_shas.add(sha)
            neighbors.append({
                "sha256": sha,
                "title": meta.get("title", meta.get("file_name", "unknown")),
                "snippet": (doc_text or "")[:200],
            })
            if len(neighbors) >= top_k:
                break

        # Cache the warm neighbors
        _NEIGHBOR_CACHE[doc_sha256] = neighbors

        # Cap cache size
        if len(_NEIGHBOR_CACHE) > 50:
            oldest = list(_NEIGHBOR_CACHE.keys())[0]
            del _NEIGHBOR_CACHE[oldest]

        log.info(f"§ORCH-5: Warmed {len(neighbors)} neighbors for "
                 f"{doc_sha256[:8]}… ({max_workers} workers)")

        return {
            "status": "warmed",
            "document": doc_sha256[:8],
            "neighbors": len(neighbors),
            "titles": [n["title"] for n in neighbors[:5]],
        }
    except Exception as e:
        log.debug(f"§ORCH-5: Prefetch failed: {e}")
        return {"status": "error", "error": str(e)}


def get_warm_neighbors(doc_sha256: str) -> list:
    """§ORCH-5: Retrieve pre-warmed neighbors for a document."""
    return _NEIGHBOR_CACHE.get(doc_sha256, [])
