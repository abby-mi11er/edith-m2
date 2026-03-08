"""
Pipeline Utilities — Performance & Reliability
===============================================
Implements: timing telemetry, source caching, prompt compression,
connection pooling, circuit breakers (with half-open state),
smart routing, quality scoring, and configurable endpoint paths.

§8.1: Half-open circuit breaker state
§8.2: Sentence-boundary truncation
§8.3: Thread-safe PipelineTimer
§8.5: Explicit httpx connect/read timeouts
§8.6: Documented score_source_relevance
§8.7: Expanded skip-retrieval patterns
§8.9: Cache key includes store_id + top_k
§8.10: Configurable OpenAI endpoint path
"""

import hashlib
import json
import logging
import os
import re
import threading
import time
from typing import Optional

import httpx

log = logging.getLogger("edith.pipeline")


# ─── 1. TIMING TELEMETRY (§8.3: thread-safe) ───────────────────────

class PipelineTimer:
    """Track per-stage timing through the pipeline.  Thread-safe (§8.3)."""

    def __init__(self):
        self._stages: dict[str, float] = {}
        self._starts: dict[str, float] = {}
        self._t0 = time.monotonic()
        self._lock = threading.Lock()

    def start(self, stage: str):
        with self._lock:
            self._starts[stage] = time.monotonic()

    def stop(self, stage: str):
        with self._lock:
            if stage in self._starts:
                self._stages[stage] = round(
                    (time.monotonic() - self._starts[stage]) * 1000
                )
                del self._starts[stage]

    def elapsed_ms(self, stage: str) -> float:
        """Return elapsed ms for a stage, or 0 if not recorded."""
        with self._lock:
            return self._stages.get(stage, 0)

    def as_dict(self) -> dict:
        with self._lock:
            total = round((time.monotonic() - self._t0) * 1000)
            return {**self._stages, "total_ms": total}

    def summary(self) -> str:
        with self._lock:
            parts = []
            for k, v in self._stages.items():
                if v >= 1000:
                    parts.append(f"{k}: {v / 1000:.1f}s")
                else:
                    parts.append(f"{k}: {v}ms")
            total = round((time.monotonic() - self._t0) * 1000)
            parts.append(f"total: {total / 1000:.1f}s")
            return " | ".join(parts)


# ─── 2. SOURCE CACHE (§8.9: key includes store_id + top_k) ─────────

_source_cache: dict[str, tuple[float, list[dict]]] = {}
_cache_lock = threading.Lock()
SOURCE_CACHE_TTL = 900  # 15 minutes


def cached_retrieve(
    query: str, store_id: str, retrieve_fn, top_k: int = 15, **kwargs
) -> tuple[list[dict], bool]:
    """Cache wrapper for source retrieval.  Returns (sources, was_cached).

    §8.9: Cache key includes store_id and top_k.
    """
    # §8.9: full key includes query, store, and k
    key = hashlib.md5(f"{query}:{store_id}:{top_k}".encode()).hexdigest()

    now = time.time()
    with _cache_lock:
        if key in _source_cache:
            ts, sources = _source_cache[key]
            if now - ts < SOURCE_CACHE_TTL:
                log.info(f"Source cache HIT for: {query[:60]}")
                return sources, True

    sources = retrieve_fn(query=query, store_id=store_id, top_k=top_k, **kwargs)

    with _cache_lock:
        if sources:
            _source_cache[key] = (now, sources)

        # Evict ALL expired entries, then cap at 200
        expired = [k for k, (ts, _) in _source_cache.items() if now - ts >= SOURCE_CACHE_TTL]
        for k in expired:
            del _source_cache[k]
        while len(_source_cache) > 200:
            oldest_key = min(_source_cache, key=lambda k: _source_cache[k][0])
            del _source_cache[oldest_key]

    return sources, False


# ─── 3. PROMPT COMPRESSION (§8.2: sentence-boundary truncation) ─────

_SENTENCE_END_RE = re.compile(r"[.!?]\s+", re.MULTILINE)


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate text at the last complete sentence before max_chars (§8.2)."""
    if len(text) <= max_chars:
        return text
    chunk = text[:max_chars]
    # Find the last sentence boundary
    matches = list(_SENTENCE_END_RE.finditer(chunk))
    if matches:
        return chunk[: matches[-1].end()].rstrip()
    # No sentence boundary found, fall back to word boundary
    last_space = chunk.rfind(" ")
    if last_space > max_chars // 2:
        return chunk[:last_space] + "…"
    return chunk + "…"


def compress_sources(
    sources: list[dict],
    max_chars_per_source: int = 800,
    max_sources: int = 8,
) -> list[dict]:
    """Compress sources to reduce token count.  §8.2: sentence-boundary truncation."""
    compressed = []
    for s in sources[:max_sources]:
        # §FIX: Check BOTH 'text' and 'snippet' — local retrieval uses 'snippet',
        # Google retrieval uses 'text'. This was causing the "unable to answer" bug.
        text = s.get("text", "") or s.get("snippet", "") or ""
        if len(text) > max_chars_per_source:
            half = max_chars_per_source // 2
            first = _truncate_at_sentence(text, half)
            last = text[-half:]
            text = first + " [...] " + last

        # Preserve the full source dict so downstream functions
        # (build_support_audit_source_blocks) can access author, year,
        # title, uri, metadata, etc. for meaningful labels.
        entry = dict(s)  # shallow copy all fields
        entry["text"] = text
        # Also set snippet to the compressed text for compatibility
        entry["snippet"] = text
        compressed.append(entry)

    return compressed


# ─── 4. CONNECTION POOLING (§8.5: explicit timeouts, §8.10: configurable path) ─────

_openai_client: Optional[httpx.Client] = None
_openai_lock = threading.Lock()

# §8.10: configurable endpoint path
_OPENAI_CHAT_PATH = os.environ.get("EDITH_OPENAI_CHAT_PATH", "/chat/completions")


def get_openai_client(
    api_key: str, base_url: str = "https://api.openai.com/v1"
) -> httpx.Client:
    """Get or create a pooled HTTP client for OpenAI API calls.

    §8.5: Explicit connect=5s, read=30s timeouts.
    """
    global _openai_client
    with _openai_lock:
        if _openai_client is None or _openai_client.is_closed:
            _openai_client = httpx.Client(
                base_url=base_url,
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                # §8.5: explicit timeouts
                timeout=httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=10.0),
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
            )
            log.info("Created pooled OpenAI HTTP client")
    return _openai_client


def shutdown_pool():
    """Close the httpx connection pool on server shutdown."""
    global _openai_client
    with _openai_lock:
        if _openai_client and not _openai_client.is_closed:
            _openai_client.close()
            log.info("Closed pooled OpenAI HTTP client")
            _openai_client = None


def call_openai_pooled(
    api_key: str,
    model: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.1,
    max_tokens: int = 2000,
    base_url: str = "https://api.openai.com/v1",
) -> tuple[str, str]:
    """Call OpenAI with connection pooling.  Returns (answer, model_name)."""
    client = get_openai_client(api_key, base_url)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = client.post(_OPENAI_CHAT_PATH, json=payload)
    resp.raise_for_status()
    data = resp.json()

    return data["choices"][0]["message"]["content"].strip(), model


def call_openai_streaming(
    api_key: str,
    model: str,
    prompt: str,
    system: str = "",
    temperature: float = 0.1,
    max_tokens: int = 2000,
    base_url: str = "https://api.openai.com/v1",
):
    """Stream OpenAI tokens.  Yields text chunks as they arrive."""
    client = get_openai_client(api_key, base_url)

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }

    with client.stream("POST", _OPENAI_CHAT_PATH, json=payload) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue


# ─── 5. CIRCUIT BREAKER (§8.1: half-open state) ─────────────────────

class CircuitBreaker:
    """Circuit breaker with three states: closed → open → half-open → closed/open.

    §8.1: Half-open state allows exactly 1 test request after cooldown.
    """

    def __init__(self, name: str, failure_threshold: int = 2, cooldown: int = 300):
        self.name = name
        self.failures = 0
        self.last_failure_time = 0.0
        self.threshold = failure_threshold
        self.cooldown = cooldown
        self._state = "closed"  # closed | open | half-open
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        """True if circuit is tripped (should not call the service)."""
        with self._lock:
            if self._state == "closed":
                return False
            if self._state == "half-open":
                return False  # allow the test request
            # state == "open"
            if time.time() - self.last_failure_time >= self.cooldown:
                self._state = "half-open"
                log.info(f"Circuit breaker [{self.name}] → HALF-OPEN (allowing test request)")
                return False
            return True

    def record_success(self):
        with self._lock:
            if self._state == "half-open":
                log.info(f"Circuit breaker [{self.name}] → CLOSED (test request succeeded)")
            self.failures = 0
            self._state = "closed"

    def record_failure(self):
        with self._lock:
            self.failures += 1
            self.last_failure_time = time.time()
            if self._state == "half-open":
                # Test failed — go back to open
                self._state = "open"
                log.warning(f"Circuit breaker [{self.name}] → OPEN (test request failed)")
            elif self.failures >= self.threshold:
                self._state = "open"
                log.warning(
                    f"Circuit breaker [{self.name}] TRIPPED after {self.failures} failures. "
                    f"Cooldown: {self.cooldown}s"
                )

    def status(self) -> dict:
        with self._lock:
            return {
                "name": self.name,
                "state": self._state,
                "failures": self.failures,
                "cooldown_remaining": max(
                    0, self.cooldown - (time.time() - self.last_failure_time)
                ) if self._state == "open" else 0,
            }


# Global circuit breakers
openai_breaker = CircuitBreaker("openai", failure_threshold=2, cooldown=300)
google_retrieval_breaker = CircuitBreaker("google_retrieval", failure_threshold=3, cooldown=120)


# ─── 7. RETRIEVAL QUALITY SCORING (§8.6: documented) ────────────────

def score_source_relevance(
    query: str, sources: list[dict], min_score: float = 0.15
) -> list[dict]:
    """Fast relevance scoring using keyword overlap.

    §8.6: Score range is [0.0, 1.0].
    - 0.0 = no query keywords found in source
    - 1.0 = all query keywords found, plus title bonus

    Filters out sources below min_score threshold.
    """
    query_words = set(w.lower() for w in query.split() if len(w) > 3)
    if not query_words:
        return sources

    scored = []
    for s in sources:
        text = s.get("text", "").lower()
        title = s.get("meta", {}).get("title", "").lower()
        combined = text + " " + title

        matches = sum(1 for w in query_words if w in combined)
        overlap = matches / len(query_words) if query_words else 0

        title_matches = sum(1 for w in query_words if w in title)
        title_boost = title_matches * 0.1

        relevance = min(1.0, overlap + title_boost)

        if relevance >= min_score:
            s_copy = dict(s)
            s_copy["relevance_score"] = round(relevance, 3)
            scored.append(s_copy)

    scored.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)

    if len(scored) < len(sources):
        log.info(
            f"Quality filter: {len(sources)} → {len(scored)} sources "
            f"(dropped {len(sources) - len(scored)} low-relevance)"
        )

    return scored


# ─── 8. SMART MODEL ROUTING (§8.7: expanded patterns) ───────────────

_RETRIEVAL_KEYWORDS = {
    "what", "explain", "describe", "compare", "analyze", "evaluate",
    "how", "why", "discuss", "define", "evidence", "research",
    "study", "studies", "paper", "papers", "author", "theory",
    "according", "literature", "findings", "data",
}

# §8.7: expanded skip patterns
_SKIP_RETRIEVAL_PATTERNS = {
    "hello", "hi ", "hey ", "hey!", "hi!", "thanks", "thank you", "help",
    "who are you", "what can you do", "how do you work",
    "good morning", "good afternoon", "good evening", "goodbye", "bye",
    "hi there", "hey there", "what's up", "sup",
}


def should_skip_retrieval(query: str) -> bool:
    """Determine if a query can be answered without retrieval.

    Returns True for greetings, meta questions, and very short queries.
    """
    q_lower = query.lower().strip()

    if len(q_lower.split()) <= 3:
        for pattern in _SKIP_RETRIEVAL_PATTERNS:
            if pattern in q_lower:
                return True

    return False
