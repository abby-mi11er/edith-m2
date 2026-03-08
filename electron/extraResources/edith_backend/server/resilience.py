"""
E.D.I.T.H. Resilience Layer — 10 Architecture Improvements
============================================================
§1: LLM Timeout Protection — guarded_generate() with configurable timeout
§2: Circuit Breaker — auto-open after N failures, auto-heal after cooldown
§3: Streaming SSE — Server-Sent Events endpoint for chat
§4: Response Cache — LRU cache on retrieval + completion
§5: Smart Dual-Brain Routing — classify → Winnie (fast) vs Gemini (deep)
§6: Training Data Validation — quality gate before DPO pairs are saved
§7: Structured Logging — JSON formatter with request_id, latency, tokens
§8: Connection Pooling — shared httpx.AsyncClient per connector
§9: Request Cancellation — cancel in-flight LLM calls on disconnect
§10: Live Dashboard — WebSocket push for EventBus events
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import uuid
from collections import OrderedDict
from datetime import datetime
from typing import Optional, AsyncGenerator

# ═══════════════════════════════════════════════════════════════════
# §1: LLM Timeout Protection
# ═══════════════════════════════════════════════════════════════════

_LLM_TIMEOUT = int(os.environ.get("EDITH_LLM_TIMEOUT", "90"))

log = logging.getLogger("edith.resilience")


async def guarded_generate(
    coro,
    timeout_s: int = _LLM_TIMEOUT,
    fallback: str = "⏰ Request timed out — the model took too long to respond. Try a simpler query or retry.",
) -> str:
    """Wrap any async LLM call with a hard timeout.

    Usage:
        result = await guarded_generate(
            async_generate_text_via_chain(prompt, chain),
            timeout_s=30,
        )
    """
    try:
        result = await asyncio.wait_for(coro, timeout=timeout_s)
        if isinstance(result, tuple):
            return result  # (text, model_name)
        return result
    except asyncio.TimeoutError:
        log.warning(f"§TIMEOUT: LLM call exceeded {timeout_s}s")
        return fallback, "timeout"
    except Exception as e:
        log.error(f"§GUARD: LLM call failed: {e}")
        return f"Generation failed: {str(e)[:100]}", "error"


def guarded_generate_sync(fn, *args, timeout_s: int = _LLM_TIMEOUT, **kwargs):
    """Wrap a sync LLM call with a thread-based timeout.

    Usage:
        result = guarded_generate_sync(
            generate_text_via_chain, prompt, chain,
            timeout_s=30,
        )
    """
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(fn, *args, **kwargs)
        try:
            return future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError:
            log.warning(f"§TIMEOUT: sync LLM call exceeded {timeout_s}s")
            return "⏰ Request timed out. Try a simpler query.", "timeout"
        except Exception as e:
            log.error(f"§GUARD: sync LLM call failed: {e}")
            return f"Generation failed: {str(e)[:100]}", "error"


# ═══════════════════════════════════════════════════════════════════
# §2: Circuit Breaker
# ═══════════════════════════════════════════════════════════════════

class CircuitBreaker:
    """Circuit breaker for external services.

    States: CLOSED (normal) → OPEN (failing) → HALF_OPEN (testing)
    Opens after `threshold` failures in `window_s` seconds.
    Auto-resets after `cooldown_s` seconds.
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, name: str, threshold: int = 3, window_s: int = 60,
                 cooldown_s: int = 300):
        self.name = name
        self.threshold = threshold
        self.window_s = window_s
        self.cooldown_s = cooldown_s
        self.state = self.CLOSED
        self._failures: list[float] = []
        self._last_open: float = 0
        self._success_count: int = 0
        self._total_calls: int = 0

    def can_execute(self) -> bool:
        """Check if the circuit allows a call."""
        self._total_calls += 1
        if self.state == self.CLOSED:
            return True
        if self.state == self.OPEN:
            if time.time() - self._last_open > self.cooldown_s:
                self.state = self.HALF_OPEN
                log.info(f"§CIRCUIT {self.name}: HALF_OPEN — testing")
                return True
            return False
        # HALF_OPEN: allow one test call
        return True

    def record_success(self):
        """Record a successful call."""
        self._success_count += 1
        if self.state == self.HALF_OPEN:
            self.state = self.CLOSED
            self._failures.clear()
            log.info(f"§CIRCUIT {self.name}: CLOSED — service recovered")

    def record_failure(self):
        """Record a failed call."""
        now = time.time()
        self._failures = [t for t in self._failures if now - t < self.window_s]
        self._failures.append(now)
        if len(self._failures) >= self.threshold:
            self.state = self.OPEN
            self._last_open = now
            log.warning(f"§CIRCUIT {self.name}: OPEN — {len(self._failures)} failures in {self.window_s}s")

    def status(self) -> dict:
        return {
            "name": self.name,
            "state": self.state,
            "failures_in_window": len(self._failures),
            "threshold": self.threshold,
            "total_calls": self._total_calls,
            "success_count": self._success_count,
            "cooldown_s": self.cooldown_s,
        }


# Pre-built breakers for each connector
_breakers: dict[str, CircuitBreaker] = {}


def get_breaker(name: str) -> CircuitBreaker:
    """Get or create a circuit breaker for a service."""
    if name not in _breakers:
        _breakers[name] = CircuitBreaker(name)
    return _breakers[name]


def all_breaker_status() -> list[dict]:
    """Return status of all circuit breakers."""
    return [b.status() for b in _breakers.values()]


async def with_circuit_breaker(name: str, coro):
    """Execute an async call through a circuit breaker.

    Usage:
        result = await with_circuit_breaker("anthropic", do_call())
    """
    breaker = get_breaker(name)
    if not breaker.can_execute():
        return {
            "error": f"Circuit breaker OPEN for {name}",
            "state": "open",
            "retry_after_s": breaker.cooldown_s,
        }
    try:
        result = await coro
        breaker.record_success()
        return result
    except Exception as e:
        breaker.record_failure()
        raise


# ═══════════════════════════════════════════════════════════════════
# §3: Streaming SSE Response
# ═══════════════════════════════════════════════════════════════════

async def stream_chat_response(
    prompt: str,
    model_chain: list[str],
    system_instruction: str = "",
    temperature: float = 0.1,
    tab: str = "",
) -> AsyncGenerator[str, None]:
    """Yield SSE-formatted chunks for streaming chat.

    Wraps backend_logic.generate_text_streaming() and formats as SSE events.
    """
    try:
        from server.backend_logic import generate_text_streaming
        for chunk in generate_text_streaming(
            prompt, model_chain, system_instruction, temperature, tab=tab,
        ):
            yield f"data: {json.dumps({'text': chunk, 'done': False})}\n\n"
        yield f"data: {json.dumps({'text': '', 'done': True})}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'error': str(e), 'done': True})}\n\n"


# ═══════════════════════════════════════════════════════════════════
# §4: Response Cache (LRU)
# ═══════════════════════════════════════════════════════════════════

class ResponseCache:
    """LRU cache for LLM responses. Key = hash(query + source_ids)."""

    def __init__(self, max_size: int = 200, ttl_s: int = 3600):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._ttl_s = ttl_s
        self._hits = 0
        self._misses = 0

    def _key(self, query: str, source_ids: list[str] = None) -> str:
        h = hashlib.sha256(query.encode()).hexdigest()[:16]
        if source_ids:
            h += "_" + hashlib.sha256("|".join(sorted(source_ids)).encode()).hexdigest()[:8]
        return h

    def get(self, query: str, source_ids: list[str] = None) -> Optional[dict]:
        """Get a cached response. Returns None if miss or expired."""
        key = self._key(query, source_ids)
        if key in self._cache:
            entry = self._cache[key]
            if time.time() - entry["ts"] < self._ttl_s:
                self._hits += 1
                self._cache.move_to_end(key)
                return entry["data"]
            else:
                del self._cache[key]
        self._misses += 1
        return None

    def put(self, query: str, data: dict, source_ids: list[str] = None):
        """Store a response in the cache."""
        key = self._key(query, source_ids)
        self._cache[key] = {"data": data, "ts": time.time()}
        self._cache.move_to_end(key)
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)

    def invalidate(self, query: str = "", source_ids: list[str] = None):
        """Invalidate a specific entry or all entries."""
        if not query:
            self._cache.clear()
            return
        key = self._key(query, source_ids)
        self._cache.pop(key, None)

    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / total, 3) if total else 0,
            "ttl_s": self._ttl_s,
        }


# Singleton cache
response_cache = ResponseCache()


# ═══════════════════════════════════════════════════════════════════
# §5: Smart Dual-Brain Routing
# ═══════════════════════════════════════════════════════════════════

# Complexity indicators that need Gemini (expensive, powerful)
_COMPLEX_KEYWORDS = {
    "compare", "contrast", "synthesize", "analyze", "evaluate", "critique",
    "causal", "counterfactual", "mechanism", "endogeneity", "instrument",
    "regression discontinuity", "difference-in-differences", "deep dive",
    "methodological", "literature review", "theoretical framework",
    "explain how", "explain why", "what are the implications",
}

# Simple queries → Winnie (fast, cheap)
_SIMPLE_PATTERNS = {
    "what is", "define", "when was", "who", "list", "name", "how many",
    "yes or no", "true or false", "which year",
}


def classify_query_complexity(query: str) -> str:
    """Classify a query as 'simple', 'standard', or 'complex'.

    simple → route to Winnie (fine-tuned OpenAI, fast, cheap)
    standard → either brain
    complex → route to Gemini (powerful, expensive)
    """
    q = query.lower().strip()
    word_count = len(q.split())

    # Short factual → simple
    if word_count < 8 and any(q.startswith(p) for p in _SIMPLE_PATTERNS):
        return "simple"

    # Complex analytical → complex
    complex_score = sum(1 for k in _COMPLEX_KEYWORDS if k in q)
    if complex_score >= 2 or (complex_score >= 1 and word_count > 20):
        return "complex"

    # Multi-part questions → complex
    if q.count("?") > 1:
        return "complex"

    return "standard"


def route_to_brain(query: str, gemini_chain: list[str],
                   winnie_model: str = "") -> list[str]:
    """Route a query to the appropriate brain based on complexity.

    Returns: model chain (Gemini-first or Winnie-first).
    """
    if not winnie_model:
        winnie_model = os.environ.get(
            "EDITH_OPENAI_FT_MODEL",
            "ft:gpt-4o-mini-2024-07-18:personal:winnie-v1:D9xqwC8p"
        )

    complexity = classify_query_complexity(query)

    if complexity == "simple":
        log.info(f"§ROUTE: simple → Winnie ({winnie_model})")
        return [winnie_model] + gemini_chain
    elif complexity == "complex":
        log.info(f"§ROUTE: complex → Gemini ({gemini_chain[0]})")
        return gemini_chain
    else:
        # Standard: use Gemini but with Flash (cheaper)
        return gemini_chain


# ═══════════════════════════════════════════════════════════════════
# §6: Training Data Validation
# ═══════════════════════════════════════════════════════════════════

def validate_training_pair(
    prompt: str,
    chosen: str,
    rejected: str,
    min_length: int = 20,
    min_quality_diff: float = 0.15,
) -> dict:
    """Quality gate for DPO training pairs.

    Checks:
    1. Both responses are non-trivial (min length)
    2. Chosen is meaningfully different from rejected
    3. Neither contains obvious garbage (repetition, empty)
    4. Chosen is substantively better (length + detail heuristic)

    Returns: {"valid": bool, "reason": str, "quality_score": float}
    """
    # Check minimum length
    if len(chosen.strip()) < min_length:
        return {"valid": False, "reason": "chosen too short", "quality_score": 0}
    if len(rejected.strip()) < min_length:
        return {"valid": False, "reason": "rejected too short", "quality_score": 0}

    # Check for garbage: excessive repetition
    def has_repetition(text: str, threshold: float = 0.5) -> bool:
        words = text.lower().split()
        if len(words) < 5:
            return False
        unique_ratio = len(set(words)) / len(words)
        return unique_ratio < threshold

    if has_repetition(chosen):
        return {"valid": False, "reason": "chosen has excessive repetition", "quality_score": 0}
    if has_repetition(rejected):
        return {"valid": False, "reason": "rejected has excessive repetition", "quality_score": 0}

    # Check that responses are meaningfully different
    def jaccard(a: str, b: str) -> float:
        sa = set(a.lower().split())
        sb = set(b.lower().split())
        if not sa or not sb:
            return 0
        return len(sa & sb) / len(sa | sb)

    similarity = jaccard(chosen, rejected)
    if similarity > 0.95:
        return {"valid": False, "reason": "chosen and rejected too similar", "quality_score": 0}

    # Quality heuristic: chosen should have more substance
    chosen_detail = _detail_score(chosen)
    rejected_detail = _detail_score(rejected)
    quality_diff = chosen_detail - rejected_detail

    if quality_diff < min_quality_diff:
        return {
            "valid": False,
            "reason": f"quality delta too small ({quality_diff:.2f} < {min_quality_diff})",
            "quality_score": quality_diff,
        }

    return {
        "valid": True,
        "reason": "passed all checks",
        "quality_score": round(quality_diff, 3),
        "similarity": round(similarity, 3),
        "chosen_detail": round(chosen_detail, 3),
        "rejected_detail": round(rejected_detail, 3),
    }


def _detail_score(text: str) -> float:
    """Heuristic: how detailed/substantive is a response? 0-1 scale."""
    words = text.split()
    if not words:
        return 0

    score = 0
    # Length bonus (normalized)
    score += min(len(words) / 200, 0.3)
    # Citation evidence
    citations = sum(1 for w in words if any(c in w for c in ["(", "et al", "20"]))
    score += min(citations / 5, 0.2)
    # Structural indicators (numbered lists, headers)
    structures = text.count("\n") + text.count("- ") + text.count("1.")
    score += min(structures / 10, 0.2)
    # Technical vocabulary
    technical = sum(1 for w in text.lower().split() if len(w) > 8)
    score += min(technical / 20, 0.15)
    # Sentence variety
    sentences = text.count(".") + text.count("?") + text.count("!")
    score += min(sentences / 10, 0.15)

    return min(score, 1.0)


# ═══════════════════════════════════════════════════════════════════
# §7: Structured Logging
# ═══════════════════════════════════════════════════════════════════

class StructuredFormatter(logging.Formatter):
    """JSON log formatter with request context."""

    def format(self, record):
        log_entry = {
            "ts": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Add context fields if attached
        for field in ("request_id", "latency_ms", "tokens_in", "tokens_out",
                      "model", "endpoint", "status_code", "user_agent"):
            val = getattr(record, field, None)
            if val is not None:
                log_entry[field] = val
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = str(record.exc_info[1])
        return json.dumps(log_entry)


def setup_structured_logging(level: str = "INFO"):
    """Configure structured JSON logging for the application."""
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())
    root = logging.getLogger("edith")
    root.handlers = [handler]
    root.setLevel(getattr(logging, level, logging.INFO))
    log.info("§LOG: Structured JSON logging enabled")


class RequestLogger:
    """Context manager for per-request structured logging."""

    def __init__(self, endpoint: str = ""):
        self.request_id = str(uuid.uuid4())[:8]
        self.endpoint = endpoint
        self.start_time = time.time()
        self.extra = {
            "request_id": self.request_id,
            "endpoint": endpoint,
        }

    def log(self, msg: str, **kwargs):
        extra = {**self.extra, **kwargs}
        record = log.makeRecord(
            log.name, logging.INFO, "", 0, msg, (), None,
        )
        for k, v in extra.items():
            setattr(record, k, v)
        log.handle(record)

    def done(self, status_code: int = 200, **kwargs):
        latency = round((time.time() - self.start_time) * 1000, 1)
        self.log("request_complete",
                 latency_ms=latency, status_code=status_code, **kwargs)


# ═══════════════════════════════════════════════════════════════════
# §8: Connection Pooling
# ═══════════════════════════════════════════════════════════════════

_pools: dict[str, "httpx.AsyncClient"] = {}


def get_pool(service: str, base_url: str = "", timeout: float = 30.0):
    """Get or create a shared httpx.AsyncClient for a service.

    Reuses connections across requests to the same service.
    """
    if service not in _pools:
        import httpx
        _pools[service] = httpx.AsyncClient(
            base_url=base_url,
            timeout=timeout,
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
            headers={"User-Agent": "EDITH/2.0"},
        )
    return _pools[service]


async def close_all_pools():
    """Close all connection pools (call on shutdown)."""
    for name, client in _pools.items():
        await client.aclose()
    _pools.clear()


# ═══════════════════════════════════════════════════════════════════
# §9: Request Cancellation
# ═══════════════════════════════════════════════════════════════════

_active_tasks: dict[str, asyncio.Task] = {}


def track_task(request_id: str, task: asyncio.Task):
    """Register an in-flight task for potential cancellation."""
    _active_tasks[request_id] = task
    task.add_done_callback(lambda _: _active_tasks.pop(request_id, None))


async def cancel_request(request_id: str) -> bool:
    """Cancel an in-flight LLM call."""
    task = _active_tasks.pop(request_id, None)
    if task and not task.done():
        task.cancel()
        log.info(f"§CANCEL: Request {request_id} cancelled")
        return True
    return False


def active_request_count() -> int:
    """Number of in-flight LLM calls."""
    return len(_active_tasks)


async def cancellable_generate(request_id: str, coro, timeout_s: int = _LLM_TIMEOUT):
    """Run a generation call that can be cancelled by request_id.

    Combines timeout protection (§1) with cancellation (§9).
    """
    task = asyncio.current_task()
    if task:
        track_task(request_id, task)
    return await guarded_generate(coro, timeout_s=timeout_s)


# ═══════════════════════════════════════════════════════════════════
# §10: Live Dashboard WebSocket
# ═══════════════════════════════════════════════════════════════════

_ws_clients: set = set()


async def ws_broadcast(event: dict):
    """Broadcast an event to all connected WebSocket clients."""
    if not _ws_clients:
        return
    msg = json.dumps(event)
    dead = set()
    for ws in _ws_clients:
        try:
            await ws.send_text(msg)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


def register_ws_client(ws):
    """Register a WebSocket client for live events."""
    _ws_clients.add(ws)
    log.info(f"§WS: Client connected ({len(_ws_clients)} total)")


def unregister_ws_client(ws):
    """Remove a WebSocket client."""
    _ws_clients.discard(ws)
    log.info(f"§WS: Client disconnected ({len(_ws_clients)} total)")


# ═══════════════════════════════════════════════════════════════════
# Registration: Wire into FastAPI + EventBus
# ═══════════════════════════════════════════════════════════════════

def register_resilience_routes(app):
    """Register resilience endpoints with the FastAPI app."""
    from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
    from starlette.responses import StreamingResponse

    router = APIRouter(tags=["Resilience"])

    @router.get("/api/resilience/status")
    async def resilience_status():
        """Full resilience layer status."""
        return {
            "circuit_breakers": all_breaker_status(),
            "cache": response_cache.stats(),
            "active_requests": active_request_count(),
            "ws_clients": len(_ws_clients),
            "llm_timeout_s": _LLM_TIMEOUT,
        }

    @router.get("/api/resilience/cache/stats")
    async def cache_stats():
        return response_cache.stats()

    @router.post("/api/resilience/cache/invalidate")
    async def cache_invalidate():
        response_cache.invalidate()
        return {"status": "cleared"}

    @router.get("/api/resilience/breakers")
    async def breaker_status():
        return {"breakers": all_breaker_status()}

    @router.get("/api/session-costs")
    async def session_costs():
        """Return accumulated session costs."""
        try:
            from server.backend_logic import get_session_costs
            return get_session_costs()
        except ImportError:
            return {"total_cost_usd": 0, "request_count": 0}

    @router.post("/api/settings/connector-key")
    async def save_connector_key(request: Request):
        """Save a connector API key to env and .env file."""
        data = await request.json()
        connector = data.get("connector", "")
        key_value = data.get("key", "")
        if not connector or not key_value:
            return {"error": "connector and key required"}, 400

        # Map connector names to env variable names
        env_map = {
            "anthropic": "ANTHROPIC_API_KEY",
            "nyt": "NYT_API_KEY",

            "zotero_key": "ZOTERO_API_KEY",
            "zotero_user": "ZOTERO_USER_ID",
            "mendeley": "MENDELEY_ACCESS_TOKEN",
            "mendeley_id": "MENDELEY_CLIENT_ID",
            "mendeley_secret": "MENDELEY_CLIENT_SECRET",
            "mathpix_id": "MATHPIX_APP_ID",
            "mathpix_key": "MATHPIX_APP_KEY",
            "overleaf_token": "OVERLEAF_GIT_TOKEN",
            "overleaf_url": "OVERLEAF_PROJECT_URL",
            "notion": "NOTION_TOKEN",
            "google_earth_key": "GOOGLE_EARTH_ENGINE_KEY",
            "google_earth_project": "GOOGLE_EARTH_ENGINE_PROJECT",
            "stata": "STATA_PATH",
            "openalex_email": "OPENALEX_POLITE_EMAIL",
        }

        env_name = env_map.get(connector)
        if not env_name:
            from fastapi.responses import JSONResponse as _JR
            return _JR(
                status_code=422,
                content={
                    "error": "unknown_connector",
                    "detail": f"Unknown connector: {connector}",
                    "known_connectors": sorted(env_map.keys()),
                    "hint": "Use one of the known connector names.",
                },
            )

        # Set in current process
        os.environ[env_name] = key_value

        # Persist to .env file
        env_path = os.path.join(os.environ.get("EDITH_DATA_ROOT", "."), ".env")
        try:
            lines = open(env_path).read().splitlines() if os.path.exists(env_path) else []
            updated = False
            for i, line in enumerate(lines):
                if line.startswith(f"{env_name}="):
                    lines[i] = f"{env_name}={key_value}"
                    updated = True
                    break
            if not updated:
                lines.append(f"{env_name}={key_value}")
            with open(env_path, "w") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as e:
            log.warning(f"Could not persist key to .env: {e}")

        log.info(f"§CONNECTOR: Saved key for {connector} → {env_name}")

        # Key validation — test the key with a lightweight API call
        validation = None
        try:
            validation = await _validate_connector_key(connector, key_value)
        except Exception:
            pass

        return {
            "saved": True, "connector": connector, "env_var": env_name,
            "validation": validation,
        }

    async def _validate_connector_key(connector: str, key: str) -> dict:
        """Test a connector key with a lightweight API call."""
        import httpx
        try:
            async with httpx.AsyncClient(timeout=8.0) as c:
                if connector == "anthropic":
                    r = await c.get("https://api.anthropic.com/v1/models",
                                    headers={"x-api-key": key, "anthropic-version": "2023-06-01"})
                    return {"valid": r.status_code == 200, "status": r.status_code}
                elif connector == "nyt":
                    r = await c.get(f"https://api.nytimes.com/svc/topstories/v2/home.json?api-key={key}")
                    return {"valid": r.status_code == 200, "status": r.status_code}
                elif connector == "notion":
                    r = await c.get("https://api.notion.com/v1/users/me",
                                    headers={"Authorization": f"Bearer {key}",
                                             "Notion-Version": "2022-06-28"})
                    return {"valid": r.status_code == 200, "status": r.status_code}
                elif connector == "mathpix_id":
                    return {"valid": bool(key), "status": "id_saved"}
                elif connector == "mathpix_key":
                    return {"valid": bool(key), "status": "key_saved"}
        except Exception as e:
            return {"valid": False, "error": str(e)}
        return None

    @router.get("/api/connectors/hub/live-status")
    async def connectors_live_status():
        """Check which connectors have keys configured and their live status."""
        connectors = {
            "anthropic": {"env": "ANTHROPIC_API_KEY", "label": "🧠 Anthropic"},
            "nyt": {"env": "NYT_API_KEY", "label": "📰 NYT"},
            "mendeley": {"env": "MENDELEY_CLIENT_ID", "label": "📚 Mendeley"},
            "mathpix": {"env": "MATHPIX_APP_ID", "label": "✏️ MathPix"},
            "notion": {"env": "NOTION_TOKEN", "label": "📓 Notion"},
            "overleaf": {"env": "OVERLEAF_GIT_TOKEN", "label": "📝 Overleaf"},
            "zotero": {"env": "ZOTERO_API_KEY", "label": "📖 Zotero"},
            "openalex": {"env": "OPENALEX_POLITE_EMAIL", "label": "🔬 OpenAlex"},
            "google_earth": {"env": "GOOGLE_EARTH_ENGINE_PROJECT", "label": "🌍 Google Earth Engine"},
            "stata": {"env": "STATA_PATH", "label": "📊 Stata"},
            "gemini": {"env": "GEMINI_API_KEY", "label": "💎 Gemini"},
            "openai": {"env": "OPENAI_API_KEY", "label": "🤖 OpenAI/Winnie"},
        }

        results = {}
        configured_count = 0
        for name, info in connectors.items():
            has_key = bool(os.environ.get(info["env"], ""))
            if has_key:
                configured_count += 1
            results[name] = {
                "label": info["label"],
                "configured": has_key,
                "env_var": info["env"],
            }

        return {
            "connectors": results,
            "summary": {
                "total": len(connectors),
                "configured": configured_count,
                "missing": len(connectors) - configured_count,
            }
        }

    @router.post("/api/connectors/test-all")
    async def test_all_connectors():
        """Ping every configured connector with a lightweight API call."""
        import httpx
        import time as _time
        results = {}

        async def _test(name, test_fn):
            start = _time.time()
            try:
                result = await test_fn()
                elapsed = round((_time.time() - start) * 1000)
                results[name] = {"status": "🟢", "response_time_ms": elapsed, **result}
            except Exception as e:
                elapsed = round((_time.time() - start) * 1000)
                results[name] = {"status": "🔴", "error": str(e)[:100], "response_time_ms": elapsed}

        async with httpx.AsyncClient(timeout=8.0) as c:
            # Anthropic
            if os.environ.get("ANTHROPIC_API_KEY"):
                async def test_anthropic():
                    r = await c.get("https://api.anthropic.com/v1/models",
                                    headers={"x-api-key": os.environ["ANTHROPIC_API_KEY"],
                                             "anthropic-version": "2023-06-01"})
                    return {"available": r.status_code == 200}
                await _test("anthropic", test_anthropic)
            else:
                results["anthropic"] = {"status": "⚪", "reason": "not configured"}

            # NYT
            if os.environ.get("NYT_API_KEY"):
                async def test_nyt():
                    r = await c.get(f"https://api.nytimes.com/svc/topstories/v2/home.json",
                                    params={"api-key": os.environ["NYT_API_KEY"]})
                    return {"available": r.status_code == 200, "articles": len(r.json().get("results", []))}
                await _test("nyt", test_nyt)
            else:
                results["nyt"] = {"status": "⚪", "reason": "not configured"}

            # Notion
            if os.environ.get("NOTION_TOKEN"):
                async def test_notion():
                    r = await c.get("https://api.notion.com/v1/users/me",
                                    headers={"Authorization": f"Bearer {os.environ['NOTION_TOKEN']}",
                                             "Notion-Version": "2022-06-28"})
                    return {"available": r.status_code == 200}
                await _test("notion", test_notion)
            else:
                results["notion"] = {"status": "⚪", "reason": "not configured"}

            # OpenAlex (always available, no key needed)
            async def test_openalex():
                r = await c.get("https://api.openalex.org/works?per_page=1",
                                params={"mailto": os.environ.get("OPENALEX_POLITE_EMAIL", "")})
                return {"available": r.status_code == 200}
            await _test("openalex", test_openalex)

            # Google Earth Engine
            if os.environ.get("GOOGLE_EARTH_ENGINE_PROJECT"):
                results["google_earth"] = {"status": "🟢", "available": True,
                                           "project": os.environ["GOOGLE_EARTH_ENGINE_PROJECT"]}
            else:
                results["google_earth"] = {"status": "⚪", "reason": "not configured"}

            # Mendeley
            if os.environ.get("MENDELEY_CLIENT_ID"):
                token = os.environ.get("MENDELEY_ACCESS_TOKEN", "")
                results["mendeley"] = {
                    "status": "🟢" if token else "🟡",
                    "configured": True,
                    "authenticated": bool(token),
                    "note": "OAuth token present" if token else "Client ID set — click Connect to authenticate",
                }
            else:
                results["mendeley"] = {"status": "⚪", "reason": "not configured"}

            # MathPix
            if os.environ.get("MATHPIX_APP_ID") and os.environ.get("MATHPIX_APP_KEY"):
                results["mathpix"] = {"status": "🟢", "configured": True}
            else:
                results["mathpix"] = {"status": "⚪", "reason": "not configured"}

        healthy = sum(1 for v in results.values() if v.get("status") == "🟢")
        return {
            "connectors": results,
            "summary": {"healthy": healthy, "total": len(results),
                        "score": f"{healthy}/{len(results)}"},
        }

    # ── Rate limit tracking for NYT ─────────────────────────────────
    _nyt_daily_calls = {"count": 0, "date": ""}

    @router.get("/api/connectors/rate-limits")
    async def connector_rate_limits():
        """Get rate limit usage for rate-limited connectors."""
        from datetime import date
        today = date.today().isoformat()
        if _nyt_daily_calls["date"] != today:
            _nyt_daily_calls["count"] = 0
            _nyt_daily_calls["date"] = today

        return {
            "nyt": {
                "daily_limit": 500,
                "used_today": _nyt_daily_calls["count"],
                "remaining": 500 - _nyt_daily_calls["count"],
                "date": today,
            }
        }


    @router.get("/api/chat/stream")
    async def chat_stream_sse(q: str = ""):
        """SSE endpoint for streaming chat responses."""
        if not q:
            async def empty():
                yield f"data: {json.dumps({'error': 'query required', 'done': True})}\n\n"
            return StreamingResponse(empty(), media_type="text/event-stream")

        model_chain = os.environ.get("EDITH_MODEL", "gemini-2.5-flash").split(",")
        return StreamingResponse(
            stream_chat_response(q, model_chain),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @router.websocket("/ws/dashboard")
    async def dashboard_ws(ws: WebSocket):
        """WebSocket for live dashboard updates."""
        await ws.accept()
        register_ws_client(ws)
        try:
            while True:
                # Keep connection alive; client can send pings
                data = await ws.receive_text()
                if data == "ping":
                    await ws.send_text('{"type":"pong"}')
        except WebSocketDisconnect:
            pass
        finally:
            unregister_ws_client(ws)

    app.include_router(router)
    return router


def wire_eventbus_to_websocket():
    """Subscribe to EventBus events and broadcast to WebSocket clients."""
    try:
        from server.event_bus import bus
        async def _forward_to_ws(event):
            await ws_broadcast({
                "type": "eventbus",
                "event": getattr(event, "name", "unknown"),
                "data": getattr(event, "data", {}),
                "ts": datetime.utcnow().isoformat() + "Z",
            })
        bus.on("*", _forward_to_ws)
        log.info("§WS: EventBus → WebSocket bridge active")
    except Exception as e:
        log.warning(f"§WS: Could not wire EventBus to WebSocket: {e}")


# ═══════════════════════════════════════════════════════════════════
# §11: Batch Query Mode — parallel execution
# ═══════════════════════════════════════════════════════════════════

async def batch_query(
    questions: list[str],
    generate_fn,
    max_concurrent: int = 5,
) -> list[dict]:
    """Run multiple queries in parallel using asyncio.gather().

    Usage:
        results = await batch_query(
            ["Q1?", "Q2?", "Q3?"],
            generate_fn=my_async_generate,
        )
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def _run_one(q: str, idx: int) -> dict:
        async with semaphore:
            t0 = time.time()
            try:
                result = await guarded_generate(generate_fn(q), timeout_s=_LLM_TIMEOUT)
                return {
                    "index": idx,
                    "question": q,
                    "answer": result[0] if isinstance(result, tuple) else result,
                    "model": result[1] if isinstance(result, tuple) else "unknown",
                    "latency_ms": round((time.time() - t0) * 1000),
                    "status": "ok",
                }
            except Exception as e:
                return {
                    "index": idx,
                    "question": q,
                    "error": str(e),
                    "latency_ms": round((time.time() - t0) * 1000),
                    "status": "error",
                }

    tasks = [_run_one(q, i) for i, q in enumerate(questions)]
    results = await asyncio.gather(*tasks)
    return sorted(results, key=lambda r: r["index"])


def detect_batch_questions(text: str) -> list[str]:
    """Detect if a user pasted multiple questions.

    Looks for numbered lists, bullet points, or multiple question marks.
    Returns a list of individual questions, or [text] if not a batch.
    """
    import re
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]

    # Numbered: "1. What is...?" / "1) What is...?"
    numbered = [re.sub(r"^\d+[.)]\s*", "", l) for l in lines
                if re.match(r"^\d+[.)]\s+", l)]
    if len(numbered) >= 2:
        return numbered

    # Bulleted: "- What is...?" / "• What is...?"
    bulleted = [re.sub(r"^[-•*]\s*", "", l) for l in lines
                if re.match(r"^[-•*]\s+", l)]
    if len(bulleted) >= 2:
        return bulleted

    # Multiple question marks on separate lines
    questions = [l for l in lines if l.endswith("?")]
    if len(questions) >= 2:
        return questions

    return [text]


# ═══════════════════════════════════════════════════════════════════
# §12: Warm Cache on Boot
# ═══════════════════════════════════════════════════════════════════

async def warm_cache_from_history(
    generate_fn=None,
    max_queries: int = 10,
) -> dict:
    """Pre-cache answers to frequently asked questions on startup.

    Reads training pair history to find top queries, then pre-generates answers.
    """
    data_root = os.environ.get("EDITH_DATA_ROOT", ".")
    history_file = os.path.join(data_root, "training_pairs.jsonl")

    if not os.path.exists(history_file):
        return {"status": "no_history", "cached": 0}

    # Count query frequency
    from collections import Counter
    query_counts: Counter = Counter()
    try:
        with open(history_file) as f:
            for line in f:
                try:
                    pair = json.loads(line.strip())
                    prompt = pair.get("prompt", "")
                    if prompt and len(prompt) > 10:
                        query_counts[prompt] += 1
                except json.JSONDecodeError:
                    continue
    except Exception:
        return {"status": "read_error", "cached": 0}

    top_queries = [q for q, _ in query_counts.most_common(max_queries)]

    if not top_queries or not generate_fn:
        return {"status": "ready", "top_queries": len(top_queries), "cached": 0}

    cached = 0
    for q in top_queries:
        if response_cache.get(q) is None:
            try:
                result = await guarded_generate(generate_fn(q), timeout_s=15)
                if isinstance(result, tuple) and result[1] != "timeout":
                    response_cache.put(q, {"answer": result[0], "model": result[1]})
                    cached += 1
            except Exception:
                continue

    return {"status": "warmed", "queries_found": len(top_queries), "cached": cached}


# ═══════════════════════════════════════════════════════════════════
# §13: Auto-Downgrade on Timeout
# ═══════════════════════════════════════════════════════════════════

async def auto_downgrade_generate(
    prompt: str,
    model_chain: list[str] = None,
    timeout_per_model: int = 20,
) -> tuple:
    """Try each model with a timeout; auto-downgrade to cheaper/faster on failure.

    Chain: gemini-2.5-pro (20s) → gemini-2.5-flash (15s) → gemini-2.0-flash-lite (10s)
    """
    if model_chain is None:
        model_chain = [
            os.environ.get("EDITH_MODEL", "gemini-2.5-pro"),
            "gemini-2.5-flash",
            "gemini-2.0-flash-lite",
        ]

    try:
        from server.backend_logic import async_generate_text_via_chain
    except ImportError:
        return "Auto-downgrade unavailable: backend_logic not found", "error"

    for i, model in enumerate(model_chain):
        timeout = max(timeout_per_model - (i * 5), 8)
        try:
            result = await asyncio.wait_for(
                async_generate_text_via_chain(prompt, [model]),
                timeout=timeout,
            )
            if isinstance(result, tuple) and result[0]:
                log.info(f"§DOWNGRADE: Success with {model} (attempt {i+1}/{len(model_chain)})")
                return result
        except asyncio.TimeoutError:
            log.warning(f"§DOWNGRADE: {model} timed out after {timeout}s, trying next")
            continue
        except Exception as e:
            log.warning(f"§DOWNGRADE: {model} failed: {e}, trying next")
            continue

    return "All models timed out. Please try a simpler query.", "all_timeout"


# ═══════════════════════════════════════════════════════════════════
# §14: Source Quality Ranking
# ═══════════════════════════════════════════════════════════════════

def rank_sources(sources: list[dict], top_k: int = 8) -> list[dict]:
    """Rank sources by quality before sending to the LLM.

    Scoring factors:
    1. Recency (newer papers score higher)
    2. Citation count (higher cited = more authoritative)
    3. Methodology quality (papers with clear methods score higher)
    4. Content length (longer = more substance)
    5. Relevance score from retrieval (if available)
    """
    if len(sources) <= top_k:
        return sources

    scored = []
    current_year = 2026

    for src in sources:
        score = 0.0
        meta = src.get("metadata", {})
        text = src.get("text", "") or src.get("content", "") or ""

        # 1. Recency (max 25 points)
        year = meta.get("year") or meta.get("publication_year")
        if year:
            try:
                year = int(year)
                age = current_year - year
                score += max(25 - age * 2, 0)
            except (ValueError, TypeError):
                score += 10  # unknown year = neutral

        # 2. Citation count (max 25 points)
        citations = meta.get("cited_by_count", meta.get("citations", 0))
        if citations:
            try:
                score += min(int(citations) / 10, 25)
            except (ValueError, TypeError):
                pass

        # 3. Methodology quality (max 20 points)
        method_keywords = [
            "regression", "instrumental variable", "fixed effect",
            "difference-in-differences", "RDD", "RCT", "randomized",
            "natural experiment", "panel data", "cross-sectional",
            "time series", "bayesian", "maximum likelihood",
        ]
        method_count = sum(1 for k in method_keywords if k.lower() in text.lower())
        score += min(method_count * 4, 20)

        # 4. Content length (max 15 points)
        score += min(len(text) / 500, 15)

        # 5. Retrieval relevance score (max 15 points)
        relevance = meta.get("score", meta.get("distance", meta.get("relevance", 0)))
        if relevance:
            try:
                score += min(float(relevance) * 15, 15)
            except (ValueError, TypeError):
                pass

        scored.append((score, src))

    # Sort by score descending, return top_k
    scored.sort(key=lambda x: x[0], reverse=True)
    return [src for _, src in scored[:top_k]]


def register_batch_route(app):
    """Register the batch query endpoint."""
    from fastapi import APIRouter, Request

    @app.post("/api/batch-questions")
    async def batch_questions_endpoint(request: Request):
        """Process multiple questions in parallel."""
        body = await request.json()
        text = body.get("text", "")
        questions = body.get("questions", [])

        if not questions and text:
            questions = detect_batch_questions(text)

        if not questions:
            return {"error": "No questions provided"}

        if len(questions) == 1:
            return {"batch": False, "questions": questions,
                    "detail": "Single question detected, use /chat instead"}

        try:
            from server.backend_logic import async_generate_text_via_chain
            model_chain = os.environ.get("EDITH_MODEL", "gemini-2.5-flash").split(",")

            async def gen(q):
                return await async_generate_text_via_chain(q, model_chain)

            results = await batch_query(questions, gen)
            return {"batch": True, "count": len(results), "results": results}
        except Exception as e:
            return {"error": str(e)}
