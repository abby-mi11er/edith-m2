"""
Server Infrastructure — B1/B3/B6/B9/B10
=========================================
Core infrastructure improvements for the E.D.I.T.H. backend:
  - Response cache (LRU) for repeated queries
  - Correlation IDs for request tracing
  - Token-bucket rate limiter per endpoint
  - Startup health gate (503 until warmup done)
  - Metrics collection and endpoint
"""

import hashlib
import json
import logging
import time
import threading
import uuid
from collections import OrderedDict
from typing import Any, Optional
from functools import wraps

from fastapi import Request, HTTPException

log = logging.getLogger("edith.infra")


# ═══════════════════════════════════════════════════════════════════
# B3: Correlation IDs — tag every request for end-to-end tracing
# ═══════════════════════════════════════════════════════════════════

_correlation_id_var: Optional[str] = None


def generate_correlation_id() -> str:
    """Generate a short correlation ID for request tracing."""
    return uuid.uuid4().hex[:12]


def get_correlation_id() -> str:
    """Get the current correlation ID (or generate one)."""
    global _correlation_id_var
    return _correlation_id_var or generate_correlation_id()


def set_correlation_id(cid: str):
    """Set the correlation ID for the current request context."""
    global _correlation_id_var
    _correlation_id_var = cid


class CorrelationMiddleware:
    """Add X-Correlation-ID to every request/response."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] == "http":
            cid = generate_correlation_id()
            set_correlation_id(cid)
            # Inject into scope for downstream access
            scope["state"] = scope.get("state", {})
            scope["state"]["correlation_id"] = cid

            async def send_with_header(message):
                if message["type"] == "http.response.start":
                    headers = list(message.get("headers", []))
                    headers.append([b"x-correlation-id", cid.encode()])
                    message["headers"] = headers
                await send(message)

            await self.app(scope, receive, send_with_header)
        else:
            await self.app(scope, receive, send)


# ═══════════════════════════════════════════════════════════════════
# B1: Response Cache — LRU cache for repeated queries
# ═══════════════════════════════════════════════════════════════════

class ResponseCache:
    """Thread-safe LRU cache for query responses.

    Key = hash(query + depth + mode + answer_length)
    Value = (response_text, sources, timestamp)
    TTL = 5 minutes (avoid stale results)
    """

    def __init__(self, max_size: int = 200, ttl_seconds: int = 300):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str, depth: str = "", mode: str = "",
                  answer_length: str = "") -> str:
        raw = f"{query.strip().lower()}|{depth}|{mode}|{answer_length}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, query: str, **kwargs) -> Optional[dict]:
        key = self._make_key(query, **kwargs)
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if time.time() - entry["ts"] < self._ttl:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    log.debug(f"§CACHE: HIT for {key[:8]} (hits={self._hits})")
                    return entry["data"]
                else:
                    del self._cache[key]
            self._misses += 1
            return None

    def put(self, query: str, data: dict, **kwargs):
        key = self._make_key(query, **kwargs)
        with self._lock:
            self._cache[key] = {"data": data, "ts": time.time()}
            self._cache.move_to_end(key)
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def invalidate_all(self):
        with self._lock:
            self._cache.clear()
            log.info("§CACHE: Invalidated all entries")

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{(self._hits / total * 100):.1f}%" if total > 0 else "0%",
        }


# Singleton
response_cache = ResponseCache()


# ═══════════════════════════════════════════════════════════════════
# B6: Rate Limiter — token bucket per endpoint
# ═══════════════════════════════════════════════════════════════════

class TokenBucket:
    """Token bucket rate limiter."""

    def __init__(self, rate: float, burst: int):
        self.rate = rate      # tokens per second
        self.burst = burst    # max burst size
        self.tokens = burst
        self.last_time = time.time()
        self._lock = threading.Lock()

    def allow(self) -> bool:
        with self._lock:
            now = time.time()
            elapsed = now - self.last_time
            self.last_time = now
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            if self.tokens >= 1:
                self.tokens -= 1
                return True
            return False


class RateLimiter:
    """Per-endpoint rate limiting with localhost exemption."""

    # Hosts that bypass rate limiting (audit scripts, dev tools)
    EXEMPT_HOSTS = {"127.0.0.1", "localhost", "::1"}

    def __init__(self):
        self._buckets: dict[str, TokenBucket] = {}
        self._denied = 0
        self._exempt_hits = 0

    def get_bucket(self, endpoint: str) -> TokenBucket:
        if endpoint not in self._buckets:
            if "chat" in endpoint:
                self._buckets[endpoint] = TokenBucket(rate=2, burst=5)
            elif "index" in endpoint:
                self._buckets[endpoint] = TokenBucket(rate=1, burst=3)
            else:
                # Default: 10 req/s, burst of 200 (supports rapid panel switching)
                self._buckets[endpoint] = TokenBucket(rate=10, burst=200)
        return self._buckets[endpoint]

    def check(self, endpoint: str, client_host: str = "") -> bool:
        # Exempt localhost (audit scripts, local dev)
        if client_host in self.EXEMPT_HOSTS:
            self._exempt_hits += 1
            return True
        bucket = self.get_bucket(endpoint)
        if bucket.allow():
            return True
        self._denied += 1
        log.warning(f"§RATE: Request denied for {endpoint} (total_denied={self._denied})")
        return False

    @property
    def stats(self) -> dict:
        return {
            "endpoints": len(self._buckets),
            "denied_total": self._denied,
            "exempt_hits": self._exempt_hits,
        }


# Singleton
rate_limiter = RateLimiter()


# ═══════════════════════════════════════════════════════════════════
# B9: Startup Health Gate — 503 until warmup complete
# ═══════════════════════════════════════════════════════════════════

class StartupGate:
    """Block requests until server warmup is complete."""

    def __init__(self):
        self._ready = False
        self._ready_time: Optional[float] = None
        self._start_time = time.time()
        self._components: dict[str, bool] = {}

    def mark_component_ready(self, name: str):
        self._components[name] = True
        log.info(f"§GATE: {name} ready ({len([v for v in self._components.values() if v])}/{len(self._components)} components)")

    def register_component(self, name: str):
        self._components[name] = False

    def mark_ready(self):
        self._ready = True
        self._ready_time = time.time()
        elapsed = self._ready_time - self._start_time
        log.info(f"§GATE: Server fully ready in {elapsed:.1f}s")

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def status(self) -> dict:
        return {
            "ready": self._ready,
            "uptime_seconds": time.time() - self._start_time,
            "components": dict(self._components),
            "warmup_time": f"{self._ready_time - self._start_time:.1f}s" if self._ready_time else "pending",
        }


# Singleton
startup_gate = StartupGate()


# ═══════════════════════════════════════════════════════════════════
# B10: Metrics Collection
# ═══════════════════════════════════════════════════════════════════

class MetricsCollector:
    """Collect latency and usage metrics."""

    def __init__(self):
        self._lock = threading.Lock()
        self._request_count = 0
        self._latencies: dict[str, list[float]] = {}  # endpoint → recent latencies
        self._model_usage: dict[str, int] = {}  # model_name → count
        self._errors = 0
        self._start_time = time.time()

    def record_request(self, endpoint: str, latency: float, model: str = ""):
        with self._lock:
            self._request_count += 1
            if endpoint not in self._latencies:
                self._latencies[endpoint] = []
            self._latencies[endpoint].append(latency)
            # Keep only last 100 latencies per endpoint
            if len(self._latencies[endpoint]) > 100:
                self._latencies[endpoint] = self._latencies[endpoint][-100:]
            if model:
                self._model_usage[model] = self._model_usage.get(model, 0) + 1

    def record_error(self):
        with self._lock:
            self._errors += 1

    def get_metrics(self) -> dict:
        with self._lock:
            result = {
                "uptime_seconds": round(time.time() - self._start_time),
                "total_requests": self._request_count,
                "total_errors": self._errors,
                "error_rate": f"{(self._errors / max(self._request_count, 1) * 100):.1f}%",
                "model_usage": dict(self._model_usage),
                "cache": response_cache.stats,
                "rate_limiter": rate_limiter.stats,
                "startup": startup_gate.status,
                "latencies": {},
            }
            for endpoint, lats in self._latencies.items():
                if lats:
                    sorted_lats = sorted(lats)
                    n = len(sorted_lats)
                    result["latencies"][endpoint] = {
                        "count": n,
                        "p50": f"{sorted_lats[n // 2]:.2f}s",
                        "p95": f"{sorted_lats[int(n * 0.95)]:.2f}s",
                        "p99": f"{sorted_lats[min(int(n * 0.99), n - 1)]:.2f}s",
                    }
            return result


# Singleton
metrics = MetricsCollector()
