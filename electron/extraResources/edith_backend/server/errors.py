"""
Structured Error Responses — A8 + Citadel Enhancements
=======================================================
Standardized error envelope for all API endpoints.
Replaces ad-hoc HTTPException details with typed, consistent responses.

§CE-1: User-friendly error messages with suggested actions
§CE-2: Error telemetry — track frequency by route and type
§CE-3: Graceful degradation — return partial results when possible
"""
from __future__ import annotations

from fastapi import HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from collections import defaultdict
import traceback
import logging
import time
import threading

log = logging.getLogger("edith.errors")


class ErrorResponse(BaseModel):
    """Standard error response body."""
    error: str  # Machine-readable error code
    detail: str  # Human-readable message
    status: int  # HTTP status code
    suggestion: Optional[str] = None  # §CE-1: user-friendly suggested fix
    request_id: Optional[str] = None  # Trace ID if available


# ═══════════════════════════════════════════════════════════════════
# §CE-1: User-friendly error messages
# ═══════════════════════════════════════════════════════════════════

_FRIENDLY_MESSAGES: dict[str, dict[str, str]] = {
    "chroma": {
        "detail": "Your library is still being indexed",
        "suggestion": "Wait 30 seconds and try again. If this persists, run Maintenance from the War Room.",
    },
    "timeout": {
        "detail": "The request took too long to complete",
        "suggestion": "Try a simpler query, or check your internet connection.",
    },
    "rate": {
        "detail": "Too many requests in a short time",
        "suggestion": "Wait a moment and try again. Winnie needs a breath between queries.",
    },
    "api_key": {
        "detail": "API key is missing or invalid",
        "suggestion": "Check your API keys in Settings. You need a valid OpenAI or Gemini key.",
    },
    "model": {
        "detail": "The AI model is temporarily unavailable",
        "suggestion": "Try again in a moment. If using Gemini, check quota at console.cloud.google.com.",
    },
    "disk": {
        "detail": "Storage issue detected",
        "suggestion": "Check that your Oyen Bolt is connected and has free space.",
    },
    "memory": {
        "detail": "System is running low on memory",
        "suggestion": "Close other applications or restart the Citadel.",
    },
    "permission": {
        "detail": "You don't have permission for this action",
        "suggestion": "Check your access level in Settings, or contact the system administrator.",
    },
    "network": {
        "detail": "Network connection issue",
        "suggestion": "Check your internet connection. Winnie can work offline with local models.",
    },
}


def _match_friendly(detail: str) -> dict[str, str] | None:
    """Match an error detail to a friendly message and suggestion."""
    detail_lower = detail.lower()
    for key, friendly in _FRIENDLY_MESSAGES.items():
        if key in detail_lower:
            return friendly
    return None


# ═══════════════════════════════════════════════════════════════════
# §CE-2: Error Telemetry
# ═══════════════════════════════════════════════════════════════════

class ErrorTelemetry:
    """Track error frequency by route and type for the Doctor panel."""

    def __init__(self):
        self._lock = threading.Lock()
        self._errors: list[dict] = []
        self._counts: dict[str, int] = defaultdict(int)

    def record(self, status: int, error: str, route: str = ""):
        """Record an error event."""
        with self._lock:
            self._errors.append({
                "status": status,
                "error": error,
                "route": route,
                "timestamp": time.time(),
            })
            # Keep only last 500 errors
            if len(self._errors) > 500:
                self._errors = self._errors[-500:]
            self._counts[error] += 1

    def get_top_errors(self, limit: int = 10) -> list[dict]:
        """Return the most frequent error types."""
        with self._lock:
            sorted_errors = sorted(
                self._counts.items(), key=lambda x: x[1], reverse=True
            )
            return [{"error": k, "count": v} for k, v in sorted_errors[:limit]]

    def get_recent(self, limit: int = 20) -> list[dict]:
        """Return recent errors."""
        with self._lock:
            return list(reversed(self._errors[-limit:]))

    def get_stats(self) -> dict:
        """Return error stats for the Doctor panel."""
        with self._lock:
            now = time.time()
            last_hour = [e for e in self._errors if now - e["timestamp"] < 3600]
            last_day = [e for e in self._errors if now - e["timestamp"] < 86400]
            return {
                "total": sum(self._counts.values()),
                "last_hour": len(last_hour),
                "last_24h": len(last_day),
                "top_errors": self.get_top_errors(5),
            }


# Global instance
error_telemetry = ErrorTelemetry()


# ═══════════════════════════════════════════════════════════════════
# §CE-3: Graceful Degradation
# ═══════════════════════════════════════════════════════════════════

def graceful_response(
    partial_data: dict | None = None,
    fallback_message: str = "",
    error_detail: str = "",
) -> dict:
    """Return partial results when a full response isn't possible.

    Instead of failing completely, return what we have with a warning.
    """
    result: dict = {"_warning": error_detail or "Partial results returned"}
    if partial_data:
        result.update(partial_data)
    if fallback_message:
        result["_fallback"] = fallback_message
    return result


# ═══════════════════════════════════════════════════════════════════
# Common error codes — enhanced with §CE-1 friendly messages
# ═══════════════════════════════════════════════════════════════════

def not_found(resource: str, detail: str = "") -> HTTPException:
    """Resource not found (404)."""
    msg = detail or f"{resource} not found"
    error_telemetry.record(404, "not_found")
    return HTTPException(status_code=404, detail=msg)


def bad_request(detail: str) -> HTTPException:
    """Invalid input (400)."""
    error_telemetry.record(400, "bad_request")
    return HTTPException(status_code=400, detail=detail)


def unauthorized(detail: str = "Authentication required") -> HTTPException:
    """Auth required (401)."""
    error_telemetry.record(401, "unauthorized")
    return HTTPException(status_code=401, detail=detail)


def forbidden(detail: str = "Access denied") -> HTTPException:
    """Permission denied (403)."""
    error_telemetry.record(403, "forbidden")
    return HTTPException(status_code=403, detail=detail)


def rate_limited(detail: str = "Rate limit exceeded") -> HTTPException:
    """Too many requests (429)."""
    error_telemetry.record(429, "rate_limited")
    return HTTPException(status_code=429, detail=detail)


def service_unavailable(detail: str = "Service temporarily unavailable") -> HTTPException:
    """Backend down (503)."""
    error_telemetry.record(503, "service_unavailable")
    return HTTPException(status_code=503, detail=detail)


def internal_error(detail: str = "An internal error occurred", log_exc: bool = True) -> HTTPException:
    """Server error (500). Optionally logs the traceback."""
    if log_exc:
        log.error(f"Internal error: {detail}\n{traceback.format_exc()}")
    error_telemetry.record(500, "internal_error")
    return HTTPException(status_code=500, detail=detail)


def error_response(status: int, error: str, detail: str, request_id: str = "") -> JSONResponse:
    """Build a structured JSON error response with friendly suggestions."""
    # §CE-1: Try to match a friendly message
    friendly = _match_friendly(detail)
    suggestion = friendly["suggestion"] if friendly else None

    body = ErrorResponse(
        error=error,
        detail=friendly["detail"] if friendly else detail,
        status=status,
        suggestion=suggestion,
        request_id=request_id or None,
    )
    error_telemetry.record(status, error)
    return JSONResponse(status_code=status, content=body.model_dump(exclude_none=True))
