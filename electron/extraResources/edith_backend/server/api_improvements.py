#!/usr/bin/env python3
"""
Server / API Layer Improvements Module
========================================
Enhancements for server/main.py:
  3.1  Route module structure (this file provides the pattern)
  3.2  Pydantic request/response schemas
  3.3  Structured error responses
  3.4  OpenAPI documentation improvements
  3.5  Background task queue
  3.6  WebSocket for chat streaming
  3.7  Health check with dependency probes
  3.8  API versioning (/v1/)
  3.9  Request tracing (X-Request-ID)
  3.10 Connection pooling for LLM clients
"""

from __future__ import annotations

import asyncio
import time
import uuid
import os
from typing import Optional, Any
from pathlib import Path

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 3.2  Pydantic Request / Response Schemas
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    """Schema for chat endpoint requests."""
    message: str = Field(..., min_length=1, max_length=50000, description="User message")
    model: str = Field(default="", description="Model override")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")
    max_tokens: int = Field(default=4096, ge=1, le=128000, description="Max response tokens")
    include_sources: bool = Field(default=True, description="Include retrieval sources")
    session_id: Optional[str] = Field(default=None, description="Session ID for context")
    top_k: int = Field(default=8, ge=1, le=50, description="Number of retrieval results")

class ChatResponse(BaseModel):
    """Schema for chat endpoint responses."""
    response: str = Field(..., description="Model response")
    model: str = Field(default="", description="Model used")
    sources: list[dict] = Field(default_factory=list, description="Retrieved sources")
    tokens_used: int = Field(default=0, description="Tokens consumed")
    latency_ms: float = Field(default=0.0, description="Response latency")
    request_id: str = Field(default="", description="Request trace ID")
    confidence: Optional[str] = Field(default=None, description="Response confidence")

class LibrarySearchRequest(BaseModel):
    """Schema for library search requests."""
    query: str = Field(..., min_length=1, max_length=5000)
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict = Field(default_factory=dict, description="Metadata filters")
    doc_type: Optional[str] = Field(default=None)
    project: Optional[str] = Field(default=None)
    year_range: Optional[list[int]] = Field(default=None, description="[start_year, end_year]")

class LibrarySearchResponse(BaseModel):
    """Schema for library search responses."""
    results: list[dict] = Field(default_factory=list)
    total: int = Field(default=0)
    query: str = Field(default="")
    latency_ms: float = Field(default=0.0)

class IndexRequest(BaseModel):
    """Schema for indexing requests."""
    path: Optional[str] = Field(default=None, description="Specific path to index")
    force_reindex: bool = Field(default=False, description="Force re-index all files")
    ocr_enabled: bool = Field(default=False, description="Enable OCR for scanned PDFs")

class IndexStatusResponse(BaseModel):
    """Schema for indexing status responses."""
    status: str = Field(default="idle")
    files_processed: int = Field(default=0)
    files_total: int = Field(default=0)
    chunks_added: int = Field(default=0)
    errors: int = Field(default=0)
    eta_minutes: Optional[float] = Field(default=None)

class HealthResponse(BaseModel):
    """Health check response with dependency probes."""
    status: str = Field(default="ok")
    version: str = Field(default="1.0.0")
    uptime_seconds: float = Field(default=0.0)
    dependencies: dict = Field(default_factory=dict)
    request_id: str = Field(default="")


# ---------------------------------------------------------------------------
# 3.3  Structured Error Responses
# ---------------------------------------------------------------------------

class ErrorResponse(BaseModel):
    """Standardized error response format."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    detail: Optional[str] = Field(default=None, description="Technical details")
    request_id: str = Field(default="", description="Request trace ID")
    status_code: int = Field(default=500)

    @classmethod
    def from_exception(cls, exc: Exception, request_id: str = "", status_code: int = 500):
        return cls(
            error=type(exc).__name__,
            message=str(exc),
            detail=None,
            request_id=request_id,
            status_code=status_code,
        )

    @classmethod
    def not_found(cls, resource: str, request_id: str = ""):
        return cls(
            error="NotFound",
            message=f"Resource not found: {resource}",
            request_id=request_id,
            status_code=404,
        )

    @classmethod
    def validation_error(cls, message: str, request_id: str = ""):
        return cls(
            error="ValidationError",
            message=message,
            request_id=request_id,
            status_code=422,
        )

    @classmethod
    def rate_limited(cls, request_id: str = ""):
        return cls(
            error="RateLimited",
            message="Too many requests. Please wait and try again.",
            request_id=request_id,
            status_code=429,
        )


# ---------------------------------------------------------------------------
# 3.5  Background Task Queue
# ---------------------------------------------------------------------------

class BackgroundTaskQueue:
    """Simple async task queue for long-running operations."""

    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self._tasks: dict[str, dict] = {}
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def submit(self, task_id: str, coroutine, description: str = ""):
        """Submit a background task."""
        self._tasks[task_id] = {
            "status": "queued",
            "description": description,
            "submitted_at": time.time(),
            "completed_at": None,
            "result": None,
            "error": None,
        }

        async def _run():
            async with self._semaphore:
                self._tasks[task_id]["status"] = "running"
                try:
                    result = await coroutine
                    self._tasks[task_id]["status"] = "completed"
                    self._tasks[task_id]["result"] = result
                except Exception as e:
                    self._tasks[task_id]["status"] = "failed"
                    self._tasks[task_id]["error"] = str(e)
                finally:
                    self._tasks[task_id]["completed_at"] = time.time()

        asyncio.create_task(_run())
        return task_id

    def get_status(self, task_id: str) -> Optional[dict]:
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[dict]:
        return [
            {"task_id": tid, **info}
            for tid, info in sorted(
                self._tasks.items(),
                key=lambda x: x[1]["submitted_at"],
                reverse=True,
            )[:50]
        ]

    def cleanup(self, max_age_seconds: int = 3600):
        """Remove completed tasks older than max_age."""
        now = time.time()
        to_remove = [
            tid for tid, info in self._tasks.items()
            if info["status"] in ("completed", "failed")
            and info.get("completed_at")
            and now - info["completed_at"] > max_age_seconds
        ]
        for tid in to_remove:
            del self._tasks[tid]


# ---------------------------------------------------------------------------
# 3.7  Health Check with Dependency Probes
# ---------------------------------------------------------------------------

_SERVER_START_TIME = time.time()

async def health_check() -> HealthResponse:
    """Comprehensive health check with dependency probes."""
    dependencies = {}

    # ChromaDB probe
    try:
        import chromadb  # type: ignore
        chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "")
        if chroma_dir and Path(chroma_dir).exists():
            client = chromadb.PersistentClient(path=chroma_dir)
            dependencies["chromadb"] = {
                "status": "ok",
                "collections": len(client.list_collections()),
            }
        else:
            dependencies["chromadb"] = {"status": "not_configured"}
    except Exception as e:
        dependencies["chromadb"] = {"status": "error", "error": str(e)}

    # Gemini API probe
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    dependencies["gemini_api"] = {"status": "configured" if api_key else "not_configured"}

    # OpenAI API probe
    openai_key = os.environ.get("OPENAI_API_KEY")
    dependencies["openai_api"] = {"status": "configured" if openai_key else "not_configured"}

    # Disk space
    try:
        import shutil
        disk = shutil.disk_usage("/")
        free_gb = disk.free / (1024**3)
        dependencies["disk"] = {
            "status": "ok" if free_gb > 1 else "warning",
            "free_gb": round(free_gb, 1),
        }
    except Exception:
        dependencies["disk"] = {"status": "unknown"}

    status = "ok"
    if any(d.get("status") == "error" for d in dependencies.values()):
        status = "degraded"

    return HealthResponse(
        status=status,
        version="2.0.0",
        uptime_seconds=round(time.time() - _SERVER_START_TIME, 1),
        dependencies=dependencies,
    )


# ---------------------------------------------------------------------------
# 3.9  Request Tracing Middleware
# ---------------------------------------------------------------------------

class RequestTracer:
    """Adds X-Request-ID header to every request for distributed tracing."""

    @staticmethod
    def generate_id() -> str:
        return str(uuid.uuid4())[:12]

    @staticmethod
    async def middleware(request, call_next):
        """FastAPI middleware to inject X-Request-ID."""
        request_id = request.headers.get("X-Request-ID") or RequestTracer.generate_id()
        request.state.request_id = request_id
        start = time.time()
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time-Ms"] = str(round((time.time() - start) * 1000, 1))
        return response


# ---------------------------------------------------------------------------
# 3.10  Connection Pooling for LLM Clients
# ---------------------------------------------------------------------------

class LLMClientPool:
    """
    Pool of LLM client connections for reuse.
    Avoids creating new client instances per request.
    """

    def __init__(self):
        self._clients: dict[str, Any] = {}

    def get_openai_client(self, api_key: str = ""):
        """Get or create OpenAI client."""
        key = f"openai:{api_key[:8]}" if api_key else "openai:env"
        if key not in self._clients:
            try:
                from openai import OpenAI  # type: ignore
                self._clients[key] = OpenAI(api_key=api_key or None)
            except Exception:
                return None
        return self._clients[key]

    def get_gemini_client(self, model: str = "gemini-2.5-flash"):
        """Get or create Gemini client."""
        key = f"gemini:{model}"
        if key not in self._clients:
            try:
                import google.generativeai as genai  # type: ignore
                api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
                if api_key:
                    genai.configure(api_key=api_key)
                self._clients[key] = genai.GenerativeModel(model)
            except Exception:
                return None
        return self._clients[key]

    def clear(self):
        self._clients.clear()


# Singleton instance
_LLM_POOL = LLMClientPool()

