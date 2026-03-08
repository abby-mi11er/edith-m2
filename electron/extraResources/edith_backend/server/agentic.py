"""
Agentic Utilities — Making E.D.I.T.H.'s Tools Think Like Agents
=================================================================
Provides four primitives that transform one-shot endpoints into
agentic, self-correcting tools:

  1. SessionStore   — persistent context across sequential API calls
  2. safe_llm_call  — never-crash LLM wrapper with structured fallback
  3. sse_progress    — SSE streaming helper for long-running operations
  4. agentic_loop   — verify-retry loop (run → check → fix → repeat)
"""

import asyncio
import json
import logging
import time
import threading
from collections import OrderedDict
from datetime import datetime
from typing import Any, Callable, Optional

log = logging.getLogger("edith.agentic")


# ═══════════════════════════════════════════════════════════════════
# §1: SESSION STORE — Persistent Context Across Calls
# "The Socratic Chamber remembers your last question."
# ═══════════════════════════════════════════════════════════════════

class SessionStore:
    """In-memory session context with TTL expiry.

    Stores conversation history keyed by session_id.
    Auto-evicts sessions older than `ttl_seconds`.
    Thread-safe via a reentrant lock.
    """

    def __init__(self, max_sessions: int = 200, ttl_seconds: int = 14400):
        self._store: OrderedDict[str, dict] = OrderedDict()
        self._lock = threading.RLock()
        self._max = max_sessions
        self._ttl = ttl_seconds

    def get(self, session_id: str) -> list[dict]:
        """Get conversation history for a session."""
        with self._lock:
            self._evict_expired()
            entry = self._store.get(session_id)
            if entry:
                entry["last_access"] = time.time()
                self._store.move_to_end(session_id)
                return list(entry["history"])
            return []

    def append(self, session_id: str, item: dict) -> int:
        """Append an item to session history. Returns new length."""
        with self._lock:
            self._evict_expired()
            if session_id not in self._store:
                if len(self._store) >= self._max:
                    self._store.popitem(last=False)  # evict oldest
                self._store[session_id] = {
                    "history": [],
                    "created": time.time(),
                    "last_access": time.time(),
                }
            entry = self._store[session_id]
            # Cap history at 50 items per session
            if len(entry["history"]) >= 50:
                entry["history"] = entry["history"][-40:]
            entry["history"].append({
                **item,
                "_ts": datetime.now().isoformat(),
            })
            entry["last_access"] = time.time()
            self._store.move_to_end(session_id)
            return len(entry["history"])

    def get_context_string(self, session_id: str, max_items: int = 10) -> str:
        """Get history formatted as a context string for LLM prompts."""
        history = self.get(session_id)
        if not history:
            return ""
        recent = history[-max_items:]
        lines = []
        for i, item in enumerate(recent, 1):
            role = item.get("role", "system")
            content = item.get("content", str(item))
            lines.append(f"[{role} #{i}]: {content}")
        return "\n".join(lines)

    def clear(self, session_id: str):
        """Clear a specific session."""
        with self._lock:
            self._store.pop(session_id, None)

    def stats(self) -> dict:
        """Return store statistics."""
        with self._lock:
            return {
                "sessions": len(self._store),
                "max_sessions": self._max,
                "ttl_seconds": self._ttl,
            }

    def _evict_expired(self):
        now = time.time()
        expired = [
            sid for sid, entry in self._store.items()
            if now - entry["last_access"] > self._ttl
        ]
        for sid in expired:
            del self._store[sid]


# Global session store — used by all endpoints
sessions = SessionStore()


# ═══════════════════════════════════════════════════════════════════
# §2: SAFE LLM CALL — Never Crash, Always Return
# "Better a template answer than a 502."
# ═══════════════════════════════════════════════════════════════════

def safe_llm_call(
    prompt: str,
    model_chain: list[str] = None,
    system_instruction: str = "",
    temperature: float = 0.1,
    fallback_text: str = "",
    fallback_fn: Callable = None,
    **kwargs,
) -> dict:
    """Never-crash LLM wrapper.

    Tries generate_text_via_chain, and on ANY failure returns
    a structured fallback instead of raising.

    Returns:
        {"text": str, "model": str, "_fallback": bool, "_reason": str}
    """
    import os
    model_chain = model_chain or [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]

    try:
        from server.backend_logic import generate_text_via_chain
        text, model = generate_text_via_chain(
            prompt, model_chain,
            system_instruction=system_instruction,
            temperature=temperature,
            **kwargs,
        )
        return {"text": text, "model": model, "_fallback": False}
    except Exception as e:
        reason = str(e)[:200]
        log.warning(f"safe_llm_call fallback: {reason}")

        # Try fallback function if provided
        if fallback_fn:
            try:
                result = fallback_fn()
                return {"text": result, "model": "fallback_fn", "_fallback": True, "_reason": reason}
            except Exception:
                pass

        # Use static fallback text
        return {
            "text": fallback_text or f"[Analysis unavailable: {reason[:80]}]",
            "model": "fallback_static",
            "_fallback": True,
            "_reason": reason,
        }


async def async_safe_llm_call(
    prompt: str,
    model_chain: list[str] = None,
    system_instruction: str = "",
    temperature: float = 0.1,
    fallback_text: str = "",
    **kwargs,
) -> dict:
    """Async version — runs the sync call in a thread pool."""
    return await asyncio.to_thread(
        safe_llm_call, prompt, model_chain,
        system_instruction=system_instruction,
        temperature=temperature,
        fallback_text=fallback_text,
        **kwargs,
    )


# ═══════════════════════════════════════════════════════════════════
# §3: SSE PROGRESS STREAMING — Show Progress, Not Spinners
# "Users see every step, not a 30-second void."
# ═══════════════════════════════════════════════════════════════════

def sse_event(data: dict) -> str:
    """Format a single SSE event."""
    return f"data: {json.dumps(data)}\n\n"


async def sse_progress_stream(
    work_fn: Callable,
    work_args: tuple = (),
    work_kwargs: dict = None,
    step_label: str = "processing",
    timeout: float = 60.0,
):
    """Async generator that streams SSE progress events.

    Runs `work_fn` in a thread, yields progress events,
    then yields the final result.

    Usage:
        return StreamingResponse(
            sse_progress_stream(heavy_fn, (arg1, arg2)),
            media_type="text/event-stream",
        )
    """
    work_kwargs = work_kwargs or {}

    # Opening event
    yield sse_event({
        "type": "progress",
        "step": step_label,
        "progress": 0,
        "message": f"Starting {step_label}...",
    })

    # Run the heavy work in a thread
    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(work_fn, *work_args, **work_kwargs),
            timeout=timeout,
        )
        yield sse_event({
            "type": "progress",
            "step": step_label,
            "progress": 100,
            "message": f"{step_label} complete",
        })
        yield sse_event({
            "type": "result",
            "data": result if isinstance(result, dict) else {"value": result},
        })

    except asyncio.TimeoutError:
        yield sse_event({
            "type": "error",
            "message": f"{step_label} timed out after {timeout}s",
            "timeout": True,
        })

    except Exception as e:
        yield sse_event({
            "type": "error",
            "message": str(e)[:300],
        })

    # Done event
    yield sse_event({"type": "done"})


# ═══════════════════════════════════════════════════════════════════
# §4: AGENTIC LOOP — Verify, Retry, Converge
# "Don't suggest a fix. Prove it works."
# ═══════════════════════════════════════════════════════════════════

def agentic_loop(
    action_fn: Callable,
    validator_fn: Callable,
    max_attempts: int = 3,
    context: dict = None,
) -> dict:
    """Run action → validate → retry loop.

    Args:
        action_fn: callable(context) -> result dict
        validator_fn: callable(result) -> {"valid": bool, "error": str}
        max_attempts: max retries
        context: mutable dict passed to action_fn (updated between attempts)

    Returns:
        {
            "result": final result,
            "attempts": number of attempts,
            "verified": whether validation passed,
            "history": list of attempt results,
        }
    """
    context = context or {}
    history = []

    for attempt in range(1, max_attempts + 1):
        context["_attempt"] = attempt
        context["_history"] = history

        try:
            result = action_fn(context)
        except Exception as e:
            result = {"error": str(e)}

        validation = {"valid": False, "error": "validator not run"}
        try:
            validation = validator_fn(result)
        except Exception as e:
            validation = {"valid": False, "error": f"Validator error: {e}"}

        attempt_record = {
            "attempt": attempt,
            "result_summary": str(result)[:200] if not isinstance(result, dict) else {
                k: str(v)[:100] for k, v in result.items()
            },
            "validation": validation,
        }
        history.append(attempt_record)

        if validation.get("valid"):
            return {
                "result": result,
                "attempts": attempt,
                "verified": True,
                "history": history,
            }

        # Feed error back into context for next attempt
        context["_last_error"] = validation.get("error", "Unknown")
        context["_last_result"] = result
        log.info(f"Agentic loop attempt {attempt}/{max_attempts} failed: {validation.get('error', '')[:100]}")

    # All attempts exhausted
    return {
        "result": result,
        "attempts": max_attempts,
        "verified": False,
        "history": history,
    }


async def async_agentic_loop(
    action_fn: Callable,
    validator_fn: Callable,
    max_attempts: int = 3,
    context: dict = None,
) -> dict:
    """Async version of the agentic loop."""
    return await asyncio.to_thread(
        agentic_loop, action_fn, validator_fn, max_attempts, context,
    )
