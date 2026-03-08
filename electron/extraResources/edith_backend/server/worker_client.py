"""
Worker Client — Async Interface to the Compute Worker
======================================================
Thin async client that API route handlers use to offload heavy compute
to the worker process (port 8002). Falls back gracefully to in-process
execution if the worker is unavailable.

Usage:
    from server.worker_client import worker

    # Async (preferred — non-blocking)
    result = await worker.generate_local(prompt, max_tokens=512)

    # Status check
    if worker.is_available:
        info = await worker.status()
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger("edith.worker_client")

WORKER_PORT = int(os.environ.get("EDITH_WORKER_PORT", "8002"))
WORKER_HOST = os.environ.get("EDITH_WORKER_HOST", "127.0.0.1")
WORKER_URL = f"http://{WORKER_HOST}:{WORKER_PORT}"
ROOT_DIR = Path(__file__).parent.parent

# Connection timeout for worker requests
_TIMEOUT = float(os.environ.get("EDITH_WORKER_TIMEOUT", "30"))


class WorkerClient:
    """Async client for the compute worker process."""

    def __init__(self):
        self._available: Optional[bool] = None
        self._worker_proc: Optional[subprocess.Popen] = None
        self._last_check = 0.0
        self._check_interval = 10.0  # seconds between availability checks

    @property
    def is_available(self) -> bool:
        """Check if worker is reachable (cached for 10s)."""
        now = time.time()
        if self._available is not None and (now - self._last_check) < self._check_interval:
            return self._available
        # Non-blocking check
        try:
            import urllib.request
            req = urllib.request.Request(f"{WORKER_URL}/status", method="GET")
            resp = urllib.request.urlopen(req, timeout=2)
            self._available = resp.status == 200
        except Exception:
            self._available = False
        self._last_check = now
        return self._available

    async def _post(self, path: str, body: dict, timeout: float = _TIMEOUT) -> dict:
        """Send async HTTP POST to worker."""
        try:
            import urllib.request
            url = f"{WORKER_URL}{path}"
            data = json.dumps(body).encode("utf-8")
            req = urllib.request.Request(
                url, data=data, method="POST",
                headers={"Content-Type": "application/json"},
            )
            # Run in thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(req, timeout=timeout),
            )
            result = json.loads(resp.read().decode("utf-8"))
            return result
        except Exception as e:
            log.warning(f"§WORKER-CLIENT: Request to {path} failed: {e}")
            return {"status": "error", "error": str(e)}

    async def _get(self, path: str, timeout: float = 5) -> dict:
        """Send async HTTP GET to worker."""
        try:
            import urllib.request
            url = f"{WORKER_URL}{path}"
            req = urllib.request.Request(url, method="GET")
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: urllib.request.urlopen(req, timeout=timeout),
            )
            return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ── Public API ─────────────────────────────────────────────────

    async def status(self) -> dict:
        """Get worker status."""
        return await self._get("/status")

    async def warmup(self) -> dict:
        """Trigger full worker warmup."""
        return await self._post("/warmup", {}, timeout=120)

    async def generate_local(self, prompt: str, system: str = "",
                             max_tokens: int = 512,
                             temperature: float = 0.1) -> dict:
        """Generate text via MLX (worker) or fall back to in-process."""
        if self.is_available:
            return await self._post("/generate", {
                "prompt": prompt,
                "system": system,
                "max_tokens": max_tokens,
                "temperature": temperature,
            })
        # Fallback: in-process (blocks event loop but works)
        try:
            from server.mlx_inference import generate
            result = generate(prompt, system_instruction=system,
                            max_tokens=max_tokens, temperature=temperature)
            return {"status": "ok" if result else "unavailable",
                    "text": result or "", "fallback": True}
        except Exception as e:
            return {"status": "error", "error": str(e), "fallback": True}

    async def embed_texts(self, texts: List[str]) -> dict:
        """Embed texts via worker or fall back to in-process."""
        if self.is_available:
            return await self._post("/embed", {"texts": texts})
        # Fallback
        try:
            from server.mlx_embeddings import embed
            vectors = embed(texts)
            return {"status": "ok", "count": len(vectors), "fallback": True}
        except Exception as e:
            return {"status": "error", "error": str(e), "fallback": True}

    async def run_pipeline(self, pipeline_name: str) -> dict:
        """Run a pipeline in the worker."""
        if self.is_available:
            return await self._post("/pipeline", {"pipeline": pipeline_name})
        return {"status": "error", "error": "Worker not available"}

    async def dispatch(self, action: str, params: dict = None) -> dict:
        """Generic dispatch to worker."""
        return await self._post("/dispatch", {
            "action": action,
            "params": params or {},
        })

    # ── Worker Process Management ──────────────────────────────────

    def ensure_worker_running(self):
        """Launch the worker process if not already running."""
        if self.is_available:
            log.info("§WORKER-CLIENT: Worker already running")
            return True

        if self._worker_proc and self._worker_proc.poll() is None:
            log.info("§WORKER-CLIENT: Worker process exists but not responding")
            return False

        try:
            python = sys.executable
            worker_script = str(ROOT_DIR / "server" / "worker.py")
            env = {**os.environ, "EDITH_WORKER_PORT": str(WORKER_PORT)}

            self._worker_proc = subprocess.Popen(
                [python, "-m", "server.worker"],
                cwd=str(ROOT_DIR),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            log.info(f"§WORKER-CLIENT: Launched worker process (pid={self._worker_proc.pid})")

            # Wait briefly for worker to start
            for _ in range(20):
                time.sleep(0.5)
                if self.is_available:
                    log.info("§WORKER-CLIENT: Worker is ready")
                    return True

            log.warning("§WORKER-CLIENT: Worker started but not responding within 10s")
            return False
        except Exception as e:
            log.error(f"§WORKER-CLIENT: Failed to launch worker: {e}")
            return False

    def stop_worker(self):
        """Stop the worker process."""
        if self._worker_proc and self._worker_proc.poll() is None:
            self._worker_proc.terminate()
            try:
                self._worker_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._worker_proc.kill()
            log.info("§WORKER-CLIENT: Worker stopped")
        self._available = False


# ── Singleton ──────────────────────────────────────────────────────
worker = WorkerClient()
