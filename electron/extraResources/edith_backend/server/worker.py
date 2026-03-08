"""
Compute Worker — Background Process for Heavy Operations
==========================================================
Runs as a separate process (port 8002) to offload heavy compute from
the main API server. Handles:
  - MLX model loading and inference
  - Embedding generation
  - ChromaDB integrity checks
  - Library scanning
  - Index rebuilding
  - Pipeline execution (build_graph, ingest_papers, etc.)

Architecture:
  API Server (port 8001)  ──async HTTP──▶  Worker (port 8002)
  - Routes, middleware                     - MLX, embeddings
  - Auth, CORS                             - Indexing, pipelines
  - ChromaDB queries (fast)                - Heavy compute (slow)

Launch:
  python -m server.worker       # standalone
  # or via the main server's startup (auto-launched as subprocess)
"""

import asyncio
import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

log = logging.getLogger("edith.worker")

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

WORKER_PORT = int(os.environ.get("EDITH_WORKER_PORT", "8002"))
WORKER_HOST = os.environ.get("EDITH_WORKER_HOST", "127.0.0.1")


class ComputeWorker:
    """Background compute worker with a simple JSON API."""

    def __init__(self):
        self._mlx_loaded = False
        self._embeddings_loaded = False
        self._warmup_complete = False
        self._start_time = time.time()
        self._task_count = 0
        self._active_tasks: dict = {}

    # ── MLX Inference ──────────────────────────────────────────────

    def load_mlx(self) -> dict:
        """Pre-load the MLX local inference model."""
        try:
            from server.mlx_inference import load_model, is_available, get_model_info
            if not is_available():
                return {"status": "unavailable", "reason": "mlx-lm not installed"}
            success = load_model()
            self._mlx_loaded = success
            info = get_model_info()
            return {"status": "loaded" if success else "failed", **info}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def generate_local(self, prompt: str, system: str = "",
                       max_tokens: int = 512, temperature: float = 0.1) -> dict:
        """Generate text via local MLX model."""
        try:
            from server.mlx_inference import generate
            result = generate(prompt, system_instruction=system,
                            max_tokens=max_tokens, temperature=temperature)
            if result is None:
                return {"status": "unavailable", "text": ""}
            return {"status": "ok", "text": result}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ── Embeddings ─────────────────────────────────────────────────

    def load_embeddings(self) -> dict:
        """Pre-load the embedding model."""
        try:
            from server.mlx_embeddings import embed, is_available, get_backend_info
            if not is_available():
                return {"status": "unavailable"}
            embed(["warmup"])  # Trigger model load
            self._embeddings_loaded = True
            return {"status": "loaded", **get_backend_info()}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def embed_texts(self, texts: list) -> dict:
        """Embed a list of texts."""
        try:
            from server.mlx_embeddings import embed
            vectors = embed(texts)
            return {"status": "ok", "count": len(vectors),
                    "dimensions": len(vectors[0]) if vectors else 0}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ── ChromaDB Health ────────────────────────────────────────────

    def chroma_health_check(self) -> dict:
        """Run ChromaDB integrity check."""
        try:
            chroma_dir = os.environ.get("EDITH_CHROMA_DIR", str(ROOT_DIR / "chroma"))
            if not os.path.isdir(chroma_dir):
                return {"status": "no_directory", "path": chroma_dir}

            import chromadb
            client = chromadb.PersistentClient(path=chroma_dir)
            collections = []
            for coll in client.list_collections():
                try:
                    c = client.get_collection(coll.name)
                    count = c.count()
                    collections.append({"name": coll.name, "count": count, "ok": True})
                except Exception as e:
                    collections.append({"name": coll.name, "ok": False, "error": str(e)})

            return {"status": "ok", "collections": collections,
                    "total_chunks": sum(c["count"] for c in collections if c.get("ok"))}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ── Library Scan ───────────────────────────────────────────────

    def scan_library(self) -> dict:
        """Scan library sources (papers, documents)."""
        try:
            from server.routes.library import _scan_library_sources
            _scan_library_sources(papers_only=True)
            return {"status": "ok"}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    # ── Pipelines ──────────────────────────────────────────────────

    def run_pipeline(self, pipeline_name: str) -> dict:
        """Run a named pipeline in the worker process."""
        pipelines = {
            "build_graph": "pipelines.build_graph",
            "extract_entities": "pipelines.extract_entities",
            "ingest_papers": "pipelines.ingest_papers",
            "run_eval": "pipelines.run_eval",
        }
        if pipeline_name not in pipelines:
            return {"status": "error", "error": f"Unknown pipeline: {pipeline_name}"}

        task_id = f"{pipeline_name}_{int(time.time())}"
        self._active_tasks[task_id] = {"status": "running", "started": time.time()}
        self._task_count += 1

        def _run():
            try:
                import importlib
                mod = importlib.import_module(pipelines[pipeline_name])
                if hasattr(mod, "run_pipeline"):
                    mod.run_pipeline()
                elif hasattr(mod, "main"):
                    mod.main()
                self._active_tasks[task_id]["status"] = "complete"
            except Exception as e:
                self._active_tasks[task_id]["status"] = "error"
                self._active_tasks[task_id]["error"] = str(e)
            self._active_tasks[task_id]["finished"] = time.time()

        threading.Thread(target=_run, daemon=True, name=f"pipeline-{pipeline_name}").start()
        return {"status": "started", "task_id": task_id}

    # ── Warmup ─────────────────────────────────────────────────────

    def warmup(self) -> dict:
        """Run full warmup sequence (called once at startup)."""
        results = {}
        t0 = time.time()

        results["mlx"] = self.load_mlx()
        results["embeddings"] = self.load_embeddings()
        results["chroma"] = self.chroma_health_check()

        # ChromaDB tuning
        try:
            chroma_dir = os.environ.get("EDITH_CHROMA_DIR", str(ROOT_DIR / "chroma"))
            if os.path.isdir(chroma_dir):
                from server.chroma_tuning import tune_chroma_db
                tune_chroma_db(chroma_dir)
                results["chroma_tuning"] = {"status": "ok"}
        except Exception as e:
            results["chroma_tuning"] = {"status": "error", "error": str(e)}

        elapsed = time.time() - t0
        self._warmup_complete = True
        results["elapsed_seconds"] = round(elapsed, 1)
        log.info(f"§WORKER: Warmup complete in {elapsed:.1f}s")
        return results

    # ── Status ─────────────────────────────────────────────────────

    def status(self) -> dict:
        return {
            "status": "running",
            "uptime_seconds": round(time.time() - self._start_time, 1),
            "mlx_loaded": self._mlx_loaded,
            "embeddings_loaded": self._embeddings_loaded,
            "warmup_complete": self._warmup_complete,
            "task_count": self._task_count,
            "active_tasks": {k: v for k, v in self._active_tasks.items()
                           if v.get("status") == "running"},
        }

    # ── Request dispatcher ─────────────────────────────────────────

    def dispatch(self, action: str, params: dict) -> dict:
        """Dispatch a request to the appropriate handler."""
        handlers = {
            "status": lambda p: self.status(),
            "warmup": lambda p: self.warmup(),
            "load_mlx": lambda p: self.load_mlx(),
            "generate_local": lambda p: self.generate_local(**p),
            "load_embeddings": lambda p: self.load_embeddings(),
            "embed_texts": lambda p: self.embed_texts(p.get("texts", [])),
            "chroma_health": lambda p: self.chroma_health_check(),
            "scan_library": lambda p: self.scan_library(),
            "run_pipeline": lambda p: self.run_pipeline(p.get("pipeline", "")),
        }
        if action not in handlers:
            return {"status": "error", "error": f"Unknown action: {action}"}
        try:
            return handlers[action](params)
        except Exception as e:
            log.error(f"§WORKER: Error in {action}: {e}", exc_info=True)
            return {"status": "error", "error": str(e)}


# ── FastAPI Worker Server ──────────────────────────────────────────

def create_worker_app():
    """Create the FastAPI app for the worker process."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse

    worker = ComputeWorker()
    app = FastAPI(title="EDITH Compute Worker", version="1.0.0")

    @app.get("/status")
    async def get_status():
        return worker.status()

    @app.post("/dispatch")
    async def dispatch_action(request: Request):
        body = await request.json()
        action = body.get("action", "")
        params = body.get("params", {})
        result = await asyncio.to_thread(worker.dispatch, action, params)
        return result

    @app.post("/warmup")
    async def do_warmup():
        result = await asyncio.to_thread(worker.warmup)
        return result

    @app.post("/generate")
    async def generate(request: Request):
        body = await request.json()
        result = await asyncio.to_thread(
            worker.generate_local,
            prompt=body.get("prompt", ""),
            system=body.get("system", ""),
            max_tokens=body.get("max_tokens", 512),
            temperature=body.get("temperature", 0.1),
        )
        return result

    @app.post("/embed")
    async def embed(request: Request):
        body = await request.json()
        result = await asyncio.to_thread(worker.embed_texts, body.get("texts", []))
        return result

    @app.post("/pipeline")
    async def run_pipeline(request: Request):
        body = await request.json()
        result = worker.run_pipeline(body.get("pipeline", ""))
        return result

    # Auto-warmup on startup
    @app.on_event("startup")
    async def startup_warmup():
        log.info(f"§WORKER: Starting warmup on port {WORKER_PORT}")
        asyncio.get_event_loop().run_in_executor(None, worker.warmup)

    return app


def main():
    """Run the worker as a standalone process."""
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    log.info(f"§WORKER: Starting compute worker on {WORKER_HOST}:{WORKER_PORT}")

    app = create_worker_app()
    uvicorn.run(app, host=WORKER_HOST, port=WORKER_PORT, log_level="info")


if __name__ == "__main__":
    main()
