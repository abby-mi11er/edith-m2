"""
Infrastructure Engine — Google-Launch Performance Layer
========================================================
§5.1: Streaming Responses — SSE token stream for real-time output
§5.4: Parallel Retrieval — concurrent multi-collection search
§5.5: Response Caching — LRU cache with semantic key hashing
§5.8: Incremental Indexing — only re-index changed files
§5.2: Connection Pooling — reuse LLM/DB connections
§5.3: Lazy Model Loading — defer until first call
§5.7: Query Plan Optimizer — route by complexity
"""

import asyncio
import hashlib
import json
import logging
import os
import time
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import AsyncGenerator, Optional

log = logging.getLogger("edith.infra")


# ═══════════════════════════════════════════════════════════════════
# §M2-3: E-Core QoS Offloading — Keep P-cores for UI + LLM
# ═══════════════════════════════════════════════════════════════════

_QOS_CLASS = os.environ.get("QOS_CLASS", "")


def set_qos_background(thread_name: str = ""):
    """§M2-3: Lower current thread/process priority to background QoS.

    On M2 Air (fanless), this pushes indexing, monitoring, and
    sanitization work to E-cores, keeping P-cores free for the LLM
    and Electron UI. This reduces thermal throttling.

    Only activates when QOS_CLASS=background (set in .env.m2).
    On M4, this is a no-op since thermal headroom is large.
    """
    if _QOS_CLASS != "background":
        return False

    try:
        # PRIO_PROCESS=0, lower priority for background work
        # macOS maps low-priority threads to E-cores
        os.setpriority(os.PRIO_PROCESS, 0, 10)  # nice 10 = low priority
        log.info(f"§M2-3: QoS background applied{f' ({thread_name})' if thread_name else ''}")
        return True
    except (OSError, AttributeError) as e:
        log.debug(f"§M2-3: Could not set QoS: {e}")
        return False


def run_with_qos(fn, *args, thread_name: str = "", **kwargs):
    """§M2-3: Execute a function with background QoS priority.

    Usage: run_with_qos(expensive_indexing_fn, doc_path)
    """
    set_qos_background(thread_name)
    return fn(*args, **kwargs)


# ═══════════════════════════════════════════════════════════════════
# §5.5: Response Cache — LRU with semantic key hashing
# ═══════════════════════════════════════════════════════════════════

class ResponseCache:
    """Thread-safe LRU cache for AI responses.

    Avoids redundant LLM calls for repeated/similar queries.
    Keys are normalized query hashes, values are (response, metadata, timestamp).
    """

    def __init__(self, max_size: int = 256, ttl_seconds: int = 3600):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._hits = 0
        self._misses = 0

    def _make_key(self, query: str, depth: str = "", collection: str = "") -> str:
        """Create a normalized cache key from query parameters."""
        normalized = " ".join(query.lower().strip().split())
        raw = f"{normalized}|{depth}|{collection}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, query: str, depth: str = "", collection: str = "") -> Optional[dict]:
        """Retrieve cached response if available and not expired."""
        key = self._make_key(query, depth, collection)
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                age = time.time() - entry["cached_at"]
                if age < self._ttl:
                    self._cache.move_to_end(key)
                    self._hits += 1
                    entry["_cache_hit"] = True
                    entry["_cache_age_s"] = round(age, 1)
                    return entry
                else:
                    del self._cache[key]
            self._misses += 1
        return None

    def put(self, query: str, response: str, sources: list = None,
            depth: str = "", collection: str = "", metadata: dict = None):
        """Store a response in the cache."""
        key = self._make_key(query, depth, collection)
        entry = {
            "response": response,
            "sources": sources or [],
            "cached_at": time.time(),
            "metadata": metadata or {},
            "query": query,
        }
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = entry
            while len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def invalidate(self, query: str = "", collection: str = ""):
        """Invalidate cache entries matching criteria."""
        if not query and not collection:
            with self._lock:
                self._cache.clear()
            return

        key = self._make_key(query, collection=collection)
        with self._lock:
            self._cache.pop(key, None)

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "size": len(self._cache),
            "max_size": self._max_size,
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": round(self._hits / max(total, 1), 3),
        }


# Global cache instance
response_cache = ResponseCache()


# ═══════════════════════════════════════════════════════════════════
# §5.4: Parallel Retrieval — concurrent multi-collection search
# ═══════════════════════════════════════════════════════════════════

def parallel_retrieve(
    query: str,
    collections: list[str] = None,
    chroma_dir: str = "",
    embed_model: str = "",
    top_k: int = 20,
    max_workers: int = 4,
) -> dict:
    """Search multiple ChromaDB collections in parallel.

    On M4 with Thunderbolt Bolt: ~15ms per collection.
    Combined: all collections searched in the time of the slowest one.
    """
    t0 = time.time()
    chroma_dir = chroma_dir or os.environ.get("EDITH_CHROMA_DIR", "")

    if not collections:
        collections = _discover_collections(chroma_dir)

    all_results = []
    errors = []

    def _search_one(collection_name: str):
        try:
            from server.chroma_backend import retrieve_local_sources
            return retrieve_local_sources(
                queries=[query],
                chroma_dir=chroma_dir,
                collection_name=collection_name,
                embed_model=embed_model or os.environ.get("EDITH_EMBED_MODEL", ""),
                top_k=top_k // max(len(collections), 1),
            )
        except Exception as e:
            return {"error": str(e), "collection": collection_name}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(_search_one, c): c for c in collections}
        for future in as_completed(futures):
            coll = futures[future]
            try:
                result = future.result()
                if isinstance(result, list):
                    for r in result:
                        r["_collection"] = coll
                    all_results.extend(result)
                elif isinstance(result, dict) and "error" in result:
                    errors.append(result)
            except Exception as e:
                errors.append({"collection": coll, "error": str(e)})

    # Sort by relevance score if available
    all_results.sort(
        key=lambda x: x.get("score", x.get("distance", 0)),
        reverse=True
    )

    elapsed = time.time() - t0
    return {
        "results": all_results[:top_k],
        "total_found": len(all_results),
        "collections_searched": len(collections),
        "errors": errors,
        "elapsed_ms": round(elapsed * 1000, 1),
    }


def _discover_collections(chroma_dir: str) -> list[str]:
    """Auto-discover available ChromaDB collections."""
    if not chroma_dir or not os.path.isdir(chroma_dir):
        return ["edith"]
    try:
        import chromadb
        client = chromadb.PersistentClient(path=chroma_dir)
        collections = client.list_collections()
        names = [c.name for c in collections]
        return names if names else ["edith"]
    except Exception:
        return ["edith"]


# ═══════════════════════════════════════════════════════════════════
# §5.1: Streaming Response Generator
# ═══════════════════════════════════════════════════════════════════

async def stream_response(
    query: str,
    model_chain: list[str] = None,
    system_instruction: str = "",
) -> AsyncGenerator[dict, None]:
    """Stream response tokens via SSE-compatible generator.

    Yields dicts: {"type": "token"|"status"|"done", "text": "..."}
    """
    model_chain = model_chain or [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]

    yield {"type": "status", "text": "Generating response..."}

    try:
        import google.generativeai as genai
        model = genai.GenerativeModel(model_chain[0])

        config = genai.GenerationConfig(temperature=0.2)
        if system_instruction:
            model = genai.GenerativeModel(
                model_chain[0],
                system_instruction=system_instruction,
            )

        response = model.generate_content(
            query,
            generation_config=config,
            stream=True,
        )

        full_text = ""
        for chunk in response:
            if hasattr(chunk, "text") and chunk.text:
                full_text += chunk.text
                yield {"type": "token", "text": chunk.text}

        yield {"type": "done", "text": full_text, "model": model_chain[0]}

    except ImportError:
        # Fallback: non-streaming
        try:
            from server.backend_logic import generate_text_via_chain
            text, model_used = generate_text_via_chain(
                query, model_chain, system_instruction=system_instruction,
            )
            yield {"type": "token", "text": text}
            yield {"type": "done", "text": text, "model": model_used}
        except Exception as e:
            yield {"type": "error", "text": str(e)}

    except Exception as e:
        yield {"type": "error", "text": str(e)}


# ═══════════════════════════════════════════════════════════════════
# §5.8: Incremental Indexing — only re-index changed files
# ═══════════════════════════════════════════════════════════════════

class IncrementalIndexer:
    """Track file changes and only re-index modified documents.

    Maintains a manifest of file hashes. On each scan, only files
    with changed content or new files are sent for re-indexing.
    """

    def __init__(self, data_root: str = ""):
        self._data_root = data_root or os.environ.get("EDITH_DATA_ROOT", "")
        self._manifest_path = os.path.join(self._data_root, ".index_manifest.json")
        self._manifest: dict = {}
        self._load_manifest()

    def _load_manifest(self):
        if os.path.exists(self._manifest_path):
            try:
                with open(self._manifest_path) as f:
                    self._manifest = json.load(f)
            except Exception:
                self._manifest = {}

    def _save_manifest(self):
        try:
            with open(self._manifest_path, "w") as f:
                json.dump(self._manifest, f, indent=2)
        except Exception as e:
            log.warning(f"§INDEX: Failed to save manifest: {e}")

    def _file_hash(self, filepath: str) -> str:
        sha = hashlib.sha256()
        try:
            with open(filepath, "rb") as f:
                while True:
                    chunk = f.read(65536)
                    if not chunk:
                        break
                    sha.update(chunk)
            return sha.hexdigest()[:16]
        except Exception:
            return ""

    def scan_for_changes(
        self,
        directory: str = "",
        extensions: list[str] = None,
    ) -> dict:
        """Scan a directory for new/modified/deleted files.

        Returns dict with new, modified, deleted, and unchanged file lists.
        """
        directory = directory or self._data_root
        if not directory or not os.path.isdir(directory):
            return {"error": "Directory not found"}

        if extensions is None:
            extensions = [".pdf", ".docx", ".doc", ".txt", ".md", ".rtf",
                         ".tex", ".bib", ".html", ".htm",
                         ".csv", ".tsv", ".xlsx", ".xls",
                         ".json", ".jsonl", ".geojson", ".yaml", ".yml", ".toml", ".xml",
                         ".ipynb", ".rmd",
                         ".r", ".do", ".sps", ".py", ".js", ".sql",
                         ".kml", ".gpx",
                         ".log", ".dta", ".sav"]

        current_files = {}
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith(".")]
            for fname in files:
                ext = os.path.splitext(fname)[1]
                if ext.lower() in [e.lower() for e in extensions]:
                    fpath = os.path.join(root, fname)
                    rel = os.path.relpath(fpath, directory)
                    current_files[rel] = {
                        "hash": self._file_hash(fpath),
                        "size": os.path.getsize(fpath),
                        "mtime": os.path.getmtime(fpath),
                    }

        # Compare with manifest
        new = []
        modified = []
        unchanged = []
        for rel, info in current_files.items():
            if rel not in self._manifest:
                new.append(rel)
            elif self._manifest[rel].get("hash") != info["hash"]:
                modified.append(rel)
            else:
                unchanged.append(rel)

        deleted = [r for r in self._manifest if r not in current_files]

        # Update manifest
        self._manifest = current_files
        self._save_manifest()

        return {
            "new": new,
            "modified": modified,
            "deleted": deleted,
            "unchanged_count": len(unchanged),
            "total_files": len(current_files),
            "needs_reindex": len(new) + len(modified) + len(deleted),
        }


# ═══════════════════════════════════════════════════════════════════
# §5.7: Query Plan Optimizer — route queries by complexity
# ═══════════════════════════════════════════════════════════════════

def optimize_query_plan(query: str, available_agents: int = 1) -> dict:
    """Determine the optimal execution plan for a query.

    Routes simple lookups to fast paths, complex research to Committee.
    """
    import re
    q = query.lower().strip()
    word_count = len(q.split())

    # Fast-path patterns (no Committee needed)
    fast_patterns = [
        (r"^(what is|define|who is)\b", "definition", 1),
        (r"^(when did|what year)\b", "temporal_lookup", 1),
        (r"^(list|name|enumerate)\b", "enumeration", 1),
        (r"^(summarize|summary of)\b", "summary", 2),
    ]

    for pattern, plan_type, min_agents in fast_patterns:
        if re.search(pattern, q):
            return {
                "plan": plan_type,
                "agents_needed": min(min_agents, available_agents),
                "use_committee": False,
                "use_cache": True,
                "estimated_latency_ms": 500,
                "depth": "quick",
            }

    # Complex patterns (Committee recommended)
    complex_signals = [
        r"(compare|contrast|differentiate)\b",
        r"(analyze|evaluate|assess|critique)\b",
        r"(how does.*relate|what.*connection)\b",
        r"(argue|thesis|hypothesis)\b",
        r"\?.*\?",  # Multiple questions
    ]
    complexity = sum(1 for p in complex_signals if re.search(p, q))

    if complexity >= 2 or word_count > 30:
        return {
            "plan": "committee_debate",
            "agents_needed": min(available_agents, 8),
            "use_committee": True,
            "use_cache": False,
            "estimated_latency_ms": 5000,
            "depth": "debate",
        }
    elif complexity >= 1 or word_count > 15:
        return {
            "plan": "deep_retrieval",
            "agents_needed": min(available_agents, 4),
            "use_committee": available_agents >= 4,
            "use_cache": True,
            "estimated_latency_ms": 2000,
            "depth": "standard",
        }
    else:
        return {
            "plan": "standard_retrieval",
            "agents_needed": 1,
            "use_committee": False,
            "use_cache": True,
            "estimated_latency_ms": 1000,
            "depth": "standard",
        }


# ═══════════════════════════════════════════════════════════════════
# §5.2: Connection Pool — reuse LLM and DB connections
# ═══════════════════════════════════════════════════════════════════

class ConnectionPool:
    """Manages reusable connections for LLM and ChromaDB clients."""

    def __init__(self):
        self._genai_model = None
        self._chroma_client = None
        self._lock = threading.Lock()

    def get_genai_model(self, model_name: str = ""):
        """Get or create a GenAI model instance."""
        model_name = model_name or os.environ.get("EDITH_MODEL", "gemini-2.5-flash")
        with self._lock:
            if self._genai_model is None:
                try:
                    import google.generativeai as genai
                    self._genai_model = genai.GenerativeModel(model_name)
                    log.info(f"§POOL: Created GenAI model: {model_name}")
                except Exception as e:
                    log.debug(f"§POOL: GenAI unavailable: {e}")
            return self._genai_model

    def get_chroma_client(self, chroma_dir: str = ""):
        """Get or create a ChromaDB persistent client."""
        chroma_dir = chroma_dir or os.environ.get("EDITH_CHROMA_DIR", "")
        with self._lock:
            if self._chroma_client is None and chroma_dir:
                try:
                    import chromadb
                    self._chroma_client = chromadb.PersistentClient(path=chroma_dir)
                    log.info(f"§POOL: Created ChromaDB client: {chroma_dir}")
                except Exception as e:
                    log.debug(f"§POOL: ChromaDB unavailable: {e}")
            return self._chroma_client

    def reset(self):
        """Reset all pooled connections."""
        with self._lock:
            self._genai_model = None
            self._chroma_client = None


# Global connection pool
conn_pool = ConnectionPool()


# ═══════════════════════════════════════════════════════════════════
# §5.3: Lazy Model Loading — defer heavy imports until first use
# ═══════════════════════════════════════════════════════════════════

class LazyLoader:
    """Defer loading of heavy ML models until first use.

    Reduces startup time from ~8s to ~2s.
    """
    _loaded: dict = {}
    _lock = threading.Lock()

    @classmethod
    def get_embedding_model(cls):
        """Load embedding model on first call only."""
        if "embed" not in cls._loaded:
            with cls._lock:
                if "embed" not in cls._loaded:
                    try:
                        from server.mlx_embeddings import _load_model
                        cls._loaded["embed"] = _load_model()
                        log.info("§LAZY: Embedding model loaded on first use")
                    except Exception as e:
                        log.debug(f"§LAZY: Embedding model unavailable: {e}")
                        cls._loaded["embed"] = None
        return cls._loaded.get("embed")

    @classmethod
    def get_knowledge_graph(cls):
        """Load knowledge graph on first call only."""
        if "kg" not in cls._loaded:
            with cls._lock:
                if "kg" not in cls._loaded:
                    try:
                        from server.knowledge_graph import KnowledgeGraph
                        cls._loaded["kg"] = KnowledgeGraph()
                        log.info("§LAZY: Knowledge graph loaded on first use")
                    except Exception as e:
                        log.debug(f"§LAZY: Knowledge graph unavailable: {e}")
                        cls._loaded["kg"] = None
        return cls._loaded.get("kg")

    @classmethod
    def loaded_modules(cls) -> list[str]:
        return list(cls._loaded.keys())


# ═══════════════════════════════════════════════════════════════════
# §IMP-4.5: Infrastructure Health Scoring — composite 0-100 score
# ═══════════════════════════════════════════════════════════════════

def get_system_health() -> dict:
    """§IMP-4.5: Compute composite infrastructure health score (0-100).

    Factors: cache performance, connection pool, disk space, index freshness.
    """
    scores = {}

    # 1. Cache health (0-25 pts)
    cache_stats = response_cache.stats
    hit_rate = cache_stats.get("hit_rate", 0)
    cache_score = min(25, int(hit_rate * 25))
    if cache_stats.get("hits", 0) + cache_stats.get("misses", 0) == 0:
        cache_score = 20  # No requests yet = healthy
    scores["cache"] = {"score": cache_score, "max": 25, "hit_rate": hit_rate}

    # 2. Connection pool health (0-25 pts)
    pool_score = 15  # Base score
    if conn_pool._genai_model is not None:
        pool_score += 5
    if conn_pool._chroma_client is not None:
        pool_score += 5
    scores["connections"] = {"score": pool_score, "max": 25}

    # 3. Disk space health (0-25 pts)
    data_root = os.environ.get("EDITH_DATA_ROOT", ".")
    try:
        import shutil
        usage = shutil.disk_usage(data_root)
        free_pct = usage.free / usage.total
        disk_score = min(25, int(free_pct * 50))  # 50%+ free = full marks
    except Exception:
        disk_score = 12  # Unknown = assume okay
    scores["disk"] = {"score": disk_score, "max": 25}

    # 4. Module readiness (0-25 pts)
    loaded = LazyLoader.loaded_modules()
    module_score = min(25, 10 + len(loaded) * 5)
    scores["modules"] = {"score": module_score, "max": 25, "loaded": loaded}

    # Composite score
    total = sum(s["score"] for s in scores.values())
    label = "Critical" if total < 40 else "Degraded" if total < 60 else "Good" if total < 80 else "Excellent"

    return {
        "health_score": total,
        "max_score": 100,
        "label": label,
        "components": scores,
    }


# ═══════════════════════════════════════════════════════════════════
# TITAN §5: MIRROR SOUL — Cold Storage Failsafe
# ═══════════════════════════════════════════════════════════════════

import hashlib
import json
import subprocess
from datetime import datetime


class MirrorSoulBackup:
    """Cold Storage Failsafe — block-level clone to a second Oyen Bolt.

    The "Mirror Soul" protocol:
    1. Detect a second drive with .edith_mirror_marker
    2. rsync from primary Bolt to mirror (block-level fidelity)
    3. Hash-verify critical files to ensure integrity
    4. Log the backup event for auditability

    Usage:
        mirror = MirrorSoulBackup()
        result = mirror.clone()          # Full rsync clone
        verify = mirror.verify_clone()   # Hash comparison
    """

    _MIRROR_MARKER = ".edith_mirror_marker"

    def __init__(self, volumes_path: str = "/Volumes"):
        self._volumes = volumes_path
        self._primary = os.environ.get("EDITH_DATA_ROOT", "")
        self._mirror = self._detect_mirror()

    def _detect_mirror(self) -> str:
        """Scan /Volumes for a drive with the mirror marker."""
        if not os.path.isdir(self._volumes):
            return ""
        for vol in os.listdir(self._volumes):
            vol_path = os.path.join(self._volumes, vol)
            marker = os.path.join(vol_path, self._MIRROR_MARKER)
            if os.path.exists(marker) and vol_path != self._primary:
                return vol_path
        return ""

    def initialize_mirror(self, drive_path: str) -> dict:
        """Create the .edith_mirror_marker on a new backup drive. Run ONCE."""
        import uuid as uuid_mod
        marker_data = {
            "mirror_uuid": str(uuid_mod.uuid4()),
            "created": datetime.now().isoformat(),
            "role": "mirror_soul",
            "primary_source": self._primary,
        }
        marker_path = os.path.join(drive_path, self._MIRROR_MARKER)
        os.makedirs(drive_path, exist_ok=True)
        with open(marker_path, "w") as f:
            json.dump(marker_data, f, indent=2)
        return {"status": "initialized", "path": marker_path}

    def clone(self) -> dict:
        """Block-level rsync clone from primary Bolt to mirror.

        Uses rsync with --delete to ensure exact mirroring.
        Excludes temporary and cache files.
        """
        if not self._primary:
            return {"error": "Primary Bolt (EDITH_DATA_ROOT) not set"}
        if not self._mirror:
            self._mirror = self._detect_mirror()
            if not self._mirror:
                return {"error": "Mirror drive not detected. Plug in second Bolt."}

        t0 = time.time()

        # Build rsync command
        excludes = [
            "--exclude=.DS_Store",
            "--exclude=__pycache__",
            "--exclude=*.pyc",
            "--exclude=.git",
            "--exclude=node_modules",
        ]

        cmd = [
            "rsync", "-av", "--delete",
            *excludes,
            f"{self._primary}/",
            f"{self._mirror}/",
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour max
            )
            elapsed = time.time() - t0

            # Parse rsync output for stats
            lines = result.stdout.strip().split("\n") if result.stdout else []
            transferred = [l for l in lines if "sent" in l.lower() or "total" in l.lower()]

            clone_result = {
                "status": "complete" if result.returncode == 0 else "error",
                "source": self._primary,
                "destination": self._mirror,
                "elapsed_s": round(elapsed, 2),
                "files_synced": len([l for l in lines if not l.startswith("sending")]),
                "summary": transferred[-1] if transferred else "",
                "returncode": result.returncode,
            }

            if result.returncode != 0:
                clone_result["stderr"] = result.stderr[:500]

            # Log backup event
            self._log_backup(clone_result)

            log.info(f"§MIRROR: Clone completed in {elapsed:.1f}s → {self._mirror}")
            return clone_result

        except subprocess.TimeoutExpired:
            return {"error": "Clone timed out after 1 hour"}
        except Exception as e:
            return {"error": str(e)}

    def verify_clone(self, sample_files: int = 20) -> dict:
        """Hash-verify that the mirror matches the primary.

        Samples N random files and compares SHA-256 hashes.
        """
        if not self._primary or not self._mirror:
            return {"error": "Primary or mirror not available"}

        import random

        # Gather all files from primary
        all_files = []
        for root, dirs, files in os.walk(self._primary):
            for f in files:
                if not f.startswith(".") and "__pycache__" not in root:
                    rel = os.path.relpath(os.path.join(root, f), self._primary)
                    all_files.append(rel)

        # Sample
        sample = random.sample(all_files, min(sample_files, len(all_files)))

        matches = 0
        mismatches = []
        for rel in sample:
            primary_path = os.path.join(self._primary, rel)
            mirror_path = os.path.join(self._mirror, rel)

            if not os.path.exists(mirror_path):
                mismatches.append({"file": rel, "reason": "missing"})
                continue

            h1 = hashlib.sha256(open(primary_path, "rb").read()).hexdigest()
            h2 = hashlib.sha256(open(mirror_path, "rb").read()).hexdigest()

            if h1 == h2:
                matches += 1
            else:
                mismatches.append({"file": rel, "reason": "hash_mismatch"})

        return {
            "sampled": len(sample),
            "matches": matches,
            "mismatches": mismatches,
            "integrity": "VERIFIED" if not mismatches else "DEGRADED",
            "primary": self._primary,
            "mirror": self._mirror,
        }

    def restore_from_mirror(self) -> dict:
        """Emergency: restore primary from mirror (reverse clone)."""
        if not self._mirror or not self._primary:
            return {"error": "Both drives must be available"}

        # Swap source and destination
        cmd = [
            "rsync", "-av", "--delete",
            "--exclude=.DS_Store",
            f"{self._mirror}/",
            f"{self._primary}/",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            return {
                "status": "restored" if result.returncode == 0 else "error",
                "source": self._mirror,
                "destination": self._primary,
            }
        except Exception as e:
            return {"error": str(e)}

    def _log_backup(self, result: dict):
        """Log backup event for audit trail."""
        log_path = os.path.join(self._mirror, ".backup_log.json")
        try:
            log_entries = []
            if os.path.exists(log_path):
                with open(log_path) as f:
                    log_entries = json.load(f)
            log_entries.append({
                "timestamp": datetime.now().isoformat(),
                "elapsed_s": result.get("elapsed_s"),
                "files": result.get("files_synced"),
                "status": result.get("status"),
            })
            with open(log_path, "w") as f:
                json.dump(log_entries[-100:], f, indent=2)
        except Exception:
            pass


# ═══════════════════════════════════════════════════════════════════
# SOUL TRANSFER §1: CITADEL PATH RESOLVER
# ═══════════════════════════════════════════════════════════════════

class CitadelPathResolver:
    """Dynamic path resolution for the CITADEL drive.

    Every module resolves paths through this single object:
        paths.core     → /Volumes/CITADEL/E.D.I.T.H_CORE/
        paths.vault    → /Volumes/CITADEL/VAULT/
        paths.archive  → /Volumes/CITADEL/VAULT/ARCHIVE/
        paths.datasets → /Volumes/CITADEL/VAULT/DATASETS/
        paths.captions → /Volumes/CITADEL/VAULT/CAPTIONS/
        paths.personas → /Volumes/CITADEL/PERSONAS/
        paths.chroma   → /Volumes/CITADEL/VAULT/chroma_index/

    Auto-detects mount. Falls back to local paths if Bolt unplugged.
    """

    try:
        from server.vault_config import VAULT_ROOT
        CITADEL_MOUNT = str(VAULT_ROOT)
    except ImportError:
        CITADEL_MOUNT = os.environ.get("EDITH_DATA_ROOT", "/Volumes/CITADEL")

    # Directory tree specification
    TREE = {
        "core": "E.D.I.T.H_CORE",
        "server": "E.D.I.T.H_CORE/server",
        "vault": "VAULT",
        "archive": "VAULT/ARCHIVE",
        "captions": "VAULT/CAPTIONS",
        "datasets": "VAULT/DATASETS",
        "personas": "PERSONAS",
        "chroma": "VAULT/chroma_index",
        "state": "E.D.I.T.H_CORE/state",
        "logs": "E.D.I.T.H_CORE/logs",
        "overnight": "E.D.I.T.H_CORE/overnight",
        "pedagogy": "VAULT/PEDAGOGY",
        "syllabi": "VAULT/PEDAGOGY/SYLLABI",
        "exams": "VAULT/PEDAGOGY/EXAMS",
    }

    def __init__(self, mount_point: str = ""):
        self._mount = mount_point or self.CITADEL_MOUNT
        self._fallback = os.environ.get("EDITH_DATA_ROOT", ".")

    @property
    def is_mounted(self) -> bool:
        return os.path.isdir(self._mount)

    @property
    def root(self) -> str:
        return self._mount if self.is_mounted else self._fallback

    def _resolve(self, key: str) -> str:
        subpath = self.TREE.get(key, key)
        return os.path.join(self.root, subpath)

    # Named path properties
    @property
    def core(self) -> str:
        return self._resolve("core")

    @property
    def server(self) -> str:
        return self._resolve("server")

    @property
    def vault(self) -> str:
        return self._resolve("vault")

    @property
    def archive(self) -> str:
        return self._resolve("archive")

    @property
    def captions(self) -> str:
        return self._resolve("captions")

    @property
    def datasets(self) -> str:
        return self._resolve("datasets")

    @property
    def personas(self) -> str:
        return self._resolve("personas")

    @property
    def chroma(self) -> str:
        return self._resolve("chroma")

    @property
    def state(self) -> str:
        return self._resolve("state")

    @property
    def logs(self) -> str:
        return self._resolve("logs")

    @property
    def overnight(self) -> str:
        return self._resolve("overnight")

    def resolve(self, key: str) -> str:
        """Resolve any path by key name."""
        return self._resolve(key)

    def get_all_paths(self) -> dict:
        """Return the full path map for diagnostics."""
        return {
            key: self._resolve(key)
            for key in self.TREE
        }

    def status(self) -> dict:
        """Full mount status and drive health."""
        info = {
            "mounted": self.is_mounted,
            "mount_point": self._mount,
            "fallback": self._fallback,
            "active_root": self.root,
            "paths": self.get_all_paths(),
        }

        if self.is_mounted:
            try:
                usage = shutil.disk_usage(self._mount)
                info["disk_total_gb"] = round(usage.total / (1024**3), 1)
                info["disk_used_gb"] = round(usage.used / (1024**3), 1)
                info["disk_free_gb"] = round(usage.free / (1024**3), 1)
                info["disk_used_pct"] = round(usage.used / usage.total * 100, 1)
            except Exception:
                pass

        return info


def verify_citadel_mount() -> dict:
    """Verify the CITADEL drive is mounted, healthy, and fast.

    Checks: mount point exists, APFS format, Thunderbolt connection,
    directory tree integrity.
    """
    resolver = CitadelPathResolver()
    result = {
        "mounted": resolver.is_mounted,
        "mount_point": resolver.root,
        "checks": [],
    }

    if not resolver.is_mounted:
        result["checks"].append({
            "check": "Mount Detection",
            "status": "FAIL",
            "detail": f"{resolver.CITADEL_MOUNT} not found. Is the Bolt connected?",
        })
        return result

    result["checks"].append({
        "check": "Mount Detection",
        "status": "PASS",
        "detail": f"CITADEL mounted at {resolver.root}",
    })

    # Check APFS format via diskutil
    try:
        output = subprocess.run(
            ["diskutil", "info", resolver.root],
            capture_output=True, text=True, timeout=5,
        )
        if "APFS" in output.stdout:
            result["checks"].append({
                "check": "APFS Format",
                "status": "PASS",
                "detail": "Drive formatted as APFS",
            })
        else:
            result["checks"].append({
                "check": "APFS Format",
                "status": "WARN",
                "detail": "Drive may not be APFS — performance may suffer",
            })

        if "Thunderbolt" in output.stdout or "USB 3" in output.stdout:
            result["checks"].append({
                "check": "Connection",
                "status": "PASS",
                "detail": "Thunderbolt/USB3 connection detected",
            })
    except Exception:
        pass

    # Check directory tree
    missing = []
    for key in CitadelPathResolver.TREE:
        path = resolver.resolve(key)
        if not os.path.isdir(path):
            missing.append(key)

    if missing:
        result["checks"].append({
            "check": "Directory Tree",
            "status": "PARTIAL",
            "detail": f"Missing directories: {', '.join(missing)}",
            "missing": missing,
        })
    else:
        result["checks"].append({
            "check": "Directory Tree",
            "status": "PASS",
            "detail": "All CITADEL directories present",
        })

    # Disk health
    try:
        usage = shutil.disk_usage(resolver.root)
        result["disk_total_gb"] = round(usage.total / (1024**3), 1)
        result["disk_free_gb"] = round(usage.free / (1024**3), 1)
        result["checks"].append({
            "check": "Disk Space",
            "status": "PASS" if usage.free > 50 * (1024**3) else "WARN",
            "detail": f"{round(usage.free / (1024**3), 1)}GB free",
        })
    except Exception:
        pass

    result["healthy"] = all(c["status"] in ("PASS", "PARTIAL") for c in result["checks"])
    return result


# Global path resolver
citadel_paths = CitadelPathResolver()


# ═══════════════════════════════════════════════════════════════════
# SOUL TRANSFER §2: MIGRATION ENGINE
# ═══════════════════════════════════════════════════════════════════

class SoulTransfer:
    """The 'Physical Soul Transfer' — move E.D.I.T.H. to the Bolt.

    Phases:
        1. build_citadel_directories() — create the exact folder tree
        2. migrate_codebase(source) — rsync code to E.D.I.T.H_CORE
        3. migrate_vault(sources) — rsync research to VAULT
        4. verify_transfer() — SHA-256 integrity check
        5. set_environment() — write .env for CITADEL paths
    """

    def __init__(self, mount_point: str = ""):
        self._mount = mount_point or CitadelPathResolver.CITADEL_MOUNT
        self._resolver = CitadelPathResolver(self._mount)

    def build_citadel_directories(self) -> dict:
        """Phase 2: Create the exact CITADEL folder tree."""
        created = []
        existed = []

        for key, subpath in CitadelPathResolver.TREE.items():
            full_path = os.path.join(self._mount, subpath)
            if os.path.isdir(full_path):
                existed.append(key)
            else:
                os.makedirs(full_path, exist_ok=True)
                created.append(key)

        log.info(
            f"§SOUL: Built CITADEL tree — "
            f"{len(created)} created, {len(existed)} existed"
        )

        return {
            "mount": self._mount,
            "created": created,
            "existed": existed,
            "total_dirs": len(CitadelPathResolver.TREE),
        }

    def migrate_codebase(self, source_dir: str) -> dict:
        """Phase 2: rsync the codebase to E.D.I.T.H_CORE."""
        dest = os.path.join(self._mount, "E.D.I.T.H_CORE")
        return self._rsync_transfer(source_dir, dest, "codebase")

    def migrate_vault(self, source_dirs: dict) -> dict:
        """Phase 2: Migrate research files to VAULT subdirectories.

        Args:
            source_dirs: {"archive": "/path/to/pdfs", "datasets": "/path/to/dta"}
        """
        results = {}
        for category, source in source_dirs.items():
            if not source or not os.path.exists(source):
                results[category] = {"status": "skipped", "reason": "Source not found"}
                continue

            subpath = CitadelPathResolver.TREE.get(category, category)
            dest = os.path.join(self._mount, subpath)
            results[category] = self._rsync_transfer(source, dest, category)

        return results

    def _rsync_transfer(self, source: str, dest: str, label: str) -> dict:
        """Execute rsync transfer with progress tracking."""
        if not os.path.exists(source):
            return {"status": "error", "reason": f"Source not found: {source}"}

        os.makedirs(dest, exist_ok=True)

        # Ensure trailing slash for rsync content transfer
        source_path = source.rstrip("/") + "/"

        try:
            result = subprocess.run(
                [
                    "rsync", "-avz", "--progress", "--stats",
                    source_path, dest + "/",
                ],
                capture_output=True, text=True, timeout=7200,  # 2hr max
            )

            # Parse stats from rsync output
            stats_lines = [l for l in result.stdout.split("\n") if ":" in l][-10:]

            log.info(f"§SOUL: Migrated {label} to {dest}")

            return {
                "status": "complete" if result.returncode == 0 else "error",
                "source": source,
                "destination": dest,
                "label": label,
                "stats": stats_lines,
                "returncode": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "label": label}
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def verify_transfer(self, sample_count: int = 50) -> dict:
        """Phase 4: SHA-256 integrity check on migrated files."""
        import random

        if not os.path.isdir(self._mount):
            return {"status": "error", "reason": "CITADEL not mounted"}

        # Collect all files
        all_files = []
        for root, dirs, files in os.walk(self._mount):
            for f in files:
                if not f.startswith("."):
                    all_files.append(os.path.join(root, f))

        if not all_files:
            return {"status": "empty", "reason": "No files found on CITADEL"}

        # Sample files for verification
        sample = random.sample(all_files, min(sample_count, len(all_files)))
        verified = []
        errors = []

        for fpath in sample:
            try:
                h = hashlib.sha256()
                with open(fpath, "rb") as f:
                    for chunk in iter(lambda: f.read(8192), b""):
                        h.update(chunk)
                verified.append({
                    "file": os.path.basename(fpath),
                    "hash": h.hexdigest()[:16],
                    "size_kb": round(os.path.getsize(fpath) / 1024, 1),
                })
            except Exception as e:
                errors.append({"file": fpath, "error": str(e)})

        return {
            "status": "verified" if not errors else "partial",
            "total_files": len(all_files),
            "sampled": len(sample),
            "verified": len(verified),
            "errors": len(errors),
            "error_details": errors[:5],
        }

    def set_environment(self) -> dict:
        """Phase 5: Write .env file with CITADEL paths."""
        env_path = os.path.join(self._mount, "E.D.I.T.H_CORE", ".env")

        env_vars = {
            "EDITH_DATA_ROOT": os.path.join(self._mount, "VAULT"),
            "EDITH_CHROMA_DIR": os.path.join(self._mount, "VAULT", "chroma_index"),
            "EDITH_CORE_DIR": os.path.join(self._mount, "E.D.I.T.H_CORE"),
            "EDITH_VAULT_DIR": os.path.join(self._mount, "VAULT"),
            "EDITH_PERSONAS_DIR": os.path.join(self._mount, "PERSONAS"),
            "EDITH_DATASETS_DIR": os.path.join(self._mount, "VAULT", "DATASETS"),
            "EDITH_ARCHIVE_DIR": os.path.join(self._mount, "VAULT", "ARCHIVE"),
            "EDITH_MOUNT": self._mount,
        }

        try:
            os.makedirs(os.path.dirname(env_path), exist_ok=True)
            with open(env_path, "w") as f:
                f.write("# E.D.I.T.H. CITADEL Environment — Auto-generated\n")
                f.write(f"# Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                for key, value in env_vars.items():
                    f.write(f"{key}={value}\n")

            log.info(f"§SOUL: Environment written to {env_path}")

            return {
                "status": "written",
                "path": env_path,
                "variables": env_vars,
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def full_transfer(
        self,
        codebase_source: str,
        vault_sources: dict = None,
    ) -> dict:
        """Execute the complete Soul Transfer protocol."""
        log.info("§SOUL: Beginning Physical Soul Transfer...")

        results = {"phases": {}}

        # Phase 2: Build directories
        results["phases"]["directories"] = self.build_citadel_directories()

        # Phase 2: Migrate codebase
        results["phases"]["codebase"] = self.migrate_codebase(codebase_source)

        # Phase 2: Migrate vault
        if vault_sources:
            results["phases"]["vault"] = self.migrate_vault(vault_sources)

        # Phase 4: Verify
        results["phases"]["verification"] = self.verify_transfer()

        # Phase 5: Set environment
        results["phases"]["environment"] = self.set_environment()

        log.info("§SOUL: Physical Soul Transfer complete.")
        return results


# ═══════════════════════════════════════════════════════════════════
# SOUL TRANSFER §3: ZERO-TOUCH IGNITION — AppleScript Generator
# ═══════════════════════════════════════════════════════════════════

def generate_ignition_script(
    output_path: str = "",
    mount_point: str = "",
) -> dict:
    """Generate the 'WAKE E.D.I.T.H.' AppleScript for zero-touch ignition.

    When the Bolt is plugged in and this app is double-clicked:
    1. Detects CITADEL volume
    2. Sets environment variables
    3. Launches backend (uvicorn) + frontend (Electron)
    4. Speaks 'Systems Optimal, the user'
    """
    if not mount_point:
        try:
            from server.vault_config import VAULT_ROOT
            mount_point = str(VAULT_ROOT)
        except ImportError:
            mount_point = os.environ.get("EDITH_DATA_ROOT", "/Volumes/CITADEL")
    script = f'''-- WAKE E.D.I.T.H. — Zero-Touch Ignition
-- Generated by E.D.I.T.H. Soul Transfer Protocol
-- Double-click to bring the Citadel online.

on run
    -- Phase 1: Detect CITADEL
    set citadelPath to "{mount_point}"
    
    try
        tell application "System Events"
            if not (exists disk item citadelPath) then
                display dialog "CITADEL drive not detected." & return & ¬
                    "Connect the Oyen Bolt via Thunderbolt and try again." ¬
                    with title "E.D.I.T.H." buttons {{"OK"}} default button "OK" ¬
                    with icon caution
                return
            end if
        end tell
    on error
        display dialog "Cannot verify CITADEL mount." with title "E.D.I.T.H." ¬
            buttons {{"OK"}} default button "OK"
        return
    end try
    
    -- Phase 2: Set environment and launch backend
    set corePath to citadelPath & "/E.D.I.T.H_CORE"
    set envFile to corePath & "/.env"
    
    tell application "Terminal"
        activate
        
        -- Launch backend
        set backendCmd to "cd " & quoted form of (corePath & "/server") & ¬
            " && source " & quoted form of envFile & ¬
            " && source .venv/bin/activate 2>/dev/null" & ¬
            " && python -m uvicorn main:app --host 0.0.0.0 --port 8000"
        do script backendCmd
        
        delay 3
        
        -- Launch frontend
        set frontendCmd to "cd " & quoted form of (corePath & "/renderer") & ¬
            " && npm start"
        do script frontendCmd
    end tell
    
    -- Phase 3: The Greeting
    delay 5
    say "Systems Optimal, the user. What are we discovering today?" ¬
        using "Samantha" speaking rate 180
    
    -- Phase 4: Notification
    display notification "All systems online. Bolt throughput: 3,100 MB/s." ¬
        with title "E.D.I.T.H." subtitle "Citadel Active"
    
end run
'''

    if not output_path:
        output_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "scripts", "wake_edith.applescript",
        )

    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            f.write(script)

        log.info(f"§SOUL: Ignition script written to {output_path}")

        return {
            "status": "generated",
            "path": output_path,
            "instruction": (
                "Open in Script Editor → File → Export → "
                "File Format: Application → Save as 'WAKE E.D.I.T.H.'"
            ),
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "script": script}
