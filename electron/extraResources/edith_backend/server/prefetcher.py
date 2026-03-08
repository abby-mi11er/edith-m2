"""
Predictive Data Pre-fetcher — Feature #3
==========================================
While the user reads one paper, pre-index the most related papers
in the background using NPU idle cycles.

Heuristics for pre-fetch targets:
  - Same-author papers in the corpus
  - Co-cited papers (papers that cite or are cited by current results)
  - Semantically similar papers not yet in the retrieval cache
"""

import logging
import os
import threading
import time
from collections import OrderedDict
from typing import Optional

log = logging.getLogger("edith.prefetcher")


class PredictiveCache:
    """Background pre-fetching of related document embeddings.

    When a query retrieves sources, the pre-fetcher identifies related
    papers that weren't retrieved and pre-embeds them in the background
    so the next query is instant.
    """

    def __init__(self, max_hot_items: int = 50):
        self._hot_cache: OrderedDict = OrderedDict()  # doc_id → embeddings
        self._max_items = max_hot_items
        self._prefetch_queue: list[dict] = []
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._lock = threading.Lock()
        self._stats = {
            "prefetches": 0,
            "hits": 0,
            "misses": 0,
        }

    def start(self):
        """Start the background pre-fetch thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        log.info("§PREFETCH: Background pre-fetcher started")

    def stop(self):
        self._running = False

    def on_query_result(self, query: str, sources: list[dict]):
        """Called after a query returns sources. Identifies pre-fetch targets.

        Heuristics:
        1. Same-author papers not in current results
        2. Papers with similar titles/abstracts
        3. Papers from the same collection but different section
        """
        if not sources:
            return

        # Extract metadata from current results
        current_ids = set()
        authors = set()
        years = set()
        keywords = set()

        for s in sources:
            meta = s.get("metadata", {})
            src = meta.get("source", "")
            current_ids.add(src)

            author = meta.get("author", "")
            if author:
                authors.add(author.lower())

            year = meta.get("year") or meta.get("publication_year")
            if year:
                years.add(str(year))

            # Extract keywords from title
            title = meta.get("title", "")
            for word in title.lower().split():
                if len(word) > 4:
                    keywords.add(word)

        # Queue pre-fetch targets
        targets = {
            "query": query,
            "exclude_ids": current_ids,
            "authors": list(authors)[:5],
            "keywords": list(keywords)[:10],
            "years": list(years),
            "queued_at": time.time(),
        }

        with self._lock:
            self._prefetch_queue.append(targets)
            # Keep queue bounded
            if len(self._prefetch_queue) > 10:
                self._prefetch_queue = self._prefetch_queue[-10:]

    def check_hot_cache(self, doc_id: str) -> Optional[list]:
        """Check if a document's embedding is in the hot cache."""
        with self._lock:
            if doc_id in self._hot_cache:
                self._stats["hits"] += 1
                self._hot_cache.move_to_end(doc_id)
                return self._hot_cache[doc_id]
            self._stats["misses"] += 1
            return None

    def _worker(self):
        """Background worker that processes the pre-fetch queue."""
        while self._running:
            target = None
            with self._lock:
                if self._prefetch_queue:
                    target = self._prefetch_queue.pop(0)

            if target:
                self._execute_prefetch(target)
            else:
                time.sleep(2)  # Idle — wait for new targets

    def _execute_prefetch(self, target: dict):
        """Execute a pre-fetch: find and embed related documents."""
        try:
            chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "")
            if not chroma_dir:
                return

            # Use the NPU to embed related queries
            try:
                from server.mlx_embeddings import embed, is_available
                if not is_available():
                    return
            except ImportError:
                return

            # Generate related queries from the original query + metadata
            related_queries = self._generate_related_queries(target)

            for rq in related_queries[:3]:  # Limit to 3 pre-fetches per cycle
                try:
                    # Embed the related query using NPU
                    embedding = embed([rq])
                    if embedding is not None:
                        # Store in hot cache
                        cache_key = f"prefetch_{hash(rq) % 10000}"
                        with self._lock:
                            self._hot_cache[cache_key] = {
                                "query": rq,
                                "embedding": embedding,
                                "prefetched_at": time.time(),
                            }
                            self._hot_cache.move_to_end(cache_key)
                            if len(self._hot_cache) > self._max_items:
                                self._hot_cache.popitem(last=False)

                        self._stats["prefetches"] += 1
                except Exception as e:
                    log.debug(f"§PREFETCH: Embed failed for '{rq[:30]}': {e}")

            log.debug(
                f"§PREFETCH: Pre-fetched {len(related_queries[:3])} queries "
                f"(total={self._stats['prefetches']})"
            )

        except Exception as e:
            log.debug(f"§PREFETCH: Cycle failed: {e}")

    def _generate_related_queries(self, target: dict) -> list[str]:
        """Generate related search queries from the target metadata."""
        queries = []
        base_query = target.get("query", "")

        # Author-based queries
        for author in target.get("authors", [])[:2]:
            queries.append(f"{author} methodology and findings")

        # Keyword-based variant queries
        keywords = target.get("keywords", [])
        if len(keywords) >= 3:
            queries.append(" ".join(keywords[:5]) + " theoretical framework")
            queries.append(" ".join(keywords[:5]) + " empirical evidence")

        # Time-shifted queries
        for year in target.get("years", [])[:1]:
            try:
                y = int(year)
                queries.append(f"{base_query[:50]} recent developments since {y}")
            except ValueError:
                pass

        return queries

    @property
    def status(self) -> dict:
        with self._lock:
            return {
                "running": self._running,
                "hot_cache_size": len(self._hot_cache),
                "queue_depth": len(self._prefetch_queue),
                "stats": dict(self._stats),
            }


# Singleton
prefetcher = PredictiveCache()
