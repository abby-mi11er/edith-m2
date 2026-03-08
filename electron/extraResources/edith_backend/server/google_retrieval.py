"""
Google File Search Retrieval Backend
====================================
§7.1: Unified client manager with lifecycle
§7.3: Real relevance scores from grounding_metadata.support
§7.4: Retry with exponential backoff (3 attempts)
§7.5: Source deduplication by (title, text_hash)
§7.7: Low-confidence marking for inline fallback
§7.9: top_k passed to system instruction
§7.10: Multi-store support (fan-out)
"""

import hashlib
import logging
import os
import time
from typing import Optional

log = logging.getLogger("edith.google_retrieval")

# Cache the file search client (created once on first use)
_FS_CLIENT = None
_client_lock = None

try:
    import threading
    _client_lock = threading.Lock()
except ImportError:
    pass


def _get_fs_client():
    """Get or create a GenAI client for file search (default API version)."""
    global _FS_CLIENT
    if _FS_CLIENT is not None:
        return _FS_CLIENT
    lock = _client_lock
    if lock:
        lock.acquire()
    try:
        if _FS_CLIENT is not None:
            return _FS_CLIENT
        from google import genai
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return None
        _FS_CLIENT = genai.Client(api_key=api_key)
        log.info("Created GenAI client for file search (default API version)")
        return _FS_CLIENT
    finally:
        if lock:
            lock.release()


def shutdown_retrieval_client():
    """Close the file search client on shutdown (§7.1)."""
    global _FS_CLIENT
    _FS_CLIENT = None
    log.info("File search client released")


def _dedup_sources(sources: list[dict]) -> list[dict]:
    """Deduplicate sources by (title, text_hash) (§7.5)."""
    seen = set()
    deduped = []
    for s in sources:
        title = s.get("meta", {}).get("title", "")
        text = s.get("text", "")
        key = (title, hashlib.md5(text.encode()).hexdigest()[:16])
        if key not in seen:
            seen.add(key)
            deduped.append(s)
    if len(deduped) < len(sources):
        log.info(f"Deduplication: {len(sources)} → {len(deduped)} sources")
    return deduped


def _retrieve_single_store(
    query: str,
    api_key: str,
    store_id: str,
    model: str = "gemini-2.5-flash",
    top_k: int = 15,
) -> list[dict]:
    """Retrieve from a single store with retry (§7.4)."""
    max_retries = 3

    for attempt in range(max_retries):
        try:
            return _retrieve_impl(query, api_key, store_id, model, top_k)
        except Exception as e:
            if attempt < max_retries - 1:
                wait = (2 ** attempt) * 0.5  # 0.5s, 1s, 2s
                log.warning(f"Google retrieval attempt {attempt + 1} failed: {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                log.error(f"Google File Search retrieval failed after {max_retries} attempts: {e}", exc_info=True)
                return []


def _retrieve_impl(
    query: str,
    api_key: str,
    store_id: str,
    model: str,
    top_k: int,
) -> list[dict]:
    """Core retrieval implementation."""
    from google.genai import types

    client = _get_fs_client()
    if client is None:
        log.error("No API key available for Google File Search")
        return []

    file_search_tool = types.Tool(
        file_search=types.FileSearch(
            file_search_store_names=[store_id],
        )
    )

    # §7.9: include top_k in system instruction
    cfg = types.GenerateContentConfig(
        tools=[file_search_tool],
        temperature=0.0,
        system_instruction=(
            "You are a research retrieval assistant. "
            "Search the file store for documents relevant to the user's query. "
            "Return the most relevant passages verbatim with their source information. "
            f"Include up to {top_k} relevant passages."
        ),
    )

    prompt = (
        f"Find all relevant passages about: {query}\n\n"
        "Return the relevant text passages with their source titles."
    )

    log.info(f"Google File Search: querying store={store_id} model={model} query={query[:80]}")

    resp = client.models.generate_content(
        model=model,
        contents=prompt,
        config=cfg,
    )

    sources = []

    if hasattr(resp, "candidates") and resp.candidates:
        candidate = resp.candidates[0]

        grounding_meta = getattr(candidate, "grounding_metadata", None)
        if grounding_meta:
            chunks = getattr(grounding_meta, "grounding_chunks", []) or []

            # §7.3: try to extract real support scores
            supports = getattr(grounding_meta, "grounding_supports", []) or []
            support_scores = {}
            for sup in supports:
                for idx_obj in getattr(sup, "grounding_chunk_indices", []) or []:
                    idx = idx_obj if isinstance(idx_obj, int) else getattr(idx_obj, "index", -1)
                    conf = getattr(sup, "confidence_scores", [None])
                    if conf and len(conf) > 0:
                        score_val = conf[0] if isinstance(conf[0], (int, float)) else 0.8
                        support_scores[idx] = max(support_scores.get(idx, 0), score_val)

            for i, chunk in enumerate(chunks):
                text = ""
                title = ""
                uri = ""

                if hasattr(chunk, "retrieved_context"):
                    ctx = chunk.retrieved_context
                    text = getattr(ctx, "text", "") or ""
                    title = getattr(ctx, "title", "") or ""
                    uri = getattr(ctx, "uri", "") or ""
                elif hasattr(chunk, "web"):
                    web = chunk.web
                    text = getattr(web, "title", "") or ""
                    uri = getattr(web, "uri", "") or ""

                if text:
                    # §7.3: use real score if available, otherwise ordered decay
                    score = support_scores.get(i, 1.0 - (i * 0.05))
                    sources.append({
                        "text": text,
                        "meta": {
                            "title": title or f"Source {i + 1}",
                            "source": uri or title or f"google_file_search_{i}",
                            "sha256": "",
                            "chunk_index": i,
                        },
                        "score": round(score, 3),
                    })

            log.info(f"Google File Search: {len(sources)} grounding chunks for query: {query[:80]}")

        # §7.7: inline text fallback marked as low-confidence
        if not sources and hasattr(candidate, "content"):
            text = ""
            for part in (candidate.content.parts or []):
                if hasattr(part, "text") and part.text:
                    text += part.text

            if text and len(text) > 50:
                sources.append({
                    "text": text,
                    "meta": {
                        "title": "Google File Search Result",
                        "source": "google_file_search",
                        "sha256": "",
                        "chunk_index": 0,
                        "confidence": "low",  # §7.7
                    },
                    "score": 0.5,  # lower score for inline fallback
                })
                log.info(f"Google File Search: using inline response (low-confidence) ({len(text)} chars)")

    if not sources:
        log.warning(f"Google File Search: no sources found for query: {query[:80]}")

    return sources[:top_k]


def retrieve_google_sources(
    query: str,
    api_key: str,
    store_id: str,
    model: str = "gemini-2.5-flash",
    top_k: int = 15,
) -> list[dict]:
    """Retrieve relevant sources from Google File Search store.

    §7.4: Automatic retry with exponential backoff.
    §7.5: Source deduplication.
    §7.10: Multi-store support if store_id contains comma-separated IDs.
    """
    # §7.10: multi-store fan-out
    store_ids = [s.strip() for s in store_id.split(",") if s.strip()]
    if len(store_ids) <= 1:
        sources = _retrieve_single_store(query, api_key, store_id, model, top_k)
    else:
        sources = []
        per_store_k = max(5, top_k // len(store_ids))
        for sid in store_ids:
            sources.extend(_retrieve_single_store(query, api_key, sid, model, per_store_k))
        log.info(f"Multi-store retrieval: {len(store_ids)} stores → {len(sources)} total sources")

    # §7.5: deduplicate
    sources = _dedup_sources(sources)

    return sources[:top_k]


def google_retrieval_available(api_key: str, store_id: str) -> bool:
    """Check if Google File Search retrieval is configured and available."""
    return bool(api_key) and bool(store_id) and "your_store_id" not in store_id
