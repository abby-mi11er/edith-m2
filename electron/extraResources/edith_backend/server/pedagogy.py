"""
Pedagogy Engine — Advanced Learning Features
==============================================
§3.4: Quiz Generation — auto-create flashcards from reading
§3.9: Interactive Citations — instant source lookup
§3.7: Comparative Summaries — side-by-side diff generation
§1.3: Semantic De-duplication — merge similar chunks

Exposed via:
  POST /api/pedagogy/quiz     — generate quiz from sources
  POST /api/pedagogy/compare  — comparative summary
  GET  /api/pedagogy/cite     — instant citation lookup
  POST /api/dedup/scan        — scan for duplicate concepts
"""

import hashlib
import json
import logging
import os
import re
import time
from collections import defaultdict
from typing import Optional

log = logging.getLogger("edith.pedagogy")


# ═══════════════════════════════════════════════════════════════════
# §3.4: Quiz Generation — auto-create Anki-style flashcards
# ═══════════════════════════════════════════════════════════════════

_QUIZ_SYSTEM = (
    "You are a quiz generator for a political science PhD student.\n"
    "Given a research source, create exactly 5 flashcards.\n"
    "Each flashcard must have:\n"
    "- FRONT: A specific question about the source\n"
    "- BACK: The precise answer, with the page reference\n"
    "- TYPE: one of [definition, methodology, finding, theory, comparison]\n\n"
    "Output as JSON array: [{\"front\": ..., \"back\": ..., \"type\": ...}]\n"
    "Focus on concepts the student is most likely to be tested on."
)


def generate_quiz(
    sources: list[dict],
    difficulty: str = "intermediate",
    num_cards: int = 5,
) -> dict:
    """Generate flashcards from reading sources.

    Difficulty: intro | intermediate | advanced | doctoral
    """
    if not sources:
        return {"cards": [], "error": "No sources provided"}

    # Combine source texts
    combined = ""
    for i, src in enumerate(sources[:5]):  # Cap at 5 sources
        text = src.get("text", "") or src.get("content", "")
        author = src.get("metadata", {}).get("author", f"Source {i+1}")
        combined += f"\n--- {author} ---\n{text[:1500]}\n"

    difficulty_mod = {
        "intro": "Use simple language. Focus on key definitions and facts.",
        "intermediate": "Include methodology questions and theoretical connections.",
        "advanced": "Test causal mechanisms, methodological critiques, debates.",
        "doctoral": "Ask about identification strategies, external validity, "
                    "and how this connects to the broader literature.",
    }.get(difficulty, "")

    try:
        from server.backend_logic import generate_text_via_chain
        model_chain = [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]

        prompt = (
            f"Generate {num_cards} flashcards from this reading:\n\n"
            f"{combined}\n\n"
            f"Difficulty: {difficulty}. {difficulty_mod}\n"
            f"Output ONLY valid JSON array."
        )

        result, model = generate_text_via_chain(
            prompt, model_chain, system_instruction=_QUIZ_SYSTEM, temperature=0.3,
        )

        # Parse JSON from response
        json_match = re.search(r'\[.*\]', result, re.DOTALL)
        if json_match:
            cards = json.loads(json_match.group())
        else:
            cards = [{"front": "Parse error", "back": result[:200], "type": "error"}]

        return {
            "cards": cards[:num_cards],
            "difficulty": difficulty,
            "source_count": len(sources),
            "model": model,
        }
    except Exception as e:
        log.error(f"§QUIZ: Generation failed: {e}")
        return {"cards": [], "error": str(e)}


def export_to_anki(cards: list[dict], output_path: str = "") -> dict:
    """Export flashcards to Anki-compatible CSV format."""
    if not output_path:
        output_path = os.path.join(
            os.environ.get("EDITH_DATA_ROOT", "."),
            f"anki_export_{int(time.time())}.csv"
        )

    lines = []
    for card in cards:
        front = card.get("front", "").replace('"', '""')
        back = card.get("back", "").replace('"', '""')
        tag = card.get("type", "general")
        lines.append(f'"{front}","{back}","edith::{tag}"')

    try:
        with open(output_path, "w") as f:
            f.write("\n".join(lines))
        return {"status": "exported", "path": output_path, "cards": len(cards)}
    except Exception as e:
        return {"status": "failed", "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# §3.9: Interactive Citations — instant source lookup
# ═══════════════════════════════════════════════════════════════════

def lookup_citation(
    citation_text: str,
    chroma_dir: str = "",
    collection_name: str = "",
    embed_model: str = "",
) -> dict:
    """Instantly find the original source for a citation.

    Given 'Mettler (2011)' or 'the submerged state thesis', find the
    exact source chunk on the Bolt within 100ms.
    """
    t0 = time.time()

    # Parse citation
    author_year = re.match(r'(\w[\w\s]+?)\s*\((\d{4})\)', citation_text)
    if author_year:
        author = author_year.group(1).strip()
        year = author_year.group(2)
        query = f"{author} {year}"
    else:
        query = citation_text

    try:
        from server.chroma_backend import retrieve_local_sources
        sources = retrieve_local_sources(
            queries=[query],
            chroma_dir=chroma_dir or os.environ.get("EDITH_CHROMA_DIR", ""),
            collection_name=collection_name or os.environ.get("EDITH_COLLECTION", "edith"),
            embed_model=embed_model or os.environ.get("EDITH_EMBED_MODEL", ""),
            top_k=3,
            min_score=0.0,
        )

        elapsed = time.time() - t0
        return {
            "query": citation_text,
            "found": len(sources) > 0,
            "sources": sources[:3],
            "elapsed_ms": round(elapsed * 1000, 1),
            "fast_enough": elapsed < 0.1,
        }
    except Exception as e:
        return {"query": citation_text, "found": False, "error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# §1.3: Semantic De-duplication — merge similar chunks
# ═══════════════════════════════════════════════════════════════════

def _compute_text_hash(text: str) -> str:
    """Compute a normalized hash for deduplication."""
    normalized = re.sub(r'\s+', ' ', text.lower().strip())
    return hashlib.md5(normalized.encode()).hexdigest()


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity between two texts."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def scan_for_duplicates(
    sources: list[dict],
    similarity_threshold: float = 0.75,
) -> dict:
    """Scan sources for semantic duplicates.

    Returns clusters of near-duplicate chunks that could be merged.
    """
    t0 = time.time()

    # Phase 1: Exact hash duplicates
    hash_groups = defaultdict(list)
    for i, src in enumerate(sources):
        text = src.get("text", "") or src.get("content", "")
        h = _compute_text_hash(text)
        hash_groups[h].append(i)

    exact_dupes = {h: indices for h, indices in hash_groups.items()
                   if len(indices) > 1}

    # Phase 2: Near-duplicate detection (Jaccard similarity)
    # Only compare within a manageable window
    near_dupes = []
    texts = [(src.get("text", "") or src.get("content", "")) for src in sources]
    checked = set()

    for i in range(min(len(texts), 500)):
        for j in range(i + 1, min(len(texts), i + 20)):  # Window of 20
            pair = (i, j)
            if pair in checked:
                continue
            checked.add(pair)

            sim = _jaccard_similarity(texts[i], texts[j])
            if sim >= similarity_threshold:
                near_dupes.append({
                    "chunk_a": i,
                    "chunk_b": j,
                    "similarity": round(sim, 3),
                    "preview_a": texts[i][:80],
                    "preview_b": texts[j][:80],
                })

    # Summary
    total_dupes = len(exact_dupes) + len(near_dupes)
    savings_pct = round(total_dupes / max(len(sources), 1) * 100, 1)

    return {
        "total_chunks": len(sources),
        "exact_duplicates": len(exact_dupes),
        "near_duplicates": len(near_dupes),
        "near_duplicate_pairs": near_dupes[:20],  # Cap output
        "estimated_savings_pct": savings_pct,
        "elapsed": round(time.time() - t0, 2),
        "recommendation": (
            f"Found {total_dupes} duplicate clusters. "
            f"Merging could save ~{savings_pct}% storage."
            if total_dupes > 0 else
            "No significant duplicates found."
        ),
    }


def merge_duplicates(
    sources: list[dict],
    duplicate_pairs: list[dict],
) -> list[dict]:
    """Merge identified duplicate chunks into Master Concepts.

    Keeps the longer/richer version of each duplicate pair.
    """
    to_remove = set()
    for pair in duplicate_pairs:
        a, b = pair["chunk_a"], pair["chunk_b"]
        text_a = sources[a].get("text", "") or sources[a].get("content", "")
        text_b = sources[b].get("text", "") or sources[b].get("content", "")
        # Keep the longer one
        if len(text_a) >= len(text_b):
            to_remove.add(b)
        else:
            to_remove.add(a)

    merged = [s for i, s in enumerate(sources) if i not in to_remove]
    return merged


# ═══════════════════════════════════════════════════════════════════
# §5.9: Graceful Degradation — capability tiers based on packages
# ═══════════════════════════════════════════════════════════════════

def get_capability_tier() -> dict:
    """Detect which features are available based on installed packages."""
    tiers = {
        "core": True,  # Always available
        "embeddings": False,
        "vibe_coding": False,
        "plotting": False,
        "statistics": False,
        "local_inference": False,
        "neural_engine": False,
    }

    # Check packages
    try:
        import sentence_transformers
        tiers["embeddings"] = True
    except ImportError:
        pass

    try:
        import pandas
        tiers["vibe_coding"] = True
    except ImportError:
        pass

    try:
        import matplotlib
        tiers["plotting"] = True
    except ImportError:
        pass

    try:
        import statsmodels
        tiers["statistics"] = True
    except ImportError:
        pass

    try:
        import mlx
        tiers["local_inference"] = True
    except ImportError:
        pass

    try:
        import onnxruntime
        providers = onnxruntime.get_available_providers()
        tiers["neural_engine"] = "CoreMLExecutionProvider" in providers
    except ImportError:
        pass

    # Compute overall tier
    available = sum(1 for v in tiers.values() if v)
    total = len(tiers)
    if available == total:
        tier_name = "FULL"
    elif available >= total * 0.7:
        tier_name = "HIGH"
    elif available >= total * 0.4:
        tier_name = "STANDARD"
    else:
        tier_name = "BASIC"

    missing = [k for k, v in tiers.items() if not v]

    return {
        "tier": tier_name,
        "capabilities": tiers,
        "available": available,
        "total": total,
        "missing": missing,
        "install_hints": {
            "embeddings": "pip install sentence-transformers onnxruntime",
            "vibe_coding": "pip install pandas",
            "plotting": "pip install matplotlib seaborn",
            "statistics": "pip install statsmodels",
            "local_inference": "pip install mlx mlx-lm",
        },
    }

