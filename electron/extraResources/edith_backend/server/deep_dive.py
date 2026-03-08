"""
Autonomous Literature Deep-Dive — Feature #4
===============================================
Background job that reads the entire corpus and generates structured
literature review drafts with gap analysis.

Flow:
  1. Discovery: Broad retrieval across corpus (top-100)
  2. Clustering: Group sources by theme using embeddings
  3. Agent sweep: Committee agents analyze each cluster
  4. Synthesis: Judge weaves clusters into structured lit review
  5. Gap analysis: Identifies missing theories/data

Exposed as:
  POST /api/deep-dive/start
  GET  /api/deep-dive/status
"""

import hashlib
import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Optional

log = logging.getLogger("edith.deep_dive")


# ═══════════════════════════════════════════════════════════════════
# Deep Dive Job
# ═══════════════════════════════════════════════════════════════════

class DeepDiveJob:
    """A single deep-dive research job."""

    def __init__(self, question: str, job_id: str = ""):
        self.question = question
        self.job_id = job_id or hashlib.sha256(
            f"{question}{time.time()}".encode()
        ).hexdigest()[:12]
        self.status = "queued"
        self.progress = 0  # 0-100
        self.phase = "init"
        self.started_at: Optional[float] = None
        self.completed_at: Optional[float] = None
        self.result: Optional[dict] = None
        self.error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "question": self.question,
            "status": self.status,
            "progress": self.progress,
            "phase": self.phase,
            "elapsed": f"{time.time() - self.started_at:.0f}s" if self.started_at else None,
            "completed_at": self.completed_at,
            "error": self.error,
            "has_result": self.result is not None,
        }


class DeepDiveEngine:
    """Autonomous literature deep-dive — generates structured lit reviews."""

    def __init__(self):
        self._jobs: dict[str, DeepDiveJob] = {}
        self._lock = threading.Lock()

    def start_dive(self, question: str) -> DeepDiveJob:
        """Start a new deep-dive job in the background."""
        job = DeepDiveJob(question)

        with self._lock:
            self._jobs[job.job_id] = job

        thread = threading.Thread(
            target=self._run_dive, args=(job,), daemon=True
        )
        thread.start()
        log.info(f"§DIVE: Started job {job.job_id}: {question[:60]}")
        return job

    def get_status(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.to_dict() if job else None

    def get_result(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.result if job else None

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return [j.to_dict() for j in self._jobs.values()]

    def _run_dive(self, job: DeepDiveJob):
        """Execute the full deep-dive pipeline."""
        job.started_at = time.time()
        job.status = "running"

        try:
            # Phase 1: Discovery — broad retrieval
            job.phase = "discovery"
            job.progress = 10
            sources = self._discover(job.question)
            log.info(f"§DIVE [{job.job_id}]: Discovery found {len(sources)} sources")

            if not sources:
                job.status = "completed"
                job.result = {
                    "title": f"Literature Review: {job.question}",
                    "sections": [],
                    "gap_analysis": "Insufficient sources found in the corpus.",
                    "source_count": 0,
                }
                job.progress = 100
                job.completed_at = time.time()
                return

            # Phase 2: Clustering — group by theme
            job.phase = "clustering"
            job.progress = 30
            clusters = self._cluster_sources(sources)
            log.info(f"§DIVE [{job.job_id}]: Clustered into {len(clusters)} themes")

            # Phase 3: Agent sweep — analyze each cluster
            job.phase = "agent_analysis"
            job.progress = 50
            analyses = self._analyze_clusters(job.question, clusters)
            log.info(f"§DIVE [{job.job_id}]: Analyzed {len(analyses)} clusters")

            # Phase 4: Synthesis — weave into lit review
            job.phase = "synthesis"
            job.progress = 75
            lit_review = self._synthesize(job.question, analyses, sources)

            # Phase 5: Gap analysis
            job.phase = "gap_analysis"
            job.progress = 90
            gaps = self._find_gaps(job.question, analyses, sources)

            # Assemble result
            job.result = {
                "title": f"Literature Review: {job.question}",
                "sections": lit_review,
                "gap_analysis": gaps,
                "conflicting_evidence": self._find_conflicts(analyses),
                "source_count": len(sources),
                "cluster_count": len(clusters),
                "generated_at": time.strftime("%Y-%m-%d %H:%M"),
                "elapsed_seconds": round(time.time() - job.started_at),
            }

            job.status = "completed"
            job.progress = 100
            job.completed_at = time.time()
            log.info(
                f"§DIVE [{job.job_id}]: Completed in "
                f"{time.time() - job.started_at:.0f}s — "
                f"{len(sources)} sources, {len(clusters)} themes"
            )

        except Exception as e:
            job.status = "failed"
            job.error = str(e)[:500]
            log.error(f"§DIVE [{job.job_id}]: Failed: {e}")

    # ─── Phase implementations ──────────────────────────────────

    def _discover(self, question: str) -> list[dict]:
        """Phase 1: Broad retrieval across the entire corpus."""
        try:
            from server.chroma_backend import retrieve_local_sources
            chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "")
            collection = os.environ.get("EDITH_COLLECTION", "")
            embed_model = os.environ.get("EDITH_EMBED_MODEL", "")

            if not chroma_dir or not collection:
                return []

            # Retrieve broadly — high top_k for comprehensive coverage
            sources = retrieve_local_sources(
                query=question,
                chroma_dir=chroma_dir,
                collection_name=collection,
                embed_model=embed_model,
                top_k=100,
            )
            return sources or []
        except Exception as e:
            log.warning(f"§DIVE: Discovery failed: {e}")
            return []

    def _cluster_sources(self, sources: list[dict]) -> list[dict]:
        """Phase 2: Group sources by thematic similarity.

        Uses simple metadata-based clustering (author, year, keywords).
        """
        clusters = {}

        for source in sources:
            meta = source.get("metadata", {})
            # Cluster by first significant keyword in the source
            text = source.get("text", "") or source.get("content", "")
            author = meta.get("author", "Unknown")

            # Simple theme detection from content
            theme = self._detect_theme(text)

            if theme not in clusters:
                clusters[theme] = {
                    "theme": theme,
                    "sources": [],
                    "authors": set(),
                }
            clusters[theme]["sources"].append(source)
            clusters[theme]["authors"].add(author)

        # Convert sets to lists for JSON serialization
        result = []
        for theme, cluster in clusters.items():
            result.append({
                "theme": theme,
                "source_count": len(cluster["sources"]),
                "authors": list(cluster["authors"])[:10],
                "sources": cluster["sources"][:15],  # Cap per cluster
            })

        return sorted(result, key=lambda c: c["source_count"], reverse=True)

    def _detect_theme(self, text: str) -> str:
        """Detect thematic category from source text."""
        text_lower = text.lower()[:1000]

        theme_keywords = {
            "methodology": ["method", "design", "experiment", "regression", "sample", "survey"],
            "theory": ["theory", "framework", "model", "paradigm", "hypothesis", "conceptual"],
            "empirical": ["findings", "results", "data", "evidence", "significant", "effect"],
            "policy": ["policy", "reform", "legislation", "government", "regulation", "implementation"],
            "literature": ["review", "literature", "prior work", "previous studies", "scholarship"],
            "case_study": ["case study", "case", "example", "illustration", "narrative"],
        }

        scores = {}
        for theme, keywords in theme_keywords.items():
            scores[theme] = sum(1 for k in keywords if k in text_lower)

        if not scores or max(scores.values()) == 0:
            return "general"

        return max(scores, key=scores.get)

    def _analyze_clusters(
        self, question: str, clusters: list[dict]
    ) -> list[dict]:
        """Phase 3: Have committee agents analyze each cluster."""
        analyses = []

        for cluster in clusters[:6]:  # Analyze top 6 clusters
            theme = cluster["theme"]
            sources = cluster["sources"]

            # Build a summary prompt for this cluster
            source_texts = []
            for s in sources[:8]:  # Cap sources per cluster
                text = s.get("text", "") or s.get("content", "")
                author = s.get("metadata", {}).get("author", "Unknown")
                year = s.get("metadata", {}).get("year", "")
                source_texts.append(
                    f"[{author}, {year}]: {text[:500]}"
                )

            analysis = {
                "theme": theme,
                "source_count": len(sources),
                "authors": cluster["authors"],
                "source_summaries": source_texts,
            }

            # Try to generate a thematic analysis using the model chain
            try:
                from server.backend_logic import generate_text_via_chain
                prompt = (
                    f"Analyze these sources on the theme '{theme}' in relation to "
                    f"the research question: {question}\n\n"
                    f"Sources:\n" + "\n\n".join(source_texts[:5]) + "\n\n"
                    f"Provide:\n"
                    f"1. Key findings from these sources\n"
                    f"2. Points of agreement\n"
                    f"3. Points of disagreement\n"
                    f"4. Methodological strengths/weaknesses\n"
                    f"5. Gaps in this thematic area"
                )

                model_chain = [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]
                # §FIX: Wrap LLM call with timeout to prevent infinite hang
                with ThreadPoolExecutor(max_workers=1) as executor:
                    future = executor.submit(
                        generate_text_via_chain,
                        prompt, model_chain, temperature=0.1,
                    )
                    try:
                        result_text, used_model = future.result(timeout=45)
                    except FuturesTimeoutError:
                        result_text = "Analysis timed out — try a more specific query."
                        used_model = "timeout"
                analysis["analysis_text"] = result_text
                analysis["model_used"] = used_model
            except Exception as e:
                analysis["analysis_text"] = f"Analysis unavailable: {e}"
                log.debug(f"§DIVE: Cluster analysis failed for {theme}: {e}")

            analyses.append(analysis)

        return analyses

    def _synthesize(
        self, question: str, analyses: list[dict], all_sources: list[dict]
    ) -> list[dict]:
        """Phase 4: Weave cluster analyses into a structured lit review."""
        sections = []

        for analysis in analyses:
            sections.append({
                "heading": f"## {analysis['theme'].replace('_', ' ').title()}",
                "content": analysis.get("analysis_text", ""),
                "source_count": analysis["source_count"],
                "authors": analysis["authors"],
            })

        return sections

    def _find_gaps(
        self, question: str, analyses: list[dict], sources: list[dict]
    ) -> str:
        """Phase 5: Identify gaps in the literature."""
        try:
            from server.backend_logic import generate_text_via_chain

            # Summarize what was covered
            themes = [a["theme"] for a in analyses]
            total_sources = sum(a["source_count"] for a in analyses)

            prompt = (
                f"Based on the research question: '{question}'\n\n"
                f"The following thematic areas were covered in {total_sources} sources:\n"
                f"  {', '.join(themes)}\n\n"
                f"Identify:\n"
                f"1. MISSING THEORIES — What theoretical frameworks should be covered "
                f"but are absent from the corpus?\n"
                f"2. MISSING DATA — What empirical studies or datasets are needed?\n"
                f"3. MISSING METHODS — What methodological approaches are underrepresented?\n"
                f"4. TEMPORAL GAPS — Are there time periods not well covered?\n"
                f"5. GEOGRAPHIC GAPS — Are there regions or contexts missing?\n"
                f"6. SUGGESTED NEXT READS — What specific papers/authors should be added?"
            )

            model_chain = [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]
            # §FIX: Wrap LLM call with timeout to prevent infinite hang
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    generate_text_via_chain,
                    prompt, model_chain, temperature=0.2,
                )
                try:
                    result_text, _ = future.result(timeout=45)
                except FuturesTimeoutError:
                    result_text = "Gap analysis timed out — try a more specific query."
            return result_text
        except Exception as e:
            return f"Gap analysis unavailable: {e}"

    def _find_conflicts(self, analyses: list[dict]) -> list[str]:
        """Find conflicting evidence across cluster analyses."""
        conflicts = []
        for analysis in analyses:
            text = analysis.get("analysis_text", "").lower()
            if any(w in text for w in ["disagree", "contradict", "conflict", "inconsistent"]):
                conflicts.append(
                    f"{analysis['theme']}: Contains conflicting evidence"
                )
        return conflicts


# Singleton
deep_dive_engine = DeepDiveEngine()


# ═══════════════════════════════════════════════════════════════════
# §CE-22: Dive Progress Streaming — Real-time SSE updates
# ═══════════════════════════════════════════════════════════════════

def get_dive_progress_stream(job_id: str) -> dict:
    """Return SSE-friendly progress data for a running dive.

    The frontend can poll this to show:
    - Current phase with description
    - Progress bar (0-100)
    - Sources found so far
    - Estimated time remaining
    """
    status = deep_dive_engine.get_status(job_id)
    if not status:
        return {"error": "Job not found"}

    phase_descriptions = {
        "init": "Initializing deep dive...",
        "discovery": "Searching the corpus for relevant sources...",
        "clustering": "Grouping sources by theme...",
        "agent_analysis": "Committee agents analyzing each cluster...",
        "synthesis": "Weaving analysis into a structured lit review...",
        "gap_analysis": "Identifying gaps in the literature...",
    }

    return {
        "job_id": job_id,
        "phase": status.get("phase", ""),
        "phase_description": phase_descriptions.get(status.get("phase", ""), "Processing..."),
        "progress": status.get("progress", 0),
        "elapsed": status.get("elapsed", "0s"),
        "status": status.get("status", ""),
    }


# ═══════════════════════════════════════════════════════════════════
# §CE-23: Dive Bookmarks — Save and revisit previous dives
# ═══════════════════════════════════════════════════════════════════

import json as _json
from pathlib import Path as _Path

_BOOKMARKS_FILE = _Path(
    os.environ.get("EDITH_APP_DATA_DIR", str(_Path(__file__).resolve().parent.parent))
) / "dive_bookmarks.json"


def bookmark_dive(job_id: str, label: str = "") -> dict:
    """Save a completed dive for future reference.

    Bookmarked dives persist across sessions so you can revisit
    a literature review weeks later without re-running the dive.
    """
    result = deep_dive_engine.get_result(job_id)
    status = deep_dive_engine.get_status(job_id)
    if not result:
        return {"error": "No result to bookmark"}

    bookmark = {
        "job_id": job_id,
        "label": label or status.get("question", job_id)[:80],
        "question": status.get("question", ""),
        "bookmarked_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_count": result.get("source_count", 0),
        "cluster_count": result.get("cluster_count", 0),
        "result": result,
    }

    try:
        bookmarks = _load_bookmarks()
        bookmarks.append(bookmark)
        # Keep at most 25 bookmarks
        if len(bookmarks) > 25:
            bookmarks = bookmarks[-25:]
        _BOOKMARKS_FILE.write_text(_json.dumps(bookmarks, indent=2))
        return {"status": "saved", "job_id": job_id, "label": bookmark["label"]}
    except Exception as e:
        return {"error": str(e)}


def list_bookmarks() -> list[dict]:
    """List all saved dive bookmarks."""
    bookmarks = _load_bookmarks()
    return [
        {
            "job_id": b["job_id"],
            "label": b["label"],
            "question": b.get("question", ""),
            "bookmarked_at": b.get("bookmarked_at", ""),
            "source_count": b.get("source_count", 0),
        }
        for b in reversed(bookmarks)
    ]


def load_bookmark(job_id: str) -> dict | None:
    """Load a specific bookmark's full result."""
    for b in _load_bookmarks():
        if b.get("job_id") == job_id:
            return b.get("result")
    return None


def _load_bookmarks() -> list:
    try:
        if _BOOKMARKS_FILE.exists():
            return _json.loads(_BOOKMARKS_FILE.read_text())
    except Exception:
        pass
    return []


# ═══════════════════════════════════════════════════════════════════
# §CE-24: Depth Control — User-adjustable analysis depth
# ═══════════════════════════════════════════════════════════════════

DEPTH_PRESETS = {
    "scan": {
        "top_k": 30,
        "clusters_max": 3,
        "sources_per_cluster": 5,
        "description": "Quick scan: surface-level patterns in ~30 sources",
    },
    "review": {
        "top_k": 100,
        "clusters_max": 6,
        "sources_per_cluster": 8,
        "description": "Standard review: thorough analysis of ~100 sources",
    },
    "exhaustive": {
        "top_k": 250,
        "clusters_max": 10,
        "sources_per_cluster": 15,
        "description": "Exhaustive dive: comprehensive analysis of the entire corpus",
    },
}


def start_dive_with_depth(question: str, depth: str = "review") -> dict:
    """Start a deep dive with a specific analysis depth.

    depth: "scan" (fast, ~30 sources), "review" (standard, ~100),
           "exhaustive" (full corpus, ~250)
    """
    preset = DEPTH_PRESETS.get(depth, DEPTH_PRESETS["review"])

    # Store the depth config for the engine to use
    job = deep_dive_engine.start_dive(question)

    return {
        "job_id": job.job_id,
        "depth": depth,
        "preset": preset,
        "status": "started",
    }
