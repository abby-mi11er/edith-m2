#!/usr/bin/env python3
"""
Discovery Mode — External Paper Search (Only When Directed)
============================================================
Winnie relies on the local corpus (26K files, 93K chunks) by default.
This module activates ONLY when the user explicitly asks for external
paper recommendations. Triggered by phrases like:
  - "recommend papers on..."
  - "find more papers about..."
  - "what else has been published on..."
  - "discover papers..."
  - "search for papers on..."

Uses Semantic Scholar (free) and OpenAlex (free) for discovery.
Optionally uses Google Scholar via SerpAPI (needs key).
"""

import json
import os
import re
import logging
import time
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.discovery")

# ---------------------------------------------------------------------------
# Discovery intent detection
# ---------------------------------------------------------------------------

DISCOVERY_TRIGGERS = [
    "recommend papers",
    "recommend articles",
    "recommend readings",
    "find papers",
    "find more papers",
    "find articles",
    "search for papers",
    "discover papers",
    "what else has been published",
    "what other papers",
    "what other research",
    "suggest readings",
    "suggest papers",
    "look for papers",
    "look outside",
    "external search",
    "search outside my files",
    "beyond my corpus",
    "explore the literature",
    "find recent papers",
    "new research on",
    "latest papers on",
    "who else has written on",
    "find similar papers",
]


def is_discovery_query(query: str) -> bool:
    """Check if the query is asking for external paper discovery."""
    q = query.lower()
    return any(trigger in q for trigger in DISCOVERY_TRIGGERS)


def extract_discovery_topic(query: str) -> str:
    """Extract the research topic from a discovery query."""
    q = query.lower()
    # Remove trigger phrases to get the topic
    for trigger in DISCOVERY_TRIGGERS:
        q = q.replace(trigger, "")
    # Clean up
    q = re.sub(r'^\s*(on|about|for|regarding|related to|in)\s+', '', q)
    q = q.strip(' ?.!')
    return q if q else query


# ---------------------------------------------------------------------------
# Multi-source discovery engine
# ---------------------------------------------------------------------------

class DiscoveryEngine:
    """Search multiple academic sources for papers the user doesn't have."""

    def __init__(self):
        self._s2 = None
        self._openalex = None
        self._serpapi_key = os.environ.get("SERPAPI_KEY", "")

    @property
    def s2(self):
        if self._s2 is None:
            from pipelines.connectors import SemanticScholarConnector
            self._s2 = SemanticScholarConnector(
                api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
            )
        return self._s2

    def search(self, topic: str, max_results: int = 15) -> dict:
        """Search multiple sources and merge results."""
        results = {
            "topic": topic,
            "papers": [],
            "sources_searched": [],
        }

        # Semantic Scholar (always available, free)
        try:
            s2_results = self.s2.search_paper(topic, limit=max_results)
            for r in s2_results:
                results["papers"].append({
                    "title": r.get("title", ""),
                    "authors": [a.get("name", "") for a in r.get("authors", [])],
                    "year": r.get("year"),
                    "abstract": r.get("abstract", ""),
                    "citations": r.get("citationCount", 0),
                    "open_access": r.get("isOpenAccess", False),
                    "fields": r.get("fieldsOfStudy", []),
                    "s2_id": r.get("paperId", ""),
                    "source": "Semantic Scholar",
                })
            results["sources_searched"].append("Semantic Scholar")
        except Exception as e:
            log.warning(f"Semantic Scholar search failed: {e}")

        # OpenAlex (free, no auth)
        try:
            oa_results = self._search_openalex(topic, limit=max_results)
            for r in oa_results:
                # Deduplicate by title similarity
                if not any(self._similar_title(r["title"], p["title"])
                          for p in results["papers"]):
                    results["papers"].append(r)
            results["sources_searched"].append("OpenAlex")
        except Exception as e:
            log.warning(f"OpenAlex search failed: {e}")

        # Google Scholar via SerpAPI (optional, needs key)
        if self._serpapi_key:
            try:
                gs_results = self._search_google_scholar(topic, limit=max_results)
                for r in gs_results:
                    if not any(self._similar_title(r["title"], p["title"])
                              for p in results["papers"]):
                        results["papers"].append(r)
                results["sources_searched"].append("Google Scholar")
            except Exception as e:
                log.warning(f"Google Scholar search failed: {e}")

        # Sort by citation count
        results["papers"].sort(key=lambda p: p.get("citations", 0), reverse=True)
        results["total_found"] = len(results["papers"])

        return results

    def _search_openalex(self, topic: str, limit: int = 10) -> list:
        """Search OpenAlex for papers (free, no auth)."""
        import urllib.request
        import urllib.parse

        q = urllib.parse.quote(topic)
        url = (
            f"https://api.openalex.org/works?"
            f"search={q}&per_page={limit}"
            f"&sort=cited_by_count:desc"
            f"&filter=type:article"
            f"&mailto=edith-assistant@example.com"
        )

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            return []

        papers = []
        for work in data.get("results", []):
            papers.append({
                "title": work.get("title", ""),
                "authors": [
                    a.get("author", {}).get("display_name", "")
                    for a in work.get("authorships", [])[:5]
                ],
                "year": work.get("publication_year"),
                "abstract": self._reconstruct_abstract(work.get("abstract_inverted_index", {})),
                "citations": work.get("cited_by_count", 0),
                "open_access": work.get("open_access", {}).get("is_oa", False),
                "doi": work.get("doi", ""),
                "source": "OpenAlex",
            })
        return papers

    def _reconstruct_abstract(self, inverted_index: dict) -> str:
        """Reconstruct abstract from OpenAlex's inverted index format."""
        if not inverted_index:
            return ""
        word_positions = []
        for word, positions in inverted_index.items():
            for pos in positions:
                word_positions.append((pos, word))
        word_positions.sort()
        return " ".join(w for _, w in word_positions)[:500]

    def _search_google_scholar(self, topic: str, limit: int = 10) -> list:
        """Search Google Scholar via SerpAPI (needs SERPAPI_KEY)."""
        import urllib.request
        import urllib.parse

        params = urllib.parse.urlencode({
            "api_key": self._serpapi_key,
            "engine": "google_scholar",
            "q": topic,
            "num": limit,
        })
        url = f"https://serpapi.com/search?{params}"

        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except Exception:
            return []

        papers = []
        for r in data.get("organic_results", []):
            info = r.get("publication_info", {})
            papers.append({
                "title": r.get("title", ""),
                "authors": info.get("authors", []),
                "year": self._extract_year(info.get("summary", "")),
                "abstract": r.get("snippet", ""),
                "citations": r.get("inline_links", {}).get("cited_by", {}).get("total", 0),
                "link": r.get("link", ""),
                "source": "Google Scholar",
            })
        return papers

    def _extract_year(self, text: str) -> Optional[int]:
        """Extract year from citation summary."""
        match = re.search(r'\b(19|20)\d{2}\b', text)
        return int(match.group()) if match else None

    def _similar_title(self, a: str, b: str) -> bool:
        """Check if two titles are similar enough to deduplicate."""
        a_clean = re.sub(r'[^a-z0-9\s]', '', a.lower()).split()
        b_clean = re.sub(r'[^a-z0-9\s]', '', b.lower()).split()
        if not a_clean or not b_clean:
            return False
        overlap = set(a_clean) & set(b_clean)
        return len(overlap) / max(len(a_clean), len(b_clean)) > 0.6

    def format_for_winnie(self, results: dict) -> str:
        """Format discovery results as context Winnie can use in her answer."""
        if not results.get("papers"):
            return ""

        lines = [
            f"\n📚 EXTERNAL DISCOVERY — Papers not in your corpus on '{results['topic']}'",
            f"   (Searched: {', '.join(results['sources_searched'])})\n",
        ]

        for i, p in enumerate(results["papers"][:10], 1):
            authors = ", ".join(p.get("authors", [])[:3])
            year = p.get("year", "?")
            cites = p.get("citations", 0)
            oa = "🔓" if p.get("open_access") else "🔒"

            lines.append(f"  {i}. {oa} [{year}] {p['title']}")
            lines.append(f"     {authors}")
            lines.append(f"     Cited: {cites:,}")
            if p.get("abstract"):
                lines.append(f"     → {p['abstract'][:150]}...")
            lines.append("")

        lines.append(
            "  💡 To add any of these to your corpus, download the PDF and "
            "place it in your Edith data folder."
        )
        return "\n".join(lines)


# Singleton
_engine = None

def get_discovery_engine() -> DiscoveryEngine:
    global _engine
    if _engine is None:
        _engine = DiscoveryEngine()
    return _engine


# ═══════════════════════════════════════════════════════════════════
# TITAN §6: SCHOLARLY PULSE — Overnight Preprint Watcher
# ═══════════════════════════════════════════════════════════════════


class ScholarlyPulseMonitor:
    """Autonomous overnight literature monitor — the "Scholarly Pulse."

    While you sleep, Winnie watches for:
    - New preprints on your research topics
    - New papers citing your key references
    - Methodology updates in your fields

    Results saved to {EDITH_DATA_ROOT}/overnight/new_ground.json
    for integration with the Daily Brief "New Ground" section.

    Usage:
        pulse = ScholarlyPulseMonitor()
        results = pulse.run_overnight_scan()
    """

    DEFAULT_WATCH_TOPICS = [
        "SNAP enrollment rural",
        "submerged state welfare",
        "Mettler policy feedback",
        "rural clientelism Texas",
        "charity welfare substitute",
        "voter turnout poverty",
        "criminal governance state capacity",
    ]

    # Key references to monitor for new citations
    DEFAULT_KEY_REFS = [
        "The Submerged State",
        "Policy feedback and political participation",
        "Why Americans Hate Welfare",
        "Criminal governance Latin America",
    ]

    def __init__(self, data_root: str = ""):
        self._root = data_root or os.environ.get("EDITH_DATA_ROOT", ".")
        self._output_dir = os.path.join(self._root, "overnight")
        self._state_path = os.path.join(self._output_dir, "pulse_state.json")
        self._state = self._load_state()
        self._engine = get_discovery_engine()

    def _load_state(self) -> dict:
        if os.path.exists(self._state_path):
            try:
                with open(self._state_path) as f:
                    return json.load(f)
            except Exception:
                pass
        return {"last_scan": None, "seen_titles": []}

    def _save_state(self):
        os.makedirs(self._output_dir, exist_ok=True)
        with open(self._state_path, "w") as f:
            json.dump(self._state, f, indent=2)

    def run_overnight_scan(
        self,
        topics: list[str] = None,
        max_per_topic: int = 5,
    ) -> dict:
        """Run an overnight scan for new papers on all watch topics.

        Queries OpenAlex (free, no auth) for each topic, filters out
        papers already seen, and saves new finds to new_ground.json.
        """
        topics = topics or self.DEFAULT_WATCH_TOPICS
        t0 = time.time()
        seen = set(self._state.get("seen_titles", []))

        new_papers = []
        for topic in topics:
            try:
                results = self._engine._search_openalex(topic, limit=max_per_topic)
                for paper in results:
                    title = paper.get("title", "")
                    if title and title not in seen:
                        paper["watch_topic"] = topic
                        new_papers.append(paper)
                        seen.add(title)
            except Exception as e:
                log.warning(f"§PULSE: Failed to scan '{topic}': {e}")

        # Update state
        self._state["last_scan"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._state["seen_titles"] = list(seen)[-500:]  # Keep last 500
        self._save_state()

        # Save results for morning brief
        output = {
            "scan_timestamp": self._state["last_scan"],
            "topics_scanned": len(topics),
            "new_papers_found": len(new_papers),
            "papers": new_papers,
        }

        output_path = os.path.join(self._output_dir, "new_ground.json")
        try:
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2, default=str)
        except Exception as e:
            log.error(f"§PULSE: Failed to save results: {e}")

        elapsed = time.time() - t0
        log.info(f"§PULSE: Overnight scan complete — {len(new_papers)} new papers "
                 f"across {len(topics)} topics in {elapsed:.1f}s")

        return output

    def check_citation_alerts(
        self,
        key_refs: list[str] = None,
    ) -> dict:
        """Check for new papers citing your key references.

        Uses OpenAlex to find papers that cite seminal works
        you're tracking.
        """
        key_refs = key_refs or self.DEFAULT_KEY_REFS
        alerts = []

        for ref in key_refs:
            try:
                results = self._engine._search_openalex(
                    f'cites:"{ref}"', limit=3
                )
                for paper in results:
                    year = paper.get("year")
                    if year and year >= 2024:
                        alerts.append({
                            "citing_paper": paper.get("title", ""),
                            "authors": paper.get("authors", []),
                            "year": year,
                            "cites": ref,
                        })
            except Exception:
                pass

        return {
            "references_monitored": len(key_refs),
            "new_citations_found": len(alerts),
            "alerts": alerts,
        }

    def format_for_daily_brief(self) -> dict:
        """Format overnight findings for the Daily Brief 'New Ground' section."""
        output_path = os.path.join(self._output_dir, "new_ground.json")
        if not os.path.exists(output_path):
            return {"new_ground": [], "summary": "No overnight scan results."}

        try:
            with open(output_path) as f:
                data = json.load(f)
        except Exception:
            return {"new_ground": [], "summary": "Could not read scan results."}

        papers = data.get("papers", [])[:5]
        brief_items = []
        for p in papers:
            brief_items.append({
                "title": p.get("title", ""),
                "authors": ", ".join(p.get("authors", [])[:2]),
                "year": p.get("year"),
                "topic": p.get("watch_topic", ""),
            })

        return {
            "new_ground": brief_items,
            "total_new": data.get("new_papers_found", 0),
            "last_scan": data.get("scan_timestamp", ""),
            "summary": (
                f"Found {data.get('new_papers_found', 0)} new papers "
                f"across {data.get('topics_scanned', 0)} topics."
            ),
        }


# ═══════════════════════════════════════════════════════════════════
# §CE-34: Serendipity Engine — Find papers you weren't looking for
# ═══════════════════════════════════════════════════════════════════

def serendipity_search(topic: str, surprise_factor: float = 0.3) -> list[dict]:
    """Search for papers that are ADJACENT to your topic but not directly about it.

    The surprise_factor (0-1) controls how far from the original topic
    to wander. 0 = exact match, 1 = maximum surprise.

    This is how breakthroughs happen: reading outside your lane.
    """
    import random

    # Adjacent fields for political science research
    adjacent_fields = {
        "accountability": ["organizational transparency", "audit society", "surveillance studies"],
        "welfare": ["basic income experiments", "care economy", "labor economics"],
        "federalism": ["comparative constitutional law", "EU multilevel governance", "municipal law"],
        "privatization": ["commons governance", "platform cooperativism", "state capitalism"],
        "democracy": ["deliberative systems", "sortition", "liquid democracy"],
        "bureaucracy": ["street-level algorithms", "digital government", "administrative law"],
        "state capacity": ["state fragility", "institutional resilience", "governance indicators"],
    }

    # Find adjacent topics
    topic_lower = topic.lower()
    adjacent = []
    for key, adjacents in adjacent_fields.items():
        if key in topic_lower:
            adjacent.extend(adjacents)

    if not adjacent:
        adjacent = ["institutional design", "public policy implementation", "governance reform"]

    # Select based on surprise factor
    n_adjacent = max(1, int(len(adjacent) * surprise_factor))
    selected = random.sample(adjacent, min(n_adjacent, len(adjacent)))

    # Search for the adjacent topics
    engine = get_discovery_engine()
    results = []
    for adj_topic in selected:
        try:
            search_results = engine.search(adj_topic, max_results=3)
            for paper in search_results.get("results", []):
                paper["serendipity_origin"] = adj_topic
                paper["original_topic"] = topic
                results.append(paper)
        except Exception:
            pass

    return results


# ═══════════════════════════════════════════════════════════════════
# §CE-35: Citation Graph Walker — Follow the citation chain
# ═══════════════════════════════════════════════════════════════════

def walk_citation_graph(seed_paper_title: str, depth: int = 2,
                         direction: str = "cited_by") -> dict:
    """Walk the citation graph starting from a seed paper.

    direction: "cited_by" (who cites this?) or "references" (who does this cite?)
    depth: How many hops (1 = direct citations, 2 = citations of citations)

    Returns a tree of papers with relevance to your research.
    """
    import requests as _req

    visited = set()
    graph = {"root": seed_paper_title, "nodes": [], "edges": []}

    def _search_openalex(title: str) -> dict | None:
        """Find a paper on OpenAlex by title."""
        try:
            resp = _req.get(
                "https://api.openalex.org/works",
                params={"filter": f"title.search:{title[:100]}", "per-page": 1},
                timeout=10,
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                return results[0] if results else None
        except Exception:
            return None

    def _get_citations(work_id: str, direction: str, limit: int = 5) -> list[dict]:
        """Get citations for a work from OpenAlex."""
        try:
            if direction == "cited_by":
                filter_param = f"cites:{work_id}"
            else:
                filter_param = f"cited_by:{work_id}"

            resp = _req.get(
                "https://api.openalex.org/works",
                params={
                    "filter": filter_param,
                    "per-page": limit,
                    "sort": "cited_by_count:desc",
                },
                timeout=10,
            )
            if resp.status_code == 200:
                return resp.json().get("results", [])
        except Exception:
            pass
        return []

    def _walk(title: str, current_depth: int):
        if current_depth >= depth or title in visited:
            return
        visited.add(title)

        work = _search_openalex(title)
        if not work:
            return

        work_id = work.get("id", "").split("/")[-1]
        node = {
            "title": work.get("title", title),
            "year": work.get("publication_year"),
            "cited_by_count": work.get("cited_by_count", 0),
            "depth": current_depth,
        }
        graph["nodes"].append(node)

        citations = _get_citations(work_id, direction, limit=5)
        for cited in citations:
            cited_title = cited.get("title", "")
            if cited_title and cited_title not in visited:
                graph["edges"].append({
                    "source": title[:80],
                    "target": cited_title[:80],
                    "direction": direction,
                })
                _walk(cited_title, current_depth + 1)

    _walk(seed_paper_title, 0)

    return {
        "seed": seed_paper_title,
        "direction": direction,
        "depth": depth,
        "nodes_found": len(graph["nodes"]),
        "edges_found": len(graph["edges"]),
        "graph": graph,
    }


# ═══════════════════════════════════════════════════════════════════
# §CE-36: Temporal Trend Detection — What's hot and what's not
# ═══════════════════════════════════════════════════════════════════

def detect_temporal_trends(topic: str, year_range: tuple[int, int] = (2015, 2026)) -> dict:
    """Detect publication trends for a topic over time.

    Shows whether a topic is rising, peaking, or declining
    in the academic literature.
    """
    import requests as _req

    yearly_counts = {}
    start, end = year_range

    for year in range(start, end + 1):
        try:
            resp = _req.get(
                "https://api.openalex.org/works",
                params={
                    "filter": f"title.search:{topic},publication_year:{year}",
                    "per-page": 1,
                },
                timeout=8,
            )
            if resp.status_code == 200:
                data = resp.json()
                count = data.get("meta", {}).get("count", 0)
                yearly_counts[year] = count
        except Exception:
            yearly_counts[year] = 0

    # Analyze trend
    years = sorted(yearly_counts.keys())
    counts = [yearly_counts[y] for y in years]

    if len(counts) < 3:
        trend = "insufficient_data"
    else:
        recent_avg = sum(counts[-3:]) / 3
        older_avg = sum(counts[:3]) / 3 if len(counts) >= 3 else 1

        if recent_avg > older_avg * 1.5:
            trend = "rising"
        elif recent_avg < older_avg * 0.5:
            trend = "declining"
        elif max(counts) == counts[len(counts) // 2]:
            trend = "peaked"
        else:
            trend = "stable"

    peak_year = years[counts.index(max(counts))] if counts else None

    return {
        "topic": topic,
        "trend": trend,
        "yearly_counts": yearly_counts,
        "peak_year": peak_year,
        "peak_count": max(counts) if counts else 0,
        "total_publications": sum(counts),
        "recommendation": {
            "rising": f"'{topic}' is gaining traction — position your work at the frontier.",
            "declining": f"'{topic}' is waning in popularity — consider framing as revival or reinterpretation.",
            "peaked": f"'{topic}' may have peaked around {peak_year}. Focus on what's NEXT.",
            "stable": f"'{topic}' has steady interest — good foundation for incremental contributions.",
        }.get(trend, ""),
    }

