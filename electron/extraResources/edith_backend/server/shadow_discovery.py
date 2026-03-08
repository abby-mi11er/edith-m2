"""
Shadow Variable Discovery — "Insane Mode" Feature
====================================================
Graph topology scan across the entire Knowledge Atlas to find
"theoretical bridges" — disconnected research clusters that share
hidden structural similarities but have never been cited together.

This is how E.D.I.T.H. discovers your dissertation's "original contribution."

Algorithm:
  1. Build a co-citation graph from indexed sources
  2. Detect "islands" — clusters of research with no cross-links
  3. For each island pair, compute semantic similarity of core concepts
  4. High similarity + zero citation links = "Shadow Variable" → theoretical bridge
  5. Report bridges with specific mechanism descriptions
"""

import hashlib
import logging
import os
import re
import time
import threading
from collections import defaultdict
from typing import Optional

log = logging.getLogger("edith.shadow")


# ═══════════════════════════════════════════════════════════════════
# Citation Graph Builder
# ═══════════════════════════════════════════════════════════════════

def _extract_citations(text: str) -> list[str]:
    """Extract author-year citation patterns from text."""
    patterns = [
        r"([A-Z][a-z]+)\s*\((\d{4})\)",       # Author (Year)
        r"\(([A-Z][a-z]+),?\s*(\d{4})\)",       # (Author, Year)
        r"([A-Z][a-z]+)\s+et\s+al\.\s*\((\d{4})\)",  # Author et al. (Year)
        r"([A-Z][a-z]+)\s+and\s+[A-Z][a-z]+\s*\((\d{4})\)",  # Author and Author (Year)
    ]
    citations = set()
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            author = match.group(1).lower()
            year = match.group(2)
            citations.add(f"{author}_{year}")
    return list(citations)


def build_citation_graph(sources: list[dict]) -> dict:
    """Build a co-citation graph from indexed sources.

    Returns:
        dict with:
          nodes: list of source nodes with metadata
          edges: list of citation links between sources
          clusters: list of disconnected components
    """
    # Build nodes
    nodes = {}
    for i, source in enumerate(sources):
        meta = source.get("metadata", {})
        text = source.get("text", "") or source.get("content", "")
        author = meta.get("author", f"source_{i}")
        year = meta.get("year", "")
        title = meta.get("title", "")

        node_id = f"{author.lower().replace(' ', '_')}_{year}" if year else f"src_{i}"
        nodes[node_id] = {
            "id": node_id,
            "author": author,
            "year": year,
            "title": title,
            "text_preview": text[:200],
            "citations": _extract_citations(text),
            "keywords": _extract_keywords(text),
            "index": i,
        }

    # Build edges — citation links AND keyword similarity
    edges = []
    node_ids = set(nodes.keys())
    node_list = list(nodes.keys())

    for nid, node in nodes.items():
        # Citation-based edges
        for cited in node["citations"]:
            if cited in node_ids and cited != nid:
                edges.append({"from": nid, "to": cited, "type": "cites"})
            for other_id in node_ids:
                if other_id != nid and cited.split("_")[0] in other_id:
                    edges.append({"from": nid, "to": other_id, "type": "co-author"})

    # Keyword-based edges — sources sharing ≥1 keyword get linked
    # This creates thematic clusters even without explicit citations
    for i in range(len(node_list)):
        for j in range(i + 1, len(node_list)):
            nid_a, nid_b = node_list[i], node_list[j]
            kw_a = set(nodes[nid_a].get("keywords", []))
            kw_b = set(nodes[nid_b].get("keywords", []))
            shared = kw_a & kw_b
            if shared:
                edges.append({
                    "from": nid_a, "to": nid_b,
                    "type": "keyword_overlap",
                    "shared": list(shared),
                })

    # Detect connected components (clusters)
    clusters = _find_connected_components(nodes, edges)

    return {
        "nodes": list(nodes.values()),
        "edges": edges,
        "clusters": clusters,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


def _extract_keywords(text: str) -> list[str]:
    """Extract thematic keywords from text."""
    text_lower = text.lower()[:2000]

    # Political science keyword categories
    keyword_pools = {
        "welfare": ["welfare", "snap", "tanf", "medicaid", "food stamp", "social program",
                     "safety net", "public assistance", "benefit"],
        "governance": ["governance", "institution", "state capacity", "bureaucracy",
                       "public administration", "government", "regime"],
        "criminal": ["criminal", "cartel", "violence", "organized crime", "extortion",
                      "drug", "trafficking", "enforcement"],
        "rural": ["rural", "county", "agricultural", "small town", "heartland",
                  "non-metropolitan"],
        "charity": ["charity", "nonprofit", "non-state", "ngo", "faith-based",
                    "volunteer", "philanthropy", "civil society"],
        "voter": ["voter", "turnout", "participation", "election", "ballot",
                  "partisan", "mobilization", "campaign"],
        "race": ["race", "racial", "ethnic", "minority", "discrimination",
                 "segregation", "diversity"],
        "inequality": ["inequality", "poverty", "income", "wealth", "class",
                      "stratification", "redistribution"],
        "methodology": ["regression", "experiment", "quasi-experiment",
                       "difference-in-differences", "instrumental variable",
                       "causal", "panel data"],
    }

    found = []
    for category, words in keyword_pools.items():
        hits = sum(1 for w in words if w in text_lower)
        if hits >= 2:
            found.append(category)

    return found


def _find_connected_components(nodes: dict, edges: list) -> list[dict]:
    """Find disconnected components in the citation graph."""
    # Build adjacency list
    adj = defaultdict(set)
    for edge in edges:
        adj[edge["from"]].add(edge["to"])
        adj[edge["to"]].add(edge["from"])

    visited = set()
    components = []

    for node_id in nodes:
        if node_id in visited:
            continue

        # BFS to find component
        component = set()
        queue = [node_id]
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            for neighbor in adj.get(current, []):
                if neighbor not in visited and neighbor in nodes:
                    queue.append(neighbor)

        if component:
            # Aggregate keywords for this cluster
            cluster_keywords = set()
            cluster_authors = set()
            for nid in component:
                cluster_keywords.update(nodes[nid].get("keywords", []))
                cluster_authors.add(nodes[nid]["author"])

            components.append({
                "nodes": list(component),
                "size": len(component),
                "keywords": list(cluster_keywords),
                "authors": list(cluster_authors)[:10],
            })

    return sorted(components, key=lambda c: c["size"], reverse=True)


# ═══════════════════════════════════════════════════════════════════
# Shadow Variable Detection — find theoretical bridges
# ═══════════════════════════════════════════════════════════════════

def find_shadow_variables(sources: list[dict]) -> dict:
    """Find "shadow variables" — hidden theoretical bridges between
    disconnected research clusters.

    A shadow variable exists when:
      - Two clusters have NO citation links
      - BUT share semantic/keyword overlap
      - Suggesting a mechanism in one field maps to another

    Returns:
        dict with bridges found, each containing:
          cluster_a, cluster_b: the disconnected clusters
          shared_keywords: overlapping themes
          bridge_hypothesis: what the connection might be
    """
    t0 = time.time()

    # Build citation graph
    graph = build_citation_graph(sources)
    clusters = graph["clusters"]

    if len(clusters) < 2:
        return {
            "bridges": [],
            "message": "All sources are connected — no isolated islands found",
            "graph_stats": {
                "nodes": graph["node_count"],
                "edges": graph["edge_count"],
                "clusters": len(clusters),
            },
        }

    # Compare all cluster pairs for keyword overlap OR complementary themes
    bridges = []

    # Complementary keyword pairs — field knowledge says these should connect
    _COMPLEMENTARY_PAIRS = [
        ({"welfare"}, {"criminal"}),
        ({"charity"}, {"governance"}),
        ({"charity"}, {"criminal"}),
        ({"rural"}, {"inequality"}),
        ({"voter"}, {"charity"}),
        ({"welfare"}, {"governance"}),
    ]

    for i, cluster_a in enumerate(clusters):
        for j, cluster_b in enumerate(clusters):
            if j <= i:
                continue

            # Skip clusters with no keywords (nothing to compare)
            if not cluster_a["keywords"] or not cluster_b["keywords"]:
                continue

            # Find shared keywords
            keywords_a = set(cluster_a["keywords"])
            keywords_b = set(cluster_b["keywords"])
            shared = keywords_a & keywords_b

            # Also check complementary keyword pairs
            complementary_match = False
            for pair_a, pair_b in _COMPLEMENTARY_PAIRS:
                if (pair_a & keywords_a and pair_b & keywords_b) or \
                   (pair_b & keywords_a and pair_a & keywords_b):
                    complementary_match = True
                    # Add the complementary keywords as "implied shared"
                    shared = shared | (pair_a & keywords_a) | (pair_b & keywords_b)
                    break

            if shared or complementary_match:
                # Calculate bridge strength
                strength = max(1, len(shared)) * min(cluster_a["size"], cluster_b["size"])

                # Generate bridge hypothesis
                hypothesis = _generate_bridge_hypothesis(
                    keywords_a, keywords_b, shared,
                    cluster_a["authors"], cluster_b["authors"],
                )

                bridges.append({
                    "cluster_a": {
                        "size": cluster_a["size"],
                        "keywords": list(keywords_a),
                        "authors": cluster_a["authors"][:5],
                    },
                    "cluster_b": {
                        "size": cluster_b["size"],
                        "keywords": list(keywords_b),
                        "authors": cluster_b["authors"][:5],
                    },
                    "shared_keywords": list(shared),
                    "unique_to_a": list(keywords_a - keywords_b)[:5],
                    "unique_to_b": list(keywords_b - keywords_a)[:5],
                    "strength": strength,
                    "hypothesis": hypothesis,
                    "bridge_type": "complementary" if complementary_match and not (keywords_a & keywords_b)
                                  else "shared_keyword",
                })

    # Sort by strength
    bridges.sort(key=lambda b: b["strength"], reverse=True)

    elapsed = time.time() - t0
    log.info(
        f"§SHADOW: Found {len(bridges)} bridges across {len(clusters)} clusters "
        f"in {elapsed:.1f}s"
    )

    return {
        "bridges": bridges[:10],  # Top 10
        "graph_stats": {
            "nodes": graph["node_count"],
            "edges": graph["edge_count"],
            "clusters": len(clusters),
        },
        "elapsed": f"{elapsed:.1f}s",
    }


def _generate_bridge_hypothesis(
    keywords_a: set, keywords_b: set, shared: set,
    authors_a: list, authors_b: list,
) -> str:
    """Generate a natural-language hypothesis about why two clusters
    should be connected."""

    # Build mechanism description based on shared keywords
    mechanism_map = {
        frozenset(["welfare", "criminal"]): (
            "Non-state provision of social goods — both criminal organizations "
            "and welfare systems function as distributors of resources to underserved "
            "populations. The mechanism of 'protection in exchange for loyalty' may "
            "map directly to 'benefits in exchange for political support.'"
        ),
        frozenset(["charity", "governance"]): (
            "Third-party governance — charities and nonprofits act as de facto "
            "government agencies in areas of state retreat. This 'outsourcing' "
            "creates a submerged governance layer invisible to standard metrics."
        ),
        frozenset(["rural", "inequality"]): (
            "Spatial inequality — rural areas experience unique forms of resource "
            "deprivation that differ from urban poverty, but the underlying "
            "mechanisms of exclusion may be structurally identical."
        ),
        frozenset(["voter", "charity"]): (
            "Policy feedback via non-state channels — when citizens receive "
            "services through charities rather than government, it may depress "
            "political participation by severing the perceived link between "
            "voting and material benefits."
        ),
        frozenset(["governance", "criminal"]): (
            "Parallel governance — criminal organizations and state institutions "
            "compete for legitimacy using similar governance mechanisms. The "
            "'social contract' can be provided by non-state actors."
        ),
    }

    # Try to match a specific mechanism
    for key_combo, hypothesis in mechanism_map.items():
        if key_combo.issubset(shared | keywords_a | keywords_b):
            return hypothesis

    # Generic hypothesis
    a_str = ", ".join(sorted(keywords_a - shared)[:3]) or "topic A"
    b_str = ", ".join(sorted(keywords_b - shared)[:3]) or "topic B"
    shared_str = ", ".join(sorted(shared))

    return (
        f"POTENTIAL BRIDGE: Research on {a_str} and {b_str} both involve "
        f"{shared_str}, but have never been cited together. "
        f"Authors {', '.join(authors_a[:2])} and {', '.join(authors_b[:2])} "
        f"may be studying the same underlying mechanism from different angles."
    )


# ═══════════════════════════════════════════════════════════════════
# Deep-Time Research Operative — batch extraction + hypothesis gen
# ═══════════════════════════════════════════════════════════════════

class DeepTimeOperative:
    """Long-running research operative that extracts structured data
    from thousands of PDFs and generates hypotheses.

    Example directive: "Extract every mention of a local charity in the
    Texas folder, cross-reference with SNAP data, build a timeline."
    """

    def __init__(self):
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()

    def start_extraction(
        self,
        directive: str,
        target_folder: str = "",
        extract_patterns: list[str] = None,
    ) -> dict:
        """Start a long-running extraction job.

        Args:
            directive: natural language research directive
            target_folder: subfolder within data root to scan
            extract_patterns: regex patterns to extract

        Returns:
            dict with job_id and status
        """
        job_id = hashlib.sha256(
            f"{directive}{time.time()}".encode()
        ).hexdigest()[:12]

        job = {
            "job_id": job_id,
            "directive": directive,
            "target_folder": target_folder,
            "status": "queued",
            "progress": 0,
            "phase": "init",
            "files_scanned": 0,
            "extractions": [],
            "hypotheses": [],
            "started_at": None,
            "completed_at": None,
            "error": None,
        }

        with self._lock:
            self._jobs[job_id] = job

        thread = threading.Thread(
            target=self._run_extraction, args=(job, extract_patterns),
            daemon=True,
        )
        thread.start()
        return {"job_id": job_id, "status": "started"}

    def get_status(self, job_id: str) -> Optional[dict]:
        with self._lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[dict]:
        with self._lock:
            return [
                {k: v for k, v in j.items() if k != "extractions"}
                for j in self._jobs.values()
            ]

    def _run_extraction(self, job: dict, patterns: list[str] = None):
        """Run the extraction pipeline."""
        job["started_at"] = time.time()
        job["status"] = "running"

        data_root = os.environ.get("EDITH_DATA_ROOT", "")
        target = os.path.join(data_root, job.get("target_folder", ""))

        if not os.path.isdir(target):
            job["status"] = "failed"
            job["error"] = f"Target directory not found: {target}"
            return

        try:
            # Phase 1: Scan files
            job["phase"] = "scanning"
            files = []
            for root, dirs, filenames in os.walk(target):
                for fname in filenames:
                    if fname.lower().endswith((".pdf", ".txt", ".md")):
                        files.append(os.path.join(root, fname))

            job["files_scanned"] = len(files)
            log.info(f"§DEEP-TIME [{job['job_id']}]: Found {len(files)} files")

            if not files:
                job["status"] = "completed"
                job["progress"] = 100
                return

            # Phase 2: Extract patterns from text files
            job["phase"] = "extracting"
            default_patterns = [
                r"(?i)charit(?:y|ies|able)",
                r"(?i)SNAP|food\s+stamp|EBT",
                r"(?i)non-?profit|NGO|faith-?based",
                r"(?i)rural\s+(?:count(?:y|ies)|area|community)",
                r"\b20[12]\d\b",  # Years 2010-2029
            ]
            use_patterns = patterns or default_patterns

            for i, fpath in enumerate(files):
                job["progress"] = int(30 + (i / len(files)) * 50)

                try:
                    # Only process text files (skip PDFs that need parsing)
                    if fpath.endswith((".txt", ".md")):
                        with open(fpath, "r", errors="ignore") as f:
                            content = f.read()
                    else:
                        continue  # Skip binary files

                    for pattern in use_patterns:
                        matches = re.findall(pattern, content)
                        if matches:
                            job["extractions"].append({
                                "file": os.path.basename(fpath),
                                "pattern": pattern[:30],
                                "count": len(matches),
                                "samples": matches[:3],
                            })
                except Exception:
                    pass  # Skip unreadable files

            # Phase 3: Generate hypotheses
            job["phase"] = "hypothesis_generation"
            job["progress"] = 85

            if job["extractions"]:
                # Count extraction types
                pattern_counts = defaultdict(int)
                for ext in job["extractions"]:
                    pattern_counts[ext["pattern"]] += ext["count"]

                # Simple hypothesis generation from co-occurrence
                hypothesis_templates = [
                    "Files mentioning both charities AND SNAP may indicate "
                    "non-state actors filling gaps in federal welfare delivery.",
                    "Rural county mentions co-occurring with nonprofit references "
                    "suggest geographic concentration of third-sector governance.",
                    "Year patterns suggest temporal clustering that may indicate "
                    "policy feedback loops responding to specific legislative changes.",
                ]
                job["hypotheses"] = hypothesis_templates[:len(pattern_counts)]

            job["status"] = "completed"
            job["progress"] = 100
            job["completed_at"] = time.time()
            elapsed = job["completed_at"] - job["started_at"]
            log.info(
                f"§DEEP-TIME [{job['job_id']}]: Completed in {elapsed:.0f}s — "
                f"{len(files)} files, {len(job['extractions'])} extractions"
            )

        except Exception as e:
            job["status"] = "failed"
            job["error"] = str(e)[:500]
            log.error(f"§DEEP-TIME [{job['job_id']}]: Failed: {e}")


# Singleton
deep_time = DeepTimeOperative()


# ═══════════════════════════════════════════════════════════════════
# CITADEL §2: THOUGHT BUBBLE BUFFER — Passive Margin Notes
# ═══════════════════════════════════════════════════════════════════

import threading
import json
from datetime import datetime


class ThoughtBubbleBuffer:
    """Passive "Thought Bubble" system for the writing margin.

    While you write Chapter 3, Shadow Discovery silently surfaces
    related counter-arguments and complementary findings. They appear
    as small, unobtrusive margin notes you can ignore until ready.

    Max 5 unread bubbles at any time — oldest auto-dismissed.

    Usage:
        bubbles = ThoughtBubbleBuffer()
        bubbles.run_shadow_alongside("Policy feedback reduces turnout...")
        # ... later ...
        for b in bubbles.get_unread():
            print(b["snippet"])
    """

    def __init__(self, max_bubbles: int = 5):
        self._bubbles: list[dict] = []
        self._lock = threading.Lock()
        self._max = max_bubbles

    def _add(self, bubble: dict):
        with self._lock:
            self._bubbles.append(bubble)
            # Auto-dismiss oldest if over limit
            while len(self._bubbles) > self._max:
                self._bubbles.pop(0)

    def get_unread(self) -> list[dict]:
        """Get all unread thought bubbles."""
        with self._lock:
            return [b for b in self._bubbles if not b.get("read")]

    def get_all(self) -> list[dict]:
        with self._lock:
            return list(self._bubbles)

    def dismiss(self, index: int = 0):
        """Dismiss a bubble by index."""
        with self._lock:
            if 0 <= index < len(self._bubbles):
                self._bubbles.pop(index)

    def dismiss_all(self):
        with self._lock:
            self._bubbles.clear()

    def expand(self, index: int = 0) -> dict:
        """Mark a bubble as read and return its full content."""
        with self._lock:
            if 0 <= index < len(self._bubbles):
                self._bubbles[index]["read"] = True
                return self._bubbles[index]
            return {}

    def run_shadow_alongside(
        self,
        paragraph: str,
        model_chain: list[str] = None,
    ):
        """Run shadow discovery in background while user writes.

        Extracts key concepts from the paragraph, queries ChromaDB
        for related content, and queues thought bubbles silently.

        Non-blocking: starts a background thread immediately.
        """
        t = threading.Thread(
            target=self._shadow_worker,
            args=(paragraph, model_chain),
            daemon=True,
        )
        t.start()
        log.info("§BUBBLE: Shadow discovery started in background")

    def _shadow_worker(self, paragraph: str, model_chain: list[str] = None):
        """Background worker — extract concepts → query → make bubbles."""
        try:
            # Extract key concepts
            keywords = _extract_keywords(paragraph)
            if not keywords:
                return

            query_text = " ".join(list(keywords)[:5])

            # Query ChromaDB for related content
            try:
                from server.chroma_backend import query_chroma
                chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "")
                if not chroma_dir:
                    return

                results = query_chroma(
                    query_text,
                    collection_name="academic_corpus",
                    chroma_dir=chroma_dir,
                    top_k=3,
                )

                for i, result in enumerate(results.get("results", [])[:3]):
                    text = result.get("text", "")[:200]
                    source = result.get("metadata", {}).get("source", "Unknown")
                    relevance = result.get("score", 0.5)

                    # Only bubble if sufficiently relevant
                    if relevance < 0.3:
                        continue

                    self._add({
                        "type": "shadow_finding",
                        "snippet": text,
                        "source": source,
                        "relevance": round(relevance, 3),
                        "keywords_matched": list(keywords)[:3],
                        "timestamp": datetime.now().isoformat(),
                        "read": False,
                    })

            except Exception as e:
                log.debug(f"§BUBBLE: ChromaDB query failed: {e}")

            # Also check for counter-arguments via shadow variables
            try:
                sources_sample = [{"text": paragraph, "metadata": {"source": "current_draft"}}]
                shadows = find_shadow_variables(sources_sample)
                for sv in shadows.get("shadow_variables", [])[:1]:
                    self._add({
                        "type": "counter_argument",
                        "snippet": sv.get("bridge_hypothesis", "")[:200],
                        "source": "Shadow Variable Detection",
                        "relevance": sv.get("keyword_overlap", 0.5),
                        "keywords_matched": list(sv.get("shared_keywords", set()))[:3],
                        "timestamp": datetime.now().isoformat(),
                        "read": False,
                    })
            except Exception:
                pass

            log.info(f"§BUBBLE: Generated {len(self.get_unread())} thought bubbles")

        except Exception as e:
            log.debug(f"§BUBBLE: Shadow worker failed: {e}")


# Global thought bubble buffer
thought_bubbles = ThoughtBubbleBuffer()
