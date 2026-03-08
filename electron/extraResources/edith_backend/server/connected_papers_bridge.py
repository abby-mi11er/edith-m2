"""
Connected Papers Bridge — Build similarity graphs from citation networks.
==========================================================================
Connected Papers doesn't expose a public API, so we build equivalent
functionality using Semantic Scholar's free citation graph API.
"""
import logging
import os
from collections import defaultdict

log = logging.getLogger("edith.connected_papers_bridge")


class ConnectedPapersBridge:
    """Build paper similarity clusters using Semantic Scholar data."""

    def __init__(self):
        self._s2 = None

    def _ensure_s2(self):
        if self._s2 is None:
            try:
                from server.connectors_full import SemanticScholarConnector
                self._s2 = SemanticScholarConnector(
                    api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY", os.environ.get("S2_API_KEY", ""))
                )
            except ImportError:
                from pipelines.connectors import SemanticScholarConnector
                self._s2 = SemanticScholarConnector(
                    api_key=os.environ.get("SEMANTIC_SCHOLAR_API_KEY", os.environ.get("S2_API_KEY", ""))
                )
        return self._s2

    def build_similarity_graph(self, seed_paper_id: str, depth: int = 1, max_nodes: int = 30) -> dict:
        """Build a similarity graph starting from a seed paper.

        Returns: {nodes: [...], edges: [...], clusters: [...]}
        """
        s2 = self._ensure_s2()
        if not s2:
            return {"error": "Semantic Scholar connector not available"}

        nodes = {}
        edges = []
        visited = set()

        def _crawl(paper_id: str, current_depth: int):
            if paper_id in visited or len(nodes) >= max_nodes:
                return
            visited.add(paper_id)

            paper = s2.get_paper(paper_id) if hasattr(s2, "get_paper") else None
            if not paper:
                return

            nodes[paper_id] = {
                "id": paper_id,
                "title": paper.get("title", ""),
                "year": paper.get("year"),
                "authors": paper.get("authors", [])[:3],
                "citation_count": paper.get("citationCount", 0),
            }

            if current_depth < depth:
                refs = s2.get_references(paper_id, limit=10) if hasattr(s2, "get_references") else []
                cites = s2.get_citations(paper_id, limit=10) if hasattr(s2, "get_citations") else []
                for ref in (refs or []):
                    ref_id = ref.get("paperId") or ref.get("id", "")
                    if ref_id:
                        edges.append({"source": paper_id, "target": ref_id, "type": "references"})
                        _crawl(ref_id, current_depth + 1)
                for cite in (cites or []):
                    cite_id = cite.get("paperId") or cite.get("id", "")
                    if cite_id:
                        edges.append({"source": cite_id, "target": paper_id, "type": "cites"})
                        _crawl(cite_id, current_depth + 1)

        _crawl(seed_paper_id, 0)

        # Simple clustering: group by shared references
        clusters = self._cluster_nodes(nodes, edges)

        return {
            "nodes": list(nodes.values()),
            "edges": edges,
            "clusters": clusters,
            "seed": seed_paper_id,
        }

    def _cluster_nodes(self, nodes: dict, edges: list) -> list[dict]:
        """Simple co-citation clustering."""
        adjacency = defaultdict(set)
        for e in edges:
            adjacency[e["source"]].add(e["target"])
            adjacency[e["target"]].add(e["source"])

        clusters = []
        assigned = set()
        for node_id in nodes:
            if node_id in assigned:
                continue
            cluster = {node_id}
            queue = [node_id]
            while queue and len(cluster) < 10:
                current = queue.pop(0)
                for neighbor in adjacency.get(current, set()):
                    if neighbor not in assigned and neighbor in nodes:
                        cluster.add(neighbor)
                        assigned.add(neighbor)
                        queue.append(neighbor)
            assigned.update(cluster)
            clusters.append({"members": list(cluster), "size": len(cluster)})
        return clusters

    def status(self) -> dict:
        s2 = self._ensure_s2()
        return {
            "available": s2 is not None,
            "configured": True,
            "note": "Uses Semantic Scholar API (no separate key needed)",
        }
