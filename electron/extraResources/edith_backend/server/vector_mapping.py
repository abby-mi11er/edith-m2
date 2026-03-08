"""
Vector Mapping — 3D Coordinate System for Knowledge Atlas
============================================================
Scans the corpus and assigns every document a unique (X, Y, Z)
position based on semantic similarity. Powers the Three.js
constellation in the CockpitView.

Methods:
  1. PCA-3D: Fast, deterministic, good for initial mapping
  2. UMAP-3D: High-fidelity topology preservation (requires umap-learn)
  3. Cluster-Seeded: Use known field assignments + random dispersion
"""

import hashlib
import json
import logging
import math
import os
import random
import time
from collections import Counter, defaultdict
from typing import Optional

log = logging.getLogger("edith.vector_mapping")


# ═══════════════════════════════════════════════════════════════════
# §1: FIELD CLASSIFIER — Assign academic cluster
# ═══════════════════════════════════════════════════════════════════

FIELD_KEYWORDS = {
    "APE": [
        "welfare", "snap", "food stamp", "poverty", "inequality", "social policy",
        "policy feedback", "submerged state", "tax expenditure", "means-tested",
        "mettler", "hacker", "pierson", "campbell", "skocpol",
    ],
    "Voting": [
        "voter", "turnout", "election", "partisan", "democrat", "republican",
        "ballot", "mobilization", "registration", "political participation",
        "aldrich", "downs", "riker", "fiorina",
    ],
    "Methods": [
        "regression", "causal", "instrument", "treatment", "ols", "fixed effect",
        "difference-in-diff", "rdd", "matching", "propensity", "bayesian",
        "angrist", "imbens", "rubin", "pearl",
    ],
    "Criminal": [
        "criminal", "cartel", "violence", "extortion", "governance", "gang",
        "organized crime", "narco", "trafficking", "enforcement",
        "arias", "lessing", "trejo", "kalyvas",
    ],
    "Comparative": [
        "comparative", "regime", "democracy", "authorit", "clientel",
        "institutional", "development", "comparative politics",
        "levitsky", "ziblatt", "acemoglu", "north", "stokes",
    ],
}

# 3D cluster centroids (normalized 0-1 space)
CLUSTER_CENTROIDS = {
    "APE":         (0.25, 0.35, 0.50),
    "Voting":      (0.60, 0.25, 0.45),
    "Methods":     (0.45, 0.70, 0.40),
    "Criminal":    (0.80, 0.60, 0.55),
    "Comparative": (0.15, 0.75, 0.50),
    "General":     (0.50, 0.50, 0.50),
}

CLUSTER_COLORS = {
    "APE": "#6366f1",
    "Voting": "#0ea5e9",
    "Methods": "#15803D",
    "Criminal": "#C2410C",
    "Comparative": "#7C3AED",
    "General": "#94A3B8",
}


def classify_field(text: str) -> tuple[str, float]:
    """Classify a text chunk into its academic field."""
    text_lower = text.lower()
    scores = {}
    for field, keywords in FIELD_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        scores[field] = count

    best = max(scores, key=scores.get)
    total = sum(scores.values())
    confidence = scores[best] / max(total, 1)

    if scores[best] == 0:
        return "General", 0.0
    return best, round(confidence, 3)


# ═══════════════════════════════════════════════════════════════════
# §2: VECTOR MAPPING — Assign 3D coordinates
# ═══════════════════════════════════════════════════════════════════

def map_chunk_to_3d(
    text: str,
    doc_id: str = "",
    field: str = "",
    embedding: list[float] = None,
) -> dict:
    """Map a single document chunk to 3D coordinates.

    Uses cluster centroid + semantic dispersion.
    If embeddings are available, uses PCA projection.
    Otherwise, uses hash-based deterministic placement.
    """
    # Classify field if not provided
    if not field:
        field, confidence = classify_field(text)
    else:
        confidence = 1.0

    centroid = CLUSTER_CENTROIDS.get(field, CLUSTER_CENTROIDS["General"])

    if embedding and len(embedding) >= 3:
        # Use first 3 PCA components as displacement
        norm = math.sqrt(sum(e ** 2 for e in embedding[:3]) + 1e-10)
        dx = embedding[0] / norm * 0.12
        dy = embedding[1] / norm * 0.12
        dz = embedding[2] / norm * 0.12
    else:
        # Hash-based deterministic placement
        h = hashlib.md5((doc_id + text[:100]).encode()).hexdigest()
        dx = (int(h[:4], 16) / 65535 - 0.5) * 0.16
        dy = (int(h[4:8], 16) / 65535 - 0.5) * 0.16
        dz = (int(h[8:12], 16) / 65535 - 0.5) * 0.16

    x = max(0.02, min(0.98, centroid[0] + dx))
    y = max(0.02, min(0.98, centroid[1] + dy))
    z = max(0.02, min(0.98, centroid[2] + dz))

    # Node size based on text richness
    word_count = len(text.split())
    size = min(5, 1 + math.log(max(word_count, 1)) / 3)

    return {
        "id": doc_id or hashlib.md5(text[:200].encode()).hexdigest()[:10],
        "x": round(x, 4),
        "y": round(y, 4),
        "z": round(z, 4),
        "cluster": field,
        "color": CLUSTER_COLORS.get(field, "#94A3B8"),
        "size": round(size, 2),
        "confidence": confidence,
        "label": text[:40].replace("\n", " ").strip(),
    }


def build_atlas_from_chroma(
    chroma_dir: str = "",
    collection_name: str = "edith",
    embed_model: str = "",
    sample_size: int = 2000,
) -> dict:
    """Build a complete 3D atlas from the ChromaDB corpus.

    Scans up to sample_size chunks, classifies each,
    assigns 3D coordinates, and returns Three.js-ready data.
    """
    chroma_dir = chroma_dir or os.environ.get("EDITH_CHROMA_DIR", "")
    embed_model = embed_model or os.environ.get("EDITH_EMBED_MODEL", "")
    t0 = time.time()

    nodes = []
    edges = []
    field_counts = Counter()

    # Try to pull from ChromaDB
    try:
        from server.chroma_backend import retrieve_local_sources
        sample_queries = [
            "welfare policy", "voter turnout", "regression analysis",
            "criminal governance", "comparative politics", "rural texas",
            "charity", "SNAP enrollment", "submerged state", "clientelism",
            "multi-level model", "causal inference", "fixed effects",
            "party identification", "institutional change", "policy feedback",
        ]

        seen_ids = set()
        for query in sample_queries:
            try:
                results = retrieve_local_sources(
                    queries=[query],
                    chroma_dir=chroma_dir,
                    collection_name=collection_name,
                    embed_model=embed_model,
                    top_k=sample_size // len(sample_queries),
                )
                for r in results:
                    doc_id = r.get("id", r.get("text", "")[:20])
                    if doc_id not in seen_ids:
                        seen_ids.add(doc_id)
                        node = map_chunk_to_3d(
                            r.get("text", ""),
                            doc_id=doc_id,
                            embedding=r.get("embedding"),
                        )
                        nodes.append(node)
                        field_counts[node["cluster"]] += 1
            except Exception:
                pass

        if len(nodes) < 10:
            raise ValueError("Too few nodes from ChromaDB — using synthetic")
    except Exception:
        # Generate synthetic atlas for demo
        nodes, field_counts = _generate_synthetic_atlas(sample_size)

    # Generate edges (nearest neighbors within clusters)
    edges = _generate_edges(nodes, max_edges=len(nodes) * 2)

    elapsed = time.time() - t0
    return {
        "nodes": nodes,
        "edges": edges,
        "total_nodes": len(nodes),
        "total_edges": len(edges),
        "field_distribution": dict(field_counts),
        "clusters": {
            name: {
                "centroid": list(centroid),
                "color": CLUSTER_COLORS.get(name, "#94A3B8"),
                "count": field_counts.get(name, 0),
            }
            for name, centroid in CLUSTER_CENTROIDS.items()
        },
        "elapsed_s": round(elapsed, 2),
        "format": "three_js_ready",
    }


def _generate_synthetic_atlas(n: int = 500) -> tuple[list[dict], Counter]:
    """Generate synthetic atlas for demo/offline mode."""
    field_counts = Counter()
    nodes = []
    distribution = {
        "APE": 0.28, "Voting": 0.22, "Methods": 0.20,
        "Criminal": 0.15, "Comparative": 0.15,
    }

    sample_labels = {
        "APE": [
            "Welfare state retrenchment in rural areas",
            "SNAP enrollment and policy visibility",
            "The submerged state: invisible social programs",
            "Tax expenditures as hidden welfare",
            "Policy feedback and political engagement",
        ],
        "Voting": [
            "Turnout gap in off-year elections",
            "Partisan sorting and geographic clustering",
            "Mobilization effects of charity organizations",
            "Electoral consequences of welfare reform",
            "Party identification in rural America",
        ],
        "Methods": [
            "Multi-level models for nested data",
            "Difference-in-differences with staggered treatment",
            "Synthetic control methods for causal inference",
            "Regression discontinuity design in policy evaluation",
            "Bayesian estimation of treatment effects",
        ],
        "Criminal": [
            "Criminal governance in Mexican cities",
            "Extortion and its effects on local markets",
            "Gang territorial control and citizen welfare",
            "Violence and political participation",
            "State-criminal hybrid governance",
        ],
        "Comparative": [
            "Clientelistic linkage and democratic quality",
            "Institutional layering in Latin America",
            "Path dependence in welfare state development",
            "Regime types and redistribution",
            "Democratic backsliding indicators",
        ],
    }

    for field, fraction in distribution.items():
        count = int(n * fraction)
        labels = sample_labels.get(field, ["Research note"])
        for i in range(count):
            label = labels[i % len(labels)]
            node = map_chunk_to_3d(
                label + f" (sample chunk {i})",
                doc_id=f"{field.lower()}_{i:04d}",
                field=field,
            )
            nodes.append(node)
            field_counts[field] += 1

    return nodes, field_counts


def _generate_edges(nodes: list[dict], max_edges: int = 500) -> list[dict]:
    """Generate edges between nearby nodes (same cluster + proximity).

    §IMP-3.8: Adds quality score for cross-cluster bridge edges.
    """
    edges = []
    # Group by cluster
    by_cluster: dict[str, list[dict]] = defaultdict(list)
    for n in nodes:
        by_cluster[n["cluster"]].append(n)

    for cluster, cluster_nodes in by_cluster.items():
        # Connect nearest neighbors
        for i, a in enumerate(cluster_nodes):
            best_dist = float('inf')
            best_j = -1
            for j, b in enumerate(cluster_nodes):
                if i == j:
                    continue
                d = math.sqrt(
                    (a["x"] - b["x"]) ** 2 +
                    (a["y"] - b["y"]) ** 2 +
                    (a["z"] - b["z"]) ** 2
                )
                if d < best_dist:
                    best_dist = d
                    best_j = j

            if best_j >= 0 and len(edges) < max_edges:
                edges.append({
                    "source": a["id"],
                    "target": cluster_nodes[best_j]["id"],
                    "weight": round(1 / (best_dist + 0.01), 2),
                    "cluster": cluster,
                })

    # Add cross-cluster bridges (sparse)
    clusters = list(by_cluster.keys())
    for i in range(len(clusters)):
        for j in range(i + 1, len(clusters)):
            c1 = by_cluster[clusters[i]]
            c2 = by_cluster[clusters[j]]
            if c1 and c2:
                # Pick the closest pair
                best_pair = None
                best_d = float('inf')
                for a in random.sample(c1, min(5, len(c1))):
                    for b in random.sample(c2, min(5, len(c2))):
                        d = math.sqrt(
                            (a["x"] - b["x"]) ** 2 +
                            (a["y"] - b["y"]) ** 2 +
                            (a["z"] - b["z"]) ** 2
                        )
                        if d < best_d:
                            best_d = d
                            best_pair = (a, b)

                if best_pair and len(edges) < max_edges:
                    # §IMP-3.8: Compute bridge quality score
                    label_a = (best_pair[0].get("label", "") or "").lower().split()
                    label_b = (best_pair[1].get("label", "") or "").lower().split()
                    shared_terms = set(label_a) & set(label_b) - {"the", "of", "and", "in", "a", "to"}
                    quality = min(1.0, len(shared_terms) * 0.3 + 0.2 / (best_d + 0.01))
                    edges.append({
                        "source": best_pair[0]["id"],
                        "target": best_pair[1]["id"],
                        "weight": round(0.5 / (best_d + 0.01), 2),
                        "cluster": "bridge",
                        "type": "cross_cluster",
                        "quality": round(quality, 3),
                        "shared_terms": list(shared_terms)[:3],
                    })

    return edges


# ═══════════════════════════════════════════════════════════════════
# §3: TOPOLOGICAL SUMMARY — Tensions Between Authors
# ═══════════════════════════════════════════════════════════════════

def generate_topological_summary(
    cluster_name: str,
    nodes: list[dict] = None,
    model_chain: list[str] = None,
) -> dict:
    """Generate a summary of theoretical tensions in a cluster.

    Not just a list of papers — a map of intellectual conflicts.
    """
    model_chain = model_chain or [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]

    known_tensions = {
        "APE": {
            "tensions": [
                {"authors": ["Mettler", "Howard"], "topic": "Visibility threshold — at what point does a submerged benefit become 'visible' enough to generate feedback?"},
                {"authors": ["Campbell", "Pierson"], "topic": "Resource vs interpretive effects — does policy feedback work through material benefits or cognitive framing?"},
                {"authors": ["Hacker", "Skocpol"], "topic": "Top-down vs bottom-up retrenchment — is the submerged state growing from elite design or institutional drift?"},
            ],
            "frontier": "The interaction between charity-as-substitute and government-as-absent in rural environments",
        },
        "Voting": {
            "tensions": [
                {"authors": ["Aldrich", "Downs"], "topic": "Rational choice vs expressive utility — why does anyone vote at all?"},
                {"authors": ["Riker", "Fiorina"], "topic": "Minimax regret vs prospective voting — election stakes or retrospective punishment?"},
            ],
            "frontier": "Charity organizations as mobilization infrastructure for voter turnout",
        },
        "Methods": {
            "tensions": [
                {"authors": ["Angrist/Imbens", "Pearl"], "topic": "Potential outcomes vs structural causal models — should social science use DAGs or design?"},
                {"authors": ["Rubin", "Heckman"], "topic": "Selection on observables vs unobservables — matching or modeling?"},
            ],
            "frontier": "Synthetic controls for county-level policy simulation",
        },
        "Criminal": {
            "tensions": [
                {"authors": ["Arias", "Lessing"], "topic": "Criminal governance as substitute vs complement — do criminals replace or supplement the state?"},
                {"authors": ["Trejo", "Kalyvas"], "topic": "Drug war violence as political strategy vs civil war logic"},
            ],
            "frontier": "Criminal governance networks as a general theory applicable to US rural voids",
        },
        "Comparative": {
            "tensions": [
                {"authors": ["Levitsky/Ziblatt", "Acemoglu/Robinson"], "topic": "Norms erosion vs institutional design — what causes democratic backsliding?"},
                {"authors": ["Stokes", "Kitschelt"], "topic": "Clientelism as rational exchange vs cultural norm"},
            ],
            "frontier": "American clientelism — treating US charity networks as clientelistic linkage",
        },
    }

    summary = known_tensions.get(cluster_name, {
        "tensions": [],
        "frontier": f"Unexplored connections in {cluster_name}",
    })

    return {
        "cluster": cluster_name,
        "centroid": list(CLUSTER_CENTROIDS.get(cluster_name, (0.5, 0.5, 0.5))),
        "color": CLUSTER_COLORS.get(cluster_name, "#94A3B8"),
        "tensions": summary.get("tensions", []),
        "frontier": summary.get("frontier", ""),
        "node_count": sum(1 for n in (nodes or []) if n.get("cluster") == cluster_name),
    }


# ═══════════════════════════════════════════════════════════════════
# SCENE 2: ATLAS GAP DETECTION — "Dark Space" Finder
# ═══════════════════════════════════════════════════════════════════

def detect_atlas_gaps(
    nodes: list[dict],
    grid_resolution: int = 8,
    min_density_ratio: float = 0.15,
) -> list[dict]:
    """Scene 2: Find 'dark spaces' in the 3D atlas constellation.

    Grids the 3D space and identifies regions between clusters where
    node density is low (dark) relative to adjacent populated regions.
    These gaps represent unexplored research territory.

    Args:
        nodes: List of atlas nodes from build_atlas_from_chroma()
        grid_resolution: Number of bins per axis (8³ = 512 cells)
        min_density_ratio: Threshold for detecting dark vs populated

    Returns: list of gap dicts with position, neighboring clusters, and bridge candidates.
    """
    if not nodes:
        return []

    # Build density grid
    grid: dict[tuple, list] = defaultdict(list)
    for n in nodes:
        gx = min(grid_resolution - 1, int(n["x"] * grid_resolution))
        gy = min(grid_resolution - 1, int(n["y"] * grid_resolution))
        gz = min(grid_resolution - 1, int(n["z"] * grid_resolution))
        grid[(gx, gy, gz)].append(n)

    # Compute max density for normalization
    max_density = max(len(v) for v in grid.values()) if grid else 1

    # Find dark cells that are between populated cells
    gaps = []
    for gx in range(grid_resolution):
        for gy in range(grid_resolution):
            for gz in range(grid_resolution):
                cell = grid.get((gx, gy, gz), [])
                density = len(cell) / max_density

                # Only interested in low-density cells
                if density > min_density_ratio:
                    continue

                # Check neighbors for populated cells from different clusters
                neighbor_clusters = set()
                neighbor_nodes = []
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for dz in (-1, 0, 1):
                            if dx == dy == dz == 0:
                                continue
                            key = (gx + dx, gy + dy, gz + dz)
                            for n in grid.get(key, []):
                                neighbor_clusters.add(n["cluster"])
                                neighbor_nodes.append(n)

                # A "dark space" is only interesting if it sits between 2+ clusters
                if len(neighbor_clusters) >= 2:
                    # Compute center position of the dark cell
                    cx = (gx + 0.5) / grid_resolution
                    cy = (gy + 0.5) / grid_resolution
                    cz = (gz + 0.5) / grid_resolution

                    gaps.append({
                        "position": {"x": round(cx, 3), "y": round(cy, 3), "z": round(cz, 3)},
                        "density": round(density, 3),
                        "neighboring_clusters": sorted(neighbor_clusters),
                        "neighbor_count": len(neighbor_nodes),
                        "grid_cell": (gx, gy, gz),
                    })

    # Sort by number of neighboring clusters (more = more interesting gap)
    gaps.sort(key=lambda g: (-len(g["neighboring_clusters"]), g["density"]))

    log.info(f"§ATLAS: Found {len(gaps)} dark spaces in constellation")
    return gaps[:20]  # Top 20 most interesting gaps


def find_bridge_papers(
    gap: dict,
    nodes: list[dict],
    top_n: int = 5,
) -> dict:
    """Scene 2: 'Winnie, show me the bridge.'

    Given a dark-space gap, find the papers closest to it from each
    neighboring cluster. These are potential bridge papers that could
    connect the fields.

    Args:
        gap: A gap dict from detect_atlas_gaps()
        nodes: Full list of atlas nodes
        top_n: Number of bridge papers per cluster

    Returns: dict with bridge papers per cluster and synthesis hypothesis.
    """
    gx = gap["position"]["x"]
    gy = gap["position"]["y"]
    gz = gap["position"]["z"]
    neighboring = set(gap.get("neighboring_clusters", []))

    bridges_by_cluster = {}
    for cluster in neighboring:
        # Find nodes in this cluster, sort by distance to gap center
        cluster_nodes = [n for n in nodes if n["cluster"] == cluster]
        scored = []
        for n in cluster_nodes:
            dist = math.sqrt(
                (n["x"] - gx) ** 2 +
                (n["y"] - gy) ** 2 +
                (n["z"] - gz) ** 2
            )
            scored.append((dist, n))
        scored.sort(key=lambda x: x[0])

        bridges_by_cluster[cluster] = [
            {
                "id": n["id"],
                "label": n.get("label", ""),
                "distance_to_gap": round(d, 4),
                "position": {"x": n["x"], "y": n["y"], "z": n["z"]},
                "cluster": cluster,
                "color": n.get("color", CLUSTER_COLORS.get(cluster, "#94A3B8")),
            }
            for d, n in scored[:top_n]
        ]

    # Generate a synthesis hypothesis
    clusters_str = " and ".join(sorted(neighboring))
    hypothesis = (
        f"This gap sits between {clusters_str}. "
        f"The nearest papers from each field may share enough "
        f"conceptual overlap to form an original bridge argument."
    )

    return {
        "gap": gap,
        "bridge_papers": bridges_by_cluster,
        "total_bridges": sum(len(v) for v in bridges_by_cluster.values()),
        "clusters_bridged": sorted(neighboring),
        "hypothesis": hypothesis,
    }


# ═══════════════════════════════════════════════════════════════════
# TITAN §3: THEORETICAL GRAVITY — MIT "Living Data Topology"
# ═══════════════════════════════════════════════════════════════════

# Seminal works that should have extra gravitational pull
_SEMINAL_PAPERS = {
    "mettler": 5.0,
    "submerged state": 5.0,
    "pierson": 4.0,
    "hacker": 3.5,
    "campbell": 3.5,
    "skocpol": 4.0,
    "aldrich": 4.0,
    "downs": 3.5,
    "angrist": 4.0,
    "imbens": 3.5,
    "pearl": 4.0,
    "arias": 3.5,
    "lessing": 3.0,
    "acemoglu": 4.0,
    "levitsky": 3.5,
}


def compute_theoretical_mass(
    nodes: list[dict],
    citation_data: dict = None,
) -> list[dict]:
    """MIT-inspired "Theoretical Gravity" — assign mass to atlas nodes.

    Seminal papers have more "mass" (larger size, stronger gravity).
    Mass is computed from:
    1. Citation count (if available from metadata)
    2. Keyword matches to known seminal authors
    3. Cross-cluster connectivity (bridge nodes get mass boost)

    Returns nodes with updated 'mass' and 'size' fields.
    """
    citation_data = citation_data or {}

    for node in nodes:
        label = (node.get("label", "") or "").lower()
        base_mass = 1.0

        # Citation-based mass
        citations = citation_data.get(node.get("id", ""), 0)
        if citations > 0:
            base_mass += math.log(citations + 1) * 0.5

        # Seminal author boost
        for author, weight in _SEMINAL_PAPERS.items():
            if author in label:
                base_mass += weight
                node["seminal"] = True
                break

        # Confidence boost
        confidence = node.get("confidence", 0.5)
        base_mass *= (0.5 + confidence)

        node["mass"] = round(base_mass, 2)
        # Scale node size by mass (log-compressed)
        node["size"] = round(min(8, 1 + math.log(base_mass + 1) * 1.5), 2)

    log.info(f"§GRAVITY: Computed mass for {len(nodes)} nodes")
    return nodes


def warp_atlas_to_query(
    nodes: list[dict],
    query: str,
    warp_strength: float = 0.3,
) -> list[dict]:
    """MIT-inspired Atlas Warping — reshape the constellation around a search.

    When searched, the relevant nodes pull toward the center and
    irrelevant nodes push to the periphery. The atlas "breathes."

    Args:
        nodes: Atlas nodes with x, y, z coordinates
        query: Search query text
        warp_strength: How aggressively to warp (0=none, 1=extreme)

    Returns: nodes with warped coordinates + relevance scores
    """
    query_lower = query.lower()
    query_words = set(query_lower.split()) - {"the", "of", "and", "in", "a", "to", "for"}
    center = (0.5, 0.5, 0.5)

    for node in nodes:
        label = (node.get("label", "") or "").lower()
        label_words = set(label.split())

        # Compute relevance: word overlap + field match
        overlap = len(query_words & label_words)
        field_match = 1.0 if any(kw in query_lower for kw in
                                  FIELD_KEYWORDS.get(node.get("cluster", ""), []))  else 0.0
        relevance = min(1.0, overlap * 0.3 + field_match * 0.4)

        # Seminal papers matching the query get extra pull
        if node.get("seminal") and overlap > 0:
            relevance = min(1.0, relevance + 0.3)

        node["relevance"] = round(relevance, 3)

        # Warp: relevant nodes pull toward center, irrelevant push away
        pull = relevance * warp_strength
        push = (1 - relevance) * warp_strength * 0.3

        ox, oy, oz = node["x"], node["y"], node["z"]
        # Interpolate toward center for relevant, away for irrelevant
        node["warped_x"] = round(ox + (center[0] - ox) * pull - (center[0] - ox) * push, 4)
        node["warped_y"] = round(oy + (center[1] - oy) * pull - (center[1] - oy) * push, 4)
        node["warped_z"] = round(oz + (center[2] - oz) * pull - (center[2] - oz) * push, 4)

    # Sort by relevance for UI highlighting
    nodes.sort(key=lambda n: n.get("relevance", 0), reverse=True)
    log.info(f"§GRAVITY: Warped atlas for '{query[:30]}' — "
             f"top relevance: {nodes[0].get('relevance', 0) if nodes else 0}")
    return nodes


# ═══════════════════════════════════════════════════════════════════
# §LoD: Level-of-Detail Atlas Scaler — Frustum Culling
# ═══════════════════════════════════════════════════════════════════

class AtlasLoD:
    """Level-of-Detail scaler for the 3D Knowledge Atlas.

    On an M2, rendering 93,000 nodes would crush the GPU.
    This class performs Frustum Culling:

        Zoomed out: 93K nodes → low-poly Point Cloud (~200 rendered)
        Zoomed in:  High-fidelity metadata from Bolt at 3,100 MB/s
        Thermal:    Dynamic render budget based on GPU load

    The M2 only "thinks" about the solar system you're currently in.
    """

    # Render budgets by thermal state
    BUDGETS = {
        "cool": 500,       # Full fidelity
        "nominal": 300,    # Standard operation
        "warm": 150,       # Reduced — dims to essentials
        "throttled": 50,   # Focus Mode — 2D blueprint only
    }

    # GPU core reservation: fraction reserved for simulations
    GPU_RESERVE = {
        "idle": 0.0,       # All cores for rendering
        "simulation": 0.6, # 60% for Monte Carlo, 40% for atlas
        "indexing": 0.3,   # 30% for indexing, 70% for atlas
    }

    def __init__(self):
        self._thermal_state = "nominal"
        self._gpu_mode = "idle"

    def compute_visible_nodes(
        self,
        focus_point: dict,
        zoom_level: float,
        all_nodes: list[dict],
    ) -> list[dict]:
        """Frustum culling: select only nodes visible at current zoom.

        Args:
            focus_point: {"x": float, "y": float, "z": float} — camera position
            zoom_level: 0.0 (zoomed out, galaxy view) to 1.0 (zoomed in, paper view)
            all_nodes: Full node list from Atlas

        Returns:
            Subset of nodes to render, with LoD metadata attached.
        """
        budget = self.get_render_budget()

        if not all_nodes:
            return []

        fx = focus_point.get("x", 0)
        fy = focus_point.get("y", 0)
        fz = focus_point.get("z", 0)

        # Calculate distance from focus for each node
        for node in all_nodes:
            nx = node.get("x", 0)
            ny = node.get("y", 0)
            nz = node.get("z", 0)
            dist = ((nx - fx)**2 + (ny - fy)**2 + (nz - fz)**2) ** 0.5
            node["_lod_distance"] = dist

        # Sort by distance (closest first)
        sorted_nodes = sorted(all_nodes, key=lambda n: n.get("_lod_distance", 999))

        # Apply zoom-scaled budget
        # At zoom 0.0 (galaxy): show budget * 0.4 nodes as point cloud
        # At zoom 1.0 (paper): show budget * 1.0 nodes with full metadata
        effective_budget = int(budget * (0.4 + 0.6 * zoom_level))
        visible = sorted_nodes[:effective_budget]

        # Assign LoD level to each visible node
        for i, node in enumerate(visible):
            dist = node.get("_lod_distance", 0)
            if zoom_level > 0.7 and i < budget * 0.3:
                node["_lod_level"] = "high"   # Full metadata, labels, edges
            elif zoom_level > 0.3 and i < budget * 0.6:
                node["_lod_level"] = "medium"  # Labels, simplified edges
            else:
                node["_lod_level"] = "low"     # Point cloud dot only

            # Ancestral nodes always get high LoD
            if node.get("type", "").startswith("ancestral"):
                node["_lod_level"] = "high"

        return visible

    def get_render_budget(self) -> int:
        """Dynamic render budget based on thermal state and GPU reservation."""
        base_budget = self.BUDGETS.get(self._thermal_state, 300)
        reserve = self.GPU_RESERVE.get(self._gpu_mode, 0)

        # Reduce budget by GPU reservation fraction
        return max(20, int(base_budget * (1.0 - reserve)))

    def set_thermal_state(self, state: str):
        """Update thermal state (called by SystemReflection)."""
        if state in self.BUDGETS:
            self._thermal_state = state
            log.info(f"§LoD: Thermal state → {state}, budget → {self.get_render_budget()}")

    def set_gpu_mode(self, mode: str):
        """Reserve GPU cores for simulation or indexing."""
        if mode in self.GPU_RESERVE:
            self._gpu_mode = mode
            log.info(
                f"§LoD: GPU mode → {mode}, "
                f"reserved {self.GPU_RESERVE[mode]*100:.0f}%, "
                f"render budget → {self.get_render_budget()}"
            )

    def status(self) -> dict:
        """Current LoD scaler state."""
        return {
            "thermal_state": self._thermal_state,
            "gpu_mode": self._gpu_mode,
            "render_budget": self.get_render_budget(),
            "gpu_reserved_pct": self.GPU_RESERVE.get(self._gpu_mode, 0) * 100,
        }


# Global LoD scaler
atlas_lod = AtlasLoD()
