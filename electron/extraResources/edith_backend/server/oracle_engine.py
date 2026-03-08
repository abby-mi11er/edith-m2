"""
Oracle Engine — The "Why Machine"
==================================
The transformation from "search engine" to "The Oracle."
Cross-domain synthesis, proactive heartbeat monitoring,
adversarial causal search, and the Causal Graph Generator.

Capabilities:
  1. Cross-Domain Synthesis — find theoretical equivalences across silos
  2. Heartbeat Monitor — ArXiv, SSRN, Federal Register, news feeds
  3. Adversarial Causal Search — attack data from 1,000 angles
  4. Causal Graph Generator — scan every claim, build visual map
"""

import hashlib
import json
import logging
import os
import random
import re
import statistics
import threading
import time
from collections import defaultdict
from datetime import datetime
from typing import Optional
from urllib.request import urlopen, Request as URLRequest
from urllib.error import URLError

log = logging.getLogger("edith.oracle")


# ═══════════════════════════════════════════════════════════════════
# §1: CROSS-DOMAIN SYNTHESIS ORACLES — Silo-Breaker
# "Your Texas charities are mathematically identical to
#  Clientelistic Linkage models in 1990s Mexico."
# ═══════════════════════════════════════════════════════════════════

# Theoretical bridge mappings — known cross-domain equivalences
THEORETICAL_BRIDGES = {
    "charity_welfare_substitution": {
        "source_field": "American Political Economy",
        "target_field": "Comparative Political Economy",
        "bridge": "Clientelistic Linkage",
        "mechanism": (
            "Non-state actors providing quasi-governmental services in exchange "
            "for political loyalty — functionally identical to clientelism"
        ),
        "key_authors": {
            "source": ["Mettler", "Campbell", "Hacker"],
            "target": ["Stokes", "Kitschelt", "Lessing", "Arias"],
        },
        "implication": (
            "Framing US charity-welfare dynamics as 'American Clientelism' "
            "bridges a 30-year literature gap between Americanist and "
            "Comparativist subfields"
        ),
    },
    "submerged_state_institutional_change": {
        "source_field": "American Political Development",
        "target_field": "Institutional Economics",
        "bridge": "Institutional Layering",
        "mechanism": (
            "Mettler's 'submerged' policies function like Mahoney & Thelen's "
            "'layering' — new institutions are stacked on top of old ones, "
            "making the system opaque"
        ),
        "key_authors": {
            "source": ["Mettler", "Howard"],
            "target": ["Mahoney", "Thelen", "Pierson", "North"],
        },
        "implication": (
            "The 'submerged state' is a special case of institutional layering "
            "unique to American tax expenditure politics"
        ),
    },
    "policy_feedback_path_dependence": {
        "source_field": "Public Policy",
        "target_field": "Historical Institutionalism",
        "bridge": "Path Dependence",
        "mechanism": (
            "Policy feedback loops (Campbell, Pierson) are a political science "
            "instance of Arthur's increasing returns — once a policy creates "
            "beneficiaries, dismantling it triggers lock-in effects"
        ),
        "key_authors": {
            "source": ["Campbell", "Pierson", "Skocpol"],
            "target": ["Arthur", "David", "North"],
        },
        "implication": (
            "Welfare state retrenchment follows identical dynamics to "
            "technological path dependence (QWERTY, VHS)"
        ),
    },
    "criminal_governance_state_failure": {
        "source_field": "Criminal Governance",
        "target_field": "Failed States Literature",
        "bridge": "Parallel Governance",
        "mechanism": (
            "Arias and Lessing's 'criminal governance' maps directly onto "
            "Reno's 'warlord politics' — non-state armed actors providing "
            "order in exchange for extraction"
        ),
        "key_authors": {
            "source": ["Arias", "Lessing", "Trejo"],
            "target": ["Reno", "Mampilly", "Kalyvas"],
        },
        "implication": (
            "Mexican cartel governance can be analyzed using Sierra Leone / "
            "Colombia frameworks, opening comparative case studies"
        ),
    },
    "voter_turnout_collective_action": {
        "source_field": "Voting Behavior",
        "target_field": "Collective Action Theory",
        "bridge": "Expressive Utility",
        "mechanism": (
            "The 'paradox of voting' dissolves under group-based utility "
            "models (Feddersen, Riker), connecting to Ostrom's commons "
            "governance and selective incentives"
        ),
        "key_authors": {
            "source": ["Aldrich", "Riker", "Feddersen"],
            "target": ["Ostrom", "Olson", "Hardin"],
        },
        "implication": (
            "Voter mobilization by charities functions as a 'selective "
            "incentive' — identical to Olson's logic of collective action"
        ),
    },
}


def find_synthesis_bridges(
    topic: str,
    sources: list[dict] = None,
    model_chain: list[str] = None,
) -> dict:
    """Find cross-domain theoretical equivalences.

    The "Silo-Breaker" — discovers connections across subfields
    that no single specialist would notice.
    """
    model_chain = model_chain or [os.environ.get("EDITH_ORACLE_MODEL", "gemini-2.5-pro")]

    # First pass: check known bridges
    topic_lower = topic.lower()
    matched_bridges = []
    for key, bridge in THEORETICAL_BRIDGES.items():
        relevance = sum(1 for term in [
            bridge["source_field"].lower(),
            bridge["target_field"].lower(),
            bridge["bridge"].lower(),
        ] if any(w in topic_lower for w in term.split()))

        if relevance > 0:
            matched_bridges.append({**bridge, "match_key": key, "relevance": relevance})

    # Second pass: LLM synthesis for novel bridges
    try:
        from server.backend_logic import generate_text_via_chain

        source_text = ""
        if sources:
            source_text = "\n".join(
                s.get("text", "")[:200] for s in sources[:10]
            )

        prompt = (
            f"RESEARCH TOPIC: {topic}\n\n"
            f"SOURCE MATERIAL:\n{source_text[:2000]}\n\n"
            f"Find theoretical equivalences across academic subfields.\n"
            f"For each bridge, provide:\n"
            f"1. SOURCE FIELD: The field this topic belongs to\n"
            f"2. TARGET FIELD: A seemingly unrelated field\n"
            f"3. BRIDGE: The theoretical concept that connects them\n"
            f"4. MECHANISM: Why these are mathematically/logically equivalent\n"
            f"5. KEY AUTHORS: Who to cite on each side\n"
            f"6. DISRUPTION: How framing this as a 'bridge' disrupts both fields\n\n"
            f"Find at least 3 bridges. Prioritize connections that no one "
            f"in either subfield has noticed."
        )
        synthesis, model = generate_text_via_chain(
            prompt, model_chain,
            system_instruction=(
                "You are a theoretical synthesizer who finds hidden connections "
                "across academic disciplines. You read American politics, "
                "comparative politics, economics, sociology, and philosophy "
                "simultaneously. Find the bridges no specialist would see."
            ),
            temperature=0.3,
        )

        return {
            "topic": topic,
            "known_bridges": matched_bridges,
            "novel_synthesis": synthesis,
            "model": model,
        }
    except Exception as e:
        return {
            "topic": topic,
            "known_bridges": matched_bridges,
            "error": str(e),
        }


# ═══════════════════════════════════════════════════════════════════
# §2: HEARTBEAT MONITOR — Proactive Research Intelligence
# "A new paper using your model was uploaded to ArXiv..."
# ═══════════════════════════════════════════════════════════════════

class HeartbeatMonitor:
    """Proactive monitoring of academic feeds and policy changes.

    Watches: ArXiv, SSRN, Federal Register, SocArXiv
    Alerts when relevant new work appears.
    """

    FEEDS = {
        "arxiv_econ": {
            "url": "https://export.arxiv.org/api/query?search_query=cat:econ.GN&max_results=10&sortBy=submittedDate",
            "parser": "_parse_arxiv",
            "interval_s": 3600,
        },
        "arxiv_cs_ai": {
            "url": "https://export.arxiv.org/api/query?search_query=cat:cs.AI+AND+all:policy&max_results=5&sortBy=submittedDate",
            "parser": "_parse_arxiv",
            "interval_s": 3600,
        },
        "federal_register": {
            "url": "https://www.federalregister.gov/api/v1/documents.json?conditions%5Bagencies%5D%5B%5D=food-and-nutrition-service&per_page=5&order=newest",
            "parser": "_parse_fed_register",
            "interval_s": 7200,
        },
        "openalex_welfare": {
            "url": "https://api.openalex.org/works?filter=default.search:welfare+policy+feedback&sort=publication_date:desc&per_page=5",
            "parser": "_parse_openalex",
            "interval_s": 7200,
        },
    }

    def __init__(self, watch_topics: list[str] = None):
        self._topics = watch_topics or [
            "SNAP", "welfare", "submerged state", "policy feedback",
            "voter turnout", "charity", "rural politics", "criminal governance",
        ]
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._alerts: list[dict] = []
        self._last_check: dict[str, float] = {}
        self._lock = threading.Lock()
        # §IMP-3.4: Persist alerts to VAULT
        self._alerts_path = os.path.join(
            os.environ.get("EDITH_DATA_ROOT", "."), ".edith_heartbeat_alerts.json"
        )
        self._load_persisted_alerts()

    def _load_persisted_alerts(self):
        """§IMP-3.4: Load previously saved alerts."""
        try:
            if os.path.exists(self._alerts_path):
                with open(self._alerts_path) as f:
                    self._alerts = json.load(f)
                log.info(f"§HEARTBEAT: Loaded {len(self._alerts)} persisted alerts")
        except Exception:
            pass

    def _persist_alerts(self):
        """§IMP-3.4: Save alerts to disk."""
        try:
            with open(self._alerts_path, "w") as f:
                json.dump(self._alerts[-100:], f, indent=2)  # Keep last 100
        except Exception:
            pass

    def start(self) -> dict:
        if self._running:
            return {"status": "already_running"}
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        log.info("§ORACLE: Heartbeat monitor started")
        return {"status": "started", "feeds": list(self.FEEDS.keys())}

    def stop(self) -> dict:
        self._running = False
        return {"status": "stopped"}

    def check_now(self) -> dict:
        """Force an immediate check of all feeds."""
        results = {}
        for feed_name, config in self.FEEDS.items():
            try:
                items = self._fetch_feed(feed_name, config)
                relevant = self._filter_relevant(items)
                results[feed_name] = {
                    "items_found": len(items),
                    "relevant": len(relevant),
                    "alerts": relevant,
                }
                for item in relevant:
                    with self._lock:
                        self._alerts.append(item)
            except Exception as e:
                results[feed_name] = {"error": str(e)}

        return results

    def get_alerts(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return list(self._alerts[-limit:])

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "feeds": list(self.FEEDS.keys()),
            "topics": self._topics,
            "alerts_count": len(self._alerts),
            "last_check": self._last_check,
        }

    def _monitor_loop(self):
        while self._running:
            for feed_name, config in self.FEEDS.items():
                last = self._last_check.get(feed_name, 0)
                if time.time() - last > config["interval_s"]:
                    try:
                        items = self._fetch_feed(feed_name, config)
                        relevant = self._filter_relevant(items)
                        for item in relevant:
                            with self._lock:
                                self._alerts.append(item)
                        self._last_check[feed_name] = time.time()
                    except Exception as e:
                        log.debug(f"§ORACLE: Feed {feed_name} error: {e}")
            time.sleep(60)

    def _fetch_feed(self, name: str, config: dict) -> list[dict]:
        """Fetch and parse a feed."""
        try:
            req = URLRequest(config["url"], headers={"User-Agent": "EDITH/2.0"})
            with urlopen(req, timeout=15) as resp:
                data = resp.read().decode("utf-8", errors="ignore")

            parser = config["parser"]
            if parser == "_parse_arxiv":
                return self._parse_arxiv(data)
            elif parser == "_parse_fed_register":
                return self._parse_fed_register(data)
            elif parser == "_parse_openalex":
                return self._parse_openalex(data)
        except Exception as e:
            return []

    def _parse_arxiv(self, xml_text: str) -> list[dict]:
        items = []
        entries = re.findall(r'<entry>(.*?)</entry>', xml_text, re.DOTALL)
        for entry in entries:
            title = re.search(r'<title>(.*?)</title>', entry, re.DOTALL)
            summary = re.search(r'<summary>(.*?)</summary>', entry, re.DOTALL)
            link = re.search(r'<id>(.*?)</id>', entry)
            published = re.search(r'<published>(.*?)</published>', entry)
            items.append({
                "source": "arxiv",
                "title": title.group(1).strip() if title else "",
                "abstract": summary.group(1).strip()[:300] if summary else "",
                "url": link.group(1).strip() if link else "",
                "date": published.group(1)[:10] if published else "",
                "timestamp": datetime.now().isoformat(),
            })
        return items

    def _parse_fed_register(self, json_text: str) -> list[dict]:
        items = []
        try:
            data = json.loads(json_text)
            for doc in data.get("results", []):
                items.append({
                    "source": "federal_register",
                    "title": doc.get("title", ""),
                    "abstract": doc.get("abstract", "")[:300],
                    "url": doc.get("html_url", ""),
                    "date": doc.get("publication_date", ""),
                    "agency": doc.get("agencies", [{}])[0].get("name", "") if doc.get("agencies") else "",
                    "timestamp": datetime.now().isoformat(),
                })
        except Exception:
            pass
        return items

    def _parse_openalex(self, json_text: str) -> list[dict]:
        items = []
        try:
            data = json.loads(json_text)
            for work in data.get("results", []):
                items.append({
                    "source": "openalex",
                    "title": work.get("title", ""),
                    "abstract": "",
                    "url": work.get("doi", work.get("id", "")),
                    "date": work.get("publication_date", ""),
                    "cited_by": work.get("cited_by_count", 0),
                    "timestamp": datetime.now().isoformat(),
                })
        except Exception:
            pass
        return items

    def _filter_relevant(self, items: list[dict]) -> list[dict]:
        """Filter items for relevance to watch topics."""
        relevant = []
        for item in items:
            text = f"{item.get('title', '')} {item.get('abstract', '')}".lower()
            matched_topics = [t for t in self._topics if t.lower() in text]
            if matched_topics:
                item["matched_topics"] = matched_topics
                item["relevance_score"] = len(matched_topics) / len(self._topics)
                relevant.append(item)
        return relevant


heartbeat_monitor = HeartbeatMonitor()


# ═══════════════════════════════════════════════════════════════════
# §3: ADVERSARIAL CAUSAL SEARCH — Attack from 1,000 Angles
# "Is the charity→SNAP relationship real or a fluke?"
# ═══════════════════════════════════════════════════════════════════

ADVERSARIAL_ATTACKS = [
    {
        "name": "Omitted Variable Bias",
        "description": "What unobserved confounder could create this relationship?",
        "prompt_fragment": "Identify 5 plausible confounders that could create a spurious relationship between {cause} and {effect}.",
    },
    {
        "name": "Reverse Causality",
        "description": "Does {effect} actually cause {cause}?",
        "prompt_fragment": "Construct a compelling argument that {effect} actually causes {cause}, not the other way around.",
    },
    {
        "name": "Selection Bias",
        "description": "Is the sample systematically non-random?",
        "prompt_fragment": "Identify how selection into the sample could create a false {cause}→{effect} relationship.",
    },
    {
        "name": "Measurement Error",
        "description": "Are {cause} and {effect} mismeasured?",
        "prompt_fragment": "How could measurement error in either {cause} or {effect} bias estimates of their relationship?",
    },
    {
        "name": "Ecological Fallacy",
        "description": "Does county-level correlation hold at individual level?",
        "prompt_fragment": "Could {cause}→{effect} at the aggregate level disappear when measured at the individual level?",
    },
    {
        "name": "Hawthorne Effect",
        "description": "Is observation altering behavior?",
        "prompt_fragment": "Could awareness of the study change how agents behave with respect to {cause} and {effect}?",
    },
    {
        "name": "Functional Form Misspecification",
        "description": "Is the relationship actually nonlinear?",
        "prompt_fragment": "Could the {cause}→{effect} relationship be nonlinear (U-shaped, threshold, logarithmic) rather than linear?",
    },
    {
        "name": "Temporal Precedence",
        "description": "Does {cause} precede {effect} in time?",
        "prompt_fragment": "Could the timing be wrong? Could {effect} have already been changing before {cause} was introduced?",
    },
    {
        "name": "SUTVA Violation",
        "description": "Are units interfering with each other?",
        "prompt_fragment": "Could spillover effects between counties/individuals violate the Stable Unit Treatment Value Assumption?",
    },
    {
        "name": "Publication Bias",
        "description": "Would null results have been published?",
        "prompt_fragment": "If {cause} had no effect on {effect}, would researchers have published? Estimate the file-drawer problem.",
    },
]


def adversarial_causal_search(
    cause: str,
    effect: str,
    data_summary: str = "",
    n_attacks: int = 10,
    model_chain: list[str] = None,
) -> dict:
    """Attack a causal claim from N angles.

    The "1,000 Angles" search — systematically tries to destroy
    your causal claim. If it survives, you have real evidence.
    """
    model_chain = model_chain or [os.environ.get("EDITH_ORACLE_MODEL", "gemini-2.5-pro")]
    t0 = time.time()

    attacks = ADVERSARIAL_ATTACKS[:min(n_attacks, len(ADVERSARIAL_ATTACKS))]
    results = []

    for attack in attacks:
        prompt = (
            f"CAUSAL CLAIM: {cause} → {effect}\n"
            f"DATA: {data_summary[:500]}\n\n"
            f"ATTACK: {attack['name']}\n"
            f"{attack['prompt_fragment'].format(cause=cause, effect=effect)}\n\n"
            f"Rate the severity of this threat (1-10) and explain specifically "
            f"how it applies to this claim. If it doesn't apply, say so."
        )

        try:
            from server.backend_logic import generate_text_via_chain
            response, model = generate_text_via_chain(
                prompt, model_chain,
                system_instruction=(
                    "You are a causal inference adversary. Your job is to find "
                    "every possible way this causal claim could be wrong. "
                    "Be specific and cite methodology literature."
                ),
                temperature=0.15,
            )

            # Extract severity rating
            severity_match = re.search(r'(\d+)\s*/\s*10|severity[:\s]+(\d+)', response.lower())
            severity = int(severity_match.group(1) or severity_match.group(2)) if severity_match else 5

            results.append({
                "attack": attack["name"],
                "description": attack["description"].format(cause=cause, effect=effect),
                "severity": severity,
                "analysis": response[:500],
                "threatens_claim": severity >= 7,
            })
        except Exception as e:
            results.append({
                "attack": attack["name"],
                "severity": 5,
                "error": str(e),
            })

    # Aggregate
    survived = sum(1 for r in results if not r.get("threatens_claim", False))
    mean_severity = statistics.mean([r.get("severity", 5) for r in results])
    max_threat = max(results, key=lambda r: r.get("severity", 0))

    return {
        "claim": f"{cause} → {effect}",
        "attacks_run": len(results),
        "survived": survived,
        "failed": len(results) - survived,
        "mean_severity": round(mean_severity, 1),
        "robustness_score": round(survived / max(len(results), 1) * 100, 1),
        "verdict": (
            "ROBUST ✅ — claim survived adversarial search"
            if survived >= len(results) * 0.7
            else f"VULNERABLE ⚠️ — Biggest threat: {max_threat['attack']}"
        ),
        "biggest_threat": max_threat["attack"],
        "results": results,
        "elapsed_s": round(time.time() - t0, 2),
    }


# ═══════════════════════════════════════════════════════════════════
# §4: CAUSAL GRAPH GENERATOR — Scan Everything, Map Everything
# "Every cause-and-effect claim you've ever recorded."
# ═══════════════════════════════════════════════════════════════════

def generate_causal_graph_from_library(
    chroma_dir: str = "",
    collection_name: str = "edith",
    embed_model: str = "",
    sample_size: int = 500,
    model_chain: list[str] = None,
) -> dict:
    """Scan the entire library and build a visual causal map.

    This is the "Foundation of the Oracle" — every cause→effect
    claim across your 93K chunks, mapped and weighted.
    """
    chroma_dir = chroma_dir or os.environ.get("EDITH_CHROMA_DIR", "")
    embed_model = embed_model or os.environ.get("EDITH_EMBED_MODEL", "")
    t0 = time.time()

    # Retrieve a sample of chunks from ChromaDB
    sources = []
    try:
        from server.chroma_backend import retrieve_local_sources
        # Use broad queries to sample the corpus
        sample_queries = [
            "causes", "leads to", "results in", "policy feedback",
            "voter turnout", "welfare state", "causal mechanism",
            "treatment effect", "independent variable",
            "dependent variable", "regression results",
        ]
        for q in sample_queries:
            try:
                results = retrieve_local_sources(
                    queries=[q],
                    chroma_dir=chroma_dir,
                    collection_name=collection_name,
                    embed_model=embed_model,
                    top_k=sample_size // len(sample_queries),
                )
                sources.extend(results)
            except Exception:
                pass
    except ImportError:
        pass

    if not sources:
        return {
            "error": "No sources retrieved — check ChromaDB connection",
            "fallback": "Use /api/causal/graph with manual sources instead",
        }

    # Deduplicate
    seen = set()
    unique_sources = []
    for s in sources:
        text = s.get("text", "")[:100]
        if text not in seen:
            seen.add(text)
            unique_sources.append(s)

    # Build causal graph
    from server.causal_engine import build_causal_graph
    graph = build_causal_graph(unique_sources)

    # Enrich with field classifications
    field_keywords = {
        "APE": ["welfare", "snap", "food stamp", "poverty", "inequality", "social policy"],
        "Voting": ["voter", "turnout", "election", "partisan", "democrat", "republican", "ballot"],
        "Methods": ["regression", "causal", "instrument", "treatment", "ols", "fixed effect"],
        "Criminal": ["criminal", "cartel", "violence", "extortion", "governance", "gang"],
        "Comparative": ["comparative", "regime", "democracy", "authorit", "clientel"],
    }

    for edge in graph.get("edges", []):
        edge_text = f"{edge['cause']} {edge['effect']}".lower()
        fields = []
        for field, keywords in field_keywords.items():
            if any(k in edge_text for k in keywords):
                fields.append(field)
        edge["fields"] = fields or ["General"]

    # Create Three.js-ready node/edge data
    nodes_3d = []
    for i, node in enumerate(graph.get("nodes", [])):
        # Position nodes in 3D space by field cluster
        node_text = node.lower()
        cluster = "General"
        for field, keywords in field_keywords.items():
            if any(k in node_text for k in keywords):
                cluster = field
                break

        cluster_offsets = {
            "APE": (0, 0, 0), "Voting": (50, 20, 0),
            "Methods": (0, 50, 20), "Criminal": (-50, 0, 20),
            "Comparative": (25, -30, 40), "General": (0, 0, 30),
        }
        base = cluster_offsets.get(cluster, (0, 0, 0))

        nodes_3d.append({
            "id": node,
            "label": node[:30],
            "cluster": cluster,
            "x": base[0] + random.uniform(-20, 20),
            "y": base[1] + random.uniform(-20, 20),
            "z": base[2] + random.uniform(-20, 20),
            "size": 1,
        })

    # Scale node sizes by connectivity
    node_connections = defaultdict(int)
    for edge in graph.get("edges", []):
        node_connections[edge["cause"]] += edge["evidence_count"]
        node_connections[edge["effect"]] += edge["evidence_count"]

    for n in nodes_3d:
        n["size"] = min(10, 1 + node_connections.get(n["id"], 0))

    edges_3d = [
        {
            "source": e["cause"],
            "target": e["effect"],
            "weight": e["weight"],
            "evidence": e["evidence_count"],
            "fields": e.get("fields", []),
        }
        for e in graph.get("edges", [])
    ]

    elapsed = time.time() - t0
    return {
        "nodes": nodes_3d,
        "edges": edges_3d,
        "total_nodes": len(nodes_3d),
        "total_edges": len(edges_3d),
        "total_claims": graph.get("total_claims_found", 0),
        "sources_scanned": len(unique_sources),
        "field_distribution": dict(
            sorted(
                defaultdict(int, {
                    f: sum(1 for e in edges_3d if f in e.get("fields", []))
                    for f in field_keywords
                }).items(),
                key=lambda x: -x[1]
            )
        ),
        "elapsed_s": round(elapsed, 2),
        "format": "three_js_ready",
    }


# ═══════════════════════════════════════════════════════════════════
# §5: THEORETICAL GAP DETECTOR — Find What's Missing
# ═══════════════════════════════════════════════════════════════════

def detect_theoretical_gaps(
    causal_graph: dict,
    model_chain: list[str] = None,
) -> dict:
    """Analyze a causal graph to find theoretical gaps.

    Gaps are: isolated clusters, weak bridges, missing mechanisms,
    and untested counterfactuals.
    """
    model_chain = model_chain or [os.environ.get("EDITH_ORACLE_MODEL", "gemini-2.5-pro")]

    edges = causal_graph.get("edges", [])
    nodes = causal_graph.get("nodes", [])

    # Find isolated nodes (low connectivity)
    node_degree = defaultdict(int)
    for e in edges:
        src = e.get("source", e.get("cause", ""))
        tgt = e.get("target", e.get("effect", ""))
        node_degree[src] += 1
        node_degree[tgt] += 1

    isolated = [n.get("id", n) if isinstance(n, dict) else n
                for n in nodes
                if node_degree.get(n.get("id", n) if isinstance(n, dict) else n, 0) <= 1]

    # Find weak bridges (low-evidence edges connecting different clusters)
    weak_bridges = [
        e for e in edges
        if e.get("evidence", e.get("evidence_count", 0)) <= 1
        and e.get("weight", 0) < 0.5
    ]

    # Find cross-cluster gaps
    clusters = defaultdict(set)
    for n in nodes:
        if isinstance(n, dict):
            clusters[n.get("cluster", "General")].add(n.get("id", ""))

    cross_cluster_edges = defaultdict(int)
    for e in edges:
        src_cluster = next(
            (c for c, ns in clusters.items()
             if e.get("source", e.get("cause", "")) in ns),
            "General"
        )
        tgt_cluster = next(
            (c for c, ns in clusters.items()
             if e.get("target", e.get("effect", "")) in ns),
            "General"
        )
        if src_cluster != tgt_cluster:
            key = f"{src_cluster}↔{tgt_cluster}"
            cross_cluster_edges[key] += 1

    # Generate gap analysis
    try:
        from server.backend_logic import generate_text_via_chain
        prompt = (
            f"CAUSAL GRAPH ANALYSIS:\n"
            f"- {len(nodes)} nodes, {len(edges)} edges\n"
            f"- Isolated concepts: {isolated[:10]}\n"
            f"- Weak bridges: {len(weak_bridges)}\n"
            f"- Cross-cluster connections: {dict(cross_cluster_edges)}\n\n"
            f"Identify the 5 most important theoretical gaps:\n"
            f"1. What mechanisms are missing?\n"
            f"2. What connections should exist but don't?\n"
            f"3. What's been assumed but never tested?\n"
            f"4. What counterfactuals would be most illuminating?"
        )
        analysis, model = generate_text_via_chain(
            prompt, model_chain,
            system_instruction=(
                "You are a theoretical gap analyst for political science. "
                "Find the missing connections that would produce breakthrough papers."
            ),
            temperature=0.3,
        )
    except Exception:
        analysis = "Gap analysis requires LLM connection."
        model = "none"

    return {
        "isolated_concepts": isolated[:20],
        "weak_bridges": len(weak_bridges),
        "cross_cluster_connections": dict(cross_cluster_edges),
        "cluster_sizes": {c: len(ns) for c, ns in clusters.items()},
        "gap_analysis": analysis,
    }


# ═══════════════════════════════════════════════════════════════════
# §6: COMMITTEE PUSHBACK GENERATOR
# "Here are the 10 most likely objections from your committee."
# ═══════════════════════════════════════════════════════════════════

def generate_committee_pushback(
    thesis: str,
    committee_members: list[str] = None,
    model_chain: list[str] = None,
) -> dict:
    """Generate the 10 most likely "Pushback" arguments from your committee.

    Each committee member attacks from their specialty.
    """
    model_chain = model_chain or [os.environ.get("EDITH_ORACLE_MODEL", "gemini-2.5-pro")]
    committee = committee_members or [
        "Methodologist (Angrist-school causal inference)",
        "Americanist (Mettler-school submerged state)",
        "Comparativist (Levitsky-school institutional analysis)",
        "Formal Theorist (game theory, rational choice)",
        "Area Specialist (Texas/Southern politics)",
    ]

    prompt = (
        f"DISSERTATION THESIS: {thesis}\n\n"
        f"COMMITTEE:\n"
        + "\n".join(f"- {m}" for m in committee)
        + "\n\n"
        f"For each committee member, generate their 2 toughest questions.\n"
        f"These should be the questions that would make a PhD student sweat.\n"
        f"Format as:\n"
        f"**[Role]**: Question\n"
        f"Then suggest how to pre-empt each objection."
    )

    try:
        from server.backend_logic import generate_text_via_chain
        pushback, model = generate_text_via_chain(
            prompt, model_chain,
            system_instruction=(
                "You are simulating a PhD dissertation committee. Each member "
                "is a world expert who will find the weakest link in the argument. "
                "Be ruthless but constructive."
            ),
            temperature=0.3,
        )
        return {
            "thesis": thesis,
            "committee": committee,
            "pushback": pushback,
            "model": model,
        }
    except Exception as e:
        return {"error": str(e)}
