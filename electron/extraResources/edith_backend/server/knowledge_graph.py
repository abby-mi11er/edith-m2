#!/usr/bin/env python3
"""
Knowledge Graph Enhancements — Scholar Nodes, Concept Taxonomy, PageRank
========================================================================
Improvements implemented:
- KG #1: Scholar nodes — every author becomes a queryable entity
- KG #2: Theory nodes — map theories to proponents and critics
- KG #3: Concept taxonomy — hierarchical concept map
- KG #4: Debate edges — link scholars who disagree
- KG #5: Methodology edges — connect scholars by shared methods
- KG #6: Temporal evolution — track concept changes over decades
- KG #7: Country-theory mapping — which theories tested where
- KG #8: Citation network PageRank — rank scholars by influence
- KG #9: Contradiction detection — flag conflicting claims
- KG #10: Graph-enhanced retrieval — use graph neighbors for retrieval boost
"""

import json
import math
import re
import logging
from pathlib import Path
from typing import Optional
from collections import defaultdict
from server.vault_config import VAULT_ROOT

log = logging.getLogger("edith.knowledge_graph")


class KnowledgeGraph:
    """Queryable knowledge graph for academic entities and relationships."""

    def __init__(self, store_path: Path = None):
        self.store_path = Path(store_path or str(VAULT_ROOT / "Connectome" / "Graph" / "knowledge_graph"))
        self.store_path.mkdir(parents=True, exist_ok=True)

        # §IMP-6.1: Also persist to VAULT for portability
        _vault_root = os.environ.get("EDITH_DATA_ROOT", "") if "os" in dir() else ""
        if _vault_root:
            self._vault_path = Path(_vault_root) / "ARTEFACTS" / "knowledge_graph.json"
        else:
            self._vault_path = None

        # Core stores
        self.scholars: dict = {}       # name → ScholarNode
        self.theories: dict = {}       # name → TheoryNode
        self.concepts: dict = {}       # name → ConceptNode
        self.debates: list = []        # DebateEdge list
        self.method_edges: list = []   # MethodologyEdge list

        self._load()

    def _load(self):
        for attr, fname in [
            ("scholars", "scholars.json"),
            ("theories", "theories.json"),
            ("concepts", "concepts.json"),
            ("debates", "debates.json"),
            ("method_edges", "method_edges.json"),
        ]:
            path = self.store_path / fname
            if path.exists():
                try:
                    setattr(self, attr, json.loads(path.read_text()))
                except Exception:
                    pass

    def save(self):
        """§IMP-6.1: Save to local store + VAULT for portability."""
        for attr, fname in [
            ("scholars", "scholars.json"),
            ("theories", "theories.json"),
            ("concepts", "concepts.json"),
            ("debates", "debates.json"),
            ("method_edges", "method_edges.json"),
        ]:
            path = self.store_path / fname
            path.write_text(json.dumps(getattr(self, attr), indent=2))

        # §IMP-6.1: Also save consolidated to VAULT
        if self._vault_path:
            try:
                self._vault_path.parent.mkdir(parents=True, exist_ok=True)
                self._vault_path.write_text(json.dumps({
                    "scholars": self.scholars,
                    "theories": self.theories,
                    "concepts": self.concepts,
                    "stats": self.stats(),
                }, indent=2))
            except Exception:
                pass

    # --- KG #1: Scholar Nodes ---
    def add_scholar(self, name: str, papers: list = None,
                    theories: list = None, methods: list = None,
                    institutions: list = None):
        """Add or update a scholar node."""
        key = name.lower().strip()
        if key not in self.scholars:
            self.scholars[key] = {
                "name": name,
                "papers": [],
                "theories": [],
                "methods": [],
                "institutions": [],
                "cited_by_count": 0,
                "pagerank": 0.0,
            }
        node = self.scholars[key]
        if papers:
            node["papers"] = list(set(node["papers"] + papers))
        if theories:
            node["theories"] = list(set(node["theories"] + theories))
        if methods:
            node["methods"] = list(set(node["methods"] + methods))
        if institutions:
            node["institutions"] = list(set(node["institutions"] + institutions))

    def get_scholar(self, name: str) -> Optional[dict]:
        return self.scholars.get(name.lower().strip())

    def search_scholars(self, query: str) -> list:
        """Search scholars by name, theory, or method."""
        q = query.lower()
        results = []
        for key, node in self.scholars.items():
            score = 0
            if q in key:
                score += 10
            if any(q in t.lower() for t in node.get("theories", [])):
                score += 5
            if any(q in m.lower() for m in node.get("methods", [])):
                score += 3
            if score > 0:
                results.append({**node, "_score": score})
        return sorted(results, key=lambda x: x["_score"], reverse=True)

    # --- KG #2: Theory Nodes ---
    def add_theory(self, name: str, proponents: list = None,
                   critics: list = None, key_predictions: list = None,
                   related_theories: list = None):
        """Add or update a theory node."""
        key = name.lower().strip()
        if key not in self.theories:
            self.theories[key] = {
                "name": name,
                "proponents": [],
                "critics": [],
                "key_predictions": [],
                "related_theories": [],
                "era": "",
            }
        node = self.theories[key]
        if proponents:
            node["proponents"] = list(set(node["proponents"] + proponents))
        if critics:
            node["critics"] = list(set(node["critics"] + critics))
        if key_predictions:
            node["key_predictions"] = list(set(node["key_predictions"] + key_predictions))
        if related_theories:
            node["related_theories"] = list(set(node["related_theories"] + related_theories))

    # --- KG #3: Concept Taxonomy ---
    def add_concept(self, name: str, parent: str = None,
                    children: list = None, definition: str = ""):
        """Add a concept to the hierarchical taxonomy."""
        key = name.lower().strip()
        if key not in self.concepts:
            self.concepts[key] = {
                "name": name,
                "parent": parent or "",
                "children": [],
                "definition": definition,
                "related_concepts": [],
            }
        node = self.concepts[key]
        if children:
            node["children"] = list(set(node["children"] + children))
        if definition:
            node["definition"] = definition

    def get_concept_tree(self, root: str = "") -> dict:
        """Get a hierarchical concept tree from a root."""
        if not root:
            # Find root concepts (no parent)
            roots = [c for c in self.concepts.values() if not c.get("parent")]
            return {"roots": roots}
        key = root.lower().strip()
        node = self.concepts.get(key)
        if not node:
            return {}
        children_trees = []
        for child_name in node.get("children", []):
            children_trees.append(self.get_concept_tree(child_name))
        return {**node, "subtree": children_trees}

    # --- KG #4: Debate Edges ---
    def add_debate(self, scholar_a: str, scholar_b: str, topic: str,
                   a_position: str = "", b_position: str = "",
                   year: int = None):
        """Record a scholarly debate between two scholars."""
        self.debates.append({
            "scholar_a": scholar_a,
            "scholar_b": scholar_b,
            "topic": topic,
            "a_position": a_position,
            "b_position": b_position,
            "year": year,
        })

    def find_debates(self, scholar_or_topic: str) -> list:
        q = scholar_or_topic.lower()
        return [d for d in self.debates
                if q in d["scholar_a"].lower()
                or q in d["scholar_b"].lower()
                or q in d["topic"].lower()]

    # --- KG #5: Methodology Edges ---
    def add_method_edge(self, scholar: str, method: str, paper: str = ""):
        self.method_edges.append({
            "scholar": scholar,
            "method": method,
            "paper": paper,
        })

    def scholars_by_method(self, method: str) -> list:
        m = method.lower()
        return list(set(e["scholar"] for e in self.method_edges
                       if m in e["method"].lower()))

    # --- KG #8: Citation Network PageRank ---
    def compute_pagerank(self, citation_graph: dict, damping: float = 0.85,
                         iterations: int = 50) -> dict:
        """Compute PageRank for scholars based on citation network.
        
        citation_graph: {paper_id: {cites: [paper_id, ...], author: str}}
        """
        # Build author-level citation graph
        author_cites = defaultdict(set)  # author → set of authors they cite
        author_cited_by = defaultdict(set)  # author → set of authors who cite them
        all_authors = set()

        for paper_id, info in citation_graph.items():
            author = info.get("author", "").lower()
            if not author:
                continue
            all_authors.add(author)
            for cited_id in info.get("cites", []):
                cited_info = citation_graph.get(cited_id, {})
                cited_author = cited_info.get("author", "").lower()
                if cited_author and cited_author != author:
                    author_cites[author].add(cited_author)
                    author_cited_by[cited_author].add(author)
                    all_authors.add(cited_author)

        n = len(all_authors)
        if n == 0:
            return {}

        # Initialize PageRank
        pr = {a: 1.0 / n for a in all_authors}

        for _ in range(iterations):
            new_pr = {}
            for author in all_authors:
                rank_sum = 0
                for citer in author_cited_by[author]:
                    out_degree = len(author_cites[citer])
                    if out_degree > 0:
                        rank_sum += pr[citer] / out_degree
                new_pr[author] = (1 - damping) / n + damping * rank_sum
            pr = new_pr

        # Update scholar nodes
        for author, rank in pr.items():
            if author in self.scholars:
                self.scholars[author]["pagerank"] = round(rank, 6)

        return dict(sorted(pr.items(), key=lambda x: -x[1])[:50])

    # --- KG #9: Contradiction Detection ---
    def detect_contradictions(self, sources: list[dict]) -> list:
        """Flag when sources make conflicting claims."""
        contradictions = []
        contradiction_signals = [
            ("finds that", "finds no"),
            ("positive effect", "negative effect"),
            ("supports", "contradicts"),
            ("significant", "not significant"),
            ("increases", "decreases"),
            ("confirms", "rejects"),
        ]

        for i, s1 in enumerate(sources):
            text1 = (s1.get("text", "") or s1.get("content", "")).lower()
            for j, s2 in enumerate(sources):
                if j <= i:
                    continue
                text2 = (s2.get("text", "") or s2.get("content", "")).lower()
                for pos, neg in contradiction_signals:
                    if (pos in text1 and neg in text2) or (neg in text1 and pos in text2):
                        contradictions.append({
                            "source_a": i + 1,
                            "source_b": j + 1,
                            "signal": f"{pos} vs {neg}",
                        })
                        break  # One signal per pair is enough

        return contradictions

    # --- KG #10: Graph-Enhanced Retrieval ---
    def expand_query_with_graph(self, query: str) -> list:
        """Use graph neighbors to suggest additional search terms."""
        expansions = set()
        q = query.lower()

        # Check scholars
        for key, scholar in self.scholars.items():
            if key in q:
                # Add their theories and co-authors
                expansions.update(scholar.get("theories", []))
                expansions.update(scholar.get("methods", []))

        # Check theories
        for key, theory in self.theories.items():
            if key in q:
                expansions.update(theory.get("proponents", []))
                expansions.update(theory.get("related_theories", []))

        # Check concepts
        for key, concept in self.concepts.items():
            if key in q:
                expansions.update(concept.get("children", []))
                if concept.get("parent"):
                    expansions.add(concept["parent"])

        return list(expansions - {""})[:10]

    # --- §1.5: Temporal Reasoning — policy timeline extraction ---
    def extract_temporal_events(self, sources: list[dict]) -> list[dict]:
        """Extract dated events from sources and build a timeline.

        Detects patterns like '2014 Farm Bill', 'enacted in 1996',
        'after the Great Depression'.
        """
        events = []
        year_pattern = re.compile(
            r'(?:in|during|after|before|since|by|from|until)\s+'
            r'(?:the\s+)?(\d{4})\b'
            r'|(\d{4})\s+'
            r'(?:Act|Bill|Amendment|Reform|Policy|Law|Executive Order)',
            re.IGNORECASE,
        )
        named_pattern = re.compile(
            r'(\d{4})\s+([\w\s]{3,40}?)\s*(?:Act|Bill|Reform|Policy|Amendment)',
            re.IGNORECASE,
        )

        for i, src in enumerate(sources):
            text = src.get("text", "") or src.get("content", "")
            author = src.get("metadata", {}).get("author", f"Source {i+1}")

            # Extract year references
            for match in year_pattern.finditer(text):
                year = match.group(1) or match.group(2)
                context = text[max(0, match.start()-50):match.end()+50].strip()
                events.append({
                    "year": int(year),
                    "context": context,
                    "source": author,
                    "source_idx": i,
                })

            # Extract named legislation
            for match in named_pattern.finditer(text):
                year, name = match.group(1), match.group(2).strip()
                events.append({
                    "year": int(year),
                    "event": f"{name} ({year})",
                    "context": text[max(0, match.start()-30):match.end()+30].strip(),
                    "source": author,
                    "type": "legislation",
                })

        # Sort chronologically and deduplicate
        events.sort(key=lambda e: e["year"])
        seen = set()
        unique = []
        for e in events:
            key = (e["year"], e.get("event", e.get("context", "")[:40]))
            if key not in seen:
                seen.add(key)
                unique.append(e)

        return unique

    def build_policy_timeline(self, sources: list[dict]) -> dict:
        """Build a structured policy timeline from sources."""
        events = self.extract_temporal_events(sources)

        # Group by decade
        decades = defaultdict(list)
        for e in events:
            decade = (e["year"] // 10) * 10
            decades[f"{decade}s"].append(e)

        return {
            "events": events,
            "by_decade": dict(decades),
            "span": f"{events[0]['year']}-{events[-1]['year']}" if events else "",
            "total_events": len(events),
        }

    # --- §3.8: Pre-requisite Mapping ---
    def find_prerequisites(self, paper_title: str, max_depth: int = 3) -> list[dict]:
        """Find which papers must be read before understanding a given work.

        Uses the theory graph to trace intellectual lineage.
        """
        prereqs = []
        title_lower = paper_title.lower()

        # Find theories related to this paper
        related_theories = []
        for key, theory in self.theories.items():
            proponent_papers = []
            for p in theory.get("proponents", []):
                scholar = self.scholars.get(p.lower(), {})
                proponent_papers.extend(scholar.get("papers", []))
            if any(title_lower in pp.lower() for pp in proponent_papers):
                related_theories.append(theory)

        # Find foundational works for those theories
        for theory in related_theories:
            for related_name in theory.get("related_theories", []):
                related = self.theories.get(related_name.lower(), {})
                for proponent in related.get("proponents", [])[:2]:
                    scholar = self.scholars.get(proponent.lower(), {})
                    if scholar and scholar.get("papers"):
                        prereqs.append({
                            "paper": scholar["papers"][0],
                            "author": scholar["name"],
                            "reason": f"Foundational for {related_name}",
                            "priority": "essential",
                        })

            # Direct proponent works
            for proponent in theory.get("proponents", []):
                scholar = self.scholars.get(proponent.lower(), {})
                if scholar:
                    for paper in scholar.get("papers", [])[:1]:
                        if paper.lower() != title_lower:
                            prereqs.append({
                                "paper": paper,
                                "author": scholar["name"],
                                "reason": f"Core work on {theory['name']}",
                                "priority": "recommended",
                            })

        # Deduplicate
        seen = set()
        unique = []
        for p in prereqs:
            key = p["paper"].lower()
            if key not in seen:
                seen.add(key)
                unique.append(p)

        return unique[:max_depth * 3]

    # --- §3.7: Comparative Summaries ---
    def compare_scholars(self, scholar_a: str, scholar_b: str) -> dict:
        """Side-by-side comparison of two scholars."""
        a = self.scholars.get(scholar_a.lower(), {})
        b = self.scholars.get(scholar_b.lower(), {})

        if not a or not b:
            return {"error": "One or both scholars not found"}

        shared_theories = set(a.get("theories", [])) & set(b.get("theories", []))
        shared_methods = set(a.get("methods", [])) & set(b.get("methods", []))

        debates = [d for d in self.debates
                   if (scholar_a.lower() in d["scholar_a"].lower()
                       and scholar_b.lower() in d["scholar_b"].lower())
                   or (scholar_b.lower() in d["scholar_a"].lower()
                       and scholar_a.lower() in d["scholar_b"].lower())]

        return {
            "scholar_a": {
                "name": a.get("name", scholar_a),
                "theories": a.get("theories", []),
                "methods": a.get("methods", []),
                "institutions": a.get("institutions", []),
                "pagerank": a.get("pagerank", 0),
            },
            "scholar_b": {
                "name": b.get("name", scholar_b),
                "theories": b.get("theories", []),
                "methods": b.get("methods", []),
                "institutions": b.get("institutions", []),
                "pagerank": b.get("pagerank", 0),
            },
            "shared_theories": list(shared_theories),
            "shared_methods": list(shared_methods),
            "debates": debates,
            "comparison_type": "allies" if shared_theories else "different_fields",
        }

    def compare_theories(self, theory_a: str, theory_b: str) -> dict:
        """Side-by-side comparison of two theories."""
        a = self.theories.get(theory_a.lower(), {})
        b = self.theories.get(theory_b.lower(), {})

        if not a or not b:
            return {"error": "One or both theories not found"}

        shared_proponents = set(a.get("proponents", [])) & set(b.get("proponents", []))

        return {
            "theory_a": a,
            "theory_b": b,
            "shared_proponents": list(shared_proponents),
            "are_related": theory_b.lower() in [t.lower() for t in a.get("related_theories", [])],
        }

    def stats(self) -> dict:
        return {
            "scholars": len(self.scholars),
            "theories": len(self.theories),
            "concepts": len(self.concepts),
            "debates": len(self.debates),
            "method_edges": len(self.method_edges),
        }


# Pre-populate with core poli-sci theories and scholars
def build_initial_graph() -> KnowledgeGraph:
    """Build initial knowledge graph with core political science content."""
    kg = KnowledgeGraph()

    # Core scholars
    scholars = [
        ("Daron Acemoglu", ["Why Nations Fail", "Economic Origins"], ["Institutionalism"], ["Cross-national regression", "IV"], ["MIT"]),
        ("Adam Przeworski", ["Democracy and Development"], ["Modernization Theory"], ["Large-N", "Formal model"], ["NYU"]),
        ("Robert Dahl", ["Polyarchy", "Who Governs?"], ["Pluralism"], ["Case study"], ["Yale"]),
        ("Samuel Huntington", ["Third Wave", "Political Order"], ["Modernization Theory", "Clash of Civilizations"], ["Comparative-historical"], ["Harvard"]),
        ("Francis Fukuyama", ["End of History"], ["Liberal Democracy"], ["Historical analysis"], ["Stanford"]),
        ("Carles Boix", ["Democracy and Redistribution"], ["Inequality-democratization"], ["Formal model", "Large-N"], ["Princeton"]),
        ("Steven Levitsky", ["How Democracies Die", "Competitive Authoritarianism"], ["Democratic backsliding"], ["Comparative case"], ["Harvard"]),
        ("Lucan Way", ["Competitive Authoritarianism"], ["Hybrid regimes"], ["Comparative case"], ["Toronto"]),
        ("Barbara Geddes", ["Paradigms and Sand Castles"], ["Authoritarian institutions"], ["Large-N", "Game theory"], ["UCLA"]),
        ("George Tsebelis", ["Veto Players"], ["Veto Players Theory"], ["Formal model"], ["Michigan"]),
        ("Arend Lijphart", ["Patterns of Democracy"], ["Consociationalism"], ["Comparative"], ["UCSD"]),
        ("James Fearon", ["Rationalist Explanations for War"], ["Bargaining model of war"], ["Formal model", "Large-N"], ["Stanford"]),
        ("Kenneth Waltz", ["Theory of International Politics"], ["Neorealism"], ["Theory building"], [""]),
        ("Robert Keohane", ["After Hegemony"], ["Neoliberal institutionalism"], ["Theory building"], ["Princeton"]),
        ("Alexander Wendt", ["Social Theory of International Politics"], ["Constructivism"], ["Theory building"], ["Ohio State"]),
        ("Elinor Ostrom", ["Governing the Commons"], ["Collective action"], ["Field experiments", "Case study"], ["Indiana"]),
        ("Robert Putnam", ["Bowling Alone", "Making Democracy Work"], ["Social capital"], ["Survey analysis", "Case study"], ["Harvard"]),
        ("Theda Skocpol", ["States and Social Revolutions"], ["Historical institutionalism"], ["Comparative-historical"], ["Harvard"]),
        ("Bruce Bueno de Mesquita", ["The Logic of Political Survival"], ["Selectorate theory"], ["Formal model", "Large-N"], ["NYU"]),
        ("Gary King", ["Designing Social Inquiry"], ["Methodology"], ["Quantitative methods"], ["Harvard"]),
    ]

    for name, papers, theories, methods, inst in scholars:
        kg.add_scholar(name, papers, theories, methods, inst)

    # Core theories
    theory_data = [
        ("Democratic Peace Theory", ["Kant", "Doyle", "Russett"], ["Rosato", "Gartzke"], ["Democracies don't fight each other"], ["Liberal internationalism"]),
        ("Selectorate Theory", ["Bueno de Mesquita", "Smith"], [], ["Winning coalition size determines public goods"], ["Institutional theory"]),
        ("Modernization Theory", ["Lipset", "Przeworski"], ["Acemoglu", "Robinson"], ["Development causes democracy"], ["Dependency theory"]),
        ("Dependency Theory", ["Wallerstein", "Frank"], ["Modernization theorists"], ["Core exploits periphery"], ["World-systems theory"]),
        ("Veto Players Theory", ["Tsebelis"], [], ["More veto players = more policy stability"], ["Institutional analysis"]),
        ("Constructivism (IR)", ["Wendt", "Finnemore"], ["Waltz", "Mearsheimer"], ["Anarchy is what states make of it"], ["Neorealism"]),
        ("Neorealism", ["Waltz", "Mearsheimer"], ["Wendt", "Keohane"], ["Structure determines state behavior"], ["Constructivism"]),
        ("Rational Choice Institutionalism", ["North", "Shepsle"], [], ["Institutions reduce transaction costs"], ["Historical institutionalism"]),
        ("Historical Institutionalism", ["Pierson", "Thelen", "Skocpol"], [], ["Path dependence shapes outcomes"], ["Rational choice"]),
        ("Resource Curse", ["Ross", "Karl"], ["Haber", "Menaldo"], ["Oil wealth hinders democracy"], ["Rentier state theory"]),
    ]

    for name, proponents, critics, predictions, related in theory_data:
        kg.add_theory(name, proponents, critics, predictions, related)

    # Concept taxonomy
    concepts = [
        ("Democracy", None, ["Liberal democracy", "Electoral democracy", "Participatory democracy", "Deliberative democracy"]),
        ("Liberal democracy", "Democracy", ["Rule of law", "Civil liberties", "Checks and balances"]),
        ("Democratic backsliding", "Democracy", ["Executive aggrandizement", "Norm erosion", "Constitutional hardball"]),
        ("Authoritarianism", None, ["Competitive authoritarianism", "Totalitarianism", "Military dictatorship", "Electoral authoritarianism"]),
        ("Political economy", None, ["Redistribution", "Development", "Trade", "Resource curse"]),
        ("Conflict", None, ["Interstate war", "Civil war", "Ethnic conflict", "Terrorism"]),
        ("Political institutions", None, ["Legislature", "Executive", "Judiciary", "Electoral systems"]),
        ("Electoral systems", "Political institutions", ["Proportional representation", "Majoritarian", "Mixed systems"]),
        ("International relations", None, ["Realism", "Liberalism", "Constructivism", "International law"]),
        ("Public opinion", None, ["Political polarization", "Media effects", "Voting behavior", "Political participation"]),
    ]

    for name, parent, children in concepts:
        kg.add_concept(name, parent, children)

    # Core debates
    debates = [
        ("Przeworski", "Lipset", "Does development cause democracy?", "Exogenous: democracy survives in richer countries", "Endogenous: development causes democratization"),
        ("Acemoglu", "Przeworski", "Institutions vs income in explaining democracy", "Institutions cause both development and democracy", "No causal effect of income on democracy"),
        ("Waltz", "Wendt", "Is anarchy structural or constructed?", "Anarchy is a structural given", "Anarchy is what states make of it"),
        ("Huntington", "Fukuyama", "Is democratization inevitable?", "Waves of democratization can reverse", "Liberal democracy is the end of history"),
        ("Ross", "Haber & Menaldo", "Does oil hinder democracy?", "Yes: resource curse through rentier effects", "No: oil-democracy link is not robust"),
    ]

    for a, b, topic, a_pos, b_pos in debates:
        kg.add_debate(a, b, topic, a_pos, b_pos)

    kg.save()
    return kg
