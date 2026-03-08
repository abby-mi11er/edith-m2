"""
Graph Vector Engine — Agentic Graph-RAG Learning Infrastructure
=================================================================
The "Brain" that turns passive search into active knowledge.

This upgrades the Citadel from:
  - Passive RAG (waiting for you to ask) →
  - Agentic Graph-RAG (proactively building a web of political theory)

Three layers work together:
1. VECTORIZE — Turn every paragraph into mathematical coordinates (embeddings)
2. GRAPH — Extract entities & relationships into a knowledge graph
3. LEARN — RLHF-Lite: learn from your corrections and note-taking paths

Architecture on M4 + Bolt:
    PDF/Notion → OCR/Parse → Chunk → Embed (MLX/HuggingFace) →
    ChromaDB (Bolt) → Entity Extraction → Knowledge Graph (NetworkX) →
    Cross-Pollination Audit → Conflict Reports → Atlas Update
"""

import hashlib
import json
import logging
import math
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.graph_vector_engine")


# ═══════════════════════════════════════════════════════════════════
# Entity & Relationship Types — The Political Science Ontology
# ═══════════════════════════════════════════════════════════════════

ENTITY_TYPES = {
    "THEORY": {
        "patterns": [
            r"(principal[- ]agent\s+theory)", r"(administrative\s+burden\s+theory)",
            r"(policy\s+feedback)", r"(state\s+capacity)", r"(welfare\s+state\s+theory)",
            r"(institutional\s+choice)", r"(street[- ]level\s+bureaucracy)",
            r"(democratic\s+erosion)", r"(rational\s+choice)", r"(new\s+public\s+management)",
            r"(collective\s+action\s+theory)", r"(social\s+capital\s+theory)",
        ],
        "color": "#FFD700",
    },
    "METHOD": {
        "patterns": [
            r"\b(OLS|RDD|DiD|2SLS|IV regression|fixed effects|random effects)\b",
            r"(propensity\s+score\s+matching)", r"(synthetic\s+control)",
            r"(qualitative\s+comparative\s+analysis|QCA)",
            r"(process\s+tracing)", r"(case\s+study\s+method)",
            r"(multilevel\s+model|HLM)", r"(event\s+study)",
            r"(Bayesian\s+(?:inference|estimation))",
        ],
        "color": "#87CEEB",
    },
    "DATASET": {
        "patterns": [
            r"\b(V-Dem|CPS|ANES|ACS|Census)\b",
            r"(World\s+Development\s+Indicators|WDI)",
            r"(Fragile\s+States\s+Index)", r"(Quality\s+of\s+Government)",
            r"(Correlates\s+of\s+War)", r"(UCDP)",
        ],
        "color": "#90EE90",
    },
    "AUTHOR": {
        "patterns": [
            r"\b(Mettler|Aldrich|Moynihan|Pierson|Skocpol|Lipsky|Ostrom|Tilly)\b",
            r"\b(Esping-Andersen|Levitsky|Fukuyama|Mann|Besley|Acemoglu)\b",
            r"\b(Herd|Campbell|Soss|Maynard-Mooney|North|Williamson)\b",
        ],
        "color": "#DDA0DD",
    },
    "CONCEPT": {
        "patterns": [
            r"(moral\s+hazard)", r"(adverse\s+selection)", r"(path\s+dependence)",
            r"(crowding\s+out)", r"(administrative\s+burden)", r"(take[- ]up\s+rates?)",
            r"(blame\s+(?:avoidance|diffusion))", r"(accountability)",
            r"(devolution)", r"(privatization)", r"(decommodification)",
            r"(infrastructural\s+power)", r"(extractive\s+capacity)",
        ],
        "color": "#FFA07A",
    },
    "PLACE": {
        "patterns": [
            r"\b(Potter\s+County|Lubbock|Texas|Mexico|Michoacán)\b",
            r"\b(United\s+States|Latin\s+America|Sub-Saharan\s+Africa)\b",
        ],
        "color": "#98FB98",
    },
}

RELATIONSHIP_PATTERNS = [
    # Causal
    (r"(\w[\w\s]+)\s+(?:causes?|leads?\s+to|results?\s+in|produces?)\s+(\w[\w\s]+)", "CAUSES"),
    (r"(\w[\w\s]+)\s+(?:reduces?|decreases?|undermines?|weakens?)\s+(\w[\w\s]+)", "REDUCES"),
    (r"(\w[\w\s]+)\s+(?:increases?|strengthens?|enhances?|promotes?)\s+(\w[\w\s]+)", "INCREASES"),
    # Theoretical
    (r"(\w[\w\s]+)\s+(?:argues?|claims?|contends?|proposes?)\s+(?:that\s+)?(\w[\w\s]+)", "ARGUES"),
    (r"(\w[\w\s]+)\s+(?:builds?\s+on|extends?|draws?\s+on)\s+(\w[\w\s]+)", "EXTENDS"),
    (r"(\w[\w\s]+)\s+(?:contradicts?|challenges?|critiques?)\s+(\w[\w\s]+)", "CONTRADICTS"),
    # Data
    (r"(\w[\w\s]+)\s+(?:uses?|employs?|analyzes?)\s+(?:data\s+from\s+)?(\w[\w\s]+)", "USES_DATA"),
    (r"(\w[\w\s]+)\s+(?:measures?|operationalizes?)\s+(\w[\w\s]+)", "MEASURES"),
    # Structural
    (r"(\w[\w\s]+)\s+(?:provides?|delivers?|supplies?)\s+(\w[\w\s]+)", "PROVIDES"),
    (r"(\w[\w\s]+)\s+(?:replaces?|substitutes?\s+for)\s+(\w[\w\s]+)", "REPLACES"),
]


@dataclass
class Entity:
    """A node in the knowledge graph."""
    entity_id: str
    name: str
    entity_type: str
    mentions: int = 1
    sources: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.entity_id,
            "name": self.name,
            "type": self.entity_type,
            "mentions": self.mentions,
            "sources": self.sources[:5],
        }


@dataclass
class Relationship:
    """An edge in the knowledge graph."""
    source_id: str
    target_id: str
    rel_type: str
    weight: float = 1.0
    evidence: list[str] = field(default_factory=list)
    user_confirmed: bool = False

    def to_dict(self) -> dict:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.rel_type,
            "weight": round(self.weight, 2),
            "confirmed": self.user_confirmed,
        }


@dataclass
class TextChunk:
    """A chunk of text ready for vectorization."""
    chunk_id: str
    text: str
    source_file: str
    page: int = 0
    entities: list[str] = field(default_factory=list)
    embedding: list[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.chunk_id,
            "text": self.text[:200],
            "source": self.source_file,
            "entities": self.entities,
            "has_embedding": len(self.embedding) > 0,
        }


# ═══════════════════════════════════════════════════════════════════
# The Graph Vector Engine
# ═══════════════════════════════════════════════════════════════════

class GraphVectorEngine:
    """Combined Knowledge Graph + Vector Engine.

    Three modes of operation:
    1. INGEST — Process new documents into chunks, embeddings, and graph nodes
    2. QUERY — Semantic search + graph traversal for deep retrieval
    3. LEARN — Update weights based on user feedback (RLHF-Lite)

    Usage:
        engine = GraphVectorEngine()
        engine.ingest_document("path/to/paper.pdf")
        results = engine.query("How does charity substitution affect state capacity?")
        engine.record_feedback("node_123", "confirmed")
    """

    def __init__(self, bolt_path: str = ""):
        try:
            from server.vault_config import VAULT_ROOT
            _default = str(VAULT_ROOT)
        except ImportError:
            _default = os.environ.get("EDITH_DATA_ROOT", ".")
        self._bolt_path = bolt_path or _default
        self._chroma_dir = os.environ.get(
            "EDITH_CHROMA_DIR",
            os.path.join(self._bolt_path, "VAULT", "CHROMA_DB")
        )

        # Knowledge Graph (in-memory, persisted to Bolt)
        self._entities: dict[str, Entity] = {}
        self._relationships: list[Relationship] = []
        self._chunks: dict[str, TextChunk] = {}

        # RLHF-Lite: user feedback weights
        self._feedback_log: list[dict] = []
        self._link_weights: dict[str, float] = {}  # edge_key → weight

        # Load existing graph
        self._load_graph()

    # ─── INGEST: Document → Chunks → Embeddings → Graph ──────────

    def ingest_document(self, file_path: str, source_label: str = "") -> dict:
        """Ingest a document into both the vector store and knowledge graph.

        This is the full pipeline:
        1. Read and chunk the text
        2. Extract entities from each chunk
        3. Extract relationships between entities
        4. Generate embeddings (if embedding model available)
        5. Store in ChromaDB + update graph
        """
        t0 = time.time()
        path = Path(file_path)
        source = source_label or path.stem

        # Step 1: Read text
        try:
            if path.suffix.lower() == ".pdf":
                from server.forensic_audit import extract_text_from_pdf
                text = extract_text_from_pdf(file_path)
            else:
                text = path.read_text(errors="ignore")
        except Exception as e:
            return {"error": f"Cannot read {file_path}: {e}"}

        if len(text) < 50:
            return {"error": "Insufficient text extracted"}

        # Step 2: Chunk the text
        chunks = self._chunk_text(text, source, chunk_size=500, overlap=50)

        # Step 3: Extract entities from each chunk
        all_entities = {}
        for chunk in chunks:
            chunk_entities = self._extract_entities(chunk.text, source)
            chunk.entities = [e.entity_id for e in chunk_entities]
            for e in chunk_entities:
                if e.entity_id in all_entities:
                    all_entities[e.entity_id].mentions += 1
                    if source not in all_entities[e.entity_id].sources:
                        all_entities[e.entity_id].sources.append(source)
                else:
                    all_entities[e.entity_id] = e

        # Step 4: Extract relationships
        relationships = self._extract_relationships(text, source)

        # Step 5: Generate embeddings (lightweight, no external deps required)
        embedded_count = self._embed_chunks(chunks)

        # Step 6: Store in ChromaDB
        chroma_result = self._store_in_chroma(chunks)

        # Step 7: Update knowledge graph
        for eid, entity in all_entities.items():
            if eid in self._entities:
                self._entities[eid].mentions += entity.mentions
                for s in entity.sources:
                    if s not in self._entities[eid].sources:
                        self._entities[eid].sources.append(s)
            else:
                self._entities[eid] = entity

        self._relationships.extend(relationships)

        # Store chunks
        for chunk in chunks:
            self._chunks[chunk.chunk_id] = chunk

        # Save graph
        self._save_graph()

        elapsed = time.time() - t0
        return {
            "source": source,
            "chunks_created": len(chunks),
            "entities_extracted": len(all_entities),
            "relationships_found": len(relationships),
            "embeddings_generated": embedded_count,
            "chroma_stored": chroma_result,
            "elapsed_seconds": round(elapsed, 2),
        }

    def _chunk_text(self, text: str, source: str,
                     chunk_size: int = 500, overlap: int = 50) -> list[TextChunk]:
        """Split text into overlapping chunks for vectorization."""
        words = text.split()
        chunks = []
        i = 0
        page_est = 1

        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunk_hash = hashlib.sha256(chunk_text[:100].encode()).hexdigest()[:12]

            chunks.append(TextChunk(
                chunk_id=f"{source}_{chunk_hash}",
                text=chunk_text,
                source_file=source,
                page=page_est,
            ))

            i += chunk_size - overlap
            page_est = i // 800 + 1  # Rough page estimate

        return chunks

    def _extract_entities(self, text: str, source: str) -> list[Entity]:
        """Extract named entities from text using pattern matching."""
        entities = []
        seen = set()

        for entity_type, config in ENTITY_TYPES.items():
            for pattern in config["patterns"]:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    name = match.group(1).strip()
                    eid = f"{entity_type}_{name.lower().replace(' ', '_')[:30]}"

                    if eid in seen:
                        continue
                    seen.add(eid)

                    entities.append(Entity(
                        entity_id=eid,
                        name=name,
                        entity_type=entity_type,
                        mentions=1,
                        sources=[source],
                    ))

        return entities

    def _extract_relationships(self, text: str, source: str) -> list[Relationship]:
        """Extract relationships between entities."""
        relationships = []
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences[:200]:  # Cap at 200 sentences
            for pattern, rel_type in RELATIONSHIP_PATTERNS:
                for match in re.finditer(pattern, sentence, re.IGNORECASE):
                    src = match.group(1).strip()[:50]
                    tgt = match.group(2).strip()[:50]

                    if len(src) < 3 or len(tgt) < 3:
                        continue

                    src_id = f"CONCEPT_{src.lower().replace(' ', '_')[:30]}"
                    tgt_id = f"CONCEPT_{tgt.lower().replace(' ', '_')[:30]}"

                    # Check if these match known entities
                    for eid, entity in self._entities.items():
                        if entity.name.lower() in src.lower():
                            src_id = eid
                        if entity.name.lower() in tgt.lower():
                            tgt_id = eid

                    relationships.append(Relationship(
                        source_id=src_id,
                        target_id=tgt_id,
                        rel_type=rel_type,
                        weight=1.0,
                        evidence=[sentence.strip()[:200]],
                    ))

        return relationships

    def _embed_chunks(self, chunks: list[TextChunk]) -> int:
        """Generate embeddings for chunks.

        Tries in order:
        1. MLX embeddings (M4-native, fastest)
        2. sentence-transformers
        3. Simple TF-IDF-like fallback
        """
        embedded = 0

        # Try MLX first (M4-optimized)
        try:
            from server.mlx_embeddings import embed_texts
            texts = [c.text[:512] for c in chunks]
            embeddings = embed_texts(texts)
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
                embedded += 1
            return embedded
        except ImportError:
            pass

        # Try sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer("all-MiniLM-L6-v2")
            texts = [c.text[:512] for c in chunks]
            embeddings = model.encode(texts).tolist()
            for chunk, emb in zip(chunks, embeddings):
                chunk.embedding = emb
                embedded += 1
            return embedded
        except ImportError:
            pass

        # Fallback: simple bag-of-words hash (not great, but functional)
        for chunk in chunks:
            words = set(chunk.text.lower().split())
            # Create a deterministic pseudo-embedding from word hashes
            pseudo_emb = [0.0] * 64
            for word in words:
                h = int(hashlib.md5(word.encode()).hexdigest()[:8], 16)
                idx = h % 64
                pseudo_emb[idx] += 1.0
            # Normalize
            norm = math.sqrt(sum(x * x for x in pseudo_emb)) or 1.0
            chunk.embedding = [x / norm for x in pseudo_emb]
            embedded += 1

        return embedded

    def _store_in_chroma(self, chunks: list[TextChunk]) -> dict:
        """Store chunks in ChromaDB on the Bolt."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self._chroma_dir)
            collection = client.get_or_create_collection("citadel_graph")

            ids = [c.chunk_id for c in chunks]
            documents = [c.text for c in chunks]
            metadatas = [{"source": c.source_file, "page": c.page,
                          "entities": ",".join(c.entities[:10])} for c in chunks]
            embeddings = [c.embedding for c in chunks if c.embedding]

            if embeddings and len(embeddings) == len(ids):
                collection.add(ids=ids, documents=documents,
                              metadatas=metadatas, embeddings=embeddings)
            else:
                collection.add(ids=ids, documents=documents, metadatas=metadatas)

            return {"stored": len(ids), "collection": "citadel_graph"}
        except Exception as e:
            log.debug(f"ChromaDB store failed: {e}")
            return {"stored": 0, "error": str(e)}

    # ─── QUERY: Semantic Search + Graph Traversal ─────────────────

    def query(self, question: str, top_k: int = 10,
              graph_depth: int = 2) -> dict:
        """Query using both vector search AND graph traversal.

        This is the "Agentic" part: it doesn't just find similar text,
        it follows the causal graph to find related entities.
        """
        # Step 1: Vector search in ChromaDB
        vector_results = self._vector_search(question, top_k)

        # Step 2: Extract entities from the question
        question_entities = self._extract_entities(question, "query")

        # Step 3: Graph traversal from question entities
        graph_context = []
        for entity in question_entities:
            neighbors = self._traverse_graph(entity.entity_id, depth=graph_depth)
            graph_context.extend(neighbors)

        # Step 4: Merge and rank results
        return {
            "question": question,
            "vector_results": vector_results,
            "graph_entities": [e.to_dict() for e in question_entities],
            "graph_context": graph_context[:20],
            "total_entities": len(self._entities),
            "total_relationships": len(self._relationships),
        }

    def _vector_search(self, question: str, top_k: int = 10) -> list[dict]:
        """Search ChromaDB for semantically similar chunks."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self._chroma_dir)
            collection = client.get_or_create_collection("citadel_graph")

            results = collection.query(query_texts=[question], n_results=top_k)

            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            return [
                {"text": doc[:300], "source": meta.get("source", ""),
                 "distance": round(dist, 3)}
                for doc, meta, dist in zip(docs, metas, distances)
            ]
        except Exception:
            return []

    def _traverse_graph(self, entity_id: str, depth: int = 2) -> list[dict]:
        """Traverse the knowledge graph from a starting entity."""
        visited = set()
        results = []

        def _walk(eid: str, current_depth: int):
            if current_depth <= 0 or eid in visited:
                return
            visited.add(eid)

            entity = self._entities.get(eid)
            if entity:
                results.append({
                    "entity": entity.to_dict(),
                    "depth": depth - current_depth,
                })

            # Find connected relationships
            for rel in self._relationships:
                neighbor_id = None
                if rel.source_id == eid:
                    neighbor_id = rel.target_id
                elif rel.target_id == eid:
                    neighbor_id = rel.source_id

                if neighbor_id and neighbor_id not in visited:
                    # Apply RLHF weight
                    edge_key = f"{rel.source_id}→{rel.target_id}"
                    weight = self._link_weights.get(edge_key, rel.weight)

                    if weight > 0.3:  # Only follow strong-enough links
                        results.append({
                            "relationship": rel.to_dict(),
                            "depth": depth - current_depth,
                        })
                        _walk(neighbor_id, current_depth - 1)

        _walk(entity_id, depth)
        return results

    # ─── LEARN: RLHF-Lite from User Feedback ─────────────────────

    def record_feedback(self, entity_or_edge: str, feedback: str,
                         context: str = "") -> dict:
        """Learn from user corrections.

        Types of feedback:
        - "confirmed" — user agrees with this link (boost weight)
        - "rejected" — user says this is wrong (reduce weight)
        - "corrected" — user provides the right answer

        "If you tell Winnie 'No, that's not what Mettler meant,' she
        updates the weight of that theoretical link."
        """
        weight_delta = {
            "confirmed": 0.2,
            "rejected": -0.3,
            "corrected": -0.1,  # Reduced but not eliminated
            "important": 0.4,
        }.get(feedback, 0)

        current_weight = self._link_weights.get(entity_or_edge, 1.0)
        new_weight = max(0.0, min(2.0, current_weight + weight_delta))
        self._link_weights[entity_or_edge] = new_weight

        self._feedback_log.append({
            "target": entity_or_edge,
            "feedback": feedback,
            "context": context[:200],
            "timestamp": time.time(),
            "old_weight": round(current_weight, 2),
            "new_weight": round(new_weight, 2),
        })

        self._save_graph()

        return {
            "recorded": True,
            "target": entity_or_edge,
            "feedback": feedback,
            "weight": round(new_weight, 2),
        }

    def record_note_path(self, from_topic: str, to_topic: str,
                          context: str = "") -> dict:
        """Record when user links two topics (note-taking path learning).

        "When you link a syllabus week to a specific dissertation idea,
        Winnie records that Path."
        """
        from_id = f"CONCEPT_{from_topic.lower().replace(' ', '_')[:30]}"
        to_id = f"CONCEPT_{to_topic.lower().replace(' ', '_')[:30]}"
        edge_key = f"{from_id}→{to_id}"

        # Boost this connection
        current = self._link_weights.get(edge_key, 0.5)
        self._link_weights[edge_key] = min(2.0, current + 0.3)

        # Ensure entities exist
        if from_id not in self._entities:
            self._entities[from_id] = Entity(
                entity_id=from_id, name=from_topic, entity_type="CONCEPT"
            )
        if to_id not in self._entities:
            self._entities[to_id] = Entity(
                entity_id=to_id, name=to_topic, entity_type="CONCEPT"
            )

        # Add relationship if not exists
        existing = any(
            r.source_id == from_id and r.target_id == to_id
            for r in self._relationships
        )
        if not existing:
            self._relationships.append(Relationship(
                source_id=from_id, target_id=to_id,
                rel_type="USER_LINKED", weight=1.3,
                evidence=[context[:200]], user_confirmed=True,
            ))

        self._save_graph()
        return {"linked": True, "from": from_topic, "to": to_topic,
                "weight": round(self._link_weights[edge_key], 2)}

    # ─── Cross-Pollination Audit ──────────────────────────────────

    def cross_pollination_audit(self) -> list[dict]:
        """Weekly audit: check if user's notes are consistent with the literature.

        "Does the data in the 2025 V-Dem update support the user's note
        from Tuesday?"
        """
        conflicts = []

        # Find relationships that contradict each other
        by_target = defaultdict(list)
        for rel in self._relationships:
            by_target[rel.target_id].append(rel)

        for target_id, rels in by_target.items():
            # Check for contradictions: INCREASES vs REDUCES on the same target
            increases = [r for r in rels if r.rel_type == "INCREASES"]
            reduces = [r for r in rels if r.rel_type == "REDUCES"]

            for inc in increases:
                for red in reduces:
                    if inc.source_id != red.source_id:
                        target = self._entities.get(target_id)
                        src_inc = self._entities.get(inc.source_id)
                        src_red = self._entities.get(red.source_id)

                        conflicts.append({
                            "type": "contradiction",
                            "target": target.name if target else target_id,
                            "claim_1": f"{src_inc.name if src_inc else inc.source_id} INCREASES {target.name if target else target_id}",
                            "claim_2": f"{src_red.name if src_red else red.source_id} REDUCES {target.name if target else target_id}",
                            "evidence_1": inc.evidence[:1],
                            "evidence_2": red.evidence[:1],
                            "severity": "high",
                        })

        return conflicts[:10]

    # ─── Graph Persistence ────────────────────────────────────────

    def _save_graph(self):
        """Persist the knowledge graph to the Bolt."""
        graph_dir = Path(self._bolt_path) / "VAULT" / "GRAPH"
        graph_dir.mkdir(parents=True, exist_ok=True)

        graph_data = {
            "entities": {eid: e.to_dict() for eid, e in self._entities.items()},
            "relationships": [r.to_dict() for r in self._relationships[-500:]],
            "link_weights": self._link_weights,
            "feedback_log": self._feedback_log[-100:],
            "saved_at": time.time(),
        }

        graph_file = graph_dir / "knowledge_graph.json"
        try:
            graph_file.write_text(json.dumps(graph_data, indent=2))
        except Exception as e:
            log.warning(f"§GRAPH: Save failed: {e}")

    def _load_graph(self):
        """Load the knowledge graph from the Bolt."""
        graph_file = Path(self._bolt_path) / "VAULT" / "GRAPH" / "knowledge_graph.json"
        if not graph_file.exists():
            return

        try:
            data = json.loads(graph_file.read_text())

            for eid, edata in data.get("entities", {}).items():
                self._entities[eid] = Entity(
                    entity_id=edata["id"], name=edata["name"],
                    entity_type=edata["type"], mentions=edata.get("mentions", 1),
                    sources=edata.get("sources", []),
                )

            for rdata in data.get("relationships", []):
                self._relationships.append(Relationship(
                    source_id=rdata["source"], target_id=rdata["target"],
                    rel_type=rdata["type"], weight=rdata.get("weight", 1.0),
                    user_confirmed=rdata.get("confirmed", False),
                ))

            self._link_weights = data.get("link_weights", {})
            self._feedback_log = data.get("feedback_log", [])

        except Exception as e:
            log.warning(f"§GRAPH: Load failed: {e}")

    # ─── Graph Statistics ─────────────────────────────────────────

    def get_graph_stats(self) -> dict:
        """Get statistics about the knowledge graph."""
        entity_counts = defaultdict(int)
        for e in self._entities.values():
            entity_counts[e.entity_type] += 1

        rel_counts = defaultdict(int)
        for r in self._relationships:
            rel_counts[r.rel_type] += 1

        # Find most connected entities
        connections = defaultdict(int)
        for r in self._relationships:
            connections[r.source_id] += 1
            connections[r.target_id] += 1

        most_connected = sorted(connections.items(), key=lambda x: -x[1])[:10]
        hub_entities = []
        for eid, count in most_connected:
            entity = self._entities.get(eid)
            hub_entities.append({
                "name": entity.name if entity else eid,
                "type": entity.entity_type if entity else "unknown",
                "connections": count,
            })

        return {
            "total_entities": len(self._entities),
            "total_relationships": len(self._relationships),
            "total_chunks": len(self._chunks),
            "entity_types": dict(entity_counts),
            "relationship_types": dict(rel_counts),
            "hub_entities": hub_entities,
            "user_feedback_count": len(self._feedback_log),
            "user_confirmed_links": sum(1 for r in self._relationships if r.user_confirmed),
        }

    @property
    def status(self) -> dict:
        return self.get_graph_stats()


# Global instance
graph_engine = GraphVectorEngine()
