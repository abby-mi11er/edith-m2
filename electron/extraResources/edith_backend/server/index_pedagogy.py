"""
Pedagogical Indexer — "Ancestral Knowledge" Training
=====================================================
Index syllabi and comprehensive exam questions as "Ancestral Nodes"
so Winnie calibrates to your departmental academic standard.

This is LOCAL RAG — your syllabi never touch OpenAI. They stay on the
Bolt, indexed via ChromaDB, and Winnie uses the GPT model only as
a "sophisticated English-speaking engine" to package YOUR knowledge.

Workflow:
    1. Crawl /VAULT/PEDAGOGY/SYLLABI and /VAULT/PEDAGOGY/EXAMS
    2. Extract authors, theories, methods, key terms
    3. Classify exam difficulty (intro/advanced/doctoral)
    4. Index to ChromaDB with source_type: "ancestral" metadata
    5. Query calibrated to specific exam standards
"""

import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

log = logging.getLogger("edith.pedagogy")


# ═══════════════════════════════════════════════════════════════════
# §1: CONCEPT EXTRACTION — Authors, Theories, Methods
# ═══════════════════════════════════════════════════════════════════

# Known political science authors for extraction
_KNOWN_AUTHORS = {
    "mettler", "aldrich", "pierson", "skocpol", "hacker",
    "campbell", "soss", "weaver", "schneider", "ingram",
    "lowi", "wilson", "ostrom", "olson", "downs",
    "kingdon", "baumgartner", "jones", "sabatier",
    "arias", "lessing", "levitsky", "ziblatt",
    "dahl", "lindblom", "schattschneider", "lasswell",
    "putnam", "verba", "schlozman", "brady",
    "key", "mayhew", "fenno", "fiorina",
}

# Methods vocabulary
_METHODS_TERMS = {
    "regression", "ols", "logistic", "probit", "tobit",
    "fixed effects", "random effects", "multi-level",
    "difference-in-differences", "did", "rdd",
    "instrumental variable", "2sls", "iv",
    "matching", "propensity score", "psm",
    "bayesian", "maximum likelihood", "mle",
    "survival analysis", "hazard", "cox",
    "panel data", "time series", "cross-section",
    "qualitative", "case study", "process tracing",
    "content analysis", "discourse analysis",
    "survey", "experiment", "field experiment",
    "quasi-experiment", "natural experiment",
}


def extract_academic_concepts(text: str) -> dict:
    """Extract academic concepts from syllabus or exam text.

    Identifies: authors, theories, methods, key terms, readings.
    """
    text_lower = text.lower()
    words = set(re.findall(r'\b[a-z]+\b', text_lower))

    # 1. Author detection
    authors_found = []
    for author in _KNOWN_AUTHORS:
        if author in text_lower:
            authors_found.append(author.title())

    # Also find capitalized name patterns (Last, Year)
    citation_pattern = re.findall(
        r'([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)*)\s*\(?(\d{4})\)?',
        text,
    )
    for author, year in citation_pattern:
        name = author.strip()
        if name not in authors_found and len(name) > 2:
            authors_found.append(name)

    # 2. Methods detection
    methods_found = []
    for method in _METHODS_TERMS:
        if method in text_lower:
            methods_found.append(method)

    # 3. Theory detection — look for patterns like "X theory" or "theory of X"
    theory_patterns = re.findall(
        r'(?:theory of |the )?(\w+(?:\s+\w+)?)\s+theory',
        text_lower,
    )
    theories = [t.strip().title() for t in theory_patterns if len(t) > 3]

    # Also find "ism" words
    isms = re.findall(r'\b(\w+ism)\b', text_lower)
    theories.extend([i.title() for i in set(isms)])

    # 4. Key terms — high-frequency academic words
    academic_terms = []
    term_patterns = [
        r'(?:dependent|independent|control)\s+variable',
        r'causal\s+\w+', r'endogen\w+', r'exogen\w+',
        r'heteroge\w+', r'counterfactual',
        r'welfare\s+state', r'policy\s+feedback',
        r'administrative\s+burden', r'political\s+participation',
        r'voter\s+turnout', r'redistributi\w+',
    ]
    for pattern in term_patterns:
        matches = re.findall(pattern, text_lower)
        academic_terms.extend(matches)

    # 5. Readings — find bibliography-style entries
    readings = re.findall(
        r'([A-Z][a-z]+,?\s+[A-Z]\.?\s*(?:&\s*[A-Z][a-z]+,?\s+[A-Z]\.?\s*)*'
        r'\(?(?:19|20)\d{2}\)?[^.]*\.)',
        text[:5000],
    )

    return {
        "authors": sorted(set(authors_found)),
        "methods": sorted(set(methods_found)),
        "theories": sorted(set(theories)),
        "key_terms": sorted(set(academic_terms))[:20],
        "readings_count": len(readings),
        "word_count": len(text.split()),
    }


# ═══════════════════════════════════════════════════════════════════
# §2: DIFFICULTY MAPPING — Exam Rigor Classification
# ═══════════════════════════════════════════════════════════════════

def map_exam_difficulty(exam_text: str) -> dict:
    """Classify the rigor level of a comprehensive exam question.

    Levels:
        intro:    Describe/define/explain
        advanced: Compare/analyze/evaluate
        doctoral: Design/critique methodology/synthesize across fields

    Returns: difficulty level, bloom's taxonomy verbs found, depth score
    """
    text_lower = exam_text.lower()

    # Bloom's Taxonomy verb detection
    bloom_levels = {
        "remember": ["define", "list", "identify", "name", "recall", "state"],
        "understand": ["describe", "explain", "summarize", "paraphrase", "discuss"],
        "apply": ["apply", "demonstrate", "use", "solve", "implement"],
        "analyze": ["analyze", "compare", "contrast", "differentiate", "examine"],
        "evaluate": ["evaluate", "assess", "critique", "justify", "defend", "argue"],
        "create": ["design", "construct", "propose", "synthesize", "formulate", "develop"],
    }

    verbs_found = {}
    for level, verbs in bloom_levels.items():
        found = [v for v in verbs if v in text_lower]
        if found:
            verbs_found[level] = found

    # Depth indicators
    depth_signals = {
        "cross_field": bool(re.search(
            r'across\s+(?:fields|disciplines|literatures)|interdisciplin',
            text_lower,
        )),
        "methodology_design": bool(re.search(
            r'design\s+(?:a|an|your)\s+(?:study|research|experiment)',
            text_lower,
        )),
        "literature_synthesis": bool(re.search(
            r'synthesiz|integrat\w+\s+(?:across|multiple|several)',
            text_lower,
        )),
        "causal_reasoning": bool(re.search(
            r'causal|counterfactual|identif\w+\s+strate',
            text_lower,
        )),
        "multiple_parts": len(re.findall(r'\b[a-d]\)|part\s+[a-d]|\d\)', exam_text)) > 2,
    }

    depth_score = sum(depth_signals.values())

    # Classification
    if "create" in verbs_found or "evaluate" in verbs_found or depth_score >= 3:
        difficulty = "doctoral"
    elif "analyze" in verbs_found or depth_score >= 2:
        difficulty = "advanced"
    else:
        difficulty = "intro"

    return {
        "difficulty": difficulty,
        "bloom_verbs": verbs_found,
        "depth_signals": depth_signals,
        "depth_score": depth_score,
        "estimated_answer_pages": (
            8 if difficulty == "doctoral" else
            5 if difficulty == "advanced" else 3
        ),
    }


# ═══════════════════════════════════════════════════════════════════
# §3: PEDAGOGICAL INDEXER — ChromaDB Integration
# ═══════════════════════════════════════════════════════════════════

class PedagogicalIndexer:
    """Crawl PEDAGOGY folder and index as 'Ancestral Nodes' in ChromaDB.

    Ancestral Nodes form the foundation of the Knowledge Atlas —
    they represent your department's academic expectations.

    Usage:
        indexer = PedagogicalIndexer()
        result = indexer.index_pedagogy()
        nodes  = indexer.get_ancestral_nodes()
    """

    COLLECTION_NAME = "pedagogy_ancestral"

    def __init__(self, data_root: str = ""):
        self._data_root = data_root or os.environ.get("EDITH_DATA_ROOT", ".")
        self._pedagogy_dir = os.path.join(self._data_root, "PEDAGOGY")
        self._syllabi_dir = os.path.join(self._pedagogy_dir, "SYLLABI")
        self._exams_dir = os.path.join(self._pedagogy_dir, "EXAMS")
        self._index_state_path = os.path.join(self._pedagogy_dir, ".pedagogy_index.json")

    def crawl_pedagogy(self) -> dict:
        """Phase 1: Find all documents in PEDAGOGY folders."""
        files = {"syllabi": [], "exams": []}
        extensions = {".pdf", ".txt", ".md", ".docx", ".doc", ".rtf",
                      ".tex", ".bib", ".html", ".htm", ".ipynb", ".rmd"}

        for folder, key in [(self._syllabi_dir, "syllabi"), (self._exams_dir, "exams")]:
            if not os.path.isdir(folder):
                log.warning(f"§PEDAGOGY: {key} folder not found: {folder}")
                continue

            for root, dirs, filenames in os.walk(folder):
                for fname in filenames:
                    ext = os.path.splitext(fname)[1].lower()
                    if ext in extensions and not fname.startswith("."):
                        fpath = os.path.join(root, fname)
                        files[key].append({
                            "path": fpath,
                            "name": fname,
                            "type": key,
                            "size_kb": round(os.path.getsize(fpath) / 1024, 1),
                        })

        log.info(
            f"§PEDAGOGY: Found {len(files['syllabi'])} syllabi, "
            f"{len(files['exams'])} exam files"
        )

        return {
            "syllabi_count": len(files["syllabi"]),
            "exams_count": len(files["exams"]),
            "files": files,
        }

    def _read_file_text(self, filepath: str) -> str:
        """Extract text from a file (PDF, txt, md, docx)."""
        ext = os.path.splitext(filepath)[1].lower()

        if ext in (".txt", ".md"):
            try:
                with open(filepath, "r", errors="replace") as f:
                    return f.read()
            except Exception:
                return ""

        if ext == ".pdf":
            try:
                import fitz  # PyMuPDF
                doc = fitz.open(filepath)
                text = ""
                for page in doc:
                    text += page.get_text()
                doc.close()
                return text
            except ImportError:
                # Fallback: try OCR
                try:
                    from server.completions import extract_text_from_image
                    result = extract_text_from_image(filepath)
                    return result.get("text", "")
                except Exception:
                    return ""
            except Exception:
                return ""

        return ""

    def index_pedagogy(self) -> dict:
        """Phase 2-4: Extract, classify, and index all pedagogical content.

        1. Crawl PEDAGOGY folders
        2. Extract text from each file
        3. Extract academic concepts
        4. Classify exam difficulty
        5. Index to ChromaDB with "ancestral" metadata
        """
        crawl = self.crawl_pedagogy()
        all_files = crawl["files"]["syllabi"] + crawl["files"]["exams"]

        if not all_files:
            return {"status": "empty", "reason": "No files found in PEDAGOGY"}

        indexed = []
        concepts_aggregate = {
            "authors": set(),
            "methods": set(),
            "theories": set(),
        }

        for fileinfo in all_files:
            filepath = fileinfo["path"]
            file_type = fileinfo["type"]

            # Extract text
            text = self._read_file_text(filepath)
            if not text or len(text) < 50:
                log.debug(f"§PEDAGOGY: Skipping {fileinfo['name']} — too short")
                continue

            # Extract concepts
            concepts = extract_academic_concepts(text)

            # Map difficulty for exams
            difficulty = None
            if file_type == "exams":
                difficulty = map_exam_difficulty(text)

            # Build metadata
            metadata = {
                "source": filepath,
                "source_type": "ancestral",
                "file_type": file_type,
                "filename": fileinfo["name"],
                "authors": json.dumps(concepts["authors"][:10]),
                "methods": json.dumps(concepts["methods"][:10]),
                "theories": json.dumps(concepts["theories"][:5]),
                "word_count": concepts["word_count"],
                "indexed_at": datetime.now().isoformat(),
            }

            if difficulty:
                metadata["difficulty"] = difficulty["difficulty"]
                metadata["depth_score"] = difficulty["depth_score"]

            # Chunk text for ChromaDB (max ~500 words per chunk)
            chunks = self._chunk_text(text, max_words=500)

            for i, chunk in enumerate(chunks):
                chunk_meta = {**metadata, "chunk_index": i, "total_chunks": len(chunks)}

                # Try to index via ChromaDB
                try:
                    self._index_chunk(chunk, chunk_meta)
                except Exception as e:
                    log.debug(f"§PEDAGOGY: Failed to index chunk: {e}")

            indexed.append({
                "file": fileinfo["name"],
                "type": file_type,
                "concepts": {
                    "authors": len(concepts["authors"]),
                    "methods": len(concepts["methods"]),
                    "theories": len(concepts["theories"]),
                },
                "chunks": len(chunks),
                "difficulty": difficulty["difficulty"] if difficulty else None,
            })

            # Aggregate concepts
            concepts_aggregate["authors"].update(concepts["authors"])
            concepts_aggregate["methods"].update(concepts["methods"])
            concepts_aggregate["theories"].update(concepts["theories"])

        # Save index state
        state = {
            "last_indexed": datetime.now().isoformat(),
            "files_indexed": len(indexed),
            "authors": sorted(concepts_aggregate["authors"]),
            "methods": sorted(concepts_aggregate["methods"]),
            "theories": sorted(concepts_aggregate["theories"]),
        }

        try:
            os.makedirs(os.path.dirname(self._index_state_path), exist_ok=True)
            with open(self._index_state_path, "w") as f:
                json.dump(state, f, indent=2)
        except Exception:
            pass

        log.info(
            f"§PEDAGOGY: Indexed {len(indexed)} files — "
            f"{len(concepts_aggregate['authors'])} authors, "
            f"{len(concepts_aggregate['methods'])} methods"
        )

        return {
            "status": "complete",
            "files_indexed": indexed,
            "aggregate": {
                "authors": sorted(concepts_aggregate["authors"]),
                "methods": sorted(concepts_aggregate["methods"]),
                "theories": sorted(concepts_aggregate["theories"]),
            },
        }

    def _chunk_text(self, text: str, max_words: int = 500) -> list[str]:
        """Split text into overlapping chunks for ChromaDB."""
        words = text.split()
        chunks = []
        overlap = 50

        for i in range(0, len(words), max_words - overlap):
            chunk = " ".join(words[i:i + max_words])
            if len(chunk.strip()) > 20:
                chunks.append(chunk)

        return chunks if chunks else [text[:2000]]

    def _index_chunk(self, text: str, metadata: dict):
        """Index a single chunk to ChromaDB."""
        try:
            from server.chroma_backend import _get_client, _get_embedder

            chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "")
            if not chroma_dir:
                return

            client = _get_client(chroma_dir)
            collection = client.get_or_create_collection(
                name=self.COLLECTION_NAME,
            )

            # Generate a deterministic ID
            import hashlib
            doc_id = hashlib.sha256(
                (metadata.get("source", "") + str(metadata.get("chunk_index", 0))).encode()
            ).hexdigest()[:16]

            collection.upsert(
                documents=[text],
                metadatas=[metadata],
                ids=[doc_id],
            )
        except Exception as e:
            log.debug(f"§PEDAGOGY: ChromaDB indexing failed: {e}")

    def get_ancestral_nodes(self) -> dict:
        """Get all indexed pedagogical concepts for the Atlas."""
        if os.path.exists(self._index_state_path):
            try:
                with open(self._index_state_path) as f:
                    state = json.load(f)

                nodes = []
                for author in state.get("authors", []):
                    nodes.append({
                        "id": f"ancestral_{author.lower().replace(' ', '_')}",
                        "label": author,
                        "type": "ancestral_author",
                        "source": "pedagogy",
                        "mass": 5.0,  # High gravitational mass
                    })

                for theory in state.get("theories", []):
                    nodes.append({
                        "id": f"ancestral_{theory.lower().replace(' ', '_')}",
                        "label": theory,
                        "type": "ancestral_theory",
                        "source": "pedagogy",
                        "mass": 3.0,
                    })

                return {
                    "nodes": nodes,
                    "total": len(nodes),
                    "last_indexed": state.get("last_indexed", ""),
                }
            except Exception:
                pass

        return {"nodes": [], "total": 0}


# ═══════════════════════════════════════════════════════════════════
# §4: EXAM-CALIBRATED QUERYING — "Answer as if for Comp Q3"
# ═══════════════════════════════════════════════════════════════════

def query_as_exam(
    question_ref: str,
    current_text: str,
    model_chain: list[str] = None,
) -> dict:
    """Calibrate Winnie's response to a specific exam question's rigor.

    Usage:
        result = query_as_exam(
            "Question 3 from my 2023 Comprehensive Exam",
            "Explain how policy feedback affects voter turnout..."
        )

    Winnie finds the exam question, analyzes its rigor, and responds
    at that academic standard — not generic AI, but YOUR department.
    """
    model_chain = model_chain or [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]

    # 1. Search for the referenced exam question
    exam_context = ""
    difficulty = None

    try:
        from server.chroma_backend import _get_client

        chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "")
        if chroma_dir:
            client = _get_client(chroma_dir)
            try:
                collection = client.get_collection(PedagogicalIndexer.COLLECTION_NAME)
                results = collection.query(
                    query_texts=[question_ref],
                    n_results=3,
                    where={"file_type": "exams"},
                )

                if results and results["documents"]:
                    exam_context = "\n\n".join(results["documents"][0][:2])
                    difficulty = map_exam_difficulty(exam_context)
            except Exception:
                pass
    except Exception:
        pass

    # 2. Build calibrated prompt
    if not difficulty:
        difficulty = {"difficulty": "doctoral", "depth_score": 3}

    rigor_instruction = {
        "doctoral": (
            "Respond at DOCTORAL COMPREHENSIVE EXAM level. "
            "Synthesize across multiple theoretical traditions. "
            "Cite specific authors and page references. "
            "Address methodological implications. "
            "Anticipate and pre-empt counterarguments."
        ),
        "advanced": (
            "Respond at ADVANCED GRADUATE level. "
            "Compare theoretical perspectives critically. "
            "Cite key authors. Discuss methodological trade-offs."
        ),
        "intro": (
            "Respond at INTRODUCTORY GRADUATE level. "
            "Define terms clearly. Explain core concepts. "
            "Reference foundational authors."
        ),
    }

    calibration = rigor_instruction.get(difficulty["difficulty"], rigor_instruction["doctoral"])

    # 3. Generate calibrated response
    try:
        from server.backend_logic import generate_text_via_chain

        prompt = (
            f"EXAM QUESTION CONTEXT:\n{exam_context[:1000]}\n\n"
            f"CURRENT TEXT TO RESPOND TO:\n{current_text}\n\n"
            f"CALIBRATION: {calibration}\n\n"
            f"Respond to the current text at the rigor level demanded "
            f"by the exam context above. This is an academic response, "
            f"not a casual answer."
        )

        response, model = generate_text_via_chain(
            prompt, model_chain,
            system_instruction=(
                "You are an expert political science advisor. Your response "
                "must match the exact academic rigor of a comprehensive "
                "examination answer in a top-20 political science department."
            ),
            temperature=0.3,
        )

        return {
            "response": response,
            "calibration_level": difficulty["difficulty"],
            "depth_score": difficulty["depth_score"],
            "exam_context_found": bool(exam_context),
            "model": model,
        }
    except Exception as e:
        return {
            "response": f"[Calibrated response unavailable: {e}]",
            "calibration_level": difficulty["difficulty"],
            "exam_context_found": bool(exam_context),
        }


# Singleton
pedagogy_indexer = PedagogicalIndexer()
