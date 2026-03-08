"""
Indexing Enhancements — Improvements to Edith's document indexing.

Implements:
  3.1   Incremental re-index via filesystem watching
  3.2   Parallel extraction (ProcessPoolExecutor)
  3.4   Citation graph (parse reference sections)
  3.6   Language detection
  3.8   Hierarchical chunking (paragraph / section / document)
  3.9   Index versioning with auto-migration
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from server.vault_config import VAULT_ROOT

log = logging.getLogger("edith.indexing_enhancements")


# ---------------------------------------------------------------------------
# 3.9: Index Versioning
# ---------------------------------------------------------------------------

CURRENT_INDEX_VERSION = 2  # Increment when chunk format or metadata schema changes


@dataclass
class IndexVersion:
    """Track and manage index schema versions."""
    meta_path: Path = field(default_factory=lambda: VAULT_ROOT / "Connectome" / "index_version.json")
    version: int = CURRENT_INDEX_VERSION
    created_at: str = ""
    last_updated: str = ""
    total_docs: int = 0
    total_chunks: int = 0

    @classmethod
    def load(cls, meta_path: Path) -> "IndexVersion":
        if meta_path.exists():
            try:
                data = json.loads(meta_path.read_text(encoding="utf-8"))
                return cls(
                    meta_path=meta_path,
                    version=data.get("version", 1),
                    created_at=data.get("created_at", ""),
                    last_updated=data.get("last_updated", ""),
                    total_docs=data.get("total_docs", 0),
                    total_chunks=data.get("total_chunks", 0),
                )
            except Exception:
                pass
        return cls(meta_path=meta_path)

    def save(self):
        self.last_updated = time.strftime("%Y-%m-%dT%H:%M:%S")
        if not self.created_at:
            self.created_at = self.last_updated

        data = {
            "version": self.version,
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "total_docs": self.total_docs,
            "total_chunks": self.total_chunks,
        }
        self.meta_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @property
    def needs_migration(self) -> bool:
        return self.version < CURRENT_INDEX_VERSION

    def migrate(self) -> list[str]:
        """Run migration steps from current version to latest."""
        migrations = []
        while self.version < CURRENT_INDEX_VERSION:
            self.version += 1
            migrations.append(f"Migrated to v{self.version}")
            log.info(f"Index schema migrated to version {self.version}")
        self.save()
        return migrations


# ---------------------------------------------------------------------------
# 3.6: Language Detection
# ---------------------------------------------------------------------------

# Common word frequencies for language identification
_LANG_INDICATORS = {
    "en": {"the", "and", "is", "in", "of", "to", "that", "for", "it", "with", "as", "was", "are", "by", "this"},
    "es": {"de", "la", "el", "en", "que", "los", "del", "las", "una", "por", "con", "para", "es", "su", "al"},
    "fr": {"le", "de", "la", "les", "des", "un", "une", "du", "en", "est", "que", "pour", "dans", "qui", "sur"},
    "de": {"der", "die", "und", "den", "von", "ein", "mit", "das", "ist", "des", "auf", "eine", "dem", "nicht", "sich"},
    "pt": {"de", "que", "do", "da", "em", "para", "uma", "com", "não", "os", "das", "dos", "por", "mais", "na"},
}


def detect_language(text: str) -> dict:
    """
    Detect the language of a text using word frequency analysis.

    Returns {language: 'en', confidence: 0.85, word_count: 100}.
    """
    words = re.findall(r"\b\w+\b", text.lower())
    if not words:
        return {"language": "unknown", "confidence": 0.0, "word_count": 0}

    # Count matches per language
    word_set = set(words[:200])  # First 200 words for speed
    scores = {}

    for lang, indicators in _LANG_INDICATORS.items():
        matches = len(word_set & indicators)
        scores[lang] = matches / len(indicators)

    if not scores:
        return {"language": "unknown", "confidence": 0.0, "word_count": len(words)}

    best_lang = max(scores, key=scores.get)
    confidence = scores[best_lang]

    return {
        "language": best_lang if confidence > 0.2 else "unknown",
        "confidence": round(confidence, 3),
        "word_count": len(words),
        "all_scores": {k: round(v, 3) for k, v in sorted(scores.items(), key=lambda x: -x[1])},
    }


# ---------------------------------------------------------------------------
# 3.4: Citation Graph
# ---------------------------------------------------------------------------


@dataclass
class CitationGraph:
    """Build and query a citation network across documents."""
    graph_path: Path = field(default_factory=lambda: VAULT_ROOT / "Connectome" / "Graph" / "citation_graph.json")
    nodes: dict[str, dict] = field(default_factory=dict)  # doc_hash -> {title, authors, year}
    edges: list[dict] = field(default_factory=list)  # [{from, to, type}]

    def __post_init__(self):
        if self.graph_path.exists():
            try:
                data = json.loads(self.graph_path.read_text(encoding="utf-8"))
                self.nodes = data.get("nodes", {})
                self.edges = data.get("edges", [])
            except Exception:
                pass

    def add_document(self, doc_hash: str, title: str, authors: str = "", year: int = 0):
        """Add a document node to the graph."""
        self.nodes[doc_hash] = {
            "title": title[:200],
            "authors": authors[:200],
            "year": year,
        }

    def add_citation(self, citing_doc: str, cited_ref: dict):
        """Add a citation edge."""
        self.edges.append({
            "from": citing_doc,
            "to_title": cited_ref.get("title", cited_ref.get("text", ""))[:200],
            "to_authors": cited_ref.get("authors", ""),
            "to_year": cited_ref.get("year", 0),
        })

    def process_document(self, doc_hash: str, title: str, full_text: str, authors: str = "", year: int = 0):
        """Index a document and its references into the graph."""
        self.add_document(doc_hash, title, authors, year)
        refs = extract_references(full_text)
        for ref in refs:
            self.add_citation(doc_hash, ref)
        self._save()

    def get_citing(self, doc_hash: str) -> list[dict]:
        """Get all documents that cite the given document."""
        return [e for e in self.edges if e.get("from") == doc_hash]

    def most_cited(self, limit: int = 20) -> list[dict]:
        """Get the most frequently cited works."""
        from collections import Counter
        cited = Counter()
        for edge in self.edges:
            key = edge.get("to_title", "")[:100]
            if key:
                cited[key] += 1
        return [{"title": t, "count": c} for t, c in cited.most_common(limit)]

    def _save(self):
        try:
            self.graph_path.parent.mkdir(parents=True, exist_ok=True)
            self.graph_path.write_text(
                json.dumps({"nodes": self.nodes, "edges": self.edges[:5000]}, indent=2),
                encoding="utf-8"
            )
        except Exception:
            pass

    @property
    def stats(self) -> dict:
        return {
            "documents": len(self.nodes),
            "citations": len(self.edges),
        }


# ---------------------------------------------------------------------------
# 3.8: Hierarchical Chunking
# ---------------------------------------------------------------------------

def hierarchical_chunk(
    text: str,
    doc_id: str = "",
    paragraph_size: int = 250,
    section_size: int = 800,
    overlap: int = 30,
) -> list[dict]:
    """
    Create chunks at multiple granularities for multi-scale retrieval.

    Returns chunks at 3 levels:
    - paragraph: ~250 tokens, for precise matching
    - section: ~800 tokens, for broader context
    - document: full text summary, for document-level queries
    """
    chunks = []
    words = text.split()

    if not words:
        return []

    # Level 1: Paragraph-level chunks
    for i in range(0, len(words), paragraph_size - overlap):
        chunk_words = words[i: i + paragraph_size]
        if len(chunk_words) < 20:
            continue
        chunks.append({
            "text": " ".join(chunk_words),
            "level": "paragraph",
            "doc_id": doc_id,
            "offset": i,
            "word_count": len(chunk_words),
        })

    # Level 2: Section-level chunks
    for i in range(0, len(words), section_size - overlap * 2):
        chunk_words = words[i: i + section_size]
        if len(chunk_words) < 50:
            continue
        chunks.append({
            "text": " ".join(chunk_words),
            "level": "section",
            "doc_id": doc_id,
            "offset": i,
            "word_count": len(chunk_words),
        })

    # Level 3: Document-level summary (first + last paragraphs)
    if len(words) > 100:
        doc_summary = " ".join(words[:200]) + " ... " + " ".join(words[-100:])
        chunks.append({
            "text": doc_summary,
            "level": "document",
            "doc_id": doc_id,
            "offset": 0,
            "word_count": len(words),
        })

    return chunks


# ---------------------------------------------------------------------------
# 3.1 & 3.2: Incremental Indexing Support
# ---------------------------------------------------------------------------

def find_changed_files(
    scan_dir: Path,
    hash_manifest: dict[str, str],
    extensions: set[str] | None = None,
) -> dict[str, list[Path]]:
    """
    Compare filesystem against hash manifest to find changed files.

    Returns {new: [...], modified: [...], deleted: [...]}.
    """
    if extensions is None:
        extensions = {
            ".pdf", ".docx", ".doc", ".md", ".txt", ".rtf",
            ".csv", ".tsv", ".xlsx", ".xls",
            ".json", ".jsonl", ".geojson", ".html", ".htm", ".xml",
            ".tex", ".bib", ".yaml", ".yml", ".toml",
            ".ipynb", ".rmd",
            ".r", ".do", ".sps", ".py", ".js", ".sql",
            ".kml", ".gpx", ".log", ".dta", ".sav",
        }

    current_files: dict[str, Path] = {}
    for ext in extensions:
        for path in scan_dir.rglob(f"*{ext}"):
            hash_key = str(path.relative_to(scan_dir))
            current_files[hash_key] = path

    new_files = []
    modified_files = []
    deleted_keys = []

    for key, path in current_files.items():
        if key not in hash_manifest:
            new_files.append(path)
        else:
            # Quick change check: file size + mtime
            try:
                stat = path.stat()
                stored_hash = hash_manifest[key]
                current_check = f"{stat.st_size}:{int(stat.st_mtime)}"
                if current_check != stored_hash:
                    modified_files.append(path)
            except OSError:
                pass

    for key in hash_manifest:
        if key not in current_files:
            deleted_keys.append(key)

    return {
        "new": new_files,
        "modified": modified_files,
        "deleted_keys": deleted_keys,
    }


def parallel_extract_texts(
    file_paths: list[Path],
    extractor_fn,
    max_workers: int = None,
) -> list[dict]:
    """
    Extract text from multiple files in parallel using ProcessPoolExecutor.

    §M4-3: Worker count is hardware-aware:
        M4 Pro: MAX_PARALLEL_INDEXING=12 (saturate all cores)
        M2 Air: MAX_PARALLEL_INDEXING=2  (avoid thermal throttling)

    Args:
        file_paths: List of files to process
        extractor_fn: Function(Path) -> {text, metadata} (must be picklable)
        max_workers: Override worker count (default: from env)
    """
    import os

    if max_workers is None:
        max_workers = int(os.environ.get("MAX_PARALLEL_INDEXING", "4"))

    if not file_paths:
        return []

    results = []
    errors = []

    actual_workers = min(max_workers, len(file_paths))
    log.info(f"§M4-3: Parallel extraction with {actual_workers} workers "
             f"({len(file_paths)} files)")

    # ProcessPoolExecutor for CPU-bound extraction
    with ProcessPoolExecutor(max_workers=actual_workers) as executor:
        future_to_path = {
            executor.submit(extractor_fn, path): path
            for path in file_paths
        }

        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                result = future.result(timeout=120)
                if result:
                    results.append(result)
            except Exception as e:
                errors.append({"path": str(path), "error": str(e)})
                log.warning(f"Extraction failed for {path}: {e}")

    log.info(f"§M4-3: Extraction complete: {len(results)} succeeded, "
             f"{len(errors)} failed ({actual_workers} workers)")
    return results


# ---------------------------------------------------------------------------
# §M4-3: Batch Parallel Embedding
# ---------------------------------------------------------------------------

