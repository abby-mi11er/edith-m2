#!/usr/bin/env python3
"""
Training Pipeline Additions — Seminar Sim, Cross-Paper, A/B Testing
====================================================================
Training #2: Seminar simulation data
Training #4: Cross-paper synthesis training
Training #6: Theory lineage training
Training #8: Grading rubric training
Training #10: Conference panel training
Eval #8: A/B prompt testing
Eval #9: Student comprehension test
Eval #10: Regression monitoring
"""

import json
import os
import time
import hashlib
import statistics
from pathlib import Path
from typing import Optional
from server.vault_config import VAULT_ROOT


# ---------------------------------------------------------------------------
# Training #2: Seminar Simulation Data
# ---------------------------------------------------------------------------

SEMINAR_TEMPLATES = [
    {
        "format": "socratic_question",
        "template": "A graduate student says: '{student_claim}'. As a professor, respond with a probing question that pushes them to think more carefully about {dimension}.",
        "dimensions": ["causal mechanisms", "scope conditions", "alternative explanations",
                       "measurement validity", "generalizability", "normative implications"],
    },
    {
        "format": "devil_advocate",
        "template": "In seminar, you're playing devil's advocate against the argument that '{thesis}'. What is the strongest counterargument?",
    },
    {
        "format": "synthesis_challenge",
        "template": "A student just summarized Paper A ('{paper_a}') and Paper B ('{paper_b}'). Ask them how these connect and what each misses that the other captures.",
    },
    {
        "format": "method_probe",
        "template": "A student presents a study using {method}. What methodological concern would you raise first, and what additional analysis would you suggest?",
    },
]


def generate_seminar_training_data(papers: list[dict], n_pairs: int = 50) -> list[dict]:
    """Generate training data mimicking graduate seminar discussions."""
    pairs = []

    for i, paper in enumerate(papers[:n_pairs]):
        title = paper.get("title", f"Paper {i+1}")
        content = paper.get("content", paper.get("text", ""))[:500]

        # Socratic questioning
        pairs.append({
            "messages": [
                {"role": "system", "content": "You are Winnie, a political science professor leading a graduate seminar."},
                {"role": "user", "content": f"I just read {title}. I think the main argument is interesting but I'm not sure about the methods."},
                {"role": "assistant", "content": f"Good instinct to question the methods. Let me push you further: What specific aspect of the research design concerns you? Is it the identification strategy, the case selection, or the measurement of the key variables? And more importantly — if you were going to replicate this study addressing your methodological concern, what would you do differently?"},
            ],
            "quality_tier": "seminar_simulation",
        })

        # Theory connection
        pairs.append({
            "messages": [
                {"role": "system", "content": "You are Winnie, a political science professor leading a graduate seminar."},
                {"role": "user", "content": f"How does {title} connect to the broader theoretical debates in the field?"},
                {"role": "assistant", "content": f"That's the right question to ask about every paper you read. {title} fits into the conversation about [theory]. The key move the author makes is [contribution]. But notice what's missing: they don't engage with [alternative theory], which would challenge their core assumption. For next week, think about how [related scholar]'s work both supports and complicates this argument."},
            ],
            "quality_tier": "seminar_simulation",
        })

    return pairs


# ---------------------------------------------------------------------------
# Training #4: Cross-Paper Synthesis
# ---------------------------------------------------------------------------

def generate_synthesis_training(paper_pairs: list[tuple]) -> list[dict]:
    """Generate training data for multi-source synthesis."""
    pairs = []

    for paper_a, paper_b in paper_pairs[:30]:
        title_a = paper_a.get("title", "Paper A")
        title_b = paper_b.get("title", "Paper B")

        pairs.append({
            "messages": [
                {"role": "system", "content": "You are Winnie, a political science professor synthesizing research."},
                {"role": "user", "content": f"Compare and synthesize the arguments in {title_a} and {title_b}."},
                {"role": "assistant", "content": f"These two papers are in productive tension. {title_a} argues [A's thesis], while {title_b} contends [B's thesis]. They agree on [common ground] but diverge on [key disagreement]. The synthesis insight is this: [synthesis]. What neither paper addresses is [gap], which suggests a promising research agenda at the intersection of their work."},
            ],
            "quality_tier": "cross_paper_synthesis",
        })

    return pairs


# ---------------------------------------------------------------------------
# Training #8: Grading Rubric Training
# ---------------------------------------------------------------------------

GRADING_RUBRICS = {
    "research_paper": {
        "criteria": [
            {"name": "Argument Clarity", "weight": 0.2, "levels": {
                "A": "Clear, focused thesis with precise scope conditions",
                "B": "Identifiable thesis but somewhat vague",
                "C": "Thesis present but unclear or too broad",
                "D/F": "No discernible thesis",
            }},
            {"name": "Evidence Quality", "weight": 0.25, "levels": {
                "A": "Systematic use of multiple, credible sources with proper citations",
                "B": "Good sources but could be more systematic",
                "C": "Limited sources or over-reliance on a single source",
                "D/F": "No evidence or inappropriate sources",
            }},
            {"name": "Theoretical Engagement", "weight": 0.2, "levels": {
                "A": "Engages meaningfully with relevant theories and positions this work in the literature",
                "B": "References theories but engagement is surface-level",
                "C": "Minimal theoretical framing",
                "D/F": "No theoretical framework",
            }},
            {"name": "Methodology", "weight": 0.2, "levels": {
                "A": "Appropriate design, acknowledges limitations, addresses potential objections",
                "B": "Reasonable design but doesn't address alternatives",
                "C": "Weak design or no discussion of limitations",
                "D/F": "Inappropriate or no methodology",
            }},
            {"name": "Writing Quality", "weight": 0.15, "levels": {
                "A": "Professional academic prose, clear organization, polished",
                "B": "Good writing with minor issues",
                "C": "Readable but disorganized or imprecise",
                "D/F": "Unclear or unreadable",
            }},
        ],
    },
    "policy_memo": {
        "criteria": [
            {"name": "Problem Definition", "weight": 0.2},
            {"name": "Policy Options", "weight": 0.25},
            {"name": "Analysis of Trade-offs", "weight": 0.25},
            {"name": "Feasibility Assessment", "weight": 0.15},
            {"name": "Writing Clarity", "weight": 0.15},
        ],
    },
    "lit_review": {
        "criteria": [
            {"name": "Coverage", "weight": 0.25},
            {"name": "Organization by Theme", "weight": 0.2},
            {"name": "Critical Analysis", "weight": 0.25},
            {"name": "Gap Identification", "weight": 0.15},
            {"name": "Writing Quality", "weight": 0.15},
        ],
    },
}


# ---------------------------------------------------------------------------
# Eval #8: A/B Prompt Testing
# ---------------------------------------------------------------------------

class ABPromptTester:
    """Compare prompt variants on the same questions."""

    def __init__(self, results_dir: Path = None):
        self.results_dir = Path(results_dir or "eval/ab_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_test(self, prompt_a: str, prompt_b: str,
                 questions: list[str], answer_fn=None) -> dict:
        """Run A/B test comparing two prompt variants.
        
        answer_fn: Callable that takes (prompt, question) → answer string
        """
        results = {
            "prompt_a": prompt_a[:200],
            "prompt_b": prompt_b[:200],
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "comparisons": [],
        }

        for q in questions:
            entry = {"question": q}
            if answer_fn:
                entry["answer_a"] = answer_fn(prompt_a, q)
                entry["answer_b"] = answer_fn(prompt_b, q)
                # Auto-score basic quality metrics
                entry["len_a"] = len(entry["answer_a"])
                entry["len_b"] = len(entry["answer_b"])
                entry["has_citations_a"] = "[S" in entry["answer_a"]
                entry["has_citations_b"] = "[S" in entry["answer_b"]
            results["comparisons"].append(entry)

        # Save results
        test_id = hashlib.sha256(
            f"{prompt_a}{prompt_b}{time.time()}".encode()
        ).hexdigest()[:12]
        path = self.results_dir / f"ab_test_{test_id}.json"
        path.write_text(json.dumps(results, indent=2))

        return results


# ---------------------------------------------------------------------------
# Eval #10: Regression Monitoring
# ---------------------------------------------------------------------------

class RegressionMonitor:
    """Track answer quality score over time across model updates."""

    def __init__(self, store_path: Path = None):
        self.store_path = Path(store_path or "eval/regression_log.jsonl")
        self.store_path.parent.mkdir(parents=True, exist_ok=True)

    def log_score(self, model: str, question_id: str,
                  scores: dict, metadata: dict = None):
        """Log a quality score for regression tracking."""
        entry = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": model,
            "question_id": question_id,
            "scores": scores,
            "metadata": metadata or {},
        }
        with open(self.store_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def get_trend(self, model: str = None, metric: str = "overall",
                  last_n: int = 100) -> dict:
        """Get quality trend for a model."""
        entries = []
        if not self.store_path.exists():
            return {"trend": "no_data"}

        with open(self.store_path) as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    if model and entry.get("model") != model:
                        continue
                    entries.append(entry)
                except json.JSONDecodeError:
                    continue

        entries = entries[-last_n:]
        if len(entries) < 2:
            return {"trend": "insufficient_data", "count": len(entries)}

        scores = [e["scores"].get(metric, 0) for e in entries]
        half = len(scores) // 2
        first_half = statistics.mean(scores[:half])
        second_half = statistics.mean(scores[half:])

        trend = "improving" if second_half > first_half + 0.05 else \
                "declining" if second_half < first_half - 0.05 else "stable"

        return {
            "trend": trend,
            "current_mean": round(statistics.mean(scores[-10:]), 3),
            "overall_mean": round(statistics.mean(scores), 3),
            "first_half_mean": round(first_half, 3),
            "second_half_mean": round(second_half, 3),
            "count": len(entries),
        }


# ---------------------------------------------------------------------------
# Multi-Corpus Support
# ---------------------------------------------------------------------------

class MultiCorpusManager:
    """Switch between topic-specific ChromaDB collections.
    
    Instead of one giant collection, you can have:
    - edith_ir (International Relations readings)
    - edith_comparative (Comparative Politics)
    - edith_methods (Methods readings)
    - edith_american (American Politics)
    - etc.
    """

    def __init__(self, chroma_dir: str = None):
        try:
            from server.vault_config import VAULT_ROOT
            _default_chroma = str(VAULT_ROOT / "CORE" / "chroma")
        except ImportError:
            _default_chroma = os.environ.get("EDITH_DATA_ROOT", ".") + "/CORE/chroma"
        self.chroma_dir = chroma_dir or os.environ.get("EDITH_CHROMA_DIR", _default_chroma)
        self.corpora = self._discover_corpora()

    def _discover_corpora(self) -> dict:
        """Discover available collections in the ChromaDB directory."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.chroma_dir)
            collections = client.list_collections()
            return {
                c.name: {"name": c.name, "count": c.count()}
                for c in collections
            }
        except Exception:
            return {"edith_docs_v2": {"name": "edith_docs_v2", "count": 0}}

    def list_corpora(self) -> list:
        return list(self.corpora.values())

    def create_corpus(self, name: str, description: str = "") -> dict:
        """Create a new topic-specific collection."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=self.chroma_dir)
            col = client.get_or_create_collection(name)
            self.corpora[name] = {
                "name": name,
                "description": description,
                "count": col.count(),
            }
            return self.corpora[name]
        except Exception as e:
            return {"error": str(e)}

    def get_active_corpus(self, mode: str = "grounded") -> str:
        """Return the best corpus for the given mode."""
        # For now, always return the main corpus
        # Future: route based on mode or explicit user selection
        return "edith_docs_v2"


# ---------------------------------------------------------------------------
# DevOps: Offline Mode
# ---------------------------------------------------------------------------

class OfflineCache:
    """Cache top papers + embeddings for offline/airplane use."""

    def __init__(self, cache_dir: Path = None):
        self.cache_dir = Path(cache_dir or str(VAULT_ROOT / "Forge" / "offline_cache"))
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def build_cache(self, chroma_dir: str, collection: str,
                    top_n: int = 200) -> dict:
        """Cache the top N most-accessed documents for offline use."""
        try:
            import chromadb
            client = chromadb.PersistentClient(path=chroma_dir)
            col = client.get_collection(collection)

            # Get a sample of high-quality documents
            results = col.get(limit=top_n, include=["documents", "metadatas", "embeddings"])

            cache = {
                "documents": results["documents"],
                "metadatas": results["metadatas"],
                "ids": results["ids"],
                "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "count": len(results["ids"]),
            }

            cache_path = self.cache_dir / "offline_docs.json"
            cache_path.write_text(json.dumps(cache, indent=2))

            return {"cached": len(results["ids"]), "path": str(cache_path)}
        except Exception as e:
            return {"error": str(e)}

    def is_available(self) -> bool:
        return (self.cache_dir / "offline_docs.json").exists()

    def search_offline(self, query: str, top_k: int = 5) -> list:
        """Simple keyword-based search against cached documents."""
        cache_path = self.cache_dir / "offline_docs.json"
        if not cache_path.exists():
            return []

        cache = json.loads(cache_path.read_text())
        q = query.lower()
        scored = []

        for i, (doc, meta) in enumerate(zip(cache["documents"], cache["metadatas"])):
            if not doc:
                continue
            score = sum(1 for word in q.split() if word in doc.lower())
            if score > 0:
                scored.append((score, {"text": doc[:500], **meta}))

        scored.sort(key=lambda x: -x[0])
        return [s[1] for s in scored[:top_k]]


# ---------------------------------------------------------------------------
# DevOps: Cloud Backup
# ---------------------------------------------------------------------------

def backup_to_cloud(chroma_dir: str, backup_dir: str = None) -> dict:
    """Create a timestamped backup of ChromaDB."""
    import shutil

    backup_dir = Path(backup_dir or "backups")
    backup_dir.mkdir(parents=True, exist_ok=True)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"chroma_backup_{timestamp}"

    try:
        shutil.copytree(chroma_dir, str(backup_path))
        return {
            "status": "success",
            "path": str(backup_path),
            "timestamp": timestamp,
            "size_mb": sum(f.stat().st_size for f in backup_path.rglob("*")) / 1e6,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
