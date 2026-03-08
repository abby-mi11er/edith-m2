"""
Advanced Flywheel Routes — Research Workflow Intelligence
==========================================================
Endpoints for 8 advanced research capabilities:
  #1 Auto-Literature-Review Generator
  #2 Paper Recommendation Engine
  #3 Replication Package Builder
  #4 Research Question Evolution Tracker
  #5 Methodology Recommender
  #6 Peer Review Simulator
  #7 Research Session Snapshots
  #8 Notes Versioning
"""

import logging
import os
import re
import json
import time
import hashlib
from collections import defaultdict
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

log = logging.getLogger("edith.routes.flywheel_advanced")
router = APIRouter()

# Shared helper
def _get_chroma_collection():
    chroma_dir = os.environ.get("EDITH_CHROMA_DIR", "")
    if not chroma_dir:
        return None
    try:
        from server.chroma_backend import _get_client
        client = _get_client(chroma_dir)
        return client.get_or_create_collection("edith_corpus", metadata={"hnsw:space": "cosine"})
    except Exception:
        return None

def _data_root():
    return os.environ.get("DATA_ROOT", os.environ.get("EDITH_DATA_ROOT", ""))


# ═══════════════════════════════════════════════════════════════════
# #1 — Auto-Literature-Review Generator
# ═══════════════════════════════════════════════════════════════════

@router.post("/api/litreview/generate", tags=["LitReview"])
async def generate_lit_review(request: Request):
    """Generate a structured literature review from indexed papers.

    Clusters papers by methodology and findings, identifies themes,
    gaps, and contradictions, then outputs a structured draft with
    proper citation flow: intro → themes → gaps → contribution.
    """
    body = await request.json()
    topic = body.get("topic", "")
    max_papers = min(body.get("max_papers", 20), 50)
    style = body.get("style", "narrative")  # narrative | thematic | chronological

    collection = _get_chroma_collection()
    if not collection:
        return JSONResponse(status_code=503, content={"error": "ChromaDB not available"})

    # Retrieve relevant papers
    if topic:
        results = collection.query(
            query_texts=[topic], n_results=max_papers,
            include=["documents", "metadatas", "distances"],
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
    else:
        results = collection.get(limit=max_papers, include=["documents", "metadatas"])
        docs = results.get("documents", [])
        metas = results.get("metadatas", [])
        distances = [0.0] * len(docs)

    if not docs:
        return {"review": "No papers found for this topic.", "papers_used": 0}

    # Build paper entries with extracted info
    papers = []
    for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
        meta = meta or {}
        text = (doc or "")[:2000]
        papers.append({
            "index": i,
            "title": meta.get("title", f"Paper {i+1}"),
            "author": meta.get("author", "Unknown"),
            "year": meta.get("year", "n.d."),
            "method": meta.get("method", ""),
            "topic": meta.get("academic_topic", ""),
            "relevance": round(1 - dist, 3) if dist else 1.0,
            "text_snippet": text[:500],
        })

    # Cluster by method
    method_clusters = defaultdict(list)
    for p in papers:
        method = p["method"] or "Unspecified"
        method_clusters[method].append(p)

    # Cluster by decade
    decade_clusters = defaultdict(list)
    for p in papers:
        try:
            year = int(p["year"])
            decade = f"{(year // 10) * 10}s"
        except (ValueError, TypeError):
            decade = "Unknown"
        decade_clusters[decade].append(p)

    # Identify themes (top topics)
    topic_counts = defaultdict(int)
    for p in papers:
        if p["topic"]:
            topic_counts[p["topic"]] += 1
    themes = sorted(topic_counts.items(), key=lambda x: -x[1])[:5]

    # Find potential gaps
    gaps = []
    methods_seen = set(p["method"] for p in papers if p["method"])
    common_methods = {"OLS", "DiD", "IV", "RDD", "Synthetic Control", "Fixed Effects", "Random Effects", "Logit", "Probit"}
    missing_methods = common_methods - methods_seen
    if missing_methods and len(methods_seen) > 0:
        gaps.append(f"No papers use {', '.join(list(missing_methods)[:3])} — methodological gap")

    years = [int(p["year"]) for p in papers if p["year"] and p["year"].isdigit()]
    if years and max(years) < 2023:
        gaps.append(f"Most recent paper is from {max(years)} — temporal gap")

    # Generate the review text
    sections = []

    # Introduction
    intro_authors = [f"{p['author'].split(',')[0]} ({p['year']})" for p in papers[:5]]
    sections.append({
        "heading": "Introduction",
        "content": (
            f"This literature review synthesizes {len(papers)} papers on "
            f"{topic or 'the research topic'}. "
            f"Key contributions include {', '.join(intro_authors[:3])}. "
            f"The literature spans methods including {', '.join(list(methods_seen)[:4]) or 'various approaches'}."
        ),
    })

    # Thematic sections
    if style == "chronological":
        for decade, cluster in sorted(decade_clusters.items()):
            citations = [f"{p['author'].split(',')[0]} ({p['year']})" for p in cluster]
            sections.append({
                "heading": f"Period: {decade}",
                "content": (
                    f"{len(cluster)} papers from the {decade}: {', '.join(citations[:5])}. "
                    f"Methods used: {', '.join(set(p['method'] for p in cluster if p['method'])) or 'various'}."
                ),
                "papers": [p["title"] for p in cluster],
            })
    else:
        for method, cluster in method_clusters.items():
            citations = [f"{p['author'].split(',')[0]} ({p['year']})" for p in cluster]
            sections.append({
                "heading": f"Studies Using {method}",
                "content": (
                    f"{len(cluster)} papers employ {method}: {', '.join(citations[:5])}. "
                ),
                "papers": [p["title"] for p in cluster],
            })

    # Gaps section
    if gaps:
        sections.append({
            "heading": "Identified Gaps",
            "content": " ".join(gaps) + " These gaps suggest opportunities for further research.",
        })

    # Conclusion
    sections.append({
        "heading": "Summary",
        "content": (
            f"This review of {len(papers)} papers reveals "
            f"{len(method_clusters)} distinct methodological approaches and "
            f"{len(themes)} major themes. "
            f"{f'Key gaps include: {gaps[0]}.' if gaps else 'The literature is well-covered.'}"
        ),
    })

    # Build full text
    review_text = ""
    for s in sections:
        review_text += f"\n## {s['heading']}\n\n{s['content']}\n"

    return {
        "review": review_text,
        "sections": sections,
        "papers_used": len(papers),
        "method_clusters": {k: len(v) for k, v in method_clusters.items()},
        "themes": [{"topic": t, "count": c} for t, c in themes],
        "gaps": gaps,
        "style": style,
    }


# ═══════════════════════════════════════════════════════════════════
# #2 — Paper Recommendation Engine
# ═══════════════════════════════════════════════════════════════════

@router.post("/api/recommend/papers", tags=["Recommend"])
async def recommend_papers(request: Request):
    """Recommend papers based on reading history and citation gaps.

    Analyzes what you've read, finds citation graph gaps, and suggests
    papers you're missing. Uses reading progress + ChromaDB + method diversity.
    """
    body = await request.json()
    max_recs = min(body.get("max", 10), 20)
    based_on_sha = body.get("sha256", "")  # recommend based on a specific paper

    collection = _get_chroma_collection()
    if not collection:
        return JSONResponse(status_code=503, content={"error": "ChromaDB not available"})

    # Load reading progress
    reading_progress = {}
    try:
        from server.routes.flywheel import _reading_progress
        reading_progress = _reading_progress
    except Exception:
        pass

    read_shas = set(sha for sha, p in reading_progress.items() if p.get("status") in ("reading", "done"))

    # Strategy 1: Find papers similar to what you've read but haven't seen
    recommendations = []

    if based_on_sha:
        # Get the seed paper
        seed = collection.get(where={"sha256": based_on_sha}, include=["documents", "metadatas"], limit=1)
        if seed["documents"]:
            text = seed["documents"][0][:1000]
            similar = collection.query(query_texts=[text], n_results=max_recs + 10, include=["metadatas", "distances"])
            for meta, dist in zip(similar["metadatas"][0], similar["distances"][0]):
                sha = meta.get("sha256", "")
                if sha == based_on_sha or sha in read_shas:
                    continue
                recommendations.append({
                    "sha256": sha,
                    "title": meta.get("title", ""),
                    "author": meta.get("author", ""),
                    "year": meta.get("year", ""),
                    "method": meta.get("method", ""),
                    "relevance": round(1 - dist, 3),
                    "reason": "similar to current paper",
                })
    elif read_shas:
        # Get a random read paper's text and find similar
        for sha in list(read_shas)[:3]:
            results = collection.get(where={"sha256": sha}, include=["documents"], limit=1)
            if results["documents"]:
                text = results["documents"][0][:800]
                similar = collection.query(query_texts=[text], n_results=max_recs, include=["metadatas", "distances"])
                for meta, dist in zip(similar["metadatas"][0], similar["distances"][0]):
                    rec_sha = meta.get("sha256", "")
                    if rec_sha in read_shas or any(r["sha256"] == rec_sha for r in recommendations):
                        continue
                    recommendations.append({
                        "sha256": rec_sha,
                        "title": meta.get("title", ""),
                        "author": meta.get("author", ""),
                        "year": meta.get("year", ""),
                        "method": meta.get("method", ""),
                        "relevance": round(1 - dist, 3),
                        "reason": "based on reading history",
                    })
    else:
        # No reading history — recommend highest-cited or most recent
        all_docs = collection.get(limit=50, include=["metadatas"])
        for meta in (all_docs.get("metadatas", []) or [])[:max_recs]:
            meta = meta or {}
            recommendations.append({
                "sha256": meta.get("sha256", ""),
                "title": meta.get("title", ""),
                "author": meta.get("author", ""),
                "year": meta.get("year", ""),
                "method": meta.get("method", ""),
                "relevance": 0.5,
                "reason": "in your library (unread)",
            })

    # Strategy 2: Method diversity — suggest papers using methods you haven't seen
    read_methods = set()
    for sha in read_shas:
        results = collection.get(where={"sha256": sha}, include=["metadatas"], limit=1)
        if results["metadatas"]:
            m = (results["metadatas"][0] or {}).get("method", "")
            if m:
                read_methods.add(m)

    if read_methods:
        all_docs = collection.get(limit=100, include=["metadatas"])
        for meta in (all_docs.get("metadatas", []) or []):
            meta = meta or {}
            method = meta.get("method", "")
            sha = meta.get("sha256", "")
            if method and method not in read_methods and sha not in read_shas:
                if not any(r["sha256"] == sha for r in recommendations):
                    recommendations.append({
                        "sha256": sha,
                        "title": meta.get("title", ""),
                        "author": meta.get("author", ""),
                        "year": meta.get("year", ""),
                        "method": method,
                        "relevance": 0.6,
                        "reason": f"uses {method} (new method for you)",
                    })

    # Sort by relevance and cap
    recommendations.sort(key=lambda r: -r["relevance"])
    recommendations = recommendations[:max_recs]

    return {
        "recommendations": recommendations,
        "total": len(recommendations),
        "papers_read": len(read_shas),
        "methods_seen": list(read_methods),
    }


# ═══════════════════════════════════════════════════════════════════
# #3 — Replication Package Builder
# ═══════════════════════════════════════════════════════════════════

@router.post("/api/replication/build", tags=["Replication"])
async def build_replication_package(request: Request):
    """Build a structured replication package for journal submission.

    Bundles Stata .do files, data paths, LaTeX output, methodology notes,
    and README into a structured archive manifest.
    """
    body = await request.json()
    project_name = body.get("project_name", "replication_package")
    # §SECURITY: Sanitize project name to prevent directory traversal
    project_name = re.sub(r'[^a-zA-Z0-9_\-]', '_', project_name)[:50]
    paper_sha = body.get("paper_sha256", "")
    stata_paths = body.get("stata_files", [])  # list of .do file paths
    data_paths = body.get("data_files", [])
    notes_text = body.get("notes", "")

    data_root = _data_root()
    package_dir = os.path.join(data_root, "replication", project_name) if data_root else ""

    manifest = {
        "project": project_name,
        "created_at": time.time(),
        "structure": {},
        "files": [],
        "readme": "",
        "status": "ok",
    }

    # Create package directory structure
    if package_dir:
        try:
            for subdir in ["code", "data", "output", "docs"]:
                os.makedirs(os.path.join(package_dir, subdir), exist_ok=True)
            manifest["structure"]["root"] = package_dir
        except Exception as e:
            manifest["status"] = f"dir_error: {e}"

    # Catalog Stata files
    for sp in stata_paths:
        if os.path.isfile(sp):
            manifest["files"].append({
                "type": "code",
                "path": sp,
                "basename": os.path.basename(sp),
                "size_kb": round(os.path.getsize(sp) / 1024, 1),
                "dest": f"code/{os.path.basename(sp)}",
            })

    # Catalog data files
    for dp in data_paths:
        if os.path.isfile(dp):
            manifest["files"].append({
                "type": "data",
                "path": dp,
                "basename": os.path.basename(dp),
                "size_kb": round(os.path.getsize(dp) / 1024, 1),
                "dest": f"data/{os.path.basename(dp)}",
            })

    # Get paper metadata for README
    paper_meta = {}
    if paper_sha:
        collection = _get_chroma_collection()
        if collection:
            results = collection.get(where={"sha256": paper_sha}, include=["metadatas"], limit=1)
            if results["metadatas"]:
                paper_meta = results["metadatas"][0] or {}

    # Generate README
    readme = f"""# Replication Package: {project_name}

## Paper
- **Title**: {paper_meta.get('title', 'TBD')}
- **Author**: {paper_meta.get('author', 'TBD')}
- **Method**: {paper_meta.get('method', 'TBD')}

## Directory Structure
```
{project_name}/
├── code/          # Stata .do files and scripts
├── data/          # Input datasets
├── output/        # Tables, figures, logs
└── docs/          # Methodology notes, codebook
```

## Software Requirements
- Stata 17+ (or compatible)
- Any additional packages listed in code/requirements.txt

## Replication Instructions
1. Set the working directory to this folder
2. Run `code/master.do` to reproduce all results
3. Output tables will appear in `output/`

## Files
{chr(10).join(f"- `{f['dest']}` ({f['size_kb']} KB)" for f in manifest['files'])}

## Notes
{notes_text or 'No additional notes provided.'}

---
Generated by E.D.I.T.H. Replication Builder · {time.strftime('%Y-%m-%d')}
"""

    manifest["readme"] = readme

    # Write README to package dir
    if package_dir and os.path.isdir(package_dir):
        try:
            with open(os.path.join(package_dir, "README.md"), 'w') as f:
                f.write(readme)
            manifest["files"].append({
                "type": "docs",
                "path": os.path.join(package_dir, "README.md"),
                "basename": "README.md",
                "size_kb": round(len(readme) / 1024, 1),
                "dest": "README.md",
            })
        except Exception:
            pass

    return manifest


# ═══════════════════════════════════════════════════════════════════
# #4 — Research Question Evolution Tracker
# ═══════════════════════════════════════════════════════════════════

_rq_history_file = ""
_rq_history: list[dict] = []


def _load_rq_history():
    global _rq_history, _rq_history_file
    root = _data_root()
    _rq_history_file = os.path.join(root, "rq_evolution.json") if root else ""
    if _rq_history_file and os.path.isfile(_rq_history_file):
        try:
            with open(_rq_history_file, 'r') as f:
                _rq_history = json.load(f)
        except Exception:
            pass


def _save_rq_history():
    if _rq_history_file:
        try:
            os.makedirs(os.path.dirname(_rq_history_file), exist_ok=True)
            with open(_rq_history_file, 'w') as f:
                json.dump(_rq_history, f, indent=2)
        except Exception:
            pass


_load_rq_history()


@router.post("/api/research-question/log", tags=["RQ"])
async def log_research_question(request: Request):
    """Log a research question evolution event.

    Tracks how the RQ changed over time with context about why.
    """
    body = await request.json()
    rq_text = body.get("question", "")
    context = body.get("context", "")  # what prompted the change
    session_id = body.get("session_id", "")

    if not rq_text:
        return JSONResponse(status_code=400, content={"error": "question required"})

    entry = {
        "question": rq_text,
        "context": context,
        "session_id": session_id,
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "version": len(_rq_history) + 1,
    }

    # Detect what changed from previous
    if _rq_history:
        prev = _rq_history[-1]["question"]
        # Simple word-level diff
        prev_words = set(prev.lower().split())
        new_words = set(rq_text.lower().split())
        added = new_words - prev_words
        removed = prev_words - new_words
        entry["diff"] = {
            "added_words": list(added)[:10],
            "removed_words": list(removed)[:10],
            "similarity": round(len(prev_words & new_words) / max(len(prev_words | new_words), 1), 3),
        }

    _rq_history.append(entry)
    _save_rq_history()

    return {"status": "logged", "version": entry["version"], "entry": entry}


@router.get("/api/research-question/history", tags=["RQ"])
async def rq_history():
    """Get the full research question evolution timeline."""
    return {
        "history": _rq_history,
        "total_versions": len(_rq_history),
        "current": _rq_history[-1] if _rq_history else None,
    }


# ═══════════════════════════════════════════════════════════════════
# #5 — Methodology Recommender
# ═══════════════════════════════════════════════════════════════════

_METHOD_DATABASE = {
    "DiD": {
        "full_name": "Difference-in-Differences",
        "best_for": "policy evaluations with treatment/control and before/after periods",
        "requires": ["panel data", "parallel trends assumption", "treatment group", "control group"],
        "strengths": ["handles time-invariant unobservables", "intuitive interpretation", "widely accepted"],
        "weaknesses": ["requires parallel trends", "sensitive to treatment timing", "SUTVA assumption"],
        "stata_command": "didregress",
        "typical_fields": ["public policy", "labor economics", "health economics"],
    },
    "IV": {
        "full_name": "Instrumental Variables",
        "best_for": "endogeneity problems where OLS is biased",
        "requires": ["valid instrument (relevance + exclusion)", "first-stage F > 10"],
        "strengths": ["addresses endogeneity", "causal identification"],
        "weaknesses": ["hard to find valid instruments", "weak instrument bias", "overidentification concerns"],
        "stata_command": "ivregress 2sls",
        "typical_fields": ["development economics", "labor", "IO"],
    },
    "RDD": {
        "full_name": "Regression Discontinuity Design",
        "best_for": "programs with eligibility cutoffs or thresholds",
        "requires": ["running variable", "discontinuity at cutoff", "no manipulation"],
        "strengths": ["strong internal validity", "minimal assumptions near cutoff"],
        "weaknesses": ["local treatment effect only", "requires density at cutoff", "bandwidth sensitivity"],
        "stata_command": "rdrobust",
        "typical_fields": ["education", "public finance", "political science"],
    },
    "Synthetic Control": {
        "full_name": "Synthetic Control Method",
        "best_for": "comparative case studies with few treated units",
        "requires": ["donor pool of untreated units", "pre-treatment outcomes", "single or few treated units"],
        "strengths": ["transparent counterfactual", "works with few treated units"],
        "weaknesses": ["sensitive to donor pool", "inference challenges", "requires long pre-period"],
        "stata_command": "synth",
        "typical_fields": ["macro policy", "regional economics", "political economy"],
    },
    "Fixed Effects": {
        "full_name": "Fixed Effects Regression",
        "best_for": "panel data with unobserved time-invariant heterogeneity",
        "requires": ["panel data", "within-variation", "strict exogeneity"],
        "strengths": ["controls for all time-invariant unobservables", "widely understood"],
        "weaknesses": ["cannot estimate time-invariant variables", "requires within-variation"],
        "stata_command": "xtreg, fe",
        "typical_fields": ["labor", "trade", "corporate finance"],
    },
    "OLS": {
        "full_name": "Ordinary Least Squares",
        "best_for": "cross-sectional analysis with exogenous regressors",
        "requires": ["exogeneity", "no multicollinearity", "homoskedasticity"],
        "strengths": ["simple", "well-understood", "BLUE under Gauss-Markov"],
        "weaknesses": ["biased with endogeneity", "no causal claims without exogeneity"],
        "stata_command": "regress",
        "typical_fields": ["all fields"],
    },
    "Event Study": {
        "full_name": "Event Study Design",
        "best_for": "estimating dynamic treatment effects over time",
        "requires": ["panel data", "event dates", "pre-event period for trends"],
        "strengths": ["visualizes treatment dynamics", "tests for pre-trends"],
        "weaknesses": ["many parameters", "heterogeneous timing issues"],
        "stata_command": "eventdd or reghdfe with interactions",
        "typical_fields": ["finance", "labor", "public policy"],
    },
    "Logit/Probit": {
        "full_name": "Logistic/Probit Regression",
        "best_for": "binary dependent variables",
        "requires": ["binary outcome", "independence assumption"],
        "strengths": ["bounded predictions", "marginal effects interpretable"],
        "weaknesses": ["distributional assumptions", "harder to compare across models"],
        "stata_command": "logit / probit",
        "typical_fields": ["health", "labor", "marketing"],
    },
}


@router.post("/api/methodology/recommend", tags=["Methodology"])
async def recommend_methodology(request: Request):
    """Recommend research methods based on data structure and question.

    Analyzes the research question, data type, and desired causal claim
    to suggest appropriate econometric methods.
    """
    body = await request.json()
    question = body.get("question", "")
    data_type = body.get("data_type", "")  # cross-section | panel | time-series
    outcome_var = body.get("outcome", "")
    treatment = body.get("treatment", "")
    has_instrument = body.get("has_instrument", False)
    has_cutoff = body.get("has_cutoff", False)
    has_control_group = body.get("has_control_group", False)
    n_treated_units = body.get("n_treated_units", 0)

    recommendations = []
    question_lower = question.lower()

    # Rule-based scoring
    for method_key, info in _METHOD_DATABASE.items():
        score = 0
        reasons = []

        # Data type matching
        if data_type == "panel":
            if method_key in ("DiD", "Fixed Effects", "Event Study", "Synthetic Control"):
                score += 3
                reasons.append(f"well-suited for panel data")
        elif data_type == "cross-section":
            if method_key in ("OLS", "IV", "RDD", "Logit/Probit"):
                score += 3
                reasons.append(f"appropriate for cross-sectional data")

        # Feature matching
        if has_instrument and method_key == "IV":
            score += 5
            reasons.append("you have an instrument available")
        if has_cutoff and method_key == "RDD":
            score += 5
            reasons.append("eligibility cutoff exists")
        if has_control_group and method_key == "DiD":
            score += 4
            reasons.append("treatment/control groups available")
        if n_treated_units and n_treated_units <= 5 and method_key == "Synthetic Control":
            score += 4
            reasons.append(f"few treated units ({n_treated_units})")

        # Keyword matching
        keywords = {
            "DiD": ["reform", "policy", "intervention", "before.*after", "treatment"],
            "IV": ["endogen", "instrument", "exogenous"],
            "RDD": ["threshold", "cutoff", "eligib", "discontinuit"],
            "Synthetic Control": ["case study", "single.*state", "one.*country"],
            "Event Study": ["event", "dynamic", "pre-trend", "timing"],
        }
        for kw_list in [keywords.get(method_key, [])]:
            for kw in kw_list:
                if re.search(kw, question_lower):
                    score += 2
                    reasons.append(f"question mentions '{kw}'")
                    break

        if score > 0:
            recommendations.append({
                "method": method_key,
                "full_name": info["full_name"],
                "score": score,
                "reasons": reasons,
                "best_for": info["best_for"],
                "requires": info["requires"],
                "strengths": info["strengths"],
                "weaknesses": info["weaknesses"],
                "stata_command": info["stata_command"],
            })

    # Sort by score
    recommendations.sort(key=lambda r: -r["score"])

    # If no matches, suggest OLS as baseline
    if not recommendations:
        ols = _METHOD_DATABASE["OLS"]
        recommendations.append({
            "method": "OLS",
            "full_name": ols["full_name"],
            "score": 1,
            "reasons": ["baseline method — consider stronger identification"],
            "best_for": ols["best_for"],
            "requires": ols["requires"],
            "strengths": ols["strengths"],
            "weaknesses": ols["weaknesses"],
            "stata_command": ols["stata_command"],
        })

    return {
        "recommendations": recommendations[:5],
        "input": {
            "question": question,
            "data_type": data_type,
            "outcome": outcome_var,
            "treatment": treatment,
        },
    }


# ═══════════════════════════════════════════════════════════════════
# #6 — Peer Review Simulator
# ═══════════════════════════════════════════════════════════════════

@router.post("/api/review/simulate", tags=["Review"])
async def simulate_peer_review(request: Request):
    """Simulate a Reviewer 2 critique of a draft.

    Analyzes the text for common methodological weaknesses, missing
    robustness checks, citation gaps, and structural issues.
    Returns a structured review with severity ratings.
    """
    body = await request.json()
    draft_text = body.get("text", "")
    paper_sha = body.get("sha256", "")

    if not draft_text and paper_sha:
        # Try to load from ChromaDB
        collection = _get_chroma_collection()
        if collection:
            results = collection.get(where={"sha256": paper_sha}, include=["documents"], limit=5)
            if results["documents"]:
                draft_text = " ".join(results["documents"][:3])[:10000]

    if not draft_text:
        return JSONResponse(status_code=400, content={"error": "Provide 'text' or 'sha256'"})

    text_lower = draft_text.lower()
    review_comments = []

    # ── Check 1: Identification Strategy ──
    id_keywords = ["identification", "causal", "endogeneity", "exogenous", "instrument"]
    if not any(kw in text_lower for kw in id_keywords):
        review_comments.append({
            "category": "Identification",
            "severity": "major",
            "comment": "The paper does not clearly articulate an identification strategy. "
                       "How do you establish causality? Please discuss potential endogeneity concerns.",
            "suggestion": "Add a subsection on identification strategy and discuss threats to validity.",
        })

    # ── Check 2: Robustness Checks ──
    robustness_keywords = ["robustness", "sensitivity", "placebo", "falsification", "alternative specification"]
    if not any(kw in text_lower for kw in robustness_keywords):
        review_comments.append({
            "category": "Robustness",
            "severity": "major",
            "comment": "The paper lacks robustness checks. How sensitive are results to alternative specifications?",
            "suggestion": "Add placebo tests, alternative control groups, different functional forms, or bandwidth sensitivity.",
        })

    # ── Check 3: Sample Size / Power ──
    n_match = re.search(r'(?:N|n|observations|sample)\s*[=:]\s*([\d,]+)', draft_text)
    if n_match:
        n = int(n_match.group(1).replace(',', ''))
        if n < 100:
            review_comments.append({
                "category": "Sample Size",
                "severity": "major",
                "comment": f"Sample size of {n} is quite small. Power concerns are significant.",
                "suggestion": "Discuss statistical power. Consider bootstrapping or exact tests.",
            })
    else:
        review_comments.append({
            "category": "Data",
            "severity": "minor",
            "comment": "Sample size is not clearly stated in the text.",
            "suggestion": "Clearly report N at the top of the results section.",
        })

    # ── Check 4: Standard Errors ──
    se_keywords = ["cluster", "heteroskedast", "robust standard", "bootstrap"]
    if not any(kw in text_lower for kw in se_keywords):
        review_comments.append({
            "category": "Standard Errors",
            "severity": "moderate",
            "comment": "No mention of clustering or robust standard errors. Are standard errors appropriately computed?",
            "suggestion": "Cluster at the appropriate level (individual, state, etc.) and report robust SEs.",
        })

    # ── Check 5: External Validity ──
    external_keywords = ["external validity", "generalizab", "other context", "scalab"]
    if not any(kw in text_lower for kw in external_keywords):
        review_comments.append({
            "category": "External Validity",
            "severity": "minor",
            "comment": "The paper does not discuss external validity. Can results generalize beyond the study context?",
            "suggestion": "Add a paragraph discussing the extent to which findings apply to other settings.",
        })

    # ── Check 6: Literature Coverage ──
    citation_count = len(re.findall(r'\(\d{4}\)', draft_text))
    if citation_count < 5:
        review_comments.append({
            "category": "Literature",
            "severity": "moderate",
            "comment": f"Only {citation_count} citations detected. The literature review appears thin.",
            "suggestion": "Expand the literature review to cover key prior work and position your contribution.",
        })

    # ── Check 7: Mechanism ──
    mechanism_keywords = ["mechanism", "channel", "pathway", "mediating", "through which"]
    if not any(kw in text_lower for kw in mechanism_keywords):
        review_comments.append({
            "category": "Mechanisms",
            "severity": "minor",
            "comment": "The paper does not discuss mechanisms. Through what channels does the treatment operate?",
            "suggestion": "Add a mechanisms section exploring mediating factors.",
        })

    # ── Check 8: Heterogeneous Effects ──
    hetero_keywords = ["heterogen", "subgroup", "by gender", "by race", "by income", "interaction"]
    if not any(kw in text_lower for kw in hetero_keywords):
        review_comments.append({
            "category": "Heterogeneity",
            "severity": "minor",
            "comment": "No analysis of heterogeneous treatment effects. Are effects uniform across subgroups?",
            "suggestion": "Explore heterogeneity by key demographic or geographic dimensions.",
        })

    # Build overall assessment
    major = sum(1 for c in review_comments if c["severity"] == "major")
    moderate = sum(1 for c in review_comments if c["severity"] == "moderate")
    minor = sum(1 for c in review_comments if c["severity"] == "minor")

    if major >= 2:
        decision = "reject (major revision required)"
    elif major == 1:
        decision = "major revision"
    elif moderate >= 2:
        decision = "minor revision"
    else:
        decision = "accept with minor revisions"

    return {
        "decision": decision,
        "comments": review_comments,
        "summary": {
            "major_issues": major,
            "moderate_issues": moderate,
            "minor_issues": minor,
            "total": len(review_comments),
        },
        "text_stats": {
            "word_count": len(draft_text.split()),
            "citations_detected": citation_count,
        },
    }


# ═══════════════════════════════════════════════════════════════════
# #7 — Research Session Snapshots
# ═══════════════════════════════════════════════════════════════════

_snapshots_file = ""
_snapshots: list[dict] = []


def _load_snapshots():
    global _snapshots, _snapshots_file
    root = _data_root()
    _snapshots_file = os.path.join(root, "session_snapshots.json") if root else ""
    if _snapshots_file and os.path.isfile(_snapshots_file):
        try:
            with open(_snapshots_file, 'r') as f:
                _snapshots = json.load(f)
        except Exception:
            pass


def _save_snapshots():
    if _snapshots_file:
        try:
            os.makedirs(os.path.dirname(_snapshots_file), exist_ok=True)
            with open(_snapshots_file, 'w') as f:
                json.dump(_snapshots, f, indent=2)
        except Exception:
            pass


_load_snapshots()


@router.post("/api/session/save", tags=["Session"])
async def save_session_snapshot(request: Request):
    """Save the current workspace state as a recoverable snapshot.

    Captures: active tab, focused paper, search filters, panel states,
    notes content, and any active mission.
    """
    body = await request.json()
    name = body.get("name", f"Session {len(_snapshots) + 1}")

    snapshot = {
        "id": hashlib.md5(f"{name}{time.time()}".encode()).hexdigest()[:12],
        "name": name,
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "state": {
            "activeTab": body.get("activeTab", ""),
            "focusedPaper": body.get("focusedPaper", {}),
            "searchQuery": body.get("searchQuery", ""),
            "filters": body.get("filters", {}),
            "panelStates": body.get("panelStates", {}),
            "notes": body.get("notes", ""),
            "missionId": body.get("missionId", ""),
            "selectedTemplate": body.get("selectedTemplate", ""),
        },
    }

    _snapshots.append(snapshot)
    _save_snapshots()

    return {"status": "saved", "snapshot": snapshot}


@router.get("/api/session/list", tags=["Session"])
async def list_session_snapshots():
    """List all saved session snapshots."""
    return {
        "snapshots": [
            {"id": s["id"], "name": s["name"], "date": s["date"], "activeTab": s["state"].get("activeTab", "")}
            for s in _snapshots
        ],
        "total": len(_snapshots),
    }


@router.get("/api/session/load/{snapshot_id}", tags=["Session"])
async def load_session_snapshot(snapshot_id: str):
    """Load a previously saved session snapshot."""
    for s in _snapshots:
        if s["id"] == snapshot_id:
            return {"status": "ok", "snapshot": s}
    return JSONResponse(status_code=404, content={"error": "Snapshot not found"})


@router.delete("/api/session/delete/{snapshot_id}", tags=["Session"])
async def delete_session_snapshot(snapshot_id: str):
    """Delete a session snapshot."""
    global _snapshots
    _snapshots = [s for s in _snapshots if s["id"] != snapshot_id]
    _save_snapshots()
    return {"status": "deleted", "remaining": len(_snapshots)}


# ═══════════════════════════════════════════════════════════════════
# #8 — Notes Versioning
# ═══════════════════════════════════════════════════════════════════

_notes_versions_file = ""
_notes_versions: list[dict] = []


def _load_notes_versions():
    global _notes_versions, _notes_versions_file
    root = _data_root()
    _notes_versions_file = os.path.join(root, "notes_versions.json") if root else ""
    if _notes_versions_file and os.path.isfile(_notes_versions_file):
        try:
            with open(_notes_versions_file, 'r') as f:
                _notes_versions = json.load(f)
        except Exception:
            pass


def _save_notes_versions():
    if _notes_versions_file:
        try:
            os.makedirs(os.path.dirname(_notes_versions_file), exist_ok=True)
            with open(_notes_versions_file, 'w') as f:
                json.dump(_notes_versions, f, indent=2)
        except Exception:
            pass


_load_notes_versions()


@router.post("/api/notes/version", tags=["Notes"])
async def save_notes_version(request: Request):
    """Save a versioned snapshot of your notes.

    Creates a new version entry with the full text,
    word-level diff from previous version, and metadata.
    """
    body = await request.json()
    text = body.get("text", "")
    label = body.get("label", "")

    if not text:
        return JSONResponse(status_code=400, content={"error": "text required"})

    version = {
        "version": len(_notes_versions) + 1,
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "label": label or f"v{len(_notes_versions) + 1}",
        "text": text,
        "word_count": len(text.split()),
        "char_count": len(text),
    }

    # Compute diff from previous version
    if _notes_versions:
        prev_text = _notes_versions[-1]["text"]
        prev_words = prev_text.split()
        new_words = text.split()

        # Simple line-level diff
        prev_lines = set(prev_text.split('\n'))
        new_lines = set(text.split('\n'))
        added_lines = new_lines - prev_lines
        removed_lines = prev_lines - new_lines

        version["diff"] = {
            "words_added": max(0, len(new_words) - len(prev_words)),
            "words_removed": max(0, len(prev_words) - len(new_words)),
            "lines_added": len(added_lines),
            "lines_removed": len(removed_lines),
            "added_preview": list(added_lines)[:5],
            "removed_preview": list(removed_lines)[:5],
        }
    else:
        version["diff"] = {"words_added": len(text.split()), "words_removed": 0,
                           "lines_added": len(text.split('\n')), "lines_removed": 0}

    _notes_versions.append(version)

    # Keep only last 50 versions
    if len(_notes_versions) > 50:
        _notes_versions[:] = _notes_versions[-50:]

    _save_notes_versions()

    return {"status": "saved", "version": version["version"], "versions_total": len(_notes_versions)}


@router.post("/api/notes/commit", tags=["Notes"])
async def commit_notes(request: Request):
    """Commit current notes to a permanent version — like 'git commit'.

    Alias for /api/notes/version with a 'commit' label.
    """
    body = await request.json()
    text = body.get("text", "")
    message = body.get("message", body.get("label", ""))
    if not text:
        return JSONResponse(status_code=400, content={"error": "text required"})

    # Delegate to save_notes_version logic
    from starlette.requests import Request as _R
    version_body = {"text": text, "label": message or f"Commit {len(_notes_versions) + 1}"}

    # Inline the version-saving logic
    version = {
        "version": len(_notes_versions) + 1,
        "timestamp": time.time(),
        "date": time.strftime("%Y-%m-%d %H:%M"),
        "label": version_body["label"],
        "text": text,
        "word_count": len(text.split()),
        "char_count": len(text),
        "committed": True,
    }
    if _notes_versions:
        prev_words = _notes_versions[-1]["text"].split()
        new_words = text.split()
        version["diff"] = {
            "words_added": max(0, len(new_words) - len(prev_words)),
            "words_removed": max(0, len(prev_words) - len(new_words)),
        }
    else:
        version["diff"] = {"words_added": len(text.split()), "words_removed": 0}
    _notes_versions.append(version)
    if len(_notes_versions) > 50:
        _notes_versions[:] = _notes_versions[-50:]
    _save_notes_versions()
    return {"status": "committed", "version": version["version"], "label": version["label"]}


@router.get("/api/notes/versions", tags=["Notes"])
async def get_notes_versions():
    """Get all note versions (without full text — use /api/notes/version/{n} for that)."""
    return {
        "versions": [
            {
                "version": v["version"],
                "date": v["date"],
                "label": v["label"],
                "word_count": v["word_count"],
                "diff": v.get("diff", {}),
            }
            for v in _notes_versions
        ],
        "total": len(_notes_versions),
    }


@router.get("/api/notes/version/{version_num}", tags=["Notes"])
async def get_notes_version(version_num: int):
    """Get a specific note version with full text."""
    for v in _notes_versions:
        if v["version"] == version_num:
            return {"status": "ok", "version": v}
    return JSONResponse(status_code=404, content={"error": f"Version {version_num} not found"})


@router.post("/api/notes/restore/{version_num}", tags=["Notes"])
async def restore_notes_version(version_num: int):
    """Restore notes to a previous version (creates a new version from old text)."""
    for v in _notes_versions:
        if v["version"] == version_num:
            # Create a new version with the restored text
            restored = {
                "version": len(_notes_versions) + 1,
                "timestamp": time.time(),
                "date": time.strftime("%Y-%m-%d %H:%M"),
                "label": f"Restored from v{version_num}",
                "text": v["text"],
                "word_count": v["word_count"],
                "char_count": v["char_count"],
                "diff": {"restored_from": version_num},
            }
            _notes_versions.append(restored)
            _save_notes_versions()
            return {"status": "restored", "new_version": restored["version"], "from_version": version_num}
    return JSONResponse(status_code=404, content={"error": f"Version {version_num} not found"})


# ═══════════════════════════════════════════════════════════════════
# Autopilot Status Dashboard
# ═══════════════════════════════════════════════════════════════════

@router.get("/api/autopilot/status", tags=["Autopilot"])
async def autopilot_status():
    """Return full autopilot intelligence status.

    Shows: auto-classified papers, relations logged, training pairs
    captured, RQs detected, and complete event bus wiring.
    """
    data_root = _data_root()
    stats = {
        "active": True,
        "triggers": [
            "paper.indexed → auto_classify + auto_relate + reading_init",
            "paper.focused → reading_focused",
            "paper.deconstructed → reading_done + concept_tracker",
            "chat.response → training_capture + method_suggest",
            "export.started → export_review + socratic_review",
        ],
    }

    # Count files
    counts = {
        "paper_relations": 0,
        "auto_claims": 0,
        "training_pairs": 0,
        "rq_versions": len(_rq_history),
        "notes_versions": len(_notes_versions),
    }
    if data_root:
        for label, filename in [("paper_relations", "paper_relations.jsonl"),
                                ("auto_claims", "auto_claims.jsonl"),
                                ("training_pairs", "edith_master_train.jsonl")]:
            fpath = os.path.join(data_root, filename)
            if os.path.isfile(fpath):
                try:
                    with open(fpath) as f:
                        counts[label] = sum(1 for _ in f)
                except Exception:
                    pass
    stats["counts"] = counts

    # Get event bus status
    try:
        from server.event_bus import bus
        stats["event_bus"] = bus.status
        stats["wiring"] = bus.wiring_map
    except Exception:
        stats["event_bus"] = {"error": "not available"}

    # Reading progress summary
    try:
        from server.routes.flywheel import _reading_progress
        status_counts = {"unread": 0, "reading": 0, "done": 0}
        for p in _reading_progress.values():
            s = p.get("status", "unread")
            status_counts[s] = status_counts.get(s, 0) + 1
        stats["reading_progress"] = status_counts
    except Exception:
        pass

    return stats

