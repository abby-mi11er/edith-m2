#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
try:
    from chroma_backend import retrieve_local_sources, format_local_context, merge_sources
except Exception:
    retrieve_local_sources = None
    format_local_context = None
    merge_sources = None


def load_env():
    candidates = []
    override = os.environ.get("EDITH_DOTENV_PATH")
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend([
        Path(__file__).parent / ".env",
        Path.cwd() / ".env",
        Path.home() / "Library" / "Application Support" / "Edith" / ".env",
    ])
    seen = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            load_dotenv(dotenv_path=p, override=False)


def build_file_search_tool(store_id: str, project: str, tag: str):
    kwargs = {"file_search_store_names": [store_id]}
    metadata = {}
    if project and project != "All":
        metadata["project"] = project
    if tag:
        metadata["tag"] = tag
    if metadata:
        kwargs["metadata_filter"] = metadata
    return types.Tool(file_search=types.FileSearch(**kwargs))


def build_web_search_tool():
    try:
        return types.Tool(google_search=types.GoogleSearch())
    except Exception:
        pass
    try:
        return types.Tool(google_search_retrieval=types.GoogleSearchRetrieval())
    except Exception:
        return None


def extract_sources(resp):
    sources = []
    candidates = getattr(resp, "candidates", []) or []
    if not candidates:
        return sources

    cand = candidates[0]
    gm = getattr(cand, "grounding_metadata", None) or getattr(cand, "groundingMetadata", None)
    if gm:
        chunks = (
            getattr(gm, "grounding_chunks", None)
            or getattr(gm, "groundingChunks", None)
            or []
        )
        for ch in chunks:
            for field in ("retrieved_context", "retrievedContext", "web", "document"):
                rc = getattr(ch, field, None)
                if not rc:
                    continue
                uri = getattr(rc, "uri", None) or getattr(rc, "url", None)
                title = (
                    getattr(rc, "title", None)
                    or getattr(rc, "displayName", None)
                    or getattr(rc, "display_name", None)
                )
                source_type = "web" if field == "web" or (uri or "").startswith("http") else "file"
                sources.append({"title": title, "uri": uri, "source_type": source_type})

    deduped = []
    seen = set()
    for s in sources:
        key = (s.get("uri") or s.get("title") or "").strip()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    return deduped


def get_text(resp):
    text = getattr(resp, "text", None)
    if text:
        return text.strip()
    cands = getattr(resp, "candidates", []) or []
    if not cands:
        return ""
    content = getattr(cands[0], "content", None)
    parts = getattr(content, "parts", None) if content else None
    if not parts:
        return ""
    out = []
    for p in parts:
        t = getattr(p, "text", None)
        if t:
            out.append(t)
    return "".join(out).strip()


def parse_backend(value: str):
    v = (value or "").strip().lower()
    return "chroma" if v == "chroma" else "google"


def main() -> int:
    load_env()

    parser = argparse.ArgumentParser(description="Smoke-test retrieval and source grounding.")
    parser.add_argument("query", help="Query text")
    parser.add_argument("--mode", default=os.environ.get("EDITH_SOURCE_MODE", "Files only"), choices=["Files only", "Web only", "Files + Web"])
    parser.add_argument("--project", default="All")
    parser.add_argument("--tag", default="")
    parser.add_argument("--section", default="")
    parser.add_argument("--doc-type", default="")
    parser.add_argument("--require-equations", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--model", default=(os.environ.get("EDITH_MODEL") or "gemini-2.5-flash"))
    parser.add_argument("--backend", default=os.environ.get("EDITH_RETRIEVAL_BACKEND", "google"), choices=["google", "chroma"])
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    store_id = os.environ.get("EDITH_STORE_ID", "").strip()
    require_citations = os.environ.get("EDITH_REQUIRE_CITATIONS", "true").lower() == "true"
    allow_web_tools = os.environ.get("EDITH_ALLOW_WEB_TOOLS", "true").lower() == "true"
    backend = parse_backend(args.backend)
    chroma_dir = os.environ.get("EDITH_CHROMA_DIR", str(Path(__file__).parent / "chroma"))
    chroma_collection = os.environ.get("EDITH_CHROMA_COLLECTION", "edith_docs")
    embed_model = os.environ.get("EDITH_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    chroma_top_k = int(os.environ.get("EDITH_CHROMA_TOP_K", "8"))
    chroma_pool_multiplier = int(os.environ.get("EDITH_CHROMA_POOL_MULTIPLIER", "4"))
    chroma_diversity_lambda = float(os.environ.get("EDITH_CHROMA_DIVERSITY_LAMBDA", "0.65"))
    chroma_bm25_weight = float(os.environ.get("EDITH_CHROMA_BM25_WEIGHT", "0.35"))
    chroma_rerank_on = os.environ.get("EDITH_CHROMA_RERANK", "true").lower() == "true"
    chroma_force_rerank_files_only = os.environ.get("EDITH_CHROMA_FORCE_RERANK_FILES_ONLY", "true").lower() == "true"
    chroma_rerank_model = os.environ.get("EDITH_CHROMA_RERANK_MODEL", "BAAI/bge-reranker-base").strip()
    chroma_rerank_top_n = int(os.environ.get("EDITH_CHROMA_RERANK_TOP_N", "14"))
    section_filter = (args.section or os.environ.get("EDITH_SECTION_FILTER", "")).strip()
    doc_type_filter = (args.doc_type or os.environ.get("EDITH_DOC_TYPE_FILTER", "")).strip()
    require_equations = bool(args.require_equations or os.environ.get("EDITH_REQUIRE_EQUATIONS", "false").lower() == "true")
    chroma_family_cap = int(os.environ.get("EDITH_CHROMA_FAMILY_CAP", "2"))

    if not api_key:
        raise SystemExit("GOOGLE_API_KEY missing.")
    if backend == "google" and not store_id:
        raise SystemExit("EDITH_STORE_ID missing for google backend.")
    if store_id and not store_id.startswith("fileSearchStores/"):
        store_id = f"fileSearchStores/{store_id}"

    client = genai.Client(api_key=api_key)

    local_sources = []
    local_context = ""
    if backend == "chroma" and args.mode in ("Files only", "Files + Web"):
        if not retrieve_local_sources:
            raise SystemExit("Chroma backend unavailable (install chromadb + sentence-transformers).")
        if args.mode == "Files only" and chroma_force_rerank_files_only:
            chroma_rerank_on = True
        local_sources = retrieve_local_sources(
            queries=[args.query],
            chroma_dir=chroma_dir,
            collection_name=chroma_collection,
            embed_model=embed_model,
            top_k=chroma_top_k,
            pool_multiplier=chroma_pool_multiplier,
            diversity_lambda=chroma_diversity_lambda,
            bm25_weight=chroma_bm25_weight,
            rerank_model=chroma_rerank_model if chroma_rerank_on else "",
            rerank_top_n=chroma_rerank_top_n,
            project=args.project,
            tag=args.tag,
            section_filter=section_filter,
            doc_type_filter=doc_type_filter,
            require_equations=require_equations,
            family_cap=max(1, int(chroma_family_cap)),
        )
        local_context = format_local_context(local_sources) if format_local_context else ""

    tools = []
    if backend == "google" and args.mode in ("Files only", "Files + Web"):
        tools.append(build_file_search_tool(store_id, args.project, args.tag))
    if allow_web_tools and args.mode in ("Web only", "Files + Web"):
        web_tool = build_web_search_tool()
        if web_tool:
            tools.append(web_tool)

    if not tools and not (backend == "chroma" and args.mode == "Files only"):
        raise SystemExit("No tools available for selected mode.")

    question = args.query
    if local_context:
        question = (
            f"{args.query}\n\n"
            "LOCAL_FILE_SOURCES:\n"
            f"{local_context}\n\n"
            "Use LOCAL_FILE_SOURCES for grounded claims. If evidence is missing, say it is not found."
        )

    prompt = (
        "Answer the question using retrieved sources only. "
        "If not found, say exactly: Not found in sources.\n\n"
        f"QUESTION:\n{question}"
    )

    if tools:
        cfg = types.GenerateContentConfig(tools=tools, temperature=0.0)
    else:
        cfg = types.GenerateContentConfig(temperature=0.0)
    resp = client.models.generate_content(model=args.model, contents=prompt, config=cfg)

    text = get_text(resp)
    remote_sources = extract_sources(resp)
    if merge_sources:
        sources = merge_sources(local_sources, remote_sources)
    else:
        sources = local_sources + remote_sources
    file_count = sum(1 for s in sources if s.get("source_type") == "file")
    web_count = sum(1 for s in sources if s.get("source_type") == "web")

    result = {
        "backend": backend,
        "mode": args.mode,
        "model": args.model,
        "project": args.project,
        "tag": args.tag,
        "section": section_filter,
        "doc_type_filter": doc_type_filter,
        "require_equations": require_equations,
        "source_count": len(sources),
        "file_source_count": file_count,
        "web_source_count": web_count,
        "local_source_count": len(local_sources),
        "remote_source_count": len(remote_sources),
        "answer_preview": text[:1200],
        "sources": sources[:20],
    }
    print(json.dumps(result, indent=2))

    if require_citations and not sources:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
