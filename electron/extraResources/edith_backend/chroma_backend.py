#!/usr/bin/env python3
import math
import re
from collections import Counter
from pathlib import Path


def _safe_imports():
    try:
        import chromadb  # type: ignore
        from sentence_transformers import SentenceTransformer  # type: ignore
        try:
            from sentence_transformers import CrossEncoder  # type: ignore
        except Exception:
            CrossEncoder = None
        return chromadb, SentenceTransformer, CrossEncoder
    except Exception:
        return None, None, None


_CHROMA = None
_ST = None
_CROSS = None
_CLIENT_CACHE = {}
_EMBEDDER_CACHE = {}
_RERANKER_CACHE = {}


def chroma_runtime_available() -> bool:
    global _CHROMA, _ST, _CROSS
    if _CHROMA is not None and _ST is not None:
        return True
    _CHROMA, _ST, _CROSS = _safe_imports()
    return _CHROMA is not None and _ST is not None


def _get_client(chroma_dir: str):
    if not chroma_runtime_available():
        raise RuntimeError("chroma runtime unavailable")
    key = str(Path(chroma_dir).expanduser().resolve())
    if key not in _CLIENT_CACHE:
        _CLIENT_CACHE[key] = _CHROMA.PersistentClient(path=key)
    return _CLIENT_CACHE[key]


def _get_embedder(model_name: str):
    if not chroma_runtime_available():
        raise RuntimeError("embedding runtime unavailable")
    key = model_name.strip()
    if key not in _EMBEDDER_CACHE:
        _EMBEDDER_CACHE[key] = _ST(key)
    return _EMBEDDER_CACHE[key]


def _get_reranker(model_name: str):
    if not model_name:
        return None
    if not chroma_runtime_available():
        return None
    if _CROSS is None:
        return None
    key = model_name.strip()
    if not key:
        return None
    if key not in _RERANKER_CACHE:
        _RERANKER_CACHE[key] = _CROSS(key)
    return _RERANKER_CACHE[key]


def _cosine_sim(a, b):
    if not a or not b:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += float(x) * float(y)
        na += float(x) * float(x)
        nb += float(y) * float(y)
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))


def _mmr_select(candidates, top_k: int, lambda_mult: float):
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates

    lm = max(0.0, min(1.0, float(lambda_mult)))
    selected = []
    remaining = candidates[:]

    # start with highest relevance
    remaining.sort(key=lambda x: x.get("relevance", 0.0), reverse=True)
    selected.append(remaining.pop(0))

    while remaining and len(selected) < top_k:
        best_idx = 0
        best_score = -1e9
        for idx, cand in enumerate(remaining):
            rel = float(cand.get("relevance", 0.0))
            emb = cand.get("embedding") or []
            redundancy = 0.0
            for s in selected:
                sim = _cosine_sim(emb, s.get("embedding") or [])
                if sim > redundancy:
                    redundancy = sim
            mmr = (lm * rel) - ((1.0 - lm) * redundancy)
            if mmr > best_score:
                best_score = mmr
                best_idx = idx
        selected.append(remaining.pop(best_idx))

    return selected


def _tokenize(text: str):
    return [x.lower() for x in re.findall(r"[A-Za-z0-9]{2,}", text or "")]


def _equation_like(text: str):
    t = str(text or "")
    return bool(re.search(r"[A-Za-z][A-Za-z0-9_]{0,20}\s*=\s*[^.;\n]{3,80}", t))


def _parse_csv_filter(raw: str):
    values = []
    for part in str(raw or "").split(","):
        token = part.strip().lower()
        if token and token not in values:
            values.append(token)
    return values


def _version_stage_boost(stage: str):
    s = str(stage or "").strip().lower()
    if s == "final":
        return 0.04
    if s in {"published", "accepted"}:
        return 0.03
    if s in {"preprint", "revision"}:
        return 0.01
    if s == "draft":
        return -0.03
    return 0.0


def _normalize_scores(values):
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi - lo <= 1e-9:
        return [0.5 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def _bm25_scores(doc_texts, query_tokens, k1=1.5, b=0.75):
    docs = [_tokenize(t) for t in (doc_texts or [])]
    if not docs:
        return []
    qtokens = [t for t in (query_tokens or []) if t]
    if not qtokens:
        return [0.0 for _ in docs]

    n_docs = len(docs)
    avgdl = sum(len(d) for d in docs) / float(max(1, n_docs))
    doc_freq = Counter()
    term_freqs = []
    for tokens in docs:
        tf = Counter(tokens)
        term_freqs.append(tf)
        for term in tf.keys():
            doc_freq[term] += 1

    idf = {}
    for term in qtokens:
        df = float(doc_freq.get(term, 0))
        idf[term] = math.log(1.0 + ((n_docs - df + 0.5) / (df + 0.5)))

    scores = []
    for tf, tokens in zip(term_freqs, docs):
        dl = float(max(1, len(tokens)))
        score = 0.0
        for term in qtokens:
            freq = float(tf.get(term, 0))
            if freq <= 0:
                continue
            denom = freq + k1 * (1.0 - b + b * (dl / max(1e-9, avgdl)))
            score += idf.get(term, 0.0) * ((freq * (k1 + 1.0)) / max(1e-9, denom))
        scores.append(score)
    return scores


def retrieve_local_sources(
    queries,
    chroma_dir: str,
    collection_name: str,
    embed_model: str,
    top_k: int = 8,
    pool_multiplier: int = 4,
    diversity_lambda: float = 0.65,
    bm25_weight: float = 0.35,
    rerank_model: str = "",
    rerank_top_n: int = 14,
    project: str = "",
    tag: str = "",
    section_filter: str = "",
    doc_type_filter: str = "",
    require_equations: bool = False,
    stitch_neighbors: int = 1,
    family_cap: int = 2,
):
    if not chroma_runtime_available():
        raise RuntimeError("Install chromadb and sentence-transformers for Chroma mode.")

    qlist = [str(q).strip() for q in (queries or []) if str(q).strip()]
    if not qlist:
        return []

    top_k = max(1, int(top_k))
    pool_multiplier = max(1, int(pool_multiplier))

    embedder = _get_embedder(embed_model)
    q_vectors = embedder.encode(qlist, normalize_embeddings=True).tolist()
    merged_q = [0.0 for _ in q_vectors[0]]
    for qv in q_vectors:
        for i, v in enumerate(qv):
            merged_q[i] += float(v)
    merged_q = [v / len(q_vectors) for v in merged_q]

    client = _get_client(chroma_dir)
    collection = client.get_or_create_collection(name=collection_name)

    where = {}
    if project and project != "All":
        where["project"] = project
    if tag:
        where["tag"] = tag
    where = where or None

    pool_n = max(top_k, top_k * pool_multiplier)
    candidate_map = {}

    section_filter = (section_filter or "").strip().lower()
    doc_type_filter_vals = set(_parse_csv_filter(doc_type_filter))
    family_cap = max(1, int(family_cap))

    for qv in q_vectors:
        res = collection.query(
            query_embeddings=[qv],
            n_results=pool_n,
            where=where,
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        ids = (res.get("ids") or [[]])[0]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        embeds = (res.get("embeddings") or [[]])[0]

        for rid, doc, meta, dist, emb in zip(ids, docs, metas, dists, embeds):
            section_heading = str((meta or {}).get("section_heading") or "").strip().lower()
            if section_filter and section_filter not in section_heading:
                continue
            doc_type = str((meta or {}).get("doc_type") or "").strip().lower()
            if doc_type_filter_vals and doc_type not in doc_type_filter_vals:
                continue
            eq_markers = str((meta or {}).get("equation_markers") or "").strip()
            if require_equations and not eq_markers and not _equation_like(doc or ""):
                continue
            rel = 1.0 - float(dist)
            prev = candidate_map.get(rid)
            if emb is None:
                emb_vec = []
            elif hasattr(emb, "tolist"):
                emb_vec = emb.tolist()
            else:
                emb_vec = list(emb)

            item = {
                "id": rid,
                "text": doc or "",
                "meta": meta or {},
                "vector_relevance": rel,
                "relevance": rel,
                "embedding": emb_vec,
            }
            if not prev or item["vector_relevance"] > prev["vector_relevance"]:
                candidate_map[rid] = item

    candidates = list(candidate_map.values())
    if not candidates:
        return []

    bm25_weight = max(0.0, min(1.0, float(bm25_weight)))
    query_tokens = _tokenize(" ".join(qlist))
    bm25_raw = _bm25_scores([c.get("text") or "" for c in candidates], query_tokens)
    vec_raw = [float(c.get("vector_relevance", 0.0)) for c in candidates]
    bm25_norm = _normalize_scores(bm25_raw)
    vec_norm = _normalize_scores(vec_raw)
    for idx, cand in enumerate(candidates):
        v = vec_norm[idx] if idx < len(vec_norm) else 0.0
        b = bm25_norm[idx] if idx < len(bm25_norm) else 0.0
        combined = ((1.0 - bm25_weight) * v) + (bm25_weight * b)
        stage_boost = _version_stage_boost((cand.get("meta") or {}).get("version_stage"))
        combined = max(0.0, min(1.0, combined + stage_boost))
        cand["vector_score"] = round(v, 4)
        cand["bm25_score"] = round(b, 4)
        cand["version_stage_boost"] = round(stage_boost, 4)
        cand["relevance"] = combined

    selected = _mmr_select(candidates, top_k=top_k, lambda_mult=diversity_lambda)

    rerank_model = (rerank_model or "").strip()
    rerank_top_n = max(1, int(rerank_top_n))
    reranker = _get_reranker(rerank_model)
    if reranker and selected:
        merged_query = " ".join(qlist[:3]).strip()
        rerank_pool = selected[: min(len(selected), rerank_top_n)]
        try:
            pairs = [(merged_query, s.get("text") or "") for s in rerank_pool]
            rr_scores = reranker.predict(pairs)
            rr_norm = _normalize_scores([float(x) for x in rr_scores])
            for i, score in enumerate(rr_norm):
                rerank_pool[i]["rerank_score"] = round(float(score), 4)
                rerank_pool[i]["relevance"] = (0.65 * float(score)) + (0.35 * float(rerank_pool[i].get("relevance", 0.0)))
            rerank_pool.sort(key=lambda x: x.get("relevance", 0.0), reverse=True)
            selected = rerank_pool + selected[len(rerank_pool):]
        except Exception:
            pass

    selected.sort(key=lambda x: x.get("relevance", 0.0), reverse=True)

    stitch_neighbors = max(0, int(stitch_neighbors))
    neighbor_text = {}
    if stitch_neighbors > 0 and selected:
        neighbor_ids = set()
        for item in selected:
            meta = item.get("meta") or {}
            sha = str(meta.get("sha256") or "").strip()
            chunk = meta.get("chunk")
            try:
                chunk_i = int(chunk)
            except Exception:
                continue
            if not sha:
                continue
            for off in range(-stitch_neighbors, stitch_neighbors + 1):
                if off == 0:
                    continue
                neighbor_ids.add(f"{sha}:{chunk_i + off}")
        if neighbor_ids:
            try:
                got = collection.get(ids=list(neighbor_ids), include=["documents"])
                ids_g = got.get("ids") or []
                docs_g = got.get("documents") or []
                for rid, doc in zip(ids_g, docs_g):
                    if isinstance(rid, str):
                        neighbor_text[rid] = (doc or "")
            except Exception:
                neighbor_text = {}

    family_counts = Counter()
    capped = []
    overflow = []
    for item in selected:
        meta = item.get("meta") or {}
        family = str(meta.get("doc_family") or meta.get("sha256") or item.get("id") or "").strip()
        if family and family_counts[family] >= family_cap:
            overflow.append(item)
            continue
        if family:
            family_counts[family] += 1
        capped.append(item)
    selected = (capped + overflow)[:top_k]

    out = []
    for item in selected:
        meta = item.get("meta") or {}
        rel_path = meta.get("rel_path") or meta.get("path") or ""
        file_name = meta.get("file_name") or (Path(rel_path).name if rel_path else "")
        page = meta.get("page")
        chunk = meta.get("chunk", 0)
        section_heading = meta.get("section_heading") or ""
        figure_table_markers = (meta.get("figure_table_markers") or "").strip()
        equation_markers = (meta.get("equation_markers") or "").strip()
        title = meta.get("title") or meta.get("title_guess") or file_name or rel_path or "local_source"
        snippet = (item.get("text") or "").strip()
        stitch_span = ""
        if stitch_neighbors > 0:
            sha = str(meta.get("sha256") or "").strip()
            try:
                chunk_i = int(chunk)
            except Exception:
                chunk_i = None
            if sha and chunk_i is not None:
                parts = []
                for off in range(-stitch_neighbors, stitch_neighbors + 1):
                    rid = f"{sha}:{chunk_i + off}"
                    if off == 0:
                        parts.append((item.get("text") or "").strip())
                    elif rid in neighbor_text:
                        parts.append((neighbor_text[rid] or "").strip())
                parts = [p for p in parts if p]
                if len(parts) > 1:
                    stitched = " ".join(parts)
                    if stitched:
                        snippet = stitched
                        stitch_span = f"{max(0, chunk_i - stitch_neighbors)}-{chunk_i + stitch_neighbors}"
        if len(snippet) > 900:
            snippet = snippet[:900]

        out.append(
            {
                "title": title,
                "uri": rel_path,
                "snippet": snippet,
                "source_type": "file",
                "rel_path": rel_path,
                "file_name": file_name,
                "sha256": meta.get("sha256"),
                "chunk": chunk,
                "page": page,
                "section_heading": section_heading,
                "doc_type": meta.get("doc_type"),
                "version_stage": meta.get("version_stage"),
                "author": meta.get("author"),
                "year": meta.get("year"),
                "citation_source": meta.get("citation_source"),
                "figure_table_markers": figure_table_markers,
                "equation_markers": equation_markers,
                "doc_family": meta.get("doc_family"),
                "vault_export_id": meta.get("vault_export_id"),
                "vault_export_date": meta.get("vault_export_date"),
                "vault_custodian": meta.get("vault_custodian"),
                "vault_matter_name": meta.get("vault_matter_name"),
                "stitch_span": stitch_span,
                "score": round(float(item.get("relevance", 0.0)), 4),
                "vector_score": item.get("vector_score"),
                "bm25_score": item.get("bm25_score"),
                "rerank_score": item.get("rerank_score"),
                "version_stage_boost": item.get("version_stage_boost"),
            }
        )

    # keep retrieval order with a deterministic score tie-break
    out.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return out


def format_local_context(sources):
    blocks = []
    for i, s in enumerate(sources or [], start=1):
        rel = (s.get("rel_path") or s.get("uri") or s.get("title") or "").strip()
        chunk = s.get("chunk")
        page = s.get("page")
        header = f"[S{i}] file={rel}"
        if page:
            header += f" page={page}"
        if chunk is not None:
            header += f" chunk={chunk}"
        section_heading = (s.get("section_heading") or "").strip()
        if section_heading:
            header += f" section={section_heading}"
        snippet = (s.get("snippet") or "").strip()
        blocks.append(f"{header}\n{snippet}")
    return "\n\n".join(blocks)


def merge_sources(primary, secondary):
    out = []
    seen = set()
    for group in (primary or [], secondary or []):
        for s in group:
            key = (s.get("uri") or s.get("title") or "").strip().lower()
            if not key:
                key = (s.get("snippet") or "").strip().lower()[:120]
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(s)
    return out
