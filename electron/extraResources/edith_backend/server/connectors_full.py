"""
§4.0: Full API Connectors
Semantic Scholar, CrossRef, ORCID, RefWorks — external academic API integrations.
"""
import os
import json
import logging
import time
from typing import Optional
from urllib.parse import quote_plus

log = logging.getLogger("edith.connectors")

try:
    import requests
except ImportError:
    requests = None


# ---------------------------------------------------------------------------
# Semantic Scholar API (free, no key required for basic use)
# ---------------------------------------------------------------------------

class SemanticScholarConnector:
    """Search and retrieve papers from Semantic Scholar's API.
    https://api.semanticscholar.org/
    """
    BASE_URL = "https://api.semanticscholar.org/graph/v1"
    
    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.environ.get("S2_API_KEY", "")
        self._session = requests.Session() if requests else None
        if self.api_key and self._session:
            self._session.headers["x-api-key"] = self.api_key
    
    def search(self, query: str, limit: int = 10, year: str = "") -> list[dict]:
        """Search papers by keyword query."""
        if not self._session:
            return []
        params = {
            "query": query,
            "limit": min(limit, 100),
            "fields": "title,authors,year,abstract,citationCount,url,externalIds,venue",
        }
        if year:
            params["year"] = year
        try:
            resp = self._session.get(f"{self.BASE_URL}/paper/search", params=params, timeout=15)
            if resp.status_code == 429:
                time.sleep(2)
                resp = self._session.get(f"{self.BASE_URL}/paper/search", params=params, timeout=15)
            if resp.status_code != 200:
                log.warning(f"S2 search failed: {resp.status_code}")
                return []
            data = resp.json()
            return [self._format_paper(p) for p in data.get("data", [])]
        except Exception as e:
            log.error(f"S2 search error: {e}")
            return []
    
    def get_paper(self, paper_id: str) -> dict:
        """Get a single paper by Semantic Scholar ID, DOI, or ArXiv ID."""
        if not self._session:
            return {}
        fields = "title,authors,year,abstract,citationCount,referenceCount,citations,references,url,externalIds,venue,tldr"
        try:
            resp = self._session.get(
                f"{self.BASE_URL}/paper/{paper_id}",
                params={"fields": fields},
                timeout=15,
            )
            if resp.status_code != 200:
                return {}
            return self._format_paper(resp.json())
        except Exception as e:
            log.error(f"S2 get_paper error: {e}")
            return {}
    
    def get_citations(self, paper_id: str, limit: int = 20) -> list[dict]:
        """Get papers that cite the given paper."""
        if not self._session:
            return []
        try:
            resp = self._session.get(
                f"{self.BASE_URL}/paper/{paper_id}/citations",
                params={"fields": "title,authors,year,citationCount,url", "limit": limit},
                timeout=15,
            )
            if resp.status_code != 200:
                return []
            return [self._format_paper(c.get("citingPaper", {})) for c in resp.json().get("data", [])]
        except Exception as e:
            log.error(f"S2 citations error: {e}")
            return []
    
    def get_references(self, paper_id: str, limit: int = 20) -> list[dict]:
        """Get papers referenced by the given paper."""
        if not self._session:
            return []
        try:
            resp = self._session.get(
                f"{self.BASE_URL}/paper/{paper_id}/references",
                params={"fields": "title,authors,year,citationCount,url", "limit": limit},
                timeout=15,
            )
            if resp.status_code != 200:
                return []
            return [self._format_paper(r.get("citedPaper", {})) for r in resp.json().get("data", [])]
        except Exception as e:
            log.error(f"S2 references error: {e}")
            return []
    
    @staticmethod
    def _format_paper(p: dict) -> dict:
        authors = p.get("authors", [])
        author_names = ", ".join(a.get("name", "") for a in authors[:5])
        if len(authors) > 5:
            author_names += f" (+{len(authors)-5} more)"
        return {
            "title": p.get("title", ""),
            "authors": author_names,
            "year": p.get("year"),
            "abstract": (p.get("abstract") or "")[:500],
            "citation_count": p.get("citationCount", 0),
            "reference_count": p.get("referenceCount", 0),
            "url": p.get("url", ""),
            "venue": p.get("venue", ""),
            "doi": (p.get("externalIds") or {}).get("DOI", ""),
            "tldr": (p.get("tldr") or {}).get("text", ""),
            "source": "semantic_scholar",
        }


# ---------------------------------------------------------------------------
# CrossRef API (free, no key required)
# ---------------------------------------------------------------------------

class CrossRefConnector:
    """Enrich papers with metadata from CrossRef using DOIs.
    https://api.crossref.org/
    """
    BASE_URL = "https://api.crossref.org"
    
    def __init__(self, email: str = ""):
        self.email = email or os.environ.get("CROSSREF_EMAIL", "")
        self._session = requests.Session() if requests else None
        if self._session:
            self._session.headers["User-Agent"] = f"EDITH/1.0 (mailto:{self.email})" if self.email else "EDITH/1.0"
    
    def lookup_doi(self, doi: str) -> dict:
        """Get full metadata for a DOI."""
        if not self._session or not doi:
            return {}
        try:
            resp = self._session.get(f"{self.BASE_URL}/works/{quote_plus(doi)}", timeout=15)
            if resp.status_code != 200:
                return {}
            msg = resp.json().get("message", {})
            return self._format_work(msg)
        except Exception as e:
            log.error(f"CrossRef DOI lookup error: {e}")
            return {}
    
    def search(self, query: str, limit: int = 10) -> list[dict]:
        """Search CrossRef for works matching query."""
        if not self._session:
            return []
        try:
            resp = self._session.get(
                f"{self.BASE_URL}/works",
                params={"query": query, "rows": limit, "sort": "relevance"},
                timeout=15,
            )
            if resp.status_code != 200:
                return []
            items = resp.json().get("message", {}).get("items", [])
            return [self._format_work(w) for w in items]
        except Exception as e:
            log.error(f"CrossRef search error: {e}")
            return []
    
    @staticmethod
    def _format_work(w: dict) -> dict:
        authors = w.get("author", [])
        author_str = ", ".join(
            f"{a.get('family', '')} {a.get('given', '')}".strip()
            for a in authors[:5]
        )
        title_list = w.get("title", [])
        return {
            "title": title_list[0] if title_list else "",
            "authors": author_str,
            "year": (w.get("published-print") or w.get("published-online") or {}).get("date-parts", [[None]])[0][0],
            "doi": w.get("DOI", ""),
            "journal": (w.get("container-title") or [""])[0],
            "volume": w.get("volume", ""),
            "issue": w.get("issue", ""),
            "pages": w.get("page", ""),
            "citation_count": w.get("is-referenced-by-count", 0),
            "type": w.get("type", ""),
            "url": w.get("URL", ""),
            "abstract": (w.get("abstract") or "")[:500],
            "source": "crossref",
        }


# ---------------------------------------------------------------------------
# ORCID API (free, no key for public data)
# ---------------------------------------------------------------------------

class ORCIDConnector:
    """Look up researchers and their works by ORCID ID.
    https://pub.orcid.org/
    """
    BASE_URL = "https://pub.orcid.org/v3.0"
    
    def __init__(self):
        self._session = requests.Session() if requests else None
        if self._session:
            self._session.headers["Accept"] = "application/json"
    
    def get_person(self, orcid_id: str) -> dict:
        """Get researcher profile by ORCID."""
        if not self._session or not orcid_id:
            return {}
        try:
            resp = self._session.get(f"{self.BASE_URL}/{orcid_id}/person", timeout=15)
            if resp.status_code != 200:
                return {}
            data = resp.json()
            name = data.get("name", {})
            return {
                "orcid": orcid_id,
                "given_name": (name.get("given-names") or {}).get("value", ""),
                "family_name": (name.get("family-name") or {}).get("value", ""),
                "biography": ((data.get("biography") or {}).get("content") or "")[:500],
            }
        except Exception as e:
            log.error(f"ORCID person error: {e}")
            return {}
    
    def get_works(self, orcid_id: str) -> list[dict]:
        """Get all works for an ORCID researcher."""
        if not self._session or not orcid_id:
            return []
        try:
            resp = self._session.get(f"{self.BASE_URL}/{orcid_id}/works", timeout=20)
            if resp.status_code != 200:
                return []
            data = resp.json()
            works = []
            for group in data.get("group", [])[:50]:
                summaries = group.get("work-summary", [])
                if not summaries:
                    continue
                s = summaries[0]
                title = (s.get("title") or {}).get("title", {}).get("value", "")
                year = None
                pub_date = s.get("publication-date") or {}
                if pub_date.get("year"):
                    year = pub_date["year"].get("value")
                
                # Extract DOI if available
                doi = ""
                for eid in (s.get("external-ids") or {}).get("external-id", []):
                    if eid.get("external-id-type") == "doi":
                        doi = eid.get("external-id-value", "")
                        break
                
                works.append({
                    "title": title,
                    "year": int(year) if year else None,
                    "doi": doi,
                    "type": s.get("type", ""),
                    "source": "orcid",
                })
            return works
        except Exception as e:
            log.error(f"ORCID works error: {e}")
            return []


# ---------------------------------------------------------------------------
# RefWorks Export
# ---------------------------------------------------------------------------

def format_refworks_export(sources: list[dict]) -> str:
    """Format sources as RefWorks Tagged Format (.txt).
    
    RefWorks uses a tagged format similar to RIS but with different tags.
    """
    lines = []
    for s in sources:
        lines.append("RT Journal Article")
        if s.get("title"):
            lines.append(f"T1 {s['title']}")
        if s.get("authors"):
            for author in s["authors"].split(","):
                author = author.strip()
                if author:
                    lines.append(f"A1 {author}")
        elif s.get("author"):
            lines.append(f"A1 {s['author']}")
        if s.get("year"):
            lines.append(f"YR {s['year']}")
        if s.get("journal") or s.get("venue"):
            lines.append(f"JF {s.get('journal') or s.get('venue')}")
        if s.get("doi"):
            lines.append(f"DO {s['doi']}")
        if s.get("url"):
            lines.append(f"UL {s['url']}")
        if s.get("abstract"):
            lines.append(f"AB {s['abstract'][:1000]}")
        if s.get("volume"):
            lines.append(f"VO {s['volume']}")
        if s.get("pages"):
            lines.append(f"SP {s['pages']}")
        lines.append("ER")
        lines.append("")
    
    return "\n".join(lines)


def format_ris_export(sources: list[dict]) -> str:
    """Format sources as RIS (Research Information Systems) format."""
    lines = []
    for s in sources:
        lines.append("TY  - JOUR")
        if s.get("title"):
            lines.append(f"TI  - {s['title']}")
        if s.get("authors"):
            for author in s["authors"].split(","):
                author = author.strip()
                if author:
                    lines.append(f"AU  - {author}")
        elif s.get("author"):
            lines.append(f"AU  - {s['author']}")
        if s.get("year"):
            lines.append(f"PY  - {s['year']}")
        if s.get("journal") or s.get("venue"):
            lines.append(f"JO  - {s.get('journal') or s.get('venue')}")
        if s.get("doi"):
            lines.append(f"DO  - {s['doi']}")
        if s.get("url"):
            lines.append(f"UR  - {s['url']}")
        if s.get("abstract"):
            lines.append(f"AB  - {s['abstract'][:1000]}")
        lines.append("ER  - ")
        lines.append("")
    
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level helpers for route integration
# ---------------------------------------------------------------------------

_CONNECTORS = {
    "semantic_scholar": {"name": "Semantic Scholar", "class": SemanticScholarConnector},
    "crossref": {"name": "CrossRef", "class": CrossRefConnector},
    "orcid": {"name": "ORCID", "class": ORCIDConnector},
}


def list_connectors() -> list[dict]:
    """List all available connector integrations."""
    return [
        {"id": cid, "name": info["name"], "available": requests is not None}
        for cid, info in _CONNECTORS.items()
    ]


def test_connection(connector: str) -> dict:
    """Test connectivity to a specific connector by making a lightweight API call."""
    if connector not in _CONNECTORS:
        return {"status": "error", "error": f"Unknown connector: {connector}",
                "available": list(_CONNECTORS.keys())}
    if not requests:
        return {"status": "error", "error": "requests library not installed"}
    try:
        info = _CONNECTORS[connector]
        instance = info["class"]()
        # Light-touch test: search for a known term
        if hasattr(instance, "search"):
            results = instance.search("test", limit=1)
            return {"status": "ok", "connector": connector, "sample_results": len(results)}
        return {"status": "ok", "connector": connector, "note": "No search method to test"}
    except Exception as e:
        return {"status": "error", "connector": connector, "error": str(e)}

