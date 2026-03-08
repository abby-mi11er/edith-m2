"""
Consensus Bridge — Evidence synthesis from Consensus.app API.
==============================================================
Uses CONSENSUS_API_KEY from environment.
Pulls yes/no synthesis from 200M+ papers to verify thesis claims.
"""
import json
import logging
import os
import urllib.request
import urllib.error

log = logging.getLogger("edith.consensus_bridge")


class ConsensusBridge:
    """Consensus.app API connector for evidence synthesis."""

    BASE_URL = "https://api.consensus.app/api/v1"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.environ.get("CONSENSUS_API_KEY", "")

    def _get(self, endpoint: str, params: dict = None) -> dict | None:
        url = f"{self.BASE_URL}{endpoint}"
        if params:
            qs = "&".join(f"{k}={urllib.parse.quote_plus(str(v))}" for k, v in params.items())
            url += f"?{qs}"
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            log.warning(f"Consensus API error: {e.code}")
            return None
        except Exception as e:
            log.warning(f"Consensus request failed: {e}")
            return None

    def check_claim(self, claim: str) -> dict:
        """Check whether scientific evidence supports a claim."""
        result = self._get("/search", {"query": claim, "page_size": 10})
        if not result:
            return {"error": "API request failed", "claim": claim}

        papers = []
        for item in result.get("results", []):
            papers.append({
                "title": item.get("title", ""),
                "authors": item.get("authors", []),
                "year": item.get("year"),
                "abstract": item.get("abstract", ""),
                "journal": item.get("journal", ""),
                "doi": item.get("doi", ""),
                "consensus_label": item.get("label", ""),
            })

        return {
            "claim": claim,
            "synthesis": result.get("synthesis", ""),
            "consensus_meter": result.get("meter", {}),
            "papers": papers,
            "total_results": result.get("total", len(papers)),
        }

    def search(self, query: str, page_size: int = 20) -> dict:
        """Search Consensus for papers relevant to a query."""
        result = self._get("/search", {"query": query, "page_size": page_size})
        if not result:
            return {"error": "API request failed", "query": query}
        return result

    def status(self) -> dict:
        if not self.api_key:
            return {"available": False, "configured": False, "reason": "CONSENSUS_API_KEY not set"}
        return {"available": True, "configured": True, "note": "Evidence synthesis from 200M+ papers"}
