"""
Zotero Bridge — Pull library, collections, and items from Zotero API v3.
===========================================================================
Uses ZOTERO_API_KEY + ZOTERO_USER_ID from environment.
"""
import json
import logging
import os
import urllib.request
import urllib.error
from pathlib import Path

log = logging.getLogger("edith.zotero_bridge")

class ZoteroBridge:
    """Connector for the Zotero REST API v3."""

    BASE_URL = "https://api.zotero.org"

    def __init__(self, api_key: str = "", user_id: str = ""):
        self.api_key = api_key or os.environ.get("ZOTERO_API_KEY", "")
        self.user_id = user_id or os.environ.get("ZOTERO_USER_ID", "")

    def _get(self, endpoint: str, params: dict = None) -> dict | list | None:
        url = f"{self.BASE_URL}/users/{self.user_id}{endpoint}"
        if params:
            qs = "&".join(f"{k}={v}" for k, v in params.items())
            url += f"?{qs}"
        req = urllib.request.Request(url, headers={
            "Zotero-API-Key": self.api_key,
            "Zotero-API-Version": "3",
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            log.warning(f"Zotero API error: {e.code} {e.reason}")
            return None
        except Exception as e:
            log.warning(f"Zotero request failed: {e}")
            return None

    def fetch_library(self, limit: int = 100) -> list[dict]:
        """Fetch all items from the user's Zotero library."""
        items = []
        start = 0
        while True:
            batch = self._get("/items", {"limit": min(limit, 100), "start": start, "format": "json"})
            if not batch:
                break
            items.extend(batch)
            if len(batch) < 100:
                break
            start += len(batch)
        return items

    def fetch_collections(self) -> list[dict]:
        """Fetch all collections."""
        return self._get("/collections", {"format": "json"}) or []

    def fetch_item_children(self, item_key: str) -> list[dict]:
        """Fetch child items (notes, attachments) for an item."""
        return self._get(f"/items/{item_key}/children", {"format": "json"}) or []

    def export_for_indexing(self) -> dict:
        """Export full Zotero library in EDITH-indexable format."""
        raw_items = self.fetch_library()
        papers = []
        for item in raw_items:
            data = item.get("data", {})
            if data.get("itemType") in ("attachment", "note"):
                continue
            authors = []
            for creator in data.get("creators", []):
                name = f"{creator.get('firstName', '')} {creator.get('lastName', '')}".strip()
                if name:
                    authors.append(name)
            papers.append({
                "title": data.get("title", ""),
                "authors": authors,
                "year": data.get("date", "")[:4],
                "abstract": data.get("abstractNote", ""),
                "tags": [t.get("tag", "") for t in data.get("tags", [])],
                "doi": data.get("DOI", ""),
                "url": data.get("url", ""),
                "source": "zotero",
                "key": item.get("key", ""),
            })
        return {
            "papers": papers,
            "stats": {"total": len(papers)},
        }

    def status(self) -> dict:
        """Check if Zotero is configured and reachable."""
        if not self.api_key or not self.user_id:
            return {"available": False, "configured": False, "reason": "ZOTERO_API_KEY or ZOTERO_USER_ID not set"}
        test = self._get("/items", {"limit": 1, "format": "json"})
        if test is not None:
            return {"available": True, "configured": True, "user_id": self.user_id}
        return {"available": False, "configured": True, "reason": "API request failed"}
