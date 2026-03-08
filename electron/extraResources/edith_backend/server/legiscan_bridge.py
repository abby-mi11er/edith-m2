"""
LegiScan Bridge — Monitor state legislature for policy keywords.
=================================================================
Uses LEGISCAN_API_KEY from environment.
Tracks bills in TX (or any state) mentioning SNAP, welfare, charity, etc.
"""
import json
import logging
import os
import urllib.request
import urllib.error
from urllib.parse import quote_plus

log = logging.getLogger("edith.legiscan_bridge")


class LegiScanBridge:
    """LegiScan API connector for legislative monitoring."""

    BASE_URL = "https://api.legiscan.com"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.environ.get("LEGISCAN_API_KEY", "")

    def _get(self, params: dict) -> dict | None:
        params["key"] = self.api_key
        qs = "&".join(f"{k}={quote_plus(str(v))}" for k, v in params.items())
        url = f"{self.BASE_URL}/?{qs}"
        req = urllib.request.Request(url)
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            log.warning(f"LegiScan API error: {e.code}")
            return None
        except Exception as e:
            log.warning(f"LegiScan request failed: {e}")
            return None

    def search_bills(self, query: str, state: str = "TX", year: int = 2) -> dict:
        """Search for bills matching keywords in a state.

        Args:
            query: Search keywords (e.g., "SNAP welfare")
            state: Two-letter state code (default: TX)
            year: 1=current session, 2=recent sessions
        """
        result = self._get({"op": "getSearch", "state": state, "query": query, "year": year})
        if not result or result.get("status") != "OK":
            return {"error": "Search failed", "query": query}

        search_result = result.get("searchresult", {})
        bills = []
        for key, item in search_result.items():
            if key in ("summary", "count", "page", "range"):
                continue
            if isinstance(item, dict):
                bills.append({
                    "bill_id": item.get("bill_id"),
                    "number": item.get("bill_number", ""),
                    "title": item.get("title", ""),
                    "state": item.get("state", state),
                    "last_action": item.get("last_action", ""),
                    "last_action_date": item.get("last_action_date", ""),
                    "url": item.get("url", ""),
                    "relevance": item.get("relevance", 0),
                })

        return {
            "query": query,
            "state": state,
            "bills": bills,
            "total": search_result.get("summary", {}).get("count", len(bills)),
        }

    def get_bill(self, bill_id: int) -> dict:
        """Get full bill details by LegiScan bill ID."""
        result = self._get({"op": "getBill", "id": bill_id})
        if not result or result.get("status") != "OK":
            return {"error": f"Bill {bill_id} not found"}
        return result.get("bill", {})

    def monitor_keywords(self, keywords: list[str], state: str = "TX") -> dict:
        """Monitor multiple keywords and return aggregated results."""
        all_bills = {}
        for keyword in keywords:
            result = self.search_bills(keyword, state)
            for bill in result.get("bills", []):
                bid = bill.get("bill_id")
                if bid and bid not in all_bills:
                    bill["matched_keywords"] = [keyword]
                    all_bills[bid] = bill
                elif bid in all_bills:
                    all_bills[bid]["matched_keywords"].append(keyword)

        bills = sorted(all_bills.values(), key=lambda b: len(b.get("matched_keywords", [])), reverse=True)
        return {
            "keywords": keywords,
            "state": state,
            "bills": bills,
            "total": len(bills),
        }

    def status(self) -> dict:
        if not self.api_key:
            return {"available": False, "configured": False, "reason": "LEGISCAN_API_KEY not set"}
        return {"available": True, "configured": True, "note": "Legislative monitoring for TX and all 50 states"}
