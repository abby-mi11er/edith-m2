"""
Perplexity Bridge — Sonar API for real-time fact-checking.
============================================================
Uses PERPLEXITY_API_KEY from environment.
Provides live verification of policy claims and recent events.
"""
import json
import logging
import os
import urllib.request
import urllib.error

log = logging.getLogger("edith.perplexity_bridge")


class PerplexityBridge:
    """Perplexity Sonar API connector for real-time verification."""

    BASE_URL = "https://api.perplexity.ai"

    def __init__(self, api_key: str = ""):
        self.api_key = api_key or os.environ.get("PERPLEXITY_API_KEY", "")

    def _post(self, endpoint: str, body: dict) -> dict | None:
        url = f"{self.BASE_URL}{endpoint}"
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method="POST", headers={
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            log.warning(f"Perplexity API error: {e.code}")
            return None
        except Exception as e:
            log.warning(f"Perplexity request failed: {e}")
            return None

    def verify_claim(self, claim: str) -> dict:
        """Fact-check a specific claim using Perplexity's real-time search."""
        body = {
            "model": "sonar",
            "messages": [
                {"role": "system", "content": (
                    "You are a fact-checker. Verify the following claim using the most recent "
                    "available information. State whether it is TRUE, FALSE, PARTIALLY TRUE, "
                    "or UNVERIFIABLE. Cite your sources."
                )},
                {"role": "user", "content": f"Verify this claim: {claim}"},
            ],
        }
        result = self._post("/chat/completions", body)
        if not result:
            return {"error": "API request failed", "verdict": "UNKNOWN"}

        answer = ""
        for choice in result.get("choices", []):
            msg = choice.get("message", {})
            answer += msg.get("content", "")

        return {
            "claim": claim,
            "verdict": answer,
            "citations": result.get("citations", []),
            "model": result.get("model", "sonar"),
        }

    def search_recent(self, query: str) -> dict:
        """Search for the most recent information on a topic."""
        body = {
            "model": "sonar",
            "messages": [
                {"role": "system", "content": "Provide the most recent information available. Include dates and sources."},
                {"role": "user", "content": query},
            ],
        }
        result = self._post("/chat/completions", body)
        if not result:
            return {"error": "API request failed"}

        answer = ""
        for choice in result.get("choices", []):
            answer += choice.get("message", {}).get("content", "")

        return {
            "query": query,
            "answer": answer,
            "citations": result.get("citations", []),
        }

    def status(self) -> dict:
        if not self.api_key:
            return {"available": False, "configured": False, "reason": "PERPLEXITY_API_KEY not set"}
        return {"available": True, "configured": True, "note": "Sonar real-time search ready"}
