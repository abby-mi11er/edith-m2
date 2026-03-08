"""
Anthropic Bridge — Claude API for long-context reasoning.
==========================================================
Uses ANTHROPIC_API_KEY from environment.
Provides high-horizon reasoning via Claude's 200K+ context window.
"""
import json
import logging
import os
import urllib.request
import urllib.error

log = logging.getLogger("edith.anthropic_bridge")


class AnthropicBridge:
    """Claude API connector for long-context academic reasoning."""

    BASE_URL = "https://api.anthropic.com/v1"
    DEFAULT_MODEL = "claude-sonnet-4-20250514"

    def __init__(self, api_key: str = "", model: str = ""):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self.model = model or os.environ.get("ANTHROPIC_MODEL", self.DEFAULT_MODEL)

    def _post(self, endpoint: str, body: dict) -> dict | None:
        url = f"{self.BASE_URL}{endpoint}"
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method="POST", headers={
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body_text = e.read().decode() if e.fp else ""
            log.warning(f"Anthropic API error: {e.code} — {body_text[:200]}")
            return None
        except Exception as e:
            log.warning(f"Anthropic request failed: {e}")
            return None

    def query(self, prompt: str, context: str = "", max_tokens: int = 4096, system: str = "") -> dict:
        """Send a query to Claude with optional long context."""
        messages = []
        if context:
            messages.append({"role": "user", "content": f"Context:\n\n{context}\n\n---\n\n{prompt}"})
        else:
            messages.append({"role": "user", "content": prompt})

        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": messages,
        }
        if system:
            body["system"] = system

        result = self._post("/messages", body)
        if not result:
            return {"error": "API request failed", "answer": ""}

        answer = ""
        for block in result.get("content", []):
            if block.get("type") == "text":
                answer += block.get("text", "")

        return {
            "answer": answer,
            "model": result.get("model", self.model),
            "usage": result.get("usage", {}),
            "stop_reason": result.get("stop_reason", ""),
        }

    def audit_document(self, text: str, focus: str = "") -> dict:
        """Upload a full document for Claude to audit for logical flaws."""
        system = (
            "You are a rigorous academic auditor specializing in political science methodology. "
            "Review the following document and identify: logical flaws, unsupported claims, "
            "missing variables, methodological weaknesses, and factual errors. "
            "Be specific and cite exact passages."
        )
        prompt = f"Audit this document{' with focus on: ' + focus if focus else ''}:\n\n{text}"
        return self.query(prompt, system=system, max_tokens=8192)

    def status(self) -> dict:
        if not self.api_key:
            return {"available": False, "configured": False, "reason": "ANTHROPIC_API_KEY not set"}
        # Light check — just verify the key format
        return {
            "available": True,
            "configured": True,
            "model": self.model,
            "note": "Claude long-context reasoning ready",
        }
