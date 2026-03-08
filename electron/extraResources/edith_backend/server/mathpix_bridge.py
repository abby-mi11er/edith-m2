"""
MathPix Bridge — OCR images and handwriting to LaTeX.
======================================================
Uses MATHPIX_APP_ID + MATHPIX_APP_KEY from environment.
Converts handwritten equations, DAGs, and images to LaTeX/tikz.
"""
import base64
import json
import logging
import os
import urllib.request
import urllib.error
from pathlib import Path

log = logging.getLogger("edith.mathpix_bridge")


class MathPixBridge:
    """MathPix OCR API connector for image → LaTeX conversion."""

    BASE_URL = "https://api.mathpix.com/v3"

    def __init__(self, app_id: str = "", app_key: str = ""):
        self.app_id = app_id or os.environ.get("MATHPIX_APP_ID", "")
        self.app_key = app_key or os.environ.get("MATHPIX_APP_KEY", "")

    def _post(self, endpoint: str, body: dict) -> dict | None:
        url = f"{self.BASE_URL}{endpoint}"
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, method="POST", headers={
            "app_id": self.app_id,
            "app_key": self.app_key,
            "Content-Type": "application/json",
        })
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            log.warning(f"MathPix API error: {e.code}")
            return None
        except Exception as e:
            log.warning(f"MathPix request failed: {e}")
            return None

    def image_to_latex(self, image_path: str) -> dict:
        """Convert an image file to LaTeX using MathPix OCR."""
        path = Path(image_path)
        if not path.exists():
            return {"error": f"File not found: {image_path}"}

        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        ext = path.suffix.lower().lstrip(".")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif"}.get(ext, "image/png")

        result = self._post("/text", {
            "src": f"data:{mime};base64,{img_b64}",
            "formats": ["latex_styled", "text"],
            "math_inline_delimiters": ["$", "$"],
            "math_display_delimiters": ["$$", "$$"],
        })
        if not result:
            return {"error": "API request failed"}

        return {
            "latex": result.get("latex_styled", ""),
            "text": result.get("text", ""),
            "confidence": result.get("confidence", 0),
            "source": image_path,
        }

    def image_bytes_to_latex(self, image_bytes: bytes, mime: str = "image/png") -> dict:
        """Convert raw image bytes to LaTeX."""
        img_b64 = base64.b64encode(image_bytes).decode()
        result = self._post("/text", {
            "src": f"data:{mime};base64,{img_b64}",
            "formats": ["latex_styled", "text"],
        })
        if not result:
            return {"error": "API request failed"}
        return {
            "latex": result.get("latex_styled", ""),
            "text": result.get("text", ""),
            "confidence": result.get("confidence", 0),
        }

    def text_to_latex(self, text: str) -> dict:
        """Convert text/markup to clean LaTeX."""
        result = self._post("/text", {
            "src": text,
            "formats": ["latex_styled"],
            "metadata": {"type": "text"},
        })
        if not result:
            return {"error": "API request failed"}
        return {"latex": result.get("latex_styled", ""), "text": text}

    def status(self) -> dict:
        if not self.app_id or not self.app_key:
            return {"available": False, "configured": False, "reason": "MATHPIX_APP_ID or MATHPIX_APP_KEY not set"}
        return {"available": True, "configured": True, "note": "Image/handwriting → LaTeX OCR ready"}
