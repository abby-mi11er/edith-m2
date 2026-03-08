"""
Overleaf Bridge — Push LaTeX drafts to Overleaf via Git integration.
=====================================================================
Uses OVERLEAF_GIT_TOKEN + OVERLEAF_PROJECT_URL from environment.
Overleaf supports Git access for v2 projects — we push via subprocess.
"""
import json
import logging
import os
import subprocess
import shutil
import tempfile
from pathlib import Path

log = logging.getLogger("edith.overleaf_bridge")


class OverleafBridge:
    """Push LaTeX content to Overleaf projects via Git."""

    def __init__(self, git_token: str = "", project_url: str = ""):
        self.git_token = git_token or os.environ.get("OVERLEAF_GIT_TOKEN", "")
        self.project_url = project_url or os.environ.get("OVERLEAF_PROJECT_URL", "")
        self._git = shutil.which("git")

    def _authenticated_url(self) -> str:
        """Insert token into the Git URL for authentication."""
        if not self.project_url or not self.git_token:
            return ""
        # Overleaf Git URL: https://git.overleaf.com/<project_id>
        # With token: https://token@git.overleaf.com/<project_id>
        if "://" in self.project_url:
            scheme, rest = self.project_url.split("://", 1)
            return f"{scheme}://{self.git_token}@{rest}"
        return self.project_url

    def push_draft(self, latex_content: str, filename: str = "edith_draft.tex",
                   commit_message: str = "E.D.I.T.H. auto-push") -> dict:
        """Push a LaTeX draft to the Overleaf project."""
        if not self._git:
            return {"error": "git not found in PATH"}
        auth_url = self._authenticated_url()
        if not auth_url:
            return {"error": "OVERLEAF_GIT_TOKEN or OVERLEAF_PROJECT_URL not set"}

        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                # Clone
                subprocess.run(
                    [self._git, "clone", "--depth", "1", auth_url, tmpdir],
                    capture_output=True, text=True, timeout=60, check=True,
                )
                # Write file
                target = Path(tmpdir) / filename
                target.write_text(latex_content, encoding="utf-8")
                # Stage + commit + push
                subprocess.run([self._git, "add", filename], cwd=tmpdir, capture_output=True, timeout=10)
                subprocess.run(
                    [self._git, "commit", "-m", commit_message, "--author", "E.D.I.T.H. <edith@localhost>"],
                    cwd=tmpdir, capture_output=True, timeout=10,
                )
                result = subprocess.run(
                    [self._git, "push", "origin", "master"],
                    cwd=tmpdir, capture_output=True, text=True, timeout=60,
                )
                if result.returncode == 0:
                    return {"status": "pushed", "filename": filename, "message": commit_message}
                else:
                    return {"error": f"git push failed: {result.stderr[:200]}"}
            except subprocess.TimeoutExpired:
                return {"error": "git operation timed out"}
            except Exception as e:
                return {"error": str(e)}

    def list_files(self) -> dict:
        """List files in the Overleaf project."""
        if not self._git:
            return {"error": "git not found"}
        auth_url = self._authenticated_url()
        if not auth_url:
            return {"error": "Not configured"}
        with tempfile.TemporaryDirectory() as tmpdir:
            try:
                subprocess.run(
                    [self._git, "clone", "--depth", "1", auth_url, tmpdir],
                    capture_output=True, timeout=60, check=True,
                )
                files = [str(p.relative_to(tmpdir)) for p in Path(tmpdir).rglob("*") if p.is_file() and ".git" not in str(p)]
                return {"files": files, "count": len(files)}
            except Exception as e:
                return {"error": str(e)}

    def status(self) -> dict:
        if not self._git:
            return {"available": False, "configured": False, "reason": "git not found in PATH"}
        if not self.git_token or not self.project_url:
            return {"available": False, "configured": False, "reason": "OVERLEAF_GIT_TOKEN or OVERLEAF_PROJECT_URL not set"}
        return {"available": True, "configured": True, "project_url": self.project_url.split("@")[-1] if "@" in self.project_url else self.project_url}
