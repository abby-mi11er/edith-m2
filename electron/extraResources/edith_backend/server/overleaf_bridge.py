"""
Overleaf Bridge — Push LaTeX drafts to Overleaf via Git integration.
=====================================================================
Supports multiple Overleaf projects via OVERLEAF_PROJECTS env var (JSON).
Also supports legacy single-project via OVERLEAF_PROJECT_URL.
Uses OVERLEAF_GIT_TOKEN for authentication.
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
        # Multi-project support: OVERLEAF_PROJECTS = JSON dict {"name": "url", ...}
        self._projects: dict[str, str] = {}
        try:
            raw = os.environ.get("OVERLEAF_PROJECTS", "")
            if raw:
                self._projects = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            log.warning("Invalid OVERLEAF_PROJECTS JSON, ignoring")
        # If single project is set but not in projects dict, add it as "default"
        if self.project_url and not self._projects:
            self._projects["Default"] = self.project_url

    def _authenticated_url(self, url: str = "") -> str:
        """Insert token into the Git URL for authentication."""
        target_url = url or self.project_url
        if not target_url or not self.git_token:
            return ""
        if "://" in target_url:
            scheme, rest = target_url.split("://", 1)
            return f"{scheme}://{self.git_token}@{rest}"
        return target_url

    def list_projects(self) -> dict:
        """Return available Overleaf projects."""
        return {"projects": {name: url.split('/')[-1] for name, url in self._projects.items()}}

    def push_draft(self, latex_content: str, filename: str = "edith_draft.tex",
                   commit_message: str = "E.D.I.T.H. auto-push",
                   project_name: str = "") -> dict:
        """Push a LaTeX draft to the Overleaf project."""
        if not self._git:
            return {"error": "git not found in PATH"}
        # Resolve project URL
        if project_name and project_name in self._projects:
            target_url = self._projects[project_name]
        elif self._projects:
            target_url = list(self._projects.values())[0]
        else:
            target_url = self.project_url
        auth_url = self._authenticated_url(target_url)
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
        if not self.git_token or (not self.project_url and not self._projects):
            return {"available": False, "configured": False, "reason": "OVERLEAF_GIT_TOKEN or OVERLEAF_PROJECT_URL not set"}
        return {
            "available": True, "configured": True,
            "projects": {name: url.split('/')[-1] for name, url in self._projects.items()},
            "project_count": len(self._projects),
        }
