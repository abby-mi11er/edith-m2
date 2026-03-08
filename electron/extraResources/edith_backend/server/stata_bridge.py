"""
Stata Bridge — Live Stata subprocess execution.
==================================================
Uses STATA_PATH env var pointing to the Stata executable.
Executes Stata code, runs models, and supports estimator switching.
"""
import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger("edith.stata_bridge")


class StataBridge:
    """Live Stata subprocess for executing code and running models."""

    def __init__(self, stata_path: str = ""):
        self.stata_path = stata_path or os.environ.get("STATA_PATH", "")
        if not self.stata_path:
            # Auto-detect common Stata paths on macOS
            candidates = [
                "/usr/local/stata18/stata-mp",
                "/usr/local/stata17/stata-mp",
                "/usr/local/stata/stata-mp",
                "/Applications/Stata/StataSE.app/Contents/MacOS/StataSE",
                "/Applications/Stata/StataMP.app/Contents/MacOS/StataMP",
                "/Applications/Stata/StataBE.app/Contents/MacOS/stata",
            ]
            for c in candidates:
                if os.path.isfile(c) and os.access(c, os.X_OK):
                    self.stata_path = c
                    break
            if not self.stata_path:
                self.stata_path = shutil.which("stata-mp") or shutil.which("stata-se") or shutil.which("stata") or ""

    def execute(self, code: str, timeout: int = 120) -> dict:
        """Execute Stata code and return output."""
        if not self.stata_path:
            return {"error": "Stata not found — set STATA_PATH", "output": ""}

        with tempfile.TemporaryDirectory() as tmpdir:
            do_file = Path(tmpdir) / "edith_run.do"
            log_file = Path(tmpdir) / "edith_run.log"

            # Write the .do file with log capture
            do_content = f'log using "{log_file}", text replace\n{code}\nlog close\n'
            do_file.write_text(do_content, encoding="utf-8")

            try:
                result = subprocess.run(
                    [self.stata_path, "-b", "do", str(do_file)],
                    capture_output=True, text=True, timeout=timeout, cwd=tmpdir,
                )
                output = ""
                if log_file.exists():
                    output = log_file.read_text(encoding="utf-8", errors="replace")
                elif result.stdout:
                    output = result.stdout

                return {
                    "output": output,
                    "return_code": result.returncode,
                    "error": result.stderr if result.returncode != 0 else "",
                }
            except subprocess.TimeoutExpired:
                return {"error": f"Stata timed out after {timeout}s", "output": ""}
            except Exception as e:
                return {"error": str(e), "output": ""}

    def run_model(self, model_spec: str, data_path: str = "") -> dict:
        """Run a regression model with optional data loading."""
        code_lines = []
        if data_path:
            if data_path.endswith(".dta"):
                code_lines.append(f'use "{data_path}", clear')
            elif data_path.endswith(".csv"):
                code_lines.append(f'import delimited "{data_path}", clear')
        code_lines.append(model_spec)
        code_lines.append("estimates store m1")
        code_lines.append("ereturn list")
        return self.execute("\n".join(code_lines))

    def switch_estimator(self, current_code: str, new_estimator: str) -> dict:
        """Switch a model's estimator (e.g., logit → probit, OLS → 2SLS).

        Args:
            current_code: Current Stata model command
            new_estimator: Target estimator (probit, logit, ivregress, etc.)
        """
        # Simple estimator substitution
        estimators = ["regress", "logit", "probit", "ologit", "oprobit", "mlogit",
                       "ivregress", "tobit", "poisson", "nbreg", "heckman"]
        new_code = current_code.strip()
        for est in estimators:
            if new_code.lower().startswith(est):
                new_code = new_estimator + new_code[len(est):]
                break

        return self.execute(new_code)

    def status(self) -> dict:
        if not self.stata_path:
            return {"available": False, "configured": False, "reason": "Stata not found — set STATA_PATH"}
        if not os.path.isfile(self.stata_path):
            return {"available": False, "configured": True, "reason": f"STATA_PATH does not exist: {self.stata_path}"}
        return {
            "available": True,
            "configured": True,
            "stata_path": self.stata_path,
            "note": "Live Stata execution ready",
        }
