"""
DAGitty Bridge — Build Directed Acyclic Graphs via R subprocess.
=================================================================
Uses R with the dagitty package to construct causal DAGs,
find adjustment sets, and identify backdoor paths.
No API key needed — requires local R installation with dagitty.
"""
import json
import logging
import os
import subprocess
import shutil
import tempfile

log = logging.getLogger("edith.dagitty_bridge")


class DAGittyBridge:
    """R-subprocess bridge for DAGitty causal graph analysis."""

    def __init__(self):
        self.r_path = shutil.which("Rscript") or shutil.which("R")

    def _run_r(self, script: str) -> dict | None:
        """Execute an R script and return JSON output."""
        if not self.r_path:
            return None
        with tempfile.NamedTemporaryFile(mode="w", suffix=".R", delete=False) as f:
            f.write(script)
            f.flush()
            try:
                result = subprocess.run(
                    [self.r_path, "--vanilla", f.name],
                    capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0:
                    # Parse the last JSON line from stdout
                    for line in reversed(result.stdout.strip().split("\n")):
                        line = line.strip()
                        if line.startswith("{") or line.startswith("["):
                            return json.loads(line)
                log.warning(f"R script error: {result.stderr[:300]}")
                return None
            except subprocess.TimeoutExpired:
                log.warning("R script timed out")
                return None
            except Exception as e:
                log.warning(f"R execution failed: {e}")
                return None
            finally:
                os.unlink(f.name)

    def build_dag(self, variables: list[str], edges: list[dict]) -> dict:
        """Build a DAG from variables and directed edges.

        Args:
            variables: List of variable names
            edges: List of {"from": "X", "to": "Y"} dicts
        """
        edge_strs = " ".join(f"{e['from']} -> {e['to']}" for e in edges)
        dag_spec = f"dag {{ {edge_strs} }}"

        script = f'''
library(dagitty)
library(jsonlite)
g <- dagitty("{dag_spec}")
result <- list(
    variables = names(g),
    edges = as.data.frame(edges(g)),
    ancestors = lapply(names(g), function(v) ancestors(g, v)),
    descendants = lapply(names(g), function(v) descendants(g, v)),
    is_acyclic = isAcyclic(g)
)
cat(toJSON(result, auto_unbox=TRUE))
'''
        result = self._run_r(script)
        if result:
            return {"dag": result, "spec": dag_spec}
        return {
            "dag": {"variables": variables, "edges": edges, "is_acyclic": True},
            "spec": dag_spec,
            "note": "R/dagitty not available — returning raw spec",
        }

    def find_adjustment_sets(self, dag_spec: str, exposure: str, outcome: str) -> dict:
        """Find minimal sufficient adjustment sets for causal identification."""
        script = f'''
library(dagitty)
library(jsonlite)
g <- dagitty("{dag_spec}")
sets <- adjustmentSets(g, exposure="{exposure}", outcome="{outcome}", type="minimal")
cat(toJSON(list(adjustment_sets=sets, exposure="{exposure}", outcome="{outcome}"), auto_unbox=TRUE))
'''
        result = self._run_r(script)
        if result:
            return result
        return {
            "adjustment_sets": [],
            "exposure": exposure,
            "outcome": outcome,
            "note": "R/dagitty not available",
        }

    def find_backdoor_paths(self, dag_spec: str, exposure: str, outcome: str) -> dict:
        """Find all backdoor paths between exposure and outcome."""
        script = f'''
library(dagitty)
library(jsonlite)
g <- dagitty("{dag_spec}")
paths <- paths(g, from="{exposure}", to="{outcome}", directed=FALSE)
cat(toJSON(list(paths=paths$paths, open=paths$open), auto_unbox=TRUE))
'''
        result = self._run_r(script)
        if result:
            return result
        return {"paths": [], "open": [], "note": "R/dagitty not available"}

    def status(self) -> dict:
        if not self.r_path:
            return {"available": False, "configured": False, "reason": "Rscript not found in PATH"}
        # Check if dagitty package is installed
        check = self._run_r('library(jsonlite); cat(toJSON(list(ok=requireNamespace("dagitty", quietly=TRUE)), auto_unbox=TRUE))')
        if check and check.get("ok"):
            return {"available": True, "configured": True, "r_path": self.r_path}
        return {
            "available": False,
            "configured": True,
            "reason": "dagitty R package not installed (run: install.packages('dagitty'))",
            "r_path": self.r_path,
        }
