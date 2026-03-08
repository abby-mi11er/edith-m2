"""
Bayesian Causal Discovery — DAGs, Backdoor Paths, and Colliders
================================================================
Pedagogical Mirror Feature 2: Force theoretical rigor through causality.

Standard regressions show correlation. This module forces you to think
in DIRECTED ACYCLIC GRAPHS. Winnie doesn't just run a model — she
generates the DAG showing:
- Which variables are confounders (backdoor paths)
- Where you have colliders (opening bad paths)
- What the identification strategy actually requires

If you miss a collider, Winnie highlights it in red and explains the bias.

Architecture:
    Theoretical Model → Variable Registry → DAG Construction →
    Backdoor Criterion Check → Collider Detection → LaTeX Output

The key insight: A model without a DAG is just curve fitting.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.causal_discovery")


# ═══════════════════════════════════════════════════════════════════
# Variable Types — The Building Blocks of Causal Reasoning
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CausalVariable:
    """A variable in the causal model."""
    name: str
    label: str  # Human-readable label
    var_type: str  # "treatment", "outcome", "confounder", "mediator", "collider", "instrument"
    description: str = ""
    observed: bool = True  # False = latent / unobserved
    measurement: str = ""  # How this is operationalized
    data_source: str = ""  # Where the data comes from

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "label": self.label,
            "type": self.var_type,
            "description": self.description,
            "observed": self.observed,
            "measurement": self.measurement,
            "data_source": self.data_source,
        }


@dataclass
class CausalEdge:
    """A directed edge in the DAG: cause → effect."""
    source: str  # Variable name
    target: str  # Variable name
    edge_type: str = "causal"  # "causal", "confounding", "selection", "mediation"
    strength: str = "hypothesized"  # "established", "hypothesized", "contested"
    mechanism: str = ""  # The theoretical reason for this arrow
    citations: list[str] = field(default_factory=list)  # Papers supporting this edge

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "type": self.edge_type,
            "strength": self.strength,
            "mechanism": self.mechanism,
            "citations": self.citations,
        }


# ═══════════════════════════════════════════════════════════════════
# Causal Model — The DAG Container
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CausalModel:
    """A Directed Acyclic Graph (DAG) for causal reasoning."""
    name: str
    description: str = ""
    variables: dict[str, CausalVariable] = field(default_factory=dict)
    edges: list[CausalEdge] = field(default_factory=list)
    treatment: str = ""
    outcome: str = ""
    created_at: str = ""

    def add_variable(self, name: str, label: str, var_type: str = "confounder",
                     **kwargs) -> CausalVariable:
        var = CausalVariable(name=name, label=label, var_type=var_type, **kwargs)
        self.variables[name] = var
        return var

    def add_edge(self, source: str, target: str, edge_type: str = "causal",
                 mechanism: str = "", **kwargs) -> CausalEdge:
        edge = CausalEdge(
            source=source, target=target, edge_type=edge_type,
            mechanism=mechanism, **kwargs,
        )
        self.edges.append(edge)
        return edge

    def get_parents(self, node: str) -> list[str]:
        """Get all direct parents (causes) of a node."""
        return [e.source for e in self.edges if e.target == node]

    def get_children(self, node: str) -> list[str]:
        """Get all direct children (effects) of a node."""
        return [e.target for e in self.edges if e.source == node]

    def get_ancestors(self, node: str, visited: set = None) -> set[str]:
        """Get all ancestors of a node (recursive)."""
        if visited is None:
            visited = set()
        parents = self.get_parents(node)
        for p in parents:
            if p not in visited:
                visited.add(p)
                self.get_ancestors(p, visited)
        return visited

    def get_descendants(self, node: str, visited: set = None) -> set[str]:
        """Get all descendants of a node (recursive)."""
        if visited is None:
            visited = set()
        children = self.get_children(node)
        for c in children:
            if c not in visited:
                visited.add(c)
                self.get_descendants(c, visited)
        return visited

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "treatment": self.treatment,
            "outcome": self.outcome,
            "variables": {k: v.to_dict() for k, v in self.variables.items()},
            "edges": [e.to_dict() for e in self.edges],
            "created_at": self.created_at,
        }


# ═══════════════════════════════════════════════════════════════════
# Causal Discovery Engine — Analysis and Diagnostics
# ═══════════════════════════════════════════════════════════════════

class CausalDiscoveryEngine:
    """Analyze DAGs for identification, confounding, and collider bias.

    Core capabilities:
    1. Backdoor criterion: Which variables must be conditioned on?
    2. Collider detection: Where are you opening bad paths?
    3. Instrument validity: Does your IV satisfy the exclusion restriction?
    4. LaTeX DAG output: Publish-ready DAG diagrams
    """

    def __init__(self, data_root: str = ""):
        self._data_root = data_root or os.environ.get("EDITH_APP_DATA_DIR", "")
        self._models: dict[str, CausalModel] = {}

    def create_model(self, name: str, treatment: str, outcome: str,
                     description: str = "") -> CausalModel:
        """Create a new causal model (DAG)."""
        model = CausalModel(
            name=name,
            description=description,
            treatment=treatment,
            outcome=outcome,
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
        )
        # Auto-add treatment and outcome
        model.add_variable(treatment, treatment.replace("_", " ").title(), "treatment")
        model.add_variable(outcome, outcome.replace("_", " ").title(), "outcome")
        model.add_edge(treatment, outcome, mechanism="Core causal hypothesis")

        self._models[name] = model
        return model

    def get_model(self, name: str) -> Optional[CausalModel]:
        return self._models.get(name)

    def list_models(self) -> list[dict]:
        return [
            {
                "name": m.name,
                "treatment": m.treatment,
                "outcome": m.outcome,
                "variable_count": len(m.variables),
                "edge_count": len(m.edges),
            }
            for m in self._models.values()
        ]

    # ──────────────────────────────────────────────────────────────
    # Backdoor Criterion — The Gold Standard
    # ──────────────────────────────────────────────────────────────

    def check_backdoor_criterion(self, model_name: str,
                                  conditioning_set: list[str] = None) -> dict:
        """Check if the backdoor criterion is satisfied.

        The backdoor criterion is met when conditioning on a set Z:
        1. No node in Z is a descendant of the treatment
        2. Z blocks every path between treatment and outcome that
           contains an arrow into the treatment

        Returns analysis with pass/fail and recommendations.
        """
        model = self._models.get(model_name)
        if not model:
            return {"error": f"Model '{model_name}' not found"}

        treatment = model.treatment
        outcome = model.outcome
        conditioning = set(conditioning_set or [])

        # Check rule 1: No conditioned variable is a descendant of treatment
        descendants = model.get_descendants(treatment)
        bad_descendants = conditioning & descendants
        rule1_pass = len(bad_descendants) == 0

        # Find all backdoor paths (paths with an arrow into treatment)
        backdoor_paths = self._find_backdoor_paths(model, treatment, outcome)

        # Check rule 2: All backdoor paths are blocked
        unblocked = []
        for path in backdoor_paths:
            if not self._is_path_blocked(model, path, conditioning):
                unblocked.append(path)

        rule2_pass = len(unblocked) == 0

        # Find confounders that SHOULD be conditioned on
        confounders = self._identify_confounders(model, treatment, outcome)
        missing_controls = [c for c in confounders if c not in conditioning]

        # Build recommendation
        if rule1_pass and rule2_pass:
            status = "IDENTIFIED"
            recommendation = (
                "The causal effect is identified. Your conditioning set blocks "
                "all backdoor paths without introducing collider bias."
            )
        elif not rule1_pass:
            status = "COLLIDER_BIAS"
            recommendation = (
                f"WARNING: You are conditioning on {bad_descendants}, which is a "
                f"descendant of the treatment '{treatment}'. This opens a collider "
                f"path and introduces bias. Remove these from your controls."
            )
        else:
            status = "CONFOUNDED"
            recommendation = (
                f"The effect is NOT identified. {len(unblocked)} backdoor path(s) "
                f"remain unblocked. Consider adding {missing_controls} to your "
                f"conditioning set."
            )

        return {
            "status": status,
            "rule1_pass": rule1_pass,
            "rule2_pass": rule2_pass,
            "backdoor_paths": [" → ".join(p) for p in backdoor_paths],
            "unblocked_paths": [" → ".join(p) for p in unblocked],
            "confounders": confounders,
            "missing_controls": missing_controls,
            "conditioning_set": list(conditioning),
            "recommendation": recommendation,
            "bad_descendants": list(bad_descendants),
        }

    def _find_backdoor_paths(self, model: CausalModel,
                              treatment: str, outcome: str) -> list[list[str]]:
        """Find all backdoor paths between treatment and outcome.

        A backdoor path is any path that:
        1. Starts at treatment
        2. Ends at outcome
        3. Has an arrow pointing INTO the treatment on the first edge
        """
        paths = []
        # Get parents of treatment — these are the starting points of backdoor paths
        parents = model.get_parents(treatment)

        for parent in parents:
            # Find all paths from parent to outcome (not through treatment)
            found = self._find_all_paths(model, parent, outcome,
                                          visited={treatment})
            for path in found:
                paths.append([treatment, parent] + path[1:])

        return paths

    def _find_all_paths(self, model: CausalModel, start: str, end: str,
                         visited: set = None, path: list = None) -> list[list[str]]:
        """Find all undirected paths between two nodes."""
        if visited is None:
            visited = set()
        if path is None:
            path = [start]

        if start == end:
            return [list(path)]

        visited = visited | {start}
        found_paths = []

        # Get all neighbors (both directions)
        neighbors = set(model.get_parents(start)) | set(model.get_children(start))

        for neighbor in neighbors:
            if neighbor not in visited:
                new_path = path + [neighbor]
                found_paths.extend(
                    self._find_all_paths(model, neighbor, end, visited, new_path)
                )

        return found_paths

    def _is_path_blocked(self, model: CausalModel, path: list[str],
                          conditioning: set[str]) -> bool:
        """Check if a path is blocked by the conditioning set.

        A path is blocked if:
        - A non-collider on the path is conditioned on, OR
        - A collider on the path is NOT conditioned on (and no descendant is)
        """
        if len(path) < 3:
            return False

        for i in range(1, len(path) - 1):
            node = path[i]
            prev_node = path[i - 1]
            next_node = path[i + 1]

            # Check if this node is a collider (arrows point IN from both sides)
            is_collider = (
                node in model.get_children(prev_node) and
                node in model.get_children(next_node)
            )

            if is_collider:
                # Collider: path is blocked UNLESS we condition on it or its descendants
                descendants = model.get_descendants(node)
                if node not in conditioning and not (descendants & conditioning):
                    return True  # Blocked by collider
            else:
                # Non-collider: path is blocked IF we condition on it
                if node in conditioning:
                    return True  # Blocked by conditioning

        return False  # Path is not blocked

    def _identify_confounders(self, model: CausalModel,
                               treatment: str, outcome: str) -> list[str]:
        """Identify variables that confound the treatment-outcome relationship."""
        treatment_ancestors = model.get_ancestors(treatment)
        outcome_ancestors = model.get_ancestors(outcome)

        # A confounder is a common cause (ancestor of both treatment and outcome)
        confounders = treatment_ancestors & outcome_ancestors
        return list(confounders)

    # ──────────────────────────────────────────────────────────────
    # Collider Detection — The Most Common Error
    # ──────────────────────────────────────────────────────────────

    def detect_colliders(self, model_name: str) -> list[dict]:
        """Find all collider variables in the DAG.

        A collider is a variable with two or more arrows pointing INTO it.
        Conditioning on a collider opens a spurious path (collider bias).
        """
        model = self._models.get(model_name)
        if not model:
            return []

        colliders = []
        for var_name, var in model.variables.items():
            parents = model.get_parents(var_name)
            if len(parents) >= 2:
                colliders.append({
                    "variable": var_name,
                    "label": var.label,
                    "parents": parents,
                    "warning": (
                        f"'{var.label}' is a collider. Conditioning on it opens "
                        f"a spurious path between {' and '.join(parents)}. "
                        f"Do NOT include this in your regression unless you intend "
                        f"to analyze the selection mechanism."
                    ),
                })

        return colliders

    # ──────────────────────────────────────────────────────────────
    # Instrument Validity Check
    # ──────────────────────────────────────────────────────────────

    def check_instrument(self, model_name: str, instrument: str) -> dict:
        """Check if a variable is a valid instrument (IV).

        A valid instrument must satisfy:
        1. Relevance: Instrument affects treatment
        2. Exclusion: Instrument affects outcome ONLY through treatment
        3. Independence: Instrument is not confounded with outcome
        """
        model = self._models.get(model_name)
        if not model:
            return {"error": f"Model '{model_name}' not found"}

        treatment = model.treatment
        outcome = model.outcome

        # Check relevance: instrument → treatment
        instrument_children = model.get_children(instrument)
        relevance = treatment in instrument_children

        # Check exclusion: no direct path from instrument to outcome except through treatment
        # Remove treatment from the graph and check if instrument can reach outcome
        descendants_without_treatment = set()
        self._get_descendants_excluding(model, instrument, {treatment}, descendants_without_treatment)
        exclusion = outcome not in descendants_without_treatment

        # Check independence: instrument has no common cause with outcome
        instrument_ancestors = model.get_ancestors(instrument)
        outcome_ancestors = model.get_ancestors(outcome)
        common_causes = instrument_ancestors & outcome_ancestors
        independence = len(common_causes) == 0

        valid = relevance and exclusion and independence

        issues = []
        if not relevance:
            issues.append(f"'{instrument}' does not directly cause '{treatment}'")
        if not exclusion:
            issues.append(f"'{instrument}' has a direct effect on '{outcome}' — exclusion restriction violated")
        if not independence:
            issues.append(f"'{instrument}' shares common causes with '{outcome}': {common_causes}")

        return {
            "instrument": instrument,
            "valid": valid,
            "relevance": relevance,
            "exclusion": exclusion,
            "independence": independence,
            "issues": issues,
            "recommendation": (
                "Valid instrument. Proceed with 2SLS estimation."
                if valid else
                "INVALID instrument. " + " AND ".join(issues) + ". Reconsider your identification strategy."
            ),
        }

    def _get_descendants_excluding(self, model: CausalModel, node: str,
                                     exclude: set, result: set):
        """Get descendants excluding certain nodes."""
        for child in model.get_children(node):
            if child not in exclude and child not in result:
                result.add(child)
                self._get_descendants_excluding(model, child, exclude, result)

    # ──────────────────────────────────────────────────────────────
    # LaTeX DAG Output — Publish-Ready Diagrams
    # ──────────────────────────────────────────────────────────────

    def generate_latex_dag(self, model_name: str) -> str:
        """Generate LaTeX code for the DAG using TikZ.

        Output is ready to paste into a paper.
        """
        model = self._models.get(model_name)
        if not model:
            return "% Model not found"

        variables = list(model.variables.values())
        n = len(variables)

        # Position variables in a circle
        positions = {}
        for i, var in enumerate(variables):
            angle = (2 * 3.14159 * i) / max(n, 1)
            x = round(3 * __import__("math").cos(angle), 1)
            y = round(3 * __import__("math").sin(angle), 1)
            positions[var.name] = (x, y)

        # Build TikZ code
        lines = [
            r"\begin{tikzpicture}[",
            r"  >=stealth, node distance=2.5cm,",
            r"  every node/.style={draw, rounded corners, minimum size=1cm, font=\small},",
            r"  treatment/.style={fill=blue!20},",
            r"  outcome/.style={fill=green!20},",
            r"  confounder/.style={fill=orange!20},",
            r"  collider/.style={fill=red!20},",
            r"  unobserved/.style={dashed, fill=gray!10},",
            r"]",
        ]

        # Nodes
        collider_names = {c["variable"] for c in self.detect_colliders(model_name)}
        for var in variables:
            x, y = positions[var.name]
            style = var.var_type
            if var.name in collider_names:
                style = "collider"
            if not var.observed:
                style = "unobserved"

            lines.append(
                f"  \\node[{style}] ({var.name}) at ({x},{y}) {{{var.label}}};"
            )

        lines.append("")

        # Edges
        for edge in model.edges:
            style = ""
            if edge.strength == "contested":
                style = "[dashed, red]"
            elif edge.strength == "hypothesized":
                style = "[dashed]"

            lines.append(
                f"  \\draw[->{style}] ({edge.source}) -- ({edge.target});"
            )

        lines.append(r"\end{tikzpicture}")
        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────
    # Pedagogical: Generate Teaching Prompt for DAG Analysis
    # ──────────────────────────────────────────────────────────────

    def generate_dag_teaching_prompt(self, model_name: str) -> str:
        """Generate a teaching prompt that walks the student through DAG analysis.

        This is what Winnie uses in Socratic mode to teach causal reasoning.
        """
        model = self._models.get(model_name)
        if not model:
            return "No model found."

        colliders = self.detect_colliders(model_name)
        backdoor = self.check_backdoor_criterion(model_name)

        prompt = (
            f"Let's analyze your causal model: '{model.name}'\n\n"
            f"TREATMENT: {model.treatment}\n"
            f"OUTCOME: {model.outcome}\n"
            f"VARIABLES: {', '.join(model.variables.keys())}\n\n"
        )

        # Teach about confounders
        if backdoor.get("confounders"):
            prompt += (
                f"CONFOUNDERS DETECTED: {', '.join(backdoor['confounders'])}\n"
                f"These share a common cause with both your treatment and outcome.\n"
                f"QUESTION: Why does failing to control for these produce bias? "
                f"Draw the path.\n\n"
            )

        # Teach about colliders
        if colliders:
            prompt += (
                f"⚠️ COLLIDER WARNING: {', '.join(c['variable'] for c in colliders)}\n"
                f"QUESTION: What happens if you condition on a collider? "
                f"Can you explain the 'selection bias' that results?\n\n"
            )

        # Identification status
        prompt += (
            f"IDENTIFICATION STATUS: {backdoor.get('status', 'UNKNOWN')}\n"
            f"{backdoor.get('recommendation', '')}\n"
        )

        return prompt

    def save_model(self, model_name: str) -> dict:
        """Save a model to disk."""
        model = self._models.get(model_name)
        if not model:
            return {"error": "Model not found"}

        save_dir = Path(self._data_root or ".") / "VAULT" / "CAUSAL_MODELS"
        save_dir.mkdir(parents=True, exist_ok=True)
        path = save_dir / f"{model_name}.json"

        try:
            path.write_text(json.dumps(model.to_dict(), indent=2))
            return {"status": "saved", "path": str(path)}
        except Exception as e:
            return {"error": str(e)}


# Global instance
causal_engine = CausalDiscoveryEngine()
