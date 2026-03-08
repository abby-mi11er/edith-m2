"""
Auto-Annotator — Stata/R Output → Dissertation Notes in Real-Time
===================================================================
Closes the loop: every regression you run is automatically "translated"
into dissertation-ready prose and pushed to your notes.

When you run a regression in Stata or R:
1. The Watcher detects the output file (.log, .smcl, .Rout)
2. The Parser extracts coefficients, standard errors, significance, R²
3. The Translator converts statistics into plain-English interpretation
4. The Annotator writes a draft paragraph citing the model
5. The Bridge pushes it to your Notion/Theory Vault notes

You never "interpret" the same data twice.
"""

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.auto_annotator")


# ═══════════════════════════════════════════════════════════════════
# Regression Output Parsers
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CoefficientResult:
    """A single coefficient from a regression."""
    variable: str
    coefficient: float
    std_error: float
    t_stat: float = 0.0
    p_value: float = 1.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    significant: bool = False
    stars: str = ""

    @property
    def significance_level(self) -> str:
        if self.p_value < 0.001:
            return "p < 0.001"
        elif self.p_value < 0.01:
            return "p < 0.01"
        elif self.p_value < 0.05:
            return "p < 0.05"
        elif self.p_value < 0.10:
            return "p < 0.10"
        return "not significant"

    @property
    def direction(self) -> str:
        if self.coefficient > 0:
            return "positive"
        elif self.coefficient < 0:
            return "negative"
        return "null"


@dataclass
class RegressionResult:
    """Full parsed regression output."""
    model_type: str  # "OLS", "IV/2SLS", "FE", "RE", "logit", "probit"
    dependent_var: str
    coefficients: list[CoefficientResult]
    n_obs: int = 0
    r_squared: float = 0.0
    adj_r_squared: float = 0.0
    f_stat: float = 0.0
    f_p_value: float = 0.0
    source_file: str = ""
    raw_output: str = ""

    def to_dict(self) -> dict:
        return {
            "model": self.model_type,
            "dv": self.dependent_var,
            "n": self.n_obs,
            "r2": round(self.r_squared, 4),
            "coefficients": [
                {"var": c.variable, "coef": round(c.coefficient, 4),
                 "se": round(c.std_error, 4), "p": round(c.p_value, 4),
                 "sig": c.significant, "stars": c.stars}
                for c in self.coefficients
            ],
        }


def parse_stata_output(text: str) -> Optional[RegressionResult]:
    """Parse Stata regression output (.log or .smcl)."""
    # Strip SMCL formatting codes
    clean = re.sub(r'\{[^}]*\}', '', text)
    clean = re.sub(r'─+|═+|─', '', clean)

    # Detect model type
    model_type = "OLS"
    if "ivregress 2sls" in clean.lower() or "ivreg2" in clean.lower():
        model_type = "IV/2SLS"
    elif "xtreg" in clean.lower() and "fe" in clean.lower():
        model_type = "Fixed Effects"
    elif "xtreg" in clean.lower() and "re" in clean.lower():
        model_type = "Random Effects"
    elif "logit" in clean.lower() or "logistic" in clean.lower():
        model_type = "Logit"
    elif "probit" in clean.lower():
        model_type = "Probit"
    elif re.search(r'\breg\b', clean.lower()):
        model_type = "OLS"

    # Extract dependent variable
    dv_match = re.search(r'(?:regress?|xtreg|ivregress|logit|probit)\s+(\w+)', clean, re.I)
    dependent_var = dv_match.group(1) if dv_match else "unknown"

    # Extract N
    n_match = re.search(r'Number of obs\s*=\s*([\d,]+)', clean)
    n_obs = int(n_match.group(1).replace(",", "")) if n_match else 0

    # Extract R²
    r2_match = re.search(r'R-squared\s*=\s*([\d.]+)', clean)
    r_squared = float(r2_match.group(1)) if r2_match else 0.0

    adj_r2_match = re.search(r'Adj R-squared\s*=\s*([\d.]+)', clean)
    adj_r_squared = float(adj_r2_match.group(1)) if adj_r2_match else 0.0

    # Extract F-stat
    f_match = re.search(r'F\(\s*\d+,\s*\d+\)\s*=\s*([\d.]+)', clean)
    f_stat = float(f_match.group(1)) if f_match else 0.0

    f_p_match = re.search(r'Prob > F\s*=\s*([\d.]+)', clean)
    f_p_value = float(f_p_match.group(1)) if f_p_match else 0.0

    # Extract coefficients
    # Stata format: variable | coef | std.err | t/z | P>|t| | [95% CI]
    coef_pattern = re.compile(
        r'(\w[\w.]*)\s*\|\s*(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+([\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)'
    )

    coefficients = []
    for match in coef_pattern.finditer(clean):
        var_name = match.group(1)
        if var_name.lower() in ("_cons", "cons"):
            var_name = "(constant)"

        coef = float(match.group(2))
        se = float(match.group(3))
        t = float(match.group(4))
        p = float(match.group(5))
        ci_lo = float(match.group(6))
        ci_hi = float(match.group(7))

        stars = ""
        if p < 0.001:
            stars = "***"
        elif p < 0.01:
            stars = "**"
        elif p < 0.05:
            stars = "*"
        elif p < 0.10:
            stars = "†"

        coefficients.append(CoefficientResult(
            variable=var_name, coefficient=coef, std_error=se,
            t_stat=t, p_value=p, ci_lower=ci_lo, ci_upper=ci_hi,
            significant=p < 0.05, stars=stars,
        ))

    if not coefficients:
        return None

    return RegressionResult(
        model_type=model_type, dependent_var=dependent_var,
        coefficients=coefficients, n_obs=n_obs,
        r_squared=r_squared, adj_r_squared=adj_r_squared,
        f_stat=f_stat, f_p_value=f_p_value,
        source_file="", raw_output=text[:2000],
    )


def parse_r_output(text: str) -> Optional[RegressionResult]:
    """Parse R regression output (lm/glm summary)."""
    # Detect model type
    model_type = "OLS"
    if "glm(" in text.lower() and "binomial" in text.lower():
        model_type = "Logit"
    elif "plm(" in text.lower():
        model_type = "Fixed Effects"
    elif "ivreg(" in text.lower():
        model_type = "IV/2SLS"

    # Extract dependent variable from Call section
    call_match = re.search(r'(?:lm|glm|plm)\((.+?)~', text)
    dependent_var = call_match.group(1).strip() if call_match else "unknown"

    # Extract coefficients
    # R format: variable   Estimate  Std. Error  t value  Pr(>|t|)
    coef_pattern = re.compile(
        r'(\w[\w.:]*)\s+(-?[\d.e+-]+)\s+(-?[\d.e+-]+)\s+(-?[\d.e+-]+)\s+([\d.e+-]+)\s*(\*{0,3}|\.)?'
    )

    coefficients = []
    in_coefficients = False
    for line in text.split("\n"):
        if "Estimate" in line and "Std. Error" in line:
            in_coefficients = True
            continue
        if in_coefficients and "---" in line:
            continue
        if in_coefficients and line.strip() == "":
            in_coefficients = False
            continue

        if in_coefficients:
            match = coef_pattern.match(line.strip())
            if match:
                var_name = match.group(1)
                if var_name == "(Intercept)":
                    var_name = "(constant)"

                try:
                    coef = float(match.group(2))
                    se = float(match.group(3))
                    t = float(match.group(4))
                    p = float(match.group(5))
                except ValueError:
                    continue

                stars = match.group(6) or ""
                coefficients.append(CoefficientResult(
                    variable=var_name, coefficient=coef, std_error=se,
                    t_stat=t, p_value=p, significant=p < 0.05, stars=stars,
                ))

    if not coefficients:
        return None

    # Extract R²
    r2_match = re.search(r'Multiple R-squared:\s*([\d.]+)', text)
    r_squared = float(r2_match.group(1)) if r2_match else 0.0

    adj_r2_match = re.search(r'Adjusted R-squared:\s*([\d.]+)', text)
    adj_r_squared = float(adj_r2_match.group(1)) if adj_r2_match else 0.0

    # Extract F-stat
    f_match = re.search(r'F-statistic:\s*([\d.]+)', text)
    f_stat = float(f_match.group(1)) if f_match else 0.0

    # Extract N (degrees of freedom)
    df_match = re.search(r'on\s+\d+\s+and\s+(\d+)\s+DF', text)
    n_obs = int(df_match.group(1)) + len(coefficients) if df_match else 0

    return RegressionResult(
        model_type=model_type, dependent_var=dependent_var,
        coefficients=coefficients, n_obs=n_obs,
        r_squared=r_squared, adj_r_squared=adj_r_squared,
        f_stat=f_stat, source_file="", raw_output=text[:2000],
    )


# ═══════════════════════════════════════════════════════════════════
# The Translator — Statistics → Dissertation Prose
# ═══════════════════════════════════════════════════════════════════

def translate_to_prose(result: RegressionResult,
                        context: str = "") -> str:
    """Convert regression output to dissertation-ready prose.

    "Every time you run a regression, the meaning of those results
    is automatically written into your notes."
    """
    if not result.coefficients:
        return "No significant results to interpret."

    # Find the key coefficient (most significant non-constant)
    key_coefs = [c for c in result.coefficients
                 if c.variable != "(constant)" and c.significant]
    key_coefs.sort(key=lambda c: c.p_value)

    prose = []

    # Opening sentence
    prose.append(
        f"Using a {result.model_type} model with "
        f"{result.dependent_var} as the dependent variable "
        f"(N = {result.n_obs:,}), "
    )

    if result.r_squared > 0:
        prose.append(
            f"the model explains {result.r_squared:.1%} of the variance "
            f"(R² = {result.r_squared:.4f}). "
        )

    # Key findings
    if key_coefs:
        for i, coef in enumerate(key_coefs[:3]):
            direction = "positively" if coef.coefficient > 0 else "negatively"
            magnitude = abs(coef.coefficient)

            if i == 0:
                prose.append(
                    f"The primary finding is that {coef.variable} is "
                    f"{direction} associated with {result.dependent_var} "
                    f"(β = {coef.coefficient:.4f}, SE = {coef.std_error:.4f}, "
                    f"{coef.significance_level}). "
                )

                # Substantive interpretation
                if magnitude > 1:
                    prose.append(
                        f"A one-unit increase in {coef.variable} is associated "
                        f"with a {magnitude:.2f}-unit {'increase' if coef.coefficient > 0 else 'decrease'} "
                        f"in {result.dependent_var}, holding other variables constant. "
                    )
                else:
                    prose.append(
                        f"The effect size suggests that {coef.variable} has a "
                        f"{'substantively meaningful' if magnitude > 0.1 else 'modest'} "
                        f"relationship with {result.dependent_var}. "
                    )
            else:
                prose.append(
                    f"Additionally, {coef.variable} shows a {direction} "
                    f"effect (β = {coef.coefficient:.4f}, {coef.significance_level}). "
                )
    else:
        # No significant results
        prose.append(
            "No statistically significant relationships were detected "
            "at conventional significance levels. "
        )

    # Non-significant controls worth noting
    non_sig = [c for c in result.coefficients
               if c.variable != "(constant)" and not c.significant]
    if non_sig and key_coefs:
        var_names = ", ".join(c.variable for c in non_sig[:3])
        prose.append(
            f"The control variables ({var_names}) did not reach "
            f"statistical significance in this specification. "
        )

    # Model fit commentary
    if result.f_p_value > 0 and result.f_p_value < 0.05:
        prose.append(
            f"The overall model is statistically significant "
            f"(F = {result.f_stat:.2f}, p = {result.f_p_value:.4f}). "
        )

    return "".join(prose)


def translate_to_latex_table(result: RegressionResult,
                              label: str = "tab:regression") -> str:
    """Convert regression output to a LaTeX table."""
    latex = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{result.model_type} Results: {result.dependent_var}}}",
        f"\\label{{{label}}}",
        "\\begin{tabular}{lcccc}",
        "\\hline\\hline",
        "Variable & Coefficient & Std. Error & $t$-stat & $p$-value \\\\",
        "\\hline",
    ]

    for coef in result.coefficients:
        stars = coef.stars.replace("†", "$^\\dagger$").replace("*", "$^*$")
        latex.append(
            f"{coef.variable} & {coef.coefficient:.4f}{stars} & "
            f"{coef.std_error:.4f} & {coef.t_stat:.2f} & {coef.p_value:.4f} \\\\"
        )

    latex.extend([
        "\\hline",
        f"$N$ & \\multicolumn{{4}}{{c}}{{{result.n_obs:,}}} \\\\",
        f"$R^2$ & \\multicolumn{{4}}{{c}}{{{result.r_squared:.4f}}} \\\\",
        "\\hline\\hline",
        "\\multicolumn{5}{l}{\\footnotesize $^\\dagger p < 0.10$, $^* p < 0.05$, "
        "$^{**} p < 0.01$, $^{***} p < 0.001$} \\\\",
        "\\end{tabular}",
        "\\end{table}",
    ])

    return "\n".join(latex)


# ═══════════════════════════════════════════════════════════════════
# The Auto-Annotator — File Watcher + Translation Pipeline
# ═══════════════════════════════════════════════════════════════════

class AutoAnnotator:
    """Watches Stata/R output and auto-generates dissertation annotations.

    Usage:
        annotator = AutoAnnotator()

        # Process a specific output file
        result = annotator.annotate_file("/path/to/model1.log")

        # Scan a directory for new outputs
        results = annotator.scan_and_annotate()

        # Get the latest annotation
        print(annotator.latest_annotation)
    """

    def __init__(self, bolt_path: str = "", watch_dirs: list[str] = None):
        try:
            from server.vault_config import VAULT_ROOT
            _default = str(VAULT_ROOT)
        except ImportError:
            _default = os.environ.get("EDITH_DATA_ROOT", ".")
        self._bolt_path = bolt_path or _default
        self._watch_dirs = watch_dirs or [
            os.path.join(self._bolt_path, "VAULT", "STATA_OUTPUT"),
            os.path.join(self._bolt_path, "VAULT", "R_OUTPUT"),
            os.path.join(self._bolt_path, "VAULT", "DATASETS"),
        ]
        self._annotations: list[dict] = []
        self._processed_files: set[str] = set()
        self._load_processed_index()

    def annotate_file(self, file_path: str) -> dict:
        """Process a single Stata/R output file and generate annotation."""
        path = Path(file_path)
        if not path.exists():
            return {"error": f"File not found: {file_path}"}

        text = path.read_text(errors="ignore")
        ext = path.suffix.lower()

        # Parse based on file type
        result = None
        if ext in (".log", ".smcl", ".do"):
            result = parse_stata_output(text)
        elif ext in (".rout", ".r", ".rmd"):
            result = parse_r_output(text)
        else:
            # Try both parsers
            result = parse_stata_output(text) or parse_r_output(text)

        if not result:
            return {"error": "Could not parse regression output", "file": file_path}

        result.source_file = str(path)

        # Generate prose translation
        prose = translate_to_prose(result)

        # Generate LaTeX table
        latex_table = translate_to_latex_table(result, label=f"tab:{path.stem}")

        # Build annotation
        annotation = {
            "source_file": str(path),
            "model_type": result.model_type,
            "dependent_var": result.dependent_var,
            "prose": prose,
            "latex_table": latex_table,
            "regression": result.to_dict(),
            "timestamp": time.time(),
        }

        self._annotations.append(annotation)
        self._processed_files.add(str(path))
        self._save_annotation(annotation, path.stem)
        self._save_processed_index()

        return annotation

    def scan_and_annotate(self) -> dict:
        """Scan watch directories for new output files and annotate them."""
        new_files = []
        for watch_dir in self._watch_dirs:
            watch_path = Path(watch_dir)
            if not watch_path.exists():
                continue

            for ext in ["*.log", "*.smcl", "*.Rout", "*.rout"]:
                for f in watch_path.rglob(ext):
                    if str(f) not in self._processed_files:
                        new_files.append(str(f))

        if not new_files:
            return {
                "scanned": True,
                "new_files": 0,
                "message": "No new output files to annotate.",
            }

        results = []
        for file_path in new_files[:20]:  # Cap at 20
            result = self.annotate_file(file_path)
            results.append({
                "file": file_path,
                "success": "error" not in result,
                "model": result.get("model_type", ""),
                "prose_preview": result.get("prose", "")[:200],
            })

        successful = sum(1 for r in results if r["success"])
        return {
            "scanned": True,
            "new_files": len(new_files),
            "annotated": successful,
            "results": results,
        }

    def _save_annotation(self, annotation: dict, stem: str):
        """Save an annotation to the Vault."""
        notes_dir = Path(self._bolt_path) / "VAULT" / "ANNOTATIONS"
        notes_dir.mkdir(parents=True, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M")

        # Save as markdown
        md_path = notes_dir / f"{stem}_{ts}.md"
        md_content = (
            f"# Auto-Annotation: {annotation['model_type']} — {annotation['dependent_var']}\n\n"
            f"**Source**: `{annotation['source_file']}`\n"
            f"**Generated**: {time.strftime('%B %d, %Y %H:%M')}\n\n"
            f"## Interpretation\n\n{annotation['prose']}\n\n"
            f"## LaTeX Table\n\n```latex\n{annotation['latex_table']}\n```\n"
        )
        try:
            md_path.write_text(md_content)
        except Exception:
            pass

    def _load_processed_index(self):
        """Load the set of already-processed files."""
        idx_path = Path(self._bolt_path) / "VAULT" / "ANNOTATIONS" / "_processed.json"
        if idx_path.exists():
            try:
                self._processed_files = set(json.loads(idx_path.read_text()))
            except Exception:
                pass

    def _save_processed_index(self):
        """Persist the processed files index."""
        idx_dir = Path(self._bolt_path) / "VAULT" / "ANNOTATIONS"
        idx_dir.mkdir(parents=True, exist_ok=True)
        idx_path = idx_dir / "_processed.json"
        try:
            idx_path.write_text(json.dumps(list(self._processed_files)))
        except Exception:
            pass

    @property
    def latest_annotation(self) -> Optional[dict]:
        return self._annotations[-1] if self._annotations else None

    @property
    def status(self) -> dict:
        return {
            "annotations_generated": len(self._annotations),
            "files_processed": len(self._processed_files),
            "watch_dirs": self._watch_dirs,
        }


# Global instance
auto_annotator = AutoAnnotator()
