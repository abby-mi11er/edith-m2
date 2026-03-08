"""
Data Vibe Coding Engine — Code Generation + Sandboxed Execution
================================================================
Winnie generates Python/R/Stata code from natural language research
directives and executes it in a sandboxed environment.

Capabilities:
  - Generate Python (pandas, statsmodels, scikit-learn)
  - Generate R (tidyverse, lm, plm)
  - Generate Stata (.do files)
  - Execute Python in subprocess sandbox
  - Capture outputs (tables, plots, statistics)
  - Auto-detect data files on the drive

Exposed as:
  POST /api/vibe/generate  — generate code from directive
  POST /api/vibe/execute   — run generated code in sandbox
  POST /api/vibe/explain   — explain existing code
  GET  /api/vibe/datasets   — list available datasets
"""

import hashlib
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import threading
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.vibe")


# ═══════════════════════════════════════════════════════════════════
# A. Dataset Discovery — find data files on the drive
# ═══════════════════════════════════════════════════════════════════

_DATA_EXTENSIONS = {
    ".csv": "CSV (comma-separated)",
    ".tsv": "TSV (tab-separated)",
    ".dta": "Stata dataset",
    ".rds": "R serialized data",
    ".rdata": "R workspace",
    ".json": "JSON",
    ".jsonl": "JSON Lines",
    ".parquet": "Apache Parquet",
    ".xlsx": "Excel",
    ".xls": "Excel (legacy)",
    ".sav": "SPSS",
    ".por": "SPSS portable",
    ".feather": "Arrow Feather",
    ".sqlite": "SQLite database",
    ".db": "SQLite database",
}


_VIBE_SKIP_DIRS = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", "backups",
    "chroma", "ChromaDB", ".Spotlight-V100", ".fseventsd", ".TemporaryItems",
    "ARTEFACTS", "Connectome", "Memory", "PERSONAS", "Sovereign_Proofs",
    "Weights", "adapters", "assets", "bin", "build", "Brain",
    "Notability", "Notion_Exports", "pedagogy", "missions", "notes",
    "GIS 2", "GIS 5300", "GIS_2", "GIS_5300",
}


def discover_datasets(data_root: str = "") -> list[dict]:
    """Scan Library/Datasets/ for analysis-ready datasets.

    Users drop .dta/.csv/.xlsx files into Library/Datasets/ and they
    appear here automatically.  No recursive whole-drive scan.
    Returns list of {path, name, extension, format, size_mb}
    """
    root = data_root or os.environ.get("EDITH_DATA_ROOT", "")
    if not root or not os.path.isdir(root):
        return []

    datasets_dir = os.path.join(root, "Library", "Datasets")
    if not os.path.isdir(datasets_dir):
        return []

    datasets = []
    for dirpath, dirs, files in os.walk(datasets_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fname in files:
            if fname.startswith("._") or fname.startswith("."):
                continue
            ext = os.path.splitext(fname)[1].lower()
            if ext in _DATA_EXTENSIONS:
                fpath = os.path.join(dirpath, fname)
                try:
                    size = os.path.getsize(fpath)
                except OSError:
                    size = 0
                datasets.append({
                    "path": fpath,
                    "name": fname,
                    "extension": ext,
                    "format": _DATA_EXTENSIONS[ext],
                    "size_mb": round(size / (1024 * 1024), 2),
                    "relative": os.path.relpath(fpath, datasets_dir),
                })

    return sorted(datasets, key=lambda d: d["size_mb"], reverse=True)


# ═══════════════════════════════════════════════════════════════════
# B. Code Generation — natural language → Python/R/Stata
# ═══════════════════════════════════════════════════════════════════

_CODEGEN_SYSTEMS = {
    "python": (
        "You are E.D.I.T.H.'s data analysis code generator. Generate ONLY Python code.\n\n"
        "RULES:\n"
        "1. Use pandas for data manipulation, statsmodels for regression, "
        "matplotlib/seaborn for plots, scikit-learn for ML\n"
        "2. ALWAYS start with `import pandas as pd` and necessary imports\n"
        "3. Use `print()` for all output — results must be visible in stdout\n"
        "4. For plots, save to a file with `plt.savefig('output_plot.png', dpi=150, bbox_inches='tight')`\n"
        "5. Include comments explaining each step\n"
        "6. Handle missing data gracefully (dropna or fillna)\n"
        "7. Print summary statistics and key findings\n"
        "8. If doing regression, print coefficients, standard errors, p-values, R²\n"
        "9. Do NOT use input() or any interactive functions\n"
        "10. Wrap main logic in try/except for graceful error handling\n\n"
        "Output ONLY the code between ```python and ``` markers. No explanation outside code."
    ),
    "r": (
        "You are E.D.I.T.H.'s data analysis code generator. Generate ONLY R code.\n\n"
        "RULES:\n"
        "1. Use tidyverse for data manipulation, ggplot2 for plots, "
        "lm/plm for regression\n"
        "2. ALWAYS start with library() calls\n"
        "3. Use cat() or print() for all output\n"
        "4. For plots, use ggsave('output_plot.png', width=10, height=6)\n"
        "5. Include comments explaining each step\n"
        "6. Handle NAs with na.rm=TRUE or drop_na()\n"
        "7. Print summary() for models\n"
        "8. Use robust standard errors when appropriate (sandwich/lmtest)\n"
        "9. Do NOT use readline() or interactive functions\n\n"
        "Output ONLY the code between ```r and ``` markers."
    ),
    "stata": (
        "You are E.D.I.T.H.'s data analysis code generator. Generate ONLY Stata .do file code.\n\n"
        "RULES:\n"
        "1. Start with `clear all` and `set more off`\n"
        "2. Use `use` to load .dta files\n"
        "3. Include comments with * or //\n"
        "4. Use `reg` for OLS, `xtreg` for panel, `logit`/`probit` for binary\n"
        "5. Always `summarize` before regression\n"
        "6. Use `estpost` and `esttab` for formatted output when possible\n"
        "7. Check for heteroscedasticity with `estat hettest`\n"
        "8. Use `robust` or `cluster()` standard errors\n"
        "9. Export results with `log using output.log, replace`\n\n"
        "Output ONLY the code between ```stata and ``` markers."
    ),
}

# Method/analysis templates for common polsci tasks
_ANALYSIS_TEMPLATES = {
    "summary": "Generate descriptive statistics (mean, sd, min, max, N) for all variables",
    "correlation": "Generate a correlation matrix with significance stars",
    "ols": "Run OLS regression with robust standard errors",
    "fixed_effects": "Run fixed effects panel regression (entity and time FE)",
    "diff_in_diff": "Run difference-in-differences with treatment × post interaction",
    "logit": "Run logistic regression with marginal effects",
    "rdd": "Run regression discontinuity design (sharp RD)",
    "iv": "Run instrumental variable regression (2SLS)",
    "cluster_se": "Run regression with clustered standard errors at specified level",
    "visualization": "Create publication-quality visualization",
}


def _generate_via_openai(system: str, prompt: str, temperature: float = 0.1) -> tuple:
    """Try generating code via OpenAI GPT-4.1 (best code model available).

    Returns (response_text, model_name) or (None, None) if unavailable.
    """
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return None, None
    code_model = os.environ.get("EDITH_CODE_MODEL", "gpt-4.1")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        resp = client.chat.completions.create(
            model=code_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=4000,
        )
        text = resp.choices[0].message.content.strip()
        log.info(f"§VIBE: OpenAI code gen OK ({code_model}, {len(text)} chars)")
        return text, code_model
    except Exception as e:
        log.warning(f"§VIBE: OpenAI code gen failed ({code_model}): {e}")
        return None, None


def generate_code(
    directive: str,
    language: str = "python",
    dataset_path: str = "",
    analysis_type: str = "",
    variables: list[str] = None,
) -> dict:
    """Generate data analysis code from a natural language directive.

    Args:
        directive: what the user wants to analyze
        language: python, r, or stata
        dataset_path: path to data file
        analysis_type: optional template key from _ANALYSIS_TEMPLATES
        variables: optional list of variable names to focus on

    Returns:
        dict with generated_code, language, explanation
    """
    lang = language.lower()
    if lang not in _CODEGEN_SYSTEMS:
        return {"error": f"Unsupported language: {lang}. Use python, r, or stata."}

    system = _CODEGEN_SYSTEMS[lang]

    # Build the prompt
    parts = [f"DIRECTIVE: {directive}"]

    if dataset_path:
        ext = os.path.splitext(dataset_path)[1].lower()
        parts.append(f"DATASET: {dataset_path} (format: {_DATA_EXTENSIONS.get(ext, 'unknown')})")

    if analysis_type and analysis_type in _ANALYSIS_TEMPLATES:
        parts.append(f"ANALYSIS TYPE: {_ANALYSIS_TEMPLATES[analysis_type]}")

    if variables:
        parts.append(f"KEY VARIABLES: {', '.join(variables)}")

    prompt = "\n".join(parts)

    try:
        # §DUAL-BRAIN: Try GPT-4.1 first (superior code gen), Gemini fallback
        response, used_model = _generate_via_openai(system, prompt, temperature=0.1)

        if not response:
            # Fallback to Gemini
            from server.backend_logic import generate_text_via_chain
            model_chain = [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]
            combined = f"{system}\n\n{prompt}"
            response, used_model = generate_text_via_chain(
                combined, model_chain,
                temperature=0.1,
            )

        # Extract code from response
        code = _extract_code_block(response, lang)

        return {
            "code": code or response,
            "language": lang,
            "model": used_model,
            "directive": directive,
            "dataset": dataset_path,
            "analysis_type": analysis_type,
        }
    except Exception as e:
        return {"error": str(e), "language": lang}


def _extract_code_block(text: str, language: str) -> str:
    """Extract code from markdown code blocks."""
    # Try language-specific block first
    patterns = [
        rf"```{language}\n(.*?)```",
        rf"```{language}\s*\n(.*?)```",
        r"```\n(.*?)```",
        r"```(.*?)```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return text.strip()


# ═══════════════════════════════════════════════════════════════════
# C. Sandboxed Execution — run code safely
# ═══════════════════════════════════════════════════════════════════

# Safety: banned patterns that should never be in generated code
_BANNED_PATTERNS = [
    r"os\.system\s*\(",
    r"subprocess\.(run|Popen|call)\s*\(",
    r"shutil\.(rmtree|move)\s*\(",
    r"__import__\s*\(",
    r"eval\s*\(",
    r"exec\s*\(",
    r"open\s*\([^)]*['\"]w['\"]",  # write mode
    r"rm\s+-rf",
    r"sys\.exit",
    r"import\s+socket",
    r"import\s+http",
    r"import\s+requests",
    r"import\s+urllib",
]


def _safety_check(code: str) -> list[str]:
    """Check code for dangerous patterns. Returns list of violations."""
    violations = []
    for pattern in _BANNED_PATTERNS:
        if re.search(pattern, code):
            violations.append(f"Banned pattern detected: {pattern}")
    return violations


class SandboxResult:
    """Result of a sandboxed code execution."""

    def __init__(self):
        self.stdout = ""
        self.stderr = ""
        self.return_code = -1
        self.elapsed = 0.0
        self.output_files: list[str] = []
        self.error: Optional[str] = None
        self.truncated = False


def execute_python(
    code: str,
    timeout: int = 60,
    working_dir: str = "",
) -> SandboxResult:
    """Execute Python code in a sandboxed subprocess.

    Safety features:
    - Runs in subprocess (isolated from main process)
    - Timeout enforcement
    - Banned pattern detection
    - Temp directory for output files
    - No network access patterns allowed

    Args:
        code: Python code to execute
        timeout: max seconds
        working_dir: directory to run in (defaults to temp)

    Returns:
        SandboxResult with stdout, stderr, output files
    """
    result = SandboxResult()

    # Safety check
    violations = _safety_check(code)
    if violations:
        result.error = f"SAFETY VIOLATION: {'; '.join(violations)}"
        result.return_code = -1
        return result

    # Create temp directory for execution
    with tempfile.TemporaryDirectory(prefix="edith_vibe_") as tmpdir:
        work_dir = working_dir or tmpdir
        script_path = os.path.join(tmpdir, "analysis_script.py")

        # Write script
        with open(script_path, "w") as f:
            f.write(code)

        # Execute in subprocess
        t0 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=work_dir,
                env={
                    **os.environ,
                    "MPLBACKEND": "Agg",  # No GUI for matplotlib
                },
            )
            result.stdout = proc.stdout[:10000]  # Cap output
            result.stderr = proc.stderr[:5000]
            result.return_code = proc.returncode
            if len(proc.stdout) > 10000:
                result.truncated = True

        except subprocess.TimeoutExpired:
            result.error = f"Execution timed out after {timeout}s"
            result.return_code = -1
        except Exception as e:
            result.error = str(e)
            result.return_code = -1

        result.elapsed = round(time.time() - t0, 2)

        # Collect output files (plots, logs)
        for fname in os.listdir(work_dir):
            if fname.endswith((".png", ".jpg", ".pdf", ".csv", ".log", ".txt")):
                result.output_files.append(os.path.join(work_dir, fname))

    return result


def execute_r(code: str, timeout: int = 120) -> SandboxResult:
    """Execute R code in a sandboxed subprocess."""
    result = SandboxResult()

    # Find R
    r_path = None
    for candidate in ["/usr/local/bin/Rscript", "/usr/bin/Rscript",
                       "/opt/homebrew/bin/Rscript"]:
        if os.path.exists(candidate):
            r_path = candidate
            break

    if not r_path:
        result.error = "R/Rscript not found on this system"
        return result

    with tempfile.TemporaryDirectory(prefix="edith_vibe_r_") as tmpdir:
        script_path = os.path.join(tmpdir, "analysis.R")
        with open(script_path, "w") as f:
            f.write(code)

        t0 = time.time()
        try:
            proc = subprocess.run(
                [r_path, script_path],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmpdir,
            )
            result.stdout = proc.stdout[:10000]
            result.stderr = proc.stderr[:5000]
            result.return_code = proc.returncode
        except subprocess.TimeoutExpired:
            result.error = f"R execution timed out after {timeout}s"
        except Exception as e:
            result.error = str(e)

        result.elapsed = round(time.time() - t0, 2)

    return result


def execute_stata(code: str, timeout: int = 120) -> SandboxResult:
    """Execute Stata code via the StataBridge.

    Uses the live Stata subprocess bridge to run .do file code.
    Requires STATA_PATH env var or Stata installed in standard location.
    """
    result = SandboxResult()
    try:
        from server.stata_bridge import StataBridge
        bridge = StataBridge()
        status = bridge.status()
        if not status.get("available"):
            result.error = status.get("reason", "Stata not available — set STATA_PATH")
            return result

        t0 = time.time()
        stata_result = bridge.execute(code, timeout=timeout)
        result.elapsed = round(time.time() - t0, 2)
        result.stdout = stata_result.get("output", "")
        result.stderr = stata_result.get("error", "")
        result.return_code = stata_result.get("return_code", 0)
        if stata_result.get("error") and not stata_result.get("output"):
            result.error = stata_result["error"]
    except Exception as e:
        result.error = str(e)

    return result


def generate_and_run(
    directive: str,
    language: str = "python",
    dataset_path: str = "",
    analysis_type: str = "",
    variables: list[str] = None,
    auto_execute: bool = False,
) -> dict:
    """Generate code AND optionally execute it.

    Returns dict with generated code, and if executed, the results.
    """
    # Step 1: Generate
    gen_result = generate_code(
        directive, language, dataset_path, analysis_type, variables,
    )

    if "error" in gen_result:
        return gen_result

    result = {
        "generated": gen_result,
        "executed": False,
    }

    # Step 2: Optionally execute
    if auto_execute and language == "python":
        exec_result = execute_python(gen_result["code"])
        result["execution"] = {
            "stdout": exec_result.stdout,
            "stderr": exec_result.stderr,
            "return_code": exec_result.return_code,
            "elapsed": exec_result.elapsed,
            "error": exec_result.error,
            "output_files": exec_result.output_files,
            "truncated": exec_result.truncated,
        }
        result["executed"] = True
    elif auto_execute and language == "r":
        exec_result = execute_r(gen_result["code"])
        result["execution"] = {
            "stdout": exec_result.stdout,
            "stderr": exec_result.stderr,
            "return_code": exec_result.return_code,
            "elapsed": exec_result.elapsed,
            "error": exec_result.error,
        }
        result["executed"] = True
    elif auto_execute and language == "stata":
        exec_result = execute_stata(gen_result["code"])
        result["execution"] = {
            "stdout": exec_result.stdout,
            "stderr": exec_result.stderr,
            "return_code": exec_result.return_code,
            "elapsed": exec_result.elapsed,
            "error": exec_result.error,
        }
        result["executed"] = True

    return result


# ═══════════════════════════════════════════════════════════════════
# D. Code Explanation — explain existing code
# ═══════════════════════════════════════════════════════════════════

def explain_code(
    code: str,
    language: str = "python",
    difficulty: str = "intermediate",
) -> dict:
    """Explain existing data analysis code line-by-line.

    Difficulty levels:
    - intro: assumes no coding knowledge, explains every concept
    - intermediate: assumes basic familiarity, focuses on logic
    - advanced: focuses on statistical methodology choices
    - doctoral: critiques methodology and suggests improvements
    """
    difficulty_prompts = {
        "intro": "Explain this code as if the reader has NEVER written code before. Define every term (variable, function, library). Use analogies.",
        "intermediate": "Explain the logic and data transformations. Assume the reader knows basic coding but not the statistical methods.",
        "advanced": "Focus on the statistical methodology: why these methods, what assumptions they make, potential issues.",
        "doctoral": "Critique the methodology: is this the correct specification? What about endogeneity, selection bias, robustness checks?",
    }

    level_prompt = difficulty_prompts.get(difficulty, difficulty_prompts["intermediate"])

    try:
        explain_system = (
            "You are E.D.I.T.H.'s code explanation engine. "
            "Explain data analysis code clearly at the requested level."
        )
        explain_prompt = (
            f"Explain this {language.upper()} data analysis code.\n\n"
            f"LEVEL: {level_prompt}\n\n"
            f"CODE:\n```{language}\n{code[:3000]}\n```\n\n"
            f"Provide:\n"
            f"1. What this code does (one paragraph)\n"
            f"2. Step-by-step walkthrough\n"
            f"3. Key statistical/methodological choices\n"
            f"4. Potential issues or improvements"
        )

        # §DUAL-BRAIN: Try GPT-4.1 first, Gemini fallback
        explanation, model = _generate_via_openai(explain_system, explain_prompt, temperature=0.2)

        if not explanation:
            from server.backend_logic import generate_text_via_chain
            model_chain = [os.environ.get("EDITH_MODEL", "gemini-2.5-flash")]
            explanation, model = generate_text_via_chain(
                f"{explain_system}\n\n{explain_prompt}", model_chain, temperature=0.2,
            )
        return {
            "explanation": explanation,
            "language": language,
            "difficulty": difficulty,
            "model": model,
        }
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════
# E. Analysis Templates — pre-built analysis pipelines
# ═══════════════════════════════════════════════════════════════════

QUICK_ANALYSES = {
    "describe": {
        "name": "Descriptive Statistics",
        "code_python": '''import pandas as pd
import sys

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data.csv")
print("=== SHAPE ===")
print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
print(f"\\nColumn types:\\n{df.dtypes}")
print(f"\\n=== DESCRIPTIVE STATISTICS ===")
print(df.describe(include='all').round(3).to_string())
print(f"\\n=== MISSING VALUES ===")
missing = df.isnull().sum()
print(missing[missing > 0].to_string() if missing.any() else "No missing values")
''',
    },
    "correlate": {
        "name": "Correlation Matrix",
        "code_python": '''import pandas as pd
import numpy as np
import sys

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data.csv")
numeric = df.select_dtypes(include=[np.number])

print("=== CORRELATION MATRIX ===")
corr = numeric.corr().round(3)
print(corr.to_string())

# Flag strong correlations
print("\\n=== STRONG CORRELATIONS (|r| > 0.5) ===")
for i in range(len(corr.columns)):
    for j in range(i+1, len(corr.columns)):
        r = corr.iloc[i, j]
        if abs(r) > 0.5:
            print(f"  {corr.columns[i]} ↔ {corr.columns[j]}: r={r:.3f}")
''',
    },
    "regression": {
        "name": "OLS Regression",
        "code_python": '''import pandas as pd
import statsmodels.api as sm
import sys

df = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "data.csv")
numeric = df.select_dtypes(include=["number"]).dropna()

if len(numeric.columns) < 2:
    print("ERROR: Need at least 2 numeric columns for regression")
    sys.exit(1)

y_col = numeric.columns[0]
x_cols = numeric.columns[1:]

print(f"=== OLS REGRESSION ===")
print(f"DV: {y_col}")
print(f"IVs: {', '.join(x_cols)}")

X = sm.add_constant(numeric[x_cols])
y = numeric[y_col]
model = sm.OLS(y, X).fit(cov_type='HC3')  # Robust SE

print(f"\\n{model.summary()}")
print(f"\\n=== KEY FINDINGS ===")
print(f"R²: {model.rsquared:.4f}")
print(f"Adj R²: {model.rsquared_adj:.4f}")
print(f"F-stat: {model.fvalue:.4f} (p={model.f_pvalue:.4f})")
for var in x_cols:
    coef = model.params[var]
    pval = model.pvalues[var]
    sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
    print(f"  {var}: β={coef:.4f}, p={pval:.4f} {sig}")
''',
    },
}


def run_quick_analysis(
    analysis_key: str,
    dataset_path: str,
    language: str = "python",
) -> dict:
    """Run a pre-built analysis template on a dataset."""
    template = QUICK_ANALYSES.get(analysis_key)
    if not template:
        return {"error": f"Unknown analysis: {analysis_key}. Options: {list(QUICK_ANALYSES.keys())}"}

    code_key = f"code_{language}"
    code = template.get(code_key)
    if not code:
        return {"error": f"No {language} template for {analysis_key}"}

    # Inject the dataset path
    code = code.replace('"data.csv"', f'"{dataset_path}"')
    code = code.replace("sys.argv[1] if len(sys.argv) > 1 else ", "")

    result = execute_python(code, timeout=30)
    return {
        "analysis": template["name"],
        "code": code,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "return_code": result.return_code,
        "elapsed": result.elapsed,
        "error": result.error,
    }
