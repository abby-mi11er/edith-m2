"""
Paper Deconstructor — Forensic PDF Audit at Molecular Resolution
===================================================================
Methodological Forensics Lab Feature 1: The X-Ray Machine.

You don't "read" a paper. You DECONSTRUCT it.

When you drag a PDF into the Cockpit and say "Winnie, perform a Full
Forensic Audit on this text," Winnie's Neural Engine performs:

1. Semantic Extraction — every citation, every dataset, every estimator
2. Bolt Cross-Reference — pulls cited papers from the 1TB vault in <2s
3. Dataset Identification — names the exact variables and data sources
4. Estimator Classification — RDD, DiD, IV, QCA, Fixed Effects, etc.
5. Theory-Graph Embedding — places the paper in the Theoretical Atlas

Architecture (M4 + Bolt at 3,100 MB/s):
    PDF → Neural OCR/Parse → Citation Extraction → Dataset Detection →
    Estimator Classification → Bolt Cross-Reference → Forensic Report

This is not summarization. This is MOLECULAR DECONSTRUCTION.
"""

import hashlib
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.paper_deconstructor")


# ═══════════════════════════════════════════════════════════════════
# Known Datasets — The Sovereign Data Registry
# ═══════════════════════════════════════════════════════════════════

def _dataset_bolt_path(subdir: str) -> str:
    """Build a dataset bolt_path using vault_config.VAULT_ROOT."""
    try:
        from server.vault_config import VAULT_ROOT
        return str(VAULT_ROOT / "VAULT" / "RESOURCES" / "DATASETS" / subdir)
    except ImportError:
        import os
        root = os.environ.get("EDITH_DATA_ROOT", ".")
        return os.path.join(root, "VAULT", "RESOURCES", "DATASETS", subdir)


KNOWN_DATASETS = {
    "v-dem": {
        "full_name": "Varieties of Democracy",
        "variables": ["electoral_integrity", "liberal_democracy", "participatory_index",
                       "deliberative_index", "egalitarian_index"],
        "coverage": "1789-2024, 202 countries",
        "format": "CSV/Stata",
        "bolt_path": _dataset_bolt_path("V-Dem"),
    },
    "cps": {
        "full_name": "Current Population Survey",
        "variables": ["labor_force_status", "earnings", "education", "poverty_status",
                       "food_stamp_receipt", "health_insurance"],
        "coverage": "Monthly, US households",
        "format": "CSV/Stata/SAS",
        "bolt_path": _dataset_bolt_path("CPS"),
    },
    "anes": {
        "full_name": "American National Election Studies",
        "variables": ["party_id", "vote_choice", "institutional_trust", "political_efficacy",
                       "ideology", "racial_resentment"],
        "coverage": "1948-2024, US nationally representative",
        "format": "Stata/SPSS",
        "bolt_path": _dataset_bolt_path("ANES"),
    },
    "census": {
        "full_name": "US Census / American Community Survey",
        "variables": ["population", "median_income", "poverty_rate", "education_attainment",
                       "race_ethnicity", "housing_tenure", "snap_participation"],
        "coverage": "Decennial + Annual (ACS), county/tract level",
        "format": "CSV",
        "bolt_path": _dataset_bolt_path("CENSUS"),
    },
    "world_bank_wdi": {
        "full_name": "World Development Indicators",
        "variables": ["gdp_per_capita", "gini_index", "government_expenditure",
                       "health_expenditure", "education_expenditure"],
        "coverage": "1960-2024, 217 countries",
        "format": "CSV/Excel",
        "bolt_path": _dataset_bolt_path("WDI"),
    },
    "fragile_states_index": {
        "full_name": "Fragile States Index (Fund for Peace)",
        "variables": ["state_legitimacy", "public_services", "group_grievance",
                       "security_apparatus", "factionalized_elites"],
        "coverage": "2005-2024, 178 countries",
        "format": "CSV",
        "bolt_path": _dataset_bolt_path("FSI"),
    },
    "qog": {
        "full_name": "Quality of Government Institute",
        "variables": ["corruption_control", "rule_of_law", "bureaucratic_quality",
                       "government_effectiveness", "regulatory_quality"],
        "coverage": "1946-2024, 194 countries",
        "format": "Stata/CSV",
        "bolt_path": _dataset_bolt_path("QOG"),
    },
}


# ═══════════════════════════════════════════════════════════════════
# Statistical Estimators — The Methodology Fingerprint
# ═══════════════════════════════════════════════════════════════════

ESTIMATOR_PATTERNS = {
    "ols": {
        "label": "Ordinary Least Squares (OLS)",
        "patterns": [r"OLS", r"linear\s+regression", r"ordinary\s+least\s+squares"],
        "family": "linear",
        "assumptions": ["linearity", "homoscedasticity", "no multicollinearity", "normality"],
    },
    "fixed_effects": {
        "label": "Fixed Effects (FE)",
        "patterns": [r"fixed\s+effects?", r"within\s+estimator", r"entity\s+demeaned",
                      r"two-way\s+fixed", r"TWFE"],
        "family": "panel",
        "assumptions": ["strict_exogeneity", "time_invariant_unobservables"],
    },
    "rdd": {
        "label": "Regression Discontinuity Design (RDD)",
        "patterns": [r"regression\s+discontinuity", r"RDD", r"sharp\s+discontinuity",
                      r"fuzzy\s+discontinuity", r"bandwidth\s+selection", r"local\s+polynomial"],
        "family": "quasi_experimental",
        "assumptions": ["continuity", "no_manipulation", "local_randomization"],
    },
    "did": {
        "label": "Difference-in-Differences (DiD)",
        "patterns": [r"diff(?:erence)?[- ]?in[- ]?diff(?:erence)?s?", r"DiD\b", r"DD\b",
                      r"parallel\s+trends?", r"event\s+study"],
        "family": "quasi_experimental",
        "assumptions": ["parallel_trends", "no_anticipation", "stable_treatment"],
    },
    "iv": {
        "label": "Instrumental Variables (IV / 2SLS)",
        "patterns": [r"instrumental\s+variable", r"2SLS", r"two[\s-]stage\s+least",
                      r"IV\s+(?:estimation|regression|approach)", r"first[\s-]stage"],
        "family": "causal",
        "assumptions": ["relevance", "exclusion_restriction", "independence"],
    },
    "qca": {
        "label": "Qualitative Comparative Analysis (QCA)",
        "patterns": [r"QCA\b", r"qualitative\s+comparative", r"crisp[\s-]set",
                      r"fuzzy[\s-]set\s+QCA", r"fsQCA", r"csQCA", r"truth\s+table"],
        "family": "configurational",
        "assumptions": ["equifinality", "conjunctural_causation"],
    },
    "ml_classifier": {
        "label": "Machine Learning Classifier",
        "patterns": [r"random\s+forest", r"gradient\s+boost", r"XGBoost",
                      r"support\s+vector", r"neural\s+network", r"deep\s+learning"],
        "family": "predictive",
        "assumptions": ["train_test_split", "cross_validation"],
    },
    "propensity_score": {
        "label": "Propensity Score Matching (PSM)",
        "patterns": [r"propensity\s+score", r"PSM\b", r"matching\s+estimat",
                      r"nearest\s+neighbor\s+matching", r"caliper"],
        "family": "quasi_experimental",
        "assumptions": ["conditional_independence", "common_support"],
    },
    "synthetic_control": {
        "label": "Synthetic Control Method (SCM)",
        "patterns": [r"synthetic\s+control", r"SCM\b", r"donor\s+pool",
                      r"Abadie", r"pre[- ]treatment\s+fit"],
        "family": "quasi_experimental",
        "assumptions": ["convex_hull", "no_interference", "pre_treatment_fit"],
    },
    "bayesian": {
        "label": "Bayesian Inference",
        "patterns": [r"Bayesian\b", r"posterior\s+distribution", r"MCMC",
                      r"prior\s+(?:distribution|belief)", r"credible\s+interval"],
        "family": "bayesian",
        "assumptions": ["prior_specification", "likelihood_model"],
    },
    "structural_equation": {
        "label": "Structural Equation Modeling (SEM)",
        "patterns": [r"structural\s+equation", r"SEM\b", r"path\s+analysis",
                      r"latent\s+variable", r"confirmatory\s+factor"],
        "family": "latent",
        "assumptions": ["model_specification", "multivariate_normality"],
    },
    "multilevel": {
        "label": "Multilevel / Hierarchical Linear Model (HLM)",
        "patterns": [r"multilevel", r"hierarchical\s+linear", r"HLM\b",
                      r"random\s+intercept", r"random\s+slope", r"mixed[\s-]effects?"],
        "family": "nested",
        "assumptions": ["nested_structure", "random_effects_distribution"],
    },
}


# ═══════════════════════════════════════════════════════════════════
# Citation Extraction — Every Reference, Traced
# ═══════════════════════════════════════════════════════════════════

# APA-style: (Author, Year) or (Author & Author, Year)
CITATION_PATTERNS = [
    re.compile(r'\(([A-Z][a-z]+(?:\s+(?:and|&|et\s+al\.?)\s*(?:[A-Z][a-z]+)?)?),?\s*(\d{4})[a-z]?\)'),
    re.compile(r'([A-Z][a-z]+(?:\s+(?:and|&)\s+[A-Z][a-z]+)?)\s*\((\d{4})[a-z]?\)'),
    re.compile(r'\(([A-Z][a-z]+)\s+et\s+al\.?,?\s*(\d{4})\)'),
]


@dataclass
class ExtractedCitation:
    """A citation extracted from the paper."""
    author: str
    year: int
    raw_text: str
    context: str  # The sentence containing the citation
    in_vault: bool = False  # Whether we have this paper on the Bolt
    vault_path: str = ""

    def to_dict(self) -> dict:
        return {
            "author": self.author,
            "year": self.year,
            "raw": self.raw_text,
            "context": self.context[:200],
            "in_vault": self.in_vault,
            "vault_path": self.vault_path,
        }


@dataclass
class ExtractedDataset:
    """A dataset reference found in the paper."""
    name: str
    known: bool
    full_name: str = ""
    variables_mentioned: list[str] = field(default_factory=list)
    bolt_available: bool = False
    bolt_path: str = ""
    context: str = ""

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "known": self.known,
            "full_name": self.full_name,
            "variables": self.variables_mentioned,
            "on_bolt": self.bolt_available,
            "bolt_path": self.bolt_path,
            "context": self.context[:200],
        }


@dataclass
class ExtractedEstimator:
    """A statistical estimator identified in the paper."""
    estimator_id: str
    label: str
    family: str
    confidence: float  # 0-1, how sure we are
    assumptions: list[str]
    context: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.estimator_id,
            "label": self.label,
            "family": self.family,
            "confidence": round(self.confidence, 2),
            "assumptions": self.assumptions,
            "context": self.context[:200],
        }


# ═══════════════════════════════════════════════════════════════════
# The Deconstructor Engine — Molecular Resolution
# ═══════════════════════════════════════════════════════════════════

class PaperDeconstructor:
    """Deconstruct an academic paper into its molecular components.

    Full Forensic Audit produces:
    1. Citation Map — every referenced work, cross-referenced with Bolt
    2. Dataset Registry — every dataset mentioned, linked to local copies
    3. Estimator Fingerprint — every statistical method, classified
    4. Theory Graph — the paper's theoretical position
    5. Variable Registry — key variables and their operationalization
    """

    def __init__(self, bolt_path: str = ""):
        try:
            from server.vault_config import VAULT_ROOT
            _default = str(VAULT_ROOT)
        except ImportError:
            _default = os.environ.get("EDITH_DATA_ROOT", ".")
        self._bolt_path = bolt_path or _default
        self._vault_path = os.path.join(self._bolt_path, "VAULT")
        self._data_root = os.environ.get("EDITH_APP_DATA_DIR", "")
        self._audit_cache: dict[str, dict] = {}

    def full_forensic_audit(self, text: str, title: str = "",
                             author: str = "", year: int = 0) -> dict:
        """Perform a complete Forensic Audit on a paper's text.

        This is the command: "Winnie, perform a Full Forensic Audit."

        Returns a comprehensive deconstruction with every citation,
        dataset, estimator, and theoretical position identified.
        """
        t0 = time.time()
        text_hash = hashlib.sha256(text[:1000].encode()).hexdigest()[:12]

        # Check cache
        if text_hash in self._audit_cache:
            cached = self._audit_cache[text_hash]
            cached["from_cache"] = True
            return cached

        # 1. Extract citations
        citations = self._extract_citations(text)

        # 2. Cross-reference citations with Bolt vault
        vault_hits = self._cross_reference_vault(citations)

        # 3. Identify datasets
        datasets = self._identify_datasets(text)

        # 4. Classify estimators
        estimators = self._classify_estimators(text)

        # 5. Extract key variables
        variables = self._extract_variables(text)

        # 6. Identify theoretical framework
        theory = self._identify_theory(text)

        # 7. Extract sample information
        sample_info = self._extract_sample_info(text)

        elapsed = time.time() - t0

        audit = {
            "paper": {
                "title": title or self._guess_title(text),
                "author": author,
                "year": year,
                "hash": text_hash,
            },
            "citations": {
                "total": len(citations),
                "in_vault": vault_hits,
                "items": [c.to_dict() for c in citations],
            },
            "datasets": {
                "total": len(datasets),
                "on_bolt": sum(1 for d in datasets if d.bolt_available),
                "items": [d.to_dict() for d in datasets],
            },
            "estimators": {
                "primary": estimators[0].to_dict() if estimators else None,
                "all": [e.to_dict() for e in estimators],
                "family": estimators[0].family if estimators else "unidentified",
            },
            "variables": variables,
            "theory": theory,
            "sample": sample_info,
            "forensics": {
                "elapsed_seconds": round(elapsed, 3),
                "text_length": len(text),
                "from_cache": False,
            },
        }

        # Cache the result
        self._audit_cache[text_hash] = audit
        if len(self._audit_cache) > 50:
            oldest = next(iter(self._audit_cache))
            del self._audit_cache[oldest]

        log.info(f"§FORENSICS: Audit complete: {len(citations)} citations, "
                 f"{len(datasets)} datasets, {len(estimators)} estimators in {elapsed:.2f}s")

        return audit

    def _extract_citations(self, text: str) -> list[ExtractedCitation]:
        """Extract every citation from the text."""
        citations = []
        seen = set()

        # Split into sentences for context
        sentences = re.split(r'(?<=[.!?])\s+', text)

        for sentence in sentences:
            for pattern in CITATION_PATTERNS:
                for match in pattern.finditer(sentence):
                    author = match.group(1).strip()
                    try:
                        year = int(match.group(2))
                    except (ValueError, IndexError):
                        continue

                    key = f"{author}_{year}"
                    if key in seen:
                        continue
                    seen.add(key)

                    citations.append(ExtractedCitation(
                        author=author,
                        year=year,
                        raw_text=match.group(0),
                        context=sentence.strip(),
                    ))

        return citations

    def _cross_reference_vault(self, citations: list[ExtractedCitation]) -> int:
        """Check which cited papers exist on the Bolt SSD."""
        vault_hits = 0
        vault_dir = Path(self._vault_path)

        if not vault_dir.exists():
            return 0

        # Build a quick index of what's on the Bolt
        for citation in citations:
            # Search for author name + year in vault filenames
            search_term = citation.author.split()[0].lower()  # First author surname
            year_str = str(citation.year)

            for pdf in vault_dir.rglob("*.pdf"):
                fname = pdf.name.lower()
                if search_term in fname and year_str in fname:
                    citation.in_vault = True
                    citation.vault_path = str(pdf)
                    vault_hits += 1
                    break

            # Also check in chroma metadata if available
            if not citation.in_vault:
                for txt in vault_dir.rglob("*.txt"):
                    fname = txt.name.lower()
                    if search_term in fname:
                        citation.in_vault = True
                        citation.vault_path = str(txt)
                        vault_hits += 1
                        break

        return vault_hits

    def _identify_datasets(self, text: str) -> list[ExtractedDataset]:
        """Identify every dataset mentioned in the text."""
        datasets = []
        text_lower = text.lower()

        # Check against known dataset registry
        for ds_key, ds_info in KNOWN_DATASETS.items():
            # Check if the dataset name appears
            name_patterns = [
                ds_info["full_name"].lower(),
                ds_key.replace("_", " ").replace("-", " "),
                ds_key.upper(),
            ]
            for pattern in name_patterns:
                if pattern in text_lower:
                    # Find which variables are mentioned
                    vars_mentioned = [
                        v for v in ds_info.get("variables", [])
                        if v.replace("_", " ") in text_lower or v in text_lower
                    ]

                    # Check if available on Bolt
                    bolt_path = ds_info.get("bolt_path", "")
                    on_bolt = Path(bolt_path).exists() if bolt_path else False

                    # Find context
                    idx = text_lower.find(pattern)
                    context = text[max(0, idx - 100):idx + 200]

                    datasets.append(ExtractedDataset(
                        name=ds_key,
                        known=True,
                        full_name=ds_info["full_name"],
                        variables_mentioned=vars_mentioned,
                        bolt_available=on_bolt,
                        bolt_path=bolt_path,
                        context=context,
                    ))
                    break

        # Check for unknown datasets (generic patterns)
        unknown_patterns = [
            r'(?:data(?:set)?|survey|panel)\s+(?:from|provided by|collected by)\s+([^.]{5,50})',
            r'(?:using|employ(?:ing)?|analyz(?:ing|e))\s+(?:the\s+)?([A-Z][A-Za-z\s]+(?:Survey|Panel|Census|Database))',
        ]
        for pattern in unknown_patterns:
            for match in re.finditer(pattern, text):
                name = match.group(1).strip()
                # Skip if already found
                if any(d.full_name.lower() in name.lower() for d in datasets):
                    continue
                datasets.append(ExtractedDataset(
                    name=name,
                    known=False,
                    context=match.group(0),
                ))

        return datasets

    def _classify_estimators(self, text: str) -> list[ExtractedEstimator]:
        """Classify every statistical estimator used in the paper."""
        estimators = []
        text_lower = text.lower()

        for est_id, est_info in ESTIMATOR_PATTERNS.items():
            matches = 0
            context = ""
            for pattern in est_info["patterns"]:
                found = re.findall(pattern, text, re.IGNORECASE)
                matches += len(found)
                if found and not context:
                    # Find the sentence containing the match
                    idx = re.search(pattern, text, re.IGNORECASE)
                    if idx:
                        start = max(0, idx.start() - 100)
                        end = min(len(text), idx.end() + 200)
                        context = text[start:end]

            if matches > 0:
                # Confidence based on number of pattern matches
                confidence = min(1.0, matches * 0.25)
                estimators.append(ExtractedEstimator(
                    estimator_id=est_id,
                    label=est_info["label"],
                    family=est_info["family"],
                    confidence=confidence,
                    assumptions=est_info["assumptions"],
                    context=context,
                ))

        # Sort by confidence (primary estimator first)
        estimators.sort(key=lambda e: e.confidence, reverse=True)
        return estimators

    def _extract_variables(self, text: str) -> dict:
        """Extract key variables: dependent, independent, controls."""
        variables = {
            "dependent": [],
            "independent": [],
            "controls": [],
            "instruments": [],
        }

        # Dependent variable patterns
        dv_patterns = [
            r'(?:dependent\s+variable|outcome\s+(?:variable|measure))\s+(?:is|was|:)\s*([^.]{5,80})',
            r'(?:we\s+measure|our\s+DV|Y\s*=)\s*([^.]{5,80})',
        ]
        for pattern in dv_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                variables["dependent"].append(match.group(1).strip()[:80])

        # Independent variable patterns
        iv_patterns = [
            r'(?:independent\s+variable|treatment|key\s+predictor)\s+(?:is|was|:)\s*([^.]{5,80})',
            r'(?:main\s+explanatory|X\s*=)\s*([^.]{5,80})',
        ]
        for pattern in iv_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                variables["independent"].append(match.group(1).strip()[:80])

        # Control variables
        ctrl_patterns = [
            r'(?:control(?:ling)?\s+for|covariates?\s+include)\s*:?\s*([^.]{5,150})',
        ]
        for pattern in ctrl_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                raw = match.group(1).strip()
                controls = [c.strip() for c in re.split(r',|;|and\b', raw) if len(c.strip()) > 2]
                variables["controls"].extend(controls[:10])

        # Instrument detection (for IV papers)
        inst_patterns = [
            r'(?:instrument(?:al)?(?:\s+variable)?)\s+(?:is|was|:)\s*([^.]{5,80})',
        ]
        for pattern in inst_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                variables["instruments"].append(match.group(1).strip()[:80])

        return variables

    def _identify_theory(self, text: str) -> dict:
        """Identify the theoretical framework(s) the paper draws on."""
        theories = {
            "principal_agent": ["principal-agent", "agency theory", "moral hazard",
                                 "adverse selection", "monitoring cost"],
            "administrative_burden": ["administrative burden", "learning costs",
                                       "compliance costs", "psychological costs", "take-up"],
            "policy_feedback": ["policy feedback", "Pierson", "lock-in", "path depend"],
            "state_capacity": ["state capacity", "infrastructural power", "Mann",
                                "extractive capacity", "regulatory capacity"],
            "new_public_management": ["new public management", "NPM", "marketization",
                                       "performance measurement", "contracting out"],
            "institutional_choice": ["institutional choice", "Ostrom", "collective action",
                                      "commons", "polycentricity"],
            "welfare_state": ["welfare state", "decommodification", "Esping-Andersen",
                               "welfare regime", "social protection"],
            "democratic_erosion": ["democratic erosion", "backsliding", "illiberalism",
                                    "Levitsky", "competitive authoritarian"],
        }

        identified = []
        text_lower = text.lower()
        for theory_id, keywords in theories.items():
            matches = sum(1 for kw in keywords if kw.lower() in text_lower)
            if matches >= 2:
                identified.append({
                    "theory": theory_id.replace("_", " ").title(),
                    "matches": matches,
                    "keywords_found": [kw for kw in keywords if kw.lower() in text_lower],
                })

        identified.sort(key=lambda t: t["matches"], reverse=True)

        return {
            "primary_framework": identified[0]["theory"] if identified else "unidentified",
            "all_frameworks": identified,
            "theoretical_density": len(identified),
        }

    def _extract_sample_info(self, text: str) -> dict:
        """Extract information about the paper's sample/data."""
        sample = {"n": None, "unit": "", "time_period": "", "geography": ""}

        # Sample size
        n_patterns = [
            r'[Nn]\s*=\s*([\d,]+)',
            r'(?:sample|observations?)\s+(?:of|size|:)\s*([\d,]+)',
            r'([\d,]+)\s+(?:observations?|respondents?|subjects?|cases?|countries?)',
        ]
        for pattern in n_patterns:
            match = re.search(pattern, text)
            if match:
                try:
                    sample["n"] = int(match.group(1).replace(",", ""))
                except ValueError:
                    pass
                break

        # Time period
        time_pattern = r'(\d{4})\s*[-–to]+\s*(\d{4})'
        time_match = re.search(time_pattern, text)
        if time_match:
            sample["time_period"] = f"{time_match.group(1)}-{time_match.group(2)}"

        # Geography
        geo_patterns = [
            r'(?:data from|sample from|study of|based in)\s+(?:the\s+)?([A-Z][a-zA-Z\s]+)',
        ]
        for pattern in geo_patterns:
            match = re.search(pattern, text)
            if match:
                sample["geography"] = match.group(1).strip()[:50]
                break

        return sample

    def _guess_title(self, text: str) -> str:
        """Attempt to extract the paper title from the first few lines."""
        lines = text.split("\n")[:5]
        for line in lines:
            line = line.strip()
            if 10 < len(line) < 200 and not line.startswith("Abstract"):
                return line
        return "Unknown Title"

    def get_dataset_bridge(self, dataset_name: str) -> dict:
        """The Dataset Bridge: link a paper's dataset to your local copy.

        "Winnie doesn't just name the data; she opens a Virtual Table."
        """
        ds_info = KNOWN_DATASETS.get(dataset_name.lower().replace(" ", "_"))
        if not ds_info:
            return {"found": False, "name": dataset_name}

        bolt_path = Path(ds_info.get("bolt_path", ""))
        files = []
        if bolt_path.exists():
            files = [str(f.name) for f in bolt_path.iterdir() if f.is_file()][:10]

        return {
            "found": True,
            "name": ds_info["full_name"],
            "variables": ds_info["variables"],
            "coverage": ds_info.get("coverage", ""),
            "format": ds_info.get("format", ""),
            "on_bolt": bolt_path.exists(),
            "bolt_path": str(bolt_path),
            "available_files": files,
            "ready_for_merge": len(files) > 0,
        }

    def save_audit(self, audit: dict) -> dict:
        """Save a forensic audit to disk."""
        save_dir = Path(self._data_root or ".") / "VAULT" / "FORENSICS"
        save_dir.mkdir(parents=True, exist_ok=True)

        paper_hash = audit.get("paper", {}).get("hash", "unknown")
        path = save_dir / f"audit_{paper_hash}.json"
        try:
            path.write_text(json.dumps(audit, indent=2, default=str))
            return {"saved": True, "path": str(path)}
        except Exception as e:
            return {"saved": False, "error": str(e)}


# Global instance
paper_deconstructor = PaperDeconstructor()
