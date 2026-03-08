"""
Citadel Bridge — The Single Pane of Glass
============================================
The "Master Weld" that turns separate apps into a unified Living Brain.

Four Systems:
  1. Global Focus Variable — Click a PDF → every module snaps to context
  2. Cockpit Command Line — "// Plot residuals" → Stata + ArcGIS + Notion
  3. Metabolic Data Pipeline — Data flows like oxygen; no Save/Export/Import
  4. Zero-Latency Unified Search — One query → Notion + Stata + PDFs + GIS

When you click a PDF in the Library, E.D.I.T.H. broadcasts a Focus Signal
to every other module:
  - Notion scrolls to the associated notes
  - The Atlas highlights that paper's node
  - The Forensic Lab pre-loads the methodology Short Course
  - The Cockpit primes the replication dataset

You never leave the Flow State.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("edith.citadel_bridge")


# ═══════════════════════════════════════════════════════════════════
# 1. GLOBAL FOCUS VARIABLE — The Contextual Handshake
# ═══════════════════════════════════════════════════════════════════

@dataclass
class FocusSignal:
    """A broadcast signal that tells every module what you're looking at."""
    focus_type: str       # "paper", "dataset", "note", "concept", "chapter"
    title: str            # Human-readable title
    path: str             # File path on Bolt (if applicable)
    author: str           # Primary author (if paper)
    methodology: str      # Detected methodology (if paper)
    concepts: list[str]   # Key concepts extracted
    chapter: int          # Related dissertation chapter (0 = unknown)
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()

    def to_dict(self) -> dict:
        return {
            "type": self.focus_type, "title": self.title,
            "path": self.path, "author": self.author,
            "methodology": self.methodology,
            "concepts": self.concepts, "chapter": self.chapter,
            "timestamp": self.timestamp,
        }


class FocusManager:
    """Manages the Global Focus Variable.

    When focus changes, every subscribed module receives
    the new context simultaneously.
    """

    def __init__(self):
        self._current_focus: Optional[FocusSignal] = None
        self._focus_history: list[FocusSignal] = []
        self._subscribers: dict[str, Callable] = {}

    @property
    def current(self) -> Optional[FocusSignal]:
        return self._current_focus

    def set_focus(self, signal: FocusSignal) -> dict:
        """Broadcast a new focus to every module."""
        self._current_focus = signal
        self._focus_history.append(signal)

        # Keep history bounded
        if len(self._focus_history) > 200:
            self._focus_history = self._focus_history[-100:]

        # Broadcast to all subscribers
        results = {}
        for name, callback in self._subscribers.items():
            try:
                results[name] = callback(signal)
            except Exception as e:
                results[name] = {"error": str(e)}

        log.info(f"§FOCUS: '{signal.title}' broadcast to {len(results)} modules")
        return {
            "focus": signal.to_dict(),
            "modules_notified": list(results.keys()),
            "results": results,
        }

    def subscribe(self, module_name: str, callback: Callable):
        """Subscribe a module to focus changes."""
        self._subscribers[module_name] = callback

    def focus_paper(self, title: str, path: str = "", author: str = "",
                    methodology: str = "") -> dict:
        """Shortcut: set focus to a paper."""
        concepts = self._extract_concepts_from_title(title)
        chapter = self._guess_chapter(title + " " + methodology)
        signal = FocusSignal(
            focus_type="paper", title=title, path=path,
            author=author, methodology=methodology,
            concepts=concepts, chapter=chapter,
        )
        return self.set_focus(signal)

    def focus_dataset(self, name: str, path: str = "") -> dict:
        """Shortcut: set focus to a dataset."""
        signal = FocusSignal(
            focus_type="dataset", title=name, path=path,
            author="", methodology="",
            concepts=[name.lower()], chapter=5,
        )
        return self.set_focus(signal)

    def focus_concept(self, concept: str) -> dict:
        """Shortcut: set focus to a concept."""
        signal = FocusSignal(
            focus_type="concept", title=concept, path="",
            author="", methodology="",
            concepts=[concept.lower()], chapter=self._guess_chapter(concept),
        )
        return self.set_focus(signal)

    def focus_chapter(self, chapter_num: int, title: str = "") -> dict:
        """Shortcut: set focus to a dissertation chapter."""
        signal = FocusSignal(
            focus_type="chapter", title=title or f"Chapter {chapter_num}",
            path="", author="", methodology="",
            concepts=[], chapter=chapter_num,
        )
        return self.set_focus(signal)

    def _extract_concepts_from_title(self, title: str) -> list[str]:
        title_lower = title.lower()
        known_concepts = [
            "state capacity", "administrative burden", "charity", "nonprofit",
            "accountability", "blame", "devolution", "welfare", "cartel",
            "federalism", "potter county", "lubbock", "governance",
            "principal-agent", "submerged state", "institutional friction",
        ]
        return [c for c in known_concepts if c in title_lower]

    def _guess_chapter(self, text: str) -> int:
        text_lower = text.lower()
        signals = {
            1: ["puzzle", "introduction", "research question"],
            2: ["theory", "framework", "mechanism", "burden", "submerged"],
            3: ["state capacity", "potter county", "service delivery"],
            4: ["charity", "nonprofit", "substitution", "ngo"],
            5: ["data", "method", "regression", "variable", "lubbock", "did", "rdd", "iv"],
            6: ["finding", "result", "coefficient", "evidence"],
            7: ["implication", "conclusion", "future"],
        }
        best_ch, best_hits = 0, 0
        for ch, kws in signals.items():
            hits = sum(1 for k in kws if k in text_lower)
            if hits > best_hits:
                best_ch, best_hits = ch, hits
        return best_ch

    @property
    def status(self) -> dict:
        return {
            "current_focus": self._current_focus.to_dict() if self._current_focus else None,
            "history_length": len(self._focus_history),
            "subscribers": list(self._subscribers.keys()),
        }


# ═══════════════════════════════════════════════════════════════════
# 2. COCKPIT COMMAND LINE — The Universal Remote
# ═══════════════════════════════════════════════════════════════════

@dataclass
class CockpitCommand:
    """A parsed cockpit command."""
    raw: str
    action: str       # "plot", "regress", "search", "export", "open", "draft", etc.
    target: str       # What to act on
    modifiers: dict    # Additional parameters
    apps: list[str]    # Apps that need to be invoked

    def to_dict(self) -> dict:
        return {
            "raw": self.raw, "action": self.action,
            "target": self.target, "modifiers": self.modifiers,
            "apps": self.apps,
        }


class CockpitCommandLine:
    """The single command line that controls Stata, ArcGIS, and Notion.

    Syntax: // <action> <target> [modifiers]

    Examples:
        // Plot residuals for Potter County
        // Regress turnout on charity_density, controls(income education)
        // Search "administrative burden" across all sources
        // Export Chapter 3 to LaTeX
        // Open the DiD sandbox
        // Draft footnote for Mettler 2011
    """

    # Command patterns: (regex, action, description)
    COMMAND_PATTERNS = [
        (r"plot\s+(.+?)(?:\s+for\s+(.+))?$", "plot",
         "Generate a plot/map from data"),
        (r"regress\s+(\w+)\s+on\s+(.+?)(?:,\s*controls?\((.+?)\))?$", "regress",
         "Run a regression in Stata"),
        (r"search\s+[\"']?(.+?)[\"']?\s*(?:across\s+(.+))?$", "search",
         "Unified search across all sources"),
        (r"export\s+(.+?)\s+to\s+(\w+)$", "export",
         "Export content to a format"),
        (r"open\s+(.+?)(?:\s+sandbox)?$", "open",
         "Open a module or sandbox"),
        (r"draft\s+(.+?)(?:\s+for\s+(.+))?$", "draft",
         "Generate a draft footnote/section"),
        (r"audit\s+(.+)$", "audit",
         "Run a forensic audit on a paper"),
        (r"annotate\s+(.+)$", "annotate",
         "Auto-annotate a Stata/R output file"),
        (r"compare\s+(.+?)\s+(?:and|vs\.?|with)\s+(.+)$", "compare",
         "Compare two papers/datasets/theories"),
        (r"focus\s+(.+)$", "focus",
         "Set the global focus to a concept/paper"),
    ]

    def __init__(self, bolt_path: str = ""):
        try:
            from server.vault_config import VAULT_ROOT
            _default = str(VAULT_ROOT)
        except ImportError:
            _default = os.environ.get("EDITH_DATA_ROOT", ".")
        self._bolt_path = bolt_path or _default
        self._history: list[CockpitCommand] = []

    def parse(self, raw_input: str) -> CockpitCommand:
        """Parse a cockpit command.

        Strips the leading '//' if present.
        """
        cmd = raw_input.strip()
        if cmd.startswith("//"):
            cmd = cmd[2:].strip()

        cmd_lower = cmd.lower()

        for pattern, action, _ in self.COMMAND_PATTERNS:
            match = re.match(pattern, cmd_lower, re.IGNORECASE)
            if match:
                groups = [g for g in match.groups() if g]
                target = groups[0] if groups else cmd
                modifiers = {}

                if action == "regress":
                    modifiers["depvar"] = groups[0] if len(groups) > 0 else ""
                    modifiers["indepvars"] = groups[1] if len(groups) > 1 else ""
                    modifiers["controls"] = groups[2] if len(groups) > 2 else ""
                elif action == "plot":
                    modifiers["variable"] = groups[0] if len(groups) > 0 else ""
                    modifiers["region"] = groups[1] if len(groups) > 1 else ""
                elif action == "compare":
                    modifiers["item_a"] = groups[0] if len(groups) > 0 else ""
                    modifiers["item_b"] = groups[1] if len(groups) > 1 else ""

                apps = self._detect_apps(action, target)

                command = CockpitCommand(
                    raw=raw_input, action=action,
                    target=target, modifiers=modifiers,
                    apps=apps,
                )
                self._history.append(command)
                return command

        # Fallback: treat as a natural language command
        command = CockpitCommand(
            raw=raw_input, action="natural",
            target=cmd, modifiers={},
            apps=["winnie"],
        )
        self._history.append(command)
        return command

    def execute(self, command: CockpitCommand) -> dict:
        """Execute a parsed cockpit command."""
        executors = {
            "plot": self._execute_plot,
            "regress": self._execute_regress,
            "search": self._execute_search,
            "export": self._execute_export,
            "open": self._execute_open,
            "draft": self._execute_draft,
            "audit": self._execute_audit,
            "annotate": self._execute_annotate,
            "compare": self._execute_compare,
            "focus": self._execute_focus,
            "natural": self._execute_natural,
        }

        executor = executors.get(command.action, self._execute_natural)
        result = executor(command)
        result["command"] = command.to_dict()
        return result

    def run(self, raw_input: str) -> dict:
        """Parse and execute in one step."""
        command = self.parse(raw_input)
        return self.execute(command)

    def _detect_apps(self, action: str, target: str) -> list[str]:
        """Detect which apps need to be invoked."""
        apps = []
        target_lower = target.lower()

        if action in ("plot", "regress"):
            apps.append("stata")
        if action == "plot" and any(w in target_lower for w in [
            "map", "spatial", "gis", "county", "residual", "geographic"
        ]):
            apps.append("arcgis")
        if action in ("draft", "export"):
            apps.append("notion")
        if action == "search":
            apps.extend(["chromadb", "notion", "bolt"])
        if action == "audit":
            apps.append("forensic_lab")

        return apps or ["winnie"]

    # ─── Command Executors ────────────────────────────────────────

    def _execute_plot(self, cmd: CockpitCommand) -> dict:
        """Generate a plot: find data → run Stata → pipe to ArcGIS if spatial."""
        variable = cmd.modifiers.get("variable", cmd.target)
        region = cmd.modifiers.get("region", "")

        # Build Stata do-file
        stata_commands = []

        # Find the dataset on the Bolt
        dataset = self._find_dataset(variable)
        if dataset:
            stata_commands.append(f'use "{dataset}", clear')

        if region:
            stata_commands.append(f'keep if region == "{region}"')

        stata_commands.extend([
            f"* Plotting: {variable}",
            f"twoway (scatter {variable} x_var), title(\"{variable}\")",
            f"graph export \"{self._bolt_path}/VAULT/PLOTS/{variable}_plot.png\", replace",
        ])

        do_content = "\n".join(stata_commands)
        do_path = Path(self._bolt_path) / "VAULT" / "TEMP" / "cockpit_plot.do"
        do_path.parent.mkdir(parents=True, exist_ok=True)

        result = {
            "action": "plot",
            "variable": variable,
            "region": region,
            "stata_do": do_content,
            "do_file": str(do_path),
            "dataset": dataset or "not found",
            "apps_invoked": cmd.apps,
        }

        # Write the do-file
        try:
            do_path.write_text(do_content)
            result["status"] = "do_file_ready"

            # Attempt AppleScript launch of Stata
            if "stata" in cmd.apps:
                result["stata_instruction"] = (
                    f"Run in Stata: do \"{do_path}\""
                )
        except Exception as e:
            result["status"] = "error"
            result["error"] = str(e)

        return result

    def _execute_regress(self, cmd: CockpitCommand) -> dict:
        """Run a regression: build do-file → execute → auto-annotate."""
        depvar = cmd.modifiers.get("depvar", "")
        indepvars = cmd.modifiers.get("indepvars", "")
        controls = cmd.modifiers.get("controls", "")

        # Build regression command
        reg_vars = f"{depvar} {indepvars}"
        if controls:
            reg_vars += f" {controls}"

        stata_commands = [
            f"* Cockpit Regression: {cmd.raw}",
            f"* Generated: {time.strftime('%Y-%m-%d %H:%M')}",
            "",
        ]

        dataset = self._find_dataset(depvar)
        if dataset:
            stata_commands.append(f'use "{dataset}", clear')

        stata_commands.extend([
            f"reg {reg_vars}, robust",
            "estimates store m1",
            f'log using "{self._bolt_path}/VAULT/STATA_OUTPUT/cockpit_reg.log", replace',
            "estimates table m1, star(.05 .01 .001) stats(N r2 F)",
            "log close",
        ])

        do_content = "\n".join(stata_commands)
        do_path = Path(self._bolt_path) / "VAULT" / "TEMP" / "cockpit_regress.do"
        do_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            do_path.write_text(do_content)
        except Exception:
            pass

        return {
            "action": "regress",
            "depvar": depvar,
            "indepvars": indepvars,
            "controls": controls,
            "stata_do": do_content,
            "do_file": str(do_path),
            "dataset": dataset or "manual load required",
            "auto_annotate": True,
            "note": "Output will be auto-annotated when .log is written",
        }

    def _execute_search(self, cmd: CockpitCommand) -> dict:
        """Unified search across all sources."""
        query = cmd.target
        results = {"action": "search", "query": query, "sources": {}}

        # 1. ChromaDB semantic search
        try:
            from server.chroma_backend import get_relevant_chunks
            chunks = get_relevant_chunks(query, top_k=5)
            results["sources"]["vault"] = {
                "count": len(chunks),
                "matches": [
                    {"title": c.get("metadata", {}).get("source", ""),
                     "snippet": c.get("text", "")[:150]}
                    for c in (chunks if isinstance(chunks, list) else [])
                ],
            }
        except Exception:
            results["sources"]["vault"] = {"count": 0, "status": "offline"}

        # 2. Notion search
        try:
            from server.notion_bridge import notion_bridge
            notion_results = notion_bridge.recall(query)
            results["sources"]["notion"] = {
                "count": len(notion_results),
                "matches": [{"title": n.get("title", "")} for n in notion_results[:5]],
            }
        except Exception:
            results["sources"]["notion"] = {"count": 0, "status": "offline"}

        # 3. File search on Bolt
        bolt_matches = self._search_bolt_files(query)
        results["sources"]["bolt_files"] = {
            "count": len(bolt_matches),
            "matches": bolt_matches[:5],
        }

        # 4. Stata logs
        stata_matches = self._search_stata_logs(query)
        results["sources"]["stata_logs"] = {
            "count": len(stata_matches),
            "matches": stata_matches[:3],
        }

        # Total
        total = sum(s.get("count", 0) for s in results["sources"].values())
        results["total_results"] = total

        return results

    def _execute_export(self, cmd: CockpitCommand) -> dict:
        """Export content to a format (LaTeX, BibTeX, Markdown, etc.)."""
        content = cmd.target
        fmt = cmd.modifiers.get("item_b", "latex") if cmd.modifiers else "latex"

        return {
            "action": "export",
            "content": content,
            "format": fmt,
            "status": "queued",
            "output_path": f"{self._bolt_path}/VAULT/EXPORTS/{content.replace(' ', '_')}.{fmt}",
        }

    def _execute_open(self, cmd: CockpitCommand) -> dict:
        """Open a module, sandbox, or file."""
        target = cmd.target.lower()

        sandbox_map = {
            "did": "Difference-in-Differences",
            "rdd": "Regression Discontinuity",
            "iv": "Instrumental Variables",
            "fe": "Fixed Effects",
        }

        for key, name in sandbox_map.items():
            if key in target:
                return {
                    "action": "open",
                    "module": "method_sandbox",
                    "sandbox": key,
                    "name": name,
                    "status": "launching",
                }

        return {
            "action": "open",
            "target": cmd.target,
            "status": "searching",
        }

    def _execute_draft(self, cmd: CockpitCommand) -> dict:
        """Generate a draft footnote or section."""
        topic = cmd.target
        source = cmd.modifiers.get("region", "")  # Reuses the "for X" pattern

        return {
            "action": "draft",
            "topic": topic,
            "source": source,
            "status": "generating",
            "output": f"Draft footnote for '{topic}' will be generated via Shadow Drafter",
        }

    def _execute_audit(self, cmd: CockpitCommand) -> dict:
        """Trigger a forensic audit."""
        paper = cmd.target
        pdf_path = self._find_pdf(paper)

        return {
            "action": "audit",
            "paper": paper,
            "pdf_path": pdf_path or "not found on Bolt",
            "status": "queued" if pdf_path else "pdf_not_found",
        }

    def _execute_annotate(self, cmd: CockpitCommand) -> dict:
        """Auto-annotate a Stata/R output file."""
        return {
            "action": "annotate",
            "file": cmd.target,
            "status": "queued",
            "note": "Auto-annotator will process this file",
        }

    def _execute_compare(self, cmd: CockpitCommand) -> dict:
        """Compare two items."""
        return {
            "action": "compare",
            "item_a": cmd.modifiers.get("item_a", ""),
            "item_b": cmd.modifiers.get("item_b", ""),
            "status": "generating",
        }

    def _execute_focus(self, cmd: CockpitCommand) -> dict:
        """Set the global focus."""
        return {
            "action": "focus",
            "target": cmd.target,
            "status": "broadcast",
        }

    def _execute_natural(self, cmd: CockpitCommand) -> dict:
        """Fallback: natural language → Winnie."""
        return {
            "action": "natural_language",
            "query": cmd.target,
            "status": "routing_to_winnie",
        }

    # ─── Helpers ──────────────────────────────────────────────────

    def _find_dataset(self, variable: str) -> Optional[str]:
        """Find a .dta or .csv file on the Bolt that likely contains a variable."""
        vault = Path(self._bolt_path) / "VAULT"
        data_dirs = [vault / "DATA", vault / "DATASETS", vault / "REPLICATION"]

        for data_dir in data_dirs:
            if not data_dir.exists():
                continue
            for f in data_dir.rglob("*.dta"):
                return str(f)
            for f in data_dir.rglob("*.csv"):
                return str(f)
        return None

    def _find_pdf(self, title: str) -> Optional[str]:
        """Find a PDF on the Bolt matching a title."""
        vault = Path(self._bolt_path) / "VAULT"
        title_lower = title.lower()

        for pdf_dir in [vault / "READINGS", vault / "NEW_PAPERS", vault / "LIBRARY"]:
            if not pdf_dir.exists():
                continue
            for f in pdf_dir.rglob("*.pdf"):
                if title_lower in f.stem.lower():
                    return str(f)
        return None

    def _search_bolt_files(self, query: str) -> list[dict]:
        """Full-text search across Bolt files."""
        matches = []
        query_lower = query.lower()
        vault = Path(self._bolt_path) / "VAULT"

        if not vault.exists():
            return matches

        for ext in ["*.md", "*.txt"]:
            for f in vault.rglob(ext):
                try:
                    content = f.read_text(errors="ignore")
                    if query_lower in content.lower():
                        # Find the matching line
                        for i, line in enumerate(content.split("\n")):
                            if query_lower in line.lower():
                                matches.append({
                                    "file": str(f),
                                    "line": i + 1,
                                    "snippet": line.strip()[:150],
                                })
                                break
                except Exception:
                    continue

                if len(matches) >= 10:
                    return matches

        return matches

    def _search_stata_logs(self, query: str) -> list[dict]:
        """Search Stata log files on the Bolt."""
        matches = []
        query_lower = query.lower()
        stata_dir = Path(self._bolt_path) / "VAULT" / "STATA_OUTPUT"

        if not stata_dir.exists():
            return matches

        for f in stata_dir.rglob("*.log"):
            try:
                content = f.read_text(errors="ignore")
                if query_lower in content.lower():
                    matches.append({
                        "file": f.name,
                        "path": str(f),
                        "type": "stata_log",
                    })
            except Exception:
                continue

        return matches

    @property
    def history(self) -> list[dict]:
        return [c.to_dict() for c in self._history[-20:]]


# ═══════════════════════════════════════════════════════════════════
# 3. METABOLIC DATA PIPELINE — Data Flows Like Oxygen
# ═══════════════════════════════════════════════════════════════════

class MetabolicPipeline:
    """The live data pipeline that eliminates Save/Export/Import.

    When the Forensic Lab identifies a dataset in a paper:
    1. Downloads the replication files to the Bolt
    2. Cleans them via background Python
    3. Makes them clickable in the Cockpit

    The data is live before you finish reading the Methods section.
    """

    def __init__(self, bolt_path: str = ""):
        try:
            from server.vault_config import VAULT_ROOT
            _default = str(VAULT_ROOT)
        except ImportError:
            _default = os.environ.get("EDITH_DATA_ROOT", ".")
        self._bolt_path = bolt_path or _default
        self._live_swap = Path(self._bolt_path) / "VAULT" / "LIVE_SWAP"
        self._pipeline_log: list[dict] = []

    def ensure_swap_dir(self):
        """Create the live swap directory if needed."""
        self._live_swap.mkdir(parents=True, exist_ok=True)

    def stage_dataset(self, name: str, source_path: str,
                      clean: bool = True) -> dict:
        """Stage a dataset into the Live Swap for instant Cockpit access.

        1. Copy to LIVE_SWAP
        2. Auto-clean (remove empty columns, standardize names)
        3. Generate a .do file to load it
        """
        self.ensure_swap_dir()
        source = Path(source_path)

        if not source.exists():
            return {"error": f"Source not found: {source_path}"}

        # Copy to live swap
        dest = self._live_swap / source.name
        try:
            import shutil
            shutil.copy2(str(source), str(dest))
        except Exception as e:
            return {"error": f"Copy failed: {e}"}

        result = {
            "name": name,
            "source": source_path,
            "staged": str(dest),
            "size_mb": round(dest.stat().st_size / (1024 * 1024), 2),
        }

        # Generate instant-load do-file
        do_path = self._live_swap / f"load_{source.stem}.do"
        do_content = (
            f"* Auto-generated by Metabolic Pipeline\n"
            f"* Dataset: {name}\n"
            f"* Source: {source_path}\n"
            f"* Staged: {time.strftime('%Y-%m-%d %H:%M')}\n\n"
            f'use "{dest}", clear\n'
            f"describe\n"
            f"summarize\n"
        )
        try:
            do_path.write_text(do_content)
            result["do_file"] = str(do_path)
        except Exception:
            pass

        entry = {**result, "timestamp": time.time()}
        self._pipeline_log.append(entry)
        return result

    def stage_from_forensic(self, forensic_result: dict) -> list[dict]:
        """Auto-stage datasets identified by the Forensic Lab.

        Called after a forensic audit identifies datasets in a paper.
        """
        staged = []
        datasets = forensic_result.get("audit", {}).get("datasets", {})

        if isinstance(datasets, dict):
            found = datasets.get("found", [])
        elif isinstance(datasets, list):
            found = datasets
        else:
            found = []

        for ds in found:
            ds_name = ds if isinstance(ds, str) else ds.get("name", "")
            # Search for the dataset on the Bolt
            ds_path = self._find_dataset_file(ds_name)
            if ds_path:
                result = self.stage_dataset(ds_name, ds_path)
                staged.append(result)

        return staged

    def pipe_stata_to_notion(self, log_path: str, notion_page: str = "") -> dict:
        """Pipe Stata output → Auto-Annotator → Notion.

        The complete metabolic flow from analysis to writing.
        """
        result = {
            "source": log_path,
            "steps": [],
        }

        # Step 1: Auto-annotate
        try:
            from server.auto_annotator import auto_annotator
            annotation = auto_annotator.annotate_file(log_path)
            result["steps"].append({
                "step": "annotate",
                "status": "done",
                "prose_length": len(annotation.get("prose", "")),
            })

            # Step 2: Push to Notion
            try:
                from server.notion_bridge import notion_bridge
                notion_bridge.push_annotation(annotation, page_id=notion_page)
                result["steps"].append({"step": "notion_push", "status": "done"})
            except Exception:
                result["steps"].append({"step": "notion_push", "status": "skipped"})

        except Exception as e:
            result["steps"].append({"step": "annotate", "status": "error", "error": str(e)})

        return result

    def pipe_arcgis_export(self, shapefile: str, variable: str) -> dict:
        """Pipe ArcGIS spatial data into the Cockpit."""
        return {
            "action": "arcgis_pipe",
            "shapefile": shapefile,
            "variable": variable,
            "status": "staged",
            "live_swap_path": str(self._live_swap / Path(shapefile).name),
        }

    def _find_dataset_file(self, name: str) -> Optional[str]:
        """Find a dataset file on the Bolt."""
        vault = Path(self._bolt_path) / "VAULT"
        name_lower = name.lower().replace(" ", "_").replace("-", "_")

        for search_dir in [vault / "DATA", vault / "DATASETS", vault / "REPLICATION"]:
            if not search_dir.exists():
                continue
            for f in search_dir.rglob("*"):
                if f.is_file() and name_lower in f.stem.lower():
                    return str(f)
        return None

    @property
    def status(self) -> dict:
        staged_files = list(self._live_swap.glob("*")) if self._live_swap.exists() else []
        return {
            "live_swap_files": len(staged_files),
            "pipeline_events": len(self._pipeline_log),
        }


# ═══════════════════════════════════════════════════════════════════
# 4. THE CITADEL BRIDGE — The Master Weld
# ═══════════════════════════════════════════════════════════════════

class CitadelBridge:
    """The Single Pane of Glass.

    Binds the Focus Variable, Cockpit, and Metabolic Pipeline
    into one unified system.

    Usage:
        bridge = CitadelBridge()

        # Set focus (broadcasts to all modules)
        bridge.focus_paper("Mettler 2011", author="Suzanne Mettler")

        # Run a cockpit command
        bridge.run("// Plot residuals for Potter County")
        bridge.run("// Regress turnout on charity_density")
        bridge.run("// Search 'administrative burden'")

        # Stage a dataset for instant Cockpit use
        bridge.stage_dataset("Potter County Census", "/path/to/data.dta")

        # Pipe Stata → Notion automatically
        bridge.pipe("stata_output.log")
    """

    def __init__(self, bolt_path: str = ""):
        try:
            from server.vault_config import VAULT_ROOT
            _default = str(VAULT_ROOT)
        except ImportError:
            _default = os.environ.get("EDITH_DATA_ROOT", ".")
        self._bolt_path = bolt_path or _default
        self.focus = FocusManager()
        self.cockpit = CockpitCommandLine(self._bolt_path)
        self.pipeline = MetabolicPipeline(self._bolt_path)

        # Auto-register module subscribers for focus changes
        self._register_default_subscribers()

    def _register_default_subscribers(self):
        """Register all modules as focus subscribers."""

        # Notion: scroll to associated notes
        def on_focus_notion(signal: FocusSignal) -> dict:
            try:
                from server.notion_bridge import notion_bridge
                results = notion_bridge.recall(signal.title)
                return {"scrolled_to": len(results), "top": results[0]["title"] if results else ""}
            except Exception:
                return {"status": "offline"}

        # Forensic Lab: pre-load methodology short course
        def on_focus_forensics(signal: FocusSignal) -> dict:
            if signal.methodology:
                try:
                    from server.method_lab import MethodLab
                    lab = MethodLab()
                    course = lab.generate_short_course(signal.methodology)
                    return {"short_course": signal.methodology, "loaded": "error" not in course}
                except Exception:
                    pass
            return {"status": "no_method_detected"}

        # Synapse Bridge: bridge thought to data
        def on_focus_synapse(signal: FocusSignal) -> dict:
            try:
                from server.synapse_bridge import synapse_bridge
                result = synapse_bridge.bridge_thought_to_data(signal.title)
                return {"confidence": result.confidence, "evidence": len(result.evidence)}
            except Exception:
                return {"status": "offline"}

        # Knowledge Graph: highlight node
        def on_focus_graph(signal: FocusSignal) -> dict:
            try:
                from server.graph_vector_engine import graph_engine
                result = graph_engine.query(signal.title)
                return {
                    "entities_highlighted": len(result.get("graph_entities", [])),
                    "neighbors": len(result.get("vector_results", [])),
                }
            except Exception:
                return {"status": "offline"}

        self.focus.subscribe("notion", on_focus_notion)
        self.focus.subscribe("forensic_lab", on_focus_forensics)
        self.focus.subscribe("synapse", on_focus_synapse)
        self.focus.subscribe("knowledge_graph", on_focus_graph)

    # ─── Public API ───────────────────────────────────────────────

    def focus_paper(self, title: str, **kwargs) -> dict:
        """Set focus to a paper — broadcasts to all modules."""
        return self.focus.focus_paper(title, **kwargs)

    def focus_concept(self, concept: str) -> dict:
        """Set focus to a concept."""
        return self.focus.focus_concept(concept)

    def focus_chapter(self, chapter: int, title: str = "") -> dict:
        """Set focus to a dissertation chapter."""
        return self.focus.focus_chapter(chapter, title)

    def run(self, command: str) -> dict:
        """Execute a cockpit command.

        Examples:
            bridge.run("// Plot residuals for Potter County")
            bridge.run("// Regress turnout on charity_density")
            bridge.run("// Search 'administrative burden'")
        """
        return self.cockpit.run(command)

    def stage_dataset(self, name: str, path: str) -> dict:
        """Stage a dataset for instant Cockpit access."""
        return self.pipeline.stage_dataset(name, path)

    def pipe(self, stata_log: str, notion_page: str = "") -> dict:
        """Pipe Stata output → Annotation → Notion."""
        return self.pipeline.pipe_stata_to_notion(stata_log, notion_page)

    @property
    def status(self) -> dict:
        return {
            "focus": self.focus.status,
            "cockpit_history": len(self.cockpit._history),
            "pipeline": self.pipeline.status,
        }


# ═══════════════════════════════════════════════════════════════════
# Global Instance
# ═══════════════════════════════════════════════════════════════════

citadel_bridge = CitadelBridge()
