#!/usr/bin/env python3
"""
Scholarly Knowledge Repositories
=================================
Extracts structured knowledge from indexed papers:
  - Datasets Repository: what datasets each paper uses
  - Methods Repository: research methods per paper with notes
  - Country-Topic Map: papers organized by country and topic
  - Intellectual Timeline: who wrote what, debates, consensus

These repositories are JSON files that Winnie can query at retrieval time
to provide professor-level context about the scholarly landscape.
"""

import json
import os
import re
import time
import logging
from pathlib import Path
from typing import Optional

log = logging.getLogger("edith.repositories")

# ---------------------------------------------------------------------------
# Dataset Patterns — common poli-sci datasets
# ---------------------------------------------------------------------------

KNOWN_DATASETS = {
    "V-Dem": ["v-dem", "varieties of democracy", "v-dem dataset"],
    "Polity IV/V": ["polity iv", "polity v", "polity score", "polity2"],
    "Freedom House": ["freedom house", "freedom in the world", "freedom score"],
    "World Bank WDI": ["world development indicators", "wdi", "world bank data"],
    "ANES": ["american national election", "anes", "national election study"],
    "CSES": ["comparative study of electoral systems", "cses"],
    "Correlates of War": ["correlates of war", "cow dataset", "cow data"],
    "UCDP": ["ucdp", "uppsala conflict data", "armed conflict data"],
    "Quality of Government": ["quality of government", "qog dataset", "qog"],
    "Manifesto Project": ["manifesto project", "cmp data", "party manifesto"],
    "ICPSR": ["icpsr", "inter-university consortium"],
    "Penn World Table": ["penn world table", "pwt"],
    "IMF WEO": ["world economic outlook", "imf data", "weo"],
    "Afrobarometer": ["afrobarometer"],
    "Eurobarometer": ["eurobarometer"],
    "Latinobarómetro": ["latinobar", "latinobarómetro", "latinobarometro"],
    "Asian Barometer": ["asian barometer", "asianbarometer"],
    "Arab Barometer": ["arab barometer"],
    "World Values Survey": ["world values survey", "wvs"],
    "General Social Survey": ["general social survey", "gss"],
    "Current Population Survey": ["current population survey", "cps"],
    "Census/ACS": ["american community survey", "acs data", "census data", "census bureau"],
    "BLS": ["bureau of labor statistics", "bls data"],
    "BEA": ["bureau of economic analysis", "bea data"],
    "FRED": ["federal reserve economic data", "fred data"],
    "DHS": ["demographic and health survey", "dhs data"],
    "ACLED": ["acled", "armed conflict location"],
    "GTD": ["global terrorism database", "gtd"],
    "SIPRI": ["sipri", "stockholm international peace"],
    "Maddison Project": ["maddison project", "maddison data"],
}

# ---------------------------------------------------------------------------
# Country/Region Detection
# ---------------------------------------------------------------------------

COUNTRY_PATTERNS = {
    # Americas
    "United States": ["united states", "u.s.", "american", "usa", "u.s.a"],
    "Mexico": ["mexico", "mexican"],
    "Brazil": ["brazil", "brazilian"],
    "Argentina": ["argentina", "argentine"],
    "Colombia": ["colombia", "colombian"],
    "Chile": ["chile", "chilean"],
    "Venezuela": ["venezuela", "venezuelan"],
    "Canada": ["canada", "canadian"],
    # Europe
    "United Kingdom": ["united kingdom", "u.k.", "british", "england", "britain"],
    "Germany": ["germany", "german", "bundesrepublik"],
    "France": ["france", "french"],
    "Italy": ["italy", "italian"],
    "Spain": ["spain", "spanish"],
    "Russia": ["russia", "russian", "soviet"],
    "Poland": ["poland", "polish"],
    "Turkey": ["turkey", "turkish", "türkiye"],
    # Asia
    "China": ["china", "chinese", "prc", "beijing"],
    "India": ["india", "indian"],
    "Japan": ["japan", "japanese"],
    "South Korea": ["south korea", "korean"],
    "Indonesia": ["indonesia", "indonesian"],
    "Pakistan": ["pakistan", "pakistani"],
    # Middle East & Africa
    "Iran": ["iran", "iranian", "persian"],
    "Israel": ["israel", "israeli"],
    "Egypt": ["egypt", "egyptian"],
    "Nigeria": ["nigeria", "nigerian"],
    "South Africa": ["south africa", "south african"],
    "Kenya": ["kenya", "kenyan"],
    # Regions
    "Latin America": ["latin america", "latin american"],
    "Sub-Saharan Africa": ["sub-saharan", "sub saharan"],
    "Middle East": ["middle east", "mena"],
    "European Union": ["european union", "eu member", "eu countries"],
    "Southeast Asia": ["southeast asia", "asean"],
    "Post-Soviet": ["post-soviet", "former soviet", "post-communist"],
}

# ---------------------------------------------------------------------------
# Topic Detection
# ---------------------------------------------------------------------------

TOPIC_PATTERNS = {
    "Democratization": ["democratization", "democratic transition", "democratic consolidation",
                         "regime change", "autocratic breakdown"],
    "Authoritarianism": ["authoritarian", "autocracy", "dictator", "one-party",
                          "competitive authoritarianism", "hybrid regime"],
    "Electoral Systems": ["electoral system", "proportional representation", "majoritarian",
                           "first-past-the-post", "mixed member", "single member district"],
    "Political Parties": ["political party", "party system", "party competition",
                           "partisan", "coalition", "party identification"],
    "Legislative Politics": ["legislature", "congress", "parliament", "roll call",
                              "committee", "filibuster", "bicameral"],
    "Executive Politics": ["executive", "presidential", "prime minister", "cabinet",
                            "executive order", "veto"],
    "Judicial Politics": ["judicial", "supreme court", "constitutional court",
                           "judicial review", "rule of law"],
    "Public Opinion": ["public opinion", "political attitude", "voter", "polling",
                        "political knowledge", "media effect"],
    "Political Economy": ["political economy", "redistribution", "inequality",
                           "trade", "economic development", "taxation"],
    "Conflict & Security": ["civil war", "interstate war", "terrorism", "insurgency",
                             "peacekeeping", "security", "military"],
    "International Relations": ["international relations", "foreign policy", "diplomacy",
                                 "international organization", "united nations"],
    "Race & Ethnicity": ["racial", "ethnic", "minority", "discrimination",
                          "identity politics", "representation"],
    "Gender Politics": ["gender", "women in politics", "feminist", "gender quota",
                         "descriptive representation"],
    "Federalism": ["federalism", "decentralization", "intergovernmental",
                    "local government", "subnational"],
    "Corruption": ["corruption", "bribery", "rent-seeking", "accountability",
                    "transparency", "anti-corruption"],
    "Social Policy": ["welfare state", "social policy", "healthcare policy",
                       "education policy", "poverty", "social protection"],
    "Environmental Politics": ["environmental", "climate change", "climate policy",
                                "green politics", "sustainability"],
    "Immigration": ["immigration", "refugee", "asylum", "migration", "border"],
}


class ScholarlyRepositories:
    """Manages all four knowledge repositories."""

    def __init__(self, store_dir: Path):
        self.store_dir = Path(store_dir)
        self.store_dir.mkdir(parents=True, exist_ok=True)
        
        self.datasets_path = self.store_dir / "datasets_repository.json"
        self.methods_path = self.store_dir / "methods_repository.json"
        self.country_map_path = self.store_dir / "country_topic_map.json"
        self.timeline_path = self.store_dir / "intellectual_timeline.json"
        
        self.datasets = self._load(self.datasets_path, default={})
        self.methods = self._load(self.methods_path, default={})
        self.country_map = self._load(self.country_map_path, default={})
        self.timeline = self._load(self.timeline_path, default={})

    def _load(self, path: Path, default=None):
        if path.exists():
            try:
                return json.loads(path.read_text())
            except Exception:
                return default or {}
        return default or {}

    def _save(self, path: Path, data):
        path.write_text(json.dumps(data, indent=2, default=str))

    # ----- Datasets Repository -----
    
    def extract_datasets(self, text: str, paper_id: str, paper_meta: dict = None):
        """Detect datasets mentioned in a text chunk and register them."""
        t = text.lower()
        meta = paper_meta or {}
        
        for dataset_name, keywords in KNOWN_DATASETS.items():
            if any(kw in t for kw in keywords):
                if dataset_name not in self.datasets:
                    self.datasets[dataset_name] = {
                        "name": dataset_name,
                        "papers": [],
                        "first_seen": time.strftime("%Y-%m-%d"),
                    }
                # Add paper reference (deduplicate)
                existing_ids = {p["id"] for p in self.datasets[dataset_name]["papers"]}
                if paper_id not in existing_ids:
                    self.datasets[dataset_name]["papers"].append({
                        "id": paper_id,
                        "title": meta.get("title_guess", ""),
                        "author": meta.get("author_guess", ""),
                        "year": meta.get("year_guess", ""),
                    })
        
        self._save(self.datasets_path, self.datasets)

    def get_dataset_info(self, dataset_name: str) -> dict:
        """Get info about a dataset and all papers that use it."""
        # Fuzzy match
        for name, info in self.datasets.items():
            if dataset_name.lower() in name.lower() or name.lower() in dataset_name.lower():
                return info
        return {}

    def list_datasets(self) -> list:
        """List all known datasets sorted by usage count."""
        return sorted(
            [{"name": k, "paper_count": len(v["papers"])} for k, v in self.datasets.items()],
            key=lambda x: x["paper_count"],
            reverse=True
        )

    # ----- Methods Repository -----

    def extract_methods(self, text: str, paper_id: str, paper_meta: dict = None):
        """Detect research methods and register with notes."""
        from server.chroma_backend import detect_methodology
        t = text.lower()
        meta = paper_meta or {}
        methods_found = detect_methodology(text)
        
        for method in methods_found:
            if method not in self.methods:
                self.methods[method] = {
                    "name": method,
                    "papers": [],
                    "notes": [],
                    "first_seen": time.strftime("%Y-%m-%d"),
                }
            existing_ids = {p["id"] for p in self.methods[method]["papers"]}
            if paper_id not in existing_ids:
                # Extract a sentence about how they used this method
                method_context = self._extract_method_context(text, method)
                self.methods[method]["papers"].append({
                    "id": paper_id,
                    "title": meta.get("title_guess", ""),
                    "author": meta.get("author_guess", ""),
                    "year": meta.get("year_guess", ""),
                    "usage_context": method_context,
                })
        
        self._save(self.methods_path, self.methods)

    def _extract_method_context(self, text: str, method: str) -> str:
        """Extract 1-2 sentences about how a method was used."""
        from server.chroma_backend import METHODOLOGY_PATTERNS
        keywords = METHODOLOGY_PATTERNS.get(method, [])
        for kw in keywords:
            idx = text.lower().find(kw)
            if idx >= 0:
                # Get surrounding sentence
                start = max(0, text.rfind(".", 0, idx) + 1)
                end = text.find(".", idx + len(kw))
                if end < 0:
                    end = min(len(text), idx + 200)
                return text[start:end + 1].strip()[:300]
        return ""

    def get_method_info(self, method_name: str) -> dict:
        """Get info about a method and all papers using it."""
        for name, info in self.methods.items():
            if method_name.lower() in name.lower() or name.lower() in method_name.lower():
                return info
        return {}

    # ----- Country-Topic Map -----

    def extract_country_topics(self, text: str, paper_id: str, paper_meta: dict = None):
        """Detect countries/regions and topics, building the map."""
        t = text.lower()
        meta = paper_meta or {}
        
        countries_found = []
        for country, keywords in COUNTRY_PATTERNS.items():
            if any(kw in t for kw in keywords):
                countries_found.append(country)
        
        topics_found = []
        for topic, keywords in TOPIC_PATTERNS.items():
            if any(kw in t for kw in keywords):
                topics_found.append(topic)
        
        for country in countries_found:
            if country not in self.country_map:
                self.country_map[country] = {"topics": {}}
            
            for topic in topics_found:
                if topic not in self.country_map[country]["topics"]:
                    self.country_map[country]["topics"][topic] = []
                
                existing_ids = {p["id"] for p in self.country_map[country]["topics"][topic]}
                if paper_id not in existing_ids:
                    self.country_map[country]["topics"][topic].append({
                        "id": paper_id,
                        "title": meta.get("title_guess", ""),
                        "author": meta.get("author_guess", ""),
                        "year": meta.get("year_guess", ""),
                    })
        
        self._save(self.country_map_path, self.country_map)

    def get_country_papers(self, country: str, topic: str = None) -> dict:
        """Get papers about a country, optionally filtered by topic."""
        for name, info in self.country_map.items():
            if country.lower() in name.lower():
                if topic:
                    for t, papers in info.get("topics", {}).items():
                        if topic.lower() in t.lower():
                            return {"country": name, "topic": t, "papers": papers}
                return info
        return {}

    # ----- Intellectual Timeline -----

    def add_timeline_entry(self, theory_or_topic: str, author: str, year: str,
                           contribution: str, paper_id: str = "",
                           entry_type: str = "contribution"):
        """Add an entry to the intellectual timeline.
        
        entry_type: 'founding', 'contribution', 'critique', 'response', 'synthesis', 'consensus'
        """
        key = theory_or_topic.lower().strip()
        if key not in self.timeline:
            self.timeline[key] = {
                "topic": theory_or_topic,
                "entries": [],
            }
        
        self.timeline[key]["entries"].append({
            "author": author,
            "year": year,
            "type": entry_type,
            "contribution": contribution[:500],
            "paper_id": paper_id,
            "added": time.strftime("%Y-%m-%d"),
        })
        
        # Sort by year
        self.timeline[key]["entries"].sort(
            key=lambda e: e.get("year", "0000")
        )
        
        self._save(self.timeline_path, self.timeline)

    def get_timeline(self, topic: str) -> dict:
        """Get the intellectual timeline for a topic/theory."""
        for key, info in self.timeline.items():
            if topic.lower() in key:
                return info
        return {}

    def get_debate_summary(self, topic: str) -> dict:
        """Summarize the debate around a topic: who argues what."""
        timeline = self.get_timeline(topic)
        if not timeline:
            return {}
        
        entries = timeline.get("entries", [])
        contributions = [e for e in entries if e["type"] == "contribution"]
        critiques = [e for e in entries if e["type"] == "critique"]
        responses = [e for e in entries if e["type"] == "response"]
        consensus = [e for e in entries if e["type"] == "consensus"]
        
        return {
            "topic": timeline.get("topic", topic),
            "total_entries": len(entries),
            "founding": [e for e in entries if e["type"] == "founding"],
            "contributions": contributions,
            "critiques": critiques,
            "responses": responses,
            "consensus_points": consensus,
            "year_range": f"{entries[0]['year']}–{entries[-1]['year']}" if entries else "",
        }

    # ----- Bulk Processing -----

    def process_chunk(self, text: str, paper_id: str, paper_meta: dict = None):
        """Process a text chunk through all four repositories."""
        self.extract_datasets(text, paper_id, paper_meta)
        self.extract_methods(text, paper_id, paper_meta)
        self.extract_country_topics(text, paper_id, paper_meta)

    def stats(self) -> dict:
        """Return repository statistics."""
        return {
            "datasets": len(self.datasets),
            "methods": len(self.methods),
            "countries": len(self.country_map),
            "timeline_topics": len(self.timeline),
            "total_dataset_refs": sum(len(v["papers"]) for v in self.datasets.values()),
            "total_method_refs": sum(len(v["papers"]) for v in self.methods.values()),
        }
