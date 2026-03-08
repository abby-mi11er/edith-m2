# E.D.I.T.H. — AI Research Assistant

**Even Dead I'm The Hero**

A local-first, privacy-hardened AI research assistant built for academic work. E.D.I.T.H. combines dual-brain AI (Winnie + Gemini), 14 external connectors, and a full citation pipeline to help you research, analyze, and write.

---

## Quick Start

### Option A: One-Click Setup

```bash
git clone https://github.com/YOUR_USERNAME/edith_safe_chat.git
cd edith_safe_chat
bash setup.sh          # installs everything
# Edit .env with your API keys
python3 desktop_launcher.py
```

### Option B: Download DMG

1. Download **Edith-1.0.0-mac-arm64.dmg** from Releases
2. Drag `Edith.app` into Applications
3. **First launch**: Right-click → Open (required for unsigned apps)
4. The setup wizard walks you through API key entry and library indexing

> No Python installation required — the backend is bundled inside the app.

### Build DMG from Source

```bash
pip install pyinstaller
bash build.sh    # → electron/dist/Edith-*.dmg
```

### First Run
The onboarding wizard will guide you through:
1. **Welcome** — overview of capabilities
2. **Connect** — paste your API keys
3. **Index** — point to your research folder

---

## Architecture

| Layer | Components |
|-------|------------|
| **Frontend** | React + Vite, 22 panels, Arctic Crystal design system |
| **Backend** | FastAPI, 475 endpoints, circuit breakers, resilience layer |
| **AI** | Dual-brain (OpenAI fine-tuned Winnie + Gemini Flash) |
| **Agentic** | Intent classification, autonomous planning, pipeline chaining |
| **Index** | ChromaDB vector store, Gemini embeddings |
| **Connectors** | 14 external APIs (see below) |

### Agentic Architecture (New)

| Feature | Description |
|---------|-------------|
| **Intent Router** | NL classification → auto-selects the right tools |
| **Autonomous Planner** | LLM plans multi-step pipelines from a goal |
| **Pipeline Chaining** | Chain tools (retrieve → extract → graph → stress-test) |
| **Proactive Suggestions** | Context-aware research suggestions |
| **Research Profile** | Persistent researcher profile that personalizes responses |
| **Session Store** | Thread-safe conversation memory with TTL |
| **Safe LLM Calls** | Never-crash wrapper with fallback chains |

### Connectors (14)

| Connector | Cost | Purpose |
|-----------|------|---------|
| OpenAI/Winnie | ~$3-8/mo | Fine-tuned research AI |
| Gemini | Free | Reasoning, second brain |
| Anthropic | ~$2-5/mo | Second-opinion analysis |
| NYT | Free | Policy journalism 1851-present |
| OpenAlex | Free | 250M+ papers, citations |
| Semantic Scholar | Free | Citation search |
| CrossRef | Free | DOI resolution |
| Connected Papers | Free | Citation graph viz |
| Mendeley | Free | Library + annotations |
| MathPix | Free tier | Equation OCR |
| Notion | Free | Workspace export |
| Overleaf | Free | LaTeX export (browser link) |
| Google Earth Engine | Free | Satellite imagery |
| Stata | License | MLE, sensitivity analysis |

### Keyboard Shortcuts

| Shortcut | Action |
|----------|--------|
| `Cmd+K` | Command palette |
| `Cmd+N` | New chat |
| `Cmd+,` | Settings |
| `Cmd+1-9` | Switch tabs |
| `Cmd+Shift+R` | Re-index |
| `Cmd+E` | Save state weld |
| `Cmd+Shift+E` | Export to Word |
| `Cmd+Shift+1/2/3` | Switch modes (Research/Analysis/Synthesis) |
| `Cmd+?` | Keyboard shortcuts help |

---

## Cost

**Realistic monthly total: $5-15/month**

Only OpenAI and Anthropic cost money. Everything else is free.

---

## Privacy

All data stays on your machine. No telemetry, no tracking, no cloud storage. See [PRIVACY_POLICY.md](docs/PRIVACY_POLICY.md).

---

## Docs

- [ARCHITECTURE.md](docs/ARCHITECTURE.md) — Full system architecture
- [CONNECTOR_SETUP.md](docs/CONNECTOR_SETUP.md) — API key setup guide
- [MODULE_WIRING_MAP.md](docs/MODULE_WIRING_MAP.md) — Module dependency map
- [PRIVACY_POLICY.md](docs/PRIVACY_POLICY.md) — Privacy policy

---

## License

MIT — see [LICENSE](LICENSE).
