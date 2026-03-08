# E.D.I.T.H. M2

**Enhanced Digital Intelligence for Thesis and Humanities**

AI-powered research assistant for graduate students. Connects your external SSD full of papers to a Gemini-powered RAG pipeline with 7 specialized research panels, integrated citation management, and multi-model AI support.

---

## What It Does

| Panel | Description |
|---|---|
| **Winnie** | AI chat grounded in your papers — 7 modes (Lit Review, Counter, Gap Analysis, Exam Prep, Teach Me, Office Hours) with Claude/Gemini model selector |
| **Library** | Browse and search all papers organized by course, with drag-and-drop indexing |
| **Search** | Multi-source search: OpenAlex, Semantic Scholar, your library, and NYT news |
| **Vibe Coder** | Generate Stata/R/Python code from natural language with dataset auto-discovery |
| **Methods Lab** | Learn causal inference methods + forensic paper audits |
| **Paper Dive** | Deep-dive analysis of any paper with peer review and optional Mathpix OCR |
| **Citations** | Citation network explorer with BibTeX export and Overleaf push to multiple projects |

### Integrations

| Service | Feature |
|---|---|
| **Gemini** | Primary AI model for all reasoning and synthesis |
| **Claude** | Alternative AI model (toggle in Winnie chat) |
| **OpenAlex** | Academic paper search (free, no key needed) |
| **Semantic Scholar** | Scholar tab — academic search with citation data |
| **NYT** | News search for current events research |
| **Mendeley** | Sync your reference library |
| **Overleaf** | Push citations to multiple LaTeX projects |
| **Mathpix** | Enhanced OCR for equations in PDFs |
| **Notion** | Export notes from the Quick Notes drawer |
| **Sentry** | Error monitoring (optional) |

---

## Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Node.js** 18+ — `brew install node`
- **Python** 3.11+ — `brew install python`
- **External SSD** with your research papers (organized by course/topic)
- **Google Gemini API key** (required) — [Get one free](https://aistudio.google.com/apikey)

---

## Setup

### 1. Clone the repo to your external drive

```bash
git clone https://github.com/YOUR-USERNAME/edith-m2.git /Volumes/YOUR_DRIVE/Edith_M2
cd /Volumes/YOUR_DRIVE/Edith_M2
```

### 2. Install frontend dependencies

```bash
npm install
```

### 3. Set up the Python backend

```bash
cd electron/extraResources/edith_backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ../../..
```

### 4. Configure your API keys

```bash
cp electron/extraResources/edith_backend/.env.example electron/extraResources/edith_backend/.env
```

Edit the `.env` file and add your keys. Only `GOOGLE_API_KEY` is required — all others are optional:

```
GOOGLE_API_KEY=your_real_gemini_key_here
```

See `.env.example` for the full list of optional integrations (Claude, Mathpix, Notion, Overleaf, Mendeley, etc.).

### 5. Organize your papers on the SSD

Create folders on your drive root for each course or topic:

```
/Volumes/YOUR_DRIVE/
├── Edith_M2/              ← this repo
├── My_Course_1/           ← drop PDFs here
├── My_Course_2/
├── Library/
│   └── Datasets/          ← .dta, .csv, .xlsx for Vibe Coder
└── Edith_M4/
    └── ChromaDB/          ← vector database (auto-created)
```

### 6. Register your courses

Edit `src/courses.json` to match your folder names:

```json
{
  "drive_root": "/Volumes/YOUR_DRIVE",
  "courses": [
    {
      "id": "my_course_1",
      "name": "My Course 1",
      "code": "Fall 2025",
      "folder": "My_Course_1",
      "category": "Methods"
    }
  ]
}
```

**Categories:** `American`, `Comparative`, `Methods`, `IR`, `Formal Theory`, `Policy`, `Reference`, `Research`

### 7. Set up the launcher

Edit `launch_m2_desktop.sh` to point to your drive, then copy to Desktop:

```bash
cp launch_m2_desktop.sh ~/Desktop/"Launch EDITH.command"
chmod +x ~/Desktop/"Launch EDITH.command"
```

### 8. Launch

```bash
# Double-click "Launch EDITH.command" on your Desktop, or:
npm run launch
```

This starts the backend (port 8003), frontend (port 5176), and opens an Electron window.

---

## How It Works

When you ask Winnie a question:

1. **Query Rewriting** — question is split into 3 retrieval queries (keyword, semantic, methodological)
2. **RAG Retrieval** — ChromaDB searches your corpus for the top 20 relevant chunks
3. **Scholarly Context** — detected methods and theories are tagged on each source
4. **Depth Classification** — question classified as quick, standard, or debate
5. **Streaming Generation** — Gemini (or Claude) generates the answer token-by-token, citing sources inline
6. **Hallucination Audit** — a second model pass checks every citation against source text
7. **Follow-up Suggestions** — 3 follow-up questions are generated

**Model chain:** `gemini-2.5-flash` → `gemini-2.0-flash` → `gemini-2.5-pro` (automatic fallback). Code generation uses `gpt-4.1` if OpenAI key is set. Claude available as alternative via toggle.

---

## Adding Content

### Papers & Readings

1. Create a folder on your drive: `/Volumes/YOUR_DRIVE/New_Course/`
2. Drop PDFs into it
3. Add an entry to `src/courses.json`
4. Restart — papers appear in Library automatically

### Datasets (for Vibe Coder)

Drop files into `/Volumes/YOUR_DRIVE/Library/Datasets/`:

Supported: `.dta`, `.csv`, `.tsv`, `.xlsx`, `.xls`, `.rds`, `.rdata`, `.sav`, `.parquet`, `.sqlite`, `.json`, `.jsonl`

No configuration needed — datasets are auto-discovered.

---

## Configuration

All configuration is through environment variables in `.env`:

| Variable | Default | Required | Description |
|---|---|---|---|
| `GOOGLE_API_KEY` | — | **Yes** | Gemini API key |
| `OPENAI_API_KEY` | — | No | OpenAI key for code generation |
| `ANTHROPIC_API_KEY` | — | No | Claude — enables model toggle in Winnie |
| `MATHPIX_APP_KEY` | — | No | Enhanced OCR for equations in PDFs |
| `NYT_API_KEY` | — | No | New York Times news search |
| `SEMANTIC_SCHOLAR_API_KEY` | — | No | Higher rate limits for Scholar tab |
| `SERPAPI_KEY` | — | No | Google Scholar search |
| `NOTION_TOKEN` | — | No | Notion note export |
| `NOTION_DATABASE_ID` | — | No | Target database for Notion exports |
| `MENDELEY_CLIENT_ID` | — | No | Mendeley reference sync |
| `OVERLEAF_GIT_TOKEN` | — | No | Overleaf LaTeX push |
| `OVERLEAF_PROJECTS` | — | No | JSON map of project names to Git URLs |
| `SENTRY_DSN` | — | No | Error monitoring |
| `EDITH_MODEL` | `gemini-2.5-flash` | No | Primary LLM model |
| `EDITH_DATA_ROOT` | Drive root | No | Root directory for papers |
| `EDITH_M2_BACKEND_PORT` | `8003` | No | Backend port |
| `EDITH_M2_FRONTEND_PORT` | `5176` | No | Frontend port |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "Drive not found" | Plug in your external SSD |
| "Offline" in browser | Check Terminal for Python errors |
| Empty Library | Verify course folders match `courses.json` |
| Empty responses | Verify `GOOGLE_API_KEY` is set in `.env` |
| Claude toggle missing | Add `ANTHROPIC_API_KEY` to `.env` |
| No Scholar results | Add `SEMANTIC_SCHOLAR_API_KEY` for higher rate limits |
| Bad code output | Add `OPENAI_API_KEY` for GPT-4.1 |
| Port in use | `lsof -ti:8003 \| xargs kill` |

---

## Tech Stack

- **Frontend:** React + TypeScript + Vite
- **Backend:** Python + FastAPI + Uvicorn
- **LLM:** Google Gemini API + Anthropic Claude API + OpenAI API
- **Vector DB:** ChromaDB with sentence-transformers
- **Desktop:** Electron (optional packaging)
- **Monitoring:** Sentry (optional)

---

## License

MIT — see [LICENSE](LICENSE)
