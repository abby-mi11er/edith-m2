# E.D.I.T.H. M2 — Architecture

> **Last updated:** 2026-03-04  
> **Version:** 2.0.0  
> **Stack:** React 19 · TypeScript · Vite 7 · Electron 36 · Zustand

---

## Overview

M2 is the **lightweight desktop research assistant** — a focused 7-panel Electron app designed for graduate-level political science research. It shares a Python backend and ChromaDB vector store with M4 (the full-featured "sovereign" model) but presents a simpler, Apple-inspired interface optimized for daily use.

```
┌─────────────────────────────────────────────────────────┐
│  Electron Shell (main.cjs)                              │
│  ┌───────────────────────────────────────────────────┐  │
│  │  React Frontend (Vite → dist/index.html)          │  │
│  │  ┌─────────┬──────────────────────────────────┐   │  │
│  │  │ PanelNav│          Active Panel             │   │  │
│  │  │ ────────│   Winnie | Library | Search       │   │  │
│  │  │ Research│   Vibe   | Methods | Dive | Cite  │   │  │
│  │  └─────────┴──────────────────────────────────┘   │  │
│  └───────────────────────────────────────────────────┘  │
│         │  Vite proxy: /api → :8003, /chat → :8003      │
│         ▼                                               │
│  ┌─────────────────────────────────┐                    │
│  │  Shared Backend (Python/FastAPI)│ ← also used by M4  │
│  │  Port 8003 (M2) or 8001 (M4)   │                    │
│  │  ChromaDB: 65,803 chunks       │                    │
│  │  Gemini API (google.genai SDK)  │                    │
│  └─────────────────────────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

---

## Port Isolation (M2 vs M4)

| Resource | M2 | M4 |
|----------|:--:|:--:|
| Frontend | `5176` | `5173` |
| Backend | `8003` | `8001` |
| ChromaDB | Shared: `/Volumes/Edith Bolt/Edith_M4/ChromaDB` | Same |
| Collection | `edith_docs_pdf` (52,655 chunks) | Auto-selects largest |
| User data | `~/Library/Application Support/Edith_M2` | Bolt SSD |

---

## File Structure

```
Edith_M2/
├── electron/
│   └── main.cjs              # Electron main process (286 lines)
├── src/
│   ├── main.tsx               # React 19 entry (StrictMode)
│   ├── App.tsx                # Panel routing, health polling, drag-to-index
│   ├── store.ts               # Zustand state (chat, tabs, backend status)
│   ├── security.ts            # sanitize(), isValidUrl(), safeFetch()
│   ├── courses.json           # 24 courses + 3 research projects
│   ├── components/
│   │   ├── PanelNav.tsx       # Top navigation + system dashboard
│   │   └── ErrorBoundary.tsx  # React error boundary
│   ├── panels/
│   │   ├── WinniePanel.tsx    # RAG chat (214 lines)
│   │   ├── LibraryPanel.tsx   # Paper library + gaps (224 lines)
│   │   ├── SearchPanel.tsx    # Multi-source search (152 lines)
│   │   ├── VibeCoderPanel.tsx # Code gen: Stata/R (181 lines)
│   │   ├── MethodsLabPanel.tsx# Method decoder + flashcards (187 lines)
│   │   ├── PaperDivePanel.tsx # Deep paper analysis (210 lines)
│   │   └── CitationsPanel.tsx # Citation network (126 lines)
│   └── styles/
│       ├── tokens.css         # Design tokens (Apple light theme)
│       └── shell.css          # Layout + component styles
├── vite.config.ts             # Dev server + API proxy
├── package.json               # Scripts, deps, electron-builder config
├── index.html                 # HTML shell
├── Courses/                   # Course reading PDFs (gitignored)
├── Research/                  # Research project files (gitignored)
└── Methods/                   # Methods reference docs (gitignored)
```

**Total source:** 19 files, ~1,800 lines of TypeScript/TSX

---

## Electron Main Process

[electron/main.cjs](file:///Volumes/Edith%20Bolt/Edith_M2/electron/main.cjs) — 286 lines

**Responsibilities:**
1. **Auto-start backend** — probes `http://127.0.0.1:8003/api/status`, if offline, spawns Python backend from shared M4 server code
2. **Runtime isolation** — sets `EDITH_PORT=8003`, `EDITH_WORKERS=1`, isolated user/session data dirs, shared ChromaDB path
3. **Shared index guard** — refuses to boot if `ChromaDB/chroma.sqlite3` is missing (`EDITH_M2_REQUIRE_SHARED_INDEX=true`)
4. **Window management** — 1400×900 BrowserWindow, hidden titlebar (`hiddenInset`), dark background `#0f1729`
5. **Dev/prod routing** — dev loads `localhost:5176`, prod loads `dist/index.html`

**Backend search order:**
1. `EDITH_M2_BACKEND_ROOT` or `EDITH_BACKEND_ROOT` env var
2. `/Applications/Edith.app/Contents/Resources/edith_backend`
3. `/Volumes/Edith Bolt/Edith_M4/electron/extraResources/edith_backend`
4. `/Volumes/Edith Bolt/Edith_M4/{dist,build}/edith_backend`
5. `~/Projects/edith_safe_chat`

---

## Frontend Architecture

### State Management — Zustand

[src/store.ts](file:///Volumes/Edith%20Bolt/Edith_M2/src/store.ts) — single store, no context providers

| State | Type | Persistence |
|-------|------|-------------|
| `activeTab` | `TabId` (7 values) | Memory only |
| `messages` | `ChatMessage[]` | `localStorage` (last 100, throttled 500ms) |
| `isStreaming` | `boolean` | Memory only |
| `committeeMode` | `boolean` | Memory only |
| `backendConnected` | `boolean` | Memory only |

Chat persistence validates each message on load — drops corrupted entries and auto-cleans.

### App Shell

[src/App.tsx](file:///Volumes/Edith%20Bolt/Edith_M2/src/App.tsx) — 109 lines

- **Panel routing:** Maps `activeTab` → component via `PANELS` record
- **Keyboard shortcuts:** `Cmd+1` through `Cmd+7` switch panels
- **Health check:** Polls `/api/status` every 10 seconds, updates status indicator
- **Drag-to-index:** Dropping files triggers `/api/index/run`
- **Error boundary:** Wraps active panel with `ErrorBoundary`

### Navigation

[src/components/PanelNav.tsx](file:///Volumes/Edith%20Bolt/Edith_M2/src/components/PanelNav.tsx) — 155 lines

Two sections with dropdown menus:

| Section | Panels |
|---------|--------|
| **Research** | Winnie · Library · Search |
| **Tools** | Vibe Coder · Methods Lab · Paper Dive · Citations |

Includes a clickable **System Dashboard** dropdown that shows uptime and subsystem status from `/api/status`.

---

## Panel Details

### 1. Winnie Chat (`WinniePanel.tsx`, 214 lines)

The primary RAG chat interface.

- **Endpoint:** `POST /chat/stream` (SSE)
- **Features:**
  - Streaming responses via `ReadableStream` with real-time token display
  - Sources shown inline (retrieved from ChromaDB index)
  - Committee mode toggle (multi-model consensus)
  - Chat persistence to `localStorage` (last 100 messages)
  - `SafeMarkdown` wrapper — error boundary around `react-markdown` to prevent render crashes
  - Input sanitization via `security.ts`
- **Keyboard:** `Enter` to send, `Shift+Enter` for newline

### 2. Library (`LibraryPanel.tsx`, 224 lines)

Browse and search the indexed paper library.

- **Endpoints:**
  - `GET /api/library/sources?course={id}` — list papers by course
  - `GET /api/library/gaps` — AI-detected research gaps
  - `GET /api/library/suggestions?course={id}` — reading suggestions
- **Features:**
  - Course selector with 24 courses + 3 research projects from `courses.json`
  - Category grouping: American, Comparative, Methods, Formal Theory, IR, Policy, Reference, Research
  - Search/filter within loaded papers
  - Gap detection sidebar
  - Paper count display
- **Data:** Courses defined in `courses.json` with `drive_root: "/Volumes/Edith Bolt"`

### 3. Search (`SearchPanel.tsx`, 152 lines)

Multi-source academic search with fallback chain.

- **Endpoints (fallback chain):**
  1. `GET /api/search?q=...&source={source}` — primary unified search
  2. `GET /api/openalex/search` — OpenAlex (250M+ papers)
  3. `GET /api/scholar/search` — Google Scholar
  4. `GET /api/crossref/search` — CrossRef
- **Sources:** All Sources · Academic (OpenAlex) · My Library · Legislation (LegiScan) · News (NYT)
- **Features:**
  - Add results to library via `POST /api/ingest`
  - Mendeley sync via `POST /api/mendeley/sync`
  - Auto-fallback: if primary returns 0 results, tries OpenAlex → Scholar → CrossRef

### 4. Vibe Coder (`VibeCoderPanel.tsx`, 181 lines)

Natural-language to statistical code generation.

- **Endpoints:**
  - `POST /api/vibe/generate` — generate code from directive
  - `POST /api/vibe/explain` — explain existing code
  - `GET /api/vibe/datasets` — discover available datasets
  - `POST /api/stata/to-latex` — convert Stata output to LaTeX
- **Languages:** Stata · R · Python (toggle buttons)
- **Features:**
  - Conversational history passed to LLM for context
  - Copy code · Save as `.do`/`.R`/`.py` · Explain · Export LaTeX
  - Dataset discovery on mount (shows count in header)
  - Syntax-aware code blocks

### 5. Methods Lab (`MethodsLabPanel.tsx`, 187 lines)

Interactive methodology decoder and flashcard creator.

- **Endpoints:**
  - `POST /api/method/decode` — decode a research method
  - `POST /api/tools/flashcard` — create study flashcard
  - `POST /api/sniper/audit` — forensic methodology audit of a paper
  - `POST /api/socratic/start` — begin Socratic coaching session
  - `POST /api/socratic/respond` — continue Socratic dialogue
  - `POST /api/streams/bridge/method-to-code` — translate method to R code
- **Methods catalog (8):**
  DiD · IV · RDD · Synth Control · FE · PSM · Event Study · OLS
- **Modes:** Learn a Method · Analyze a Paper
- **Socratic toggle:** When enabled, uses `/api/socratic/*` endpoints for guided questioning instead of direct answers
- **Features:**
  - Select a method, then ask questions about it
  - Flashcard creation with question/answer/method
  - Side-by-side method comparison
  - Markdown-rendered analysis output

### 6. Paper Dive (`PaperDivePanel.tsx`, ~220 lines)

Deep analysis of a single paper.

- **Endpoints:**
  - `POST /api/deep-dive/start` — start deep dive (returns structured sections)
  - `POST /api/tools/ocr` — OCR upload for images/PDFs
  - `POST /api/peer-review` — get R1/R2/R3-style peer review
  - `POST /api/chat/stream` — follow-up questions with paper context
- **Features:**
  - Enter paper title or paste abstract for comprehensive breakdown
  - OCR upload: scan image/PDF → auto-start dive with extracted text
  - Multi-section output: summary, methodology, findings, contributions, limitations
  - Peer review on demand (simulated reviewer feedback)
  - Follow-up Q&A within paper context
  - Proactive suggestions for follow-up questions (via `useSuggestions` hook)

### 7. Citations (`CitationsPanel.tsx`, 126 lines)

Citation network explorer.

- **Endpoints:**
  - `POST /api/citations/works-cited` — forward citations
  - `POST /api/openalex/cited-by` — reverse citations
  - `POST /api/connectors/connected-papers/search` — connected papers graph
  - `POST /api/citations/suggestions` — AI reading suggestions
- **Views:** Works Cited · Cited By · Connected Papers · Suggestions
- **Features:**
  - Enter paper title or DOI
  - BibTeX export via `POST /api/export/bibtex`

---

## Shared Components

### NotesDrawer (`src/components/NotesDrawer.tsx`)
Slide-out notes panel accessible via `Cmd+N` from any panel.
- **Persistence:** `localStorage` key `edith_m2_notes`
- **Backend sync:** Fire-and-forget `POST /api/notes` (optional)
- **Animation:** `slideInRight` 200ms + `fadeIn` 150ms overlay

### SuggestionsBar (`src/components/SuggestionsBar.tsx`)
Smart intent detection bar in Winnie panel. Shows contextual action chips based on user input.
- **Backend:** `POST /api/suggestions/context` (debounced 400ms)
- **Offline fallback:** Local keyword matching (methods → Methods Lab, code → Vibe Coder, paper → Paper Dive, etc.)
- **Action:** Clicking a chip navigates to the target panel via `setActiveTab`

### useSuggestions (`src/components/useSuggestions.ts`)
Shared hook for proactive next-step suggestions.
- **API:** `POST /api/suggestions` (debounced 600ms)
- **Used by:** VibeCoderPanel, PaperDivePanel, CitationsPanel
- **Renders via:** `SuggestionChips` presentational component

### ErrorBoundary (`src/components/ErrorBoundary.tsx`)
React error boundary wrapping each panel to prevent cascading UI crashes.

### PanelNav (`src/components/PanelNav.tsx`)
Title bar with dropdown navigation and system dashboard popover.
- **Sections:** Research (Winnie, Library, Search) · Tools (Vibe Coder, Methods Lab, Paper Dive, Citations)
- **Dashboard:** Shows backend uptime and subsystem status via `GET /api/status`

---

## Design System

[src/styles/tokens.css](file:///Volumes/Edith%20Bolt/Edith_M2/src/styles/tokens.css) — Apple-inspired light theme

| Token | Value | Purpose |
|-------|-------|---------|
| `--bg-primary` | `#ffffff` | Main background |
| `--bg-secondary` | `#f5f5f7` | Sidebar, cards |
| `--accent` | `#0071e3` | Apple blue (links, active states) |
| `--text-primary` | `#1d1d1f` | Body text |
| `--font-sans` | Inter, SF Pro | UI text |
| `--font-serif` | Source Serif 4 | Reading content |
| `--font-mono` | JetBrains Mono | Code blocks |

**Design philosophy:** Clean, quiet, functional. Lots of white space, subtle depth via `box-shadow`, smooth 200ms transitions.

---

## Security

[src/security.ts](file:///Volumes/Edith%20Bolt/Edith_M2/src/security.ts)

| Function | Purpose |
|----------|---------|
| `sanitize(input, maxLength)` | Strip HTML tags, JS protocol, event handlers, limit length |
| `isValidUrl(url)` | Validate URL protocol (http/https only) |
| `safeFetch(url, options, timeout)` | Fetch with AbortController timeout (default 30s) |

Additional security headers set in `vite.config.ts`:
- `X-Content-Type-Options: nosniff`
- `X-Frame-Options: DENY`
- `Referrer-Policy: strict-origin-when-cross-origin`
- `Permissions-Policy: camera=(), microphone=(), geolocation=()`

---

## Backend Integration

M2 does **not** have its own backend — it reuses the shared Python/FastAPI server from M4 at `/Volumes/Edith Bolt/Edith_M4/server/`.

**Vite proxy** (dev mode) forwards all API calls:

```
/api/*  →  http://127.0.0.1:8003
/chat/* →  http://127.0.0.1:8003
```

### Key API Dependencies

| Panel | Endpoints Used |
|-------|---------------|
| Winnie | `POST /chat/stream`, `GET /api/status` |
| Library | `GET /api/library/sources`, `/gaps`, `/suggestions` |
| Search | `GET /api/search`, `/api/openalex/search`, `/api/scholar/search`, `/api/crossref/search`, `POST /api/ingest`, `/api/mendeley/sync` |
| Vibe Coder | `POST /api/vibe/generate`, `/explain`, `GET /api/vibe/datasets`, `POST /api/stata/to-latex` |
| Methods Lab | `POST /api/method/decode`, `/api/tools/flashcard`, `/api/sniper/audit`, `/api/socratic/start`, `/api/socratic/respond`, `/api/streams/bridge/method-to-code` |
| Paper Dive | `POST /api/deep-dive/start`, `/api/tools/ocr`, `/api/peer-review`, `/api/chat/stream` |
| Citations | `POST /api/citations/works-cited`, `/api/openalex/cited-by`, `/api/connectors/connected-papers/search`, `/api/citations/suggestions`, `/api/export/bibtex` |
| System | `GET /api/status`, `POST /api/index/run` |

---

## Course & Research Data

From `courses.json` — 24 courses across 7 categories + 3 research projects:

**Categories:**
- **American** (3): BECO, Electoral Politics, Congress Sp 2025
- **Comparative** (4): Comp. Poli. Econ, Comp. Poli. Pro-Sem, Institutions, Political Econ Institutions
- **Methods** (10): Causal Inference, Data Analysis 1&2, GIS 2&5300, Game Theory, Machine Learning (×2), MLE, Network Analysis, Research Design, Time Series, R Programming, Python Methods
- **IR** (1): IR Pro-Sem
- **Policy** (1): Food Policy
- **Formal Theory** (1): Game Theory
- **Reference** (1): Syllabus

**Research Projects:** SNAP Paper, Second Year Paper, Nationalized Politics

**File filters:**
- **Show:** `.pdf`, `.pptx`, `.docx`, `.md`, `.txt`, `.do`, `.R`
- **Hide:** `.csv`, `.dta`, `.xlsx`, `.shp`, `.dbf`, `.json`, `.DS_Store`

---

## Build & Deploy

| Command | What it does |
|---------|-------------|
| `npm run dev` | Vite dev server on `:5176` |
| `npm run electron` | Launch Electron (expects Vite running) |
| `npm run electron:dev` | Vite + Electron together via `concurrently` |
| `npm run build` | `tsc` + `vite build` → `dist/` |
| `npm run dist` | Build + `electron-builder` → `release/` |

**Dependencies:**
- **Runtime:** React 19, ReactDOM 19, react-markdown, Zustand 5
- **Dev:** Vite 7, TypeScript 5.9, Electron 36, electron-builder, concurrently, wait-on

---

## M2 vs M4 Comparison

| Dimension | M2 | M4 |
|-----------|:--:|:--:|
| **Panels** | 7 | 35+ |
| **Source files** | 22 | 200+ |
| **Total LoC** | ~2,200 | ~45,000+ |
| **Design** | Apple light, clean | Dark neural HUD |
| **State mgmt** | Zustand (localStorage) | Zustand (localStorage + server sync) |
| **Backend** | Shared (M4 server) | Own server + modules |
| **ChromaDB** | Shared (65,803 chunks) | Same |
| **Target user** | Daily coursework | Advanced research workflows |
| **Auth** | None | Session tokens, RBAC |
| **Offline** | Graceful degradation | Full offline mode |
