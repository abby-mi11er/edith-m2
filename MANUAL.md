# E.D.I.T.H. M2 — Complete User Manual

> **Even Dead, I'm The Hero**
> An AI-powered research assistant for political science graduate students

---

## Table of Contents
1. [Getting Started](#getting-started)
2. [Winnie Chat](#1-winnie-chat--the-research-brain)
3. [Library](#2-library--your-paper-collection)
4. [Search](#3-search--find-sources-everywhere)
5. [Citations](#4-citations--citation-network-explorer)
6. [Methods Lab](#5-methods-lab--learn--audit-methods)
7. [Vibe Coder](#6-vibe-coder--ai-code-generator)
8. [Paper Dive](#7-paper-dive--deep-paper-analysis)
9. [Backend Architecture](#backend-architecture)
10. [Configuration](#configuration)

---

## Getting Started

### Launching E.D.I.T.H.
Double-click the **E.D.I.T.H. M2** desktop launcher. This starts:
1. **Backend** — FastAPI server on port `8003` (Python, Uvicorn)
2. **Frontend** — React/Vite app served by Electron

The top-right corner shows **🟢 Connected** when the backend is running. If offline, panels gracefully degrade — most features require the backend.

### Navigation
- **Research** dropdown → Winnie, Library, Search, Citations
- **Tools** dropdown → Vibe Coder, Methods Lab, Paper Dive

---

## 1. Winnie Chat — The Research Brain

Winnie is your primary research interface. It's a RAG-powered (Retrieval-Augmented Generation) chat that draws from your indexed paper library stored in ChromaDB.

### 7 Chat Modes

| Mode | What It Does |
|---|---|
| **Grounded** | Default. Sources claims from your library using author-date citations `(Author, Year)` |
| **Lit Review** | Synthesizes a structured literature review across your sources |
| **Counter** | Generates counterarguments and opposing perspectives |
| **Gap Analysis** | Identifies gaps in existing literature that your research could fill |
| **Exam Prep** | Creates exam-style questions and answers from your readings |
| **Teach Me** | Explains concepts in plain language like a patient tutor |
| **Office Hours** | Simulates a professor answering questions about a topic |

### Features
- **Committee Mode** — A checkbox that activates multi-perspective synthesis. Instead of a single AI response, it convenes a virtual "committee" that deliberates from different theoretical angles and returns a unified synthesis.
- **Model Toggle** — When Claude is configured, a dropdown appears to switch between **Gemini** (default) and **Claude** models.
- **Streaming** — Responses stream in real-time via Server-Sent Events (SSE), so you see tokens appearing as they're generated.
- **Source Chips** — After each response, source chips appear showing `Author (Year) p.X` with click-through links.
- **Follow-Up Suggestions** — After each response, the AI generates 2-3 suggested follow-up questions.
- **Chat History Sidebar** — Click the clock icon (top-right) to open a sidebar showing all past conversations, organized by mode. Click any past chat to reload it. Chats auto-save after each response.
- **Suggestion Chips** — A context-aware row of clickable suggestions below the input to help continue conversations.

### How to Use
1. Select a mode (Grounded for most research questions)
2. Type your question and press **Enter** (or Shift+Enter for newlines)
3. Wait for the streamed response
4. Click source chips to verify citations
5. Use follow-up suggestions or ask your own follow-up

### Citation Format
All citations use **author-date format**: `(Author, Year)`. Source blocks reference your indexed PDFs by their actual bibliographic metadata.

---

## 2. Library — Your Paper Collection

A browsable catalog of all PDFs across your course folders.

### Layout
- **Left Sidebar** — Course categories with paper counts per course
- **"All Papers" button** — Shows all papers across every course
- **Search bar** — Filters papers by title and filename (text match)

### Course Categories
Papers are organized by categories defined in `courses.json`:
- **American** — American politics courses
- **Comparative** — Comparative politics courses
- **Methods** — Methodology courses
- **Research Projects** — Your own research paper collections

### Paper Actions
Click any paper card to expand it. You'll see:
- **File path** — The full path to the PDF on your drive
- **Paper Dive** — Jump directly to Paper Dive with this paper
- **Citations** — Jump to Citations panel to explore its citation network
- **Ask Winnie** — Jump to Winnie Chat to ask questions about this paper

### Additional Features
- **Export BibTeX** — Generates BibTeX entries for displayed papers and copies to clipboard
- **Gap Analysis Banner** — Shows detected literature gaps at the top of the All Papers view
- **Smart Reading Suggestions** — When viewing a course, similar papers are suggested based on content similarity

---

## 3. Search — Find Sources Everywhere

A unified search interface that queries multiple academic databases simultaneously.

### 5 Source Tabs

| Tab | Source | What It Searches |
|---|---|---|
| **All Sources** | Unified | Queries all backends and merges results |
| **Academic** | OpenAlex | 250M+ works from the OpenAlex academic database |
| **My Library** | ChromaDB | Your locally indexed papers via RAG embeddings |
| **Scholar** | Semantic Scholar | Semantic Scholar's academic search API |
| **News** | New York Times | NYT API for news coverage of your topic |

### Features
- **Fallback Chain** — If the primary source returns 0 results, E.D.I.T.H. automatically tries OpenAlex → Semantic Scholar → Crossref as fallbacks
- **Deduplication** — Results are deduplicated by DOI, OpenAlex ID, and title
- **Add to Library** — Each result has an **Add** button that imports the paper into your library via OpenAlex
- **Open Source** — Click "Open" to access the paper in its original journal or open-access repository

### Result Cards
Each result shows:
- Title, authors, year
- Abstract (if available)
- DOI link
- Source badge (e.g., "openalex", "scholar", "nyt")

---

## 4. Citations — Citation Network Explorer

Maps the full citation network of any paper using OpenAlex.

### 4 Views

| View | What It Shows |
|---|---|
| **Works Cited** | Papers that your seed paper cites (references) |
| **Cited By** | Papers that cite your seed paper (forward citations) |
| **Connected Papers** | Papers related by co-citation and bibliographic coupling (similarity graph) |
| **Suggestions** | AI-recommended papers based on your local library and reading patterns |

### How It Works
1. Type a paper title or keyword in the search box
2. E.D.I.T.H. resolves it to an OpenAlex work ID
3. Depending on the active view, it fetches references, citing works, or connected papers
4. Results display with title, authors, year, and DOI

### Overleaf Integration
- **Project Dropdown** — Select from your linked Overleaf projects
- **Push to Overleaf** — Push selected citations directly into your Overleaf project's bibliography
- **Export BibTeX** — Copies formatted BibTeX entries to clipboard for manual paste

### Citation Card Actions
Each citation card includes:
- **Open source** link (DOI or OpenAlex URL)
- Year and author metadata

---

## 5. Methods Lab — Learn & Audit Methods

A specialized tool for understanding and evaluating quantitative methods.

### Two Modes

#### Learn a Method
Choose from 8 pre-built method cards:

| Method | Focus |
|---|---|
| **Difference-in-Differences** | Parallel trends, causal effects |
| **Instrumental Variables** | Endogeneity, exclusion restriction |
| **Regression Discontinuity** | Assignment thresholds |
| **Synthetic Control** | Counterfactual from donor pool |
| **Fixed Effects** | Time-invariant unobservables |
| **Propensity Score Matching** | Covariate balance |
| **Event Study** | Dynamic treatment effects |
| **OLS Regression** | The workhorse |

**Workflow:**
1. Click a method card
2. Ask any question (e.g., "What is the parallel trends assumption?")
3. Get a detailed structured response
4. Use the action buttons:
   - **Make Flashcards** — AI generates Q&A flashcards from the explanation
   - **Translate to R** — Generates R code implementing the method

#### Socratic Mode
Toggle the **Socratic** checkbox to switch to guided learning. Instead of giving direct answers, E.D.I.T.H. asks probing questions back, guiding you to understand the method through back-and-forth dialogue. The session persists across turns via a session ID.

#### Analyze a Paper
Switch to the **Analyze a Paper** tab to run a forensic methodological audit:
1. Enter a paper title or DOI
2. E.D.I.T.H. identifies the method used
3. Returns a detailed report evaluating:
   - Identification strategy
   - Robustness assumptions
   - Potential threats to validity
   - Statistical power considerations

---

## 6. Vibe Coder — AI Code Generator

Write statistical analysis code by describing what you want in natural language.

### 3 Language Tabs
- **Stata** (.do files) — Default
- **R** (.R files)
- **Python** (.py files)

### How to Use
1. Select your language
2. Describe your analysis: *"Write a fixed effects regression with year and state fixed effects"*
3. Click **Generate**
4. Generated code appears in a code block

### Code Actions
After code is generated, you get 4 buttons:
- **Copy** — Copies code to clipboard
- **Save as .do / .R / .py** — Downloads the code file
- **Explain** — AI explains the generated code line by line
- **Export LaTeX** — (Stata only) Converts Stata log output to LaTeX table format

### Dataset Detection
E.D.I.T.H. scans your project directory for datasets and shows a **"X datasets available"** badge. The AI is aware of these datasets and can reference them in generated code.

### Context-Aware Suggestions
After generating code, suggestion chips appear (e.g., "Check robustness assumptions", "Generate a structured outline") for follow-up actions.

---

## 7. Paper Dive — Deep Paper Analysis

Performs a comprehensive breakdown of any research paper.

### How to Use
1. Enter a paper title or paste an abstract
2. Click **Deep Dive**
3. Wait 15-30 seconds (the backend fetches and analyzes the paper)
4. Get a structured breakdown in 6 sections:
   - **Summary** — What the paper is about
   - **Methodology** — How the research was conducted
   - **Key Findings** — Main results
   - **Contributions** — Novel contributions to the field
   - **Limitations** — Weaknesses and caveats
   - **Related Work** — How it connects to other literature

### Enhanced OCR (Mathpix)
Toggle **Enhanced OCR** when diving into papers with:
- Mathematical equations
- Complex tables
- Scanned/image-based PDFs

This uses Mathpix's API for higher-quality text extraction.

### Scan Button
The **Scan** button opens a file picker to upload an image or PDF directly for OCR processing. The extracted text is then displayed and ready for analysis.

### Follow-Up Questions
After a deep dive, a follow-up input appears. Ask specific questions about the paper (e.g., "What are the threats to internal validity?") and get answers grounded in the paper's content.

### Peer Review
Click **Get Peer Review** to receive an AI-generated peer review of the paper, covering:
- Theoretical contribution assessment
- Methodological rigor evaluation
- Suggested revisions
- Overall assessment

---

## Backend Architecture

### Core Stack
| Component | Technology |
|---|---|
| **API Server** | FastAPI + Uvicorn (port 8003) |
| **AI Models** | Google Gemini (primary), Anthropic Claude (optional) |
| **Vector DB** | ChromaDB (local embeddings for RAG) |
| **Academic API** | OpenAlex (250M+ works) |
| **Citations** | Semantic Scholar, Crossref |
| **News** | New York Times API |
| **LaTeX** | Overleaf API integration |
| **OCR** | Mathpix API (optional) |

### RAG Pipeline
1. PDFs are chunked and embedded into ChromaDB during indexing
2. User queries are embedded and matched against the vector store
3. Top-k relevant chunks are retrieved with metadata (author, year, page)
4. Chunks are formatted as source blocks with author-date labels
5. The LLM generates responses citing sources using `(Author, Year)` format

### Key Endpoints
| Endpoint | Purpose |
|---|---|
| `POST /chat/stream` | Streaming chat with RAG context |
| `POST /api/socratic/committee` | Multi-perspective committee mode |
| `GET /api/library/sources` | List all indexed papers |
| `GET /api/research/search` | Unified search across all sources |
| `POST /api/vibe/generate` | Generate analysis code |
| `POST /api/deep-dive/start` | Start paper deep dive |
| `POST /api/method/decode` | Methods lab Q&A |
| `POST /api/sniper/audit` | Forensic method audit |
| `GET /api/openalex/search` | OpenAlex academic search |
| `GET /api/openalex/citations/{id}` | Citation network |
| `POST /api/export/bibtex` | Generate BibTeX entries |
| `POST /api/connectors/overleaf/push` | Push to Overleaf |
| `POST /api/peer-review` | AI peer review |
| `POST /api/tools/flashcard` | Generate flashcards |
| `POST /api/chat/followups` | Generate follow-up suggestions |

---

## Configuration

### Environment Variables (`.env`)
| Variable | Required | Purpose |
|---|---|---|
| `GOOGLE_API_KEY` | Yes | Gemini API access |
| `ANTHROPIC_API_KEY` | No | Claude model access |
| `MATHPIX_APP_ID` | No | Enhanced OCR |
| `MATHPIX_APP_KEY` | No | Enhanced OCR |
| `OVERLEAF_EMAIL` | No | Overleaf push integration |
| `OVERLEAF_PASSWORD` | No | Overleaf push integration |
| `NYT_API_KEY` | No | News search |
| `SEMANTIC_SCHOLAR_API_KEY` | No | Scholar search |

### Course Configuration (`courses.json`)
Defines how your PDF folders map to course categories:
```json
{
  "courses": [
    { "id": "course_id", "name": "Course Name", "folder": "/path/to/pdfs", "category": "Category" }
  ],
  "research_projects": [
    { "id": "project_id", "name": "Project Name", "folder": "/path/to/papers", "category": "Research" }
  ]
}
```

### ChromaDB
The vector database lives at `electron/extraResources/edith_backend/chroma_store/`. It contains embedded chunks from all your PDFs. To re-index after adding new papers, run the indexer endpoint or use the Library panel.

---

## Tips & Tricks

1. **Use Grounded mode** for sourced answers, **Lit Review** when synthesizing across papers
2. **Committee mode** is great for seeing multiple theoretical perspectives on a debate
3. **Methods Lab + Vibe Coder** workflow: Learn a method → Translate to R → Refine in Vibe Coder
4. **Paper Dive → Citations** workflow: Deep dive a paper → Explore its citation network → Add relevant papers to library
5. **Search fallback**: If "My Library" returns nothing, the "All Sources" tab automatically queries OpenAlex and Semantic Scholar
6. **Exam Prep** mode with Socratic toggle is perfect for qualifying exam preparation
7. **Flashcards** from Methods Lab explanations help build long-term retention
