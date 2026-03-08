# E.D.I.T.H. M2 — Complete Setup & Usage Guide

---

## Part 0: Download & Install

### Prerequisites

You need these installed on your Mac before starting:

| Tool | Install Command | Check |
|---|---|---|
| **Git** | `xcode-select --install` | `git --version` |
| **Node.js 18+** | `brew install node` | `node --version` |
| **Python 3.11+** | `brew install python` | `python3 --version` |
| **Homebrew** (if needed) | `/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"` | `brew --version` |

### Step 1: Download from GitHub

**Option A — Git clone (recommended):**
```bash
# Clone directly to your external SSD
git clone https://github.com/REPO_OWNER/edith-m2.git "/Volumes/YOUR_DRIVE/Edith_M2"
cd "/Volumes/YOUR_DRIVE/Edith_M2"
```

**Option B — Download ZIP:**
1. Go to the GitHub repo page
2. Click the green **Code** button → **Download ZIP**
3. Unzip to your external SSD as `/Volumes/YOUR_DRIVE/Edith_M2/`

### Step 2: Install Frontend Dependencies

```bash
cd "/Volumes/YOUR_DRIVE/Edith_M2"
npm install
```

This installs React, Vite, and all frontend packages (~2 minutes).

### Step 3: Install Backend Dependencies

```bash
cd electron/extraResources/edith_backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This creates an isolated Python environment and installs FastAPI, ChromaDB, sentence-transformers, and all backend packages (~3 minutes).

### Step 4: Configure API Keys

```bash
# Still in the edith_backend directory:
cp .env.example .env
```

Open `.env` in any text editor and paste your Google Gemini key (see Part 1 below for details):
```
GOOGLE_API_KEY=AIza...your_real_key_here...
```

### Step 5: Set Up the One-Click Launcher

```bash
cd "/Volumes/YOUR_DRIVE/Edith_M2"
```

Edit `launch_m2_desktop.sh` — change line 6 to your drive name:
```bash
BOLT="/Volumes/YOUR_DRIVE/Edith_M2"
```

Copy to Desktop:
```bash
cp launch_m2_desktop.sh ~/Desktop/"Launch EDITH.command"
chmod +x ~/Desktop/"Launch EDITH.command"
```

### Step 6: Launch

**Double-click `Launch EDITH.command` on your Desktop.**

Chrome opens to E.D.I.T.H. You're ready to go.

---

## Part 1: API Keys

### All Available API Keys

| Key | Required? | What It Powers | Free Tier? |
|---|---|---|---|
| **Google Gemini** | **Yes** | Winnie chat, Paper Dive, Methods Lab, Citations, peer review | Yes — generous free tier |
| **OpenAI** | Recommended | Vibe Coder code generation (GPT-4.1), fine-tuned Winnie models | Pay-as-you-go |
| **Semantic Scholar** | No | Enhanced citation search with higher rate limits | Yes — request access |
| **LegiScan** | No | US legislation search in Search panel | Yes — free tier |
| **NYT** | No | New York Times article search | Yes — free tier |
| **SerpAPI** | No | Google Scholar search results | Free trial |
| **Perplexity** | No | AI-powered search | Pay-as-you-go |
| **Mendeley** | No | Citation library sync from Mendeley account | Yes |
| **Zotero** | No | Citation library sync from Zotero | Yes |
| **Notion** | No | Note sync to/from Notion databases | Yes |
| **Overleaf** | No | LaTeX document sync | Yes |
| **Anthropic** | No | Claude as an alternative LLM | Pay-as-you-go |
| **Mathpix** | No | OCR for equations in scanned PDFs | Free trial |
| **Google Earth Engine** | No | Spatial/GIS data features | Yes — academic |
| **Sentry** | No | Error monitoring (for developers) | Yes — free tier |

> **Only `GOOGLE_API_KEY` is required.** Everything else is optional and progressively unlocks more features. Start with just Gemini and add more keys as you want more capabilities.

### How to Get Each Key

**Google Gemini (required):**
1. Go to [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Click "Create API Key"
3. Copy the key (starts with `AIza...`)

**OpenAI (recommended):**
1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Click "Create new secret key"
3. Copy the key (starts with `sk-...`)
4. Add a payment method at [platform.openai.com/account/billing](https://platform.openai.com/account/billing)

**Semantic Scholar:**
1. Go to [semanticscholar.org/product/api](https://www.semanticscholar.org/product/api#Partner-Form)
2. Request an API key (usually approved in 1-2 days)

**LegiScan:**
1. Go to [legiscan.com/legiscan](https://legiscan.com/legiscan)
2. Create account → get API key

**NYT:**
1. Go to [developer.nytimes.com](https://developer.nytimes.com/get-started)
2. Create an app → copy API key

**Mendeley:**
1. Go to [dev.mendeley.com](https://dev.mendeley.com)
2. Register an application
3. Copy the Client ID and Client Secret
4. Use E.D.I.T.H.'s OAuth flow (Settings → Connectors) to get access/refresh tokens

**Zotero:**
1. Go to [zotero.org/settings/keys](https://www.zotero.org/settings/keys)
2. Create a new private key

**Notion:**
1. Go to [notion.so/my-integrations](https://www.notion.so/my-integrations)
2. Click "New integration" → copy token
3. Share your database with the integration

### Where to Put Them

```bash
# Copy the template:
cp electron/extraResources/edith_backend/.env.example \
   electron/extraResources/edith_backend/.env
```

Open `.env` in any text editor and fill in the keys you have. See `.env.example` for the full list with links to get each key.

> **Never commit `.env` to git.** It is already in `.gitignore`.

---

## Part 2: Your Drive (SSD Setup)

### Folder Structure

Plug in an external SSD and create this layout:

```
/Volumes/YOUR_DRIVE/
│
├── Edith_M2/                    ← this repo (clone here)
│
├── Your_Course_1/               ← drop PDFs here
│   ├── paper1.pdf
│   ├── lecture_notes.pdf
│   └── syllabus.pdf
│
├── Your_Course_2/
│   └── readings/
│       ├── author2020.pdf
│       └── author2021.pdf
│
├── Library/
│   └── Datasets/                ← for Vibe Coder
│       ├── my_data.dta          ← Stata
│       ├── survey.csv           ← CSV
│       └── panel_data.xlsx      ← Excel
│
└── Edith_M4/
    └── ChromaDB/                ← vector database (auto-created on first index)
```

### Register Your Courses

Edit `src/courses.json`:

```json
{
  "drive_root": "/Volumes/YOUR_DRIVE",
  "courses": [
    {
      "id": "my_course_1",
      "name": "My Course 1",
      "code": "Fall 2025",
      "folder": "Your_Course_1",
      "category": "Methods"
    },
    {
      "id": "my_course_2",
      "name": "My Course 2",
      "code": "Spring 2026",
      "folder": "Your_Course_2",
      "category": "Comparative"
    }
  ],
  "research_projects": [
    {
      "id": "my_thesis",
      "name": "My Thesis",
      "folder": "Thesis_Project",
      "category": "Research"
    }
  ]
}
```

**Categories:** `American`, `Comparative`, `Methods`, `IR`, `Formal Theory`, `Policy`, `Reference`, `Research`

---

## Part 3: Training & Indexing

### What "Training" Means

E.D.I.T.H. uses **Retrieval-Augmented Generation (RAG)**. Your PDFs are chunked into text segments, converted to vector embeddings, and stored in ChromaDB. When you ask a question, the system finds the most relevant chunks and sends them to Gemini alongside your question.

### How to Index Your Papers

**Option A — Automatic (on launch):**
The launcher script validates ChromaDB on startup. If the index exists, it uses it. If not, it creates an empty one.

**Option B — Manual indexing:**
```bash
cd electron/extraResources/edith_backend
source .venv/bin/activate
python chroma_index.py --data-root /Volumes/YOUR_DRIVE --collection edith_docs_pdf
```

This scans all PDF files in your course folders and indexes them into ChromaDB. It takes ~5-15 minutes depending on how many papers you have.

### Re-indexing After Adding Papers

Drop new PDFs into your course folders, then re-run the index command. It will add new documents without removing existing ones.

### Key Settings

| Variable | Default | Description |
|---|---|---|
| `EDITH_CHROMA_DIR` | `Edith_M4/ChromaDB` | Where the vector database lives |
| `EDITH_CHROMA_COLLECTION` | `edith_docs_pdf` | Collection name |
| `EDITH_EMBED_MODEL` | `all-MiniLM-L6-v2` | Embedding model (sentence-transformers) |

---

## Part 4: Launching E.D.I.T.H.

### First-Time Setup

```bash
# 1. Install frontend
cd /Volumes/YOUR_DRIVE/Edith_M2
npm install

# 2. Install backend
cd electron/extraResources/edith_backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure (see Part 1 above)
cp .env.example .env
# Edit .env with your keys

# 4. Set up the launcher
cd /Volumes/YOUR_DRIVE/Edith_M2
# Edit launch_m2_desktop.sh line 6:
#   BOLT="/Volumes/YOUR_DRIVE/Edith_M2"
cp launch_m2_desktop.sh ~/Desktop/"Launch EDITH.command"
chmod +x ~/Desktop/"Launch EDITH.command"
```

### Daily Launch

**Double-click `Launch EDITH.command` on your Desktop.** That's it.

It does:
1. Checks if your drive is connected
2. Kills any stale processes on ports 8003/5176
3. Starts the Python backend (FastAPI on port 8003)
4. Starts the React frontend (Vite on port 5176)
5. Waits for both to be healthy
6. Opens Chrome to `http://localhost:5176`

### Shutting Down

Press `Ctrl+C` in the Terminal window. Both backend and frontend stop cleanly.

---

## Part 5: Using Every Tool

### 🟣 Winnie (Chat Panel)

**What it is:** Your AI research assistant grounded in your paper library.

**How to use:**
1. Type a question in the input box at the bottom
2. Hit Enter or click Send
3. Winnie retrieves relevant passages from your papers, then generates a cited answer
4. Follow-up suggestions appear below — click any to continue the thread

**7 Modes** (buttons above the input):

| Mode | Best For | Example |
|---|---|---|
| **Grounded** | Standard questions | "What does Acemoglu argue about institutions?" |
| **Lit Review** | Literature synthesis | "Review the literature on democratic backsliding" |
| **Counter** | Stress-testing claims | "Counter the argument that trade promotes peace" |
| **Gap Analysis** | Finding research gaps | "What gaps exist in the electoral accountability literature?" |
| **Exam Prep** | Study questions | "Generate exam questions on principal-agent theory" |
| **Teach Me** | Explaining concepts | "Teach me about regression discontinuity design" |
| **Office Hours** | Professor-style Q&A | "How should I structure my literature review chapter?" |

**Committee Mode** (checkbox): Sends your question to 3 simulated committee members who each respond from their perspective, then synthesizes their feedback.

**Keyboard shortcuts:** `Cmd+1` thru `Cmd+7` switch panels. `Cmd+N` opens notes drawer.

---

### 📚 Library Panel

**What it is:** Browse all papers organized by course.

**How to use:**
1. Navigate: Research → Library
2. Left sidebar shows courses grouped by category with paper counts
3. Click a course to filter papers
4. Search bar at top filters by title
5. Click a paper card to expand it — reveals:
   - **Paper Dive** button → deep analysis
   - **Citations** button → citation network
   - **Ask Winnie** button → discuss with Winnie

---

### 🔍 Search Panel

**What it is:** Multi-source academic search.

**How to use:**
1. Navigate: Research → Search
2. Type a query and hit Enter
3. Choose a source tab:

| Tab | Source | What It Searches |
|---|---|---|
| **All** | Everything | Combined results |
| **Academic** | OpenAlex | 100M+ academic papers worldwide |
| **My Library** | ChromaDB | Your indexed papers |
| **Legislation** | LegiScan | US legislation (needs API key) |
| **News** | NYT | New York Times articles (needs API key) |

4. Click **Import** on any result to add it to your library

---

### 💻 Vibe Coder Panel

**What it is:** Generate statistical code from plain English.

**How to use:**
1. Navigate: Tools → Vibe Coder
2. Pick a language: **Stata**, **R**, or **Python**
3. Describe what you want:
   - "Run a difference-in-differences with state and year fixed effects on voter_turnout"
   - "Create a scatter plot of GDP vs democracy scores with a LOESS line"
4. Click **Generate**
5. Review the code — then use the buttons:

| Button | What It Does |
|---|---|
| **Copy** | Copy code to clipboard |
| **Save** | Download as `.do` / `.R` / `.py` file |
| **Explain** | Get a line-by-line explanation |
| **Export LaTeX** | Convert output tables to LaTeX |

**Datasets:** Drop `.dta`, `.csv`, `.xlsx` files into `/Volumes/YOUR_DRIVE/Library/Datasets/` — they appear automatically in the dataset selector.

**Model used:** GPT-4.1 (if OpenAI key set) → Gemini fallback.

---

### 🔬 Methods Lab Panel

**What it is:** Learn causal inference methods + audit paper methodologies.

**How to use — Learn Mode:**
1. Navigate: Tools → Methods Lab
2. Click one of the 8 method cards:

| Card | Method |
|---|---|
| DID | Difference-in-Differences |
| IV | Instrumental Variables |
| RDD | Regression Discontinuity |
| Synth | Synthetic Control |
| FE | Fixed Effects |
| PSM | Propensity Score Matching |
| Event | Event Study |
| OLS | Ordinary Least Squares |

3. Ask a question about that method
4. Use buttons on the response:
   - **Make Flashcards** → Generate Anki-style study cards
   - **Translate to R/Stata** → Get code implementing the method

**How to use — Paper Audit Mode:**
1. Toggle to "Analyze a Paper"
2. Paste paper text
3. Get a forensic methodology audit: what method they used, whether assumptions hold, what's missing

---

### 📖 Paper Dive Panel

**What it is:** Deep analysis of any paper.

**How to use:**
1. Navigate: Research → Paper Dive
2. Type a paper title or topic
3. Click **Deep Dive** or press Enter
4. E.D.I.T.H. generates a structured analysis:
   - Summary
   - Methodology breakdown
   - Key findings
   - Contributions to the field
   - Limitations
   - Related work
5. **Peer Review** button → 3 simulated reviewers critique the paper
6. Ask follow-up questions in the input box

---

### 📑 Citations Panel

**What it is:** Citation network explorer.

**How to use:**
1. Navigate: Tools → Citations
2. Search for any paper by title
3. Results show citation counts and metadata
4. Click a result to see tabs:

| Tab | What It Shows |
|---|---|
| **Works Cited** | What this paper cites |
| **Cited By** | Who cites this paper |
| **Connected** | Related papers (co-citation network) |
| **Suggestions** | Recommended reads based on this paper |

5. **Export BibTeX** → download citation in `.bib` format for LaTeX

---

## Part 6: Configuration Reference

All optional — the defaults work out of the box with just `GOOGLE_API_KEY`.

| Variable | Default | What It Does |
|---|---|---|
| `EDITH_MODEL` | `gemini-2.5-flash` | Primary LLM |
| `EDITH_ORACLE_MODEL` | `gemini-2.5-pro` | Model for deep analysis |
| `EDITH_CODE_MODEL` | `gpt-4.1` | Model for code generation |
| `EDITH_M2_BACKEND_PORT` | `8003` | Backend port |
| `EDITH_M2_FRONTEND_PORT` | `5176` | Frontend port |
| `EDITH_DATA_ROOT` | Drive root | Root for papers/data |
| `EDITH_CHROMA_DIR` | `Edith_M4/ChromaDB` | Vector DB location |
| `EDITH_CHROMA_COLLECTION` | `edith_docs_pdf` | Collection name |
| `EDITH_GEN_TIMEOUT` | `120` | LLM response timeout (seconds) |

---

## Part 7: Fine-Tuning Your Own Winnie Model

E.D.I.T.H. supports using a **fine-tuned OpenAI model** as Winnie's brain. This makes responses more personalized to your research style, vocabulary, and field.

### What Fine-Tuning Does

Instead of using a generic GPT-4 model, you can train a custom model on:
- Your own Q&A pairs (how you'd answer research questions)
- Your writing style and academic voice
- Your field's terminology and conventions
- Your preferred citation and explanation patterns

### How to Fine-Tune

1. **Prepare training data** — create a `.jsonl` file with prompt/completion pairs:
```json
{"messages": [{"role": "system", "content": "You are Winnie, a research assistant."}, {"role": "user", "content": "What is the principal-agent problem?"}, {"role": "assistant", "content": "The principal-agent problem arises when..."}]}
{"messages": [{"role": "system", "content": "You are Winnie, a research assistant."}, {"role": "user", "content": "Explain difference-in-differences."}, {"role": "assistant", "content": "Difference-in-differences (DID) is a causal inference method..."}]}
```

2. **Upload and train** via OpenAI:
```bash
# Upload training file
openai api files.create -f my_training_data.jsonl -p fine-tune

# Start fine-tuning job (on GPT-4o)
openai api fine_tuning.jobs.create \
  -m gpt-4o-2024-08-06 \
  -t file-abc123

# Check status
openai api fine_tuning.jobs.list
```

3. **Use your model** — when training completes, you get a model ID like:
```
ft:gpt-4o-2024-08-06:personal:winnie-v1:XXXXXXXX
```

4. **Add to `.env`:**
```
OPENAI_FT_MODEL=ft:gpt-4o-2024-08-06:personal:winnie-v1:XXXXXXXX
```

E.D.I.T.H. will automatically use your fine-tuned model for Winnie's responses when this is set.

### Tips for Good Training Data

- **50-100 examples minimum** for noticeable improvement
- Include a mix of: short answers, long lit reviews, methodology explanations
- Match the tone and depth you want in responses
- Include examples from different subfields/courses
- Quality over quantity — 50 great examples beat 500 mediocre ones

### Cost

Fine-tuning GPT-4o costs ~$25 per million training tokens. A typical 100-example dataset runs about $3-5 to train.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| "Drive not found" | Plug in your external SSD |
| "Offline" status | Backend didn't start — check Terminal for Python errors |
| No papers in Library | Course folders must match `folder` in `courses.json` |
| Winnie gives empty answers | Check `GOOGLE_API_KEY` in `.env` |
| Vibe Coder outputs bad code | Add `OPENAI_API_KEY` for GPT-4.1 |
| Port already in use | `lsof -ti:8003 | xargs kill` then relaunch |
| ChromaDB not found | Run `mkdir -p /Volumes/YOUR_DRIVE/Edith_M4/ChromaDB` |
| Slow first response | Normal — ChromaDB loads on first query (~5s) |
