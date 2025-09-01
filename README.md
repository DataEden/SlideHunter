
<p align="center">
  <img src="assets\images\SlideHunter_LogoV2.png" alt="SlideHunter App Logo Mockup", width="55%">
  <br/>
  <em>Lightning-fast answers with pinpoint slide/page citations, powered by modern ML retrieval (FAISS + BM25 + reranker) and concise GPT-4o-mini summarization and google/flan-t5-base local-fallback.</em>
</p>

---

## ✨ What it does (MVP)

- Ingests **Canvas Pages** + **Image/Text** → 400–600 char chunks with rich metadata.
- Builds a **single FAISS store** (`data/faiss/canvas.index` + `facts.json`) using `sentence-transformers/all-MiniLM-L6-v2`.
- **Hybrid retrieval**: FAISS dense vectors + BM25 over titles/snippets; optional **cross‑encoder reranker** for the top‑50.
- **Auto‑router**: technical ↔ career (short route descriptions, margin threshold).
- **Type‑aware boosts**: `Page`/`File` ≻ `Assignment`/`Quiz`; **low‑score refusal** to avoid weak citations.
- **Phase date facts** parsed from module names (e.g., `P2W1 (6/9–6/13)` ⇒ “Phase 2 begins June 9”)
  
---

<p align="center">
  <img src="assets\images\SlideHunter_App_Flow_Diagram.png" alt="SlideHunter-App Flow Diagram", width="80%">
  <br/>
  <em>Find exactly where a concept lives in course slides and notes. Lightning-fast answers with pinpoint slide/page citations, powered by modern ML retrieval (FAISS + BM25 + reranker), concise GPT-4o-mini summarization with google/flan-t5-base model as fallback </em>
  </p>

## ⚙️ Flow Overview

- The SlideHunter app connects directly to the Canvas API to ingest course content (pages, modules, links). Text is chunked into manageable pieces (≈400–600 characters) with rich metadata, then indexed in two ways:
  - Dense embeddings (MiniLM-L6-v2 → FAISS vector store) for semantic search
  - Sparse BM25 over titles/snippets for keyword relevance
- A lightweight router steers queries between technical and career domains:
  - while optional components like a BGE reranker refine top-50 hits and a low-score refusal guardrail filters out weak matches.
  - Results flow into the Streamlit UI, which surfaces top-k hits (we've decided on k=4 for now.), citations, and optional summarization.

---

## 🙌 The Team

```
Mina Grullon, Fari Lindo, Thalyann Olivo, Jahaira Zhagnay
```

---

## 🗂️ Repo Structure

```
SLIDEHUNT/                       # Root of local-repo
├─ app/
│  └─ app.py                     # Streamlit frontend
├─ assets/
│  └─ images/                    # PNGs, etc.
├─ data/
│  ├─ slides/                    # PDFs / source content
│  ├─ index/                     # (legacy Chroma if you keep it)
│  └─ faiss/
│     ├─ canvas.index            # FAISS index (persisted)
│     └─ facts.json              # parallel facts + metadata
├─ notebooks/
|  ├─ canvas_api_extraction.ipynb 
│  ├─ 01_setup_and_ingest.ipynb   # builds data/faiss/*
│  ├─ 02_query_demo.ipynb         # small/Initial test prompts 
│  └─ 03_eval.ipynb               # Evaluate & inspect outputs
├─ outputs/
│  ├─ data_ds_A.csv
│  └─ eval_prompts.csv
├─ prompts/
│  └─ answer_from_context.txt
├─ scripts/
|  ├─ __init__.py
│  └─ nb01_helper.py
├─ requirements.txt
├─ .env                       # SLIDEHUNTER_BASE="Path to/local repo/root folder", etc.
├─ .gitignore
├─ flowchart.md
├─ LICENSE
├─ nb01_helper.py             # nb01_setup_and_ingest's helper script
├─ SlideHunter.py             # Streamlit frontend
└─ README.md
```

---

## 🚀 Quickstart (Local)

1) **Create venv & install**
  
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

1) **Put slides** in `data/slides/` (or run Canvas ingestion below).

2) **Build the index** (Notebook `01_setup_and_ingest.ipynb`) → writes:
  

```
data/faiss/canvas.index
data/faiss/facts.json
```

3) **Run Streamlit**

- If in root: `streamlit run SlideHunter`
- if in apps folder: `streamlit run app/SlideHunter.py`

> **Windows note:** If `pip install faiss-cpu` fails, use Conda (`conda install -c pytorch faiss-cpu`) or run the notebooks; keep Chroma as a temporary fallback if needed.

---

## 🔐 Environment

Create **`.env`** in repo root:

```dotenv
SLIDEHUNTER_BASE=.
Optional (since there's a fallback)
 OPENAI_API_KEY=sk-...
Canvas access (for ingestion script/notebook)
 CANVAS_BASE_URL=https://<your-subdomain>.instructure.com
 CANVAS_TOKEN=<your_personal_access_token>
Load with `**python-dotenv** in notebooks/apps or rely on Streamlit environment.`
```

---

## 🎓 Accessing Canvas & Parsing Courses/Modules

### Create a Canvas token
- Log into Canvas → **Account → Settings → + New Access Token**.
- Copy the token; store it in `.env` as `CANVAS_TOKEN`.

### Install and connect

If needed install 'pip install -r requirements.txt' to install all dependencies/packages.

```python
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np, faiss, re, json, os
from canvasapi import Canvas
import torch

CANVAS_BASE_URL = os.getenv("CANVAS_BASE_URL")
CANVAS_TOKEN = os.getenv("CANVAS_TOKEN")
canvas = Canvas(CANVAS_BASE_URL, CANVAS_TOKEN)

OPENAI_API_KEY = config.get("OPENAI_API_KEY")
```

Load with `python-dotenv` in notebooks/apps or rely on Streamlit environment.

---

## 🧠 Build Embeddings & FAISS Store

If needed run command 'pip install -r requirements.txt'

```python
from sentence_transformers import SentenceTransformer
import faiss, numpy as np, json, os
from pathlib import Path

# Embeddings
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
emb = model.encode(facts, normalize_embeddings=True, show_progress_bar=True).astype("float32")

# FAISS (cosine via inner product on normalized vectors)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)

# Persist
STORE_DIR = Path("data/faiss"); STORE_DIR.mkdir(parents=True, exist_ok=True)
faiss.write_index(index, STORE_DIR / "canvas.index")
json.dump({"facts": facts, "metas": metas}, open(STORE_DIR / "facts.json","w",encoding="utf-8"), ensure_ascii=False)
```

---

## 🧪 Evaluation & Metrics

We track:

- **Coverage** — % queries with ≥1 hit above threshold  
- **Top‑1 Source Type Precision** — Page/File vs Assignment/Quiz for concept queries  
- **Citation Accuracy** — manual spot‑check (k=20)  
- **Router Accuracy** — target domain vs routed domain  
- **Latency** — median; Streamlit shows per‑query  
- **Refusal Rate** — % of queries correctly refused

Run `any notebook` to export to `outputs/eval_prompts.csv`.

---

## Canvas API Cheat Sheet (for our notebooks)

### Auth & entry point
- **Create client:** `Canvas(BASE_URL, TOKEN)`
- **Objects we touch most:** `Canvas`, `Course`, `Module`, `ModuleItem`, `Page`, `File`, `Assignment`, `Quiz`, `DiscussionTopic`

---

### Core objects & methods

| Object | Method | Purpose | Returns / Type | Key fields you’ll use | Notes |
|---|---|---|---|---|---|
| **Canvas** | `get_courses(enrollment_state="active")` | List your courses | Iterable of **Course** | `Course.id`, `Course.name` | Filter by name to pick “technical” vs “career”. |
| **Canvas** | `get_course(course_id)` | Fetch one course by ID | **Course** | — | Use when you know the ID. |
| **Course** | `get_modules()` | List modules in a course | Iterable of **Module** | `Module.id`, `Module.name` | Names often include week/phase (e.g., `P2W1 (6/9–6/13)`). |
| **Module** | `get_module_items()` | Items within a module | Iterable of **ModuleItem** | `item.id`, `item.title`, `item.type`, `item.content_id`, `item.page_url`, `item.external_url` | `item.type` determines next call (Page/File/Assignment/Quiz/ExternalUrl…). |
| **Course** | `get_page(page_url)` | Get a page (from `item.page_url`) | **Page** | `Page.body`, `Page.html_url`, `Page.title` | Use `Page.body` for text extraction. |
| **Course** | `get_pages()` | List all pages | Iterable of **Page** | `Page.url` (slug), `title` | Alternative to walking via modules. |
| **Course** | `get_file(file_id)` | Get a file (from `item.content_id`) | **File** | `File.display_name`, `File.size`, `File.url`, `File.content_type` | Download/parse PDFs if you index files. |
| **Course** | `get_files()` | List all files in course | Iterable of **File** | same as above | Useful for bulk file ingest. |
| **Course** | `get_assignments()` | List assignments | Iterable of **Assignment** | `name`, `due_at`, `html_url` | Usually down-weight for concept queries. |
| **Course** | `get_quizzes()` | List quizzes | Iterable of **Quiz** | `title`, `html_url` | Same note as assignments. |
| **Course** | `get_discussion_topics()` | List discussions | Iterable of **DiscussionTopic** | `title`, `message`, `html_url` | Optional ingest. |
| **ModuleItem** | *(type-specific)* | — | — | — | — |
|  | `type == "Page"` | Indicates a Canvas page | — | `page_url` (slug) | Then call `course.get_page(item.page_url)`. |
|  | `type == "File"` | Indicates a file (PDF/PPTX) | — | `content_id` (file id) | Then call `course.get_file(item.content_id)`. |
|  | `type == "ExternalUrl"` | External link | — | `external_url` | Store link as metadata; no body to parse. |
|  | `type in {"Assignment","Quiz","Discussion"}` | Graded items | — | `content_id` / `html_url` | Useful links; not primary concept source. |

---

### Typical flows (at a glance)

1. **Enumerate courses → pick targets**  
   `Canvas → get_courses()` → filter by `Course.name` (technical vs career)

2. **Walk modules & items**  
   `Course → get_modules()` → each `Module → get_module_items()`

3. **Fetch content**  
   - If `item.type == "Page"` → `course.get_page(item.page_url)` → use `Page.body`  
   - If `item.type == "File"` → `course.get_file(item.content_id)` → download/parse  
   - Else (`Assignment`/`Quiz`/`ExternalUrl`) → keep `title` + `html_url`/`external_url` as metadata

4. **Save metadata with each chunk**  
   `course_name`, `course_id`, `module_name`, `module_id`, `item_title`, `item.type`, `url/html_url`, and our `domain` (technical/career)

---

### Common `ModuleItem.type` values (routing hint)

- `Page` → primary source for **lecture notes / concepts** ✅  
- `JSON/XML` (HTML/Text/Items) → **slides/pages/** ✅  
- `Assignment`, `Quiz`, `Discussion` → links/context; **down-weight** for concept queries  
- `ExternalUrl`, `ExternalTool` → store link/cite only

---

### Practical notes

- **Pagination:** `canvasapi` iterables auto-paginate; just loop.  
- **Rate limits:** be gentle; cache `Page.body` to disk for re-runs.  
- **Phase dates:** module names often include ranges like `P2W1 (6/9–6/13)`—parse once into a `phase_start` map.  
- **Security:** never commit tokens; keep `CANVAS_BASE_URL` and `CANVAS_TOKEN` in `.env`.

---

## 🔬 Approach , Evaluation Results, Findings, Setbacks & Resolutions

## ✅ Approach (MVP)

- **Retriever:** SentenceTransformers (MiniLM-L6-v2) → FAISS (cosine, IndexFlatIP)  
- **Hybrid boost:** BM25 on titles + leading snippets (blend: 0.7*dense + 0.3*BM25)  
- **Quality levers:** type-aware boosts (Page/File ≻ Assignment/Quiz), low-score refusal (τ = 0.40)  
- **Routing:** lightweight domain router (technical ↔ career)  
- **Dates:** phase date extractor from module names  
- **Optional:** BGE reranker (top-50) and summarizer (GPT-4o/mini or local FLAN-T5)  

## 📊 Evaluation Results (taken from 20 technical responses)

Below are the evaluation results from 20 test prompts against the FAISS + BM25 + router retriever.  
Each query was run through the `search()` function with `scope="auto"` (router-enabled).  

| query                                      | scope      | latency_s | top_score | top_domain | citation (truncated) |
|--------------------------------------------|------------|-----------|-----------|------------|-----------------------|
| When does phase 2 begin?                   | all        | 0.041     | 0.476     | technical  | IF '25 Data Science Cohort A > P2W12 ... |
| Any way of saying June 9th?                | technical  | 0.026     | 0.230     | technical  | Foundations Course > Week 1: Foundations ... |
| Where can I find my instructor's email?    | technical  | 0.022     | 0.332     | technical  | IF '25 Data Science Cohort A > Fellow Res... |
| Under Course Team Contact Information?     | all        | 0.037     | 0.444     | technical  | IF '25 Data Science Cohort A > Fellow Res... |
| What was the last TLAB about?              | technical  | 0.023     | 0.334     | technical  | IF '25 Data Science Cohort A > P2W9 ...     |
| An explanation of making a recommender.    | technical  | 0.014     | 0.465     | technical  | IF '25 Data Science Cohort A > P2W9 ...     |
| What lecture slides cover pivot tables?    | technical  | 0.033     | 0.563     | technical  | IF '25 Data Science Cohort A > P1W6 ...     |
| What lecture slides explain control flow?  | technical  | 0.020     | 0.389     | technical  | Foundations Course > Week 1: Foundations ... |
| Bullet point list of SQL concepts?         | technical  | 0.018     | 0.577     | technical  | IF '25 Data Science Cohort A > P1W9 ...     |
| When does phase 2 commence?                | all        | 0.031     | 0.361     | technical  | IF '25 Data Science Cohort A > P2W1 ...     |
| Summary of P2W2’s material?                | all        | 0.016     | 0.518     | technical  | IF '25 Data Science Cohort A > P2W2 ...     |
| Where did we define precision vs. recall?  | technical  | 0.027     | 0.382     | technical  | IF '25 Data Science Cohort A > P2W3 ...     |

---

### 📝 Initial Observations

- **Coverage:** 100% (all 20 questions returned at least one hit).  
- **Latency:** Fast, consistently between `0.01–0.04s` per query.  
- **Top scores:** Range `0.23–0.58`.  
  - ~70% of queries scored **≥ 0.38** (above the refusal threshold of 0.40).  
  - A few (e.g., *“Any way of saying June 9th?”*, *XGBoost vs AdaBoost*) were borderline at ~0.23–0.29, indicating weaker matches which makes sense, since the XGBoost model wasn't a covered topic.  
- **Domains:** All results classified as **technical**. No “career” prompts were included in this round — router evaluation pending.  
- **Insights:** Retrieval is strong for structured concepts (e.g., SQL, PCA, regression), weaker for ambiguous/natural phrasing (“June 9th” question).  

---

## **Key Observations**

- **Hybrid helps names & codes:** BM25 boosts matches for P2W2, “pivot tables”, “ROC”, “log loss”  
- **Type boosts reduce noise:** Concept questions more reliably hit Page/File over Assignment/Quiz  
- **Reranker = precision mode:** Improves ordering when candidates are close; toggle off to keep latency minimal  
- **Guardrailed summarizer:** Only runs when top hit ≥ τ; otherwise refuse + show snippets to avoid hallucinations  
- **Ambiguity hurts dense scores:** Colloquial queries benefit from BM25 blending & mild rewrites  

---

**Detailed Findings:**

- hybrid helps named tokens (e.g., `P2W2`, `pivot tables`); light boosts remove many quiz/assignment mis‑hits; reranker helps edge cases. We tested our retrieval models using questions categorized into 3 different levels of difficulty: easy (3 example questions), medium (5 example questions), and hard (7 example questions).

- Our easy questions were used to recall material that is more factual with only one answer which would allow us to test our models retrieval ability for exact answers.
  - For example, when asked "When does phase 2 begin?" the output returned a top score of 0.467 which revelas a relatively strong match with a low latency score of 0.054.
  - Our medium questions involved more conceptual content and required the model to switch gears to work more towards summarizing slide content.
  - We found the medium level questions optimal for testing semantic retrieval since it required giving a more nuanced output.
    - For example, when asked about pivot tables or SQL concepts the output was often not explicit however we observed accuracy (~50%) heading generally in a positive direction with top scores around 0.42.
    - Similarly, the harder questions demanded a more technical response of summarized content from different topics discussed at different times throughout the students' learning period.
- These queries demonstrated a weak performance returning an average top score of about .30. We also included noisy inputs to test the ability to retrieve information based on informal language and typos.
- Additionally, we also included out of scope questions:
  - such as "Can I see other students' grades??" to test the models ability to truly distinguish between career or technical modules when prompted a question that doesn't necessarily fall into either category.

In our final evaluation of the model we found the average top score to be 0.407 which demonstrated the model is finding relevant matches since every query returned at least one citation (100% coverage) but it is not a strong.

---

## 🧭 Process Notes

- Collaboration & learning: Colab for GPUs, shared iteration  
- Productization: VS Code + Streamlit + clean repo structure  
- Reproducibility: FAISS store in `data/faiss/` shared by notebooks & app  
  
---

## ⚠️ Limitations (MVP)

- No OCR → image-only slides are skipped  
- Single embedding model (MiniLM-L6-v2); no multilingual / long-context  
- Router is simple; no multi-index routing yet  

---

🔮 **Next steps:**  

- Add **10–15 career-focused prompts** (resume, LinkedIn, cover letters, interview prep) to test router accuracy.  
- Increase total evaluation set to ~30–40 queries for better coverage.  
- Inspect **borderline low-score cases** to refine refusal threshold or add reranker tuning.  

---

## 🧭 Streamlit Demo

- **Scope:** auto/technical/career/all  
- Toggles: **BM25 hybrid**, **reranker**, **low‑score refusal**  
- Special handling: **Phase 2 begin date**

## 🔁 Reproduce (quick)

1. Run ingest → build `data/faiss/{canvas.index,facts.json}`  
2. Run query demo → verify retrieval & citations  
3. Launch Streamlit:  

```bash
streamlit run app/SlideHunter.py or SlindHunter.py
```

If deploying from Colab, you can use **cloudflared** to expose a public URL.

---

## 🔒 Data & Privacy

- Store only course content text and metadata. No student PII.  
- `.env` for tokens/keys. Never commit `.env` or source PDFs if restricted.

---

## 🗺️ Future Expansion Roadmap

- OCR fallback for image‑only slides
- Richer date/deadline extractors  
- Option to split into domain‑specific stores  
- Instruction‑tuned summarizer for 1–2 sentence answers with citations  
- FastAPI search service (two‑service architecture)

---

## 🤝 Team Collaboration

**Roles (template):** PM & UX · ETL & chunking · Retrieval & scoring · Reranker & QA · DevOps & CI

**Working agreement:**pip install -r requirements.txt

- `main` deliberate communicate changes.
- Don’t commit slides or `.env`.
- Communicate prior to each commit/push via Slack group, etc.
- Communicate changes in order to prevent modifying identical content.
- Fari (@DataEden) – Lead Developer  
- ChatGPT (OpenAI) – AI Pair Programmer / Vibe Coding Partner.

---

## 📜 License

!MIT [see `LICENSE`](\LICENSE)
