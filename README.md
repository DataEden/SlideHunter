
# SlideHunter — Lecture Navigator MVP (Multi‑Modal RAG)

<p align="center">
  <img src="images/SlideHunter_Logo.png" alt="SlideHunter-App Flow Diagram", width="70%">
  <br/>
  <em>Ingestion → Retrieval (FAISS + BM25) → Routing/Rerank → Streamlit UI with citations</em>
</p>

> Lightning-fast answers with pinpoint slide/page citations, powered by modern ML retrieval (FAISS + BM25 + reranker) and concise GPT-4o-mini summarization.  

---

## ✨ What it does (MVP)

- Ingests **Canvas Pages** + **PDF slides** → 400–600 char chunks with rich metadata.
- Builds a **single FAISS store** (`data/faiss/canvas.index` + `facts.json`) using `sentence-transformers/all-MiniLM-L6-v2`.
- **Hybrid retrieval**: FAISS dense vectors + BM25 over titles/snippets; optional **cross‑encoder reranker** for the top‑50.
- **Auto‑router**: technical ↔ career (short route descriptions, margin threshold).
- **Type‑aware boosts**: `Page`/`File` ≻ `Assignment`/`Quiz`; **low‑score refusal** to avoid weak citations.
- **Phase date facts** parsed from module names (e.g., `P2W1 (6/9–6/13)` ⇒ “Phase 2 begins June 9”)
  
---

<p align="center">
  <img src="images/SlideHunter_App_Flow_Diagram.png" alt="SlideHunter-App Flow Diagram", width="70%">
  <br/>
  <em>Find exactly where a concept lives in course slides and notes. Lightning-fast answers with pinpoint slide/page citations, powered by modern ML retrieval (FAISS + BM25 + reranker), concise GPT-4o-mini summarization with google/flan-t5-base model as fallback </em>
  </p>

---

## 🙌 The Team

```

Mina Grullon, Fari Lindo, Thalyann Olivo, Jahaira Zhagnay

```

---

## 🗂️ Repo Structure

```
SLIDEHUNTER/
├─ app/
│  └─ app.py                     # Streamlit frontend
├─ data/
│  ├─ slides/                    # PDFs / source content
│  ├─ index/                     # (legacy Chroma if you keep it)
│  └─ faiss/
│     ├─ canvas.index            # FAISS index (persisted)
│     └─ facts.json              # parallel facts + metadata
├─ images/
│  └─ image\SlideHunter_App_Flow_Diagram.png 
├─ notebooks/
│  ├─ 01_setup_and_ingest.ipynb  # builds data/faiss/*
│  ├─ 02_query_demo.ipynb        # quick search & inspect
│  └─ 03_eval.ipynb              # test prompts & metrics
├─ prompts/
│  └─ answer_from_context.txt
├─ requirements.txt
├─ .env                          # SLIDEHUNT_BASE=.; (optional keys)
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

2) **Put slides** in `data/slides/` (or run Canvas ingestion below).

3) **Build the index** (Notebook `01_setup_and_ingest.ipynb`) → writes:
```
data/faiss/canvas.index
data/faiss/facts.json
```

4) **Run Streamlit**
```bash
streamlit run app/app.py
```

> **Windows note:** If `pip install faiss-cpu` fails, use Conda (`conda install -c pytorch faiss-cpu`) or run the notebooks; keep Chroma as a temporary fallback if needed.

---

## 🔐 Environment

Create **`.env`** in repo root:

```dotenv
SLIDEHUNT_BASE=.
# Optional (only if you add generation)
# OPENAI_API_KEY=sk-...
# Canvas access (for ingestion script/notebook)
CANVAS_BASE_URL=https://<your-subdomain>.instructure.com
CANVAS_TOKEN=<your_personal_access_token>
```

Load with `python-dotenv` in notebooks/apps or rely on Streamlit environment.

---

## 🎓 Accessing Canvas & Parsing Courses/Modules

### Create a Canvas token
- Log into Canvas → **Account → Settings → + New Access Token**.
- Copy the token; store it in `.env` as `CANVAS_TOKEN`.

### Install and connect
```bash
pip install canvasapi beautifulsoup4
```

```python
import os, re
from canvasapi import Canvas
from bs4 import BeautifulSoup

BASE_URL = os.getenv("CANVAS_BASE_URL")
TOKEN    = os.getenv("CANVAS_TOKEN")
canvas   = Canvas(BASE_URL, TOKEN)
```

---

## 🧠 Build Embeddings & FAISS Store

```bash
pip install sentence-transformers faiss-cpu rank-bm25
```

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

Run `03_eval.ipynb` to export `outputs/eval_prompts.csv`.

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
- `File` (PDF/PPTX) → **slides** ✅  
- `Assignment`, `Quiz`, `Discussion` → links/context; **down-weight** for concept queries  
- `ExternalUrl`, `ExternalTool` → store link/cite only

---

### Practical notes

- **Pagination:** `canvasapi` iterables auto-paginate; just loop.  
- **Rate limits:** be gentle; cache `Page.body` to disk for re-runs.  
- **Phase dates:** module names often include ranges like `P2W1 (6/9–6/13)`—parse once into a `phase_start` map.  
- **Security:** never commit tokens; keep `CANVAS_BASE_URL` and `CANVAS_TOKEN` in `.env`.

---

## 🤝 Team Collaboration

**Roles (template):** PM & UX · ETL & chunking · Retrieval & scoring · Reranker & QA · DevOps & CI

**Working agreement:**
- `main` protected; feature branches → PRs (small, focused).
- Don’t commit slides or `.env`.
- Issues labeled by area: `etl`, `retrieval`, `frontend`, `eval`, `infra`.

---

## 🔬 Findings, Approach, Setbacks & Resolutions

**Approach:** single FAISS store; BM25 hybrid; type‑aware boosts; low‑score refusal; simple router; phase date extractor.

**Findings:** hybrid helps named tokens (e.g., `P2W2`, `pivot tables`); light boosts remove many quiz/assignment mis‑hits; reranker helps edge cases.

**Setbacks:** sparse PDFs → PyMuPDF + optional OCR; assignment citations on concept queries → boosts + rerank; routing ambiguity → margin to “all”.

**Resolutions/Results:** cleaner top‑1 slide citations; stable latency; reproducible builds via persisted index.

---

## 🧭 Streamlit Demo

- **Scope:** auto/technical/career/all  
- Toggles: **BM25 hybrid**, **reranker**, **low‑score refusal**  
- Special handling: **Phase 2 begin date**

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

## 📜 License

!MIT [(see `LICENSE`)](\LICENSE)
