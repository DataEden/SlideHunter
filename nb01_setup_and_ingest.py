# Imports
from dotenv import dotenv_values
import json, os
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import numpy as np, faiss, re, json, os
from canvasapi import Canvas
import torch

# Load environment variables from .env file
config = dotenv_values() # load .env file

# Injest and process data from Canvas Sections
# Set up Canva API client
CANVAS_BASE_URL = config.get("CANVAS_BASE_URL")
CANVAS_TOKEN=config.get("CANVAS_TOKEN")
OPENAI_API_KEY = config.get("OPENAI_API_KEY")

# Initialize Canvas API client 
canvas = Canvas(CANVAS_BASE_URL, CANVAS_TOKEN)

# Getting the list of courses
my_courses = canvas.get_courses()

# Pulling courses from Canvas
my_courses = canvas.get_courses()
course_list = []

for course in my_courses:
    print(course.name)
    course_list.append(course)

# Pulling modules from courses on Canvas 
modules = course.get_modules()

for module in modules:
    print(f"  Module_id: {module.id}")
    print(f"  Module: {module.name}")
    module_items = module.get_module_items()
    for item in module_items:
        print(f" - Item: {item.title} ({item.type})")
    
# Embedding Tokenized Canvas modules (Texts/items).
# Then Turning Those Embddings into a facts list + FAISS index that we can query.

# Multi-course to ONE FAISS store + simple router using career and technical courses

# 0) CONFIG: map course names to domain buckets
DOMAINS = {
    "technical": [
        "Foundations '25 Data Science",
        "Foundations Course",
        "IF '25 Data Science Cohort A",
    ],
    "career": [
        "IF '25 NY Career Readiness and Success",
    ],
}
# Short route descriptions--We can add more if needed (used for auto routing purpos)
ROUTE_DESC = {
    "technical": "Technical class content: Python, SQL, statistics, machine learning, slides, labs, code, algorithms, data science, lecture notes.",
    "career":    "Career readiness content: resumes, cover letters, job search, interviews, career prep, LinkedIn, networking, internship resources.",
}

# 1) Utility: HTML → text, light chunking
def strip_html(html: str) -> str:
    if not html: return ""
    txt = " ".join(BeautifulSoup(html, "html.parser").stripped_strings)
    return re.sub(r"\s+", " ", txt).strip()

def chunk_text(text, max_chars=600):
    if not text: return []
    parts = re.split(r"(\n|\.\s+)", text)
    buf, chunks = "", []
    for p in parts:
        buf += p
        if len(buf) >= max_chars:
            chunks.append(buf.strip()); buf = ""
    if buf.strip(): chunks.append(buf.strip())
    return [c for c in chunks if c]

# 2) Select courses by name (use your Canvas client `canvas`)
def course_domain(course_name: str):
    for dom, names in DOMAINS.items():
        if any(course_name.startswith(n) for n in names):
            return dom
    return "other"

wanted_prefixes = sum(DOMAINS.values(), [])
all_courses = [c for c in canvas.get_courses(enrollment_state="active")
               if any(c.name.startswith(p) for p in wanted_prefixes)]

print("Selected courses:", [c.name for c in all_courses])

# 3) Build facts + metas from ALL selected courses
facts, metas = [], []
for crs in all_courses:
    dom = course_domain(crs.name)
    for module in crs.get_modules():
        for item in module.get_module_items():
            t = (item.type or "").strip()
            if t == "Page":
                page = crs.get_page(item.page_url)
                text = strip_html(getattr(page, "body", ""))
                for chunk in chunk_text(text, max_chars=600):
                    facts.append(f"[{dom}] {crs.name} > {module.name} > {item.title}: {chunk}")
                    metas.append({
                        "domain": dom,
                        "course_id": crs.id, "course_name": crs.name,
                        "module_id": module.id, "module_name": module.name,
                        "item_title": item.title, "type": "Page",
                        "url": getattr(page, "html_url", None)
                    })
            elif t in ("ExternalUrl", "ExternalTool"):
                facts.append(f"[{dom}] {crs.name} > {module.name} > {item.title}: external link {getattr(item, 'external_url', '')}")
                metas.append({
                    "domain": dom, "course_id": crs.id, "course_name": crs.name,
                    "module_id": module.id, "module_name": module.name,
                    "item_title": item.title, "type": t,
                    "url": getattr(item, "external_url", None)
                })
            elif t == "File":
                facts.append(f"[{dom}] {crs.name} > {module.name} > {item.title} (file)")
                metas.append({
                    "domain": dom, "course_id": crs.id, "course_name": crs.name,
                    "module_id": module.id, "module_name": module.name,
                    "item_title": item.title, "type": "File", "file_id": item.content_id
                })
            elif t == "SubHeader":
                continue
            else:
                facts.append(f"[{dom}] {crs.name} > {module.name} > {item.title} ({t})")
                metas.append({
                    "domain": dom, "course_id": crs.id, "course_name": crs.name,
                    "module_id": module.id, "module_name": module.name,
                    "item_title": item.title, "type": t
                })

print(f"Built {len(facts)} facts")

# 4) Embed — use GPU if available, else CPU
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
print("model device:", model.device)

# (optional) quick warm-up on GPU
if DEVICE == "cuda":
    _ = model.encode(["warm up"], show_progress_bar=False)

# pick a sensible batch size per device
BATCH = 192 if DEVICE == "cuda" else 64

emb = model.encode(
    facts,
    batch_size=BATCH,
    normalize_embeddings=True,   # cosine-ready
    convert_to_numpy=True,       # returns NumPy on CPU for FAISS
    show_progress_bar=True
).astype("float32")

d = emb.shape[1]
index = faiss.IndexFlatIP(d)               # cosine (vectors normalized)
index.add(emb)
print("FAISS ntotal:", index.ntotal)

# 5) Router: choose technical / career / all based on similarity to route descriptions
route_emb = {k: model.encode([v], normalize_embeddings=True).astype("float32") for k,v in ROUTE_DESC.items()}

def choose_scope(query, margin=0.05):
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    sims = {k: float((q @ route_emb[k].T)[0,0]) for k in ROUTE_DESC}
    best = max(sims, key=sims.get)
    # if not clearly better, use 'all'
    ordered = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    if ordered[0][1] - ordered[1][1] < margin:
        return "all", sims
    return best, sims

# 6) Search with optional scope filter (auto by default)
def search(query, k=5, scope="auto"):
    if scope == "auto":
        scope, sims = choose_scope(query)
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    # pull more then filter by domain
    D, I = index.search(q, k*8)
    hits = []
    for score, idx in zip(D[0], I[0]):
        if idx == -1: continue
        m = metas[idx]
        if scope != "all" and m["domain"] != scope:
            continue
        hits.append({"score": float(score), "fact": facts[idx], "meta": m})
        if len(hits) >= k: break
    # if not enough in-scope, backfill with any
    if len(hits) < k:
        for score, idx in zip(D[0], I[0]):
            if idx == -1: continue
            if any(h["meta"] is metas[idx] for h in hits): continue
            hits.append({"score": float(score), "fact": facts[idx], "meta": metas[idx]})
            if len(hits) >= k: break
    return scope, hits

# 7) Try it out with some pre-test test-prompts
tests = [
    "Where did we define precision vs. recall?",
    "tips for a resume and cover letter?",
    "What lecture slides did we learn about control flow?",
  ]
for q in tests:
    scope, hits = search(q, k=4, scope="auto")
    print(f"\nQ: {q}   [scope={scope}]")
    if not hits: print("  (no hits)"); continue
    for h in hits:
        m = h["meta"]
        cite = f"{m['course_name']} > {m['module_name']} > {m['item_title']} ({m['type']})"
        if m.get("url"): cite += f"  [{m['url']}]"
        print(f"  {h['score']:.3f} :: {cite}")

# --- Persist FAISS + metadata to the repo root (SLIDEHUNTER/) ---
from pathlib import Path
import os, json, faiss

# 0) Resolve project base: prefer env var; else step out of notebooks/
ENV_BASE = os.getenv("SLIDEHUNTER_BASE") or os.getenv("SLIDEHUNT_BASE")
if ENV_BASE:
    BASE = Path(ENV_BASE).resolve()
else:
    here = Path.cwd().resolve()
    BASE = here.parent if here.name.lower() == "notebooks" else here  # run from repo root if you're inside notebooks/

# 1) Paths under the repo
STORE_DIR  = BASE / "data" / "faiss"
INDEX_PATH = STORE_DIR / "canvas.index"
FACTS_PATH = STORE_DIR / "facts.json"
STORE_DIR.mkdir(parents=True, exist_ok=True)

# 2) Save / Load helpers
def save_store(index, facts, metas, index_path=INDEX_PATH, facts_path=FACTS_PATH):
    faiss.write_index(index, str(index_path))
    with open(facts_path, "w", encoding="utf-8") as f:
        json.dump({"facts": facts, "metas": metas}, f, ensure_ascii=False)
    print("saved:", index_path)
    print("saved:", facts_path)

def load_store(index_path=INDEX_PATH, facts_path=FACTS_PATH):
    idx = faiss.read_index(str(index_path))
    data = json.load(open(facts_path, "r", encoding="utf-8"))
    print("loaded:", index_path, "and", facts_path)
    return idx, data["facts"], data["metas"]

# 3) Save right after you build `index`, `facts`, `metas`
save_store(index, facts, metas)
