"""
nb01_helper.py — Canvas ingest → facts/metas → embeddings → FAISS index
Safe to import in notebooks; does not execute side-effect code at import.
"""

from pathlib import Path
import os, re, json
from typing import List, Dict, Tuple, Iterable

# --- Light imports at top; heavy ones inside functions to speed up import time
from bs4 import BeautifulSoup

# ---------- Config & small utilities ----------
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

ROUTE_DESC = {
    "technical": "Technical class content: Python, SQL, statistics, machine learning, slides, labs, code, algorithms, data science, lecture notes.",
    "career":    "Career readiness content: resumes, cover letters, job search, interviews, career prep, LinkedIn, networking, internship resources.",
}

def strip_html(html: str) -> str:
    """HTML → text (single-spaced)."""
    if not html:
        return ""
    txt = " ".join(BeautifulSoup(html, "html.parser").stripped_strings)
    return re.sub(r"\s+", " ", txt).strip()

def chunk_text(text: str, max_chars: int = 600) -> List[str]:
    """Chunk by sentence-ish boundaries up to ~max_chars."""
    if not text:
        return []
    parts = re.split(r"(\n|\.\s+)", text)
    buf, chunks = "", []
    for p in parts:
        buf += p
        if len(buf) >= max_chars:
            chunks.append(buf.strip()); buf = ""
    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if c]

def course_domain(course_name: str) -> str:
    """
    Return domain key for course_name based on DOMAINS prefixes.
    Args:
        course_name: Full course name string
    Returns:
        domain key (e.g. "technical", "career", or "other")
    """
    for dom, names in DOMAINS.items():
        if any(course_name.startswith(n) for n in names):
            return dom
    return "other"

# ---------- Canvas access (wrapped) ----------
def get_canvas_client() -> "Canvas":
    """Create a Canvas client from .env (dotenv optional).
       Requires CANVAS_BASE_URL and CANVAS_TOKEN in .env or environment.
    Args:
        None
    Returns: 
        Canvas client"""
    try:
        from dotenv import dotenv_values
        cfg = dotenv_values()
    except Exception:
        cfg = os.environ

    from canvasapi import Canvas
    base = cfg.get("CANVAS_BASE_URL")
    token = cfg.get("CANVAS_TOKEN")
    if not base or not token:
        raise RuntimeError("Missing CANVAS_BASE_URL / CANVAS_TOKEN in .env or environment.")
    return Canvas(base, token)

def select_courses(canvas) -> list:
    """Return active courses filtered by DOMAINS prefixes.
    Args:
        canvas: Canvas client
    Returns: List of Course objects
    """
    wanted_prefixes = sum(DOMAINS.values(), [])
    return [c for c in canvas.get_courses(enrollment_state="active")
            if any(c.name.startswith(p) for p in wanted_prefixes)]

# ---------- Ingest → facts/metas ----------
def build_facts_and_metas(canvas) -> Tuple[List[str], List[Dict]]:
    """
    Traverse selected courses/modules/items and build facts + metas.
    Facts are strings; metas are dicts with context.
    Args:
        canvas: Canvas client
    Returns:
        (facts, metas)
    """
    facts, metas = [], []
    all_courses = select_courses(canvas)

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
                    url = getattr(item, "external_url", "")
                    facts.append(f"[{dom}] {crs.name} > {module.name} > {item.title}: external link {url}")
                    metas.append({
                        "domain": dom, "course_id": crs.id, "course_name": crs.name,
                        "module_id": module.id, "module_name": module.name,
                        "item_title": item.title, "type": t,
                        "url": url
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
    return facts, metas

# ---------- Embedding + FAISS ----------
def embed_facts(facts: List[str], device: str = None, batch_gpu: int = 192, batch_cpu: int = 64):
    """Return (model, embeddings np.array[float32], device_str)."""
    import numpy as np
    import torch
    from sentence_transformers import SentenceTransformer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)

    if device == "cuda":
        _ = model.encode(["warm up"], show_progress_bar=False)

    batch = batch_gpu if device == "cuda" else batch_cpu
    emb = model.encode(
        facts,
        batch_size=batch,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True
    ).astype("float32")
    return model, emb, device

def build_faiss_index(emb):
    """Create inner-product FAISS index for normalized vectors."""
    import faiss
    d = emb.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(emb)
    return index

# ---------- Save / Load ----------
def resolve_store_base() -> Path:
    """Resolve repo BASE for data/faiss. Honors SLIDEHUNTER_BASE else CWD."""
    base_env = os.getenv("SLIDEHUNTER_BASE")
    if base_env:
        return Path(base_env).expanduser().resolve()
    return Path.cwd()

def store_paths(base: Path = None):
    base = base or resolve_store_base()
    store_dir = base / "data" / "faiss"
    index_path = store_dir / "canvas.index"
    facts_path = store_dir / "facts.json"
    return store_dir, index_path, facts_path

def save_store(index, facts, metas, index_path=None, facts_path=None):
    """Persist FAISS index + facts/metas JSON."""
    import faiss
    store_dir, default_index, default_facts = store_paths()
    index_path = Path(index_path or default_index)
    facts_path = Path(facts_path or default_facts)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    with open(facts_path, "w", encoding="utf-8") as f:
        json.dump({"facts": facts, "metas": metas}, f, ensure_ascii=False)
    return index_path, facts_path

def load_store(index_path=None, facts_path=None):
    """Load FAISS index + facts/metas JSON."""
    import faiss
    _, default_index, default_facts = store_paths()
    index_path = Path(index_path or default_index)
    facts_path = Path(facts_path or default_facts)
    idx = faiss.read_index(str(index_path))
    with open(facts_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return idx, data["facts"], data["metas"]

# ---------- Optional: small search helper ----------
def make_router(model) -> dict:
    """Precompute route embeddings for auto scope."""
    import numpy as np
    return {k: model.encode([v], normalize_embeddings=True).astype("float32") for k, v in ROUTE_DESC.items()}

def choose_scope(model, router_emb: dict, query: str, margin: float = 0.05):
    """Return ('technical' | 'career' | 'all', sims_dict)."""
    q = model.encode([query], normalize_embeddings=True).astype("float32")
    sims = {k: float((q @ router_emb[k].T)[0, 0]) for k in ROUTE_DESC}
    ordered = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    if ordered[0][1] - ordered[1][1] < margin:
        return "all", sims
    return ordered[0][0], sims

def search(query: str, model, index, facts, metas, router_emb: dict, k: int = 5, scope: str = "auto", margin: float = 0.05):
    """
    Search the FAISS index for relevant facts based on the query and scope.

    Args:
        query (str): The input query to search for.
        model: A SentenceTransformers model used to encode the query.
        index: A FAISS index built over normalized embeddings (IndexFlatIP).
        facts (List[str]): Fact snippets aligned 1:1 with the index vectors.
        metas (List[dict]): Metadata aligned 1:1 with `facts`.
        router_emb (dict): Precomputed route embeddings from `make_router(model)`.
        k (int): Number of top results to return.
        scope (str): "technical", "career", "all", or "auto" to choose automatically.
        margin (float): Router margin; if best−second < margin, use "all".

    Returns:
        Tuple[str, List[dict]]:
            (resolved_scope, hits) where each hit is:
            {"score": float, "fact": str, "meta": dict}
    """
    import numpy as np

    # Auto-scope via router if requested
    if scope == "auto":
        scope, _ = choose_scope(model, router_emb, query, margin=margin)

    # Encode query (cosine-ready since the index is IP over normalized vectors)
    q = model.encode([query], normalize_embeddings=True).astype("float32")

    # Overfetch to allow domain filtering, then backfill if needed
    D, I = index.search(q, k * 8)

    hits, seen = [], set()

    # Pass 1: keep only items in the chosen domain (unless scope == "all")
    for score, idx in zip(D[0], I[0]):
        if idx == -1:
            continue
        m = metas[idx]
        if scope != "all" and m.get("domain") != scope:
            continue
        if idx in seen:
            continue
        seen.add(idx)
        hits.append({"score": float(score), "fact": facts[idx], "meta": m})
        if len(hits) >= k:
            break

    # Pass 2: backfill with any remaining high-score items regardless of domain
    if len(hits) < k:
        for score, idx in zip(D[0], I[0]):
            if idx == -1 or idx in seen:
                continue
            seen.add(idx)
            hits.append({"score": float(score), "fact": facts[idx], "meta": metas[idx]})
            if len(hits) >= k:
                break

    return scope, hits

# ---------- CLI/script entrypoint (won't run on import) ----------
if __name__ == "__main__":
    # Example end-to-end run when executed as a script
    print("Running nb01_helper as a script: ingest → embed → index → save")
    canvas = get_canvas_client()
    facts, metas = build_facts_and_metas(canvas)
    print(f"Built {len(facts)} facts")
    model, emb, device = embed_facts(facts)
    print("Model device:", device)
    index = build_faiss_index(emb)
    print("FAISS ntotal:", index.ntotal)
    idx_path, facts_path = save_store(index, facts, metas)
    print("Saved:", idx_path)
    print("Saved:", facts_path)
