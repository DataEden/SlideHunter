# SlideHunter.py — Streamlit app (local-ready)
# Usage: streamlit run app/SlideHunter.py
# Env (optional):
#   SLIDEHUNT_BASE=/path/to/repo/root    # where data/faiss/{canvas.index,facts.json} live
#   OPENAI_API_KEY=sk-...                # enables GPT-4o summarizer option

import os, json, re, time, numpy as np
import streamlit as st
from pathlib import Path
from rank_bm25 import BM25Okapi

# Keep Transformers in PyTorch mode only; silence TF logs if present
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---------- Path resolution (robust + .env support) ----------
import os
from pathlib import Path

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()  # loads variables like SLIDEHUNTER_BASE / SLIDEHUNT_BASE
except Exception:
    pass  # python-dotenv not installed is fine; env vars may still be set

def _norm(p: str) -> Path:
    # Normalize Windows/Unix paths, expand ~, and resolve
    return Path(p.replace("\\", "/")).expanduser().resolve()

def get_repo_base() -> Path:
    """
    Resolve repo root so that data/faiss/* are found.
    Priority:
      1) SLIDEHUNTER_BASE (from .env or env)
      2) SLIDEHUNT_BASE   (legacy name)
      3) Auto-detect:
         - this file at repo root:  <repo>/SlideHunter.py
         - this file in app/:       <repo>/app/SlideHunter.py
      4) Walk upward looking for data/faiss
    """
    # 1) Preferred var
    v = os.getenv("SLIDEHUNTER_BASE")
    if v:
        return _norm(v)

    
    """
    # 2) Legacy var
    v = os.getenv("SLIDEHUNT_BASE")
    if v:
        return _norm(v)
    """

    # 3) Auto-detect
    here = Path(__file__).resolve()
    root = here.parent
    if (root / "data" / "faiss").exists():
        return root
    if root.name.lower() == "app" and (root.parent / "data" / "faiss").exists():
        return root.parent

    # 4) Walk up a few levels
    probe = root
    for _ in range(4):
        if (probe / "data" / "faiss").exists():
            return probe
        probe = probe.parent

    # Fallback (likely repo root if you run from there)
    return root

BASE = get_repo_base()
STORE_DIR  = BASE / "data" / "faiss"
SLIDES_DIR = BASE / "data" / "slides"
INDEX_PATH = STORE_DIR / "canvas.index"
FACTS_PATH = STORE_DIR / "facts.json"

# ---------- Router + scoring knobs ----------
ROUTE_DESC = {
    "technical": "Technical class content: Python, SQL, statistics, machine learning, slides, labs, code, algorithms, data science, lecture notes.",
    "career":    "Career readiness content: resumes, cover letters, job search, interviews, career prep, LinkedIn, networking, internship resources.",
}
TYPE_BOOST = {"Page": 0.10, "SlidePage": 0.08, "File": 0.06, "Assignment": -0.06, "Quiz": -0.08}
LOW_SCORE_REFUSAL = 0.40

# ---------- Cached loaders ----------
@st.cache_resource(show_spinner=True)
def load_embedder():
    from sentence_transformers import SentenceTransformer
    # Auto-select device (GPU if available); SentenceTransformers handles this
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=True)
def load_faiss_store():
    import faiss
    idx = faiss.read_index(str(INDEX_PATH))
    data = json.load(open(FACTS_PATH, "r", encoding="utf-8"))
    return idx, data["facts"], data["metas"]

@st.cache_resource(show_spinner=True)
def make_bm25(facts, metas):
    docs = []
    for f, m in zip(facts, metas):
        title = " ".join(str(x) for x in [m.get("course_name",""), m.get("module_name",""), m.get("item_title","")])
        docs.append((title + " " + f[:300]).lower().split())
    return BM25Okapi(docs)

@st.cache_resource(show_spinner=True)
def load_reranker(model_name="BAAI/bge-reranker-base"):
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch
        tok = AutoTokenizer.from_pretrained(model_name)
        mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        mdl.to(device)
        return tok, mdl, device
    except Exception as e:
        st.warning(f"Reranker not available: {e}")
        return None, None, None

# ---------- Summarizer helpers ----------
def _build_context(hits, cap=1400):
    from textwrap import shorten
    blocks, used = [], 0
    for i, h in enumerate(hits, 1):
        snippet = shorten((h.get("text","") or "").replace("\n", " ").strip(), width=600, placeholder="…")
        cite = h.get("citation") or ""
        block = f"[{i}] {snippet}\nSOURCE: {cite}"
        if used + len(block) > cap:
            break
        blocks.append(block); used += len(block)
    return "\n\n".join(blocks)

def have_openai_key():  # cached by env; cheap
    return bool(os.getenv("OPENAI_API_KEY"))

def summarize_gpt4o(question, hits, model="gpt-4o-mini", max_context_chars=1400):
    from openai import OpenAI
    client  = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    context = _build_context(hits, cap=max_context_chars)
    prompt  = (
        "You are a careful teaching assistant.\n"
        "Answer the question in 1–2 sentences using ONLY the Context.\n"
        "If the Context is insufficient, say you can't find it in the slides/pages.\n"
        "Add inline citation markers like [1], [2] that refer to the numbered sources.\n\n"
        f"Question: {question}\n\nContext:\n{context}"
    )
    resp = client.responses.create(model=model, input=prompt)
    return resp.output_text.strip()

@st.cache_resource(show_spinner=True)
def _load_local_summarizer():
    from transformers import pipeline
    return pipeline("text2text-generation", model="google/flan-t5-base")

def summarize_local(question, hits, max_context_chars=1400):
    pipe    = _load_local_summarizer()
    context = _build_context(hits, cap=max_context_chars)
    p = (
        "Using ONLY the Context, answer the Question in 1–2 sentences. "
        "Add inline citation markers like [1], [2] that refer to the source numbers in Context. "
        "If insufficient, say you cannot find it in the slides/pages.\n\n"
        f"Question: {question}\n\nContext:\n{context}"
    )
    return pipe(p, max_new_tokens=128, do_sample=False)[0]["generated_text"].strip()

# ---------- Utilities ----------
def cosine_query(index, embedder, q, k):
    import faiss
    qv = embedder.encode([q], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, k)
    return D[0], I[0]

def extract_phase_dates(metas):
    phase_start = {}
    patt = re.compile(r"P(\d+)W\d+\s*\((\d{1,2})/(\d{1,2})-(\d{1,2})/(\d{1,2})\)")
    for m in metas:
        s = m.get("module_name") or ""
        mt = patt.search(s)
        if mt:
            phase = int(mt.group(1))
            start = (int(mt.group(2)), int(mt.group(3)))
            if phase not in phase_start or start < phase_start[phase]:
                phase_start[phase] = start
    return phase_start

def format_mmdd(mmdd):
    import calendar
    mm, dd = mmdd
    return f"{calendar.month_name[mm]} {dd}"

def domain_of(meta):
    d = meta.get("domain")
    if d: return d
    name = (meta.get("course_name") or "").lower()
    if any(x in name for x in ["career", "success"]): return "career"
    return "technical"

# ---------- App UI ----------
st.set_page_config(page_title="SlideHunter - TKH", page_icon="🧭", layout="wide")
st.title("🧭 SlideHunter — TKH Lecture Navigator (MVP)")

if not INDEX_PATH.exists() or not FACTS_PATH.exists():
    st.error(
        "Index or facts not found.\n\n"
        f"INDEX_PATH = {INDEX_PATH}\nFACTS_PATH = {FACTS_PATH}\n\n"
        "Run ingest notebook first, or set SLIDEHUNT_BASE to the repo root."
    )
    st.stop()

embedder = load_embedder()
index, facts, metas = load_faiss_store()
bm25 = make_bm25(facts, metas)
phase_start = extract_phase_dates(metas)

# Precompute router descriptions once
_route_emb = {
    k: embedder.encode([desc], normalize_embeddings=True).astype("float32")
    for k, desc in ROUTE_DESC.items()
}

def choose_scope(embedder, q, route_desc, margin=0.07):
    qv = embedder.encode([q], normalize_embeddings=True).astype("float32")
    sims = {k: float((qv @ _route_emb[k].T)[0,0]) for k in route_desc}
    ordered = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    if len(ordered) < 2 or (ordered[0][1] - ordered[1][1]) >= margin:
        return ordered[0][0], sims
    return "all", sims

colL, colR = st.columns([3,2])
with colR:
    st.markdown("### Options")
    scope_opt = st.selectbox("Search scope", ["auto","technical","career","all"], index=0)
    topk = st.slider("Top-k", 1, 10, 4)
    use_bm25 = st.checkbox("Use BM25 hybrid", value=True)
    use_reranker = st.checkbox("Use reranker (BGE)", value=False)

    default_sum = have_openai_key()
    gen_summary = st.checkbox("Generate concise answer (summarizer)", value=default_sum)
    sum_model = st.selectbox(
        "Summarizer model", ["gpt-4o-mini", "gpt-4o", "local-fallback"],
        index=(0 if default_sum else 2)
    )
    max_ctx = st.slider("Summary context size (chars)", 600, 2400, 1400, step=200)

    rr_tok, rr_model, rr_device = load_reranker() if use_reranker else (None, None, None)

with colL:
    q = st.text_input("Search for Course Material", placeholder="e.g., Where did we define precision vs. recall?")
    go = st.button("Search")

# ---------- Search ----------
def search(q, k=4, scope="auto"):
    t0 = time.time()

    # special-case: Phase 2 begin
    if re.search(r"\bphase\s*2\b.*(begin|start|commence|when)", q, flags=re.I) and 2 in phase_start:
        mmdd = phase_start[2]
        ans = f"Phase 2 begins on {format_mmdd(mmdd)}."
        return {"latency": time.time()-t0, "special": ans, "hits": [], "scope": "all"}

    if scope == "auto":
        scope, _ = choose_scope(embedder, q, ROUTE_DESC)

    # dense retrieval (grab extra, then filter)
    D, I = cosine_query(index, embedder, q, k*10)
    cand = []
    for score, idx in zip(D, I):
        if idx == -1: continue
        m = metas[idx]
        dom = domain_of(m)
        if scope != "all" and dom != scope:
            continue
        cand.append((float(score) + TYPE_BOOST.get(m.get("type"), 0.0), idx))

    # hybrid: BM25 reweight
    if use_bm25 and cand:
        tokens = q.lower().split()
        bm_scores = bm25.get_scores(tokens)
        bmin, bmax = float(np.min(bm_scores)), float(np.max(bm_scores))
        def bnorm(i): return 0.0 if bmax == bmin else (bm_scores[i]-bmin)/(bmax-bmin)
        cand = [(0.7*s + 0.3*bnorm(idx), idx) for (s, idx) in cand]

    # optional rerank (top-50)
    cand = sorted(cand, key=lambda x: x[0], reverse=True)[:max(k*12, 50)]
    if use_reranker and rr_model is not None and cand:
        import torch
        pairs = [(q, facts[i]) for _, i in cand]
        inputs = rr_tok(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512).to(rr_device)
        with torch.no_grad():
            scores = rr_model(**inputs).logits.squeeze(-1).cpu().numpy()
        cand = list(zip(scores, [i for _, i in cand]))

    # finalize top-k
    cand = sorted(cand, key=lambda x: x[0], reverse=True)[:k]
    hits = []
    for s, idx in cand:
        m = metas[idx]
        cite = f"{m.get('course_name','')} > {m.get('module_name','')} > {m.get('item_title','')} ({m.get('type','')})"
        if m.get("url"):  cite += f"  [{m['url']}]"
        if m.get("deck"): cite += f"  [{m['deck']} p.{m.get('page')}]"
        hits.append({"score": round(float(s), 3), "citation": cite, "domain": domain_of(m), "text": facts[idx]})

    if hits and hits[0]["score"] < LOW_SCORE_REFUSAL:
        return {"latency": time.time()-t0, "refusal": True, "scope": scope, "hits": hits}
    return {"latency": time.time()-t0, "scope": scope, "hits": hits}

# ---------- Render ----------
if go and q.strip():
    out = search(q, k=topk, scope=scope_opt)

    st.caption(f"Latency: {out.get('latency', 0):.3f}s   |   Scope: {out.get('scope','auto')}")
    if out.get("special"):
        st.success(out["special"])

    if out.get("refusal"):
        st.warning("I couldn't find a confident slide/page for that. Here are the closest matches:")
    if not out.get("hits"):
        st.info("No hits in the selected scope.")
    else:
        # summarizer (if enabled + above threshold)
        if gen_summary and out["hits"] and (out["hits"][0]["score"] >= LOW_SCORE_REFUSAL):
            with st.spinner("Summarizing…"):
                try:
                    if sum_model == "local-fallback" or (sum_model.startswith("gpt-") and not have_openai_key()):
                        ans = summarize_local(q, out["hits"], max_context_chars=max_ctx)
                        st.success(ans); st.caption("Model: local (FLAN-T5)")
                    else:
                        ans = summarize_gpt4o(q, out["hits"], model=sum_model, max_context_chars=max_ctx)
                        st.success(ans); st.caption(f"Model: {sum_model}")
                except Exception as e:
                    st.warning(f"Summarizer unavailable: {e}")

        for h in out["hits"]:
            with st.expander(f"★ {h['score']:.3f} — {h['citation']}"):
                st.write(h["text"])

    with st.sidebar:
        st.markdown("### Router scores (debug)")
        sc, sims = choose_scope(embedder, q, ROUTE_DESC)
        st.json({k: round(v,3) for k,v in sims.items()})
