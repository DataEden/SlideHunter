## Streamlit Application User Interface (UI)

# SlideHunter.py — Streamlit app (MVP + optional summarizer)
# -----------------------------------------------------------------------------
# Usage: streamlit run SlideHunter.py
# Env:
#   SLIDEHUNT_BASE=/path/to/project           # where data/faiss/*.{index,json} live
#   OPENAI_API_KEY=sk-...                     # optional (enables GPT-4o summarizer)

import os, json, re, time, numpy as np
import streamlit as st
from pathlib import Path
from rank_bm25 import BM25Okapi

# Keep Transformers in PyTorch mode only; silence TF logs if present
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TRANSFORMERS_NO_FLAX", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# ---- CONFIG -----------------------------------------------------------------
BASE = os.environ.get("SLIDEHUNT_BASE", "/content/drive/MyDrive/SlideHunt")
STORE_DIR  = f"{BASE}/data/faiss"
INDEX_PATH = f"{STORE_DIR}/canvas.index"
FACTS_PATH = f"{STORE_DIR}/facts.json"

ROUTE_DESC = {
    "technical": "Technical class content: Python, SQL, statistics, machine learning, slides, labs, code, algorithms, data science, lecture notes.",
    "career":    "Career readiness content: resumes, cover letters, job search, interviews, career prep, LinkedIn, networking, internship resources."
}

TYPE_BOOST = {
    "Page":       0.10,
    "SlidePage":  0.08,
    "File":       0.06,
    "Assignment": -0.06,
    "Quiz":       -0.08
}
LOW_SCORE_REFUSAL = 0.40   # extractive refusal; summarizer won’t run below this

# ---- LOAD MODELS / STORE ----------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_embedder():
    from sentence_transformers import SentenceTransformer
    # Auto-select device inside ST; fast startup + GPU if available
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=True)
def load_faiss():
    import faiss
    idx = faiss.read_index(INDEX_PATH)
    data = json.load(open(FACTS_PATH, "r", encoding="utf-8"))
    return idx, data["facts"], data["metas"]

@st.cache_resource(show_spinner=True)
def make_bm25(facts, metas):
    # Build a simple corpus: (course + module + item title) + leading snippet
    docs = []
    for f, m in zip(facts, metas):
        title = " ".join(str(x) for x in [m.get("course_name",""), m.get("module_name",""), m.get("item_title","")])
        txt = f"{title} {f[:300]}"
        docs.append(txt.lower().split())
    return BM25Okapi(docs)

@st.cache_resource(show_spinner=True)
def load_reranker(model_name="BAAI/bge-reranker-base"):
    # Optional reranker (CPU or GPU); loaded only when checkbox is on
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

# ---- SUMMARIZER (OpenAI or local fallback) ----------------------------------
def _build_context(hits, cap=1400):
    """Make a numbered context block the LLM can truly read (trimmed)."""
    from textwrap import shorten
    blocks, used = [], 0
    for i, h in enumerate(hits, 1):
        snippet = (h.get("text","") or "").replace("\n", " ").strip()
        snippet = shorten(snippet, width=600, placeholder="…")
        cite = h.get("citation") or ""
        block = f"[{i}] {snippet}\nSOURCE: {cite}"
        if used + len(block) > cap:
            break
        blocks.append(block); used += len(block)
    return "\n\n".join(blocks)

@st.cache_resource(show_spinner=False)
def have_openai_key():
    return bool(os.getenv("OPENAI_API_KEY"))

def summarize_gpt4o(question, hits, model="gpt-4o-mini", max_context_chars=1400):
    """Guardrailed GPT-4o/mini summarizer. Requires OPENAI_API_KEY."""
    
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    context = _build_context(hits, cap=max_context_chars)
    prompt = f"""You are a careful teaching assistant.
Answer the question in 1-2 sentences using ONLY the Context.
If the Context is insufficient, say you can't find it in the slides/pages.
Add inline citation markers like [1], [2] that refer to the numbered sources.

Question: {question}

Context:
{context}
"""
    resp = client.responses.create(model=model, input=prompt)
    return resp.output_text.strip()

@st.cache_resource(show_spinner=True)
def _load_local_summarizer():
    from transformers import pipeline
    # FLAN-T5 base: decent quality, runs on CPU for short outputs
    return pipeline("text2text-generation", model="google/flan-t5-base")

def summarize_local(question, hits, max_context_chars=1400):
    """Local fallback summarizer (no API key needed)."""
    pipe = _load_local_summarizer()
    context = _build_context(hits, cap=max_context_chars)
    p = ("Using ONLY the Context, answer the Question in 1-2 sentences. "
         "Add inline citation markers like [1], [2] that refer to the source numbers in Context. "
         "If insufficient, say you cannot find it in the slides/pages.\n\n"
         f"Question: {question}\n\nContext:\n{context}")
    out = pipe(p, max_new_tokens=128, do_sample=False)[0]["generated_text"]
    return out.strip()

# ---- UTILITIES --------------------------------------------------------------
def cosine_query(index, embedder, q, k):
    import faiss
    qv = embedder.encode([q], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, k)
    return D[0], I[0]

def extract_phase_dates(metas):
    # Find patterns like: P2W1 (6/9-6/13) in module_name
    phase_start = {}
    patt = re.compile(r"P(\d+)W\d+\s*\((\d{1,2})/(\d{1,2})-(\d{1,2})/(\d{1,2})\)")
    for m in metas:
        s = m.get("module_name") or ""
        mt = patt.search(s)
        if mt:
            phase = int(mt.group(1))
            start = (int(mt.group(2)), int(mt.group(3)))  # (month, day)
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

# ---- APP UI -----------------------------------------------------------------
st.set_page_config(page_title="SlideHunter - TKH", page_icon="🧭", layout="wide")
st.title("🧭 SlideHunter — TKH Lecture Navigator (MVP)")

# file presence check
if not Path(INDEX_PATH).exists() or not Path(FACTS_PATH).exists():
    st.error(f"Index or facts not found.\n\nINDEX_PATH={INDEX_PATH}\nFACTS_PATH={FACTS_PATH}\n\nSet SLIDEHUNT_BASE or move your store.")
    st.stop()

embedder = load_embedder()
index, facts, metas = load_faiss()
bm25 = make_bm25(facts, metas)
phase_start = extract_phase_dates(metas)

# Precompute router embeddings once (faster than recomputing per query)
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

    # Summarizer controls
    default_sum = have_openai_key()  # on by default if API key exists
    gen_summary = st.checkbox("Generate concise answer (summarizer)", value=default_sum)
    sum_model = st.selectbox("Summarizer model", ["gpt-4o-mini", "gpt-4o", "local-fallback"],
                             index=(0 if have_openai_key() else 2))
    max_ctx = st.slider("Summary context size (chars)", 600, 2400, 1400, step=200)

    rr_tok, rr_model, rr_device = load_reranker() if use_reranker else (None, None, None)

with colL:
    q = st.text_input("Ask a question", placeholder="e.g., Where did we define precision vs. recall?")
    go = st.button("Search")

# ---- SEARCH LOGIC -----------------------------------------------------------
def search(q, k=4, scope="auto"):
    t0 = time.time()

    # special-case: phase begin date
    if re.search(r"\bphase\s*2\b.*(begin|start|commence|when)", q, flags=re.I) and 2 in phase_start:
        mmdd = phase_start[2]
        ans = f"Phase 2 begins on {format_mmdd(mmdd)}."
        return {"latency": time.time()-t0, "special": ans, "hits": [], "scope": "all"}

    # scope
    if scope == "auto":
        scope, sims = choose_scope(embedder, q, ROUTE_DESC)

    # dense retrieval (grab extra for filtering)
    D, I = cosine_query(index, embedder, q, k*10)
    cand = []
    for score, idx in zip(D, I):
        if idx == -1: continue
        m = metas[idx]
        dom = domain_of(m)
        if scope != "all" and dom != scope:
            continue
        base = float(score) + TYPE_BOOST.get(m.get("type"), 0.0)
        cand.append((base, idx))

    # hybrid: blend with BM25 on titles+snippets
    if use_bm25 and len(cand):
        tokens = q.lower().split()
        bm_scores = bm25.get_scores(tokens)
        bmin, bmax = float(np.min(bm_scores)), float(np.max(bm_scores))
        def bnorm(i):
            if bmax == bmin: return 0.0
            return (bm_scores[i]-bmin)/(bmax-bmin)
        cand = [(0.7*s + 0.3*bnorm(idx), idx) for (s, idx) in cand]

    # pick top-50 for optional rerank
    cand = sorted(cand, key=lambda x: x[0], reverse=True)[:max(k*12, 50)]

    # optional: BGE reranker
    if use_reranker and rr_model is not None and len(cand):
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
        if m.get("url"): cite += f"  [{m['url']}]"
        if m.get("deck"): cite += f"  [{m['deck']} p.{m.get('page')}]"
        hits.append({
            "score": round(float(s), 3),
            "citation": cite,
            "domain": domain_of(m),
            "text": facts[idx]
        })

    # refusal if top score too low
    if hits and hits[0]["score"] < LOW_SCORE_REFUSAL:
        return {"latency": time.time()-t0, "refusal": True, "scope": scope, "hits": hits}
    return {"latency": time.time()-t0, "scope": scope, "hits": hits}

# ---- UI RENDER --------------------------------------------------------------
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
        # Optional summarizer (guarded by threshold + toggle)
        if gen_summary and out["hits"] and (out["hits"][0]["score"] >= LOW_SCORE_REFUSAL):
            with st.spinner("Summarizing…"):
                try:
                    if sum_model == "local-fallback" or (sum_model.startswith("gpt-") and not have_openai_key()):
                        ans = summarize_local(q, out["hits"], max_context_chars=max_ctx)
                        st.success(ans)
                        st.caption("Model: local (FLAN-T5)")
                    else:
                        ans = summarize_gpt4o(q, out["hits"], model=sum_model, max_context_chars=max_ctx)
                        st.success(ans)
                        st.caption(f"Model: {sum_model}")
                except Exception as e:
                    st.warning(f"Summarizer unavailable: {e}")

        # Show hits with citations
        for h in out["hits"]:
            with st.expander(f"★ {h['score']:.3f} — {h['citation']}"):
                st.write(h["text"])

    # Debug router scores
    with st.sidebar:
        st.markdown("### Router scores (debug)")
        sc, sims = choose_scope(embedder, q, ROUTE_DESC)
        st.json({k: round(v,3) for k,v in sims.items()})

# Using Cloudflare  

# Mount Drive (so app can read FAISS store)
from google.colab import drive
drive.mount('/content/drive')

# Config: where FAISS files live & where app.py is
import os, re, time, subprocess, shutil, pathlib
BASE   = "/content/drive/MyDrive/SlideHunt"   # Base directory
APPDIR = "/content"                           # SlideHunter.py folder
os.environ["SLIDEHUNT_BASE"] = BASE

# Sanity check saved files
idx   = pathlib.Path(BASE) / "data/faiss/canvas.index"
facts = pathlib.Path(BASE) / "data/faiss/facts.json"
print("Index exists:", idx.exists(), idx)
print("Facts exists:", facts.exists(), facts)

# Dependencies
!pip -q install streamlit sentence-transformers faiss-cpu rank-bm25

# Kill previous runs (ignore errors)
!pkill -f "streamlit run" || true
!pkill -f "cloudflared"   || true

# Start streamlit in background
%cd $APPDIR
!streamlit run SlideHunters.py --server.port 8501 --server.headless true &

# Ensure cloudflared is available: try pip, else download binary
path = shutil.which("cloudflared")
if not path:
    try:
        import sys
        !pip -q install cloudflared
        path = shutil.which("cloudflared")
    except Exception as e:
        path = None

if not path:
    # Download static binary (linux x86_64)
    !wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /content/cloudflared
    !chmod +x /content/cloudflared
    path = "/content/cloudflared"

print("Using cloudflared at:", path)

# Open tunnel and print public URL
p = subprocess.Popen([path, "tunnel", "--url", "http://localhost:8501", "--no-autoupdate"],
                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

url = None
for _ in range(300):
    line = p.stdout.readline()
    if "trycloudflare.com" in line:
        m = re.search(r"https?://[^\s]*trycloudflare\.com", line)
        if m:
            url = m.group(0); break
    time.sleep(0.05)

print("Public URL:", url or "(not found yet — scroll the logs above)")

