import os, json, re, time, numpy as np
import streamlit as st
from pathlib import Path
from rank_bm25 import BM25Okapi

# ---- CONFIG -----------------------------------------------------------------
# FAISS store directory or Drive path if in Google Colab
BASE = os.environ.get("SLIDEHUNT_BASE", "/content/drive/MyDrive/SlideHunt")
STORE_DIR  = f"{BASE}/data/faiss"
INDEX_PATH = f"{STORE_DIR}/canvas.index"
FACTS_PATH = f"{STORE_DIR}/facts.json"

# Route descriptions (auto-router)
ROUTE_DESC = {
    "technical": "Technical class content: Python, SQL, statistics, machine learning, slides, labs, code, algorithms, data science, lecture notes.",
    "career":    "Career readiness content: resumes, cover letters, job search, interviews, career prep, LinkedIn, networking, internship resources."
}

# Type boosts (light nudges)
TYPE_BOOST = {
    "Page":       0.10,
    "File":       0.06,
    "Assignment": -0.06,
    "Quiz":       -0.08
}
LOW_SCORE_REFUSAL = 0.40   # below this, refuse for extractive MVP

# ---- LOAD MODELS / STORE ----------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=True)
def load_faiss():
    import faiss
    idx = faiss.read_index(INDEX_PATH)
    data = json.load(open(FACTS_PATH, "r", encoding="utf-8"))
    return idx, data["facts"], data["metas"]

@st.cache_resource(show_spinner=True)
def make_bm25(facts, metas):
    # light “titles+context” corpus for BM25 (helps names like P2W2, pivot tables)
    docs = []
    for f, m in zip(facts, metas):
        title = " ".join(str(x) for x in [m.get("course_name",""), m.get("module_name",""), m.get("item_title","")])
        txt = f"{title} {f[:300]}"
        docs.append(txt.lower().split())
    return BM25Okapi(docs)

@st.cache_resource(show_spinner=True)
def load_reranker(model_name="BAAI/bge-reranker-base"):
    # Optional; enabled via checkbox
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

# ---- UTILITIES --------------------------------------------------------------
def cosine_query(index, embedder, q, k):
    # FAISS IndexFlatIP with normalized vectors
    import faiss
    qv = embedder.encode([q], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, k)
    return D[0], I[0]

def choose_scope(embedder, q, route_desc, margin=0.07):
    qv = embedder.encode([q], normalize_embeddings=True)
    sims = {}
    for k,desc in route_desc.items():
        dv = embedder.encode([desc], normalize_embeddings=True)
        sims[k] = float((qv @ dv.T)[0,0])
    best = max(sims, key=sims.get)
    ordered = sorted(sims.items(), key=lambda x: x[1], reverse=True)
    if ordered[0][1] - ordered[1][1] < margin:
        return "all", sims
    return best, sims

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
    return phase_start  # {phase: (mm, dd)}

def format_mmdd(mmdd):
    import calendar
    mm, dd = mmdd
    return f"{calendar.month_name[mm]} {dd}"

def domain_of(meta):
    # derive from meta['domain'] if present; else from course_name heuristics
    d = meta.get("domain")
    if d: return d
    name = (meta.get("course_name") or "").lower()
    if any(x in name for x in ["career", "success"]): return "career"
    return "technical"

# ---- APP UI -----------------------------------------------------------------
st.set_page_config(page_title="SlideHunt – TKH", page_icon="🧭", layout="wide")
st.title("🧭 SlideHunt — TKH Lecture Navigator (MVP)")

# Paths sanity
if not Path(INDEX_PATH).exists() or not Path(FACTS_PATH).exists():
    st.error(f"Index or facts not found.\n\nINDEX_PATH={INDEX_PATH}\nFACTS_PATH={FACTS_PATH}\n\nSet SLIDEHUNT_BASE or move your store.")
    st.stop()

embedder = load_embedder()
index, facts, metas = load_faiss()
bm25 = make_bm25(facts, metas)
phase_start = extract_phase_dates(metas)

colL, colR = st.columns([3,2])
with colR:
    st.markdown("### Options")
    scope_opt = st.selectbox("Search scope", ["auto","technical","career","all"], index=0)
    topk = st.slider("Top-k", 1, 10, 4)
    use_bm25 = st.checkbox("Use BM25 hybrid", value=True)
    use_reranker = st.checkbox("Use reranker (BGE)", value=False)
    tok, rerank_model, device = load_reranker() if use_reranker else (None, None, None)

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
        return {"latency": time.time()-t0, "special": ans, "hits": []}

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
        base = float(score)

        # light type boost/penalty
        t = m.get("type")
        base += TYPE_BOOST.get(t, 0.0)

        cand.append((base, idx))

    # hybrid: blend with BM25 on titles+snippets
    if use_bm25:
        tokens = q.lower().split()
        bm_scores = bm25.get_scores(tokens)
        # normalize to 0..1 (avoid div-by-zero)
        bmin, bmax = float(np.min(bm_scores)), float(np.max(bm_scores))
        def bnorm(i):
            if bmax == bmin: return 0.0
            return (bm_scores[i]-bmin)/(bmax-bmin)

        cand = [ (0.7*s + 0.3*bnorm(idx), idx) for (s, idx) in cand ]

    # pick top-50 for optional rerank
    cand = sorted(cand, key=lambda x: x[0], reverse=True)[:max(k*12, 50)]

    if use_reranker and rerank_model is not None:
        from transformers import AutoTokenizer
        import torch
        tok = tok or None
        pairs = [(q, facts[i]) for _,i in cand]
        # batch score
        inputs = tok(pairs, padding=True, truncation=True, return_tensors="pt", max_length=512).to(device)
        with torch.no_grad():
            scores = rerank_model(**inputs).logits.squeeze(-1).cpu().numpy()
        cand = list(zip(scores, [i for _,i in cand]))

    # finalize top-k
    cand = sorted(cand, key=lambda x: x[0], reverse=True)[:k]
    hits = []
    for s, idx in cand:
        m = metas[idx]
        cite = f"{m.get('course_name','')} › {m.get('module_name','')} › {m.get('item_title','')} ({m.get('type','')})"
        if m.get("url"): cite += f"  [{m['url']}]"
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
        st.warning("I couldn’t find a confident slide/page for that. Here are the closest matches:")
    if not out.get("hits"):
        st.info("No hits in the selected scope.")
    else:
        for h in out["hits"]:
            with st.expander(f"★ {h['score']:.3f} — {h['citation']}"):
                st.write(h["text"])

    with st.sidebar:
        st.markdown("### Router scores (debug)")
        sc, sims = choose_scope(embedder, q, ROUTE_DESC)
        st.json({k: round(v,3) for k,v in sims.items()})
