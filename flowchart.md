
<h2 align="center">SlideHunter-App Flow Diagram <a href="https://mermaid.js.org/">mermaid</a></h2>

<div align="center">

```mermaid
flowchart TB
  %% Sources
  A["Canvas API<br/>(Pages, Modules, Links)"]

  %% Ingestion
  C["Ingestion & Chunking<br/>(Canvas text/items · 400-600 char chunks · metadata)"]

  %% Retrieval building blocks
  D["Embeddings<br/>Sentence-Transformers<br/>MiniLM-L6-v2"]
  E["Vector Store<br/>FAISS (cosine)<br/>canvas.index + facts.json"]
  F["BM25 over titles + snippets<br/>(rank-bm25)"]

  %% Logic
  G["Router: technical <-> career<br/>(route descriptions)"]
  H["Optional Reranker<br/>BGE reranker (top-50)"]
  I["Low-score Refusal<br/>(type boosts + threshold)"]

  %% UI
  J["Streamlit UI<br/>Top-k hits + citations<br/>Answer-from-context"]

  %% Edges
  A --> C
  C --> D
  C --> E
  C --> F
  D --> G
  E --> H
  F --> I
  G --> J
  H --> J
  I --> J
