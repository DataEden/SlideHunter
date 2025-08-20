# SlideHunt — Lecture Navigator MVP (Multi-Modal RAG)

Find exactly where a topic was covered in your course slides. Notebook-first workflow for groups.

## How to use
1) Drop the desired PDF, etc., files into `data/slides/`.
2) Create a virtual env, then install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3) Open the notebooks in order:
   - `01_setup_and_ingest.ipynb` → builds the Chroma index from PDFs
   - `02_query_demo.ipynb` → ask questions, get answers + (deck, page) citations
   - `03_eval.ipynb` → run a tiny test set to sanity-check the MVP?
4) We can Set `OPENAI_API_KEY` in `.env` for generative answers. Without it, the notebooks fall back to local embeddings + extractive snippets.

## Philosophy (least stress)
- PDFs only for now, page-level chunks
- One index (Chroma), one collection (`slides`)
- Strict "answer-from-context" prompting, graceful refusal
- Small eval set to prove it works

## Repo Tree
```
slideHunt-notebooks/
├─ README.md
├─ requirements.txt
├─ data/
│  ├─ slides/         
│  └─ index/          # chroma index
├─ prompts/
│  └─ answer_from_context.txt
├─ notebooks/
│  ├─ 01_setup_and_ingest.ipynb
│  ├─ 02_query_demo.ipynb
│  └─ 03_eval.ipynb

```
