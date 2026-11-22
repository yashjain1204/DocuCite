# DocuCite

Upload your docs and get cited answers. A from-scratch RAG pipeline (sentence-transformers + FAISS + FastAPI), benchmarked on the FastAPI docs: recall@5 [X]%, held-out accuracy [Y]/10.

> 🚧 Work in progress.

## Project structure

```
DocuCite/
├── backend/              # Python API + RAG pipeline
│   ├── app/
│   │   ├── core/         # config, logging
│   │   ├── api/routes/   # HTTP endpoints
│   │   ├── schemas/      # request/response models
│   │   ├── services/     # use cases that tie the pieces together
│   │   ├── llm/          # swappable LLM providers
│   │   ├── rag/          # ingestion → embeddings → vectorstore → retrieval → generation
│   │   └── evaluation/   # recall@k, MRR, accuracy
│   ├── scripts/          # CLI entry points (build index, run eval)
│   └── tests/
├── frontend/             # Next.js app (Vercel)
├── data/raw/             # source documents
├── eval/datasets/        # hand-written test set
└── storage/              # generated indexes and uploads (git-ignored)
```

## Setup

```bash
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
copy backend\.env.example backend\.env   # then add your LLM API key
```

## Data

`data/raw/fastapi/` contains the English FastAPI documentation (Markdown), © Sebastián Ramírez, MIT License. See `data/raw/fastapi/LICENSE-FastAPI`.
