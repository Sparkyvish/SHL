# SHL Assessment Recommender

A conversational agent that helps hiring managers find the right SHL Individual
Test Solutions through dialogue, not keyword search.

## Architecture

```
User → POST /chat (FastAPI)
         │
         ▼
      agent.py
     ┌──────────────────────────────────────────────┐
     │  1. Build retrieval query from history        │
     │  2. Semantic search → top-15 catalog items   │
     │  3. Inject items into Claude system prompt    │
     │  4. Claude generates structured JSON reply   │
     │  5. Validate schema + strip hallucinated URLs│
     └──────────────────────────────────────────────┘
         │
         ▼
      retriever.py  ←  index.faiss  +  index_meta.json
                        (built from catalog.json)
```

**Stack choices:**
- **LLM**: Anthropic Claude Sonnet (reliable JSON output, strong instruction following)
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2` — 80 MB, fast, no API cost
- **Vector store**: FAISS `IndexFlatIP` on L2-normalised embeddings (= cosine similarity, exact search, no approximate errors)
- **API**: FastAPI with Pydantic v2 for strict schema enforcement

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set environment variables

```bash
export ANTHROPIC_API_KEY=sk-ant-...
```

### 3. Scrape the SHL catalog

```bash
python scraper.py
# Writes: catalog.json
```

### 4. Build the FAISS index

```bash
python build_index.py
# Writes: index.faiss, index_meta.json
```

### 5. Start the server

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 6. Test it

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "I am hiring a Java developer who works with stakeholders"}
    ]
  }'
```

## Evaluation

```bash
# Run all traces
python eval.py --traces traces/

# Run a single trace
python eval.py --traces traces/trace_01_java_dev.json
```

Metrics reported: **Mean Recall@10**, turn count, schema errors, hallucination rate.

## Design decisions & trade-offs

### Why RAG instead of full catalog in context?
The SHL catalog has ~400 individual tests. Injecting all of them exceeds practical
context limits and degrades instruction-following. Retrieving the top 15 keeps the
prompt focused while covering recall needs.

### Why FAISS Flat (exact search) instead of HNSW?
With ~400 vectors, exact search is faster than approximate. HNSW only wins above
~100k vectors. No accuracy loss, simpler code.

### Why stateless API?
The spec requires it. Statelessness means horizontal scaling is trivial — any
replica handles any request.

### Anti-hallucination URL guard
After the LLM responds, `_validate_and_clean()` checks every recommendation URL
against the scraped catalog. Any URL not in the catalog is silently dropped before
the response is returned. This is a hard post-processing guarantee independent of
the LLM's behavior.

### Turn-cap pressure
When `len(messages) >= 6`, the system prompt adds an explicit instruction to
prioritise giving a shortlist. This ensures the agent commits to recommendations
before the evaluator's 8-turn cap is hit.

## Deployment (Render / Railway / Fly)

1. Push this repo to GitHub.
2. Create a new Web Service on Render (or equivalent).
3. Set build command: `pip install -r requirements.txt && python scraper.py && python build_index.py`
4. Set start command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add `ANTHROPIC_API_KEY` as an environment variable.
6. Note: first `/health` call may take up to 2 minutes on a cold start (model + index load).

## File structure

```
shl-recommender/
├── scraper.py         # Scrapes SHL catalog → catalog.json
├── build_index.py     # Embeds catalog → index.faiss + index_meta.json
├── retriever.py       # Semantic search wrapper (loaded once at startup)
├── agent.py           # Conversational logic + Claude prompting
├── main.py            # FastAPI app (/health, /chat)
├── eval.py            # Local evaluation harness
├── requirements.txt
├── traces/            # Conversation trace JSON files for development
│   ├── trace_01_java_dev.json
│   ├── trace_02_vague_query.json
│   └── trace_03_off_topic.json
└── README.md
```
