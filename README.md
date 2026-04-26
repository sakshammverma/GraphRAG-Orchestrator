# GraphRAG-Orchestrator

A production grade agentic RAG system built with LangGraph, ChromaDB, FastAPI, and Ollama. Four specialized agents collaborate to plan, retrieve, critique, and synthesize answers from academic PDFs, with real benchmark results validating every claim.

---

## Benchmark Results
 
| Metric | Result |
|---|---|
| Answer Faithfulness (RAGAS, 10 questions) | **87%** |
| Hallucination Rate — Bare LLM baseline | 66.7% |
| Hallucination Rate — RAG Pipeline | 44.4% |
| Hallucination Reduction | **22 percentage points** |
| Load Test Failure Rate (20 concurrent users) | **0%** |
 
Full results: [`ragas_results.json`](ragas_results.json) · [`hallucination_results.json`](hallucination_results.json) · [`locust_summary.json`](locust_summary.json)
 
---

## Architecture 

<img width="1368" height="1062" alt="shapes at 26-04-26 01 15 18" src="https://github.com/user-attachments/assets/e3736503-4b5b-43aa-8299-68fb26d4abd2" />


**Key design decision:** when the critic rejects retrieved context, the loop returns to the Planner, not the Retriever. This forces generation of genuinely different search queries at each retry (specific → broad → keyword fallback), rather than re-running the same queries against the same index.
 
---

## Tech Stack
 
| Component | Technology |
|---|---|
| Agent orchestration | LangGraph (StateGraph) |
| LLM | llama3 via Ollama |
| Embeddings | nomic-embed-text via Ollama |
| Vector store | ChromaDB + MMR retrieval |
| Multi-query expansion | LangChain MultiQueryRetriever |
| API server | FastAPI + SSE streaming |
| RAG evaluation | RAGAS (answer faithfulness) |
| Hallucination eval | LLM-as-judge (llama3) |
| Load testing | Locust |
| Containerization | Docker |
 
---

## Project Structure

```
MULTI-AGENT-RESEARCH-SYSTEM/
│
├── graph_engine.py              # LangGraph pipeline — 4 agents + routing logic
├── main.py                      # FastAPI server with SSE streaming endpoint
├── rag.py                       # PDF ingestion pipeline → ChromaDB
│
├── benchmark_ragas.py           # RAGAS faithfulness benchmark (parallelized + cached)
├── benchmark_hallucination.py   # LLM-as-judge hallucination benchmark
├── benchmark_timing.py          # Query time benchmark
├── locustfile.py                # Locust load test — 20 concurrent users
│
├── ragas_results.json           # RAGAS benchmark output — 87% faithfulness
├── hallucination_results.json   # Hallucination output — 22pp reduction
├── locust_summary.json          # Locust output — 0% failure rate
├── locust_report.html           # Locust HTML report
├── timing_results.json          # Query timing output
├── pipeline_cache.json          # RAGAS Phase 1 cache — safe to delete to rerun
│
├── Research PDFs/               # Source documents ingested into ChromaDB
├── chroma-db/                   # Persisted vector store (not committed to git)
│
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup & Running
 
### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) installed and running
### 1. Install dependencies
```bash
pip install -r requirements.txt
```
 
### 2. Pull models
```bash
ollama pull llama3
ollama pull nomic-embed-text
```
 
### 3. Add PDFs and ingest
Place your PDF files in the project root, then update `PDF_PATHS` in `rag.py` with your filenames. Then run:
```bash
python rag.py
```
 
### 4. Start the API server
```bash
python main.py
```
Server starts at `http://localhost:8000`
 
### 5. Send a research query
```bash
curl -X POST http://localhost:8000/api/research \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the core mechanism of the Transformer architecture?"}'
```
 
The response streams as Server-Sent Events, you'll see each agent completing in real time before the final answer arrives.
 
---

## Running Benchmarks
 
### RAGAS (answer faithfulness)
```bash
# Smoke test (10 questions, ~15 min)
python benchmark_ragas.py --quick --workers 1
 
# Full run (50 questions)
python benchmark_ragas.py --workers 1
```
Results saved to `ragas_results.json`. Safe to Ctrl+C and resume, completed questions are cached in `pipeline_cache.json`.
 
### Hallucination (LLM-as-judge)
```bash
python benchmark_hallucination.py
```
Compares bare LLM vs full RAG pipeline on adversarial questions. Results saved to `hallucination_results.json`.
 
### Load test (Locust)
```bash
# Start the API server first
python main.py
 
# In a second terminal
locust -f locustfile.py --headless -u 20 -r 5 -t 60s \
       --host http://localhost:8000 \
       --html locust_report.html
```
Results saved to `locust_summary.json` and `locust_report.html`.
 
---

## Reproducing Results
 
All benchmark numbersn are reproducible:
 
```bash
# Delete cached results
rm pipeline_cache.json ragas_results.json hallucination_results.json locust_summary.json
 
# Reingest PDFs (if chroma-db is missing)
python rag.py
 
# Run all benchmarks
python benchmark_ragas.py --quick --workers 1
python benchmark_hallucination.py
locust -f locustfile.py --headless -u 20 -r 5 -t 60s --host http://localhost:8000
```


