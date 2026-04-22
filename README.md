# GraphRAG-Orchestrator

A local multi-agent RAG prototype built with **LangGraph + FastAPI + Chroma + Ollama**.

## Current Progress 

- **4-agent workflow** wired in LangGraph: `planner -> retriever -> critic -> publisher` with a retry loop from critic back to retriever. 
- **Retriever stack** using Chroma vector DB + MMR + MultiQueryRetriever for better recall.
- **Ingestion pipeline** (`rag.py`) that chunks and embeds PDF content into a persistent Chroma store.
- **Streaming API** (`main.py`) with Server-Sent Events (SSE) so clients can receive node-by-node updates and final answers.

## Overall Workflow
  <img width="1477" height="863" alt="shapes at 26-04-22 04 01 34" src="https://github.com/user-attachments/assets/bad39273-eba6-48f2-a44c-d4e71aff32d4" />
