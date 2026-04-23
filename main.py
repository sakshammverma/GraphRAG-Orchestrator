import json
import asyncio
import threading
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
 
from graph_engine import app as graph_engine_app

# APi Initialization 
app = FastAPI(title="Agentic RAG Engine API", version="1.0.0")

# Enable CORS  
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data Models 
class ResearchRequest(BaseModel):
    query: str


def format_sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def build_node_payload(
    node_name: str,
    node_state: dict[str, Any],
    current_loops: int,
) -> dict[str, Any]:
    payload = {
        "node": node_name,
        "status": f"Agent '{node_name.upper()}' completed its task.",
        "loops": current_loops,
    }

    if node_name == "critic" and node_state.get("critic_decision") == "INVALID":
        payload["status"] = "CRITIC REJECTED DATA: Triggering fallback retrieval loop."

    if node_name == "publisher":
        payload["final_answer"] = node_state.get("final_answer", "")
        payload["done"] = True

    return payload


# The Streaming Generator
async def stream_graph_updates(user_query: str) -> AsyncIterator[str]:
    
    # Yield Server-Sent Events (SSE) for each LangGraph node update. 
    initial_state = {"original_question": user_query, "loop_count": 0}
    event_queue: asyncio.Queue[str | None] = asyncio.Queue()
    loop = asyncio.get_running_loop()

    yield format_sse(
        {
            "node": "system",
            "status": "Research request accepted. Starting agent workflow.",
            "loops": 0,
        }
    )

    def run_graph() -> None:
        current_loops = initial_state["loop_count"]
        try:
            for output in graph_engine_app.stream(initial_state):
                for node_name, node_state in output.items():
                    if "loop_count" in node_state:
                        current_loops = node_state["loop_count"]

                    payload = build_node_payload(node_name, node_state, current_loops)
                    loop.call_soon_threadsafe(
                        event_queue.put_nowait,
                        format_sse(payload),
                    )
        except Exception as exc:
            loop.call_soon_threadsafe(
                event_queue.put_nowait,
                format_sse(
                    {
                        "node": "system",
                        "status": "Agent workflow failed.",
                        "error": str(exc),
                        "done": True,
                    }
                ),
            )
        finally:
            loop.call_soon_threadsafe(event_queue.put_nowait, None)

    threading.Thread(target=run_graph, daemon=True).start()

    while True:
        event = await event_queue.get()
        if event is None:
            break

        yield event

# The API Endpoint 
@app.get("/health")
async def healthcheck():
    return {"status": "ok"}


@app.post("/api/research")
async def research_endpoint(request: ResearchRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    
    # Return the StreamingResponse with the specific SSE media type
    return StreamingResponse(
        stream_graph_updates(request.query),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

if __name__ == "__main__":
    import uvicorn
    print("\nStarting API Server on http://localhost:8000 ...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
