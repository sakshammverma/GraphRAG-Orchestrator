import logging
import re
import sys
from pathlib import Path
from typing import TypedDict

from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_core.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langgraph.graph import END, StateGraph

PDF_DB_DIRECTORY = Path(__file__).with_name("chroma-db")
CHAT_MODEL = "llama3"
EMBEDDING_MODEL = "nomic-embed-text"
DEFAULT_QUERY = "What is the core mechanism of the Transformer architecture?"


class AgentState(TypedDict, total=False):
    original_question: str
    sub_queries: list[str]
    retrieved_chunks: list[str]
    critic_decision: str
    loop_count: int
    final_answer: str


# ── Utilities 
def safe_for_console(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def parse_queries(raw_text: str, fallback_question: str) -> list[str]:
    candidates = []
    for line in raw_text.splitlines():
        cleaned = re.sub(r"^\s*\d+[\).\-\s]*", "", line).strip()
        if cleaned:
            candidates.append(cleaned)
    if not candidates and "," in raw_text:
        candidates = [part.strip() for part in raw_text.split(",") if part.strip()]
    unique_queries = list(dict.fromkeys(candidates))
    return unique_queries[:2] or [fallback_question]


def normalize_decision(raw_decision: str) -> str:
    cleaned = raw_decision.strip().upper()
    return "VALID" if cleaned == "VALID" else "INVALID"


# ── Model & DB setup 
print("Booting AI Models and Database...")
logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

llm = ChatOllama(model=CHAT_MODEL, temperature=0)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

vectorstore = Chroma(
    persist_directory=str(PDF_DB_DIRECTORY),
    embedding_function=embeddings,
)
base_retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 3, "fetch_k": 15},
)
advanced_retriever = MultiQueryRetriever.from_llm(
    retriever=base_retriever,
    llm=llm,
)


# ── Agents 
def planner_agent(state: AgentState) -> AgentState:
    print("\n[Node] PLANNER AGENT")
    question = state["original_question"]  # type: ignore
    loop_count = state.get("loop_count", 0)
 
    if loop_count == 0:
        instruction = (
            "Break this question into 2 distinct and specific search queries. "
            "Output each query on its own line with no extra commentary."
        )
    elif loop_count == 1:
        instruction = (
            "The previous search did not find enough information. "
            "Generate 2 broader, more general search queries for the same question. "
            "Try different angles or synonyms. "
            "Output each query on its own line with no extra commentary."
        )
    else:
        instruction = (
            "Previous searches failed. Generate 2 very simple keyword-based queries "
            "to find any remotely related content. "
            "Output each query on its own line with no extra commentary."
        )

    prompt = PromptTemplate.from_template(
        instruction + "\nQuestion: {question}\nQueries:"
    )
    response = (prompt | llm).invoke({"question": question})
    queries = parse_queries(response.content, question)  # type: ignore

    print(f"Generated Queries: {queries}")
   
    return {"sub_queries": queries}


def retriever_agent(state: AgentState) -> AgentState:
    print("\n[Node] RETRIEVER AGENT")
    queries = state["sub_queries"]  # type: ignore
    all_chunks: list[str] = []

    for query in queries:
        docs = advanced_retriever.invoke(query)
        all_chunks.extend(doc.page_content for doc in docs)

    unique_chunks = list(dict.fromkeys(all_chunks))
    print(f" -> Retrieved {len(unique_chunks)} unique chunks.")
 
    current_loops = state.get("loop_count", 0) + 1
    return {"retrieved_chunks": unique_chunks, "loop_count": current_loops}


def critic_agent(state: AgentState) -> AgentState:
    print("\n[Node] CRITIC AGENT")
    question = state["original_question"]  # type: ignore
    chunks = state.get("retrieved_chunks", [])
    loops = state.get("loop_count", 0)

    if loops >= 3:
        print(" -> Max loops reached. Forcing to Publisher.")
        return {"critic_decision": "VALID"}

    context = "\n\n".join(chunks)
    prompt = PromptTemplate.from_template(
        """Does this context contain ANY information relevant to answering the question?
            Even partial information counts as YES.
            If YES output exactly VALID.
            If the context is completely unrelated output INVALID.
            Question: {question}
            Context: {context}
            Decision:"""
    )

    raw_decision = (prompt | llm).invoke(
        {"question": question, "context": context}
    ).content
    decision = normalize_decision(raw_decision)  # type: ignore
    print(f" -> Critic Decision: {decision}")
    return {"critic_decision": decision}


def publisher_agent(state: AgentState) -> AgentState:
    print("\n[Node] PUBLISHER AGENT")
    question = state["original_question"]  # type: ignore
    context = "\n\n".join(state.get("retrieved_chunks", []))

    prompt = PromptTemplate.from_template(
        """You are a senior technical writer. Answer the question using ONLY the context.
            If the context is insufficient, say so clearly.
            Question: {question}
            Context: {context}
            Answer:"""
    )

    print(" -> Generating final response...")
    response = (prompt | llm).invoke({"question": question, "context": context})
    return {"final_answer": response.content}  # type: ignore


# ── Routing  
def routing_logic(state: AgentState) -> str:
    decision = state.get("critic_decision", "INVALID")
    if decision == "INVALID":
        print(" [ROUTER] Context rejected. Looping back to Planner for new queries...")
        return "planner" 
    print(" [ROUTER] Context approved. Moving to Publisher...")
    return "publisher"


# ── Graph 
print("\nCompiling LangGraph...")
workflow = StateGraph(AgentState)

workflow.add_node("planner", planner_agent)
workflow.add_node("retriever", retriever_agent)
workflow.add_node("critic", critic_agent)
workflow.add_node("publisher", publisher_agent)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "critic")
workflow.add_conditional_edges(
    "critic",
    routing_logic,
    {
        "planner": "planner",    
        "publisher": "publisher",
    },
)
workflow.add_edge("publisher", END)

app = workflow.compile()
print("Graph compiled successfully!")


if __name__ == "__main__":
    print(f"\n--- STARTING SYSTEM FOR QUERY: '{DEFAULT_QUERY}' ---")
    final_state = app.invoke({"original_question": DEFAULT_QUERY, "loop_count": 0})

    print("\n================ FINAL OUTPUT ================\n")
    print(safe_for_console(final_state.get("final_answer", "No final answer generated.")))
    print("\n==============================================")