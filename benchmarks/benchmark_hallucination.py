from pathlib import Path
from typing import cast
import json

from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

from core.graph_engine import app as graph_engine_app

llm = ChatOllama(model="llama3", temperature=0)

BARE_PROMPT = PromptTemplate.from_template(
    "Answer this question about Attention Is All You Need.\nQuestion: {question}\nAnswer:"
)


def get_bare_answers(question: str) -> str:
    response = (BARE_PROMPT | llm).invoke({"question": question}).content
    return cast(str, response)


def get_rag_ans(question: str) -> str:
    final_state = graph_engine_app.invoke({"original_question": question, "loop_count": 0})
    return cast(str, final_state.get("final_answer", "No answer generated."))


def judge_answer(question: str, answer: str) -> dict:
    JUDGE_PROMPT = PromptTemplate.from_template(
    """You are a factual accuracy judge. Follow these steps exactly.

    STEP 1 - List the specific factual claims made in the answer.
    STEP 2 - For each claim, write CORRECT or WRONG and why.
    STEP 3 - Count: how many claims are WRONG out of total claims.
    STEP 4 - Assign score using this exact rule:
        0 wrong out of total  → score = 9
        1 wrong out of total  → score = 6
        2 wrong out of total  → score = 4
        3+ wrong out of total → score = 2
        Answer says "I don't know" or is empty → score = 3

    Question: {question}
    Answer: {answer}

    After your analysis output ONLY this JSON on the last line, nothing after it:
    {{"score": <integer>, "reason": "<one sentence summary>"}}"""
)

    response = (JUDGE_PROMPT | llm).invoke({
        "question": question,
        "answer": answer
    }).content.strip()

    # Clean common LLM formatting issues
    if response.startswith("```"):
        response = response.split("```")[1]
        if response.startswith("json"):
            response = response[4:]
    response = response.strip()

    try:
        result = json.loads(response)
        # Ensure score is a number
        result["score"] = float(result.get("score", 5))
        return result
    except json.JSONDecodeError:
        # Try extracting score with regex as fallback
        import re
        match = re.search(r'"score"\s*:\s*(\d+)', response)
        score = float(match.group(1)) if match else 5.0
        return {"score": score, "reason": f"parse fallback: {response[:80]}"}

def parse_score(judgment: dict) -> float:
    try:
        return float(judgment.get("score", 5))
    except (TypeError, ValueError):
        return 5.0


THRESHOLD = 7  # scores at or below this threshold are counted as hallucinations


def run_benchmark(
    questions: list[str], output_file: str = "hallucination_results.json"
) -> dict[str, float | int]:
    bare_hallucinations = 0
    rag_hallucinations = 0
    records = []

    for i, question in enumerate(questions, 1):
        print(f"[{i:03d}/{len(questions)}] {question[:60]}...")

        print("  Getting bare answer...")
        bare_answer = get_bare_answers(question)

        print("  Getting RAG answer...")
        rag_answer = get_rag_ans(question)

        print("  Judging...")
        bare_judge = judge_answer(question, bare_answer)
        rag_judge = judge_answer(question, rag_answer)

        bare_score = parse_score(bare_judge)
        rag_score = parse_score(rag_judge)

        bare_hallucinated = bare_score <= THRESHOLD
        rag_hallucinated = rag_score <= THRESHOLD

        bare_hallucinations += int(bare_hallucinated)
        rag_hallucinations += int(rag_hallucinated)

        print(
            f"  bare={bare_score:.1f}/10 {'(hallucinated)' if bare_hallucinated else '(accurate)'} | "
            f"rag={rag_score:.1f}/10 {'(hallucinated)' if rag_hallucinated else '(accurate)'}"
        )

        records.append(
            {
                "question": question,
                "bare_answer": bare_answer,
                "bare_score": bare_score,
                "bare_hallucinated": bare_hallucinated,
                "rag_answer": rag_answer,
                "rag_score": rag_score,
                "rag_hallucinated": rag_hallucinated,
            }
        )

    n = len(questions)
    bare_rate = bare_hallucinations / n * 100 if n else 0.0
    rag_rate = rag_hallucinations / n * 100 if n else 0.0

    summary = {
        "total_questions": n,
        "bare_hallucination_rate_pct": round(bare_rate, 1),
        "rag_hallucination_rate_pct": round(rag_rate, 1),
        "reduction_percentage_points": round(bare_rate - rag_rate, 1),
    }

    print(f"\n{'=' * 50}")
    print(f"  Bare LLM hallucination : {bare_rate:.1f}%")
    print(f"  RAG  hallucination     : {rag_rate:.1f}%")
    print(f"  Reduction              : {bare_rate - rag_rate:.1f} pp")
    print(f"{'=' * 50}\n")

    Path(output_file).write_text(
        json.dumps({"summary": summary, "records": records}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved -> {output_file}")
    return summary


if __name__ == "__main__":
    questions = [ 
    "What does Mamba's selective SSM make input-dependent that traditional SSMs do not?",
    "What is the exact name of the selection mechanism Mamba uses to filter information?", 
    "What sequence length does Mamba achieve 5x faster inference than Transformer at?",
    "What specific architecture block does Mamba introduce combining SSM with gating?",

    "What does this paper propose as an alternative to vector similarity search in RAG?", 
    "What reasoning approach does the paper propose instead of embedding-based retrieval?",
    "What does the paper call its proposed non-vector retrieval system?",
    "What specific failure mode of cosine similarity does the paper demonstrate?",
    "What evaluation benchmark does the paper use to test its proposed approach?", 
]
    run_benchmark(questions)
