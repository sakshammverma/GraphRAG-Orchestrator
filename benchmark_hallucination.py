from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from graph_engine import app as graph_engine_app
from typing import cast
import json

llm = ChatOllama(model="llama3", temperature=0)

BARE_PROMPT= PromptTemplate.from_template(
    "Answer this quesiton about attention is all you need.\nQuestion:{question}\nAnswer:"
)
 
def get_bare_answers(question: str) -> str:
    response = (BARE_PROMPT | llm).invoke({"question": question}).content
    return cast(str, response)

def get_rag_ans(question: str) -> str:
     final_state =  graph_engine_app.invoke({"original_question": question, "loop_count": 0})
     return str(final_state)

def judge_answer(question: str, answer: str) -> dict:
     JUDGE_PROMPT = PromptTemplate.from_template('''
        You are a judge. Score the answer on sale of 0 to 1 given for the factual accuracy
        Question: {question}
        Answer: {answer}
        Output: strictly in JSON {'score': 7, 'reason': "..."}''')
     response = (JUDGE_PROMPT | llm). invoke({"question": question, "answer": answer}).content
     output= cast(str, response)
     try:
          return json.loads(output)
     except json.JSONDecodeError:
          return {'score': 0.0, 'reason': "Failed to parse JSON Response"}
     
def run_benchmark()