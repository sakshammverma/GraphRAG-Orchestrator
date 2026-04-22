import json
import time
from pathlib import Path

from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, faithfulness

ragas_llm = LangchainLLMWrapper(ChatOllama(model="llama3", temperature=0))
ragas_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

EVAL_QUESTIONS = [
    {
        "question": "What is the core mechanism of the Transformer architecture?",
        "ground_truth": "The Transformer relies entirely on self-attention mechanisms to compute representations of input and output sequences, dispensing with recurrence and convolutions.",
    },
    {
        "question": "What is multi-head attention?",
        "ground_truth": "Multi-head attention runs attention in parallel across multiple learned linear projections, allowing the model to attend to different representation subspaces simultaneously.",
    },
]
