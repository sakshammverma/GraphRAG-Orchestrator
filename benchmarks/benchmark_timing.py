import time
from core.graph_engine import app

QUERIES = [
    "What is the core mechanism of the Transformer architecture?",
    "How does multi-head attention differ from single-head attention?",
    "What is the role of positional encoding?", 
    "What is the significance of residual connections in the Transformer?",
    "How does the Transformer model long-range dependencies?",
    "What is layer normalization and why is it used instead of batch normalization?",
    "What is the role of the final linear + softmax layer in the decoder?",
    "How does the Transformer achieve parallelism during training?",
    "What is the maximum path length between positions in self-attention?",
    "What is the Transformer big model configuration?",
    "How are embeddings scaled in the Transformer input?",
]

results=[]
for i, query in enumerate(QUERIES, start=1):
    start = time.perf_counter()
    app.invoke({"original_question": query, "loop_count": 0})
    end = time.perf_counter()
    elapsed= end-start
    print(f"Query no. {i}\nTotal time = {elapsed:.2f}s")
    results.append({"query": query, "elapsed_seconds": round(elapsed, 2)})


import json
with open("timing_results.json", "w") as f:
    json.dump(results, f, indent=2)

avg = sum(r["elapsed_seconds"] for r in results) / len(results)
print(f"\nAverage pipeline time: {avg:.2f}s")
print(f"Manual research baseline: 900s (15 mins)")
print(f"Reduction: {round((1 - avg/900)*100)}%")
    
