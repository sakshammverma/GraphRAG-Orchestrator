
import json
import random
import time
from locust import HttpUser, between, task, events
QUERY_POOL = [
    "What is the core mechanism of the Transformer architecture?",
    "How does multi-head attention differ from single-head attention?",
    "Why did the Transformer abandon recurrence?",
    "What is the role of positional encoding?",
    "Explain scaled dot-product attention step by step.",
    "How does the encoder process input sequences?",
    "What is the decoder's causal masking mechanism?",
    "What optimizer and learning rate schedule was used to train the Transformer?",
    "How does the Transformer handle variable-length sequences?",
    "What is the difference between encoder self-attention and decoder cross-attention?",
    "Explain label smoothing and its effect on Transformer training.",
    "What is the feed-forward sub-layer in the Transformer?",
    "How does the Transformer compare to LSTMs in computational complexity?",
    "What BLEU scores did the Transformer achieve on WMT benchmarks?",
    "How does byte-pair encoding work and why is it used?",
    "What is the significance of residual connections in the Transformer?",
    "How does the Transformer model long-range dependencies?",
    "What is layer normalization and why is it used instead of batch normalization?",
    "What is the role of the final linear + softmax layer in the decoder?",
    "How does the Transformer achieve parallelism during training?",
    "What is the maximum path length between positions in self-attention?",
    "What is the Transformer big model configuration?",
    "How are embeddings scaled in the Transformer input?",
    "What is the perplexity achieved by the Transformer on English-German?",
    "Why does scaling by 1/sqrt(d_k) prevent vanishing gradients in attention?",
]

class ResearchUser(HttpUser):
    wait_time = between(1, 3)   

    @task
    def research_query(self):
        query = random.choice(QUERY_POOL)
        payload = {"query": query}

        start = time.perf_counter()
        first_byte_time = None
        full_response_chunks = []
 
        with self.client.post(
            "/api/research",
            json=payload,
            stream=True,
            catch_response=True,
        ) as response:
            if response.status_code != 200:
                response.failure(f"Non-200 status: {response.status_code}")
                return

            try:
                for line in response.iter_lines():
                    if first_byte_time is None:
                        first_byte_time = time.perf_counter()

                    if not line:
                        continue
 
                    if isinstance(line, bytes):
                        line = line.decode("utf-8")

                    if line.startswith("data: "):
                        raw_json = line[len("data: "):]
                        try:
                            event = json.loads(raw_json)
                            full_response_chunks.append(event)
                            if event.get("done"):
                                break
                        except json.JSONDecodeError:
                            pass

                elapsed = time.perf_counter() - start

                if not full_response_chunks:
                    response.failure("No SSE events received")
                    return

                final_event = full_response_chunks[-1]
                if not final_event.get("done"):
                    response.failure("Stream ended without 'done' event")
                    return
 
                response.success()

            except Exception as exc:
                response.failure(f"Streaming error: {exc}")

    @task(weight=1)
    def health_check(self):
        """Lightweight health check to verify server responsiveness."""
        self.client.get("/health")
 
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    print("\n" + "=" * 60)
    print("LOCUST LOAD TEST STARTED")
    print("Target: p50 latency < 2400ms at 20 concurrent users")
    print("=" * 60 + "\n")


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    stats = environment.stats.total
    p50 = stats.get_response_time_percentile(0.5)
    p95 = stats.get_response_time_percentile(0.95)
    p99 = stats.get_response_time_percentile(0.99)
    rps = stats.current_rps
    failures = stats.num_failures
    total = stats.num_requests

    print("\n" + "=" * 60)
    print("LOCUST LOAD TEST RESULTS")
    print("=" * 60)
    print(f"  Total requests  : {total}")
    print(f"  Failures        : {failures} ({failures/max(total,1)*100:.1f}%)")
    print(f"  Requests/sec    : {rps:.2f}")
    print(f"  p50 latency     : {p50:.0f}ms  {'✓ PASS' if p50 < 2400 else '✗ FAIL (target < 2400ms)'}")
    print(f"  p95 latency     : {p95:.0f}ms")
    print(f"  p99 latency     : {p99:.0f}ms")
    print("=" * 60 + "\n")
 
    result = {
        "total_requests": total,
        "failures": failures,
        "failure_rate_pct": round(failures / max(total, 1) * 100, 2),
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "target_p50_ms": 2400,
        "p50_pass": p50 < 2400,
    }
    import json
    from pathlib import Path
    Path("locust_summary.json").write_text(json.dumps(result, indent=2))
    print("Results saved → locust_summary.json")
    