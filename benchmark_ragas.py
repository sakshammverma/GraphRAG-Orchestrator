from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
from pathlib import Path

from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, answer_faithfulness

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
 
ragas_llm = LangchainLLMWrapper(ChatOllama(model="llama3", temperature=0))
ragas_embeddings = LangchainEmbeddingsWrapper(OllamaEmbeddings(model="nomic-embed-text"))

from graph_engine import app as graph_engine_app

CACHE_FILE = Path("pipeline_cache.json")
# Test Set
EVAL_QUESTIONS = [
    {
        "question": "What is the core mechanism of the Transformer architecture?",
        "ground_truth": "The Transformer relies entirely on self-attention mechanisms to compute representations of input and output sequences, dispensing with recurrence and convolutions.",
    },
    {
        "question": "What is multi-head attention?",
        "ground_truth": "Multi-head attention runs attention in parallel across multiple learned linear projections, allowing the model to attend to different representation subspaces simultaneously.",
    },
    {
        "question": "Why did the authors abandon recurrence in the Transformer?",
        "ground_truth": "Recurrence prevents parallelization during training and makes learning dependencies across long sequences harder. Self-attention handles this in constant path length.",
    },
    {
        "question": "What is the role of positional encoding in the Transformer?",
        "ground_truth": "Positional encodings inject information about token order into embeddings using sine and cosine functions of different frequencies, since the model has no recurrence or convolution.",
    },
    {
        "question": "What is scaled dot-product attention?",
        "ground_truth": "Scaled dot-product attention computes the dot products of queries with keys, scales by 1/sqrt(d_k) to prevent vanishing gradients, applies softmax, and multiplies by values.",
    },
    {
        "question": "How does the encoder in the Transformer work?",
        "ground_truth": "The encoder is composed of N identical layers, each with two sub-layers: multi-head self-attention and a position-wise feed-forward network, with residual connections and layer normalization.",
    },
    {
        "question": "What is the decoder's masking mechanism and why is it used?",
        "ground_truth": "The decoder uses a causal mask in self-attention to prevent positions from attending to subsequent positions, ensuring autoregressive generation.",
    },
    {
        "question": "What training data was used to train the Transformer?",
        "ground_truth": "The Transformer was trained on the WMT 2014 English-German dataset (4.5M sentence pairs) and the larger WMT 2014 English-French dataset (36M sentence pairs).",
    },
    {
        "question": "What BLEU score did the Transformer achieve on English-to-German translation?",
        "ground_truth": "The Transformer achieved 28.4 BLEU on the WMT 2014 English-to-German task, outperforming all previously reported models.",
    },
    {
        "question": "What optimizer and learning rate schedule was used?",
        "ground_truth": "Adam optimizer was used with beta1=0.9, beta2=0.98. The learning rate was warmed up linearly for 4000 steps then decayed proportionally to the inverse square root of the step number.",
    },
    {
        "question": "What is the feed-forward sub-layer in the Transformer?",
        "ground_truth": "Each encoder and decoder layer contains a position-wise feed-forward network with two linear transformations and a ReLU activation, with inner dimensionality 2048.",
    },
    {
        "question": "How does the Transformer handle variable-length sequences?",
        "ground_truth": "It uses padding masks to ignore pad tokens during attention and processes all positions simultaneously through self-attention rather than sequentially.",
    },
    {
        "question": "What is the difference between encoder self-attention and decoder cross-attention?",
        "ground_truth": "Encoder self-attention allows all positions to attend to all others. Decoder cross-attention attends over encoder output. Decoder self-attention is masked to prevent future token visibility.",
    },
    {
        "question": "What is label smoothing and how was it used in the paper?",
        "ground_truth": "Label smoothing with epsilon=0.1 was applied during training, which hurt perplexity but improved BLEU and accuracy by teaching the model to be less confident.",
    },
    {
        "question": "How many layers (N) does the base Transformer model use?",
        "ground_truth": "The base Transformer uses N=6 identical layers in both encoder and decoder stacks.",
    },
    {
        "question": "What is the model dimensionality d_model in the base Transformer?",
        "ground_truth": "The base model uses d_model=512 for embedding and sub-layer output dimensions.",
    },
    {
        "question": "How does the Transformer compare to RNNs in computational complexity?",
        "ground_truth": "Self-attention layers are O(1) sequential operations vs O(n) for RNNs, and O(n^2 * d) per-layer complexity vs O(n * d^2) for recurrent layers.",
    },
    {
        "question": "What is the Transformer big model?",
        "ground_truth": "The Transformer big model uses d_model=1024, 16 attention heads, feed-forward dimensionality 4096, and dropout 0.3, trained on 8 P100 GPUs for 300K steps.",
    },
    {
        "question": "What is dropout regularization in the Transformer?",
        "ground_truth": "Dropout is applied to attention weights and to the output of each sub-layer before adding the residual connection, with a rate of 0.1 for the base model.",
    },
    {
        "question": "What task other than translation was Transformer applied to?",
        "ground_truth": "The Transformer was applied to English constituency parsing, achieving competitive results with 40K training examples and even 17K examples (semi-supervised).",
    },
    {
        "question": "What is byte pair encoding and was it used in this paper?",
        "ground_truth": "The paper used byte-pair encoding with 37,000 tokens for English-German and a wordpiece vocabulary of 32,000 for English-French.",
    },
    {
        "question": "How many attention heads are used in the base Transformer?",
        "ground_truth": "The base Transformer uses h=8 parallel attention heads, each with d_k = d_v = d_model/h = 64 dimensions.",
    },
    {
        "question": "What are the benefits of self-attention over convolutional layers?",
        "ground_truth": "Self-attention connects all positions in O(1) sequential operations vs O(log_k(n)) for dilated convolutions, and provides more interpretable attention patterns.",
    },
    {
        "question": "What beam search parameters were used for decoding?",
        "ground_truth": "Beam search with beam size 4 and length penalty alpha=0.6 was used, with max output length = input length + 50.",
    },
    {
        "question": "What is a residual connection and why is it important?",
        "ground_truth": "Residual connections add the input of each sub-layer to its output before normalization, helping gradients flow during training and stabilizing deep networks.",
    },
    {
        "question": "What is layer normalization in the Transformer context?",
        "ground_truth": "Layer normalization normalizes activations across the feature dimension rather than batch dimension, applied after each sub-layer residual addition.",
    },
    {
        "question": "How does the model share weights between embedding and output layers?",
        "ground_truth": "The model uses the same weight matrix for the encoder/decoder input embeddings and the pre-softmax linear transformation, multiplied by sqrt(d_model).",
    },
    {
        "question": "What were the training hardware and time requirements?",
        "ground_truth": "The base model was trained for 100,000 steps on 8 NVIDIA P100 GPUs taking ~12 hours. The big model was trained for 300,000 steps taking ~3.5 days.",
    },
    {
        "question": "Why does the paper argue attention is more interpretable?",
        "ground_truth": "Attention weights can be visualized and analyzed, revealing that different heads learn syntactic and coreference relationships, providing insight into model behavior.",
    },
    {
        "question": "What is the maximum path length between any two positions in self-attention?",
        "ground_truth": "The maximum path length between any two positions in a self-attention layer is O(1), making it easier to learn long-range dependencies compared to O(n) in RNNs.",
    },
    {
        "question": "What is the purpose of the warmup steps in the learning rate schedule?",
        "ground_truth": "The 4000 warmup steps prevent excessively large updates in early training when parameters are randomly initialized and gradients are noisy.",
    },
    {
        "question": "What vocabulary size was used for English-German?",
        "ground_truth": "A shared source-target vocabulary of about 37,000 tokens using byte-pair encoding was used for English-German.",
    },
    {
        "question": "What is the WMT 2014 English-French BLEU result for the big Transformer?",
        "ground_truth": "The big Transformer achieved 41.0 BLEU on WMT 2014 English-French, a new state-of-the-art at the time, trained with dropout 0.1.",
    },
    {
        "question": "How does the Transformer decoder generate output tokens?",
        "ground_truth": "The decoder generates tokens autoregressively: at each step it attends to the encoder output and its own previously generated tokens, predicting one token at a time.",
    },
    {
        "question": "Why does scaling by 1/sqrt(d_k) matter in dot-product attention?",
        "ground_truth": "Without scaling, large d_k values push dot products into regions with extremely small softmax gradients. Dividing by sqrt(d_k) stabilizes gradient flow.",
    },
    {
        "question": "What is checkpoint averaging and was it used?",
        "ground_truth": "The authors averaged the last K checkpoints written at regular intervals rather than using just the final checkpoint, improving performance without additional training cost.",
    },
    {
        "question": "How does the Transformer achieve parallelism during training?",
        "ground_truth": "Self-attention computes representations for all positions simultaneously in matrix operations, unlike RNNs that must process tokens sequentially.",
    },
    {
        "question": "How are encoder-decoder attention keys and values derived?",
        "ground_truth": "In encoder-decoder attention, queries come from the previous decoder layer, while keys and values come from the encoder output, allowing every decoder position to attend to all encoder positions.",
    },
    {
        "question": "What is the significance of the Transformer for NLP history?",
        "ground_truth": "The Transformer demonstrated that attention alone can achieve state-of-the-art results, becoming the foundation for BERT, GPT and all modern LLMs.",
    },
    {
        "question": "What is position-wise application in feed-forward layers?",
        "ground_truth": "The FFN is applied to each position independently and identically: the same linear transformation is applied to each token position separately with shared weights across positions.",
    },
    {
        "question": "What is the difference between additive and dot-product attention?",
        "ground_truth": "Additive attention uses a feed-forward network to compute compatibility, while dot-product attention multiplies queries and keys directly. Dot-product is faster and more space-efficient.",
    },
    {
        "question": "What is the encoder-decoder attention mechanism?",
        "ground_truth": "Encoder-decoder attention allows decoder positions to attend over all encoder output positions, using encoder outputs as keys and values, and the decoder's previous layer output as queries.",
    },
    {
        "question": "What parsing dataset was used to test the Transformer?",
        "ground_truth": "The Wall Street Journal portion of the Penn Treebank was used for the English constituency parsing experiments.",
    },
    {
        "question": "Can positional encoding be replaced with learned embeddings?",
        "ground_truth": "Yes, the authors experimented with learned positional embeddings and found nearly identical results to sinusoidal encoding, so sinusoidal was kept for its ability to extrapolate to longer sequences.",
    },
    {
        "question": "What is the number of tokens per batch during training?",
        "ground_truth": "Each training batch contained a set of sentence pairs containing approximately 25,000 source tokens and 25,000 target tokens.",
    },
    {
        "question": "What is the perplexity of the base Transformer on English-German?",
        "ground_truth": "The base Transformer achieved 4.92 cross-entropy loss on the English-German development set.",
    },
    {
        "question": "What is epsilon in label smoothing for the Transformer?",
        "ground_truth": "The label smoothing value epsilon_ls = 0.1 was used, distributing 0.1 probability mass across all tokens in the vocabulary uniformly.",
    },
    {
        "question": "What is the difference between the base and big Transformer?",
        "ground_truth": "The big model has d_model=1024, 16 heads, ffn_dim=4096 and dropout 0.3, vs base model's d_model=512, 8 heads, ffn_dim=2048, dropout 0.1.",
    },
    {
        "question": "Why is O(1) path length important for learning?",
        "ground_truth": "Shorter paths between positions in the network mean gradients flow more easily and the model can learn long-range dependencies without gradient degradation.",
    },
    {
        "question": "How many P100 GPUs were used for training?",
        "ground_truth": "8 NVIDIA P100 GPUs were used for training both the base and big Transformer models.",
    },
    {
        "question": "What framework was the Transformer implemented in?",
        "ground_truth": "The Transformer was implemented in TensorFlow and trained using the tensor2tensor library.",
    },
]
assert len(EVAL_QUESTIONS) == 50


# Cache helpers 
def load_cache() -> dict:
    if CACHE_FILE.exists():
        return json.loads(CACHE_FILE.read_text())
    return {}
 
 
def save_cache(cache: dict):
    CACHE_FILE.write_text(json.dumps(cache, indent=2))
 

# ── Phase 1: Run one question through pipeline  
def run_single(item: dict) -> dict:
    q = item["question"]
    t0 = time.perf_counter()
    state = graph_engine_app.invoke({"original_question": q, "loop_count": 0})
    elapsed = time.perf_counter() - t0
    return {
        "question": q,
        "ground_truth": item["ground_truth"],
        "answer": state.get("final_answer", ""),
        "contexts": state.get("retrieved_chunks", []) or ["No context retrieved."],
        "elapsed_seconds": round(elapsed, 2),
    }
 
 
 