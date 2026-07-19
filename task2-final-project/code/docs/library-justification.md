# Library Justification: GPTCache

**Author:** Nissim Brami · Caching in LLMs · Ben-Gurion University

## Choice

I chose **GPTCache** (Zilliz, MIT license, ~8 k GitHub stars, v0.1.x) as
the baseline framework for this project.

GPTCache is a Python library that caches LLM API responses using
semantic similarity: incoming prompts are embedded, compared against
stored embeddings (via FAISS or Milvus), and if similarity exceeds a
threshold the cached response is returned. This eliminates redundant
LLM calls, cutting both latency and dollar cost.

## Why GPTCache and not something else

The course lists four candidate frameworks. My reasoning for each:

| Library | Language | Cache target | Eviction | Modularity | Rejected because |
|---|---|---|---|---|---|
| **GPTCache** | Python | API responses | LRU / FIFO | Plugin (`EvictionBase`) | *chosen* |
| LMCache | Python | GPU KV tensors | GPU-tied | Low | Needs a GPU-heavy inference stack; not reproducible on a laptop |
| vLLM | Python | KV tensors | Coupled to scheduler | Monolithic | 78 kLoC engine; caching is intertwined with paged attention |
| Caffeine | Java | generic objects | W-TinyLFU | High | Already near-optimal; also wrong language for the class ecosystem |
| LangChain Cache | Python | API responses | — | Minimal | Has no real eviction to extend |

GPTCache is the only option that combines:

1. A clean plugin architecture — the `EvictionBase` abstract class is small,
   documented, and every policy in the codebase already subclasses it.
2. **Weak defaults** (LRU + FIFO). LLM workloads have heterogeneous per-query
   costs, so recency-only eviction throws away exactly the kind of value
   cost-aware policies capture. This is where the improvement lives.
3. Pure Python, MIT-licensed, no GPU dependency — trivially reproducible
   inside a Docker container for the grader.

## Default eviction policy

`gptcache.manager.eviction.MemoryCacheEviction` supports two strategies:

- **LRU** — evict the least-recently-accessed key.
- **FIFO** — evict in insertion order.

Both are recency-only. Neither considers query cost, per-token pricing,
frequency, or entry size. Given that the whole *point* of an LLM
cache is to save API dollars, this is a real gap and the natural place
to add a Greedy-Dual-Size-Frequency (GDSF) variant.

## Ease of modification

Adding a new eviction policy takes three steps:

1. Subclass `EvictionBase`.
2. Implement `put()`, `get()`, and `is_evict()`.
3. Register the class in the eviction manager factory.

My `GDSFEvictionPlugin` in `src/cost_aware_eviction/gptcache_plugin.py`
does exactly this — the enhancement lives as a self-contained module
with zero modifications to upstream GPTCache code.

## Relevance to course themes

The course covers LRU, LFU, and the Greedy-Dual family (Greedy-Dual, GD-Size,
GDSF, LFU-DA). GDSF was introduced by Cherkasova (1998) for web proxy
caches, where fetching cost is heterogeneous — the same structural
argument applies word-for-word to LLM API caches. So this project
directly instantiates classical eviction theory in a domain where the
"cost per miss" is measurable in dollars.

## Reproducibility features I rely on

- `pip install gptcache` (no build step, no CUDA)
- Runs on commodity hardware (`faiss-cpu`, no GPU)
- MIT license permits fork and redistribution
- Deterministic given a fixed embedding model and seed
