# Project Overview — everything you need to talk about it

Nissim Brami · Caching in LLMs · Ben-Gurion University · Prof. Gil Einziger

This document is the "explain-me-the-whole-project" single page. It is
written for someone who has **not read the paper** and has **not used
GPTCache**. Read the four sections below in order. Each takes about 10
minutes at a normal reading pace. If you can hold these four sections
in your head, you can explain the whole project to anyone.

Full detail lives in the rest of the repo:

- Task 1 (StreamingLLM presentation): `task1-presentation/`
- Task 2 (GDSF cost-aware eviction for GPTCache): `task2-final-project/`

---

## Part 1 — What the StreamingLLM paper is (10 minutes)

### 1.1 The problem, in one sentence

Large language models cannot **keep talking to you** for hours on end
without either running out of memory or producing gibberish. The
StreamingLLM paper explains why, and fixes it in about 50 lines of
code.

### 1.2 The setting

When you type into ChatGPT or a similar system, the model generates
one token at a time. To generate token *T + 1*, it has to look at
every token from *1* through *T*. Recomputing that lookup from scratch
every step would be quadratic (`O(T²)`), so the model keeps a
**KV cache** — a list of key/value vectors, one per past token, sitting
in GPU memory. This is the object the paper is about.

Two ugly facts about the KV cache:

1. It **grows linearly** with the length of the conversation. A 7-billion-parameter model at 4,000 tokens uses ~2 GB of KV cache in FP16. Long conversations eventually don't fit in GPU memory.
2. Language models were only pre-trained on some fixed window (say 4,000 tokens). If the cache holds more, the positional encodings inside the model go out of distribution and quality collapses.

So there is a hard cap on how long a conversation can be. Once you hit
it, you have three obvious options — and **all three are broken**.

### 1.3 The three broken baselines

The authors ran a very specific experiment: Llama-2-13B, reading a
65,000-token book from the PG19 corpus, and reported **perplexity**
(a measure of how well the model predicts the next word — lower is
better).

| Baseline                        | Cost    | Perplexity |
|---|---|---|
| **Dense attention** (keep all)  | O(T²)   | 5641 (already broken, then OOM)     |
| **Window attention** (keep last L) | O(T·L)  | **5158** — total gibberish          |
| **Window + recompute** (rebuild the cache every step) | O(T·L²) | 5.43 (correct, but far too slow) |

The mystery is line 2. Keeping only the most recent tokens should be
the cheapest reasonable strategy. But the model completely falls apart
— from a fluent Perplexity of 5-ish, to 5158. Why?

### 1.4 The observation — "attention sinks"

When the authors visualized where a language model looks (its
attention maps), they saw something odd: **starting from layer 3
onward, every attention head, in every layer, in every model they
tested (Llama-2, MPT, Falcon, Pythia), heavily attends to the very
first few tokens** — regardless of what those tokens actually say.
They named these initial tokens **attention sinks**.

Why? Because the softmax function that produces attention weights
**must sum to one**. That means the model always has a fixed budget of
attention mass to spend at every step, whether the context deserves it
or not. That excess mass has to land somewhere. And in a decoder-only
model, the tokens visible to *every* subsequent position are the
initial ones. They become the natural dumping ground.

The moment you evict the first token from the cache, you rip a huge
chunk out of the softmax denominator, and every remaining attention
weight in the row gets renormalized. The whole attention distribution
warps. Perplexity → 5158.

### 1.5 The clincher experiment

Is it the content of the first tokens that matters, or just their
*position*? The paper's cleanest ablation: replace the first four
tokens with linebreak characters (`\n`). Nothing else changes.

| Configuration                              | Perplexity |
|---|---|
| Window, no sinks                           | 5158.07    |
| **StreamingLLM** (4 real tokens + 1020 rolling) | **5.40**    |
| StreamingLLM with **linebreaks at positions 0–3** | 5.60 |

Almost identical. **The tokens' positions do the work, not their
content.** Sinks are structural, not semantic. This kills the "maybe
the first tokens carry irreplaceable meaning" theory in one row of a
table.

### 1.6 The fix (StreamingLLM)

Trivial once you've seen the mechanism:

1. Always keep the **first 4 tokens** in the cache. Never evict them.
2. Also keep the **last L tokens** in a rolling window (FIFO — the
   oldest one falls out when a new one arrives).
3. Everything between the sinks and the rolling window is discarded.
4. **Renumber position IDs inside the cache**, not in the original
   text. So if the cache holds tokens with original positions
   [0, 1, 2, 3, 6, 7, 8] and we're decoding position 9, the model
   sees positions [0, 1, 2, 3, 4, 5, 6, 7] — the sinks and the
   rolling window are made *adjacent* in position space.

That's the whole method. No fine-tuning. No new model architecture.
About 50 lines of PyTorch on top of a Hugging Face model.

### 1.7 The results

- **Perplexity flat past 4,000,000 tokens** on ten models: Llama-2
  {7B, 13B, 70B}, MPT {7B, 30B}, Falcon {7B, 40B}, Pythia {2.8B,
  6.9B, 12B}.
- **Up to 22.2× faster** per token than the "correct" baseline
  (sliding window with recomputation), because that baseline rebuilds
  the cache every step and StreamingLLM doesn't.
- **Streaming QA** (concatenating every ARC question into one stream)
  matches or slightly beats the one-shot-per-question oracle. Meanwhile
  the window-attention baseline collapses to 0.12% accuracy on 70B.

### 1.8 The limitations (which the paper is honest about)

- StreamingLLM does **not extend the context window**. If the answer to
  your question is older than the rolling window, StreamingLLM cannot
  find it.
- It underperforms simple truncation on long-document QA (LongBench).
- Bigger cache does not always mean lower perplexity.
- It does not compare with content-aware evictors like H2O or SnapKV
  — it evicts purely by position.

Adopted upstream within months: NVIDIA TensorRT-LLM, Hugging Face
Transformers, Intel Extension for Transformers, MLC LLM.

**One-line summary:** StreamingLLM is a small, robust, honest empirical
fix for a softmax quirk, and it is exactly the kind of paper a caching
class should present because it *is* a cache eviction policy.

---

## Part 2 — What we did for Task 1 (10 minutes)

### 2.1 The assignment

Prof. Einziger's Task 1: pick a recent paper on caching in LLMs,
present it to the class in a ~60-minute lecture. Deliverables: a
PowerPoint deck + speaker notes + optional supporting materials.

### 2.2 The paper we chose

*Efficient Streaming Language Models with Attention Sinks* by Xiao,
Tian, Chen, Han, Lewis. ICLR 2024. It is the paper summarized in
Part 1 above.

Why this paper for a caching audience:

- It is *literally* a paper about a cache eviction policy — the KV
  cache in a language model.
- It uses course vocabulary (capacity, eviction, hit, miss) without
  ever using the word "cache."
- It has a clean empirical result and an honest limits section, which
  makes for a good 60-minute talk.
- It connects cleanly to my final project (Task 2), because both
  projects are eviction policies on bounded caches — at different
  layers of the LLM serving stack.

### 2.3 What we built

A single directory, `task1-presentation/`, with:

- **The paper itself** — `papers/streaming-llm/streaming-llm.pdf`, plus
  our own `deep-read.md` (paragraph-by-paragraph summary with page
  references) and `critical-analysis.md` (our critique).
- **The deck** — `StreamingLLM_Presentation.pptx`, **46 slides**,
  built programmatically by `build_streaming_llm_deck.py` so every
  slide is version-controlled Python.
- **Speaker notes** — `notes/speaker-notes.md`, ~8,000 words, one
  section per slide, every numeric claim tied to a page of the paper.
- **Runnable demo** — `code/streaming_llm_demo.py`, ~200 lines of
  PyTorch + Hugging Face, implementing `SinkKVCache`, `WindowKVCache`,
  `DenseKVCache` side by side. Includes unit tests.
- **Reference decks** — `reference-decks/`, four prior decks from the
  course (SCALM, Chapter 1 - Cache, Intro, Extended pipeline cache),
  kept as a style reference.

### 2.4 How the deck is structured (60 minutes)

Twelve sections, each anchored to what the audience needs at that
point in the talk:

1. **Basics** (5 min) — remind the audience of LLM decoding and the KV
   cache. Anchor everything to Chapter 1 course vocabulary.
2. **Related work** (5 min) — where StreamingLLM sits in the 2024
   landscape (RoPE, ALiBi, Position Interpolation, YaRN, Longformer,
   LM-Infinite). Prevents "isn't this just Longformer?" questions.
3. **The observation** (6 min) — attention sinks, the softmax
   argument, and the linebreak clincher.
4. **The mechanism** (10 min) — 4 + L cache layout, position
   re-indexing, worked example, RoPE vs ALiBi variants, algorithm in
   one box.
5. **Sink-token pre-training** (4 min) — §3.3 refinement + the 160M
   caveat.
6. **Results** (12 min) — perplexity to 4M tokens, 22.2× speedup,
   streaming QA, StreamEval distance test.
7. **Limits deep-dive** (6 min) — LongBench, non-monotone cache size,
   positional-only ceiling.
8. **Modern alternatives** (3 min) — H2O, Scissorhands, FastGen,
   SnapKV, Keyformer.
9. **Implementation walkthrough** (4 min) — the actual PyTorch code.
10. **What I would change** (2 min) — four concrete extensions.
11. **Bridge to Task 2** (1 min) — the two-caches side-by-side.
12. **Q&A prep** (leftover time) — expected + hard questions.

### 2.5 One thing to be able to say out loud

If someone stops you in the hallway and asks what your talk is about:

> "A 2024 ICLR paper explaining why language models fall apart on long
> conversations, and a small trick — always keep the first 4 tokens in
> the cache plus a rolling window of the last L — that fixes it. It's
> a cache eviction policy, which is why we picked it for a caching
> class."

---

## Part 3 — What GPTCache is (10 minutes)

### 3.1 The problem, in one sentence

Calling GPT-4 twice with **semantically equivalent** prompts costs
twice as much and takes twice as long — even though the model would
give roughly the same answer. GPTCache puts a **response cache** in
front of the LLM so that "similar enough" prompts return the cached
answer instead of hitting the paid API.

### 3.2 Where it lives

- Repository: **`zilliztech/GPTCache`** on GitHub. Company: Zilliz
  (the people behind the Milvus vector database).
- License: **Apache 2.0**.
- Stars: **~12,000**.
- First release: mid-2023, riding the wave of production LLM apps.
- Language: Python.
- Position in the stack: sits *between your application and the
  OpenAI/Anthropic/HuggingFace API*. You wrap your normal call in a
  GPTCache-aware call and it decides whether to serve from cache or
  actually make the API request.

### 3.3 How the caching works

GPTCache is not a normal key-value cache because prompts are almost
never *exactly* equal. Two things need to be the same for a cache hit:

1. **The prompt has to be semantically similar** to a previously
   cached prompt. GPTCache does this with a **vector store**: embed
   the prompt into a dense vector (via SentenceTransformers, OpenAI
   embeddings, etc.), and look up the nearest neighbours in a vector
   database (FAISS, Milvus, ChromaDB, etc.). If the distance is below
   a threshold, it's a match.
2. **The cached answer has to still be valid.** Every entry has
   metadata (model, temperature, system prompt) and only matches with
   matching metadata are used.

There are three main components you interact with:

- **`CacheBase`** — where the prompts and answers are stored
  (SQLite, PostgreSQL, MySQL, MongoDB, DuckDB…).
- **`VectorBase`** — where the embeddings are stored for similarity
  lookup (FAISS, Milvus, Chroma, Hnswlib, PGVector…).
- **`EvictionBase`** — the eviction policy for when the caches get
  full. And this is where our contribution goes.

### 3.4 The eviction question

Before our change, `EvictionBase(name="memory", policy=P)` accepted
four options for `P`:

- **LRU** (Least Recently Used) — the default. Evict whichever entry
  hasn't been touched in the longest time.
- **LFU** (Least Frequently Used) — evict whichever entry has the
  smallest hit count.
- **FIFO** — evict in insertion order, no matter what.
- **RR** (Random Replacement) — evict a uniformly random entry.

These are the canonical page-cache policies from an operating systems
textbook. They all share a problem for LLM traffic: **they treat every
cached entry as equally valuable**. But in an LLM cache, regenerating
one entry might cost 20 cents (a long GPT-4 completion) and
regenerating another might cost 0.02 cents (a short GPT-3.5 answer).

LRU has no way to know that. LFU has no way to know that. So on
"realistic" LLM traffic — where cost varies by 100x between entries —
these policies **evict expensive answers as readily as cheap ones**,
and every avoidable eviction of an expensive answer costs real money
next time someone asks the same question.

### 3.5 What GPTCache already did well

- **Backend flexibility.** You can swap SQLite for MongoDB, or FAISS
  for Milvus, without touching the application code.
- **Similarity strategies.** Exact match, kNN, distance thresholds —
  all pluggable.
- **Integration with major LLM frameworks.** LangChain, LlamaIndex,
  and the OpenAI SDK all support GPTCache as a drop-in.
- **Working reference eviction policies.** LRU/LFU/FIFO/RR out of the
  box, all wired through a clean `EvictionBase` factory.

### 3.6 What was missing

- **No cost-aware eviction policy.** Every existing option treats
  entries as fungible. There is no built-in way to say "this response
  cost 10× more to generate than that one; don't evict it first."
- **No size-aware eviction policy.** Long responses take up more
  memory (or more Redis quota) than short ones, but nothing in the
  eviction path knows about that either.

### 3.7 The upstream contribution point

Because `EvictionBase` is already a factory with a clean `policy=…`
argument, adding a new policy is exactly the kind of contribution
maintainers welcome. That's the surface where our GDSF policy plugs
in.

---

## Part 4 — What we did for Task 2 (10 minutes)

### 4.1 The assignment

Prof. Einziger's Task 2: an open-source contribution and a research
paper on a caching-in-LLMs topic. Deliverable: (a) a real
pull-request-ready contribution to an upstream repo, (b) a
7-page ACM-style paper written up in `acmart` sigconf.

### 4.2 What we chose

Add a new eviction policy to GPTCache: **GDSF — Greedy-Dual-Size-
Frequency**. Written by Cao & Irani (USITS 1997) and Cherkasova (HP
Labs 1998) for **web proxy caches** in the late 1990s. It's a
weighted-caching classic that has never been ported to LLM caches.

### 4.3 The GDSF idea, in one line

Instead of evicting by "last touched" or "count of hits," GDSF assigns
each entry a **priority score** that folds in three signals:

```
Priority(i) = L + freq(i)^α · cost(i)^β / size(i)
```

- `freq(i)` — how many times this entry has been requested. Rewards
  hot entries.
- `cost(i)` — how expensive it was to generate. Rewards entries you
  really don't want to regenerate. In LLM terms: expensive GPT-4
  answers.
- `size(i)` — how much space this entry takes. Penalizes bloat.
- `L` — a global "clock" that inflates every time an item is evicted.
  It grows monotonically. It lets recent entries stay competitive
  even if their frequency is still low. This is the "aging" mechanism
  that keeps GDSF from getting stuck.
- `α`, `β` — exponents that let you tune how much frequency vs cost
  matters.

Evict the entry with the **lowest priority**. Cao & Irani proved this
is competitively-optimal (ratio *k*) for weighted online caching among
deterministic algorithms.

### 4.4 What we built

Two things:

1. **The upstream contribution** — a real PR-ready branch on my fork
   of `zilliztech/GPTCache`.
2. **The research paper** — a 7-page ACM `acmart` sigconf write-up in
   `task2-final-project/report-latex/report.pdf`.

Plus a substantial amount of supporting code:

- `task2-final-project/code/src/cost_aware_eviction/` — 1,060 lines of
  production Python. The heart is `gdsf_eviction.py`, which is a
  clean implementation of GDSF backed by an indexed min-heap
  (O(log n) priority updates) with a single `RLock` for thread
  safety. Wired into GPTCache's `EvictionBase` factory as
  `policy="GDSF"`.
- `task2-final-project/code/tests/` — **259 unit tests**, 3,332 lines,
  all passing. Cover factory wiring, backwards-compat API,
  cost-aware ranking, size penalization, clock monotonicity,
  tie-breaking, input validation.
- `task2-final-project/code/benchmarks/` — 6 workload generators
  (uniform-cost, high-variance-cost, bursty, size-varying,
  adversarial anti-LRU, Zipf-fits-in-cache), the runner, the metrics.
- `task2-final-project/code/scripts/compute_statistics.py` — the
  paired-t + Bonferroni + BCa bootstrap analysis that produced the
  numbers in the paper.

### 4.5 The API design

Preserved backwards compatibility while adding a new capability:

- `EvictionBase("memory", policy="GDSF")` — the factory dispatches to
  the new class automatically. Existing users see no change.
- `put(keys)` — still works. Defaults `cost=1, size=1`, which reduces
  GDSF to LFU-with-clock-aging (a strict improvement over pure LFU on
  bursty traffic).
- `put_with_metadata([(key, cost, size), ...])` — the new extension.
  Callers who *know* the dollar cost and response size get the full
  cost-aware behaviour.

### 4.6 The benchmark

**3,600 runs** total: 30 random seeds × 6 workloads × 4 cache sizes ×
5 policies (LRU, LFU, FIFO, RR, GDSF). Every run reports:

- **CWHR** — Cost-Weighted Hit Rate (a hit on an expensive entry
  counts more than a hit on a cheap one).
- **Dollar spend** — total regeneration cost.

Then, statistical analysis on the paired policy comparisons: paired-t
tests, Bonferroni-corrected p-values, and 95% BCa bootstrap
confidence intervals with 10,000 resamples and seed pinned to
`20260721` for reproducibility.

### 4.7 The results

| Workload             | Δ CWHR (GDSF − LRU) | 95% BCa CI       | Bonf. p     | Dollar Δ  |
|----------------------|---------------------|------------------|-------------|-----------|
| high-variance cost   | **+0.1190**         | [+0.102, +0.136] | 8.6e-26     | **+25.7%** |
| bursty               | **+0.1626**         | [+0.150, +0.175] | 5.9e-49     | **+32.3%** |
| size-varying         | **+0.1632**         | [+0.153, +0.173] | 1.6e-58     | **+91.0%** |
| adversarial anti-LRU | **+0.1419**         | [+0.100, +0.189] | 3.4e-8      | **+18.8%** |
| uniform cost         | +0.00003            | [−0.0004, +0.0005] | 1.00      | +0.005%    |
| Zipf, fits in cache  | −0.0003             | [−0.0005, −0.0002] | 4.9e-4    | −0.037%    |

Read the numbers this way:

- On the four workloads where a cost-aware policy *should* help, GDSF
  wins **by 18.8% to 91.0% in dollar savings** vs LRU. The
  Bonferroni-corrected p-values are astronomically small. The
  confidence intervals do not overlap zero.
- On the two workloads where a cost-aware policy *can't* help
  (uniform cost, cache-fits-Zipf), GDSF is statistically
  indistinguishable from LRU. No meaningful regression.

### 4.8 The upstream PR

Branch `feat/gdsf-cost-aware-eviction` on my fork
`nissimbrami/GPTCache`, ready to open against `zilliztech/GPTCache`.
3 files changed, +389 lines:

- `gptcache/manager/eviction/gdsf_eviction.py` (new — the policy).
- `gptcache/manager/eviction/manager.py` (edit — wire `policy="GDSF"`
  through the factory).
- `tests/unit_tests/manager/test_gdsf_eviction.py` (new — 8 unit tests).

Draft body of the PR is in
`task2-final-project/GPTCACHE_PR.md`. The PR itself is a single-click
away.

### 4.9 One thing to be able to say out loud

If someone stops you in the hallway and asks what your final project
is about:

> "I ported GDSF — a classic weighted-caching policy from the 1990s
> for web proxies — into GPTCache, the ~12,000-star Zilliz project
> that puts a semantic response cache in front of LLM APIs. On
> realistic LLM workloads where regeneration cost varies by 100×
> between entries, my policy saves **up to 91%** more money than the
> default LRU with p < 1e-58. Contribution is a PR-ready branch on
> zilliztech/GPTCache; the write-up is a 7-page ACM sigconf paper."

---

## Cross-reference: how Task 1 and Task 2 relate

Two caches, one idea:

| Layer                  | Task 1 — StreamingLLM             | Task 2 — GDSF on GPTCache        |
|---|---|---|
| What is cached         | Key/Value vectors of past tokens  | Whole (prompt → response) pairs  |
| Where in the stack     | Inside the model, per attention layer | In front of the model, at the API layer |
| Capacity unit          | S sinks + L rolling tokens        | N entries or M bytes             |
| Eviction signal        | Position (first S + last L)       | Frequency × cost / size + clock  |
| Failure mode when dumb | PPL 5158 (softmax collapse)       | Overspend on regeneration $$$    |
| Fix                    | Positional eviction rule          | Cost-aware eviction rule         |

Both caches are bounded. Both suffer under naive eviction. Both admit
a small, elegant policy fix that doesn't require retraining the model.
That is the through-line for the whole project and it is what the
last slide of the presentation says.

---

## Where to look for more detail

- **Full paper summary with page references:** `task1-presentation/papers/streaming-llm/deep-read.md`
- **Critical analysis of the paper:** `task1-presentation/papers/streaming-llm/critical-analysis.md`
- **Slide-by-slide speaker notes:** `task1-presentation/notes/speaker-notes.md`
- **The GDSF plugin itself:** `task2-final-project/code/src/cost_aware_eviction/gdsf_eviction.py`
- **The research paper PDF:** `task2-final-project/report-latex/report.pdf`
- **The upstream PR draft:** `task2-final-project/GPTCACHE_PR.md`
