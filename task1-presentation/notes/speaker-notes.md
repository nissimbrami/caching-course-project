# SCALM — Presentation Speaker Notes

**Paper:** SCALM: Towards Semantic Caching for Automated Chat Services with Large Language Models
**Authors:** Chen Wang, Xinyi Feng, Ao Cheng, Junchen Wang
**Venue:** arXiv:2406.00025 (June 2024)
**Presenter:** Nissim Brami (`nissimbrami@post.bgu.ac.il`)
**Slot:** #8 on the course paper list

## One-sentence pitch

SCALM is a semantic cache that groups paraphrases of the same underlying
intent into *clusters* and stores one representative response per cluster,
which roughly triples the hit rate versus a plain semantic cache on real chat
workloads.

## What is being cached and why does it work

- **What:** LLM chat responses, keyed by the *semantic cluster* of the incoming prompt.
- **Why it works:** Real chat traffic has strong intent locality — the same
  underlying question comes up many times in many wordings. Exact-match
  caches see all wordings as different. A plain semantic cache stores every
  paraphrase separately and fights itself for capacity. SCALM collapses all
  paraphrases of the same intent into one entry, so the cache spends
  capacity on *distinct topics*, not on paraphrase duplicates.

## The mechanism, precisely

1. **Embedding.** Each prompt is embedded with a small encoder (sentence
   transformer).
2. **Online clustering.** A streaming k-means-style step assigns the
   embedding to the nearest existing cluster if the distance is below radius
   `r`, or opens a new cluster otherwise.
3. **Centroid update.** The cluster centroid is updated via exponential
   moving average with decay `λ` — new members nudge it, old members anchor
   it.
4. **Hit check.** A prompt is served from cache when its distance to the
   assigned cluster centroid is below the response threshold `τ`.
5. **Eviction.** Priority ≈ `(freq × cost) / age`. Low-value clusters die
   first.
6. **Drift.** When intra-cluster variance rises above a threshold, a
   re-clustering pass runs on the affected region.

## Positioning in course vocabulary

- **Cache target:** response-level, semantic-key. Not KV-tensor caching.
- **Eviction family:** cost-frequency weighted priority — a coarse cousin of
  Greedy-Dual-Size-Frequency (GDSF), which is exactly the family I use in my
  final project.
- **Admission control:** implicit — a new cluster is only opened when a
  prompt is sufficiently isolated from existing clusters. That is
  structurally similar to the admission filter in W-TinyLFU.
- **Neighbours in the literature:**
  - GPTCache (baseline for my project) — semantic cache without clustering.
  - MeanCache (paper #7) — user-centric semantic cache; different angle.
  - Cache-me-if-you-can — KV-level, orthogonal.

## Headline numbers to quote

From the paper's evaluation on real chat traces at cache size 10 000:

- Exact-match cache: ≈ 12 % hit rate.
- Vanilla semantic cache: ≈ 28 % hit rate.
- **SCALM: 40–45 % hit rate.**
- ≈ 60 % p95 latency reduction vs no-cache baseline.
- ≈ 55 % LLM-API dollar reduction at steady state.
- Sweet-spot similarity threshold `τ ∈ [0.85, 0.92]`.

## Strengths I plan to highlight

- Real workload, real numbers — not a synthetic microbenchmark.
- Bounded memory: `k` clusters, not `n` prompts.
- Handles paraphrase / typo naturally, which is where exact match dies.
- Drift-aware — degrades gracefully as topics shift.

## Limitations I plan to be honest about

- **Threshold sensitivity.** τ too low ⇒ semantic false hits ⇒ wrong
  answer. That risk is worse than a cache miss.
- **All-clusters-equal.** SCALM's eviction is cost-frequency weighted but
  its *admission* treats all clusters the same. A cost-aware admission
  filter would fit naturally.
- **Cold start.** Clustering needs a warmup before hit rate lifts.
- **Response quality validation** in the paper leans on LLM-as-judge
  scoring, which has its own biases.

## Improvements I would propose (link to my project)

1. Replace SCALM's coarse cost-frequency ratio with a **proper GDSF priority**:
   `Priority(c) = Clock + freq(c)^α · cost(c)^β / size(c)`.
   This is exactly the algorithm I implemented for my final project on top of
   GPTCache — the theoretical lineage carries over cleanly.
2. **Cost-aware admission.** Currently any new isolated prompt opens a
   cluster. Weighting admission by predicted regeneration cost would keep
   expensive intents even at low frequency.
3. **Dollarize the cost signal.** Use per-model pricing (tokens × rate)
   rather than a proxy — that's what makes GDSF's `cost` term interpretable
   in dollars.
4. **Two-tier arrangement.** Hot exact-match layer catches the trivial
   repeats fast; SCALM layer catches paraphrases. My project's benchmarks
   already show single-tier GDSF wins on cost-heterogeneous workloads;
   layering with SCALM should compound the effect.
5. **Better drift detection.** Statistical process control (CUSUM, EWMA on
   centroid variance) instead of an ad-hoc threshold.

## Anticipated Q&A

**Q: Is SCALM safe? Could a false hit return a wrong answer?**
Yes, that's the fundamental risk of any semantic cache. Mitigated by (a)
tuning `τ` conservatively, (b) LLM-as-judge validation for high-stakes
queries, (c) fallback to real LLM call on low confidence.

**Q: Why not just use a larger vanilla semantic cache?**
Because near-duplicates crowd out distinct intents. The paper's ablation
shows that at any capacity, clustering beats storing individual embeddings.

**Q: How does this differ from MeanCache (paper #7)?**
MeanCache is user-centric — clusters per-user context. SCALM is
service-centric — global clusters across all users. Different assumptions
about locality.

**Q: What's the O(·) cost per operation?**
Embed: fixed cost of the encoder. Nearest-cluster ANN: O(log k) or
approximate O(1) depending on index. Eviction: O(log k) via a priority
queue. All dominated in practice by the embedding step.

**Q: Does this connect to your final project?**
Directly. My project replaces GPTCache's LRU with a full GDSF eviction
policy. SCALM adds a semantic-clustering *layer* on top of the cache; my
work improves the *eviction rule* underneath any such layer. Both changes
attack the same underlying inefficiency — cost-oblivious eviction — from
different sides of the stack.
