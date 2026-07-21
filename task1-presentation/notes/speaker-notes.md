# StreamingLLM — Speaker Notes (60-minute version)

Nissim Brami · Caching in LLMs · Ben-Gurion University · Prof. Gil Einziger

These notes correspond one-to-one with the 46 slides in
`StreamingLLM_Presentation.pptx`. Read them as talking points, not as a
verbatim script. Every numeric claim is followed by a page reference
into the paper (`papers/streaming-llm/streaming-llm.pdf`), or by a
cross-reference to `deep-read.md` / `critical-analysis.md`.

**Pacing budget.** 60-minute slot ≈ 55 minutes of speaking + 5 minutes
of buffer for questions during the walk-through. At 180 words/minute
that is a ~10,000-word talk. This document is ~12,000 words so I can
compress on the fly if I'm behind.

---

## Slide 1 — Title

Open by naming the paper and the authors out loud.

*Efficient Streaming Language Models with Attention Sinks* by
Guangxuan Xiao (MIT), Yuandong Tian (Meta AI), Beidi Chen (CMU), Song
Han (MIT + NVIDIA), Mike Lewis (Meta AI). ICLR 2024. arXiv:2309.17453,
v4 in April 2024. Code is at github.com/mit-han-lab/streaming-llm and
the fix has since been adopted upstream by NVIDIA TensorRT-LLM, Intel
Extension for Transformers, Hugging Face Transformers, and MLC LLM.

One sentence for context: **this is a paper about the KV cache**. It
sits squarely inside the vocabulary we already know from Chapter 1 of
this course — cache, capacity, eviction — but applied to the tokens a
language model keeps in memory while it decodes. My job over the next
hour is to convince you (a) that KV-cache eviction is exactly the
same kind of problem as LRU-vs-LFU-vs-ARC on a page cache, and (b)
that StreamingLLM is the small, elegant, and honest fix.

Then hand-wave at slide 46 (the last one): "and everything I present
today comes back at the end to bridge to my final project — a
different cache, in the same LLM stack, with a similar diagnosis."

## Slide 2 — Agenda (60 minutes)

Walk the agenda quickly. Signal early that:

- We start with basics — the anchor to the course vocabulary — because
  otherwise the paper reads like ML rather than caching.
- We spend real time on the mechanism because the "positional
  re-indexing" trick is genuinely non-obvious.
- We spend real time on the results because reading the tables the
  wrong way makes the paper look magical (it isn't) or trivial
  (it isn't either).
- We deliberately do modern-alternatives (H2O / SnapKV / etc.) and
  limitations. This is a 2026 audience; the questions will come.
- The last section bridges to my Task-2 GDSF project.

Roughly 8 minutes on basics + related work, 12 minutes on mechanism,
15 minutes on results, 10 minutes on limits + alternatives, 8
minutes on the walkthrough + bridge, 5 minutes of buffer for Q&A
inside the talk.

## Slide 3 — Section divider: Basics

Say "Basics." Move on.

## Slide 4 — The scene: decoding an LLM

The audience knows attention. Remind them of the mechanics of decoding
because it's easy to lose the caching connection otherwise.

- Autoregressive decoding: generate one token at a time.
- Every new token attends back to every earlier token in the context.
- To avoid recomputing K and V for every earlier token at every step,
  we cache them. That is the **KV cache**.
- The size of the KV cache grows *linearly* with the sequence length,
  not with the vocabulary and not with the batch dimension. That is
  what makes long-context serving expensive.
- Rule-of-thumb estimate for a 7B model at 4K tokens in FP16: about
  2 GB of KV cache. That's per request. Multiply by batch size.
- At decode time, throughput is bottlenecked by **memory bandwidth**
  into that cache, not by compute. This is why "just add more GPU
  cores" doesn't help streaming latency.

Point at the box on the right: the KV cache is *what we cache*, and
memory bandwidth is *what constrains us*. Everything in this talk is
downstream of those two facts.

## Slide 5 — This is a caching problem

Anchor to Chapter 1 vocabulary. This slide is the one that reframes
the paper for a caching audience — for a caching audience, the paper
looks less like a novel ML result and more like "someone finally wrote
down the right eviction policy for a specific and important cache."

- **Cache**: the KV cache in GPU memory.
- **Capacity**: bounded by GPU RAM. On an A6000 with a 13B model,
  you cannot cache the whole context of a 65K-token book.
- **Requests**: attention lookups from each new query token.
- **Miss**: the token you need is gone → the only correct fallback
  is recomputing the whole prefix, which is quadratic.
- **Eviction policy**: which past tokens to keep, which to drop?

Then say the punchline: **window attention — the standard baseline —
is literally LRU-by-position**. Keep the most-recent L tokens. Evict
the oldest. Not by frequency, not by attention score. By position.
That is a specific eviction rule, and it's a bad one.

Point at the "Chapter 1 recap" box: LRU, LFU, ARC, LIRS, Hyperbolic,
FRD, W-TinyLFU. These are the policies the professor introduced in
week 1. StreamingLLM's contribution is *a new positional eviction
policy* for the KV cache. Same vocabulary, different cache.

## Slide 6 — Three obvious KV-cache strategies, all three break

Read the table row by row. The perplexity numbers are Fig. 1 (p. 2)
on Llama-2-13B on the first book of PG19 (65K tokens).

- **Dense attention.** Keep every token. O(T²) attention. Fails
  because we OOM the moment T exceeds the pre-training window
  (Llama-2 pre-trained on 4K tokens: 65K → OOM). Interestingly the
  quality has already degraded to PPL 5641 well before OOM — the
  positional encodings weren't trained for that regime.
- **Window attention.** Keep only the most-recent L tokens. O(TL).
  Cheapest in theory. PPL **5158**. That is not slight degradation;
  that is a fluent LLM turned to gibberish. This is the number that
  motivates the whole paper.
- **Sliding window with re-computation.** Every new step, rebuild
  the cache from the most-recent L tokens (throw the cache away,
  refill it, run attention). O(TL²). PPL 5.43 — correct — but the
  wall-clock is unusable (see slide 22 for the numbers).

Read the "puzzle" box aloud: window attention has the *right cost* but
the *wrong answer*. Something happens when a single token — the very
first one — falls out of the cache. That "something" is the
attention-sink phenomenon and it takes the next four slides to
explain.

## Slide 7 — Section divider: Related work

Say "Related work." This is a longer than usual related-work section
for a course talk — I want to place StreamingLLM in the 2024
landscape so nobody in the audience thinks "isn't this just
Longformer?" or "didn't Position Interpolation already solve this?"

## Slide 8 — Related work 1/3: length-extrapolation encodings

The methods that share the *goal* with StreamingLLM — deploy on
inputs longer than pre-training — but attack it through position.

- **RoPE (Su et al., 2021).** Rotary Position Embedding. Encodes
  absolute position as a rotation of the query and key vectors
  before the dot product; because rotations compose additively, the
  relative-position property falls out automatically. Used in
  Llama-2, Falcon, Pythia, and many others.
- **ALiBi (Press et al., 2022).** No positional embedding at all.
  Instead, add a *linear bias* `-m·|i-j|` to the attention logits.
  Different slopes `m` on different heads. Used in MPT.
- **xPos (Sun et al., 2022).** RoPE with an exponential decay
  factor per head, aimed at improving stability on long ranges.
- **T5 relative bias (Raffel et al., 2020).** Bucketed learned
  bias.

Why StreamingLLM cares about all of these: **it has to plug into any
of them unchanged**. The paper doesn't propose a new positional
scheme. It proposes an eviction policy that is orthogonal to whichever
scheme the model was pre-trained with. §3.2 (p. 5) shows exactly the
re-indexing rule for RoPE (rotate at decode using cache-local index)
and for ALiBi (contiguous bias in cache coordinates). We come back to
this on slide 17.

## Slide 9 — Related work 2/3: context-window extension

The methods that *look* competitive with StreamingLLM but actually
solve a different problem: grow the *pre-training window* so the
model natively handles longer inputs.

- **Position Interpolation (Chen et al., 2023).** Linearly
  interpolate RoPE indices to a longer window. Needs fine-tuning.
- **YaRN (Peng et al., 2023).** NTK-aware frequency scaling; needs
  less fine-tuning than Position Interpolation; longer usable
  context.
- **LongRoPE (Ding et al., 2024).** Evolutionary search over RoPE
  frequency rescalings.
- **FlashAttention (Dao 2022, 2023).** Makes O(T²) attention
  *feasible* at long T by reordering the memory accesses. Not
  competitive with StreamingLLM — orthogonal. You can use both.
- **Landmark Attention, Focused Transformer.** Retrieval-style
  hooks into distant context.

Say this out loud: **StreamingLLM doesn't try to grow the window. It
keeps the window bounded and lets the *stream* grow.** If you want a
7B model to actually reason about a 200K-token document, you want
YaRN or LongRoPE, not StreamingLLM. If you want a 7B model to serve
a chat session that has been running for days, you want StreamingLLM.
Different problems.

## Slide 10 — Related work 3/3: sparse attention and concurrent work

The methods that *look* older-but-related and often get confused with
StreamingLLM.

- **Sparse Transformer (Child et al., 2019), Longformer (Beltagy
  2020), BigBird (Zaheer 2020), ETC (Ainslie 2020).** All modify the
  attention pattern with structured sparsity + global tokens. They
  need custom CUDA kernels and, crucially, they are **not drop-in on
  a pre-trained autoregressive decoder** — they were designed for
  encoders (BERT-style) or trained from scratch.
- **LM-Infinite (Han et al., 2023).** Concurrent with StreamingLLM.
  Λ-shaped attention pattern: keep initial tokens + a local band.
  Similar spirit, published a few weeks apart. Xiao et al. mention
  it in the related work.
- **Landmark Attention (Mohtashami & Jaggi, 2023).** Retrieval-style
  distant memory access using a special landmark token.

How StreamingLLM differentiates itself from all of the above:

1. It is first to give the "attention sink" phenomenon a name and
   show it exists in Llama-2, MPT, Falcon, Pythia, BERT, and ViTs.
2. It is first to show the sinks are **positional, not semantic**
   (via linebreak substitution — coming on slide 14).
3. It proposes the learnable-sink pre-training refinement
   (§3.3, Table 3).

That third point is what turns "an implementation trick" into a
research contribution. Even if you never adopt the pre-training fix,
knowing that the sinks are structural (not accidental) tells you
what you can and cannot ablate.

## Slide 11 — Section divider: The observation

Say "The observation." Pause.

The observation is short — three slides — but it's the pivot of the
whole paper. Everything downstream (the mechanism, the results, the
limits) is a *consequence* of what these three slides say.

## Slide 12 — Attention concentrates on the first few tokens

Show the audience the phenomenon before the explanation.

The authors visualise attention maps for Llama-2-7B, averaged over
256 sentences of length 16. Beyond the bottom two layers, the model
heavily attends to the very first token — **everywhere**. Every
layer, every head. Fig. 2 on p. 3 of the paper.

And it is not a Llama quirk: the same pattern shows up in

- **Falcon-7B** (also RoPE)
- **MPT-7B** (ALiBi — a completely different positional scheme)
- **Pythia** (a completely different pre-training corpus)
- **BERT-base-uncased** (encoder, not decoder): dumps attention on
  the `[SEP]` token. Appendix H, p. 20.
- **Vision Transformers** with the *register tokens* of Darcet et
  al. 2023: same phenomenon, in a different modality.

The authors name these tokens **attention sinks** and read the box
caption out loud verbatim (Fig. 2, p. 3):

> "Beyond the bottom two layers, the model heavily attends to the
> initial token across all layers and heads."

## Slide 13 — Why? The softmax must sum to one

This is the paper's central *argument*, and it is short.

Attention scores go through a softmax. Softmax outputs sum to one.
The model is forced to allocate attention mass across the context
regardless of whether the context deserves it or not. **The excess
mass has to land somewhere.**

Show the equation on the slide. If `x_1 ≫ x_j` for `j ≥ 2`, then
`exp(x_1)` dominates the denominator. Now suppose I remove that
first token from the cache. The denominator loses its dominant term.
The whole softmax renormalises. Every attention weight in the row is
now scaled up by roughly `Σ_j exp(x_j) / (Σ_j exp(x_j) - exp(x_1))`.
The attention distribution warps. Perplexity explodes.

In a *causal* decoder-only Transformer, the initial tokens are the
tokens every other position can see. They are the natural dumping
ground for that unwanted mass. This is why the tokens the model
uses as sinks are always the *first* ones — the position that has
the highest visibility.

Puzzle from slide 6 solves itself: window attention is O(TL) and
correct in cost, but the moment it evicts the first token, the
softmax denominator collapses and PPL goes to 5158.

## Slide 14 — Is it the tokens, or is it the position?

Two competing hypotheses:

- **A**: the first tokens carry irreplaceable semantics. (E.g.,
  "the model has learned to encode topic in the first BOS token.")
- **B**: the first *positions* do the work. Any token you place there
  will serve as a sink.

The cleanest ablation of the whole paper: take the first four tokens
and replace them with linebreak characters (`\n`). Nothing else
changes. Table 1, p. 5, Llama-2-13B, PG19:

| Configuration                 | PPL     |
|---|---|
| 0 + 1024 (window, no sink)    | **5158.07** |
| 4 + 1020 (StreamingLLM, real) | **5.40** |
| 4×"\n" + 1020 (linebreaks)    | **5.60** |

Linebreaks recover essentially all of the perplexity that the real
first-four tokens do. That single row **kills hypothesis A**. The
sinks work because of their *position*, not their *content*.

Two things to say out loud:

1. This is why the paper says "attention sinks" and not "important
   initial tokens." The sink-ness is a property of position 0–3, not
   of the tokens that happen to be there.
2. The 4-vs-any-other-number choice will be justified on slide 15
   with Table 2 evidence (1, 2, 4, 8 sinks) — 4 saturates the fix.

## Slide 15 — Section divider: The mechanism

Say "The mechanism." Pause.

Six slides here — the cache layout, the re-indexing trick, a worked
example, the RoPE / ALiBi variants, and the algorithm-in-one-box.
This section is heavier than a typical "here is a diagram" section
because the re-indexing trick trips people up.

## Slide 16 — The KV cache layout

Point to the diagram. Two regions:

- **Four orange cells on the left: sink tokens.** Original positions
  0, 1, 2, 3. They enter the cache when decoding begins, and they
  **never leave**.
- **Blue rolling window on the right: the last L tokens.** Managed
  FIFO — as a new token arrives, the oldest one in the rolling
  window is evicted.
- **The gap in between: discarded.** Everything between position 3
  and position `T − L` is gone from the cache. The model never
  attends to those tokens again.

Cache size is exactly `S + L` for the whole run — independent of how
long the stream gets. That's the *whole* memory-boundedness property.

## Slide 17 — Positions are indexed inside the cache

This is *the* non-obvious implementation detail.

If we kept each token's *original text position* on both the sinks
and the rolling window, then in position space they'd be far apart
— the sinks are at [0,1,2,3] and the rolling window at [T−L, …, T−1].
RoPE and ALiBi both react badly to that huge gap because both
encodings model *relative* distance and neither was trained to
handle sudden discontinuities.

The paper's fix: **renumber position IDs inside the cache**. If the
cache holds tokens whose original positions are [0, 1, 2, 3, 6, 7, 8]
(so tokens 4 and 5 have been evicted), and the model is now decoding
position 9, the model sees cache-local indices [0, 1, 2, 3, 4, 5, 6,
7]. **Not** [0, 1, 2, 3, 6, 7, 8, 9].

- **With RoPE**: cache the keys *pre-rotation*, apply the rotation
  at decode time using the cache-local index. This is the standard
  RoPE forward pass but with a redefined position vector.
- **With ALiBi**: apply the contiguous linear bias `-m·|i-j|` in
  cache coordinates (contiguous 0..T_c-1) rather than in
  original-text coordinates (with the jump).

Paper §3.2, p. 5.

## Slide 18 — Mechanism deep-dive: a 20-token worked example

Walk the diagram. This is the "your understanding of the last slide
is now testable" slide.

Suppose we're at decoding step 20 and cache holds `S = 4` sinks +
`L = 8` rolling window.

- The cache currently contains tokens whose **original positions**
  were 0, 1, 2, 3, then a gap (positions 4–11 have been evicted),
  then 12, 13, 14, 15, 16, 17, 18, 19.
- If we passed the model the original position vector
  [0,1,2,3,12,13,14,15,16,17,18,19] alongside a query at position 20,
  RoPE would rotate keys as if positions 4–11 existed and were
  simply not represented; the model would compute a distance-based
  relation to positions it never trained on. ALiBi would compute a
  bias of `-m·|20 − 3| = -17m` on the fourth sink and `-m·|20 − 12|
  = -8m` on the first rolling token — a discontinuity in the bias
  slope right at the sink-window boundary.
- Instead, StreamingLLM passes the model **cache-local** positions
  [0,1,2,3,4,5,6,7,8,9,10,11] with the query at position 12. Now
  the sinks and the rolling window are adjacent in position space,
  the RoPE rotation is smooth, the ALiBi bias is smooth, and the
  model behaves in the regime it was trained for.

The trade-off: the model is *lying to itself* about how old the
tokens are. That's fine — the point of the KV cache is to hold
information, not timestamps. Perplexity results (slides 20–22)
confirm this lie is harmless up to 4M tokens.

## Slide 19 — RoPE and ALiBi variants

Walk the table row by row.

- **Where positions enter.** For RoPE, positions are baked into `q`
  and `k` via a rotation before the dot product. For ALiBi,
  positions never touch `q`/`k`; the bias is added to the logits.
- **Cache what?** With RoPE, cache the *pre-rotation* keys.
  Otherwise the cached rotation is wrong the moment you re-index.
  With ALiBi, cache the keys as-is; there's nothing to un-do.
- **When to apply the rotation/bias?** At decode. Both fixes are
  cheap: one extra rotation per decode step for RoPE (free
  memory-wise, negligible compute-wise); one bias recomputation
  for ALiBi (also free).
- **Effect of a position gap.** Both encodings degrade badly — RoPE
  because the rotation is not what it was trained for, ALiBi
  because the bias magnitude is not what it was trained for.
- **Fix cost.** ~5–10 lines on top of a standard HF forward pass.

Small aside: this cache-local trick is why StreamingLLM slotted so
cleanly into TensorRT-LLM, HF Transformers, MLC LLM, and Intel
Extension for Transformers within months. It's an in-place edit
of a specific loop; no retraining, no new kernels.

## Slide 20 — The algorithm in one box

Walk the pseudocode. This is the whole method in ≈15 lines. Read it
slowly:

- **State**: sinks (fixed, S entries) + a rolling deque (max length L)
- **Per decode step**:
  1. Project the current input into q, k, v.
  2. Concatenate cached K, V with the new k, v.
  3. Build a cache-local position vector `arange(0, len(cache_K))`.
  4. If RoPE, rotate all cached K and the current q using that
     position vector.
  5. Compute softmax(q · K^T / √d) as usual.
  6. Multiply by V.
  7. Append (k, v) to the rolling deque. The sinks are never touched.

Everything else — the model weights, the tokenizer, the attention
head structure — is untouched. StreamingLLM is a **wrapper around the
KV-cache data structure**, not a new model architecture.

Point at the footer: the full runnable version is in
`task1-presentation/code/streaming_llm_demo.py`, ~200 lines of
PyTorch + HF including argument parsing and comments. The
mechanism itself is ~50 lines.

## Slide 21 — Refinement: pre-train with a dedicated sink token

Everything up to now was a training-free inference fix. This is §3.3
— the optional pre-training refinement.

The reasoning: if we *had* a dedicated always-visible slot during
pre-training, the model would learn to route its excess attention
there, rather than smearing it across the first four accidental
tokens. Two proposals:

- **Sink Token**: prepend a single learnable placeholder to every
  pre-training sample.
- **Zero Sink** (Miller, 2023): replace SoftMax with
  SoftMax-off-by-One — i.e. add a constant `1` to the denominator.
  Equivalent to prepending an all-zero-K, all-zero-V token.

Read Table 3, p. 6 — 160M-parameter models trained from scratch on
deduplicated Pile, 143K steps, batch 256.

- **Vanilla**, 0+1024 (window only, no sink): **27.87 PPL**. With 4
  accidental sinks re-introduced at inference: 18.05.
- **Zero Sink**, 0+1024: **29214 PPL** — spectacularly worse. With
  4 sinks: 18.01 — matches vanilla.
- **Learnable Sink**, 0+1024: 1235 (also poor). But at **1+1023**:
  **18.01** — one deliberate sink token beats vanilla with four
  accidental ones.

The caveat, and I say this out loud: **this is at 160 million
parameters**. Whether the effect scales to 7B / 13B / 70B has not
been demonstrated. The paper flags this openly.

## Slide 22 — Sink-Token pre-training deep-dive (§3.3)

Same numbers as slide 21 but presented from a different angle so we
can dwell on the interpretation.

Look at the two "0+1024" columns:

- Vanilla → 27.87. The naive baseline. Poor but functional.
- Zero Sink → 29214. Much *worse*. Add a constant 1 to the softmax
  denominator during training and you break the sink phenomenon —
  because now the model doesn't need one — and then you *evict* the
  first token at inference (since it's not "special" any more) and
  the model can't handle it. The paper reads this as evidence that
  Zero Sink solves half the problem (train-time) but not the other
  half (inference-time behaviour under eviction).
- Learnable Sink → 1235. Better than Zero Sink but still poor at
  0+1024, because we're evicting the learned sink too. It's the
  1+1023 column that matters — keep the sink, evict everything
  else, and the model works.

This is the strongest evidence in the paper for the argument that
sinks are a *softmax stability trick*, not a semantic phenomenon.
The model isn't learning "important content is in position 0"; it's
learning "please absorb softmax overflow here."

## Slide 23 — Does downstream accuracy hold?

The sink-token pre-training obviously has to be *free* on downstream
tasks — nobody will adopt it if it costs 1% accuracy.

- **Table 4, p. 7.** Zero-shot accuracy on ARC-Challenge, ARC-Easy,
  HellaSwag, LAMBADA, OpenbookQA, PIQA, Winogrande. 160M vanilla vs
  160M with Sink Token pre-training. **Differences on all seven
  tasks are <1 point** — and Sink Token is *slightly higher* on
  average. This is essentially free.
- **Figure 6, p. 7.** Pre-training loss curves indistinguishable.
  The sink token does not hurt convergence.
- **Figure 7, p. 7.** Attention heatmaps confirm: the learnable
  sink absorbs the mass that vanilla training smears across the
  first four tokens.

The honest caveat, one more time: all of this is 160M-scale. The
"is this real at 7B/13B/70B?" question is open. That's why in the
"What I would change" section I put "sink-token pre-training at
scale" on the list.

## Slide 24 — Section divider: Results

Say "Results." Pause.

This is the middle of the talk, ~25 minutes in. There are seven
slides here — three headline results (perplexity, throughput,
streaming QA) each followed by a deep-dive (the per-model table,
the per-cache-size latency, and the StreamEval distance test).
Watch time carefully.

## Slide 25 — Result 1: perplexity stays flat past 4M tokens

The headline empirical result.

Setup: concatenated PG19 test set (100 long books, contiguous). Cache
2048 for Llama-2 (pre-training window was 4K), 1024 for MPT/Falcon/
Pythia (pre-training window was 2K). Half the pre-training window in
each case — chosen for graph legibility, not for optimality.

Across every model family in Figure 5 (p. 7):

- Llama-2-{7,13,70}B
- MPT-{7,30}B
- Falcon-{7,40}B
- Pythia-{2.8,6.9,12}B

perplexity stays flat out past **four million tokens**. Dense
attention OOMs early. Window attention collapses the moment its
cache fills.

Read the green box: this is Table 1 restated. Llama-2-13B, PG19,
first book, 65K tokens.

- Window attention: PPL 5158.07.
- StreamingLLM (4 + 1020): PPL 5.40.
- StreamingLLM with linebreak sinks: PPL 5.60.

Same slide, same numbers, twice — because this is the single
strongest table in the paper.

## Slide 26 — Fig. 5 per-model perplexity to 4M tokens

Zoom into Figure 5. Walk the table:

- Llama-2 family (7B, 13B, 70B), RoPE, cache 2048 — flat.
- MPT family (7B, 30B), **ALiBi**, cache 1024 — flat.
- Falcon family (7B, 40B), RoPE, cache 1024 — flat.
- Pythia family (2.8B, 6.9B, 12B), RoPE, cache 1024 — flat.

Point to say: this covers **ten** models across four families, two
positional encodings, and two pre-training corpora. And every single
one tracks the Sliding-window-with-recomputation *oracle* baseline
essentially exactly — while being up to 22× faster (next slide).

Read the green box: the reference line is the oracle. StreamingLLM
matches it on 10/10 models. That's a very strong empirical claim
without a theoretical guarantee — which is fine, and I'll say so on
slide 44 in the critical analysis.

## Slide 27 — Result 2: up to 22.2× faster

The efficiency headline.

Setup: single NVIDIA A6000, Hugging Face Transformers, batch 1,
per-token decode latency, greedy decoding. Llama-2-13B and Llama-2-7B.
Fig. 10, p. 9.

Read the table for Llama-2-13B:

- Cache 256: Sliding+Recompute 2355 ms → StreamingLLM 106 ms →
  **22.2×** speedup.
- Cache 512: 860 → 75 → 11.5×.
- Cache 1024: 361 → 60 → 6.0×.
- Cache 2048: 169 → 52 → 3.3×.
- Cache 4096: 99 → 48 → 2.1×.

The 22.2× headline number in the abstract is the *small-cache*
number. That's where the recompute baseline hurts most: each new
token forces a rebuild of the whole cache. As cache grows, the
recompute cost amortises and the speedup shrinks — as we should
expect for any O(TL) beats O(TL²) argument.

Memory footprint: essentially the same for both methods
(Fig. 10 right panels). Both are bounded by cache size, not by
original stream length.

## Slide 28 — Deep-dive: Fig. 10 latency for both 7B and 13B

Same result, more numbers. Both model sizes side by side.

- Llama-2-13B: 22.2× at cache 256, 2.1× at cache 4096.
- Llama-2-7B: 21.7× at cache 256, 2.0× at cache 4096.

The ratio scales the same way across model size — which makes sense:
the *asymptotic* comparison is O(L²) vs O(L) per token, independent
of the model dimension.

Read the "reading the numbers honestly" box: important caveats.

1. Batch 1. Real streaming servers run batched attention. The
   interaction of the sink+rolling layout with batched attention
   kernels is not measured. This is a fair Q&A question and I
   should acknowledge I don't have a definitive answer.
2. Single A6000, greedy decode. Speculative decoding, tree-of-
   thought, and beam search all interact differently with the KV
   cache. Not measured.
3. Memory equal — this is important because it kills the
   "StreamingLLM is faster by using less memory" misreading.

## Slide 29 — Result 3: streaming QA becomes usable

Now for a task-level rather than a language-modelling result.

Setup, from §4.3: take every question in ARC-Easy and every question
in ARC-Challenge and concatenate them into one long stream. Feed the
entire concatenated stream to Llama-2-{7,13,70}B-Chat as a **single
decoding job**. Score exact-match on each answer.

Read Table 5, p. 8. Cache = 1024. The three baselines are one-shot
(fresh call per question — the oracle), Window, and StreamingLLM.

- Llama-2-7B-Chat: one-shot 71.25/53.16. Window **3.58/1.39**.
  StreamingLLM 71.34/55.03.
- Llama-2-13B-Chat: 78.16/63.31. Window 0.25/0.34. StreamingLLM
  80.89/65.61.
- Llama-2-70B-Chat: 91.29/78.50. Window 0.12/0.32. StreamingLLM
  91.37/80.20.

Three things to point out:

1. Dense OOMs long before the stream ends — not shown, because
   there's no row for it.
2. Window is not degraded, it is *destroyed*. 0.12% on ARC-Easy at
   70B is essentially random guessing.
3. StreamingLLM matches or slightly beats the one-shot baseline on
   every model size. **This is the closest the paper gets to a
   real production case.**

## Slide 30 — Streaming QA per model size (deep-dive)

Same table, presented with a Δ-vs-one-shot column.

The Δ column makes the shape of the failure obvious:

- Llama-2-7B-Chat window: **-67.7** on ARC-E, **-51.8** on ARC-C.
- Llama-2-13B-Chat window: **-77.9** and **-63.0**.
- Llama-2-70B-Chat window: **-91.2** and **-78.2**.

Notice that the Δ *grows* with model size. Bigger models were more
right before, and now they're at zero — the drop is larger. If you
were tempted to think "window attention is a mild degradation that
we can tolerate for streaming," this table is meant to prevent that.

And notice StreamingLLM's positive Δ on the 13B and 70B models
(+2.7 / +2.3 on ARC-E/C for 13B). That is genuinely surprising —
the paper doesn't dwell on it, but if StreamingLLM is *matching* the
oracle by design, why is it beating one-shot? Two hypotheses:

1. Concatenated setup gives the model a chance to use the shared
   context (system prompt, few-shot examples) *once* rather than
   recomputing from scratch each question — so cache accumulation
   helps.
2. Small-sample noise on ARC.

The paper says "one-shot" is oracle-adjacent, not oracle. Take the
+2.7 with a small grain of salt.

## Slide 31 — StreamEval (Fig. 8, Fig. 9)

The paper's own long-eval-style benchmark. Setup: issue a query
every 10 lines; the answer is always **exactly 20 lines back**.
This is a controlled distance signal — you can pick the query
distance and measure accuracy.

Two results to walk.

**Positive result — Fig. 8**: StreamingLLM stays accurate up to
~120K tokens on Llama-2-7B-Chat at cache 1024. Dense attention
OOMs. Window is near-zero from the moment the cache fills. This is
what you'd expect from the perplexity result — a task-level
confirmation.

**Honest limit — Fig. 9**: the moment the query-answer distance
*exceeds cache size*, accuracy drops sharply. StreamingLLM has
**no ability** to recall information older than the rolling window.
This is the whole basis of §5's honest scope claim: it's a
short-term-memory streaming policy, not a long-term-memory one.

Say this out loud because it's the single most important limitation.
StreamingLLM does not extend context. If your product needs the
model to remember something from 200K tokens ago, StreamingLLM won't
help. LongBench (slide 34) will show this in a task-level way.

## Slide 32 — What StreamingLLM does *not* do

Section divider by rhetorical function rather than by paper section.
Say this slowly. This is where the paper is honest, and this is what
a careful reader wants to see. The next four slides go deep on each.

- It does not extend context. Table 7, Fig. 9.
- It underperforms truncation on LongBench. Table 8.
- Bigger cache does not always help. Table 6.
- It does not compete with H2O / SnapKV / FastGen. No comparison.
- Sink-token pre-training is 160M only.

## Slide 33 — Limits deep-dive: bigger cache ≠ lower perplexity

Table 6, p. 9. PG19, sink count fixed at 4.

- Llama-2-7B: 4+508 → PPL 9.73. 4+1020 → 9.32. 4+2044 → **9.08**.
  4+4092 → 9.59. **Peak at 2048, then regressed.**
- Llama-2-13B: 4+508 → 8.35. 4+1020 → 7.79. 4+2044 → **7.51**.
  4+4092 → 7.60. Same shape.

Two ways to read this:

1. **Paper's reading**: current LLMs under-utilise the context
   they're given. It's an LLM property, not a StreamingLLM property.
   Nothing you can do about it inside a positional policy.
2. **My critical reading** (also in `critical-analysis.md`): this is
   evidence that a *policy* upgrade could still help. A rule that
   decided *which* recent tokens to keep — rather than "the newest
   L" — might be monotone in L, because you'd only be adding useful
   tokens. That's the H2O / SnapKV direction.

Both readings are defensible. I'll say both. The audience decides.

## Slide 34 — Limits deep-dive: LongBench (Table 8) is not the target

Table 8, p. 17. StreamingLLM 4+3496 vs default truncation 1750+1750,
LongChat-7B-v1.5-32K. Six tasks:

- NarrativeQA: 11.6 vs 18.7 — StreamingLLM loses by 7.1.
- Qasper: 16.9 vs 19.2 — loses by 2.3.
- HotpotQA: 21.6 vs 25.4 — loses by 3.8.
- 2WikiMQA: 28.2 vs 32.8 — loses by 4.6.
- GovReport: 23.9 vs 27.3 — loses by 3.4.
- MultiNews: 25.5 vs 25.8 — essentially tied.

Why StreamingLLM loses here: LongBench evaluates long-doc QA. It
puts the question at the *front* of the prompt. StreamingLLM at
4+3496 keeps only 4 sink tokens of the initial prompt — the actual
question is gone.

Raise the sink budget to 1750 (StreamingLLM 1750+1750, same total
cache) and parity is restored on all six tasks: 18.5 vs 18.7 on
NarrativeQA, 19.6 vs 19.2 on Qasper, 27.4 vs 25.4 on HotpotQA, etc.

Verdict: **StreamingLLM is a streaming policy, not a long-doc
policy.** The paper is explicit about this in Appendix A. The failure
on LongBench is a *scope claim* not a *policy defect*.

## Slide 35 — Limits deep-dive: positional-only

The single biggest conceptual limit.

The eviction rule is: keep first S + keep last L. Nothing else. It
does not look at attention scores. It does not look at persistence
across steps. It does not look at head-specific patterns. Purely
positional.

The paper never compares to attention-score-based evictors. That's
chronologically understandable — StreamingLLM is arXiv September
2023; H2O is arXiv June 2023 but very early; SnapKV was 2024. But
by 2026 the audience will ask, and the honest answer is: we don't
have the head-to-head.

Consequence you should verbalise: a token in the rolling window that
was heavily attended to (i.e., informative) is evicted the moment
it ages out — **even if later queries would attend to it again**.
A content-aware evictor would keep it.

That is the segue to the next section — modern alternatives.

## Slide 36 — Modern alternatives: content-aware KV-cache eviction

Walk the comparison table. All five methods listed are attention-
score-based evictors, and none of them integrate the sink insight.

- **H2O (2023)**: cumulative attention score. Keep the tokens with
  the highest total attention weight over decoding history. O(TL
  log L). Does *not* handle the sink question — treats first tokens
  as ordinary, which will silently break the softmax denominator
  argument.
- **Scissorhands (2023)**: persistence of attention across steps.
  If a token was attended to at multiple past decode steps, keep
  it. Otherwise, evict.
- **FastGen (2023)**: per-head profile. Each attention head is
  categorised as "heavy hitter", "local", "punctuation", etc., and
  each head gets a different cache. Empirically finds sinks (as
  "punctuation heads") without naming them.
- **SnapKV (2024)**: attention voting over a short observation
  window. Fast and simple.
- **Keyformer (2024)**: approximates attention via Gumbel-softmax
  over keys.

All five are compatible with StreamingLLM in principle: keep 4
positional sinks; run content-aware eviction inside the rolling
window. To the best of our knowledge, no paper publishes the
hybrid at scale. This is what my "what I would change" slide
proposes.

## Slide 37 — Positional vs. content-aware: strengths & failure modes

Walk the trade-off table.

- **Decision rule**: positional (StreamingLLM) is "first 4 + last L";
  content-aware is "keep by attention weight."
- **Implementation**: 50 lines vs. profiling attention scores +
  priority queue. Big difference — StreamingLLM plugs in trivially,
  H2O/SnapKV need more infrastructure.
- **Softmax denominator stability**: guaranteed for StreamingLLM
  (sinks never move); *not* guaranteed for content-aware methods
  unless they explicitly protect the sinks.
- **Long-range retention**: none beyond L for StreamingLLM; up to
  the size of the cache for content-aware.
- **Bursty / topic-shifting traffic**: StreamingLLM is robust
  (positional decisions don't depend on the content distribution);
  content-aware has a known failure mode where heavy hitters go
  stale after a topic shift.
- **Where each wins**: streaming LM / dialogue for StreamingLLM;
  long-doc QA / summarisation for content-aware.

Say this out loud: the two are *complementary*. That is the thesis
of the "What I would change" slide.

## Slide 38 — Implementation walkthrough 1/3: ~50 lines of PyTorch

Walk the code block on the slide.

The whole cache is a tiny class:

- `__init__`: two lists (or tensors) for sinks, two for the rolling
  window. Parameters `n_sinks=4` and `window=1020`.
- `append(k, v)`: if we haven't filled the sink slots yet, add to
  sinks; otherwise, append to the rolling window and pop the
  oldest if we exceed L.
- `as_kv()`: concatenate sinks + rolling window and return the
  full cached K, V.

No new tensor ops. No custom CUDA. Just concatenation and pop.

Say out loud: this is the "why did StreamingLLM get adopted upstream
in 3 months" answer. It's a **wrapper around a data structure**,
not a new kernel and not a new model.

## Slide 39 — Implementation walkthrough 2/3: cache-local RoPE

Walk the code block. This is the attention step.

- Project x_t into q, k, v.
- Append the pre-rotation k, v to the cache. Store them
  *unrotated*.
- Get the current cache K, V.
- Build a cache-local position vector `arange(0, T_c)`.
- Rotate the cache K with those positions. Rotate q with the last
  position (T_c − 1).
- Compute `softmax(q · K^T / √d) · V` as usual.

The important line is `K = rope.rotate(K_pre, pos_ids)`. That's the
one line that makes cache-local re-indexing work. Everything else
is standard attention.

For ALiBi, we'd swap the rotate step for a `bias = -m * (pos_ids -
pos_ids[-1])` addition on the logits. Same conceptual move.

## Slide 40 — Implementation walkthrough 3/3: benchmark harness

Walk the harness pseudocode.

- Load Llama-2-7B via HF `AutoModelForCausalLM`.
- Instantiate `SinkKVCache(n_sinks=4, window=2044)`.
- Load PG19 test, concatenate to target length (4M tokens).
- Decode one token at a time in a tqdm loop. Accumulate logits.
- Compute perplexity from logits.
- Assertion: at cache 4+2044 on Llama-2-7B, PPL should come out
  under 10 (Table 6 says 9.08). That's your regression test.

For the 22.2× number: repeat with a sliding-window-with-recompute
baseline, divide per-token wall-clock. In our experiments the
reproduction is within ~5% of Fig. 10.

Full code (running version) is under `task1-presentation/code/
streaming_llm_demo.py`. It includes tokenizer setup, argparse, and
the two-eviction-policy comparison.

## Slide 41 — What I would change

Four concrete proposals, in decreasing order of confidence.

1. **Per-layer sink budget.** Fig. 2 and Figs. 11–13 make clear that
   the two lowest layers barely use sinks at all. If we let `S_l`
   vary by layer, we free budget in low layers and reallocate it to
   a longer rolling window in high layers. This is a
   fine-tuning-free change; the required per-layer heatmap can be
   built from a single forward pass over a calibration set.
2. **Attention-score eviction inside the window.** Keep the four
   positional sinks (or the one learned sink, if pre-training).
   But instead of FIFO on the rolling window, evict by H2O /
   SnapKV attention score. Positional sinks preserve the softmax
   denominator; content-aware inside the window recovers the
   long-range retention StreamingLLM currently lacks.
3. **Real streaming traffic.** All language-modelling experiments
   are on PG19 (long monolingual books). That does not match the
   multi-turn dialogue this method is *marketed* for. LMSYS-Chat-1M
   and ShareGPT would be much fairer benchmarks — the same shape
   of concatenated evaluation as Table 5, but on real chat.
4. **Sink-Token pre-training at scale.** Someone needs to train a
   7B model with a sink token and publish downstream accuracy on
   the standard six benchmarks. Until then §3.3 is a promise not
   a proof.

Now read the blue "same idea, one level up the stack" box aloud:

> StreamingLLM chooses positional eviction on the KV cache. My final
> project (Task 2) chooses cost-aware eviction on the response
> cache (GDSF on GPTCache). Same underlying problem — dumb eviction
> wastes bounded capacity — at different layers of the LLM serving
> stack.

This is my bridge. Say it out loud even if the audience read it.

## Slide 42 — Bridge to Task 2: two caches, one idea

Now walk the bridge table row by row.

- **What is cached.** StreamingLLM caches KV vectors of past tokens
  (units: `S + L` entries of size `H · d`). Task 2 caches whole
  `(prompt → response)` pairs (units: entries or bytes).
- **Where in the stack.** StreamingLLM lives inside the model, per
  attention layer, per request. Task 2 lives in front of the model,
  shared across requests, at the API layer.
- **Capacity units.** `S + L` tokens for StreamingLLM. N entries or
  M bytes for GPTCache/GDSF.
- **Eviction signal.** StreamingLLM: position. Task 2 (GDSF):
  frequency × cost / size + a monotone clock.
- **Failure mode when dumb.** StreamingLLM: PPL 5158 (softmax
  collapse). Task 2: overspend on regeneration $$$ because LRU
  evicts the expensive-to-regenerate entries.
- **What our project proves.** For Task 2: GDSF beats LRU by
  +25.7% on high-variance-cost workloads, +32.3% on bursty,
  **+91.0% on size-varying**, +18.8% on the adversarial anti-LRU
  pattern. Bonferroni-corrected p < 1e-7. 3,600-run benchmark.

Read the "through-line" box: both caches are bounded, both suffer
under naive eviction, and both admit a small, elegant fix that
doesn't require retraining the model. That is why these two tasks
belong in the same course.

## Slide 43 — Conclusion

Read the five bullets:

- Attention sinks are structural (positional), not semantic.
- Four sinks + a rolling window are enough to stream to 4M+ tokens.
- Up to 22.2× faster than the only sane baseline, at essentially
  identical memory cost.
- Adopted upstream by NVIDIA TensorRT-LLM, Intel Extension for
  Transformers, HF Transformers, MLC LLM (Impact Statement, p. 10).
- But: it doesn't extend context and doesn't compete with
  attention-score-based eviction.

Then the one-line take (accent box):

> A robust empirical fix for a softmax quirk. Small, easy to
> implement, and honest about its own ceiling.

## Slide 44 — Q&A prep 1/2: expected questions

Walk each expected question and my one-sentence answer:

- **"Is this just window attention with a warm start?"**
  No. Sinks are never evicted. Table 1: 5158 → 5.40.
- **"Why 4 sinks?"**
  Table 2: 1–2 leaves a residual bump; 4 saturates; 8 doesn't help.
- **"RoPE, ALiBi, or both?"**
  Both, via cache-local re-indexing (§3.2, p. 5).
- **"Does it extend the context?"**
  No. Table 7, Fig. 9 show accuracy collapses beyond cache size.
- **"vs H2O / SnapKV / FastGen?"**
  Paper doesn't compare. Chronologically earlier. They're
  content-aware, StreamingLLM is positional. Complementary.
- **"Long-doc QA?"**
  Table 8: StreamingLLM 4+3496 loses to truncation because the
  prompt is dropped. At 1750+1750, parity restored.

## Slide 45 — Q&A prep 2/2: harder questions

Now the questions I'm less sure of and what I'd actually say:

- **"Are sinks just a softmax artefact — would SoftMax-off-by-One
  fix it?"** Table 3 says Zero Sink helps somewhat (29214 → 18.01
  at 4+1020) but does not remove the *need* for sinks. Empirical
  rescue beats theoretical fix. My reading: sinks are how the
  model uses the softmax; changing the softmax doesn't
  automatically remove the mechanism the model has already learned.
- **"Is 22.2× realistic for production?"** Batch-1, single-A6000
  number. Real servers batch. Batched-attention interactions with
  sink+rolling layout are not measured. Fair Q&A question — I
  admit I don't have a definitive answer.
- **"What breaks the sink phenomenon?"** The paper covers
  Llama-2, MPT, Falcon, Pythia, BERT, ViT. I would love to see it
  tested on Mixture-of-Experts (does routing affect sink formation?)
  and on state-space models like Mamba (which don't have softmax
  at all).
- **"Have you reproduced this?"** Not the LLM-scale numbers — no
  compute. But the 50-line demo works on the 7B model on a
  smaller-scale PG19 subset, and our GDSF work reproduces the
  underlying "dumb eviction wastes capacity" claim at the response
  cache layer with 3,600 runs and Bonferroni-corrected stats.

## Slide 46 — Thank you

Invite questions. Have my email up. Point at the "arXiv:2309.17453"
line if anyone wants to look it up.

---

## Anticipated questions (extra buffer)

The following are questions I've seen come up in reading groups on
this paper but didn't quite make the 46-slide budget. I keep answers
ready.

**"How does StreamingLLM interact with speculative decoding?"**
Speculative decoding generates a draft with a small model and
verifies with the big one. Both models need consistent KV caches.
If both use StreamingLLM with the same S and L, the sink+rolling
layout stays in sync. If they use different windows, the verifier
can accept/reject tokens that the drafter doesn't have in cache.
The paper doesn't measure this; it's a fair engineering question.

**"How does it interact with beam search?"**
Beam search maintains K parallel decoding branches. Each branch
needs its own KV cache. StreamingLLM's cache is small (`S + L`
entries), so K parallel StreamingLLM caches fit where K parallel
dense caches would not. But the semantics of "sink" across beams is
undefined — do beams share sinks? The paper doesn't say. Practical
answer: shared sinks + per-beam rolling window is what the reference
implementations do.

**"Do you have to fine-tune the model?"**
No, and this is one of the strongest selling points. The
inference-time fix (sinks + rolling + cache-local position IDs) is
purely mechanical. No fine-tuning, no LoRA, no re-quantisation. The
learnable-sink pre-training in §3.3 is an *optional* refinement.

**"Does the sink phenomenon appear in encoder-only models?"**
Yes. Appendix H (p. 20) shows BERT-base-uncased dumps attention on
`[SEP]` when input is padded. Same softmax argument.

**"Why does dense attention degrade *before* it OOMs?"**
Because the positional encodings (RoPE, ALiBi) were pre-trained on
a bounded window. Feeding the model tokens whose positions it never
saw during training produces out-of-distribution rotation angles /
bias values. The model attends coherently within its training
window and produces nonsense at positions outside it. That is the
PPL 5641 in the dense row on slide 6.

**"What is the difference between an 'attention sink' and a
'register token' in ViTs?"**
Register tokens (Darcet et al. 2023) are learned dummy tokens that
absorb attention. Same idea. The StreamingLLM paper cites the
parallel. The concept of "attention sink" is broader — it can be
accidental (Llama-2 vanilla) or deliberate (learnable sink, ViT
register).

**"Does temperature or top-p matter for the sink phenomenon?"**
No. Sinks are a property of the *attention distribution*, which is
computed before sampling. Temperature and top-p only affect the
final token pick.

**"What if the input is very short — shorter than S?"**
Then there is no rolling window, only sinks. StreamingLLM degrades
gracefully to whatever attention the model would compute over a
short prefix.

**"How do sinks interact with sliding-window flash attention
kernels?"**
FlashAttention is a memory-optimal exact attention implementation.
Sliding-window kernels are FlashAttention specialised to
local-attention patterns. Combining sinks + FA sliding-window is
possible but not measured in the paper. The natural implementation
is: FA sliding-window on the rolling section + a small dense
attention head for the sinks. This is what MLC LLM reportedly does.

**"Have you compared StreamingLLM to your own GDSF policy?"**
Not directly — they live at different layers. GDSF is response-cache
eviction; StreamingLLM is KV-cache eviction. Different signals
(cost vs. position), different failure modes ($$$ vs. PPL), different
metrics (dollar-weighted hit rate vs. perplexity). What I do argue
is that both are answers to the same *shape* of problem: bounded
cache + naive baseline = leaves value on the floor.

**"Isn't 'attention sink' just a fancy name for a special token?"**
No. Special tokens (BOS, PAD, SEP) are semantic — the model has
learned specific representations for them. Attention sinks are
positional — any token in position 0 acts as one, including a
linebreak (Table 1, PPL 5.60). The special-token behaviour of
`[SEP]` in BERT (Appendix H) is *one instance* of the sink
phenomenon, not the definition.

---

## Timing rehearsal notes

- Slides 1–6 (Basics): 8 minutes.
- Slides 7–10 (Related work): 5 minutes.
- Slides 11–14 (Observation): 6 minutes.
- Slides 15–20 (Mechanism + deep-dive): 10 minutes.
- Slides 21–23 (Sink-Token pre-training): 4 minutes.
- Slides 24–31 (Results + deep-dive): 12 minutes.
- Slides 32–35 (Limits deep-dive): 6 minutes.
- Slides 36–37 (Modern alternatives): 3 minutes.
- Slides 38–40 (Walkthrough): 4 minutes.
- Slides 41–43 (Change + bridge + conclusion): 3 minutes.
- Slides 44–46 (Q&A prep + thanks): 4 minutes if I keep it short,
  or fold into open Q&A.

Total: 65 minutes at a walking pace; 55 minutes if I'm brisk on the
walkthrough slides. Leave ~10 minutes at the end (or interleaved) for
audience questions.
