# StreamingLLM — Speaker Notes

Nissim Brami · Caching in LLMs · Ben-Gurion University · Prof. Gil Einziger

These notes correspond one-to-one with the slides in
`StreamingLLM_Presentation.pptx`. Read them as talking points, not as a
script. Every numeric claim is followed by a page reference into the
paper (`papers/streaming-llm/streaming-llm.pdf`).

---

## Slide 1: Title

Open by naming the paper and the authors out loud. The paper is
*Efficient Streaming Language Models with Attention Sinks* by Xiao,
Tian, Chen, Han, and Lewis, published at ICLR 2024. First author is
at MIT; the work was done partly during an internship at Meta AI.

One sentence for context: this is a paper about the KV cache. It sits
squarely inside the vocabulary we already know from Chapter 1 of the
course (cache, capacity, eviction) but applied to the tokens a
language model keeps in memory while it decodes.

## Slide 2: Agenda

Walk the agenda quickly. Signal early that the last slide bridges to
my final project (Task 2), which is about a different cache in the
same LLM stack.

## Slide 3: Section divider — Basics

Say "Basics" and move on.

## Slide 4: The scene, decoding an LLM

The audience knows attention. Remind them of the mechanics:
autoregressive decoding, one token at a time, every new token attends
back to every earlier token, and to avoid recomputing we keep every
past token's Key and Value vectors in GPU memory. That is the KV
cache. Paper §1 (p. 2) says KV caching "consumes extensive memory";
a rule-of-thumb estimate for a 7B model at 4K tokens with FP16 is
about 2 GB. At decode time the bottleneck is memory
bandwidth into that cache, not compute.

## Slide 5: This is a caching problem

Anchor to Chapter 1 vocabulary. The KV cache has a fixed capacity
(GPU RAM). Requests are attention lookups from each new query
token. A miss means recomputing the whole context, which is
quadratic. What we need is an eviction policy. Window attention (the standard baseline) is literally LRU-by-position: keep the
most-recent L tokens.

Point at the box on the right: LRU, LFU, ARC, LIRS, Hyperbolic, FRD,
W-TinyLFU. Those are the professor's cache policies from Chapter 1.
StreamingLLM picks a positional policy, but with one non-obvious
twist we are about to see.

## Slide 6: Three obvious KV-cache strategies, all three break

Read the table row by row. The perplexity numbers are the paper's
own Fig. 1 (p. 2), measured on Llama-2-13B on the first book of
PG19 (65K tokens).

- Dense attention: keeps every token; O(T²); goes out of memory the
  moment T exceeds the pre-training window. Perplexity 5641. The model has degraded well before it OOMs.
- Window attention: keeps only the last L tokens; O(TL); the cheapest
  strategy that theoretically fits. Perplexity **5158**. Effectively
  gibberish.
- Sliding window with re-computation: rebuild the cache from the last
  L tokens for every new step; O(TL²); correct (PPL 5.43) but
  crushingly slow.

Read the "puzzle" box aloud: window attention has the right cost but
the wrong answer. What breaks when a single token, the first one, leaves the cache?

## Slide 7: Section divider — The observation

Say "The observation," pause, then advance.

## Slide 8: Attention concentrates on the first few tokens

Show the audience the phenomenon before the explanation.

The authors visualise attention maps for Llama-2-7B, averaged over
256 sentences of length 16. Beyond the bottom two layers, the model
heavily attends to the very first token. Everywhere. Every layer,
every head. This is not a Llama quirk: the same pattern appears in
Falcon, MPT (which uses ALiBi rather than RoPE), Pythia; in
BERT-base-uncased (where the model dumps attention onto `[SEP]`,
Appendix H); and in Vision Transformers with register tokens.

The authors name these tokens **attention sinks**.

Read the box out loud with the paper's exact wording from Fig. 2's
caption on p. 3.

## Slide 9: Why? The softmax must sum to one

This is the paper's central argument, and it is short.

Attention scores go through a softmax. Softmax outputs sum to one,
which means the model is forced to spread attention mass across the
context whether the context deserves it or not. That excess mass has
to land somewhere.

In a causal, decoder-only Transformer, the initial tokens are the
tokens every other position can see. They are the natural dumping
ground for that excess. The model learns to route unwanted
attention to them.

Now the puzzle from slide 6 solves itself: if you evict the first
token, you remove a large chunk of the softmax denominator; the
remaining logits get re-normalised; the whole attention distribution
warps; and perplexity explodes.

## Slide 10: Is it the tokens, or is it the position?

Two hypotheses:
- A: the first tokens carry irreplaceable semantics.
- B: the first *positions* do the work; the token identity is
  incidental.

The paper runs the cleanest ablation of the whole submission. Take
the first four tokens and replace them with linebreak characters.
Nothing else changes.

Read the table (Table 1, p. 5, Llama-2-13B, PG19):

- Window attention, no sinks: **PPL 5158.07**.
- StreamingLLM with the real first four tokens as sinks: **PPL 5.40**.
- StreamingLLM with **linebreaks** at those four positions: **PPL 5.60**.

Linebreaks recover almost all of the perplexity that real first-four
tokens do. Hypothesis B wins: the positions do the work, not the
content.

## Slide 11: Section divider — The mechanism

Say "The mechanism," pause, advance.

## Slide 12: The KV cache layout

Point to the diagram. Two regions:

- Four orange cells on the left: sink tokens. Positions 0, 1, 2, 3.
  They enter the cache when decoding begins and they never leave.
- A blue rolling window on the right: the last L tokens. Evicted
  FIFO as new tokens arrive.
- Everything between the sinks and the rolling window is discarded.

Cache size is exactly `S + L` (four plus L) for the whole run.
Independent of how long the stream gets.

## Slide 13: Positions are indexed inside the cache

This is the one implementation detail that catches people off guard.

If you keep the original text positions on the four sinks and on the
rolling window, the two regions are separated by a huge gap in
position space. RoPE and ALiBi both react badly to that gap.

The paper's fix: renumber positions **inside the cache**. If the
cache holds tokens whose original positions were `[0, 1, 2, 3, 6, 7,
8]`, and the model is now predicting position 9, it sees cache-local
indices `[0, 1, 2, 3, 4, 5, 6, 7]`. Not `[0, 1, 2, 3, 6, 7, 8, 9]`.

For RoPE, this means caching the keys pre-rotation and applying the
rotation at decode using the cache-local index. For ALiBi, it means
applying a contiguous linear bias in cache coordinates. Paper §3.2,
p. 5.

## Slide 14: Pre-training with a dedicated sink token

Everything up to now was a training-free inference fix. This slide
covers §3.3, the optional pre-training refinement.

If you control pre-training, you can prepend one learnable
placeholder token to every training sample. Call it the sink token.
The model learns to route its excess attention there.

Read Table 3 (p. 6): three 160M-parameter models trained from
scratch on deduplicated Pile, 143K steps.

- **Vanilla** column, 0+1024 (window only, no sink): PPL 27.87.
  With 4 sinks re-introduced at inference: PPL 18.05.
- **Zero Sink**: SoftMax-off-by-One, i.e. add a constant 1 to the
  denominator (Miller, 2023). Helps a little (18.01 at 4+1020) but
  27.87 → 29214 without any sink, so it does not solve the problem.
- **Learnable Sink** column: 1+1023 already gives PPL 18.01. One
  learnable sink token in pre-training replaces the four accidental
  ones at inference.

The caveat, stated openly: this was only validated at 160M
parameters. Whether it scales to 7B, 13B, 70B is left as future
work.

## Slide 15: Section divider — Results

Say "Results," pause, advance.

## Slide 16: Result 1 — perplexity stays flat past 4 million tokens

Setup: concatenated PG19 test set (100 long books). Cache 2048 for
Llama-2, 1024 for MPT, Falcon, Pythia (half the pre-training window
in each case).

Across every model family in Figure 5 (Llama-2-7B, 13B, 70B;
MPT-7B, 30B; Falcon-7B, 40B; Pythia-2.8B, 6.9B, 12B) perplexity
stays flat out past four million tokens. Dense attention OOMs early.
Window attention collapses the moment its cache fills.

Read the green box: this is Table 1 restated, Llama-2-13B, PG19,
first 65K tokens. Window attention gives PPL 5158.07; StreamingLLM
gives 5.40; linebreak-sinks give 5.60. Same slide, twice, because
it is the strongest single table in the paper.

## Slide 17: Result 2 — up to 22.2× faster

The efficiency benchmark. Single NVIDIA A6000, Hugging Face
Transformers, batch 1, Llama-2-13B, per-token decode latency.

Read the table from Fig. 10 on p. 9:

- Cache 256: sliding+recompute 2355 ms; StreamingLLM 106 ms; **22.2×**.
- Cache 512: 860 vs 75; **11.5×**.
- Cache 1024: 361 vs 60; **6.0×**.
- Cache 2048: 169 vs 52; **3.3×**.
- Cache 4096: 99 vs 48; **2.1×**.

The 22.2× headline number in the abstract is the small-cache
number. That is where the recompute baseline hurts most. Memory footprint
of the two methods is essentially the same (Fig. 10 right panels)
because both are bounded by cache size, not by original stream
length.

## Slide 18: Result 3 — streaming QA becomes usable

The setup here is more interesting than it looks. Take every
question in ARC-Easy and ARC-Challenge and concatenate them into
one very long stream. Feed the whole thing to Llama-2-Chat as a
single decoding job. Score exact match on each answer.

Read Table 5, p. 8, cache = 1024:

- Llama-2-7B-Chat: one-shot 71.25 / 53.16; window **3.58 / 1.39** (broken);
  StreamingLLM 71.34 / 55.03 (recovered).
- Llama-2-13B-Chat: one-shot 78.16 / 63.31; window 0.25 / 0.34;
  StreamingLLM 80.89 / 65.61.
- Llama-2-70B-Chat: one-shot 91.29 / 78.50; window 0.12 / 0.32;
  StreamingLLM 91.37 / 80.20.

Two things to point out:
1. Dense OOMs long before the stream ends.
2. StreamingLLM matches or slightly beats the one-shot baseline on
   every model size. This is a proxy for what a chatbot deployment
   would experience with many short turns fed back-to-back.

## Slide 19: What StreamingLLM does not do

Say this slowly. This is where the paper is honest, and this is
what a careful reader wants to see.

- It does **not** extend the context window. If the answer is older
  than the rolling window, the model cannot find it. Table 7 on
  p. 16 shows accuracy collapsing the moment query-answer distance
  exceeds cache size.
- On LongBench (Table 8, p. 17), StreamingLLM 4+3496
  underperforms the default 1750+1750 truncation baseline on all
  six tasks: NarrativeQA 11.6 vs 18.7, Qasper 16.9 vs 19.2,
  HotpotQA 21.6 vs 25.4, 2WikiMQA 28.2 vs 32.8, GovReport 23.9
  vs 27.3, MultiNews 25.5 vs 25.8. Parity is only restored when
  the sink budget is grown to 1750.
- Bigger cache does not always help. Llama-2-7B at 4+2044 gives
  9.08 PPL; at 4+4092 it gives 9.59. Table 6, p. 9. The paper
  reads this as an LLM limitation rather than a StreamingLLM one.
- The paper does not compare against any attention-score-based
  eviction policy (no H2O, SnapKV, FastGen, Keyformer). It is
  purely positional. That is its virtue (implementation cost) and
  its ceiling.
- The Sink-Token pre-training result is 160M parameters only.

## Slide 20: What I would change

Four concrete changes, in decreasing order of how confident I am
that they help:

1. **Per-layer sink budget.** Fig. 2 and Figs. 11-13 make clear
   that the two lowest layers barely use sinks at all. If we let
   `S_l` vary by layer, we can free budget in low layers and
   reallocate it to a longer rolling window in high layers.
2. **Attention-score eviction inside the window.** Keep the four
   positional sinks. But instead of FIFO on the rolling window,
   evict by H2O / SnapKV-style attention score. Positional plus
   content-aware.
3. **Real streaming traffic.** All language-modelling experiments
   are on PG19, long monolingual books. That does not match the
   multi-turn dialogue this method is marketed for. LMSYS-Chat-1M
   or ShareGPT would be a much fairer benchmark.
4. **Sink-Token pre-training at scale.** Section 3.3 is a
   promise; someone should train a 7B model with a sink token and
   publish downstream accuracy.

Then read the blue "same idea, one level up the stack" box aloud:

> StreamingLLM chooses positional eviction on the KV cache. My
> final project (Task 2) chooses cost-aware eviction on the
> response cache (GDSF on GPTCache). Same underlying problem,
> dumb eviction wastes bounded capacity, at different layers of
> the LLM serving stack.

This is my bridge. Say it out loud even if the audience read it.

## Slide 21: Conclusion

Read the five bullets, then the one-line take.

Key facts to hit:
- Attention sinks are structural (positional), not semantic.
- Four sinks plus a rolling window are enough to stream to 4M+
  tokens.
- Up to 22.2× faster than the only sane baseline, at essentially
  identical memory cost.
- Adopted upstream by NVIDIA TensorRT-LLM, Intel Extension for
  Transformers, Hugging Face Transformers, and MLC LLM within
  months (paper Impact Statement, p. 10).
- But: bounded cache, positional-only eviction, no comparison
  with content-aware methods.

One-line take: a robust empirical fix for a softmax quirk. Small,
easy to implement, and honest about its own ceiling.

## Slide 22: Thank you

Invite questions.

---

## Anticipated questions

**"Isn't this just window attention with a warm-up cache?"**
No. Window attention keeps the most-recent L tokens and lets the
initial ones fall out. That is exactly what breaks it (PPL 5158
in Table 1). StreamingLLM keeps four initial tokens permanently
*plus* a rolling window. That is the whole algorithmic difference,
and it is what keeps the softmax denominator stable.

**"Why exactly four sinks?"**
Table 2 (p. 5). One or two sinks leave a residual PPL bump; four
saturates the recovery; eight adds nothing. The paper's
hypothesis: vanilla pre-training has no single dedicated
start-token slot, so the model spreads its sinkage over the first
few positions.

**"Does it work with RoPE, with ALiBi, or with neither?"**
Both, as long as position IDs are indexed inside the cache rather
than in original-text coordinates. §3.2, p. 5.

**"Does it extend the context window?"**
No, and the paper is explicit about this. Table 7 (p. 16) and
Table 8 (p. 17) show accuracy collapsing when the query is older
than the rolling window. StreamingLLM is a policy for bounded
cache, not for infinite context.

**"How does it compare to H2O, SnapKV, FastGen, Keyformer?"**
The paper does not compare. Those methods evict by attention score
or by per-head profile (content-aware). StreamingLLM is
positional-only. A fair head-to-head would need a follow-up
paper.

**"Does it help long-doc QA or summarisation?"**
No. Appendix D, Table 8, p. 17. StreamingLLM 4+3496 underperforms
the default LongBench truncation baseline on all six tasks
because it loses the initial prompt. Only when the sink budget is
raised to 1750 does it match truncation. This is stated up front
in Appendix A.

**"How does this connect to your final project?"**
StreamingLLM decides what to keep in the KV cache by position;
my Task 2 project decides what to keep in the response cache by
cost (GDSF: Priority = Clock + freq^α · cost^β / size). Different
levels of the LLM serving stack; same underlying idea: a smart
eviction rule beats a naive one at fixed capacity.
