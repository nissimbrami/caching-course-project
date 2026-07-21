# StreamingLLM — Deep-Read

Every claim below is grounded in the paper text I have read. Every number
is followed by a `[p.N]` reference. The point of this file is that anyone
building the deck can trace every slide claim back to a specific page of
the PDF at `task1-presentation/papers/streaming-llm/streaming-llm.pdf`.

## Bibliographic block

- **Title:** Efficient Streaming Language Models with Attention Sinks
- **Authors (exact order, from p.1):** Guangxuan Xiao, Yuandong Tian,
  Beidi Chen, Song Han, Mike Lewis.
- **Affiliations (from p.1 superscripts):**
  1. Guangxuan Xiao — MIT
  2. Yuandong Tian — Meta AI
  3. Beidi Chen — Carnegie Mellon University
  4. Song Han — MIT and NVIDIA (joint superscript 1,4)
  5. Mike Lewis — Meta AI
  Footnote on p.1: "Part of the work done during an internship at Meta AI."
- **Venue:** ICLR 2024. The paper's running header says "Published as a
  conference paper at ICLR 2024" on every page.
- **arXiv:** 2309.17453v4, dated 7 Apr 2024 (bottom of p.1).
- **Code:** https://github.com/mit-han-lab/streaming-llm (p.1 title block).

## One-paragraph pitch (paraphrased faithfully)

Naive sliding-window attention breaks the moment the *first* tokens fall
out of the KV cache. The reason is not that those first tokens carry
irreplaceable semantics — it is that softmax over the attention logits
must sum to one, and the model has learned to *dump excess attention* on
whichever tokens are visible to everyone. In a decoder-only causal model,
the tokens visible to everyone are the initial ones. They become
"attention sinks." Evicting them takes a large chunk out of the softmax
denominator, the distribution collapses, and perplexity explodes.
StreamingLLM fixes this at inference time by always keeping the first
four tokens in the KV cache (as "sink" tokens) plus a rolling window of
the most recent L tokens. Position IDs are re-assigned within the cache,
not from the original text. No fine-tuning is required. Optionally, if
you can pre-train from scratch, one dedicated learnable sink token
prepended to every training sample lets a single token play the role of
the four initial ones.

## Three pillars of the paper

1. **The empirical phenomenon** — "attention sink" (§3.1).
   Beyond the bottom two layers, attention concentrates heavily on the
   initial tokens across every layer and head, regardless of their
   semantic content [p.3 Fig.2, p.4].

2. **The training-free fix** — sink + rolling KV cache (§3.2).
   Keep `S` sink tokens (default 4) plus the last `L` tokens. Position
   IDs are indexed inside the cache, not in the original stream. Works
   with RoPE and ALiBi [p.5 Fig.4].

3. **The pre-training refinement** — learnable Sink Token (§3.3).
   Prepend a single learnable token to every pre-training sample. Same
   convergence, same downstream accuracy, but one sink token suffices at
   inference [p.6 Table 3, p.7 Fig.7].

## §1 — Introduction

- Two challenges stated for streaming inference: (i) KV cache grows
  unboundedly with sequence length; (ii) LLMs pretrained at a fixed
  window `L_train` do not generalise to lengths `>> L_train` [p.2].
- The paper's central question: *Can we deploy an LLM for infinite-length
  inputs without sacrificing efficiency and performance?* [p.1, verbatim].
- StreamingLLM enables Llama-2-{7,13,70}B, MPT-{7,30}B,
  Falcon-{7,40}B, and Pythia-{2.8,6.9,12}B to stably model **4M+ tokens**
  [p.2, p.6].
- Against sliding-window-with-recomputation (the only baseline with
  acceptable quality), StreamingLLM achieves **up to 22.2× per-token
  speedup** [p.2 abstract; p.9 Fig.10].

## §2 — Related work

Three axes are named [pp.3–4]:

1. **Length Extrapolation** — RoPE (Su et al. 2021), ALiBi
   (Press et al. 2022). StreamingLLM sits in this bucket.
2. **Context Window Extension** — FlashAttention (Dao 2022, 2023),
   Position Interpolation (Chen 2023), YaRN (Peng 2023), etc.
3. **Improving Utilisation of Long Text** — Liu et al.
   "Lost in the Middle" and similar.

The paper is explicit that it does **not** try to expand the context
window nor improve memory-of-long-text — it only stably operates on the
most recent tokens with a bounded cache.

## §3 — The mechanism, precisely

### §3.1 Why window attention fails

- Figure 3 (p.4): perplexity of dense attention, window attention,
  sliding-with-recomputation, and StreamingLLM on a 20K-token text.
  Perplexity spikes for window attention exactly at the point the cache
  fills and the first token is evicted.
- Equation 1 (p.4): softmax denominator argument. If the initial-token
  logit `x_1 ≫ x_j` for `j ≥ 2`, dropping `x_1` removes a huge chunk of
  `sum_j exp(x_j)` and warps the distribution.
- **Table 1 (p.5) — the crown-jewel result of §3.1:**

  | Cache config (Llama-2-13B, PG19 first book, 65K tokens) | PPL ↓ |
  |---|---|
  | `0 + 1024` (window, no sink)     | **5158.07** |
  | `4 + 1020` (StreamingLLM)         | **5.40** |
  | `4×"\n" + 1020` (linebreak sinks) | **5.60** |

  The linebreak-substitution shows that the sinks work because of their
  *position*, not their *content*.

### §3.1 continued — Table 2 (p.5): how many sinks are enough?

  Across Llama-2-7B, Falcon-7B, MPT-7B, Pythia-12B, at cache size = half
  the pre-training window, adding **4 sinks is enough**; 1 or 2 sinks
  leave a residual PPL bump; adding 8 gives diminishing returns.

  Falcon-7B example: 0+2048 → 17.90 PPL, 1+2047 → 12.12, 4+2044 → 12.12,
  8+2040 → 12.12.
  MPT-7B is the loudest example of window failure: 0+2048 → 460.29,
  1+2047 → 14.99, 4+2044 → 14.99.
  Llama-2-7B, cache 4096: 0+4096 → 3359.95, 1+4095 → 11.88, 2+4094 →
  10.51, 4+4092 → 9.59, 8+4088 → 9.54.

### §3.2 Rolling KV cache with attention sinks

- Cache layout: `[S sink tokens] + [rolling window of length L]`
  (Fig.4, p.5).
- **Critical implementation detail (p.5):** relative position IDs are
  computed on *cache indices*, not on original-text token indices.
  So if the cache holds tokens with original positions [0,1,2,3,6,7,8]
  and is decoding position 9, the model sees relative positions
  [0,1,2,3,4,5,6,7], not [0,1,2,3,6,7,8,9].
- With RoPE: cache the pre-rotary keys and re-apply RoPE at decode time
  using the cache-local index.
- With ALiBi: apply a contiguous linear bias in cache coordinates rather
  than a "jumping" bias in original coordinates.

### §3.3 Pre-training with a sink token

- The paper argues that if a model *had* a dedicated always-visible sink
  slot during pre-training, it would concentrate all excess attention
  there rather than smearing it across the first 4 real tokens.
- Two alternatives are proposed:
  - **Sink Token** — a learnable placeholder prepended to every
    training sample.
  - **Zero Sink** — replace SoftMax with SoftMax-off-by-One (Miller
    2023), Eq.2 (p.6). Adds a constant `1` to the denominator, which is
    equivalent to prepending an all-zero-K, all-zero-V token.
- **Table 3 (p.6):** 160M-parameter models trained from scratch on
  deduplicated Pile, 143K steps, batch 256.

  | Cache config       | 0+1024 | 1+1023 | 2+1022 | 4+1020 |
  |---|---|---|---|---|
  | Vanilla            | 27.87  | 18.49  | 18.05  | 18.05  |
  | Zero Sink          | 29214  | 19.90  | 18.27  | 18.01  |
  | Learnable Sink     | 1235   | 18.01  | 18.01  | 18.02  |

  Takeaway: the learnable sink token alone (1+1023) already matches what
  vanilla needs 4 tokens to achieve. Zero Sink barely helps.

## §4 — Experiments

### §4.1 Language modelling on long texts (p.6)

- Dataset: concatenated PG19 test set — 100 long books.
- Cache: 2048 for Llama-2, 1024 for Falcon/Pythia/MPT (half the
  pre-training window, chosen for readability).
- Figure 3 shows StreamingLLM tracking the sliding-window-with-recomp
  baseline on all four model families up to 20K tokens.
- Figure 5 (p.7) shows perplexity stays flat for **4M+ tokens** across
  Llama-2-[7,13,70]B, Falcon-[7,40]B, Pythia-[2.8,6.9,12]B, and
  MPT-[7,30]B.

### §4.2 Sink-token pre-training results (p.7)

- Figure 6 shows the sink-token pre-training loss curve is
  indistinguishable from vanilla.
- Table 4 (p.7): zero-shot accuracy across ARC-c, ARC-e, HellaSwag,
  LAMBADA, OpenbookQA, PIQA, Winogrande. Vanilla vs +Sink Token differs
  by <1 point on every task and slightly *higher* in the +Sink column.
- Figure 7 (p.7): attention heatmaps confirm the sink token draws the
  attention that would otherwise smear over multiple initial tokens.

### §4.3 Streaming QA with instruction-tuned models (p.8)

- **Table 5 (p.8): concatenated ARC on Llama-2-{7,13,70}B-Chat, cache
  1024:**

  | Model                        | ARC-E | ARC-C |
  |---|---|---|
  | Llama-2-7B-Chat, one-shot    | 71.25 | 53.16 |
  | Llama-2-7B-Chat, Window      | 3.58  | 1.39  |
  | Llama-2-7B-Chat, StreamingLLM| 71.34 | 55.03 |
  | Llama-2-13B-Chat, one-shot   | 78.16 | 63.31 |
  | Llama-2-13B-Chat, Window     | 0.25  | 0.34  |
  | Llama-2-13B-Chat, StreamingLLM| 80.89 | 65.61|
  | Llama-2-70B-Chat, one-shot   | 91.29 | 78.50 |
  | Llama-2-70B-Chat, Window     | 0.12  | 0.32  |
  | Llama-2-70B-Chat, StreamingLLM| 91.37 | 80.20|

  Dense OOMs. Window is essentially random. StreamingLLM matches or
  slightly beats one-shot.

- StreamEval (Fig.8, Fig.9, p.8): the paper's own long-eval-style
  benchmark. A query is issued every 10 lines; the answer is always 20
  lines back. StreamingLLM stays accurate up to ~120K tokens where
  dense/window collapse.

### §4.4 Ablations (p.9)

- **Number of sinks:** 4 is the sweet spot (Table 2 result restated).
- **Cache size (Table 6, p.9):** increasing the rolling window does *not*
  monotonically lower PPL. Example, Llama-2-7B: 4+508 → 9.73, 4+1020 →
  9.32, 4+2044 → 9.08, 4+4092 → 9.59. The paper reads this as an
  under-utilisation limitation of current LLMs, not a StreamingLLM bug.

### §4.5 Efficiency results (p.9)

- Figure 10 (p.9): single NVIDIA A6000, Hugging Face Transformers,
  Llama-2-7B and Llama-2-13B, cache sizes 256/512/1024/2048/4096.
- On Llama-2-13B at cache **256**, per-token latency:
  - Sliding-window-with-recomputation: **2355 ms**
  - StreamingLLM: **106 ms**
  - Ratio: ~22.2× — this is exactly the "up to 22.2×" number in the
    abstract.
- Full Llama-2-13B latency table (Fig. 10, p. 9):

  | Cache | Sliding+recompute | StreamingLLM | Speed-up |
  |---|---|---|---|
  | 256  | 2355 | 106 | 22.2× |
  | 512  | 860  | 75  | 11.5× |
  | 1024 | 361  | 60  | 6.0×  |
  | 2048 | 169  | 52  | 3.3×  |
  | 4096 | 99   | 48  | 2.1×  |

- Llama-2-7B latency table (Fig. 10, p. 9):

  | Cache | Sliding+recompute | StreamingLLM |
  |---|---|---|
  | 256  | 1411 | 65 |
  | 512  | 523  | 45 |
  | 1024 | 223  | 35 |
  | 2048 | 103  | 31 |
  | 4096 | 63   | 31 |

- Memory footprint (both models, both baselines) is essentially
  identical — both bounded by cache size, not by original length.

## §5 — Conclusion (p.9)

- "StreamingLLM firstly decouples the LLM's pre-training window size and
  its actual text generation length."
- No fine-tuning required.
- Learnable sink token further helps if you control pre-training.
- Impact statement (p.10): adopted by NVIDIA TensorRT-LLM, Intel
  Extension for Transformers, Hugging Face Transformers, MLC LLM.

## Appendices worth knowing about

- **A. Applications & limitations (p.15).** StreamingLLM is for
  short-term-memory streaming (multi-round dialogue). It does *not*
  extend context or add long-term memory. Long-doc QA and summarisation
  are explicit non-goals.
- **B. Sparse transformers comparison (p.15).** Sparse Transformer,
  Longformer, ETC, BigBird — all incompatible with pretrained
  autoregressive LMs and mostly need custom kernels. StreamingLLM plugs
  into standard GPU kernels and standard HF models.
- **B. Concurrent work — LM-Infinite (Han et al. 2023) (p.15).**
  Λ-shaped attention pattern; similar spirit. StreamingLLM's separate
  contribution is naming and characterising the "attention sink"
  phenomenon and showing it exists in encoders and ViTs too.
- **C. StreamEval distance-vs-accuracy (Table 7, p.16).** With
  Llama-2-7B-32K-Instruct + StreamingLLM at cache 4+16380, accuracy
  drops off the moment the query-answer distance exceeds cache size.
  This is presented as a *limitation*: the paper does not extend
  context.
- **D. LongBench (Table 8, p.17).** StreamingLLM 4+3496 underperforms
  truncation 1750+1750 on all six tasks — NarrativeQA (11.6 vs 18.7),
  Qasper (16.9 vs 19.2), HotpotQA (21.6 vs 25.4), 2WikiMQA (28.2 vs
  32.8), GovReport (23.9 vs 27.3), MultiNews (25.5 vs 25.8, essentially
  tied). Once the sink budget is raised to 1750, StreamingLLM 1750+1750
  matches truncation. Again: bounded by what fits in the cache.
- **E–G. Attention visualisations on longer sequences and Llama-2-70B
  (pp.17–19).** The sink phenomenon holds across depths and model
  scales.
- **H. Attention sinks in BERT (p.20).** BERT-base-uncased dumps
  attention onto `[SEP]`. Same phenomenon in an encoder.
- **I. More sinks in pre-training (p.21).** Two sink tokens don't help
  more than one on downstream tasks. Different from ViT registers
  (Darcet et al. 2023), which benefit from multiple.

## Datasets and models used (exhaustive)

- **Language modelling:** PG19 (100 long books).
- **NLP zero-shot:** ARC-Challenge, ARC-Easy, HellaSwag, LAMBADA,
  OpenbookQA, PIQA, Winogrande (§4.2, Table 4).
- **Instruction-tuned streaming QA:** ARC-Easy, ARC-Challenge
  concatenated (§4.3, Table 5).
- **Long-QA style:** StreamEval (the paper's own, inspired by LongEval;
  §4.3, Fig.8, Fig.9).
- **Long-context truncation baseline:** LongBench (NarrativeQA, Qasper,
  HotpotQA, 2WikiMQA, GovReport, MultiNews; Table 8).
- **Pre-training corpus:** deduplicated Pile (§4.2).
- **Models evaluated:** Llama-2-{7,13,70}B and their Chat variants,
  MPT-{7,30}B, Falcon-{7,40}B, Pythia-{2.8,6.9,12}B, Llama-2-7B-32K-Instruct,
  LongChat-7B-v1.5-32K.
- **Pre-training experiments:** 160M-parameter Pythia-160M-style models
  trained from scratch on 8× A6000, batch 256, 143K steps.

## Baselines the paper explicitly compares against

- **Dense attention** — full O(T²) cache. Fails on inputs longer than
  the pre-training window.
- **Window attention (Beltagy 2020)** — most recent L tokens only.
  Fails the moment the first token is evicted.
- **Sliding window with re-computation** — the "oracle" baseline;
  matches quality but is O(TL²) and slow.
- **Truncation 1750+1750** — LongBench's own default (Table 8).
- **One-shot sample-by-sample** — ARC baseline (Table 5).

## Headline numbers (curated, with sources)

- **22.2× per-token speedup vs sliding-window-with-recomputation** —
  Llama-2-13B, **cache 256**, 2355 ms → 106 ms [p.9 Fig.10].
- **PPL 5158.07 → 5.40** on Llama-2-13B, PG19, first 65K tokens, by
  adding 4 sinks to a 1024-token window [p.5 Table 1].
- **4M+ tokens** — stable perplexity on Llama-2-{7,13,70}B, Falcon,
  Pythia, MPT [p.7 Fig.5].
- **1 learnable sink token** replaces 4 accidental sinks after
  pre-training with the fix [p.6 Table 3].
- **Adopted upstream** by NVIDIA TensorRT-LLM, Intel Extension for
  Transformers, Hugging Face Transformers, MLC LLM [p.10 Impact].

## Where the paper is honest about limits

Pulled straight from the appendix (A, C, D). The deck must show these:

1. **Does not extend context** — the model can only reason about the
   tokens *inside* the cache. If the answer is older than the rolling
   window, StreamingLLM cannot find it (Table 7, Fig.9).
2. **Not for long-doc QA or summarisation** — cache-window bounded.
3. **Cache size ≠ perplexity monotonic** (Table 6, Llama-2-7B goes
   4+2044: 9.08 → 4+4092: 9.59). Current LLMs don't fully use the cache
   they're given.
4. **Sink Token pre-training is expensive** — the fix is free if you
   have a working LLM; the *better* fix requires re-pre-training.

## Where a course project could plausibly extend the paper

Only for our internal planning (this section is not for the deck).

- Adaptive number of sink tokens per layer (Fig.11/Fig.13 show layer
  variance).
- Learned per-head cache budget.
- Combine sinks with H2O/SnapKV-style attention-based eviction *inside*
  the rolling window.
- Sink-aware quantisation.
- StreamingLLM + a cost-aware admission/eviction *at the response
  level* — this is exactly the bridge to the Task 2 project on GPTCache
  + GDSF: StreamingLLM shows a positional cache strategy; my Task 2
  shows a cost-aware one, at a different level of the stack.
