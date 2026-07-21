# StreamingLLM demo code

Minimal PyTorch + Hugging Face implementation of the sink + rolling KV
cache from Xiao et al., ICLR 2024, for the class presentation.

## Files

- `streaming_llm_demo.py` — the `SinkKVCache` data structure, alongside
  `WindowKVCache` and `DenseKVCache` baselines, plus a runnable `main()`
  that compares dense perplexity against a StreamingLLM-emulated pass.
- `test_sink_kv_cache.py` — pytest/unittest suite for the cache data
  structure. Runs in <1 s and does not need a downloaded model.

## Quick start (CPU, ~10 s)

```bash
python streaming_llm_demo.py \
    --model sshleifer/tiny-gpt2 \
    --sinks 4 --window 32 --stream-len 256
```

Expected output shape:

```
Model            : sshleifer/tiny-gpt2
Device           : cpu
Stream tokens    : 256
Sinks + window   : 4 + 32
Dense PPL        : ...
StreamingLLM PPL : ...
SinkKVCache OK   : final size = 36 entries (4 sinks + 32 rolling)
```

## Running the tests

```bash
python -m unittest test_sink_kv_cache.py
```

## Scaling up (GPU, real Llama)

```bash
python streaming_llm_demo.py \
    --model meta-llama/Llama-2-7b-hf \
    --sinks 4 --window 1020 \
    --stream-len 8192
```

Note: for the paper's actual 22.2× speedup and PPL-flat-to-4M-tokens
result, you need an A6000-class GPU, the Llama-2 weights, and PG19 as
input. The demo here reproduces the *qualitative* behaviour on a
small model in seconds; it's not a benchmark harness.

## What the demo *does* and *does not* prove

**Does**:

- Show that `SinkKVCache` is a small data structure (a `dataclass` with
  two lists and two tensors) — this is the "50 lines" claim from the
  paper.
- Show that the mechanism preserves the sink tokens across arbitrary
  overfilling (`test_sinks_never_touched`).
- Show that all three policies (Dense / Window / SinkKVCache) coexist
  under the same interface, so drop-in comparisons are trivial.

**Does not**:

- Reproduce paper Table 1 numbers on Llama-2-13B. That needs the actual
  weights and a GPU.
- Implement the RoPE cache-local re-indexing at the kernel level —
  we use the HF `SinkCache`-compatible reset trick in
  `streaming_llm_perplexity`. Production paths (TensorRT-LLM,
  HF Transformers) do this inside the attention kernel.

## References

- Xiao, Tian, Chen, Han, Lewis. *Efficient Streaming Language Models
  with Attention Sinks.* ICLR 2024. arXiv:2309.17453.
- Reference implementation: https://github.com/mit-han-lab/streaming-llm
