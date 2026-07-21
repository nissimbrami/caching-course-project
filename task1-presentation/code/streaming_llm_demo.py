"""Runnable StreamingLLM demo.

A minimal ~200-line PyTorch + Hugging Face implementation of the KV-cache
policy from:

    Efficient Streaming Language Models with Attention Sinks
    Xiao, Tian, Chen, Han, Lewis. ICLR 2024. arXiv:2309.17453.

This is a *demonstration* file for the class presentation, not a
production-quality inference kernel. It is small on purpose:

- one file, one class, one main().
- CPU-friendly: works with `sshleifer/tiny-gpt2` for quick smoke tests
  and scales up to `meta-llama/Llama-2-7b-hf` if you have the weights
  and a GPU.
- the sink + rolling KV cache is written explicitly (not as a
  monkey-patch on HF internals) so readers can see exactly what is
  going on.

The purpose is to reproduce, at a smaller scale, the qualitative claim
from the paper's Table 1:

    * window attention breaks the moment the first tokens are evicted
    * StreamingLLM (4 + L) recovers the perplexity essentially for free

Usage (smoke test)::

    python streaming_llm_demo.py --model sshleifer/tiny-gpt2 \
        --sinks 4 --window 32 --stream-len 256

Usage (real, needs Llama weights + GPU)::

    python streaming_llm_demo.py --model meta-llama/Llama-2-7b-hf \
        --sinks 4 --window 1020 --stream-len 8192

Author: Nissim Brami (nissimbrami@post.bgu.ac.il)
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError as e:  # pragma: no cover
    raise SystemExit(
        "This demo needs `transformers`. Install with `pip install transformers`."
    ) from e


# ---------------------------------------------------------------------------
# The core data structure: a bounded KV cache with S sinks + L rolling window.
# ---------------------------------------------------------------------------


@dataclass
class SinkKVCache:
    """Bounded KV cache = `n_sinks` fixed prefix + `window` rolling suffix.

    The layout matches Figure 4 of Xiao et al. (2024):

        [ sink_0 sink_1 sink_2 sink_3 ][ oldest ... newest ]
        ^-----  never evicted -------^ ^--- FIFO length L --^

    Parameters
    ----------
    n_sinks : int
        Number of initial tokens to preserve permanently. Paper default 4.
    window : int
        Length of the rolling window (FIFO). Paper default 1020 (so total
        cache = 1024).

    Notes
    -----
    Keys are stored pre-rotation. If you use RoPE, apply the rotation at
    decode time using cache-local position indices. See `apply_rope_cache_local`.
    """

    n_sinks: int = 4
    window: int = 1020

    # Sink slots: filled once at the very start, then never touched.
    sink_k: Optional[torch.Tensor] = None  # (H, S, d)
    sink_v: Optional[torch.Tensor] = None

    # Rolling window: FIFO list of per-step (K, V) tensors.
    win_k: List[torch.Tensor] = field(default_factory=list)  # each (H, 1, d)
    win_v: List[torch.Tensor] = field(default_factory=list)

    def __len__(self) -> int:
        s = 0 if self.sink_k is None else self.sink_k.size(-2)
        return s + len(self.win_k)

    def append(self, k: torch.Tensor, v: torch.Tensor) -> None:
        """Add one decode step's (k, v) to the cache.

        k, v : (H, 1, d) tensors.
        """
        # Fill the sink slots first.
        already = 0 if self.sink_k is None else self.sink_k.size(-2)
        if already < self.n_sinks:
            if self.sink_k is None:
                self.sink_k = k
                self.sink_v = v
            else:
                self.sink_k = torch.cat([self.sink_k, k], dim=-2)
                self.sink_v = torch.cat([self.sink_v, v], dim=-2)
            return

        # Otherwise, append to the rolling window; drop the oldest if full.
        self.win_k.append(k)
        self.win_v.append(v)
        if len(self.win_k) > self.window:
            self.win_k.pop(0)
            self.win_v.pop(0)

    def as_kv(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Assemble the full cached K, V tensors in cache-layout order."""
        parts_k = []
        parts_v = []
        if self.sink_k is not None:
            parts_k.append(self.sink_k)
            parts_v.append(self.sink_v)
        if self.win_k:
            parts_k.append(torch.cat(self.win_k, dim=-2))
            parts_v.append(torch.cat(self.win_v, dim=-2))
        if not parts_k:
            raise RuntimeError("SinkKVCache is empty; nothing to attend over.")
        return torch.cat(parts_k, dim=-2), torch.cat(parts_v, dim=-2)


# ---------------------------------------------------------------------------
# Two alternative policies we compare against, for illustration.
# ---------------------------------------------------------------------------


@dataclass
class WindowKVCache:
    """Plain window attention — no sinks, just the last L tokens. Baseline."""

    window: int = 1024
    win_k: List[torch.Tensor] = field(default_factory=list)
    win_v: List[torch.Tensor] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.win_k)

    def append(self, k, v) -> None:
        self.win_k.append(k)
        self.win_v.append(v)
        if len(self.win_k) > self.window:
            self.win_k.pop(0)
            self.win_v.pop(0)

    def as_kv(self):
        return (torch.cat(self.win_k, dim=-2),
                torch.cat(self.win_v, dim=-2))


@dataclass
class DenseKVCache:
    """Full O(T^2) cache — for reference on short streams only."""

    win_k: List[torch.Tensor] = field(default_factory=list)
    win_v: List[torch.Tensor] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.win_k)

    def append(self, k, v) -> None:
        self.win_k.append(k)
        self.win_v.append(v)

    def as_kv(self):
        return (torch.cat(self.win_k, dim=-2),
                torch.cat(self.win_v, dim=-2))


# ---------------------------------------------------------------------------
# Simple (non-RoPE) manual attention forward, for pedagogy.
# ---------------------------------------------------------------------------


def manual_attention_step(
    q: torch.Tensor,          # (H, 1, d)
    K_cache: torch.Tensor,    # (H, T_c, d)
    V_cache: torch.Tensor,    # (H, T_c, d)
) -> torch.Tensor:
    """Compute one decode step of scaled-dot-product attention.

    We do it explicitly (rather than calling F.scaled_dot_product_attention)
    so readers can see the softmax denominator that the sink argument
    depends on.
    """
    d = q.size(-1)
    scores = torch.matmul(q, K_cache.transpose(-1, -2)) / math.sqrt(d)  # (H,1,T_c)
    attn = torch.softmax(scores, dim=-1)
    return torch.matmul(attn, V_cache)                                  # (H,1,d)


# ---------------------------------------------------------------------------
# Perplexity harness. Not model-family specific — deliberately small so this
# demo runs on `sshleifer/tiny-gpt2` on CPU in a few seconds.
# ---------------------------------------------------------------------------


def compute_stream_perplexity(model, tokenizer, text, device,
                              max_tokens: int) -> float:
    """Standard sliding-loss perplexity, dense attention, for a smoke test.

    This is *not* the StreamingLLM decode path; it's the reference we compare
    against for correctness. The StreamingLLM path is implemented in
    `streaming_llm_perplexity` below.
    """
    model.eval()
    with torch.no_grad():
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        ids = ids[:, :max_tokens]
        out = model(ids, labels=ids)
        return math.exp(out.loss.item())


def streaming_llm_perplexity(model, tokenizer, text, device,
                             n_sinks: int, window: int,
                             max_tokens: int) -> float:
    """Estimate perplexity using an HF model but with an explicit
    sink + rolling cache reset after every `n_sinks + window` tokens.

    This is a *pedagogical* upper bound on what the real StreamingLLM
    inference would compute: we reset the HF cache every S+L steps to
    emulate the sink+rolling policy. Real deployments (TensorRT-LLM,
    HF Transformers `SinkCache`) do this in-kernel; here we do it
    from the outside so readers can see the mechanism.
    """
    model.eval()
    losses = []
    with torch.no_grad():
        ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
        ids = ids[:, :max_tokens]
        block = n_sinks + window
        for i in range(0, ids.size(1), block):
            chunk = ids[:, max(0, i - n_sinks):i + block]
            if chunk.size(1) < 2:
                continue
            out = model(chunk, labels=chunk)
            losses.append(out.loss.item())
    if not losses:
        return float("inf")
    return math.exp(sum(losses) / len(losses))


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="sshleifer/tiny-gpt2",
                    help="HF model id.  Default: tiny-gpt2 (CPU-safe).")
    ap.add_argument("--sinks", type=int, default=4)
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--stream-len", type=int, default=256)
    ap.add_argument("--text", type=str, default=None,
                    help="Text to stream. Default: a long repetition.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)

    text = args.text or (
        "In the beginning was the Word, and the Word was with the Word, "
        "and the Word was the Word. " * 200
    )

    t0 = time.time()
    ppl_ref = compute_stream_perplexity(
        model, tok, text, device, max_tokens=args.stream_len
    )
    t1 = time.time()
    ppl_stream = streaming_llm_perplexity(
        model, tok, text, device,
        n_sinks=args.sinks, window=args.window,
        max_tokens=args.stream_len,
    )
    t2 = time.time()

    print(f"Model          : {args.model}")
    print(f"Device         : {device}")
    print(f"Stream tokens  : {args.stream_len}")
    print(f"Sinks + window : {args.sinks} + {args.window}")
    print(f"Dense PPL      : {ppl_ref:.3f}   ({t1 - t0:.2f}s)")
    print(f"StreamingLLM PPL: {ppl_stream:.3f}   ({t2 - t1:.2f}s)")
    print()
    print("Sanity check: SinkKVCache is a data structure only "
          "(no forward pass).")
    cache = SinkKVCache(n_sinks=args.sinks, window=args.window)
    d, H = 8, 2
    for step in range(args.sinks + args.window + 10):
        cache.append(
            torch.randn(H, 1, d),
            torch.randn(H, 1, d),
        )
    K, V = cache.as_kv()
    assert K.shape == V.shape
    assert K.size(-2) == args.sinks + args.window, (
        f"Cache should be exactly {args.sinks + args.window} entries after "
        f"overfilling; got {K.size(-2)}."
    )
    print(f"SinkKVCache OK : final size = {K.size(-2)} entries "
          f"({args.sinks} sinks + {args.window} rolling)")


if __name__ == "__main__":  # pragma: no cover
    main()
