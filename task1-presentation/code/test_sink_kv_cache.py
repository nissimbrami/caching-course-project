"""Small unit test for the SinkKVCache data structure.

Runs in <1s. Does not require Hugging Face or a downloaded model —
just PyTorch. This is the regression net for the demo code.

Run with::

    python -m unittest task1-presentation/code/test_sink_kv_cache.py
"""

import unittest

import torch

from streaming_llm_demo import SinkKVCache, WindowKVCache, DenseKVCache


H, D = 2, 8


def _rand():
    return torch.randn(H, 1, D), torch.randn(H, 1, D)


class TestSinkKVCache(unittest.TestCase):

    def test_sinks_fill_first(self):
        c = SinkKVCache(n_sinks=4, window=32)
        for _ in range(4):
            c.append(*_rand())
        # After 4 appends, cache is 4 sinks + 0 rolling = 4 total.
        self.assertEqual(len(c), 4)
        self.assertIsNotNone(c.sink_k)
        self.assertEqual(c.sink_k.size(-2), 4)
        self.assertEqual(len(c.win_k), 0)

    def test_rolling_bounded(self):
        c = SinkKVCache(n_sinks=4, window=32)
        # Fill sinks + rolling + 10 extras.
        for _ in range(4 + 32 + 10):
            c.append(*_rand())
        self.assertEqual(len(c), 4 + 32)
        K, V = c.as_kv()
        self.assertEqual(K.size(-2), 4 + 32)
        self.assertEqual(V.size(-2), 4 + 32)

    def test_sinks_never_touched(self):
        c = SinkKVCache(n_sinks=4, window=8)
        # Seed sinks with distinguishable keys (all ones).
        for _ in range(4):
            k = torch.ones(H, 1, D)
            v = torch.ones(H, 1, D)
            c.append(k, v)
        # Overfill the rolling window with zeros; sinks must survive.
        for _ in range(100):
            c.append(torch.zeros(H, 1, D), torch.zeros(H, 1, D))
        K, _ = c.as_kv()
        # First 4 rows are the "all ones" sinks (per head).
        self.assertTrue(torch.all(K[:, :4, :] == 1.0))

    def test_window_cache_is_bounded(self):
        c = WindowKVCache(window=16)
        for _ in range(50):
            c.append(*_rand())
        K, _ = c.as_kv()
        self.assertEqual(K.size(-2), 16)

    def test_dense_cache_grows(self):
        c = DenseKVCache()
        for _ in range(20):
            c.append(*_rand())
        K, _ = c.as_kv()
        self.assertEqual(K.size(-2), 20)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
