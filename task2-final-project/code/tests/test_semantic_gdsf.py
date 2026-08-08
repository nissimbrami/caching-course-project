"""Tests for SemanticGDSFManager and its POLICY_REGISTRY adapter."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cost_aware_eviction.semantic_gdsf import (
    SemanticGDSFManager,
    _Embedder,
    _hash_project,
    cosine,
)
from benchmarks.policies import POLICY_REGISTRY, create_policy


# --------------------------------------------------------------------------- #
# Hash-projection embedder                                                    #
# --------------------------------------------------------------------------- #

def test_hash_projection_deterministic():
    v1 = _hash_project("hello world", dim=64)
    v2 = _hash_project("hello world", dim=64)
    assert np.allclose(v1, v2)


def test_hash_projection_norm_one_or_zero():
    v = _hash_project("something meaningful here", dim=128)
    n = np.linalg.norm(v)
    assert n == pytest.approx(1.0, abs=1e-6) or n == 0.0


def test_hash_projection_different_texts_differ():
    v1 = _hash_project("the quick brown fox", dim=128)
    v2 = _hash_project("something entirely different", dim=128)
    assert cosine(v1, v2) < 0.9  # nowhere near identical


# --------------------------------------------------------------------------- #
# Embedder fallback ordering                                                  #
# --------------------------------------------------------------------------- #

def test_embedder_hash_backend_direct():
    e = _Embedder(backend="hash", dim=64)
    assert e.backend == "hash"
    v = e.embed("test prompt")
    assert v.shape == (64,)


def test_embedder_auto_backend_lands_somewhere():
    e = _Embedder(backend="auto", dim=64)
    assert e.backend in ("sentence-transformers", "tfidf", "hash")
    v = e.embed("hi")
    assert v.ndim == 1


# --------------------------------------------------------------------------- #
# SemanticGDSFManager: basic contract                                         #
# --------------------------------------------------------------------------- #

def test_semantic_manager_fresh_miss():
    m = SemanticGDSFManager(max_size=1000, embedder_backend="hash")
    hit, _, sim = m.semantic_lookup("anything")
    assert hit is False


def test_semantic_manager_exact_hit():
    m = SemanticGDSFManager(max_size=1000, embedder_backend="hash", tau=0.85)
    m.semantic_put("k1", prompt="how do I train a neural network", size=100, cost=1.0)
    hit, key, sim = m.semantic_lookup("how do I train a neural network")
    assert hit is True
    assert key == "k1"
    assert sim >= 0.999


def test_semantic_manager_semantic_hit_below_exact():
    """A paraphrase should still hit under a moderate tau."""
    m = SemanticGDSFManager(max_size=1000, embedder_backend="hash", tau=0.3)
    m.semantic_put("k1", prompt="the quick brown fox jumps", size=100, cost=1.0)
    # A partial-overlap prompt: hash projection preserves token identity so
    # cosine remains substantial for overlapping vocabulary.
    hit, key, sim = m.semantic_lookup("the quick brown fox")
    assert hit is True
    assert key == "k1"


def test_semantic_manager_semantic_miss_when_tau_high():
    m = SemanticGDSFManager(max_size=1000, embedder_backend="hash", tau=0.999)
    m.semantic_put("k1", prompt="one string", size=100, cost=1.0)
    hit, _, _ = m.semantic_lookup("a completely different string")
    assert hit is False


# --------------------------------------------------------------------------- #
# Radius bookkeeping                                                          #
# --------------------------------------------------------------------------- #

def test_radius_increments_on_hits():
    m = SemanticGDSFManager(max_size=1000, embedder_backend="hash", tau=0.5)
    m.semantic_put("k1", prompt="foo bar baz", size=100, cost=1.0)
    r0 = m._radius["k1"]
    for _ in range(5):
        m.semantic_lookup("foo bar baz")
    assert m._radius["k1"] >= r0 + 5


def test_radius_wiped_on_eviction():
    """A key evicted to make room must have its embedding + radius cleaned."""
    m = SemanticGDSFManager(max_size=200, embedder_backend="hash")
    m.semantic_put("k1", prompt="alpha", size=100, cost=0.01)
    m.semantic_put("k2", prompt="beta", size=100, cost=0.01)
    # Force eviction with a third insert
    m.semantic_put("k3", prompt="gamma", size=100, cost=10.0)
    # At least one of {k1, k2} evicted; check bookkeeping consistent
    resident = set(m._embeddings.keys())
    assert resident.issubset({"k1", "k2", "k3"})
    for k in ("k1", "k2", "k3"):
        if k not in resident:
            assert k not in m._radius


# --------------------------------------------------------------------------- #
# Priority formula includes radius                                            #
# --------------------------------------------------------------------------- #

def test_priority_grows_with_radius():
    m = SemanticGDSFManager(
        max_size=1000, alpha=1.0, beta=1.0, gamma=1.0,
        tau=0.5, embedder_backend="hash",
    )
    m.semantic_put("k1", prompt="hello world", size=100, cost=1.0)
    p0 = m.get_priority("k1")
    for _ in range(10):
        m.semantic_lookup("hello world")
    p1 = m.get_priority("k1")
    assert p1 > p0


# --------------------------------------------------------------------------- #
# Determinism                                                                 #
# --------------------------------------------------------------------------- #

def test_determinism_across_runs():
    """Same trace + same tau + hash embedder ==> identical hit patterns."""
    def run():
        m = SemanticGDSFManager(max_size=500, embedder_backend="hash", tau=0.9)
        hits = []
        for i, prompt in enumerate([
            "alpha beta gamma", "alpha beta gamma", "delta epsilon zeta",
            "alpha beta gamma", "eta theta iota",
        ]):
            hit, _, _ = m.semantic_lookup(prompt)
            hits.append(hit)
            if not hit:
                m.semantic_put(f"k{i}", prompt=prompt, size=50, cost=1.0)
        return hits
    assert run() == run()


# --------------------------------------------------------------------------- #
# Registry integration                                                        #
# --------------------------------------------------------------------------- #

def test_registered_in_policy_registry():
    assert "SemanticGDSF" in POLICY_REGISTRY


def test_factory_creates_policy():
    p = create_policy(
        "SemanticGDSF", max_size=1024,
        alpha=1.0, beta=1.0, gamma=0.5, tau=0.85,
        embedder_backend="hash", embed_dim=64,
    )
    assert p.max_size == 1024
    assert p.current_size == 0
    # Put/access smoke
    ev = p.put("hello", size=100, cost=1.0)
    assert ev == []
    assert p.access("hello") is True
