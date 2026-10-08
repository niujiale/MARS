"""
The joint substitution-rate tensor has two implementations that must agree.

One forms the Gram matrix of a dense one-hot tensor; the other counts the
co-occurring substitutions that actually exist. MARS picks between them per
batch based on how sparse the input is, so a user's results would silently
depend on their data unless the two produce the same bits.

They do, and not by luck: every entry is a count of reads, float32 represents
such integers exactly, and so the order the counts are accumulated in cannot
change the answer. These tests hold that guarantee to the standard of exact
equality across substitution rates, coverage levels and both strands, and also
check that the cheaper path is still the one chosen on realistic input -- an
optimization that quietly stops applying is worth catching.
"""

from __future__ import annotations

import numpy as np
import pytest

from nfl_py import stats as nfl_stats
from nfl_py.stats import pairwise_joint_sr_matrix


BLOCK = 21
DENSE_ONLY = 10 ** 18   # prices a counted pair so high the dense path wins


def make_batch(n_loci, n_reads, sub_rate, cover_frac, seed):
    """A batch tensor with a given substitution rate and read coverage."""
    rng = np.random.default_rng(seed)
    ref = rng.integers(1, 5, size=(n_loci, BLOCK)).astype(np.int8)

    covered = rng.random((n_loci, n_reads)) < cover_frac
    M_base = np.repeat(covered[:, :, None], BLOCK, axis=2)
    # Ragged edges: a read rarely spans a whole window.
    M_base &= rng.random((n_loci, n_reads, BLOCK)) < 0.9

    B = np.broadcast_to(ref[:, None, :], (n_loci, n_reads, BLOCK)).copy()
    subs = rng.random((n_loci, n_reads, BLOCK)) < sub_rate
    shift = rng.integers(1, 5, size=B.shape).astype(np.int8)
    B = np.where(subs, ((B - 1 + shift) % 5) + 1, B).astype(np.int8)
    B[~M_base] = 0
    return B, M_base, ref.astype(np.int64)


def both_paths(B, M_base, ref, reverse):
    saved = nfl_stats._SPARSE_JOINT_PAIR_COST
    try:
        nfl_stats._SPARSE_JOINT_PAIR_COST = DENSE_ONLY
        dense = pairwise_joint_sr_matrix(B, M_base, ref, reverse=reverse)
        nfl_stats._SPARSE_JOINT_PAIR_COST = saved
        auto = pairwise_joint_sr_matrix(B, M_base, ref, reverse=reverse)
    finally:
        nfl_stats._SPARSE_JOINT_PAIR_COST = saved
    return dense, auto


@pytest.mark.parametrize("sub_rate", [0.0, 0.02, 0.07, 0.3, 0.75, 1.0])
@pytest.mark.parametrize("cover_frac", [0.02, 0.5, 1.0])
@pytest.mark.parametrize("reverse", [False, True])
def test_joint_sr_paths_are_bit_identical(sub_rate, cover_frac, reverse):
    B, M_base, ref = make_batch(
        n_loci=8, n_reads=60, sub_rate=sub_rate, cover_frac=cover_frac,
        seed=hash((sub_rate, cover_frac, reverse)) % (2 ** 32),
    )
    dense, auto = both_paths(B, M_base, ref, reverse)

    assert dense.shape == auto.shape
    assert dense.dtype == auto.dtype
    assert np.array_equal(dense, auto), (
        f"max|diff|={np.abs(dense - auto).max()} at sub_rate={sub_rate}, "
        f"cover={cover_frac}, reverse={reverse}"
    )


def test_counting_path_is_used_on_realistic_input():
    """Sparse input should take the cheaper route, not silently fall back."""
    B, M_base, ref = make_batch(n_loci=40, n_reads=400, sub_rate=0.07,
                                cover_frac=0.05, seed=5)
    sel = (B != ref[:, None, :]) & M_base & (B >= 1) & (B <= 5)
    counts = nfl_stats._joint_sr_counts_sparse(sel, B, BLOCK, reverse=False)
    assert counts is not None


def test_dense_path_is_used_when_counting_would_be_worse():
    """A saturated batch must fall back rather than count every pair."""
    B, M_base, ref = make_batch(n_loci=40, n_reads=400, sub_rate=1.0,
                                cover_frac=1.0, seed=6)
    sel = (B != ref[:, None, :]) & M_base & (B >= 1) & (B <= 5)
    counts = nfl_stats._joint_sr_counts_sparse(sel, B, BLOCK, reverse=False)
    assert counts is None


def test_empty_substitutions_still_normalizes():
    """No substitutions anywhere: result is all zeros, not a divide-by-zero."""
    B, M_base, ref = make_batch(n_loci=4, n_reads=30, sub_rate=0.0,
                                cover_frac=1.0, seed=7)
    dense, auto = both_paths(B, M_base, ref, reverse=False)
    assert np.array_equal(dense, auto)
    assert np.all(np.isfinite(auto))
    assert not np.any(auto)
