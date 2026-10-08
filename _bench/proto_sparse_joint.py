#!/usr/bin/env python3
"""
Prototype: joint substitution-rate tensor by sparse co-occurrence counting.

The dense path materializes a (n_loci, n_reads, 5*block) one-hot tensor and
takes its Gram matrix through BLAS. Two facts make that wasteful:

  * A read covers one 21-base window, so for a batch whose loci are spread out,
    almost every (locus, read) pair is empty.
  * Within a covered window only substituted positions contribute, and real
    nanopore data substitutes a few percent of bases.

The products being summed are all 0.0 or 1.0, so every entry of the result is
an exact integer count well below 2**24. Float32 can represent those exactly,
which means the summation order is irrelevant and a counting implementation is
bit-identical to the matmul rather than merely close.

This script checks that claim and measures both paths as the substitution rate
varies from realistic to pathological.
"""

import argparse
import time

import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nfl_py.stats import (  # noqa: E402
    pairwise_joint_sr_matrix, _COMPLEMENT_PERM,
)


def joint_sr_sparse(B, M_base, ref_idx, reverse=False):
    """Counting implementation of pairwise_joint_sr_matrix for 3D batches."""
    n_loci, n_reads, block = B.shape
    flat = 5 * block

    sel = (B != ref_idx[:, None, :]) & M_base & (B >= 1) & (B <= 5)

    l_idx, r_idx, s_idx = np.nonzero(sel)
    b_idx = B[l_idx, r_idx, s_idx].astype(np.int64) - 1
    if reverse:
        b_idx = _COMPLEMENT_PERM[b_idx]
    slot = b_idx * block + s_idx

    # Entries are already grouped by (locus, read) because np.nonzero walks in
    # C order. Each group contributes every ordered pair of its own entries.
    counts = sel.sum(-1).ravel()                       # per (locus, read)
    counts = counts[counts > 0]
    ends = np.cumsum(counts)
    starts = ends - counts

    per_entry_count = np.repeat(counts, counts)
    per_entry_start = np.repeat(starts, counts)
    total = int(per_entry_count.sum())

    left = np.repeat(np.arange(l_idx.size, dtype=np.int64), per_entry_count)
    within = (np.arange(total, dtype=np.int64)
              - np.repeat(np.cumsum(per_entry_count) - per_entry_count,
                          per_entry_count))
    right = np.repeat(per_entry_start, per_entry_count) + within

    dest = (l_idx[left] * (flat * flat) + slot[left] * flat + slot[right])
    var_flat = np.bincount(dest, minlength=n_loci * flat * flat)
    var_flat = var_flat.astype(np.float32).reshape(n_loci, flat, flat)

    var_joint = var_flat.reshape(n_loci, 5, block, 5, block).transpose(
        0, 1, 3, 2, 4
    )

    M_f = M_base.astype(np.float32)
    n_ij = M_f.swapaxes(-1, -2) @ M_f
    denom = np.where(n_ij > 0, n_ij, 1.0)
    return (var_joint / denom[:, None, None, :, :]).astype(np.float32)


def make_batch(n_loci, n_reads, block, sub_rate, cover_frac, rng):
    """
    Synthesize a batch tensor with a given substitution rate and coverage.

    cover_frac is the fraction of (locus, read) pairs that overlap at all,
    which is what a batch spanning many loci actually looks like.
    """
    ref = rng.integers(1, 5, size=(n_loci, block)).astype(np.int8)

    covered = rng.random((n_loci, n_reads)) < cover_frac
    M_base = np.broadcast_to(covered[:, :, None], (n_loci, n_reads, block))
    M_base = M_base.copy()

    B = np.broadcast_to(ref[:, None, :], (n_loci, n_reads, block)).copy()
    subs = rng.random((n_loci, n_reads, block)) < sub_rate
    shift = rng.integers(1, 5, size=B.shape).astype(np.int8)
    B = np.where(subs, ((B - 1 + shift) % 5) + 1, B).astype(np.int8)
    B[~M_base] = 0
    return B, M_base, ref.astype(np.int64)


def timed(fn, repeat=3):
    fn()
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    return (time.perf_counter() - t0) / repeat, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loci", type=int, default=100)
    ap.add_argument("--reads", type=int, default=976)
    ap.add_argument("--context", type=int, default=10)
    args = ap.parse_args()

    block = 2 * args.context + 1
    rng = np.random.default_rng(4242)

    print(f"loci={args.loci} reads={args.reads} block={block}")
    print(f"{'sub_rate':>9} {'cover':>7} {'nnz':>10} {'pairs':>10} "
          f"{'dense':>10} {'sparse':>10} {'ratio':>7}  exact")

    for cover_frac in (0.03, 0.3, 1.0):
        for sub_rate in (0.07, 0.25, 0.75):
            B, M_base, ref = make_batch(args.loci, args.reads, block,
                                        sub_rate, cover_frac, rng)
            for reverse in (False,):
                t_dense, dense = timed(
                    lambda: pairwise_joint_sr_matrix(B, M_base, ref,
                                                     reverse=reverse))
                t_sparse, sparse = timed(
                    lambda: joint_sr_sparse(B, M_base, ref, reverse=reverse))

                sel = (B != ref[:, None, :]) & M_base & (B >= 1)
                nnz = int(sel.sum())
                c = sel.sum(-1)
                pairs = int((c.astype(np.int64) ** 2).sum())
                exact = np.array_equal(dense, sparse)

                print(f"{sub_rate:>9.2f} {cover_frac:>7.2f} {nnz:>10} "
                      f"{pairs:>10} {t_dense * 1000:>9.1f}m "
                      f"{t_sparse * 1000:>9.1f}m "
                      f"{t_dense / t_sparse:>6.2f}x  {exact}")


if __name__ == "__main__":
    main()
