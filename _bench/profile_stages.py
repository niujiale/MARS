#!/usr/bin/env python3
"""
Stage-level timing for the optimized prepdata path, across data sizes.

Splits the per-batch cost into pileup / QV stats / joint-SR stats / context /
sparse-encode so it is clear what the remaining bottleneck is, and compares
against the original dictionary pileup at each size.
"""

import os
import random
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from equiv_check import build_bam, make_reference  # noqa: E402
from nfl_py.pileup import (  # noqa: E402
    pileup_bam, pileup_matrix_bam, extend_loci, loci_keep_mask,
    close_cached_bams,
)
from nfl_py.boosting import (  # noqa: E402
    pileup_to_boosting_data, pileup_matrix_to_boosting_data,
    _build_pileup_matrices, _encode_ref_window,
)
from nfl_py.stats import (  # noqa: E402
    pairwise_qv_stats_matrix, pairwise_joint_sr_matrix,
)
from scipy.sparse import csr_matrix  # noqa: E402


def bench(n_reads, n_loci, reflen, context=10, repeat=3, seed=7, sub_rate=0.07):
    rng = random.Random(seed)
    refname = "synth_chr"
    refseq = make_reference(reflen, rng)
    tmpdir = tempfile.mkdtemp(prefix="mars_prof_")
    bam = os.path.join(tmpdir, "synth.bam")
    build_bam(bam, refname, reflen, n_reads, rng,
              refseq=refseq, sub_rate=sub_rate)

    c = context
    loci = sorted(rng.sample(range(c + 1, reflen - c), n_loci))
    loci_arr = np.asarray(loci, dtype=np.int64)
    tstart, tend = loci[0] - c, loci[-1] + c
    loci_ext = extend_loci(loci, c)

    def timed(fn):
        fn()  # warm up
        t0 = time.perf_counter()
        for _ in range(repeat):
            out = fn()
        return (time.perf_counter() - t0) / repeat, out

    # ---- old path ----
    t_old_pileup, pu = timed(lambda: pileup_bam(
        bam, refname, tstart, tend, isqv=True, isins=False, loci_ext=loci_ext))
    t_old_fold, _ = timed(lambda: _build_pileup_matrices(pu.bup_qv, pu.bup))
    t_old_total, X_old = timed(lambda: pileup_to_boosting_data(
        pu, refseq, loci, context_size=c, reverse=False,
        run_qv_mean=12.5, run_qv_sd=7.25, use_gpu=False))
    t_old_all = t_old_pileup + t_old_total

    # ---- new path ----
    t_new_mask, keep = timed(lambda: loci_keep_mask(tstart, tend, loci_arr, c))
    t_new_pileup, pm = timed(lambda: pileup_matrix_bam(
        bam, refname, tstart, tend, reverse=False, keep_mask=keep))
    t_new_feat, X_new = timed(lambda: pileup_matrix_to_boosting_data(
        pm, refseq, loci, context_size=c,
        run_qv_mean=12.5, run_qv_sd=7.25, use_gpu=False))
    t_new_all = t_new_mask + t_new_pileup + t_new_feat

    # ---- inside the feature step ----
    block = 2 * c + 1
    p_starts = loci_arr - c - 1 - pm.blocus
    offsets = np.arange(block, dtype=np.int64)
    win_idx = p_starts[:, None] + offsets[None, :]
    col_idx = pm.col_of_pos[win_idx]
    ref_idx_full = _encode_ref_window(refseq, pm.blocus, pm.col_of_pos.size)

    t_gather, batches = timed(lambda: (
        np.ascontiguousarray(np.transpose(pm.Q[:, col_idx], (1, 0, 2))),
        np.ascontiguousarray(np.transpose(pm.M_qv[:, col_idx], (1, 0, 2))),
        np.ascontiguousarray(np.transpose(pm.B[:, col_idx], (1, 0, 2))),
        np.ascontiguousarray(np.transpose(pm.M_base[:, col_idx], (1, 0, 2))),
        ref_idx_full[win_idx],
    ))
    Qb, Mqb, Bb, Mbb, refb = batches
    t_qv, _ = timed(lambda: pairwise_qv_stats_matrix(Qb, Mqb))
    t_joint, _ = timed(lambda: pairwise_joint_sr_matrix(Bb, Mbb, refb, reverse=False))
    t_sparse, _ = timed(lambda: csr_matrix(np.round(X_new, 4)))

    identical = np.array_equal(X_old, X_new, equal_nan=True)
    close_cached_bams()

    print(f"\n=== reads={n_reads} loci={n_loci} reflen={reflen} "
          f"sub_rate={sub_rate} (bit-identical={identical}) ===")
    print(f"  OLD  pileup(dict)  {t_old_pileup * 1000:8.1f} ms")
    print(f"       ...fold only  {t_old_fold * 1000:8.1f} ms  (part of features)")
    print(f"       features      {t_old_total * 1000:8.1f} ms")
    print(f"       TOTAL         {t_old_all * 1000:8.1f} ms")
    print(f"  NEW  keep-mask     {t_new_mask * 1000:8.1f} ms")
    print(f"       pileup(matrix){t_new_pileup * 1000:8.1f} ms")
    print(f"       features      {t_new_feat * 1000:8.1f} ms")
    print(f"       TOTAL         {t_new_all * 1000:8.1f} ms   "
          f"speedup={t_old_all / t_new_all:.2f}x")
    print(f"  breakdown of NEW features:")
    print(f"       window gather {t_gather * 1000:8.1f} ms")
    print(f"       QV stats      {t_qv * 1000:8.1f} ms")
    print(f"       joint-SR stats{t_joint * 1000:8.1f} ms  <-- GPU target")
    print(f"       round+csr     {t_sparse * 1000:8.1f} ms")
    print(f"  matrix cols: old={pu.dp.size} new={pm.col_of_pos.size} "
          f"materialized={pm.Q.shape[1] - 1} reads={pm.Q.shape[0]}")
    return t_old_all, t_new_all


if __name__ == "__main__":
    import sys
    # Default to a realistic mismatch rate; pass "random" to drive the dense
    # paths with reads that disagree with the reference three times in four.
    sub_rate = None if "random" in sys.argv[1:] else 0.07
    totals = []
    for n_reads, n_loci, reflen in (
        (200, 100, 20000),
        (800, 100, 20000),
        (2000, 100, 20000),
        (800, 100, 200000),
    ):
        totals.append(bench(n_reads, n_loci, reflen, sub_rate=sub_rate))
    o = sum(t[0] for t in totals)
    n = sum(t[1] for t in totals)
    print(f"\nAGGREGATE old={o * 1000:.0f} ms new={n * 1000:.0f} ms "
          f"speedup={o / n:.2f}x")
