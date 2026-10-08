#!/usr/bin/env python3
"""
Fine-grained profile of what is left in a prepdata batch.

profile_stages.py showed the split between pileup / stats / encode. This goes
one level further into the two areas that stage left ambiguous:

  * inside joint-SR: building the one-hot substitution tensor vs the matmul
    over it, and how sparse that tensor actually is.
  * the tail after features are computed: assembling the feature matrix,
    rounding, CSR conversion and the npz write, including how much the
    F-contiguous layout produced by `vstack(...).T` costs the CSR step.
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
    pileup_matrix_bam, loci_keep_mask, close_cached_bams,
)
from nfl_py.boosting import (  # noqa: E402
    pileup_matrix_to_boosting_data, _encode_ref_window,
)
from nfl_py.stats import _f32, _COMPLEMENT_PERM  # noqa: E402
from scipy.sparse import csr_matrix  # noqa: E402


def timed(fn, repeat=5):
    fn()
    t0 = time.perf_counter()
    for _ in range(repeat):
        out = fn()
    return (time.perf_counter() - t0) / repeat, out


def run(n_reads, n_loci, reflen, context=10, seed=11):
    rng = random.Random(seed)
    refname = "synth_chr"
    refseq = make_reference(reflen, rng)
    tmpdir = tempfile.mkdtemp(prefix="mars_tail_")
    bam = os.path.join(tmpdir, "synth.bam")
    build_bam(bam, refname, reflen, n_reads, rng)

    c = context
    bs = 2 * c + 1
    loci = sorted(rng.sample(range(c + 1, reflen - c), n_loci))
    loci_arr = np.asarray(loci, dtype=np.int64)
    tstart, tend = loci[0] - c, loci[-1] + c

    keep = loci_keep_mask(tstart, tend, loci_arr, c)
    pm = pileup_matrix_bam(bam, refname, tstart, tend, reverse=False,
                           keep_mask=keep)

    # Rebuild the per-locus window batches exactly as _features_from_matrices
    # does, so the numbers below are attributable.
    p_starts = loci_arr - c - 1 - pm.blocus
    offsets = np.arange(bs, dtype=np.int64)
    win_idx = p_starts[:, None] + offsets[None, :]
    col_idx = pm.col_of_pos[win_idx]
    ref_idx_full = _encode_ref_window(refseq, pm.blocus, pm.col_of_pos.size)

    B = np.ascontiguousarray(np.transpose(pm.B[:, col_idx], (1, 0, 2)))
    M_base = np.ascontiguousarray(np.transpose(pm.M_base[:, col_idx], (1, 0, 2)))
    M_qv = np.ascontiguousarray(np.transpose(pm.M_qv[:, col_idx], (1, 0, 2)))
    ref_idx = ref_idx_full[win_idx]
    nr = B.shape[1]

    # ---- inside joint-SR ----
    is_sub = (B != ref_idx[..., None, :]) & M_base

    def build_onehot():
        O = np.zeros((n_loci, nr, 5, bs), dtype=np.float32)
        for b in range(5):
            O[..., b, :] = _f32((B == b + 1) & is_sub)
        return O

    t_onehot, O = timed(build_onehot)
    A = O.reshape(n_loci, nr, 5 * bs)
    t_matmul, _ = timed(lambda: A.swapaxes(-1, -2) @ A)
    t_nij, _ = timed(lambda: _f32(M_base).swapaxes(-1, -2) @ _f32(M_base))

    density = float(A.astype(bool).mean())
    masks_equal = bool(np.array_equal(M_base, M_qv))

    # ---- the tail ----
    X = pileup_matrix_to_boosting_data(pm, refseq, loci, context_size=c,
                                       run_qv_mean=12.5, run_qv_sd=7.25,
                                       use_gpu=False)
    t_round, Xr = timed(lambda: np.round(X, decimals=4))
    Xc = np.ascontiguousarray(Xr)
    t_csr_f, _ = timed(lambda: csr_matrix(Xr))       # as produced today
    t_csr_c, _ = timed(lambda: csr_matrix(Xc))       # C-contiguous
    sp = csr_matrix(Xc)
    outfile = os.path.join(tmpdir, "X_probe.npz")

    def write():
        np.savez(outfile,
                 data=sp.data.astype(np.float32), indices=sp.indices,
                 indptr=sp.indptr, shape=sp.shape, loci=loci,
                 chromosome=refname, strand='+', depth=np.zeros(n_loci))
    t_save, _ = timed(write, repeat=3)

    close_cached_bams()

    print(f"\n=== reads={n_reads} loci={n_loci} reflen={reflen} "
          f"(materialized reads={nr}) ===")
    print(f"  joint-SR one-hot build   {t_onehot * 1000:8.1f} ms  "
          f"({O.nbytes / 1e6:.1f} MB, density={density:.3%})")
    print(f"  joint-SR matmul          {t_matmul * 1000:8.1f} ms")
    print(f"  n_ij (M_base.T @ M_base) {t_nij * 1000:8.1f} ms   "
          f"M_base == M_qv: {masks_equal}")
    print(f"  --")
    print(f"  X layout                 {'F' if Xr.flags.f_contiguous else 'C'}"
          f"-contiguous, {X.shape}, {X.nbytes / 1e6:.2f} MB")
    print(f"  round                    {t_round * 1000:8.1f} ms")
    print(f"  csr from F-contiguous    {t_csr_f * 1000:8.1f} ms  <-- today")
    print(f"  csr from C-contiguous    {t_csr_c * 1000:8.1f} ms")
    print(f"  savez                    {t_save * 1000:8.1f} ms  "
          f"(nnz={sp.nnz}, {sp.nnz / (X.shape[0] * X.shape[1]):.1%} dense)")


if __name__ == "__main__":
    for n_reads, n_loci, reflen in ((800, 100, 20000), (2000, 100, 20000)):
        run(n_reads, n_loci, reflen)
