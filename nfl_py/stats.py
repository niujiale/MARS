#!/usr/bin/env python3
"""
Statistical Functions

Implementation of joint moment statistics:
- Covariance
- Coskewness (pairwise)
- Cokurtosis (pairwise)
- Joint substitution rate

Author: MARS Team
"""

import logging
from typing import Dict, List, Optional, Tuple
import numpy as np

from .gpu_utils import GPUAccelerator, is_gpu_available

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend dispatch shims
# ---------------------------------------------------------------------------
# The vectorized pairwise statistics below work on either NumPy arrays or
# PyTorch tensors. We detect the array type at call time and route a small
# set of operations through backend-specific shims; the bulk of the code
# (matmul, broadcasting arithmetic, boolean masks) uses identical API on
# both backends. This keeps a single source of truth for the math.

try:
    import torch as _torch  # noqa: F401
    _TORCH_IMPORTED = True
except ImportError:
    _TORCH_IMPORTED = False


def _is_torch(a) -> bool:
    return type(a).__module__.startswith('torch')


def _f32(a):
    """Cast to float32 on whatever backend a lives on."""
    if _is_torch(a):
        import torch
        return a.to(torch.float32)
    return a.astype(np.float32)


def _diag_last2(a):
    """Diagonal along the last two axes, backend-agnostic."""
    if _is_torch(a):
        import torch
        return torch.diagonal(a, dim1=-2, dim2=-1)
    return np.diagonal(a, axis1=-2, axis2=-1)


def _zeros_like_backend(shape, reference):
    """Zeros of `shape` on the same backend/device as `reference`, float32."""
    if _is_torch(reference):
        import torch
        return torch.zeros(shape, dtype=torch.float32, device=reference.device)
    return np.zeros(shape, dtype=np.float32)


def _permute(a, perm):
    if _is_torch(a):
        return a.permute(*perm)
    return np.transpose(a, perm)


def _where(cond, x, y):
    if _is_torch(cond):
        import torch
        return torch.where(cond, x, y)
    return np.where(cond, x, y)


def _sqrt(a):
    if _is_torch(a):
        import torch
        return torch.sqrt(a)
    return np.sqrt(a)


def _maximum(a, b):
    if _is_torch(a):
        import torch
        if not _is_torch(b):
            b = torch.as_tensor(b, dtype=a.dtype, device=a.device)
        return torch.maximum(a, b)
    return np.maximum(a, b)


def _to_numpy(a):
    """Bring a (possibly-GPU) tensor back to a numpy array; no-op on numpy."""
    if _is_torch(a):
        return a.detach().cpu().numpy()
    return a


def covariance(
    x: Dict[str, int],
    y: Dict[str, int],
    mu_x: Optional[float] = None,
    mu_y: Optional[float] = None
) -> float:
    """
    Calculate covariance between two dictionaries of values.

    Computes covariance only for keys present in both dictionaries.

    Args:
        x: Dictionary mapping read names to values
        y: Dictionary mapping read names to values
        mu_x: Pre-computed mean of x (optional)
        mu_y: Pre-computed mean of y (optional)

    Returns:
        Covariance value (0 if fewer than 3 common keys)

    Example:
        >>> x = {"read1": 30, "read2": 35, "read3": 32}
        >>> y = {"read1": 28, "read2": 33, "read3": 30}
        >>> covariance(x, y)
        4.333...
    """
    # Find common keys
    common_keys = set(x.keys()) & set(y.keys())
    n = len(common_keys)

    if n < 3:
        return 0.0

    x_vals = np.array([x[k] for k in common_keys], dtype=np.float32)
    y_vals = np.array([y[k] for k in common_keys], dtype=np.float32)

    if mu_x is None:
        mu_x = np.mean(x_vals)
    if mu_y is None:
        mu_y = np.mean(y_vals)

    cov = np.sum((x_vals - mu_x) * (y_vals - mu_y)) / (n - 1)
    return float(cov)


def coskewness(
    x: Dict[str, int],
    y: Dict[str, int],
    mu_x: Optional[float] = None,
    mu_y: Optional[float] = None,
    sigma_x: Optional[float] = None,
    sigma_y: Optional[float] = None
) -> float:
    """
    Calculate pairwise coskewness based on standardized third centralized moments.

    Formula: E[(x - mu_x)^2 * (y - mu_y)] / (sigma_x^2 * sigma_y)

    Args:
        x: Dictionary mapping read names to values
        y: Dictionary mapping read names to values
        mu_x: Pre-computed mean of x (optional)
        mu_y: Pre-computed mean of y (optional)
        sigma_x: Pre-computed std of x (optional)
        sigma_y: Pre-computed std of y (optional)

    Returns:
        Coskewness value (0 if fewer than 3 common keys or zero variance)
    """
    # Find common keys
    common_keys = set(x.keys()) & set(y.keys())
    n = len(common_keys)

    if n < 3:
        return 0.0

    x_vals = np.array([x[k] for k in common_keys], dtype=np.float32)
    y_vals = np.array([y[k] for k in common_keys], dtype=np.float32)

    if mu_x is None or mu_y is None or sigma_x is None or sigma_y is None:
        mu_x = np.mean(x_vals)
        mu_y = np.mean(y_vals)
        sigma_x = np.std(x_vals, ddof=1)
        sigma_y = np.std(y_vals, ddof=1)

    if sigma_x == 0 or sigma_y == 0:
        return 0.0

    sk = np.sum((x_vals - mu_x)**2 * (y_vals - mu_y)) / n
    result = sk / (sigma_x**2 * sigma_y)

    if np.isnan(result):
        return 0.0

    return float(result)


def cokurtosis(
    x: Dict[str, int],
    y: Dict[str, int],
    mu_x: Optional[float] = None,
    mu_y: Optional[float] = None,
    sigma_x: Optional[float] = None,
    sigma_y: Optional[float] = None
) -> float:
    """
    Calculate pairwise symmetric cokurtosis based on standardized fourth centralized moments.

    Formula: E[(x - mu_x)^2 * (y - mu_y)^2] / (sigma_x^2 * sigma_y^2)

    Args:
        x: Dictionary mapping read names to values
        y: Dictionary mapping read names to values
        mu_x: Pre-computed mean of x (optional)
        mu_y: Pre-computed mean of y (optional)
        sigma_x: Pre-computed std of x (optional)
        sigma_y: Pre-computed std of y (optional)

    Returns:
        Cokurtosis value (0 if fewer than 3 common keys or zero variance)
    """
    # Find common keys
    common_keys = set(x.keys()) & set(y.keys())
    n = len(common_keys)

    if n < 3:
        return 0.0

    x_vals = np.array([x[k] for k in common_keys], dtype=np.float32)
    y_vals = np.array([y[k] for k in common_keys], dtype=np.float32)

    if mu_x is None or mu_y is None or sigma_x is None or sigma_y is None:
        mu_x = np.mean(x_vals)
        mu_y = np.mean(y_vals)
        sigma_x = np.std(x_vals, ddof=1)
        sigma_y = np.std(y_vals, ddof=1)

    if sigma_x == 0 or sigma_y == 0:
        return 0.0

    kt = np.sum((x_vals - mu_x)**2 * (y_vals - mu_y)**2) / n
    result = kt / (sigma_x**2 * sigma_y**2)

    if np.isnan(result):
        return 0.0

    return float(result)


def base2shift(base: str) -> int:
    """
    Convert DNA base to integer index.

    Args:
        base: DNA base character

    Returns:
        Integer: A=1, C=2, G=3, T=4, Gap=5, others=0
    """
    base = base.upper()
    mapping = {'A': 1, 'C': 2, 'G': 3, 'T': 4, '-': 5}
    return mapping.get(base, 0)


def shift2base(num: int) -> str:
    """
    Convert integer index to DNA base.

    Args:
        num: Integer index

    Returns:
        DNA base character
    """
    mapping = {1: 'A', 2: 'C', 3: 'G', 4: 'T', 5: '-'}
    return mapping.get(num, 'N')


def complement(base: str) -> str:
    """
    Get complement of a DNA base.

    Args:
        base: DNA base character

    Returns:
        Complement base
    """
    comp = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N', '-': '-'}
    return comp.get(base.upper(), 'N')


def joint_sr(
    x: Dict[str, str],
    y: Dict[str, str],
    refbase_i: str,
    refbase_j: str,
    reverse: bool = False
) -> np.ndarray:
    """
    Calculate joint substitution rate matrix.

    Args:
        x: Dictionary mapping read names to bases at position i
        y: Dictionary mapping read names to bases at position j
        refbase_i: Reference base at position i
        refbase_j: Reference base at position j
        reverse: If True, use complement bases for reverse strand

    Returns:
        Flattened 5x5 substitution rate matrix (A, C, G, T, Gap)
    """
    var_joint = np.zeros((5, 5), dtype=np.float32)
    n = 0

    common_keys = set(x.keys()) & set(y.keys())

    for key in common_keys:
        base_x = x[key].upper()
        base_y = y[key].upper()

        # Only count if both differ from reference
        if base_x != refbase_i.upper() and base_y != refbase_j.upper():
            if reverse:
                base_x = complement(base_x)
                base_y = complement(base_y)

            idx_x = base2shift(base_x) - 1  # Convert to 0-indexed
            idx_y = base2shift(base_y) - 1

            if idx_x >= 0 and idx_y >= 0:
                var_joint[idx_x, idx_y] += 1

        n += 1

    if n > 0:
        var_joint /= n

    return var_joint.flatten()


def joint_sr_arrays(
    x_bases: np.ndarray,
    y_bases: np.ndarray,
    refbase_i: str,
    refbase_j: str,
    reverse: bool = False,
    use_gpu: bool = True
) -> np.ndarray:
    """
    Calculate joint substitution rate from arrays (GPU-accelerated).

    Args:
        x_bases: Array of base indices at position i (1-indexed)
        y_bases: Array of base indices at position j (1-indexed)
        refbase_i: Reference base at position i
        refbase_j: Reference base at position j
        reverse: If True, use complement bases
        use_gpu: Whether to use GPU acceleration

    Returns:
        Flattened 5x5 substitution rate matrix
    """
    ref_i_idx = base2shift(refbase_i)
    ref_j_idx = base2shift(refbase_j)

    if use_gpu and is_gpu_available():
        with GPUAccelerator(use_gpu) as xp:
            x = xp.asarray(x_bases)
            y = xp.asarray(y_bases)

            # Filter for substitutions
            mask = (x != ref_i_idx) & (y != ref_j_idx)
            x_sub = x[mask]
            y_sub = y[mask]

            # Build joint matrix
            var_joint = xp.zeros((5, 5), dtype=xp.float32)
            for xi, yi in zip(x_sub.get() if hasattr(x_sub, 'get') else x_sub,
                              y_sub.get() if hasattr(y_sub, 'get') else y_sub):
                if xi > 0 and yi > 0:
                    var_joint[xi - 1, yi - 1] += 1

            n = len(x)
            if n > 0:
                var_joint /= n

            return var_joint.flatten().get() if hasattr(var_joint, 'get') else var_joint.flatten()
    else:
        var_joint = np.zeros((5, 5), dtype=np.float32)

        mask = (x_bases != ref_i_idx) & (y_bases != ref_j_idx)
        x_sub = x_bases[mask]
        y_sub = y_bases[mask]

        for xi, yi in zip(x_sub, y_sub):
            if xi > 0 and yi > 0:
                var_joint[xi - 1, yi - 1] += 1

        n = len(x_bases)
        if n > 0:
            var_joint /= n

        return var_joint.flatten()


def pairwise_qv_stats_matrix(
    Q: np.ndarray,
    M: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized pairwise QV moments from a (..., n_reads, block_size) tensor.

    Accepts any leading batch dims — pass a 2D (n_reads, block_size) input to
    process a single locus, or a 3D (n_loci, n_reads, block_size) input to
    fuse a whole batch of loci into a single BLAS matmul per statistic.

    Numerically equivalent (up to float32 round-off) to the per-pair
    covariance/coskewness/cokurtosis in this module, but does the 21x21 work
    in a few BLAS calls instead of nested Python loops over dicts.

    Conventions preserved from the scalar versions:
      * mean_i uses reads present at position i (position-wise mean).
      * cov[i,j] uses reads common to positions i and j, with n_ij - 1 in
        the denominator. Returns 0 when n_ij < 3.
      * sigma_i = sqrt(cov[i,i]) — position-wise std derived from the
        common (self-self) path.
      * coskewness is asymmetric: sum((x-mu_x)^2 * (y-mu_y)) / n_ij
        then divided by sigma_i^2 * sigma_j.
      * cokurtosis is symmetric: sum((x-mu_x)^2 * (y-mu_y)^2) / n_ij
        then divided by sigma_i^2 * sigma_j^2.
      * Returns 0 when n_ij < 3 or sigma_i == 0 or sigma_j == 0.

    Args:
        Q: (n_reads, block_size) float array; value at (r, i) is the QV
           of read r at position i, or NaN if the read has no QV here.
        M: (n_reads, block_size) bool mask; True where Q has a real value.

    Returns:
        qv_mean: (block_size,) float32
        qv_cov:  (block_size, block_size) float32
        qv_sk:   (block_size, block_size) float32
        qv_kt:   (block_size, block_size) float32
    """
    M_f = _f32(M)
    n_i = M_f.sum(-2)  # (..., block_size)

    Q_zero = _f32(_where(M, Q, 0.0))
    safe_n_i = _where(n_i > 0, n_i, 1.0)
    mean_i = _f32(_where(n_i > 0, Q_zero.sum(-2) / safe_n_i, 0.0))

    Q_c = _f32(_where(M, Q - mean_i[..., None, :], 0.0))

    M_f_T = M_f.swapaxes(-1, -2)
    Q_c_T = Q_c.swapaxes(-1, -2)
    n_ij = M_f_T @ M_f  # (..., bs, bs)

    S1 = Q_c_T @ Q_c
    valid_cov = n_ij >= 3
    denom_cov = _where(n_ij > 1, n_ij - 1, 1.0)
    cov = _f32(_where(valid_cov, S1 / denom_cov, 0.0))

    sigma = _f32(_sqrt(_maximum(_diag_last2(cov), 0.0)))

    safe_n_ij = _where(n_ij > 0, n_ij, 1.0)
    Q_c_sq = Q_c * Q_c
    Q_c_sq_T = Q_c_sq.swapaxes(-1, -2)

    sig2 = sigma * sigma
    denom_sk = sig2[..., :, None] * sigma[..., None, :]
    denom_kt = sig2[..., :, None] * sig2[..., None, :]
    sigma_valid = (sigma[..., :, None] > 0) & (sigma[..., None, :] > 0)
    valid = valid_cov & sigma_valid

    S2 = Q_c_sq_T @ Q_c
    sk = _f32(_where(
        valid,
        (S2 / safe_n_ij) / _where(denom_sk > 0, denom_sk, 1.0),
        0.0,
    ))

    S3 = Q_c_sq_T @ Q_c_sq
    kt = _f32(_where(
        valid,
        (S3 / safe_n_ij) / _where(denom_kt > 0, denom_kt, 1.0),
        0.0,
    ))

    return mean_i, cov, sk, kt


# Complement permutation over 0-indexed base space [A, C, G, T, -]:
# A(0)->T(3), C(1)->G(2), G(2)->C(1), T(3)->A(0), -(4)->-(4)
_COMPLEMENT_PERM = np.array([3, 2, 1, 0, 4], dtype=np.int64)


# How much more expensive one counted pair is than one BLAS multiply-add, with
# margin. Below this ratio the counting path wins; above it the dense Gram
# matrix does. Calibrated on the measurements in _bench/proto_sparse_joint.py.
_SPARSE_JOINT_PAIR_COST = 1000


def _joint_sr_counts_sparse(sel, B, block, reverse):
    """
    Gram matrix of the one-hot substitution tensor by counting co-occurrences.

    Returns (n_loci, 5, block, 5, block) counts as float32, or None if this
    input is not one the counting path should handle.

    Every entry of the dense ``A.T @ A`` is a sum of products of 0.0 and 1.0,
    so it is an integer no larger than the read count. float32 represents such
    integers exactly, which is why counting them in a different order gives a
    bit-identical answer rather than an approximate one.
    """
    n_loci, n_reads, _ = B.shape
    flat = 5 * block

    # Number of contributing positions per (locus, read). A read covers one
    # window, so for a batch spanning many loci most of these are zero.
    per_read = sel.sum(-1, dtype=np.int64)
    n_pairs = int((per_read * per_read).sum())

    dense_cost = n_loci * flat * flat * n_reads
    if n_pairs * _SPARSE_JOINT_PAIR_COST >= dense_cost:
        return None

    l_idx, r_idx, s_idx = np.nonzero(sel)
    b_idx = B[l_idx, r_idx, s_idx].astype(np.int64) - 1
    if reverse:
        b_idx = _COMPLEMENT_PERM[b_idx]
    slot = b_idx * block + s_idx

    # np.nonzero walks in C order, so entries are already grouped by
    # (locus, read) and each group is a contiguous run.
    group_sizes = per_read.ravel()
    group_sizes = group_sizes[group_sizes > 0]
    group_starts = np.cumsum(group_sizes) - group_sizes

    size_of = np.repeat(group_sizes, group_sizes)
    start_of = np.repeat(group_starts, group_sizes)

    # For every entry, pair it with all entries of its own group.
    left = np.repeat(np.arange(l_idx.size, dtype=np.int64), size_of)
    within = (np.arange(n_pairs, dtype=np.int64)
              - np.repeat(np.cumsum(size_of) - size_of, size_of))
    right = np.repeat(start_of, size_of) + within

    dest = l_idx[left] * (flat * flat) + slot[left] * flat + slot[right]
    counts = np.bincount(dest, minlength=n_loci * flat * flat)
    return counts.astype(np.float32).reshape(n_loci, 5, block, 5, block)


def pairwise_joint_sr_matrix(
    B: np.ndarray,
    M_base: np.ndarray,
    ref_idx: np.ndarray,
    reverse: bool = False,
) -> np.ndarray:
    """
    Vectorized joint substitution-rate tensor from base matrices.

    Accepts any leading batch dims on B / M_base / ref_idx — pass 2D+1D
    for a single locus, or 3D+2D (n_loci, n_reads, block_size) +
    (n_loci, block_size) to process a whole batch in one matmul.

    Numerically equivalent to calling `joint_sr` over every (i, j) pair.

    For a NumPy batch this may count co-occurring substitutions instead of
    forming the Gram matrix of the one-hot tensor, whichever is cheaper for the
    data at hand. Both give the same bits: every entry is a count of reads, and
    float32 holds such integers exactly.

    Conventions preserved:
      * A read contributes to var_joint[x, y, i, j] only if its base at i
        AND at j both differ from the forward-strand reference (the
        substitution test uses raw bases, not complemented, even on the
        reverse strand — matches the scalar version exactly).
      * On the reverse strand, the stored base index is complemented
        (A<->T, C<->G); deletion stays as '-'.
      * Normalization is by the count of reads present at both i and j
        (whether or not they are substitutions), per the scalar version.

    Args:
        B: (n_reads, block_size) int; 0 = no base at (r, i), 1..5 = A/C/G/T/-.
        M_base: (n_reads, block_size) bool; True where B has a valid base.
        ref_idx: (block_size,) int; 1..5 forward-strand ref base index, or
                 0 for invalid. Always the forward ref, even for reverse.
        reverse: True on the reverse strand (applies complement to the
                 base axis of the returned tensor).

    Returns:
        var_joint: (5, 5, block_size, block_size) float32, normalized by
                   the number of reads common to positions i and j.
    """
    block_size = B.shape[-1]
    n_reads = B.shape[-2]
    batch_shape = tuple(B.shape[:-2])  # leading batch dims (possibly empty)

    is_sub = (B != ref_idx[..., None, :]) & M_base  # (..., n_reads, block_size)

    var_joint = None
    if not _is_torch(B) and B.ndim == 3:
        # The dense Gram matrix costs the same whether the substitution tensor
        # is full or nearly empty. Real data substitutes a few percent of bases
        # and a read covers only one window, so counting the co-occurrences
        # that exist is usually far less work. Falls through when it is not.
        sel = is_sub & (B >= 1) & (B <= 5)
        counts = _joint_sr_counts_sparse(sel, B, block_size, reverse)
        if counts is not None:
            var_joint = counts.transpose(0, 1, 3, 2, 4)

    if var_joint is None:
        O_sub = _zeros_like_backend(
            batch_shape + (n_reads, 5, block_size), reference=M_base
        )
        if reverse:
            for b in range(5):
                O_sub[..., _COMPLEMENT_PERM[b], :] = _f32((B == b + 1) & is_sub)
        else:
            for b in range(5):
                O_sub[..., b, :] = _f32((B == b + 1) & is_sub)

        # Flatten (5, block_size) into a single axis so A.T @ A runs through
        # batched BLAS matmul (much faster than einsum on >=4D).
        flat = 5 * block_size
        A = O_sub.reshape(batch_shape + (n_reads, flat))
        var_flat = A.swapaxes(-1, -2) @ A  # (..., 5*bs, 5*bs)
        var_joint = var_flat.reshape(
            batch_shape + (5, block_size, 5, block_size)
        )
        # (..., x, i, y, j) -> (..., x, y, i, j)
        perm = tuple(range(len(batch_shape))) + tuple(
            len(batch_shape) + a for a in (0, 2, 1, 3)
        )
        var_joint = _permute(var_joint, perm)

    M_base_f = _f32(M_base)
    n_ij_base = M_base_f.swapaxes(-1, -2) @ M_base_f
    denom = _where(n_ij_base > 0, n_ij_base, 1.0)
    var_joint = var_joint / denom[..., None, None, :, :]

    return _f32(var_joint)


def compute_pairwise_qv_stats(
    qv_dicts: List[Dict[str, int]],
    use_gpu: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute pairwise QV statistics (mean, covariance, coskewness, cokurtosis).

    Args:
        qv_dicts: List of dictionaries mapping read names to QV values
                  (one dict per position in context)
        use_gpu: Whether to use GPU acceleration

    Returns:
        Tuple of (qv_mean, qv_cov, qv_sk, qv_kt) arrays
    """
    n_pos = len(qv_dicts)

    qv_mean = np.zeros(n_pos, dtype=np.float32)
    qv_cov = np.zeros((n_pos, n_pos), dtype=np.float32)
    qv_sk = np.zeros((n_pos, n_pos), dtype=np.float32)
    qv_kt = np.zeros((n_pos, n_pos), dtype=np.float32)

    # Compute means and standard deviations
    means = []
    stds = []
    for i in range(n_pos):
        if qv_dicts[i]:
            vals = np.array(list(qv_dicts[i].values()), dtype=np.float32)
            qv_mean[i] = np.nanmean(vals) if len(vals) > 0 else 0.0
            means.append(qv_mean[i])
            stds.append(np.nanstd(vals, ddof=1) if len(vals) > 1 else 0.0)
        else:
            means.append(0.0)
            stds.append(0.0)

    # Compute pairwise statistics
    for i in range(n_pos):
        for j in range(n_pos):
            if qv_dicts[i] and qv_dicts[j]:
                qv_cov[i, j] = covariance(
                    qv_dicts[i], qv_dicts[j],
                    means[i], means[j]
                )
                qv_sk[i, j] = coskewness(
                    qv_dicts[i], qv_dicts[j],
                    means[i], means[j],
                    stds[i], stds[j]
                )
                qv_kt[i, j] = cokurtosis(
                    qv_dicts[i], qv_dicts[j],
                    means[i], means[j],
                    stds[i], stds[j]
                )

    return qv_mean, qv_cov, qv_sk, qv_kt
