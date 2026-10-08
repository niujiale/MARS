#!/usr/bin/env python3
"""
Boosting Module

XGBoost model training, prediction, and feature extraction.
Core functions for converting pileup data to feature matrices.

Author: MARS Team
"""

import os
import logging
import warnings
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, save_npz, load_npz
import xgboost as xgb
from tqdm.auto import tqdm

# Suppress semaphore leak warnings (Python 3.12+ issue)
warnings.filterwarnings("ignore", message=".*leaked semaphore.*", category=UserWarning)

from .pileup import (
    pileup_bam, PileupData, get_mean_qv, extend_loci,
    pileup_matrix_bam, PileupMatrix, loci_keep_mask, BASE_CODE_LUT,
)
from .stats import (
    covariance, coskewness, cokurtosis, joint_sr,
    base2shift, compute_pairwise_qv_stats,
    pairwise_qv_stats_matrix, pairwise_joint_sr_matrix,
)
from .iofastx import load_fasta_dna, context_to_encoded_vec, reverse_complement
from .gpu_utils import is_gpu_available, GPUAccelerator

# Optional torch import for the GPU path. If torch is missing or CUDA is
# unavailable, we silently fall back to NumPy — the only behavioural change
# is that `use_gpu=True` no longer triggers the GPU branch.
try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None
    _TORCH_AVAILABLE = False


def _torch_cuda_ready() -> bool:
    return _TORCH_AVAILABLE and _torch.cuda.is_available()

# ASCII code point -> base index (A=1, C=2, G=3, T=4, '-'=5, else 0).
_BASE_LUT = BASE_CODE_LUT

# Byte-level lookup tables replacing the per-character Python work in
# iofastx.context_to_encoded_vec / reverse_complement. The context encoding is
# A=[0,0], C=[1,0], G=[0,1], T=[1,1], anything else [0,0], so the first bit is
# set for C/T and the second for G/T.
_CTX_BIT0 = np.zeros(256, dtype=np.float32)
_CTX_BIT1 = np.zeros(256, dtype=np.float32)
for _b in 'CTct':
    _CTX_BIT0[ord(_b)] = 1.0
for _b in 'GTgt':
    _CTX_BIT1[ord(_b)] = 1.0

_COMPLEMENT_BYTE = np.full(256, ord('N'), dtype=np.uint8)
for _a, _c in (('A', 'T'), ('T', 'A'), ('C', 'G'), ('G', 'C'), ('N', 'N')):
    _COMPLEMENT_BYTE[ord(_a)] = ord(_c)
    _COMPLEMENT_BYTE[ord(_a.lower())] = ord(_c)
del _b, _a, _c

# One reference chromosome is reused across every batch of a run, so its byte
# view is worth keeping. Identity comparison keeps this free of hashing cost
# and the single slot bounds memory to one chromosome.
_REFSEQ_BYTES_CACHE: Dict[str, Any] = {"seq": None, "bytes": None}

# Which compute backend we have already logged about, so the one-shot notice
# is not repeated for every batch.
_ANNOUNCED_MODES = set()


def _refseq_bytes(refseq: str) -> np.ndarray:
    """Byte view of a reference sequence, cached for the current chromosome."""
    if _REFSEQ_BYTES_CACHE["seq"] is refseq:
        return _REFSEQ_BYTES_CACHE["bytes"]
    arr = np.frombuffer(refseq.encode('latin-1', 'replace'), dtype=np.uint8)
    _REFSEQ_BYTES_CACHE["seq"] = refseq
    _REFSEQ_BYTES_CACHE["bytes"] = arr
    return arr


def _encode_ref_window(refseq: str, blocus: int, tlen: int) -> np.ndarray:
    """
    Base indices of the reference across a pileup window.

    Position ``p`` of the window is reference position ``blocus + 1 + p``
    (1-based); offsets that fall outside the sequence stay 0.
    """
    ref_idx = np.zeros(tlen, dtype=np.int64)
    lo = max(0, -blocus)
    hi = min(tlen, len(refseq) - blocus)
    if hi > lo:
        seg = _refseq_bytes(refseq)[blocus + lo:blocus + hi]
        ref_idx[lo:hi] = BASE_CODE_LUT[seg]
    return ref_idx


def _build_pileup_matrices(
    qv_list,
    base_list,
    dtype_qv: np.dtype = np.float32,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Fold a list of per-position {read: value} dicts into dense (n_reads x tlen)
    matrices with presence masks. The read axis is the union of read names
    across all positions, with a stable ordering.

    Returns (Q, M_qv, B, M_base, read_to_ridx).
    """
    tlen = len(qv_list)
    all_reads = set()
    for d in qv_list:
        all_reads.update(d.keys())
    for d in base_list:
        all_reads.update(d.keys())
    read_list = sorted(all_reads)
    read_to_ridx = {r: i for i, r in enumerate(read_list)}
    n_reads = len(read_list)

    Q = np.full((n_reads, tlen), np.nan, dtype=dtype_qv)
    M_qv = np.zeros((n_reads, tlen), dtype=bool)
    B = np.zeros((n_reads, tlen), dtype=np.int8)
    M_base = np.zeros((n_reads, tlen), dtype=bool)

    if n_reads == 0:
        return Q, M_qv, B, M_base, read_to_ridx

    for p in range(tlen):
        d_qv = qv_list[p]
        if d_qv:
            for rname, qv in d_qv.items():
                ri = read_to_ridx[rname]
                Q[ri, p] = qv
                M_qv[ri, p] = True
        d_b = base_list[p]
        if d_b:
            for rname, base_char in d_b.items():
                ri = read_to_ridx[rname]
                # Single-char string -> base code via LUT
                B[ri, p] = _BASE_LUT[ord(base_char[0])]
                M_base[ri, p] = True

    return Q, M_qv, B, M_base, read_to_ridx

logger = logging.getLogger(__name__)


@dataclass
class BoostingArgs:
    """XGBoost training parameters."""
    eta: float = 0.1
    num_round: int = 200
    max_depth: int = 6
    objective: str = "reg:squarederror"


def pileup_to_boosting_data(
    pu_dat: PileupData,
    refseq: str,
    loci: List[int],
    context_size: int = 10,
    reverse: bool = False,
    run_qv_mean: float = 0.0,
    run_qv_sd: float = 1.0,
    use_gpu: bool = True
) -> np.ndarray:
    """
    Convert pileup data to feature matrix for XGBoost (fast pairwise version).

    Extracts features:
    - QV mean (2*context_size + 1 features)
    - QV covariance (context_size^2 features)
    - QV coskewness (context_size^2 features)
    - QV cokurtosis (context_size^2 features)
    - Joint substitution rate (25 * context_size^2 features)
    - Context encoding (2 * (2*context_size + 1) features)

    Args:
        pu_dat: PileupData from pileup_bam
        refseq: Reference sequence string for chromosome
        loci: List of 1-based positions to process
        context_size: Number of positions on each side of target
        reverse: Whether processing reverse strand
        run_qv_mean: Run-level QV mean for normalization
        run_qv_sd: Run-level QV standard deviation
        use_gpu: Whether to use GPU acceleration

    Returns:
        Feature matrix (n_loci x n_features)
    """
    if pu_dat.isref:
        raise ValueError("pileup_to_boosting_data: pu_dat.isref should be False")

    # Build pu_dat-level dense matrices once per batch. Every locus in this
    # call shares the same pileup, so we can slice 21-column windows out of a
    # single (n_reads, tlen) table instead of rebuilding per locus.
    if not reverse:
        Q_full, M_qv_full, B_full, M_base_full, _ = _build_pileup_matrices(
            pu_dat.bup_qv, pu_dat.bup
        )
    else:
        Q_full, M_qv_full, B_full, M_base_full, _ = _build_pileup_matrices(
            pu_dat.bun_qv, pu_dat.bun
        )

    # Every window position has its own column here, so the position -> column
    # map is the identity.
    col_of_pos = np.arange(Q_full.shape[1], dtype=np.int64)

    return _features_from_matrices(
        Q_full, M_qv_full, B_full, M_base_full, col_of_pos, pu_dat.blocus,
        refseq, loci, context_size, reverse, run_qv_mean, run_qv_sd, use_gpu
    )


def pileup_matrix_to_boosting_data(
    pu_mat: PileupMatrix,
    refseq: str,
    loci: List[int],
    context_size: int = 10,
    run_qv_mean: float = 0.0,
    run_qv_sd: float = 1.0,
    use_gpu: bool = True
) -> np.ndarray:
    """
    Convert a matrix pileup to a feature matrix.

    Same features and same values as :func:`pileup_to_boosting_data`, but
    reading from :class:`PileupMatrix` — which the pileup produced directly —
    instead of rebuilding the matrices from per-position dictionaries.

    Args:
        pu_mat: PileupMatrix from pileup_matrix_bam
        refseq: Reference sequence string for the chromosome
        loci: List of 1-based positions to process
        context_size: Number of positions on each side of target
        run_qv_mean: Run-level QV mean for normalization
        run_qv_sd: Run-level QV standard deviation
        use_gpu: Whether to use GPU acceleration

    Returns:
        Feature matrix (n_loci x n_features)
    """
    return _features_from_matrices(
        pu_mat.Q, pu_mat.M_qv, pu_mat.B, pu_mat.M_base, pu_mat.col_of_pos,
        pu_mat.blocus, refseq, loci, context_size, pu_mat.reverse,
        run_qv_mean, run_qv_sd, use_gpu
    )


def _features_from_matrices(
    Q_full: np.ndarray,
    M_qv_full: np.ndarray,
    B_full: np.ndarray,
    M_base_full: np.ndarray,
    col_of_pos: np.ndarray,
    blocus: int,
    refseq: str,
    loci: List[int],
    context_size: int,
    reverse: bool,
    run_qv_mean: float,
    run_qv_sd: float,
    use_gpu: bool
) -> np.ndarray:
    """
    Shared feature extraction over dense per-read pileup matrices.

    ``col_of_pos`` maps a 0-based window offset to a column of the matrices,
    which lets a caller materialize only the positions it cares about and
    point everything else at a permanently empty column. Its length defines
    the window, independently of how many columns the matrices actually have.
    """
    if run_qv_sd <= 0:
        raise ValueError("run_qv_sd must be positive")

    block_size = 2 * context_size + 1
    n_loci = len(loci)

    if n_loci == 0:
        return np.zeros((0, 0), dtype=np.float32)

    # The feature matrix is built directly in its final (locus, feature)
    # layout. The statistics functions already return locus-major arrays, so
    # accumulating into (feature, locus) blocks and transposing at the end
    # meant every block -- including the 11025-column joint tensor -- made two
    # strided passes through memory for no change in value.
    pair = block_size * block_size
    widths = (block_size, pair, pair, pair, 25 * pair, 2 * block_size)
    offsets = np.cumsum((0,) + widths)
    feature_matrix = np.zeros((n_loci, offsets[-1]), dtype=np.float32)

    qv_mean = feature_matrix[:, offsets[0]:offsets[1]]
    qv_cov = feature_matrix[:, offsets[1]:offsets[2]]
    qv_sk = feature_matrix[:, offsets[2]:offsets[3]]
    qv_kt = feature_matrix[:, offsets[3]:offsets[4]]
    sr_cov = feature_matrix[:, offsets[4]:offsets[5]]
    context_encoded = feature_matrix[:, offsets[5]:offsets[6]]

    tlen = col_of_pos.size
    n_reads = Q_full.shape[0]
    refseq_len = len(refseq)

    ref_idx_full = _encode_ref_window(refseq, blocus, tlen)

    # Partition loci into valid (window fully inside the pileup) and invalid.
    # Invalid slots keep their zero entries, matching the scalar version.
    loci_arr = np.asarray(loci, dtype=np.int64)
    p_starts = loci_arr - context_size - 1 - blocus
    valid_mask = (p_starts >= 0) & (p_starts + block_size <= tlen)
    valid_idx = np.flatnonzero(valid_mask)
    n_valid = valid_idx.size

    if n_valid > 0 and n_reads > 0:
        # Gather 21-column windows for every valid locus in one fancy-index op.
        offsets = np.arange(block_size, dtype=np.int64)
        win_idx = p_starts[valid_idx, None] + offsets[None, :]  # (n_valid, block_size)
        col_idx = col_of_pos[win_idx]

        # Q_full[:, col_idx] -> (n_reads, n_valid, block_size); move loci first.
        Q_batch = np.ascontiguousarray(np.transpose(Q_full[:, col_idx], (1, 0, 2)))
        M_qv_batch = np.ascontiguousarray(np.transpose(M_qv_full[:, col_idx], (1, 0, 2)))
        B_batch = np.ascontiguousarray(np.transpose(B_full[:, col_idx], (1, 0, 2)))
        M_base_batch = np.ascontiguousarray(np.transpose(M_base_full[:, col_idx], (1, 0, 2)))
        ref_idx_batch = ref_idx_full[win_idx]  # (n_valid, block_size)

        if reverse:
            # Mirror the 21 positions (scalar version walks loci_range[::-1]).
            Q_batch = np.ascontiguousarray(Q_batch[:, :, ::-1])
            M_qv_batch = np.ascontiguousarray(M_qv_batch[:, :, ::-1])
            B_batch = np.ascontiguousarray(B_batch[:, :, ::-1])
            M_base_batch = np.ascontiguousarray(M_base_batch[:, :, ::-1])
            ref_idx_batch = np.ascontiguousarray(ref_idx_batch[:, ::-1])

        # Optionally move the batch to CUDA and run statistics there. The
        # stats functions auto-detect torch inputs; results come back to
        # NumPy via .cpu().numpy() before assignment into the output arrays.
        on_gpu = use_gpu and _torch_cuda_ready()
        # One-shot log so the user can see unambiguously which path is in use.
        mode = 'gpu' if on_gpu else 'cpu'
        if mode not in _ANNOUNCED_MODES:
            if on_gpu:
                logger.info(
                    "Feature statistics running on GPU (torch.cuda: %s)",
                    _torch.cuda.get_device_name(0),
                )
            else:
                reason = (
                    "use_gpu=False" if not use_gpu
                    else "torch.cuda not available"
                )
                logger.info("Feature statistics running on CPU (%s)", reason)
            _ANNOUNCED_MODES.add(mode)
        if on_gpu:
            dev = _torch.device('cuda')
            Q_t = _torch.from_numpy(Q_batch).to(dev)
            M_qv_t = _torch.from_numpy(M_qv_batch).to(dev)
            B_t = _torch.from_numpy(B_batch).to(dev)
            M_base_t = _torch.from_numpy(M_base_batch).to(dev)
            ref_idx_t = _torch.from_numpy(ref_idx_batch).to(dev)

            mean_b, cov_b, sk_b, kt_b = pairwise_qv_stats_matrix(Q_t, M_qv_t)
            joint_b = pairwise_joint_sr_matrix(
                B_t, M_base_t, ref_idx_t, reverse=reverse
            )
            mean_b = mean_b.cpu().numpy()
            cov_b = cov_b.cpu().numpy()
            sk_b = sk_b.cpu().numpy()
            kt_b = kt_b.cpu().numpy()
            joint_b = joint_b.cpu().numpy()
        else:
            mean_b, cov_b, sk_b, kt_b = pairwise_qv_stats_matrix(Q_batch, M_qv_batch)
            joint_b = pairwise_joint_sr_matrix(
                B_batch, M_base_batch, ref_idx_batch, reverse=reverse
            )

        qv_mean[valid_idx] = mean_b
        qv_cov[valid_idx] = cov_b.reshape(n_valid, pair)
        qv_sk[valid_idx] = sk_b.reshape(n_valid, pair)
        qv_kt[valid_idx] = kt_b.reshape(n_valid, pair)

        # The (5, 5) base axes flatten row-major, matching the scalar
        # version's var_joint.flatten().
        sr_cov[valid_idx] = joint_b.reshape(n_valid, 25 * pair)

    # Context encoding: gather every locus's 21-base window out of the
    # reference in one shot, then apply the 2-bit encoding through lookup
    # tables. Loci whose window runs off either end keep their zero column.
    ctx_starts = loci_arr - context_size - 1
    ctx_idx = np.flatnonzero(
        (ctx_starts >= 0) & (loci_arr + context_size <= refseq_len)
    )
    if ctx_idx.size > 0:
        ctx_offsets = np.arange(block_size, dtype=np.int64)
        ctx_chars = _refseq_bytes(refseq)[
            ctx_starts[ctx_idx, None] + ctx_offsets[None, :]
        ]
        if reverse:
            ctx_chars = _COMPLEMENT_BYTE[ctx_chars][:, ::-1]
        # Per base the encoding emits two adjacent values, so build
        # (n, block_size, 2) and flatten the trailing pair.
        encoded = np.empty((ctx_idx.size, block_size, 2), dtype=np.float32)
        encoded[:, :, 0] = _CTX_BIT0[ctx_chars]
        encoded[:, :, 1] = _CTX_BIT1[ctx_chars]
        context_encoded[ctx_idx] = encoded.reshape(
            ctx_idx.size, 2 * block_size
        )

    # Run-level normalization, matching the scalar version (applied only to
    # mean and cov; sk/kt are already scaled by per-locus sigma). It covers
    # loci with no pileup too, whose zeros therefore become -mean/sd. Written
    # as the same expressions the stacked version used, so scalar promotion
    # rounds exactly the way it did before.
    qv_mean[...] = (qv_mean - run_qv_mean) / run_qv_sd
    qv_cov[...] = qv_cov / (run_qv_sd ** 2)

    return feature_matrix


def train_boosting_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    args: BoostingArgs = None,
    verbosity: int = 1
) -> xgb.Booster:
    """
    Train XGBoost model.

    Args:
        X_train: Training feature matrix
        y_train: Training labels
        args: Training parameters
        verbosity: Verbosity level

    Returns:
        Trained XGBoost Booster model
    """
    if args is None:
        args = BoostingArgs()

    dtrain = xgb.DMatrix(X_train, label=y_train)

    params = {
        'eta': args.eta,
        'max_depth': args.max_depth,
        'objective': args.objective,
        'verbosity': verbosity
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=args.num_round
    )

    return model


def predict_boosting_model(
    model: xgb.Booster,
    X: np.ndarray
) -> np.ndarray:
    """
    Make predictions with XGBoost model.

    Args:
        model: Trained XGBoost model
        X: Feature matrix

    Returns:
        Prediction array
    """
    dmatrix = xgb.DMatrix(X)
    return model.predict(dmatrix)


def save_boosting_model(
    model: xgb.Booster,
    outfile: str
) -> None:
    """
    Save XGBoost model to file.

    Args:
        model: Trained model
        outfile: Output file path
    """
    model.save_model(outfile)
    logger.info(f"Model saved to {outfile}")


def load_boosting_model(infile: str) -> xgb.Booster:
    """
    Load XGBoost model from file.

    Args:
        infile: Input file path

    Returns:
        Loaded XGBoost model
    """
    model = xgb.Booster()
    model.load_model(infile)
    return model


def cv_boosting_model(
    X: np.ndarray,
    y: np.ndarray,
    args: BoostingArgs = None,
    nfold: int = 5
) -> dict:
    """
    Cross-validate XGBoost model.

    Args:
        X: Feature matrix
        y: Labels
        args: Training parameters
        nfold: Number of folds

    Returns:
        CV results dictionary
    """
    if args is None:
        args = BoostingArgs()

    dtrain = xgb.DMatrix(X, label=y)

    params = {
        'eta': args.eta,
        'max_depth': args.max_depth,
        'objective': args.objective
    }

    return xgb.cv(
        params,
        dtrain,
        num_boost_round=args.num_round,
        nfold=nfold
    )


@dataclass
class FeatureJob:
    """
    One (chromosome, strand) unit of feature extraction.

    A MARS run produces one of these per transcript per strand. Collecting them
    all before extraction starts is what lets a single worker pool cover the
    whole run; see :func:`bam_to_boosting_jobs`.
    """
    chromosome: str
    loci: List[int]
    outdir: str
    reverse: bool = False


# Run-wide, read-only state for the worker processes: the BAM path, the
# normalization constants, and the fully expanded job list including each job's
# reference sequence. Workers receive it once for the whole run rather than once
# per job, and under the fork start method they inherit it without any pickling
# at all.
_RUN_CTX: Dict[str, Any] = {}


def _init_run_worker(ctx: Dict[str, Any]) -> None:
    """Pool initializer for start methods that cannot inherit memory."""
    _RUN_CTX.clear()
    _RUN_CTX.update(ctx)


def _write_batch(
    run: Dict[str, Any],
    job: Dict[str, Any],
    batch_idx: int,
    loci_batch: List[int],
) -> None:
    """Pileup one batch of loci, extract features, and save the .npz."""
    context_size = run['context_size']
    tstart = loci_batch[0] - context_size
    tend = loci_batch[-1] + context_size

    # Only the context windows of the job's loci are materialized; everything
    # else in the batch span is left out of the matrices entirely.
    keep_mask = loci_keep_mask(tstart, tend, job['loci_sorted'], context_size)

    pu_mat = pileup_matrix_bam(
        run['bamfile'], job['chromosome'], tstart, tend,
        reverse=job['reverse'], keep_mask=keep_mask
    )

    X = pileup_matrix_to_boosting_data(
        pu_mat, job['refseq_chr'], loci_batch,
        context_size=context_size,
        run_qv_mean=run['run_qv_mean'],
        run_qv_sd=run['run_qv_sd'],
        use_gpu=run['use_gpu']
    )

    # Round to 4 decimal places and convert to sparse matrix. X is ours alone,
    # so round in place rather than allocating a second copy of it.
    np.round(X, decimals=4, out=X)
    X_sparse = csr_matrix(X)

    strand = '-' if job['reverse'] else '+'
    depth = pu_mat.depth[
        np.asarray(loci_batch, dtype=np.int64) - pu_mat.blocus - 1
    ]

    outfile = Path(job['outdir']) / f"X_{batch_idx + 1}.npz"
    np.savez(
        outfile,
        data=X_sparse.data.astype(np.float32),
        indices=X_sparse.indices,
        indptr=X_sparse.indptr,
        shape=X_sparse.shape,
        loci=loci_batch,
        chromosome=job['chromosome'],
        strand=strand,
        depth=depth
    )


def _process_batch(task: tuple) -> Optional[str]:
    """
    Process a single batch in a worker process.

    Args:
        task: Tuple of (job_idx, batch_idx, loci_batch). Everything else comes
            from the run context, inherited or installed at pool startup.

    Returns:
        None on success, or a message describing the failure.
    """
    job_idx, batch_idx, loci_batch = task
    job = _RUN_CTX['jobs'][job_idx]
    try:
        _write_batch(_RUN_CTX, job, batch_idx, loci_batch)
        return None
    except Exception as e:
        strand = '-' if job['reverse'] else '+'
        return (f"{job['chromosome']} strand {strand} batch {batch_idx + 1}: "
                f"{e}")


def bam_to_boosting_jobs(
    bamfile: str,
    refseq: Dict[str, str],
    jobs: List[FeatureJob],
    context_size: int = 10,
    batch_size: int = 100,
    run_qv_mean: float = 0.0,
    run_qv_sd: float = 1.0,
    use_gpu: bool = True,
    showprogress: bool = True,
    num_workers: int = 1,
    progress_desc: str = "Feature extraction",
    strict: bool = False,
) -> List[FeatureJob]:
    """
    Extract features for many (chromosome, strand) jobs with one worker pool.

    Parallelism is over every batch of every job at once. This matters because
    the unit of work MARS actually has is small: mapped against a
    transcriptome, most transcripts carry fewer loci than a single batch, so
    parallelising within one job leaves nearly all workers idle while still
    paying to start them. Pooling across jobs instead keeps the workers fed and
    pays the startup cost once per run rather than once per transcript.

    Results are independent of ``num_workers``: each batch writes its own file
    and the arithmetic within a batch does not depend on how the batches were
    scheduled.

    Args:
        bamfile: Path to BAM file
        refseq: Dictionary of reference sequences
        jobs: Feature extraction jobs, one per chromosome and strand
        context_size: Bases of context on each side of a target site. Any
            value is accepted; feature width grows as its square, and loci
            whose window overhangs the reference are dropped
        batch_size: Number of loci per batch
        run_qv_mean: Run-level QV mean
        run_qv_sd: Run-level QV standard deviation
        use_gpu: Whether to use GPU (for statistics computation). Each worker
            process holds its own CUDA context, so high worker counts multiply
            GPU memory use.
        showprogress: Whether to show a progress bar over all batches
        num_workers: Maximum number of worker processes (1 = sequential)
        progress_desc: Label for the progress bar
        strict: Raise instead of dropping loci that do not fit their reference
            sequence, or skipping a job left with none

    Returns:
        The jobs that were accepted, in input order, each carrying only the
        loci that survived the context check. Callers that write per-job side
        files should key off this rather than the input list, so that a
        skipped job does not leave a half-populated directory behind and a
        label vector never outruns its feature rows.
    """
    accepted: List[FeatureJob] = []
    job_ctxs: List[Dict[str, Any]] = []
    tasks: List[tuple] = []

    for job in jobs:
        Path(job.outdir).mkdir(parents=True, exist_ok=True)
        if len(job.loci) == 0:
            continue

        loci = sorted(job.loci)
        strand_label = '-' if job.reverse else '+'

        # Unless the caller asked for strictness, a job that cannot be laid out
        # is dropped with a warning rather than aborting the run: one malformed
        # transcript should not cost the user every other transcript.
        try:
            refseq_chr = refseq[job.chromosome]
        except KeyError:
            if strict:
                raise
            logger.warning(
                "Skipping %s strand %s: not found in reference",
                job.chromosome, strand_label,
            )
            continue

        # A locus is usable when its whole context window lies inside the
        # reference. Only the loci that overhang are dropped, so a large
        # context costs the transcript its outermost sites instead of costing
        # the run the whole transcript.
        loci_arr = np.asarray(loci, dtype=np.int64)
        fits = ((loci_arr - context_size >= 1)
                & (loci_arr + context_size <= len(refseq_chr)))
        n_dropped = int((~fits).sum())
        if n_dropped:
            reason = (
                f"{n_dropped}/{loci_arr.size} loci of {job.chromosome} strand "
                f"{strand_label} have no room for a context of {context_size} "
                f"in a {len(refseq_chr)} nt reference"
            )
            if strict:
                raise ValueError(reason)
            logger.warning("Dropping %s", reason)

        loci_arr = loci_arr[fits]
        if loci_arr.size == 0:
            if strict:
                raise ValueError(
                    f"No locus of {job.chromosome} strand {strand_label} has "
                    f"room for a context of {context_size}"
                )
            logger.warning(
                "Skipping %s strand %s: no locus has room for a context of %d",
                job.chromosome, strand_label, context_size,
            )
            continue
        loci = loci_arr.tolist()

        job_idx = len(job_ctxs)
        # The caller needs the surviving loci to line up its labels with the
        # rows that were actually written.
        accepted.append(replace(job, loci=loci))
        job_ctxs.append({
            'chromosome': job.chromosome,
            'refseq_chr': refseq_chr,
            'loci_sorted': np.asarray(loci, dtype=np.int64),
            'reverse': job.reverse,
            'outdir': str(job.outdir),
        })

        for batch_idx, start in enumerate(range(0, len(loci), batch_size)):
            tasks.append((job_idx, batch_idx, loci[start:start + batch_size]))

    if not tasks:
        logger.warning(
            "No loci left to extract: a context of %d does not fit any "
            "target site in the supplied references",
            context_size,
        )
        return accepted

    # Feature width grows with the square of the window, so say what a run is
    # about to cost before it allocates anything.
    block_size = 2 * context_size + 1
    n_features = 3 * block_size + 28 * block_size * block_size
    logger.info(
        "Context %d -> window %d nt, %s features per locus (%.1f MB per batch "
        "of %d)",
        context_size, block_size, f"{n_features:,}",
        n_features * batch_size * 4 / 1e6, batch_size,
    )
    if n_features > 200_000:
        logger.warning(
            "Feature width %s is large; lower --batch-size and the worker "
            "count if extraction runs out of memory",
            f"{n_features:,}",
        )

    run_ctx = {
        'bamfile': bamfile,
        'context_size': context_size,
        'run_qv_mean': run_qv_mean,
        'run_qv_sd': run_qv_sd,
        'use_gpu': use_gpu,
        'jobs': job_ctxs,
    }

    # More workers than batches would only add startup cost.
    workers = max(1, min(num_workers, len(tasks)))

    def track(iterator):
        if not showprogress:
            return iterator
        return tqdm(
            iterator,
            total=len(tasks),
            desc=progress_desc,
            unit="batch",
            dynamic_ncols=True,
            leave=False,
        )

    if workers == 1:
        _RUN_CTX.clear()
        _RUN_CTX.update(run_ctx)
        try:
            for task in track(iter(tasks)):
                failure = _process_batch(task)
                if failure:
                    logger.warning("Feature extraction failed for %s", failure)
        finally:
            _RUN_CTX.clear()
        return accepted

    import multiprocessing as mp

    mp_ctx = mp.get_context()
    forking = mp_ctx.get_start_method() == 'fork'
    if forking:
        # Children inherit the context as copy-on-write memory. Sending it
        # through initargs instead would pickle every reference sequence once
        # per worker, which for a transcriptome is most of the run's work.
        _RUN_CTX.clear()
        _RUN_CTX.update(run_ctx)
        pool_kwargs: Dict[str, Any] = {}
    else:
        pool_kwargs = {
            'initializer': _init_run_worker,
            'initargs': (run_ctx,),
        }

    # Hand out work in groups so that IPC does not dominate for short batches,
    # but keep the groups small enough that the tail stays balanced.
    chunksize = max(1, len(tasks) // (workers * 8))

    logger.info(
        "Extracting features for %d batches across %d jobs using %d worker(s)",
        len(tasks), len(job_ctxs), workers,
    )

    try:
        with mp_ctx.Pool(processes=workers, **pool_kwargs) as pool:
            results = pool.imap_unordered(_process_batch, tasks, chunksize)
            for failure in track(results):
                if failure:
                    logger.warning("Feature extraction failed for %s", failure)
    finally:
        if forking:
            _RUN_CTX.clear()

    return accepted


def bam_to_boosting_data(
    bamfile: str,
    refseq: Dict[str, str],
    chromosome: str,
    loci: List[int],
    outdir: str,
    context_size: int = 10,
    batch_size: int = 100,
    reverse: bool = False,
    run_qv_mean: float = 0.0,
    run_qv_sd: float = 1.0,
    use_gpu: bool = True,
    showprogress: bool = True,
    num_workers: int = 1
) -> None:
    """
    Convert BAM file to boosting data (sparse matrices saved to files).

    Single-job form of :func:`bam_to_boosting_jobs`. Callers that have more
    than one chromosome or strand to process should build a list of
    :class:`FeatureJob` and call that function once instead of calling this one
    in a loop, so that the worker pool is created once for the whole run.

    Args:
        bamfile: Path to BAM file
        refseq: Dictionary of reference sequences
        chromosome: Chromosome name
        loci: List of 1-based positions
        outdir: Output directory for sparse matrices
        context_size: Context window size
        batch_size: Number of loci per batch
        reverse: Whether processing reverse strand
        run_qv_mean: Run-level QV mean
        run_qv_sd: Run-level QV standard deviation
        use_gpu: Whether to use GPU (for statistics computation)
        showprogress: Whether to show progress
        num_workers: Number of parallel workers (1 = sequential)
    """
    bam_to_boosting_jobs(
        bamfile,
        refseq,
        [FeatureJob(chromosome, list(loci), str(outdir), reverse)],
        context_size=context_size,
        batch_size=batch_size,
        run_qv_mean=run_qv_mean,
        run_qv_sd=run_qv_sd,
        use_gpu=use_gpu,
        showprogress=showprogress,
        num_workers=num_workers,
        progress_desc=f"{chromosome} batches",
        strict=True,
    )


def load_boosting_data(
    indir: str,
    showprogress: bool = False
) -> csr_matrix:
    """
    Load boosting data from directory.

    Args:
        indir: Directory containing .npz batch files
        showprogress: Whether to show progress

    Returns:
        Concatenated sparse feature matrix
    """
    indir = Path(indir)
    if not indir.is_dir():
        raise ValueError(f"{indir} is not a directory")

    # Find batch files
    batch_files = sorted(indir.glob("X_*.npz"))
    if not batch_files:
        raise ValueError(f"No batch files found in {indir}")

    matrices = []
    for batch_file in batch_files:
        data = np.load(batch_file, allow_pickle=True)
        X_sparse = csr_matrix(
            (data['data'], data['indices'], data['indptr']),
            shape=tuple(data['shape'])
        )
        matrices.append(X_sparse)

        if showprogress:
            logger.info(f"Loaded {batch_file}")

    # Concatenate vertically
    from scipy.sparse import vstack
    return vstack(matrices)


def load_boosting_data_info(indir: str) -> 'pd.DataFrame':
    """
    Load metadata from boosting data directory.

    Args:
        indir: Directory containing .npz batch files

    Returns:
        DataFrame with chr, loci, strand, depth columns
    """
    import pandas as pd

    indir = Path(indir)
    batch_files = sorted(indir.glob("X_*.npz"))

    all_data = []
    for batch_file in batch_files:
        data = np.load(batch_file, allow_pickle=True)
        n = len(data['loci'])
        for i in range(n):
            all_data.append({
                'chr': str(data['chromosome']),
                'loci': int(data['loci'][i]),
                'strand': str(data['strand']),
                'depth': int(data['depth'][i])
            })

    return pd.DataFrame(all_data)
