#!/usr/bin/env python3
"""
Pileup Functions

BAM file pileup processing for extracting base calls and quality values.
Provides functions for per-read pileup data extraction.

Author: MARS Team
"""

import logging
import os
from typing import Dict, List, Optional, Set, Tuple, NamedTuple
from dataclasses import dataclass, field
import numpy as np
import pysam

from .ioxam import get_chrinfo_from_bam_header, get_locus_shift_from_cigar, get_nread_from_bam

logger = logging.getLogger(__name__)


# Byte-code lookup table: ASCII code point -> base index (A=1, C=2, G=3, T=4,
# '-'=5, anything else 0). Indexing this with a uint8 array is the vectorized
# equivalent of calling base2shift() on each character. Lowercase is mapped
# too because pysam can return soft-masked bases in lowercase.
BASE_CODE_LUT = np.zeros(256, dtype=np.int8)
for _code, _bases in ((1, 'Aa'), (2, 'Cc'), (3, 'Gg'), (4, 'Tt'), (5, '-')):
    for _b in _bases:
        BASE_CODE_LUT[ord(_b)] = _code
del _code, _bases, _b


# Per-CIGAR-op consumption of the query and the reference, as 0/1 multipliers
# indexed by the pysam op code. These encode exactly the same table as
# get_locus_shift_from_cigar(), but in a form that can be applied to a whole
# CIGAR at once. _CIGAR_SUPPORTED mirrors that function's ValueError.
_CIGAR_SUPPORTED = np.zeros(16, dtype=bool)
_CIGAR_SUPPORTED[[0, 1, 2, 3, 4, 5, 7, 8]] = True
_CIGAR_QMUL = np.zeros(16, dtype=np.int64)
_CIGAR_QMUL[[0, 1, 4, 7, 8]] = 1          # M, I, S, =, X consume query
_CIGAR_TMUL = np.zeros(16, dtype=np.int64)
_CIGAR_TMUL[[0, 2, 3, 7, 8]] = 1          # M, D, N, =, X consume reference


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


def extend_loci(
    loci: List[int],
    context_size: int = 10
) -> Set[int]:
    """
    Extend list of loci to include context positions.

    Args:
        loci: List of 1-based positions
        context_size: Number of positions on each side

    Returns:
        Set of all positions including context
    """
    if len(loci) == 0:
        return set()
    arr = np.asarray(loci, dtype=np.int64)
    offsets = np.arange(-context_size, context_size + 1, dtype=np.int64)
    return set((arr[:, None] + offsets[None, :]).ravel().tolist())


def loci_keep_mask(
    tstart: int,
    tend: int,
    loci: np.ndarray,
    context_size: int = 10
) -> np.ndarray:
    """
    Build a boolean mask over [tstart, tend] marking context windows.

    Equivalent to testing ``pos in extend_loci(loci, context_size)`` for every
    position in the window, but as an O(1)-lookup array instead of a Python
    set. Only the loci that can reach the window are inspected.

    Args:
        tstart: Window start (1-based, inclusive)
        tend: Window end (1-based, inclusive)
        loci: Sorted array of 1-based target positions
        context_size: Number of positions on each side of each locus

    Returns:
        Boolean array of length (tend - tstart + 1); True where the position
        falls within context_size of at least one locus.
    """
    tlen = tend - tstart + 1
    if tlen <= 0:
        return np.zeros(0, dtype=bool)

    loci = np.asarray(loci, dtype=np.int64)
    if loci.size == 0:
        return np.zeros(tlen, dtype=bool)

    lo = np.searchsorted(loci, tstart - context_size, side='left')
    hi = np.searchsorted(loci, tend + context_size, side='right')
    sel = loci[lo:hi]
    if sel.size == 0:
        return np.zeros(tlen, dtype=bool)

    # Mark the [locus - c, locus + c] intervals with a difference array so
    # overlapping windows cost nothing extra.
    delta = np.zeros(tlen + 1, dtype=np.int32)
    starts = np.clip(sel - context_size - tstart, 0, tlen)
    stops = np.clip(sel + context_size + 1 - tstart, 0, tlen)
    np.add.at(delta, starts, 1)
    np.add.at(delta, stops, -1)
    return np.cumsum(delta[:tlen]) > 0


def _expand_ranges(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """
    Concatenate ``range(s, s + l)`` for every (s, l) pair into one array.

    Zero-length entries drop out. This is the vectorized replacement for the
    per-base ``for j in range(qlen)`` loops of the scalar pileup.
    """
    total = int(lengths.sum())
    if total == 0:
        return np.empty(0, dtype=np.int64)
    ends = np.cumsum(lengths)
    offsets = np.arange(total, dtype=np.int64) - np.repeat(ends - lengths, lengths)
    return np.repeat(starts, lengths) + offsets


# Reopening a BAM (and re-reading its header) once per batch is a measurable
# cost when a chromosome is split into hundreds of batches. Handles are cached
# per (pid, path) so a forked worker never inherits its parent's handle, which
# htslib does not support.
_BAM_HANDLES: Dict[Tuple[int, str], pysam.AlignmentFile] = {}


def open_bam_cached(bamfile: str) -> pysam.AlignmentFile:
    """
    Return a process-local, cached read handle for a BAM file.

    The handle stays open for the lifetime of the process. Callers must not
    close it.
    """
    key = (os.getpid(), str(bamfile))
    handle = _BAM_HANDLES.get(key)
    if handle is None:
        handle = pysam.AlignmentFile(bamfile, "rb")
        _BAM_HANDLES[key] = handle
    return handle


def close_cached_bams() -> None:
    """Close every cached handle owned by this process."""
    pid = os.getpid()
    for key in [k for k in _BAM_HANDLES if k[0] == pid]:
        try:
            _BAM_HANDLES.pop(key).close()
        except Exception:  # pragma: no cover - best-effort cleanup
            pass


@dataclass
class PileupMatrix:
    """
    Per-read pileup for one strand, already folded into dense matrices.

    This is the same information as :class:`PileupData` for a single strand,
    but skipping the intermediate ``{read_name: value}`` dictionaries. Those
    dicts were only ever rebuilt into exactly these matrices downstream, so
    producing them directly removes both the Python-level per-base loop and
    the dict round-trip.

    Only positions selected by ``keep_mask`` get a column. ``col_of_pos`` maps
    a 0-based window offset to its column; positions that were not selected
    map to the last column, which is a *void* column left permanently empty.
    An empty column is exactly what the dict version produced for a filtered
    position, so the two representations stay interchangeable.
    """
    Q: np.ndarray            # (n_reads, n_cols + 1) float32, NaN where absent
    M_qv: np.ndarray         # (n_reads, n_cols + 1) bool
    B: np.ndarray            # (n_reads, n_cols + 1) int8, 1..5
    M_base: np.ndarray       # (n_reads, n_cols + 1) bool
    col_of_pos: np.ndarray   # (tlen,) int64, window offset -> column index
    read_names: List[str]
    depth: np.ndarray        # (tlen,) int32, depth on the requested strand
    blocus: int
    chromosome: str
    tstart: int
    tend: int
    reverse: bool


def pileup_matrix_bam(
    bamfile: str,
    chromosome: str,
    tstart: int,
    tend: int = 0,
    reverse: bool = False,
    keep_mask: Optional[np.ndarray] = None,
    read_level_normalize: bool = False,
    chunk_bases: int = 2_000_000
) -> PileupMatrix:
    """
    Pileup one strand of a BAM region directly into dense matrices.

    Numerically equivalent to ``pileup_bam`` followed by folding the
    per-position dicts of the requested strand into (n_reads x n_positions)
    matrices, with two differences that do not affect the result:
    insertions are not tracked, and only the requested strand is built.

    CIGARs are decoded for many reads at a time rather than one read at a
    time, so the NumPy call overhead is amortized over a whole chunk instead
    of being paid per read. Reads are ordered by read name and only reads that
    actually land on a selected position get a row, which keeps the reduction
    over the read axis — and therefore float32 round-off in the downstream
    statistics — identical to the dict-based path.

    Assumes at most one alignment per read name in the region, which is what
    the MARS pipeline produces (secondary and supplementary alignments are
    filtered out before this point).

    Args:
        bamfile: Path to indexed BAM file
        chromosome: Chromosome name
        tstart: Start position (1-based)
        tend: End position (0 = chromosome length)
        reverse: Build the negative strand instead of the positive one
        keep_mask: Boolean array of length (tend - tstart + 1) selecting the
            positions to materialize (None = all of them)
        read_level_normalize: Whether to normalize QV by read mean
        chunk_bases: Soft cap on the query bases decoded per vectorized chunk,
            bounding peak temporary memory on very deep pileups

    Returns:
        PileupMatrix for the requested strand
    """
    bam = open_bam_cached(bamfile)
    chrinfo = dict(zip(bam.references, bam.lengths))
    if chromosome not in chrinfo:
        raise ValueError(f"{chromosome} not found in {bamfile}")

    chrlen = chrinfo[chromosome]
    if tstart < 1 or tstart > chrlen:
        raise ValueError(f"tstart={tstart}, should be 1 <= tstart <= {chrlen}")
    if tend < 0 or tend > chrlen:
        raise ValueError(f"tend={tend}, should be 0 <= tend <= {chrlen}")
    if tend == 0:
        tend = chrlen

    tlen = tend - tstart + 1
    blocus = tstart - 1

    if keep_mask is None:
        keep_mask = np.ones(tlen, dtype=bool)
    elif keep_mask.shape[0] != tlen:
        raise ValueError(
            f"keep_mask has length {keep_mask.shape[0]}, expected {tlen}"
        )

    n_cols = int(keep_mask.sum())
    void_col = n_cols
    col_of_pos = np.full(tlen, void_col, dtype=np.int64)
    col_of_pos[keep_mask] = np.arange(n_cols, dtype=np.int64)
    # Prefix sum of the mask, so "does this read reach any selected position?"
    # is two lookups instead of a scan.
    keep_prefix = np.concatenate(
        ([0], np.cumsum(keep_mask, dtype=np.int64))
    )

    names: List[str] = []
    depth_lo: List[int] = []
    depth_hi: List[int] = []

    # Filtered (row, column, value) triples, accumulated across chunks. These
    # only hold selected positions, so they stay small even though the
    # intermediate per-chunk expansion does not.
    out_rows_b: List[np.ndarray] = []
    out_cols_b: List[np.ndarray] = []
    out_codes: List[np.ndarray] = []
    out_rows_q: List[np.ndarray] = []
    out_cols_q: List[np.ndarray] = []
    out_vals_q: List[np.ndarray] = []

    # Current chunk: one entry per buffered read.
    buf_cigar: List[np.ndarray] = []
    buf_counts: List[int] = []
    buf_left: List[int] = []
    buf_qoff: List[int] = []
    buf_ridx: List[int] = []
    buf_seq: List[str] = []
    buf_qual: List[bytes] = []
    buf_hasqv: List[bool] = []
    state = {"qlen": 0}

    def flush_chunk() -> None:
        if not buf_ridx:
            return

        counts = np.asarray(buf_counts, dtype=np.int64)
        cig = np.concatenate(buf_cigar)
        ops = cig[:, 0]
        lens = cig[:, 1]
        if not _CIGAR_SUPPORTED[ops].all():
            bad = int(ops[~_CIGAR_SUPPORTED[ops]][0])
            raise ValueError(f"Unsupported CIGAR operation: {bad}")

        n_buf = counts.size
        op_read = np.repeat(np.arange(n_buf, dtype=np.int64), counts)
        op_first = np.concatenate(([0], np.cumsum(counts)[:-1]))

        qmul = _CIGAR_QMUL[ops]
        tmul = _CIGAR_TMUL[ops]
        q_shift = qmul * lens
        t_shift = tmul * lens
        # Exclusive cumulative shift, rebased so each read restarts from its
        # own query offset / left-most reference position.
        q_excl = np.cumsum(q_shift) - q_shift
        t_excl = np.cumsum(t_shift) - t_shift
        q_at = (q_excl - q_excl[op_first][op_read]
                + np.asarray(buf_qoff, dtype=np.int64)[op_read])
        t_at = (t_excl - t_excl[op_first][op_read]
                + np.asarray(buf_left, dtype=np.int64)[op_read])

        seq_all = np.frombuffer(
            "".join(buf_seq).encode('latin-1', 'replace'), dtype=np.uint8
        )
        qual_all = np.frombuffer(b"".join(buf_qual), dtype=np.uint8)
        has_qv = np.asarray(buf_hasqv, dtype=bool)
        ridx = np.asarray(buf_ridx, dtype=np.int64)

        def select(op_mask):
            """Expand the marked ops and keep only selected positions."""
            t_pos = _expand_ranges(t_at[op_mask], lens[op_mask])
            reads = np.repeat(op_read[op_mask], lens[op_mask])
            offs = t_pos - tstart
            take = (offs >= 0) & (offs < tlen)
            cols = col_of_pos[offs[take]]
            in_scope = cols != void_col
            return take, in_scope, cols[in_scope], reads[take][in_scope]

        # Aligned columns come from ops that consume both query and reference
        # (M, =, X); '-' columns come from D. N/S/H/I contribute nothing, which
        # is what the scalar version's if/elif chain does.
        aligned = (qmul == 1) & (tmul == 1)
        if aligned.any():
            q_pos = _expand_ranges(q_at[aligned], lens[aligned])
            take, in_scope, cols, reads = select(aligned)
            if cols.size:
                q_pos = q_pos[take][in_scope]
                out_rows_b.append(ridx[reads])
                out_cols_b.append(cols)
                out_codes.append(BASE_CODE_LUT[seq_all[q_pos]])
                with_qv = has_qv[reads]
                if with_qv.any():
                    out_rows_q.append(ridx[reads[with_qv]])
                    out_cols_q.append(cols[with_qv])
                    out_vals_q.append(qual_all[q_pos[with_qv]])

        deleted = ops == 2
        if deleted.any():
            _, _, cols, reads = select(deleted)
            if cols.size:
                out_rows_b.append(ridx[reads])
                out_cols_b.append(cols)
                out_codes.append(np.full(cols.size, 5, dtype=np.int8))

        buf_cigar.clear()
        buf_counts.clear()
        buf_left.clear()
        buf_qoff.clear()
        buf_ridx.clear()
        buf_seq.clear()
        buf_qual.clear()
        buf_hasqv.clear()
        state["qlen"] = 0

    for read in bam.fetch(chromosome, tstart - 1, tend):
        if read.is_unmapped or read.is_reverse != reverse:
            continue
        qseq = read.query_sequence
        if qseq is None:
            continue

        leftpos = read.reference_start + 1
        rightpos = read.reference_end

        # Depth counts every read overlapping the window regardless of
        # keep_mask, matching the scalar version.
        lo = max(leftpos, tstart) - tstart
        hi = min(rightpos, tend) - tstart + 1
        if hi > lo:
            depth_lo.append(lo)
            depth_hi.append(hi)
        else:
            # Cannot reach a selected position either.
            continue

        cigar = read.cigartuples
        if not cigar:
            continue
        # Skip reads whose reference span holds no selected position at all.
        if keep_prefix[hi] == keep_prefix[lo]:
            continue

        raw_qv = read.query_qualities
        if raw_qv is None:
            qual_bytes = bytes(len(qseq))
            hasqv = False
        elif read_level_normalize:
            qual_bytes = _normalize_by_read(
                np.frombuffer(raw_qv, dtype=np.uint8)
            ).tobytes()
            hasqv = True
        else:
            qual_bytes = raw_qv.tobytes()
            hasqv = True

        buf_cigar.append(np.asarray(cigar, dtype=np.int64))
        buf_counts.append(len(cigar))
        buf_left.append(leftpos)
        buf_qoff.append(state["qlen"])
        buf_ridx.append(len(names))
        buf_seq.append(qseq)
        buf_qual.append(qual_bytes)
        buf_hasqv.append(hasqv)
        names.append(read.query_name)
        state["qlen"] += len(qseq)

        if state["qlen"] >= chunk_bases:
            flush_chunk()

    flush_chunk()

    depth = np.zeros(tlen, dtype=np.int32)
    if depth_lo:
        delta = np.zeros(tlen + 1, dtype=np.int32)
        np.add.at(delta, np.asarray(depth_lo, dtype=np.int64), 1)
        np.add.at(delta, np.asarray(depth_hi, dtype=np.int64), -1)
        depth = np.cumsum(delta[:tlen]).astype(np.int32)

    # The read axis is exactly the set of reads that landed on a selected
    # position, sorted by name — the same set and order the dict version's
    # sorted(union of dict keys) produced.
    if out_rows_b:
        all_rows = np.concatenate(out_rows_b)
        contributing = np.unique(all_rows)
    else:
        all_rows = np.empty(0, dtype=np.int64)
        contributing = np.empty(0, dtype=np.int64)

    read_list = sorted({names[i] for i in contributing.tolist()})
    rank_of_name = {name: i for i, name in enumerate(read_list)}
    n_reads = len(read_list)
    row_of_buffered = np.zeros(len(names), dtype=np.int64)
    for i in contributing.tolist():
        row_of_buffered[i] = rank_of_name[names[i]]

    Q = np.full((n_reads, n_cols + 1), np.nan, dtype=np.float32)
    M_qv = np.zeros((n_reads, n_cols + 1), dtype=bool)
    B = np.zeros((n_reads, n_cols + 1), dtype=np.int8)
    M_base = np.zeros((n_reads, n_cols + 1), dtype=bool)

    if n_reads:
        rows_b = row_of_buffered[all_rows]
        cols_b = np.concatenate(out_cols_b)
        B[rows_b, cols_b] = np.concatenate(out_codes)
        M_base[rows_b, cols_b] = True
        if out_rows_q:
            rows_q = row_of_buffered[np.concatenate(out_rows_q)]
            cols_q = np.concatenate(out_cols_q)
            Q[rows_q, cols_q] = np.concatenate(out_vals_q)
            M_qv[rows_q, cols_q] = True

    return PileupMatrix(
        Q=Q, M_qv=M_qv, B=B, M_base=M_base,
        col_of_pos=col_of_pos,
        read_names=read_list,
        depth=depth,
        blocus=blocus,
        chromosome=chromosome,
        tstart=tstart, tend=tend,
        reverse=reverse,
    )


@dataclass
class PileupData:
    """Container for pileup data at multiple positions."""
    # Base pileup: Dict[str, str] = {read_name: base}
    bup: List[Dict[str, str]] = field(default_factory=list)  # positive strand
    bun: List[Dict[str, str]] = field(default_factory=list)  # negative strand

    # Insertion pileup: Dict[str, str] = {read_name: inserted_sequence}
    bup_ins: List[Dict[str, str]] = field(default_factory=list)
    bun_ins: List[Dict[str, str]] = field(default_factory=list)

    # Quality value pileup: Dict[str, int] = {read_name: qv}
    bup_qv: List[Dict[str, int]] = field(default_factory=list)
    bun_qv: List[Dict[str, int]] = field(default_factory=list)

    # Depth arrays
    dp: Optional[np.ndarray] = None  # positive strand depth
    dn: Optional[np.ndarray] = None  # negative strand depth

    # Metadata
    blocus: int = 0  # base locus (tstart - 1)
    chromosome: str = ""
    tstart: int = 0
    tend: int = 0
    isref: bool = False
    isins: bool = True
    isqv: bool = False


def pileup_bam(
    bamfile: str,
    chromosome: str,
    tstart: int,
    tend: int = 0,
    isqv: bool = True,
    isins: bool = True,
    showprogress: bool = False,
    loci_ext: Optional[Set[int]] = None,
    read_level_normalize: bool = False
) -> PileupData:
    """
    Pileup reads from BAM file in specified region.

    Extracts per-read base calls and quality values for each position.

    Args:
        bamfile: Path to indexed BAM file
        chromosome: Chromosome name
        tstart: Start position (1-based)
        tend: End position (0 = chromosome length)
        isqv: Whether to extract quality values
        isins: Whether to track insertions
        showprogress: Whether to show progress bar
        loci_ext: Set of positions to include (None = all in range)
        read_level_normalize: Whether to normalize QV by read mean

    Returns:
        PileupData object with per-read base and QV information

    Example:
        >>> pu = pileup_bam("sample.bam", "chr1", 1000, 2000)
        >>> print(len(pu.bup[0]))  # Number of reads at first position
    """
    # Get chromosome info
    chrinfo = get_chrinfo_from_bam_header(bamfile)
    if chromosome not in chrinfo:
        raise ValueError(f"{chromosome} not found in {bamfile}")

    chrlen = chrinfo[chromosome]
    if tstart < 1 or tstart > chrlen:
        raise ValueError(f"tstart={tstart}, should be 1 <= tstart <= {chrlen}")
    if tend < 0 or tend > chrlen:
        raise ValueError(f"tend={tend}, should be 0 <= tend <= {chrlen}")

    if tend == 0:
        tend = chrlen

    tlen = tend - tstart + 1
    blocus = tstart - 1

    # Initialize output
    bup = [{} for _ in range(tlen)]
    bun = [{} for _ in range(tlen)]
    bup_ins = [{} for _ in range(tlen)]
    bun_ins = [{} for _ in range(tlen)]
    bup_qv = [{} for _ in range(tlen)]
    bun_qv = [{} for _ in range(tlen)]
    dp = np.zeros(tlen, dtype=np.int32)
    dn = np.zeros(tlen, dtype=np.int32)

    # Setup loci filter
    if loci_ext is None:
        loci_ext = set(range(tstart, tend + 1))

    # Process reads
    with pysam.AlignmentFile(bamfile, "rb") as bam:
        for read in bam.fetch(chromosome, tstart - 1, tend):
            if read.is_unmapped:
                continue

            _pileup_read(
                read, tstart, tend, blocus,
                bup, bun, bup_ins, bun_ins, bup_qv, bun_qv, dp, dn,
                isqv, isins, loci_ext, read_level_normalize
            )

    return PileupData(
        bup=bup, bun=bun,
        bup_ins=bup_ins, bun_ins=bun_ins,
        bup_qv=bup_qv, bun_qv=bun_qv,
        dp=dp, dn=dn,
        blocus=blocus,
        chromosome=chromosome,
        tstart=tstart, tend=tend,
        isref=False, isins=isins, isqv=isqv
    )


def _pileup_read(
    read: pysam.AlignedSegment,
    tstart: int,
    tend: int,
    blocus: int,
    bup: List[Dict[str, str]],
    bun: List[Dict[str, str]],
    bup_ins: List[Dict[str, str]],
    bun_ins: List[Dict[str, str]],
    bup_qv: List[Dict[str, int]],
    bun_qv: List[Dict[str, int]],
    dp: np.ndarray,
    dn: np.ndarray,
    isqv: bool,
    isins: bool,
    loci_ext: Set[int],
    read_level_normalize: bool
) -> None:
    """
    Process a single read for pileup.

    Args:
        read: pysam AlignedSegment
        tstart: Start of region
        tend: End of region
        blocus: Base locus offset
        bup, bun: Base pileup dicts for +/- strand
        bup_ins, bun_ins: Insertion pileup dicts
        bup_qv, bun_qv: QV pileup dicts
        dp, dn: Depth arrays
        isqv: Whether to extract QV
        isins: Whether to track insertions
        loci_ext: Set of positions to include
        read_level_normalize: Whether to normalize QV by read mean
    """
    leftpos = read.reference_start + 1  # Convert to 1-based
    rightpos = read.reference_end
    qseq = read.query_sequence
    qname = read.query_name
    is_positive = not read.is_reverse

    # Get quality values
    qv = None
    if isqv:
        qv = read.query_qualities
        if qv is not None and read_level_normalize:
            qv = _normalize_by_read(qv)

    if qseq is None:
        return

    # Update depth
    leftpos_trim = max(leftpos, tstart)
    rightpos_trim = min(rightpos, tend)
    if is_positive:
        dp[(leftpos_trim - blocus - 1):(rightpos_trim - blocus)] += 1
    else:
        dn[(leftpos_trim - blocus - 1):(rightpos_trim - blocus)] += 1

    # Process CIGAR
    if read.cigartuples is None:
        return

    qnextpos = 0
    tnextpos = leftpos

    for op, op_len in read.cigartuples:
        qstart = qnextpos
        tstart_op = tnextpos

        q_shift, t_shift = get_locus_shift_from_cigar(op, op_len)

        qnextpos = qstart + q_shift
        tnextpos = tstart_op + t_shift

        qend = qnextpos
        tend_op = tnextpos

        qlen = qend - qstart
        tlen = tend_op - tstart_op

        # Match or mismatch (M, =, X)
        if qlen > 0 and tlen > 0:
            for j in range(qlen):
                cur_qpos = qstart + j
                cur_tpos = tstart_op + j

                # Check range
                if cur_tpos < tstart or cur_tpos > tend:
                    continue
                if cur_tpos not in loci_ext:
                    continue

                idx = cur_tpos - blocus - 1
                base = qseq[cur_qpos]

                if is_positive:
                    bup[idx][qname] = base
                    if isqv and qv is not None:
                        bup_qv[idx][qname] = qv[cur_qpos]
                else:
                    bun[idx][qname] = base
                    if isqv and qv is not None:
                        bun_qv[idx][qname] = qv[cur_qpos]

        # Deletion (D)
        elif op == 2:
            for j in range(tlen):
                cur_tpos = tstart_op + j
                if cur_tpos < tstart or cur_tpos > tend:
                    continue
                if cur_tpos not in loci_ext:
                    continue

                idx = cur_tpos - blocus - 1

                if is_positive:
                    bup[idx][qname] = '-'
                else:
                    bun[idx][qname] = '-'

        # Insertion (I)
        elif op == 1 and isins:
            cur_tpos = tstart_op
            if cur_tpos < tstart or cur_tpos > tend:
                continue
            if cur_tpos not in loci_ext:
                continue

            idx = cur_tpos - blocus - 1
            ins_seq = qseq[qstart:qend]

            if is_positive:
                bup_ins[idx][qname] = ins_seq
            else:
                bun_ins[idx][qname] = ins_seq


def _normalize_by_read(qv: np.ndarray, prop: float = 0.1) -> np.ndarray:
    """
    Normalize quality values by read mean.

    Args:
        qv: Quality value array
        prop: Proportion to trim for robust mean

    Returns:
        Normalized quality values
    """
    qv_f32 = np.array(qv, dtype=np.float32)

    # Compute trimmed mean
    n = len(qv_f32)
    trim_n = int(n * prop)
    if trim_n > 0:
        sorted_qv = np.sort(qv_f32)
        read_mean_qv = np.mean(sorted_qv[trim_n:-trim_n])
    else:
        read_mean_qv = np.mean(qv_f32)

    # Normalize to center around 91 (like Julia version)
    qv_norm = np.clip(qv_f32 - read_mean_qv + 91.0, 0, 255).astype(np.uint8)
    return qv_norm


@dataclass
class PileupCountData:
    """Container for pileup count data."""
    fqp: np.ndarray  # Positive strand counts [A, C, G, T, Del, Depth]
    fqn: np.ndarray  # Negative strand counts
    fqp_qv: np.ndarray  # Positive strand avg QV per base
    fqn_qv: np.ndarray  # Negative strand avg QV per base
    fqp_ins: List[Dict[str, int]]  # Insertion counts
    fqn_ins: List[Dict[str, int]]
    blocus: int
    chromosome: str
    tstart: int
    tend: int


def pileup_count_bam(
    bamfile: str,
    chromosome: str,
    tstart: int,
    tend: int = 0
) -> PileupCountData:
    """
    Pileup reads and count base frequencies.

    Args:
        bamfile: Path to BAM file
        chromosome: Chromosome name
        tstart: Start position (1-based)
        tend: End position (0 = chromosome length)

    Returns:
        PileupCountData with base frequency matrices
    """
    chrinfo = get_chrinfo_from_bam_header(bamfile)
    if chromosome not in chrinfo:
        raise ValueError(f"{chromosome} not found in {bamfile}")

    chrlen = chrinfo[chromosome]
    if tend == 0:
        tend = chrlen

    tlen = tend - tstart + 1
    blocus = tstart - 1

    # Initialize count matrices
    # Columns: A, C, G, T, Del, Depth
    fqp = np.zeros((tlen, 6), dtype=np.int32)
    fqn = np.zeros((tlen, 6), dtype=np.int32)
    fqp_qv = np.zeros((tlen, 6), dtype=np.float32)
    fqn_qv = np.zeros((tlen, 6), dtype=np.float32)
    fqp_ins = [{} for _ in range(tlen)]
    fqn_ins = [{} for _ in range(tlen)]

    with pysam.AlignmentFile(bamfile, "rb") as bam:
        for read in bam.fetch(chromosome, tstart - 1, tend):
            if read.is_unmapped:
                continue
            _pileup_count_read(
                read, tstart, tend, blocus,
                fqp, fqn, fqp_qv, fqn_qv, fqp_ins, fqn_ins
            )

    # Normalize QV by counts
    with np.errstate(divide='ignore', invalid='ignore'):
        fqp_qv[:, :4] = np.where(fqp[:, :4] > 0, fqp_qv[:, :4] / fqp[:, :4], 0)
        fqn_qv[:, :4] = np.where(fqn[:, :4] > 0, fqn_qv[:, :4] / fqn[:, :4], 0)
        fqp_qv[:, 4] = np.where(fqp_qv[:, 5] > 0, fqp_qv[:, 4] / fqp_qv[:, 5], 0)
        fqn_qv[:, 4] = np.where(fqn_qv[:, 5] > 0, fqn_qv[:, 4] / fqn_qv[:, 5], 0)

    return PileupCountData(
        fqp=fqp, fqn=fqn,
        fqp_qv=fqp_qv, fqn_qv=fqn_qv,
        fqp_ins=fqp_ins, fqn_ins=fqn_ins,
        blocus=blocus,
        chromosome=chromosome,
        tstart=tstart, tend=tend
    )


def _pileup_count_read(
    read: pysam.AlignedSegment,
    tstart: int,
    tend: int,
    blocus: int,
    fqp: np.ndarray,
    fqn: np.ndarray,
    fqp_qv: np.ndarray,
    fqn_qv: np.ndarray,
    fqp_ins: List[Dict[str, int]],
    fqn_ins: List[Dict[str, int]]
) -> None:
    """Process a single read for pileup counting."""
    leftpos = read.reference_start + 1
    rightpos = read.reference_end
    qseq = read.query_sequence
    is_positive = not read.is_reverse
    qv = read.query_qualities

    if qseq is None or read.cigartuples is None:
        return

    # Update depth
    if is_positive:
        fqp[(leftpos - blocus - 1):(rightpos - blocus), 5] += 1
    else:
        fqn[(leftpos - blocus - 1):(rightpos - blocus), 5] += 1

    # Process CIGAR
    qnextpos = 0
    tnextpos = leftpos

    for op, op_len in read.cigartuples:
        qstart = qnextpos
        tstart_op = tnextpos

        q_shift, t_shift = get_locus_shift_from_cigar(op, op_len)

        qnextpos = qstart + q_shift
        tnextpos = tstart_op + t_shift

        qlen = qnextpos - qstart
        tlen = tnextpos - tstart_op

        # Match/mismatch
        if qlen > 0 and tlen > 0:
            for j in range(qlen):
                cur_qpos = qstart + j
                cur_tpos = tstart_op + j

                if cur_tpos < tstart or cur_tpos > tend:
                    continue

                idx = cur_tpos - blocus - 1
                base_idx = base2shift(qseq[cur_qpos]) - 1  # 0-indexed

                if base_idx >= 0 and base_idx < 4:
                    if is_positive:
                        fqp[idx, base_idx] += 1
                        if qv is not None:
                            fqp_qv[idx, base_idx] += qv[cur_qpos]
                            fqp_qv[idx, 4] += qv[cur_qpos]
                            fqp_qv[idx, 5] += 1
                    else:
                        fqn[idx, base_idx] += 1
                        if qv is not None:
                            fqn_qv[idx, base_idx] += qv[cur_qpos]
                            fqn_qv[idx, 4] += qv[cur_qpos]
                            fqn_qv[idx, 5] += 1

        # Deletion
        elif op == 2:
            for j in range(tlen):
                cur_tpos = tstart_op + j
                if cur_tpos < tstart or cur_tpos > tend:
                    continue
                idx = cur_tpos - blocus - 1
                if is_positive:
                    fqp[idx, 4] += 1
                else:
                    fqn[idx, 4] += 1

        # Insertion
        elif op == 1:
            if tstart_op < tstart or tstart_op > tend:
                continue
            idx = tstart_op - blocus - 1
            ins_seq = qseq[qstart:qnextpos]

            if is_positive:
                fqp_ins[idx][ins_seq] = fqp_ins[idx].get(ins_seq, 0) + 1
            else:
                fqn_ins[idx][ins_seq] = fqn_ins[idx].get(ins_seq, 0) + 1


def get_mean_qv(
    bamfile: str,
    showprogress: bool = False
) -> Tuple[float, float]:
    """
    Get mean and standard deviation of quality values from BAM file.

    Args:
        bamfile: Path to BAM file
        showprogress: Whether to show progress

    Returns:
        Tuple of (mean_qv, std_qv)
    """
    mean_qv = 0.0
    mean_qv_2 = 0.0
    n_qv = 0

    with pysam.AlignmentFile(bamfile, "rb") as bam:
        for read in bam.fetch():
            qv = read.query_qualities
            if qv is not None:
                qv_arr = np.array(qv, dtype=np.float64)
                mean_qv += np.sum(qv_arr)
                mean_qv_2 += np.sum(qv_arr ** 2)
                n_qv += len(qv_arr)

    if n_qv > 0:
        mean = mean_qv / n_qv
        std = np.sqrt(mean_qv_2 / n_qv - mean ** 2)
        return mean, std

    return 0.0, 1.0
