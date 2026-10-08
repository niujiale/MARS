#!/usr/bin/env python3
"""
BAM/SAM I/O Utilities

Functions for reading and processing BAM/SAM files using pysam.
Provides chromosome info extraction, CIGAR parsing, and read range utilities.

Author: MARS Team
"""

import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import pysam

logger = logging.getLogger(__name__)


def get_chrinfo_from_bam_header(bamfile: str) -> Dict[str, int]:
    """
    Get chromosome names and lengths from BAM file header.

    Args:
        bamfile: Path to sorted and indexed BAM file

    Returns:
        Dictionary mapping chromosome names to lengths

    Example:
        >>> chrinfo = get_chrinfo_from_bam_header("sample.bam")
        >>> print(chrinfo["chr1"])
        248956422
    """
    chrinfo = {}
    with pysam.AlignmentFile(bamfile, "rb") as bam:
        for ref_name, ref_length in zip(bam.references, bam.lengths):
            chrinfo[ref_name] = ref_length
    return chrinfo


def get_locus_shift_from_cigar(
    op: int,
    length: int
) -> Tuple[int, int]:
    """
    Get locus shift for read and reference from a CIGAR operation.

    CIGAR operations:
        M (0): Match or mismatch - consumes both query and reference
        I (1): Insertion - consumes query only
        D (2): Deletion - consumes reference only
        N (3): Skip - consumes reference only
        S (4): Soft clip - consumes query only
        H (5): Hard clip - consumes neither
        = (7): Sequence match - consumes both
        X (8): Sequence mismatch - consumes both

    Args:
        op: CIGAR operation code (pysam convention)
        length: Length of the operation

    Returns:
        Tuple of (query_shift, reference_shift)

    Example:
        >>> get_locus_shift_from_cigar(0, 10)  # Match
        (10, 10)
        >>> get_locus_shift_from_cigar(1, 5)   # Insertion
        (5, 0)
    """
    # M, =, X: consumes both query and reference
    if op in (0, 7, 8):
        return length, length
    # D, N: consumes reference only
    elif op in (2, 3):
        return 0, length
    # I, S: consumes query only
    elif op in (1, 4):
        return length, 0
    # H: consumes neither
    elif op == 5:
        return 0, 0
    else:
        raise ValueError(f"Unsupported CIGAR operation: {op}")


def get_nread_from_bam(
    bamfile: str,
    chromosome: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None
) -> int:
    """
    Count number of reads in BAM file or region.

    Args:
        bamfile: Path to BAM file
        chromosome: Chromosome name (optional, for region)
        start: Start position (1-based, optional)
        end: End position (optional)

    Returns:
        Number of reads

    Example:
        >>> nread = get_nread_from_bam("sample.bam", "chr1", 1000, 2000)
    """
    nread = 0
    with pysam.AlignmentFile(bamfile, "rb") as bam:
        if chromosome:
            # Convert to 0-based coordinates
            region_start = start - 1 if start else None
            region_end = end if end else None
            for _ in bam.fetch(chromosome, region_start, region_end):
                nread += 1
        else:
            for _ in bam.fetch():
                nread += 1
    return nread


def get_readrange_from_bam(
    bamfile: str,
    chromosome: Optional[str] = None,
    start: Optional[int] = None,
    end: Optional[int] = None
) -> Dict[str, List[int]]:
    """
    Get read names and their alignment ranges.

    Args:
        bamfile: Path to BAM file
        chromosome: Chromosome name (optional)
        start: Start position (1-based, optional)
        end: End position (optional)

    Returns:
        Dictionary mapping read names to [start, end] positions

    Example:
        >>> ranges = get_readrange_from_bam("sample.bam", "chr1")
        >>> print(ranges["read_001"])
        [1000, 1500]
    """
    readrange = {}
    with pysam.AlignmentFile(bamfile, "rb") as bam:
        if chromosome:
            region_start = start - 1 if start else None
            region_end = end if end else None
            iterator = bam.fetch(chromosome, region_start, region_end)
        else:
            iterator = bam.fetch()

        for read in iterator:
            if read.is_unmapped:
                continue
            # Convert to 1-based coordinates
            readrange[read.query_name] = [
                read.reference_start + 1,
                read.reference_end
            ]

    return readrange


def get_readrange_dataframe(
    bamfile: str,
    chromosome: str,
    start: int = 1,
    end: int = 0
) -> 'pd.DataFrame':
    """
    Get read ranges as a pandas DataFrame.

    Args:
        bamfile: Path to BAM file
        chromosome: Chromosome name
        start: Start position (1-based)
        end: End position (0 = chromosome length)

    Returns:
        DataFrame with columns: qname, is_positive_strand, tstart, tend
    """
    import pandas as pd

    chrinfo = get_chrinfo_from_bam_header(bamfile)
    if chromosome not in chrinfo:
        raise ValueError(f"{chromosome} not in {bamfile}")

    chrlen = chrinfo[chromosome]
    if start < 1 or start > chrlen:
        raise ValueError(f"start={start}, should be 1 <= start <= {chrlen}")
    if end < 0 or end > chrlen:
        raise ValueError(f"end={end}, should be 0 <= end <= {chrlen}")

    if end == 0:
        end = chrlen

    data = []
    with pysam.AlignmentFile(bamfile, "rb") as bam:
        for read in bam.fetch(chromosome, start - 1, end):
            if read.is_unmapped:
                continue
            data.append({
                'qname': read.query_name,
                'is_positive_strand': not read.is_reverse,
                'tstart': read.reference_start + 1,
                'tend': read.reference_end
            })

    df = pd.DataFrame(data)
    if len(df) > 0:
        df = df.sort_values(['tstart', 'tend']).reset_index(drop=True)

    return df


def encode_read_names(bamfile: str) -> Dict[str, str]:
    """
    Encode read names to shorter strings for memory efficiency.

    Args:
        bamfile: Path to BAM file

    Returns:
        Dictionary mapping original names to encoded names
    """
    import base64
    encode_dict = {}
    nread = 0

    with pysam.AlignmentFile(bamfile, "rb") as bam:
        for read in bam.fetch():
            if read.query_name not in encode_dict:
                # Encode using base62-like encoding
                encoded = base64.b64encode(
                    nread.to_bytes(8, 'big')
                ).decode('ascii').rstrip('=')
                encode_dict[read.query_name] = encoded
                nread += 1

    return encode_dict


@dataclass
class CigarRange:
    """CIGAR operation with genomic range."""
    operation: int
    op_length: int
    ref_start: int
    ref_end: int
    query_start: int
    query_end: int


def get_cigar_ranges(
    cigar_tuples: List[Tuple[int, int]],
    leftpos: int
) -> List[CigarRange]:
    """
    Get genomic and query ranges for each CIGAR operation.

    Args:
        cigar_tuples: List of (operation, length) tuples from pysam
        leftpos: 1-based left position of alignment

    Returns:
        List of CigarRange objects

    Example:
        >>> read = bam.fetch(...).__next__()
        >>> ranges = get_cigar_ranges(read.cigartuples, read.reference_start + 1)
    """
    ranges = []
    ref_pos = leftpos
    query_pos = 1

    for op, length in cigar_tuples:
        q_shift, r_shift = get_locus_shift_from_cigar(op, length)

        cigar_range = CigarRange(
            operation=op,
            op_length=length,
            ref_start=ref_pos,
            ref_end=ref_pos + r_shift - 1 if r_shift > 0 else ref_pos - 1,
            query_start=query_pos,
            query_end=query_pos + q_shift - 1 if q_shift > 0 else query_pos - 1
        )
        ranges.append(cigar_range)

        ref_pos += r_shift
        query_pos += q_shift

    return ranges


def split_to_ranges(
    value: int,
    chunk_size: int
) -> List[Tuple[int, int]]:
    """
    Split an integer range [1, value] into chunks.

    Args:
        value: End of range
        chunk_size: Size of each chunk

    Returns:
        List of (start, end) tuples

    Example:
        >>> split_to_ranges(20, 6)
        [(1, 6), (7, 12), (13, 20)]
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if value <= chunk_size:
        return [(1, value)]

    ranges = []
    start = 1
    while start <= value:
        end = min(start + chunk_size - 1, value)
        ranges.append((start, end))
        start = end + 1

    return ranges


def split_chr(
    chrinfo: Dict[str, int],
    chunk_size: int
) -> Dict[str, List[Tuple[int, int]]]:
    """
    Split each chromosome into ranges of specified size.

    Args:
        chrinfo: Dictionary of chromosome lengths
        chunk_size: Size of each chunk

    Returns:
        Dictionary mapping chromosome names to list of (start, end) ranges

    Example:
        >>> chrinfo = {"chr1": 1000000, "chr2": 500000}
        >>> chunks = split_chr(chrinfo, 100000)
    """
    chrinfo_split = {}
    for chrom, length in chrinfo.items():
        chrinfo_split[chrom] = split_to_ranges(length, chunk_size)
    return chrinfo_split
