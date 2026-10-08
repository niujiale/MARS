#!/usr/bin/env python3
"""
FASTA I/O Utilities

Functions for reading FASTA files using BioPython.

Author: MARS Team
"""

import logging
import gzip
from typing import Dict, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


def load_fasta_dna(
    fafile: str,
    chromosome: Optional[str] = None
) -> Dict[str, str]:
    """
    Load DNA sequences from a FASTA file.

    Args:
        fafile: Path to FASTA file (.fa, .fasta, .fna, or .gz compressed)
        chromosome: Specific chromosome to load (optional, loads all if None)

    Returns:
        Dictionary mapping sequence IDs to sequences.
        If chromosome is specified, returns single-entry dict.

    Example:
        >>> seqs = load_fasta_dna("reference.fa")
        >>> print(seqs["chr1"][:50])
        NNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNNN

        >>> chr1_seq = load_fasta_dna("reference.fa", "chr1")
        >>> print(len(chr1_seq["chr1"]))
        248956422
    """
    try:
        from Bio import SeqIO
    except ImportError:
        raise ImportError(
            "BioPython is required for FASTA parsing. "
            "Install with: pip install biopython"
        )

    seqdata = {}
    file_ext = _get_file_extension(fafile)

    # Open file (handle gzip compression)
    if file_ext == '.gz':
        handle = gzip.open(fafile, 'rt')
    elif file_ext in ('.fa', '.fasta', '.fna'):
        handle = open(fafile, 'r')
    else:
        raise ValueError(
            f"Unsupported file extension: {file_ext}. "
            "Expected .fa, .fasta, .fna, or .gz"
        )

    try:
        for record in SeqIO.parse(handle, "fasta"):
            seq_id = record.id
            if chromosome is not None:
                if seq_id == chromosome:
                    seqdata[seq_id] = str(record.seq).upper()
                    break
            else:
                seqdata[seq_id] = str(record.seq).upper()
    finally:
        handle.close()

    if chromosome is not None and chromosome not in seqdata:
        raise ValueError(f"{chromosome} not found in {fafile}")

    logger.info(f"Loaded {len(seqdata)} sequence(s) from {fafile}")
    return seqdata


def _get_file_extension(filepath: str) -> str:
    """
    Get file extension, handling double extensions like .fa.gz.

    Args:
        filepath: Path to file

    Returns:
        File extension (e.g., '.fa' or '.gz')
    """
    path = Path(filepath)
    suffixes = path.suffixes

    if len(suffixes) >= 2 and suffixes[-1] == '.gz':
        return '.gz'
    elif suffixes:
        return suffixes[-1]
    return ''


def get_sequence_context(
    sequence: str,
    position: int,
    context_size: int = 2
) -> str:
    """
    Extract sequence context around a position.

    Args:
        sequence: DNA sequence string
        position: 1-based position in sequence
        context_size: Number of bases on each side

    Returns:
        Sequence context string (length = 2*context_size + 1)

    Example:
        >>> seq = "ACGTACGTACGT"
        >>> get_sequence_context(seq, 6, 2)
        'TACGT'
    """
    # Convert to 0-based index
    pos_0based = position - 1

    start = max(0, pos_0based - context_size)
    end = min(len(sequence), pos_0based + context_size + 1)

    return sequence[start:end]


def context_to_encoded_vec(context: str) -> list:
    """
    Encode DNA context to numerical vector.

    Uses binary encoding:
        A = [0, 0], C = [1, 0], G = [0, 1], T = [1, 1]

    Args:
        context: DNA sequence string

    Returns:
        List of integers (length = 2 * len(context))

    Example:
        >>> context_to_encoded_vec("ACGT")
        [0, 0, 1, 0, 0, 1, 1, 1]
    """
    encoding = {
        'A': [0, 0],
        'C': [1, 0],
        'G': [0, 1],
        'T': [1, 1],
        'N': [0, 0],  # Default for unknown
    }

    result = []
    for base in context.upper():
        result.extend(encoding.get(base, [0, 0]))

    return result


def reverse_complement(sequence: str) -> str:
    """
    Get reverse complement of DNA sequence.

    Args:
        sequence: DNA sequence string

    Returns:
        Reverse complement sequence

    Example:
        >>> reverse_complement("ACGT")
        'ACGT'
        >>> reverse_complement("AACG")
        'CGTT'
    """
    complement = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}
    return ''.join(complement.get(base, 'N') for base in reversed(sequence.upper()))
