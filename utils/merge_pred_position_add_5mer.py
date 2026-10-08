#!/usr/bin/env python3
"""
Add 5-mer Context to Prediction Positions

Extract 5-mer sequence context around each prediction position from reference transcripts.
The 5-mer is centered on the modification site (2 bases upstream, the site, 2 bases downstream).

Usage:
    python merge_pred_position_add_5mer.py -i predictions.csv -f transcripts.fa -o output.csv

Author: MARS Team
"""

import sys
import logging
import argparse
from typing import Dict, Optional

import pandas as pd
from Bio import SeqIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_reference_transcripts(fasta_file: str) -> Dict[str, str]:
    """
    Load reference transcript sequences from FASTA file.

    Args:
        fasta_file: Path to reference FASTA file

    Returns:
        Dictionary mapping transcript IDs to sequences
    """
    logger.info(f"Loading reference transcripts from: {fasta_file}")

    reference_dict = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        reference_dict[record.id] = str(record.seq)

    logger.info(f"Loaded {len(reference_dict)} transcripts")
    return reference_dict


def extract_5mer_context(
    reference_dict: Dict[str, str],
    transcript_id: str,
    position: int,
    context_size: int = 2
) -> Optional[str]:
    """
    Extract 5-mer sequence context around a position.

    The position uses 1-based indexing. Returns a sequence of length
    (2 * context_size + 1), centered on the given position.

    Args:
        reference_dict: Dictionary of transcript sequences
        transcript_id: Transcript identifier
        position: 1-based position in transcript
        context_size: Number of bases on each side (default: 2 for 5-mer)

    Returns:
        5-mer sequence string, or None if position is invalid
    """
    seq = reference_dict.get(transcript_id)

    if seq is None:
        return None

    # Convert to 0-based index
    pos_0based = position - 1

    # Calculate window bounds
    start = max(0, pos_0based - context_size)
    end = min(len(seq), pos_0based + context_size + 1)

    # Check if we can extract full context
    if end - start < 2 * context_size + 1:
        # Partial context at sequence boundaries
        return seq[start:end]

    return seq[start:end]


def add_5mer_to_dataframe(
    df: pd.DataFrame,
    reference_dict: Dict[str, str],
    transcript_col: str = 'transcript_id',
    position_col: str = 'tx_pos',
    context_size: int = 2
) -> pd.DataFrame:
    """
    Add 5-mer context column to DataFrame.

    Args:
        df: Input DataFrame with transcript and position columns
        reference_dict: Dictionary of transcript sequences
        transcript_col: Name of transcript ID column
        position_col: Name of position column
        context_size: Number of bases on each side

    Returns:
        DataFrame with added '5mer' column
    """
    logger.info("Extracting 5-mer contexts...")

    df = df.copy()
    df['5mer'] = df.apply(
        lambda row: extract_5mer_context(
            reference_dict,
            row[transcript_col],
            int(row[position_col]),
            context_size
        ),
        axis=1
    )

    # Count successful extractions
    valid_count = df['5mer'].notna().sum()
    logger.info(f"Extracted {valid_count}/{len(df)} 5-mer contexts")

    return df


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Add 5-mer sequence context to prediction positions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python merge_pred_position_add_5mer.py -i predictions.csv -f transcripts.fa -o with_5mer.csv
    python merge_pred_position_add_5mer.py -i data.csv -f ref.fa -o output.csv --context-size 3
        """
    )

    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Input CSV file with predictions'
    )
    parser.add_argument(
        '-f', '--fasta',
        required=True,
        help='Reference transcript FASTA file'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output CSV file path'
    )
    parser.add_argument(
        '--transcript-col',
        default='transcript_id',
        help='Column name for transcript ID (default: transcript_id)'
    )
    parser.add_argument(
        '--position-col',
        default='tx_pos',
        help='Column name for position (default: tx_pos)'
    )
    parser.add_argument(
        '--context-size',
        type=int,
        default=2,
        help='Number of bases on each side of position (default: 2 for 5-mer)'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    logger.info("="*60)
    logger.info("5-mer Context Extractor")
    logger.info("="*60)

    # Load reference sequences
    reference_dict = load_reference_transcripts(args.fasta)

    # Load input data
    logger.info(f"Loading input data from: {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Input data shape: {df.shape}")

    # Add 5-mer context
    df = add_5mer_to_dataframe(
        df,
        reference_dict,
        transcript_col=args.transcript_col,
        position_col=args.position_col,
        context_size=args.context_size
    )

    # Save output
    df.to_csv(args.output, index=False)
    logger.info(f"Output saved to: {args.output}")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
