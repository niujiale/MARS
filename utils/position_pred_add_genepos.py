#!/usr/bin/env python3
"""
Add Gene Positions to Predictions

Match predictions with GTF annotation and BED ground truth data to add genomic coordinates.
Supports matching by transcript position with coordinate system adjustments.

Usage:
    python position_pred_add_genepos.py -p predictions.csv -g gtf_positions.csv -o output.csv
    python position_pred_add_genepos.py -p pred.csv -g gtf.csv -b ground_truth.bed -o output.csv

Author: MARS Team
"""

import sys
import logging
import argparse
from typing import Optional

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


def load_gtf_positions(gtf_file: str) -> pd.DataFrame:
    """
    Load GTF-derived position information.

    Args:
        gtf_file: Path to GTF positions CSV file

    Returns:
        DataFrame with transcript positions and genomic coordinates
    """
    logger.info(f"Loading GTF positions from: {gtf_file}")
    df = pd.read_csv(gtf_file)
    logger.info(f"Loaded {len(df)} GTF positions")
    return df


def load_predictions(predictions_file: str) -> pd.DataFrame:
    """
    Load prediction results.

    Args:
        predictions_file: Path to predictions CSV file

    Returns:
        DataFrame with predictions
    """
    logger.info(f"Loading predictions from: {predictions_file}")
    df = pd.read_csv(predictions_file)
    logger.info(f"Loaded {len(df)} predictions")
    return df


def load_bed_file(
    bed_file: str,
    add_chr_prefix: bool = True
) -> pd.DataFrame:
    """
    Load BED file with ground truth positions.

    Args:
        bed_file: Path to BED file
        add_chr_prefix: Add 'chr' prefix to chromosome names

    Returns:
        DataFrame with BED positions
    """
    logger.info(f"Loading BED file from: {bed_file}")

    df = pd.read_csv(
        bed_file,
        sep='\t',
        header=None,
        names=['chromosome', 'start', 'end']
    )

    if add_chr_prefix:
        df['chromosome'] = 'chr' + df['chromosome'].astype(str)

    logger.info(f"Loaded {len(df)} BED positions")
    return df


def merge_with_gtf(
    predictions: pd.DataFrame,
    gtf_positions: pd.DataFrame,
    pred_trans_col: str = 'trans',
    pred_pos_col: str = 'trans_pos',
    gtf_trans_col: str = 'transcript_id',
    gtf_pos_col: str = 'tx_pos',
    pos_offset: int = -1
) -> pd.DataFrame:
    """
    Merge predictions with GTF position data.

    Args:
        predictions: DataFrame with predictions
        gtf_positions: DataFrame with GTF positions
        pred_trans_col: Transcript column in predictions
        pred_pos_col: Position column in predictions
        gtf_trans_col: Transcript column in GTF
        gtf_pos_col: Position column in GTF
        pos_offset: Offset to apply to prediction position

    Returns:
        Merged DataFrame
    """
    # Remove version numbers from transcript IDs if present
    predictions = predictions.copy()
    predictions[pred_trans_col] = predictions[pred_trans_col].str.split('.').str[0]

    # Apply position offset
    predictions['_merge_pos'] = predictions[pred_pos_col] + pos_offset

    # Merge
    merged = pd.merge(
        gtf_positions,
        predictions,
        how='inner',
        left_on=[gtf_trans_col, gtf_pos_col],
        right_on=[pred_trans_col, '_merge_pos']
    )

    # Clean up temporary column
    merged = merged.drop(columns=['_merge_pos'])

    logger.info(f"Merged {len(merged)} rows with GTF positions")
    return merged


def merge_with_bed(
    data: pd.DataFrame,
    bed_data: pd.DataFrame,
    chrom_col: str = 'chromosome',
    pos_col: str = 'g_pos',
    pos_offset: int = 1
) -> pd.DataFrame:
    """
    Merge data with BED ground truth positions.

    Args:
        data: DataFrame to merge
        bed_data: DataFrame with BED positions
        chrom_col: Chromosome column name
        pos_col: Genomic position column name
        pos_offset: Offset to apply to position

    Returns:
        Merged DataFrame with matching BED positions
    """
    # Normalize chromosome format
    data = data.copy()
    data[chrom_col] = data[chrom_col].str.replace('chr', '', regex=False)

    # Apply position offset
    data['_merge_g_pos'] = data[pos_col] + pos_offset

    # Normalize BED chromosome format
    bed_data = bed_data.copy()
    bed_data['chromosome'] = bed_data['chromosome'].str.replace('chr', '', regex=False)

    # Merge
    merged = pd.merge(
        data,
        bed_data,
        left_on=[chrom_col, '_merge_g_pos'],
        right_on=['chromosome', 'start'],
        how='inner'
    )

    # Clean up
    merged = merged.drop(columns=['_merge_g_pos', 'start', 'end'])
    if 'chromosome_y' in merged.columns:
        merged = merged.drop(columns=['chromosome_y'])
        merged = merged.rename(columns={'chromosome_x': 'chromosome'})

    logger.info(f"Matched {len(merged)} rows with BED positions")
    return merged


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description='Add genomic coordinates to predictions.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Merge predictions with GTF positions only
    python position_pred_add_genepos.py -p predictions.csv -g gtf_positions.csv -o output.csv

    # Also match with BED ground truth
    python position_pred_add_genepos.py -p pred.csv -g gtf.csv -b truth.bed -o output.csv
        """
    )

    parser.add_argument(
        '-p', '--predictions',
        required=True,
        help='Input predictions CSV file'
    )
    parser.add_argument(
        '-g', '--gtf',
        required=True,
        help='GTF positions CSV file'
    )
    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output CSV file'
    )
    parser.add_argument(
        '-b', '--bed',
        help='Optional BED file with ground truth positions'
    )
    parser.add_argument(
        '--no-chr-prefix',
        action='store_true',
        help='Do not add chr prefix to BED chromosome names'
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
    logger.info("Gene Position Annotator")
    logger.info("="*60)

    # Load data
    predictions = load_predictions(args.predictions)
    gtf_positions = load_gtf_positions(args.gtf)

    # Merge with GTF
    result = merge_with_gtf(predictions, gtf_positions)

    # Optionally merge with BED
    if args.bed:
        bed_data = load_bed_file(args.bed, add_chr_prefix=not args.no_chr_prefix)
        result = merge_with_bed(result, bed_data)

    # Save output
    result.to_csv(args.output, index=False)
    logger.info(f"Output saved to: {args.output}")
    logger.info(f"Final shape: {result.shape}")
    logger.info("="*60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)
