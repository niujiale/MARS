"""
MARS Utility Scripts

This module contains utility scripts for post-processing and analysis:

- merge_bed.py: Merge BED and CSV files
- merge_pred_position.py: Merge prediction position files
- merge_pred_position_add_5mer.py: Add 5-mer context to predictions
- pred_merge_gtf.py: Merge predictions with GTF annotations
- position_pred_add_genepos.py: Add gene positions to predictions
- python_version.py: Check Python package versions

Usage:
    python -m utils.merge_bed -c predictions.csv -b ground_truth.bed -o output.csv
    python -m utils.python_version
"""

__version__ = "1.0.0"
