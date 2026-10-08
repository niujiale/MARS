"""
NFL-Py: Python version of NanoFreeLunch

A Python package for RNA modification detection from Nanopore sequencing data.
This is a Python implementation of NanoFreeLunch (NFL) with GPU acceleration support.

Features:
- BAM file pileup processing
- Quality value (QV) statistics extraction (mean, covariance, coskewness, cokurtosis)
- Substitution rate calculation
- Sparse feature matrix generation for XGBoost
- GPU acceleration via PyTorch/CuPy (optional)

Author: MARS Team
"""

# nfl_py ships inside the MARS distribution, so it reports the MARS version
# rather than maintaining a second version number that can drift.
from mars._version import __version__

from .ioxam import (
    get_chrinfo_from_bam_header,
    get_locus_shift_from_cigar,
    get_nread_from_bam,
    get_readrange_from_bam,
)

from .iofastx import load_fasta_dna

from .pileup import (
    pileup_bam,
    pileup_matrix_bam,
    pileup_count_bam,
    base2shift,
    shift2base,
    extend_loci,
    loci_keep_mask,
)

from .stats import (
    covariance,
    coskewness,
    cokurtosis,
    joint_sr,
)

from .boosting import (
    pileup_to_boosting_data,
    pileup_matrix_to_boosting_data,
    train_boosting_model,
    predict_boosting_model,
    save_boosting_model,
    load_boosting_model,
    bam_to_boosting_data,
    bam_to_boosting_jobs,
    FeatureJob,
    load_boosting_data,
)

from .gpu_utils import (
    is_gpu_available,
    get_device,
    to_device,
)

__all__ = [
    # ioxam
    'get_chrinfo_from_bam_header',
    'get_locus_shift_from_cigar',
    'get_nread_from_bam',
    'get_readrange_from_bam',
    # iofastx
    'load_fasta_dna',
    # pileup
    'pileup_bam',
    'pileup_matrix_bam',
    'pileup_count_bam',
    'base2shift',
    'shift2base',
    'extend_loci',
    'loci_keep_mask',
    # stats
    'covariance',
    'coskewness',
    'cokurtosis',
    'joint_sr',
    # boosting
    'pileup_to_boosting_data',
    'pileup_matrix_to_boosting_data',
    'train_boosting_model',
    'predict_boosting_model',
    'save_boosting_model',
    'load_boosting_model',
    'bam_to_boosting_data',
    'bam_to_boosting_jobs',
    'FeatureJob',
    'load_boosting_data',
    # gpu_utils
    'is_gpu_available',
    'get_device',
    'to_device',
]
