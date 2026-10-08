"""MARS: RNA modification detection from Nanopore sequencing data.

This package provides the installed command-line surface:

    mars                 umbrella CLI (run / predict / train / prepdata / ...)
    mars-pipeline        alignment + feature extraction
    mars-predict         apply an XGBoost model to extracted features
    mars-train           train a new XGBoost model
    xgb_model_pred       alias of mars-predict (legacy script name)
    xgb_model            alias of mars-train (legacy script name)

Nothing heavy is imported here on purpose: `mars --version` and shell tab
completion must not pay for pysam / xgboost / matplotlib import time.
"""

from ._version import __version__

__all__ = ["__version__"]
