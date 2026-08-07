"""
serialization.py
================
High-level helpers for saving and loading TeasCo-FT pipelines.

Uses joblib for serialization (memory-mapped arrays, efficient for sklearn models).

Usage
-----
    from teascoFT.utils import save_pipeline, load_pipeline

    save_pipeline(pipeline, "models/my_classifier.pkl")
    pipeline = load_pipeline("models/my_classifier.pkl")
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Union

logger = logging.getLogger(__name__)


def save_pipeline(pipeline, path: Union[str, Path]) -> None:
    """
    Save a CascadePipeline to disk.

    Parameters
    ----------
    pipeline : CascadePipeline
    path : str or Path — destination file path (e.g., "models/classifier.pkl")
    """
    pipeline.save(path)


def load_pipeline(path: Union[str, Path]):
    """
    Load a CascadePipeline from disk.

    Parameters
    ----------
    path : str or Path — path to the saved .pkl file

    Returns
    -------
    CascadePipeline
    """
    from teascoFT.cascade.pipeline import CascadePipeline
    return CascadePipeline.load(path)
