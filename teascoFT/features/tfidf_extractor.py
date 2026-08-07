"""
tfidf_extractor.py
==================
TF-IDF based feature extractor — fully offline, zero extra dependencies.

Uses scikit-learn's TfidfVectorizer under the hood. Produces sparse matrices
that are directly compatible with all scikit-learn classifiers.

Usage
-----
    extractor = TFIDFExtractor(max_features=30_000, ngram_range=(1, 2))
    extractor.fit(train_texts)

    X_train = extractor.transform(train_texts)   # sparse matrix
    X_test  = extractor.transform(test_texts)
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Tuple, Union

import numpy as np
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)


class TFIDFExtractor:
    """
    Wraps sklearn's TfidfVectorizer with a clean fit/transform interface.

    Parameters
    ----------
    max_features : int, optional
        Maximum vocabulary size. Default: 50_000.
    ngram_range : tuple of (int, int)
        Min and max n-gram lengths. Default: (1, 2) — unigrams + bigrams.
    sublinear_tf : bool
        Apply sublinear TF scaling (log(1+tf)). Improves performance on
        long documents. Default: True.
    strip_accents : str, optional
        "unicode" to strip accents; None to keep them. Default: "unicode".
    analyzer : str
        "word" or "char" or "char_wb". Default: "word".
    min_df : int or float
        Minimum document frequency for a term to be included. Default: 2.
    max_df : float
        Maximum document frequency (fraction of corpus). Default: 0.95.
    """

    def __init__(
        self,
        max_features: int = 50_000,
        ngram_range: Tuple[int, int] = (1, 2),
        sublinear_tf: bool = True,
        strip_accents: Optional[str] = "unicode",
        analyzer: str = "word",
        min_df: Union[int, float] = 2,
        max_df: float = 0.95,
    ) -> None:
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=sublinear_tf,
            strip_accents=strip_accents,
            analyzer=analyzer,
            min_df=min_df,
            max_df=max_df,
        )
        self._fitted = False
        logger.info(
            "TFIDFExtractor created | max_features=%d | ngram_range=%s",
            max_features, ngram_range,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def fit(self, texts: List[str]) -> "TFIDFExtractor":
        """
        Fit the vocabulary on training texts.

        Parameters
        ----------
        texts : list of str

        Returns
        -------
        self
        """
        logger.info("Fitting TF-IDF vocabulary on %d texts...", len(texts))
        self._vectorizer.fit(texts)
        self._fitted = True
        vocab_size = len(self._vectorizer.vocabulary_)
        logger.info("TF-IDF vocabulary fitted | vocab_size=%d", vocab_size)
        return self

    def transform(self, texts: List[str]) -> spmatrix:
        """
        Transform texts into a sparse TF-IDF matrix.

        Parameters
        ----------
        texts : list of str

        Returns
        -------
        scipy.sparse.spmatrix of shape (n_texts, vocab_size)
        """
        self._check_fitted()
        return self._vectorizer.transform(texts)

    def fit_transform(self, texts: List[str]) -> spmatrix:
        """Fit and transform in one step."""
        self.fit(texts)
        return self.transform(texts)

    @property
    def vocab_size(self) -> int:
        """Number of unique tokens in the fitted vocabulary."""
        self._check_fitted()
        return len(self._vectorizer.vocabulary_)

    @property
    def feature_names(self) -> List[str]:
        """List of feature names (tokens / n-grams)."""
        self._check_fitted()
        return self._vectorizer.get_feature_names_out().tolist()

    @property
    def vectorizer(self) -> TfidfVectorizer:
        """Direct access to the underlying TfidfVectorizer."""
        return self._vectorizer

    # ── Private ────────────────────────────────────────────────────────────

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "TFIDFExtractor has not been fitted yet. Call .fit(texts) first."
            )
