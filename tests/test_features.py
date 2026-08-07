"""
test_features.py
================
Unit tests for TFIDFExtractor and EmbeddingExtractor.
"""

import numpy as np
import pytest
from scipy.sparse import issparse

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from teascoFT.features import TFIDFExtractor


SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog",
    "Machine learning is a subset of artificial intelligence",
    "Natural language processing enables text classification",
    "Deep learning models require large amounts of training data",
    "Knowledge distillation compresses large models into smaller ones",
]


class TestTFIDFExtractor:
    """Tests for TFIDFExtractor."""

    def test_fit_transform_shape(self):
        extractor = TFIDFExtractor(max_features=100)
        X = extractor.fit_transform(SAMPLE_TEXTS)
        assert X.shape[0] == len(SAMPLE_TEXTS)
        assert X.shape[1] <= 100

    def test_output_is_sparse(self):
        extractor = TFIDFExtractor()
        X = extractor.fit_transform(SAMPLE_TEXTS)
        assert issparse(X), "TF-IDF output should be a sparse matrix"

    def test_transform_without_fit_raises(self):
        extractor = TFIDFExtractor()
        with pytest.raises(RuntimeError, match="not been fitted"):
            extractor.transform(SAMPLE_TEXTS)

    def test_fit_then_transform(self):
        extractor = TFIDFExtractor(max_features=200)
        extractor.fit(SAMPLE_TEXTS)
        X = extractor.transform(SAMPLE_TEXTS[:2])
        assert X.shape[0] == 2

    def test_vocab_size(self):
        extractor = TFIDFExtractor(max_features=50)
        extractor.fit(SAMPLE_TEXTS)
        assert 0 < extractor.vocab_size <= 50

    def test_feature_names(self):
        extractor = TFIDFExtractor(max_features=50)
        extractor.fit(SAMPLE_TEXTS)
        names = extractor.feature_names
        assert isinstance(names, list)
        assert len(names) == extractor.vocab_size

    def test_values_nonnegative(self):
        extractor = TFIDFExtractor()
        X = extractor.fit_transform(SAMPLE_TEXTS)
        assert (X.data >= 0).all(), "TF-IDF values should be non-negative"

    def test_ngram_range(self):
        extractor_unigram = TFIDFExtractor(max_features=500, ngram_range=(1, 1))
        extractor_bigram  = TFIDFExtractor(max_features=500, ngram_range=(1, 2))
        extractor_unigram.fit(SAMPLE_TEXTS)
        extractor_bigram.fit(SAMPLE_TEXTS)
        # Bigrams should have a larger or equal vocabulary
        assert extractor_bigram.vocab_size >= extractor_unigram.vocab_size

    def test_different_test_texts(self):
        extractor = TFIDFExtractor(max_features=100)
        extractor.fit(SAMPLE_TEXTS)
        X_test = extractor.transform(["A completely new sentence about rockets."])
        assert X_test.shape == (1, extractor.vocab_size)
