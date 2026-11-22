"""
Sentence embedders producing the 384 embedding columns of the feature matrix.

``SentenceEmbedder`` is the real one (all-MiniLM-L6-v2, runs locally, ~90 MB
download on first use). ``HashingEmbedder`` needs nothing downloaded and is
used by the offline demo and the tests.
"""

from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 384


class SentenceEmbedder:
    """all-MiniLM-L6-v2 sentence embeddings, returned as float32 numpy."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2", batch_size: int = 64,
                 device: str = "auto"):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required: pip install sentence-transformers"
            ) from e
        device = None if self.device == "auto" else self.device
        logger.info("Loading sentence encoder %s", self.model_name)
        self._model = SentenceTransformer(self.model_name, device=device)
        return self._model

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM

    def embed(self, texts: List[str]) -> np.ndarray:
        vectors = self._load().encode(
            list(texts), batch_size=self.batch_size,
            show_progress_bar=False, convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)


class HashingEmbedder:
    """Deterministic 384-dim bag-of-words hashing — no model download.

    Much weaker than MiniLM; it exists so the pipeline can be demonstrated and
    tested offline.
    """

    model_name = "hashing-384"

    def __init__(self):
        from sklearn.feature_extraction.text import HashingVectorizer

        self._vectorizer = HashingVectorizer(
            n_features=EMBEDDING_DIM, ngram_range=(1, 2), alternate_sign=False,
            norm="l2", lowercase=True,
        )

    @property
    def embedding_dim(self) -> int:
        return EMBEDDING_DIM

    def embed(self, texts: List[str]) -> np.ndarray:
        return self._vectorizer.transform(list(texts)).toarray().astype(np.float32)
