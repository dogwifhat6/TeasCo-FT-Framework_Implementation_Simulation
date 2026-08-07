"""
embedding_extractor.py
======================
Sentence-embedding feature extractor using a pretrained MiniLM model.
Produces dense 384-dimensional vectors that capture semantic meaning.

Requires the optional `sentence-transformers` package:
    pip install sentence-transformers

Usage
-----
    extractor = EmbeddingExtractor(model_name="all-MiniLM-L6-v2")
    extractor.fit(train_texts)   # no-op — embeddings are frozen/pretrained

    X_train = extractor.transform(train_texts)   # dense numpy array
    X_test  = extractor.transform(test_texts)

Disk Caching
------------
Pass `cache_dir` to cache embeddings so that repeated .transform() calls
on the same text list don't re-encode. Embeddings are stored as .npy files
keyed by a hash of the input texts.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """
    Dense sentence-embedding extractor using a SentenceTransformer model.

    Parameters
    ----------
    model_name : str
        Any model from https://huggingface.co/sentence-transformers.
        Default: "all-MiniLM-L6-v2" (~80MB, 384-dim, excellent quality/speed).
    batch_size : int
        Number of texts per encoding batch. Default: 64.
    device : str, optional
        "cpu", "cuda", or "mps". If None, auto-detected.
    cache_dir : str or Path, optional
        If provided, computed embeddings are cached on disk to avoid
        redundant computation.
    normalize : bool
        Whether to L2-normalize embedding vectors. Default: True.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 64,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        normalize: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self.normalize = normalize
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._model = None  # lazy-loaded on first use
        self._fitted = True  # embeddings are frozen; fit() is a no-op

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "EmbeddingExtractor created | model=%s | cache=%s",
            model_name, cache_dir,
        )

    # ── Public API ─────────────────────────────────────────────────────────

    def fit(self, texts: List[str]) -> "EmbeddingExtractor":
        """
        No-op: pretrained embedding models don't need fitting.
        Provided for API compatibility with TFIDFExtractor.
        """
        logger.info(
            "EmbeddingExtractor.fit() called — model is pretrained, no fitting needed."
        )
        self._load_model()
        return self

    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Encode texts into dense embedding vectors.

        Parameters
        ----------
        texts : list of str

        Returns
        -------
        np.ndarray of shape (n_texts, embedding_dim)
        """
        if self.cache_dir:
            return self._transform_with_cache(texts)
        return self._encode(texts)

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """Fit (no-op) and transform in one step."""
        self.fit(texts)
        return self.transform(texts)

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the embedding vectors."""
        self._load_model()
        return self._model.get_sentence_embedding_dimension()

    # ── Private ────────────────────────────────────────────────────────────

    def _load_model(self) -> None:
        """Lazily load the SentenceTransformer model."""
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for EmbeddingExtractor. "
                "Install it with: pip install sentence-transformers"
            ) from e

        logger.info("Loading SentenceTransformer model: %s", self.model_name)
        self._model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(
            "Model loaded | embedding_dim=%d",
            self._model.get_sentence_embedding_dimension(),
        )

    def _encode(self, texts: List[str]) -> np.ndarray:
        """Run the encoding."""
        self._load_model()
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )
        return embeddings

    def _transform_with_cache(self, texts: List[str]) -> np.ndarray:
        """Check disk cache before encoding."""
        cache_key = hashlib.md5(
            (self.model_name + "".join(texts)).encode()
        ).hexdigest()
        cache_path = self.cache_dir / f"{cache_key}.npy"

        if cache_path.exists():
            logger.info("Cache hit — loading embeddings from %s", cache_path)
            return np.load(str(cache_path))

        embeddings = self._encode(texts)
        np.save(str(cache_path), embeddings)
        logger.info("Embeddings cached to %s", cache_path)
        return embeddings
