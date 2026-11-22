"""
features.py — Shared 385-dim Feature Construction with Content-Hash Caching
============================================================================

Every adapter (MLP, XGBoost, CatBoost, LightGBM) consumes the *same* input
representation defined by the V2 specification:

    [ sentence embedding (384) | frozen-LLM confidence scalar (1) ] = 385 dims

Building that representation is by far the most expensive part of a run: it
requires encoding every text with all-MiniLM-L6-v2 and querying the frozen LLM
once per text. Previously this work happened inside ``CascadeTrainerV2._prepare_data``,
so it was repeated for every trainer instance — in a benchmark over 4 adapters x
5 folds that is 20x redundant work on identical inputs.

``FeatureBuilder`` computes the matrix once and caches it on disk under a key
derived from the *content* of the inputs (encoder name + LLM name + the texts
themselves). That makes the cache safe to reuse: changing the texts, the encoder,
or the LLM changes the key, so a stale cache can never silently poison a run —
which is a real failure mode of the older shape-only cache check.
"""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _to_numpy(values) -> np.ndarray:
    """float32 array from a numpy array, list, or torch tensor (duck-typed, so
    this module never imports torch itself)."""
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float32)

EMBEDDING_DIM = 384
FEATURE_DIM = 385  # 384 embedding + 1 LLM confidence scalar


class FeatureBuilder:
    """
    Build (and cache) the 385-dim input matrix shared by all adapters.

    Parameters
    ----------
    embedder : object
        Any object exposing ``embed(texts) -> array(N, 384)``. Use
        :class:`~teasco_tree.features.embeddings.SentenceEmbedder` for real
        MiniLM embeddings, or ``MockEmbedder`` for tests.
    llm : object or None
        Frozen LLM exposing ``predict(texts) -> (List[str], array(N,))``.
        Supplies the confidence scalar in column 384. If ``None``, that column
        is filled with zeros (documented no-signal mode).
    cache_dir : str or Path, optional
        Directory for the ``.npz`` feature cache. ``None`` disables caching.
    verbose : bool
        Log progress. Default: True.

    Examples
    --------
    >>> builder = FeatureBuilder(embedder, llm, cache_dir=".cache")
    >>> X = builder.build(texts)          # (N, 385) float32, cached
    >>> y = builder.encode_labels(labels, classes=["FR", "NFR"])
    """

    def __init__(
        self,
        embedder,
        llm=None,
        cache_dir: Optional[str] = None,
        verbose: bool = True,
    ):
        self.embedder = embedder
        self.llm = llm
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.verbose = verbose

        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────

    @property
    def feature_dim(self) -> int:
        """Dimensionality of the built feature matrix (385)."""
        return FEATURE_DIM

    def build(self, texts: Sequence[str], progress_callback=None) -> np.ndarray:
        """
        Build the 385-dim feature matrix for ``texts``.

        Returns
        -------
        np.ndarray, shape (N, 385), dtype float32
            Columns 0..383 are the sentence embedding, column 384 is the
            frozen-LLM confidence scalar.

        ``progress_callback(done, total)`` is forwarded to the LLM, whose
        per-text pass is the slow part of building features.
        """
        texts = list(texts)

        cache_path = self._cache_path(texts)
        if cache_path is not None and cache_path.exists():
            X = np.load(cache_path)["X"]
            if X.shape == (len(texts), FEATURE_DIM):
                if self.verbose:
                    logger.info("Feature cache hit: %s (%s)", cache_path.name, X.shape)
                return X
            # Hash collision or a truncated write — fall through and rebuild.
            logger.warning("Cache shape mismatch at %s; rebuilding.", cache_path)

        t0 = time.time()

        # ── Sentence embeddings (384-dim) ──────────────────────────────────
        if self.verbose:
            logger.info("Embedding %d texts...", len(texts))
        embeddings = self.embedder.embed(texts)
        embeddings = _to_numpy(embeddings)

        if embeddings.shape != (len(texts), EMBEDDING_DIM):
            raise ValueError(
                f"Embedder returned {embeddings.shape}, expected "
                f"({len(texts)}, {EMBEDDING_DIM})."
            )

        # ── Frozen-LLM confidence scalar (1-dim) ───────────────────────────
        if self.llm is not None:
            if self.verbose:
                logger.info("Querying frozen LLM for confidence scalars...")
            if progress_callback is not None:
                _, confidences = self.llm.predict(
                    texts, progress_callback=progress_callback
                )
            else:
                _, confidences = self.llm.predict(texts)
            confidences = _to_numpy(confidences).reshape(-1, 1)
            if confidences.shape[0] != len(texts):
                raise ValueError(
                    f"LLM returned {confidences.shape[0]} confidences for "
                    f"{len(texts)} texts."
                )
        else:
            confidences = np.zeros((len(texts), 1), dtype=np.float32)

        X = np.hstack([embeddings, confidences]).astype(np.float32)

        if self.verbose:
            logger.info(
                "Built features %s in %.1fs", X.shape, time.time() - t0
            )

        if cache_path is not None:
            np.savez_compressed(cache_path, X=X)
            if self.verbose:
                logger.info("Cached features to %s", cache_path)

        return X

    @staticmethod
    def encode_labels(labels: Sequence[str], classes: List[str]) -> np.ndarray:
        """
        Map string labels to integer class indices.

        Raises
        ------
        ValueError
            If a label is not present in ``classes`` — surfacing the problem
            here is far easier to debug than a downstream KeyError.
        """
        lookup = {c: i for i, c in enumerate(classes)}
        unknown = sorted(set(labels) - set(lookup))
        if unknown:
            raise ValueError(
                f"Labels {unknown} are not in classes={classes}."
            )
        return np.asarray([lookup[l] for l in labels], dtype=np.int64)

    # ── Private ────────────────────────────────────────────────────────────

    def _cache_path(self, texts: List[str]) -> Optional[Path]:
        """Content-addressed cache path, or None when caching is disabled."""
        if self.cache_dir is None:
            return None

        embedder_id = getattr(self.embedder, "model_name", type(self.embedder).__name__)
        llm_id = getattr(self.llm, "model_name", type(self.llm).__name__) if self.llm else "none"

        digest = hashlib.sha256()
        digest.update(str(embedder_id).encode())
        digest.update(b"\x00")
        digest.update(str(llm_id).encode())
        digest.update(b"\x00")
        for t in texts:
            digest.update(t.encode("utf-8", errors="replace"))
            digest.update(b"\x00")

        return self.cache_dir / f"feat_{digest.hexdigest()[:16]}.npz"
