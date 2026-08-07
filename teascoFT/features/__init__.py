"""
Features module — converts raw text into numerical vectors for student models.

Two extractors are provided:
- TFIDFExtractor  : Lightweight, fully offline, zero extra dependencies.
- EmbeddingExtractor : MiniLM sentence embeddings (requires sentence-transformers).
"""

from teascoFT.features.tfidf_extractor import TFIDFExtractor
from teascoFT.features.embedding_extractor import EmbeddingExtractor

__all__ = ["TFIDFExtractor", "EmbeddingExtractor"]
