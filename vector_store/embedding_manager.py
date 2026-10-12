"""
embedding_manager.py – Sentence-transformer encoder for event text.
"""

import logging
from typing import Any

import numpy as np

logger = logging.getLogger("cctv_assistant")


class EmbeddingManager:
    """
    Thin wrapper around a ``sentence-transformers`` model that encodes
    text strings into dense vectors.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (default ``"all-MiniLM-L6-v2"``).
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model: Any = None
        self.dimension: int = 384               # default for MiniLM-L6-v2

    def load_model(self) -> None:
        """Lazy-load the sentence-transformer model."""
        from sentence_transformers import SentenceTransformer

        logger.info("Loading embedding model: %s", self.model_name)
        self.model = SentenceTransformer(self.model_name)
        # Infer the actual dimension from the model
        self.dimension = self.model.get_sentence_embedding_dimension()
        logger.info(
            "Embedding model loaded  dim=%d", self.dimension
        )

    def encode(self, text: str | list[str]) -> np.ndarray:
        """
        Encode one or more strings.

        Parameters
        ----------
        text : str or list[str]
            Input text(s).

        Returns
        -------
        np.ndarray
            Shape ``(dim,)`` for a single string, ``(n, dim)`` for a list.
        """
        if self.model is None:
            self.load_model()

        single = isinstance(text, str)
        if single:
            text = [text]

        embeddings = self.model.encode(text, show_progress_bar=False)
        return embeddings[0] if single else embeddings
