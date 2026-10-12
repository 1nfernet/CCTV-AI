"""
faiss_index.py – FAISS vector index wrapper: add, search, save, load.
"""

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("cctv_assistant")


class FaissIndex:
    """
    Manages a FAISS flat-L2 index together with a parallel metadata list
    so each vector can be mapped back to its originating event.

    Parameters
    ----------
    dimension : int
        Embedding dimensionality (e.g. 384 for ``all-MiniLM-L6-v2``).
    index_path : str | Path
        File path for persisting the FAISS index.
    metadata_path : str | Path
        File path for persisting the metadata list.
    """

    def __init__(
        self,
        dimension: int = 384,
        index_path: str | Path = "vector_store/faiss_store.index",
        metadata_path: str | Path = "vector_store/event_metadata.pkl",
    ):
        self.dimension = dimension
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index: Any = None
        self.metadata: list[dict] = []
        self._faiss: Any = None

    # ── lazy import ────────────────────────────────────────────────────
    def _import_faiss(self):
        if self._faiss is None:
            import faiss
            self._faiss = faiss
        return self._faiss

    # ── lifecycle ──────────────────────────────────────────────────────
    def build(self) -> None:
        """Create a fresh in-memory FAISS index."""
        faiss = self._import_faiss()
        self.index = faiss.IndexFlatL2(self.dimension)
        self.metadata = []
        logger.info("Created new FAISS index  dim=%d", self.dimension)

    def load(self) -> bool:
        """Load a persisted index + metadata.  Returns ``True`` on success."""
        faiss = self._import_faiss()
        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            with open(self.metadata_path, "rb") as fh:
                self.metadata = pickle.load(fh)
            logger.info(
                "Loaded FAISS index  vectors=%d", self.index.ntotal
            )
            return True
        logger.info("No existing FAISS index found – will create a new one")
        self.build()
        return False

    def save(self) -> None:
        """Persist the index and metadata to disk."""
        faiss = self._import_faiss()
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        with open(self.metadata_path, "wb") as fh:
            pickle.dump(self.metadata, fh)
        logger.debug("Saved FAISS index  vectors=%d", self.index.ntotal)

    # ── read / write ───────────────────────────────────────────────────
    def add(self, vector: np.ndarray, metadata: dict) -> None:
        """
        Add a single vector (shape ``(dim,)`` or ``(1, dim)``) with its metadata.
        """
        if self.index is None:
            self.build()
        vec = np.atleast_2d(vector).astype("float32")
        self.index.add(vec)
        self.metadata.append(metadata)

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Return the *top_k* nearest metadata dicts for a query vector.

        Each result dict is augmented with a ``"score"`` key (L2 distance).
        """
        if self.index is None or self.index.ntotal == 0:
            return []

        vec = np.atleast_2d(query_vector).astype("float32")
        distances, indices = self.index.search(vec, min(top_k, self.index.ntotal))

        results: list[dict] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            entry = dict(self.metadata[idx])
            entry["score"] = float(dist)
            results.append(entry)
        return results

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index else 0
