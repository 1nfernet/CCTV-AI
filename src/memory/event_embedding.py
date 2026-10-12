"""
event_embedding.py – Convert event summaries to embeddings and store in FAISS.
"""

import logging

logger = logging.getLogger("cctv_assistant")


class EventEmbedder:
    """
    Bridges the event pipeline with the vector store.

    * Receives an event summary string.
    * Encodes it via ``EmbeddingManager``.
    * Stores the resulting vector + metadata in the ``FaissIndex``.
    * Periodically persists the index to disk.

    Parameters
    ----------
    embedding_manager
        Instance of ``vector_store.embedding_manager.EmbeddingManager``.
    faiss_index
        Instance of ``vector_store.faiss_index.FaissIndex``.
    auto_save_interval : int
        Persist the index to disk every *N* insertions.
    """

    def __init__(self, embedding_manager, faiss_index, auto_save_interval: int = 10):
        self.emb = embedding_manager
        self.idx = faiss_index
        self.auto_save_interval = auto_save_interval
        self._inserts_since_save = 0

    def embed_and_store(self, summary: str, metadata: dict | None = None) -> None:
        """
        Encode ``summary``, add the vector to the FAISS index, and auto-save
        when the interval threshold is reached.
        """
        vector = self.emb.encode(summary)
        meta = metadata.copy() if metadata else {}
        meta.setdefault("summary", summary)

        self.idx.add(vector, meta)
        self._inserts_since_save += 1

        if self._inserts_since_save >= self.auto_save_interval:
            self.idx.save()
            self._inserts_since_save = 0
            logger.debug("Auto-saved FAISS index  total=%d", self.idx.size)

    def flush(self) -> None:
        """Force-save the index to disk."""
        self.idx.save()
        self._inserts_since_save = 0
