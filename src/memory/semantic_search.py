"""
semantic_search.py – Query the FAISS vector store with natural language.
"""

import logging

logger = logging.getLogger("cctv_assistant")


class SemanticSearch:
    """
    Accepts a natural-language query, converts it to an embedding, and
    retrieves the most relevant event summaries from the FAISS index.

    Parameters
    ----------
    embedding_manager
        ``vector_store.embedding_manager.EmbeddingManager`` instance.
    faiss_index
        ``vector_store.faiss_index.FaissIndex`` instance.
    """

    def __init__(self, embedding_manager, faiss_index):
        self.emb = embedding_manager
        self.idx = faiss_index

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Search for events semantically similar to ``query``.

        Returns
        -------
        list[dict]
            Metadata dicts, each augmented with a ``"score"`` key (L2 distance).
        """
        if self.idx.size == 0:
            logger.info("FAISS index is empty – no results")
            return []

        vector = self.emb.encode(query)
        results = self.idx.search(vector, top_k=top_k)
        logger.debug("Semantic search  query=%r  hits=%d", query, len(results))
        return results

    def search_summaries(self, query: str, top_k: int = 5) -> list[str]:
        """Convenience: return just the summary strings."""
        results = self.search(query, top_k=top_k)
        return [r.get("summary", "") for r in results if r.get("summary")]
