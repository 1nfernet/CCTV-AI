"""
query_engine.py – Orchestrates semantic search + DB lookup for user queries.
"""

import logging

from src.memory.semantic_search import SemanticSearch
from src.chatbot.response_generator import ResponseGenerator

logger = logging.getLogger("cctv_assistant")


class QueryEngine:
    """
    Receives a natural-language question from the Telegram bot,
    retrieves matching events, and generates a conversational answer.

    Parameters
    ----------
    semantic_search : SemanticSearch
    response_generator : ResponseGenerator
    db_manager : optional
        ``database.db_manager.DatabaseManager`` for supplementary SQL queries.
    """

    def __init__(
        self,
        semantic_search: SemanticSearch,
        response_generator: ResponseGenerator,
        db_manager=None,
    ):
        self.search = semantic_search
        self.responder = response_generator
        self.db = db_manager

    def answer(self, question: str) -> str:
        """
        End-to-end: question → semantic search → LLM → answer string.
        """
        logger.info("Query received: %s", question)

        # 1. Semantic search over event embeddings
        summaries = self.search.search_summaries(question, top_k=10)

        # 2. Supplement with keyword-based DB search if we have few hits
        if self.db and len(summaries) < 3:
            keywords = question.lower().split()
            for kw in keywords:
                if len(kw) > 3:
                    db_results = self.db.search_events(kw, limit=5)
                    for row in db_results:
                        s = row.get("summary", "")
                        if s and s not in summaries:
                            summaries.append(s)

        # 3. Generate natural-language response
        answer = self.responder.generate(question, summaries[:10])
        logger.info("Answer: %s", answer[:120])
        return answer
