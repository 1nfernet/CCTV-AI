"""
response_generator.py – LLM-based natural language answer generation.

Uses OpenAI (or a local fallback) to convert retrieved event records
into conversational responses.
"""

import os
import logging

logger = logging.getLogger("cctv_assistant")


class ResponseGenerator:
    """
    Takes a user question and a list of relevant event summaries,
    then uses an LLM to produce a natural-language answer.

    Parameters
    ----------
    provider : str
        ``"openai"`` or ``"local"`` (local = rule-based fallback).
    model : str
        Chat model name (e.g. ``"gpt-3.5-turbo"``).
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum response tokens.
    """

    SYSTEM_PROMPT = (
        "You are a smart home surveillance assistant. "
        "You receive a list of detected events (with timestamps) from CCTV cameras. "
        "Answer the user's question conversationally, referencing specific times and people. "
        "If no relevant events were found, say so politely. "
        "Keep your answer concise (1-3 sentences)."
    )

    def __init__(
        self,
        provider: str = "openai",
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.3,
        max_tokens: int = 300,
    ):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    # ── public API ─────────────────────────────────────────────────────
    def generate(self, question: str, event_summaries: list[str]) -> str:
        """
        Generate a natural-language answer.

        Parameters
        ----------
        question : str
            The user's query.
        event_summaries : list[str]
            Matching event summaries retrieved from the semantic search.

        Returns
        -------
        str   The assistant's response.
        """
        if self.provider == "openai":
            return self._generate_openai(question, event_summaries)
        return self._generate_local(question, event_summaries)

    # ── OpenAI backend ─────────────────────────────────────────────────
    def _generate_openai(self, question: str, summaries: list[str]) -> str:
        try:
            from openai import OpenAI

            api_key = os.getenv("OPENAI_API_KEY", "")
            if not api_key:
                logger.warning("OPENAI_API_KEY not set – falling back to local mode")
                return self._generate_local(question, summaries)

            client = OpenAI(api_key=api_key)

            events_block = "\n".join(f"- {s}" for s in summaries) if summaries else "No events found."

            user_msg = (
                f"User question: {question}\n\n"
                f"Detected events:\n{events_block}"
            )

            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": user_msg},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content.strip()

        except Exception as exc:
            logger.error("OpenAI call failed: %s – using local fallback", exc)
            return self._generate_local(question, summaries)

    # ── Local fallback ─────────────────────────────────────────────────
    @staticmethod
    def _generate_local(question: str, summaries: list[str]) -> str:
        """
        Simple rule-based fallback when no LLM API is available.
        Returns the most relevant event summaries directly.
        """
        if not summaries:
            return "I don't have any recorded events matching your question."

        if len(summaries) == 1:
            return f"Yes. {summaries[0]}"

        intro = "Here's what I found:\n"
        body = "\n".join(f"• {s}" for s in summaries)
        return intro + body
