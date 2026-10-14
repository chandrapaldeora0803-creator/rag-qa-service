"""LLM clients.

MockLLM keeps tests, CI and the evaluation step fully offline and deterministic.
OpenAICompatibleLLM talks to any /chat/completions API (OpenAI, Azure OpenAI, vLLM, Ollama, ...).
"""
import os
import re
from typing import Protocol

import httpx

from rag.retriever import tokenize

NO_ANSWER = "I don't know based on the provided documents."


class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class MockLLM:
    """Picks the context sentence with the largest word overlap with the question and cites its source.

    It is not a real model, but it follows the same contract (answer + [chunk_id] citation, or NO_ANSWER),
    which is enough to test the pipeline and the evaluation metrics end to end.
    """

    def complete(self, prompt: str) -> str:
        question_words = set(tokenize(_section(prompt, "Question:", "Answer:")))
        best_sentence, best_source, best_overlap = "", "", 0
        for line in _section(prompt, "Context:", "Question:").splitlines():
            match = re.match(r"\[(?P<source>[^\]]+)\]\s*(?P<text>.*)", line.strip())
            if not match:
                continue
            for sentence in re.split(r"(?<=[.!?])\s+", match.group("text")):
                overlap = len(question_words & set(tokenize(sentence)))
                if overlap > best_overlap:
                    best_sentence, best_source, best_overlap = sentence.strip(), match.group("source"), overlap
        if not best_sentence:
            return NO_ANSWER
        return f"{best_sentence} [{best_source}]"


class OpenAICompatibleLLM:
    def __init__(self, base_url: str, api_key: str, model: str, timeout: float = 30.0):
        self.model = model
        self._client = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            headers={"Authorization": f"Bearer {api_key}"},
        )

    def complete(self, prompt: str) -> str:
        response = self._client.post("/chat/completions", json={
            "model": self.model,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        })
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()


def llm_from_env() -> LLMClient:
    """Use a real model when LLM_API_KEY is set, otherwise fall back to the offline mock."""
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        return MockLLM()
    return OpenAICompatibleLLM(
        base_url=os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
        api_key=api_key,
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
    )


def _section(prompt: str, start: str, end: str) -> str:
    _, _, rest = prompt.partition(start)
    section, _, _ = rest.partition(end)
    return section.strip()
