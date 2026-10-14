"""BM25 retriever implemented from scratch, so the project needs no search engine or embedding API.

BM25 score of a chunk for a query = sum over query terms of
    idf(term) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * chunk_len / avg_len))
k1 controls term-frequency saturation, b controls length normalisation.
"""
import math
import re
from collections import Counter
from dataclasses import dataclass

from rag.chunking import Chunk

_TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "of", "to", "in", "on", "for", "and",
    "or", "what", "how", "does", "do", "which", "with", "by", "it", "as", "at", "can", "i",
    "you", "your", "we", "our", "my", "me",
})


def tokenize(text: str) -> list[str]:
    """Lowercase, split on non-alphanumerics and drop stopwords."""
    return [token for token in _TOKEN.findall(text.lower()) if token not in STOPWORDS]


@dataclass(frozen=True)
class ScoredChunk:
    chunk: Chunk
    score: float


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self._chunks: list[Chunk] = []
        self._term_freqs: list[Counter] = []
        self._lengths: list[int] = []
        self._doc_freq: Counter = Counter()

    def __len__(self) -> int:
        return len(self._chunks)

    def add(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            tokens = tokenize(chunk.text)
            term_freq = Counter(tokens)
            self._chunks.append(chunk)
            self._term_freqs.append(term_freq)
            self._lengths.append(len(tokens))
            self._doc_freq.update(term_freq.keys())

    def idf(self, term: str) -> float:
        n = len(self._chunks)
        df = self._doc_freq.get(term, 0)
        # The +1 inside the log keeps idf positive even for very common terms.
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 3) -> list[ScoredChunk]:
        if not self._chunks:
            return []
        terms = tokenize(query)
        avg_len = (sum(self._lengths) / len(self._lengths)) or 1.0
        results: list[ScoredChunk] = []
        for chunk, term_freq, length in zip(self._chunks, self._term_freqs, self._lengths):
            score = 0.0
            for term in terms:
                tf = term_freq.get(term, 0)
                if tf == 0:
                    continue
                norm = tf + self.k1 * (1 - self.b + self.b * length / avg_len)
                score += self.idf(term) * tf * (self.k1 + 1) / norm
            if score > 0:
                results.append(ScoredChunk(chunk, score))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:k]
