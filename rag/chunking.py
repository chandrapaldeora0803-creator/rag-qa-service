"""Split documents into overlapping word-window chunks.

Overlap keeps a fact that sits on a chunk boundary retrievable from at least one chunk.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<doc_id>#<index>", used as the citation id
    doc_id: str
    text: str


def chunk_text(doc_id: str, text: str, size: int = 120, overlap: int = 30) -> list[Chunk]:
    """Split text into chunks of `size` words; each chunk shares `overlap` words with the previous one."""
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be >= 0 and smaller than size")

    words = text.split()
    if not words:
        return []

    step = size - overlap
    chunks: list[Chunk] = []
    for index, start in enumerate(range(0, len(words), step)):
        window = words[start:start + size]
        chunks.append(Chunk(chunk_id=f"{doc_id}#{index}", doc_id=doc_id, text=" ".join(window)))
        if start + size >= len(words):
            break
    return chunks
