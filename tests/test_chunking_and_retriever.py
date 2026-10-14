import pytest

from rag.chunking import Chunk, chunk_text
from rag.retriever import BM25Retriever, tokenize


# ---------- chunking ----------

def test_short_text_is_a_single_chunk():
    chunks = chunk_text("doc", "one two three", size=10, overlap=2)
    assert [c.chunk_id for c in chunks] == ["doc#0"]
    assert chunks[0].text == "one two three"


def test_chunks_overlap_and_cover_every_word():
    words = [f"w{i}" for i in range(25)]
    chunks = chunk_text("doc", " ".join(words), size=10, overlap=3)
    assert len(chunks) == 4
    assert chunks[0].text.split()[-3:] == chunks[1].text.split()[:3]
    assert chunks[-1].text.split()[-1] == "w24"


def test_empty_text_gives_no_chunks():
    assert chunk_text("doc", "   ") == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (5, 5), (5, -1)])
def test_invalid_parameters_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("doc", "some text", size=size, overlap=overlap)


# ---------- retriever ----------

def _retriever() -> BM25Retriever:
    retriever = BM25Retriever()
    retriever.add([
        Chunk("refunds#0", "refunds", "Refunds are processed within 30 days of purchase."),
        Chunk("shipping#0", "shipping", "Express shipping delivers within one business day."),
        Chunk("security#0", "security", "Passwords must be at least 12 characters long."),
    ])
    return retriever


def test_tokenize_lowercases_and_drops_stopwords():
    assert tokenize("What is the Refund policy?") == ["refund", "policy"]


def test_most_relevant_chunk_ranks_first():
    results = _retriever().search("how fast is express shipping")
    assert results[0].chunk.doc_id == "shipping"


def test_no_match_returns_empty_list():
    assert _retriever().search("student discount") == []


def test_empty_index_returns_empty_list():
    assert BM25Retriever().search("anything") == []


def test_top_k_limits_results():
    assert len(_retriever().search("within", k=1)) == 1


def test_rare_terms_have_higher_idf_than_common_terms():
    retriever = _retriever()
    assert retriever.idf("passwords") > retriever.idf("within")
