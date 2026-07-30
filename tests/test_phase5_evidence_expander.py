import pytest
from app.retrieval.models import RetrievedChunk
from app.retrieval.evidence_expander import EvidenceExpander


def test_evidence_expansion():
    c1 = RetrievedChunk(
        paper_id="paper_a",
        chunk_id="paper_a_chunk_5",
        text="Target paragraph in section 2.",
        score=0.9,
        page_number=2,
        chunk_index=5,
        char_start=0,
        char_end=100,
    )

    c_neighbor = RetrievedChunk(
        paper_id="paper_a",
        chunk_id="paper_a_chunk_6",
        text="Neighboring paragraph in section 2.",
        score=0.7,
        page_number=2,
        chunk_index=6,
        char_start=101,
        char_end=200,
    )

    c_unrelated = RetrievedChunk(
        paper_id="paper_b",
        chunk_id="paper_b_chunk_1",
        text="Unrelated paper paragraph.",
        score=0.2,
        page_number=1,
        chunk_index=1,
        char_start=0,
        char_end=100,
    )

    expanded = EvidenceExpander.expand(
        retrieved_chunks=[c1],
        available_pool=[c1, c_neighbor, c_unrelated],
    )

    chunk_ids = [c.chunk_id for c in expanded]
    assert "paper_a_chunk_5" in chunk_ids
    assert "paper_a_chunk_6" in chunk_ids
    assert len(chunk_ids) == len(set(chunk_ids))  # Guarantee no duplicates
