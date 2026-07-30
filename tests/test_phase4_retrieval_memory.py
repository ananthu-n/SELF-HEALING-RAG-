import pytest
from app.self_healing.retrieval_memory import RetrievalMemory


def test_retrieval_memory_tracking():
    mem = RetrievalMemory()

    mem.record_attempt(
        strategy="hybrid",
        query="GraphRAG definition",
        paper_ids=["2401.15884v3", "2601.05264v1"],
        chunk_ids=["chunk_1", "chunk_2"],
    )

    assert "2401.15884v3" in mem.seen_paper_ids
    assert "chunk_1" in mem.seen_chunk_ids
    assert mem.is_chunk_seen("chunk_1")
    assert not mem.is_chunk_seen("chunk_99")
