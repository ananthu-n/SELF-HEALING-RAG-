import pytest
from app.self_healing.query_rewriter import QueryRewriter
from app.pipeline.intent import QueryIntent


def test_query_rewriter_no_keyword_stuffing():
    rewriter = QueryRewriter()

    # Definition query rewrite test
    query = "What is GraphRAG?"
    rewritten = rewriter.rewrite(query=query, reason="IRRELEVANT_RETRIEVAL", intent=QueryIntent.DEFINITION)

    # Must be clean, no comma-separated list dumps
    assert isinstance(rewritten, str)
    assert len(rewritten) > 0
    assert "," not in rewritten or len(rewritten.split(",")) <= 2
