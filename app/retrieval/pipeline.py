from __future__ import annotations

from app.retrieval.models import RetrievedChunk
from app.retrieval.processors.duplicate_filter import DuplicateFilter
from app.retrieval.processors.noise_filter import NoiseFilter
from app.retrieval.processors.score_filter import ScoreFilter
from app.retrieval.processors.topk_filter import TopKFilter


class RetrievalPipeline:
    """
    Sequential retrieval processing pipeline.
    """

    def __init__(
        self,
        top_k: int,
        minimum_score: float = 0.55,
    ) -> None:

        self.processors = [
            NoiseFilter(),
            DuplicateFilter(),
            ScoreFilter(minimum_score),
            TopKFilter(top_k),
        ]

    def process(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:

        for processor in self.processors:
            chunks = processor.process(chunks)

        return chunks
