from __future__ import annotations

from pydantic import BaseModel, Field
from app.core.logger import logger


class RetrievalMemory(BaseModel):
    """
    Tracks previous retrieval attempts to prevent repeated retrieval of identical evidence.
    """

    seen_paper_ids: set[str] = Field(default_factory=set)
    seen_chunk_ids: set[str] = Field(default_factory=set)
    used_strategies: list[str] = Field(default_factory=list)
    rewritten_queries: list[str] = Field(default_factory=list)

    def record_attempt(
        self,
        strategy: str,
        query: str,
        paper_ids: list[str],
        chunk_ids: list[str],
    ) -> None:
        self.used_strategies.append(strategy)
        self.rewritten_queries.append(query)
        prev_chunk_count = len(self.seen_chunk_ids)

        self.seen_paper_ids.update(paper_ids)
        self.seen_chunk_ids.update(chunk_ids)

        new_chunks = len(self.seen_chunk_ids) - prev_chunk_count
        logger.info(
            f"RetrievalMemory updated: {len(paper_ids)} paper IDs, {len(chunk_ids)} chunk IDs recorded. "
            f"({new_chunks} new unique chunks added)."
        )

    def is_chunk_seen(self, chunk_id: str) -> bool:
        return chunk_id in self.seen_chunk_ids
