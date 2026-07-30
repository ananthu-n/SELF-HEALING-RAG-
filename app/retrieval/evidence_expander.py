from __future__ import annotations

from typing import Sequence
from app.core.logger import logger
from app.retrieval.models import RetrievedChunk


class EvidenceExpander:
    """
    Expands retrieved evidence when context is insufficient.

    Strategies:
    1. Neighboring chunks (chunk_index - 1, chunk_index + 1)
    2. Same-section / same-page chunks
    3. Same-paper chunks
    4. Merges without duplicates
    """

    @classmethod
    def expand(
        cls,
        retrieved_chunks: Sequence[RetrievedChunk],
        available_pool: Sequence[RetrievedChunk] | None = None,
    ) -> list[RetrievedChunk]:
        """
        Merges retrieved chunks with neighboring and same-paper chunks from the available pool.
        """
        if not retrieved_chunks:
            return []

        logger.info(f"EvidenceExpander: Expanding context for {len(retrieved_chunks)} target chunks...")
        existing_ids = {c.chunk_id for c in retrieved_chunks}
        expanded: list[RetrievedChunk] = list(retrieved_chunks)

        if not available_pool:
            return expanded

        # Target paper IDs and neighbor indices
        target_neighbors = {(c.paper_id, c.chunk_index - 1) for c in retrieved_chunks}
        target_neighbors.update({(c.paper_id, c.chunk_index + 1) for c in retrieved_chunks})
        target_papers = {c.paper_id for c in retrieved_chunks}

        added_count = 0
        # 1. Neighboring chunk expansion
        for candidate in available_pool:
            if candidate.chunk_id in existing_ids:
                continue

            # Neighbor check
            if (candidate.paper_id, candidate.chunk_index) in target_neighbors:
                expanded.append(candidate)
                existing_ids.add(candidate.chunk_id)
                added_count += 1

        # 2. Same-paper fallback expansion
        for candidate in available_pool:
            if candidate.chunk_id in existing_ids:
                continue

            if candidate.paper_id in target_papers and added_count < 10:
                expanded.append(candidate)
                existing_ids.add(candidate.chunk_id)
                added_count += 1

        logger.success(f"EvidenceExpander: Added {added_count} expanded neighbor/same-paper chunks (Total: {len(expanded)}).")
        return expanded
