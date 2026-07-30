from __future__ import annotations

from abc import ABC
from abc import abstractmethod

from app.retrieval.models import RetrievalResult


class BaseRetriever(ABC):
    """
    Abstract base class for all retrieval strategies.

    Every retriever must return a RetrievalResult,
    regardless of the retrieval algorithm.

    Examples
    --------
    - DenseRetriever
    - BM25Retriever
    - HybridRetriever
    - HyDERetriever (future)
    """

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int,
    ) -> RetrievalResult:
        """
        Retrieve relevant document chunks.

        Parameters
        ----------
        query:
            User query.

        top_k:
            Number of chunks to retrieve.

        Returns
        -------
        RetrievalResult
        """
        raise NotImplementedError