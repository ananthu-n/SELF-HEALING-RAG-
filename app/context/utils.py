from __future__ import annotations

import math


class TokenEstimator:
    """
    Lightweight token estimator.

    Used only for budgeting before prompt construction.

    Approximation:
        1 token ≈ 4 characters
    """

    CHARS_PER_TOKEN = 4

    @classmethod
    def estimate(cls, text: str) -> int:

        if not text:
            return 0

        return math.ceil(
            len(text) / cls.CHARS_PER_TOKEN
        )

    @classmethod
    def estimate_many(
        cls,
        texts: list[str],
    ) -> int:

        return sum(
            cls.estimate(text)
            for text in texts
        )