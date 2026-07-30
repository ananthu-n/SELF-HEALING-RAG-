from __future__ import annotations

from app.core.logger import logger

from app.evaluation.decision import (
    DecisionAction,
    RetryDecision,
)

from app.evaluation.failure_models import (
    FailureAnalysis,
    FailureType,
)

from app.self_healing.strategy_registry import get_failure_strategy


class DecisionEngine:
    """
    Converts a FailureAnalysis into a retry decision.

    The FailureAnalyzer determines WHY the answer failed.

    The DecisionEngine determines WHAT to do next.
    """

    def decide(
        self,
        failure: FailureAnalysis,
    ) -> RetryDecision:

        logger.info(
            f"Decision Engine received: "
            f"{failure.failure_type.value}"
        )

        strategy = get_failure_strategy(failure.failure_type)

        if failure.failure_type == FailureType.NONE:

            logger.success("Grounded answer accepted.")

        return RetryDecision(

            should_retry=strategy.should_retry,

            action=strategy.decision_action,

            reason=strategy.decision_reason,

        )
