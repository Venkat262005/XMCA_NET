"""Baseline hallucination detection methods for comparative evaluation.

Status: PENDING TASK AUTHORIZATION
Baseline methods (e.g. logit-entropy, visual contrastive decoding, attention probing)
will be implemented without placeholder shortcuts or hardcoded outputs.
"""


class BaselineEvaluator:
    """Base interface for baseline comparison methods."""

    def __init__(self, baseline_name: str):
        self.baseline_name = baseline_name

    def score(self, *args, **kwargs) -> float:
        raise NotImplementedError(
            f"Baseline '{self.baseline_name}' is pending task implementation. "
            "Dummy scores and hardcoded outputs are strictly prohibited."
        )

