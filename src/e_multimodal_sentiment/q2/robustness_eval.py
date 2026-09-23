"""Q2 experiment orchestration placeholder; metrics remain shared."""
from collections.abc import Iterable
from typing import Any
from e_multimodal_sentiment.training.trainer import Trainer


def evaluate_robustness(trainer: Trainer, batches: Iterable[dict[str, Any]], *, seed: int = 0) -> dict[str, float]:
    """Await confirmed missingness scenarios and aggregation protocol."""
    raise NotImplementedError("Register an experiment and confirm robustness protocol first.")
