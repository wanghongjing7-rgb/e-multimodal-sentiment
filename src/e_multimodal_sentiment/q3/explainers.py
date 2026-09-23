"""Q3 attribution entry point, deliberately separate from Trainer.predict."""
from typing import Any

from e_multimodal_sentiment.models.api import ModelOutput, MultimodalModel

from .attribution import AttributionMethod, AttributionResult, AttributionTarget


def explain(
    model: MultimodalModel,
    batch: dict[str, Any],
    target: AttributionTarget,
    method: AttributionMethod,
    output: ModelOutput | None = None,
) -> AttributionResult:
    """Delegate directly to a method that may enable gradients as needed."""
    return method.explain(model=model, batch=batch, target=target, output=output)
