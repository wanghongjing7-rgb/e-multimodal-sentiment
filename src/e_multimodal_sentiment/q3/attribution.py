"""Attribution contracts only; no IG, SHAP, or ablation implementation."""
from dataclasses import dataclass, field
from typing import Any, Protocol

import torch

from e_multimodal_sentiment.models.api import ModelOutput, MultimodalModel


@dataclass(frozen=True)
class AttributionTarget:
    """Output target whose evidence is requested."""

    task: str
    class_index: int | None = None

    def __post_init__(self) -> None:
        if self.task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'.")
        if self.task == "classification":
            if self.class_index is None or self.class_index < 0:
                raise ValueError("classification attribution requires a non-negative class_index.")
        elif self.class_index is not None:
            raise ValueError("regression attribution must not set class_index.")


@dataclass
class AttributionResult:
    """Attribution values on each modality's native time axis."""

    modality_contributions: torch.Tensor | None = None
    temporal_importance: dict[str, torch.Tensor] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class AttributionMethod(Protocol):
    """Methods may run gradients and repeated forwards directly on the model."""

    def explain(
        self,
        model: MultimodalModel,
        batch: dict[str, Any],
        target: AttributionTarget,
        output: ModelOutput | None = None,
    ) -> AttributionResult:
        ...
