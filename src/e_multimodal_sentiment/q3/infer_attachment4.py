"""Complete-modality specialty inference over the shared trainer."""
from typing import Any
from e_multimodal_sentiment.models.api import ModelOutput
from e_multimodal_sentiment.training.trainer import Trainer


def predict_batch(trainer: Trainer, batch: dict[str, Any]) -> ModelOutput:
    """Request diagnostics; actual attribution requires an explicit explainer."""
    if any(mask.any() for mask in batch["missing_masks"].values()):
        raise ValueError("Q3 requires complete modalities at valid time positions.")
    return trainer.predict(batch, return_native_diagnostics=True)
