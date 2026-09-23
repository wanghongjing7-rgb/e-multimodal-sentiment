"""Unlabeled specialty inference using the shared trainer."""
from typing import Any
from e_multimodal_sentiment.models.api import ModelOutput
from e_multimodal_sentiment.training.trainer import Trainer


def predict_batch(trainer: Trainer, batch: dict[str, Any]) -> ModelOutput:
    """In-memory prediction only; file format and export await attachment audit."""
    return trainer.predict(batch)
