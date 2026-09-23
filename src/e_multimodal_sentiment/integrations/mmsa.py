"""Bridge from BatchSchema to MMSA MulT at commit a94e65d."""
from typing import NamedTuple

import torch

from e_multimodal_sentiment.common.schema import BatchSchema, ModalitySequence


class MMSAMulTInputs(NamedTuple):
    """Positional arguments for MMSA MULT.forward(text, audio, video)."""

    text: torch.Tensor
    audio: torch.Tensor
    video: torch.Tensor


def _available_features(sequence: ModalitySequence) -> torch.Tensor:
    return torch.where(
        sequence.available_mask.unsqueeze(-1),
        sequence.features,
        torch.zeros_like(sequence.features),
    ).to(dtype=torch.float32)


def batch_to_mmsa_mult_inputs(batch: BatchSchema) -> MMSAMulTInputs:
    """Return dense batch-first tensors required by the fixed MMSA MulT."""
    if batch.text_representation != "dense":
        raise ValueError("MMSA MulT bridge currently accepts dense text only.")
    expected_shapes = {
        "text": (50, 768),
        "audio": (50, 74),
        "vision": (50, 35),
    }
    for name, (time_steps, feature_dim) in expected_shapes.items():
        shape = getattr(batch, name).features.shape
        if shape[1:] != (time_steps, feature_dim):
            raise ValueError(
                f"{name} must have shape [B,{time_steps},{feature_dim}], got {tuple(shape)}."
            )
    return MMSAMulTInputs(
        text=_available_features(batch.text),
        audio=_available_features(batch.audio),
        video=_available_features(batch.vision),
    )
