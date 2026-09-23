"""Common Q2/Q3 contracts with independent modality time axes."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import nn


@dataclass
class ModelOutput:
    """Predictions plus optional native model diagnostics."""

    class_logits: torch.Tensor
    regression: torch.Tensor
    text_hidden: torch.Tensor | None = None
    audio_hidden: torch.Tensor | None = None
    vision_hidden: torch.Tensor | None = None
    fused_hidden: torch.Tensor | None = None
    modality_hidden: dict[str, torch.Tensor] | None = None
    fusion_weights: torch.Tensor | None = None
    temporal_scores: dict[str, torch.Tensor] | None = None
    auxiliary: dict[str, torch.Tensor] = field(default_factory=dict)


class MultimodalModel(nn.Module, ABC):
    """Base API for aligned or unaligned batched features."""

    @abstractmethod
    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        valid_masks: dict[str, torch.Tensor] | None = None,
        missing_masks: dict[str, torch.Tensor] | None = None,
        return_hidden: bool = False,
        return_native_diagnostics: bool = False,
    ) -> ModelOutput:
        """Predict without requiring equal text, audio, and vision lengths."""
        raise NotImplementedError
