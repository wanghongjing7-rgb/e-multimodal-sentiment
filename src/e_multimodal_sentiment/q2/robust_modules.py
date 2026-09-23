"""Interfaces for future Q2 robustness mechanisms."""
from typing import Protocol
import torch


class RobustnessModule(Protocol):
    """Transform modality-specific hidden features using independent masks."""
    def __call__(
        self,
        hidden: dict[str, torch.Tensor],
        missing_masks: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        ...
