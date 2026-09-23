"""Shared contracts for aligned and unaligned multimodal samples."""
from dataclasses import dataclass, field
from typing import Any

import torch

MODALITIES = ("text", "audio", "vision")


@dataclass
class ModalitySequence:
    """One modality on its native time axis.

    valid_mask identifies real rather than padded positions.
    missing_mask identifies unavailable data at otherwise valid positions.
    """

    features: torch.Tensor
    valid_mask: torch.Tensor | None = None
    missing_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.features, torch.Tensor) or self.features.ndim != 2:
            raise ValueError("features must be a torch.Tensor with shape [T_m, D_m].")
        if not self.features.is_floating_point():
            raise ValueError("features must use a floating-point dtype.")
        if self.features.shape[0] < 1 or self.features.shape[1] < 1:
            raise ValueError("features must have non-zero time and feature dimensions.")
        for name, mask in (("valid_mask", self.valid_mask), ("missing_mask", self.missing_mask)):
            if mask is None:
                continue
            if not isinstance(mask, torch.Tensor) or mask.dtype != torch.bool:
                raise ValueError(f"{name} must be a boolean torch.Tensor or None.")
            if mask.shape != (self.length,):
                raise ValueError(f"{name} must have shape [{self.length}], got {tuple(mask.shape)}.")
            if mask.device != self.features.device:
                raise ValueError(f"{name} and features must be on the same device.")
        if self.missing_mask is not None:
            valid = self.valid_mask if self.valid_mask is not None else torch.ones_like(self.missing_mask)
            if torch.any(self.missing_mask & ~valid):
                raise ValueError("missing_mask cannot mark invalid or padded positions as missing.")

    @property
    def length(self) -> int:
        """Return the native temporal length T_m."""
        return self.features.shape[0]


@dataclass
class SampleSchema:
    """A sample whose three modalities may use independent time axes."""

    id: str
    text: ModalitySequence
    audio: ModalitySequence
    vision: ModalitySequence
    class_label: torch.Tensor | int | None = None
    reg_label: torch.Tensor | float | None = None
    raw_text: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("id must be a non-empty string.")
        for name in MODALITIES:
            if not isinstance(getattr(self, name), ModalitySequence):
                raise ValueError(f"{name} must be a ModalitySequence.")
        self._validate_scalar_label("class_label", self.class_label)
        self._validate_scalar_label("reg_label", self.reg_label)
        if isinstance(self.class_label, torch.Tensor) and self.class_label.is_floating_point():
            raise ValueError("class_label tensor must use an integer dtype.")
        class_value = self.class_label.item() if isinstance(self.class_label, torch.Tensor) else self.class_label
        if class_value is not None and int(class_value) not in (0, 1, 2):
            raise ValueError("class_label must be 0, 1, 2, or None; formal mapping awaits audit.")
        reg_value = self.reg_label.item() if isinstance(self.reg_label, torch.Tensor) else self.reg_label
        if reg_value is not None and not torch.isfinite(torch.tensor(float(reg_value))):
            raise ValueError("reg_label must be finite or None.")

    @staticmethod
    def _validate_scalar_label(name: str, value: torch.Tensor | int | float | None) -> None:
        if isinstance(value, torch.Tensor) and value.numel() != 1:
            raise ValueError(f"{name} tensor must contain exactly one value.")

    @property
    def is_aligned(self) -> bool:
        """Whether native temporal lengths happen to be equal."""
        return self.text.length == self.audio.length == self.vision.length
