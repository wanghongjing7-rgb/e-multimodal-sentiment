"""Frozen common data contracts and mask semantics."""
from dataclasses import dataclass

import torch

MODALITIES = ("text", "audio", "vision")
TEXT_REPRESENTATIONS = ("dense", "bert_tokens")


@dataclass
class ModalitySequence:
    """One modality sequence for a sample or padded batch.

    valid_mask marks structural validity, chiefly batch padding.
    observed_mask marks whether the source contains a modality observation.
    missing_mask marks only missingness introduced or identified for Q2.
    No mask is inferred from feature values.
    """

    features: torch.Tensor
    valid_mask: torch.Tensor
    observed_mask: torch.Tensor
    missing_mask: torch.Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.features, torch.Tensor):
            raise TypeError("features must be a torch.Tensor.")
        if self.features.ndim not in (2, 3):
            raise ValueError("features must have shape [T,D] or [B,T,D].")
        expected_shape = self.features.shape[:-1]
        for name in ("valid_mask", "observed_mask", "missing_mask"):
            mask = getattr(self, name)
            if not isinstance(mask, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor.")
            if mask.ndim != self.features.ndim - 1 or mask.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {tuple(expected_shape)} for features "
                    f"with shape {tuple(self.features.shape)}."
                )
            if mask.device != self.features.device:
                raise ValueError(f"{name} and features must be on the same device.")
            if mask.dtype != torch.bool:
                setattr(self, name, mask.to(dtype=torch.bool))

    @property
    def available_mask(self) -> torch.Tensor:
        """Positions available to downstream computation."""
        return self.valid_mask & self.observed_mask & ~self.missing_mask

    @property
    def length(self) -> int:
        """Temporal length T for either [T,D] or [B,T,D] features."""
        return self.features.shape[-2]


@dataclass
class SampleSchema:
    """One aligned or unaligned multimodal sample."""

    sample_id: str
    text: ModalitySequence
    audio: ModalitySequence
    vision: ModalitySequence
    text_representation: str = "dense"
    classification_label: int | None = None
    regression_label: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("sample_id must be a non-empty string.")
        for name in MODALITIES:
            sequence = getattr(self, name)
            if not isinstance(sequence, ModalitySequence):
                raise TypeError(f"{name} must be a ModalitySequence.")
            if sequence.features.ndim != 2:
                raise ValueError(f"{name} in SampleSchema must contain [T,D] features.")
        if self.text_representation not in TEXT_REPRESENTATIONS:
            raise ValueError(
                f"text_representation must be one of {TEXT_REPRESENTATIONS}, "
                f"got {self.text_representation!r}."
            )

    @property
    def is_aligned(self) -> bool:
        """Whether the three native temporal lengths are equal."""
        return self.text.length == self.audio.length == self.vision.length

    @property
    def id(self) -> str:
        """Compatibility alias for the previous field name."""
        return self.sample_id

    @property
    def class_label(self) -> int | None:
        """Compatibility alias for the previous field name."""
        return self.classification_label

    @property
    def reg_label(self) -> float | None:
        """Compatibility alias for the previous field name."""
        return self.regression_label


@dataclass
class BatchSchema:
    """Padded batch with modality-specific temporal lengths."""

    sample_ids: list[str]
    text: ModalitySequence
    audio: ModalitySequence
    vision: ModalitySequence
    text_representation: str
    classification_labels: torch.Tensor | None = None
    regression_labels: torch.Tensor | None = None
    classification_label_mask: torch.Tensor | None = None
    regression_label_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        batch_size = len(self.sample_ids)
        for name in MODALITIES:
            sequence = getattr(self, name)
            if not isinstance(sequence, ModalitySequence):
                raise TypeError(f"{name} must be a ModalitySequence.")
            if sequence.features.ndim != 3:
                raise ValueError(f"{name} in BatchSchema must contain [B,T,D] features.")
            if sequence.features.shape[0] != batch_size:
                raise ValueError(f"{name} batch size must match sample_ids.")
        if self.text_representation not in TEXT_REPRESENTATIONS:
            raise ValueError(f"text_representation must be one of {TEXT_REPRESENTATIONS}.")

    @property
    def valid_masks(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name).valid_mask for name in MODALITIES}

    @property
    def observed_masks(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name).observed_mask for name in MODALITIES}

    @property
    def missing_masks(self) -> dict[str, torch.Tensor]:
        return {name: getattr(self, name).missing_mask for name in MODALITIES}

    def __getitem__(self, key: str):
        """Minimal mapping compatibility for existing common consumers."""
        aliases = {
            "id": self.sample_ids,
            "sample_ids": self.sample_ids,
            "text": self.text.features,
            "audio": self.audio.features,
            "vision": self.vision.features,
            "valid_masks": self.valid_masks,
            "observed_masks": self.observed_masks,
            "missing_masks": self.missing_masks,
            "class_label": self.classification_labels,
            "classification_labels": self.classification_labels,
            "reg_label": self.regression_labels,
            "regression_labels": self.regression_labels,
            "class_label_mask": self.classification_label_mask,
            "classification_label_mask": self.classification_label_mask,
            "reg_label_mask": self.regression_label_mask,
            "regression_label_mask": self.regression_label_mask,
        }
        try:
            return aliases[key]
        except KeyError as error:
            raise KeyError(key) from error

    def get(self, key: str, default=None):
        """Dictionary-style get used by existing training code."""
        try:
            return self[key]
        except KeyError:
            return default
