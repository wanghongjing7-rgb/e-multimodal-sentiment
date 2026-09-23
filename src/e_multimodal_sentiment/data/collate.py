"""Batch collation for independent text, audio, and vision time axes."""
from collections.abc import Sequence
from typing import Any

import torch

from e_multimodal_sentiment.common.schema import MODALITIES, ModalitySequence, SampleSchema


def _materialize_masks(sequence: ModalitySequence) -> tuple[torch.Tensor, torch.Tensor]:
    valid = sequence.valid_mask
    if valid is None:
        valid = torch.ones(sequence.length, dtype=torch.bool, device=sequence.features.device)
    missing = sequence.missing_mask
    if missing is None:
        missing = torch.zeros(sequence.length, dtype=torch.bool, device=sequence.features.device)
    return valid, missing


def _collate_label(
    samples: Sequence[SampleSchema], name: str, dtype: torch.dtype
) -> tuple[torch.Tensor | None, torch.Tensor]:
    present = torch.tensor([getattr(sample, name) is not None for sample in samples], dtype=torch.bool)
    if not present.any():
        return None, present
    values = []
    for sample in samples:
        value = getattr(sample, name)
        if isinstance(value, torch.Tensor):
            value = value.item()
        values.append(value if value is not None else 0)
    return torch.tensor(values, dtype=dtype), present


def collate_samples(samples: Sequence[SampleSchema]) -> dict[str, Any]:
    """Pad each modality separately and preserve missing versus padding."""
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    batch: dict[str, Any] = {"valid_masks": {}, "missing_masks": {}}
    for modality in MODALITIES:
        sequences = [getattr(sample, modality) for sample in samples]
        feature_dims = {sequence.features.shape[1] for sequence in sequences}
        if len(feature_dims) != 1:
            raise ValueError(f"{modality} feature dimensions must match within a batch.")
        batch[modality] = torch.nn.utils.rnn.pad_sequence(
            [sequence.features for sequence in sequences], batch_first=True, padding_value=0.0
        )
        valid_and_missing = [_materialize_masks(sequence) for sequence in sequences]
        batch["valid_masks"][modality] = torch.nn.utils.rnn.pad_sequence(
            [item[0] for item in valid_and_missing], batch_first=True, padding_value=False
        )
        batch["missing_masks"][modality] = torch.nn.utils.rnn.pad_sequence(
            [item[1] for item in valid_and_missing], batch_first=True, padding_value=False
        )
    batch["id"] = [sample.id for sample in samples]
    batch["raw_text"] = [sample.raw_text for sample in samples]
    batch["metadata"] = [sample.metadata for sample in samples]
    batch["class_label"], batch["class_label_mask"] = _collate_label(samples, "class_label", torch.long)
    batch["reg_label"], batch["reg_label_mask"] = _collate_label(samples, "reg_label", torch.float32)
    return batch
