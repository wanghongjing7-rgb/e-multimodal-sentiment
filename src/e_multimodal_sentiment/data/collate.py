"""Minimal collation into the frozen BatchSchema."""
from collections.abc import Sequence

import torch

from e_multimodal_sentiment.common.schema import (
    MODALITIES,
    BatchSchema,
    ModalitySequence,
    SampleSchema,
)


def _pad_modality(samples: Sequence[SampleSchema], modality: str) -> ModalitySequence:
    sequences = [getattr(sample, modality) for sample in samples]
    feature_dims = {sequence.features.shape[-1] for sequence in sequences}
    if len(feature_dims) != 1:
        raise ValueError(f"{modality} feature dimensions must match within a batch.")
    return ModalitySequence(
        features=torch.nn.utils.rnn.pad_sequence(
            [sequence.features for sequence in sequences],
            batch_first=True,
            padding_value=0.0,
        ),
        valid_mask=torch.nn.utils.rnn.pad_sequence(
            [sequence.valid_mask for sequence in sequences],
            batch_first=True,
            padding_value=False,
        ),
        observed_mask=torch.nn.utils.rnn.pad_sequence(
            [sequence.observed_mask for sequence in sequences],
            batch_first=True,
            padding_value=False,
        ),
        missing_mask=torch.nn.utils.rnn.pad_sequence(
            [sequence.missing_mask for sequence in sequences],
            batch_first=True,
            padding_value=False,
        ),
    )


def _collate_label(
    samples: Sequence[SampleSchema],
    name: str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor | None, torch.Tensor]:
    present = torch.tensor([getattr(sample, name) is not None for sample in samples], dtype=torch.bool)
    if not present.any():
        return None, present
    values = [
        getattr(sample, name) if getattr(sample, name) is not None else 0
        for sample in samples
    ]
    return torch.tensor(values, dtype=dtype), present


def collate_samples(samples: Sequence[SampleSchema]) -> BatchSchema:
    """Pad each modality independently; all padding masks are False."""
    if not samples:
        raise ValueError("Cannot collate an empty batch.")
    representations = {sample.text_representation for sample in samples}
    if len(representations) != 1:
        raise ValueError("All samples in a batch must share text_representation.")
    classification_labels, classification_label_mask = _collate_label(
        samples, "classification_label", torch.long
    )
    regression_labels, regression_label_mask = _collate_label(
        samples, "regression_label", torch.float32
    )
    modalities = {name: _pad_modality(samples, name) for name in MODALITIES}
    return BatchSchema(
        sample_ids=[sample.sample_id for sample in samples],
        text=modalities["text"],
        audio=modalities["audio"],
        vision=modalities["vision"],
        text_representation=representations.pop(),
        classification_labels=classification_labels,
        regression_labels=regression_labels,
        classification_label_mask=classification_label_mask,
        regression_label_mask=regression_label_mask,
    )
