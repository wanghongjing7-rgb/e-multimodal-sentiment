"""Read-only adapter for the confirmed Attachment2 aligned pickle layout."""
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from e_multimodal_sentiment.common.schema import ModalitySequence, SampleSchema

from .base import BaseAdapter

_SPLITS = ("train", "valid", "test")
_MODALITY_SHAPES = {
    "text": (50, 768),
    "audio": (50, 74),
    "vision": (50, 35),
}
_REQUIRED_FIELDS = (
    "text",
    "audio",
    "vision",
    "classification_labels",
    "regression_labels",
    "id",
)


class Attachment2AlignedAdapter(BaseAdapter):
    """Convert a trusted Attachment2 aligned pickle into shared samples."""

    def __init__(self, path: str | Path, *, trusted: bool = False) -> None:
        self.path = Path(path)
        self.trusted = trusted
        self._data: Mapping[str, Any] | None = None

    def _load(self) -> Mapping[str, Any]:
        if not self.trusted:
            raise ValueError("Only load a trusted pickle; construct with trusted=True.")
        if self._data is None:
            with self.path.open("rb") as stream:
                data = pickle.load(stream)
            if not isinstance(data, Mapping):
                raise ValueError("Attachment2 must contain a top-level mapping.")
            self._data = data
        return self._data

    def _split(self, split: str) -> Mapping[str, Any]:
        if split not in _SPLITS:
            raise ValueError(f"split must be one of {_SPLITS}.")
        value = self._load().get(split)
        if not isinstance(value, Mapping):
            raise ValueError(f"Attachment2 split {split!r} must be a mapping.")
        missing = [field for field in _REQUIRED_FIELDS if field not in value]
        if missing:
            raise ValueError(f"Attachment2 split {split!r} is missing fields: {missing}.")
        return value

    def inspect_schema(self) -> dict[str, Any]:
        data = self._load()
        return {
            "source_name": self.path.name,
            "splits": {
                split: {
                    "present": split in data,
                    "fields": self.list_fields(split) if split in data else [],
                }
                for split in _SPLITS
            },
        }

    def list_fields(self, split: str) -> list[str]:
        return [str(field) for field in self._split(split)]

    @staticmethod
    def _features(value: Any, modality: str) -> torch.Tensor:
        tensor = torch.as_tensor(value).to(dtype=torch.float32)
        expected_time, expected_dim = _MODALITY_SHAPES[modality]
        if tensor.ndim != 3 or tensor.shape[1:] != (expected_time, expected_dim):
            raise ValueError(
                f"{modality} must have shape [N,{expected_time},{expected_dim}], "
                f"got {tuple(tensor.shape)}."
            )
        return tensor

    def to_samples(self, split: str) -> Sequence[SampleSchema]:
        raw = self._split(split)
        modalities = {
            name: self._features(raw[name], name)
            for name in _MODALITY_SHAPES
        }
        sample_count = modalities["text"].shape[0]
        if any(tensor.shape[0] != sample_count for tensor in modalities.values()):
            raise ValueError("All Attachment2 modalities must have the same sample count.")

        classification = torch.as_tensor(raw["classification_labels"]).reshape(-1)
        regression = torch.as_tensor(raw["regression_labels"]).reshape(-1)
        ids = list(raw["id"])
        if len(classification) != sample_count or len(regression) != sample_count or len(ids) != sample_count:
            raise ValueError("Labels and ids must match the modality sample count.")
        classification = classification.to(dtype=torch.int64)
        regression = regression.to(dtype=torch.float32)

        samples: list[SampleSchema] = []
        for index in range(sample_count):
            sequences: dict[str, ModalitySequence] = {}
            for name, values in modalities.items():
                features = values[index]
                valid = torch.ones(features.shape[0], dtype=torch.bool, device=features.device)
                observed = torch.any(features != 0, dim=-1)
                missing = torch.zeros_like(valid)
                sequences[name] = ModalitySequence(features, valid, observed, missing)
            samples.append(
                SampleSchema(
                    sample_id=str(ids[index]),
                    text=sequences["text"],
                    audio=sequences["audio"],
                    vision=sequences["vision"],
                    text_representation="dense",
                    classification_label=int(classification[index].item()),
                    regression_label=float(regression[index].item()),
                )
            )
        return samples
