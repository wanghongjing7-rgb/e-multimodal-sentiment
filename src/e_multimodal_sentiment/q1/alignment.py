"""Alignment contract for raw asynchronous modality streams."""
from typing import Protocol
import torch
from e_multimodal_sentiment.common.schema import SampleSchema


class TemporalAligner(Protocol):
    def align(self, sample_id: str, streams: dict[str, torch.Tensor]) -> SampleSchema:
        """Create aligned features and preserve evidence timestamps in metadata."""
        ...
