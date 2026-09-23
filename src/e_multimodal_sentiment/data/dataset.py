"""Dataset over validated samples; serialization belongs to adapters."""
from collections.abc import Sequence

from torch.utils.data import Dataset

from e_multimodal_sentiment.common.schema import SampleSchema


class MultimodalDataset(Dataset[SampleSchema]):
    """Hold already adapted samples without parsing pickle files."""

    def __init__(self, samples: Sequence[SampleSchema]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> SampleSchema:
        return self.samples[index]


# Compatibility import for callers that used the original module location.
from .collate import collate_samples  # noqa: E402,F401
