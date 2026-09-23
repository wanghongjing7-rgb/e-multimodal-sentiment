"""Base interface for source-specific, read-only data adapters."""
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

from e_multimodal_sentiment.common.schema import SampleSchema


class BaseAdapter(ABC):
    """Inspect a source and convert an explicitly mapped split."""

    @abstractmethod
    def inspect_schema(self) -> dict[str, Any]:
        """Return source structure without guessing semantic field mappings."""
        raise NotImplementedError

    @abstractmethod
    def list_fields(self, split: str) -> list[str]:
        """List raw fields present in one split."""
        raise NotImplementedError

    @abstractmethod
    def to_samples(self, split: str) -> Sequence[SampleSchema]:
        """Convert one split after its exact field mapping is confirmed."""
        raise NotImplementedError
