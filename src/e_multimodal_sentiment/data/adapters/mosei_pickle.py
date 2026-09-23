"""Conservative read-only adapter shell for MOSEI-style pickle dictionaries."""
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from e_multimodal_sentiment.common.schema import SampleSchema

from .base import BaseAdapter


class MoseiPickleAdapter(BaseAdapter):
    """Inspect trusted pickle content; mapping awaits D-E-AUDIT-Q2-001.

    Pickle loading can execute code. The caller must explicitly mark the local
    file as trusted. No aligned or unaligned layout is assumed.
    """

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
                raise ValueError("Expected a top-level pickle mapping.")
            self._data = data
        return self._data

    def inspect_schema(self) -> dict[str, Any]:
        data = self._load()
        return {
            "path_name": self.path.name,
            "top_level_keys": [str(key) for key in data],
            "splits": {
                split: {
                    "present": split in data,
                    "fields": self.list_fields(split) if split in data else [],
                    "type": type(data.get(split)).__name__,
                }
                for split in ("train", "valid", "test")
            },
            "mapping_status": "unconfirmed: run D-E-AUDIT-Q2-001",
        }

    def list_fields(self, split: str) -> list[str]:
        value = self._load().get(split)
        if not isinstance(value, Mapping):
            return []
        return [str(key) for key in value]

    def to_samples(self, split: str) -> Sequence[SampleSchema]:
        raise NotImplementedError(
            "Exact competition field mapping is intentionally deferred until D-E-AUDIT-Q2-001."
        )
