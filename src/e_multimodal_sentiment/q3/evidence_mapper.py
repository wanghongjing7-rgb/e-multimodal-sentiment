"""Map temporal importance back to source evidence after alignment audit."""
from typing import Any, Protocol
from .attribution import AttributionResult


class EvidenceMapper(Protocol):
    def map_evidence(self, attribution: AttributionResult, metadata: dict[str, Any]) -> list[dict[str, Any]]:
        """Return timestamp/token/frame evidence using real source metadata."""
        ...
