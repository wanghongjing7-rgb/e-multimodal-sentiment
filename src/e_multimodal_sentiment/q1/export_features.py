"""Feature export contract; destination is always supplied by the caller."""
from collections.abc import Sequence
from pathlib import Path
from e_multimodal_sentiment.common.schema import SampleSchema


def export_features(samples: Sequence[SampleSchema], output_path: Path) -> None:
    """Storage format and provenance schema await confirmation."""
    raise NotImplementedError("Confirm feature storage format before export.")
