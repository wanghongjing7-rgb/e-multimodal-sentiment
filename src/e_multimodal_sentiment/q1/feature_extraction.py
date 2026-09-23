"""Feature extraction contracts; never downloads or initializes pretrained models."""
from pathlib import Path
from typing import Protocol
import torch


class FeatureExtractor(Protocol):
    def extract(self, video_path: Path, *, seed: int = 0) -> dict[str, torch.Tensor]:
        """Return features and timestamps using explicitly provided local resources."""
        ...
