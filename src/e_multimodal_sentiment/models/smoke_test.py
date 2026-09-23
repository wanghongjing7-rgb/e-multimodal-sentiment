"""Smoke-test model only. Not a formal competition model."""
import torch
from torch import nn

from e_multimodal_sentiment.common.schema import MODALITIES

from .api import ModelOutput, MultimodalModel
from .components import SmokeTestEncoder, SmokeTestFusion, SmokeTestPredictionHead


class SmokeTestModel(MultimodalModel):
    """Independent masked pooling used only to validate shared interfaces."""

    def __init__(self, input_dims: tuple[int, int, int], hidden_dim: int = 8, seed: int = 0) -> None:
        super().__init__()
        if len(input_dims) != 3 or min(*input_dims, hidden_dim) <= 0:
            raise ValueError("Three positive feature dimensions are required.")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.encoders = nn.ModuleList([SmokeTestEncoder(d, hidden_dim) for d in input_dims])
            self.head = SmokeTestPredictionHead(hidden_dim)
        self.fusion = SmokeTestFusion()

    @staticmethod
    def _mask(
        modality: str,
        features: torch.Tensor,
        masks: dict[str, torch.Tensor] | None,
        default: bool,
    ) -> torch.Tensor:
        if masks is None or modality not in masks:
            return torch.full(features.shape[:2], default, dtype=torch.bool, device=features.device)
        mask = masks[modality]
        if mask.dtype != torch.bool or mask.shape != features.shape[:2] or mask.device != features.device:
            raise ValueError(
                f"{modality} mask must be boolean {list(features.shape[:2])} on the feature device."
            )
        return mask

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        valid_masks: dict[str, torch.Tensor] | None = None,
        missing_masks: dict[str, torch.Tensor] | None = None,
        return_hidden: bool = False,
        return_native_diagnostics: bool = False,
    ) -> ModelOutput:
        features_by_name = dict(zip(MODALITIES, (text, audio, vision)))
        batch_sizes = set()
        for name, features in features_by_name.items():
            if features.ndim != 3:
                raise ValueError(f"{name} features must have shape [B, T_{name}, D_{name}].")
            batch_sizes.add(features.shape[0])
        if len(batch_sizes) != 1:
            raise ValueError("All modalities must share the batch dimension.")

        hidden_by_name: dict[str, torch.Tensor] = {}
        pooled, available = [], []
        temporal_scores: dict[str, torch.Tensor] = {}
        for encoder, (name, features) in zip(self.encoders, features_by_name.items()):
            valid = self._mask(name, features, valid_masks, True)
            missing = self._mask(name, features, missing_masks, False)
            if torch.any(missing & ~valid):
                raise ValueError(f"{name} missing_mask cannot cover invalid or padded positions.")
            observed = valid & ~missing
            safe_features = torch.where(observed.unsqueeze(-1), features, torch.zeros_like(features))
            hidden = encoder(safe_features)
            hidden = torch.where(observed.unsqueeze(-1), hidden, torch.zeros_like(hidden))
            count = observed.sum(1)
            hidden_by_name[name] = hidden
            pooled.append(hidden.sum(1) / count.unsqueeze(-1).clamp_min(1))
            available.append(count > 0)
            temporal_scores[name] = observed.to(features.dtype) / count.unsqueeze(-1).clamp_min(1)

        fused, fusion_weights = self.fusion(torch.stack(pooled, dim=1), torch.stack(available, dim=1))
        class_logits, regression = self.head(fused)
        return ModelOutput(
            class_logits=class_logits,
            regression=regression,
            text_hidden=hidden_by_name["text"] if return_hidden else None,
            audio_hidden=hidden_by_name["audio"] if return_hidden else None,
            vision_hidden=hidden_by_name["vision"] if return_hidden else None,
            fusion_weights=fusion_weights if return_native_diagnostics else None,
            temporal_scores=temporal_scores if return_native_diagnostics else None,
        )
