"""Classification/regression heads over an injected MMSA MulT instance."""
from collections.abc import Mapping

import torch
from torch import nn

from e_multimodal_sentiment.models.api import ModelOutput


class MMSAMulTMultiTask(nn.Module):
    """Reuse MMSA MulT's projected shared representation without copying it."""

    def __init__(self, base_model: nn.Module) -> None:
        super().__init__()
        if not hasattr(base_model, "out_layer"):
            raise ValueError("MMSA MulT base_model must expose out_layer.")
        out_layer = base_model.out_layer
        if not hasattr(out_layer, "in_features"):
            raise ValueError("MMSA MulT base_model.out_layer must expose in_features.")
        shared_dim = out_layer.in_features
        if not isinstance(shared_dim, int) or shared_dim < 1:
            raise ValueError("MMSA MulT out_layer.in_features must be a positive integer.")

        self.shared_dim = shared_dim
        self.base_model = base_model
        self.base_model.out_layer = nn.Identity()
        self.classification_head = nn.Linear(shared_dim, 3)
        self.regression_head = nn.Linear(shared_dim, 1)

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
    ) -> ModelOutput:
        """Return common dual-task output from MMSA's projected fusion state."""
        native = self.base_model(text, audio, video)
        if not isinstance(native, Mapping):
            raise TypeError("MMSA MulT forward must return a mapping.")
        required = ("M", "Feature_t", "Feature_a", "Feature_v", "Feature_f")
        missing = [name for name in required if name not in native]
        if missing:
            raise ValueError(f"MMSA MulT output is missing keys: {missing}.")

        shared = native["M"]
        if shared.ndim != 2 or shared.shape[-1] != self.shared_dim:
            raise ValueError(
                f"MMSA shared representation must have shape [B,{self.shared_dim}], "
                f"got {tuple(shared.shape)}."
            )
        modality_hidden = {
            "text": native["Feature_t"],
            "audio": native["Feature_a"],
            "vision": native["Feature_v"],
        }
        return ModelOutput(
            class_logits=self.classification_head(shared),
            regression=self.regression_head(shared).squeeze(-1),
            text_hidden=modality_hidden["text"],
            audio_hidden=modality_hidden["audio"],
            vision_hidden=modality_hidden["vision"],
            fused_hidden=shared,
            modality_hidden=modality_hidden,
            auxiliary={"mmsa_feature_f": native["Feature_f"]},
        )
