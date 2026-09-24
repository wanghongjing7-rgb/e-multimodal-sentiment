"""Exploratory Q2 ratio-only / geometry calibration over fixed MMSA MulT.

No MMSA forward body is copied. Temporary hooks change only the three
pre-projection branches and the corresponding residual during one forward.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from e_multimodal_sentiment.models.api import ModelOutput
from e_multimodal_sentiment.models.backbones.mmsa_multitask import MMSAMulTMultiTask
from e_multimodal_sentiment.q2.gap_proxy import (
    GeometryNormalization,
    extract_observable_gap_proxy,
    observable_gap_geometry,
)
from e_multimodal_sentiment.training.losses import MultiTaskCriterion


class CalibratedMMSAMulT(nn.Module):
    """Constrained residual calibration at the verified MulT branch boundary.

    Public inputs contain observations and structural validity only. No true
    simulated missing mask can be supplied to this forward method.
    """

    def __init__(
        self,
        base: MMSAMulTMultiTask,
        *,
        mode: str,
        epsilon: float,
        hidden_dim: int,
        normalization: GeometryNormalization,
    ) -> None:
        super().__init__()
        if mode not in ("ratio", "geometry"):
            raise ValueError("mode must be ratio or geometry")
        if not 0 < epsilon < 1:
            raise ValueError("epsilon must be in (0,1)")
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        native = base.base_model
        if not isinstance(native.proj1, nn.Linear) or not isinstance(native.proj2, nn.Linear):
            raise ValueError("Fixed MMSA MulT must expose Linear proj1 and proj2")
        self.branch_dims = (2 * native.d_l, 2 * native.d_a, 2 * native.d_v)
        if sum(self.branch_dims) != base.shared_dim or native.proj1.in_features != base.shared_dim:
            raise ValueError("MulT branch dimensions do not match verified fusion boundary")
        if native.proj2.out_features != base.shared_dim:
            raise ValueError("MulT proj2 must return shared fusion dimension")
        self.base = base
        self.mode = mode
        self.epsilon = float(epsilon)
        self.normalization = normalization
        self.calibrator = nn.Sequential(
            nn.Linear(3 if mode == "ratio" else 18, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
        )
        nn.init.zeros_(self.calibrator[-1].weight)
        nn.init.zeros_(self.calibrator[-1].bias)

    def _descriptor(
        self,
        text_bert: torch.Tensor,
        audio: torch.Tensor,
        vision: torch.Tensor,
        valid_masks: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:
        if text_bert.ndim != 3 or text_bert.shape[1] != 3:
            raise ValueError("text_bert must be [B,3,T]")
        if set(valid_masks) != {"text", "audio", "vision"}:
            raise ValueError("valid_masks must contain text/audio/vision")
        descriptors = []
        for index in range(len(text_bert)):
            sample_valid = {name: mask[index] for name, mask in valid_masks.items()}
            proxy = extract_observable_gap_proxy(
                text_bert=text_bert[index], audio=audio[index], vision=vision[index],
                valid_masks=sample_valid,
            )
            if self.mode == "ratio":
                ratios = []
                for name in ("text", "audio", "vision"):
                    eligible = sample_valid[name].bool()
                    denominator = (int(eligible.sum()) if self.normalization.ratio == "eligible"
                                   else eligible.numel())
                    ratios.append(float((proxy[name] & eligible).sum()) / max(denominator, 1))
                descriptors.append(torch.tensor(ratios, dtype=torch.float32))
            else:
                descriptors.append(observable_gap_geometry(proxy, sample_valid, self.normalization))
        return torch.stack(descriptors).to(device=audio.device)

    def forward(
        self,
        text: torch.Tensor,
        audio: torch.Tensor,
        video: torch.Tensor,
        *,
        text_bert: torch.Tensor,
        valid_masks: Mapping[str, torch.Tensor],
    ) -> ModelOutput:
        """Infer descriptor from observed inputs, then run one MMSA forward."""
        descriptor = self._descriptor(text_bert, audio, video, valid_masks)
        if not torch.isfinite(descriptor).all():
            raise FloatingPointError("Gap descriptor contains NaN/Inf")
        scores = self.calibrator(descriptor)
        alpha = 1.0 + self.epsilon * torch.tanh(scores)
        if not torch.isfinite(alpha).all():
            raise FloatingPointError("Calibration alpha contains NaN/Inf")
        native = self.base.base_model
        context: dict[str, torch.Tensor] = {}

        def before_projection(_module, inputs):
            original = inputs[0]
            if original.ndim != 2 or original.shape[1] != sum(self.branch_dims):
                raise ValueError("Unexpected MulT branch-concatenation shape")
            branches = original.split(self.branch_dims, dim=-1)
            calibrated = torch.cat(
                [branch * alpha[:, index : index + 1] for index, branch in enumerate(branches)],
                dim=-1,
            )
            context["delta"] = calibrated - original
            return (calibrated,)

        def after_projection(_module, _inputs, output):
            if "delta" not in context:
                raise RuntimeError("MulT projection hook order changed")
            return output + context["delta"]

        first = native.proj1.register_forward_pre_hook(before_projection)
        second = native.proj2.register_forward_hook(after_projection)
        try:
            output = self.base(text, audio, video)
        finally:
            second.remove()
            first.remove()
        output.auxiliary = {
            **output.auxiliary,
            "calibration_alpha": alpha,
            "observable_descriptor": descriptor,
        }
        return output


class CalibratedCriterion:
    """COMMON CE/MSE plus the only permitted calibration penalty."""

    def __init__(self, *, lambda_cls: float, lambda_reg: float, beta: float) -> None:
        if beta < 0:
            raise ValueError("beta must be nonnegative")
        self.base = MultiTaskCriterion(lambda_cls=lambda_cls, lambda_reg=lambda_reg)
        self.beta = float(beta)

    def __call__(self, output: ModelOutput, batch):
        total, components = self.base(output, batch)
        alpha = output.auxiliary.get("calibration_alpha")
        if alpha is None:
            raise ValueError("Calibrated output must contain calibration_alpha")
        penalty = ((alpha - 1.0) ** 2).sum(dim=-1).mean()
        total = total + self.beta * penalty
        return total, {**components, "calibration": penalty, "total": total}
