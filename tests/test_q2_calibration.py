"""Minimal gates for ratio-only and 18-D observable calibration."""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

from e_multimodal_sentiment.models.backbones.mmsa_multitask import MMSAMulTMultiTask
from e_multimodal_sentiment.q2.gap_proxy import GeometryNormalization
from e_multimodal_sentiment.q2.robust_modules import CalibratedMMSAMulT


class FakeMulT(nn.Module):
    """Tiny source-shaped MulT boundary; only the branch/projection contract."""

    def __init__(self) -> None:
        super().__init__()
        self.d_l = self.d_a = self.d_v = 2
        self.proj1 = nn.Linear(12, 12)
        self.proj2 = nn.Linear(12, 12)
        self.out_layer = nn.Linear(12, 1)

    def forward(self, text, audio, video):
        branches = [item.mean(dim=(1, 2)).unsqueeze(-1).expand(-1, 4)
                    for item in (text, audio, video)]
        fused = torch.cat(branches, dim=-1)
        shared = self.proj2(torch.relu(self.proj1(fused))) + fused
        return {"M": self.out_layer(shared), "Feature_t": branches[0],
                "Feature_a": branches[1], "Feature_v": branches[2], "Feature_f": fused}


def observation():
    text = torch.ones(2, 20, 8)
    audio = torch.ones(2, 20, 4)
    vision = torch.ones(2, 20, 3)
    tokens = torch.zeros(2, 3, 20)
    tokens[:, 0, 0] = 101
    tokens[:, 0, 1:17] = 200
    tokens[:, 0, 17] = 102
    tokens[:, 1, :18] = 1
    valid = {name: tokens[:, 1].bool() for name in ("text", "audio", "vision")}
    return text, audio, vision, tokens, valid


def calibrated(mode="geometry"):
    base = MMSAMulTMultiTask(FakeMulT())
    norm = GeometryNormalization("eligible", "eligible", "eligible", 4.0, "union")
    return CalibratedMMSAMulT(base, mode=mode, epsilon=0.2, hidden_dim=8, normalization=norm)


@pytest.mark.parametrize("mode,dimension", [("ratio", 3), ("geometry", 18)])
def test_zero_init_is_exact_b1_branch_path(mode, dimension):
    model = calibrated(mode).eval()
    text, audio, vision, tokens, valid = observation()
    with torch.no_grad():
        expected = model.base(text, audio, vision)
        actual = model(text, audio, vision, text_bert=tokens, valid_masks=valid)
    assert model.calibrator[0].in_features == dimension
    torch.testing.assert_close(actual.auxiliary["calibration_alpha"], torch.ones(2, 3))
    torch.testing.assert_close(actual.fused_hidden, expected.fused_hidden)
    torch.testing.assert_close(actual.class_logits, expected.class_logits)
    torch.testing.assert_close(actual.regression, expected.regression)
    assert torch.isfinite(actual.class_logits).all()
    assert torch.isfinite(actual.regression).all()
    assert torch.isfinite(actual.auxiliary["observable_descriptor"]).all()


def test_true_mask_cannot_enter_public_forward_and_gate_is_bounded():
    model = calibrated()
    assert "true_simulated_missing_mask" not in inspect.signature(model.forward).parameters
    text, audio, vision, tokens, valid = observation()
    with pytest.raises(TypeError):
        model(text, audio, vision, text_bert=tokens, valid_masks=valid,
              true_simulated_missing_mask=valid)
    with torch.no_grad():
        model.calibrator[-1].bias[:] = torch.tensor([100.0, -100.0, 0.0])
    result = model(text, audio, vision, text_bert=tokens, valid_masks=valid)
    alpha = result.auxiliary["calibration_alpha"]
    assert (alpha >= 0.8).all() and (alpha <= 1.2).all()


def test_calibration_final_layer_receives_gradient():
    model = calibrated("geometry")
    text, audio, vision, tokens, valid = observation()
    tokens[:, 0, 5:7] = 100
    output = model(text, audio, vision, text_bert=tokens, valid_masks=valid)
    loss = output.class_logits.square().sum() + output.regression.square().sum()
    loss.backward()
    gradient = model.calibrator[-1].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert bool((gradient.abs() > 0).any())
