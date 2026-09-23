"""Minimal gradient and shape checks for the MMSA MulT multitask wrapper."""
import torch
from torch import nn

from e_multimodal_sentiment.models.backbones import MMSAMulTMultiTask


class FakeMulT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Parameter(torch.randn(1, 180))
        self.out_layer = nn.Linear(180, 1)

    def forward(self, text, audio, video):
        batch_size = text.shape[0]
        shared = self.shared.expand(batch_size, -1)
        return {
            "Feature_t": shared[:, :60],
            "Feature_a": shared[:, 60:120],
            "Feature_v": shared[:, 120:180],
            "Feature_f": shared,
            "M": self.out_layer(shared),
        }


def test_mmsa_multitask_shapes_and_gradients() -> None:
    base = FakeMulT()
    model = MMSAMulTMultiTask(base)
    assert isinstance(base.out_layer, nn.Identity)
    assert model.shared_dim == 180

    output = model(torch.ones(2, 1, 1), torch.ones(2, 1, 1), torch.ones(2, 1, 1))
    assert output.fused_hidden.shape == (2, 180)
    assert output.class_logits.shape == (2, 3)
    assert output.regression.shape == (2,)
    assert set(output.modality_hidden) == {"text", "audio", "vision"}
    assert output.auxiliary["mmsa_feature_f"].shape == (2, 180)

    class_gradient = torch.autograd.grad(
        output.class_logits.sum(), output.fused_hidden, retain_graph=True
    )[0]
    regression_gradient = torch.autograd.grad(
        output.regression.sum(), output.fused_hidden
    )[0]
    assert torch.count_nonzero(class_gradient)
    assert torch.count_nonzero(regression_gradient)
