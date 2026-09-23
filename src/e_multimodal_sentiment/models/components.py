"""Small shared smoke-test components, replaceable independently."""
import torch
from torch import nn


class SmokeTestEncoder(nn.Module):
    """Linear projection only; not a proposed competition encoder."""
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, hidden_dim)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.projection(features)


class SmokeTestFusion(nn.Module):
    """Average available modalities; weights are diagnostics, not attribution."""
    def forward(self, pooled: torch.Tensor, available: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weights = available.to(pooled.dtype)
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1)
        return (pooled * weights.unsqueeze(-1)).sum(1), weights


class SmokeTestPredictionHead(nn.Module):
    """Shared dual-task linear head."""
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, 3)
        self.regressor = nn.Linear(hidden_dim, 1)

    def forward(self, fused: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.classifier(fused), self.regressor(fused).squeeze(-1)
