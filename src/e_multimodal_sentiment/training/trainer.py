"""Shared optimization shell; all loss policy lives in a Criterion."""
from typing import Any

import torch

from e_multimodal_sentiment.models.api import ModelOutput, MultimodalModel

from .hooks import TaskHook
from .losses import Criterion


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


class Trainer:
    """Run forward/backward/update without choosing task losses."""

    def __init__(
        self,
        model: MultimodalModel,
        optimizer: torch.optim.Optimizer,
        criterion: Criterion,
        device: str = "cpu",
        hook: TaskHook | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.hook = hook or TaskHook()

    def _prepare(self, batch: dict[str, Any]) -> dict[str, Any]:
        return self.hook.preprocess_batch(_move_to_device(batch, self.device))

    def _forward(self, batch: dict[str, Any], return_native_diagnostics: bool = False) -> ModelOutput:
        return self.model(
            text=batch["text"],
            audio=batch["audio"],
            vision=batch["vision"],
            valid_masks=batch.get("valid_masks"),
            missing_masks=batch.get("missing_masks"),
            return_native_diagnostics=return_native_diagnostics,
        )

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        """Optimize one batch using the injected criterion."""
        self.model.train()
        batch = self._prepare(batch)
        self.optimizer.zero_grad(set_to_none=True)
        output = self._forward(batch)
        loss, components = self.criterion(output, batch)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise ValueError("Criterion must return a finite scalar loss.")
        loss.backward()
        self.optimizer.step()
        self.hook.postprocess_output(batch, output)
        return {name: value.detach().item() for name, value in components.items()}

    @torch.no_grad()
    def predict(
        self, batch: dict[str, Any], return_native_diagnostics: bool = False
    ) -> ModelOutput:
        """Perform ordinary prediction without gradients."""
        self.model.eval()
        batch = self._prepare(batch)
        output = self._forward(batch, return_native_diagnostics)
        return self.hook.postprocess_output(batch, output)
