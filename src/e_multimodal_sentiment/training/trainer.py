"""Shared minimal training and validation loops."""
from collections.abc import Callable, Iterable
from dataclasses import fields, is_dataclass, replace
from typing import Any

import torch
from torch import nn

from e_multimodal_sentiment.evaluation.metrics import compute_metrics
from e_multimodal_sentiment.models.api import ModelOutput

from .hooks import TaskHook
from .losses import Criterion

ForwardFunction = Callable[[nn.Module, Any, bool], ModelOutput]


def _move_to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if is_dataclass(value) and not isinstance(value, type):
        return replace(
            value,
            **{
                item.name: _move_to_device(getattr(value, item.name), device)
                for item in fields(value)
            },
        )
    if isinstance(value, dict):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _default_forward(model: nn.Module, batch: Any, diagnostics: bool) -> ModelOutput:
    return model(
        text=batch["text"],
        audio=batch["audio"],
        vision=batch["vision"],
        valid_masks=batch.get("valid_masks"),
        missing_masks=batch.get("missing_masks"),
        return_native_diagnostics=diagnostics,
    )


class Trainer:
    """Run optimization and validation without choosing model or loss policy."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: Criterion,
        device: str = "cpu",
        hook: TaskHook | None = None,
        forward_fn: ForwardFunction | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.hook = hook or TaskHook()
        self.forward_fn = forward_fn or _default_forward

    def _prepare(self, batch: Any) -> Any:
        return self.hook.preprocess_batch(_move_to_device(batch, self.device))

    def _forward(self, batch: Any, return_native_diagnostics: bool = False) -> ModelOutput:
        output = self.forward_fn(self.model, batch, return_native_diagnostics)
        if not torch.isfinite(output.class_logits).all():
            raise FloatingPointError("class_logits contains NaN or Inf.")
        if not torch.isfinite(output.regression).all():
            raise FloatingPointError("regression contains NaN or Inf.")
        return output

    def train_step(self, batch: Any) -> dict[str, float]:
        """Optimize one batch using the injected criterion."""
        self.model.train()
        batch = self._prepare(batch)
        output = self._forward(batch)
        loss, components = self.criterion(output, batch)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError("Criterion must return a finite scalar loss.")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.hook.postprocess_output(batch, output)
        return {name: value.detach().item() for name, value in components.items()}

    def train_epoch(
        self,
        dataloader: Iterable[Any],
        *,
        max_batches: int | None = None,
    ) -> dict[str, float | int]:
        """Train over at most max_batches and return mean loss components."""
        if max_batches is not None and max_batches < 1:
            raise ValueError("max_batches must be positive or None.")
        totals: dict[str, float] = {}
        count = 0
        for batch in dataloader:
            result = self.train_step(batch)
            count += 1
            for name, value in result.items():
                totals[name] = totals.get(name, 0.0) + value
            if max_batches is not None and count >= max_batches:
                break
        if count == 0:
            raise ValueError("Training dataloader produced no batches.")
        return {"batches": count, **{name: value / count for name, value in totals.items()}}

    @torch.no_grad()
    def validate_epoch(
        self,
        dataloader: Iterable[Any],
        *,
        max_batches: int | None = None,
    ) -> dict[str, float | int]:
        """Aggregate labeled predictions, then compute shared metrics once."""
        if max_batches is not None and max_batches < 1:
            raise ValueError("max_batches must be positive or None.")
        self.model.eval()
        class_logits, class_labels = [], []
        regression, reg_labels = [], []
        count = 0
        for raw_batch in dataloader:
            batch = self._prepare(raw_batch)
            output = self._forward(batch)
            class_mask = batch["class_label_mask"]
            reg_mask = batch["reg_label_mask"]
            if class_mask.any():
                class_logits.append(output.class_logits[class_mask])
                class_labels.append(batch["class_label"][class_mask])
            if reg_mask.any():
                regression.append(output.regression[reg_mask])
                reg_labels.append(batch["reg_label"][reg_mask])
            count += 1
            if max_batches is not None and count >= max_batches:
                break
        if count == 0:
            raise ValueError("Validation dataloader produced no batches.")
        metrics = compute_metrics(
            class_logits=torch.cat(class_logits) if class_logits else None,
            class_labels=torch.cat(class_labels) if class_labels else None,
            regression=torch.cat(regression) if regression else None,
            reg_labels=torch.cat(reg_labels) if reg_labels else None,
        )
        return {"batches": count, **metrics}

    @torch.no_grad()
    def predict(self, batch: Any, return_native_diagnostics: bool = False) -> ModelOutput:
        """Perform ordinary prediction without gradients."""
        self.model.eval()
        batch = self._prepare(batch)
        output = self._forward(batch, return_native_diagnostics)
        return self.hook.postprocess_output(batch, output)
