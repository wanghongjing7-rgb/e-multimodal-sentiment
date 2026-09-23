"""Pluggable loss contracts for shared training."""
from typing import Any, Protocol

import torch
from torch import nn

from e_multimodal_sentiment.models.api import ModelOutput


class Criterion(Protocol):
    """Compute the total scalar loss and reportable components."""

    def __call__(
        self, output: ModelOutput, batch: Any
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ...


class MultiTaskCriterion:
    """Exploratory weighted classification and regression objective."""

    def __init__(self, lambda_cls: float = 1.0, lambda_reg: float = 1.0) -> None:
        if lambda_cls < 0 or lambda_reg < 0:
            raise ValueError("Loss weights must be non-negative.")
        if lambda_cls == 0 and lambda_reg == 0:
            raise ValueError("At least one loss weight must be positive.")
        self.lambda_cls = float(lambda_cls)
        self.lambda_reg = float(lambda_reg)
        self.classification_loss = nn.CrossEntropyLoss()
        self.regression_loss = nn.MSELoss()

    def __call__(
        self, output: ModelOutput, batch: Any
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        class_mask = batch["class_label_mask"]
        reg_mask = batch["reg_label_mask"]
        components: dict[str, torch.Tensor] = {}
        total = output.regression.new_zeros(())

        if class_mask.any():
            labels = batch["class_label"]
            if labels is None:
                raise ValueError("classification labels are missing for labeled positions.")
            if labels.dtype != torch.int64:
                raise TypeError("classification labels must use torch.int64.")
            selected = labels[class_mask]
            if not torch.isin(selected, selected.new_tensor([0, 1, 2])).all():
                raise ValueError("classification labels must contain only 0, 1, or 2.")
            components["classification"] = self.classification_loss(
                output.class_logits[class_mask], selected
            )
            total = total + self.lambda_cls * components["classification"]

        if reg_mask.any():
            labels = batch["reg_label"]
            if labels is None:
                raise ValueError("regression labels are missing for labeled positions.")
            if labels.dtype != torch.float32:
                raise TypeError("regression labels must use torch.float32.")
            selected = labels[reg_mask]
            if not torch.isfinite(selected).all():
                raise ValueError("regression labels must be finite.")
            components["regression"] = self.regression_loss(
                output.regression[reg_mask], selected
            )
            total = total + self.lambda_reg * components["regression"]

        if not components:
            raise ValueError("Unlabeled specialty data cannot be used for supervised training.")
        components["total"] = total
        return total, components


class SmokeTestMultiTaskCriterion(MultiTaskCriterion):
    """Backward-compatible equal-weight criterion used by synthetic smoke tests."""
