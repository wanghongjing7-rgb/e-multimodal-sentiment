"""Pluggable loss contracts for shared training."""
from typing import Any, Protocol

import torch
from torch import nn

from e_multimodal_sentiment.models.api import ModelOutput


class Criterion(Protocol):
    """Compute the total scalar loss and detached/reportable components."""

    def __call__(
        self, output: ModelOutput, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        ...


class SmokeTestMultiTaskCriterion:
    """Smoke-test only CE + MSE. Not a formal competition loss."""

    def __call__(
        self, output: ModelOutput, batch: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        class_mask = batch["class_label_mask"]
        reg_mask = batch["reg_label_mask"]
        components: dict[str, torch.Tensor] = {}
        loss = output.regression.new_zeros(())
        if class_mask.any():
            labels = batch["class_label"]
            if labels is None:
                raise ValueError("class_label cannot be None when class_label_mask contains True.")
            components["classification"] = nn.functional.cross_entropy(
                output.class_logits[class_mask], labels[class_mask]
            )
            loss = loss + components["classification"]
        if reg_mask.any():
            labels = batch["reg_label"]
            if labels is None:
                raise ValueError("reg_label cannot be None when reg_label_mask contains True.")
            components["regression"] = nn.functional.mse_loss(
                output.regression[reg_mask], labels[reg_mask]
            )
            loss = loss + components["regression"]
        if not components:
            raise ValueError("Unlabeled specialty data cannot be used for supervised training.")
        components["total"] = loss
        return loss, components
