"""Unaligned model, Criterion, Trainer, and attribution interface tests."""
import inspect
from typing import Any

import pytest
import torch

from e_multimodal_sentiment.common.schema import ModalitySequence, SampleSchema
from e_multimodal_sentiment.data.collate import collate_samples
from e_multimodal_sentiment.models.smoke_test import SmokeTestModel
from e_multimodal_sentiment.q2.infer_attachment3 import predict_batch
from e_multimodal_sentiment.q3.attribution import (
    AttributionResult,
    AttributionTarget,
)
from e_multimodal_sentiment.q3.explainers import explain
from e_multimodal_sentiment.q3.infer_attachment4 import predict_batch as predict_complete
from e_multimodal_sentiment.training.losses import SmokeTestMultiTaskCriterion
from e_multimodal_sentiment.training.trainer import Trainer


def make_batch(labeled: bool = True) -> dict[str, Any]:
    sample = SampleSchema(
        "synthetic",
        ModalitySequence(torch.ones(4, 2)),
        ModalitySequence(torch.ones(7, 3)),
        ModalitySequence(torch.ones(10, 4)),
        class_label=1 if labeled else None,
        reg_label=0.2 if labeled else None,
    )
    return collate_samples([sample])


def forward(model: SmokeTestModel, batch: dict[str, Any], **kwargs):
    return model(
        batch["text"],
        batch["audio"],
        batch["vision"],
        valid_masks=batch["valid_masks"],
        missing_masks=batch["missing_masks"],
        **kwargs,
    )


def test_unaligned_model_and_output_diagnostics() -> None:
    model = SmokeTestModel((2, 3, 4), seed=4)
    output = forward(model, make_batch(), return_hidden=True, return_native_diagnostics=True)
    assert output.class_logits.shape == (1, 3)
    assert output.regression.shape == (1,)
    assert output.text_hidden.shape == (1, 4, 8)
    assert output.audio_hidden.shape == (1, 7, 8)
    assert output.vision_hidden.shape == (1, 10, 8)
    assert output.fusion_weights.shape == (1, 3)
    assert set(output.temporal_scores) == {"text", "audio", "vision"}
    assert output.temporal_scores["text"].shape == (1, 4)
    assert output.temporal_scores["audio"].shape == (1, 7)
    assert output.temporal_scores["vision"].shape == (1, 10)
    assert output.auxiliary == {}


def test_masks_are_applied_safely() -> None:
    model = SmokeTestModel((2, 3, 4))
    batch = make_batch()
    batch["missing_masks"]["text"][:, :2] = True
    first = forward(model, batch)
    batch["text"][:, :2] = float("nan")
    second = forward(model, batch)
    torch.testing.assert_close(first.class_logits, second.class_logits)
    batch["valid_masks"]["text"][:, -1] = False
    batch["missing_masks"]["text"][:, -1] = True
    with pytest.raises(ValueError, match="invalid"):
        forward(model, batch)


def test_smoke_criterion_and_trainer_separation() -> None:
    model = SmokeTestModel((2, 3, 4))
    criterion = SmokeTestMultiTaskCriterion()
    batch = make_batch()
    loss, components = criterion(forward(model, batch), batch)
    assert loss.ndim == 0 and set(components) == {"classification", "regression", "total"}
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    trainer = Trainer(model, optimizer, criterion)
    result = trainer.train_step(batch)
    assert result["total"] >= 0
    trainer_source = inspect.getsource(Trainer)
    assert "cross_entropy" not in trainer_source
    assert "mse_loss" not in trainer_source


def test_unlabeled_prediction_and_complete_modality_guard() -> None:
    model = SmokeTestModel((2, 3, 4))
    trainer = Trainer(
        model,
        torch.optim.SGD(model.parameters(), lr=0.01),
        SmokeTestMultiTaskCriterion(),
    )
    batch = make_batch(labeled=False)
    assert predict_batch(trainer, batch).regression.shape == (1,)
    assert predict_complete(trainer, batch).fusion_weights is not None
    with pytest.raises(ValueError, match="Unlabeled"):
        trainer.train_step(batch)
    batch["missing_masks"]["vision"][:, 0] = True
    with pytest.raises(ValueError, match="complete"):
        predict_complete(trainer, batch)


def test_attribution_api_receives_model_batch_target() -> None:
    class RecordingMethod:
        def explain(self, model, batch, target, output=None):
            assert isinstance(model, SmokeTestModel)
            assert batch["audio"].shape[1] == 7
            assert target == AttributionTarget("classification", 2)
            with torch.enable_grad():
                value = model.head.classifier.weight.sum() * 0
            return AttributionResult(
                modality_contributions=torch.zeros(1, 3) + value,
                temporal_importance={
                    "text": torch.zeros(1, 4),
                    "audio": torch.zeros(1, 7),
                    "vision": torch.zeros(1, 10),
                },
                metadata={"method": "recording-test"},
            )

    model = SmokeTestModel((2, 3, 4))
    result = explain(
        model,
        make_batch(),
        AttributionTarget(task="classification", class_index=2),
        RecordingMethod(),
    )
    assert result.temporal_importance["audio"].shape == (1, 7)
    assert result.metadata["method"] == "recording-test"


def test_config_layering_and_machine_isolation() -> None:
    from pathlib import Path

    from e_multimodal_sentiment.common.config import load_config

    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs/base.yaml")
    combined = load_config(
        root / "configs/base.yaml",
        root / "configs/q2.yaml",
        root / "configs/local.example.yaml",
    )
    assert "missingness" not in base and "device" not in base and "paths" not in base
    assert combined["missingness"]["modalities"] == ["text"]
    assert combined["paths"]["data_root"] is None
