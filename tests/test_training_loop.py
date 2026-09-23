"""Minimal dual-task loss, optimization, and validation metric test."""
import torch

from e_multimodal_sentiment.common.schema import ModalitySequence, SampleSchema
from e_multimodal_sentiment.data.collate import collate_samples
from e_multimodal_sentiment.models.smoke_test import SmokeTestModel
from e_multimodal_sentiment.training.losses import MultiTaskCriterion
from e_multimodal_sentiment.training.trainer import Trainer


def _sequence(value: float, length: int, dim: int) -> ModalitySequence:
    return ModalitySequence(
        features=torch.full((length, dim), value),
        valid_mask=torch.ones(length, dtype=torch.bool),
        observed_mask=torch.ones(length, dtype=torch.bool),
        missing_mask=torch.zeros(length, dtype=torch.bool),
    )


def test_minimal_multitask_training_loop() -> None:
    samples = [
        SampleSchema(
            sample_id=f"sample-{index}",
            text=_sequence(float(index + 1), 4, 2),
            audio=_sequence(float(index + 1), 4, 3),
            vision=_sequence(float(index + 1), 4, 4),
            classification_label=index,
            regression_label=float(index - 1),
        )
        for index in range(3)
    ]
    batch = collate_samples(samples)
    model = SmokeTestModel((2, 3, 4), seed=3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = MultiTaskCriterion(lambda_cls=0.5, lambda_reg=2.0)
    trainer = Trainer(model, optimizer, criterion)

    before = model.head.classifier.weight.detach().clone()
    train = trainer.train_epoch([batch], max_batches=1)
    assert set(train) == {"batches", "classification", "regression", "total"}
    assert train["total"] >= 0
    assert not torch.equal(before, model.head.classifier.weight)

    valid = trainer.validate_epoch([batch], max_batches=1)
    assert {
        "accuracy",
        "f1_macro",
        "f1_weighted",
        "mae",
        "pearson",
    }.issubset(valid)
