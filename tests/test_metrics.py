"""Known arithmetic expectations, not competition results."""
import math
import pytest
import torch
from e_multimodal_sentiment.evaluation.metrics import compute_metrics


def test_known_metrics() -> None:
    logits = torch.eye(3)[torch.tensor([0, 1, 1, 2])]
    result = compute_metrics(logits, torch.tensor([0, 0, 1, 2]),
                             torch.tensor([1., 2., 3.]), torch.tensor([2., 3., 4.]))
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["macro_f1"] == pytest.approx(7 / 9)
    assert result["weighted_f1"] == pytest.approx(0.75)
    assert result["mae"] == pytest.approx(1)
    assert result["pearson"] == pytest.approx(1)


def test_undefined_pearson() -> None:
    result = compute_metrics(regression=torch.ones(3), reg_labels=torch.arange(3.))
    assert math.isnan(result["pearson"])


def test_missing_class_fixed_three_class_macro() -> None:
    result = compute_metrics(torch.tensor([[1., 0., 0.]]), torch.tensor([0]))
    assert result["macro_f1"] == pytest.approx(1 / 3)
    assert result["weighted_f1"] == 1


def test_invalid_metric_input() -> None:
    with pytest.raises(ValueError):
        compute_metrics(regression=torch.tensor([float("nan")]), reg_labels=torch.ones(1))
    with pytest.raises(ValueError):
        compute_metrics(class_labels=torch.tensor([1]))
    assert compute_metrics() == {}
