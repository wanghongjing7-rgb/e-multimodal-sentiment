"""Unified metrics; unlabeled targets must be filtered by the caller."""
import math
import torch


def compute_metrics(
    class_logits: torch.Tensor | None = None,
    class_labels: torch.Tensor | None = None,
    regression: torch.Tensor | None = None,
    reg_labels: torch.Tensor | None = None,
) -> dict[str, float]:
    """Calculate supported metrics. Undefined Pearson is NaN, never fabricated."""
    result: dict[str, float] = {}
    if (class_logits is None) != (class_labels is None):
        raise ValueError("Classification predictions and labels must be paired.")
    if class_logits is not None and class_labels is not None:
        logits = class_logits.detach().cpu()
        labels = class_labels.detach().cpu()
        if logits.ndim != 2 or logits.shape[1] != 3 or labels.shape != (len(logits),) or not len(labels):
            raise ValueError("Expected nonempty logits [N,3] and labels [N].")
        if not torch.isfinite(logits).all() or not torch.isin(labels, torch.tensor([0, 1, 2])).all():
            raise ValueError("Invalid logits or three-class labels.")
        predicted = logits.argmax(-1)
        result["accuracy"] = (predicted == labels).double().mean().item()
        f1s, supports = [], []
        for c in range(3):
            tp = ((predicted == c) & (labels == c)).sum().item()
            support = (labels == c).sum().item()
            denominator = (predicted == c).sum().item() + support
            f1s.append(2 * tp / denominator if denominator else 0.0)
            supports.append(support)
        # 最终正式口径待确认：同时报告 macro 和 weighted；固定三类，zero_division=0。
        result["macro_f1"] = sum(f1s) / 3
        result["weighted_f1"] = sum(f * n for f, n in zip(f1s, supports)) / len(labels)
    if (regression is None) != (reg_labels is None):
        raise ValueError("Regression predictions and labels must be paired.")
    if regression is not None and reg_labels is not None:
        pred, target = regression.detach().cpu().double(), reg_labels.detach().cpu().double()
        if pred.ndim != 1 or pred.shape != target.shape or not pred.numel():
            raise ValueError("Expected nonempty regression vectors [N].")
        if not torch.isfinite(pred).all() or not torch.isfinite(target).all():
            raise ValueError("Regression values must be finite.")
        result["mae"] = (pred - target).abs().mean().item()
        x, y = pred - pred.mean(), target - target.mean()
        denominator = x.norm() * y.norm()
        result["pearson"] = (torch.dot(x, y) / denominator).clamp(-1, 1).item() if denominator > 0 else math.nan
    return result
