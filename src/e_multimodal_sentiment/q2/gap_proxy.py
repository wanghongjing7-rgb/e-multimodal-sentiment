"""Observable gap proxies and explicitly parameterized 18-D geometry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import torch

MODALITIES = ("text", "audio", "vision")


@dataclass(frozen=True)
class GeometryNormalization:
    """Exploratory normalization; callers must choose and record a convention.

    ``span_count_divisor`` is retained for existing runner/checkpoint arguments.
    It no longer determines k: k uses the eligible axis's feasible span count.
    """

    ratio: Literal["eligible", "sequence"]
    longest: Literal["eligible", "sequence"]
    center: Literal["eligible", "sequence"]
    span_count_divisor: float
    overlap: Literal["union", "eligible"]

    def __post_init__(self) -> None:
        if self.span_count_divisor <= 0:
            raise ValueError("span_count_divisor must be positive.")


def extract_observable_gap_proxy(
    *,
    text_bert: torch.Tensor,
    audio: torch.Tensor,
    vision: torch.Tensor,
    valid_masks: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Infer candidate gaps solely from the corrupted observation.

    Vision all-zero vectors can be natural and therefore give a noisy proxy.
    No true simulation mask is accepted by this function.
    """
    if text_bert.ndim != 2 or text_bert.shape[0] != 3:
        raise ValueError("text_bert must have shape [3,T].")
    if set(valid_masks) != set(MODALITIES):
        raise ValueError("valid_masks must contain text, audio and vision.")
    token, attention = text_bert[0], text_bert[1]
    text_valid = valid_masks["text"] & (attention == 1) & (token != 101) & (token != 102)
    return {
        "text": text_valid & (token == 100),
        "audio": valid_masks["audio"] & (audio == 0).all(dim=-1),
        "vision": valid_masks["vision"] & (vision == 0).all(dim=-1),
    }


def _span_lengths(mask: torch.Tensor) -> list[int]:
    indices = mask.nonzero(as_tuple=False).flatten().tolist()
    if not indices:
        return []
    lengths: list[int] = []
    length = 1
    for previous, current in zip(indices, indices[1:]):
        if current == previous + 1:
            length += 1
        else:
            lengths.append(length)
            length = 1
    lengths.append(length)
    return lengths


def observable_gap_geometry(
    proxy: Mapping[str, torch.Tensor],
    eligible_masks: Mapping[str, torch.Tensor],
    normalization: GeometryNormalization,
) -> torch.Tensor:
    """Build [r,l,c,k,b] x 3 plus pair overlaps from observable proxies.

    k = K / K_max, where K_max is the maximum number of disjoint one-step
    spans on the eligible axis (sum of ceil(run_length / 2) over eligible runs).
    This is observable at inference and does not use the simulator's 2-4 cap.
    """
    if set(proxy) != set(MODALITIES) or set(eligible_masks) != set(MODALITIES):
        raise ValueError("proxy and eligible_masks must contain all three modalities.")
    values: list[float] = []
    for name in MODALITIES:
        observed = proxy[name].bool() & eligible_masks[name].bool()
        eligible = eligible_masks[name].bool()
        n = observed.numel()
        eligible_count = int(eligible.sum())
        count = int(observed.sum())
        lengths = _span_lengths(observed)
        max_separated_spans = sum((length + 1) // 2 for length in _span_lengths(eligible))
        ratio_den = eligible_count if normalization.ratio == "eligible" else n
        longest_den = eligible_count if normalization.longest == "eligible" else n
        if normalization.center == "eligible":
            eligible_index = eligible.nonzero(as_tuple=False).flatten().tolist()
            rank = {index: order for order, index in enumerate(eligible_index)}
            centers = [rank[int(index)] for index in observed.nonzero(as_tuple=False).flatten()]
            center = sum(centers) / count / max(eligible_count - 1, 1) if count else 0.0
        else:
            centers = observed.nonzero(as_tuple=False).flatten()
            center = float(centers.float().mean()) / max(n - 1, 1) if count else 0.0
        values.extend([
            count / max(ratio_den, 1),
            max(lengths, default=0) / max(longest_den, 1),
            center,
            len(lengths) / max(max_separated_spans, 1),
            float(count > 0),
        ])
    for left, right in (("text", "audio"), ("text", "vision"), ("audio", "vision")):
        if proxy[left].numel() != proxy[right].numel():
            raise ValueError("Pairwise overlap requires aligned time axes.")
        intersection = int((proxy[left].bool() & proxy[right].bool()).sum())
        if normalization.overlap == "union":
            denominator = int((proxy[left].bool() | proxy[right].bool()).sum())
        else:
            denominator = int((eligible_masks[left].bool() & eligible_masks[right].bool()).sum())
        values.append(intersection / max(denominator, 1))
    result = torch.tensor(values, dtype=torch.float32)
    if not torch.isfinite(result).all():
        raise ValueError("Observable geometry contains NaN or Inf.")
    return result
