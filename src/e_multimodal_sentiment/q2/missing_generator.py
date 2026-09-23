"""Contiguous missing events on independent modality time axes."""
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

from e_multimodal_sentiment.common.schema import MODALITIES

Unit = Literal["steps", "ratio"]


@dataclass
class MissingGenerationResult:
    """Non-mutating result of one synthetic missingness request."""

    modified_features: dict[str, torch.Tensor]
    missing_masks: dict[str, torch.Tensor]
    events: list[dict[str, Any]]


def _validate_axis(features: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, bool]:
    if features.ndim not in (2, 3):
        raise ValueError("features must have shape [T,D] or [B,T,D].")
    if valid_mask.dtype != torch.bool or valid_mask.ndim not in (1, 2):
        raise ValueError("valid_mask must be boolean [T] or [B,T].")
    single = features.ndim == 2
    if single != (valid_mask.ndim == 1):
        raise ValueError("features and valid_mask must both be batched or both be unbatched.")
    if features.shape[:-1] != valid_mask.shape:
        raise ValueError("valid_mask shape must match the feature time axes.")
    return (
        features.unsqueeze(0) if single else features,
        valid_mask.unsqueeze(0) if single else valid_mask,
        single,
    )


def _requested_length(valid_count: int, duration: int | float, unit: Unit) -> int:
    if unit == "steps":
        if isinstance(duration, bool) or int(duration) != duration or duration <= 0:
            raise ValueError("steps duration must be a positive integer.")
        return int(duration)
    if unit == "ratio":
        if not 0 < float(duration) <= 1:
            raise ValueError("ratio duration must be in (0, 1].")
        return max(1, round(valid_count * float(duration)))
    raise ValueError("unit must be 'steps' or 'ratio'.")


def _choose_start(
    valid: torch.Tensor,
    length: int,
    position: int | float,
    unit: Unit,
    generator: torch.Generator,
) -> int:
    starts = [
        start for start in range(valid.numel() - length + 1)
        if bool(valid[start : start + length].all())
    ]
    if not starts:
        raise ValueError("No fully valid contiguous interval satisfies the requested duration.")
    if unit == "steps":
        if isinstance(position, bool) or int(position) != position or position < 0:
            raise ValueError("steps position must be a non-negative integer.")
        requested = int(position)
    else:
        if not 0 <= float(position) <= 1:
            raise ValueError("ratio position must be in [0, 1].")
        requested = round(float(position) * max(valid.numel() - length, 0))
    distances = [abs(start - requested) for start in starts]
    best = [start for start, distance in zip(starts, distances) if distance == min(distances)]
    return best[torch.randint(len(best), (1,), generator=generator).item()]


def generate_single_modality_missing(
    valid_mask: torch.Tensor,
    *,
    features: torch.Tensor,
    position: int | float,
    duration: int | float,
    unit: Unit,
    seed: int = 0,
    modality: str = "unknown",
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Mask one contiguous event per sample on one modality's native axis."""
    features_b, valid_b, single = _validate_axis(features, valid_mask)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    missing = torch.zeros_like(valid_b)
    events: list[dict[str, Any]] = []
    for batch_index, valid in enumerate(valid_b.detach().cpu()):
        length = _requested_length(int(valid.sum().item()), duration, unit)
        start = _choose_start(valid, length, position, unit, generator)
        end = start + length
        missing[batch_index, start:end] = True
        events.append(
            {
                "modality": modality,
                "batch_index": batch_index,
                "start": start,
                "end": end,
                "length": length,
                "unit": unit,
                "requested_position": position,
                "requested_duration": duration,
            }
        )
    missing = missing.to(valid_mask.device)
    modified = torch.where(missing.unsqueeze(-1), torch.zeros_like(features_b), features_b)
    if single:
        return modified[0], missing[0], events
    return modified, missing, events


def generate_multimodal_missing(
    valid_masks: Mapping[str, torch.Tensor],
    modalities: Sequence[str],
    position: int | float,
    duration: int | float,
    unit: Unit,
    seed: int,
    *,
    features: Mapping[str, torch.Tensor],
) -> MissingGenerationResult:
    """Apply equivalent requests independently on selected native time axes."""
    selected = tuple(modalities)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("modalities must contain distinct modality names.")
    if any(name not in MODALITIES for name in selected):
        raise ValueError(f"modalities must be selected from {MODALITIES}.")
    if set(features) != set(MODALITIES) or set(valid_masks) != set(MODALITIES):
        raise ValueError("features and valid_masks must each contain text, audio, and vision.")

    modified = {name: tensor.clone() for name, tensor in features.items()}
    masks = {name: torch.zeros_like(valid_masks[name]) for name in MODALITIES}
    events: list[dict[str, Any]] = []
    for offset, name in enumerate(selected):
        changed, mask, modality_events = generate_single_modality_missing(
            valid_masks[name],
            features=features[name],
            position=position,
            duration=duration,
            unit=unit,
            seed=seed + offset,
            modality=name,
        )
        modified[name] = changed
        masks[name] = mask
        events.extend(modality_events)
    return MissingGenerationResult(modified, masks, events)
