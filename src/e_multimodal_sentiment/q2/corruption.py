"""Q2 exploratory contiguous corruption on one aligned sample.

The returned true mask is for simulation bookkeeping only. It must not be
passed to a prediction model; observable proxies are extracted separately.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal, Mapping

import torch

ModalitySet = Literal["T", "A", "V", "TA", "TV", "AV", "TAV"]
Position = Literal["front", "middle", "rear", "random"]
Shape = Literal["single_block", "multi_block"]
Relation = Literal["synchronous", "staggered"]
NAMES = {"T": "text", "A": "audio", "V": "vision"}


@dataclass(frozen=True)
class CorruptionResult:
    """Non-mutating corrupted view, simulation truth and event metadata."""

    observation: dict[str, torch.Tensor]
    true_simulated_missing_mask: dict[str, torch.Tensor]
    metadata: dict[str, object]


def _runs(mask: torch.Tensor) -> list[tuple[int, int]]:
    indices = mask.nonzero(as_tuple=False).flatten().tolist()
    if not indices:
        return []
    result: list[tuple[int, int]] = []
    start = previous = indices[0]
    for index in indices[1:]:
        if index != previous + 1:
            result.append((start, previous + 1))
            start = index
        previous = index
    result.append((start, previous + 1))
    return result


def _choose_spans(
    eligible: torch.Tensor,
    ratio: float,
    position: Position,
    shape: Shape,
    span_count_weights: Mapping[int, float],
    rng: random.Random,
    anchor: float | None = None,
) -> list[tuple[int, int]]:
    """Select separated spans with the same rounded total length for both shapes."""
    intervals = _runs(eligible)
    eligible_count = int(eligible.sum())
    length = max(1, round(eligible_count * ratio))
    if not intervals:
        raise ValueError("No eligible timestep exists.")
    if length > max(stop - start for start, stop in intervals):
        raise ValueError("Requested missing length exceeds every contiguous valid interval.")
    if shape == "single_block":
        count = 1
    else:
        candidates = [
            k for k, weight in span_count_weights.items()
            if 2 <= k <= 4 and weight > 0 and k <= length
            and any(stop - start >= length + k - 1 for start, stop in intervals)
        ]
        if not candidates:
            raise ValueError("Multi-block infeasible at this ratio and eligible length.")
        count = rng.choices(candidates, weights=[span_count_weights[k] for k in candidates])[0]
    width = length + count - 1
    feasible = [(start, stop) for start, stop in intervals if stop - start >= width]
    if not feasible:
        raise ValueError("Requested separated spans do not fit a valid interval.")
    run_start, run_stop = max(feasible, key=lambda pair: pair[1] - pair[0])
    fraction = anchor if anchor is not None else {
        "front": 0.0, "middle": 0.5, "rear": 1.0, "random": rng.random()
    }[position]
    first = run_start + round((run_stop - run_start - width) * fraction)
    if count == 1:
        return [(first, first + length)]
    cuts = sorted(rng.sample(range(1, length), count - 1))
    parts = [b - a for a, b in zip([0, *cuts], [*cuts, length])]
    spans: list[tuple[int, int]] = []
    cursor = first
    for part in parts:
        spans.append((cursor, cursor + part))
        cursor += part + 1
    return spans


def corrupt_sample(
    *,
    text_bert: torch.Tensor,
    audio: torch.Tensor,
    vision: torch.Tensor,
    valid_masks: Mapping[str, torch.Tensor],
    source_sample_id: str,
    seed: int,
    modality_set: ModalitySet,
    missing_ratio: float,
    position_type: Position,
    shape_type: Shape,
    overlap_type: Relation,
    span_count_weights: Mapping[int, float],
) -> CorruptionResult:
    """Corrupt source observations while preserving CLS/SEP/padding and inputs."""
    if modality_set not in ("T", "A", "V", "TA", "TV", "AV", "TAV"):
        raise ValueError("Invalid modality_set.")
    if position_type not in ("front", "middle", "rear", "random"):
        raise ValueError("Invalid position_type.")
    if shape_type not in ("single_block", "multi_block"):
        raise ValueError("Invalid shape_type.")
    if overlap_type not in ("synchronous", "staggered"):
        raise ValueError("Invalid overlap_type.")
    if not 0 <= missing_ratio <= 1:
        raise ValueError("missing_ratio must be in [0,1].")
    if text_bert.ndim != 2 or text_bert.shape[0] != 3:
        raise ValueError("text_bert must be [3,T].")
    if audio.ndim != 2 or vision.ndim != 2:
        raise ValueError("audio and vision must be [T,D].")
    inputs = {"text": text_bert, "audio": audio, "vision": vision}
    if set(valid_masks) != set(inputs):
        raise ValueError("valid_masks must contain text, audio and vision.")
    for name, features in inputs.items():
        mask = valid_masks[name]
        if mask.dtype != torch.bool or mask.ndim != 1 or mask.shape[0] != features.shape[-1 if name == "text" else 0]:
            raise ValueError(f"{name} valid_mask has the wrong shape/dtype.")
    observation = {name: value.clone() for name, value in inputs.items()}
    truth = {name: torch.zeros_like(valid_masks[name]) for name in inputs}
    selected = [NAMES[symbol] for symbol in modality_set]
    span_lists: dict[str, list[list[int]]] = {name: [] for name in inputs}
    if missing_ratio:
        rng = random.Random(seed)
        base = {"front": 0.0, "middle": 0.5, "rear": 1.0, "random": rng.random()}[position_type]
        for index, name in enumerate(selected):
            eligible = valid_masks[name].clone()
            if name == "text":
                token = text_bert[0]
                attention = text_bert[1] == 1
                eligible &= attention & (token != 101) & (token != 102)
            # Staggered means distinct normalized anchors for affected modalities.
            anchor = base if overlap_type == "synchronous" or len(selected) == 1 else (base + index / len(selected)) % 1.0
            spans = _choose_spans(eligible, missing_ratio, position_type, shape_type, span_count_weights, rng, anchor)
            for start, stop in spans:
                truth[name][start:stop] = True
                if name == "text":
                    observation[name][0, start:stop] = 100
                else:
                    observation[name][start:stop, :] = 0
                span_lists[name].append([start, stop])
    metadata: dict[str, object] = {
        "source_sample_id": source_sample_id,
        "seed": seed,
        "modality_set": modality_set,
        "missing_ratio": missing_ratio,
        "position_type": position_type,
        "shape_type": shape_type,
        "num_spans": {name: len(items) for name, items in span_lists.items()},
        "span_list": span_lists,
        "overlap_type": overlap_type,
    }
    return CorruptionResult(observation, truth, metadata)
