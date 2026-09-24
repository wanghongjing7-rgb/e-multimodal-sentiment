"""Hard Q2-01 gates for local contiguous corruption and observable proxies."""

from __future__ import annotations

import pytest
import torch

from e_multimodal_sentiment.q2.corruption import corrupt_sample
from e_multimodal_sentiment.q2.gap_proxy import (
    GeometryNormalization,
    extract_observable_gap_proxy,
    observable_gap_geometry,
)


def fixture_observation() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Use a short valid text with explicit CLS, SEP and tail padding."""
    text = torch.zeros(3, 20)
    text[0, 0] = 101
    text[0, 1:17] = torch.arange(200, 216)
    text[0, 17] = 102
    text[1, :18] = 1
    audio = torch.ones(20, 4)
    vision = torch.ones(20, 3)
    valid = {name: torch.ones(20, dtype=torch.bool) for name in ("text", "audio", "vision")}
    valid["text"][18:] = False
    valid["audio"][18:] = False
    valid["vision"][18:] = False
    return text, audio, vision, valid


def corrupt(**overrides):
    """Create one deterministic exploratory event with overridable factors."""
    text, audio, vision, valid = fixture_observation()
    kwargs = dict(
        text_bert=text,
        audio=audio,
        vision=vision,
        valid_masks=valid,
        source_sample_id="synthetic-1",
        seed=42,
        modality_set="TAV",
        missing_ratio=0.5,
        position_type="middle",
        shape_type="single_block",
        overlap_type="synchronous",
        span_count_weights={2: 1, 3: 1, 4: 1},
    )
    kwargs.update(overrides)
    return corrupt_sample(**kwargs)


def normalization() -> GeometryNormalization:
    """Explicit test-only geometry convention, not a frozen research setting."""
    return GeometryNormalization("eligible", "eligible", "eligible", 4.0, "union")


def test_single_and_multi_are_separated_and_match_total_length() -> None:
    single = corrupt(shape_type="single_block")
    multi = corrupt(shape_type="multi_block")
    for name in ("text", "audio", "vision"):
        one = single.metadata["span_list"][name]
        many = multi.metadata["span_list"][name]
        assert len(one) == 1
        assert 2 <= len(many) <= 4
        assert sum(stop - start for start, stop in one) == sum(stop - start for start, stop in many)
        assert all(left[1] < right[0] for left, right in zip(many, many[1:]))
        assert all(0 <= start < stop <= 18 for start, stop in many)
        assert int(single.true_simulated_missing_mask[name].sum()) == int(multi.true_simulated_missing_mask[name].sum())


def test_seed_reproduces_spans_without_mutating_source() -> None:
    text, audio, vision, valid = fixture_observation()
    original = (text.clone(), audio.clone(), vision.clone())
    arguments = dict(text_bert=text, audio=audio, vision=vision, valid_masks=valid,
                     source_sample_id="x", seed=33, modality_set="TAV", missing_ratio=0.4,
                     position_type="random", shape_type="multi_block", overlap_type="staggered",
                     span_count_weights={2: 1, 3: 1, 4: 1})
    a = corrupt_sample(**arguments)
    b = corrupt_sample(**arguments)
    assert a.metadata["span_list"] == b.metadata["span_list"]
    for actual, expected in zip((text, audio, vision), original):
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("modality_set", ["T", "A", "V", "TA", "TV", "AV", "TAV"])
def test_modality_combinations_only_change_selected(modality_set: str) -> None:
    event = corrupt(modality_set=modality_set)
    for symbol, name in (("T", "text"), ("A", "audio"), ("V", "vision")):
        assert bool(event.true_simulated_missing_mask[name].any()) == (symbol in modality_set)


@pytest.mark.parametrize("position", ["front", "middle", "rear", "random"])
@pytest.mark.parametrize("relation", ["synchronous", "staggered"])
def test_positions_and_relations_produce_valid_spans(position: str, relation: str) -> None:
    event = corrupt(position_type=position, overlap_type=relation)
    for name, spans in event.metadata["span_list"].items():
        assert len(spans) == 1
        assert not bool((event.true_simulated_missing_mask[name] & ~fixture_observation()[3][name]).any())


def test_text_special_tokens_and_padding_are_protected_and_other_modalities_zero() -> None:
    event = corrupt()
    text = event.observation["text"]
    assert text[0, 0] == 101 and text[0, 17] == 102
    assert torch.equal(text[1], fixture_observation()[0][1])
    assert torch.equal(text[2], fixture_observation()[0][2])
    assert torch.equal(text[:, 18:], fixture_observation()[0][:, 18:])
    assert bool((text[0, event.true_simulated_missing_mask["text"]] == 100).all())
    for name in ("audio", "vision"):
        assert not bool(event.observation[name][event.true_simulated_missing_mask[name]].any())


def test_proxy_is_independent_and_vision_natural_zero_is_noisy() -> None:
    event = corrupt(modality_set="T")
    observation = event.observation
    observation["vision"][4] = 0  # natural zero, not a simulated event
    proxy = extract_observable_gap_proxy(
        text_bert=observation["text"], audio=observation["audio"],
        vision=observation["vision"], valid_masks=fixture_observation()[3],
    )
    assert proxy["vision"][4]
    assert not event.true_simulated_missing_mask["vision"][4]
    assert torch.equal(proxy["text"], event.true_simulated_missing_mask["text"])


def test_geometry_distinguishes_span_count_and_longest_and_is_finite() -> None:
    valid = fixture_observation()[3]
    single = corrupt(modality_set="T", shape_type="single_block")
    multi = corrupt(modality_set="T", shape_type="multi_block")
    descriptions = []
    for event in (single, multi):
        p = extract_observable_gap_proxy(
            text_bert=event.observation["text"], audio=event.observation["audio"],
            vision=event.observation["vision"], valid_masks=valid,
        )
        descriptions.append(observable_gap_geometry(p, valid, normalization()))
    assert all(item.shape == (18,) and torch.isfinite(item).all() for item in descriptions)
    assert descriptions[0][0] == descriptions[1][0]  # same missing ratio
    assert descriptions[0][1] > descriptions[1][1]  # longest span
    assert descriptions[0][3] < descriptions[1][3]  # number of spans


def test_pair_overlap_and_no_gap_are_finite() -> None:
    valid = fixture_observation()[3]
    clean = corrupt(missing_ratio=0)
    zero = extract_observable_gap_proxy(
        text_bert=clean.observation["text"], audio=clean.observation["audio"],
        vision=clean.observation["vision"], valid_masks=valid,
    )
    assert observable_gap_geometry(zero, valid, normalization()).shape == (18,)
    joint = corrupt(modality_set="TA", overlap_type="synchronous")
    p = extract_observable_gap_proxy(
        text_bert=joint.observation["text"], audio=joint.observation["audio"],
        vision=joint.observation["vision"], valid_masks=valid,
    )
    assert observable_gap_geometry(p, valid, normalization())[15] > 0


def test_multi_block_infeasible_short_text_is_reported_not_faked() -> None:
    text, audio, vision, valid = fixture_observation()
    with pytest.raises(ValueError, match="Multi-block infeasible"):
        corrupt_sample(
            text_bert=text, audio=audio, vision=vision, valid_masks=valid,
            source_sample_id="short", seed=1, modality_set="T", missing_ratio=0.01,
            position_type="front", shape_type="multi_block", overlap_type="synchronous",
            span_count_weights={2: 1, 3: 1, 4: 1},
        )
