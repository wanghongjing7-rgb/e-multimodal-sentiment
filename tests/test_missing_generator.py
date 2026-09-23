"""Contiguous missingness tests on independent modality time axes."""
import pytest
import torch

from e_multimodal_sentiment.q2.missing_generator import (
    generate_multimodal_missing,
    generate_single_modality_missing,
)


def test_single_modality_contiguous_and_reproducible() -> None:
    features = torch.ones(2, 10, 3)
    valid = torch.ones(2, 10, dtype=torch.bool)
    valid[:, 8:] = False
    first = generate_single_modality_missing(
        valid, features=features, position=2, duration=3, unit="steps", seed=7, modality="audio"
    )
    second = generate_single_modality_missing(
        valid, features=features, position=2, duration=3, unit="steps", seed=7, modality="audio"
    )
    torch.testing.assert_close(first[0], second[0])
    assert torch.equal(first[1], second[1])
    assert not (first[1] & ~valid).any()
    for row in first[1]:
        indices = row.nonzero().flatten()
        assert len(indices) == 3
        assert torch.equal(indices[1:] - indices[:-1], torch.ones(2, dtype=torch.long))


def test_ratio_maps_to_each_native_time_axis() -> None:
    features = {
        "text": torch.ones(50, 2),
        "audio": torch.ones(500, 3),
        "vision": torch.ones(500, 4),
    }
    valid = {name: torch.ones(value.shape[0], dtype=torch.bool) for name, value in features.items()}
    result = generate_multimodal_missing(
        valid,
        ["text", "audio"],
        position=0.4,
        duration=0.2,
        unit="ratio",
        seed=42,
        features=features,
    )
    assert result.missing_masks["text"].sum().item() == 10
    assert result.missing_masks["audio"].sum().item() == 100
    assert result.missing_masks["vision"].sum().item() == 0
    assert [event["length"] for event in result.events] == [10, 100]
    assert {event["unit"] for event in result.events} == {"ratio"}
    assert torch.equal(features["text"], torch.ones_like(features["text"]))
    assert not result.modified_features["text"][result.missing_masks["text"]].any()


def test_missing_cannot_cross_invalid_hole() -> None:
    features = torch.ones(5, 2)
    valid = torch.tensor([True, True, False, True, True])
    with pytest.raises(ValueError, match="No fully valid"):
        generate_single_modality_missing(
            valid, features=features, position=0, duration=3, unit="steps"
        )


@pytest.mark.parametrize(
    "position,duration,unit",
    [(0, 0, "steps"), (-1, 1, "steps"), (0.5, 0, "ratio"), (1.2, 0.2, "ratio")],
)
def test_invalid_requests(position, duration, unit) -> None:
    with pytest.raises(ValueError):
        generate_single_modality_missing(
            torch.ones(5, dtype=torch.bool),
            features=torch.ones(5, 2),
            position=position,
            duration=duration,
            unit=unit,
        )
