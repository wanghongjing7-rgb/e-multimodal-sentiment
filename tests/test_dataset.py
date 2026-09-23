"""Synthetic aligned/unaligned schema, collate, adapter, and audit tests."""
import pickle

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from e_multimodal_sentiment.common.schema import ModalitySequence, SampleSchema
from e_multimodal_sentiment.data.adapters import MoseiPickleAdapter
from e_multimodal_sentiment.data.collate import collate_samples
from e_multimodal_sentiment.data.dataset import MultimodalDataset
from scripts.audit_data import audit_pickle


def sequence(length: int, dim: int, *, valid: torch.Tensor | None = None) -> ModalitySequence:
    return ModalitySequence(torch.ones(length, dim), valid_mask=valid)


def sample(lengths: tuple[int, int, int], labeled: bool = True, sample_id: str = "sample") -> SampleSchema:
    return SampleSchema(
        sample_id,
        sequence(lengths[0], 2),
        sequence(lengths[1], 3),
        sequence(lengths[2], 4),
        class_label=1 if labeled else None,
        reg_label=0.5 if labeled else None,
    )


def test_aligned_and_unaligned_schema() -> None:
    assert sample((4, 4, 4)).is_aligned
    unaligned = sample((4, 7, 10))
    assert not unaligned.is_aligned
    assert (unaligned.text.length, unaligned.audio.length, unaligned.vision.length) == (4, 7, 10)


def test_modality_sequence_validation() -> None:
    with pytest.raises(ValueError, match="shape"):
        ModalitySequence(torch.ones(3, 2), valid_mask=torch.ones(2, dtype=torch.bool))
    with pytest.raises(ValueError, match="invalid"):
        ModalitySequence(
            torch.ones(3, 2),
            valid_mask=torch.tensor([True, False, True]),
            missing_mask=torch.tensor([False, True, False]),
        )


def test_collate_independent_axes_and_padding() -> None:
    first = sample((4, 7, 10), sample_id="long")
    second = sample((2, 3, 5), sample_id="short")
    loader = DataLoader(MultimodalDataset([first, second]), batch_size=2, collate_fn=collate_samples)
    batch = next(iter(loader))
    assert batch["text"].shape == (2, 4, 2)
    assert batch["audio"].shape == (2, 7, 3)
    assert batch["vision"].shape == (2, 10, 4)
    assert batch["valid_masks"]["text"].shape == (2, 4)
    assert batch["valid_masks"]["audio"].shape == (2, 7)
    assert batch["valid_masks"]["vision"].shape == (2, 10)
    assert batch["missing_masks"]["text"].shape == (2, 4)
    assert not batch["valid_masks"]["text"][1, 2:].any()
    assert not batch["missing_masks"]["text"][1, 2:].any()


def test_unlabeled_batch_uses_none_labels() -> None:
    batch = collate_samples([sample((4, 7, 10), labeled=False)])
    assert batch["class_label"] is None
    assert batch["reg_label"] is None
    assert not batch["class_label_mask"].any()
    assert not batch["reg_label_mask"].any()


def test_audit_quick_full_and_read_only(tmp_path) -> None:
    path = tmp_path / "fixture.pkl"
    path.write_bytes(
        pickle.dumps(
            {
                "train": {
                    "text": np.array([[0.0, np.nan, np.inf]]),
                    "labels": np.array([-2.0, 1.0]),
                },
                "test": {"text": np.zeros((1, 2))},
            }
        )
    )
    before = path.read_bytes()
    quick = audit_pickle(path, trusted=True)
    full = audit_pickle(path, trusted=True, mode="full")
    assert path.read_bytes() == before
    assert quick["audit_mode"] == "quick"
    assert "nan" not in quick["fields"]["train"]["text"]
    assert quick["fields"]["train"]["text"]["sample_count"] == 1
    assert full["fields"]["train"]["text"]["nan"] == 1
    assert full["fields"]["train"]["text"]["inf"] == 1
    assert full["fields"]["train"]["labels"]["label_range"] == [-2.0, 1.0]
    with pytest.raises(ValueError, match="trusted"):
        audit_pickle(path)


def test_adapter_inspection_defers_mapping(tmp_path) -> None:
    path = tmp_path / "unaligned.pkl"
    path.write_bytes(pickle.dumps({"train": {"text": [1], "audio": [2]}}))
    adapter = MoseiPickleAdapter(path, trusted=True)
    report = adapter.inspect_schema()
    assert report["splits"]["train"]["fields"] == ["text", "audio"]
    assert "D-E-AUDIT-Q2-001" in report["mapping_status"]
    with pytest.raises(NotImplementedError, match="D-E-AUDIT-Q2-001"):
        adapter.to_samples("train")
