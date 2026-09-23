"""Minimal Attachment2 aligned adapter and fixed MMSA MulT bridge tests."""
import pickle

import numpy as np
import torch

from e_multimodal_sentiment.data.adapters import Attachment2AlignedAdapter
from e_multimodal_sentiment.data.collate import collate_samples
from e_multimodal_sentiment.integrations.mmsa import batch_to_mmsa_mult_inputs


def _split(prefix: str) -> dict[str, object]:
    text = np.ones((2, 50, 768), dtype=np.float64)
    audio = np.ones((2, 50, 74), dtype=np.float64)
    vision = np.ones((2, 50, 35), dtype=np.float64)
    text[0, 7] = 0
    audio[0, 8] = 0
    vision[0, 9] = 0
    return {
        "text": text,
        "audio": audio,
        "vision": vision,
        "classification_labels": np.array([[0], [2]], dtype=np.int32),
        "regression_labels": np.array([[0.25], [-0.5]], dtype=np.float64),
        "id": [f"{prefix}-0", f"{prefix}-1"],
    }


def test_attachment2_aligned_adapter_dtype_and_masks(tmp_path) -> None:
    path = tmp_path / "attachment2.pkl"
    path.write_bytes(pickle.dumps({split: _split(split) for split in ("train", "valid", "test")}))

    samples = Attachment2AlignedAdapter(path, trusted=True).to_samples("train")
    first = samples[0]
    assert first.sample_id == "train-0"
    assert first.is_aligned
    assert first.text.features.dtype == torch.float32
    assert first.audio.features.dtype == torch.float32
    assert first.vision.features.dtype == torch.float32
    assert isinstance(first.classification_label, int)
    assert isinstance(first.regression_label, float)
    assert first.text.valid_mask.all()
    assert not first.text.observed_mask[7]
    assert not first.text.missing_mask.any()
    assert not first.text.available_mask[7]


def test_mmsa_mult_bridge_shapes_and_order(tmp_path) -> None:
    path = tmp_path / "attachment2.pkl"
    path.write_bytes(pickle.dumps({split: _split(split) for split in ("train", "valid", "test")}))
    batch = collate_samples(Attachment2AlignedAdapter(path, trusted=True).to_samples("valid"))

    inputs = batch_to_mmsa_mult_inputs(batch)
    assert inputs.text.shape == (2, 50, 768)
    assert inputs.audio.shape == (2, 50, 74)
    assert inputs.video.shape == (2, 50, 35)
    assert inputs.text.dtype == inputs.audio.dtype == inputs.video.dtype == torch.float32
    assert batch.classification_labels.dtype == torch.int64
    assert batch.regression_labels.dtype == torch.float32
    assert not inputs.text[0, 7].any()
