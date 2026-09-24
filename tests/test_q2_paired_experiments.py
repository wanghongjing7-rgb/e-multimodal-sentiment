"""Minimal gates for order-independent Q2 events and paired initialization."""

from __future__ import annotations

import json
import inspect
import os
import random
import subprocess
import sys
from pathlib import Path

import torch

from e_multimodal_sentiment.models.backbones.mmsa_multitask import MMSAMulTMultiTask
from e_multimodal_sentiment.q2.corruption import (
    TRAIN_CORRUPTION_RNG_VERSION, corrupt_sample, sample_training_corruption_plan,
    training_corruption_seed,
)
from e_multimodal_sentiment.q2.gap_proxy import GeometryNormalization
from e_multimodal_sentiment.q2.robust_modules import CalibratedMMSAMulT

from test_q2_calibration import FakeMulT, observation
from test_q2_corruption import fixture_observation

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from q2_shared_init import SHARED_INIT_KIND, file_sha256, load_shared_init  # noqa: E402
from q2_mmsa_bootstrap import EXPECTED_COMMIT  # noqa: E402


def plan(sample_id: str, epoch: int = 1) -> dict[str, object]:
    """Use identical corruption settings for B1/B2/B3."""
    return sample_training_corruption_plan(
        global_seed=20260923, epoch=epoch, source_sample_id=sample_id,
        clean_probability=0.2, shape_probs=(0.5, 0.5),
    )


def test_stable_corruption_is_model_free_order_independent_and_versioned() -> None:
    ids = ("sample-a", "sample-b", "sample-c")
    ordered = {sample_id: plan(sample_id) for sample_id in ids}
    reversed_order = {sample_id: plan(sample_id) for sample_id in reversed(ids)}
    assert ordered == reversed_order
    for model_type in ("B1", "B2", "B3"):
        assert {sample_id: plan(sample_id) for sample_id in ids} == ordered, model_type
    assert ordered["sample-a"] == plan("sample-a")
    assert ordered["sample-a"]["rng_version"] == TRAIN_CORRUPTION_RNG_VERSION
    assert "model" not in inspect.signature(sample_training_corruption_plan).parameters
    assert training_corruption_seed(20260923, 1, "sample-a") != training_corruption_seed(20260923, 2, "sample-a")
    assert plan("sample-a", 1)["seed"] != plan("sample-a", 2)["seed"]


def test_corruption_plan_matches_fresh_process_without_global_rng_changes() -> None:
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state().clone()
    expected = plan("same-across-processes")
    assert random.getstate() == python_state
    assert torch.equal(torch.random.get_rng_state(), torch_state)
    code = (
        "import json, sys, types; sys.modules['torch'] = types.ModuleType('torch'); "
        "from e_multimodal_sentiment.q2.corruption import sample_training_corruption_plan; "
        "print(json.dumps(sample_training_corruption_plan(global_seed=20260923, epoch=1, "
        "source_sample_id='same-across-processes', clean_probability=0.2, shape_probs=(0.5, 0.5))))"
    )
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    actual = json.loads(subprocess.check_output([sys.executable, "-c", code], env=env, text=True))
    assert actual == expected


def test_full_corruption_metadata_matches_for_three_variants() -> None:
    text, audio, vision, valid = fixture_observation()
    python_state = random.getstate()
    torch_state = torch.random.get_rng_state().clone()
    events = []
    for _model_type in ("B1", "B2", "B3"):
        drawn = sample_training_corruption_plan(
            global_seed=20260923, epoch=1, source_sample_id="paired-event",
            clean_probability=0.0, shape_probs=(1.0, 0.0),
        )
        event = corrupt_sample(
            text_bert=text, audio=audio, vision=vision, valid_masks=valid,
            source_sample_id="paired-event", seed=drawn["seed"],
            modality_set=drawn["modality_set"], missing_ratio=drawn["missing_ratio"],
            position_type=drawn["position_type"], shape_type=drawn["shape_type"],
            overlap_type=drawn["overlap_type"], span_count_weights={2: 1, 3: 1, 4: 1},
        )
        events.append({key: event.metadata[key] for key in (
            "modality_set", "missing_ratio", "position_type", "shape_type",
            "num_spans", "span_list", "overlap_type",
        )})
    assert events[0] == events[1] == events[2]
    assert random.getstate() == python_state
    assert torch.equal(torch.random.get_rng_state(), torch_state)


def test_shared_base_state_and_four_clean_outputs_match(tmp_path: Path) -> None:
    initial = MMSAMulTMultiTask(FakeMulT())
    artifact_path = tmp_path / "shared_init.pt"
    torch.save({"kind": SHARED_INIT_KIND, "model": initial.state_dict(), "seed": 9,
                "mmsa_sha": EXPECTED_COMMIT, "source_commit": "test"}, artifact_path)
    expected_sha = file_sha256(artifact_path)
    text, audio, vision, tokens, valid = observation()
    outputs = {}
    states = {}
    norm = GeometryNormalization("eligible", "eligible", "eligible", 4.0, "union")
    for name in ("B0", "B1", "B2", "B3"):
        base = MMSAMulTMultiTask(FakeMulT())
        assert load_shared_init(artifact_path, base, expected_sha256=expected_sha)["seed"] == 9
        states[name] = {key: value.clone() for key, value in base.state_dict().items()}
        model = (CalibratedMMSAMulT(base, mode="ratio" if name == "B2" else "geometry",
                                    epsilon=0.2, hidden_dim=8, normalization=norm)
                 if name in ("B2", "B3") else base).eval()
        with torch.no_grad():
            outputs[name] = (model(text, audio, vision, text_bert=tokens, valid_masks=valid)
                             if name in ("B2", "B3") else model(text, audio, vision))
    for name in ("B1", "B2", "B3"):
        assert all(torch.equal(value, states["B0"][key]) for key, value in states[name].items())
        torch.testing.assert_close(outputs[name].class_logits, outputs["B0"].class_logits, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(outputs[name].regression, outputs["B0"].regression, atol=1e-6, rtol=1e-6)
