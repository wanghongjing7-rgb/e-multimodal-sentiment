"""Create and verify one explicit shared MulT/dual-head initialization; no training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_multimodal_sentiment.models.backbones import MMSAMulTMultiTask
from e_multimodal_sentiment.q2.gap_proxy import GeometryNormalization
from e_multimodal_sentiment.q2.robust_modules import CalibratedMMSAMulT

from q2_mmsa_bootstrap import EXPECTED_COMMIT, load_fixed_mmsa
from smoke_mmsa_mult import _mult_args

SHARED_INIT_KIND = "q2_shared_init_v1"


def file_sha256(path: Path) -> str:
    """Hash the exact serialized artifact used by all four runs."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_shared_init(path: Path, model: MMSAMulTMultiTask, *, expected_sha256: str | None = None) -> dict:
    """Load only a tagged, fixed-MMSA base state with strict key matching."""
    actual_sha256 = file_sha256(path)
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise ValueError("shared_init SHA256 changed before loading")
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    if artifact.get("kind") != SHARED_INIT_KIND or artifact.get("mmsa_sha") != EXPECTED_COMMIT:
        raise ValueError("Expected tagged Q2 shared_init for the fixed MMSA commit")
    model.load_state_dict(artifact["model"], strict=True)
    return {"sha256": actual_sha256, "seed": artifact["seed"], "source_commit": artifact["source_commit"]}


def main() -> None:
    """Save shared weights, then compare four separately constructed eval models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--mmsa-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory exists; refusing to overwrite")
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    torch.manual_seed(args.seed)
    started = datetime.now().astimezone().isoformat()
    device = torch.device(args.device)
    mult_cls, _ = load_fixed_mmsa(args.mmsa_root)
    source_commit = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    args.output_dir.mkdir(parents=True)
    artifact_path = args.output_dir / "shared_init.pt"
    initial = MMSAMulTMultiTask(mult_cls(_mult_args(args.mmsa_root)))
    torch.save({"kind": SHARED_INIT_KIND, "model": initial.state_dict(),
                "seed": args.seed, "mmsa_sha": EXPECTED_COMMIT,
                "source_commit": source_commit}, artifact_path)
    shared_sha256 = file_sha256(artifact_path)

    with args.data_path.open("rb") as stream:
        train = pickle.load(stream)["train"]
    count = min(args.batch_size, len(train["id"]))
    if count < 1:
        raise ValueError("Empty Attachment2 train split")
    text = torch.as_tensor(train["text"][:count], dtype=torch.float32, device=device)
    audio = torch.as_tensor(train["audio"][:count], dtype=torch.float32, device=device)
    vision = torch.as_tensor(train["vision"][:count], dtype=torch.float32, device=device)
    tokens = torch.as_tensor(train["text_bert"][:count], dtype=torch.float32, device=device)
    structural = tokens[:, 1, :] == 1
    valid = {name: structural for name in ("text", "audio", "vision")}
    norm = GeometryNormalization("eligible", "eligible", "eligible", 4.0, "union")
    outputs = {}
    base_states = {}
    alpha_max = {}
    with torch.no_grad():
        for name in ("B0", "B1", "B2", "B3"):
            base = MMSAMulTMultiTask(mult_cls(_mult_args(args.mmsa_root)))
            load_shared_init(artifact_path, base, expected_sha256=shared_sha256)
            base_states[name] = {key: value.clone() for key, value in base.state_dict().items()}
            model = (CalibratedMMSAMulT(
                base, mode="ratio" if name == "B2" else "geometry",
                epsilon=args.epsilon, hidden_dim=args.hidden_dim, normalization=norm,
            ) if name in ("B2", "B3") else base).to(device).eval()
            output = (model(text, audio, vision, text_bert=tokens, valid_masks=valid)
                      if name in ("B2", "B3") else model(text, audio, vision))
            if not torch.isfinite(output.class_logits).all() or not torch.isfinite(output.regression).all():
                raise FloatingPointError(f"{name} initialization output contains NaN/Inf")
            outputs[name] = output
            if name in ("B2", "B3"):
                alpha_max[name] = float((output.auxiliary["calibration_alpha"] - 1).abs().max())
    base_state_equal = all(
        torch.equal(value, base_states["B0"][key])
        for name in ("B1", "B2", "B3")
        for key, value in base_states[name].items()
    )
    comparisons = {}
    for left, right in (("B0", "B1"), ("B1", "B2"), ("B1", "B3")):
        comparisons[f"{left}_vs_{right}"] = {
            "classification_logits_max_abs_diff": float((outputs[left].class_logits - outputs[right].class_logits).abs().max()),
            "regression_max_abs_diff": float((outputs[left].regression - outputs[right].regression).abs().max()),
        }
    tolerance = 1e-5
    passed = base_state_equal and all(
        all(value <= tolerance for value in pair.values()) for pair in comparisons.values()
    ) and all(value == 0.0 for value in alpha_max.values())
    report = {
        "status": "PASS" if passed else "FAIL",
        "source_commit": source_commit,
        "mmsa_sha": EXPECTED_COMMIT,
        "shared_init_path": str(artifact_path.resolve()),
        "shared_init_sha256": shared_sha256,
        "shared_seed": args.seed,
        "device": str(device),
        "batch_size": count,
        "source_split": "Attachment2 train",
        "source_sample_ids": [str(item) for item in train["id"][:count]],
        "epsilon": args.epsilon,
        "hidden_dim": args.hidden_dim,
        "tolerance": tolerance,
        "base_state_bitwise_equal": base_state_equal,
        "alpha_max_abs_from_one": alpha_max,
        "comparisons": comparisons,
        "start_time": started,
        "end_time": datetime.now().astimezone().isoformat(),
        "training_performed": False,
    }
    (args.output_dir / "equivalence.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report), flush=True)
    if not passed:
        raise RuntimeError("Shared initialization equivalence failed; stop before training")


if __name__ == "__main__":
    main()
