"""Check zero-initialized B2/B3 against the same real B1 MulT checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_multimodal_sentiment.models.backbones import MMSAMulTMultiTask
from e_multimodal_sentiment.q2.gap_proxy import GeometryNormalization
from e_multimodal_sentiment.q2.robust_modules import CalibratedMMSAMulT

from q2_mmsa_bootstrap import load_fixed_mmsa
from smoke_mmsa_mult import _mult_args


def main() -> None:
    """Record exact-source shape and initialization equivalence without training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--mmsa-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--hidden-dim", type=int, required=True)
    args = parser.parse_args()
    if args.output_path.exists():
        parser.error("output already exists")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    with args.data_path.open("rb") as stream:
        raw = pickle.load(stream)["valid"]
    mult_cls, _ = load_fixed_mmsa(args.mmsa_root)
    base = MMSAMulTMultiTask(mult_cls(_mult_args(args.mmsa_root)))
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    base.load_state_dict(checkpoint["model"])
    base.eval()
    text = torch.as_tensor(raw["text"][:2], dtype=torch.float32)
    audio = torch.as_tensor(raw["audio"][:2], dtype=torch.float32)
    vision = torch.as_tensor(raw["vision"][:2], dtype=torch.float32)
    tokens = torch.as_tensor(raw["text_bert"][:2], dtype=torch.float32)
    mask = tokens[:, 1, :] == 1
    valid = {name: mask for name in ("text", "audio", "vision")}
    with torch.no_grad():
        reference = base(text, audio, vision)
    records = []
    for mode in ("ratio", "geometry"):
        model = CalibratedMMSAMulT(
            base, mode=mode, epsilon=args.epsilon, hidden_dim=args.hidden_dim,
            normalization=GeometryNormalization("eligible", "eligible", "eligible", 4.0, "union"),
        ).eval()
        with torch.no_grad():
            result = model(text, audio, vision, text_bert=tokens, valid_masks=valid)
        record = {
            "mode": mode,
            "source_sample_ids": [str(value) for value in raw["id"][:2]],
            "descriptor_shape": list(result.auxiliary["observable_descriptor"].shape),
            "class_logits_shape": list(result.class_logits.shape),
            "regression_shape": list(result.regression.shape),
            "fused_hidden_shape": list(result.fused_hidden.shape),
            "alpha_max_abs_from_one": float((result.auxiliary["calibration_alpha"] - 1).abs().max()),
            "logits_max_abs_from_b1": float((result.class_logits - reference.class_logits).abs().max()),
            "regression_max_abs_from_b1": float((result.regression - reference.regression).abs().max()),
            "finite": bool(torch.isfinite(result.class_logits).all() and torch.isfinite(result.regression).all()),
        }
        records.append(record)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps({"checkpoint": str(args.checkpoint.resolve()),
                                           "epsilon": args.epsilon, "hidden_dim": args.hidden_dim,
                                           "normalization": "test-only: eligible/eligible/eligible, span divisor 4, overlap union",
                                           "records": records}, indent=2), encoding="utf-8")
    print(json.dumps(records))


if __name__ == "__main__":
    main()
