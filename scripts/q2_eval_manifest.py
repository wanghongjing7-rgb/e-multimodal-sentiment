"""Evaluate a fixed manifest scene; subset results remain exploratory smoke."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_multimodal_sentiment.evaluation.metrics import compute_metrics
from e_multimodal_sentiment.models.backbones import MMSAMulTMultiTask
from e_multimodal_sentiment.q2.corruption import mask_aligned_dense_text
from e_multimodal_sentiment.q2.gap_proxy import GeometryNormalization
from e_multimodal_sentiment.q2.robust_modules import CalibratedMMSAMulT

from q2_mmsa_bootstrap import load_fixed_mmsa
from smoke_mmsa_mult import _mult_args


def main() -> None:
    """Apply recorded spans without resampling and retain per-source outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--mmsa-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory exists; refusing to overwrite")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device(args.device)
    rows = []
    with args.manifest.open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["scenario_id"] == args.scenario_id:
                rows.append(row)
    if not rows:
        raise ValueError(f"Scenario {args.scenario_id} absent from manifest")
    with args.data_path.open("rb") as stream:
        valid = pickle.load(stream)["valid"]
    mult_cls, _ = load_fixed_mmsa(args.mmsa_root)
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = state["config"]
    model = MMSAMulTMultiTask(mult_cls(_mult_args(args.mmsa_root)))
    if config["model"] in ("B2", "B3"):
        model = CalibratedMMSAMulT(
            model, mode="ratio" if config["model"] == "B2" else "geometry",
            epsilon=config["epsilon"], hidden_dim=config["hidden_dim"],
            normalization=GeometryNormalization(
                config["geometry_ratio_den"], config["geometry_longest_den"],
                config["geometry_center_den"], config["geometry_span_divisor"],
                config["geometry_overlap_den"],
            ),
        )
    model = model.to(device)
    model.load_state_dict(state["model"])
    model.eval()
    args.output_dir.mkdir(parents=True)
    logits, classes, predicted, targets = [], [], [], []
    alpha_rows: list[list[float]] = []
    counts = {"ready": 0, "infeasible": 0}
    started = time.perf_counter()
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_sample_id", "scenario_id", "class_label", "predicted_class",
                         "reg_label", "predicted_regression", "alpha_T", "alpha_A", "alpha_V"])
        with torch.no_grad():
            for row in rows:
                counts[row["status"]] += 1
                if row["status"] != "ready":
                    continue
                index = row["sample_index"]
                if str(valid["id"][index]) != row["source_sample_id"]:
                    raise ValueError("Manifest source_sample_id does not match valid data")
                text = torch.as_tensor(valid["text"][index], dtype=torch.float32, device=device)
                tokens = torch.as_tensor(valid["text_bert"][index], dtype=torch.float32, device=device).clone()
                audio = torch.as_tensor(valid["audio"][index], dtype=torch.float32, device=device).clone()
                vision = torch.as_tensor(valid["vision"][index], dtype=torch.float32, device=device).clone()
                spans = row["span_list"]
                for start, stop in spans["audio"]:
                    audio[start:stop] = 0
                for start, stop in spans["vision"]:
                    vision[start:stop] = 0
                if spans["text"]:
                    for start, stop in spans["text"]:
                        tokens[0, start:stop] = 100
                    text = mask_aligned_dense_text(text, spans["text"])
                if isinstance(model, CalibratedMMSAMulT):
                    structural = (tokens[1] == 1).unsqueeze(0)
                    output = model(
                        text.unsqueeze(0), audio.unsqueeze(0), vision.unsqueeze(0),
                        text_bert=tokens.unsqueeze(0),
                        valid_masks={name: structural for name in ("text", "audio", "vision")},
                    )
                else:
                    output = model(text.unsqueeze(0), audio.unsqueeze(0), vision.unsqueeze(0))
                if not torch.isfinite(output.class_logits).all() or not torch.isfinite(output.regression).all():
                    raise FloatingPointError("Prediction contains NaN/Inf")
                truth_class = int(valid["classification_labels"][index])
                truth_reg = float(valid["regression_labels"][index])
                logits.append(output.class_logits.cpu())
                classes.append(truth_class)
                predicted.append(float(output.regression.item()))
                targets.append(truth_reg)
                alpha = (output.auxiliary["calibration_alpha"][0].cpu().tolist()
                         if isinstance(model, CalibratedMMSAMulT) else [None, None, None])
                if isinstance(model, CalibratedMMSAMulT):
                    alpha_rows.append(alpha)
                writer.writerow([row["source_sample_id"], row["scenario_id"], truth_class,
                                 int(output.class_logits.argmax(-1).item()), truth_reg, predicted[-1], *alpha])
    if not logits:
        raise ValueError("No ready samples to evaluate")
    metrics = compute_metrics(torch.cat(logits), torch.tensor(classes, dtype=torch.int64),
                              torch.tensor(predicted, dtype=torch.float32),
                              torch.tensor(targets, dtype=torch.float32))
    metrics.update(scenario_id=args.scenario_id, manifest=str(args.manifest.resolve()),
                   checkpoint=str(args.checkpoint.resolve()), samples=len(logits),
                   status_counts=counts, elapsed_seconds=time.perf_counter() - started,
                   pearson_valid_count=int(math.isfinite(metrics["pearson"])),
                   pearson_nan_count=int(not math.isfinite(metrics["pearson"])),
                   scope="single fixed scenario smoke; not full robustness grid")
    if alpha_rows:
        alpha_tensor = torch.tensor(alpha_rows)
        metrics["alpha_summary"] = {
            name: {"min": float(alpha_tensor[:, index].min()),
                   "max": float(alpha_tensor[:, index].max()),
                   "mean": float(alpha_tensor[:, index].mean())}
            for index, name in enumerate(("text", "audio", "vision"))
        }
    metrics["parameter_count"] = sum(parameter.numel() for parameter in model.parameters())
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps(metrics, allow_nan=True), flush=True)


if __name__ == "__main__":
    main()
