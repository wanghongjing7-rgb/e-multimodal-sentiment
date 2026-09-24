"""Summarize only observed B0–B3 clean and one fixed-scene paired smoke."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_multimodal_sentiment.evaluation.metrics import compute_metrics

NAMES = {
    "B0": "EXP-E-Q2-MULT-CLEAN-001",
    "B1": "EXP-E-Q2-MULT-CMISS-001",
    "B2": "EXP-E-Q2-RATIO-001",
    "B3": "EXP-E-Q2-CGRC-001",
}


def main() -> None:
    """Check paired IDs and report real metrics without model selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, required=True)
    parser.add_argument("--scenario-id", required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    if args.output_path.exists():
        parser.error("output exists; refusing overwrite")
    summary = {"scope": "one fixed scenario smoke; incomplete Q2-05/Q2-06",
               "scenario_id": args.scenario_id, "models": {}, "paired_source_ids": True}
    reference_ids = None
    for name, exp_id in NAMES.items():
        experiment = args.experiments_dir / exp_id
        evaluation = args.eval_dir / f"{name}_{args.scenario_id}"
        epoch = json.loads((experiment / "epochs.jsonl").read_text(encoding="utf-8").splitlines()[-1])
        corrupted = json.loads((evaluation / "metrics.json").read_text(encoding="utf-8"))
        with (evaluation / "predictions.csv").open(newline="", encoding="utf-8") as stream:
            ids = [row["source_sample_id"] for row in csv.DictReader(stream)]
        if reference_ids is None:
            reference_ids = ids
        elif ids != reference_ids:
            raise ValueError(f"{name} source IDs are not paired with previous models")
        # Corruption may be infeasible for short sequences. Compare against
        # clean predictions for exactly the same source IDs, not all 728.
        with (experiment / f"valid_predictions_epoch_{epoch['epoch']}.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            by_id = {row["source_sample_id"]: row for row in csv.DictReader(stream)}
        selected = [by_id[source_id] for source_id in ids]
        predicted_class = torch.tensor([int(row["predicted_class"]) for row in selected])
        clean = compute_metrics(
            class_logits=torch.nn.functional.one_hot(predicted_class, num_classes=3).float(),
            class_labels=torch.tensor([int(row["class_label"]) for row in selected]),
            regression=torch.tensor([float(row["predicted_regression"]) for row in selected]),
            reg_labels=torch.tensor([float(row["reg_label"]) for row in selected]),
        )
        summary["models"][name] = {
            "exp_id": exp_id,
            "checkpoint": str((experiment / "best_val_total_loss.pt").resolve()),
            "clean": clean,
            "clean_full_valid": epoch["valid"],
            "corrupted": corrupted,
            "delta_f1_macro": corrupted["f1_macro"] - clean["f1_macro"],
            "delta_mae": corrupted["mae"] - clean["mae"],
            "delta_pearson": corrupted["pearson"] - clean["pearson"],
            "train_epoch_seconds": epoch["elapsed_seconds"],
        }
    summary["paired_sample_count"] = len(reference_ids)
    summary["remaining_scenes_note"] = "Only one of 265 scenes was evaluated; no worst-case or average-grid claim."
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=True), encoding="utf-8")
    print(json.dumps({name: {"clean_f1": item["clean"]["f1_macro"],
                             "corrupted_f1": item["corrupted"]["f1_macro"],
                             "clean_mae": item["clean"]["mae"],
                             "corrupted_mae": item["corrupted"]["mae"]}
                      for name, item in summary["models"].items()}))


if __name__ == "__main__":
    main()
