"""Compile predeclared paired smoke summaries into a factual subset CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    """Write observed cells only; never extrapolate missing manifest scenes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    args = parser.parse_args()
    if args.output_path.exists():
        parser.error("output exists; refusing to overwrite")
    files = sorted(args.paired_dir.glob("paired_V02-*_subset_clean.json"))
    if not files:
        parser.error("no paired subset summaries found")
    desired = {json.loads(path.read_text(encoding="utf-8"))["scenario_id"] for path in files}
    scenes = {}
    with args.manifest.open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            scene = item["scenario_id"]
            if scene in desired and scene not in scenes:
                scenes[scene] = {key: item[key] for key in (
                    "modality_set", "missing_ratio", "position_type", "shape_type", "overlap_type"
                )}
                if len(scenes) == len(desired):
                    break
    records = []
    for path in files:
        summary = json.loads(path.read_text(encoding="utf-8"))
        if not summary["paired_source_ids"]:
            raise ValueError(f"Unpaired source IDs in {path}")
        for name, item in summary["models"].items():
            clean, corrupted = item["clean"], item["corrupted"]
            records.append({
                "scope": "predeclared_subset_smoke_only",
                "scenario_id": summary["scenario_id"],
                **scenes[summary["scenario_id"]],
                "model": name,
                "exp_id": item["exp_id"],
                "paired_source_count": summary["paired_sample_count"],
                "accuracy": corrupted["accuracy"],
                "f1_macro": corrupted["f1_macro"],
                "f1_weighted": corrupted["f1_weighted"],
                "mae": corrupted["mae"],
                "pearson": corrupted["pearson"],
                "pearson_valid_count": corrupted["pearson_valid_count"],
                "pearson_nan_count": corrupted["pearson_nan_count"],
                "delta_f1_macro_vs_paired_clean": item["delta_f1_macro"],
                "delta_mae_vs_paired_clean": item["delta_mae"],
                "delta_pearson_vs_paired_clean": item["delta_pearson"],
                "train_epoch_seconds": item["train_epoch_seconds"],
                "inference_seconds": corrupted["elapsed_seconds"],
                "parameter_count": corrupted.get("parameter_count"),
                "checkpoint": item["checkpoint"],
            })
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps({"scenes": len(desired), "models": 4, "observed_rows": len(records),
                      "output": str(args.output_path.resolve()),
                      "scope": "subset smoke; full 265-scene grid remains pending"}))


if __name__ == "__main__":
    main()
