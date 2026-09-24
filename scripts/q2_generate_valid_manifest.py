"""Generate the fixed exploratory Q2 valid corruption grid, without model calls."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import Counter
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from e_multimodal_sentiment.q2.corruption import corrupt_sample

MODALITY_SETS = ("T", "A", "V", "TA", "TV", "AV", "TAV")
RATIOS = (0.0, 0.10, 0.20, 0.40, 0.60)
POSITIONS = ("front", "middle", "rear")
SHAPES = ("single_block", "multi_block")


def scenes() -> list[dict[str, object]]:
    """Return one clean scene and the complete nonzero factor grid."""
    result = [dict(modality_set="none", missing_ratio=0.0, position_type="none",
                   shape_type="none", overlap_type="none")]
    for ratio in RATIOS[1:]:
        for modality in MODALITY_SETS:
            relations = ("synchronous",) if len(modality) == 1 else ("synchronous", "staggered")
            for position in POSITIONS:
                for shape in SHAPES:
                    for relation in relations:
                        result.append(dict(modality_set=modality, missing_ratio=ratio,
                                           position_type=position, shape_type=shape,
                                           overlap_type=relation))
    return result


def main() -> None:
    """Write immutable-by-convention JSONL events and a summary beside them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--span-weights", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    args = parser.parse_args()
    if any(weight < 0 for weight in args.span_weights) or sum(args.span_weights) == 0:
        parser.error("--span-weights must be nonnegative with positive total")
    if args.output_path.exists():
        parser.error("output already exists; this script never overwrites a manifest")
    weights = dict(zip((2, 3, 4), args.span_weights))
    with args.data_path.open("rb") as stream:
        valid = pickle.load(stream)["valid"]
    grid = scenes()
    counts: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with args.output_path.open("w", encoding="utf-8") as stream:
        for sample_index, source_id in enumerate(valid["id"]):
            text = torch.as_tensor(valid["text_bert"][sample_index])
            audio = torch.as_tensor(valid["audio"][sample_index])
            vision = torch.as_tensor(valid["vision"][sample_index])
            attention_valid = text[1] == 1
            valid_masks = {name: attention_valid for name in ("text", "audio", "vision")}
            for scene_index, scene in enumerate(grid):
                seed = (args.seed + sample_index * 1000003 + scene_index * 9176) % (2**31)
                row = {
                    "source_sample_id": str(source_id),
                    "sample_index": sample_index,
                    "scenario_id": f"V02-{scene_index:03d}",
                    "seed": seed,
                    **scene,
                }
                if scene_index == 0:
                    row.update(status="ready", num_spans={name: 0 for name in valid_masks},
                               span_list={name: [] for name in valid_masks})
                else:
                    try:
                        event = corrupt_sample(
                            text_bert=text, audio=audio, vision=vision, valid_masks=valid_masks,
                            source_sample_id=str(source_id), seed=seed,
                            modality_set=scene["modality_set"], missing_ratio=scene["missing_ratio"],
                            position_type=scene["position_type"], shape_type=scene["shape_type"],
                            overlap_type=scene["overlap_type"], span_count_weights=weights,
                        )
                        row.update(status="ready", num_spans=event.metadata["num_spans"],
                                   span_list=event.metadata["span_list"])
                    except ValueError as error:
                        row.update(status="infeasible", reason=str(error), num_spans=None, span_list=None)
                        reasons[str(error)] += 1
                counts[row["status"]] += 1
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = {
        "manifest": str(args.output_path.resolve()),
        "source_data": str(args.data_path.resolve()),
        "seed": args.seed,
        "span_weights": weights,
        "samples": len(valid["id"]),
        "scenes": len(grid),
        "rows": len(valid["id"]) * len(grid),
        "status_counts": dict(counts),
        "infeasible_reasons": dict(reasons),
        "expected_forward_per_model_if_all_ready": counts["ready"],
        "complete_grid_forward_per_model": len(valid["id"]) * len(grid),
    }
    summary_path = args.output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
