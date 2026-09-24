"""Inference-only frozen L0W over audited recovered Attachment3 dense inputs."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "outputs/q2"))

from q2_architecture_sprint import LiteAligned

CHECKPOINT = ROOT / "outputs/q2/architecture_sprint/EXP-E-Q2-LITE-WCE-001/best_val_total_loss.pt"
INPUT = ROOT / "outputs/q2/attachment3_dense_recovered.npz"
INPUT_METADATA = ROOT / "outputs/q2/attachment3_dense_recovered.metadata.json"
OUTPUT = ROOT / "outputs/q2/final_inference/recovered_l0w"
CLASS_MAPPING = {0: "Negative", 1: "Neutral", 2: "Positive"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    if not all(path.is_file() for path in (CHECKPOINT, INPUT, INPUT_METADATA)):
        raise FileNotFoundError("checkpoint or recovered audited input missing")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state = torch.load(CHECKPOINT, map_location=device, weights_only=False)
    if state["epoch"] != 11 or state["config"]["model"] != "L0W":
        raise ValueError("frozen L0W checkpoint identity mismatch")
    source = np.load(INPUT, allow_pickle=False)
    ids = [str(value) for value in source["sample_ids"].tolist()]
    text = np.asarray(source["text"], dtype=np.float32)
    audio = np.asarray(source["audio"], dtype=np.float32)
    vision = np.asarray(source["vision"], dtype=np.float32)
    expected_shapes = {"text": (30, 50, 768), "audio": (30, 50, 74), "vision": (30, 50, 35)}
    actual_shapes = {"text": text.shape, "audio": audio.shape, "vision": vision.shape}
    if actual_shapes != expected_shapes or len(ids) != 30:
        raise ValueError(f"recovered input shape/count mismatch: {actual_shapes}, ids={len(ids)}")
    if len(set(ids)) != len(ids) or not all(np.isfinite(values).all() for values in (text, audio, vision)):
        raise ValueError("recovered inputs failed duplicate/finite check")
    model = LiteAligned().to(device)
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    with torch.no_grad():
        prediction = model(torch.from_numpy(text).to(device), torch.from_numpy(audio).to(device),
                           torch.from_numpy(vision).to(device))
        logits = prediction.class_logits.cpu()
        probabilities = torch.softmax(logits, dim=-1)
        classes = logits.argmax(dim=-1)
        regression = prediction.regression.cpu()
    nonfinite = int((~torch.isfinite(logits)).any(dim=1).sum() +
                    (~torch.isfinite(probabilities)).any(dim=1).sum() +
                    (~torch.isfinite(regression)).sum())
    invalid_class = sum(int(value) not in CLASS_MAPPING for value in classes.tolist())
    output_ids = list(ids)
    duplicates = len(output_ids) - len(set(output_ids))
    missing = len(set(ids) - set(output_ids))
    if nonfinite or invalid_class or duplicates or missing:
        raise ValueError("prediction QC failed")
    OUTPUT.mkdir(parents=True)
    prediction_path = OUTPUT / "attachment3_predictions.csv"
    fields = ["sample_id", "logit_negative", "logit_neutral", "logit_positive",
              "probability_negative", "probability_neutral", "probability_positive",
              "predicted_class", "predicted_sentiment_label", "regression_prediction"]
    with prediction_path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for index, sample_id in enumerate(ids):
            predicted = int(classes[index])
            writer.writerow({"sample_id": sample_id,
                             "logit_negative": float(logits[index, 0]),
                             "logit_neutral": float(logits[index, 1]),
                             "logit_positive": float(logits[index, 2]),
                             "probability_negative": float(probabilities[index, 0]),
                             "probability_neutral": float(probabilities[index, 1]),
                             "probability_positive": float(probabilities[index, 2]),
                             "predicted_class": predicted,
                             "predicted_sentiment_label": CLASS_MAPPING[predicted],
                             "regression_prediction": float(regression[index])})
    with prediction_path.open(encoding="utf-8-sig", newline="") as stream:
        written = list(csv.DictReader(stream))
    qc = {"duplicates": duplicates, "missing": missing, "nonfinite": nonfinite,
          "invalid_class": invalid_class, "output_rows": len(written),
          "input_rows": len(ids), "row_count_matches": len(written) == len(ids)}
    if not qc["row_count_matches"]:
        raise ValueError("written CSV row count mismatch")
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    timestamp = datetime.now().astimezone().isoformat()
    recovery = json.loads(INPUT_METADATA.read_text(encoding="utf-8"))
    metadata = {"status": "PASS", "git_sha": head,
                "checkpoint_path": str(CHECKPOINT.resolve()), "checkpoint_sha256": sha256(CHECKPOINT),
                "checkpoint_epoch": state["epoch"],
                "model_version": "L0W Lite-Aligned + weighted CE; frozen best epoch 11",
                "feature_version": "aligned_50", "class_mapping": {str(k): v for k, v in CLASS_MAPPING.items()},
                "sample_count": len(ids), "timestamp": timestamp, "device": str(device),
                "input_shapes": {key: list(value) for key, value in actual_shapes.items()},
                "text_interface_recovery": recovery,
                "recovered_input_path": str(INPUT.resolve()), "recovered_input_sha256": sha256(INPUT),
                "regression_postprocessing": "none; raw model output", "model_eval": True,
                "torch_no_grad": True, "training_performed": False, "qc": qc}
    metadata_path = OUTPUT / "inference_metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUTPUT / "run.log").write_text(
        f"{timestamp} loaded frozen L0W epoch 11 checkpoint\n"
        f"{timestamp} loaded audited recovered Attachment3 dense inputs; rows={len(ids)}\n"
        f"{timestamp} model.eval and torch.no_grad inference complete; device={device}\n"
        f"{timestamp} QC PASS {json.dumps(qc)}\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "rows": len(written), "qc": qc,
                      "csv": str(prediction_path.resolve()), "metadata": str(metadata_path.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
