"""Run one real Attachment2 aligned batch through the fixed MMSA MulT."""
import argparse
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from e_multimodal_sentiment.data.adapters import Attachment2AlignedAdapter
from e_multimodal_sentiment.data.collate import collate_samples
from e_multimodal_sentiment.integrations.mmsa import batch_to_mmsa_mult_inputs
from e_multimodal_sentiment.models.backbones import MMSAMulTMultiTask

MMSA_COMMIT = "a94e65d07fa1ae0d44e552390074b29b0898edfd"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "valid", "test"), default="test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mmsa-root", type=Path, required=True)
    parser.add_argument("--multitask", action="store_true")
    return parser.parse_args()


def _verify_mmsa(mmsa_root: Path) -> None:
    model_path = mmsa_root / "src" / "MMSA" / "models" / "singleTask" / "MULT.py"
    config_path = mmsa_root / "src" / "MMSA" / "config" / "config_regression.json"
    if not model_path.is_file() or not config_path.is_file():
        raise FileNotFoundError("--mmsa-root does not contain the expected MMSA source tree.")
    result = subprocess.run(
        ["git", "-C", str(mmsa_root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip() != MMSA_COMMIT:
        raise RuntimeError(
            f"MMSA commit mismatch: expected {MMSA_COMMIT}, got {result.stdout.strip()}."
        )


def _mult_args(mmsa_root: Path) -> SimpleNamespace:
    config_path = mmsa_root / "src" / "MMSA" / "config" / "config_regression.json"
    with config_path.open(encoding="utf-8") as stream:
        config = json.load(stream)
    aligned = config["datasetCommonParams"]["mosei"]["aligned"]
    mult = config["mult"]
    values: dict[str, Any] = {}
    values.update(aligned)
    values.update(mult["commonParams"])
    values.update(mult["datasetParams"]["mosei"])
    values["train_mode"] = "regression"
    return SimpleNamespace(**values)


def _load_mult_class(mmsa_root: Path):
    mmsa_src = mmsa_root / "src"
    if str(mmsa_src) not in sys.path:
        sys.path.insert(0, str(mmsa_src))
    module = importlib.import_module("MMSA.models.singleTask.MULT")
    return module.MULT


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    _verify_mmsa(args.mmsa_root)

    samples = Attachment2AlignedAdapter(args.data_path, trusted=True).to_samples(args.split)
    loader = DataLoader(
        samples,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_samples,
    )
    batch = next(iter(loader))
    inputs = batch_to_mmsa_mult_inputs(batch)

    mult_class = _load_mult_class(args.mmsa_root)
    model = mult_class(_mult_args(args.mmsa_root))
    if args.multitask:
        model = MMSAMulTMultiTask(model)
    model.eval()
    started = time.perf_counter()
    with torch.no_grad():
        output = model(inputs.text, inputs.audio, inputs.video)
    elapsed = time.perf_counter() - started

    print(f"text shape: {tuple(inputs.text.shape)}")
    print(f"audio shape: {tuple(inputs.audio.shape)}")
    print(f"video shape: {tuple(inputs.video.shape)}")
    print(f"output type: {type(output).__name__}")
    if args.multitask:
        print(f"class_logits shape: {tuple(output.class_logits.shape)}")
        print(f"regression shape: {tuple(output.regression.shape)}")
        print(f"fused_hidden shape: {tuple(output.fused_hidden.shape)}")
    else:
        prediction = output["M"]
        print(f"output shape: {tuple(prediction.shape)}")
        print(f"output dtype: {prediction.dtype}")
    print(f"forward elapsed time: {elapsed:.6f} s")


if __name__ == "__main__":
    main()
