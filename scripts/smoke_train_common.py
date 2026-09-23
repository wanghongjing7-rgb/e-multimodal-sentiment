"""Run a bounded train/validation smoke loop on real Attachment2 aligned data."""
import argparse
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from smoke_mmsa_mult import _load_mult_class, _mult_args, _verify_mmsa

from e_multimodal_sentiment.data.adapters import Attachment2AlignedAdapter
from e_multimodal_sentiment.data.collate import collate_samples
from e_multimodal_sentiment.models.backbones import MMSAMulTMultiTask
from e_multimodal_sentiment.training.losses import MultiTaskCriterion
from e_multimodal_sentiment.training.trainer import Trainer


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--mmsa-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-train-batches", type=int, default=5)
    parser.add_argument("--max-valid-batches", type=int, default=3)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    return parser.parse_args()


def _multitask_forward(model, batch, return_native_diagnostics):
    del return_native_diagnostics
    return model(batch["text"], batch["audio"], batch["vision"])


def main() -> None:
    args = _parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    if args.lr <= 0:
        raise ValueError("--lr must be positive.")
    if args.max_train_batches < 1 or args.max_valid_batches < 1:
        raise ValueError("Maximum batch counts must be positive.")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    _verify_mmsa(args.mmsa_root)
    adapter = Attachment2AlignedAdapter(args.data_path, trusted=True)
    train_loader = DataLoader(
        adapter.to_samples("train"),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_samples,
    )
    valid_loader = DataLoader(
        adapter.to_samples("valid"),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_samples,
    )

    mult_class = _load_mult_class(args.mmsa_root)
    model = MMSAMulTMultiTask(mult_class(_mult_args(args.mmsa_root)))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = MultiTaskCriterion(
        lambda_cls=args.lambda_cls,
        lambda_reg=args.lambda_reg,
    )
    trainer = Trainer(
        model,
        optimizer,
        criterion,
        device=str(device),
        forward_fn=_multitask_forward,
    )

    started = time.perf_counter()
    train = trainer.train_epoch(
        train_loader,
        max_batches=args.max_train_batches,
    )
    valid = trainer.validate_epoch(
        valid_loader,
        max_batches=args.max_valid_batches,
    )
    elapsed = time.perf_counter() - started

    print(f"device: {device}")
    print(f"train batches: {train['batches']}")
    print(f"valid batches: {valid['batches']}")
    print("train:")
    print(f"  total_loss: {train['total']:.6f}")
    print(f"  classification_loss: {train['classification']:.6f}")
    print(f"  regression_loss: {train['regression']:.6f}")
    print("valid:")
    print(f"  accuracy: {valid['accuracy']:.6f}")
    print(f"  f1_macro: {valid['f1_macro']:.6f}")
    print(f"  f1_weighted: {valid['f1_weighted']:.6f}")
    print(f"  mae: {valid['mae']:.6f}")
    print(f"  pearson: {valid['pearson']:.6f}")
    print(f"elapsed time: {elapsed:.6f} s")


if __name__ == "__main__":
    main()
