"""Exploratory full-split B0/B1 training with retained source IDs and checkpoints."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from e_multimodal_sentiment.data.adapters import Attachment2AlignedAdapter
from e_multimodal_sentiment.data.collate import collate_samples
from e_multimodal_sentiment.evaluation.metrics import compute_metrics
from e_multimodal_sentiment.models.backbones import MMSAMulTMultiTask
from e_multimodal_sentiment.q2.corruption import corrupt_sample
from e_multimodal_sentiment.q2.gap_proxy import GeometryNormalization
from e_multimodal_sentiment.q2.robust_modules import CalibratedCriterion, CalibratedMMSAMulT
from e_multimodal_sentiment.training.hooks import TaskHook
from e_multimodal_sentiment.training.losses import MultiTaskCriterion
from e_multimodal_sentiment.training.trainer import Trainer

from q2_mmsa_bootstrap import load_fixed_mmsa
from smoke_mmsa_mult import _mult_args


class PairedSamples(Dataset):
    """Pair shared aligned samples with source token IDs without copying data."""

    def __init__(self, samples, tokens) -> None:
        self.samples, self.tokens = samples, tokens

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index], torch.as_tensor(self.tokens[index])


def collate_pairs(items):
    """Expose COMMON labels and Q2 raw tokens in a single mapping."""
    batch = collate_samples([item[0] for item in items])
    return {
        "source_sample_ids": batch.sample_ids,
        "text": batch.text.features,
        "audio": batch.audio.features,
        "vision": batch.vision.features,
        "text_bert": torch.stack([item[1] for item in items]),
        "class_label": batch.classification_labels,
        "reg_label": batch.regression_labels,
        "class_label_mask": batch.classification_label_mask,
        "reg_label_mask": batch.regression_label_mask,
    }


class ContinuousCorruptionHook(TaskHook):
    """B1-only batch preprocessing; the model receives no simulation truth."""

    def __init__(self, encoder, seed: int, clean_probability: float, shape_probs: tuple[float, float]):
        self.encoder = encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.rng = random.Random(seed)
        self.enabled = False
        self.clean_probability = clean_probability
        self.shape_probs = shape_probs
        self.last_events: list[dict[str, object]] = []

    def preprocess_batch(self, batch):
        self.last_events = []
        if not self.enabled:
            return batch
        text = batch["text"].clone()
        audio = batch["audio"].clone()
        vision = batch["vision"].clone()
        tokens_batch = batch["text_bert"].clone()
        for index, source_id in enumerate(batch["source_sample_ids"]):
            if self.rng.random() < self.clean_probability:
                self.last_events.append({"source_sample_id": source_id, "clean": True})
                continue
            tokens = batch["text_bert"][index]
            valid = tokens[1] == 1
            masks = {name: valid for name in ("text", "audio", "vision")}
            modality = self.rng.choice(("T", "A", "V", "TA", "TV", "AV", "TAV"))
            ratio = self.rng.uniform(0.10, 0.50)
            shape = self.rng.choices(("single_block", "multi_block"), weights=self.shape_probs)[0]
            relation = self.rng.choice(("synchronous", "staggered"))
            event_seed = self.rng.randrange(2**31)
            try:
                event = corrupt_sample(
                    text_bert=tokens, audio=audio[index], vision=vision[index],
                    valid_masks=masks, source_sample_id=source_id, seed=event_seed,
                    modality_set=modality, missing_ratio=ratio, position_type="random",
                    shape_type=shape, overlap_type=relation, span_count_weights={2: 1, 3: 1, 4: 1},
                )
            except ValueError as error:
                # Preserve the sample and record the infeasible planned view.
                self.last_events.append({"source_sample_id": source_id, "clean": True,
                                         "infeasible_corruption": str(error), "requested_shape": shape})
                continue
            audio[index] = event.observation["audio"]
            vision[index] = event.observation["vision"]
            tokens_batch[index] = event.observation["text"]
            if "T" in modality:
                with torch.no_grad():
                    text[index] = self.encoder(event.observation["text"].unsqueeze(0).float())[0]
            self.last_events.append({**event.metadata, "clean": False})
        batch["text"], batch["audio"], batch["vision"] = text, audio, vision
        batch["text_bert"] = tokens_batch
        return batch


def forward_multitask(model, batch, diagnostics):
    """Connect COMMON Trainer to the existing three-argument wrapper."""
    del diagnostics
    if isinstance(model, CalibratedMMSAMulT):
        valid = batch["text_bert"][:, 1, :] == 1
        return model(batch["text"], batch["audio"], batch["vision"],
                     text_bert=batch["text_bert"],
                     valid_masks={name: valid for name in ("text", "audio", "vision")})
    return model(batch["text"], batch["audio"], batch["vision"])


@torch.no_grad()
def validate(model, criterion, loader, device, output_path: Path):
    """Evaluate all valid samples once and save per-source predictions."""
    model.eval()
    all_logits, all_class, all_regression, all_targets = [], [], [], []
    total_loss = 0.0
    count = 0
    started = time.perf_counter()
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["source_sample_id", "class_label", "predicted_class", "reg_label", "predicted_regression"])
        for batch in loader:
            ids = batch["source_sample_ids"]
            batch = {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}
            result = forward_multitask(model, batch, False)
            if not torch.isfinite(result.class_logits).all() or not torch.isfinite(result.regression).all():
                raise FloatingPointError("Validation predictions contain NaN/Inf")
            loss, _ = criterion(result, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Validation loss contains NaN/Inf")
            total_loss += float(loss) * len(ids)
            count += len(ids)
            all_logits.append(result.class_logits.cpu())
            all_class.append(batch["class_label"].cpu())
            all_regression.append(result.regression.cpu())
            all_targets.append(batch["reg_label"].cpu())
            for sample_id, truth_class, predicted_class, truth_reg, predicted_reg in zip(
                ids, batch["class_label"].cpu().tolist(), result.class_logits.argmax(-1).cpu().tolist(),
                batch["reg_label"].cpu().tolist(), result.regression.cpu().tolist(),
            ):
                writer.writerow([sample_id, truth_class, predicted_class, truth_reg, predicted_reg])
    metrics = compute_metrics(torch.cat(all_logits), torch.cat(all_class), torch.cat(all_regression), torch.cat(all_targets))
    metrics.update(total_loss=total_loss / count, samples=count, seconds=time.perf_counter() - started,
                   pearson_valid_count=int(math.isfinite(metrics["pearson"])),
                   pearson_nan_count=int(not math.isfinite(metrics["pearson"])))
    return metrics


def main() -> None:
    """Train B0 or B1 over complete train split and validate only on valid."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--mmsa-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", choices=("B0", "B1", "B2", "B3"), required=True)
    parser.add_argument("--exp-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--lambda-reg", type=float, default=1.0)
    parser.add_argument("--clean-probability", type=float, default=0.20)
    parser.add_argument("--shape-probs", type=float, nargs=2, default=(0.5, 0.5))
    parser.add_argument("--init-checkpoint", type=Path)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--beta", type=float)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--geometry-ratio-den", choices=("eligible", "sequence"))
    parser.add_argument("--geometry-longest-den", choices=("eligible", "sequence"))
    parser.add_argument("--geometry-center-den", choices=("eligible", "sequence"))
    parser.add_argument("--geometry-span-divisor", type=float)
    parser.add_argument("--geometry-overlap-den", choices=("union", "eligible"))
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, batch-size and lr must be positive")
    if args.output_dir.exists():
        parser.error("output directory already exists; training never overwrites prior experiments")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.model in ("B2", "B3") and any(value is None for value in (
        args.init_checkpoint, args.epsilon, args.beta, args.hidden_dim,
        args.geometry_ratio_den, args.geometry_longest_den, args.geometry_center_den,
        args.geometry_span_divisor, args.geometry_overlap_den,
    )):
        parser.error("B2/B3 require traceable init-checkpoint, epsilon, beta, hidden width and geometry normalization")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True)
    source_paths = [
        ROOT / "scripts" / "q2_train_baselines.py",
        ROOT / "scripts" / "q2_mmsa_bootstrap.py",
        ROOT / "src" / "e_multimodal_sentiment" / "q2" / "corruption.py",
        ROOT / "src" / "e_multimodal_sentiment" / "q2" / "gap_proxy.py",
        ROOT / "src" / "e_multimodal_sentiment" / "q2" / "robust_modules.py",
        ROOT / "configs" / "q2_cgrc_exploratory.yaml",
    ]
    config = vars(args).copy()
    config.update(code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                  torch=torch.__version__, cuda=torch.version.cuda, device=str(device),
                  data_version="Attachment2 aligned_50.pkl; train/valid only",
                  model_version=f"MMSA MulT + COMMON dual heads {args.model}; exploratory",
                  command=sys.argv, start_time=datetime.now().astimezone().isoformat(),
                  end_time=None,
                  source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in source_paths if path.exists()},
                  status="running")
    (args.output_dir / "config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")
    mult_cls, encoder_cls = load_fixed_mmsa(args.mmsa_root)
    adapter = Attachment2AlignedAdapter(args.data_path, trusted=True)
    train = adapter.to_samples("train")
    valid = adapter.to_samples("valid")
    tokens_train = adapter._split("train")["text_bert"]
    tokens_valid = adapter._split("valid")["text_bert"]
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(PairedSamples(train, tokens_train), batch_size=args.batch_size,
                              shuffle=True, generator=generator, collate_fn=collate_pairs)
    valid_loader = DataLoader(PairedSamples(valid, tokens_valid), batch_size=args.batch_size,
                              shuffle=False, collate_fn=collate_pairs)
    base = mult_cls(_mult_args(args.mmsa_root))
    model = MMSAMulTMultiTask(base)
    if args.model in ("B2", "B3"):
        initial = torch.load(args.init_checkpoint, map_location="cpu", weights_only=False)
        model.load_state_dict(initial["model"])
        model = CalibratedMMSAMulT(
            model, mode="ratio" if args.model == "B2" else "geometry",
            epsilon=args.epsilon, hidden_dim=args.hidden_dim,
            normalization=GeometryNormalization(
                args.geometry_ratio_den, args.geometry_longest_den,
                args.geometry_center_den, args.geometry_span_divisor,
                args.geometry_overlap_den,
            ),
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = (CalibratedCriterion(lambda_cls=args.lambda_cls, lambda_reg=args.lambda_reg,
                                     beta=args.beta) if args.model in ("B2", "B3")
                 else MultiTaskCriterion(args.lambda_cls, args.lambda_reg))
    hook = ContinuousCorruptionHook(encoder_cls(use_finetune=False).to(device), args.seed,
                                    args.clean_probability, tuple(args.shape_probs)) if args.model in ("B1", "B2", "B3") else TaskHook()
    trainer = Trainer(model, optimizer, criterion, device=str(device), hook=hook, forward_fn=forward_multitask)
    best = math.inf
    with (args.output_dir / "train_batches.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            start = time.perf_counter()
            if isinstance(hook, ContinuousCorruptionHook):
                hook.enabled = True
            for number, batch in enumerate(train_loader, 1):
                losses = trainer.train_step(batch)
                row = {"epoch": epoch, "batch": number, "source_sample_ids": batch["source_sample_ids"],
                       "losses": losses, "corruption_events": hook.last_events if isinstance(hook, ContinuousCorruptionHook) else []}
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                if number % 20 == 0:
                    log.flush()
                    print(f"epoch={epoch} batch={number}/{len(train_loader)} loss={losses['total']:.6f}", flush=True)
            if isinstance(hook, ContinuousCorruptionHook):
                hook.enabled = False
            valid_metrics = validate(trainer.model, criterion, valid_loader, device,
                                     args.output_dir / f"valid_predictions_epoch_{epoch}.csv")
            epoch_result = {"epoch": epoch, "train_batches": len(train_loader), "valid": valid_metrics,
                            "elapsed_seconds": time.perf_counter() - start}
            with (args.output_dir / "epochs.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(epoch_result, allow_nan=True) + "\n")
            state = {"model": trainer.model.state_dict(), "optimizer": optimizer.state_dict(),
                     "epoch": epoch, "seed": args.seed, "config": config, "valid": valid_metrics}
            torch.save(state, args.output_dir / "last.pt")
            if valid_metrics["total_loss"] < best:
                best = valid_metrics["total_loss"]
                torch.save(state, args.output_dir / "best_val_total_loss.pt")
            print(json.dumps(epoch_result, allow_nan=True), flush=True)
    config["status"] = "completed"
    config["end_time"] = datetime.now().astimezone().isoformat()
    (args.output_dir / "config.json").write_text(json.dumps(config, default=str, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
