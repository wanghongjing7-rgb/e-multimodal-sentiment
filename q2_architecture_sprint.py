"""Aligned Q2 A1/L0 exploratory runs using the established A0 validation rule."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts"), str(ROOT / "outputs" / "q2")]

from e_multimodal_sentiment.data.adapters import Attachment2AlignedAdapter
from e_multimodal_sentiment.models.api import ModelOutput
from e_multimodal_sentiment.models.backbones import MMSAMulTMultiTask
from e_multimodal_sentiment.training.losses import MultiTaskCriterion
from e_multimodal_sentiment.training.trainer import Trainer
from q2_mmsa_bootstrap import EXPECTED_COMMIT, load_fixed_mmsa
from q2_shared_init import file_sha256, load_shared_init
from q2_train_baselines import PairedSamples, collate_pairs, forward_multitask, validate
from q2_train_convergence import classification_diagnostics, class_distribution, write_distribution
from smoke_mmsa_mult import _mult_args


class LiteAligned(nn.Module):
    def __init__(self):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(dim, out) for dim, out in ((768, 128), (74, 64), (35, 64))])
        self.encoders = nn.ModuleList([nn.GRU(dim, 64, batch_first=True, bidirectional=True) for dim in (128, 64, 64)])
        self.attention = nn.ModuleList([nn.Linear(128, 1) for _ in range(3)])
        self.gate = nn.Sequential(nn.Linear(384, 3), nn.Sigmoid())
        self.shared = nn.Sequential(nn.Linear(384, 128), nn.ReLU(), nn.Dropout(0.1))
        self.classification_head = nn.Linear(128, 3)
        self.regression_head = nn.Linear(128, 1)

    def forward(self, text, audio, vision):
        pooled = []
        for values, projection, encoder, attention in zip((text, audio, vision), self.projections, self.encoders, self.attention):
            encoded, _ = encoder(torch.relu(projection(values)))
            scores = torch.softmax(attention(encoded).squeeze(-1), dim=1)
            pooled.append((encoded * scores.unsqueeze(-1)).sum(dim=1))
        joined = torch.cat(pooled, dim=-1)
        reliability = self.gate(joined)
        fused = torch.cat([item * reliability[:, index:index + 1] for index, item in enumerate(pooled)], dim=-1)
        hidden = self.shared(fused)
        return ModelOutput(self.classification_head(hidden), self.regression_head(hidden).squeeze(-1),
                           text_hidden=pooled[0], audio_hidden=pooled[1], vision_hidden=pooled[2],
                           fused_hidden=hidden, fusion_weights=reliability)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("A1", "L0", "L0W"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--mmsa-root", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output directory exists; refusing to overwrite")
    if (args.seed, args.batch_size, args.lr, args.max_epochs, args.patience) != (20260923, 32, 1e-4, 15, 3):
        parser.error("predeclared protocol requires seed=20260923, batch=32, lr=1e-4, epochs=15, patience=3")
    if not torch.cuda.is_available() or torch.cuda.get_device_name(0) != "NVIDIA GeForce RTX 4090 D":
        parser.error("designated RTX 4090 D unavailable")
    if not args.data_path.is_file() or not args.init_checkpoint.is_file():
        parser.error("confirmed data or shared initialization missing")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = torch.device("cuda:0")
    args.output_dir.mkdir(parents=True)
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    source_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config = dict(model=args.model, data_path=str(args.data_path.resolve()), mmsa_root=str(args.mmsa_root.resolve()),
                  init_checkpoint=str(args.init_checkpoint.resolve()), shared_init_sha256=file_sha256(args.init_checkpoint),
                  output_dir=str(args.output_dir.resolve()), seed=args.seed, batch_size=args.batch_size, lr=args.lr,
                  max_epochs=args.max_epochs, patience=args.patience, code_commit=head, source_sha256=source_hash,
                  mmsa_commit=EXPECTED_COMMIT, torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0), optimizer="Adam", weight_decay=0,
                  scheduler=None, lambda_cls=1, lambda_reg=1,
                  train_classification_loss="weighted CE" if args.model in ("A1", "L0W") else "ordinary CE",
                  validation_classification_loss="ordinary CE", regression_loss="MSE",
                  early_stop_metric="minimum unweighted validation total loss", start_time=datetime.now().astimezone().isoformat(),
                  status="running")
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    adapter = Attachment2AlignedAdapter(args.data_path, trusted=True)
    distribution = class_distribution(adapter)
    write_distribution(args.output_dir, distribution)
    train, valid = adapter.to_samples("train"), adapter.to_samples("valid")
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(PairedSamples(train, adapter._split("train")["text_bert"]), batch_size=32,
                              shuffle=True, generator=generator, num_workers=0, collate_fn=collate_pairs)
    valid_loader = DataLoader(PairedSamples(valid, adapter._split("valid")["text_bert"]), batch_size=32,
                              shuffle=False, num_workers=0, collate_fn=collate_pairs)
    counts = torch.tensor([distribution["train"][name]["count"] for name in ("Negative", "Neutral", "Positive")], dtype=torch.float32)
    weights = counts.rsqrt()
    weights /= weights.mean()
    config["class_counts"] = counts.int().tolist()
    config["class_weights"] = weights.tolist() if args.model in ("A1", "L0W") else None
    config["weight_formula"] = "inverse sqrt(train count), normalized to mean 1" if args.model in ("A1", "L0W") else None
    if args.model == "A1":
        mult_cls, _ = load_fixed_mmsa(args.mmsa_root)
        model = MMSAMulTMultiTask(mult_cls(_mult_args(args.mmsa_root)))
        shared = load_shared_init(args.init_checkpoint, model, expected_sha256=config["shared_init_sha256"])
        if shared["source_commit"] != "9a44ba5c9c6035905de5585909c93f2d8279ce8c":
            raise ValueError("unexpected shared initialization provenance")
    else:
        model = LiteAligned()
    config["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=0)
    train_criterion = MultiTaskCriterion(1, 1)
    if args.model in ("A1", "L0W"):
        train_criterion.classification_loss = nn.CrossEntropyLoss(weight=weights.to(device))
    valid_criterion = MultiTaskCriterion(1, 1)
    trainer = Trainer(model, optimizer, train_criterion, device=str(device), forward_fn=forward_multitask)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    fields = ["epoch", "train_total_loss", "train_cls_loss", "train_reg_loss", "valid_total_loss",
              "accuracy", "macro_f1", "weighted_f1", "mae", "pearson", "learning_rate",
              "optimizer_steps", "wall_time_seconds"]
    with (args.output_dir / "epoch_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        csv.DictWriter(stream, fieldnames=fields).writeheader()
    best, best_epoch, bad, steps, history = math.inf, None, 0, 0, []
    started = time.perf_counter()
    with (args.output_dir / "train_batches.jsonl").open("w", encoding="utf-8") as batch_log:
        for epoch in range(1, args.max_epochs + 1):
            epoch_started = time.perf_counter()
            totals = dict(total=0., classification=0., regression=0.)
            sample_count = 0
            for number, batch in enumerate(train_loader, 1):
                size = len(batch["source_sample_ids"])
                losses = trainer.train_step(batch)
                steps += 1
                sample_count += size
                for key in totals:
                    totals[key] += losses[key] * size
                batch_log.write(json.dumps(dict(epoch=epoch, batch=number, source_sample_ids=batch["source_sample_ids"], losses=losses)) + "\n")
            prediction_path = args.output_dir / f"valid_predictions_epoch_{epoch}.csv"
            metrics = validate(trainer.model, valid_criterion, valid_loader, device, prediction_path)
            diagnostics = classification_diagnostics(prediction_path)
            diagnostic_path = args.output_dir / f"classification_diagnostics_epoch_{epoch}.json"
            diagnostic_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
            row = dict(epoch=epoch, train_total_loss=totals["total"] / sample_count,
                       train_cls_loss=totals["classification"] / sample_count,
                       train_reg_loss=totals["regression"] / sample_count,
                       valid_total_loss=metrics["total_loss"], accuracy=metrics["accuracy"], macro_f1=metrics["macro_f1"],
                       weighted_f1=metrics["weighted_f1"], mae=metrics["mae"], pearson=metrics["pearson"],
                       learning_rate=args.lr, optimizer_steps=steps, wall_time_seconds=time.perf_counter() - epoch_started)
            history.append(row)
            with (args.output_dir / "epoch_metrics.csv").open("a", newline="", encoding="utf-8") as stream:
                csv.DictWriter(stream, fieldnames=fields).writerow(row)
            with (args.output_dir / "epochs.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            improved = metrics["total_loss"] < best
            if improved:
                best, best_epoch, bad = metrics["total_loss"], epoch, 0
                shutil.copy2(diagnostic_path, args.output_dir / "classification_diagnostics_best.json")
            else:
                bad += 1
            shutil.copy2(diagnostic_path, args.output_dir / "classification_diagnostics_last.json")
            state = dict(model=trainer.model.state_dict(), optimizer=optimizer.state_dict(), epoch=epoch,
                         seed=args.seed, config=config.copy(), valid=metrics, diagnostics=diagnostics)
            torch.save(state, args.output_dir / "last.pt")
            if improved:
                torch.save(state, args.output_dir / "best_val_total_loss.pt")
            print(json.dumps(dict(model=args.model, **row, best_epoch=best_epoch, non_improving_epochs=bad)), flush=True)
            if bad >= args.patience:
                break
    config.update(status="early_stopped" if epoch < args.max_epochs else "completed_max_epochs",
                  end_time=datetime.now().astimezone().isoformat(), best_epoch=best_epoch, stop_epoch=epoch,
                  best_val_total_loss=best, optimizer_steps=steps, elapsed_seconds=time.perf_counter() - started)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (args.output_dir / "result.json").write_text(json.dumps(dict(config=config, class_distribution=distribution, history=history), indent=2), encoding="utf-8")
    for name in ("last.pt", "best_val_total_loss.pt"):
        path = args.output_dir / name
        state = torch.load(path, map_location="cpu", weights_only=False)
        state["config"] = config.copy()
        torch.save(state, path)


if __name__ == "__main__":
    main()
