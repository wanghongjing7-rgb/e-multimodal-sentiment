"""Use the previously audited fixed MMSA BertTextEncoder to encode Attachment3."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attachment3-dir", type=Path, required=True)
    parser.add_argument("--mmsa-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--bert-snapshot", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.metadata.exists():
        parser.error("output exists; refusing to overwrite")
    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        parser.error("offline Hugging Face environment is required")
    mmsa_commit = subprocess.check_output(["git", "-C", str(args.mmsa_root), "rev-parse", "HEAD"], text=True).strip()
    if mmsa_commit != "a94e65d07fa1ae0d44e552390074b29b0898edfd":
        raise RuntimeError(f"fixed MMSA commit mismatch: {mmsa_commit}")
    files = sorted(args.attachment3_dir.glob("*.pkl"))
    if len(files) != 30:
        raise ValueError(f"expected 30 aligned Attachment3 files, got {len(files)}")
    ids, tokens, audio, vision = [], [], [], []
    for path in files:
        with path.open("rb") as stream:
            item = pickle.load(stream)
        if sorted(item) != ["test"] or sorted(item["test"]) != ["audio", "text_bert", "vision"]:
            raise ValueError(f"unexpected schema: {path}")
        block = item["test"]
        ids.append(path.stem)
        tokens.append(np.asarray(block["text_bert"], dtype=np.float32))
        audio.append(np.asarray(block["audio"], dtype=np.float32))
        vision.append(np.asarray(block["vision"], dtype=np.float32))
    text_bert = np.concatenate(tokens)
    audio_values = np.concatenate(audio)
    vision_values = np.concatenate(vision)
    if text_bert.shape != (30, 3, 50) or audio_values.shape != (30, 50, 74) or vision_values.shape != (30, 50, 35):
        raise ValueError("Attachment3 aligned shape mismatch")
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate Attachment3 sample IDs")

    for name, directory in (
        ("MMSA", args.mmsa_root / "src" / "MMSA"),
        ("MMSA.models", args.mmsa_root / "src" / "MMSA" / "models"),
        ("MMSA.models.subNets", args.mmsa_root / "src" / "MMSA" / "models" / "subNets"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules.setdefault(name, package)
    sys.path.insert(0, str(args.mmsa_root / "src"))
    encoder_module = __import__("MMSA.models.subNets.BertTextEncoder", fromlist=["BertTextEncoder"])
    encoder = encoder_module.BertTextEncoder(use_finetune=False).eval()
    with torch.no_grad():
        dense = encoder(torch.from_numpy(text_bert)).cpu().numpy().astype(np.float32, copy=False)
    if dense.shape != (30, 50, 768) or not np.isfinite(dense).all():
        raise ValueError("recovered dense text failed shape/finite check")
    if not np.isfinite(audio_values).all() or not np.isfinite(vision_values).all():
        raise ValueError("Attachment3 audio/vision contains nonfinite values")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, sample_ids=np.asarray(ids), text=dense,
                        audio=audio_values, vision=vision_values, text_bert=text_bert)
    metadata = {
        "status": "PASS", "mmsa_commit": mmsa_commit,
        "encoder_source": str((args.mmsa_root / "src/MMSA/models/subNets/BertTextEncoder.py").resolve()),
        "encoder_class": "MMSA.models.subNets.BertTextEncoder.BertTextEncoder",
        "pretrained": "bert-base-uncased", "bert_snapshot": str(args.bert_snapshot.resolve()),
        "bert_snapshot_commit": args.bert_snapshot.name, "offline": True,
        "sample_count": len(ids), "sample_ids": ids,
        "shapes": {"text_bert": list(text_bert.shape), "text": list(dense.shape),
                   "audio": list(audio_values.shape), "vision": list(vision_values.shape)},
        "finite": {"text": bool(np.isfinite(dense).all()), "audio": bool(np.isfinite(audio_values).all()),
                   "vision": bool(np.isfinite(vision_values).all())},
        "source_files": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in files],
    }
    args.metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"status": "PASS", "sample_count": len(ids), "shapes": metadata["shapes"],
                      "output": str(args.output), "output_sha256": sha256(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
