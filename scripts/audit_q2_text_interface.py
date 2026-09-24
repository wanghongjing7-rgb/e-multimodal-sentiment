"""Read-only, one-off audit of Attachment2/3 text and fixed MMSA interfaces.

The script never writes data or downloads weights. Set HF_HUB_OFFLINE and
TRANSFORMERS_OFFLINE before running; model checks use cached weights only.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import platform
import subprocess
import sys
import traceback
import types
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def jsonable(value: Any) -> Any:
    """Convert NumPy scalars and arrays to JSON-compatible values."""
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def channel_summary(values: np.ndarray, channel: int) -> dict[str, Any]:
    """Summarize a token channel without changing its data."""
    flat = values[:, channel, :].reshape(-1)
    finite = flat[np.isfinite(flat)]
    unique, counts = np.unique(finite, return_counts=True)
    top = np.argsort(counts)[-8:][::-1]
    return {
        "min": float(finite.min()) if finite.size else None,
        "max": float(finite.max()) if finite.size else None,
        "unique_count": int(unique.size),
        "common": [[float(unique[i]), int(counts[i])] for i in top],
        "nan": int(np.isnan(flat).sum()),
        "inf": int(np.isinf(flat).sum()),
        "integer_valued": bool(np.all(finite == np.floor(finite))),
    }


def runs(indices: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open contiguous runs of selected indices."""
    if not len(indices):
        return []
    cuts = np.flatnonzero(np.diff(indices) != 1) + 1
    return [(int(part[0]), int(part[-1]) + 1) for part in np.split(indices, cuts)]


def text_structure(values: np.ndarray) -> dict[str, Any]:
    """Separate CLS/SEP, tail padding and zero runs inside those markers."""
    counts: Counter[str] = Counter()
    spans: dict[str, list[list[int]]] = {name: [] for name in ("id_zero", "attention_zero", "all_zero", "unk_100")}
    lengths: list[int] = []
    tail_lengths: list[int] = []
    for sample in values:
        token, attention, segment = sample
        sep = np.flatnonzero(token == 102)
        counts["cls_first"] += int(token[0] == 101)
        counts["sep_present"] += int(bool(len(sep)))
        # Last SEP is the source's explicit end marker, not an inferred zero boundary.
        end = int(sep[-1]) if len(sep) else int(np.flatnonzero(np.any(sample != 0, axis=0))[-1]) if np.any(sample) else 0
        lengths.append(end + 1)
        tail_lengths.append(sample.shape[1] - end - 1)
        counts["tail_nonzero"] += int(np.any(sample[:, end + 1 :] != 0))
        interior = np.arange(1, end)
        conditions = {
            "id_zero": token[interior] == 0,
            "attention_zero": attention[interior] == 0,
            "all_zero": (token[interior] == 0) & (attention[interior] == 0) & (segment[interior] == 0),
            "unk_100": token[interior] == 100,
        }
        for name, condition in conditions.items():
            current = runs(interior[condition])
            counts[f"{name}_samples"] += int(bool(current))
            counts[f"{name}_positions"] += int(condition.sum())
            spans[name].extend([[start, stop] for start, stop in current])
    return {
        "samples": int(len(values)),
        "counts": dict(counts),
        "sequence_length": {"min": min(lengths), "max": max(lengths), "median": float(np.median(lengths))},
        "tail_padding": {"min": min(tail_lengths), "max": max(tail_lengths), "median": float(np.median(tail_lengths))},
        "spans": {
            name: {
                "count": len(items),
                "longest": max((stop - start for start, stop in items), default=0),
                "length_histogram": dict(sorted(Counter(stop - start for start, stop in items).items())),
                "start_histogram": dict(sorted(Counter(start for start, _ in items).items())),
                "examples": items[:12],
            }
            for name, items in spans.items()
        },
    }


def audit_block(block: dict[str, Any], name: str) -> dict[str, Any]:
    """Audit one in-memory split or concatenated Attachment3 set."""
    result: dict[str, Any] = {"name": name, "keys": sorted(block)}
    result["fields"] = {
        key: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype)}
        for key, value in block.items()
    }
    text = np.asarray(block["text_bert"])
    result["channels"] = [channel_summary(text, index) for index in range(3)]
    result["structure"] = text_structure(text)
    return result


def tensor_summary(value: Any) -> dict[str, Any]:
    """Summarize a finite tensor without printing its contents."""
    import torch

    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std(unbiased=False)),
        "nan": int(torch.isnan(value).sum()),
        "inf": int(torch.isinf(value).sum()),
    }


def comparisons(dense: Any, encoded: Any, attention: np.ndarray) -> dict[str, Any]:
    """Compare two tensors over all and attention-valid positions."""
    import torch
    import torch.nn.functional as F

    def one(mask: Any) -> dict[str, Any]:
        left = dense[mask]
        right = encoded[mask]
        diff = (left - right).abs()
        cosine = F.cosine_similarity(left, right, dim=-1, eps=1e-8)
        return {
            "positions": int(left.shape[0]),
            "mean_abs_diff": float(diff.mean()),
            "max_abs_diff": float(diff.max()),
            "mse": float(((left - right) ** 2).mean()),
            "timestep_cosine": {
                "mean": float(cosine.mean()),
                "median": float(cosine.median()),
                "min": float(cosine.min()),
                "max": float(cosine.max()),
            },
        }

    all_mask = torch.ones(dense.shape[:2], dtype=torch.bool)
    valid_mask = torch.from_numpy(attention == 1)
    flat_cos = F.cosine_similarity(dense.flatten(1), encoded.flatten(1), dim=-1, eps=1e-8)
    return {
        "dense": tensor_summary(dense),
        "encoded": tensor_summary(encoded),
        "all": one(all_mask),
        "attention_valid": one(valid_mask),
        "sample_flatten_cosine": flat_cos.tolist(),
        "dense_norm_per_timestep": {"mean": float(dense.norm(dim=-1).mean()), "median": float(dense.norm(dim=-1).median())},
        "encoded_norm_per_timestep": {"mean": float(encoded.norm(dim=-1).mean()), "median": float(encoded.norm(dim=-1).median())},
    }


def model_audit(mmsa_root: Path, project_root: Path, a2: dict[str, np.ndarray], a3: dict[str, np.ndarray]) -> dict[str, Any]:
    """Run fixed MMSA BERT on the same small real batches, then COMMON smoke."""
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    result: dict[str, Any] = {"offline": True}
    try:
        import torch
        import transformers

        result["torch"] = torch.__version__
        result["transformers"] = transformers.__version__
        result["device"] = "cpu"
        # Load the exact upstream source modules without executing MMSA's broad
        # package __init__, which imports optional training dependencies.
        for name, directory in (
            ("MMSA", mmsa_root / "src" / "MMSA"),
            ("MMSA.models", mmsa_root / "src" / "MMSA" / "models"),
            ("MMSA.models.subNets", mmsa_root / "src" / "MMSA" / "models" / "subNets"),
            ("MMSA.models.singleTask", mmsa_root / "src" / "MMSA" / "models" / "singleTask"),
        ):
            package = types.ModuleType(name)
            package.__path__ = [str(directory)]
            sys.modules.setdefault(name, package)
        sys.path.insert(0, str(mmsa_root / "src"))
        sys.path.insert(0, str(project_root / "src"))
        encoder_module = __import__("MMSA.models.subNets.BertTextEncoder", fromlist=["BertTextEncoder"])
        sys.modules["MMSA.models.subNets"].BertTextEncoder = encoder_module.BertTextEncoder
        encoder = encoder_module.BertTextEncoder(use_finetune=False).eval()
        with torch.no_grad():
            a2_input = torch.from_numpy(a2["text_bert"].astype(np.float32))
            a3_input = torch.from_numpy(a3["text_bert"].astype(np.float32))
            a2_encoded = encoder(a2_input)
            a3_encoded = encoder(a3_input)
        result["bert"] = {
            "a2_raw_shape": list(a2["text_bert"].shape),
            "a2_canonical_shape": list(a2_input.shape),
            "a2_output": tensor_summary(a2_encoded),
            "a3_raw_shape": list(a3["text_bert"].shape),
            "a3_canonical_shape": list(a3_input.shape),
            "a3_output": tensor_summary(a3_encoded),
        }
        dense = torch.from_numpy(a2["text"].astype(np.float32))
        result["comparison"] = comparisons(dense, a2_encoded, a2["text_bert"][:, 1, :])
        try:
            from e_multimodal_sentiment.common.schema import BatchSchema, ModalitySequence
            from e_multimodal_sentiment.integrations.mmsa import batch_to_mmsa_mult_inputs
            from e_multimodal_sentiment.models.backbones import MMSAMulTMultiTask

            sys.path.insert(0, str(project_root / "scripts"))
            from smoke_mmsa_mult import _load_mult_class, _mult_args, _verify_mmsa

            _verify_mmsa(mmsa_root)

            def sequence(features: Any) -> ModalitySequence:
                valid = torch.ones(features.shape[:2], dtype=torch.bool)
                # An explicit Q2 mask cannot be proven solely from feature zeros.
                return ModalitySequence(features, valid, valid.clone(), torch.zeros_like(valid))

            batch = BatchSchema(
                sample_ids=[f"audit-{index}" for index in range(len(a3_encoded))],
                text=sequence(a3_encoded.float()),
                audio=sequence(torch.from_numpy(a3["audio"].astype(np.float32))),
                vision=sequence(torch.from_numpy(a3["vision"].astype(np.float32))),
                text_representation="dense",
            )
            bridge = batch_to_mmsa_mult_inputs(batch)
            model = MMSAMulTMultiTask(_load_mult_class(mmsa_root)(_mult_args(mmsa_root))).eval()
            with torch.no_grad():
                output = model(bridge.text, bridge.audio, bridge.video)
            result["common_smoke"] = {
                "text": tensor_summary(bridge.text),
                "audio": tensor_summary(bridge.audio),
                "vision": tensor_summary(bridge.video),
                "class_logits": tensor_summary(output.class_logits),
                "regression": tensor_summary(output.regression),
                "fused_hidden": tensor_summary(output.fused_hidden),
                "mask_note": "All-true valid/observed and all-false missing for shape smoke only; no inferred Q2 mask.",
            }
        except Exception as error:  # preserve independent BERT result
            result["common_smoke_error"] = f"{type(error).__name__}: {error}"
    except Exception as error:
        result["bert_error"] = f"{type(error).__name__}: {error}"
        result["bert_error_trace_tail"] = traceback.format_exc().splitlines()[-5:]
    return result


def main() -> None:
    """Audit exactly the paths supplied by the caller and print JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attachment2", type=Path, required=True)
    parser.add_argument("--attachment3-dir", type=Path, required=True)
    parser.add_argument("--mmsa-root", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    if not 2 <= args.samples <= 4:
        parser.error("--samples must be 2..4")
    root = Path(__file__).resolve().parents[1]
    output: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "project_commit": subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        "mmsa_commit": subprocess.check_output(["git", "-C", str(args.mmsa_root), "rev-parse", "HEAD"], text=True).strip(),
        "attachment2": str(args.attachment2.resolve()),
        "attachment3_dir": str(args.attachment3_dir.resolve()),
    }
    with args.attachment2.open("rb") as stream:
        a2 = pickle.load(stream)
    output["attachment2_top_keys"] = sorted(a2)
    output["attachment2_audit"] = {split: audit_block(a2[split], split) for split in ("train", "valid", "test")}
    a2_sample = {
        "text_bert": np.array(a2["train"]["text_bert"][: args.samples], copy=True),
        "text": np.array(a2["train"]["text"][: args.samples], copy=True),
    }
    del a2
    gc.collect()
    files = sorted(args.attachment3_dir.glob("*.pkl"))
    output["attachment3_files"] = [str(path.resolve()) for path in files]
    blocks: list[dict[str, Any]] = []
    output["attachment3_file_schema"] = []
    for path in files:
        with path.open("rb") as stream:
            item = pickle.load(stream)
        if sorted(item) != ["test"]:
            raise ValueError(f"Unexpected Attachment3 top-level keys in {path}: {list(item)}")
        block = item["test"]
        output["attachment3_file_schema"].append({
            "file": str(path.resolve()),
            "top_keys": sorted(item),
            "fields": {key: {"shape": list(np.asarray(value).shape), "dtype": str(np.asarray(value).dtype)} for key, value in block.items()},
        })
        blocks.append(block)
    combined = {key: np.concatenate([np.asarray(block[key]) for block in blocks]) for key in ("text_bert", "audio", "vision")}
    output["attachment3_audit"] = audit_block(combined, "attachment3_aligned_all")
    a3_sample = {key: values[: args.samples].copy() for key, values in combined.items()}
    output["model_audit"] = model_audit(args.mmsa_root, root, a2_sample, a3_sample)
    print(json.dumps(jsonable(output), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
