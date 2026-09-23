"""Read-only quick/full audit of a trusted local pickle dictionary."""
import argparse
import json
import pickle
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

AuditMode = Literal["quick", "full"]
_CHUNK_SIZE = 1_000_000


def _as_array(value: Any) -> np.ndarray | None:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    try:
        return np.asarray(value)
    except (ValueError, TypeError):
        return None


def _full_numeric_stats(array: np.ndarray, *, is_label: bool) -> dict[str, Any]:
    """Scan flat chunks so temporary boolean arrays stay bounded."""
    nan_count = inf_count = zero_count = finite_count = 0
    minimum: float | None = None
    maximum: float | None = None
    flat = array.reshape(-1)
    for offset in range(0, flat.size, _CHUNK_SIZE):
        chunk = flat[offset : offset + _CHUNK_SIZE]
        nan_count += int(np.isnan(chunk).sum())
        inf_count += int(np.isinf(chunk).sum())
        zero_count += int(np.count_nonzero(chunk == 0))
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            chunk_min = float(finite.min())
            chunk_max = float(finite.max())
            minimum = chunk_min if minimum is None else min(minimum, chunk_min)
            maximum = chunk_max if maximum is None else max(maximum, chunk_max)
            finite_count += int(finite.size)
    result: dict[str, Any] = {
        "nan": nan_count,
        "inf": inf_count,
        "zero_ratio": zero_count / flat.size if flat.size else None,
        "numeric_range": [minimum, maximum] if finite_count else None,
    }
    if is_label:
        result["label_range"] = result["numeric_range"]
    return result


def describe(value: Any, *, mode: AuditMode = "quick", is_label: bool = False) -> dict[str, Any]:
    """Describe structure; expensive content statistics run only in full mode."""
    if isinstance(value, Mapping):
        return {
            str(key): describe(
                item,
                mode=mode,
                is_label=is_label or "label" in str(key).lower(),
            )
            for key, item in value.items()
        }
    array = _as_array(value)
    if array is None:
        return {
            "type": type(value).__name__,
            "shape": None,
            "dtype": None,
            "sample_count": None,
            "note": "Ragged or unsupported value",
        }
    report: dict[str, Any] = {
        "type": type(value).__name__,
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sample_count": int(array.shape[0]) if array.ndim else 1,
    }
    if is_label:
        report["label_shape"] = list(array.shape)
    if mode == "full":
        if array.dtype.kind in "biuf":
            report.update(_full_numeric_stats(array, is_label=is_label))
        elif array.dtype.kind == "c":
            report["note"] = "Complex values: full numeric statistics are not reported."
        else:
            report["note"] = "Non-numeric values: full numeric statistics are unavailable."
    return report


def audit_pickle(
    path: str | Path,
    *,
    trusted: bool = False,
    mode: AuditMode = "quick",
) -> dict[str, Any]:
    """Load once in read-only mode and return a structured audit report."""
    if not trusted:
        raise ValueError("Only load trusted pickle files; pass trusted=True explicitly.")
    if mode not in ("quick", "full"):
        raise ValueError("mode must be 'quick' or 'full'.")
    with Path(path).open("rb") as stream:
        data = pickle.load(stream)
    if not isinstance(data, Mapping):
        raise ValueError("Expected a top-level pickle dictionary.")
    return {
        "audit_mode": mode,
        "source_name": Path(path).name,
        "top_level_keys": [str(key) for key in data],
        "splits": {
            split: {
                "present": split in data,
                "type": type(data.get(split)).__name__ if split in data else None,
                "fields": (
                    [str(key) for key in data[split]]
                    if isinstance(data.get(split), Mapping)
                    else []
                ),
            }
            for split in ("train", "valid", "test")
        },
        "fields": describe(data, mode=mode),
    }


def main() -> None:
    """Print JSON to stdout; never create or overwrite data files."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Explicit path to a trusted local pickle")
    parser.add_argument("--trusted-pickle", action="store_true", help="Acknowledge pickle execution risk")
    parser.add_argument("--mode", choices=("quick", "full"), default="quick")
    args = parser.parse_args()
    if not args.trusted_pickle:
        parser.error("Use --trusted-pickle only for files from a trusted source.")
    if args.mode == "full":
        print(
            "Full audit selected. May require substantial RAM/time for large unaligned files.",
            file=sys.stderr,
        )
    print(
        json.dumps(
            audit_pickle(args.path, trusted=True, mode=args.mode),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
