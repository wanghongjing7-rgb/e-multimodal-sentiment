"""Import fixed MMSA source modules without optional package-level dependencies."""

from __future__ import annotations

import importlib
import subprocess
import sys
import types
from pathlib import Path

EXPECTED_COMMIT = "a94e65d07fa1ae0d44e552390074b29b0898edfd"


def load_fixed_mmsa(mmsa_root: Path):
    """Return the unchanged upstream MULT and BertTextEncoder classes."""
    actual = subprocess.check_output(
        ["git", "-C", str(mmsa_root), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != EXPECTED_COMMIT:
        raise RuntimeError(f"MMSA commit mismatch: expected {EXPECTED_COMMIT}, got {actual}")
    for name, directory in (
        ("MMSA", mmsa_root / "src" / "MMSA"),
        ("MMSA.models", mmsa_root / "src" / "MMSA" / "models"),
        ("MMSA.models.subNets", mmsa_root / "src" / "MMSA" / "models" / "subNets"),
        ("MMSA.models.singleTask", mmsa_root / "src" / "MMSA" / "models" / "singleTask"),
    ):
        package = types.ModuleType(name)
        package.__path__ = [str(directory)]
        sys.modules.setdefault(name, package)
    encoder = importlib.import_module("MMSA.models.subNets.BertTextEncoder").BertTextEncoder
    sys.modules["MMSA.models.subNets"].BertTextEncoder = encoder
    mult = importlib.import_module("MMSA.models.singleTask.MULT").MULT
    return mult, encoder
