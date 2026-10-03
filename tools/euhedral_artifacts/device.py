"""The quantization device: "cpu" (NumPy) or "cuda" (PyTorch), and the lookup tables on it."""

from __future__ import annotations

import os
from typing import Any

import numpy as np

DEVICE = "cpu"
_TABLES: dict[str, Any] = {}


def default_device() -> str:
    # Ask NVML, not the CUDA runtime: initializing CUDA in the parent would make every forked --jobs
    # worker fail with "Cannot re-initialize CUDA in forked subprocess".
    os.environ.setdefault("PYTORCH_NVML_BASED_CUDA_CHECK", "1")
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_device(name: str) -> None:
    global DEVICE
    DEVICE = name
    _TABLES.clear()


def torch():
    import torch  # imported lazily: the CPU path needs only NumPy
    return torch


def table(name: str, values: np.ndarray):
    if name not in _TABLES:
        _TABLES[name] = torch().from_numpy(values).to(DEVICE)
    return _TABLES[name]
