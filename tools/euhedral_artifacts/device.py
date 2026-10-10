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


# CUDA's default synchronization spins: a host thread waiting for the device keeps a core at 100%. Each
# process that uses the GPU asks the driver for blocking synchronization instead, before PyTorch creates the
# primary context. It must happen in each forked worker (the parent never initializes CUDA).
_BLOCKING_SYNC_PID = None
CU_CTX_SCHED_BLOCKING_SYNC = 0x4


def _request_blocking_sync() -> None:
    global _BLOCKING_SYNC_PID
    if _BLOCKING_SYNC_PID == os.getpid():
        return
    _BLOCKING_SYNC_PID = os.getpid()
    import ctypes
    try:
        driver = ctypes.CDLL("libcuda.so.1")
    except OSError:
        return
    handle = ctypes.c_int()
    if driver.cuInit(0) != 0 or driver.cuDeviceGet(ctypes.byref(handle), 0) != 0:
        return
    driver.cuDevicePrimaryCtxSetFlags(handle, CU_CTX_SCHED_BLOCKING_SYNC)


def torch():
    import torch  # imported lazily: the CPU path needs only NumPy
    if DEVICE == "cuda":
        _request_blocking_sync()
    return torch


def table(name: str, values: np.ndarray):
    if name not in _TABLES:
        _TABLES[name] = torch().from_numpy(values).to(DEVICE)
    return _TABLES[name]
