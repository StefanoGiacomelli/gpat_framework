"""Torch device selection utilities."""

from __future__ import annotations

import torch


def resolve_torch_device(requested: str = "auto") -> str:
    """Resolve ``auto`` to CUDA, MPS, or CPU in priority order."""

    value = (requested or "auto").strip().lower()
    if value == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available on this system")
    if value == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available on this system")
    if value not in {"cpu", "cuda", "mps"}:
        raise ValueError("Device must be one of: auto, cpu, cuda, mps")
    return value
