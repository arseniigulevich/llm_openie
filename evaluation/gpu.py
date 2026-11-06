"""GPU selection and memory release for the model-based metrics (Entailment, SS)."""

from __future__ import annotations

import gc
import os
import subprocess


def select_cuda_device_with_most_free_memory() -> None:
    """Expose only the GPU with the most free memory (sets CUDA_VISIBLE_DEVICES).

    CUDA reads CUDA_VISIBLE_DEVICES once, when the first model is placed on a GPU, so
    this must run before any model is loaded; the runner calls it once at the start of a
    run. An explicit CUDA_VISIBLE_DEVICES set by the user is respected.
    """
    preset = os.environ.get("CUDA_VISIBLE_DEVICES")
    if preset:
        print(f"Using preconfigured CUDA_VISIBLE_DEVICES={preset}")
        return
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return  # no NVIDIA runtime: keep the default device

    best_idx, best_free = None, -1
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            idx, free = int(parts[0]), int(parts[1])
        except ValueError:
            continue
        if free > best_free:
            best_idx, best_free = idx, free
    if best_idx is None:
        return

    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)
    print(f"Auto-selected GPU {best_idx} with {best_free} MiB free (set CUDA_VISIBLE_DEVICES).")


def release_cuda_memory() -> None:
    """Best-effort CUDA memory cleanup after a model is unloaded."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
    except Exception:
        pass
