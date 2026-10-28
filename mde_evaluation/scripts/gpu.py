"""Pick the fastest available device: NVIDIA GPU (CUDA, Linux), Apple GPU (MPS, Mac) or CPU.

The scripts only call these three helpers, so they run unchanged on the Mac and on the Linux machine.
"""

import torch


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sync(device: torch.device):
    """Wait until the GPU has finished, so time measurements are correct (GPUs work asynchronously)."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def free_memory(device: torch.device):
    """Release GPU memory after a model is deleted, before loading the next one."""
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()
