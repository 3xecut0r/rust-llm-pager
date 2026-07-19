from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KVBlockStoreStats:
    gpu_to_cpu_bytes: int = 0
    cpu_to_gpu_bytes: int = 0
    gpu_to_cpu_copies: int = 0
    cpu_to_gpu_copies: int = 0
    gpu_to_cpu_sec: float = 0.0
    cpu_to_gpu_sec: float = 0.0


def tensor_nbytes(tensor: torch.Tensor) -> int:
    """Return how many bytes a tensor occupies."""
    return tensor.numel() * tensor.element_size()


def kv_nbytes(key: torch.Tensor, value: torch.Tensor) -> int:
    """Return the combined byte size of a key/value tensor pair."""
    return tensor_nbytes(key) + tensor_nbytes(value)


class KVBlockStore:
    # Runtime-agnostic primitive that physically moves KV-cache blocks
    # between GPU and CPU memory. Knows nothing about HuggingFace, only
    # about (key, value) tensor pairs indexed by block id.

    def __init__(self, tokens_per_block: int):
        self.tokens_per_block = tokens_per_block

        self.gpu_blocks: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.cpu_blocks: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

        self.stats = KVBlockStoreStats()

    def put_gpu(self, block_id: int, key: torch.Tensor, value: torch.Tensor) -> None:
        """Register a block that already lives on the GPU."""
        if not key.is_cuda or not value.is_cuda:
            raise ValueError("put_gpu expects CUDA tensors.")

        if block_id in self.cpu_blocks:
            raise ValueError(f"Block {block_id} already exists on CPU.")

        self.gpu_blocks[block_id] = (key.detach().contiguous(), value.detach().contiguous())

    def has_gpu(self, block_id: int) -> bool:
        return block_id in self.gpu_blocks

    def has_cpu(self, block_id: int) -> bool:
        return block_id in self.cpu_blocks

    def offload_to_cpu(self, block_id: int) -> None:
        """Move a block from GPU to CPU and record how long it took."""
        if block_id not in self.gpu_blocks:
            raise KeyError(f"Block {block_id} is not on GPU.")

        key, value = self.gpu_blocks.pop(block_id)

        if key.is_cuda:
            torch.cuda.synchronize(key.device)

        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)

        started.record()
        cpu_key = key.to("cpu", non_blocking=False).contiguous()
        cpu_value = value.to("cpu", non_blocking=False).contiguous()
        ended.record()

        torch.cuda.synchronize(key.device)

        self.cpu_blocks[block_id] = (cpu_key, cpu_value)

        self.stats.gpu_to_cpu_bytes += kv_nbytes(key, value)
        self.stats.gpu_to_cpu_copies += 1
        self.stats.gpu_to_cpu_sec += started.elapsed_time(ended) / 1000.0

    def load_to_gpu(self, block_id: int, device: torch.device | str) -> None:
        """Move a block from CPU to GPU and record how long it took."""
        if block_id not in self.cpu_blocks:
            raise KeyError(f"Block {block_id} is not on CPU.")

        key, value = self.cpu_blocks.pop(block_id)
        device = torch.device(device)

        if device.type != "cuda":
            raise ValueError("load_to_gpu expects a CUDA device.")

        torch.cuda.synchronize(device)

        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)

        started.record()
        gpu_key = key.to(device, non_blocking=False).contiguous()
        gpu_value = value.to(device, non_blocking=False).contiguous()
        ended.record()

        torch.cuda.synchronize(device)

        self.gpu_blocks[block_id] = (gpu_key, gpu_value)

        self.stats.cpu_to_gpu_bytes += kv_nbytes(key, value)
        self.stats.cpu_to_gpu_copies += 1
        self.stats.cpu_to_gpu_sec += started.elapsed_time(ended) / 1000.0

    def ensure_gpu(self, block_id: int, device: torch.device | str) -> None:
        """Load a block onto the GPU if it isn't already there."""
        if self.has_gpu(block_id):
            return

        self.load_to_gpu(block_id, device)

    def ensure_cpu(self, block_id: int) -> None:
        """Offload a block to the CPU if it isn't already there."""
        if self.has_cpu(block_id):
            return

        self.offload_to_cpu(block_id)

    def resident_gpu_bytes(self) -> int:
        return sum(kv_nbytes(key, value) for key, value in self.gpu_blocks.values())

    def resident_cpu_bytes(self) -> int:
        return sum(kv_nbytes(key, value) for key, value in self.cpu_blocks.values())

    def gpu_block_ids(self) -> list[int]:
        return sorted(self.gpu_blocks)

    def cpu_block_ids(self) -> list[int]:
        return sorted(self.cpu_blocks)

    def summary(self) -> dict:
        """Return resident bytes and cumulative transfer stats as a plain dict."""
        return {
            "gpu_blocks": len(self.gpu_blocks),
            "cpu_blocks": len(self.cpu_blocks),
            "resident_gpu_bytes": self.resident_gpu_bytes(),
            "resident_cpu_bytes": self.resident_cpu_bytes(),
            "gpu_to_cpu_bytes": self.stats.gpu_to_cpu_bytes,
            "cpu_to_gpu_bytes": self.stats.cpu_to_gpu_bytes,
            "gpu_to_cpu_copies": self.stats.gpu_to_cpu_copies,
            "cpu_to_gpu_copies": self.stats.cpu_to_gpu_copies,
            "gpu_to_cpu_sec": self.stats.gpu_to_cpu_sec,
            "cpu_to_gpu_sec": self.stats.cpu_to_gpu_sec,
        }

    def apply_tiers(self, tiers: list[int], device: torch.device | str) -> dict[str, list[int]]:
        """
        Move each block to the tier the pager assigned it: 0 is GPU, anything
        else is CPU for now (a disk tier will get its own number later).
        """
        device = torch.device(device)

        moved_to_gpu: list[int] = []
        moved_to_cpu: list[int] = []

        for block_id in sorted(set(self.gpu_blocks) | set(self.cpu_blocks)):
            desired_tier = tiers[block_id] if block_id < len(tiers) else 2

            if desired_tier == 0:
                if not self.has_gpu(block_id):
                    self.ensure_gpu(block_id, device)
                    moved_to_gpu.append(block_id)
            elif not self.has_cpu(block_id):
                self.ensure_cpu(block_id)
                moved_to_cpu.append(block_id)

        return {"to_gpu": moved_to_gpu, "to_cpu": moved_to_cpu}

    def get_gpu(self, block_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        if block_id not in self.gpu_blocks:
            raise KeyError(f"Block {block_id} is not on GPU.")

        return self.gpu_blocks[block_id]
