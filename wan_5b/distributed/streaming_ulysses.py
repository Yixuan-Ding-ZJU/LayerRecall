# Copyright 2024-2026 LongLive Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Chunk-level Ulysses collectives shared by streaming model paths.

The helpers in this module deliberately use the sequence-parallel state owned
by :mod:`wan_5b.distributed.sp_training`.  They do not initialize or retain a
second process group.
"""

from __future__ import annotations

import threading
import time
from numbers import Integral
from typing import Dict, Iterable, Tuple

import torch
import torch.distributed as dist

from wan_5b.distributed import sp_training as _sp_training


_YX_MAX_DEBUG_METADATA_VALUES = 4096
_YX_TELEMETRY_LOCK = threading.Lock()
_YX_TELEMETRY = {
    "collective_count": 0,
    "estimated_bytes": 0,
    "collective_time_s": 0.0,
    "operations": {},
}


def _YX_sp_context():
    if not dist.is_available() or not dist.is_initialized():
        return None, 1, 0
    configured_group = _sp_training.get_sequence_parallel_group()
    if configured_group is None:
        return None, 1, 0
    group = _sp_training.resolve_sequence_parallel_group(configured_group)
    return group, _sp_training.get_sp_world_size(), _sp_training.get_sp_rank()


def streaming_sp_info() -> Tuple[object, int, int]:
    """Return ``(group, sp_size, sp_rank)`` with disabled SP mapped to size 1."""
    return _YX_sp_context()


def streaming_sp_enabled() -> bool:
    return _YX_sp_context()[1] > 1


def local_frame_bounds(global_frames: int) -> Tuple[int, int]:
    """Return this SP rank's contiguous frame interval within one chunk."""
    global_frames = int(global_frames)
    if global_frames <= 0:
        raise ValueError(f"global_frames must be positive, got {global_frames}")
    _, world_size, rank = _YX_sp_context()
    if global_frames % world_size != 0:
        raise ValueError(
            f"global_frames ({global_frames}) must be divisible by SP size "
            f"({world_size})"
        )
    local_frames = global_frames // world_size
    start = rank * local_frames
    return start, start + local_frames


def _YX_tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _YX_estimated_collective_bytes(
    tensor: torch.Tensor,
    world_size: int,
    collective: str,
) -> int:
    """Estimate per-rank payload bytes, including sends and receives."""
    if world_size <= 1:
        return 0
    nbytes = _YX_tensor_nbytes(tensor)
    if collective == "all_to_all":
        return 2 * nbytes * (world_size - 1) // world_size
    if collective == "all_reduce":
        return 2 * nbytes * (world_size - 1) // world_size
    if collective == "all_gather":
        return 2 * nbytes * (world_size - 1)
    raise ValueError(f"Unsupported collective kind: {collective}")


def _YX_record_collective(
    operation: str,
    estimated_bytes: int,
    elapsed_s: float,
) -> None:
    with _YX_TELEMETRY_LOCK:
        _YX_TELEMETRY["collective_count"] += 1
        _YX_TELEMETRY["estimated_bytes"] += int(estimated_bytes)
        _YX_TELEMETRY["collective_time_s"] += float(elapsed_s)
        operation_stats = _YX_TELEMETRY["operations"].setdefault(
            operation,
            {
                "collective_count": 0,
                "estimated_bytes": 0,
                "collective_time_s": 0.0,
            },
        )
        operation_stats["collective_count"] += 1
        operation_stats["estimated_bytes"] += int(estimated_bytes)
        operation_stats["collective_time_s"] += float(elapsed_s)


def collective_telemetry_snapshot() -> Dict[str, object]:
    """Return a process-local copy of the streaming collective counters."""
    with _YX_TELEMETRY_LOCK:
        return {
            "collective_count": int(_YX_TELEMETRY["collective_count"]),
            "estimated_bytes": int(_YX_TELEMETRY["estimated_bytes"]),
            "collective_time_s": float(_YX_TELEMETRY["collective_time_s"]),
            "operations": {
                name: dict(values)
                for name, values in _YX_TELEMETRY["operations"].items()
            },
        }


def reset_collective_telemetry() -> None:
    with _YX_TELEMETRY_LOCK:
        _YX_TELEMETRY["collective_count"] = 0
        _YX_TELEMETRY["estimated_bytes"] = 0
        _YX_TELEMETRY["collective_time_s"] = 0.0
        _YX_TELEMETRY["operations"] = {}


def _YX_validate_dense_tensor(tensor: torch.Tensor, name: str, ndim: int) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}, got shape {tuple(tensor.shape)}")
    if any(int(size) <= 0 for size in tensor.shape):
        raise ValueError(f"{name} dimensions must be positive, got {tuple(tensor.shape)}")


def _YX_gloo_all_to_all_impl(
    tensor: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group,
    world_size: int,
    rank: int,
) -> torch.Tensor:
    """All-to-all layout via all-gather for Gloo builds lacking alltoall."""
    if int(tensor.shape[scatter_dim]) % world_size != 0:
        raise ValueError(
            f"scatter dimension {scatter_dim} with size {tensor.shape[scatter_dim]} "
            f"must be divisible by SP world size ({world_size})"
        )
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor.contiguous(), group=group)
    rank_chunks = [
        source.chunk(world_size, dim=scatter_dim)[rank].contiguous()
        for source in gathered
    ]
    return torch.cat(rank_chunks, dim=gather_dim).contiguous()


class _YXGlooAllToAllWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, scatter_dim, gather_dim, group, world_size, rank):
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.group = group
        ctx.world_size = world_size
        ctx.rank = rank
        return _YX_gloo_all_to_all_impl(
            tensor,
            scatter_dim,
            gather_dim,
            group,
            world_size,
            rank,
        )

    @staticmethod
    def backward(ctx, grad_output):
        return (
            _YX_gloo_all_to_all_impl(
                grad_output,
                ctx.gather_dim,
                ctx.scatter_dim,
                ctx.group,
                ctx.world_size,
                ctx.rank,
            ),
            None,
            None,
            None,
            None,
            None,
        )


def _YX_timed_all_to_all_with_grad(
    tensor: torch.Tensor,
    scatter_dim: int,
    gather_dim: int,
    group,
    world_size: int,
    operation: str,
) -> torch.Tensor:
    start = time.perf_counter()
    backend = str(dist.get_backend(group)).lower()
    if "gloo" in backend:
        rank = _sp_training.get_sp_rank()
        output = _YXGlooAllToAllWithGrad.apply(
            tensor,
            scatter_dim,
            gather_dim,
            group,
            world_size,
            rank,
        )
    else:
        output = _sp_training.all_to_all_with_grad(
            tensor,
            scatter_dim=scatter_dim,
            gather_dim=gather_dim,
            group=group,
        )
    elapsed_s = time.perf_counter() - start
    _YX_record_collective(
        operation,
        _YX_estimated_collective_bytes(tensor, world_size, "all_to_all"),
        elapsed_s,
    )
    return output


def ulysses_seq_to_head(tensor: torch.Tensor) -> torch.Tensor:
    """Exchange ``[B, S_local, H, D]`` to ``[B, S_global, H / P, D]``."""
    _YX_validate_dense_tensor(tensor, "tensor", ndim=4)
    group, world_size, _ = _YX_sp_context()
    if world_size == 1:
        return tensor
    num_heads = int(tensor.shape[2])
    if num_heads % world_size != 0:
        raise ValueError(
            f"head count ({num_heads}) must be divisible by SP world size "
            f"({world_size})"
        )
    return _YX_timed_all_to_all_with_grad(
        tensor,
        scatter_dim=2,
        gather_dim=1,
        group=group,
        world_size=world_size,
        operation="seq_to_head",
    )


def ulysses_packed_qkv_seq_to_head(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exchange Q/K/V from ``[B, S_local, H, D]`` in one collective.

    The returned tensors each use ``[B, S_global, H / P, D]``.  Q/K/V are
    packed along their feature dimension before invoking the shared
    autograd-safe all-to-all implementation.
    """
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        _YX_validate_dense_tensor(tensor, name, ndim=4)
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError(
            "q, k, and v must have identical [B, S_local, H, D] shapes; "
            f"got q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}"
        )
    if q.device != k.device or q.device != v.device:
        raise ValueError(
            f"q, k, and v must share a device; got {q.device}, {k.device}, {v.device}"
        )
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError(
            f"q, k, and v must share a dtype; got {q.dtype}, {k.dtype}, {v.dtype}"
        )

    group, world_size, _ = _YX_sp_context()
    if world_size == 1:
        return q, k, v
    num_heads = int(q.shape[2])
    if num_heads % world_size != 0:
        raise ValueError(
            f"QKV head count ({num_heads}) must be divisible by SP world size "
            f"({world_size})"
        )

    head_dim = int(q.shape[-1])
    packed_qkv = torch.cat((q, k, v), dim=-1)
    packed_output = _YX_timed_all_to_all_with_grad(
        packed_qkv,
        scatter_dim=2,
        gather_dim=1,
        group=group,
        world_size=world_size,
        operation="packed_qkv_seq_to_head",
    )
    q_global, k_global, v_global = packed_output.split(head_dim, dim=-1)
    return q_global, k_global, v_global


def ulysses_head_to_seq(output: torch.Tensor) -> torch.Tensor:
    """Invert Ulysses output to ``[B, S_local, H, D]`` autograd-safely."""
    _YX_validate_dense_tensor(output, "output", ndim=4)
    group, world_size, _ = _YX_sp_context()
    if world_size == 1:
        return output
    global_sequence = int(output.shape[1])
    if global_sequence % world_size != 0:
        raise ValueError(
            f"global sequence length ({global_sequence}) must be divisible by "
            f"SP world size ({world_size})"
        )
    return _YX_timed_all_to_all_with_grad(
        output,
        scatter_dim=1,
        gather_dim=2,
        group=group,
        world_size=world_size,
        operation="head_to_seq",
    )


def _YX_all_reduce_sum_in_place(
    tensor: torch.Tensor,
    group,
    world_size: int,
    operation: str,
) -> None:
    start = time.perf_counter()
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group)
    elapsed_s = time.perf_counter() - start
    _YX_record_collective(
        operation,
        _YX_estimated_collective_bytes(tensor, world_size, "all_reduce"),
        elapsed_s,
    )


class _YXAllReduceSumWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, group, world_size):
        ctx.group = group
        ctx.world_size = world_size
        output = tensor.contiguous().clone()
        _YX_all_reduce_sum_in_place(
            output,
            group,
            world_size,
            operation="all_reduce_sum_forward",
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.contiguous().clone()
        _YX_all_reduce_sum_in_place(
            grad_input,
            ctx.group,
            ctx.world_size,
            operation="all_reduce_sum_backward",
        )
        return grad_input, None, None


def sp_sum(tensor: torch.Tensor, detach: bool = False) -> torch.Tensor:
    """Elementwise sum over SP ranks with an autograd-safe backward."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.numel() == 0:
        raise ValueError("SP sum does not support empty tensors")
    source = tensor.detach() if detach else tensor
    group, world_size, _ = _YX_sp_context()
    if world_size == 1:
        return source
    if detach:
        output = source.contiguous().clone()
        _YX_all_reduce_sum_in_place(
            output,
            group,
            world_size,
            operation="all_reduce_sum_detached",
        )
        return output
    return _YXAllReduceSumWithGrad.apply(source, group, world_size)


def sp_global_mean(tensor: torch.Tensor, detach: bool = False) -> torch.Tensor:
    """Elementwise arithmetic mean over SP ranks."""
    summed = sp_sum(tensor, detach=detach)
    _, world_size, _ = _YX_sp_context()
    if world_size == 1:
        return summed
    return summed / world_size


def current_token_summary(
    tensor: torch.Tensor,
    token_dim: int = 1,
    detach: bool = False,
    tokens_per_frame: int | None = None,
) -> torch.Tensor:
    """Globally average tokens while preserving every batch entry.

    For the common ``[B, T_local, D]`` input this returns ``[B, D]``.  The
    batch dimension is never included in the local reduction or SP reduction.
    SP chunks are required to have the same local token count.
    """
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"tensor must be a torch.Tensor, got {type(tensor).__name__}")
    if tensor.ndim < 2:
        raise ValueError(
            f"current summary expects at least [B, T], got shape {tuple(tensor.shape)}"
        )
    token_dim = int(token_dim)
    if token_dim < 0:
        token_dim += tensor.ndim
    if token_dim <= 0 or token_dim >= tensor.ndim:
        raise ValueError(
            f"token_dim must name a non-batch dimension, got {token_dim} for "
            f"shape {tuple(tensor.shape)}"
        )
    if int(tensor.shape[token_dim]) <= 0:
        raise ValueError("current summary does not support an empty token dimension")
    source = tensor.detach() if detach else tensor
    if tokens_per_frame is None:
        local_batch_summary = source.mean(dim=token_dim)
        return sp_global_mean(local_batch_summary, detach=False)

    if token_dim != 1 or tensor.ndim != 3:
        raise ValueError(
            "frame-indexed current summary requires [B, T, D] with token_dim=1, "
            f"got shape={tuple(tensor.shape)}, token_dim={token_dim}"
        )
    tokens_per_frame = int(tokens_per_frame)
    if tokens_per_frame <= 0 or int(tensor.shape[1]) % tokens_per_frame != 0:
        raise ValueError(
            f"token count {tensor.shape[1]} must be divisible by "
            f"tokens_per_frame={tokens_per_frame}"
        )
    group, world_size, rank = _YX_sp_context()
    del group
    local_frames = int(tensor.shape[1]) // tokens_per_frame
    global_frames = local_frames * world_size
    per_frame = source.float().reshape(
        int(source.shape[0]), local_frames, tokens_per_frame, int(source.shape[2])
    ).mean(dim=2)
    frame_slots = per_frame.new_zeros(
        int(per_frame.shape[0]), global_frames, int(per_frame.shape[2])
    )
    start = rank * local_frames
    frame_slots[:, start:start + local_frames] = per_frame
    # Frame ownership is disjoint across ranks. The all-reduce therefore
    # reconstructs global frame order without a BF16 reduction-tree change.
    global_per_frame = sp_sum(frame_slots, detach=False)
    return global_per_frame.mean(dim=1)


def k_summary(
    k: torch.Tensor,
    detach: bool = False,
    tokens_per_frame: int | None = None,
) -> torch.Tensor:
    """Average a local ``[B, T, H, D]`` K tensor into one global ``[D]``."""
    _YX_validate_dense_tensor(k, "k", ndim=4)
    source = k.detach() if detach else k
    if tokens_per_frame is None:
        local_summary = source.mean(dim=(0, 1, 2))
        return sp_global_mean(local_summary, detach=False)

    tokens_per_frame = int(tokens_per_frame)
    if tokens_per_frame <= 0 or int(k.shape[1]) % tokens_per_frame != 0:
        raise ValueError(
            f"K token count {k.shape[1]} must be divisible by "
            f"tokens_per_frame={tokens_per_frame}"
        )
    group, world_size, rank = _YX_sp_context()
    del group
    local_frames = int(k.shape[1]) // tokens_per_frame
    global_frames = local_frames * world_size
    per_frame = source.float().reshape(
        int(source.shape[0]),
        local_frames,
        tokens_per_frame,
        int(source.shape[2]),
        int(source.shape[3]),
    ).mean(dim=(0, 2, 3))
    frame_slots = per_frame.new_zeros(global_frames, int(per_frame.shape[1]))
    start = rank * local_frames
    frame_slots[start:start + local_frames] = per_frame
    global_per_frame = sp_sum(frame_slots, detach=False)
    return global_per_frame.mean(dim=0)


def _YX_timed_all_gather(
    tensor: torch.Tensor,
    group,
    world_size: int,
    operation: str,
) -> Tuple[torch.Tensor, ...]:
    gathered = [torch.empty_like(tensor) for _ in range(world_size)]
    start = time.perf_counter()
    dist.all_gather(gathered, tensor, group=group)
    elapsed_s = time.perf_counter() - start
    _YX_record_collective(
        operation,
        _YX_estimated_collective_bytes(tensor, world_size, "all_gather"),
        elapsed_s,
    )
    return tuple(gathered)


def all_gather_detached_frames(local_frames: torch.Tensor) -> torch.Tensor:
    """Gather ``[B, F_local, C, H, W]`` frame chunks in SP-rank order."""
    _YX_validate_dense_tensor(local_frames, "local_frames", ndim=5)
    if local_frames.requires_grad:
        raise ValueError(
            "local_frames requires gradients; callers must pass local_frames.detach()"
        )
    group, world_size, _ = _YX_sp_context()
    if world_size == 1:
        return local_frames
    gathered = _YX_timed_all_gather(
        local_frames.contiguous(),
        group,
        world_size,
        operation="detached_frame_all_gather",
    )
    return torch.cat(gathered, dim=1).contiguous()


def _YX_metadata_device(group) -> torch.device:
    backend = str(dist.get_backend(group)).lower()
    if "nccl" in backend:
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL metadata checks require a CUDA device")
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _YX_metadata_tensor(
    metadata: Iterable[int] | int | torch.Tensor,
    group=None,
) -> torch.Tensor:
    device = torch.device("cpu") if group is None else _YX_metadata_device(group)
    if isinstance(metadata, torch.Tensor):
        if metadata.requires_grad:
            raise ValueError("debug metadata must not require gradients")
        if metadata.dtype not in {
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.bool,
        }:
            raise TypeError(f"debug metadata must be integer-valued, got {metadata.dtype}")
        values = metadata.detach().reshape(-1).to(device=device, dtype=torch.int64)
    elif isinstance(metadata, Integral):
        values = torch.tensor([int(metadata)], device=device, dtype=torch.int64)
    else:
        try:
            items = list(metadata)
        except TypeError as exc:
            raise TypeError("debug metadata must contain only integers") from exc
        if any(not isinstance(item, Integral) for item in items):
            raise TypeError("debug metadata must contain only integers")
        values = torch.tensor(
            [int(item) for item in items],
            device=device,
            dtype=torch.int64,
        )
    if values.numel() > _YX_MAX_DEBUG_METADATA_VALUES:
        raise ValueError(
            f"debug metadata has {values.numel()} values; maximum is "
            f"{_YX_MAX_DEBUG_METADATA_VALUES}"
        )
    return values.contiguous()


def assert_sp_metadata_consistent(
    metadata: Iterable[int] | int | torch.Tensor,
    label: str = "metadata",
) -> Tuple[Tuple[int, ...], ...]:
    """All-gather integer IDs and raise on every rank if values differ."""
    group, world_size, _ = _YX_sp_context()
    if world_size == 1:
        values = _YX_metadata_tensor(metadata)
        return (tuple(int(value) for value in values.tolist()),)

    values = _YX_metadata_tensor(metadata, group)
    length = torch.tensor([values.numel()], device=values.device, dtype=torch.int64)
    gathered_lengths = _YX_timed_all_gather(
        length,
        group,
        world_size,
        operation="metadata_length_all_gather",
    )
    lengths = tuple(int(item.item()) for item in gathered_lengths)
    if len(set(lengths)) != 1:
        raise RuntimeError(f"SP {label} lengths differ across ranks: {lengths}")
    if lengths[0] == 0:
        return tuple(() for _ in range(world_size))

    gathered_values = _YX_timed_all_gather(
        values,
        group,
        world_size,
        operation="metadata_value_all_gather",
    )
    rows = tuple(
        tuple(int(value) for value in row.to(device="cpu").tolist())
        for row in gathered_values
    )
    if any(row != rows[0] for row in rows[1:]):
        raise RuntimeError(f"SP {label} differs across ranks: {rows}")
    return rows


# Short integration aliases keep call sites readable while retaining YX-prefixed
# canonical names for this experimental module.
YX_packed_qkv_seq_to_head = ulysses_packed_qkv_seq_to_head
YX_inverse_output = ulysses_head_to_seq
YX_current_summary = current_token_summary
YX_pre_rope_k_summary = k_summary
YX_detached_frame_all_gather = all_gather_detached_frames
YX_assert_consistent_metadata = assert_sp_metadata_consistent
YX_get_collective_telemetry = collective_telemetry_snapshot


__all__ = [
    "all_gather_detached_frames",
    "YX_assert_consistent_metadata",
    "assert_sp_metadata_consistent",
    "collective_telemetry_snapshot",
    "YX_current_summary",
    "current_token_summary",
    "YX_detached_frame_all_gather",
    "YX_get_collective_telemetry",
    "YX_inverse_output",
    "k_summary",
    "local_frame_bounds",
    "YX_packed_qkv_seq_to_head",
    "YX_pre_rope_k_summary",
    "reset_collective_telemetry",
    "sp_global_mean",
    "sp_sum",
    "streaming_sp_enabled",
    "streaming_sp_info",
    "ulysses_head_to_seq",
    "ulysses_packed_qkv_seq_to_head",
    "ulysses_seq_to_head",
]
