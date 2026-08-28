"""Gradient math helpers for replicated LayerRecall parameters under streaming SP.

The intended order is:

1. Normalize each rank's prediction ``local_sum`` by an SP-global, detached
   valid count.
2. Scale regularization that is replicated on every SP rank by ``1 / sp_size``.
3. Run immediate backward on every rank.
4. Sum replicated LayerRecall gradients over WORLD and divide by ``dp_size``.
5. Clip the synchronized gradients, if clipping is enabled.

``sp_group=None`` deliberately means SP=1 for the loss-side helpers.  In
contrast, ``world_group=None`` in the gradient helper means the default WORLD
process group, matching ``torch.distributed`` conventions.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Iterable

import torch
import torch.distributed as dist


__all__ = [
    "ReplicatedGradientSyncResult",
    "clip_synced_grad_norm_",
    "normalize_sp_prediction_local_sum",
    "scale_replicated_regularization",
    "sp_global_detached_count",
    "sp_global_detached_sum_count",
    "sync_replicated_layer_recall_gradients_",
]


@dataclass(frozen=True)
class ReplicatedGradientSyncResult:
    """Summary of an in-place WORLD gradient synchronization."""

    world_size: int
    dp_size: int
    sp_size: int
    synchronized_parameter_count: int
    collective_count: int


def _YX_detached_scalar(value, *, name: str) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"{name} must contain exactly one value, got {value.shape}")
        return value.detach().clone().reshape(())
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a scalar tensor or real number")
    return torch.tensor(value).reshape(())


def _YX_sp_size(sp_group) -> int:
    # The trainer represents disabled sequence parallelism with no subgroup,
    # even when the default WORLD process group has more than one rank.
    if sp_group is None:
        return 1
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("an explicit sp_group requires initialized torch.distributed")
    return dist.get_world_size(group=sp_group)


def sp_global_detached_count(local_valid_count, *, sp_group=None) -> torch.Tensor:
    """Return the detached sum of valid counts over one SP replica.

    The returned scalar is safe to use as a loss denominator: no gradient can
    flow through either the count or its collective.  The caller should place
    tensor counts on a device supported by the process-group backend.
    """

    global_count = _YX_detached_scalar(
        local_valid_count,
        name="local_valid_count",
    )
    if _YX_sp_size(sp_group) > 1:
        dist.all_reduce(global_count, op=dist.ReduceOp.SUM, group=sp_group)
    return global_count


def normalize_sp_prediction_local_sum(
    local_sum: torch.Tensor,
    local_valid_count,
    *,
    sp_group=None,
) -> torch.Tensor:
    """Compute ``local_sum / SP-global-valid-count`` for immediate backward."""

    if not isinstance(local_sum, torch.Tensor):
        raise TypeError("local_sum must be a tensor so prediction gradients are preserved")
    if local_sum.numel() != 1:
        raise ValueError(f"local_sum must contain exactly one value, got {local_sum.shape}")

    global_count = sp_global_detached_count(
        local_valid_count,
        sp_group=sp_group,
    ).to(device=local_sum.device)
    if global_count.item() <= 0:
        raise ValueError(
            "SP-global valid count must be positive before prediction normalization"
        )
    return local_sum.reshape(()) / global_count


def sp_global_detached_sum_count(
    local_sum,
    local_count,
    *,
    sp_group=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return detached SP-global sum/count scalars for logging.

    Sum and count are packed into one float64 collective.  This path is for
    metrics only; prediction loss normalization uses the count-only helper so
    the differentiable local sum is never reduced.
    """

    detached_sum = _YX_detached_scalar(local_sum, name="local_sum")
    detached_count = _YX_detached_scalar(local_count, name="local_count")
    pair = torch.stack(
        (
            detached_sum.to(dtype=torch.float64),
            detached_count.to(device=detached_sum.device, dtype=torch.float64),
        )
    )
    if _YX_sp_size(sp_group) > 1:
        dist.all_reduce(pair, op=dist.ReduceOp.SUM, group=sp_group)
    return pair[0], pair[1]


def scale_replicated_regularization(
    loss_reg: torch.Tensor,
    sp_size: int,
) -> torch.Tensor:
    """Scale regularization replicated on every SP rank by ``1 / sp_size``."""

    if not isinstance(loss_reg, torch.Tensor):
        raise TypeError("loss_reg must be a tensor")
    if isinstance(sp_size, bool) or not isinstance(sp_size, Integral) or sp_size < 1:
        raise ValueError(f"sp_size must be a positive integer, got {sp_size!r}")
    if sp_size == 1:
        return loss_reg
    return loss_reg / int(sp_size)


def _YX_named_tensors(parameters: Iterable) -> list[tuple[str, torch.Tensor]]:
    named_parameters = []
    seen_ids = set()
    for index, item in enumerate(parameters):
        if (
            isinstance(item, tuple)
            and len(item) == 2
            and isinstance(item[0], str)
        ):
            name, parameter = item
        else:
            name, parameter = f"parameter[{index}]", item
        if not isinstance(parameter, torch.Tensor):
            raise TypeError(f"{name} must be a tensor or torch.nn.Parameter")
        if id(parameter) in seen_ids:
            raise ValueError(f"{name} is duplicated in the parameter iterable")
        seen_ids.add(id(parameter))
        named_parameters.append((name, parameter))
    return named_parameters


def _YX_world_size(world_group) -> int:
    if not dist.is_available() or not dist.is_initialized():
        if world_group is not None:
            raise RuntimeError("world_group requires initialized torch.distributed")
        return 1

    global_world_size = dist.get_world_size()
    group_world_size = dist.get_world_size(group=world_group)
    if group_world_size != global_world_size:
        raise ValueError(
            "world_group must span the full default WORLD process group: "
            f"got {group_world_size} ranks, expected {global_world_size}"
        )
    return group_world_size


def _YX_validate_dp_size(dp_size: int, world_size: int) -> int:
    if isinstance(dp_size, bool) or not isinstance(dp_size, Integral) or dp_size < 1:
        raise ValueError(f"dp_size must be a positive integer, got {dp_size!r}")
    dp_size = int(dp_size)
    if world_size % dp_size != 0:
        raise ValueError(
            f"WORLD size {world_size} must be divisible by dp_size {dp_size}"
        )
    return dp_size


@torch.no_grad()
def sync_replicated_layer_recall_gradients_(
    parameters: Iterable,
    *,
    dp_size: int,
    world_group=None,
) -> ReplicatedGradientSyncResult:
    """Synchronize replicated LayerRecall gradients as ``WORLD_SUM / dp_size``.

    Parameters may be tensors, parameters, or ``(name, parameter)`` pairs.
    Before gradient values are reduced, one presence/status vector per device
    is reduced over WORLD.  Thus every rank observes and raises on a mismatched
    ``None`` gradient instead of entering differently shaped value collectives.

    Dense gradients are flattened by ``(device, dtype)`` and reduced once per
    bucket.  Sparse or non-strided gradients are rejected consistently on all
    ranks.  With WORLD=DP (SP=1), the same operation becomes the ordinary DP
    average; with an uninitialized single-rank process it is a no-op.
    """

    named_parameters = _YX_named_tensors(parameters)
    world_size = _YX_world_size(world_group)
    dp_size = _YX_validate_dp_size(dp_size, world_size)
    sp_size = world_size // dp_size

    parameters_by_device = defaultdict(list)
    for index, (name, parameter) in enumerate(named_parameters):
        parameters_by_device[parameter.device].append((index, name, parameter))

    active_indices = set()
    collective_count = 0
    for device, entries in parameters_by_device.items():
        # For each parameter, reduce [has_grad, has_unsupported_grad].
        status = torch.zeros(2 * len(entries), dtype=torch.int64, device=device)
        for local_index, (_, _, parameter) in enumerate(entries):
            gradient = parameter.grad
            if gradient is None:
                continue
            status[2 * local_index] = 1
            if gradient.layout != torch.strided or gradient.device != parameter.device:
                status[2 * local_index + 1] = 1

        if world_size > 1:
            dist.all_reduce(status, op=dist.ReduceOp.SUM, group=world_group)
            collective_count += 1

        for local_index, (index, name, _) in enumerate(entries):
            present_count = int(status[2 * local_index].item())
            unsupported_count = int(status[2 * local_index + 1].item())
            if unsupported_count:
                raise RuntimeError(
                    f"{name} has a sparse, non-strided, or wrong-device gradient "
                    f"on {unsupported_count}/{world_size} WORLD ranks; only dense "
                    "same-device gradients are supported"
                )
            if present_count not in (0, world_size):
                raise RuntimeError(
                    f"inconsistent None gradient for {name}: present on "
                    f"{present_count}/{world_size} WORLD ranks"
                )
            if present_count == world_size:
                active_indices.add(index)

    gradients_by_bucket = defaultdict(list)
    for index in sorted(active_indices):
        name, parameter = named_parameters[index]
        gradient = parameter.grad
        # Presence/status synchronization above guarantees this on every rank.
        if gradient is None:
            raise RuntimeError(f"internal gradient-presence error for {name}")
        gradients_by_bucket[(gradient.device, gradient.dtype)].append(
            (name, gradient)
        )

    for _, entries in sorted(
        gradients_by_bucket.items(),
        key=lambda item: (str(item[0][0]), str(item[0][1])),
    ):
        total_numel = sum(gradient.numel() for _, gradient in entries)
        if total_numel == 0:
            continue
        flat_gradient = torch.cat(
            [gradient.detach().reshape(-1) for _, gradient in entries],
            dim=0,
        )
        if world_size > 1:
            dist.all_reduce(
                flat_gradient,
                op=dist.ReduceOp.SUM,
                group=world_group,
            )
            collective_count += 1
        flat_gradient.div_(dp_size)

        offset = 0
        for _, gradient in entries:
            numel = gradient.numel()
            gradient.copy_(flat_gradient.narrow(0, offset, numel).view_as(gradient))
            offset += numel

    return ReplicatedGradientSyncResult(
        world_size=world_size,
        dp_size=dp_size,
        sp_size=sp_size,
        synchronized_parameter_count=len(active_indices),
        collective_count=collective_count,
    )


@torch.no_grad()
def clip_synced_grad_norm_(
    parameters: Iterable,
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
) -> torch.Tensor:
    """Clip gradients after ``sync_replicated_layer_recall_gradients_`` has completed."""

    named_parameters = _YX_named_tensors(parameters)
    return torch.nn.utils.clip_grad_norm_(
        [parameter for _, parameter in named_parameters],
        max_norm=max_norm,
        norm_type=norm_type,
        error_if_nonfinite=error_if_nonfinite,
    )
