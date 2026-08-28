# Adopted from trainer/diffusion.py
# SPDX-License-Identifier: Apache-2.0

import gc
import hashlib
import json
import logging
import os
import random
import shutil
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    StateDictType,
)

try:
    from tqdm import tqdm
except ModuleNotFoundError:
    tqdm = None

from model.chpm import CHPMModel
from utils.config import wan_default_config
from utils.dataset import MultiTextConcatDataset, MultiVideoConcatDataset, eval_collate_fn, multi_video_collate_fn
from utils.distributed import FSDP, barrier, fsdp_wrap, launch_distributed_job
from utils.misc import set_seed
from utils.chpm_resume import (
    CHPMPromptStream,
    canonical_sha256,
    capture_rng_state,
    restore_rng_state,
    validate_rng_state,
)
from wan_5b.distributed import sp_training
from wan_5b.distributed.streaming_ulysses import (
    collective_telemetry_snapshot,
    reset_collective_telemetry,
)
from wan_5b.distributed.sp_training import SequenceParallelHelper
from wan_5b.modules.causal_model import CausalWanAttentionBlock


_YX_SUPPORTED_MODEL = "Wan2.2-TI2V-5B"


def _YX_host_memory_snapshot():
    values = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as handle:
            for line in handle:
                key, raw = line.split(":", 1)
                if key in {"MemTotal", "MemAvailable", "SwapTotal", "SwapFree"}:
                    values[key] = int(raw.strip().split()[0]) / 1024.0
    except (OSError, ValueError, IndexError):
        pass
    return {
        "mem_total_mb": float(values.get("MemTotal", 0.0)),
        "mem_available_mb": float(values.get("MemAvailable", 0.0)),
        "swap_total_mb": float(values.get("SwapTotal", 0.0)),
        "swap_used_mb": float(
            max(0.0, values.get("SwapTotal", 0.0) - values.get("SwapFree", 0.0))
        ),
    }


@dataclass(frozen=True)
class _YXPredictionSPTopology:
    global_rank: int
    world_size: int
    sp_rank: int
    dp_rank: int
    sp_size: int
    dp_size: int
    local_frames: int
    local_heads: int
    effective_global_batch: int
    streaming_sequence_parallel_mode: str
    sp_group_ranks: tuple
    dp_group_ranks: tuple


def _YX_process_group_rank_lists(world_size, sp_size):
    world_size = int(world_size)
    sp_size = int(sp_size)
    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if sp_size <= 0:
        raise ValueError(f"sequence_parallel_size must be positive, got {sp_size}")
    if world_size % sp_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by sequence_parallel_size ({sp_size})"
        )
    if sp_size == 1:
        return (), ()
    dp_size = world_size // sp_size
    sp_groups = tuple(
        tuple(range(replica * sp_size, (replica + 1) * sp_size))
        for replica in range(dp_size)
    )
    dp_groups = tuple(
        tuple(replica * sp_size + sp_rank for replica in range(dp_size))
        for sp_rank in range(sp_size)
    )
    return sp_groups, dp_groups


def _YX_resolve_prediction_sp_topology(
    *,
    global_rank,
    world_size,
    sequence_parallel_size=1,
    streaming_sequence_parallel_mode="disabled",
    model_name=_YX_SUPPORTED_MODEL,
    num_heads=24,
    num_frame_per_block=8,
    batch_size=1,
    gradient_accumulation_steps=1,
    layer_recall_replicated_params=None,
):
    global_rank = int(global_rank)
    world_size = int(world_size)
    sp_size = int(sequence_parallel_size)
    mode = str(streaming_sequence_parallel_mode or "disabled").strip().lower()
    num_heads = int(num_heads)
    num_frame_per_block = int(num_frame_per_block)
    batch_size = int(batch_size)
    gradient_accumulation_steps = int(gradient_accumulation_steps)

    if world_size <= 0:
        raise ValueError(f"world_size must be positive, got {world_size}")
    if global_rank < 0 or global_rank >= world_size:
        raise ValueError(f"global_rank ({global_rank}) must be in [0, {world_size})")
    if sp_size <= 0:
        raise ValueError(f"sequence_parallel_size must be positive, got {sp_size}")
    if str(model_name) != _YX_SUPPORTED_MODEL:
        raise ValueError(
            "chpm sequence topology only supports "
            f"{_YX_SUPPORTED_MODEL}, got {model_name}"
        )
    if sp_size > 1 and sp_size != 2:
        raise ValueError(
            "chpm only supports sequence_parallel_size=2; "
            f"SP={sp_size} is not supported"
        )
    if sp_size > 1 and mode != "ulysses_chunk":
        raise ValueError(
            "streaming_sequence_parallel_mode must be 'ulysses_chunk' when "
            f"sequence_parallel_size={sp_size}, got {mode!r}"
        )
    if (
        sp_size > 1
        and layer_recall_replicated_params is not None
        and not bool(layer_recall_replicated_params)
    ):
        raise ValueError(
            "sequence_parallel_size=2 requires layer_recall_replicated_params=true so "
            "LayerRecall parameters can be excluded from FSDP and synchronized explicitly"
        )
    if sp_size == 1 and mode != "disabled":
        raise ValueError(
            "streaming_sequence_parallel_mode must be 'disabled' when "
            f"sequence_parallel_size=1, got {mode!r}"
        )
    if world_size % sp_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by sequence_parallel_size ({sp_size})"
        )
    if num_heads % sp_size != 0:
        raise ValueError(
            f"num_heads ({num_heads}) must be divisible by sequence_parallel_size ({sp_size})"
        )
    if num_frame_per_block % sp_size != 0:
        raise ValueError(
            "num_frame_per_block "
            f"({num_frame_per_block}) must be divisible by sequence_parallel_size ({sp_size})"
        )
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if gradient_accumulation_steps <= 0:
        raise ValueError(
            "gradient_accumulation_steps must be positive, got "
            f"{gradient_accumulation_steps}"
        )

    dp_size = world_size // sp_size
    sp_rank = global_rank % sp_size
    dp_rank = global_rank // sp_size
    sp_groups, dp_groups = _YX_process_group_rank_lists(world_size, sp_size)
    if sp_size == 1:
        sp_group_ranks = (global_rank,)
        dp_group_ranks = tuple(range(world_size))
    else:
        sp_group_ranks = sp_groups[dp_rank]
        dp_group_ranks = dp_groups[sp_rank]
    return _YXPredictionSPTopology(
        global_rank=global_rank,
        world_size=world_size,
        sp_rank=sp_rank,
        dp_rank=dp_rank,
        sp_size=sp_size,
        dp_size=dp_size,
        local_frames=num_frame_per_block // sp_size,
        local_heads=num_heads // sp_size,
        effective_global_batch=batch_size * gradient_accumulation_steps * dp_size,
        streaming_sequence_parallel_mode=mode,
        sp_group_ranks=sp_group_ranks,
        dp_group_ranks=dp_group_ranks,
    )


def _YX_create_prediction_process_groups(topology, dist_api=dist):
    if int(topology.sp_size) == 1:
        return None, None
    sp_groups, dp_groups = _YX_process_group_rank_lists(
        topology.world_size,
        topology.sp_size,
    )
    current_sp_group = None
    current_dp_group = None
    for ranks in sp_groups:
        group = dist_api.new_group(ranks=list(ranks))
        if topology.global_rank in ranks:
            current_sp_group = group
    for ranks in dp_groups:
        group = dist_api.new_group(ranks=list(ranks))
        if topology.global_rank in ranks:
            current_dp_group = group
    return current_sp_group, current_dp_group


def _YX_initialize_prediction_process_groups(
    topology,
    dist_api=dist,
    sp_training_api=sp_training,
):
    sp_group, dp_group = _YX_create_prediction_process_groups(topology, dist_api=dist_api)
    sp_training_api.set_sequence_parallel_group(sp_group)
    sp_training_api.set_data_parallel_group(dp_group)
    return sp_group, dp_group


def _YX_sampler_rank_and_replicas(topology):
    return int(topology.dp_rank), int(topology.dp_size)


def _YX_micro_step_seed(
    *,
    base_seed,
    dp_rank,
    dp_size,
    step,
    accumulation_step,
    accumulation_steps,
):
    base_seed = int(base_seed)
    dp_rank = int(dp_rank)
    dp_size = int(dp_size)
    step = int(step)
    accumulation_step = int(accumulation_step)
    accumulation_steps = int(accumulation_steps)
    if dp_size <= 0 or dp_rank < 0 or dp_rank >= dp_size:
        raise ValueError(f"invalid DP rank topology: dp_rank={dp_rank}, dp_size={dp_size}")
    if accumulation_steps <= 0 or accumulation_step < 0 or accumulation_step >= accumulation_steps:
        raise ValueError(
            "invalid accumulation position: "
            f"accumulation_step={accumulation_step}, accumulation_steps={accumulation_steps}"
        )
    micro_step = step * accumulation_steps + accumulation_step
    return base_seed + micro_step * dp_size + dp_rank


def _YX_epoch_aware_iterator(dataloader, sampler, start_epoch=0):
    epoch = int(start_epoch)
    while True:
        sampler.set_epoch(epoch)
        for batch in dataloader:
            yield batch
        epoch += 1


def _YX_get(config, key, default=None):
    getter = getattr(config, "get", None)
    if getter is not None:
        try:
            return getter(key, default)
        except Exception:
            pass
    return getattr(config, key, default)


def _layer_recall_replicated_params_enabled(config):
    prediction_config = _YX_get(config, "chpm", None)
    nested_value = _YX_get(prediction_config, "layer_recall_replicated_params", False)
    return bool(_YX_get(config, "layer_recall_replicated_params", nested_value))


@torch.no_grad()
def _YX_move_replicated_layer_recall_params_(named_params, *, device):
    """Move FSDP-ignored LayerRecall FP32 masters without replacing Parameter objects."""
    for name, param in named_params:
        if not isinstance(param, torch.nn.Parameter):
            raise TypeError(f"{name} must be a torch.nn.Parameter")
        param.data = param.data.to(device=device, dtype=torch.float32)
        if param.grad is not None:
            param.grad.data = param.grad.data.to(
                device=device,
                dtype=torch.float32,
            )


def _YX_assert_replicated_layer_recall_optimizer_fp32(
    optimizer,
    named_params,
    *,
    require_state=False,
):
    """Fail fast if replicated LayerRecall masters or Adam moments lose FP32 precision."""
    for name, param in named_params:
        if param.dtype != torch.float32:
            raise RuntimeError(
                f"Replicated LayerRecall parameter {name} must stay FP32, got {param.dtype}"
            )
        state = optimizer.state.get(param, {})
        for state_name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = state.get(state_name, None)
            if value is None:
                if require_state and state_name in {"exp_avg", "exp_avg_sq"}:
                    raise RuntimeError(
                        f"Replicated LayerRecall optimizer state {name}.{state_name} is missing"
                    )
                continue
            if not torch.is_tensor(value) or value.dtype != torch.float32:
                dtype = value.dtype if torch.is_tensor(value) else type(value).__name__
                raise RuntimeError(
                    f"Replicated LayerRecall optimizer state {name}.{state_name} must be FP32, "
                    f"got {dtype}"
                )


def _YX_flatten_prompts(prompts):
    if len(prompts) == 0:
        return []
    if isinstance(prompts[0], (list, tuple)):
        return [prompt for prompt_group in prompts for prompt in prompt_group]
    return list(prompts)


def _YX_normalize_prompt_payload(value):
    if isinstance(value, (list, tuple)):
        return [_YX_normalize_prompt_payload(item) for item in value]
    if value is None:
        return None
    return str(value)


def _YX_prompt_sha256(prompts):
    normalized = _YX_normalize_prompt_payload(prompts)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _YX_sp_parity_tensor_stats(tensor):
    source = tensor.detach().float()
    finite = torch.isfinite(source)
    finite_count = finite.sum(dtype=torch.float32)
    count = source.new_tensor(source.numel(), dtype=torch.float32)
    denominator = finite_count.clamp(min=1.0)
    finite_values = torch.where(finite, source, torch.zeros_like(source))
    mean = finite_values.sum() / denominator
    centered = torch.where(finite, source - mean, torch.zeros_like(source))
    std = (centered.square().sum() / denominator).sqrt()
    minimum = torch.where(
        finite_count > 0,
        torch.where(finite, source, torch.full_like(source, float("inf"))).amin(),
        source.new_tensor(float("nan")),
    )
    maximum = torch.where(
        finite_count > 0,
        torch.where(finite, source, torch.full_like(source, float("-inf"))).amax(),
        source.new_tensor(float("nan")),
    )
    return {
        "count": count.detach().to(device="cpu", dtype=torch.float32),
        "finite_count": finite_count.detach().to(device="cpu", dtype=torch.float32),
        "nonfinite_count": (count - finite_count).detach().to(device="cpu", dtype=torch.float32),
        "all_finite": (finite_count == count).detach().to(device="cpu", dtype=torch.float32),
        "mean": mean.detach().to(device="cpu", dtype=torch.float32),
        "std": std.detach().to(device="cpu", dtype=torch.float32),
        "min": minimum.detach().to(device="cpu", dtype=torch.float32),
        "max": maximum.detach().to(device="cpu", dtype=torch.float32),
        "abs_max": torch.maximum(minimum.abs(), maximum.abs()).detach().to(
            device="cpu",
            dtype=torch.float32,
        ),
    }


def _YX_sp_parity_cpu_stats(stats):
    return {
        str(name): {
            str(metric): value.detach().to(device="cpu", dtype=torch.float32)
            if torch.is_tensor(value)
            else torch.tensor(float(value), dtype=torch.float32)
            for metric, value in values.items()
        }
        for name, values in stats.items()
    }


def _YX_sp_parity_cpu_tensor(tensor):
    if tensor.requires_grad:
        raise ValueError("SP parity capture tensors must be detached before trainer gather/save")
    output_dtype = torch.bfloat16 if tensor.dtype == torch.bfloat16 else torch.float16
    return tensor.detach().to(device="cpu", dtype=output_dtype).contiguous()


def _YX_sp_parity_gather_mode(frame_metadata, *, sp_size):
    global_frames = int(frame_metadata["global_frames"])
    local_frames = int(frame_metadata["local_frames"])
    recorded_sp_size = int(frame_metadata.get("sequence_parallel_size", sp_size))
    if recorded_sp_size != int(sp_size):
        raise ValueError(
            "SP parity capture metadata topology mismatch: "
            f"recorded SP={recorded_sp_size}, trainer SP={sp_size}"
        )
    if local_frames == global_frames:
        return False
    if (
        int(sp_size) > 1
        and bool(frame_metadata.get("is_sp_shard", False))
        and local_frames * int(sp_size) == global_frames
    ):
        return True
    raise ValueError(
        "SP parity capture has inconsistent frame metadata: "
        f"global_frames={global_frames}, local_frames={local_frames}, sp_size={sp_size}, "
        f"is_sp_shard={frame_metadata.get('is_sp_shard')}"
    )


def _YX_sp_parity_materialize(
    capture,
    *,
    global_rank,
    sp_rank,
    dp_rank,
    sp_size,
    dp_size,
    step,
    actual_micro_step_seed,
    batch_idx,
    prompts,
    gather_frames_fn=None,
):
    anchors = list(capture.get("anchors", []))
    materialized_anchors = []
    output_frame_metadata = []

    for source_record in anchors:
        record = dict(source_record)
        frame_metadata = dict(record["frame_metadata"])
        should_gather = _YX_sp_parity_gather_mode(frame_metadata, sp_size=sp_size)
        if should_gather and gather_frames_fn is None:
            from wan_5b.distributed.streaming_ulysses import (
                all_gather_detached_frames,
            )

            gather_frames_fn = all_gather_detached_frames

        output_record = {
            "chunk_index": int(record["chunk_index"]),
            "start_frame": int(record["start_frame"]),
            "end_frame": int(record["end_frame"]),
            "prediction_target": str(record["prediction_target"]),
            "full_tensors_included": bool(record.get("full_tensors_included", False)),
            "local_stats": _YX_sp_parity_cpu_stats(record.get("stats", {})),
        }
        global_frames = int(frame_metadata["global_frames"])
        local_frames = int(frame_metadata["local_frames"])
        saved_frames = local_frames
        full_tensor_values = {}
        if bool(record.get("full_tensors_included", False)):
            for name in ("chunk_noisy", "teacher_target", "student_prediction"):
                tensor = record[name]
                if tensor.requires_grad:
                    raise ValueError(f"SP parity capture {name} must be detached before gather")
                if int(tensor.shape[1]) != local_frames:
                    raise ValueError(
                        f"SP parity capture {name} has {tensor.shape[1]} frames, "
                        f"metadata says {local_frames}"
                    )
                full_tensor_values[name] = (
                    gather_frames_fn(tensor.detach()) if should_gather else tensor.detach()
                )
                if int(full_tensor_values[name].shape[1]) != global_frames:
                    raise ValueError(
                        f"SP parity capture {name} materialized "
                        f"{full_tensor_values[name].shape[1]} frames, expected {global_frames}"
                    )
            local_timestep = record["local_timestep"]
            if local_timestep.requires_grad:
                raise ValueError("SP parity capture timestep must be detached before gather")
            if int(local_timestep.shape[1]) != local_frames:
                raise ValueError(
                    "SP parity capture timestep frame count does not match frame metadata"
                )
            if should_gather:
                gathered_timestep = gather_frames_fn(
                    local_timestep.detach().reshape(
                        int(local_timestep.shape[0]),
                        local_frames,
                        1,
                        1,
                        1,
                    )
                )
                full_tensor_values["local_timestep"] = gathered_timestep.reshape(
                    int(local_timestep.shape[0]),
                    global_frames,
                )
            else:
                full_tensor_values["local_timestep"] = local_timestep.detach()
            saved_frames = global_frames

        scalar_values = {}
        for name in ("chunk_sum", "chunk_count"):
            tensor = record[name]
            if tensor.requires_grad:
                raise ValueError(f"SP parity capture {name} must be detached before gather")
            scalar = tensor.detach().float().reshape(())
            if should_gather:
                gathered = gather_frames_fn(scalar.reshape(1, 1, 1, 1, 1))
                scalar = gathered.float().sum()
            scalar_values[name] = scalar.detach()
            output_record[name] = scalar.to(device="cpu", dtype=torch.float32)

        output_stats = {
            name: _YX_sp_parity_tensor_stats(tensor)
            for name, tensor in {**full_tensor_values, **scalar_values}.items()
        }
        if not full_tensor_values:
            for name, values in _YX_sp_parity_cpu_stats(record.get("stats", {})).items():
                output_stats.setdefault(name, values)
        output_record["stats"] = output_stats
        output_record["finite"] = {
            name: values["all_finite"]
            for name, values in output_stats.items()
        }
        for name, tensor in full_tensor_values.items():
            output_record[name] = _YX_sp_parity_cpu_tensor(tensor)

        materialized_metadata = {
            "global_frames": int(global_frames),
            "local_frames": int(local_frames),
            "saved_frames": int(saved_frames),
            "input_has_full_frames": bool(local_frames == global_frames),
            "input_was_sp_shard": bool(should_gather),
            "gathered_across_sp": bool(should_gather),
        }
        output_record["frame_metadata"] = materialized_metadata
        output_frame_metadata.append(dict(materialized_metadata))
        materialized_anchors.append(output_record)

    normalized_prompts = _YX_normalize_prompt_payload(prompts)
    flattened_prompts = _YX_flatten_prompts(normalized_prompts)
    return {
        "artifact_type": "chpm_sp_parity",
        "global_rank": int(global_rank),
        "sp_rank": int(sp_rank),
        "dp_rank": int(dp_rank),
        "sp_size": int(sp_size),
        "dp_size": int(dp_size),
        "step": int(step),
        "actual_micro_step_seed": int(actual_micro_step_seed),
        "batch_idx": int(batch_idx),
        "prompts": normalized_prompts,
        "prompts_sha256": _YX_prompt_sha256(normalized_prompts),
        "prompt_sha256": [_YX_prompt_sha256(prompt) for prompt in flattened_prompts],
        "prediction_target": str(capture.get("prediction_target", "")),
        "full_tensors": bool(capture.get("full_tensors", False)),
        "anchor_schedule": dict(capture.get("anchor_schedule", {})),
        "anchor_count": int(len(materialized_anchors)),
        "frame_metadata": output_frame_metadata,
        "anchors": materialized_anchors,
    }


def _YX_save_sp_parity_capture(
    capture,
    *,
    logdir,
    global_rank,
    sp_rank,
    dp_rank,
    sp_size,
    dp_size,
    step,
    actual_micro_step_seed,
    batch_idx,
    prompts,
    gather_frames_fn=None,
):
    if capture is None:
        return {
            "sp_parity/capture_saved": 0,
            "sp_parity/capture_count": 0,
            "sp_parity/capture_path": "",
        }
    if not logdir:
        raise ValueError("sp_parity_capture_enabled requires a non-empty logdir")

    artifact = _YX_sp_parity_materialize(
        capture,
        global_rank=global_rank,
        sp_rank=sp_rank,
        dp_rank=dp_rank,
        sp_size=sp_size,
        dp_size=dp_size,
        step=step,
        actual_micro_step_seed=actual_micro_step_seed,
        batch_idx=batch_idx,
        prompts=prompts,
        gather_frames_fn=gather_frames_fn,
    )
    capture_dir = os.path.join(str(logdir), "sp_parity")
    output_path = os.path.join(
        capture_dir,
        f"YX_dp{int(dp_rank):03d}_step{int(step):06d}_anchors.pt",
    )
    is_sp_root = int(sp_rank) == 0
    if is_sp_root:
        os.makedirs(capture_dir, exist_ok=True)
        tmp_path = f"{output_path}.tmp.rank{int(global_rank)}"
        torch.save(artifact, tmp_path)
        os.replace(tmp_path, output_path)
    return {
        "sp_parity/capture_saved": int(is_sp_root),
        "sp_parity/capture_count": int(artifact["anchor_count"]),
        "sp_parity/capture_path": output_path if is_sp_root else "",
    }



class Trainer:
    CHECKPOINT_FORMAT = "chpm"
    CHECKPOINT_VERSION = 3
    CHECKPOINT_COMPLETE_MARKER = "COMPLETE"
    CRITICAL_RESUME_CONTRACT_VERSION = 1
    EXPECTED_LAYER_RECALL_TENSORS = 11
    EXPECTED_LAYER_RECALL_NUMEL = 1_648_416
    CHECKPOINT_REQUIRED_KEYS = frozenset(
        {
            "trainer",
            "checkpoint_version",
            "layer_recall_state_dict",
            "student_optimizer",
            "step",
            "global_step",
            "global_micro_step",
            "accumulation_step",
            "data_stream_states",
            "rng_states",
            "dataset_manifest",
            "dataset_manifest_hash",
            "trainable_schema",
            "critical_resume_contract",
            "critical_resume_fingerprint",
            "config",
        }
    )

    def __init__(self, config):
        self.config = config
        self.step = 0
        self.global_micro_step = 0
        self.accumulation_step = 0
        self.resume_checkpoint_path = None
        self.prompt_stream = None
        self.dataset_manifest = None
        self.dataset_manifest_hash = None

        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        launch_distributed_job()
        global_rank = dist.get_rank()
        self.global_rank = int(global_rank)
        self.world_size = dist.get_world_size()
        self.dtype = torch.bfloat16 if bool(_YX_get(config, "mixed_precision", False)) else torch.float32
        self.device = torch.cuda.current_device()
        self.is_main_process = global_rank == 0
        self.disable_wandb = bool(_YX_get(config, "disable_wandb", False))
        self.output_path = _YX_get(config, "logdir", "")
        if self.is_main_process and self.output_path:
            os.makedirs(self.output_path, exist_ok=True)
        self.metrics_path = os.path.join(self.output_path, "chpm_metrics.jsonl") if self.output_path else ""
        self.sampled_path = os.path.join(self.output_path, "chpm_samples.jsonl") if self.output_path else ""
        self.summary_path = os.path.join(self.output_path, "run_summary.json") if self.output_path else ""
        self.rank_telemetry_path = (
            os.path.join(self.output_path, f"rank{self.global_rank:03d}_telemetry.jsonl")
            if self.output_path
            else ""
        )
        self.resume_trace_path = (
            os.path.join(
                self.output_path,
                f"rank{self.global_rank:03d}_exact_resume_trace.jsonl",
            )
            if self.output_path
            else ""
        )
        self.prompt_only_data = not os.path.isdir(os.path.join(str(_YX_get(config, "data_path", "")), "video"))
        self.exact_resume_enabled = bool(
            _YX_get(config, "exact_resume_enabled", self.prompt_only_data)
        )
        if self.exact_resume_enabled and not self.prompt_only_data:
            raise ValueError(
                "exact_resume_enabled=true currently requires prompt-only data; "
                "raw-video worker prefetch/RNG cannot be resumed exactly"
            )
        self.latest_train_status = {}
        self._progress_bar_active = False
        self._configure_runtime_config()
        trainable_filter = str(_YX_get(config, "trainable_param_filter", "layer_recall_only"))
        if trainable_filter != "layer_recall_only":
            raise ValueError(
                "chpm only supports trainable_param_filter=layer_recall_only, "
                f"got {trainable_filter}"
            )
        self.gradient_accumulation_steps = int(_YX_get(config, "gradient_accumulation_steps", 1))
        self.max_grad_norm = float(_YX_get(config, "max_grad_norm", 10.0))
        self.layer_recall_replicated_params = _layer_recall_replicated_params_enabled(config)
        model_name = str(config.model_kwargs.model_name)
        num_heads = int(wan_default_config.get(model_name, {}).get("num_heads", 0))
        self.topology = _YX_resolve_prediction_sp_topology(
            global_rank=self.global_rank,
            world_size=self.world_size,
            sequence_parallel_size=int(_YX_get(config, "sequence_parallel_size", 1)),
            streaming_sequence_parallel_mode=_YX_get(
                config,
                "streaming_sequence_parallel_mode",
                "disabled",
            ),
            model_name=model_name,
            num_heads=num_heads,
            num_frame_per_block=int(_YX_get(config, "num_frame_per_block", 1)),
            batch_size=int(_YX_get(config, "batch_size", 1)),
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            layer_recall_replicated_params=self.layer_recall_replicated_params,
        )
        self.sequence_parallel_size = int(self.topology.sp_size)
        self.data_parallel_size = int(self.topology.dp_size)
        self.sp_rank = int(self.topology.sp_rank)
        self.dp_rank = int(self.topology.dp_rank)
        self.local_frames = int(self.topology.local_frames)
        self.local_heads = int(self.topology.local_heads)
        self.effective_global_batch = int(self.topology.effective_global_batch)
        self.streaming_sequence_parallel_mode = self.topology.streaming_sequence_parallel_mode
        self.sp_group, self.dp_group = _YX_initialize_prediction_process_groups(self.topology)

        if self.sequence_parallel_size > 1:
            if bool(_YX_get(config, "load_raw_video", False)):
                raise ValueError(
                    "sequence_parallel_size=2 prediction distillation currently supports "
                    "prompt-only synthetic latents only; raw-video VAE input is not supported"
                )
            if not self.prompt_only_data:
                raise ValueError(
                    "sequence_parallel_size=2 prediction distillation currently supports "
                    "prompt-only synthetic latents only; raw-video and ODE-latent datasets "
                    "are not supported"
                )

        if "seed" not in config:
            config.seed = 0
        self.base_seed = int(config.seed)
        self.model_init_seed = int(_YX_get(config, "model_init_seed", self.base_seed))
        self.data_seed = int(_YX_get(config, "data_seed", self.base_seed))
        set_seed(self.model_init_seed)

        if self.is_main_process:
            print(
                "[CHPM][Topology] "
                f"world={self.world_size}, sp={self.sequence_parallel_size}, "
                f"dp={self.data_parallel_size}, local_frames={self.local_frames}, "
                f"local_heads={self.local_heads}, effective_global_batch={self.effective_global_batch}"
            )

        if self.is_main_process and not self.disable_wandb:
            if _YX_get(config, "wandb_key", None):
                wandb.login(host=_YX_get(config, "wandb_host", "https://api.wandb.ai"), key=config.wandb_key)
            wandb_name = os.environ.get("WANDB_NAME") or _YX_get(
                config,
                "wandb_name",
                _YX_get(config, "config_name", "chpm"),
            )
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=wandb_name,
                mode="online",
                entity=_YX_get(config, "wandb_entity", None),
                project=_YX_get(config, "wandb_project", "LongLive2-LayerRecall"),
                dir=_YX_get(config, "wandb_save_dir", ""),
            )

        # Rank-0-only setup above may consume RNG state; reset immediately before construction.
        set_seed(self.model_init_seed)
        self.model = CHPMModel(config, device=self.device)
        if bool(self.model.layer_recall_replicated_params) != bool(self.layer_recall_replicated_params):
            raise RuntimeError(
                "Trainer/model layer_recall_replicated_params configuration mismatch"
            )
        self.sp_helper = SequenceParallelHelper(self)

        raw_state = None
        base_student_ckpt = _YX_get(config, "student_ckpt", _YX_get(config, "generator_ckpt", None))
        base_teacher_ckpt = _YX_get(config, "teacher_ckpt", base_student_ckpt)
        if base_student_ckpt:
            if self.is_main_process:
                print(f"[CHPM] Loading base student weights from {base_student_ckpt}")
            checkpoint = torch.load(base_student_ckpt, map_location="cpu")
            student_missing, student_unexpected = self.model.load_student_weights(checkpoint)
            del checkpoint
            if base_teacher_ckpt and str(base_teacher_ckpt) != str(base_student_ckpt):
                if self.is_main_process:
                    print(f"[CHPM] Loading base teacher weights from {base_teacher_ckpt}")
                checkpoint = torch.load(base_teacher_ckpt, map_location="cpu")
            else:
                checkpoint = torch.load(base_student_ckpt, map_location="cpu")
            teacher_missing, teacher_unexpected = self.model.load_teacher_weights(checkpoint)
            del checkpoint
            load_info = {
                "student_missing": list(student_missing),
                "student_unexpected": list(student_unexpected),
                "teacher_missing": list(teacher_missing),
                "teacher_unexpected": list(teacher_unexpected),
            }
            if self.is_main_process:
                self._print_load_info(load_info)
            gc.collect()
        elif self.is_main_process:
            print("[CHPM] No base checkpoint provided; training starts from initialized weights.")

        resume_checkpoint_path = self._find_resume_checkpoint()
        if resume_checkpoint_path:
            if self.is_main_process:
                print(f"[CHPM] Loading LayerRecall resume state from {resume_checkpoint_path}")
            checkpoint = torch.load(
                resume_checkpoint_path, map_location="cpu", weights_only=False
            )
            if self.exact_resume_enabled:
                checkpoint = self._validate_resume_checkpoint_schema(
                    checkpoint, checkpoint_path=resume_checkpoint_path
                )
            layer_recall_missing, layer_recall_unexpected = self.model.load_layer_recall_state_dict(checkpoint)
            if layer_recall_missing or layer_recall_unexpected:
                raise RuntimeError(
                    "LayerRecall resume state mismatch: "
                    f"missing={list(layer_recall_missing)[:12]}, "
                    f"unexpected={list(layer_recall_unexpected)[:12]}"
                )
            if isinstance(checkpoint, dict) and "step" in checkpoint:
                self.step = int(checkpoint.get("global_step", checkpoint["step"]))
                self.global_micro_step = int(
                    checkpoint.get(
                        "global_micro_step",
                        self.step * self.gradient_accumulation_steps,
                    )
                )
                self.accumulation_step = int(checkpoint.get("accumulation_step", 0))
                self.resume_checkpoint_path = str(resume_checkpoint_path)
                if self.is_main_process:
                    print(
                        "[CHPM] Resuming from "
                        f"step {self.step}, global_micro_step={self.global_micro_step}, "
                        f"accumulation_step={self.accumulation_step}"
                    )
            raw_state = checkpoint if isinstance(checkpoint, dict) else None
            gc.collect()
        else:
            layer_recall_init_ckpt = str(
                _YX_get(config, "layer_recall_init_ckpt", "") or ""
            )
            if layer_recall_init_ckpt:
                if not os.path.isfile(layer_recall_init_ckpt):
                    raise FileNotFoundError(
                        "LayerRecall initialization checkpoint does not exist: "
                        f"{layer_recall_init_ckpt}"
                    )
                if self.is_main_process:
                    print(
                        "[CHPM] Initializing LayerRecall weights from "
                        f"{layer_recall_init_ckpt}"
                    )
                checkpoint = torch.load(
                    layer_recall_init_ckpt, map_location="cpu", weights_only=False
                )
                layer_recall_missing, layer_recall_unexpected = (
                    self.model.load_layer_recall_state_dict(checkpoint)
                )
                if layer_recall_missing or layer_recall_unexpected:
                    raise RuntimeError(
                        "LayerRecall initialization state mismatch: "
                        f"missing={list(layer_recall_missing)[:12]}, "
                        f"unexpected={list(layer_recall_unexpected)[:12]}"
                    )
                del checkpoint
                gc.collect()

        pre_fsdp_trainable_names = self.model.capture_pre_fsdp_trainable_layer_recall_params()
        pre_fsdp_trainable_named_params = list(
            self.model.pre_fsdp_trainable_layer_recall_named_param_objects()
        )
        pre_fsdp_trainable_tensor_count = int(
            self.model.pre_fsdp_trainable_layer_recall_param_tensor_count
        )
        pre_fsdp_trainable_count = int(self.model.pre_fsdp_trainable_layer_recall_param_count)
        if self.layer_recall_replicated_params:
            _YX_move_replicated_layer_recall_params_(
                pre_fsdp_trainable_named_params,
                device=torch.device("cuda", self.device),
            )

        transformer_wrap = (CausalWanAttentionBlock,)
        student_fsdp_kwargs = dict(
            sharding_strategy=_YX_get(config, "sharding_strategy", "full"),
            mixed_precision=bool(_YX_get(config, "mixed_precision", False)),
            wrap_strategy=_YX_get(config, "generator_fsdp_wrap_strategy", "size"),
            transformer_module=transformer_wrap,
        )
        if self.layer_recall_replicated_params:
            student_fsdp_kwargs["ignored_states"] = [
                param for _, param in pre_fsdp_trainable_named_params
            ]
        self.model.student = fsdp_wrap(
            self.model.student,
            **student_fsdp_kwargs,
        )
        teacher_wrap_strategy = str(_YX_get(config, "teacher_fsdp_wrap_strategy", _YX_get(config, "generator_fsdp_wrap_strategy", "size")))
        teacher_wrapped_by_fsdp = teacher_wrap_strategy.lower() != "none"
        if teacher_wrapped_by_fsdp:
            self.model.teacher = fsdp_wrap(
                self.model.teacher,
                sharding_strategy=_YX_get(config, "teacher_sharding_strategy", _YX_get(config, "sharding_strategy", "full")),
                mixed_precision=bool(_YX_get(config, "mixed_precision", False)),
                wrap_strategy=teacher_wrap_strategy,
                transformer_module=transformer_wrap,
                cpu_offload=bool(_YX_get(config, "teacher_cpu_offload", False)),
            )
        else:
            self.model.teacher_runtime_cpu_offload = True
            self.model.teacher.to(device="cpu")
        self.model.text_encoder = fsdp_wrap(
            self.model.text_encoder,
            sharding_strategy=_YX_get(config, "sharding_strategy", "full"),
            mixed_precision=bool(_YX_get(config, "mixed_precision", False)),
            wrap_strategy=_YX_get(config, "text_encoder_fsdp_wrap_strategy", "size"),
            cpu_offload=bool(_YX_get(config, "text_encoder_cpu_offload", False)),
        )
        if self.model.vae is not None:
            self.model.vae = self.model.vae.to(device=self.device, dtype=self.dtype)

        self.model.teacher.requires_grad_(False)
        self.model.configure_student_layer_recall_trainable_params()
        if self.layer_recall_replicated_params:
            replicated_ids = {id(param) for _, param in pre_fsdp_trainable_named_params}
            wrapped_replicated = [
                (name, param)
                for name, param in self.model.student_layer_recall_named_parameters()
                if id(param) in replicated_ids
            ]
            if len(wrapped_replicated) != len(pre_fsdp_trainable_named_params):
                raise RuntimeError(
                    "FSDP ignored_states did not preserve all captured LayerRecall Parameter objects: "
                    f"captured={len(pre_fsdp_trainable_named_params)}, "
                    f"visible={len(wrapped_replicated)}"
                )
            self.replicated_layer_recall_named_params = list(pre_fsdp_trainable_named_params)
            trainable_params = [
                param for _, param in self.replicated_layer_recall_named_params
            ]
        else:
            self.replicated_layer_recall_named_params = []
            trainable_params = [
                param for _, param in self.model.student_layer_recall_named_parameters()
            ]
        trainable_names = list(pre_fsdp_trainable_names)
        trainable_count = pre_fsdp_trainable_count
        layer_recall_summary = self.model.layer_recall_architecture_summary()
        if self.is_main_process:
            print(
                "[CHPM] Trainable LayerRecall params (full pre-FSDP): "
                f"{pre_fsdp_trainable_tensor_count} tensors, {trainable_count} scalars"
            )
            for name in trainable_names[:20]:
                print(f"  trainable: {name}")
            if len(trainable_names) > 20:
                print(f"  ... {len(trainable_names) - 20} more")
            self._write_run_summary(
                {
                    "trainer": "chpm",
                    "exact_resume_enabled": bool(self.exact_resume_enabled),
                    "checkpoint_version": int(self.CHECKPOINT_VERSION),
                    "checkpoint_complete_marker": self.CHECKPOINT_COMPLETE_MARKER,
                    "resume_checkpoint_path": self.resume_checkpoint_path,
                    "resume_global_step": int(self.step),
                    "resume_global_micro_step": int(self.global_micro_step),
                    "critical_resume_fingerprint": canonical_sha256(
                        self._build_critical_resume_contract()
                    ),
                    "world_size": int(self.world_size),
                    "sequence_parallel_size": int(self.sequence_parallel_size),
                    "global_rank": int(self.global_rank),
                    "sp_rank": int(self.sp_rank),
                    "dp_rank": int(self.dp_rank),
                    "sp_size": int(self.sequence_parallel_size),
                    "dp_size": int(self.data_parallel_size),
                    "streaming_sequence_parallel_mode": str(self.streaming_sequence_parallel_mode),
                    "sp_group_ranks": list(self.topology.sp_group_ranks),
                    "dp_group_ranks": list(self.topology.dp_group_ranks),
                    "local_frames": int(self.local_frames),
                    "local_heads": int(self.local_heads),
                    "student_kv_cache_heads": int(self.local_heads),
                    "teacher_kv_cache_heads": int(self.local_heads),
                    "effective_global_batch": int(self.effective_global_batch),
                    "base_seed": int(self.base_seed),
                    "model_init_seed": int(self.model_init_seed),
                    "data_seed": int(self.data_seed),
                    "micro_step_seed_formula": (
                        "data_seed + (step * gradient_accumulation_steps + accumulation_step) "
                        "* dp_size + dp_rank"
                    ),
                    "student_local_attn_size": int(self.model.student_model_kwargs.get("local_attn_size", -1)),
                    "teacher_local_attn_size": int(self.model.teacher_model_kwargs.get("local_attn_size", -1)),
                    "teacher_requested_cache_frames": int(_YX_get(
                        _YX_get(config, "chpm", None),
                        "teacher_physical_cache_frames",
                        0,
                    ) or 0),
                    "teacher_effective_cache_frames": int(max(
                        3 * int(_YX_get(config, "num_frame_per_block", 8)),
                        int(_YX_get(
                            _YX_get(config, "chpm", None),
                            "teacher_physical_cache_frames",
                            0,
                        ) or 0),
                        int(self.model.teacher_model_kwargs.get("local_attn_size", -1)),
                    )),
                    "student_physical_cache_frames": int(getattr(self.model.layer_recall_config, "layer_recall_physical_cache_frames", 0) or 0),
                    "student_sink_size": int(self.model.student_model_kwargs.get("sink_size", 0)),
                    "teacher_sink_size": int(self.model.teacher_model_kwargs.get("sink_size", 0)),
                    "teacher_wrapped_by_fsdp": bool(teacher_wrapped_by_fsdp),
                    "teacher_cpu_offload": bool(_YX_get(config, "teacher_cpu_offload", False)),
                    "layer_recall_log_path": str(getattr(self.model.layer_recall_config, "layer_recall_log_path", "")),
                    **layer_recall_summary,
                    "prediction_target": str(self.model.prediction_target),
                    "clean_latent_source": str(self.model.clean_latent_source),
                    "rollout_mode": str(self.model.rollout_mode),
                    "anchor_every_n_frames": int(self.model.anchor_every_n_frames),
                    "anchor_backward_mode": str(self.model.anchor_backward_mode),
                    "student_prefix_context_gradient": "detached",
                    "layer_recall_replicated_params": bool(self.layer_recall_replicated_params),
                    "layer_recall_replicated_param_dtype": (
                        str(trainable_params[0].dtype)
                        if self.layer_recall_replicated_params and trainable_params
                        else "fsdp_managed"
                    ),
                    "layer_recall_replicated_optimizer_state_dtype": (
                        "torch.float32"
                        if self.layer_recall_replicated_params
                        else "fsdp_managed"
                    ),
                    "layer_recall_fp32_island_modules": list(
                        getattr(self.model, "layer_recall_fp32_island_modules", ())
                    ),
                    "layer_recall_forward_compute_policy": str(
                        getattr(self.model, "layer_recall_forward_compute_policy", "fsdp_managed")
                    ),
                    "layer_recall_gradient_sync_mode": (
                        "world_sum_div_dp_size"
                        if self.layer_recall_replicated_params
                        else "fsdp_managed"
                    ),
                    "layer_recall_gradient_sync_divisor": (
                        int(self.data_parallel_size)
                        if self.layer_recall_replicated_params
                        else None
                    ),
                    "layer_recall_regularization_sp_scale": (
                        1.0 / float(self.sequence_parallel_size)
                        if self.layer_recall_replicated_params
                        else 1.0
                    ),
                    "sp_parity_capture_enabled": bool(
                        self.model.sp_parity_capture_enabled
                    ),
                    "sp_parity_capture_full_tensors": bool(
                        self.model.sp_parity_capture_full_tensors
                    ),
                    "teacher_target_device": str(self.model.teacher_target_device),
                    "teacher_runtime_cpu_offload": bool(self.model.teacher_runtime_cpu_offload),
                    "prompt_only_data": bool(getattr(self, "prompt_only_data", False)),
                    "teacher_trainable_param_count": 0,
                    "student_trainable_layer_recall_param_count": int(trainable_count),
                    "student_trainable_layer_recall_param_tensor_count": pre_fsdp_trainable_tensor_count,
                    "student_trainable_layer_recall_param_count_scope": "full_unsharded_pre_fsdp",
                    "trainable_param_names": trainable_names,
                }
            )

        self.student_optimizer = torch.optim.AdamW(
            trainable_params,
            lr=float(_YX_get(config, "lr", 1.0e-5)),
            betas=(float(_YX_get(config, "beta1", 0.0)), float(_YX_get(config, "beta2", 0.999))),
            weight_decay=float(_YX_get(config, "weight_decay", 0.0)),
        )
        if self.layer_recall_replicated_params:
            _YX_assert_replicated_layer_recall_optimizer_fp32(
                self.student_optimizer,
                self.replicated_layer_recall_named_params,
            )

        if raw_state is not None:
            if "student_optimizer" in raw_state:
                osd = FSDP.optim_state_dict_to_load(
                    self.model.student,
                    self.student_optimizer,
                    raw_state["student_optimizer"],
                )
                self.student_optimizer.load_state_dict(osd)
                del osd
                if self.layer_recall_replicated_params:
                    _YX_assert_replicated_layer_recall_optimizer_fp32(
                        self.student_optimizer,
                        self.replicated_layer_recall_named_params,
                        require_state=True,
                    )
                if self.is_main_process:
                    print("[CHPM] Resumed optimizer state from key 'student_optimizer'")
            else:
                raise RuntimeError("CHPM resume checkpoint is missing student_optimizer")
            gc.collect()

        self._build_dataloader(raw_state)
        self._restore_runtime_state(raw_state)
        self._update_run_summary(
            {
                "dataset_manifest_hash": self.dataset_manifest_hash,
                "data_stream_state": (
                    self.prompt_stream.state_dict()
                    if self.prompt_stream is not None
                    else None
                ),
                "rng_state_restored": bool(
                    raw_state is not None and self.exact_resume_enabled
                ),
            }
        )
        if raw_state is not None:
            del raw_state
            gc.collect()
        self.previous_time = None

    def _print_progress_safe(self, message):
        if self._progress_bar_active and tqdm is not None:
            tqdm.write(message)
        else:
            print(message)

    def _print_load_info(self, load_info):
        for key, values in load_info.items():
            layer_recall_values = [value for value in values if "layer_recall" in value]
            other_values = [value for value in values if "layer_recall" not in value]
            if layer_recall_values:
                print(f"[CHPM] {key} LayerRecall keys: {layer_recall_values[:12]}")
            if other_values:
                print(f"[CHPM][Warning] {key} non-LayerRecall keys: {other_values[:12]}")

    @staticmethod
    def _normalize_contract_path(path):
        if not path:
            return None
        return os.path.abspath(os.path.normpath(os.path.expanduser(str(path))))

    @staticmethod
    def _normalize_contract_int_list(value, *, field):
        if value is None:
            return []
        try:
            return sorted(set(int(item) for item in value))
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"critical resume field {field} must be an integer list"
            ) from exc

    @staticmethod
    def _contract_mapping(value):
        if value is None:
            return {}
        if OmegaConf.is_config(value):
            return OmegaConf.to_container(value, resolve=True)
        if isinstance(value, Mapping):
            return dict(value)
        raise RuntimeError(
            f"critical resume value must be a mapping, got {type(value).__name__}"
        )

    def _build_critical_resume_contract(self):
        config = self.config
        prediction = _YX_get(config, "chpm", {})
        layer_recall = _YX_get(config, "layer_recall", {})
        model_kwargs = _YX_get(config, "model_kwargs", {})
        student_kwargs = _YX_get(config, "student_model_kwargs", model_kwargs)
        teacher_kwargs = _YX_get(config, "teacher_model_kwargs", model_kwargs)

        layer_recall_semantic_keys = (
            "layer_recall_enabled",
            "layer_recall_selection_mode",
            "layer_recall_temperature",
            "layer_recall_candidate_pool_size",
            "layer_recall_normalize_scores",
            "layer_recall_score_mode",
            "layer_recall_physical_cache_frames",
            "layer_recall_current_conditioned_enabled",
            "layer_recall_current_hidden_dim",
            "layer_recall_current_alpha",
            "layer_recall_current_detach_summary",
            "layer_recall_current_zero_init",
            "layer_recall_use_layer_gamma",
            "memory_sensitive_layers",
        )
        layer_recall_contract = {}
        for key in layer_recall_semantic_keys:
            value = _YX_get(layer_recall, key, None)
            if key == "memory_sensitive_layers":
                value = self._normalize_contract_int_list(value, field=key)
            layer_recall_contract[key] = value

        prediction_keys = (
            "prediction_target",
            "clean_latent_source",
            "anchor_every_n_frames",
            "anchor_include_last_chunk",
            "teacher_target_device",
            "teacher_cpu_offload",
            "teacher_runtime_cpu_offload",
            "loss_type",
            "teacher_local_attn_size",
            "student_local_attn_size",
            "student_physical_cache_frames",
            "teacher_physical_cache_frames",
            "min_history_chunks",
            "prediction_loss_weight",
            "regularization_weight",
            "layer_recall_replicated_params",
        )
        prediction_contract = {
            key: _YX_get(prediction, key, _YX_get(config, key, None))
            for key in prediction_keys
        }

        return {
            "contract_version": self.CRITICAL_RESUME_CONTRACT_VERSION,
            "trainer": str(_YX_get(config, "trainer", "")),
            "exact_resume_enabled": bool(self.exact_resume_enabled),
            "distributed": {
                "world_size": int(self.world_size),
                "sequence_parallel_size": int(self.sequence_parallel_size),
                "data_parallel_size": int(self.data_parallel_size),
                "streaming_sequence_parallel_mode": str(
                    self.streaming_sequence_parallel_mode
                ),
                "per_rank_batch_size": int(_YX_get(config, "batch_size", 1)),
                "gradient_accumulation_steps": int(
                    self.gradient_accumulation_steps
                ),
                "layer_recall_replicated_params": bool(self.layer_recall_replicated_params),
                "mixed_precision": bool(_YX_get(config, "mixed_precision", False)),
                "sharding_strategy": str(_YX_get(config, "sharding_strategy", "")),
            },
            "model": {
                "name": str(_YX_get(model_kwargs, "model_name", "")),
                "num_frame_per_block": int(_YX_get(config, "num_frame_per_block", 1)),
                "student_local_attn_size": int(
                    _YX_get(student_kwargs, "local_attn_size", -1)
                ),
                "teacher_local_attn_size": int(
                    _YX_get(teacher_kwargs, "local_attn_size", -1)
                ),
                "image_or_video_shape": [
                    int(value) for value in list(_YX_get(config, "image_or_video_shape", []))
                ],
            },
            "chpm": prediction_contract,
            "layer_recall": layer_recall_contract,
            "optimizer": {
                "lr": float(_YX_get(config, "lr", 1.0e-5)),
                "weight_decay": float(_YX_get(config, "weight_decay", 0.0)),
                "beta1": float(_YX_get(config, "beta1", 0.0)),
                "beta2": float(_YX_get(config, "beta2", 0.999)),
                "max_grad_norm": float(_YX_get(config, "max_grad_norm", 10.0)),
            },
            "schedule": {
                "max_iters": int(_YX_get(config, "max_iters", 1)),
                "num_training_frames": int(
                    _YX_get(config, "num_training_frames", 0)
                ),
                "min_num_training_frames": int(
                    _YX_get(config, "min_num_training_frames", 0)
                ),
                "slice_last_frames": int(_YX_get(config, "slice_last_frames", 0)),
                "sampling_steps": int(_YX_get(config, "sampling_steps", 0)),
            },
            "data": {
                "data_path": self._normalize_contract_path(
                    _YX_get(config, "data_path", None)
                ),
                "chunks_per_shot": int(_YX_get(config, "chunks_per_shot", 0)),
                "scene_cut_prefix": str(_YX_get(config, "scene_cut_prefix", "")),
                "data_seed": int(self.data_seed),
                "prompt_only": bool(self.prompt_only_data),
            },
            "seeds": {
                "base_seed": int(self.base_seed),
                "model_init_seed": int(self.model_init_seed),
                "data_seed": int(self.data_seed),
            },
            "base_checkpoints": {
                "generator": self._normalize_contract_path(
                    _YX_get(config, "generator_ckpt", None)
                ),
                "student": self._normalize_contract_path(
                    _YX_get(config, "student_ckpt", None)
                ),
                "teacher": self._normalize_contract_path(
                    _YX_get(config, "teacher_ckpt", None)
                ),
                "lora": self._normalize_contract_path(
                    _YX_get(config, "lora_ckpt", None)
                ),
                "layer_recall_init": self._normalize_contract_path(
                    _YX_get(config, "layer_recall_init_ckpt", None)
                ),
            },
        }

    @staticmethod
    def _critical_contract_differences(saved, current, *, prefix=""):
        if isinstance(saved, Mapping) and isinstance(current, Mapping):
            differences = []
            for key in sorted(set(saved).union(current)):
                field = f"{prefix}.{key}" if prefix else str(key)
                if key not in saved:
                    differences.append(f"{field}: missing from checkpoint")
                elif key not in current:
                    differences.append(f"{field}: absent from current contract")
                else:
                    differences.extend(
                        Trainer._critical_contract_differences(
                            saved[key], current[key], prefix=field
                        )
                    )
            return differences
        if saved != current:
            return [f"{prefix}: saved={saved!r}, current={current!r}"]
        return []

    def _current_trainable_schema(self):
        named_parameters = list(
            self.model.pre_fsdp_trainable_layer_recall_named_param_objects()
        )
        return {
            name: {
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "numel": int(parameter.numel()),
            }
            for name, parameter in sorted(named_parameters)
        }

    @staticmethod
    def _optimizer_step_values(optimizer_state):
        values = []
        for state in optimizer_state.get("state", {}).values():
            if not isinstance(state, Mapping) or "step" not in state:
                continue
            value = state["step"]
            values.append(int(value.item() if torch.is_tensor(value) else value))
        return values

    def _validate_resume_checkpoint_schema(self, checkpoint, *, checkpoint_path):
        if not isinstance(checkpoint, Mapping):
            raise RuntimeError(
                f"LayerRecall prediction resume checkpoint must be a mapping: {checkpoint_path}"
            )
        missing = self.CHECKPOINT_REQUIRED_KEYS.difference(checkpoint)
        if missing:
            raise RuntimeError(
                "LayerRecall prediction resume checkpoint is incomplete or unsupported; missing "
                f"required fields: {sorted(missing)}"
            )
        if checkpoint["trainer"] != self.CHECKPOINT_FORMAT:
            raise RuntimeError(
                "Refusing checkpoint from another trainer: "
                f"expected={self.CHECKPOINT_FORMAT!r}, got={checkpoint['trainer']!r}"
            )
        if checkpoint["checkpoint_version"] != self.CHECKPOINT_VERSION:
            raise RuntimeError(
                "Unsupported LayerRecall prediction checkpoint version: "
                f"expected={self.CHECKPOINT_VERSION}, "
                f"got={checkpoint['checkpoint_version']!r}"
            )

        global_step = checkpoint["global_step"]
        global_micro_step = checkpoint["global_micro_step"]
        accumulation_step = checkpoint["accumulation_step"]
        for name, value in (
            ("global_step", global_step),
            ("global_micro_step", global_micro_step),
            ("accumulation_step", accumulation_step),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise RuntimeError(
                    f"resume counter {name} must be a non-negative int, got {value!r}"
                )
        if checkpoint["step"] != global_step:
            raise RuntimeError(
                f"resume step alias mismatch: step={checkpoint['step']}, "
                f"global_step={global_step}"
            )
        if accumulation_step != 0:
            raise RuntimeError(
                "LayerRecall prediction checkpoints are only valid at optimizer boundaries; "
                f"accumulation_step must be 0, got {accumulation_step}"
            )
        expected_micro_step = global_step * self.gradient_accumulation_steps
        if global_micro_step != expected_micro_step:
            raise RuntimeError(
                "global/micro step mismatch: "
                f"global_micro_step={global_micro_step}, expected={expected_micro_step}"
            )

        layer_recall_state = checkpoint["layer_recall_state_dict"]
        if not isinstance(layer_recall_state, Mapping) or not layer_recall_state:
            raise RuntimeError("layer_recall_state_dict must be a non-empty mapping")
        if any(not torch.is_tensor(value) for value in layer_recall_state.values()):
            raise RuntimeError("layer_recall_state_dict contains non-tensor values")
        if len(layer_recall_state) != self.EXPECTED_LAYER_RECALL_TENSORS:
            raise RuntimeError(
                f"LayerRecall tensor count mismatch: {len(layer_recall_state)}/{self.EXPECTED_LAYER_RECALL_TENSORS}"
            )
        layer_recall_numel = sum(int(value.numel()) for value in layer_recall_state.values())
        if layer_recall_numel != self.EXPECTED_LAYER_RECALL_NUMEL:
            raise RuntimeError(
                f"LayerRecall scalar count mismatch: {layer_recall_numel}/{self.EXPECTED_LAYER_RECALL_NUMEL}"
            )
        nonfinite = [
            name for name, value in layer_recall_state.items() if not torch.isfinite(value).all()
        ]
        if nonfinite:
            raise RuntimeError(f"resume LayerRecall state contains non-finite tensors: {nonfinite[:8]}")

        optimizer_state = checkpoint["student_optimizer"]
        if not isinstance(optimizer_state, Mapping) or {
            "state",
            "param_groups",
        }.difference(optimizer_state):
            raise RuntimeError("student_optimizer state is incomplete")
        if global_step > 0 and len(optimizer_state["state"]) != self.EXPECTED_LAYER_RECALL_TENSORS:
            raise RuntimeError(
                "student_optimizer parameter state count mismatch: "
                f"{len(optimizer_state['state'])}/{self.EXPECTED_LAYER_RECALL_TENSORS}"
            )
        optimizer_steps = self._optimizer_step_values(optimizer_state)
        if optimizer_steps and set(optimizer_steps) != {global_step}:
            raise RuntimeError(
                "optimizer internal step mismatch: "
                f"saved={sorted(set(optimizer_steps))}, global_step={global_step}"
            )

        saved_schema = checkpoint["trainable_schema"]
        current_schema = self._current_trainable_schema()
        if saved_schema != current_schema:
            differences = self._critical_contract_differences(
                saved_schema, current_schema, prefix="trainable_schema"
            )
            raise RuntimeError(
                "trainable LayerRecall schema mismatch:\n  - "
                + "\n  - ".join(differences[:32])
            )

        data_stream_states = checkpoint["data_stream_states"]
        if not isinstance(data_stream_states, (list, tuple)) or len(
            data_stream_states
        ) != self.world_size:
            raise RuntimeError(
                "data_stream_states must contain one state per global rank"
            )
        progress = set()
        for expected_rank, entry in enumerate(data_stream_states):
            if not isinstance(entry, Mapping) or {
                "global_rank",
                "sp_rank",
                "dp_rank",
                "stream",
            }.difference(entry):
                raise RuntimeError(
                    f"data stream state for rank {expected_rank} is incomplete"
                )
            expected_sp_rank = expected_rank % self.sequence_parallel_size
            expected_dp_rank = expected_rank // self.sequence_parallel_size
            if (
                entry["global_rank"] != expected_rank
                or entry["sp_rank"] != expected_sp_rank
                or entry["dp_rank"] != expected_dp_rank
            ):
                raise RuntimeError(
                    f"data stream topology mismatch at global rank {expected_rank}: "
                    f"saved={dict(entry)}"
                )
            stream = entry["stream"]
            if not isinstance(stream, Mapping):
                raise RuntimeError(
                    f"data stream payload for rank {expected_rank} must be a mapping"
                )
            missing_stream = CHPMPromptStream.REQUIRED_STATE_KEYS.difference(
                stream
            )
            if missing_stream:
                raise RuntimeError(
                    f"data stream rank {expected_rank} missing {sorted(missing_stream)}"
                )
            if stream["global_micro_step"] != global_micro_step:
                raise RuntimeError(
                    f"data stream rank {expected_rank} global_micro_step mismatch: "
                    f"{stream['global_micro_step']} != {global_micro_step}"
                )
            if stream["rank"] != expected_dp_rank or stream["num_replicas"] != self.data_parallel_size:
                raise RuntimeError(
                    f"data stream DP topology mismatch at global rank {expected_rank}"
                )
            progress.add(
                (
                    int(stream["epoch"]),
                    int(stream["sample_cursor"]),
                    int(stream["global_micro_step"]),
                )
            )
        if len(progress) != 1:
            raise RuntimeError(
                f"data stream ranks are not synchronized: {sorted(progress)}"
            )

        rng_states = checkpoint["rng_states"]
        if not isinstance(rng_states, (list, tuple)) or len(rng_states) != self.world_size:
            raise RuntimeError("rng_states must contain one state per global rank")
        for expected_rank, entry in enumerate(rng_states):
            if not isinstance(entry, Mapping) or entry.get("global_rank") != expected_rank:
                raise RuntimeError(
                    f"RNG state rank ordering mismatch at rank {expected_rank}"
                )
            validate_rng_state(entry.get("state"))

        manifest = checkpoint["dataset_manifest"]
        manifest_hash = checkpoint["dataset_manifest_hash"]
        if not isinstance(manifest, Mapping):
            raise RuntimeError("dataset_manifest must be a mapping")
        if not isinstance(manifest_hash, str) or len(manifest_hash) != 64:
            raise RuntimeError("dataset_manifest_hash must be a SHA-256 digest")
        if canonical_sha256(manifest) != manifest_hash:
            raise RuntimeError("dataset manifest fingerprint is invalid")

        saved_contract = checkpoint["critical_resume_contract"]
        saved_fingerprint = checkpoint["critical_resume_fingerprint"]
        if not isinstance(saved_contract, Mapping):
            raise RuntimeError("critical_resume_contract must be a mapping")
        if canonical_sha256(saved_contract) != saved_fingerprint:
            raise RuntimeError("critical resume contract fingerprint is invalid")
        current_contract = self._build_critical_resume_contract()
        differences = self._critical_contract_differences(
            saved_contract, current_contract
        )
        if differences:
            raise RuntimeError(
                "critical resume contract mismatch; refusing to continue:\n  - "
                + "\n  - ".join(differences[:32])
            )

        checkpoint_dir = os.path.basename(os.path.dirname(checkpoint_path))
        if checkpoint_dir.startswith("checkpoint_model_"):
            try:
                directory_step = int(checkpoint_dir.rsplit("_", 1)[-1])
            except ValueError as exc:
                raise RuntimeError(
                    f"invalid checkpoint directory name: {checkpoint_dir}"
                ) from exc
            if directory_step != global_step:
                raise RuntimeError(
                    "checkpoint directory/global step mismatch: "
                    f"directory={directory_step}, payload={global_step}"
                )
        if not isinstance(checkpoint["config"], Mapping):
            raise RuntimeError("checkpoint config must be a mapping")
        return checkpoint

    def _find_resume_checkpoint(self):
        explicit_checkpoint = str(_YX_get(self.config, "resume_checkpoint", "") or "")
        if explicit_checkpoint:
            checkpoint_path = explicit_checkpoint
            if os.path.isdir(checkpoint_path):
                checkpoint_path = os.path.join(checkpoint_path, "model.pt")
            if not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"Explicit resume checkpoint does not exist: {checkpoint_path}"
                )
            if self.exact_resume_enabled:
                marker = os.path.join(
                    os.path.dirname(checkpoint_path), self.CHECKPOINT_COMPLETE_MARKER
                )
                if not os.path.isfile(marker):
                    raise RuntimeError(
                        "Explicit exact resume requires a COMPLETE marker next to model.pt"
                    )
            return checkpoint_path
        auto_resume = bool(_YX_get(self.config, "auto_resume", True))
        if auto_resume and self.output_path:
            latest_checkpoint = self.find_latest_checkpoint(
                self.output_path, require_complete=self.exact_resume_enabled
            )
            if latest_checkpoint:
                return latest_checkpoint
            checkpoint_dirs = [
                item
                for item in os.listdir(self.output_path)
                if item.startswith("checkpoint_model_")
                and os.path.isdir(os.path.join(self.output_path, item))
            ] if os.path.isdir(self.output_path) else []
            if self.exact_resume_enabled and checkpoint_dirs:
                raise RuntimeError(
                    "Checkpoint directories exist but none has a valid COMPLETE marker; "
                    "refusing to silently restart. Legacy v1 checkpoints are not exact-resume "
                    f"compatible: {sorted(checkpoint_dirs)[-5:]}"
                )
            if self.is_main_process:
                print("[CHPM] Auto resume found no checkpoint in logdir.")
        return None

    def _write_jsonl(self, path, payload):
        if not self.is_main_process or not path:
            return
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _write_rank_jsonl(self, payload):
        if not self.rank_telemetry_path:
            return
        with open(self.rank_telemetry_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _write_exact_resume_trace(self, payload):
        if not self.exact_resume_enabled or not self.resume_trace_path:
            return
        with open(self.resume_trace_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")

    def _write_run_summary(self, payload):
        if not self.is_main_process or not self.summary_path:
            return
        with open(self.summary_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)

    def _update_run_summary(self, payload):
        if not self.is_main_process or not self.summary_path:
            return
        current = {}
        if os.path.isfile(self.summary_path):
            with open(self.summary_path, "r", encoding="utf-8") as handle:
                current = json.load(handle)
        current.update(payload)
        with open(self.summary_path, "w", encoding="utf-8") as handle:
            json.dump(current, handle, indent=2, sort_keys=True)


    def _configure_runtime_config(self):
        if not self.output_path:
            self.output_path = ""
        if _YX_get(self.config, "layer_recall", None) is None:
            self.config.layer_recall = OmegaConf.create({})
        prediction_cfg = _YX_get(self.config, "chpm", None)
        mapping = {
            "student_physical_cache_frames": "layer_recall_physical_cache_frames",
        }
        for source_key, target_key in mapping.items():
            value = _YX_get(prediction_cfg, source_key, None)
            if value is not None:
                existing = _YX_get(self.config.layer_recall, target_key, None)
                if existing is not None and int(existing) != int(value):
                    raise ValueError(
                        f"chpm.{source_key}={value} conflicts with "
                        f"layer_recall.{target_key}={existing}"
                    )
                self.config.layer_recall[target_key] = value
        if self.output_path:
            self.config.layer_recall.layer_recall_log_path = os.path.join(self.output_path, "layer_recall_selection.jsonl")
        loss_type = str(_YX_get(prediction_cfg, "loss_type", "mse")).lower()
        if loss_type != "mse":
            raise ValueError(f"chpm currently supports loss_type=mse only, got {loss_type}")

    def _build_dataset_manifest(self):
        manifest = {
            "mode": str(self.dataset._mode),
            "dataset_size": int(len(self.dataset)),
            "num_blocks": int(self.dataset.num_blocks),
            "chunks_per_shot": int(self.dataset.chunks_per_shot),
            "scene_cut_prefix": str(self.dataset.scene_cut_prefix),
            "caption_field": str(self.dataset.caption_field),
            "prompt_content_hash_format": "sha256-canonical-json-v1",
        }
        if self.dataset._mode == "dir":
            manifest["ordered_cases"] = [
                folder.name for folder in self.dataset._folders
            ]
        prompt_hashes = []
        for index in range(len(self.dataset)):
            item = self.dataset[index]
            prompts = item.get("prompts") if isinstance(item, Mapping) else None
            if not isinstance(prompts, list) or len(prompts) != self.dataset.num_blocks:
                raise RuntimeError(
                    f"dataset item {index} must contain exactly "
                    f"{self.dataset.num_blocks} prompts"
                )
            if any(not isinstance(prompt, str) for prompt in prompts):
                raise RuntimeError(f"dataset item {index} contains a non-string prompt")
            prompt_hashes.append(canonical_sha256(prompts))
        manifest["ordered_training_prompt_hashes"] = prompt_hashes
        return manifest

    def _build_dataloader(self, resume_state=None):
        model_name = self.config.model_kwargs.model_name
        data_path = str(self.config.data_path)
        sampler_rank, sampler_replicas = _YX_sampler_rank_and_replicas(self.topology)
        self.data_generator = torch.Generator()
        self.data_generator.manual_seed(self.data_seed)
        num_workers = int(_YX_get(self.config, "num_workers", 2))
        worker_kwargs = {"num_workers": num_workers}
        if num_workers > 0:
            worker_kwargs["prefetch_factor"] = 1
        if self.prompt_only_data:
            num_blocks = int(list(self.config.image_or_video_shape)[1]) // int(_YX_get(self.config, "num_frame_per_block", 1))
            dataset = MultiTextConcatDataset(
                data_path=data_path,
                num_blocks=num_blocks,
                chunks_per_shot=int(_YX_get(self.config, "chunks_per_shot", 0)),
                scene_cut_prefix=str(_YX_get(self.config, "scene_cut_prefix", "")),
                deterministic=False,
            )
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=sampler_replicas,
                rank=sampler_rank,
                shuffle=True,
                drop_last=True,
                seed=self.data_seed,
            )
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=int(_YX_get(self.config, "batch_size", 1)),
                sampler=sampler,
                pin_memory=False,
                persistent_workers=False,
                collate_fn=eval_collate_fn,
                generator=self.data_generator,
                **worker_kwargs,
            )
            if self.is_main_process:
                print(f"[CHPM] Using prompt-only dataset: {data_path}")
                print(f"[CHPM] DATASET SIZE {len(dataset)}")
            self.dataset = dataset
            if self.exact_resume_enabled:
                if int(_YX_get(self.config, "batch_size", 1)) != 1:
                    raise ValueError(
                        "Exact LayerRecall prediction resume currently requires batch_size=1"
                    )
                manifest_payload = [
                    self._build_dataset_manifest() if self.is_main_process else None
                ]
                dist.broadcast_object_list(manifest_payload, src=0)
                self.dataset_manifest = manifest_payload[0]
                self.dataset_manifest_hash = canonical_sha256(
                    self.dataset_manifest
                )
                if resume_state is not None:
                    saved_manifest_hash = resume_state.get("dataset_manifest_hash")
                    if saved_manifest_hash != self.dataset_manifest_hash:
                        raise RuntimeError(
                            "dataset manifest changed since checkpoint: "
                            f"saved={saved_manifest_hash}, "
                            f"current={self.dataset_manifest_hash}"
                        )
                self.prompt_stream = CHPMPromptStream(
                    len(dataset),
                    rank=sampler_rank,
                    num_replicas=sampler_replicas,
                    batch_size=int(_YX_get(self.config, "batch_size", 1)),
                    shuffle=True,
                    drop_last=True,
                    seed=self.data_seed,
                )
                if resume_state is not None:
                    stream_entries = resume_state["data_stream_states"]
                    entry = stream_entries[self.global_rank]
                    self.prompt_stream.load_state_dict(entry["stream"])
                if self.prompt_stream.global_micro_step != self.global_micro_step:
                    raise RuntimeError(
                        "prompt stream/global micro-step mismatch after restore: "
                        f"stream={self.prompt_stream.global_micro_step}, "
                        f"trainer={self.global_micro_step}"
                    )
                self.sampler = None
                self.dataloader = None
                if self.is_main_process:
                    state = self.prompt_stream.state_dict()
                    print(
                        "[CHPM][ExactResume] "
                        f"dataset_manifest={self.dataset_manifest_hash}, "
                        f"epoch={state['epoch']}, sample_cursor={state['sample_cursor']}, "
                        f"global_micro_step={state['global_micro_step']}"
                    )
            else:
                self.sampler = sampler
                self.dataloader = _YX_epoch_aware_iterator(dataloader, sampler)
            return

        frame_raw_height = (
            list(self.config.image_or_video_shape)[3]
            * wan_default_config[model_name]["spatial_compression_ratio"]
        )
        frame_raw_width = (
            list(self.config.image_or_video_shape)[4]
            * wan_default_config[model_name]["spatial_compression_ratio"]
        )
        total_frames = (
            (list(self.config.image_or_video_shape)[1] - 1)
            * wan_default_config[model_name]["temporal_compression_ratio"]
            + 1
        )
        self.fps = wan_default_config[model_name].get("fps", 16)
        dataset = MultiVideoConcatDataset(
            data_dir=self.config.data_path,
            video_size=(frame_raw_height, frame_raw_width),
            total_frames=total_frames,
            deterministic=False,
            num_frame_per_block=int(_YX_get(self.config, "num_frame_per_block", 1)),
            temporal_compression_ratio=wan_default_config[model_name]["temporal_compression_ratio"],
            target_fps=self.fps,
            allow_padding=bool(_YX_get(self.config, "allow_padding", False)),
            min_latent_frames=int(_YX_get(self.config, "min_latent_frames", 0)),
            single_video_only=bool(_YX_get(self.config, "uniform_prompt", False)),
            independent_first_frame=bool(_YX_get(self.config, "independent_first_frame", False)),
            return_image=bool(_YX_get(self.config, "i2v", False)),
            max_chunks_per_shot=int(_YX_get(self.config, "max_chunks_per_shot", 0)),
            sample_warning_seconds=float(_YX_get(self.config, "dataset_sample_warning_seconds", 60.0)),
            sample_warning_interval_seconds=float(
                _YX_get(self.config, "dataset_sample_warning_interval_seconds", 60.0)
            ),
        )
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=sampler_replicas,
            rank=sampler_rank,
            shuffle=True,
            drop_last=True,
            seed=self.data_seed,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=int(_YX_get(self.config, "batch_size", 1)),
            sampler=sampler,
            pin_memory=False,
            persistent_workers=False,
            collate_fn=multi_video_collate_fn,
            generator=self.data_generator,
            **worker_kwargs,
        )
        if self.is_main_process:
            print(f"[CHPM] DATASET SIZE {len(dataset)}")
        self.sampler = sampler
        self.dataloader = _YX_epoch_aware_iterator(dataloader, sampler)

    def _peek_training_batch(self):
        if not self.exact_resume_enabled:
            return next(self.dataloader), None
        indices = self.prompt_stream.peek_batch()
        items = [self.dataset[index] for index in indices]
        batch = eval_collate_fn(items)
        batch["YX_dataset_indices"] = list(indices)
        batch["YX_data_epoch"] = int(self.prompt_stream.epoch)
        batch["YX_data_sample_cursor"] = int(self.prompt_stream.sample_cursor)
        batch["YX_global_micro_step"] = int(self.prompt_stream.global_micro_step)
        return batch, indices

    def _commit_training_batch(self, indices):
        if not self.exact_resume_enabled:
            self.global_micro_step += 1
            return
        self.prompt_stream.commit_batch(indices)
        self.global_micro_step = int(self.prompt_stream.global_micro_step)

    def _local_rng_state(self):
        return {
            "global_rank": int(self.global_rank),
            "state": capture_rng_state(
                device=self.device,
                data_generator=getattr(self, "data_generator", None),
            ),
        }

    def _local_data_stream_state(self):
        if self.prompt_stream is None:
            raise RuntimeError("exact resume prompt stream has not been initialized")
        return {
            "global_rank": int(self.global_rank),
            "sp_rank": int(self.sp_rank),
            "dp_rank": int(self.dp_rank),
            "stream": self.prompt_stream.state_dict(),
        }

    def _restore_runtime_state(self, checkpoint):
        if checkpoint is None or not self.exact_resume_enabled:
            return
        rng_entries = checkpoint["rng_states"]
        entry = rng_entries[self.global_rank]
        restore_rng_state(
            entry["state"],
            device=self.device,
            data_generator=getattr(self, "data_generator", None),
        )
        if self.is_main_process:
            print(
                "[CHPM][ExactResume] Restored per-rank RNG and data stream "
                f"at global_micro_step={self.global_micro_step}"
            )

    def find_latest_checkpoint(self, logdir, *, require_complete=False):
        if not logdir or not os.path.exists(logdir):
            return None
        checkpoint_dirs = []
        for item in os.listdir(logdir):
            if item.startswith("checkpoint_model_") and os.path.isdir(os.path.join(logdir, item)):
                try:
                    step = int(item.replace("checkpoint_model_", ""))
                except ValueError:
                    continue
                checkpoint_path = os.path.join(logdir, item, "model.pt")
                marker_path = os.path.join(
                    logdir, item, self.CHECKPOINT_COMPLETE_MARKER
                )
                if os.path.exists(checkpoint_path) and (
                    not require_complete or os.path.isfile(marker_path)
                ):
                    checkpoint_dirs.append((step, checkpoint_path))
        if not checkpoint_dirs:
            return None
        checkpoint_dirs.sort(key=lambda pair: pair[0])
        return checkpoint_dirs[-1][1]

    def save(self):
        if self.exact_resume_enabled:
            if self.accumulation_step != 0:
                raise RuntimeError(
                    "Refusing checkpoint inside gradient accumulation: "
                    f"accumulation_step={self.accumulation_step}"
                )
            expected_micro_step = self.step * self.gradient_accumulation_steps
            if self.global_micro_step != expected_micro_step:
                raise RuntimeError(
                    "Refusing checkpoint with inconsistent global/micro step: "
                    f"global_step={self.step}, global_micro_step={self.global_micro_step}, "
                    f"expected={expected_micro_step}"
                )
            if self.prompt_stream is None or self.prompt_stream.has_pending_batch:
                raise RuntimeError(
                    "Refusing checkpoint before prompt-stream micro-batch commit"
                )
        if self.is_main_process:
            print("[CHPM] Start gathering student states...")
        with FSDP.state_dict_type(
            self.model.student,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            student_state_dict = self.model.student.state_dict()
            layer_recall_state_dict = {
                key.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", ""): value
                for key, value in student_state_dict.items()
                if "layer_recall" in key
            }
            optimizer_state_dict = FSDP.optim_state_dict(self.model.student, self.student_optimizer)

        checkpoint_dir = os.path.join(self.output_path, f"checkpoint_model_{self.step:06d}")
        if self.exact_resume_enabled:
            data_stream_states = [None for _ in range(self.world_size)]
            dist.all_gather_object(
                data_stream_states, self._local_data_stream_state()
            )
            rng_states = [None for _ in range(self.world_size)]
            dist.all_gather_object(rng_states, self._local_rng_state())
            critical_resume_contract = self._build_critical_resume_contract()
            critical_resume_fingerprint = canonical_sha256(
                critical_resume_contract
            )
            payload = {
                "trainer": self.CHECKPOINT_FORMAT,
                "checkpoint_version": self.CHECKPOINT_VERSION,
                "layer_recall_state_dict": layer_recall_state_dict,
                "student_optimizer": optimizer_state_dict,
                "step": int(self.step),
                "global_step": int(self.step),
                "global_micro_step": int(self.global_micro_step),
                "accumulation_step": int(self.accumulation_step),
                "data_stream_states": data_stream_states,
                "rng_states": rng_states,
                "dataset_manifest": self.dataset_manifest,
                "dataset_manifest_hash": self.dataset_manifest_hash,
                "trainable_schema": self._current_trainable_schema(),
                "critical_resume_contract": critical_resume_contract,
                "critical_resume_fingerprint": critical_resume_fingerprint,
                "config": OmegaConf.to_container(self.config, resolve=True),
            }
            if self.is_main_process:
                os.makedirs(checkpoint_dir, exist_ok=True)
                checkpoint_file = os.path.join(checkpoint_dir, "model.pt")
                tmp_file = checkpoint_file + ".tmp"
                marker_file = os.path.join(
                    checkpoint_dir, self.CHECKPOINT_COMPLETE_MARKER
                )
                marker_tmp = marker_file + ".tmp"
                if os.path.exists(marker_file):
                    raise RuntimeError(
                        f"Refusing to overwrite completed checkpoint: {checkpoint_dir}"
                    )
                with open(tmp_file, "wb") as handle:
                    torch.save(payload, handle)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp_file, checkpoint_file)
                with open(marker_tmp, "w", encoding="ascii") as handle:
                    handle.write(
                        f"global_step={self.step}\n"
                        f"global_micro_step={self.global_micro_step}\n"
                        f"contract={critical_resume_fingerprint}\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(marker_tmp, marker_file)
                directory_fd = None
                try:
                    directory_fd = os.open(checkpoint_dir, os.O_RDONLY)
                    os.fsync(directory_fd)
                except OSError:
                    pass
                finally:
                    if directory_fd is not None:
                        os.close(directory_fd)
                self._prune_completed_checkpoints()
                print("[CHPM] Model saved to", checkpoint_file)
        elif self.is_main_process:
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_file = os.path.join(checkpoint_dir, "model.pt")
            torch.save(
                {
                    "checkpoint_version": self.CHECKPOINT_VERSION,
                    "layer_recall_state_dict": layer_recall_state_dict,
                    "student_optimizer": optimizer_state_dict,
                    "step": self.step,
                    "trainer": self.CHECKPOINT_FORMAT,
                    "config": OmegaConf.to_container(self.config, resolve=True),
                },
                checkpoint_file,
            )
            print("[CHPM] Model saved to", checkpoint_file)
        barrier()
        torch.cuda.empty_cache()
        gc.collect()

    def _prune_completed_checkpoints(self):
        keep = int(_YX_get(self.config, "max_checkpoints", 0) or 0)
        if keep <= 0:
            return
        completed = []
        for item in os.listdir(self.output_path):
            if not item.startswith("checkpoint_model_"):
                continue
            checkpoint_dir = os.path.join(self.output_path, item)
            model_path = os.path.join(checkpoint_dir, "model.pt")
            marker_path = os.path.join(
                checkpoint_dir, self.CHECKPOINT_COMPLETE_MARKER
            )
            if not os.path.isfile(model_path) or not os.path.isfile(marker_path):
                continue
            try:
                step = int(item.rsplit("_", 1)[-1])
            except ValueError:
                continue
            completed.append((step, checkpoint_dir))
        completed.sort()
        for _, checkpoint_dir in completed[:-keep]:
            shutil.rmtree(checkpoint_dir)

    def train_one_step(self, batch, accumulation_step=0, accumulation_steps=None):
        step_start_time = time.perf_counter()
        reset_collective_telemetry()
        accumulation_steps = accumulation_steps or self.gradient_accumulation_steps
        batch_idx = int(self.step) * int(accumulation_steps) + int(accumulation_step)
        micro_step_seed = _YX_micro_step_seed(
            base_seed=self.data_seed,
            dp_rank=self.dp_rank,
            dp_size=self.data_parallel_size,
            step=self.step,
            accumulation_step=accumulation_step,
            accumulation_steps=accumulation_steps,
        )
        set_seed(micro_step_seed)
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        if self.sequence_parallel_size > 1 and ("ode_latent" in batch or "frames" in batch):
            input_kind = "ode_latent" if "ode_latent" in batch else "raw_video"
            raise ValueError(
                "sequence_parallel_size=2 prediction distillation currently supports "
                f"prompt-only synthetic latents only; received {input_kind} input"
            )

        text_prompts = batch["prompts"]
        batch_size = len(text_prompts)
        dataset_indices = [int(value) for value in batch.get("YX_dataset_indices", [])]
        data_epoch = int(batch.get("YX_data_epoch", -1))
        data_sample_cursor = int(batch.get("YX_data_sample_cursor", -1))
        clean_latent_is_sp_sharded = False
        synthetic_clean_latent = False
        if "ode_latent" in batch and not bool(_YX_get(self.config, "load_raw_video", False)):
            clean_latent = batch["ode_latent"][:, -1].to(device=self.device, dtype=self.dtype)
            image_latent = clean_latent[:, 0:1]
            clean_latent_source = "dataset_ode_latent"
        elif bool(_YX_get(self.config, "load_raw_video", False)) and "frames" in batch:
            clean_latent, image_latent, clean_latent_is_sp_sharded = self.sp_helper.encode_raw_video_latents(
                batch,
                batch_size=batch_size,
            )
            clean_latent_source = "raw_video_vae_encode"
        elif "frames" not in batch:
            shape = list(self.config.image_or_video_shape)
            clean_latent = torch.randn(
                [batch_size, int(shape[1]), int(shape[2]), int(shape[3]), int(shape[4])],
                device=self.device,
                dtype=self.dtype,
            )
            image_latent = clean_latent[:, 0:1]
            synthetic_clean_latent = True
            clean_latent_source = "synthetic_prompt_only"
        else:
            raise ValueError(
                "Batch has raw frames but load_raw_video is false and no ode_latent is available. "
                "Use load_raw_video=true or provide precomputed ode_latent."
            )

        loss_mask = self.sp_helper.build_loss_mask(batch, clean_latent, clean_latent_is_sp_sharded)
        if self.sequence_parallel_size > 1:
            # Streaming SP shards each 8-frame chunk inside the causal model.
            # Keep the sequence mask global so the rollout can apply the same
            # per-chunk local frame bounds as flow/x0/timestep.
            loss_mask_global_valid_count = None
        else:
            loss_mask, loss_mask_global_valid_count = self.sp_helper.partition_loss_mask(
                loss_mask,
                already_sharded=clean_latent_is_sp_sharded,
            )
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size

        with torch.no_grad():
            text_prompts_flat = _YX_flatten_prompts(text_prompts)
            conditional_dict = self.model.text_encoder(text_prompts=text_prompts_flat)

        immediate_backward = True
        if immediate_backward and accumulation_step == 0:
            self.student_optimizer.zero_grad(set_to_none=True)

        backward_called = False

        def _backward_anchor(anchor_loss: torch.Tensor) -> None:
            nonlocal backward_called
            (anchor_loss / accumulation_steps).backward()
            backward_called = True

        loss, log_dict = self.model.chpm_loss(
            image_or_video_shape=image_or_video_shape,
            conditional_dict=conditional_dict,
            clean_latent=clean_latent,
            initial_latent=image_latent,
            loss_mask=loss_mask,
            loss_mask_global_valid_count=loss_mask_global_valid_count,
            global_step=int(self.step) + 1,
            backward_callback=_backward_anchor,
        )
        log_dict["dataset_indices_csv"] = ",".join(
            str(value) for value in dataset_indices
        )
        log_dict["data_epoch"] = int(data_epoch)
        log_dict["data_sample_cursor"] = int(data_sample_cursor)
        log_dict["global_micro_step_before_commit"] = int(
            batch.get("YX_global_micro_step", batch_idx)
        )
        log_dict["micro_step_seed"] = int(micro_step_seed)
        sp_parity_capture = log_dict.pop("_sp_parity_capture", None)

        if (not immediate_backward) and accumulation_step == 0:
            self.student_optimizer.zero_grad(set_to_none=True)
        if not immediate_backward:
            (loss / accumulation_steps).backward()
        elif not backward_called:
            raise RuntimeError("Immediate anchor backward was enabled but no backward callback was called.")
        if accumulation_step == accumulation_steps - 1:
            replicated_sync_result = None
            if self.layer_recall_replicated_params:
                from utils.chpm_sp import (
                    clip_synced_grad_norm_,
                    sync_replicated_layer_recall_gradients_,
                )

                replicated_sync_result = sync_replicated_layer_recall_gradients_(
                    self.replicated_layer_recall_named_params,
                    dp_size=self.data_parallel_size,
                )
                grad_norm = clip_synced_grad_norm_(
                    self.replicated_layer_recall_named_params,
                    max_norm=self.max_grad_norm,
                )
            else:
                grad_norm = self.model.student.clip_grad_norm_(self.max_grad_norm)
            self.student_optimizer.step()
            if self.layer_recall_replicated_params:
                _YX_assert_replicated_layer_recall_optimizer_fp32(
                    self.student_optimizer,
                    self.replicated_layer_recall_named_params,
                    require_state=True,
                )
            self.step += 1
        else:
            grad_norm = torch.tensor(0.0, device=self.device)
            return
        sp_parity_metrics = _YX_save_sp_parity_capture(
            sp_parity_capture,
            logdir=self.output_path,
            global_rank=self.global_rank,
            sp_rank=self.sp_rank,
            dp_rank=self.dp_rank,
            sp_size=self.sequence_parallel_size,
            dp_size=self.data_parallel_size,
            step=self.step,
            actual_micro_step_seed=micro_step_seed,
            batch_idx=batch_idx,
            prompts=text_prompts,
        )

        wandb_loss_dict = {
            "chpm/loss": float(log_dict["loss_total"]),
            "chpm/loss_pred": float(log_dict["loss_pred"]),
            "chpm/loss_reg": float(log_dict["loss_reg"]),
            "chpm/grad_norm": float(grad_norm.detach().float().cpu().item()),
            "chpm/num_pred_chunks": int(log_dict["num_pred_chunks"]),
            "chpm/num_chunks": int(log_dict["num_chunks"]),
            "chpm/layer_recall_gate_active_events": int(log_dict.get("layer_recall_gate_active_events", 0)),
            "chpm/layer_recall_soft_or_st_events": int(log_dict.get("layer_recall_soft_or_st_events", 0)),
            "chpm/layer_recall_candidate_chunks_max": int(log_dict.get("layer_recall_candidate_chunks_max", 0)),
            "chpm/layer_recall_soft_memory_positive_events": int(log_dict.get("layer_recall_soft_memory_positive_events", 0)),
            "chpm/memory_sensitive_layer_count": int(log_dict.get("memory_sensitive_layer_count", 0)),
            "chpm/original_window_layer_count": int(log_dict.get("original_window_layer_count", 0)),
            "chpm/memory_sensitive_layer_events": int(log_dict.get("memory_sensitive_layer_events", 0)),
            "chpm/original_window_layer_events": int(log_dict.get("original_window_layer_events", 0)),
            "chpm/layer_recall_active_events": int(log_dict.get("layer_recall_active_events", 0)),
            "chpm/sp_parity_capture_count": int(
                sp_parity_metrics["sp_parity/capture_count"]
            ),
            "chpm/layer_recall_replicated_params": int(self.layer_recall_replicated_params),
            "chpm/gradient_sync_divisor": int(
                self.data_parallel_size if self.layer_recall_replicated_params else 1
            ),
            "chpm/regularization_sp_scale": float(
                1.0 / float(self.sequence_parallel_size)
                if self.layer_recall_replicated_params
                else 1.0
            ),
            "chpm/gradient_sync_parameter_count": int(
                replicated_sync_result.synchronized_parameter_count
                if replicated_sync_result is not None
                else 0
            ),
            "chpm/gradient_sync_collective_count": int(
                replicated_sync_result.collective_count
                if replicated_sync_result is not None
                else 0
            ),
        }
        collective_telemetry = collective_telemetry_snapshot()
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        rank_telemetry = {
            "step": int(self.step),
            "global_rank": int(self.global_rank),
            "sp_rank": int(self.sp_rank),
            "dp_rank": int(self.dp_rank),
            "sp_size": int(self.sequence_parallel_size),
            "dp_size": int(self.data_parallel_size),
            "step_time_s": float(time.perf_counter() - step_start_time),
            "cuda": {
                "allocated_mb": float(torch.cuda.memory_allocated(self.device) / (1024**2)),
                "reserved_mb": float(torch.cuda.memory_reserved(self.device) / (1024**2)),
                "max_allocated_mb": float(torch.cuda.max_memory_allocated(self.device) / (1024**2)),
                "max_reserved_mb": float(torch.cuda.max_memory_reserved(self.device) / (1024**2)),
                "physical_free_mb": float(free_bytes / (1024**2)),
                "physical_total_mb": float(total_bytes / (1024**2)),
            },
            "host": _YX_host_memory_snapshot(),
            "collectives": collective_telemetry,
            "teacher": {
                key: log_dict.get(key)
                for key in (
                    "teacher_requested_cache_frames",
                    "teacher_effective_cache_frames",
                    "teacher_local_attn_frames",
                    "teacher_max_visible_frames",
                    "teacher_cache_shape",
                    "teacher_phase_time_s",
                    "teacher_memory_before_model_move",
                    "teacher_memory_after_model_move",
                    "teacher_memory_after_cache_allocation",
                    "teacher_memory_after_release",
                    "teacher_phase_peak",
                    "teacher_memory_trace",
                )
            },
            "student": {
                key: log_dict.get(key)
                for key in (
                    "student_effective_cache_frames",
                    "student_cache_shape",
                    "student_phase_time_s",
                    "student_memory_after_cache_allocation",
                    "student_phase_peak",
                    "student_memory_trace",
                )
            },
            "loss": {
                "total": float(log_dict["loss_total"]),
                "prediction": float(log_dict["loss_pred"]),
                "regularization": float(log_dict["loss_reg"]),
                "grad_norm": float(grad_norm.detach().float().cpu().item()),
            },
            "rollout": {
                "num_chunks": int(log_dict["num_chunks"]),
                "num_pred_chunks": int(log_dict["num_pred_chunks"]),
                "layer_recall_gate_active_events": int(log_dict.get("layer_recall_gate_active_events", 0)),
                "layer_recall_soft_or_st_events": int(log_dict.get("layer_recall_soft_or_st_events", 0)),
                "memory_sensitive_layer_events": int(
                    log_dict.get("memory_sensitive_layer_events", 0)
                ),
                "original_window_layer_events": int(
                    log_dict.get("original_window_layer_events", 0)
                ),
                "layer_recall_active_events": int(
                    log_dict.get("layer_recall_active_events", 0)
                ),
            },
        }
        self._write_rank_jsonl(rank_telemetry)
        if self.is_main_process:
            if not self.disable_wandb:
                wandb.log(wandb_loss_dict, step=self.step)
            self.latest_train_status = {
                "loss": wandb_loss_dict["chpm/loss"],
                "L_pred": wandb_loss_dict["chpm/loss_pred"],
                "L_reg": wandb_loss_dict["chpm/loss_reg"],
                "grad_norm": wandb_loss_dict["chpm/grad_norm"],
                "pred_chunks": wandb_loss_dict["chpm/num_pred_chunks"],
                "num_chunks": wandb_loss_dict["chpm/num_chunks"],
                "layer_recall_active": wandb_loss_dict["chpm/layer_recall_gate_active_events"],
                "layer_recall_st": wandb_loss_dict["chpm/layer_recall_soft_or_st_events"],
                "memory_sensitive_events": wandb_loss_dict["chpm/memory_sensitive_layer_events"],
            }
            self._print_progress_safe(
                f"[step {self.step:07d}] "
                f"loss={self.latest_train_status['loss']:.6f}, "
                f"L_pred={self.latest_train_status['L_pred']:.6f}, "
                f"L_reg={self.latest_train_status['L_reg']:.6f}, "
                f"grad_norm={self.latest_train_status['grad_norm']:.6f}, "
                f"pred_chunks={self.latest_train_status['pred_chunks']}/"
                f"{self.latest_train_status['num_chunks']}, "
                f"layer_recall_active={self.latest_train_status['layer_recall_active']}, "
                f"layer_recall_st={self.latest_train_status['layer_recall_st']}, "
                f"memory_sensitive_events={self.latest_train_status['memory_sensitive_events']}"
            )
            metric_payload = {
                "step": int(self.step),
                "time": float(time.time()),
                **wandb_loss_dict,
                **{
                    f"raw/{key}": value
                    for key, value in log_dict.items()
                    if isinstance(value, (int, float, str, bool))
                },
                "memory/max_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024**2)),
                "memory/max_reserved_mb": float(torch.cuda.max_memory_reserved() / (1024**2)),
                "sp/collective_count": int(collective_telemetry["collective_count"]),
                "sp/collective_bytes": int(collective_telemetry["estimated_bytes"]),
                "sp/collective_time_s": float(collective_telemetry["collective_time_s"]),
                "runtime/step_time_s": float(time.perf_counter() - step_start_time),
                "data/clean_latent_source": clean_latent_source,
                "data/synthetic_clean_latent": bool(synthetic_clean_latent),
            }
            metric_payload.update(sp_parity_metrics)
            self._write_jsonl(self.metrics_path, metric_payload)
            self._write_jsonl(
                self.sampled_path,
                {
                    "step": int(self.step),
                    "global_micro_step": int(self.global_micro_step + 1),
                    "micro_step_seed": int(micro_step_seed),
                    "dataset_indices_csv": ",".join(
                        str(value) for value in dataset_indices
                    ),
                    "data_epoch": int(data_epoch),
                    "data_sample_cursor": int(data_sample_cursor),
                    "prediction_target": str(log_dict.get("prediction_target", "")),
                    "num_pred_chunks": int(log_dict.get("num_pred_chunks", 0)),
                    "num_chunks": int(log_dict.get("num_chunks", 0)),
                    "layer_recall_gate_active_events": int(log_dict.get("layer_recall_gate_active_events", 0)),
                    "layer_recall_soft_or_st_events": int(log_dict.get("layer_recall_soft_or_st_events", 0)),
                    "layer_recall_candidate_chunks_max": int(log_dict.get("layer_recall_candidate_chunks_max", 0)),
                    "memory_sensitive_layers_csv": str(log_dict.get("memory_sensitive_layers_csv", "")),
                    "memory_sensitive_layer_count": int(log_dict.get("memory_sensitive_layer_count", 0)),
                    "original_window_layer_count": int(log_dict.get("original_window_layer_count", 0)),
                    "layer_recall_visible_layout": str(log_dict.get("layer_recall_visible_layout", "")),
                    "disabled_layer_visible_layout": str(log_dict.get("disabled_layer_visible_layout", "")),
                    "memory_sensitive_layer_events": int(log_dict.get("memory_sensitive_layer_events", 0)),
                    "original_window_layer_events": int(log_dict.get("original_window_layer_events", 0)),
                    "layer_recall_active_events": int(log_dict.get("layer_recall_active_events", 0)),
                },
            )

        if self.step % int(_YX_get(self.config, "gc_interval", 100)) == 0:
            if self.is_main_process:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()

    def train(self):
        if bool(_YX_get(self.config, "generate_before_train", False)) and self.is_main_process:
            print("[CHPM] generate_before_train is ignored: this trainer does not run VAE decode/evaluation.")

        max_iters = int(_YX_get(self.config, "max_iters", 1))
        if self.step >= max_iters:
            if self.is_main_process:
                print(
                    f"[CHPM] Resume step {self.step} already reached max_iters={max_iters}; "
                    "skipping training."
                )
            return
        progress_enabled = self.is_main_process and bool(_YX_get(self.config, "progress_bar", True))
        progress_bar = None
        if progress_enabled and tqdm is not None:
            progress_bar = tqdm(
                total=max_iters,
                initial=min(int(self.step), max_iters),
                desc="YX LayerRecall train",
                unit="step",
                dynamic_ncols=True,
                leave=True,
            )
            self._progress_bar_active = True
        elif progress_enabled:
            print(f"[progress] step {self.step}/{max_iters}")

        try:
            while True:
                step_before = int(self.step)
                for acc in range(self.gradient_accumulation_steps):
                    expected_accumulation = int(self.global_micro_step) % int(
                        self.gradient_accumulation_steps
                    )
                    if expected_accumulation != acc:
                        raise RuntimeError(
                            "global micro-step/accumulation cursor mismatch: "
                            f"global_micro_step={self.global_micro_step}, "
                            f"expected_accumulation={expected_accumulation}, loop={acc}"
                        )
                    self.accumulation_step = int(acc)
                    batch, pending_indices = self._peek_training_batch()
                    micro_before = int(self.global_micro_step)
                    data_epoch = int(batch.get("YX_data_epoch", -1))
                    data_cursor = int(batch.get("YX_data_sample_cursor", -1))
                    dataset_indices = [
                        int(value) for value in batch.get("YX_dataset_indices", [])
                    ]
                    prompt_hashes = [
                        canonical_sha256(prompts)
                        for prompts in batch.get("prompts", [])
                    ]
                    micro_step_seed = _YX_micro_step_seed(
                        base_seed=self.data_seed,
                        dp_rank=self.dp_rank,
                        dp_size=self.data_parallel_size,
                        step=self.step,
                        accumulation_step=acc,
                        accumulation_steps=self.gradient_accumulation_steps,
                    )
                    self.train_one_step(
                        batch,
                        accumulation_step=acc,
                        accumulation_steps=self.gradient_accumulation_steps,
                    )
                    self._commit_training_batch(pending_indices)
                    self.accumulation_step = int(
                        (acc + 1) % self.gradient_accumulation_steps
                    )
                    self._write_exact_resume_trace(
                        {
                            "time": float(time.time()),
                            "global_rank": int(self.global_rank),
                            "sp_rank": int(self.sp_rank),
                            "dp_rank": int(self.dp_rank),
                            "optimizer_step_before": int(step_before),
                            "optimizer_step_after": int(self.step),
                            "accumulation_step": int(acc),
                            "global_micro_step_before": micro_before,
                            "global_micro_step_after": int(self.global_micro_step),
                            "micro_step_seed": int(micro_step_seed),
                            "data_epoch_before": data_epoch,
                            "data_sample_cursor_before": data_cursor,
                            "dataset_indices": dataset_indices,
                            "prompt_hashes": prompt_hashes,
                            "committed": True,
                            "resume_checkpoint_path": self.resume_checkpoint_path,
                        }
                    )

                if self.accumulation_step != 0:
                    raise RuntimeError(
                        "optimizer step completed with nonzero accumulation cursor: "
                        f"{self.accumulation_step}"
                    )
                if self.global_micro_step != self.step * self.gradient_accumulation_steps:
                    raise RuntimeError(
                        "optimizer/global micro-step commit mismatch: "
                        f"step={self.step}, global_micro_step={self.global_micro_step}"
                    )

                if (not bool(_YX_get(self.config, "no_save", False))) and self.step % int(_YX_get(self.config, "log_iters", 1)) == 0:
                    torch.cuda.empty_cache()
                    self.save()
                    torch.cuda.empty_cache()

                barrier()
                if self.is_main_process:
                    current_time = time.time()
                    if self.previous_time is not None and not self.disable_wandb:
                        wandb.log({"per iteration time": current_time - self.previous_time}, step=self.step)
                    self.previous_time = current_time

                if progress_enabled and int(self.step) > step_before:
                    if progress_bar is not None:
                        progress_bar.update(int(self.step) - step_before)
                        if self.latest_train_status:
                            progress_bar.set_postfix(
                                loss=f"{self.latest_train_status['loss']:.4f}",
                                pred=f"{self.latest_train_status['L_pred']:.4f}",
                                grad=f"{self.latest_train_status['grad_norm']:.3g}",
                                layer_recall=int(self.latest_train_status["layer_recall_active"]),
                            )
                        progress_bar.refresh()
                    else:
                        pct = 100.0 * float(self.step) / max(1.0, float(max_iters))
                        print(f"[progress] step {self.step}/{max_iters} ({pct:.2f}%)")

                if self.step >= max_iters:
                    break
        finally:
            self._progress_bar_active = False
            if progress_bar is not None:
                progress_bar.close()
