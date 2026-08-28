# SPDX-License-Identifier: Apache-2.0

import math
import os
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn

from utils.config import wan_default_config
from utils.wan_5b_wrapper import WanDiffusionWrapper, WanTextEncoder, WanVAEWrapper
from utils.layer_recall import (
    LayerRecallConfig,
    clear_layer_recall_context,
    set_layer_recall_context,
)


def _YX_cuda_memory_snapshot(device: torch.device | int) -> Dict[str, float]:
    if not torch.cuda.is_available():
        return {
            "allocated_mb": 0.0,
            "reserved_mb": 0.0,
            "max_allocated_mb": 0.0,
            "max_reserved_mb": 0.0,
            "physical_free_mb": 0.0,
            "physical_total_mb": 0.0,
        }
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    divisor = float(1024**2)
    return {
        "allocated_mb": float(torch.cuda.memory_allocated(device) / divisor),
        "reserved_mb": float(torch.cuda.memory_reserved(device) / divisor),
        "max_allocated_mb": float(torch.cuda.max_memory_allocated(device) / divisor),
        "max_reserved_mb": float(torch.cuda.max_memory_reserved(device) / divisor),
        "physical_free_mb": float(free_bytes / divisor),
        "physical_total_mb": float(total_bytes / divisor),
    }


def _YX_to_container(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if OmegaConf.is_config(value):
        return dict(OmegaConf.to_container(value, resolve=True))
    return dict(value)


def _YX_get(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(key, default)
    getter = getattr(value, "get", None)
    if getter is not None:
        try:
            return getter(key, default)
        except Exception:
            pass
    return getattr(value, key, default)


def _YX_section_get(args: Any, section: str, key: str, default: Any = None, aliases: Tuple[str, ...] = ()) -> Any:
    for candidate in (key, *aliases):
        if hasattr(args, candidate):
            return getattr(args, candidate)
    section_value = getattr(args, section, None)
    for candidate in (key, *aliases):
        value = _YX_get(section_value, candidate, None)
        if value is not None:
            return value
    return default


def _YX_clean_param_name(name: str) -> str:
    return (
        name.replace("_fsdp_wrapped_module.", "")
        .replace("_checkpoint_wrapped_module.", "")
        .replace("_orig_mod.", "")
    )


@dataclass(frozen=True)
class _YXFrameShardMetadata:
    sp_size: int
    sp_rank: int
    global_frames: int
    local_frame_start: int
    local_frame_end: int

    @property
    def local_frames(self) -> int:
        return int(self.local_frame_end) - int(self.local_frame_start)

    @property
    def is_sp_shard(self) -> bool:
        return int(self.sp_size) > 1


@dataclass
class _YXStreamChunkOutput:
    local_flow: torch.Tensor
    local_x0: torch.Tensor
    full_context_x0: Optional[torch.Tensor]
    frame_shard: _YXFrameShardMetadata

    def __iter__(self):
        # Keep the historical two-value unpacking available to external probes.
        yield self.local_flow
        yield self.local_x0


def _YX_streaming_frame_shard(global_frames: int) -> Tuple[object, _YXFrameShardMetadata]:
    from wan_5b.distributed.streaming_ulysses import (
        local_frame_bounds,
        streaming_sp_info,
    )

    sp_group, sp_size, sp_rank = streaming_sp_info()
    local_start, local_end = local_frame_bounds(int(global_frames))
    return sp_group, _YXFrameShardMetadata(
        sp_size=int(sp_size),
        sp_rank=int(sp_rank),
        global_frames=int(global_frames),
        local_frame_start=int(local_start),
        local_frame_end=int(local_end),
    )


def _YX_slice_chunk_to_local(
    tensor: Optional[torch.Tensor],
    frame_shard: _YXFrameShardMetadata,
) -> Optional[torch.Tensor]:
    if tensor is None:
        return None
    if int(tensor.shape[1]) != int(frame_shard.global_frames):
        raise ValueError(
            "chunk tensor frame count does not match shard metadata: "
            f"tensor={tensor.shape[1]}, global={frame_shard.global_frames}"
        )
    return tensor[
        :,
        int(frame_shard.local_frame_start):int(frame_shard.local_frame_end),
    ].contiguous()


def _YX_all_gather_detached_context(local_x0: torch.Tensor) -> torch.Tensor:
    if local_x0.requires_grad:
        raise ValueError("Streaming SP context gather requires a detached x0 tensor")
    from wan_5b.distributed.streaming_ulysses import (
        all_gather_detached_frames,
    )

    return all_gather_detached_frames(local_x0)


def _YX_normalize_replicated_prediction(
    local_sum: torch.Tensor,
    local_valid_count: torch.Tensor,
    *,
    sp_group,
) -> torch.Tensor:
    from utils.chpm_sp import normalize_sp_prediction_local_sum

    return normalize_sp_prediction_local_sum(
        local_sum,
        local_valid_count,
        sp_group=sp_group,
    )


def _YX_replicated_prediction_log_values(
    local_sum: torch.Tensor,
    local_count: torch.Tensor,
    *,
    sp_group,
) -> Tuple[torch.Tensor, torch.Tensor]:
    from utils.chpm_sp import sp_global_detached_sum_count

    return sp_global_detached_sum_count(
        local_sum,
        local_count,
        sp_group=sp_group,
    )


def _YX_scale_replicated_reg(loss_reg: torch.Tensor, sp_size: int) -> torch.Tensor:
    from utils.chpm_sp import scale_replicated_regularization

    return scale_replicated_regularization(loss_reg, sp_size=sp_size)


def _YX_sp_parity_tensor_stats(tensor: torch.Tensor) -> Dict[str, torch.Tensor]:
    source = tensor.detach()
    values = source.float()
    finite = torch.isfinite(values)
    finite_count = finite.sum(dtype=torch.float32)
    count = values.new_tensor(values.numel(), dtype=torch.float32)
    denominator = finite_count.clamp(min=1.0)
    finite_values = torch.where(finite, values, torch.zeros_like(values))
    mean = finite_values.sum() / denominator
    centered = torch.where(finite, values - mean, torch.zeros_like(values))
    std = (centered.square().sum() / denominator).sqrt()
    minimum = torch.where(
        finite_count > 0,
        torch.where(finite, values, torch.full_like(values, float("inf"))).amin(),
        values.new_tensor(float("nan")),
    )
    maximum = torch.where(
        finite_count > 0,
        torch.where(finite, values, torch.full_like(values, float("-inf"))).amax(),
        values.new_tensor(float("nan")),
    )
    return {
        "count": count.detach(),
        "finite_count": finite_count.detach(),
        "nonfinite_count": (count - finite_count).detach(),
        "all_finite": (finite_count == count).to(dtype=torch.float32).detach(),
        "mean": mean.detach(),
        "std": std.detach(),
        "min": minimum.detach(),
        "max": maximum.detach(),
        "abs_max": torch.maximum(minimum.abs(), maximum.abs()).detach(),
    }


def _YX_sp_parity_anchor_record(
    *,
    enabled: bool,
    full_tensors: bool,
    sequence_parallel_size: int,
    chunk_index: int,
    start_frame: int,
    end_frame: int,
    prediction_target: str,
    chunk_noisy: torch.Tensor,
    local_timestep: torch.Tensor,
    teacher_target: torch.Tensor,
    student_prediction: torch.Tensor,
    chunk_sum: torch.Tensor,
    chunk_count: torch.Tensor,
) -> Optional[Dict[str, Any]]:
    if not enabled:
        return None

    tensors = {
        "chunk_noisy": chunk_noisy.detach(),
        "local_timestep": local_timestep.detach(),
        "teacher_target": teacher_target.detach(),
        "student_prediction": student_prediction.detach(),
        "chunk_sum": chunk_sum.detach(),
        "chunk_count": chunk_count.detach(),
    }
    global_frames = int(end_frame) - int(start_frame)
    local_frames = int(tensors["chunk_noisy"].shape[1])
    sp_size = max(1, int(sequence_parallel_size))
    record: Dict[str, Any] = {
        "chunk_index": int(chunk_index),
        "start_frame": int(start_frame),
        "end_frame": int(end_frame),
        "prediction_target": str(prediction_target),
        "full_tensors_included": bool(full_tensors),
        "frame_metadata": {
            "global_frames": int(global_frames),
            "local_frames": int(local_frames),
            "sequence_parallel_size": int(sp_size),
            "has_full_frames": bool(local_frames == global_frames),
            "is_sp_shard": bool(
                sp_size > 1
                and local_frames < global_frames
                and local_frames * sp_size == global_frames
            ),
        },
        "chunk_sum": tensors["chunk_sum"],
        "chunk_count": tensors["chunk_count"],
        "stats": {
            name: _YX_sp_parity_tensor_stats(tensor)
            for name, tensor in tensors.items()
        },
    }
    record["finite"] = {
        name: stats["all_finite"]
        for name, stats in record["stats"].items()
    }
    if full_tensors:
        for name in (
            "chunk_noisy",
            "local_timestep",
            "teacher_target",
            "student_prediction",
        ):
            record[name] = tensors[name]
    return record


def _YX_sp_parity_capture_payload(
    *,
    enabled: bool,
    full_tensors: bool,
    prediction_target: str,
    rollout_mode: str,
    anchor_every_n_frames: int,
    anchor_include_last_chunk: bool,
    num_frame_per_block: int,
    target_chunk_indices: List[int],
    records: Optional[List[Dict[str, Any]]],
) -> Optional[Dict[str, Any]]:
    if not enabled:
        return None
    return {
        "enabled": True,
        "full_tensors": bool(full_tensors),
        "prediction_target": str(prediction_target),
        "anchor_schedule": {
            "rollout_mode": str(rollout_mode),
            "anchor_every_n_frames": int(anchor_every_n_frames),
            "anchor_include_last_chunk": bool(anchor_include_last_chunk),
            "num_frame_per_block": int(num_frame_per_block),
            "chunk_indices": [int(index) for index in target_chunk_indices],
            "anchor_end_frames": [
                int((index + 1) * num_frame_per_block)
                for index in target_chunk_indices
            ],
        },
        "anchors": list(records or []),
    }


class CHPMModel(nn.Module):
    """Teacher/student prediction distillation for current-conditioned LayerRecall.

    This wrapper intentionally avoids DMD, fake_score/real_score, CFG, and VAE
    decode.  The teacher is a frozen causal Wan generator; the student is the
    same generator with LayerRecall enabled and only LayerRecall parameters trainable.
    """

    def __init__(self, args, device):
        super().__init__()
        self.args = args
        self.device = device
        self.dtype = torch.bfloat16 if getattr(args, "mixed_precision", False) else torch.float32
        self.num_frame_per_block = int(getattr(args, "num_frame_per_block", 1))
        self.sequence_parallel_size = int(getattr(args, "sequence_parallel_size", 1))
        self.layer_recall_replicated_params = bool(_YX_section_get(
            args,
            "chpm",
            "layer_recall_replicated_params",
            False,
        ))
        self.sp_parity_capture_enabled = bool(_YX_section_get(
            args,
            "chpm",
            "sp_parity_capture_enabled",
            False,
        ))
        self.sp_parity_capture_full_tensors = bool(_YX_section_get(
            args,
            "chpm",
            "sp_parity_capture_full_tensors",
            True,
        ))
        self.num_train_timestep = int(getattr(args, "num_train_timestep", 1000))
        self.min_step = int(getattr(args, "min_step", 0))
        self.max_step = int(getattr(args, "max_step", self.num_train_timestep))
        self.min_history_chunks = int(_YX_section_get(args, "chpm", "min_history_chunks", 1, ()))
        self.prediction_loss_weight = float(_YX_section_get(args, "chpm", "prediction_loss_weight", 1.0, ()))
        self.reg_weight = float(_YX_section_get(args, "chpm", "regularization_weight", 1.0e-4, ()))
        self.use_training_weight = bool(_YX_section_get(args, "chpm", "use_training_weight", False, ()))
        self.rollout_mode = "student_full_rollout"
        self.anchor_every_n_frames = int(_YX_section_get(
            args,
            "chpm",
            "anchor_every_n_frames",
            0,
            (),
        ) or 0)
        self.anchor_include_last_chunk = bool(_YX_section_get(
            args,
            "chpm",
            "anchor_include_last_chunk",
            True,
            (),
        ))
        self.anchor_backward_mode = "immediate"
        self.prediction_target = str(_YX_section_get(
            args,
            "chpm",
            "prediction_target",
            "denoised_latent",
            (),
        )).lower()
        self.clean_latent_source = str(_YX_section_get(
            args,
            "chpm",
            "clean_latent_source",
            "dataset",
            (),
        )).lower()
        self.teacher_target_device = str(_YX_section_get(
            args,
            "chpm",
            "teacher_target_device",
            "cpu",
            (),
        )).lower()
        if self.teacher_target_device not in {"cpu", "cuda"}:
            raise ValueError(f"Unsupported teacher_target_device: {self.teacher_target_device}")
        self.teacher_runtime_cpu_offload = bool(_YX_section_get(
            args,
            "chpm",
            "teacher_runtime_cpu_offload",
            False,
            (),
        ))

        self.trainable_allowlist = ("layer_recall",)

        base_model_kwargs = _YX_to_container(getattr(args, "model_kwargs", {}))
        student_kwargs = dict(base_model_kwargs)
        student_kwargs.update(_YX_to_container(getattr(args, "student_model_kwargs", None)))
        teacher_kwargs = dict(base_model_kwargs)
        teacher_kwargs.update(_YX_to_container(getattr(args, "teacher_model_kwargs", None)))
        student_local_attn = _YX_section_get(args, "chpm", "student_local_attn_size", None)
        teacher_local_attn = _YX_section_get(args, "chpm", "teacher_local_attn_size", None)
        if student_local_attn is not None:
            student_kwargs["local_attn_size"] = int(student_local_attn)
        if teacher_local_attn is not None:
            teacher_kwargs["local_attn_size"] = int(teacher_local_attn)
        sink_size = _YX_section_get(
            args,
            "chpm",
            "sink_size",
            _YX_section_get(args, "inference", "sink_size", getattr(args, "sink_size", None)),
            (),
        )
        if sink_size is not None:
            student_kwargs["sink_size"] = int(sink_size)
            teacher_kwargs["sink_size"] = int(sink_size)
        self.student_model_kwargs = dict(student_kwargs)
        self.teacher_model_kwargs = dict(teacher_kwargs)

        model_name = str(student_kwargs.get("model_name", "Wan2.2-TI2V-5B"))
        if "5B" not in model_name:
            raise ValueError(f"Only Wan2.2-TI2V-5B is supported, got {model_name}")
        self.model_name = model_name
        self.frame_seq_length = math.prod(list(args.image_or_video_shape)[-2:]) // 4
        model_defaults = wan_default_config[self.model_name]
        self.num_transformer_blocks = int(model_defaults["num_transformer_blocks"])
        self.num_heads = int(model_defaults["num_heads"])
        self.head_dim = int(model_defaults["head_dim"])

        YX_wan_model_root = getattr(args, "YX_wan_model_root", None) or student_kwargs.get("YX_wan_model_root", None)

        self.student = WanDiffusionWrapper(**student_kwargs, is_causal=True)
        self.teacher = WanDiffusionWrapper(**teacher_kwargs, is_causal=True)
        for wrapper in (self.student, self.teacher):
            wrapper.model.num_frame_per_block = self.num_frame_per_block
            if getattr(args, "independent_first_frame", False) and not getattr(args, "i2v", False):
                wrapper.model.independent_first_frame = True

        allow_gradient_checkpointing = bool(_YX_section_get(
            args,
            "chpm",
            "allow_gradient_checkpointing",
            False,
            (),
        ))
        if bool(getattr(args, "gradient_checkpointing", False)) and allow_gradient_checkpointing:
            self.student.enable_gradient_checkpointing()

        self.scheduler = self.student.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        layer_recall_config = LayerRecallConfig.from_repo_config(args, YX_rank=rank)
        if not layer_recall_config.layer_recall_enabled:
            raise ValueError("CHPM requires layer_recall.layer_recall_enabled=true")
        if not layer_recall_config.layer_recall_current_conditioned_enabled:
            raise ValueError("CHPM requires current-conditioned LayerRecall")
        self.layer_recall_config = layer_recall_config
        self._dit(self.student).configure_layer_recall(self.layer_recall_config)
        if self.layer_recall_replicated_params:
            self.configure_student_layer_recall_fp32_island()

        self.teacher.requires_grad_(False)
        self.teacher.eval()
        self.student.requires_grad_(False)
        self.capture_pre_fsdp_trainable_layer_recall_params()

        self.text_encoder = WanTextEncoder(YX_wan_model_root=YX_wan_model_root)
        self.text_encoder.requires_grad_(False)
        self.vae = None
        if bool(getattr(args, "load_raw_video", False)):
            self.vae = WanVAEWrapper(YX_wan_model_root=YX_wan_model_root)
            self.vae.requires_grad_(False)

    @staticmethod
    def _unwrap(wrapper):
        return wrapper.module if hasattr(wrapper, "module") else wrapper

    @classmethod
    def _dit(cls, wrapper):
        return cls._unwrap(wrapper).model

    def configure_student_layer_recall_trainable_params(self) -> List[str]:
        self.student.requires_grad_(False)
        trainable_names = []
        for name, param in self.student.named_parameters():
            clean_name = _YX_clean_param_name(name)
            if any(token in clean_name for token in self.trainable_allowlist):
                param.requires_grad_(True)
                trainable_names.append(clean_name)
        if len(trainable_names) == 0:
            raise RuntimeError(
                "No student LayerRecall parameters matched allowlist "
                f"{self.trainable_allowlist}. Ensure configure_layer_recall ran first."
            )
        return trainable_names

    def configure_student_layer_recall_fp32_island(self) -> Tuple[str, ...]:
        """Validate modules using FP32 masters with causal BF16 compute casts."""
        dit = self._dit(self.student)
        installed = []
        for name in (
            "layer_recall_current_norm",
            "layer_recall_current_mlp",
            "layer_recall_current_gate",
        ):
            module = getattr(dit, name, None)
            if module is None:
                continue
            installed.append(name)
        if bool(getattr(self.layer_recall_config, "layer_recall_current_conditioned_enabled", False)):
            expected = {
                "layer_recall_current_norm",
                "layer_recall_current_mlp",
                "layer_recall_current_gate",
            }
            missing = sorted(expected.difference(installed))
            if missing:
                raise RuntimeError(
                    "Current-conditioned replicated LayerRecall FP32 island is incomplete: "
                    f"missing {missing}"
                )
        self.layer_recall_fp32_island_modules = tuple(installed)
        self.layer_recall_forward_compute_policy = "fp32_master_differentiable_activation_dtype_cast"
        return self.layer_recall_fp32_island_modules

    def student_layer_recall_named_parameters(self) -> Iterable[Tuple[str, torch.nn.Parameter]]:
        for name, param in self.student.named_parameters():
            clean_name = _YX_clean_param_name(name)
            if param.requires_grad and any(token in clean_name for token in self.trainable_allowlist):
                yield clean_name, param

    def capture_pre_fsdp_trainable_layer_recall_params(self) -> List[str]:
        """Capture full LayerRecall parameter metadata while parameters are still unsharded."""
        trainable_names = self.configure_student_layer_recall_trainable_params()
        trainable_params = list(self.student_layer_recall_named_parameters())
        self._pre_fsdp_trainable_layer_recall_named_param_objects = tuple(trainable_params)
        self.pre_fsdp_trainable_layer_recall_param_names = list(trainable_names)
        self.pre_fsdp_trainable_layer_recall_param_tensor_count = len(trainable_params)
        self.pre_fsdp_trainable_layer_recall_param_count = sum(
            int(param.numel()) for _, param in trainable_params
        )
        return list(trainable_names)

    def pre_fsdp_trainable_layer_recall_named_param_objects(
        self,
    ) -> Tuple[Tuple[str, torch.nn.Parameter], ...]:
        captured = getattr(
            self,
            "_pre_fsdp_trainable_layer_recall_named_param_objects",
            (),
        )
        if not captured:
            raise RuntimeError("Pre-FSDP LayerRecall Parameter objects have not been captured")
        return tuple(captured)

    def layer_recall_architecture_summary(self) -> Dict[str, Any]:
        config = self.layer_recall_config
        num_layers = max(
            0,
            int(getattr(config, "layer_recall_num_layers", self.num_transformer_blocks)),
        )
        all_layer_ids = list(range(num_layers))
        enabled_layer_ids = sorted({
            int(layer_id)
            for layer_id in config.memory_sensitive_layers
            if 0 <= int(layer_id) < num_layers
        })
        enabled_layer_set = set(enabled_layer_ids)
        disabled_layer_ids = [
            layer_id for layer_id in all_layer_ids if layer_id not in enabled_layer_set
        ]
        return {
            "layer_recall_num_layers": num_layers,
            "memory_sensitive_layers": enabled_layer_ids,
            "memory_sensitive_layers_csv": ",".join(str(item) for item in enabled_layer_ids),
            "memory_sensitive_layer_count": len(enabled_layer_ids),
            "original_window_layers": disabled_layer_ids,
            "original_window_layer_count": len(disabled_layer_ids),
            "layer_recall_visible_layout": "sink_selected_current",
            "disabled_layer_visible_layout": "original_window",
        }

    @staticmethod
    def _layer_recall_event_metrics(layer_recall_counters: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "memory_sensitive_layer_events": int(
                layer_recall_counters.get("memory_sensitive_layer_events", 0) or 0
            ),
            "original_window_layer_events": int(
                layer_recall_counters.get("original_window_layer_events", 0) or 0
            ),
            "layer_recall_active_events": int(
                layer_recall_counters.get("layer_recall_layout_applied_events", 0) or 0
            ),
        }

    def load_generator_weights(self, checkpoint: Any) -> Dict[str, Any]:
        state_dict = self._extract_generator_state_dict(checkpoint)
        student_missing, student_unexpected = self.load_student_weights(state_dict)
        teacher_missing, teacher_unexpected = self.load_teacher_weights(state_dict)
        return {
            "student_missing": list(student_missing),
            "student_unexpected": list(student_unexpected),
            "teacher_missing": list(teacher_missing),
            "teacher_unexpected": list(teacher_unexpected),
        }

    def load_student_weights(self, checkpoint: Any) -> Tuple[List[str], List[str]]:
        state_dict = checkpoint if isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()) else self._extract_generator_state_dict(checkpoint)
        return self.student.load_state_dict(state_dict, strict=False)

    def load_teacher_weights(self, checkpoint: Any) -> Tuple[List[str], List[str]]:
        state_dict = checkpoint if isinstance(checkpoint, dict) and all(torch.is_tensor(v) for v in checkpoint.values()) else self._extract_generator_state_dict(checkpoint)
        return self.teacher.load_state_dict(state_dict, strict=False)

    def load_layer_recall_state_dict(self, checkpoint: Any) -> Tuple[List[str], List[str]]:
        if not isinstance(checkpoint, dict) or checkpoint.get("trainer") != "chpm":
            raise ValueError("LayerRecall weights must come from a CHPM checkpoint")
        if int(checkpoint.get("checkpoint_version", -1)) != 3:
            raise ValueError("LayerRecall weights require CHPM checkpoint version 3")
        layer_recall_state = checkpoint.get("layer_recall_state_dict")
        if not isinstance(layer_recall_state, dict) or not layer_recall_state:
            raise ValueError("CHPM checkpoint is missing layer_recall_state_dict")
        incompatible = self.student.load_state_dict(layer_recall_state, strict=False)
        missing = [key for key in incompatible.missing_keys if "layer_recall" in key]
        unexpected = list(incompatible.unexpected_keys)
        return missing, unexpected

    @staticmethod
    def _extract_generator_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
        if isinstance(checkpoint, dict):
            for key in ("generator", "student_generator", "student", "model"):
                value = checkpoint.get(key, None)
                if isinstance(value, dict):
                    return value
            if all(torch.is_tensor(value) for value in checkpoint.values()):
                return checkpoint
        raise ValueError(
            "Checkpoint does not contain a generator state_dict under one of "
            "'generator', 'student_generator', 'student', 'model', or as a raw state_dict."
        )

    def _get_timestep(
        self,
        min_timestep: int,
        max_timestep: int,
        batch_size: int,
        num_frame: int,
        num_frame_per_block: int,
    ) -> torch.Tensor:
        timestep = torch.randint(
            int(min_timestep),
            int(max_timestep),
            [batch_size, num_frame],
            device=self.device,
            dtype=torch.long,
        )
        timestep = timestep.reshape(timestep.shape[0], -1, num_frame_per_block)
        timestep[:, :, 1:] = timestep[:, :, 0:1]
        return timestep.reshape(timestep.shape[0], -1)

    def _new_kv_cache(
        self,
        *,
        batch_size: int,
        dtype: torch.dtype,
        device: torch.device,
        num_frames: int,
        role: str,
    ) -> List[Dict[str, Any]]:
        sp_size = max(1, int(self.sequence_parallel_size))
        if int(self.num_heads) % sp_size != 0:
            raise ValueError(
                f"num_heads ({self.num_heads}) must be divisible by SP size ({sp_size})"
            )
        cache_heads = int(self.num_heads) // sp_size
        block_token_size = self.num_frame_per_block * self.frame_seq_length
        min_frames = max(3 * self.num_frame_per_block, self.num_frame_per_block)
        if role == "teacher":
            configured = int(_YX_section_get(
                self.args,
                "chpm",
                "teacher_physical_cache_frames",
                0,
                (),
            ) or 0)
            local_attn = int(self.teacher_model_kwargs.get("local_attn_size", -1) or -1)
            base_frames = 3 * self.num_frame_per_block if local_attn == -1 else local_attn
            capacity_frames = max(min_frames, configured, base_frames)
        else:
            configured = int(_YX_section_get(
                self.args,
                "chpm",
                "student_physical_cache_frames",
                0,
                (),
            ) or 0)
            physical = int(getattr(self.layer_recall_config, "layer_recall_physical_cache_frames", 0) or 0)
            local_attn = self.student_model_kwargs.get("local_attn_size", -1)
            local_attn = int(local_attn if local_attn is not None else -1)
            base_frames = 3 * self.num_frame_per_block if local_attn == -1 else local_attn
            capacity_frames = max(min_frames, configured, physical, base_frames)

        capacity_blocks = max(1, math.ceil(int(capacity_frames) / self.num_frame_per_block))
        kv_cache_size = capacity_blocks * block_token_size
        cache = []
        for _ in range(self.num_transformer_blocks):
            cache.append(
                {
                    "k": torch.zeros(
                        [batch_size, kv_cache_size, cache_heads, self.head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "v": torch.zeros(
                        [batch_size, kv_cache_size, cache_heads, self.head_dim],
                        dtype=dtype,
                        device=device,
                    ),
                    "block_token_size": block_token_size,
                    "max_blocks": capacity_blocks,
                    "num_heads": cache_heads,
                    "global_num_heads": self.num_heads,
                    "num_filled_blocks": 0,
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "pinned_start": torch.tensor([-1], dtype=torch.long, device=device),
                    "pinned_len": torch.tensor([0], dtype=torch.long, device=device),
                }
            )
        return cache

    def _reset_layer_recall_memory(self, wrapper) -> None:
        dit = self._dit(wrapper)
        if hasattr(dit, "reset_layer_recall_memory"):
            dit.reset_layer_recall_memory()

    def _condition_for_chunk(
        self,
        conditional_dict: Dict[str, Any],
        *,
        batch_size: int,
        chunk_index: int,
    ) -> Dict[str, Any]:
        result = {}
        for key, value in conditional_dict.items():
            if not torch.is_tensor(value) or value.shape[0] == batch_size:
                result[key] = value
                continue
            if value.shape[0] % batch_size != 0:
                result[key] = value
                continue
            reshaped = value.reshape(batch_size, -1, *value.shape[1:])
            selected = min(int(chunk_index), int(reshaped.shape[1]) - 1)
            result[key] = reshaped[:, selected].contiguous()
        return result

    def _forward_stream_chunk(
        self,
        wrapper,
        *,
        chunk_latent: torch.Tensor,
        conditional_dict: Dict[str, torch.Tensor],
        timestep: torch.Tensor,
        kv_cache: List[Dict[str, Any]],
        crossattn_cache: List[Any],
        chunk_index: int,
        chunk_start_frame: int,
        call_type: str,
        skip_cache_update: bool = False,
        gather_detached_context_x0: bool = False,
        global_step: Optional[int] = None,
        denoising_step_index: int = 0,
    ) -> _YXStreamChunkOutput:
        sp_group, frame_shard = _YX_streaming_frame_shard(
            int(chunk_latent.shape[1])
        )
        del sp_group
        if int(frame_shard.sp_size) != int(self.sequence_parallel_size):
            raise RuntimeError(
                "Streaming SP runtime/config mismatch: "
                f"runtime={frame_shard.sp_size}, configured={self.sequence_parallel_size}"
            )
        set_layer_recall_context(
            YX_step=int(global_step) if global_step is not None else -1,
            YX_call_type=str(call_type),
            YX_cfg_branch="pos",
            YX_chunk_index=int(chunk_index),
            YX_chunk_start_frame=int(chunk_start_frame),
            YX_cache_start_frame=int(chunk_start_frame),
            YX_num_frames=int(chunk_latent.shape[1]),
            YX_frame_seq_length=int(self.frame_seq_length),
            YX_denoising_step_index=int(denoising_step_index),
            YX_skip_cache_update=bool(skip_cache_update),
        )
        try:
            local_flow, local_x0 = wrapper(
                noisy_image_or_video=chunk_latent,
                conditional_dict=conditional_dict,
                timestep=timestep,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=int(chunk_start_frame) * int(self.frame_seq_length),
                cache_start=int(chunk_start_frame) * int(self.frame_seq_length),
            )
            expected_shape = (
                int(chunk_latent.shape[0]),
                int(frame_shard.local_frames),
            )
            for name, tensor in (("flow", local_flow), ("x0", local_x0)):
                if tuple(int(value) for value in tensor.shape[:2]) != expected_shape:
                    raise ValueError(
                        f"Streaming {name} shape {tuple(tensor.shape[:2])} does not "
                        f"match local [B, F] shard {expected_shape}"
                    )
            full_context_x0 = None
            if gather_detached_context_x0:
                full_context_x0 = _YX_all_gather_detached_context(
                    local_x0.detach()
                )
                if int(full_context_x0.shape[1]) != int(frame_shard.global_frames):
                    raise ValueError(
                        "Gathered context x0 does not reconstruct the full chunk: "
                        f"got {full_context_x0.shape[1]}, expected {frame_shard.global_frames}"
                    )
            return _YXStreamChunkOutput(
                local_flow=local_flow,
                local_x0=local_x0,
                full_context_x0=full_context_x0,
                frame_shard=frame_shard,
            )
        finally:
            clear_layer_recall_context()

    def _prediction_chunk_loss(
        self,
        student_flow: torch.Tensor,
        teacher_flow: torch.Tensor,
        timestep: torch.Tensor,
        loss_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        loss = F.mse_loss(student_flow.float(), teacher_flow.float(), reduction="none").mean(dim=(2, 3, 4))
        if self.use_training_weight:
            weight = self.scheduler.training_weight(timestep).to(device=loss.device, dtype=loss.dtype)
            loss = loss * weight
        if loss_mask is not None:
            mask = loss_mask.to(device=loss.device, dtype=loss.dtype)
            loss = loss * mask
            return loss.sum(), mask.sum()
        return loss.sum(), loss.new_tensor(loss.numel(), dtype=loss.dtype)

    def _regularization_loss(self, reference: torch.Tensor) -> torch.Tensor:
        reg_terms = []
        for _, param in self.student_layer_recall_named_parameters():
            if param.numel() > 0:
                reg_terms.append(param.float().pow(2).mean())
        if not reg_terms:
            return reference.float().new_zeros(())
        return torch.stack(reg_terms).mean()

    def _store_teacher_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = tensor.detach()
        if self.teacher_target_device == "cpu":
            return tensor.to(device="cpu", dtype=self.dtype, copy=True)
        return tensor.to(device=self.device, dtype=self.dtype, copy=True)

    def _load_teacher_tensor(self, tensor: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=reference.device, dtype=reference.dtype, non_blocking=True)

    def _prepare_teacher_phase(self) -> None:
        if not self.teacher_runtime_cpu_offload:
            return
        self.teacher.to(device=self.device, dtype=self.dtype)
        self.teacher.eval()

    def _finish_teacher_phase(self) -> None:
        if not self.teacher_runtime_cpu_offload:
            return
        self.teacher.to(device="cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _prediction_chunk_indices(self, num_chunks: int) -> List[int]:
        if self.anchor_every_n_frames <= 0:
            raise ValueError("anchor_every_n_frames must be greater than zero")
        if self.anchor_every_n_frames % self.num_frame_per_block != 0:
            raise ValueError(
                "anchor_every_n_frames must be divisible by num_frame_per_block, got "
                f"{self.anchor_every_n_frames} and {self.num_frame_per_block}"
            )
        anchor_every_chunks = max(1, self.anchor_every_n_frames // self.num_frame_per_block)
        selected = [
            int(chunk_index)
            for chunk_index in range(int(num_chunks))
            if (chunk_index + 1) % anchor_every_chunks == 0
        ]
        if self.anchor_include_last_chunk:
            selected.append(int(num_chunks) - 1)
        selected = sorted(set(chunk for chunk in selected if chunk >= self.min_history_chunks))
        if not selected:
            selected = [int(num_chunks) - 1]
        return selected

    def _expected_prediction_loss_count(
        self,
        *,
        target_chunk_indices: List[int],
        loss_mask: Optional[torch.Tensor],
        reference: torch.Tensor,
        local_frame_start: int = 0,
        local_frame_end: Optional[int] = None,
        clamp_min: bool = True,
    ) -> torch.Tensor:
        local_frame_end = (
            self.num_frame_per_block
            if local_frame_end is None
            else int(local_frame_end)
        )
        local_frame_start = int(local_frame_start)
        local_frames = local_frame_end - local_frame_start
        if local_frame_start < 0 or local_frame_end > self.num_frame_per_block or local_frames <= 0:
            raise ValueError(
                "invalid local prediction-loss frame interval: "
                f"[{local_frame_start}, {local_frame_end})"
            )
        total = reference.float().new_zeros(())
        for chunk_index in target_chunk_indices:
            start = int(chunk_index) * self.num_frame_per_block + local_frame_start
            end = int(chunk_index) * self.num_frame_per_block + local_frame_end
            if loss_mask is None:
                total = total + total.new_tensor(
                    int(reference.shape[0]) * int(local_frames),
                    dtype=total.dtype,
                )
            else:
                total = total + loss_mask[:, start:end].to(device=total.device, dtype=total.dtype).sum()
        return total.clamp(min=1.0) if clamp_min else total

    def chpm_loss(
        self,
        *,
        image_or_video_shape: List[int],
        conditional_dict: Dict[str, torch.Tensor],
        clean_latent: torch.Tensor,
        initial_latent: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        loss_mask_global_valid_count: Optional[torch.Tensor] = None,
        global_step: Optional[int] = None,
        backward_callback: Optional[Callable[[torch.Tensor], None]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        del loss_mask_global_valid_count
        batch_size, num_frame = int(image_or_video_shape[0]), int(image_or_video_shape[1])
        if num_frame % self.num_frame_per_block != 0:
            raise ValueError(
                f"num_frame ({num_frame}) must be divisible by "
                f"num_frame_per_block ({self.num_frame_per_block})"
            )
        if loss_mask is not None and int(loss_mask.shape[1]) != num_frame:
            raise ValueError(
                "chpm_loss expects a global-frame loss mask; "
                f"got {loss_mask.shape[1]} mask frames for {num_frame} latent frames"
            )

        self._reset_layer_recall_memory(self.student)
        self._reset_layer_recall_memory(self.teacher)

        noise = torch.randn_like(clean_latent)
        index = self._get_timestep(
            self.min_step,
            self.max_step,
            batch_size,
            num_frame,
            self.num_frame_per_block,
        )
        timestep = self.scheduler.timesteps[index].to(dtype=self.dtype, device=self.device)

        context_frames = 0
        if getattr(self.args, "i2v", False) and initial_latent is not None:
            context_frames = int(initial_latent.shape[1])
            timestep[:, :context_frames] = 0

        noisy_latents = self.scheduler.add_noise(
            clean_latent.flatten(0, 1),
            noise.flatten(0, 1),
            timestep.flatten(0, 1),
        ).unflatten(0, (batch_size, num_frame))
        if context_frames > 0:
            noisy_latents[:, :context_frames] = initial_latent.to(
                device=noisy_latents.device,
                dtype=noisy_latents.dtype,
            )

        num_chunks = num_frame // self.num_frame_per_block
        target_chunk_indices = self._prediction_chunk_indices(num_chunks)
        target_chunk_set = set(target_chunk_indices)
        sp_group, chunk_frame_shard = _YX_streaming_frame_shard(
            self.num_frame_per_block
        )
        if int(chunk_frame_shard.sp_size) != int(self.sequence_parallel_size):
            raise RuntimeError(
                "Streaming SP runtime/config mismatch before rollout: "
                f"runtime={chunk_frame_shard.sp_size}, configured={self.sequence_parallel_size}"
            )
        expected_pred_loss_count = self._expected_prediction_loss_count(
            target_chunk_indices=target_chunk_indices,
            loss_mask=loss_mask,
            reference=clean_latent,
        )
        local_expected_pred_loss_count = self._expected_prediction_loss_count(
            target_chunk_indices=target_chunk_indices,
            loss_mask=loss_mask,
            reference=clean_latent,
            local_frame_start=chunk_frame_shard.local_frame_start,
            local_frame_end=chunk_frame_shard.local_frame_end,
            clamp_min=False,
        )
        replicated_loss_math = bool(self.layer_recall_replicated_params)
        immediate_backward = backward_callback is not None
        sp_parity_records: Optional[List[Dict[str, Any]]] = (
            [] if self.sp_parity_capture_enabled else None
        )
        teacher_phase_start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        teacher_memory_before_model_move = _YX_cuda_memory_snapshot(self.device)
        self._prepare_teacher_phase()
        teacher_memory_after_model_move = _YX_cuda_memory_snapshot(self.device)
        teacher_cache = self._new_kv_cache(
            batch_size=batch_size,
            dtype=clean_latent.dtype,
            device=clean_latent.device,
            num_frames=num_frame,
            role="teacher",
        )
        teacher_cache_shape = list(teacher_cache[0]["k"].shape)
        teacher_effective_cache_frames = int(teacher_cache[0]["k"].shape[1]) // int(
            self.frame_seq_length
        )
        teacher_requested_cache_frames = int(_YX_section_get(
            self.args,
            "chpm",
            "teacher_physical_cache_frames",
            0,
            (),
        ) or 0)
        teacher_local_attn_frames = int(self.teacher_model_kwargs.get("local_attn_size", -1) or -1)
        teacher_memory_after_cache_allocation = _YX_cuda_memory_snapshot(self.device)
        teacher_memory_trace: List[Dict[str, Any]] = []
        teacher_max_visible_frames = 0
        teacher_crossattn = [None] * self.num_transformer_blocks

        teacher_records: List[Dict[str, Any]] = []
        for chunk_index in range(num_chunks):
            start = chunk_index * self.num_frame_per_block
            end = start + self.num_frame_per_block
            chunk_slice = slice(start, end)
            chunk_condition = self._condition_for_chunk(
                conditional_dict,
                batch_size=batch_size,
                chunk_index=chunk_index,
            )
            chunk_noisy = noisy_latents[:, chunk_slice].contiguous()
            chunk_clean = clean_latent[:, chunk_slice].contiguous()
            chunk_timestep = timestep[:, chunk_slice].contiguous()
            should_predict = chunk_index in target_chunk_set

            teacher_flow = None
            teacher_x0 = None
            teacher_full_context_x0 = None
            if should_predict or self.clean_latent_source == "teacher_rollout":
                with torch.no_grad():
                    teacher_output = self._forward_stream_chunk(
                        self.teacher,
                        chunk_latent=chunk_noisy,
                        conditional_dict=chunk_condition,
                        timestep=chunk_timestep,
                        kv_cache=teacher_cache,
                        crossattn_cache=teacher_crossattn,
                        chunk_index=chunk_index,
                        chunk_start_frame=start,
                        call_type="denoise",
                        skip_cache_update=True,
                        gather_detached_context_x0=(
                            self.clean_latent_source == "teacher_rollout"
                            and chunk_index < num_chunks - 1
                        ),
                        global_step=global_step,
                        denoising_step_index=0,
                    )
                    teacher_flow = teacher_output.local_flow
                    teacher_x0 = teacher_output.local_x0
                    teacher_full_context_x0 = teacher_output.full_context_x0

            if self.clean_latent_source == "teacher_rollout":
                if teacher_x0 is None:
                    raise RuntimeError("teacher_rollout requires teacher denoised latent for context_update")
                if chunk_index < num_chunks - 1:
                    if teacher_full_context_x0 is None:
                        raise RuntimeError(
                            "teacher_rollout requires detached full-chunk x0 for context_update"
                        )
                    context_chunk = teacher_full_context_x0
                else:
                    context_chunk = None
            else:
                context_chunk = chunk_clean.detach()

            record: Dict[str, Any] = {
                "chunk_index": int(chunk_index),
                "chunk_start": int(start),
                "chunk_end": int(end),
                "should_predict": bool(should_predict),
            }
            if should_predict:
                if self.prediction_target in {"denoised_latent", "x0", "pred_x0", "clean_latent"}:
                    record["target"] = self._store_teacher_tensor(teacher_x0)
                    record["target_name"] = "x0"
                elif self.prediction_target in {"flow", "velocity", "v", "flow_pred"}:
                    record["target"] = self._store_teacher_tensor(teacher_flow)
                    record["target_name"] = "flow"
                else:
                    raise ValueError(f"Unsupported prediction_target: {self.prediction_target}")
            teacher_records.append(record)

            zero_timestep = torch.zeros_like(chunk_timestep)
            if chunk_index < num_chunks - 1:
                with torch.no_grad():
                    self._forward_stream_chunk(
                        self.teacher,
                        chunk_latent=context_chunk,
                        conditional_dict=chunk_condition,
                        timestep=zero_timestep,
                        kv_cache=teacher_cache,
                        crossattn_cache=teacher_crossattn,
                        chunk_index=chunk_index,
                        chunk_start_frame=start,
                        call_type="context_update",
                        global_step=global_step,
                        denoising_step_index=-1,
                    )

            del chunk_noisy, chunk_clean, chunk_timestep, context_chunk
            if teacher_flow is not None:
                del teacher_flow
            if teacher_x0 is not None:
                del teacher_x0
            if teacher_full_context_x0 is not None:
                del teacher_full_context_x0

            effective_attn_frames = (
                teacher_effective_cache_frames
                if teacher_local_attn_frames == -1
                else teacher_local_attn_frames
            )
            teacher_max_visible_frames = max(
                teacher_max_visible_frames,
                min(int((chunk_index + 1) * self.num_frame_per_block), int(effective_attn_frames)),
            )
            if (chunk_index + 1) % 8 == 0 or chunk_index == num_chunks - 1:
                snapshot = _YX_cuda_memory_snapshot(self.device)
                snapshot.update({
                    "chunk_index": int(chunk_index),
                    "completed_frames": int((chunk_index + 1) * self.num_frame_per_block),
                    "cache_local_end_tokens": int(teacher_cache[0]["local_end_index"].item()),
                    "max_visible_frames": int(teacher_max_visible_frames),
                })
                teacher_memory_trace.append(snapshot)

        teacher_phase_peak = _YX_cuda_memory_snapshot(self.device)
        del teacher_cache, teacher_crossattn
        self._finish_teacher_phase()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        teacher_memory_after_release = _YX_cuda_memory_snapshot(self.device)
        teacher_phase_time_s = float(time.perf_counter() - teacher_phase_start_time)

        student_phase_start_time = time.perf_counter()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)
        student_cache = self._new_kv_cache(
            batch_size=batch_size,
            dtype=clean_latent.dtype,
            device=clean_latent.device,
            num_frames=num_frame,
            role="student",
        )
        student_cache_shape = list(student_cache[0]["k"].shape)
        student_effective_cache_frames = int(student_cache[0]["k"].shape[1]) // int(
            self.frame_seq_length
        )
        student_memory_after_cache_allocation = _YX_cuda_memory_snapshot(self.device)
        student_memory_trace: List[Dict[str, Any]] = []
        student_crossattn = [None] * self.num_transformer_blocks

        pred_loss_sum = clean_latent.float().new_zeros(())
        pred_loss_count = clean_latent.float().new_zeros(())
        pred_backward_loss = clean_latent.float().new_zeros(())
        pred_chunks = 0
        student_prefix_denoise_chunks = 0
        student_context_source_counts: Dict[str, int] = {}

        for chunk_index in range(num_chunks):
            start = chunk_index * self.num_frame_per_block
            end = start + self.num_frame_per_block
            chunk_slice = slice(start, end)
            chunk_condition = self._condition_for_chunk(
                conditional_dict,
                batch_size=batch_size,
                chunk_index=chunk_index,
            )
            chunk_noisy = noisy_latents[:, chunk_slice].contiguous()
            chunk_timestep = timestep[:, chunk_slice].contiguous()
            local_chunk_noisy = _YX_slice_chunk_to_local(
                chunk_noisy,
                chunk_frame_shard,
            )
            local_chunk_timestep = _YX_slice_chunk_to_local(
                chunk_timestep,
                chunk_frame_shard,
            )
            global_chunk_mask = (
                loss_mask[:, chunk_slice].contiguous()
                if loss_mask is not None
                else None
            )
            chunk_mask = _YX_slice_chunk_to_local(
                global_chunk_mask,
                chunk_frame_shard,
            )
            teacher_record = teacher_records[chunk_index]

            should_predict = bool(teacher_record["should_predict"])
            student_context_chunk = None
            student_context_source = "student_pred_detached"
            if should_predict:
                student_output = self._forward_stream_chunk(
                    self.student,
                    chunk_latent=chunk_noisy,
                    conditional_dict=chunk_condition,
                    timestep=chunk_timestep,
                    kv_cache=student_cache,
                    crossattn_cache=student_crossattn,
                    chunk_index=chunk_index,
                    chunk_start_frame=start,
                    call_type="denoise",
                    skip_cache_update=True,
                    gather_detached_context_x0=chunk_index < num_chunks - 1,
                    global_step=global_step,
                    denoising_step_index=0,
                )
                student_flow = student_output.local_flow
                student_x0 = student_output.local_x0
                if self.prediction_target in {"denoised_latent", "x0", "pred_x0", "clean_latent"}:
                    student_pred = student_x0
                elif self.prediction_target in {"flow", "velocity", "v", "flow_pred"}:
                    student_pred = student_flow
                else:
                    raise ValueError(f"Unsupported prediction_target: {self.prediction_target}")
                teacher_pred = self._load_teacher_tensor(teacher_record["target"], student_pred)
                chunk_sum, chunk_count = self._prediction_chunk_loss(
                    student_pred,
                    teacher_pred,
                    local_chunk_timestep,
                    chunk_mask,
                )
                if replicated_loss_math:
                    normalized_chunk_loss = _YX_normalize_replicated_prediction(
                        chunk_sum,
                        local_expected_pred_loss_count,
                        sp_group=sp_group,
                    )
                    logging_sum, logging_count = _YX_replicated_prediction_log_values(
                        chunk_sum,
                        chunk_count,
                        sp_group=sp_group,
                    )
                else:
                    normalized_chunk_loss = chunk_sum / expected_pred_loss_count
                    logging_sum, logging_count = chunk_sum, chunk_count
                if sp_parity_records is not None:
                    capture_record = _YX_sp_parity_anchor_record(
                        enabled=True,
                        full_tensors=self.sp_parity_capture_full_tensors,
                        sequence_parallel_size=self.sequence_parallel_size,
                        chunk_index=chunk_index,
                        start_frame=start,
                        end_frame=end,
                        prediction_target=self.prediction_target,
                        chunk_noisy=local_chunk_noisy,
                        local_timestep=local_chunk_timestep,
                        teacher_target=teacher_pred,
                        student_prediction=student_pred,
                        chunk_sum=chunk_sum,
                        chunk_count=chunk_count,
                    )
                    if capture_record is not None:
                        sp_parity_records.append(capture_record)
                if immediate_backward:
                    backward_callback(
                        self.prediction_loss_weight * normalized_chunk_loss
                    )
                    pred_backward_loss = pred_backward_loss + normalized_chunk_loss.detach()
                else:
                    pred_backward_loss = pred_backward_loss + normalized_chunk_loss
                pred_loss_sum = pred_loss_sum + (
                    logging_sum.detach() if replicated_loss_math else logging_sum
                )
                pred_loss_count = pred_loss_count + (
                    logging_count.detach() if replicated_loss_math else logging_count
                )
                pred_chunks += 1
                student_context_chunk = student_output.full_context_x0
                if chunk_index < num_chunks - 1 and student_context_chunk is None:
                    raise RuntimeError("CHPM anchor requires full detached student x0 context")
                del teacher_pred, student_pred, student_flow, student_x0, student_output
            else:
                with torch.no_grad():
                    student_output = self._forward_stream_chunk(
                        self.student,
                        chunk_latent=chunk_noisy,
                        conditional_dict=chunk_condition,
                        timestep=chunk_timestep,
                        kv_cache=student_cache,
                        crossattn_cache=student_crossattn,
                        chunk_index=chunk_index,
                        chunk_start_frame=start,
                        call_type="denoise",
                        skip_cache_update=True,
                        gather_detached_context_x0=chunk_index < num_chunks - 1,
                        global_step=global_step,
                        denoising_step_index=0,
                    )
                    _student_flow = student_output.local_flow
                    student_x0 = student_output.local_x0
                student_context_chunk = student_output.full_context_x0
                if chunk_index < num_chunks - 1 and student_context_chunk is None:
                    raise RuntimeError(
                        "student_rollout_detached requires full detached x0 context"
                    )
                student_context_source = "student_rollout_detached"
                student_prefix_denoise_chunks += 1
                del _student_flow, student_x0, student_output

            zero_timestep = torch.zeros_like(chunk_timestep)
            if chunk_index < num_chunks - 1:
                if student_context_chunk is None:
                    raise RuntimeError("CHPM rollout requires detached student context for every chunk")
                context_chunk = student_context_chunk
                student_context_source_counts[student_context_source] = (
                    student_context_source_counts.get(student_context_source, 0) + 1
                )
                with torch.no_grad():
                    self._forward_stream_chunk(
                        self.student,
                        chunk_latent=context_chunk,
                        conditional_dict=chunk_condition,
                        timestep=zero_timestep,
                        kv_cache=student_cache,
                        crossattn_cache=student_crossattn,
                        chunk_index=chunk_index,
                        chunk_start_frame=start,
                        call_type="context_update",
                        global_step=global_step,
                        denoising_step_index=-1,
                    )
                del context_chunk
            del chunk_noisy, local_chunk_noisy, chunk_timestep, local_chunk_timestep
            if global_chunk_mask is not None:
                del global_chunk_mask
            if chunk_mask is not None:
                del chunk_mask

            if (chunk_index + 1) % 8 == 0 or chunk_index == num_chunks - 1:
                snapshot = _YX_cuda_memory_snapshot(self.device)
                snapshot.update({
                    "chunk_index": int(chunk_index),
                    "completed_frames": int((chunk_index + 1) * self.num_frame_per_block),
                    "cache_local_end_tokens": int(student_cache[0]["local_end_index"].item()),
                })
                student_memory_trace.append(snapshot)

        student_phase_peak = _YX_cuda_memory_snapshot(self.device)
        student_phase_time_s = float(time.perf_counter() - student_phase_start_time)

        loss_pred = pred_loss_sum / pred_loss_count.clamp(min=1.0)
        loss_reg = self._regularization_loss(clean_latent)
        loss_reg_backward = (
            _YX_scale_replicated_reg(loss_reg, chunk_frame_shard.sp_size)
            if replicated_loss_math
            else loss_reg
        )
        reg_backward_used = False
        if immediate_backward:
            if self.reg_weight != 0.0 and loss_reg_backward.requires_grad:
                backward_callback(self.reg_weight * loss_reg_backward)
                reg_backward_used = True
            total_loss = (
                self.prediction_loss_weight * loss_pred.detach()
                + self.reg_weight * loss_reg.detach()
            )
        else:
            total_loss = (
                self.prediction_loss_weight * pred_backward_loss
                + self.reg_weight * loss_reg_backward
            )
        logged_total_loss = (
            self.prediction_loss_weight * loss_pred.detach()
            + self.reg_weight * loss_reg.detach()
        )
        layer_recall_counters = {}
        student_logger = getattr(self._dit(self.student), "layer_recall_logger", None)
        if student_logger is not None:
            layer_recall_counters = student_logger.snapshot_counters()
        log_dict = {
            "loss_pred": float(loss_pred.detach().cpu().item()),
            "loss_reg": float(loss_reg.detach().cpu().item()),
            "loss_total": float(logged_total_loss.cpu().item()),
            "prediction_target": self.prediction_target,
            "clean_latent_source": self.clean_latent_source,
            "teacher_target_device": self.teacher_target_device,
            "teacher_runtime_cpu_offload": bool(self.teacher_runtime_cpu_offload),
            "teacher_student_schedule": "teacher_phase_cpu_targets_then_student_phase",
            "rollout_mode": self.rollout_mode,
            "anchor_every_n_frames": int(self.anchor_every_n_frames),
            "anchor_include_last_chunk": bool(self.anchor_include_last_chunk),
            "anchor_backward_mode": self.anchor_backward_mode,
            "immediate_backward_used": bool(immediate_backward),
            "reg_backward_used": bool(reg_backward_used),
            "pred_chunk_indices_csv": ",".join(str(index) for index in target_chunk_indices),
            "pred_anchor_end_frames_csv": ",".join(
                str((index + 1) * self.num_frame_per_block) for index in target_chunk_indices
            ),
            "expected_pred_loss_count": float(
                (
                    pred_loss_count
                    if replicated_loss_math
                    else expected_pred_loss_count
                ).detach().cpu().item()
            ),
            "prediction_loss_normalization": (
                "local_sum_over_sp_global_valid_count"
                if replicated_loss_math
                else "managed_full_sequence_sum_over_valid_count"
            ),
            "layer_recall_replicated_params": bool(self.layer_recall_replicated_params),
            "regularization_sp_scale": (
                1.0 / float(chunk_frame_shard.sp_size)
                if replicated_loss_math
                else 1.0
            ),
            "student_prefix_context_gradient": "detached",
            "student_prefix_denoise_chunks": int(student_prefix_denoise_chunks),
            "student_context_student_rollout_detached_chunks": int(
                student_context_source_counts.get("student_rollout_detached", 0)
            ),
            "student_context_student_pred_detached_chunks": int(
                student_context_source_counts.get("student_pred_detached", 0)
            ),
            "student_context_source_counts": dict(student_context_source_counts),
            "num_pred_chunks": int(pred_chunks),
            "num_chunks": int(num_chunks),
            "min_history_chunks": int(self.min_history_chunks),
            "layer_recall_log_events": int(layer_recall_counters.get("YX_log_events", 0)),
            "layer_recall_gate_active_events": int(layer_recall_counters.get("YX_gate_active_events", 0)),
            "layer_recall_soft_or_st_events": int(layer_recall_counters.get("YX_soft_or_st_events", 0)),
            "layer_recall_soft_memory_positive_events": int(layer_recall_counters.get("YX_soft_memory_positive_events", 0)),
            "layer_recall_candidate_chunks_max": int(layer_recall_counters.get("YX_candidate_chunks_max", 0)),
            "teacher_requested_cache_frames": int(teacher_requested_cache_frames),
            "teacher_effective_cache_frames": int(teacher_effective_cache_frames),
            "teacher_local_attn_frames": int(teacher_local_attn_frames),
            "teacher_max_visible_frames": int(teacher_max_visible_frames),
            "teacher_cache_shape": teacher_cache_shape,
            "student_effective_cache_frames": int(student_effective_cache_frames),
            "student_cache_shape": student_cache_shape,
            "teacher_phase_time_s": float(teacher_phase_time_s),
            "student_phase_time_s": float(student_phase_time_s),
            "teacher_memory_before_model_move": teacher_memory_before_model_move,
            "teacher_memory_after_model_move": teacher_memory_after_model_move,
            "teacher_memory_after_cache_allocation": teacher_memory_after_cache_allocation,
            "teacher_memory_after_release": teacher_memory_after_release,
            "teacher_phase_peak": teacher_phase_peak,
            "teacher_memory_trace": teacher_memory_trace,
            "student_memory_after_cache_allocation": student_memory_after_cache_allocation,
            "student_phase_peak": student_phase_peak,
            "student_memory_trace": student_memory_trace,
        }
        log_dict.update(self.layer_recall_architecture_summary())
        log_dict.update(self._layer_recall_event_metrics(layer_recall_counters))
        sp_parity_payload = _YX_sp_parity_capture_payload(
            enabled=self.sp_parity_capture_enabled,
            full_tensors=self.sp_parity_capture_full_tensors,
            prediction_target=self.prediction_target,
            rollout_mode=self.rollout_mode,
            anchor_every_n_frames=self.anchor_every_n_frames,
            anchor_include_last_chunk=self.anchor_include_last_chunk,
            num_frame_per_block=self.num_frame_per_block,
            target_chunk_indices=target_chunk_indices,
            records=sp_parity_records,
        )
        if sp_parity_payload is not None:
            log_dict["_sp_parity_capture"] = sp_parity_payload
        return total_loss, log_dict
