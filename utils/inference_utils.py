# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0
"""Small helpers for release inference examples."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Sequence

import torch
from einops import rearrange
from torchvision.io import write_video

def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def unwrap_generator_state_dict(checkpoint: object, use_ema: bool = False) -> object:
    """Extract generator weights from supported LongLive checkpoint layouts."""
    if not isinstance(checkpoint, Mapping):
        return checkpoint
    if "generator" in checkpoint or "generator_ema" in checkpoint:
        key = "generator_ema" if use_ema and "generator_ema" in checkpoint else "generator"
        return checkpoint[key]
    if "model" in checkpoint:
        return checkpoint["model"]
    return checkpoint


def clean_fsdp_state_dict_keys(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Remove FSDP wrapper prefixes used by some EMA checkpoints."""
    return {str(key).replace("_fsdp_wrapped_module.", ""): value for key, value in state_dict.items()}


def load_generator_checkpoint(generator, checkpoint_path: str, *, use_ema: bool = False, strict: bool | None = None):
    """Load a LongLive generator checkpoint into ``generator``."""
    checkpoint = _torch_load(checkpoint_path)
    state_dict = unwrap_generator_state_dict(checkpoint, use_ema=use_ema)
    if use_ema:
        state_dict = clean_fsdp_state_dict_keys(state_dict)
    if strict is None:
        strict = not use_ema
    return generator.load_state_dict(state_dict, strict=strict)


def _load_lora_state_dict(lora_ckpt_path: str) -> Mapping[str, torch.Tensor]:
    """Load a LoRA checkpoint, unwrapping ``generator_lora`` when present."""
    checkpoint = _torch_load(lora_ckpt_path)
    if isinstance(checkpoint, Mapping) and "generator_lora" in checkpoint:
        return checkpoint["generator_lora"]
    return checkpoint


def apply_and_merge_lora(
    pipeline,
    config,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.bfloat16,
    verbose: bool = False,
):
    """Wrap ``pipeline.generator.model`` with a LoRA adapter, load weights, and merge.

    The merged module ends up structurally identical to the original generator,
    with each ``nn.Linear`` carrying the base weight plus the LoRA delta.

    Returns ``True`` when LoRA was applied and merged, ``False`` when the config
    did not request a LoRA adapter.
    """
    adapter_cfg = getattr(config, "adapter", None)
    lora_ckpt = getattr(config, "lora_ckpt", None)
    if adapter_cfg is None or not lora_ckpt:
        return False

    import peft
    from utils.lora_utils import configure_lora_for_model

    if device is not None:
        pipeline.generator.to(device=torch.device(device), dtype=dtype)
    else:
        pipeline.generator.to(dtype=dtype)

    if verbose:
        print(f"[LoRA] Wrapping generator with adapter config: {adapter_cfg}")
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=adapter_cfg,
        is_main_process=verbose,
    )

    if verbose:
        print(f"[LoRA] Loading LoRA weights from: {lora_ckpt}")
    lora_state = _load_lora_state_dict(lora_ckpt)
    peft.set_peft_model_state_dict(pipeline.generator.model, lora_state)  # type: ignore[arg-type]

    if verbose:
        print("[LoRA] Merging LoRA delta into base weights (merge_and_unload)...")
    pipeline.generator.model = pipeline.generator.model.merge_and_unload(safe_merge=True)
    pipeline.generator.model.eval().requires_grad_(False)
    pipeline.is_lora_enabled = False
    pipeline.is_lora_merged = True
    return True


def place_vae_for_streaming(pipeline, config) -> torch.device | None:
    """Move ``pipeline.vae`` to ``config.vae_device`` for streaming-pipeline decode.

    Only acts when both ``streaming_vae`` and ``vae_device`` are set; otherwise
    leaves the VAE on whatever device the rest of the pipeline already uses.
    Mirrors the relocation done in ``inference.py`` so that quick-start scripts
    can opt in to the streaming-pipeline VAE simply by enabling those config
    fields.
    """
    if not bool(getattr(config, "streaming_vae", False)):
        return None
    vae_device_str = getattr(config, "vae_device", None)
    if not vae_device_str:
        return None

    vae_device = torch.device(vae_device_str)
    pipeline.vae.to(device="cpu")
    pipeline.vae.to(device=vae_device)
    if hasattr(pipeline.vae, "mean"):
        pipeline.vae.mean = pipeline.vae.mean.to(device=vae_device)
        pipeline.vae.std = pipeline.vae.std.to(device=vae_device)
    return vae_device


def prepare_single_prompt_inputs(
    config,
    prompt: str,
    device: torch.device | str,
    *,
    dtype: torch.dtype = torch.bfloat16,
    batch_size: int = 1,
    generator: torch.Generator | None = None,
):
    """Create the per-block prompt list and latent noise for one text prompt."""
    num_frames = int(getattr(config, "num_output_frames", config.image_or_video_shape[1]))
    frames_per_block = int(getattr(config, "num_frame_per_block", 1))
    if num_frames % frames_per_block != 0:
        raise ValueError(f"num_frames={num_frames} must be divisible by num_frame_per_block={frames_per_block}")

    latent_shape = list(config.image_or_video_shape[2:])
    if len(latent_shape) != 3:
        raise ValueError(f"Expected latent shape [C, H, W], got {latent_shape}")

    num_blocks = num_frames // frames_per_block
    prompts = [[prompt] * num_blocks for _ in range(batch_size)]
    noise = torch.randn(
        [batch_size, num_frames, *latent_shape],
        device=device,
        dtype=dtype,
        generator=generator,
    )
    return noise, prompts


def video_to_uint8(video: torch.Tensor) -> torch.Tensor:
    """Convert a generated video tensor from [T, C, H, W] or [1, T, C, H, W] to uint8 THWC."""
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError("video_to_uint8 expects a single sample when a batch dimension is present.")
        video = video[0]
    if video.ndim != 4:
        raise ValueError(f"Expected video tensor with 4 dims, got shape={tuple(video.shape)}")
    if video.shape[1] in (1, 3):
        video = rearrange(video, "t c h w -> t h w c")
    return (255.0 * video.cpu()).clamp(0, 255).to(torch.uint8)


def save_video(video: torch.Tensor, output_path: str | os.PathLike, *, fps: int = 24) -> None:
    """Save a generated LongLive video tensor as an mp4 file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_video(str(output_path), video_to_uint8(video), fps=fps)
