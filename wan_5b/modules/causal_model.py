# Adopted from https://github.com/guandeh17/Self-Forcing
# SPDX-License-Identifier: Apache-2.0
# # Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

try:
    from transformers.models.x_clip.modeling_x_clip import x_clip_loss
except ImportError:
    x_clip_loss = None
from wan_5b.modules.attention import attention
from wan_5b.modules.model import (
    WanRMSNorm,
    rope_apply,
    WanLayerNorm,
    WanCrossAttention,
    rope_params,
    sinusoidal_embedding_1d,
    WanCrossAttention,
    flash_attention
)
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from diffusers.configuration_utils import ConfigMixin, register_to_config
from torch.nn.attention.flex_attention import BlockMask
from diffusers.models.modeling_utils import ModelMixin
import os
import torch.nn as nn
import torch
import math
import torch.distributed as dist
from typing import Any, Dict, List, Sequence, Tuple

from wan_5b.distributed.streaming_ulysses import (
    assert_sp_metadata_consistent,
    current_token_summary,
    k_summary,
    local_frame_bounds,
    streaming_sp_info,
    ulysses_head_to_seq,
    ulysses_packed_qkv_seq_to_head,
)

# wan 5b model compilation for flexattention
flex_attention = torch.compile(
    flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


from utils.position_embedding_utils import (
    compute_temporal_freqs as _compute_temporal_freqs,
    select_temporal_offset_for_sample,
)
from utils.layer_recall import (
    HistoryChunkRecord,
    LayerRecallConfig,
    LayerRecallSelectionLogger,
    LayerRecallMemoryBank,
    assemble_slot_visible_plan,
    get_layer_recall_context,
    is_layer_recall_enabled_for_layer,
    materialize_layer_recall_slot,
    pool_pre_rope_k,
    should_use_layer_recall_selection,
    stable_topk,
    straight_through_hard_value,
)


# iter-21: cache freqs_i across causal_rope_apply calls within a chunk.
# All ~60 layer Q/K calls in one chunk share identical (f,h,w,start_frame,
# t_scale,temporal_offset_i,method,original_seq_len) but recompute the same
# concatenated freqs tensor each time. LRU keeps memory bounded.
# NOTE: this cache holds tensors across torch.compile step boundaries which
# is incompatible with cudagraphs (mode=reduce-overhead). If cudagraphs
# path is enabled in the future, this cache must be removed alongside
# refactoring of the KV cache scalar tensors (global_end_index, etc.).
_FREQS_I_CACHE: "dict[tuple, torch.Tensor]" = {}
_FREQS_I_CACHE_MAX = 16
# iter-21 + iter-41: cache is on by default (iter-21 win). Set
# LLV2_FREQS_I_CACHE=0 to disable for future cudagraphs experiments (the
# cache holds tensors created inside torch.compile that get marked as
# cudagraph-pool memory; reading them on a later compile step crashes with
# "accessing tensor output of CUDAGraphs that has been overwritten").
_FREQS_I_CACHE_ENABLED = os.environ.get("LLV2_FREQS_I_CACHE", "1") == "1"

# iter-42: Triton fp32 RoPE kernel (utils/rope_triton.py). Default ON.
# Replaces the fp64 complex view_as_complex × complex_freqs × view_as_real
# chain with a single fused Triton kernel. Quality validated bit-exact at
# bf16 (unit test agent/rope_unit_test.py: max|Δ|=7.8e-3 = single bf16 ULP).
# Set LLV2_TRITON_ROPE=0 to revert to the fp64 path.
# When enabled, _FREQS_I_CACHE stores (freqs_i_complex, cos_f32, sin_f32);
# when disabled, stores (freqs_i_complex, None, None).
_TRITON_ROPE_ENABLED = os.environ.get("LLV2_TRITON_ROPE", "1") == "1"

# Cudagraph experiment only. Default OFF because the out-of-place temp-KV
# construction removes mutated-input skips but is materially slower than the
# in-place temporary cache update path.
_CGRAPH_OUTPLACE_KV_ENABLED = os.environ.get("LLV2_CGRAPH_OUTPLACE_KV", "0") == "1"

# iter-43/44: Triton fused adaLN-modulate kernel (utils/adaln_triton.py).
# Default ON after iter-44 added `@triton.autotune` over (num_warps, num_stages).
# iter-43 (no autotune) was FLAT vs iter-42 (median -1.0%, p90 +5.8%, total
# identical) — fixed config beat the eager median but jitter on tail.
# iter-44 (autotuned) is WIN: median -1.7%, p90 -1.6%, total -1.5%, FPS +1.5%
# vs iter-42, quality in run-to-run noise floor (mean|Δ|=0.68 vs noise=0.69).
# Unit test agent/adaln_unit_test.py: max|Δ|=3.1e-2 (1 bf16 ULP), mean=1.1e-3.
# Set LLV2_TRITON_ADALN=0 to fall back to eager nn.LayerNorm + Python modulate.
_TRITON_ADALN_ENABLED = os.environ.get("LLV2_TRITON_ADALN", "1") == "1"

# iter-31: per-chunk Python-int metadata published by CausalWanModel.forward
# so attention forwards can read Python ints without `.item()` graph breaks.
# Single-thread inference assumption — overwritten before each model() call.
_CURRENT_GRID_META: "dict[str, int]" = {}


def _layer_recall_module_forward_in_dtype(
    module: nn.Module,
    value: torch.Tensor,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    """Run an FP32-master LayerRecall module with differentiable BF16 compute weights."""
    floating_params = {
        name: parameter
        for name, parameter in module.named_parameters()
        if torch.is_floating_point(parameter)
    }
    if value.dtype == compute_dtype and all(
        parameter.dtype == compute_dtype
        for parameter in floating_params.values()
    ):
        return module(value)

    state = {
        name: (
            parameter.to(dtype=compute_dtype)
            if torch.is_floating_point(parameter)
            else parameter
        )
        for name, parameter in module.named_parameters()
    }
    state.update(
        {
            name: (
                buffer.to(dtype=compute_dtype)
                if torch.is_floating_point(buffer)
                else buffer
            )
            for name, buffer in module.named_buffers()
        }
    )
    return torch.func.functional_call(
        module,
        state,
        (value.to(dtype=compute_dtype),),
        strict=True,
    )

# iter-35: removed (LOST). Consolidating duplicate .item() reads caused
# p90 latency to spike +10% — dynamo apparently traced more specialized
# paths when local vars were used in branches vs fresh .item() reads each
# time. Restored original .item() per-use pattern.


def _YX_validate_streaming_sp_preflight(
    *,
    YX_sp_size: int,
    YX_global_frames: int,
    YX_num_heads: int,
    YX_use_relative_rope: bool = False,
    YX_temporal_offset=0.0,
    YX_current_conditioned_enabled: bool = False,
    YX_current_detach_summary: bool = True,
    YX_pinned_start: int = -1,
    YX_pinned_len: int = 0,
    YX_global_sink_size: int = 0,
) -> None:
    """Validate the training-compatible cached path before its first collective."""
    YX_sp_size = int(YX_sp_size)
    if YX_sp_size == 1:
        return
    if YX_sp_size != 2:
        raise ValueError(
            "Training-compatible Streaming Ulysses cached path supports SP=2 only, "
            f"got SP={YX_sp_size}"
        )
    if int(YX_global_frames) % YX_sp_size != 0:
        raise ValueError(
            f"chunk frames ({YX_global_frames}) must be divisible by SP size ({YX_sp_size})"
        )
    if int(YX_num_heads) % YX_sp_size != 0:
        raise ValueError(
            f"attention heads ({YX_num_heads}) must be divisible by SP size ({YX_sp_size})"
        )

    if bool(YX_use_relative_rope):
        raise ValueError("Streaming Ulysses does not support relative RoPE")
    if bool(YX_current_conditioned_enabled) and not bool(YX_current_detach_summary):
        raise ValueError(
            "Training-compatible Streaming Ulysses cached path requires "
            "layer_recall_current_detach_summary=true when current-conditioned LayerRecall is enabled"
        )
    if torch.is_tensor(YX_temporal_offset):
        raise ValueError("Streaming Ulysses does not support tensor temporal_offset")
    if float(YX_temporal_offset) != 0.0:
        raise ValueError("Streaming Ulysses requires temporal_offset=0")
    if (
        int(YX_global_sink_size) > 0
        or (int(YX_pinned_start) >= 0 and int(YX_pinned_len) > 0)
    ):
        raise ValueError("Streaming Ulysses does not support active pinned/multi-shot sink")


def _YX_prepare_streaming_sp_kv_cache(
    YX_kv_cache,
    *,
    YX_sp_size: int,
    YX_sp_rank: int,
    YX_num_heads: int,
):
    """Lazily shard externally allocated full-head caches for the cached path."""
    if int(YX_sp_size) == 1 or YX_kv_cache is None:
        return YX_kv_cache
    local_heads = int(YX_num_heads) // int(YX_sp_size)
    head_start = int(YX_sp_rank) * local_heads
    head_end = head_start + local_heads
    for layer_index, cache in enumerate(YX_kv_cache):
        for name in ("k", "v"):
            tensor = cache.get(name)
            if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
                raise ValueError(
                    f"layer {layer_index} cache {name} must be [B, T, H, D]"
                )
            cache_heads = int(tensor.shape[2])
            if cache_heads == int(YX_num_heads):
                cache[name] = tensor[:, :, head_start:head_end, :].contiguous()
            elif cache_heads != local_heads:
                raise ValueError(
                    f"layer {layer_index} cache {name} has {cache_heads} heads; "
                    f"expected {YX_num_heads} full or {local_heads} local heads"
                )
        cache["YX_streaming_sp_size"] = int(YX_sp_size)
        cache["YX_streaming_sp_rank"] = int(YX_sp_rank)
    return YX_kv_cache


def _YX_slice_streaming_sp_chunk(
    YX_patched_x,
    YX_e: torch.Tensor,
    YX_e0: torch.Tensor,
    YX_grid_sizes: torch.Tensor,
    *,
    YX_frame_start: int,
    YX_frame_end: int,
):
    """Slice one contiguous frame shard while keeping x/e/e0/grid aligned."""
    frame_start = int(YX_frame_start)
    frame_end = int(YX_frame_end)
    if frame_start < 0 or frame_end <= frame_start:
        raise ValueError(f"invalid frame interval [{frame_start}, {frame_end})")
    if not isinstance(YX_grid_sizes, torch.Tensor) or YX_grid_sizes.ndim != 2:
        raise ValueError("grid_sizes must be a [B, 3] tensor")
    if int(YX_grid_sizes.shape[1]) != 3:
        raise ValueError("grid_sizes must contain [F, H, W]")
    batch_size = len(YX_patched_x)
    if int(YX_grid_sizes.shape[0]) != batch_size:
        raise ValueError("patched inputs and grid_sizes must have the same batch size")
    if int(YX_e.shape[0]) != batch_size or int(YX_e0.shape[0]) != batch_size:
        raise ValueError("patched inputs, e, and e0 must have the same batch size")

    local_x = []
    for batch_index, tensor in enumerate(YX_patched_x):
        if tensor.ndim != 5:
            raise ValueError("patch embeddings must have shape [1, C, F, H, W]")
        global_frames = int(YX_grid_sizes[batch_index, 0])
        if frame_end > global_frames or int(tensor.shape[2]) != global_frames:
            raise ValueError("frame interval must fit the full patch-embedding chunk")
        local_x.append(tensor[:, :, frame_start:frame_end].contiguous())

    if frame_end > int(YX_e.shape[1]) or frame_end > int(YX_e0.shape[1]):
        raise ValueError("frame interval must fit both e and e0")
    local_e = YX_e[:, frame_start:frame_end].contiguous()
    local_e0 = YX_e0[:, frame_start:frame_end].contiguous()
    local_grid_sizes = YX_grid_sizes.clone()
    local_grid_sizes[:, 0] = frame_end - frame_start
    return local_x, local_e, local_e0, local_grid_sizes


def _YX_make_chunk_memory_record(
    *,
    YX_chunk_index: int,
    YX_start_frame: int,
    YX_num_frames: int,
    YX_cache_start_token: int,
    YX_cache_end_token: int,
    YX_global_start_token: int,
    YX_global_end_token: int,
    YX_summary: torch.Tensor,
) -> HistoryChunkRecord:
    return HistoryChunkRecord(
        chunk_index=int(YX_chunk_index),
        start_frame=int(YX_start_frame),
        num_frames=int(YX_num_frames),
        cache_start_token=int(YX_cache_start_token),
        cache_end_token=int(YX_cache_end_token),
        global_start_token=int(YX_global_start_token),
        global_end_token=int(YX_global_end_token),
        summary=YX_summary,
    )


def _YX_assert_layer_recall_sp_debug_consistent(
    *,
    YX_layer_index: int,
    YX_candidate_records,
    YX_selected_chunk_ids,
    YX_current_start: int,
    YX_current_end: int,
    YX_local_start: int,
    YX_local_end: int,
) -> None:
    candidate_ids = [int(record.chunk_index) for record in YX_candidate_records]
    global_ranges = [int(YX_current_start), int(YX_current_end)]
    local_ranges = [int(YX_local_start), int(YX_local_end)]
    for record in YX_candidate_records:
        global_ranges.extend(int(value) for value in record.global_token_range)
        local_ranges.extend(int(value) for value in record.token_range)
    label_prefix = f"layer_{int(YX_layer_index)}"
    assert_sp_metadata_consistent(
        candidate_ids,
        label=f"{label_prefix}_candidate_chunk_ids",
    )
    assert_sp_metadata_consistent(
        [int(item) for item in YX_selected_chunk_ids],
        label=f"{label_prefix}_selected_chunk_ids",
    )
    assert_sp_metadata_consistent(
        global_ranges,
        label=f"{label_prefix}_global_cache_ranges",
    )
    assert_sp_metadata_consistent(
        local_ranges,
        label=f"{label_prefix}_local_cache_ranges",
    )


def causal_rope_apply(x, grid_sizes, freqs, start_frame=0, t_scale=1.0,
                      method="linear", original_seq_len=None,
                      temporal_offset=0.0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    # iter-30: accept Python list/tuple to skip the .tolist() graph break.
    # Callers that already have Python ints (sink_grid, local_grid, window_grid_sizes)
    # now pass a plain list instead of `torch.tensor([[..]]).expand(...)`.
    if isinstance(grid_sizes, (list, tuple)):
        fwh_list = grid_sizes
    else:
        fwh_list = grid_sizes.tolist()
    for i, (f, h, w) in enumerate(fwh_list):
        seq_len = f * h * w

        # precompute multipliers — only needed for the fp64 complex path.
        # iter-42: skip the bf16→fp64 cast + view_as_complex when the Triton
        # kernel will be used (it consumes bf16 directly).
        if not _TRITON_ROPE_ENABLED:
            x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
                seq_len, n, -1, 2))
        temporal_offset_i = select_temporal_offset_for_sample(
            temporal_offset, i, f, start_frame=start_frame)

        # iter-21: cache freqs_i. iter-41: gate the cache behind
        # LLV2_FREQS_I_CACHE=1 (default off). The cache stores tensors
        # created inside torch.compile, which cudagraph allocator considers
        # owned by the per-step memory pool — reading them on a later step
        # races with the pool's reuse. Disabling the cache unblocks
        # `mode=reduce-overhead` for cudagraphs; the recomputation cost is
        # tiny (60 layer calls × per-chunk concat ≈ 0.5% wall) compared to
        # the cudagraphs unlock potential.
        if _FREQS_I_CACHE_ENABLED:
            if torch.is_tensor(temporal_offset_i):
                if temporal_offset_i.ndim == 0:
                    offset_key = float(temporal_offset_i.item())
                else:
                    offset_key = ("tensor", id(temporal_offset_i))
            else:
                offset_key = float(temporal_offset_i)
            cache_key = (
                f, h, w, start_frame, t_scale, method,
                original_seq_len, offset_key, x.device.type, x.device.index,
            )
            cache_entry = _FREQS_I_CACHE.get(cache_key)
        else:
            cache_entry = None
            cache_key = None

        if cache_entry is None:
            temporal_freqs = _compute_temporal_freqs(
                freqs[0], f, start_frame, t_scale, x.device,
                method=method, original_seq_len=original_seq_len,
                temporal_offset=temporal_offset_i)
            freqs_i_complex = torch.cat([
                temporal_freqs.view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ], dim=-1).reshape(seq_len, 1, -1)
            if _TRITON_ROPE_ENABLED:
                # iter-42: store (cos, sin) fp32 derived once; freqs_i_complex
                # itself only kept for the legacy fp64 path.
                from utils.rope_triton import _split_complex_to_cos_sin
                cos_f32, sin_f32 = _split_complex_to_cos_sin(freqs_i_complex)
                cache_entry = (freqs_i_complex, cos_f32, sin_f32)
            else:
                cache_entry = (freqs_i_complex, None, None)
            if _FREQS_I_CACHE_ENABLED:
                if len(_FREQS_I_CACHE) >= _FREQS_I_CACHE_MAX:
                    _FREQS_I_CACHE.pop(next(iter(_FREQS_I_CACHE)))
                _FREQS_I_CACHE[cache_key] = cache_entry
        freqs_i, cos_f32, sin_f32 = cache_entry

        # apply rotary embedding
        if _TRITON_ROPE_ENABLED:
            # iter-42: Triton fp32 kernel — replaces the fp64 complex128 path.
            # iter-46: kernel takes full x[i] + seq_len and emits rotated-or-
            # passthrough output in a single launch, eliminating the
            # `.contiguous()` slice + outer `torch.cat`. Bit-exact preserved.
            from utils.rope_triton import rope_apply_triton
            x_i = rope_apply_triton(x[i], cos_f32, sin_f32, seq_len=seq_len)
        else:
            x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
            x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


class MultiShotT2VCrossAttention(WanCrossAttention):

    def forward(self, x, context, context_lens, is_teacher_forcing=False, crossattn_cache=None):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B * num_chunks, L2, C]
            context_lens(Tensor): Shape [B] or [B * num_chunks]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        # Original batch size (videos)
        b_orig, L1, C = x.size()
        n, d = self.num_heads, self.head_dim

        # Effective batch size for cross-attention (videos * chunks)
        b_ctx = context.size(0)
        assert b_ctx % b_orig == 0, f"context batch ({b_ctx}) must be a multiple of x batch ({b_orig})"
        num_chunks = b_ctx // b_orig

        # Prepare context_lens for [B * num_chunks] if needed
        if context_lens is not None and context_lens.numel() == b_orig:
            context_lens = context_lens.repeat_interleave(num_chunks)
        elif context_lens is not None:
            assert context_lens.numel() == b_ctx, \
                f"context_lens must have length {b_orig} or {b_ctx}, got {context_lens.numel()}"
        # Helper to run standard cross-attention on a given x_chunk of shape [B * num_chunks, L_chunk, C]
        def _cross_attend(x_chunk):
            b_eff, L_chunk, _ = x_chunk.size()

            # compute query, key, value
            q = self.norm_q(self.q(x_chunk)).view(b_eff, -1, n, d)

            # iter-24: Bypass crossattn_cache. Cached K/V tensors escape the
            # cudagraph memory pool across torch.compile step boundaries and
            # block `mode=reduce-overhead`; recompute K/V for this path.
            k = self.norm_k(self.k(context)).view(b_eff, -1, n, d)
            v = self.v(context).view(b_eff, -1, n, d)

            # compute attention
            x_attn = attention(q, k, v, k_lens=context_lens)

            # output projection
            x_attn = x_attn.flatten(2)
            x_attn = self.o(x_attn)
            return x_attn

        if not is_teacher_forcing:
            # -------------------------------
            # Regular multi-shot: all tokens attend text, we just chunk along L1
            # x: [B, L1, C] -> [B * num_chunks, L1 / num_chunks, C]
            # -------------------------------
            assert L1 % num_chunks == 0, \
                f"L1 ({L1}) must be divisible by num_chunks ({num_chunks})"
            tokens_per_chunk = L1 // num_chunks

            x_chunked = x.view(b_orig, num_chunks, tokens_per_chunk, C)
            x_chunked = x_chunked.reshape(b_ctx, tokens_per_chunk, C)

            x_attn = _cross_attend(x_chunked)  # [B * num_chunks, tokens_per_chunk, C]

            # reshape back to [B, L1, C]
            x_attn = x_attn.view(b_orig, num_chunks, tokens_per_chunk, C)
            x_attn = x_attn.reshape(b_orig, L1, C)
            return x_attn

        # -------------------------------
        # Teacher forcing:
        # x is typically [B, 2 * L_tf, C], where the first half is clean and
        # the second half is noisy. Apply multi-shot cross-attention to both
        # halves.
        # -------------------------------
        assert L1 % 2 == 0, f"In teacher-forcing mode, L1 ({L1}) should be even."
        half = L1 // 2
        x_clean = x[:, :half, :]       # [B, L_tf, C]
        x_noisy = x[:, half:, :]       # [B, L_tf, C]

        def _chunk_and_attend(x_part):
            L_part = x_part.size(1)
            assert L_part % num_chunks == 0, \
                f"Segment length ({L_part}) must be divisible by num_chunks ({num_chunks})"
            tokens_per_chunk = L_part // num_chunks

            # [B, L_part, C] -> [B * num_chunks, L_part / num_chunks, C]
            x_chunked = x_part.view(b_orig, num_chunks, tokens_per_chunk, C)
            x_chunked = x_chunked.reshape(b_ctx, tokens_per_chunk, C)

            x_attn = _cross_attend(x_chunked)  # [B * num_chunks, tokens_per_chunk, C]
            x_attn = x_attn.view(b_orig, num_chunks, tokens_per_chunk, C)
            x_attn = x_attn.reshape(b_orig, L_part, C)
            return x_attn

        x_clean_attn = _chunk_and_attend(x_clean)
        x_noisy_attn = _chunk_and_attend(x_noisy)

        # Reassemble the full sequence from cross-attended clean and noisy halves.
        x_out = torch.cat([x_clean_attn, x_noisy_attn], dim=1)  # [B, L1, C]
        return x_out


class CausalWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size if local_attn_size != -1 else 24
        self.sink_size = sink_size
        self.global_sink_size = 0
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 24 * 880 if local_attn_size == -1 else local_attn_size * 880

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(
        self,
        x,
        seq_lens,
        grid_sizes,
        freqs,
        block_mask,
        kv_cache=None,
        current_start=0,
        cache_start=None,
        t_scale=1.0,
        use_relative_rope=False,
        method="linear",
        original_seq_len=None,
        temporal_offset=0.0,
        layer_recall_config=None,
        layer_recall_bank=None,
        layer_recall_query=None,
        layer_recall_logger=None,
        layer_recall_layer_index=-1,
        layer_recall_current_info=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
            t_scale (float): Temporal RoPE interpolation scale. <1.0 compresses positions.
            use_relative_rope (bool): If True, store raw K in cache and apply RoPE
                with window-relative positions at attention time.
            method (str): RoPE method. This release supports "linear".
            original_seq_len (int): Unused by the release linear RoPE path.
            temporal_offset (float): Multi-shot RoPE offset (shot_index * phi).
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        layer_recall_context = get_layer_recall_context()
        YX_sp_size = int(_CURRENT_GRID_META.get("YX_sp_size", 1))
        YX_global_k_summary = None
        YX_transient_bank_records = None
        YX_summary_tokens_per_frame = int(_CURRENT_GRID_META.get("frame_seqlen", 0) or 0)
        if (
            kv_cache is not None
            and layer_recall_config is not None
            and bool(getattr(layer_recall_config, "layer_recall_enabled", False))
            and str(layer_recall_context.get("YX_call_type", "")) == "context_update"
            and str(layer_recall_context.get("YX_cfg_branch", "pos")) == "pos"
        ):
            YX_global_k_summary = k_summary(
                k,
                detach=True,
                tokens_per_frame=(
                    YX_summary_tokens_per_frame
                    if YX_summary_tokens_per_frame > 0
                    else None
                ),
            )
        if kv_cache is not None and YX_sp_size > 1:
            q, k, v = ulysses_packed_qkv_seq_to_head(q, k, v)

        if kv_cache is None:
            # Teacher-forcing training doubles sequence length with clean/noisy halves.
            is_tf = (s == seq_lens[0].item() * 2)
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs, t_scale=t_scale,
                                    method=method, original_seq_len=original_seq_len,
                                    temporal_offset=temporal_offset).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs, t_scale=t_scale,
                                    method=method, original_seq_len=original_seq_len,
                                    temporal_offset=temporal_offset).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )
                x = x[:, :, :(-padded_length)] if padded_length > 0 else x
                x = x.transpose(2, 1)

            else:
                roped_query = rope_apply(q, grid_sizes, freqs, t_scale=t_scale,
                                         method=method, original_seq_len=original_seq_len,
                                         temporal_offset=temporal_offset).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs, t_scale=t_scale,
                                       method=method, original_seq_len=original_seq_len,
                                       temporal_offset=temporal_offset).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [roped_query,
                     torch.zeros([q.shape[0], padded_length, q.shape[2], q.shape[3]],
                                 device=q.device, dtype=v.dtype)],
                    dim=1
                )

                padded_roped_key = torch.cat(
                    [roped_key, torch.zeros([k.shape[0], padded_length, k.shape[2], k.shape[3]],
                                            device=k.device, dtype=v.dtype)],
                    dim=1
                )

                padded_v = torch.cat(
                    [v, torch.zeros([v.shape[0], padded_length, v.shape[2], v.shape[3]],
                                    device=v.device, dtype=v.dtype)],
                    dim=1
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask
                )
                x = x[:, :, :(-padded_length)] if padded_length > 0 else x
                x = x.transpose(2, 1)
        else:
            # iter-31: read Python ints from module-level dict (set by
            # CausalWanModel.forward) instead of `.item()` calls on
            # grid_sizes, removing 4 graph breaks per attention forward.
            if _CURRENT_GRID_META:
                frame_seqlen = _CURRENT_GRID_META["frame_seqlen"]
                num_new_frames = _CURRENT_GRID_META.get(
                    "global_num_new_frames",
                    _CURRENT_GRID_META["num_new_frames"],
                )
                h = _CURRENT_GRID_META["h"]
                w = _CURRENT_GRID_META["w"]
            else:
                frame_seqlen = math.prod(grid_sizes[0][1:]).item()
                num_new_frames = grid_sizes[0][0].item()
                h, w = grid_sizes[0][1].item(), grid_sizes[0][2].item()
            num_new_tokens = q.shape[1]
            current_end = current_start + num_new_tokens
            # iter-30: build Python-int grid once; pass to all rope_apply calls
            # below so they skip the .tolist() graph break.
            b = q.shape[0]
            grid_py = [(num_new_frames, h, w)] * b

            if not use_relative_rope:
                current_start_frame = current_start // frame_seqlen
                roped_query = causal_rope_apply(
                    q, grid_py, freqs, start_frame=current_start_frame, t_scale=t_scale,
                    method=method, original_seq_len=original_seq_len,
                    temporal_offset=temporal_offset).type_as(v)
                roped_key = causal_rope_apply(
                    k, grid_py, freqs, start_frame=current_start_frame, t_scale=t_scale,
                    method=method, original_seq_len=original_seq_len,
                    temporal_offset=temporal_offset).type_as(v)
                key_to_cache = roped_key
            else:
                key_to_cache = k

            sink_tokens = self.sink_size * frame_seqlen
            global_sink_tokens = getattr(self, "global_sink_size", 0) * frame_seqlen
            kv_cache_size = kv_cache["k"].shape[1]

            # ----- global + multi-shot pinned-sink support -----
            # Two protection mechanisms (independent, both optional):
            #   * global_sink_tokens: first N frames are permanently anchored
            #     (set via global_sink_size; never moves, always attended).
            #   * pinned region (pinned_start/pinned_len): multi-shot sink put
            #     on a scene cut. The pinned chunk lives at its original buffer
            #     position; rolling shifts non-pinned data around it.
            # effective_sink = leading buffer prefix that rolling MUST keep:
            #   pinned right after global (pinned_start == global_sink_tokens)
            #     -> global_sink_tokens + pinned_len
            #   pinned elsewhere (floating)
            #     -> global_sink_tokens
            #   no pinned
            #     -> max(global_sink_tokens, sink_tokens)  # legacy compat
            # iter-39: read pinned state from _CURRENT_GRID_META (published
            # once per chunk in CausalWanModel._forward_inference). Falls
            # back to `.item()` if the dict was not initialized (e.g. unit
            # test exercising the attention block directly).
            if _CURRENT_GRID_META and "pinned_start" in _CURRENT_GRID_META:
                pinned_start_val = _CURRENT_GRID_META["pinned_start"]
                pinned_len_val = _CURRENT_GRID_META["pinned_len"]
                pinned_chunk_index_val = _CURRENT_GRID_META.get("pinned_chunk_index", -1)
            else:
                pinned_start_t = kv_cache.get("pinned_start", None)
                pinned_len_val = 0
                pinned_chunk_index_val = -1
                if pinned_start_t is not None and hasattr(pinned_start_t, 'item'):
                    pinned_start_val = pinned_start_t.item()
                    pinned_len_val = kv_cache["pinned_len"].item()
                    pinned_chunk_t = kv_cache.get("pinned_chunk_index", None)
                    if pinned_chunk_t is not None and hasattr(pinned_chunk_t, "item"):
                        pinned_chunk_index_val = pinned_chunk_t.item()
                else:
                    pinned_start_val = -1
            has_pinned = pinned_start_val >= 0 and pinned_len_val > 0
            if has_pinned and pinned_start_val == global_sink_tokens:
                effective_sink = global_sink_tokens + pinned_len_val
            elif has_pinned:
                effective_sink = global_sink_tokens
            else:
                effective_sink = max(global_sink_tokens, sink_tokens)

            # iter-39: read cache indices from _CURRENT_GRID_META (published
            # by CausalWanModel._forward_inference) to avoid 6+ `.item()`
            # syncs per block forward. Falls back to .item() when the dict
            # is not initialized (direct attention-block unit tests).
            if _CURRENT_GRID_META and "global_end_index" in _CURRENT_GRID_META:
                _cache_global_end = _CURRENT_GRID_META["global_end_index"]
                _cache_local_end = _CURRENT_GRID_META["local_end_index"]
            else:
                _cache_global_end = kv_cache["global_end_index"].item()
                _cache_local_end = kv_cache["local_end_index"].item()

            layer_recall_enabled = (
                layer_recall_config is not None
                and bool(getattr(layer_recall_config, "layer_recall_enabled", False))
                and layer_recall_bank is not None
                and layer_recall_query is not None
            )
            YX_sp_debug_enabled = bool(
                YX_sp_size > 1
                and layer_recall_config is not None
                and getattr(layer_recall_config, "layer_recall_sp_debug_consistency", False)
            )
            YX_sp_debug_checked = False
            YX_visible_layout_pre = "sink_selected_current" if layer_recall_config is not None else ""
            YX_memory_sensitive_layer = True
            YX_layer_role = "layer_recall_not_evaluated"
            if layer_recall_enabled:
                YX_memory_sensitive_layer, YX_layer_role = is_layer_recall_enabled_for_layer(
                    layer_recall_config,
                    int(layer_recall_layer_index),
                )
            YX_pre_gate_active = False
            if (
                layer_recall_enabled
                and str(layer_recall_context.get("YX_call_type", "")) == "denoise"
                and str(layer_recall_context.get("YX_cfg_branch", "pos")) == "pos"
            ):
                YX_pre_gate_active, _ = should_use_layer_recall_selection(
                    YX_local_attn_size=self.local_attn_size,
                    YX_current_start_token=int(current_start),
                    YX_max_attention_size=int(self.max_attention_size),
                )
                YX_pre_gate_active = bool(
                    YX_pre_gate_active and YX_memory_sensitive_layer
                )
            YX_use_logical_temp_kv = (
                YX_pre_gate_active
                and YX_visible_layout_pre == "sink_selected_current"
                and not bool(use_relative_rope)
                and not bool(has_pinned)
            )

            cache_update_info = None
            if self.local_attn_size != -1 and (current_end > _cache_global_end) and (
                    num_new_tokens + _cache_local_end > kv_cache_size):
                num_evicted_tokens = num_new_tokens + _cache_local_end - kv_cache_size
                num_rolled_tokens = _cache_local_end - num_evicted_tokens - effective_sink

                local_end_index = _cache_local_end + current_end - \
                    _cache_global_end - num_evicted_tokens
                local_start_index = local_end_index - num_new_tokens

                cache_k = kv_cache["k"]
                cache_v = kv_cache["v"]
                new_k_for_cache = key_to_cache

                if YX_use_logical_temp_kv:
                    temp_k = None
                    temp_v = None
                elif _CGRAPH_OUTPLACE_KV_ENABLED:
                    # Cudagraph experiment: build the post-roll cache view
                    # out-of-place. Slice assignment here forces Inductor
                    # cudagraph partitions to mutate inputs.
                    temp_k = torch.cat([
                        cache_k[:, :effective_sink],
                        cache_k[:, effective_sink + num_evicted_tokens:
                                effective_sink + num_evicted_tokens + num_rolled_tokens],
                        new_k_for_cache,
                    ], dim=1)
                    temp_v = torch.cat([
                        cache_v[:, :effective_sink],
                        cache_v[:, effective_sink + num_evicted_tokens:
                                effective_sink + num_evicted_tokens + num_rolled_tokens],
                        v,
                    ], dim=1)
                else:
                    temp_k = cache_k.clone()
                    temp_v = cache_v.clone()

                    temp_k[:, effective_sink:effective_sink + num_rolled_tokens] = \
                        temp_k[:, effective_sink + num_evicted_tokens:effective_sink + num_evicted_tokens + num_rolled_tokens].clone()
                    temp_v[:, effective_sink:effective_sink + num_rolled_tokens] = \
                        temp_v[:, effective_sink + num_evicted_tokens:effective_sink + num_evicted_tokens + num_rolled_tokens].clone()

                    temp_k[:, local_start_index:local_end_index] = new_k_for_cache
                    temp_v[:, local_start_index:local_end_index] = v

                # When pinned is "floating" (lives outside effective_sink), the
                # rolling shifted non-pinned data left by num_evicted_tokens;
                # the pinned anchor must follow that shift to keep tracking the
                # same data. When pinned sits inside effective_sink (i.e. right
                # after the global region), it is part of the protected prefix
                # and rolling does not move it.
                pinned_shift = num_evicted_tokens if (has_pinned and pinned_start_val >= effective_sink) else 0

                cache_update_info = {
                    "action": "roll_and_insert",
                    "sink_tokens": effective_sink,
                    "num_rolled_tokens": num_rolled_tokens,
                    "num_evicted_tokens": num_evicted_tokens,
                    "local_start_index": local_start_index,
                    "local_end_index": local_end_index,
                    "new_k": key_to_cache,
                    "new_v": v,
                    "current_end": current_end,
                    "pinned_shift": pinned_shift,
                }

            else:
                # iter-39: reuse the dict-cached scalars from above.
                local_end_index = _cache_local_end + current_end - _cache_global_end
                local_start_index = local_end_index - num_new_tokens

                if YX_use_logical_temp_kv:
                    temp_k = None
                    temp_v = None
                else:
                    temp_k = torch.cat([kv_cache["k"][:, :local_start_index], key_to_cache], dim=1)
                    temp_v = torch.cat([kv_cache["v"][:, :local_start_index], v], dim=1)

                cache_update_info = {
                    "action": "direct_insert",
                    "local_start_index": local_start_index,
                    "local_end_index": local_end_index,
                    "new_k": key_to_cache,
                    "new_v": v,
                    "current_end": current_end,
                    "pinned_shift": 0,
                }

            YX_bank_roll_applied = False
            YX_bank_pruned_count = 0
            if (
                layer_recall_enabled
                and cache_update_info is not None
                and cache_update_info.get("action") == "roll_and_insert"
            ):
                if bool(layer_recall_context.get("YX_skip_cache_update", False)):
                    YX_transient_bank_records = layer_recall_bank.records_for_layer(
                        int(layer_recall_layer_index)
                    )
                before_count = len(layer_recall_bank.records_for_layer(int(layer_recall_layer_index)))
                layer_recall_bank.apply_cache_roll(
                    YX_layer_index=int(layer_recall_layer_index),
                    YX_sink_tokens=int(cache_update_info.get("sink_tokens", effective_sink)),
                    YX_num_evicted_tokens=int(cache_update_info.get("num_evicted_tokens", 0)),
                )
                layer_recall_bank.prune_by_cache_capacity(
                    YX_layer_index=int(layer_recall_layer_index),
                    YX_cache_tokens=int(kv_cache_size),
                )
                after_count = len(layer_recall_bank.records_for_layer(int(layer_recall_layer_index)))
                YX_bank_roll_applied = True
                YX_bank_pruned_count = max(0, int(before_count) - int(after_count))
            if (
                layer_recall_enabled
                and str(layer_recall_context.get("YX_call_type", "")) == "context_update"
                and str(layer_recall_context.get("YX_cfg_branch", "pos")) == "pos"
            ):
                layer_recall_bank.add_or_replace(
                    int(layer_recall_layer_index),
                    _YX_make_chunk_memory_record(
                        YX_chunk_index=int(
                            layer_recall_context.get(
                                "YX_chunk_index",
                                current_start // max(1, num_new_tokens),
                            )
                        ),
                        YX_start_frame=int(
                            layer_recall_context.get(
                                "YX_chunk_start_frame",
                                current_start // max(1, frame_seqlen),
                            )
                        ),
                        YX_num_frames=int(
                            layer_recall_context.get("YX_num_frames", num_new_frames)
                        ),
                        YX_cache_start_token=int(local_start_index),
                        YX_cache_end_token=int(local_end_index),
                        YX_global_start_token=int(current_start),
                        YX_global_end_token=int(current_end),
                        YX_summary=(
                            YX_global_k_summary.detach().float()
                            if YX_global_k_summary is not None
                            else pool_pre_rope_k(k)
                        ),
                    )
                )
                if YX_sp_debug_enabled:
                    _YX_assert_layer_recall_sp_debug_consistent(
                        YX_layer_index=layer_recall_layer_index,
                        YX_candidate_records=[],
                        YX_selected_chunk_ids=[],
                        YX_current_start=current_start,
                        YX_current_end=current_end,
                        YX_local_start=local_start_index,
                        YX_local_end=local_end_index,
                    )
                    YX_sp_debug_checked = True
                if layer_recall_logger is not None:
                    layer_recall_logger.log({
                        **layer_recall_context,
                        "YX_layer_index": int(layer_recall_layer_index),
                        "layer_recall_gate_active": False,
                        "layer_recall_gate_reason": "context_update_summary_write",
                        "layer_recall_selection_mode": str(getattr(layer_recall_config, "layer_recall_selection_mode", "hard")).lower(),
                        "YX_current_start_token": int(current_start),
                        "YX_local_start_index": int(local_start_index),
                        "YX_local_end_index": int(local_end_index),
                        "YX_frame_seq_length": int(frame_seqlen),
                        "YX_chunk_token_size": int(num_new_tokens),
                        "YX_logical_temp_kv": bool(YX_use_logical_temp_kv),
                        "YX_skip_cache_update": bool(layer_recall_context.get("YX_skip_cache_update", False)),
                    })

            window_start = max(0, local_end_index - self.max_attention_size)

            def YX_get_logical_kv_range(YX_start, YX_end):
                YX_start = int(YX_start)
                YX_end = int(YX_end)
                if YX_end <= YX_start:
                    empty_k = key_to_cache[:, :0]
                    empty_v = v[:, :0]
                    return empty_k, empty_v
                if not YX_use_logical_temp_kv:
                    return temp_k[:, YX_start:YX_end], temp_v[:, YX_start:YX_end]

                action = str(cache_update_info.get("action", "")) if cache_update_info is not None else ""
                base_k = kv_cache["k"]
                base_v = kv_cache["v"]
                k_segments = []
                v_segments = []
                cursor = YX_start

                if action == "direct_insert":
                    if cursor < local_start_index:
                        seg_end = min(YX_end, int(local_start_index))
                        k_segments.append(base_k[:, cursor:seg_end])
                        v_segments.append(base_v[:, cursor:seg_end])
                        cursor = seg_end
                    if cursor < YX_end:
                        seg_end = min(YX_end, int(local_end_index))
                        rel_start = max(0, cursor - int(local_start_index))
                        rel_end = max(0, seg_end - int(local_start_index))
                        k_segments.append(key_to_cache[:, rel_start:rel_end])
                        v_segments.append(v[:, rel_start:rel_end])
                        cursor = seg_end
                elif action == "roll_and_insert":
                    sink_tokens = int(cache_update_info.get("sink_tokens", effective_sink))
                    evicted_tokens = int(cache_update_info.get("num_evicted_tokens", 0))
                    while cursor < YX_end:
                        if cursor < sink_tokens:
                            seg_end = min(YX_end, sink_tokens)
                            k_segments.append(base_k[:, cursor:seg_end])
                            v_segments.append(base_v[:, cursor:seg_end])
                            cursor = seg_end
                        elif cursor < local_start_index:
                            seg_end = min(YX_end, int(local_start_index))
                            pre_start = cursor + evicted_tokens
                            pre_end = seg_end + evicted_tokens
                            k_segments.append(base_k[:, pre_start:pre_end])
                            v_segments.append(base_v[:, pre_start:pre_end])
                            cursor = seg_end
                        else:
                            seg_end = min(YX_end, int(local_end_index))
                            rel_start = max(0, cursor - int(local_start_index))
                            rel_end = max(0, seg_end - int(local_start_index))
                            k_segments.append(key_to_cache[:, rel_start:rel_end])
                            v_segments.append(v[:, rel_start:rel_end])
                            cursor = seg_end
                else:
                    k_segments.append(base_k[:, YX_start:YX_end])
                    v_segments.append(base_v[:, YX_start:YX_end])
                    cursor = YX_end

                if cursor < YX_end:
                    raise RuntimeError(
                        f"YX logical KV range mapping failed for range=({YX_start}, {YX_end}), "
                        f"cursor={cursor}, action={action}"
                    )
                if len(k_segments) == 1:
                    return k_segments[0], v_segments[0]
                return torch.cat(k_segments, dim=1), torch.cat(v_segments, dim=1)

            def YX_concat_logical_kv_ranges(YX_ranges):
                parts_k = []
                parts_v = []
                for YX_start, YX_end in YX_ranges:
                    part_k, part_v = YX_get_logical_kv_range(YX_start, YX_end)
                    parts_k.append(part_k)
                    parts_v.append(part_v)
                if not parts_k:
                    return key_to_cache[:, :0], v[:, :0]
                if len(parts_k) == 1:
                    return parts_k[0], parts_v[0]
                return torch.cat(parts_k, dim=1), torch.cat(parts_v, dim=1)

            # Build the K/V actually attended over.
            # Cases:
            #   (a) prepend_sink  : effective_sink > 0 and out of window
            #                       -> prepend [:effective_sink] (covers global
            #                          and any pinned-merged-to-front)
            #   (b) prepend_pinned: a floating pinned region (pinned_start
            #                       >= effective_sink) lives outside the window
            #                       -> additionally prepend that pinned slice
            #   (c) otherwise     : plain sliding window
            # Note (a) and (b) are not mutually exclusive: when global is
            # enabled AND there is a separate floating pinned region outside
            # the window, both prefixes must be prepended.
            prepend_sink = effective_sink > 0 and window_start > 0
            prepend_pinned = (
                has_pinned and pinned_start_val >= effective_sink
                and pinned_start_val < window_start
            )

            if YX_use_logical_temp_kv:
                if prepend_sink and prepend_pinned:
                    extra = effective_sink + pinned_len_val
                    effective_local_size = self.max_attention_size - extra
                    local_window_start = max(effective_sink, local_end_index - effective_local_size)
                    window_k, window_v = YX_concat_logical_kv_ranges([
                        (0, effective_sink),
                        (pinned_start_val, pinned_start_val + pinned_len_val),
                        (local_window_start, local_end_index),
                    ])
                elif prepend_sink:
                    effective_local_size = self.max_attention_size - effective_sink
                    local_window_start = max(effective_sink, local_end_index - effective_local_size)
                    window_k, window_v = YX_concat_logical_kv_ranges([
                        (0, effective_sink),
                        (local_window_start, local_end_index),
                    ])
                elif prepend_pinned:
                    effective_local_size = self.max_attention_size - pinned_len_val
                    local_window_start = max(0, local_end_index - effective_local_size)
                    window_k, window_v = YX_concat_logical_kv_ranges([
                        (pinned_start_val, pinned_start_val + pinned_len_val),
                        (local_window_start, local_end_index),
                    ])
                else:
                    window_k, window_v = YX_get_logical_kv_range(window_start, local_end_index)
            elif prepend_sink and prepend_pinned:
                # [global+sink] + [pinned] + [local window]
                extra = effective_sink + pinned_len_val
                effective_local_size = self.max_attention_size - extra
                local_window_start = max(effective_sink, local_end_index - effective_local_size)
                window_k = torch.cat([
                    temp_k[:, :effective_sink],
                    temp_k[:, pinned_start_val:pinned_start_val + pinned_len_val],
                    temp_k[:, local_window_start:local_end_index],
                ], dim=1)
                window_v = torch.cat([
                    temp_v[:, :effective_sink],
                    temp_v[:, pinned_start_val:pinned_start_val + pinned_len_val],
                    temp_v[:, local_window_start:local_end_index],
                ], dim=1)
            elif prepend_sink:
                effective_local_size = self.max_attention_size - effective_sink
                local_window_start = max(effective_sink, local_end_index - effective_local_size)
                window_k = torch.cat([temp_k[:, :effective_sink], temp_k[:, local_window_start:local_end_index]], dim=1)
                window_v = torch.cat([temp_v[:, :effective_sink], temp_v[:, local_window_start:local_end_index]], dim=1)
            elif prepend_pinned:
                effective_local_size = self.max_attention_size - pinned_len_val
                local_window_start = max(0, local_end_index - effective_local_size)
                window_k = torch.cat(
                    [temp_k[:, pinned_start_val:pinned_start_val + pinned_len_val],
                     temp_k[:, local_window_start:local_end_index]], dim=1)
                window_v = torch.cat(
                    [temp_v[:, pinned_start_val:pinned_start_val + pinned_len_val],
                     temp_v[:, local_window_start:local_end_index]], dim=1)
            else:
                window_k = temp_k[:, window_start:local_end_index]
                window_v = temp_v[:, window_start:local_end_index]

            if (
                layer_recall_enabled
                and str(layer_recall_context.get("YX_call_type", "")) == "denoise"
                and str(layer_recall_context.get("YX_cfg_branch", "pos")) == "pos"
            ):
                YX_selection_mode = str(getattr(layer_recall_config, "layer_recall_selection_mode", "hard")).lower()
                YX_visible_layout = "sink_selected_current"
                YX_gate_active, YX_gate_reason = should_use_layer_recall_selection(
                    YX_local_attn_size=self.local_attn_size,
                    YX_current_start_token=int(current_start),
                    YX_max_attention_size=int(self.max_attention_size),
                )
                YX_original_window_layer_active = bool(
                    YX_gate_active and not YX_memory_sensitive_layer
                )
                if YX_original_window_layer_active:
                    YX_gate_active = False
                    YX_gate_reason = YX_layer_role
                if use_relative_rope:
                    YX_gate_active = False
                    YX_gate_reason = "relative_rope_not_supported_for_layer_recall_v1"
                if prepend_pinned:
                    YX_gate_active = False
                    YX_gate_reason = "multi_shot_sink_not_supported_with_layer_recall"

                YX_candidate_records = []
                YX_candidate_scores = layer_recall_query.new_empty((0,), dtype=torch.float32)
                YX_candidate_weights = layer_recall_query.new_empty((0,), dtype=torch.float32)
                layer_recall_layout_applied = False
                YX_soft_memory_tokens = 0
                YX_recent_ranges = []
                YX_current_range = (int(local_start_index), int(local_end_index))
                YX_base_sink_tokens = int(effective_sink)
                YX_sink_ranges = [(0, int(YX_base_sink_tokens))] if int(YX_base_sink_tokens) > 0 else []
                YX_pinned_is_floating = bool(has_pinned and int(pinned_start_val) >= int(effective_sink))
                YX_candidate_count_before_filter = 0
                YX_candidate_count_after_filter = 0
                YX_filter_unequal_length_count = 0
                YX_filter_nonresident_count = 0
                YX_filter_pinned_count = 0
                YX_memory_slots_requested = 0
                YX_memory_slots_filled = 0
                YX_memory_slots_unfilled = 0
                YX_visible_underfill_tokens = 0
                YX_selected_ranges = []
                YX_selected_chunk_ids = []
                if YX_gate_active:
                    YX_candidate_records, YX_candidate_scores = layer_recall_bank.score_all(
                        YX_layer_index=int(layer_recall_layer_index),
                        YX_query=layer_recall_query,
                        YX_current_start_token=int(current_start),
                        YX_normalize=bool(getattr(layer_recall_config, "layer_recall_normalize_scores", False))
                        or str(getattr(layer_recall_config, "layer_recall_score_mode", "dot")).lower() == "cosine",
                    )
                    YX_candidate_count_before_filter = len(YX_candidate_records)
                    filtered_records = []
                    filtered_scores = []
                    for record, score in zip(YX_candidate_records, YX_candidate_scores):
                        start, end = record.token_range
                        if int(end) - int(start) != int(num_new_tokens):
                            YX_filter_unequal_length_count += 1
                            continue
                        if int(start) < int(YX_base_sink_tokens) or int(end) > int(local_start_index):
                            YX_filter_nonresident_count += 1
                            continue
                        filtered_records.append(record)
                        filtered_scores.append(score)
                    YX_candidate_records = filtered_records
                    YX_candidate_scores = (
                        torch.stack(filtered_scores, dim=0).float()
                        if filtered_scores
                        else layer_recall_query.new_empty((0,), dtype=torch.float32)
                    )
                    YX_candidate_count_after_filter = len(YX_candidate_records)

                    if YX_candidate_scores.numel() > 0:
                        max_candidates = int(
                            getattr(layer_recall_config, "layer_recall_candidate_pool_size", 0) or 0
                        )
                        if max_candidates > 0 and YX_candidate_scores.numel() > max_candidates:
                            top_result = stable_topk(
                                YX_candidate_scores,
                                YX_candidate_records,
                                max_candidates,
                            )
                            YX_candidate_records = [
                                YX_candidate_records[int(idx)]
                                for idx in top_result.indices.detach().cpu().tolist()
                            ]
                            YX_candidate_scores = top_result.values
                            YX_candidate_count_after_filter = len(YX_candidate_records)

                    if YX_candidate_scores.numel() > 0:
                        sort_result = stable_topk(
                            YX_candidate_scores,
                            YX_candidate_records,
                            int(YX_candidate_scores.numel()),
                        )
                        sorted_records = [
                            YX_candidate_records[int(idx)]
                            for idx in sort_result.indices.detach().cpu().tolist()
                        ]
                    else:
                        sorted_records = []

                    YX_plan = assemble_slot_visible_plan(
                            YX_selected_records=sorted_records,
                            YX_sink_tokens=int(YX_base_sink_tokens),
                            YX_current_start_token=int(local_start_index),
                            YX_current_end_token=int(local_end_index),
                            YX_max_attention_size=int(self.max_attention_size),
                            YX_chunk_token_size=int(num_new_tokens),
                        )
                    YX_memory_slots_requested = int(YX_plan.get("YX_memory_slots_requested", 0) or 0)
                    YX_memory_slots_filled = int(YX_plan.get("YX_memory_slots_filled", 0) or 0)
                    YX_memory_slots_unfilled = int(YX_plan.get("YX_memory_slots_unfilled", 0) or 0)
                    YX_visible_underfill_tokens = int(YX_plan.get("YX_visible_underfill_tokens", 0) or 0)
                    YX_selected_ranges = list(YX_plan.get("YX_selected_ranges", []))
                    YX_selected_chunk_ids = list(YX_plan.get("YX_selected_chunk_ids", []))
                    YX_soft_memory_tokens = int(YX_plan.get("YX_selected_tokens", 0) or 0)

                    if YX_plan:
                        k_parts = []
                        v_parts = []
                        if YX_plan["YX_sink_ranges"]:
                            sink_k, sink_v = YX_concat_logical_kv_ranges(YX_plan["YX_sink_ranges"])
                            k_parts.append(sink_k)
                            v_parts.append(sink_v)
                        if YX_memory_slots_filled > 0:
                            temperature = max(float(getattr(layer_recall_config, "layer_recall_temperature", 1.0)), 1e-6)
                            used_score_positions = []
                            hard_position_by_chunk = {
                                int(record.chunk_index): pos for pos, record in enumerate(YX_candidate_records)
                            }
                            slot_k_parts = []
                            slot_v_parts = []
                            first_slot_weights = None
                            selected_records_for_plan = list(YX_plan.get("YX_selected_records", []))
                            for YX_slot_index, selected_record in enumerate(selected_records_for_plan):
                                hard_pos = hard_position_by_chunk.get(int(selected_record.chunk_index), None)
                                if hard_pos is None:
                                    continue
                                masked_scores = YX_candidate_scores.float().clone()
                                for prev_pos in used_score_positions:
                                    masked_scores[int(prev_pos)] = -torch.inf
                                slot_weights = torch.softmax(masked_scores / temperature, dim=0).to(dtype=key_to_cache.dtype)
                                if first_slot_weights is None:
                                    first_slot_weights = slot_weights
                                soft_k = None
                                soft_v = None
                                for cand_pos, candidate_record in enumerate(YX_candidate_records):
                                    cand_start = int(candidate_record.cache_start_token)
                                    cand_end = int(candidate_record.cache_end_token)
                                    weight = slot_weights[int(cand_pos)].to(dtype=key_to_cache.dtype).view(1, 1, 1, 1)
                                    k_slice, v_slice = YX_get_logical_kv_range(cand_start, cand_end)
                                    soft_k = k_slice * weight if soft_k is None else soft_k + k_slice * weight
                                    soft_v = v_slice * weight if soft_v is None else soft_v + v_slice * weight
                                hard_record = YX_candidate_records[int(hard_pos)]
                                hard_k, hard_v = YX_get_logical_kv_range(
                                    int(hard_record.cache_start_token),
                                    int(hard_record.cache_end_token),
                                )
                                slot_k_parts.append(
                                    materialize_layer_recall_slot(
                                        hard_k, soft_k, YX_selection_mode
                                    )
                                )
                                slot_v_parts.append(
                                    materialize_layer_recall_slot(
                                        hard_v, soft_v, YX_selection_mode
                                    )
                                )
                                used_score_positions.append(int(hard_pos))
                            if first_slot_weights is not None:
                                YX_candidate_weights = first_slot_weights
                            if slot_k_parts:
                                layer_recall_k = torch.cat(slot_k_parts, dim=1)
                                layer_recall_v = torch.cat(slot_v_parts, dim=1)
                                k_parts.append(layer_recall_k)
                                v_parts.append(layer_recall_v)

                        current_k, current_v = YX_get_logical_kv_range(local_start_index, local_end_index)
                        k_parts.append(current_k)
                        v_parts.append(current_v)
                        window_k = torch.cat(k_parts, dim=1)
                        window_v = torch.cat(v_parts, dim=1)
                        layer_recall_layout_applied = True
                        YX_sink_ranges = list(YX_plan["YX_sink_ranges"])
                        if YX_memory_slots_unfilled > 0:
                            YX_gate_reason = "sink_selected_current_underfilled"


                if YX_sp_debug_enabled:
                    _YX_assert_layer_recall_sp_debug_consistent(
                        YX_layer_index=layer_recall_layer_index,
                        YX_candidate_records=YX_candidate_records,
                        YX_selected_chunk_ids=YX_selected_chunk_ids,
                        YX_current_start=current_start,
                        YX_current_end=current_end,
                        YX_local_start=local_start_index,
                        YX_local_end=local_end_index,
                    )
                    YX_sp_debug_checked = True

                if layer_recall_logger is not None:
                    YX_current_info = dict(layer_recall_current_info or {})
                    YX_candidate_count = len(YX_candidate_records)
                    YX_selection_entropy = None
                    YX_top1_weight = None
                    YX_top1_margin = None
                    YX_top1_chunk_id = None
                    YX_top1_temporal_distance = None
                    YX_score_margin = None
                    YX_score_top_ids = []
                    YX_hard_top_ids = [int(item) for item in YX_selected_chunk_ids]
                    YX_temporal_distances = [
                        int(layer_recall_context.get("YX_chunk_index", 0)) - int(record.chunk_index)
                        for record in YX_candidate_records
                    ]
                    if YX_candidate_weights.numel() > 0:
                        weights_float = YX_candidate_weights.detach().float()
                        YX_selection_entropy = float(
                            (-(weights_float * torch.log(weights_float.clamp_min(1e-12))).sum()).item()
                        )
                        sorted_weight_result = stable_topk(
                            weights_float,
                            YX_candidate_records,
                            int(weights_float.numel()),
                        )
                        sorted_weights = sorted_weight_result.values
                        top1_index = int(
                            sorted_weight_result.indices[0].detach().cpu().item()
                        )
                        YX_top1_weight = float(weights_float[top1_index].item())
                        top2_weight = float(sorted_weights[1].item()) if int(sorted_weights.numel()) > 1 else 0.0
                        YX_top1_margin = float(YX_top1_weight - top2_weight)
                        if 0 <= top1_index < len(YX_candidate_records):
                            top1_record = YX_candidate_records[top1_index]
                            YX_top1_chunk_id = int(top1_record.chunk_index)
                            YX_top1_temporal_distance = (
                                int(layer_recall_context.get("YX_chunk_index", 0)) - int(top1_record.chunk_index)
                            )
                    if YX_candidate_scores.numel() > 0:
                        score_topk = min(
                            int(YX_candidate_scores.numel()),
                            max(1, int(YX_memory_slots_requested)),
                        )
                        score_top = stable_topk(
                            YX_candidate_scores.detach().float(),
                            YX_candidate_records,
                            score_topk,
                        )
                        YX_score_top_ids = [
                            int(YX_candidate_records[int(idx)].chunk_index)
                            for idx in score_top.indices.detach().cpu().tolist()
                            if int(idx) < len(YX_candidate_records)
                        ]
                    if YX_candidate_scores.numel() >= 2:
                        score_top2 = stable_topk(
                            YX_candidate_scores.detach().float(),
                            YX_candidate_records,
                            2,
                        ).values
                        YX_score_margin = float((score_top2[0] - score_top2[1]).item())
                    YX_sink_token_count = int(sum(end - start for start, end in YX_sink_ranges))
                    YX_effective_visible_layout = (
                        str(YX_visible_layout)
                        if layer_recall_layout_applied
                        else "original_window"
                    )
                    YX_selection_payload = {
                        **layer_recall_context,
                        **YX_current_info,
                        "YX_event_schema_version": 1,
                        "YX_rank": int(dist.get_rank()) if dist.is_initialized() else 0,
                        "YX_layer_index": int(layer_recall_layer_index),
                        "layer_recall_gate_active": bool(YX_gate_active),
                        "layer_recall_gate_reason": YX_gate_reason,
                        "memory_sensitive_layer": bool(YX_memory_sensitive_layer),
                        "layer_role": str(YX_layer_role),
                        "memory_sensitive_layers": [
                            int(item)
                            for item in layer_recall_config.memory_sensitive_layers
                        ],
                        "layer_recall_selection_mode": YX_selection_mode,
                        "layer_recall_attention_kv_mode": (
                            "soft_slots"
                            if layer_recall_layout_applied and YX_selection_mode == "soft"
                            else "hard_slots"
                            if layer_recall_layout_applied
                            else "original_window"
                        ),
                        "layer_recall_cache_write_rope_mode": "absolute",
                        "layer_recall_temperature": float(getattr(layer_recall_config, "layer_recall_temperature", 1.0)),
                        "YX_score_query_type": str(YX_current_info.get("YX_score_query_type", "global")),
                        "YX_current_start_token": int(current_start),
                        "YX_local_start_index": int(local_start_index),
                        "YX_local_end_index": int(local_end_index),
                        "YX_frame_seq_length": int(frame_seqlen),
                        "YX_chunk_token_size": int(num_new_tokens),
                        "YX_candidate_count": int(YX_candidate_count),
                        "YX_candidate_count_before_filter": int(YX_candidate_count_before_filter),
                        "YX_candidate_count_after_filter": int(YX_candidate_count_after_filter),
                        "YX_filter_unequal_length_count": int(YX_filter_unequal_length_count),
                        "YX_filter_nonresident_count": int(YX_filter_nonresident_count),
                        "YX_filter_pinned_count": int(YX_filter_pinned_count),
                        "YX_candidate_chunk_ids": [int(record.chunk_index) for record in YX_candidate_records],
                        "YX_candidate_ranges": [record.token_range for record in YX_candidate_records],
                        "YX_candidate_token_ranges": [record.token_range for record in YX_candidate_records],
                        "YX_candidate_global_token_ranges": [record.global_token_range for record in YX_candidate_records],
                        "YX_candidate_start_frames": [int(record.start_frame) for record in YX_candidate_records],
                        "YX_candidate_num_frames": [int(record.num_frames) for record in YX_candidate_records],
                        "YX_candidate_scores": YX_candidate_scores,
                        "YX_candidate_weights": YX_candidate_weights,
                        "YX_candidate_scores_requires_grad": bool(getattr(YX_candidate_scores, "requires_grad", False)),
                        "YX_candidate_weights_requires_grad": bool(getattr(YX_candidate_weights, "requires_grad", False)),
                        "YX_grad_enabled": bool(torch.is_grad_enabled()),
                        "YX_soft_distribution_grad_path": bool(
                            torch.is_grad_enabled() and getattr(YX_candidate_weights, "requires_grad", False)
                        ),
                        "YX_score_top_ids": YX_score_top_ids,
                        "YX_hard_top_ids": YX_hard_top_ids,
                        "YX_selection_entropy": YX_selection_entropy,
                        "YX_top1_weight": YX_top1_weight,
                        "YX_top1_margin": YX_top1_margin,
                        "YX_score_margin": YX_score_margin,
                        "YX_selected_temporal_distances": YX_temporal_distances,
                        "YX_top1_chunk_id": YX_top1_chunk_id,
                        "YX_top1_temporal_distance": YX_top1_temporal_distance,
                        "YX_soft_memory_tokens": int(YX_soft_memory_tokens),
                        "YX_visible_layout": str(YX_visible_layout),
                        "YX_effective_visible_layout": str(YX_effective_visible_layout),
                        "layer_recall_layout_applied": bool(layer_recall_layout_applied),
                        "original_window_layer_active": bool(YX_original_window_layer_active),
                        "YX_max_attention_size": int(self.max_attention_size),
                        "YX_effective_sink_tokens": int(effective_sink),
                        "YX_global_sink_tokens": int(global_sink_tokens),
                        "YX_global_sink_token_range": [0, int(global_sink_tokens)] if int(global_sink_tokens) > 0 else [],
                        "YX_pinned_start_token": int(pinned_start_val),
                        "YX_pinned_len_tokens": int(pinned_len_val),
                        "YX_pinned_chunk_index": int(pinned_chunk_index_val),
                        "YX_pinned_is_floating": bool(YX_pinned_is_floating),
                        "YX_forced_tokens": int(
                            YX_sink_token_count
                            + int(local_end_index - local_start_index)
                        ),
                        "YX_memory_slot_budget_tokens": int(
                            max(
                                0,
                                int(self.max_attention_size)
                                - YX_sink_token_count
                                - int(local_end_index - local_start_index),
                            )
                        ),
                        "YX_memory_slots_requested": int(YX_memory_slots_requested),
                        "YX_memory_slots_filled": int(YX_memory_slots_filled),
                        "YX_memory_slots_unfilled": int(YX_memory_slots_unfilled),
                        "YX_recent_window_disabled": bool(
                            layer_recall_layout_applied
                            and YX_visible_layout == "sink_selected_current"
                        ),
                        "YX_visible_underfill_tokens": int(YX_visible_underfill_tokens),
                        "YX_selected_chunk_ids": [int(item) for item in YX_selected_chunk_ids],
                        "YX_selected_ranges": YX_selected_ranges,
                        "YX_bank_roll_applied": bool(YX_bank_roll_applied),
                        "YX_bank_pruned_count": int(YX_bank_pruned_count),
                        "YX_logical_temp_kv": bool(YX_use_logical_temp_kv),
                        "YX_skip_cache_update": bool(layer_recall_context.get("YX_skip_cache_update", False)),
                        "YX_sink_tokens": YX_sink_token_count,
                        "YX_recent_tokens": sum(end - start for start, end in YX_recent_ranges),
                        "YX_current_tokens": int(local_end_index - local_start_index),
                        "YX_visible_tokens": int(window_k.shape[1]),
                        "YX_query_norm": float(layer_recall_query.detach().float().norm().item()),
                    }
                    layer_recall_logger.log(YX_selection_payload)

            if YX_sp_debug_enabled and not YX_sp_debug_checked:
                _YX_assert_layer_recall_sp_debug_consistent(
                    YX_layer_index=layer_recall_layer_index,
                    YX_candidate_records=[],
                    YX_selected_chunk_ids=[],
                    YX_current_start=current_start,
                    YX_current_end=current_end,
                    YX_local_start=local_start_index,
                    YX_local_end=local_end_index,
                )

            if use_relative_rope:
                if prepend_sink:
                    # Sink and local window tokens get separate RoPE in a
                    # virtual contiguous layout: [sink_frames | local_frames].
                    sink_frame_count = effective_sink // frame_seqlen
                    local_tokens = window_k.shape[1] - effective_sink
                    local_frame_count = local_tokens // frame_seqlen
                    combined_frames = sink_frame_count + local_frame_count

                    # iter-30: pass Python list instead of expanded tensor;
                    # causal_rope_apply skips .tolist() graph break this way.
                    sink_grid = [(sink_frame_count, h, w)] * b
                    roped_sink_k = causal_rope_apply(
                        window_k[:, :effective_sink], sink_grid, freqs,
                        start_frame=0, t_scale=t_scale,
                        method=method, original_seq_len=original_seq_len,
                    ).type_as(v)

                    local_grid = [(local_frame_count, h, w)] * b
                    roped_local_k = causal_rope_apply(
                        window_k[:, effective_sink:], local_grid, freqs,
                        start_frame=sink_frame_count, t_scale=t_scale,
                        method=method, original_seq_len=original_seq_len,
                    ).type_as(v)

                    roped_window_k = torch.cat([roped_sink_k, roped_local_k], dim=1)

                    q_start_frame = combined_frames - num_new_frames
                    roped_query = causal_rope_apply(
                        q, grid_py, freqs,
                        start_frame=q_start_frame, t_scale=t_scale,
                        method=method, original_seq_len=original_seq_len,
                    ).type_as(v)
                else:
                    window_tokens = window_k.shape[1]
                    window_frames = window_tokens // frame_seqlen

                    # iter-30: Python list to skip .tolist() break.
                    window_grid_sizes = [(window_frames, h, w)] * b

                    roped_window_k = causal_rope_apply(
                        window_k, window_grid_sizes, freqs,
                        start_frame=0, t_scale=t_scale,
                        method=method, original_seq_len=original_seq_len,
                    ).type_as(v)

                    q_start_frame = window_frames - num_new_frames
                    roped_query = causal_rope_apply(
                        q, grid_py, freqs,
                        start_frame=q_start_frame, t_scale=t_scale,
                        method=method, original_seq_len=original_seq_len,
                    ).type_as(v)
                x = attention(roped_query, roped_window_k, window_v)
            else:
                x = attention(roped_query, window_k, window_v)

        # Restore the local sequence before the full-head output projection.
        if kv_cache is not None and YX_sp_size > 1:
            x = ulysses_head_to_seq(x)

        # output
        x = x.flatten(2)
        x = self.o(x)

        if YX_transient_bank_records is not None:
            layer_recall_bank.YX_records_by_layer[int(layer_recall_layer_index)] = list(
                YX_transient_bank_records
            )
        
        # Return both output and cache update info
        if kv_cache is not None:
            if bool(layer_recall_context.get("YX_skip_cache_update", False)):
                return x, None
            return x, (current_end, local_end_index, cache_update_info)
        return x


class CausalWanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 local_attn_size=-1,
                 sink_size=0,
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = MultiShotT2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
        t_scale=1.0,
        use_relative_rope=False,
        method="linear",
        original_seq_len=None,
        temporal_offset=0.0,
        layer_recall_config=None,
        layer_recall_bank=None,
        layer_recall_query=None,
        layer_recall_logger=None,
        layer_recall_current_norm=None,
        layer_recall_current_mlp=None,
        layer_recall_current_gate=None,
        layer_recall_layer_gamma=None,
        layer_recall_current_alpha=None,
        layer_recall_layer_index=-1,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, F, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            t_scale (float): Temporal RoPE interpolation scale. <1.0 compresses positions.
            use_relative_rope (bool): If True, use window-relative RoPE positions in KV cache path.
            method (str): RoPE method. This release supports "linear".
            original_seq_len (int): Unused by the release linear RoPE path.
            temporal_offset (float): Multi-shot RoPE offset (shot_index * phi).
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        use_triton_adaln = _TRITON_ADALN_ENABLED and not torch.is_grad_enabled()

        # self-attention
        if use_triton_adaln:
            # iter-43: fused LayerNorm + (1+e[1])*x + e[0] in one Triton kernel.
            from utils.adaln_triton import adaln_modulate_triton
            modulated_x = adaln_modulate_triton(
                x, e[1], e[0], frame_seqlen,
                eps=self.norm1.eps, add_one_to_scale=True,
            )
        else:
            modulated_x = (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2)

        layer_recall_compute_dtype = modulated_x.dtype
        layer_recall_query_for_layer = (
            layer_recall_query.to(dtype=layer_recall_compute_dtype)
            if layer_recall_query is not None
            else None
        )
        layer_recall_current_info = {
            "YX_score_query_type": "global",
            "layer_recall_current_conditioned_used": False,
        }
        if (
            layer_recall_config is not None
            and bool(getattr(layer_recall_config, "layer_recall_enabled", False))
            and bool(getattr(layer_recall_config, "layer_recall_current_conditioned_enabled", False))
            and layer_recall_query is not None
            and layer_recall_current_norm is not None
            and layer_recall_current_mlp is not None
        ):
            YX_detach_current_summary = bool(
                getattr(layer_recall_config, "layer_recall_current_detach_summary", True)
            )
            YX_sp_size = int(_CURRENT_GRID_META.get("YX_sp_size", 1))
            if kv_cache is not None:
                YX_current_summary = current_token_summary(
                    modulated_x,
                    token_dim=1,
                    detach=YX_detach_current_summary,
                    tokens_per_frame=(
                        int(_CURRENT_GRID_META.get("frame_seqlen", 0) or 0)
                        or None
                    ),
                )
            else:
                YX_current_summary = modulated_x.mean(dim=1)
                if YX_detach_current_summary:
                    YX_current_summary = YX_current_summary.detach()
            YX_current_summary_normed = _layer_recall_module_forward_in_dtype(
                layer_recall_current_norm,
                YX_current_summary,
                layer_recall_compute_dtype,
            )
            YX_query_delta_by_batch = _layer_recall_module_forward_in_dtype(
                layer_recall_current_mlp,
                YX_current_summary_normed,
                layer_recall_compute_dtype,
            )
            if kv_cache is not None and YX_sp_size > 1:
                if int(YX_query_delta_by_batch.shape[0]) != 1:
                    raise ValueError(
                        "Streaming Ulysses current-conditioned LayerRecall requires B=1; "
                        "batch entries are never merged"
                    )
                YX_query_delta = YX_query_delta_by_batch[0]
            else:
                YX_query_delta = YX_query_delta_by_batch.mean(dim=0)
            YX_query_delta = YX_query_delta.to(
                device=layer_recall_query_for_layer.device,
                dtype=layer_recall_compute_dtype,
            )
            if layer_recall_current_gate is not None:
                YX_current_gate_by_batch = torch.sigmoid(
                    _layer_recall_module_forward_in_dtype(
                        layer_recall_current_gate,
                        YX_current_summary_normed,
                        layer_recall_compute_dtype,
                    )
                )
                if kv_cache is not None and YX_sp_size > 1:
                    YX_current_gate = YX_current_gate_by_batch[0, 0]
                else:
                    YX_current_gate = YX_current_gate_by_batch.mean()
            else:
                YX_current_gate = YX_query_delta.new_tensor(1.0)
            if (
                layer_recall_layer_gamma is not None
                and bool(getattr(layer_recall_config, "layer_recall_use_layer_gamma", True))
                and 0 <= int(layer_recall_layer_index) < int(layer_recall_layer_gamma.numel())
            ):
                YX_layer_gamma = layer_recall_layer_gamma[int(layer_recall_layer_index)].to(
                    device=layer_recall_query_for_layer.device,
                    dtype=layer_recall_compute_dtype,
                )
            else:
                YX_layer_gamma = YX_query_delta.new_tensor(1.0)
            if layer_recall_current_alpha is not None:
                YX_current_alpha = layer_recall_current_alpha.to(
                    device=layer_recall_query_for_layer.device,
                    dtype=layer_recall_compute_dtype,
                )
            else:
                YX_current_alpha = YX_query_delta.new_tensor(
                    float(getattr(layer_recall_config, "layer_recall_current_alpha", 0.1))
                )
            layer_recall_query_for_layer = (
                layer_recall_query_for_layer
                + YX_current_alpha
                * YX_layer_gamma
                * YX_current_gate.to(layer_recall_compute_dtype)
                * YX_query_delta
            )
            layer_recall_current_info = {
                "YX_score_query_type": "current_conditioned",
                "layer_recall_current_conditioned_used": True,
                "layer_recall_current_summary_norm": float(YX_current_summary.detach().float().norm(dim=-1).mean().item()),
                "layer_recall_current_delta_norm": float(YX_query_delta.detach().float().norm().item()),
                "layer_recall_current_gate": float(YX_current_gate.detach().float().item()),
                "layer_recall_current_alpha": float(YX_current_alpha.detach().float().item()),
                "layer_recall_layer_gamma": float(YX_layer_gamma.detach().float().item()),
                "layer_recall_global_query_norm": float(
                    layer_recall_query.to(dtype=layer_recall_compute_dtype).detach().float().norm().item()
                ),
                "layer_recall_current_query_norm": float(layer_recall_query_for_layer.detach().float().norm().item()),
            }
        self_attn_result = self.self_attn(
            modulated_x,
            seq_lens, grid_sizes,
            freqs, block_mask, kv_cache, current_start, cache_start, t_scale=t_scale,
            use_relative_rope=use_relative_rope,
            method=method, original_seq_len=original_seq_len,
            temporal_offset=temporal_offset,
            layer_recall_config=layer_recall_config,
            layer_recall_bank=layer_recall_bank,
            layer_recall_query=layer_recall_query_for_layer,
            layer_recall_logger=layer_recall_logger,
            layer_recall_layer_index=layer_recall_layer_index,
            layer_recall_current_info=layer_recall_current_info)
        
        if kv_cache is not None:
            y, cache_update_info = self_attn_result
        else:
            y = self_attn_result
            cache_update_info = None
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        # iter-40: avoid `seq_lens[0].item()` graph break. seq_lens[0] equals
        # num_new_frames * frame_seqlen at inference time, both of which are
        # Python ints already in _CURRENT_GRID_META (published by iter-31).
        # `is_tf` is True only in teacher-forcing training where x.shape[1]
        # is the doubled (clean+noisy) sequence — never at inference.
        if _CURRENT_GRID_META and "frame_seqlen" in _CURRENT_GRID_META:
            seq_len_py = (
                _CURRENT_GRID_META["frame_seqlen"]
                * _CURRENT_GRID_META["num_new_frames"]
            )
            is_tf = (x.shape[1] == seq_len_py * 2)
        else:
            is_tf = (x.shape[1] == seq_lens[0].item() * 2)
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context,
                                    context_lens, crossattn_cache=crossattn_cache, is_teacher_forcing=is_tf)
            if use_triton_adaln:
                # iter-43: fused LayerNorm + (1+e[4])*x + e[3] in one Triton kernel.
                from utils.adaln_triton import adaln_modulate_triton
                ffn_in = adaln_modulate_triton(
                    x, e[4], e[3], frame_seqlen,
                    eps=self.norm2.eps, add_one_to_scale=True,
                )
            else:
                ffn_in = (self.norm2(x).unflatten(dim=1, sizes=(num_frames,
                          frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            y = self.ffn(ffn_in)
            x = x + (y.unflatten(dim=1, sizes=(num_frames,
                     frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        
        if cache_update_info is not None:
            # cache_update_info is already formatted as
            # (current_end, local_end_index, cache_update_info).
            return x, cache_update_info
        else:
            return x


class CausalHead(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = (self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]))
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video with causal attention.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 local_attn_size=-1,
                 sink_size=0,
                 num_frame_per_block=1,
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video), 'i2v' (image-to-video), or 'ti2v'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            local_attn_size (`int`, *optional*, defaults to -1):
                Window size for temporal local attention (-1 indicates global attention)
            sink_size (`int`, *optional*, defaults to 0):
                Size of the attention sink, we keep the first `sink_size` frames unchanged when rolling the KV cache
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            CausalWanAttentionBlock(dim, ffn_dim, num_heads,
                                  local_attn_size, sink_size, qk_norm, cross_attn_norm, eps)
            for _ in range(num_layers)
        ])
        for YX_layer_index, YX_block in enumerate(self.blocks):
            YX_block.YX_layer_index = int(YX_layer_index)

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)
        self.layer_recall_config = LayerRecallConfig(layer_recall_enabled=False)
        self.layer_recall_bank = LayerRecallMemoryBank()
        self.layer_recall_logger = None
        self.register_parameter("layer_recall_base_query", None)
        self.layer_recall_current_norm = None
        self.layer_recall_current_mlp = None
        self.layer_recall_current_gate = None
        self.register_parameter("layer_recall_layer_gamma", None)
        self.register_parameter("layer_recall_current_alpha", None)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
            dim=1)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False

        self.block_mask = None
        self._block_mask_batch_size = 0

        self.num_frame_per_block = num_frame_per_block
        self.independent_first_frame = False
        self.t_scale = 1.0
        self.use_relative_rope = False
        self.rope_method = "linear"
        self.original_seq_len = None
        self.rope_temporal_offset = 0.0
    def configure_layer_recall(self, layer_recall_config: LayerRecallConfig) -> None:
        layer_recall_config.validate(len(self.blocks))
        self.layer_recall_config = layer_recall_config
        self.layer_recall_bank = LayerRecallMemoryBank()
        self.layer_recall_logger = LayerRecallSelectionLogger(layer_recall_config)
        head_dim = self.dim // self.num_heads
        if bool(layer_recall_config.layer_recall_enabled) and self.layer_recall_base_query is None:
            self.layer_recall_base_query = nn.Parameter(torch.randn(head_dim) * 0.02)
        if bool(layer_recall_config.layer_recall_enabled) and bool(getattr(layer_recall_config, "layer_recall_current_conditioned_enabled", False)):
            hidden_dim = max(1, int(getattr(layer_recall_config, "layer_recall_current_hidden_dim", 512)))
            if self.layer_recall_current_norm is None:
                self.layer_recall_current_norm = nn.LayerNorm(self.dim)
            if self.layer_recall_current_mlp is None:
                self.layer_recall_current_mlp = nn.Sequential(
                    nn.Linear(self.dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, head_dim),
                )
                if bool(getattr(layer_recall_config, "layer_recall_current_zero_init", True)):
                    nn.init.zeros_(self.layer_recall_current_mlp[-1].weight)
                    nn.init.zeros_(self.layer_recall_current_mlp[-1].bias)
            if self.layer_recall_current_gate is None:
                self.layer_recall_current_gate = nn.Linear(self.dim, 1)
            if self.layer_recall_layer_gamma is None:
                gamma = torch.ones(len(self.blocks), dtype=torch.float32)
                self.layer_recall_layer_gamma = nn.Parameter(gamma)
            if self.layer_recall_current_alpha is None:
                self.layer_recall_current_alpha = nn.Parameter(
                    torch.tensor([float(getattr(layer_recall_config, "layer_recall_current_alpha", 0.1))], dtype=torch.float32)
                )

    def reset_layer_recall_memory(self) -> None:
        if hasattr(self, "layer_recall_bank") and self.layer_recall_bank is not None:
            self.layer_recall_bank.clear()
        if hasattr(self, "layer_recall_logger") and self.layer_recall_logger is not None:
            self.layer_recall_logger.reset_counters()

    def _set_gradient_checkpointing(self, module, value=False):
        self.gradient_checkpointing = value

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 31,
        frame_seqlen: int = 880, num_frame_per_block=3,
        batch_size=None,
    ) -> BlockMask:

        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            is_real_q = q_idx < total_length
            is_real_k = kv_idx < total_length
            return (q_idx == kv_idx) | (is_real_q & is_real_k & (kv_idx < ends[q_idx]))

        block_mask = create_block_mask(attention_mask, B=batch_size, H=None, Q_LEN=total_length + padded_length,
                                       KV_LEN=total_length + padded_length, _compile=False, device=device)
        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 31,
        frame_seqlen: int = 880, num_frame_per_block=1,
        batch_size=None,
    ) -> BlockMask:
        """
        Block-wise causal mask. The mask is defined only by the AR chunk size:
        a token can attend to all tokens before the end of its current
        num_frame_per_block chunk.
        """
        print(f"num_frame_per_block: {num_frame_per_block}")
        total_length = num_frames * frame_seqlen
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        block_size = frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            # Apply only to real tokens in [0, total_length); padding keeps only
            # self-loops.
            is_real_q = q_idx < total_length
            is_real_k = kv_idx < total_length

            # End position of the block containing the current token.
            current_block_end = ((q_idx // block_size) + 1) * block_size

            clean_mask = is_real_q & is_real_k & (kv_idx < current_block_end)
            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask

        block_mask = create_block_mask(
            attention_mask,
            B=batch_size,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=True,
            device=device,
        )

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f" cache a block wise causal mask with block size of {num_frame_per_block} frames"
            )
            print(block_mask)

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str,
        num_frames: int = 31,
        frame_seqlen: int = 880,
        num_frame_per_block: int = 1,
        batch_size: int | None = None,
    ):
        total_length = num_frames * frame_seqlen * 2
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        clean_ends = num_frames * frame_seqlen
        attention_block_size = frame_seqlen * num_frame_per_block

        # Use pure mathematical coordinates; do not introduce external tensor
        # lookup tables here.
        def attention_mask(b, h, q_idx, kv_idx):
            is_real_q = q_idx < total_length
            is_real_k = kv_idx < total_length

            # ==========================================
            # 1. Clean-frame mask.
            # ==========================================
            is_clean_q = q_idx < clean_ends

            # End position of the block containing the current token.
            clean_block_idx = q_idx // attention_block_size
            current_clean_block_end = (clean_block_idx + 1) * attention_block_size

            clean_mask = (
                is_clean_q
                & (kv_idx < current_clean_block_end)
            )

            # ==========================================
            # 2. Noisy-frame mask.
            # ==========================================
            is_noisy_q = q_idx >= clean_ends

            noisy_rel_idx = q_idx - clean_ends
            block_index = noisy_rel_idx // attention_block_size

            # C1: noisy tokens in the same block.
            noisy_block_start = clean_ends + (block_index * attention_block_size)
            noisy_block_end = noisy_block_start + attention_block_size
            C1 = (kv_idx >= noisy_block_start) & (kv_idx < noisy_block_end)

            # C2: clean context tokens from previous blocks.
            context_end_for_noisy = block_index * attention_block_size

            C2 = kv_idx < context_end_for_noisy
            noise_mask = is_noisy_q & (C1 | C2)

            # ==========================================
            # 3. Final merge.
            # ==========================================
            eye_mask = q_idx == kv_idx
            return eye_mask | (is_real_q & is_real_k & (clean_mask | noise_mask))

        # _compile=True is required here. Triton compiles the mathematical
        # formula above directly into a memory-efficient block-sparse matrix.
        block_mask = create_block_mask(
            attention_mask,
            B=batch_size,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=True,
            device=device,
        )

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(block_mask)
        
        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask_natural(
        device: torch.device | str,
        num_frames: int,
        frame_seqlen: int,
        num_frame_per_block: int = 1,
        sp_size: int = 1,
        batch_size: int | None = None,
    ):
        """Teacher-Forcing attention mask for the *natural* interleaved layout
        produced directly by `all_to_all(scatter=head, gather=seq)`:

            [r0_clean, r0_noisy, r1_clean, r1_noisy, ..., r_{sp-1}_clean, r_{sp-1}_noisy]

        Semantically equivalent to :func:`_prepare_teacher_forcing_mask` (which
        assumes the [all_clean; all_noisy] layout), but the mask decodes
        ``is_noisy`` and ``global_frame`` directly from the token index, so
        :func:`distributed_flex_attention` no longer has to reshape/permute
        tokens after all_to_all.
        """
        assert num_frames % sp_size == 0, (
            f"num_frames ({num_frames}) must be divisible by sp_size ({sp_size}) "
            f"for natural TF layout"
        )

        F_local = num_frames // sp_size
        clean_half = F_local * frame_seqlen            # per-rank, clean side
        per_rank_len = 2 * clean_half                  # per-rank, clean + noisy
        total_length = num_frames * frame_seqlen * 2
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        def attention_mask(b, h, q_idx, kv_idx):
            is_real_q = q_idx < total_length
            is_real_k = kv_idx < total_length

            # ---- decode q ----
            r_q = q_idx // per_rank_len
            in_rank_q = q_idx % per_rank_len
            is_noisy_q = in_rank_q >= clean_half
            side_q = in_rank_q % clean_half              # offset within clean/noisy half
            global_f_q = r_q * F_local + side_q // frame_seqlen
            block_q = global_f_q // num_frame_per_block

            # ---- decode k ----
            r_k = kv_idx // per_rank_len
            in_rank_k = kv_idx % per_rank_len
            is_noisy_k = in_rank_k >= clean_half
            side_k = in_rank_k % clean_half
            global_f_k = r_k * F_local + side_k // frame_seqlen
            block_k = global_f_k // num_frame_per_block

            # 1. clean_q -> clean_k: blockwise causal.
            clean2clean = (
                (~is_noisy_q) & (~is_noisy_k)
                & (block_k <= block_q)
            )

            # 2. noisy_q -> clean_k: strictly earlier blocks.
            noisy2clean = (
                is_noisy_q & (~is_noisy_k)
                & (block_k < block_q)
            )

            # 3. noisy_q -> noisy_k: only tokens within the same block.
            noisy2noisy = (
                is_noisy_q & is_noisy_k
                & (block_k == block_q)
            )

            eye_mask = q_idx == kv_idx
            return eye_mask | (
                is_real_q & is_real_k
                & (clean2clean | noisy2clean | noisy2noisy)
            )

        block_mask = create_block_mask(
            attention_mask,
            B=batch_size,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=True,
            device=device,
        )

        import torch.distributed as dist
        if not dist.is_initialized() or dist.get_rank() == 0:
            print(
                f"[TF mask natural] sp_size={sp_size} F_local={F_local} "
                f"clean_half={clean_half} per_rank_len={per_rank_len} "
                f"total_length={total_length} "
                f"num_frame_per_block={num_frame_per_block}"
            )
            print(block_mask)

        return block_mask

    def _apply_cache_updates(self, kv_cache, cache_update_infos):
        """
        Applies cache updates collected from multiple blocks.
        Args:
            kv_cache: List of cache dictionaries for each block
            cache_update_infos: List of (block_index, cache_update_info) tuples
        """
        for block_index, (current_end, local_end_index, update_info) in cache_update_infos:
            if update_info is not None:
                cache = kv_cache[block_index]
                if update_info["action"] == "roll_and_insert":
                    # Apply the rolling update.
                    sink_tokens = update_info["sink_tokens"]
                    num_rolled_tokens = update_info["num_rolled_tokens"]
                    num_evicted_tokens = update_info["num_evicted_tokens"]
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]

                    # Roll cached tokens.
                    cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

                    # Insert the new key/value tensors.
                    cache["k"][:, local_start_index:local_end_index] = new_k
                    cache["v"][:, local_start_index:local_end_index] = new_v

                    # If a pinned multi-shot sink lives outside position 0,
                    # the rolling shifted everything left by num_evicted_tokens;
                    # pinned_start must follow so it tracks the same data.
                    pinned_shift = update_info.get("pinned_shift", 0)
                    if pinned_shift > 0 and "pinned_start" in cache:
                        cache["pinned_start"].sub_(pinned_shift)

                elif update_info["action"] == "direct_insert":
                    # Insert directly.
                    local_start_index = update_info["local_start_index"]
                    local_end_index = update_info["local_end_index"]
                    new_k = update_info["new_k"]
                    new_v = update_info["new_v"]
                    # Insert the new key/value tensors.
                    cache["k"][:, local_start_index:local_end_index] = new_k
                    cache["v"][:, local_start_index:local_end_index] = new_v
            
            # Update cache indices.
            kv_cache[block_index]["global_end_index"].fill_(current_end)
            kv_cache[block_index]["local_end_index"].fill_(local_end_index)

    def _forward_inference(
        self,
        x,
        t,
        context,
        seq_len,
        clip_fea=None,
        y=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        current_start: int = 0,
        cache_start: int = 0,
        defer_cache_updates: bool = False,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (880 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B, F]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
        global_grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x]
        )
        first_shape = tuple(x[0].shape[2:])
        global_num_frames = int(first_shape[0])
        frame_seqlen = int(first_shape[1] * first_shape[2])
        _, YX_sp_size, YX_sp_rank = streaming_sp_info()

        pinned_start = int(_CURRENT_GRID_META.get("pinned_start", -1))
        pinned_len = int(_CURRENT_GRID_META.get("pinned_len", 0))
        if (
            "pinned_start" not in _CURRENT_GRID_META
            and kv_cache
            and isinstance(kv_cache[0], dict)
        ):
            pinned_start_value = kv_cache[0].get("pinned_start", -1)
            pinned_len_value = kv_cache[0].get("pinned_len", 0)
            pinned_start = int(
                pinned_start_value.item()
                if torch.is_tensor(pinned_start_value)
                else pinned_start_value
            )
            pinned_len = int(
                pinned_len_value.item()
                if torch.is_tensor(pinned_len_value)
                else pinned_len_value
            )
        YX_global_sink_size = max(
            (int(getattr(block.self_attn, "global_sink_size", 0)) for block in self.blocks),
            default=0,
        )
        _YX_validate_streaming_sp_preflight(
            YX_sp_size=YX_sp_size,
            YX_global_frames=global_num_frames,
            YX_num_heads=self.num_heads,
            YX_use_relative_rope=self.use_relative_rope,
            YX_temporal_offset=self.rope_temporal_offset,
            YX_current_conditioned_enabled=bool(
                getattr(self.layer_recall_config, "layer_recall_enabled", False)
                and getattr(
                    self.layer_recall_config,
                    "layer_recall_current_conditioned_enabled",
                    False,
                )
            ),
            YX_current_detach_summary=bool(
                getattr(self.layer_recall_config, "layer_recall_current_detach_summary", True)
            ),
            YX_pinned_start=pinned_start,
            YX_pinned_len=pinned_len,
            YX_global_sink_size=YX_global_sink_size,
        )
        _YX_prepare_streaming_sp_kv_cache(
            kv_cache,
            YX_sp_size=YX_sp_size,
            YX_sp_rank=YX_sp_rank,
            YX_num_heads=self.num_heads,
        )
        local_frame_start, local_frame_end = local_frame_bounds(global_num_frames)

        # iter-39 v2: kv_cache scalars (global_end_index, local_end_index,
        # pinned_start, pinned_len) are published into _CURRENT_GRID_META by
        # the eager `_call_model` wrapper (utils/wan_5b_wrapper.py) BEFORE
        # this compiled forward runs. Reading them here via `.item()` would
        # trigger graph breaks; the wrapper does it in eager Python instead.

        # time embeddings
        if t.dim() == 1:
            raise NotImplementedError(f"t.shape should be [B, F], but got {t.shape}")

        bt = t.size(0)
        t_len = t.size(1)
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim,
                                    t).unflatten(0, (bt, t_len)).type_as(x[0]))
        e0 = self.time_projection(e).unflatten(2, (6, self.dim)) # B, F, 6, C

        x, e, e0, grid_sizes = _YX_slice_streaming_sp_chunk(
            x,
            e,
            e0,
            global_grid_sizes,
            YX_frame_start=local_frame_start,
            YX_frame_end=local_frame_end,
        )
        local_num_frames = int(local_frame_end - local_frame_start)
        _CURRENT_GRID_META["frame_seqlen"] = frame_seqlen
        _CURRENT_GRID_META["num_new_frames"] = local_num_frames
        _CURRENT_GRID_META["global_num_new_frames"] = global_num_frames
        _CURRENT_GRID_META["local_num_new_frames"] = local_num_frames
        _CURRENT_GRID_META["h"] = int(first_shape[1])
        _CURRENT_GRID_META["w"] = int(first_shape[2])
        _CURRENT_GRID_META["YX_sp_size"] = int(YX_sp_size)
        _CURRENT_GRID_META["YX_sp_rank"] = int(YX_sp_rank)

        x = [u.flatten(2).transpose(1, 2) for u in x]
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        x = torch.cat(x)

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            t_scale=self.t_scale,
            use_relative_rope=self.use_relative_rope,
            method=self.rope_method,
            original_seq_len=self.original_seq_len,
            temporal_offset=self.rope_temporal_offset,
            layer_recall_config=self.layer_recall_config,
            layer_recall_bank=self.layer_recall_bank,
            layer_recall_query=self.layer_recall_base_query,
            layer_recall_logger=self.layer_recall_logger,
            layer_recall_current_norm=self.layer_recall_current_norm,
            layer_recall_current_mlp=self.layer_recall_current_mlp,
            layer_recall_current_gate=self.layer_recall_current_gate,
            layer_recall_layer_gamma=self.layer_recall_layer_gamma,
            layer_recall_current_alpha=self.layer_recall_current_alpha,
        )

        def create_custom_forward(module, layer_index):
            def custom_forward(*inputs, **kwargs):
                try:
                    return module(*inputs, **kwargs)
                except torch.OutOfMemoryError as exc:
                    raise torch.OutOfMemoryError(
                        f"YX causal transformer block OOM at layer={int(layer_index)}: {exc}"
                    ) from exc
            return custom_forward

        cache_update_info = None
        cache_update_infos = []  # Collect cache updates from every block.
        for block_index, block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "layer_recall_layer_index": block_index,
                    }
                )
                result = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block, block_index),
                    x, **kwargs,
                    use_reentrant=False,
                )
                # Handle the result
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    # Keep only basic metadata for later blocks, without the
                    # concrete cache-update payload.
                    cache_update_info = block_cache_update_info[:2]  # (current_end, local_end_index)
                else:
                    x = result
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[block_index],
                        "crossattn_cache": crossattn_cache[block_index],
                        "current_start": current_start,
                        "cache_start": cache_start,
                        "layer_recall_layer_index": block_index,
                    }
                )
                result = create_custom_forward(block, block_index)(x, **kwargs)
                # Handle the result
                if kv_cache is not None and isinstance(result, tuple):
                    x, block_cache_update_info = result
                    cache_update_infos.append((block_index, block_cache_update_info))
                    # Keep only basic metadata for later blocks, without the
                    # concrete cache-update payload.
                    cache_update_info = block_cache_update_info[:2]  # (current_end, local_end_index)
                else:
                    x = result

        # Apply all cache updates after every block has run. For cudagraphs
        # experiments this can be deferred to the eager wrapper so cache
        # mutation does not happen inside the compiled forward.
        if kv_cache is not None and cache_update_infos and not defer_cache_updates:
            self._apply_cache_updates(kv_cache, cache_update_infos)

        # head
        x = self.head(x, e.unsqueeze(2))
        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        output = torch.stack(x)
        if kv_cache is not None and defer_cache_updates:
            return output, cache_update_infos
        return output

    def _forward_train(
        self,
        x,
        t,
        context,
        seq_len,
        clean_x=None,
        aug_t=None,
        clip_fea=None,
        y=None,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert clip_fea is not None and y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        # Construct blockwise causal attn mask
        # Recreate mask when batch size changes to avoid Triton broadcasting bug
        current_batch_size = x.shape[0]
        if self.block_mask is None or self._block_mask_batch_size != current_batch_size:
            self._block_mask_batch_size = current_batch_size
            if clean_x is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        batch_size=current_batch_size,
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        batch_size=current_batch_size,
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=x.shape[2],
                        frame_seqlen=x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2]),
                        num_frame_per_block=self.num_frame_per_block,
                        batch_size=current_batch_size,
                    )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]

        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        max_len = int(seq_lens.max().item())
        assert max_len > 0, "Token sequence length is zero after patch embedding"
        # Pad all samples to the batch max length instead of the first sample length
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, max_len - u.size(1), u.size(2))], dim=1)
            for u in x
        ])

        # time embeddings
        if t.dim() == 1:
            raise NotImplementedError(f"t.shape should be [B, F], but got {t.shape}")
        bt = t.size(0)
        t_len = t.size(1)
        t_ori_shape = t.shape
        t = t.flatten()
        e = self.time_embedding(
            sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, t_len)).type_as(x))
        e0 = self.time_projection(e).unflatten(2, (6, self.dim)) # B, F, 6, C

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        if clean_x is not None:
            clean_x = [self.patch_embedding(u.unsqueeze(0)) for u in clean_x]
            clean_x = [u.flatten(2).transpose(1, 2) for u in clean_x]

            seq_lens_clean = torch.tensor([u.size(1) for u in clean_x], dtype=torch.long)
            clean_x = torch.cat([
                torch.cat([u, u.new_zeros(1, seq_lens_clean[0] - u.size(1), u.size(2))], dim=1) for u in clean_x
            ])

            x = torch.cat([clean_x, x], dim=1)
            if aug_t is None:
                aug_t = torch.zeros(t_ori_shape, device=t.device, dtype=t.dtype)
            bt_clean = aug_t.size(0)
            t_clean_len = aug_t.size(1)
            aug_t = aug_t.flatten()
            e_clean = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim, aug_t).unflatten(0, (bt_clean, t_clean_len)).type_as(x))
            e0_clean = self.time_projection(e_clean).unflatten(2, (6, self.dim))
            e0 = torch.cat([e0_clean, e0], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
            t_scale=self.t_scale,
            method=self.rope_method,
            original_seq_len=self.original_seq_len,
            temporal_offset=self.rope_temporal_offset,
        )

        def create_custom_forward(module):
            def custom_forward(*inputs, **kwargs):
                return module(*inputs, **kwargs)
            return custom_forward

        for block in self.blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    x, **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)

        if clean_x is not None:
            x = x[:, x.shape[1] // 2:]

        # head
        x = self.head(x, e.unsqueeze(2))

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return torch.stack(x)

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
