from __future__ import annotations

import os
import sys
import unittest
from datetime import timedelta
from pathlib import Path


# Keep this parity test on the portable eager RoPE path. Attention itself still
# dispatches through the repository's current CUDA attention implementation.
os.environ.setdefault("LLV2_TRITON_ROPE", "0")
os.environ.setdefault("LLV2_FREQS_I_CACHE", "0")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.distributed as dist
import torch.nn as nn

from utils.layer_recall import (
    HistoryChunkRecord,
    LayerRecallConfig,
    LayerRecallMemoryBank,
    clear_layer_recall_context,
    set_layer_recall_context,
)
from utils.chpm_sp import sync_replicated_layer_recall_gradients_
from wan_5b.distributed import sp_training
from wan_5b.distributed.streaming_ulysses import (
    collective_telemetry_snapshot,
    reset_collective_telemetry,
)
from wan_5b.modules import causal_model


_YX_BATCH = 1
_YX_DIM = 64
_YX_HEADS = 4
_YX_HEAD_DIM = _YX_DIM // _YX_HEADS
_YX_GLOBAL_TOKENS = 4
_YX_LOCAL_TOKENS = _YX_GLOBAL_TOKENS // 2
_YX_CACHE_TOKENS = 16
_YX_CACHE_PREFILL = 8
_YX_CURRENT_START = 884
_YX_EXPECTED_SELECTED_IDS = [7, 3]


class _YXSelectionCapture:
    def __init__(self) -> None:
        self.events = []

    def log(self, payload) -> None:
        self.events.append(payload)

    def selection_event(self):
        matches = [
            event
            for event in self.events
            if event.get("YX_call_type") == "denoise"
            and "YX_selected_chunk_ids" in event
        ]
        if len(matches) != 1:
            raise AssertionError(f"Expected one LayerRecall selection event, got {len(matches)}")
        return matches[0]


def _YX_make_attention(device: torch.device) -> causal_model.CausalWanSelfAttention:
    module = causal_model.CausalWanSelfAttention(
        dim=_YX_DIM,
        num_heads=_YX_HEADS,
        local_attn_size=1,
        sink_size=0,
        qk_norm=False,
    )
    with torch.no_grad():
        for parameter_index, parameter in enumerate(module.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape_as(
                parameter
            )
            values = ((values.remainder(31) - 15.0) / 50.0) + (
                parameter_index + 1
            ) * 0.002
            parameter.copy_(values)
    module.requires_grad_(False)
    return module.to(device=device, dtype=torch.bfloat16).eval()


def _YX_full_input(device: torch.device) -> torch.Tensor:
    values = torch.arange(
        _YX_BATCH * _YX_GLOBAL_TOKENS * _YX_DIM,
        dtype=torch.float32,
        device=device,
    ).reshape(_YX_BATCH, _YX_GLOBAL_TOKENS, _YX_DIM)
    return (0.35 * torch.sin(values * 0.17) + 0.15 * torch.cos(values * 0.11)).to(
        torch.bfloat16
    )


def _YX_loss_weights(device: torch.device) -> torch.Tensor:
    values = torch.arange(
        _YX_BATCH * _YX_GLOBAL_TOKENS * _YX_DIM,
        dtype=torch.float32,
        device=device,
    ).reshape(_YX_BATCH, _YX_GLOBAL_TOKENS, _YX_DIM)
    return 0.2 + values.remainder(23) / 29.0


def _YX_make_cache(device: torch.device):
    values = torch.arange(
        _YX_BATCH * _YX_CACHE_TOKENS * _YX_HEADS * _YX_HEAD_DIM,
        dtype=torch.float32,
        device=device,
    ).reshape(_YX_BATCH, _YX_CACHE_TOKENS, _YX_HEADS, _YX_HEAD_DIM)
    return {
        "k": (0.45 * torch.sin(values * 0.037)).to(torch.bfloat16),
        "v": (0.40 * torch.cos(values * 0.053)).to(torch.bfloat16),
        "global_end_index": torch.tensor(
            _YX_CURRENT_START, dtype=torch.int64, device=device
        ),
        "local_end_index": torch.tensor(
            _YX_CACHE_PREFILL, dtype=torch.int64, device=device
        ),
    }


def _YX_make_bank(device: torch.device) -> LayerRecallMemoryBank:
    bank = LayerRecallMemoryBank()
    summary_low = torch.zeros(_YX_HEAD_DIM, dtype=torch.float32, device=device)
    summary_low[1] = 1.0
    summary_high = torch.zeros(_YX_HEAD_DIM, dtype=torch.float32, device=device)
    summary_high[0] = 1.0
    bank.add_or_replace(
        0,
        HistoryChunkRecord(
            chunk_index=3,
            start_frame=_YX_CURRENT_START - 8,
            num_frames=_YX_GLOBAL_TOKENS,
            cache_start_token=0,
            cache_end_token=4,
            global_start_token=_YX_CURRENT_START - 8,
            global_end_token=_YX_CURRENT_START - 4,
            summary=summary_low,
        ),
    )
    bank.add_or_replace(
        0,
        HistoryChunkRecord(
            chunk_index=7,
            start_frame=_YX_CURRENT_START - 4,
            num_frames=_YX_GLOBAL_TOKENS,
            cache_start_token=4,
            cache_end_token=8,
            global_start_token=_YX_CURRENT_START - 4,
            global_end_token=_YX_CURRENT_START,
            summary=summary_high,
        ),
    )
    return bank


def _YX_make_layer_recall_config() -> LayerRecallConfig:
    config = LayerRecallConfig(
        layer_recall_enabled=True,
        layer_recall_normalize_scores=False,
        layer_recall_selection_mode="straight_through_topk",
        layer_recall_temperature=1.0,
        layer_recall_candidate_pool_size=2,
        layer_recall_num_heads=_YX_HEADS,
        layer_recall_head_dim=_YX_HEAD_DIM,
        layer_recall_num_layers=1,
        layer_recall_frame_seq_length=1,
        layer_recall_chunk_token_size=_YX_GLOBAL_TOKENS,
        layer_recall_current_conditioned_enabled=False,
        memory_sensitive_layers=(0,),
    )
    # This runtime-only debug knob exercises rank-consistency checks for the
    # candidate IDs, selected IDs, and global/local cache coordinates.
    config.layer_recall_sp_debug_consistency = True
    return config


def _YX_make_query(device: torch.device) -> nn.Parameter:
    values = torch.linspace(-0.08, 0.08, _YX_HEAD_DIM, device=device)
    values[0] = 0.20
    values[1] = -0.10
    return nn.Parameter(values.to(torch.float32))


def _YX_freqs(device: torch.device) -> torch.Tensor:
    spatial_pairs = _YX_HEAD_DIM // 6
    return torch.cat(
        [
            causal_model.rope_params(1024, _YX_HEAD_DIM - 4 * spatial_pairs),
            causal_model.rope_params(1024, 2 * spatial_pairs),
            causal_model.rope_params(1024, 2 * spatial_pairs),
        ],
        dim=1,
    ).to(device)


def _YX_set_grid_metadata(*, sp_size: int, sp_rank: int) -> None:
    local_tokens = _YX_GLOBAL_TOKENS if sp_size == 1 else _YX_LOCAL_TOKENS
    causal_model._CURRENT_GRID_META.clear()
    causal_model._CURRENT_GRID_META.update(
        {
            "frame_seqlen": 1,
            "num_new_frames": local_tokens,
            "global_num_new_frames": _YX_GLOBAL_TOKENS,
            "local_num_new_frames": local_tokens,
            "h": 1,
            "w": 1,
            "YX_sp_size": sp_size,
            "YX_sp_rank": sp_rank,
        }
    )


def _YX_forward(
    module,
    x,
    cache,
    query,
    bank,
    logger,
    config,
    freqs,
    *,
    local_tokens,
):
    return module(
        x=x,
        seq_lens=torch.tensor([_YX_GLOBAL_TOKENS], device=x.device),
        grid_sizes=torch.tensor([[local_tokens, 1, 1]], device=x.device),
        freqs=freqs,
        block_mask=None,
        kv_cache=cache,
        current_start=_YX_CURRENT_START,
        cache_start=_YX_CURRENT_START,
        layer_recall_config=config,
        layer_recall_bank=bank,
        layer_recall_query=query,
        layer_recall_logger=logger,
        layer_recall_layer_index=0,
    )


class YXCausalStreamingUlyssesCudaParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not torch.cuda.is_available() or not dist.is_nccl_available():
            raise unittest.SkipTest("two CUDA devices and NCCL are required")
        if int(os.environ.get("WORLD_SIZE", "1")) != 2:
            raise unittest.SkipTest(
                "launch with torchrun --standalone --nproc_per_node=2"
            )
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if local_rank < 0 or local_rank >= torch.cuda.device_count():
            raise unittest.SkipTest("LOCAL_RANK must identify a visible CUDA device")

        torch.cuda.set_device(local_rank)
        cls.device = torch.device("cuda", local_rank)
        cls._owns_process_group = not dist.is_initialized()
        if cls._owns_process_group:
            dist.init_process_group(backend="nccl", timeout=timedelta(seconds=120))
        if dist.get_world_size() != 2 or str(dist.get_backend()).lower() != "nccl":
            raise RuntimeError("This test requires an NCCL WORLD process group of size 2")
        cls.rank = dist.get_rank()

    @classmethod
    def tearDownClass(cls) -> None:
        sp_training.set_sequence_parallel_group(None)
        clear_layer_recall_context()
        causal_model._CURRENT_GRID_META.clear()
        if getattr(cls, "_owns_process_group", False) and dist.is_initialized():
            dist.destroy_process_group()

    def test_sp1_sp2_forward_backward_cache_and_selection_parity(self) -> None:
        original_grid_metadata = dict(causal_model._CURRENT_GRID_META)
        device = self.device
        rank = self.rank
        token_start = rank * _YX_LOCAL_TOKENS
        token_end = token_start + _YX_LOCAL_TOKENS
        head_start = rank * (_YX_HEADS // 2)
        head_end = head_start + (_YX_HEADS // 2)

        set_layer_recall_context(
            YX_call_type="denoise",
            YX_cfg_branch="pos",
            YX_chunk_index=221,
            YX_chunk_start_frame=_YX_CURRENT_START,
            YX_num_frames=_YX_GLOBAL_TOKENS,
            YX_execution_phase="training",
        )
        try:
            full_input = _YX_full_input(device)
            full_loss_weights = _YX_loss_weights(device)
            freqs = _YX_freqs(device)
            config = _YX_make_layer_recall_config()

            # SP=1 oracle: WORLD remains initialized, but no configured SP group
            # means the streaming helpers take their full-sequence fast path.
            sp_training.set_sequence_parallel_group(None)
            _YX_set_grid_metadata(sp_size=1, sp_rank=0)
            reference_module = _YX_make_attention(device)
            reference_cache = _YX_make_cache(device)
            causal_model._YX_prepare_streaming_sp_kv_cache(
                [reference_cache],
                YX_sp_size=1,
                YX_sp_rank=0,
                YX_num_heads=_YX_HEADS,
            )
            reference_query = _YX_make_query(device)
            reference_logger = _YXSelectionCapture()
            reference_input = full_input.detach().clone().requires_grad_(True)
            reference_output, reference_indices = _YX_forward(
                reference_module,
                reference_input,
                reference_cache,
                reference_query,
                _YX_make_bank(device),
                reference_logger,
                config,
                freqs,
                local_tokens=_YX_GLOBAL_TOKENS,
            )
            reference_loss = (
                reference_output.float() * full_loss_weights
            ).sum()
            reference_loss.backward()
            reference_input_grad = reference_input.grad.detach().clone()
            reference_query_grad = reference_query.grad.detach().clone()

            # SP=2 candidate: each rank owns a contiguous token shard while the
            # cache helper shards full caches into contiguous local head ranges.
            sp_training.set_sequence_parallel_group(dist.group.WORLD)
            _YX_set_grid_metadata(sp_size=2, sp_rank=rank)
            sharded_module = _YX_make_attention(device)
            sharded_cache = _YX_make_cache(device)
            full_cache_k = sharded_cache["k"].clone()
            full_cache_v = sharded_cache["v"].clone()
            causal_model._YX_prepare_streaming_sp_kv_cache(
                [sharded_cache],
                YX_sp_size=2,
                YX_sp_rank=rank,
                YX_num_heads=_YX_HEADS,
            )
            sharded_query = _YX_make_query(device)
            sharded_logger = _YXSelectionCapture()
            sharded_input = (
                full_input[:, token_start:token_end]
                .detach()
                .contiguous()
                .requires_grad_(True)
            )

            reset_collective_telemetry()
            sharded_output, sharded_indices = _YX_forward(
                sharded_module,
                sharded_input,
                sharded_cache,
                sharded_query,
                _YX_make_bank(device),
                sharded_logger,
                config,
                freqs,
                local_tokens=_YX_LOCAL_TOKENS,
            )
            sharded_loss = (
                sharded_output.float()
                * full_loss_weights[:, token_start:token_end]
            ).sum()
            sharded_loss.backward()
            sync_result = sync_replicated_layer_recall_gradients_(
                [("layer_recall_base_query", sharded_query)],
                dp_size=1,
                world_group=dist.group.WORLD,
            )
            telemetry = collective_telemetry_snapshot()

            reference_event = reference_logger.selection_event()
            sharded_event = sharded_logger.selection_event()
            reference_global_end, reference_local_end, reference_update = (
                reference_indices
            )
            sharded_global_end, sharded_local_end, sharded_update = sharded_indices

            self.assertEqual(tuple(reference_output.shape), (1, 4, _YX_DIM))
            self.assertEqual(tuple(sharded_output.shape), (1, 2, _YX_DIM))
            torch.testing.assert_close(
                sharded_output.float(),
                reference_output[:, token_start:token_end].float(),
                rtol=0.035,
                atol=0.025,
            )
            torch.testing.assert_close(
                sharded_input.grad.float(),
                reference_input_grad[:, token_start:token_end].float(),
                rtol=0.05,
                atol=0.035,
            )

            self.assertGreater(float(reference_query_grad.norm().item()), 1.0e-7)
            self.assertGreater(float(sharded_query.grad.norm().item()), 1.0e-7)
            torch.testing.assert_close(
                sharded_query.grad.float(),
                reference_query_grad.float(),
                rtol=0.08,
                atol=0.003,
            )
            self.assertEqual(sync_result.world_size, 2)
            self.assertEqual(sync_result.dp_size, 1)
            self.assertEqual(sync_result.sp_size, 2)
            self.assertEqual(sync_result.synchronized_parameter_count, 1)

            self.assertEqual(tuple(reference_cache["k"].shape), (1, 16, 4, 16))
            self.assertEqual(tuple(sharded_cache["k"].shape), (1, 16, 2, 16))
            self.assertTrue(sharded_cache["k"].is_contiguous())
            self.assertEqual(sharded_cache["YX_streaming_sp_size"], 2)
            self.assertEqual(sharded_cache["YX_streaming_sp_rank"], rank)
            torch.testing.assert_close(
                sharded_cache["k"], full_cache_k[:, :, head_start:head_end]
            )
            torch.testing.assert_close(
                sharded_cache["v"], full_cache_v[:, :, head_start:head_end]
            )

            expected_global_end = _YX_CURRENT_START + _YX_GLOBAL_TOKENS
            expected_local_end = _YX_CACHE_PREFILL + _YX_GLOBAL_TOKENS
            self.assertEqual(reference_global_end, expected_global_end)
            self.assertEqual(sharded_global_end, expected_global_end)
            self.assertEqual(reference_local_end, expected_local_end)
            self.assertEqual(sharded_local_end, expected_local_end)
            self.assertEqual(reference_update["action"], "direct_insert")
            self.assertEqual(sharded_update["action"], "direct_insert")
            self.assertEqual(reference_update["local_start_index"], _YX_CACHE_PREFILL)
            self.assertEqual(sharded_update["local_start_index"], _YX_CACHE_PREFILL)
            self.assertEqual(reference_update["local_end_index"], expected_local_end)
            self.assertEqual(sharded_update["local_end_index"], expected_local_end)
            self.assertEqual(
                tuple(reference_update["new_k"].shape), (1, 4, 4, 16)
            )
            self.assertEqual(tuple(sharded_update["new_k"].shape), (1, 4, 2, 16))
            torch.testing.assert_close(
                sharded_update["new_k"].float(),
                reference_update["new_k"][:, :, head_start:head_end].float(),
                rtol=0.025,
                atol=0.02,
            )
            torch.testing.assert_close(
                sharded_update["new_v"].float(),
                reference_update["new_v"][:, :, head_start:head_end].float(),
                rtol=0.025,
                atol=0.02,
            )

            self.assertEqual(
                reference_event["YX_selected_chunk_ids"],
                _YX_EXPECTED_SELECTED_IDS,
            )
            self.assertEqual(
                sharded_event["YX_selected_chunk_ids"],
                _YX_EXPECTED_SELECTED_IDS,
            )
            self.assertEqual(reference_event["YX_score_top_ids"], [7, 3])
            self.assertEqual(sharded_event["YX_score_top_ids"], [7, 3])
            self.assertTrue(reference_event["YX_candidate_scores_requires_grad"])
            self.assertTrue(sharded_event["YX_candidate_scores_requires_grad"])

            operations = telemetry["operations"]
            self.assertEqual(
                operations["packed_qkv_seq_to_head"]["collective_count"], 1
            )
            self.assertEqual(operations["head_to_seq"]["collective_count"], 1)
            self.assertGreater(telemetry["estimated_bytes"], 0)
        finally:
            sp_training.set_sequence_parallel_group(None)
            clear_layer_recall_context()
            causal_model._CURRENT_GRID_META.clear()
            causal_model._CURRENT_GRID_META.update(original_grid_metadata)


if __name__ == "__main__":
    unittest.main()
