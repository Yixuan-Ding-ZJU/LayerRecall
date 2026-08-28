import inspect
import unittest

import torch
import torch.nn as nn

from wan_5b.modules import causal_model


def _cache(*, heads=24):
    return {
        "k": torch.zeros(1, 14080, heads, 2),
        "v": torch.zeros(1, 14080, heads, 2),
    }


def _preflight_kwargs():
    return {
        "YX_sp_size": 2,
        "YX_global_frames": 8,
        "YX_num_heads": 24,
        "YX_use_relative_rope": False,
        "YX_temporal_offset": 0.0,
        "YX_current_conditioned_enabled": False,
        "YX_current_detach_summary": True,
        "YX_pinned_start": -1,
        "YX_pinned_len": 0,
        "YX_global_sink_size": 0,
    }


class CausalStreamingUlyssesIntegrationTest(unittest.TestCase):
    def test_fp32_layer_recall_master_uses_differentiable_bf16_compute_cast(self):
        module = nn.Sequential(
            nn.Linear(4, 3),
            nn.SiLU(),
            nn.Linear(3, 2),
        ).float()
        value = torch.arange(8, dtype=torch.bfloat16).reshape(2, 4)

        output = causal_model._layer_recall_module_forward_in_dtype(
            module,
            value,
            torch.bfloat16,
        )
        output.float().sum().backward()

        self.assertEqual(output.dtype, torch.bfloat16)
        for parameter in module.parameters():
            self.assertEqual(parameter.dtype, torch.float32)
            self.assertIsNotNone(parameter.grad)
            self.assertEqual(parameter.grad.dtype, torch.float32)

    def test_frame_slice_keeps_patch_grid_and_time_embeddings_aligned(self):
        patched = [
            torch.arange(8, dtype=torch.float32).view(1, 1, 8, 1, 1),
            torch.arange(100, 108, dtype=torch.float32).view(1, 1, 8, 1, 1),
        ]
        e = torch.arange(2 * 8 * 3, dtype=torch.float32).view(2, 8, 3)
        e0 = torch.arange(2 * 8 * 6 * 3, dtype=torch.float32).view(2, 8, 6, 3)
        grid = torch.tensor([[8, 1, 1], [8, 1, 1]], dtype=torch.long)

        local_x, local_e, local_e0, local_grid = (
            causal_model._YX_slice_streaming_sp_chunk(
                patched,
                e,
                e0,
                grid,
                YX_frame_start=4,
                YX_frame_end=8,
            )
        )

        self.assertEqual(local_x[0].shape, (1, 1, 4, 1, 1))
        self.assertEqual(local_x[0].flatten().tolist(), [4.0, 5.0, 6.0, 7.0])
        self.assertEqual(
            local_x[1].flatten().tolist(),
            [104.0, 105.0, 106.0, 107.0],
        )
        self.assertTrue(torch.equal(local_e, e[:, 4:8]))
        self.assertTrue(torch.equal(local_e0, e0[:, 4:8]))
        self.assertEqual(local_grid.tolist(), [[4, 1, 1], [4, 1, 1]])
        self.assertEqual(grid.tolist(), [[8, 1, 1], [8, 1, 1]])

    def test_sp1_preflight_and_cache_prepare_are_noops(self):
        cache = _cache()
        original_k = cache["k"]
        original_v = cache["v"]
        causal_model._YX_validate_streaming_sp_preflight(
            YX_sp_size=1,
            YX_global_frames=7,
            YX_num_heads=23,
            YX_use_relative_rope=True,
            YX_temporal_offset=torch.tensor(1.0),
            YX_current_conditioned_enabled=True,
            YX_current_detach_summary=False,
            YX_pinned_start=0,
            YX_pinned_len=1,
            YX_global_sink_size=1,
        )
        result = causal_model._YX_prepare_streaming_sp_kv_cache(
            [cache],
            YX_sp_size=1,
            YX_sp_rank=0,
            YX_num_heads=24,
        )

        self.assertIs(result[0]["k"], original_k)
        self.assertIs(result[0]["v"], original_v)
        self.assertNotIn("YX_streaming_sp_size", cache)

    def test_streaming_sp_preflight_rejects_unsupported_modes(self):
        cases = [
            ({"YX_sp_size": 3}, "SP=2 only"),
            ({"YX_global_frames": 7}, "frames"),
            ({"YX_num_heads": 23}, "heads"),
            ({"YX_use_relative_rope": True}, "relative RoPE"),
            (
                {
                    "YX_current_conditioned_enabled": True,
                    "YX_current_detach_summary": False,
                },
                "layer_recall_current_detach_summary=true",
            ),
            ({"YX_temporal_offset": torch.tensor(0.0)}, "tensor temporal_offset"),
            ({"YX_temporal_offset": 1.0}, "temporal_offset=0"),
            ({"YX_pinned_start": 0, "YX_pinned_len": 7040}, "pinned"),
            ({"YX_global_sink_size": 8}, "pinned/multi-shot"),
        ]
        for updates, message in cases:
            with self.subTest(updates=updates, message=message):
                kwargs = _preflight_kwargs()
                kwargs.update(updates)
                with self.assertRaisesRegex(ValueError, message):
                    causal_model._YX_validate_streaming_sp_preflight(**kwargs)

    def test_streaming_sp_preflight_does_not_forbid_autograd(self):
        with torch.enable_grad():
            causal_model._YX_validate_streaming_sp_preflight(
                **_preflight_kwargs()
            )

    def test_cache_head_shard_is_contiguous_and_idempotent(self):
        head_values = torch.arange(24).view(1, 1, 24, 1).expand(1, 16, 24, 2)
        cache = {
            "k": head_values.clone(),
            "v": (head_values + 100).clone(),
        }
        caches = [cache]

        causal_model._YX_prepare_streaming_sp_kv_cache(
            caches,
            YX_sp_size=2,
            YX_sp_rank=1,
            YX_num_heads=24,
        )
        self.assertEqual(cache["k"].shape, (1, 16, 12, 2))
        self.assertEqual(cache["k"][0, 0, :, 0].tolist(), list(range(12, 24)))
        self.assertEqual(cache["v"][0, 0, :, 0].tolist(), list(range(112, 124)))
        self.assertTrue(cache["k"].is_contiguous())

        first_local_k = cache["k"]
        causal_model._YX_prepare_streaming_sp_kv_cache(
            caches,
            YX_sp_size=2,
            YX_sp_rank=1,
            YX_num_heads=24,
        )
        self.assertIs(cache["k"], first_local_k)

    def test_context_record_keeps_global_and_local_token_coordinates(self):
        record = causal_model._YX_make_chunk_memory_record(
            YX_chunk_index=2,
            YX_start_frame=16,
            YX_num_frames=8,
            YX_cache_start_token=7040,
            YX_cache_end_token=14080,
            YX_global_start_token=14080,
            YX_global_end_token=21120,
            YX_summary=torch.ones(128),
        )

        self.assertEqual(record.token_range, (7040, 14080))
        self.assertEqual(record.global_token_range, (14080, 21120))
        self.assertEqual(record.num_tokens, 7040)

    def test_self_attention_exchange_order_and_inverse_before_output_projection(self):
        source = inspect.getsource(causal_model.CausalWanSelfAttention.forward)
        summary_pos = source.index("YX_global_k_summary = k_summary(")
        packed_pos = source.index("ulysses_packed_qkv_seq_to_head(q, k, v)")
        inverse_pos = source.index("ulysses_head_to_seq(x)")
        output_projection_pos = source.index("self.o(x)")

        self.assertLess(summary_pos, packed_pos)
        self.assertLess(packed_pos, inverse_pos)
        self.assertLess(inverse_pos, output_projection_pos)
        self.assertEqual(
            source.count("ulysses_packed_qkv_seq_to_head(q, k, v)"),
            1,
        )
        self.assertIn("YX_current_start_token=int(current_start)", source)
        self.assertIn("YX_global_start_token=int(current_start)", source)
        self.assertNotIn("torch.topk", source)
        self.assertNotIn("torch.sort", source)
        self.assertIn("YX_transient_bank_records", source)
        self.assertLess(
            source.index("layer_recall_bank.apply_cache_roll("),
            source.index("layer_recall_bank.YX_records_by_layer"),
        )

    def test_inference_keeps_global_and_local_grid_metadata_and_unified_cache_apply(self):
        source = inspect.getsource(causal_model.CausalWanModel._forward_inference)
        self.assertIn('"global_num_new_frames"', source)
        self.assertIn('"local_num_new_frames"', source)
        self.assertLess(
            source.index("for block_index, block in enumerate(self.blocks)"),
            source.index("self._apply_cache_updates(kv_cache, cache_update_infos)"),
        )


if __name__ == "__main__":
    unittest.main()
