import os
import tempfile
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from wan_5b.distributed import sp_training
from wan_5b.distributed.streaming_ulysses import (
    all_gather_detached_frames,
    assert_sp_metadata_consistent,
    collective_telemetry_snapshot,
    current_token_summary,
    k_summary,
    reset_collective_telemetry,
    sp_global_mean,
    sp_sum,
    ulysses_head_to_seq,
    ulysses_packed_qkv_seq_to_head,
    ulysses_seq_to_head,
)


def _YX_sentinel_qkv(rank):
    q = torch.empty(1, 2, 4, 1, dtype=torch.float64)
    for sequence_index in range(2):
        for head_index in range(4):
            q[0, sequence_index, head_index, 0] = (
                rank * 1000 + sequence_index * 100 + head_index
            )
    return q, q + 10_000, q + 20_000


def _YX_expected_seq_to_head(rank, offset=0):
    expected = torch.empty(1, 4, 2, 1, dtype=torch.float64)
    for source_rank in range(2):
        for sequence_index in range(2):
            global_sequence_index = source_rank * 2 + sequence_index
            for local_head_index in range(2):
                global_head_index = rank * 2 + local_head_index
                expected[0, global_sequence_index, local_head_index, 0] = (
                    offset
                    + source_rank * 1000
                    + sequence_index * 100
                    + global_head_index
                )
    return expected


def _YX_world_two_worker(rank, world_size, init_method):
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    sp_training.set_sequence_parallel_group(dist.group.WORLD)
    try:
        q, k, v = _YX_sentinel_qkv(rank)

        reset_collective_telemetry()
        q_global, k_global, v_global = ulysses_packed_qkv_seq_to_head(q, k, v)
        torch.testing.assert_close(q_global, _YX_expected_seq_to_head(rank))
        torch.testing.assert_close(k_global, _YX_expected_seq_to_head(rank, 10_000))
        torch.testing.assert_close(v_global, _YX_expected_seq_to_head(rank, 20_000))
        packed_stats = collective_telemetry_snapshot()
        assert packed_stats["collective_count"] == 1
        assert packed_stats["operations"]["packed_qkv_seq_to_head"]["collective_count"] == 1

        q_oracle = ulysses_seq_to_head(q)
        k_oracle = ulysses_seq_to_head(k)
        v_oracle = ulysses_seq_to_head(v)
        torch.testing.assert_close(q_global, q_oracle)
        torch.testing.assert_close(k_global, k_oracle)
        torch.testing.assert_close(v_global, v_oracle)

        q_roundtrip = ulysses_head_to_seq(q_global)
        torch.testing.assert_close(q_roundtrip, q)

        q_grad = q.clone().requires_grad_(True)
        k_grad = k.clone().requires_grad_(True)
        v_grad = v.clone().requires_grad_(True)
        q_exchanged, k_exchanged, v_exchanged = ulysses_packed_qkv_seq_to_head(
            q_grad, k_grad, v_grad
        )
        q_local = ulysses_head_to_seq(q_exchanged)
        loss = q_local.sum() + 2.0 * k_exchanged.sum() + 3.0 * v_exchanged.sum()
        loss.backward()
        torch.testing.assert_close(q_grad.grad, torch.ones_like(q_grad))
        torch.testing.assert_close(k_grad.grad, torch.full_like(k_grad, 2.0))
        torch.testing.assert_close(v_grad.grad, torch.full_like(v_grad, 3.0))

        rank_value = torch.tensor([float(rank + 1)], requires_grad=True)
        global_sum = sp_sum(rank_value)
        global_mean = sp_global_mean(rank_value)
        torch.testing.assert_close(global_sum, torch.tensor([3.0]))
        torch.testing.assert_close(global_mean, torch.tensor([1.5]))
        (global_mean.sum() / world_size).backward()
        torch.testing.assert_close(rank_value.grad, torch.tensor([0.5]))

        current = torch.tensor(
            [
                [[rank * 4.0], [rank * 4.0 + 2.0]],
                [[rank * 4.0 + 10.0], [rank * 4.0 + 12.0]],
            ],
            requires_grad=True,
        )
        current_summary = current_token_summary(current, detach=True)
        torch.testing.assert_close(current_summary, torch.tensor([[3.0], [13.0]]))
        assert current_summary.shape == (2, 1)
        assert not current_summary.requires_grad

        framewise_current = current_token_summary(
            current,
            detach=True,
            tokens_per_frame=2,
        )
        torch.testing.assert_close(framewise_current, torch.tensor([[3.0], [13.0]]))
        assert framewise_current.dtype == torch.float32

        k_local = (
            torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
            + rank * 100
        ).requires_grad_(True)
        pooled_k = k_summary(k_local, detach=True)
        torch.testing.assert_close(pooled_k, torch.tensor([53.0, 54.0]))
        assert pooled_k.shape == (2,)
        assert not pooled_k.requires_grad

        framewise_k = k_summary(k_local, detach=True, tokens_per_frame=2)
        torch.testing.assert_close(framewise_k, torch.tensor([53.0, 54.0]))

        local_frames = torch.tensor(
            [float(rank), float(10 + rank)], dtype=torch.float32
        ).reshape(2, 1, 1, 1, 1)
        gathered_frames = all_gather_detached_frames(local_frames)
        expected_frames = torch.tensor(
            [[0.0, 1.0], [10.0, 11.0]], dtype=torch.float32
        ).reshape(2, 2, 1, 1, 1)
        torch.testing.assert_close(gathered_frames, expected_frames)
        with torch.enable_grad():
            with _YX_assert_raises(ValueError, "must pass local_frames.detach"):
                all_gather_detached_frames(local_frames.requires_grad_(True))

        with _YX_assert_raises(ValueError, "head divisibility"):
            invalid_q = torch.zeros(1, 2, 3, 1)
            ulysses_packed_qkv_seq_to_head(invalid_q, invalid_q, invalid_q)
        with _YX_assert_raises(ValueError, "sequence divisibility"):
            ulysses_head_to_seq(torch.zeros(1, 3, 2, 1))

        consistent_rows = assert_sp_metadata_consistent([17, 29], label="chunk IDs")
        assert consistent_rows == ((17, 29), (17, 29))
        with _YX_assert_raises(RuntimeError, "metadata mismatch"):
            assert_sp_metadata_consistent([17, rank], label="chunk IDs")

        stats = collective_telemetry_snapshot()
        assert stats["collective_count"] > 1
        assert stats["estimated_bytes"] > 0
        assert stats["collective_time_s"] >= 0.0

        # WORLD remains size=2, but a missing SP group explicitly means SP=1.
        sp_training.set_sequence_parallel_group(None)
        reset_collective_telemetry()
        local_q, local_k, local_v = ulysses_packed_qkv_seq_to_head(q, k, v)
        assert local_q is q
        assert local_k is k
        assert local_v is v
        assert ulysses_head_to_seq(q) is q
        assert sp_sum(q) is q
        assert sp_global_mean(q) is q
        detached_local_frames = local_frames.detach()
        assert (
            all_gather_detached_frames(detached_local_frames)
            is detached_local_frames
        )
        assert assert_sp_metadata_consistent([17, rank]) == ((17, rank),)
        disabled_stats = collective_telemetry_snapshot()
        assert disabled_stats["collective_count"] == 0
        assert disabled_stats["estimated_bytes"] == 0
    finally:
        sp_training.set_sequence_parallel_group(None)
        dist.destroy_process_group()


class _YX_assert_raises:
    def __init__(self, exception_type, description):
        self.exception_type = exception_type
        self.description = description

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception, traceback):
        if exception_type is None:
            raise AssertionError(f"Expected {self.exception_type.__name__}: {self.description}")
        if not issubclass(exception_type, self.exception_type):
            return False
        return True


class YXStreamingUlyssesCollectivesTest(unittest.TestCase):
    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "gloo required")
    def test_world_two_collectives(self):
        with tempfile.TemporaryDirectory(prefix="YX_streaming_ulysses_") as temp_dir:
            init_method = f"file://{os.path.join(temp_dir, 'process_group_init')}"
            mp.spawn(
                _YX_world_two_worker,
                args=(2, init_method),
                nprocs=2,
                join=True,
            )

    def test_sp_one_fast_paths(self):
        self.assertFalse(dist.is_initialized())
        q = torch.randn(2, 3, 4, 5, requires_grad=True)
        k = torch.randn_like(q)
        v = torch.randn_like(q)

        q_output, k_output, v_output = ulysses_packed_qkv_seq_to_head(q, k, v)
        self.assertIs(q_output, q)
        self.assertIs(k_output, k)
        self.assertIs(v_output, v)
        self.assertIs(ulysses_head_to_seq(q), q)
        self.assertIs(sp_sum(q), q)
        self.assertIs(sp_global_mean(q), q)

        detached_sum = sp_sum(q, detach=True)
        self.assertFalse(detached_sum.requires_grad)
        torch.testing.assert_close(detached_sum, q.detach())

        current_summary = current_token_summary(q, detach=True)
        torch.testing.assert_close(current_summary, q.detach().mean(dim=1))
        self.assertFalse(current_summary.requires_grad)

        pooled_k = k_summary(k, detach=True)
        torch.testing.assert_close(pooled_k, k.detach().mean(dim=(0, 1, 2)))
        self.assertEqual(pooled_k.shape, (5,))
        self.assertFalse(pooled_k.requires_grad)

        frames = torch.randn(2, 3, 1, 4, 5)
        self.assertIs(all_gather_detached_frames(frames), frames)
        self.assertEqual(
            assert_sp_metadata_consistent(torch.tensor([4, 8])),
            ((4, 8),),
        )
        with self.assertRaises(TypeError):
            assert_sp_metadata_consistent([4.5, 8.5])

        reset_collective_telemetry()
        stats = collective_telemetry_snapshot()
        self.assertEqual(stats["collective_count"], 0)
        self.assertEqual(stats["estimated_bytes"], 0)


if __name__ == "__main__":
    unittest.main()
