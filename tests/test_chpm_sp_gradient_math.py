from __future__ import annotations

import os
import tempfile
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from utils.chpm_sp import (
    clip_synced_grad_norm_,
    normalize_sp_prediction_local_sum,
    scale_replicated_regularization,
    sp_global_detached_count,
    sp_global_detached_sum_count,
    sync_replicated_layer_recall_gradients_,
)


_PREDICTION_COEFFICIENTS = (
    ((1.0, 2.0), (3.0, 4.0)),
    ((2.0, -1.0), (0.0, 5.0)),
    ((-2.0, 3.0), (4.0, 1.0)),
)
_LOCAL_VALID_COUNTS = ((1, 2), (2, 3), (3, 4))
_REGULARIZATION_COEFFICIENTS = (
    (0.5, -0.25),
    (1.0, -0.5),
    (1.5, -0.75),
)


def _YX_world_six_worker(rank: int, world_size: int, init_method: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=90),
    )
    try:
        sp_groups = [
            dist.new_group(ranks=[0, 1]),
            dist.new_group(ranks=[2, 3]),
            dist.new_group(ranks=[4, 5]),
        ]
        dp_rank = rank // 2
        sp_rank = rank % 2
        sp_group = sp_groups[dp_rank]

        layer_recall = torch.tensor([1.5, -0.5], dtype=torch.float64, requires_grad=True)
        local_coefficient = torch.tensor(
            _PREDICTION_COEFFICIENTS[dp_rank][sp_rank],
            dtype=layer_recall.dtype,
        )
        local_count = torch.tensor(
            _LOCAL_VALID_COUNTS[dp_rank][sp_rank],
            dtype=torch.int64,
        )
        regularization_coefficient = torch.tensor(
            _REGULARIZATION_COEFFICIENTS[dp_rank],
            dtype=layer_recall.dtype,
        )

        local_prediction_sum = (local_coefficient * layer_recall).sum()
        prediction_loss = normalize_sp_prediction_local_sum(
            local_prediction_sum,
            local_count,
            sp_group=sp_group,
        )
        regularization_loss = (regularization_coefficient * layer_recall).sum()
        scaled_regularization_loss = scale_replicated_regularization(
            regularization_loss,
            sp_size=2,
        )

        logging_sum, logging_count = sp_global_detached_sum_count(
            local_prediction_sum,
            local_count,
            sp_group=sp_group,
        )
        expected_pair_sum = sum(
            torch.tensor(coefficients, dtype=layer_recall.dtype).dot(layer_recall.detach()).item()
            for coefficients in _PREDICTION_COEFFICIENTS[dp_rank]
        )
        expected_pair_count = sum(_LOCAL_VALID_COUNTS[dp_rank])
        torch.testing.assert_close(
            logging_sum,
            torch.tensor(expected_pair_sum, dtype=torch.float64),
        )
        torch.testing.assert_close(
            logging_count,
            torch.tensor(float(expected_pair_count), dtype=torch.float64),
        )
        assert not logging_sum.requires_grad
        assert not logging_count.requires_grad

        # Immediate local backward, followed by one WORLD synchronization.
        (prediction_loss + scaled_regularization_loss).backward()
        sync_result = sync_replicated_layer_recall_gradients_(
            [("layer_recall", layer_recall)],
            dp_size=3,
        )

        # Hand calculation by DP replica:
        #   DP0: [4/3, 2]   + [1/2, -1/4] = [11/6, 7/4]
        #   DP1: [2/5, 4/5] + [1,   -1/2] = [7/5,  3/10]
        #   DP2: [2/7, 4/7] + [3/2, -3/4] = [25/14, -5/28]
        # Their DP average is [527/315, 131/210].
        expected_dp_average = torch.tensor(
            [527.0 / 315.0, 131.0 / 210.0],
            dtype=layer_recall.dtype,
        )
        torch.testing.assert_close(layer_recall.grad, expected_dp_average)
        assert sync_result.world_size == 6
        assert sync_result.dp_size == 3
        assert sync_result.sp_size == 2
        assert sync_result.synchronized_parameter_count == 1

        # SP=1 under an initialized WORLD: no SP collective or reg division;
        # WORLD=DP, so the final operation is an ordinary six-way DP average.
        sp1_layer_recall = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
        sp1_local_count = torch.tensor(rank + 1, dtype=torch.int64)
        sp1_global_count = sp_global_detached_count(
            sp1_local_count,
            sp_group=None,
        )
        torch.testing.assert_close(sp1_global_count, sp1_local_count)
        sp1_prediction = normalize_sp_prediction_local_sum(
            sp1_layer_recall * float(rank + 1),
            sp1_local_count,
            sp_group=None,
        )
        sp1_regularization = scale_replicated_regularization(
            sp1_layer_recall * float(rank),
            sp_size=1,
        )
        (sp1_prediction + sp1_regularization).backward()
        sp1_result = sync_replicated_layer_recall_gradients_([sp1_layer_recall], dp_size=6)
        # Local prediction gradients are all 1; reg gradients are rank 0..5.
        torch.testing.assert_close(
            sp1_layer_recall.grad,
            torch.tensor(3.5, dtype=sp1_layer_recall.dtype),
        )
        assert sp1_result.sp_size == 1

        # Every rank must observe the mismatch and raise before value reduction.
        inconsistent = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
        if rank != 0:
            inconsistent.grad = torch.ones_like(inconsistent)
        try:
            sync_replicated_layer_recall_gradients_(
                [("inconsistent", inconsistent)],
                dp_size=3,
            )
        except RuntimeError as error:
            assert "inconsistent None gradient" in str(error)
            assert "5/6 WORLD ranks" in str(error)
        else:
            raise AssertionError("expected inconsistent None gradients to fail")
    finally:
        dist.destroy_process_group()


class YXLayerRecallSPGradientPureMathTest(unittest.TestCase):
    def test_sp1_count_and_logging_are_detached(self):
        local_count = torch.tensor(7.0, requires_grad=True)
        global_count = sp_global_detached_count(local_count, sp_group=None)
        self.assertFalse(global_count.requires_grad)
        self.assertIsNot(global_count, local_count)
        torch.testing.assert_close(global_count, torch.tensor(7.0))

        local_sum = torch.tensor(11.0, requires_grad=True)
        logging_sum, logging_count = sp_global_detached_sum_count(
            local_sum,
            local_count,
            sp_group=None,
        )
        self.assertFalse(logging_sum.requires_grad)
        self.assertFalse(logging_count.requires_grad)
        torch.testing.assert_close(logging_sum, torch.tensor(11.0, dtype=torch.float64))
        torch.testing.assert_close(logging_count, torch.tensor(7.0, dtype=torch.float64))

    def test_prediction_and_replicated_regularization_scaling(self):
        parameter = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        prediction = normalize_sp_prediction_local_sum(
            10.0 * parameter,
            torch.tensor(5),
            sp_group=None,
        )
        regularization = scale_replicated_regularization(
            6.0 * parameter,
            sp_size=2,
        )
        (prediction + regularization).backward()
        torch.testing.assert_close(parameter.grad, torch.tensor(5.0, dtype=torch.float64))

        identity_loss = parameter.square()
        self.assertIs(
            scale_replicated_regularization(identity_loss, sp_size=1),
            identity_loss,
        )

    def test_single_process_sync_and_post_sync_clip(self):
        fp32 = torch.nn.Parameter(torch.tensor([3.0, 4.0], dtype=torch.float32))
        fp64 = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float64))
        (fp32.sum() + 2.0 * fp64.sum()).backward()

        result = sync_replicated_layer_recall_gradients_(
            [("fp32", fp32), ("fp64", fp64)],
            dp_size=1,
        )
        torch.testing.assert_close(fp32.grad, torch.ones_like(fp32))
        torch.testing.assert_close(fp64.grad, torch.full_like(fp64, 2.0))
        self.assertEqual(result.world_size, 1)
        self.assertEqual(result.sp_size, 1)
        self.assertEqual(result.collective_count, 0)

        total_norm = clip_synced_grad_norm_([fp32, fp64], max_norm=1.0)
        self.assertGreater(total_norm.item(), 1.0)
        clipped_norm = torch.sqrt(fp32.grad.float().square().sum() + fp64.grad.float().square().sum())
        torch.testing.assert_close(clipped_norm, torch.tensor(1.0), atol=1e-6, rtol=1e-6)

    def test_invalid_scaling_and_topology_fail_early(self):
        loss = torch.tensor(1.0, requires_grad=True)
        with self.assertRaisesRegex(ValueError, "sp_size"):
            scale_replicated_regularization(loss, sp_size=0)
        with self.assertRaisesRegex(ValueError, "divisible"):
            sync_replicated_layer_recall_gradients_([loss], dp_size=2)
        with self.assertRaisesRegex(ValueError, "positive"):
            normalize_sp_prediction_local_sum(
                loss,
                torch.tensor(0),
                sp_group=None,
            )


class YXLayerRecallSPGradientDistributedMathTest(unittest.TestCase):
    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(),
        "torch.distributed Gloo is required",
    )
    def test_world6_sp2_dp3_prediction_and_reg_gradient(self):
        with tempfile.TemporaryDirectory(prefix="layer_recall_sp_gradients_") as temp_dir:
            init_method = f"file://{os.path.join(temp_dir, 'process_group_init')}"
            mp.spawn(
                _YX_world_six_worker,
                args=(6, init_method),
                nprocs=6,
                join=True,
            )


if __name__ == "__main__":
    unittest.main()
