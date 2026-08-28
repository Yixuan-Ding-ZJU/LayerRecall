from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
TARGET_TRAINER = REPO_ROOT / "trainer" / "chpm.py"


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_trainer_module():
    model_package = _module("model")
    model_package.__path__ = []
    model_module = _module(
        "model.chpm",
        CHPMModel=object,
    )

    utils_package = _module("utils")
    utils_package.__path__ = []
    config_module = _module(
        "utils.config",
        wan_default_config={
            "Wan2.2-TI2V-5B": {
                "num_heads": 24,
            }
        },
    )
    dataset_module = _module(
        "utils.dataset",
        MultiTextConcatDataset=object,
        MultiVideoConcatDataset=object,
        eval_collate_fn=lambda batch: batch,
        multi_video_collate_fn=lambda batch: batch,
    )
    distributed_module = _module(
        "utils.distributed",
        FSDP=object,
        barrier=lambda: None,
        fsdp_wrap=lambda model, **kwargs: model,
        launch_distributed_job=lambda: None,
    )
    misc_module = _module("utils.misc", set_seed=lambda seed: None)
    resume_module = _module(
        "utils.chpm_resume",
        CHPMPromptStream=type(
            "CHPMPromptStreamStub",
            (),
            {"REQUIRED_STATE_KEYS": frozenset()},
        ),
        canonical_sha256=lambda value: "stub",
        capture_rng_state=lambda *args, **kwargs: {},
        restore_rng_state=lambda *args, **kwargs: None,
        validate_rng_state=lambda *args, **kwargs: None,
    )

    wan_package = _module("wan_5b")
    wan_package.__path__ = []
    wan_distributed_package = _module("wan_5b.distributed")
    wan_distributed_package.__path__ = []
    sp_training_module = _module(
        "wan_5b.distributed.sp_training",
        SequenceParallelHelper=object,
        set_sequence_parallel_group=lambda group: None,
        set_data_parallel_group=lambda group: None,
    )
    streaming_ulysses_module = _module(
        "wan_5b.distributed.streaming_ulysses",
        collective_telemetry_snapshot=lambda: {},
        reset_collective_telemetry=lambda: None,
    )
    wan_distributed_package.sp_training = sp_training_module
    wan_modules_package = _module("wan_5b.modules")
    wan_modules_package.__path__ = []
    causal_model_module = _module(
        "wan_5b.modules.causal_model",
        CausalWanAttentionBlock=object,
    )

    omegaconf_module = _module("omegaconf", OmegaConf=object)
    wandb_module = _module("wandb")
    stubs = {
        "model": model_package,
        "model.chpm": model_module,
        "utils": utils_package,
        "utils.config": config_module,
        "utils.dataset": dataset_module,
        "utils.distributed": distributed_module,
        "utils.misc": misc_module,
        "utils.chpm_resume": resume_module,
        "wan_5b": wan_package,
        "wan_5b.distributed": wan_distributed_package,
        "wan_5b.distributed.sp_training": sp_training_module,
        "wan_5b.distributed.streaming_ulysses": streaming_ulysses_module,
        "wan_5b.modules": wan_modules_package,
        "wan_5b.modules.causal_model": causal_model_module,
        "omegaconf": omegaconf_module,
        "wandb": wandb_module,
    }

    module_name = "YX_chpm_topology_under_test"
    spec = importlib.util.spec_from_file_location(module_name, TARGET_TRAINER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load trainer from {TARGET_TRAINER}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


class _FakeDist:
    def __init__(self):
        self.calls = []

    def new_group(self, ranks):
        group = tuple(ranks)
        self.calls.append(group)
        return group


class _FakeSPTraining:
    def __init__(self):
        self.sp_groups = []
        self.dp_groups = []

    def set_sequence_parallel_group(self, group):
        self.sp_groups.append(group)

    def set_data_parallel_group(self, group):
        self.dp_groups.append(group)


class _FakeSampler:
    def __init__(self):
        self.epochs = []

    def set_epoch(self, epoch):
        self.epochs.append(epoch)


class YXLayerRecallPredictionSPTopologyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer = _load_trainer_module()

    def _topology(self, rank=0, world=6, sp=2, **overrides):
        values = {
            "global_rank": rank,
            "world_size": world,
            "sequence_parallel_size": sp,
            "streaming_sequence_parallel_mode": "ulysses_chunk" if sp > 1 else "disabled",
            "model_name": "Wan2.2-TI2V-5B",
            "num_heads": 24,
            "num_frame_per_block": 8,
            "batch_size": 2,
            "gradient_accumulation_steps": 3,
        }
        values.update(overrides)
        return self.trainer._YX_resolve_prediction_sp_topology(**values)

    def test_world6_sp2_topology_and_group_layout(self):
        topology = self._topology(rank=4)

        self.assertEqual(topology.sp_rank, 0)
        self.assertEqual(topology.dp_rank, 2)
        self.assertEqual(topology.sp_size, 2)
        self.assertEqual(topology.dp_size, 3)
        self.assertEqual(topology.local_frames, 4)
        self.assertEqual(topology.local_heads, 12)
        self.assertEqual(topology.effective_global_batch, 18)
        self.assertEqual(topology.sp_group_ranks, (4, 5))
        self.assertEqual(topology.dp_group_ranks, (0, 2, 4))

        fake_dist = _FakeDist()
        sp_group, dp_group = self.trainer._YX_create_prediction_process_groups(
            topology,
            dist_api=fake_dist,
        )
        self.assertEqual(
            fake_dist.calls,
            [(0, 1), (2, 3), (4, 5), (0, 2, 4), (1, 3, 5)],
        )
        self.assertEqual(sp_group, (4, 5))
        self.assertEqual(dp_group, (0, 2, 4))

    def test_sp1_degrades_without_subgroups_and_clears_global_state(self):
        topology = self._topology(rank=4, sp=1)
        fake_dist = _FakeDist()
        fake_sp_training = _FakeSPTraining()

        sp_group, dp_group = self.trainer._YX_initialize_prediction_process_groups(
            topology,
            dist_api=fake_dist,
            sp_training_api=fake_sp_training,
        )

        self.assertEqual(topology.sp_rank, 0)
        self.assertEqual(topology.dp_rank, 4)
        self.assertEqual(topology.sp_size, 1)
        self.assertEqual(topology.dp_size, 6)
        self.assertEqual(topology.local_frames, 8)
        self.assertEqual(topology.local_heads, 24)
        self.assertEqual(topology.effective_global_batch, 36)
        self.assertEqual(self.trainer._YX_sampler_rank_and_replicas(topology), (4, 6))
        self.assertIsNone(sp_group)
        self.assertIsNone(dp_group)
        self.assertEqual(fake_dist.calls, [])
        self.assertEqual(fake_sp_training.sp_groups, [None])
        self.assertEqual(fake_sp_training.dp_groups, [None])

    def test_sp3_and_sp6_are_rejected(self):
        for sp_size in (3, 6):
            with self.subTest(sp_size=sp_size):
                with self.assertRaisesRegex(ValueError, "only supports sequence_parallel_size=2"):
                    self._topology(sp=sp_size, world=6)

    def test_world_size_must_be_divisible_by_sp(self):
        with self.assertRaisesRegex(ValueError, "world_size .* divisible"):
            self._topology(world=5, sp=2)

    def test_num_heads_must_be_divisible_by_sp(self):
        with self.assertRaisesRegex(ValueError, "num_heads .* divisible"):
            self._topology(num_heads=23)

    def test_num_frame_per_block_must_be_divisible_by_sp(self):
        with self.assertRaisesRegex(ValueError, "num_frame_per_block .* divisible"):
            self._topology(num_frame_per_block=7)

    def test_only_wan22_ti2v_5b_is_allowed(self):
        with self.assertRaisesRegex(ValueError, "only supports Wan2.2-TI2V-5B"):
            self._topology(model_name="Wan2.1-T2V-1.3B")

    def test_sp2_requires_ulysses_chunk_mode(self):
        with self.assertRaisesRegex(ValueError, "must be 'ulysses_chunk'"):
            self._topology(streaming_sequence_parallel_mode="disabled")

    def test_same_sp_pair_uses_same_sampler_dp_rank(self):
        rank0 = self._topology(rank=0)
        rank1 = self._topology(rank=1)
        rank2 = self._topology(rank=2)

        self.assertEqual(self.trainer._YX_sampler_rank_and_replicas(rank0), (0, 3))
        self.assertEqual(self.trainer._YX_sampler_rank_and_replicas(rank1), (0, 3))
        self.assertEqual(self.trainer._YX_sampler_rank_and_replicas(rank2), (1, 3))

    def test_micro_step_seed_matches_within_sp_pair_and_differs_across_dp(self):
        common = {
            "base_seed": 123,
            "dp_size": 3,
            "step": 7,
            "accumulation_step": 1,
            "accumulation_steps": 2,
        }
        rank0_seed = self.trainer._YX_micro_step_seed(dp_rank=0, **common)
        rank1_seed = self.trainer._YX_micro_step_seed(dp_rank=0, **common)
        rank2_seed = self.trainer._YX_micro_step_seed(dp_rank=1, **common)

        self.assertEqual(rank0_seed, rank1_seed)
        self.assertNotEqual(rank0_seed, rank2_seed)

    def test_epoch_aware_iterator_advances_sampler_epoch(self):
        sampler = _FakeSampler()
        iterator = self.trainer._YX_epoch_aware_iterator(["a", "b"], sampler)

        self.assertEqual([next(iterator) for _ in range(5)], ["a", "b", "a", "b", "a"])
        self.assertEqual(sampler.epochs, [0, 1, 2])


if __name__ == "__main__":
    unittest.main()
