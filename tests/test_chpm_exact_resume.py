from __future__ import annotations

import copy
import random
import tempfile
import unittest
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DistributedSampler

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from utils.chpm_resume import (
    CHPMPromptStream,
    canonical_sha256,
)
from trainer.chpm import Trainer


class PredictionPromptStreamTests(unittest.TestCase):
    def test_matches_distributed_sampler(self):
        for dataset_size in (3, 8, 9, 10, 17, 1600):
            for num_replicas in (2, 3, 6):
                if dataset_size < num_replicas:
                    continue
                for epoch in (0, 1, 7):
                    for rank in range(num_replicas):
                        with self.subTest(
                            dataset_size=dataset_size,
                            num_replicas=num_replicas,
                            epoch=epoch,
                            rank=rank,
                        ):
                            expected = DistributedSampler(
                                list(range(dataset_size)),
                                num_replicas=num_replicas,
                                rank=rank,
                                shuffle=True,
                                drop_last=True,
                                seed=17,
                            )
                            expected.set_epoch(epoch)
                            stream = CHPMPromptStream(
                                dataset_size,
                                rank=rank,
                                num_replicas=num_replicas,
                                seed=17,
                                epoch=epoch,
                            )
                            actual = []
                            for _ in range(stream.num_batches_per_epoch):
                                batch = stream.peek_batch()
                                actual.extend(batch)
                                stream.commit_batch(batch)
                            self.assertEqual(actual, list(expected))

    def test_resume_mid_epoch_is_exact(self):
        uninterrupted = CHPMPromptStream(
            17, rank=1, num_replicas=3, seed=20260720
        )
        for _ in range(2):
            batch = uninterrupted.peek_batch()
            uninterrupted.commit_batch(batch)
        checkpoint_state = uninterrupted.state_dict()
        suffix = []
        for _ in range(6):
            batch = uninterrupted.peek_batch()
            suffix.extend(batch)
            uninterrupted.commit_batch(batch)

        resumed = CHPMPromptStream(
            17, rank=1, num_replicas=3, seed=20260720
        )
        resumed.load_state_dict(checkpoint_state)
        resumed_suffix = []
        for _ in range(6):
            batch = resumed.peek_batch()
            resumed_suffix.extend(batch)
            resumed.commit_batch(batch)

        self.assertEqual(resumed_suffix, suffix)
        self.assertEqual(resumed.state_dict(), uninterrupted.state_dict())

    def test_resume_across_epoch_boundary_is_exact(self):
        stream = CHPMPromptStream(17, rank=0, num_replicas=3, seed=9)
        for _ in range(stream.num_batches_per_epoch):
            batch = stream.peek_batch()
            stream.commit_batch(batch)
        self.assertEqual(stream.epoch, 1)
        self.assertEqual(stream.sample_cursor, 0)
        checkpoint_state = stream.state_dict()
        expected = stream.peek_batch()

        resumed = CHPMPromptStream(17, rank=0, num_replicas=3, seed=9)
        resumed.load_state_dict(checkpoint_state)
        self.assertEqual(resumed.peek_batch(), expected)

    def test_refuses_snapshot_before_commit(self):
        stream = CHPMPromptStream(17, rank=0, num_replicas=3)
        stream.peek_batch()
        with self.assertRaisesRegex(RuntimeError, "uncommitted batch"):
            stream.state_dict()

    def test_rejects_topology_change(self):
        source = CHPMPromptStream(17, rank=0, num_replicas=3)
        state = source.state_dict()
        target = CHPMPromptStream(17, rank=0, num_replicas=2)
        with self.assertRaisesRegex(ValueError, "num_replicas mismatch"):
            target.load_state_dict(state)

    def test_canonical_hash_is_order_independent_for_mapping(self):
        self.assertEqual(
            canonical_sha256({"a": 1, "b": [2, 3]}),
            canonical_sha256({"b": [2, 3], "a": 1}),
        )

    def test_complete_marker_filters_partial_checkpoint(self):
        trainer = Trainer.__new__(Trainer)
        with tempfile.TemporaryDirectory(prefix="YX_resume_test_") as tmp:
            root = Path(tmp)
            partial = root / "checkpoint_model_000002"
            partial.mkdir()
            torch.save({"step": 2}, partial / "model.pt")
            self.assertIsNone(
                trainer.find_latest_checkpoint(str(root), require_complete=True)
            )

            complete = root / "checkpoint_model_000001"
            complete.mkdir()
            torch.save({"step": 1}, complete / "model.pt")
            (complete / Trainer.CHECKPOINT_COMPLETE_MARKER).write_text("step=1\n")
            self.assertEqual(
                trainer.find_latest_checkpoint(str(root), require_complete=True),
                str(complete / "model.pt"),
            )


class PredictionResumeCheckpointSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trainer = Trainer.__new__(Trainer)
        cls.trainer.gradient_accumulation_steps = 1
        cls.trainer.world_size = 2
        cls.trainer.sequence_parallel_size = 2
        cls.trainer.data_parallel_size = 1

        tensor_sizes = [1] * 10 + [Trainer.EXPECTED_LAYER_RECALL_NUMEL - 10]
        layer_recall_state = {
            f"layer_recall_tensor_{index}": torch.zeros(size, dtype=torch.float32)
            for index, size in enumerate(tensor_sizes)
        }
        named_parameters = [
            (name, torch.nn.Parameter(value)) for name, value in layer_recall_state.items()
        ]
        cls.trainer.model = SimpleNamespace(
            pre_fsdp_trainable_layer_recall_named_param_objects=lambda: named_parameters
        )
        cls.contract = {"contract_version": 1, "semantic_token": "fixed"}
        cls.trainer._build_critical_resume_contract = lambda: copy.deepcopy(
            cls.contract
        )

        stream_states = []
        rng_states = []
        for global_rank in range(2):
            stream_states.append(
                {
                    "global_rank": global_rank,
                    "sp_rank": global_rank,
                    "dp_rank": 0,
                    "stream": {
                        "version": 1,
                        "dataset_size": 17,
                        "rank": 0,
                        "num_replicas": 1,
                        "batch_size": 1,
                        "shuffle": True,
                        "drop_last": True,
                        "seed": 7,
                        "epoch": 0,
                        "sample_cursor": 2,
                        "batch_cursor_in_epoch": 2,
                        "global_micro_step": 2,
                    },
                }
            )
            rng_states.append(
                {
                    "global_rank": global_rank,
                    "state": {
                        "version": 1,
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch_cpu": torch.get_rng_state(),
                        "torch_cuda": torch.zeros(1, dtype=torch.uint8),
                        "data_generator": torch.Generator().get_state(),
                    },
                }
            )

        manifest = {
            "mode": "txt",
            "dataset_size": 17,
            "ordered_training_prompt_hashes": ["a", "b"],
        }
        optimizer_state = {
            "state": {
                index: {
                    "step": torch.tensor(2.0),
                    "exp_avg": torch.zeros(1),
                    "exp_avg_sq": torch.zeros(1),
                }
                for index in range(Trainer.EXPECTED_LAYER_RECALL_TENSORS)
            },
            "param_groups": [{"params": list(range(Trainer.EXPECTED_LAYER_RECALL_TENSORS))}],
        }
        cls.valid_checkpoint = {
            "trainer": Trainer.CHECKPOINT_FORMAT,
            "checkpoint_version": Trainer.CHECKPOINT_VERSION,
            "layer_recall_state_dict": layer_recall_state,
            "student_optimizer": optimizer_state,
            "step": 2,
            "global_step": 2,
            "global_micro_step": 2,
            "accumulation_step": 0,
            "data_stream_states": stream_states,
            "rng_states": rng_states,
            "dataset_manifest": manifest,
            "dataset_manifest_hash": canonical_sha256(manifest),
            "trainable_schema": cls.trainer._current_trainable_schema(),
            "critical_resume_contract": cls.contract,
            "critical_resume_fingerprint": canonical_sha256(cls.contract),
            "config": {"trainer": Trainer.CHECKPOINT_FORMAT},
        }

    def _validate(self, checkpoint):
        return self.trainer._validate_resume_checkpoint_schema(
            checkpoint,
            checkpoint_path="/tmp/checkpoint_model_000002/model.pt",
        )

    def test_accepts_complete_consistent_schema(self):
        validated = self._validate(copy.deepcopy(self.valid_checkpoint))
        self.assertEqual(validated["global_step"], 2)

    def test_rejects_global_micro_step_mismatch(self):
        checkpoint = copy.deepcopy(self.valid_checkpoint)
        checkpoint["global_micro_step"] = 1
        with self.assertRaisesRegex(RuntimeError, "global/micro step mismatch"):
            self._validate(checkpoint)

    def test_rejects_optimizer_step_mismatch(self):
        checkpoint = copy.deepcopy(self.valid_checkpoint)
        checkpoint["student_optimizer"]["state"][0]["step"] = torch.tensor(1.0)
        with self.assertRaisesRegex(RuntimeError, "optimizer internal step mismatch"):
            self._validate(checkpoint)

    def test_rejects_sp_rank_data_stream_divergence(self):
        checkpoint = copy.deepcopy(self.valid_checkpoint)
        checkpoint["data_stream_states"][1]["stream"]["sample_cursor"] = 3
        with self.assertRaisesRegex(RuntimeError, "data stream ranks are not synchronized"):
            self._validate(checkpoint)

    def test_rejects_incomplete_rank_rng(self):
        checkpoint = copy.deepcopy(self.valid_checkpoint)
        del checkpoint["rng_states"][1]["state"]["torch_cuda"]
        with self.assertRaisesRegex(RuntimeError, "RNG state is incomplete"):
            self._validate(checkpoint)

    def test_rejects_dataset_manifest_tampering(self):
        checkpoint = copy.deepcopy(self.valid_checkpoint)
        checkpoint["dataset_manifest"]["dataset_size"] = 18
        with self.assertRaisesRegex(RuntimeError, "dataset manifest fingerprint"):
            self._validate(checkpoint)

    def test_rejects_critical_contract_change(self):
        checkpoint = copy.deepcopy(self.valid_checkpoint)
        checkpoint["critical_resume_contract"]["semantic_token"] = "changed"
        checkpoint["critical_resume_fingerprint"] = canonical_sha256(
            checkpoint["critical_resume_contract"]
        )
        with self.assertRaisesRegex(RuntimeError, "critical resume contract mismatch"):
            self._validate(checkpoint)


if __name__ == "__main__":
    unittest.main()
