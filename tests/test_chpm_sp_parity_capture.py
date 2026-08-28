from __future__ import annotations

import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.distributed.fsdp  # noqa: F401

# FSDP registers process-global c10d Meta kernels on first import. Keep the
# real module loaded before any temporary sys.modules stubs are installed.


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "model" / "chpm.py"
TRAINER_PATH = REPO_ROOT / "trainer" / "chpm.py"


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


class _OmegaConfStub:
    @staticmethod
    def is_config(value):
        return False


def _load_model_module():
    utils_package = _module("utils")
    utils_package.__path__ = []
    stubs = {
        "omegaconf": _module("omegaconf", OmegaConf=_OmegaConfStub),
        "utils": utils_package,
        "utils.config": _module("utils.config", wan_default_config={}),
        "utils.wan_5b_wrapper": _module(
            "utils.wan_5b_wrapper",
            WanDiffusionWrapper=object,
            WanTextEncoder=object,
            WanVAEWrapper=object,
        ),
        "utils.layer_recall": _module(
            "utils.layer_recall",
            LayerRecallConfig=object,
            clear_layer_recall_context=lambda: None,
            set_layer_recall_context=lambda **kwargs: None,
        ),
    }
    module_name = "layer_recall_sp_parity_model_under_test"
    spec = importlib.util.spec_from_file_location(module_name, MODEL_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {MODEL_PATH}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


def _load_trainer_module():
    model_package = _module("model")
    model_package.__path__ = []
    utils_package = _module("utils")
    utils_package.__path__ = []
    wan_package = _module("wan_5b")
    wan_package.__path__ = []
    wan_distributed = _module("wan_5b.distributed")
    wan_distributed.__path__ = []
    sp_training = _module(
        "wan_5b.distributed.sp_training",
        SequenceParallelHelper=object,
        set_sequence_parallel_group=lambda group: None,
        set_data_parallel_group=lambda group: None,
    )
    streaming_ulysses = _module(
        "wan_5b.distributed.streaming_ulysses",
        collective_telemetry_snapshot=lambda: {},
        reset_collective_telemetry=lambda: None,
    )
    prediction_resume = _module(
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
    wan_distributed.sp_training = sp_training
    wan_modules = _module("wan_5b.modules")
    wan_modules.__path__ = []
    stubs = {
        "model": model_package,
        "model.chpm": _module(
            "model.chpm",
            CHPMModel=object,
        ),
        "utils": utils_package,
        "utils.config": _module(
            "utils.config",
            wan_default_config={"Wan2.2-TI2V-5B": {"num_heads": 24}},
        ),
        "utils.dataset": _module(
            "utils.dataset",
            MultiTextConcatDataset=object,
            MultiVideoConcatDataset=object,
            eval_collate_fn=lambda batch: batch,
            multi_video_collate_fn=lambda batch: batch,
        ),
        "utils.distributed": _module(
            "utils.distributed",
            FSDP=object,
            barrier=lambda: None,
            fsdp_wrap=lambda model, **kwargs: model,
            launch_distributed_job=lambda: None,
        ),
        "utils.misc": _module("utils.misc", set_seed=lambda seed: None),
        "utils.chpm_resume": prediction_resume,
        "wan_5b": wan_package,
        "wan_5b.distributed": wan_distributed,
        "wan_5b.distributed.sp_training": sp_training,
        "wan_5b.distributed.streaming_ulysses": streaming_ulysses,
        "wan_5b.modules": wan_modules,
        "wan_5b.modules.causal_model": _module(
            "wan_5b.modules.causal_model",
            CausalWanAttentionBlock=object,
        ),
        "omegaconf": _module("omegaconf", OmegaConf=object),
        "wandb": _module("wandb"),
    }
    module_name = "layer_recall_sp_parity_trainer_under_test"
    spec = importlib.util.spec_from_file_location(module_name, TRAINER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {TRAINER_PATH}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


MODEL = _load_model_module()
TRAINER = _load_trainer_module()


def _anchor_record(*, chunk_index=0, local_frames=4, global_frames=4, sp_size=1, full_tensors=True):
    start = chunk_index * global_frames
    noisy = torch.arange(local_frames, dtype=torch.float32).reshape(1, local_frames, 1, 1, 1)
    teacher = (noisy + 1.0).requires_grad_()
    student = (noisy + 2.0).requires_grad_()
    timestep = torch.full((1, local_frames), 500.0, dtype=torch.float32)
    chunk_sum = (student - teacher).square().sum()
    chunk_count = torch.tensor(float(local_frames), dtype=torch.float32)
    return MODEL._YX_sp_parity_anchor_record(
        enabled=True,
        full_tensors=full_tensors,
        sequence_parallel_size=sp_size,
        chunk_index=chunk_index,
        start_frame=start,
        end_frame=start + global_frames,
        prediction_target="denoised_latent",
        chunk_noisy=noisy,
        local_timestep=timestep,
        teacher_target=teacher,
        student_prediction=student,
        chunk_sum=chunk_sum,
        chunk_count=chunk_count,
    )


def _capture(*, sp_size=1, local_frames=4, global_frames=4, anchor_count=1):
    records = [
        _anchor_record(
            chunk_index=index,
            local_frames=local_frames,
            global_frames=global_frames,
            sp_size=sp_size,
        )
        for index in range(anchor_count)
    ]
    return MODEL._YX_sp_parity_capture_payload(
        enabled=True,
        full_tensors=True,
        prediction_target="denoised_latent",
        rollout_mode="student_full_rollout",
        anchor_every_n_frames=global_frames,
        anchor_include_last_chunk=True,
        num_frame_per_block=global_frames,
        target_chunk_indices=list(range(anchor_count)),
        records=records,
    )


class YXLayerRecallSPParityCaptureTest(unittest.TestCase):
    def test_disabled_capture_has_no_payload(self):
        payload = MODEL._YX_sp_parity_capture_payload(
            enabled=False,
            full_tensors=True,
            prediction_target="denoised_latent",
            rollout_mode="student_full_rollout",
            anchor_every_n_frames=8,
            anchor_include_last_chunk=True,
            num_frame_per_block=8,
            target_chunk_indices=list(range(6)),
            records=None,
        )
        self.assertIsNone(payload)

        record = MODEL._YX_sp_parity_anchor_record(
            enabled=False,
            full_tensors=True,
            sequence_parallel_size=1,
            chunk_index=0,
            start_frame=0,
            end_frame=4,
            prediction_target="denoised_latent",
            chunk_noisy=torch.empty(1, 4, 1, 1, 1),
            local_timestep=torch.empty(1, 4),
            teacher_target=torch.empty(1, 4, 1, 1, 1),
            student_prediction=torch.empty(1, 4, 1, 1, 1),
            chunk_sum=torch.tensor(0.0),
            chunk_count=torch.tensor(0.0),
        )
        self.assertIsNone(record)

    def test_enabled_capture_collects_six_detached_anchor_records(self):
        capture = _capture(anchor_count=6)

        self.assertEqual(capture["anchor_schedule"]["chunk_indices"], list(range(6)))
        self.assertEqual(len(capture["anchors"]), 6)
        required = {
            "chunk_index",
            "start_frame",
            "end_frame",
            "prediction_target",
            "chunk_noisy",
            "local_timestep",
            "teacher_target",
            "student_prediction",
            "chunk_sum",
            "chunk_count",
            "finite",
            "stats",
            "frame_metadata",
        }
        for record in capture["anchors"]:
            self.assertTrue(required.issubset(record))
            for name in (
                "chunk_noisy",
                "local_timestep",
                "teacher_target",
                "student_prediction",
                "chunk_sum",
                "chunk_count",
            ):
                self.assertFalse(record[name].requires_grad)
                self.assertEqual(record[name].device.type, "cpu")
            self.assertEqual(record["finite"]["teacher_target"].dtype, torch.float32)

        stats_only = _anchor_record(full_tensors=False)
        self.assertNotIn("teacher_target", stats_only)
        self.assertIn("teacher_target", stats_only["stats"])

    def test_sp1_saver_path_content_dtype_and_hash(self):
        prompts = [["first prompt", "second prompt"]]
        capture = _capture(anchor_count=1)
        with tempfile.TemporaryDirectory() as directory:
            metrics = TRAINER._YX_save_sp_parity_capture(
                capture,
                logdir=directory,
                global_rank=3,
                sp_rank=0,
                dp_rank=3,
                sp_size=1,
                dp_size=4,
                step=12,
                actual_micro_step_seed=9876,
                batch_idx=23,
                prompts=prompts,
            )
            expected = Path(directory) / "sp_parity" / "YX_dp003_step000012_anchors.pt"
            self.assertEqual(Path(metrics["sp_parity/capture_path"]), expected)
            self.assertEqual(metrics["sp_parity/capture_count"], 1)
            self.assertTrue(expected.is_file())

            artifact = torch.load(expected, map_location="cpu", weights_only=False)
            self.assertEqual(artifact["global_rank"], 3)
            self.assertEqual(artifact["sp_rank"], 0)
            self.assertEqual(artifact["dp_rank"], 3)
            self.assertEqual(artifact["sp_size"], 1)
            self.assertEqual(artifact["actual_micro_step_seed"], 9876)
            self.assertEqual(artifact["batch_idx"], 23)
            self.assertEqual(artifact["prompts"], prompts)
            self.assertEqual(artifact["prompts_sha256"], TRAINER._YX_prompt_sha256(prompts))
            anchor = artifact["anchors"][0]
            for name in (
                "chunk_noisy",
                "local_timestep",
                "teacher_target",
                "student_prediction",
            ):
                self.assertEqual(anchor[name].device.type, "cpu")
                self.assertIn(anchor[name].dtype, {torch.float16, torch.bfloat16})
                self.assertFalse(anchor[name].requires_grad)
            self.assertEqual(anchor["chunk_sum"].dtype, torch.float32)
            self.assertEqual(anchor["stats"]["teacher_target"]["mean"].dtype, torch.float32)
            self.assertFalse(anchor["frame_metadata"]["gathered_across_sp"])

    def test_sp2_local_frames_gather_and_full_frames_skip_duplicate_gather(self):
        calls = []

        def fake_gather(tensor):
            self.assertFalse(tensor.requires_grad)
            calls.append(tuple(tensor.shape))
            return torch.cat((tensor, tensor), dim=1)

        local_artifact = TRAINER._YX_sp_parity_materialize(
            _capture(sp_size=2, local_frames=4, global_frames=8),
            global_rank=0,
            sp_rank=0,
            dp_rank=0,
            sp_size=2,
            dp_size=1,
            step=1,
            actual_micro_step_seed=10,
            batch_idx=0,
            prompts=["prompt"],
            gather_frames_fn=fake_gather,
        )
        self.assertEqual(len(calls), 6)
        local_anchor = local_artifact["anchors"][0]
        self.assertEqual(local_anchor["teacher_target"].shape[1], 8)
        self.assertTrue(local_anchor["frame_metadata"]["gathered_across_sp"])
        self.assertEqual(float(local_anchor["chunk_count"]), 8.0)

        def forbidden_gather(tensor):
            raise AssertionError("full 8-frame tensors must not be gathered again")

        full_artifact = TRAINER._YX_sp_parity_materialize(
            _capture(sp_size=2, local_frames=8, global_frames=8),
            global_rank=0,
            sp_rank=0,
            dp_rank=0,
            sp_size=2,
            dp_size=1,
            step=1,
            actual_micro_step_seed=10,
            batch_idx=0,
            prompts=["prompt"],
            gather_frames_fn=forbidden_gather,
        )
        self.assertEqual(full_artifact["anchors"][0]["teacher_target"].shape[1], 8)
        self.assertFalse(full_artifact["anchors"][0]["frame_metadata"]["gathered_across_sp"])

    def test_prompt_hash_is_stable(self):
        prompts = [["alpha", "beta"], ["gamma"]]
        first = TRAINER._YX_prompt_sha256(prompts)
        second = TRAINER._YX_prompt_sha256(tuple(tuple(group) for group in prompts))
        changed = TRAINER._YX_prompt_sha256([["beta", "alpha"], ["gamma"]])

        self.assertEqual(first, second)
        self.assertEqual(len(first), 64)
        self.assertNotEqual(first, changed)


if __name__ == "__main__":
    unittest.main()
