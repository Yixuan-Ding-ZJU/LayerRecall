from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn
import torch.distributed.fsdp  # noqa: F401


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
WRAPPER_PATH = REPO_ROOT / "utils" / "wan_5b_wrapper.py"
MODEL_PATH = REPO_ROOT / "model" / "chpm.py"
TRAINER_PATH = REPO_ROOT / "trainer" / "chpm.py"


def _module(name, **attributes):
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    return module


def _load_module(name, path, stubs):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, stubs):
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return module


class _OmegaConfStub:
    @staticmethod
    def is_config(value):
        return False


def _load_wrapper_module():
    utils_package = _module("utils")
    utils_package.__path__ = []
    wan_package = _module("wan_5b")
    wan_package.__path__ = []
    wan_modules = _module("wan_5b.modules")
    wan_modules.__path__ = []
    return _load_module(
        "YX_streaming_sp_wrapper_under_test",
        WRAPPER_PATH,
        {
            "utils": utils_package,
            "utils.scheduler": _module(
                "utils.scheduler",
                SchedulerInterface=object,
                FlowMatchScheduler=object,
            ),
            "wan_5b": wan_package,
            "wan_5b.modules": wan_modules,
            "wan_5b.modules.tokenizers": _module(
                "wan_5b.modules.tokenizers",
                HuggingfaceTokenizer=object,
            ),
            "wan_5b.modules.model": _module("wan_5b.modules.model", WanModel=object),
            "wan_5b.modules.vae2_2": _module(
                "wan_5b.modules.vae2_2",
                _video_vae=lambda **kwargs: None,
            ),
            "wan_5b.modules.t5": _module("wan_5b.modules.t5", umt5_xxl=lambda **kwargs: None),
            "wan_5b.modules.causal_model": _module(
                "wan_5b.modules.causal_model",
                CausalWanModel=object,
            ),
        },
    )


def _load_model_module():
    utils_package = _module("utils")
    utils_package.__path__ = []
    return _load_module(
        "YX_streaming_sp_model_under_test",
        MODEL_PATH,
        {
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
        },
    )


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
        collective_telemetry_snapshot=lambda: {
            "collective_count": 0,
            "estimated_bytes": 0,
            "collective_time_s": 0.0,
            "operations": {},
        },
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
    return _load_module(
        "YX_streaming_sp_trainer_under_test",
        TRAINER_PATH,
        {
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
        },
    )


WRAPPER = _load_wrapper_module()
MODEL = _load_model_module()
TRAINER = _load_trainer_module()


def _bare_model(*, sp_size):
    instance = MODEL.CHPMModel.__new__(
        MODEL.CHPMModel
    )
    nn.Module.__init__(instance)
    instance.args = SimpleNamespace(chpm={})
    instance.sequence_parallel_size = int(sp_size)
    instance.num_frame_per_block = 8
    instance.frame_seq_length = 2
    instance.num_transformer_blocks = 1
    instance.num_heads = 24
    instance.head_dim = 128
    instance.student_model_kwargs = {"local_attn_size": -1}
    instance.teacher_model_kwargs = {"local_attn_size": -1}
    instance.layer_recall_config = SimpleNamespace(layer_recall_physical_cache_frames=0)
    return instance


class _LocalFlowWrapper:
    def __init__(self, local_x0):
        self.local_x0 = local_x0

    def __call__(self, **kwargs):
        return torch.zeros_like(self.local_x0), self.local_x0


class YXStreamingSPRolloutTest(unittest.TestCase):
    def test_runtime_telemetry_records_phase_cache_and_collectives(self):
        model_source = MODEL_PATH.read_text(encoding="utf-8")
        trainer_source = TRAINER_PATH.read_text(encoding="utf-8")

        for field in (
            "teacher_memory_after_model_move",
            "teacher_memory_after_cache_allocation",
            "teacher_memory_after_release",
            "teacher_max_visible_frames",
            "teacher_memory_trace",
            "student_memory_trace",
        ):
            self.assertIn(field, model_source)
        self.assertIn("rank{self.global_rank:03d}_telemetry.jsonl", trainer_source)
        self.assertIn("collective_telemetry_snapshot", trainer_source)

    def test_sp1_cache_has_24_heads_and_sp2_cache_has_12(self):
        for sp_size, expected_heads in ((1, 24), (2, 12)):
            with self.subTest(sp_size=sp_size):
                model = _bare_model(sp_size=sp_size)
                cache = model._new_kv_cache(
                    batch_size=1,
                    dtype=torch.float32,
                    device=torch.device("cpu"),
                    num_frames=80,
                    role="student",
                )
                self.assertEqual(cache[0]["k"].shape, (1, 48, expected_heads, 128))
                self.assertEqual(cache[0]["v"].shape, (1, 48, expected_heads, 128))
                self.assertEqual(cache[0]["num_heads"], expected_heads)
                self.assertEqual(cache[0]["global_num_heads"], 24)

    def test_wrapper_x0_inputs_follow_local_flow_frame_shard(self):
        noisy = torch.arange(8.0).reshape(1, 8, 1, 1, 1)
        timestep = torch.arange(100, 108).reshape(1, 8)
        local_flow = torch.zeros(1, 4, 1, 1, 1)
        local_noisy, local_timestep = WRAPPER._YX_select_local_x0_inputs(
            local_flow,
            noisy,
            timestep,
            sp_size=2,
            local_frame_start=4,
            local_frame_end=8,
        )
        torch.testing.assert_close(local_noisy, noisy[:, 4:8])
        torch.testing.assert_close(local_timestep, timestep[:, 4:8])

        sp1_noisy, sp1_timestep = WRAPPER._YX_select_local_x0_inputs(
            torch.zeros_like(noisy),
            noisy,
            timestep,
            sp_size=1,
            local_frame_start=0,
            local_frame_end=8,
        )
        self.assertIs(sp1_noisy, noisy)
        self.assertIs(sp1_timestep, timestep)

    def test_local_mask_and_timestep_use_the_same_frame_bounds(self):
        shard = MODEL._YXFrameShardMetadata(
            sp_size=2,
            sp_rank=1,
            global_frames=8,
            local_frame_start=4,
            local_frame_end=8,
        )
        timestep = torch.arange(8).reshape(1, 8)
        mask = torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]], dtype=torch.float32)
        torch.testing.assert_close(
            MODEL._YX_slice_chunk_to_local(timestep, shard),
            timestep[:, 4:8],
        )
        torch.testing.assert_close(
            MODEL._YX_slice_chunk_to_local(mask, shard),
            mask[:, 4:8],
        )

    def test_forward_exposes_local_outputs_and_only_gathers_detached_x0(self):
        model = _bare_model(sp_size=2)
        local_x0 = torch.arange(4.0, requires_grad=True).reshape(1, 4, 1, 1, 1)
        gathered_requires_grad = []

        def fake_gather(tensor):
            gathered_requires_grad.append(tensor.requires_grad)
            return torch.cat((tensor, tensor), dim=1)

        shard = MODEL._YXFrameShardMetadata(
            sp_size=2,
            sp_rank=0,
            global_frames=8,
            local_frame_start=0,
            local_frame_end=4,
        )
        with mock.patch.object(
            MODEL,
            "_YX_streaming_frame_shard",
            return_value=(object(), shard),
        ), mock.patch.object(
            MODEL,
            "_YX_all_gather_detached_context",
            side_effect=fake_gather,
        ):
            output = model._forward_stream_chunk(
                _LocalFlowWrapper(local_x0),
                chunk_latent=torch.zeros(1, 8, 1, 1, 1),
                conditional_dict={},
                timestep=torch.zeros(1, 8),
                kv_cache=[],
                crossattn_cache=[],
                chunk_index=0,
                chunk_start_frame=0,
                call_type="denoise",
                gather_detached_context_x0=True,
            )

        self.assertEqual(output.local_flow.shape[1], 4)
        self.assertEqual(output.local_x0.shape[1], 4)
        self.assertTrue(output.local_x0.requires_grad)
        self.assertEqual(output.full_context_x0.shape[1], 8)
        self.assertFalse(output.full_context_x0.requires_grad)
        self.assertEqual(gathered_requires_grad, [False])
        self.assertTrue(output.frame_shard.is_sp_shard)

    def test_context_gather_rejects_a_tensor_that_still_requires_grad(self):
        with self.assertRaisesRegex(ValueError, "detached"):
            MODEL._YX_all_gather_detached_context(
                torch.ones(1, 4, 1, 1, 1, requires_grad=True)
            )

    def test_prediction_and_regularization_use_replicated_sp_scaling(self):
        parameter = torch.tensor(3.0, dtype=torch.float64, requires_grad=True)
        prediction = MODEL._YX_normalize_replicated_prediction(
            10.0 * parameter,
            torch.tensor(5),
            sp_group=None,
        )
        regularization = MODEL._YX_scale_replicated_reg(
            6.0 * parameter,
            sp_size=2,
        )
        (prediction + regularization).backward()
        torch.testing.assert_close(
            parameter.grad,
            torch.tensor(5.0, dtype=torch.float64),
        )

    def test_fp32_master_policy_does_not_install_fp32_input_hooks(self):
        model = _bare_model(sp_size=2)
        dit = nn.Module()
        dit.layer_recall_current_norm = nn.LayerNorm(4)
        dit.layer_recall_current_mlp = nn.Sequential(
            nn.Linear(4, 3),
            nn.SiLU(),
            nn.Linear(3, 2),
        )
        dit.layer_recall_current_gate = nn.Linear(4, 1)
        dit.layer_recall_base_query = nn.Parameter(torch.ones(2, dtype=torch.float32))
        wrapper = nn.Module()
        wrapper.model = dit
        model.student = wrapper
        model.layer_recall_config = SimpleNamespace(
            layer_recall_current_conditioned_enabled=True,
        )

        installed = model.configure_student_layer_recall_fp32_island()
        self.assertEqual(
            set(installed),
            {
                "layer_recall_current_norm",
                "layer_recall_current_mlp",
                "layer_recall_current_gate",
            },
        )
        for name in installed:
            self.assertEqual(len(getattr(dit, name)._forward_pre_hooks), 0)
        self.assertEqual(
            model.layer_recall_forward_compute_policy,
            "fp32_master_differentiable_activation_dtype_cast",
        )

    def test_sp2_requires_replicated_layer_recall_but_sp1_default_remains_valid(self):
        common = {
            "global_rank": 0,
            "world_size": 2,
            "sequence_parallel_size": 2,
            "streaming_sequence_parallel_mode": "ulysses_chunk",
            "model_name": "Wan2.2-TI2V-5B",
            "num_heads": 24,
            "num_frame_per_block": 8,
        }
        with self.assertRaisesRegex(ValueError, "layer_recall_replicated_params=true"):
            TRAINER._YX_resolve_prediction_sp_topology(
                **common,
                layer_recall_replicated_params=False,
            )
        topology = TRAINER._YX_resolve_prediction_sp_topology(
            **common,
            layer_recall_replicated_params=True,
        )
        self.assertEqual(topology.local_frames, 4)
        self.assertEqual(topology.local_heads, 12)

        sp1 = TRAINER._YX_resolve_prediction_sp_topology(
            global_rank=0,
            world_size=1,
            sequence_parallel_size=1,
            streaming_sequence_parallel_mode="disabled",
            model_name="Wan2.2-TI2V-5B",
            num_heads=24,
            num_frame_per_block=8,
        )
        self.assertEqual(sp1.local_frames, 8)
        self.assertEqual(sp1.local_heads, 24)
        self.assertFalse(
            TRAINER._layer_recall_replicated_params_enabled(
                SimpleNamespace(chpm={})
            )
        )

    def test_pre_fsdp_capture_retains_exact_layer_recall_parameter_objects(self):
        model = _bare_model(sp_size=1)
        model.trainable_allowlist = ("layer_recall",)
        model.student = nn.Module()
        model.student.register_parameter(
            "layer_recall_weight",
            nn.Parameter(torch.tensor([1.0, 2.0])),
        )
        original = model.student.layer_recall_weight
        names = model.capture_pre_fsdp_trainable_layer_recall_params()
        captured = model.pre_fsdp_trainable_layer_recall_named_param_objects()
        self.assertEqual(names, ["layer_recall_weight"])
        self.assertEqual(len(captured), 1)
        self.assertIs(captured[0][1], original)
        self.assertTrue(original.requires_grad)
        original.data = original.data.to(dtype=torch.bfloat16)
        TRAINER._YX_move_replicated_layer_recall_params_(
            captured,
            device=torch.device("cpu"),
        )
        self.assertIs(captured[0][1], original)
        self.assertEqual(original.dtype, torch.float32)

        optimizer = torch.optim.AdamW([original], lr=1.0e-2)
        optimizer.zero_grad(set_to_none=True)
        original.square().sum().backward()
        optimizer.step()
        TRAINER._YX_assert_replicated_layer_recall_optimizer_fp32(
            optimizer,
            captured,
            require_state=True,
        )
        self.assertEqual(optimizer.state[original]["exp_avg"].dtype, torch.float32)
        self.assertEqual(optimizer.state[original]["exp_avg_sq"].dtype, torch.float32)

    def test_layer_recall_loader_accepts_only_chpm_v3_schema(self):
        model = _bare_model(sp_size=1)
        model.student = nn.Module()
        model.student.register_parameter(
            "layer_recall_weight",
            nn.Parameter(torch.zeros(2)),
        )
        state = {"layer_recall_weight": torch.ones(2)}

        with self.assertRaisesRegex(ValueError, "CHPM checkpoint"):
            model.load_layer_recall_state_dict(state)
        with self.assertRaisesRegex(ValueError, "version 3"):
            model.load_layer_recall_state_dict(
                {
                    "trainer": "chpm",
                    "checkpoint_version": 2,
                    "layer_recall_state_dict": state,
                }
            )

        missing, unexpected = model.load_layer_recall_state_dict(
            {
                "trainer": "chpm",
                "checkpoint_version": 3,
                "layer_recall_state_dict": state,
            }
        )
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        torch.testing.assert_close(model.student.layer_recall_weight, torch.ones(2))


if __name__ == "__main__":
    unittest.main()
