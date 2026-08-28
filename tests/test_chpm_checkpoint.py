#!/usr/bin/env python3
"""Two-rank GPU check for FSDP-ignored LayerRecall checkpoint and optimizer state."""

import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    StateDictType,
)


SEED = 20260718
EXPECTED_LAYER_RECALL_KEYS = {"layer_recall_gain", "layer_recall_bias"}


class YXToyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(16, 32, bias=False),
            nn.GELU(),
            nn.Linear(32, 16, bias=False),
        )
        self.backbone.requires_grad_(False)
        self.layer_recall_gain = nn.Parameter(torch.linspace(0.8, 1.2, 16))
        self.layer_recall_bias = nn.Parameter(torch.linspace(-0.1, 0.1, 16))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.backbone(inputs)
        return features * self.layer_recall_gain + self.layer_recall_bias


def clean_key(name: str) -> str:
    return name.replace("_fsdp_wrapped_module.", "").replace(
        "_checkpoint_wrapped_module.", ""
    )


def build_fsdp_model(device: torch.device):
    torch.manual_seed(SEED)
    model = YXToyModel().to(device)
    full_backbone_numel = sum(param.numel() for param in model.backbone.parameters())
    layer_recall_named_params = {
        name: param for name, param in model.named_parameters() if "layer_recall" in name
    }
    assert set(layer_recall_named_params) == EXPECTED_LAYER_RECALL_KEYS
    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=device,
        use_orig_params=True,
        ignored_states=list(layer_recall_named_params.values()),
        sync_module_states=False,
    )
    wrapped_layer_recall_named_params = {
        clean_key(name): param
        for name, param in fsdp_model.named_parameters()
        if "layer_recall" in name
    }
    assert set(wrapped_layer_recall_named_params) == EXPECTED_LAYER_RECALL_KEYS
    optimizer = torch.optim.AdamW(
        list(wrapped_layer_recall_named_params.values()),
        lr=2.0e-2,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    local_backbone_numel = sum(
        param.numel() for param in fsdp_model.module.backbone.parameters()
    )
    return (
        fsdp_model,
        wrapped_layer_recall_named_params,
        optimizer,
        full_backbone_numel,
        local_backbone_numel,
    )


def fixed_batch(device: torch.device) -> torch.Tensor:
    return torch.linspace(-1.0, 1.0, 64, device=device).reshape(4, 16)


def train_step(model: FSDP, optimizer: torch.optim.Optimizer, inputs: torch.Tensor):
    optimizer.zero_grad(set_to_none=True)
    output = model(inputs)
    loss = output.square().mean() + 0.1 * output.mean()
    loss.backward()
    optimizer.step()
    return loss.detach()


def layer_recall_vector(named_params) -> torch.Tensor:
    return torch.cat(
        [named_params[name].detach().reshape(-1) for name in sorted(named_params)]
    )


def assert_replicated(local_tensor: torch.Tensor, label: str) -> None:
    gathered = [torch.empty_like(local_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, local_tensor)
    for rank, value in enumerate(gathered[1:], start=1):
        torch.testing.assert_close(
            value,
            gathered[0],
            rtol=0.0,
            atol=0.0,
            msg=lambda message: f"{label} differs between rank 0 and rank {rank}: {message}",
        )


def snapshot_optimizer(optimizer, named_params):
    snapshot = {}
    for name, param in named_params.items():
        assert param in optimizer.state, f"optimizer has no state for {name}"
        snapshot[name] = {}
        for state_name, value in optimizer.state[param].items():
            if torch.is_tensor(value):
                snapshot[name][state_name] = value.detach().cpu().clone()
            else:
                snapshot[name][state_name] = value
    return snapshot


def assert_optimizer_snapshot_equal(actual, expected) -> None:
    assert set(actual) == set(expected)
    for param_name in expected:
        assert set(actual[param_name]) == set(expected[param_name])
        for state_name, expected_value in expected[param_name].items():
            actual_value = actual[param_name][state_name]
            if torch.is_tensor(expected_value):
                torch.testing.assert_close(
                    actual_value,
                    expected_value,
                    rtol=0.0,
                    atol=0.0,
                    msg=lambda message: (
                        f"optimizer state {param_name}.{state_name} was not restored: "
                        f"{message}"
                    ),
                )
            else:
                assert actual_value == expected_value


def optimizer_fqn_keys(full_optimizer_state) -> set[str]:
    keys = set()
    for key in full_optimizer_state.get("state", {}):
        if isinstance(key, str):
            keys.add(clean_key(key))
    for group in full_optimizer_state.get("param_groups", []):
        for key in group.get("params", []):
            if isinstance(key, str):
                keys.add(clean_key(key))
    return keys


def main() -> None:
    assert torch.cuda.is_available(), "this test requires CUDA"
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == 2, f"expected exactly 2 ranks, got {world_size}"

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="nccl")
    checkpoint_path = Path(
        os.environ.get(
            "YX_FSDP_TEST_CHECKPOINT",
            f"/tmp/YX_test_fsdp_ignored_layer_recall_checkpoint_{os.environ['MASTER_PORT']}.pt",
        )
    )

    try:
        (
            model,
            layer_recall_named_params,
            optimizer,
            full_backbone_numel,
            local_backbone_numel,
        ) = build_fsdp_model(device)
        assert model.sharding_strategy == ShardingStrategy.FULL_SHARD
        assert all(param.requires_grad for param in layer_recall_named_params.values())
        assert all(not param.requires_grad for param in model.module.backbone.parameters())
        shard_counts = [None] * world_size
        dist.all_gather_object(shard_counts, local_backbone_numel)
        assert sum(shard_counts) == full_backbone_numel
        assert all(count < full_backbone_numel for count in shard_counts)

        inputs = fixed_batch(device)
        first_loss = train_step(model, optimizer, inputs)
        assert_replicated(layer_recall_vector(layer_recall_named_params), "LayerRecall parameters after first update")
        optimizer_after_first_step = snapshot_optimizer(optimizer, layer_recall_named_params)

        with FSDP.state_dict_type(
            model,
            StateDictType.FULL_STATE_DICT,
            FullStateDictConfig(rank0_only=True, offload_to_cpu=True),
            FullOptimStateDictConfig(rank0_only=True, offload_to_cpu=True),
        ):
            full_state = model.state_dict()
            layer_recall_state = {
                clean_key(key): value
                for key, value in full_state.items()
                if "layer_recall" in key
            }
            full_optimizer_state = FSDP.optim_state_dict(model, optimizer)

        if rank == 0:
            full_state_keys = {clean_key(key) for key in full_state}
            optimizer_keys = optimizer_fqn_keys(full_optimizer_state)
            ignored_in_full_state = EXPECTED_LAYER_RECALL_KEYS.issubset(full_state_keys)
            ignored_in_optimizer_state = EXPECTED_LAYER_RECALL_KEYS.issubset(optimizer_keys)
            assert ignored_in_full_state
            assert ignored_in_optimizer_state
            assert set(layer_recall_state) == EXPECTED_LAYER_RECALL_KEYS
            assert all("_fsdp_wrapped_module" not in key for key in layer_recall_state)
            assert all("_checkpoint_wrapped_module" not in key for key in layer_recall_state)
            torch.save(
                {
                    "trainer": "chpm",
                    "checkpoint_version": 3,
                    "layer_recall_state_dict": layer_recall_state,
                    "student_optimizer": full_optimizer_state,
                    "step": 1,
                },
                checkpoint_path,
            )
            print(
                f"[ENV] torch={torch.__version__} cuda={torch.version.cuda} "
                f"world_size={world_size}"
            )
            print(
                "[CHECK] frozen backbone uses FSDP FULL_SHARD: PASS; "
                f"full_numel={full_backbone_numel} local_shards={shard_counts}"
            )
            print(
                "[SCOPE] identical per-rank inputs isolate checkpoint behavior; "
                "FSDP ignored_states does not synchronize LayerRecall gradients"
            )
            print(
                "[CHECK] ignored LayerRecall parameters are replicated after first update: "
                "PASS"
            )
            print(
                "[OBSERVATION][torch 2.8] ignored LayerRecall params in "
                f"FSDP FULL_STATE_DICT: {'YES' if ignored_in_full_state else 'NO'}; "
                f"keys={sorted(key for key in full_state_keys if 'layer_recall' in key)}"
            )
            print(
                "[OBSERVATION][torch 2.8] ignored LayerRecall params in "
                f"FSDP.optim_state_dict: {'YES' if ignored_in_optimizer_state else 'NO'}; "
                f"keys={sorted(key for key in optimizer_keys if 'layer_recall' in key)}"
            )
            print(
                f"[CHECK] external LayerRecall checkpoint keys have no FSDP prefix: PASS; "
                f"keys={sorted(layer_recall_state)}"
            )
        dist.barrier()

        second_loss = train_step(model, optimizer, inputs)
        uninterrupted_second_step_layer_recall = layer_recall_vector(layer_recall_named_params).clone()
        assert_replicated(
            uninterrupted_second_step_layer_recall,
            "uninterrupted LayerRecall parameters after second update",
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        (
            restored_model,
            restored_layer_recall_named_params,
            restored_optimizer,
            _,
            _,
        ) = build_fsdp_model(device)
        load_result = restored_model.module.load_state_dict(
            checkpoint["layer_recall_state_dict"], strict=False
        )
        layer_recall_missing = [key for key in load_result.missing_keys if "layer_recall" in key]
        assert not layer_recall_missing, f"missing restored LayerRecall keys: {layer_recall_missing}"
        assert not load_result.unexpected_keys

        optimizer_state_to_load = FSDP.optim_state_dict_to_load(
            restored_model,
            restored_optimizer,
            checkpoint["student_optimizer"],
        )
        restored_optimizer.load_state_dict(optimizer_state_to_load)
        restored_optimizer_snapshot = snapshot_optimizer(
            restored_optimizer, restored_layer_recall_named_params
        )
        assert_optimizer_snapshot_equal(
            restored_optimizer_snapshot, optimizer_after_first_step
        )
        assert_replicated(
            layer_recall_vector(restored_layer_recall_named_params), "restored LayerRecall parameters"
        )

        restored_second_loss = train_step(
            restored_model, restored_optimizer, inputs
        )
        restored_second_step_layer_recall = layer_recall_vector(restored_layer_recall_named_params)
        assert_replicated(
            restored_second_step_layer_recall, "restored LayerRecall parameters after next update"
        )
        torch.testing.assert_close(
            restored_second_step_layer_recall,
            uninterrupted_second_step_layer_recall,
            rtol=0.0,
            atol=0.0,
        )
        torch.testing.assert_close(
            restored_second_loss,
            second_loss,
            rtol=0.0,
            atol=0.0,
        )

        if rank == 0:
            print("[CHECK] optimizer state restored through FSDP conversion: PASS")
            print(
                "[CHECK] resumed next update matches uninterrupted training and "
                "both ranks agree: PASS"
            )
            print(
                f"[RESULT] PASS first_loss={first_loss.item():.9f} "
                f"second_loss={second_loss.item():.9f}"
            )
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.barrier()
        if rank == 0:
            checkpoint_path.unlink(missing_ok=True)
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
