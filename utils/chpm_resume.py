"""Strict checkpoint/resume helpers for LayerRecall prediction distillation."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections.abc import Mapping
from typing import Any

import numpy as np
import torch


_STREAM_STATE_VERSION = 1
_RNG_STATE_VERSION = 1


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class CHPMPromptStream:
    """Deterministic DP-sharded prompt stream with committed resume cursor.

    The per-epoch ordering matches ``DistributedSampler(..., shuffle=True,
    drop_last=True)``. ``peek_batch`` does not advance committed state;
    ``commit_batch`` is called only after the corresponding training micro-step
    has completed.
    """

    REQUIRED_STATE_KEYS = frozenset(
        {
            "version",
            "dataset_size",
            "rank",
            "num_replicas",
            "batch_size",
            "shuffle",
            "drop_last",
            "seed",
            "epoch",
            "sample_cursor",
            "global_micro_step",
        }
    )

    def __init__(
        self,
        dataset_size: int,
        *,
        rank: int,
        num_replicas: int,
        batch_size: int = 1,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int = 0,
        epoch: int = 0,
        sample_cursor: int = 0,
        global_micro_step: int = 0,
    ) -> None:
        self.dataset_size = self._non_negative_int("dataset_size", dataset_size)
        self.num_replicas = self._positive_int("num_replicas", num_replicas)
        self.rank = self._non_negative_int("rank", rank)
        if self.rank >= self.num_replicas:
            raise ValueError(
                f"rank must be in [0, {self.num_replicas}), got {self.rank}"
            )
        self.batch_size = self._positive_int("batch_size", batch_size)
        if not isinstance(shuffle, bool) or not isinstance(drop_last, bool):
            raise TypeError("shuffle and drop_last must be bool")
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = self._int("seed", seed)
        self.epoch = self._non_negative_int("epoch", epoch)
        self.sample_cursor = self._non_negative_int("sample_cursor", sample_cursor)
        self.global_micro_step = self._non_negative_int(
            "global_micro_step", global_micro_step
        )
        self._pending_indices: tuple[int, ...] | None = None
        self._rebuild_epoch()
        self._validate_cursor()

    @staticmethod
    def _int(name: str, value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{name} must be int, got {type(value).__name__}")
        return value

    @classmethod
    def _non_negative_int(cls, name: str, value: Any) -> int:
        value = cls._int(name, value)
        if value < 0:
            raise ValueError(f"{name} must be non-negative, got {value}")
        return value

    @classmethod
    def _positive_int(cls, name: str, value: Any) -> int:
        value = cls._int(name, value)
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
        return value

    @property
    def num_samples(self) -> int:
        if self.drop_last and self.dataset_size % self.num_replicas != 0:
            return math.ceil((self.dataset_size - self.num_replicas) / self.num_replicas)
        return math.ceil(self.dataset_size / self.num_replicas)

    @property
    def num_batches_per_epoch(self) -> int:
        return math.ceil(self.num_samples / self.batch_size)

    @property
    def batch_cursor_in_epoch(self) -> int:
        return self.sample_cursor // self.batch_size

    @property
    def has_pending_batch(self) -> bool:
        return self._pending_indices is not None

    def _rebuild_epoch(self) -> None:
        if self.dataset_size <= 0:
            raise ValueError("dataset_size must be positive for prompt training")
        if self.num_samples <= 0:
            raise ValueError(
                "drop_last=True leaves no samples: "
                f"dataset_size={self.dataset_size}, num_replicas={self.num_replicas}"
            )
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(self.dataset_size, generator=generator).tolist()
        else:
            indices = list(range(self.dataset_size))

        total_size = self.num_samples * self.num_replicas
        if self.drop_last:
            indices = indices[:total_size]
        elif total_size > len(indices):
            padding_size = total_size - len(indices)
            repeats = math.ceil(padding_size / len(indices))
            indices += (indices * repeats)[:padding_size]
        self._local_indices = indices[self.rank:total_size:self.num_replicas]
        if len(self._local_indices) != self.num_samples:
            raise RuntimeError(
                "prompt stream local shard length mismatch: "
                f"expected={self.num_samples}, actual={len(self._local_indices)}"
            )

    def _validate_cursor(self) -> None:
        if self.sample_cursor >= self.num_samples:
            raise ValueError(
                "sample_cursor must identify the next sample inside the current epoch: "
                f"cursor={self.sample_cursor}, num_samples={self.num_samples}"
            )

    def peek_batch(self) -> list[int]:
        if self._pending_indices is not None:
            raise RuntimeError("a prompt batch is already pending commit")
        end = min(self.num_samples, self.sample_cursor + self.batch_size)
        self._pending_indices = tuple(self._local_indices[self.sample_cursor:end])
        if not self._pending_indices:
            raise RuntimeError("prompt stream produced an empty batch")
        return list(self._pending_indices)

    def commit_batch(self, expected_indices: list[int] | tuple[int, ...] | None = None) -> None:
        if self._pending_indices is None:
            raise RuntimeError("no pending prompt batch to commit")
        if expected_indices is not None and tuple(expected_indices) != self._pending_indices:
            raise RuntimeError(
                "committed prompt indices differ from pending batch: "
                f"pending={list(self._pending_indices)}, committed={list(expected_indices)}"
            )
        self.sample_cursor += len(self._pending_indices)
        self.global_micro_step += 1
        self._pending_indices = None
        if self.sample_cursor == self.num_samples:
            self.epoch += 1
            self.sample_cursor = 0
            self._rebuild_epoch()
        elif self.sample_cursor > self.num_samples:
            raise RuntimeError("prompt stream cursor advanced beyond the epoch boundary")

    def state_dict(self) -> dict[str, Any]:
        if self._pending_indices is not None:
            raise RuntimeError("refusing to snapshot prompt stream with an uncommitted batch")
        return {
            "version": _STREAM_STATE_VERSION,
            "dataset_size": self.dataset_size,
            "rank": self.rank,
            "num_replicas": self.num_replicas,
            "batch_size": self.batch_size,
            "shuffle": self.shuffle,
            "drop_last": self.drop_last,
            "seed": self.seed,
            "epoch": self.epoch,
            "sample_cursor": self.sample_cursor,
            "batch_cursor_in_epoch": self.batch_cursor_in_epoch,
            "global_micro_step": self.global_micro_step,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError(f"prompt stream state must be a mapping, got {type(state).__name__}")
        missing = self.REQUIRED_STATE_KEYS.difference(state)
        if missing:
            raise KeyError(f"prompt stream state is missing keys: {sorted(missing)}")
        if state["version"] != _STREAM_STATE_VERSION:
            raise ValueError(
                f"unsupported prompt stream version {state['version']}; "
                f"expected {_STREAM_STATE_VERSION}"
            )
        for name in (
            "dataset_size",
            "rank",
            "num_replicas",
            "batch_size",
            "shuffle",
            "drop_last",
            "seed",
        ):
            if state[name] != getattr(self, name):
                raise ValueError(
                    f"prompt stream {name} mismatch: "
                    f"saved={state[name]!r}, current={getattr(self, name)!r}"
                )
        self.epoch = self._non_negative_int("state['epoch']", state["epoch"])
        self.sample_cursor = self._non_negative_int(
            "state['sample_cursor']", state["sample_cursor"]
        )
        self.global_micro_step = self._non_negative_int(
            "state['global_micro_step']", state["global_micro_step"]
        )
        self._pending_indices = None
        self._rebuild_epoch()
        self._validate_cursor()
        saved_batch_cursor = state.get("batch_cursor_in_epoch")
        if saved_batch_cursor is not None and int(saved_batch_cursor) != self.batch_cursor_in_epoch:
            raise ValueError(
                "prompt stream batch cursor is inconsistent with sample cursor: "
                f"saved={saved_batch_cursor}, derived={self.batch_cursor_in_epoch}"
            )


def capture_rng_state(
    *,
    device: int | torch.device,
    data_generator: torch.Generator | None = None,
) -> dict[str, Any]:
    state = {
        "version": _RNG_STATE_VERSION,
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state(device),
        "data_generator": None,
    }
    if data_generator is not None:
        state["data_generator"] = data_generator.get_state()
    return state


def validate_rng_state(state: Mapping[str, Any]) -> None:
    if not isinstance(state, Mapping):
        raise RuntimeError(f"RNG state must be a mapping, got {type(state).__name__}")
    required = {
        "version",
        "python",
        "numpy",
        "torch_cpu",
        "torch_cuda",
        "data_generator",
    }
    missing = required.difference(state)
    if missing:
        raise RuntimeError(f"RNG state is incomplete; missing {sorted(missing)}")
    if state["version"] != _RNG_STATE_VERSION:
        raise RuntimeError(
            f"unsupported RNG state version {state['version']}; expected {_RNG_STATE_VERSION}"
        )
    for key in ("torch_cpu", "torch_cuda"):
        if not torch.is_tensor(state[key]):
            raise RuntimeError(f"RNG state {key!r} must be a tensor")
    if state["data_generator"] is not None and not torch.is_tensor(
        state["data_generator"]
    ):
        raise RuntimeError("RNG state 'data_generator' must be a tensor or None")


def restore_rng_state(
    state: Mapping[str, Any],
    *,
    device: int | torch.device,
    data_generator: torch.Generator | None = None,
) -> None:
    validate_rng_state(state)
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"].cpu())
    torch.cuda.set_rng_state(state["torch_cuda"].cpu(), device)
    if data_generator is not None:
        generator_state = state["data_generator"]
        if generator_state is None:
            raise RuntimeError("resume RNG state is missing DataLoader generator state")
        data_generator.set_state(generator_state.cpu())


__all__ = [
    "CHPMPromptStream",
    "canonical_sha256",
    "capture_rng_state",
    "restore_rng_state",
    "validate_rng_state",
]
